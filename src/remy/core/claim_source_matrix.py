"""Explainable claim-to-source binding for chat and research outputs.

Counting accepted URLs is not citation coverage. This module builds the
actual matrix: each claim is mapped to the fetched source material that
supports it, only partially overlaps, fails identity checks, or participates
in an explicit contradiction. Everything is deterministic and local.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from remy.core.local_search_reranker import LocalMultilingualReranker
from remy.core.search_gateway import canonicalize_url


_URL_RE = re.compile(r"https?://[^\s<>\]\)\"',]+", re.IGNORECASE)
_FETCH_TOOLS = frozenset(
    {"browse_page", "browser_act", "http_get", "extract_content", "fetch_url"}
)


def _claim_text(claim: Any) -> str:
    if isinstance(claim, str):
        return claim.strip()
    if isinstance(claim, Mapping):
        return str(
            claim.get("summary")
            or claim.get("text")
            or claim.get("claim")
            or claim.get("content")
            or ""
        ).strip()
    return str(getattr(claim, "text", "") or "").strip()


def _claim_class(claim: Any) -> str:
    if isinstance(claim, Mapping):
        return str(claim.get("claim_class") or claim.get("type") or "finding")
    return str(getattr(claim, "claim_class", "finding") or "finding")


def _explicit_urls(claim: Any, text: str) -> list[str]:
    values: list[str] = []
    if isinstance(claim, Mapping):
        for key in ("source_url", "url"):
            value = str(claim.get(key) or "").strip()
            if value:
                values.append(value)
        for key in ("source_urls", "citations", "sources"):
            raw = claim.get(key) or []
            if isinstance(raw, str):
                raw = [raw]
            for value in raw if isinstance(raw, Sequence) else []:
                if isinstance(value, Mapping):
                    value = value.get("url") or value.get("uri") or ""
                value = str(value or "").strip()
                if value.startswith(("http://", "https://")):
                    values.append(value)
    values.extend(_URL_RE.findall(text))
    return list(
        dict.fromkeys(url for value in values if (url := canonicalize_url(value)))
    )


def _source_payload(source: Mapping[str, Any], position: int) -> dict[str, Any] | None:
    result = source.get("result") if isinstance(source.get("result"), Mapping) else {}
    nested = source.get("source") if isinstance(source.get("source"), Mapping) else {}
    url = canonicalize_url(
        str(
            source.get("url")
            or source.get("uri")
            or result.get("url")
            or nested.get("uri")
            or nested.get("url")
            or ""
        )
    )
    if not url:
        return None
    packet = source.get("evidence_packet") or result.get("evidence_packet") or {}
    identity_mismatch = bool(
        isinstance(packet, Mapping)
        and (packet.get("has_mismatch") or packet.get("ok") is False)
    )
    content = str(
        source.get("content")
        or result.get("content")
        or source.get("body")
        or ""
    )
    return {
        "source_id": str(source.get("source_id") or f"source-{position}"),
        "url": url,
        "domain": (urlsplit(url).hostname or "").lower().removeprefix("www."),
        "title": str(source.get("title") or result.get("title") or nested.get("title") or ""),
        "snippet": str(
            source.get("snippet")
            or nested.get("snippet")
            or result.get("description")
            or ""
        ),
        "content": content[:20_000],
        "source_class": str(
            source.get("source_class") or nested.get("source_class") or "unknown"
        ),
        "identity_mismatch": identity_mismatch,
        "evidence_packet": dict(packet) if isinstance(packet, Mapping) else {},
        "original_url": str(
            source.get("original_url")
            or result.get("original_url")
            or nested.get("original_url")
            or ""
        ),
        "original_source_url": str(
            source.get("original_source_url")
            or result.get("original_source_url")
            or nested.get("original_source_url")
            or ""
        ),
        "canonical_source_url": str(
            source.get("canonical_source_url")
            or result.get("canonical_source_url")
            or nested.get("canonical_source_url")
            or ""
        ),
        "syndicated_from": str(
            source.get("syndicated_from")
            or result.get("syndicated_from")
            or nested.get("syndicated_from")
            or ""
        ),
        "metadata": dict(
            source.get("metadata")
            or result.get("metadata")
            or nested.get("metadata")
            or {}
        )
        if isinstance(
            source.get("metadata")
            or result.get("metadata")
            or nested.get("metadata"),
            Mapping,
        )
        else {},
        "citations": (
            source.get("citations")
            or result.get("citations")
            or nested.get("citations")
            or []
        ),
        "date": str(
            source.get("date") or result.get("date") or nested.get("date") or ""
        ),
        "published_at": str(
            source.get("published_at")
            or result.get("published_at")
            or nested.get("published_at")
            or ""
        ),
        "publication_date": str(
            source.get("publication_date")
            or result.get("publication_date")
            or nested.get("publication_date")
            or ""
        ),
        "date_published": str(
            source.get("date_published")
            or result.get("date_published")
            or nested.get("date_published")
            or source.get("datePublished")
            or result.get("datePublished")
            or nested.get("datePublished")
            or ""
        ),
        "modified_at": str(
            source.get("modified_at")
            or result.get("modified_at")
            or nested.get("modified_at")
            or source.get("dateModified")
            or result.get("dateModified")
            or nested.get("dateModified")
            or ""
        ),
        "last_modified": str(
            source.get("last_modified")
            or result.get("last_modified")
            or nested.get("last_modified")
            or ""
        ),
    }


def normalize_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and fuse source material by canonical URL."""
    fused: dict[str, dict[str, Any]] = {}
    for position, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            continue
        payload = _source_payload(source, position)
        if payload is None:
            continue
        existing = fused.get(payload["url"])
        if existing is None:
            fused[payload["url"]] = payload
            continue
        for key in ("title", "snippet", "content"):
            if len(payload[key]) > len(existing[key]):
                existing[key] = payload[key]
        existing["identity_mismatch"] = bool(
            existing["identity_mismatch"] or payload["identity_mismatch"]
        )
        if payload["evidence_packet"]:
            existing["evidence_packet"] = payload["evidence_packet"]
        for key in (
            "original_url",
            "original_source_url",
            "canonical_source_url",
            "syndicated_from",
            "metadata",
            "citations",
            "date",
            "published_at",
            "publication_date",
            "date_published",
            "modified_at",
            "last_modified",
        ):
            if not existing.get(key) and payload.get(key):
                existing[key] = payload[key]
    return list(fused.values())


def sources_from_session_log(session_log: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract fetched evidence payloads from a chat/agent session log."""
    sources: list[dict[str, Any]] = []
    for entry in session_log or []:
        if not isinstance(entry, Mapping) or entry.get("type") != "tool_call":
            continue
        if str(entry.get("tool") or "") not in _FETCH_TOOLS:
            continue
        payload: dict[str, Any] = {}
        for key in ("result_full", "result"):
            raw = entry.get(key)
            if isinstance(raw, Mapping):
                payload = dict(raw)
                break
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed, dict):
                    payload = parsed
                    break
        args = entry.get("args_full") or entry.get("args") or {}
        if not isinstance(args, Mapping):
            args = {}
        if not payload.get("url") and args.get("url"):
            payload["url"] = args.get("url")
        if payload.get("error"):
            continue
        sources.append(payload)
    return normalize_sources(sources)


def _claim_id(index: int, text: str) -> str:
    digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()[:12]
    return f"claim-{index}-{digest}"


def _contradiction_matches(
    claim_text: str,
    explicit_urls: set[str],
    contradiction: Mapping[str, Any],
    reranker: LocalMultilingualReranker,
) -> bool:
    contradiction_urls = {
        canonicalize_url(str(contradiction.get(key) or ""))
        for key in ("source_a", "source_b", "url_a", "url_b")
    }
    contradiction_urls.discard("")
    if explicit_urls & contradiction_urls:
        return True
    texts = [
        str(contradiction.get(key) or "").strip()
        for key in ("claim_a", "claim_b", "summary", "text")
    ]
    texts = [text for text in texts if text and not text.startswith(("http://", "https://"))]
    if not texts:
        return False
    scores = reranker.score_candidates(
        claim_text,
        [{"title": text, "snippet": "", "uri": f"https://conflict.local/{i}"}
         for i, text in enumerate(texts)],
    )
    return max((float(score.get("score") or 0.0) for score in scores), default=0.0) >= 0.68


def _normalized_claim_text(text: str) -> str:
    return " ".join(str(text or "").casefold().split()).strip(" .")


def _contradiction_is_active(contradiction: Mapping[str, Any]) -> bool:
    status = str(
        contradiction.get("resolution_status")
        or contradiction.get("status")
        or "unresolved"
    ).casefold()
    return status not in {
        "resolved",
        "superseded",
        "resolved_by_temporal_supersession",
        "already_resolved",
        "dismissed",
    }


def _repair_query(text: str, status: str) -> str:
    compact = " ".join(text.split())[:180]
    if status == "conflict":
        return f'{compact} independent primary source resolve conflicting evidence'
    return f'{compact} official primary source evidence'


def _provenance_repair_query(text: str) -> str:
    compact = " ".join(text.split())[:180]
    return (
        f'{compact} independent primary source not derived from existing citations'
    )


def build_claim_source_matrix(
    claims: Sequence[Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    contradictions: Sequence[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build claim rows and aggregate coverage from fetched source material."""
    normalized_sources = normalize_sources(sources)
    from remy.core.source_provenance_graph import build_source_provenance_graph

    provenance_graph = build_source_provenance_graph(
        normalized_sources,
        minimum_content_chars=20,
    )
    provenance_nodes = {
        str(node.get("url") or ""): node
        for node in provenance_graph.get("nodes") or []
        if isinstance(node, Mapping)
    }
    from remy.core.claim_temporal_evidence import evaluate_claim_temporal_evidence

    sources_by_url = {source["url"]: source for source in normalized_sources}
    from remy.core.temporal_supersession import evaluate_temporal_supersessions

    evaluated_contradictions = evaluate_temporal_supersessions(
        contradictions,
        normalized_sources,
        now=now,
    )
    reranker = LocalMultilingualReranker()
    rows: list[dict[str, Any]] = []

    for index, claim in enumerate(claims, start=1):
        text = _claim_text(claim)
        if not text:
            continue
        explicit = set(_explicit_urls(claim, text))
        score_inputs = [
            {
                "title": source["title"],
                "snippet": "\n".join(
                    part for part in (source["snippet"], source["content"]) if part
                )[:20_000],
                "uri": source["url"],
            }
            for source in normalized_sources
        ]
        scores = reranker.score_candidates(text, score_inputs) if score_inputs else []
        relations: list[dict[str, Any]] = []
        for source, score in zip(normalized_sources, scores):
            # The reranker's final score intentionally discounts snippet/body
            # matches versus titles. Here the fetched body *is* the evidence,
            # so binding uses its undiluted weighted coverage when stronger.
            lexical = max(
                float(score.get("score") or 0.0),
                float(score.get("coverage") or 0.0),
            )
            direct = source["url"] in explicit
            has_material = len((source["content"] or source["snippet"]).strip()) >= 20
            if source["identity_mismatch"] and direct:
                relation = "identity_mismatch"
                reason = "explicit citation failed evidence-packet identity checks"
            elif direct and has_material and lexical >= 0.34:
                relation = "supports"
                reason = "explicit citation with matching fetched content"
            elif direct:
                relation = "partial"
                reason = (
                    "explicit citation has insufficient readable content"
                    if not has_material
                    else "explicit citation has weak claim-to-content overlap"
                )
            elif has_material and lexical >= 0.72:
                relation = "supports"
                reason = "fetched content strongly matches the claim"
            elif has_material and lexical >= 0.48:
                relation = "partial"
                reason = "fetched content partially matches the claim"
            else:
                continue
            provenance = provenance_nodes.get(source["url"], {})
            relations.append(
                {
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "domain": source["domain"],
                    "relation": relation,
                    "score": round(lexical, 4),
                    "reason": reason,
                    "matched_exact": list(score.get("matched_exact") or []),
                    "matched_fuzzy": dict(score.get("matched_fuzzy") or {}),
                    "evidence_root": str(
                        provenance.get("evidence_root") or source["url"]
                    ),
                    "authority_role": str(
                        provenance.get("authority_role") or "unknown"
                    ),
                    "authority_score": float(
                        provenance.get("authority_score") or 0.0
                    ),
                }
            )

        matching_contradictions = [
            contradiction
            for contradiction in evaluated_contradictions
            if _contradiction_matches(text, explicit, contradiction, reranker)
        ]
        active_contradictions = [
            contradiction
            for contradiction in matching_contradictions
            if _contradiction_is_active(contradiction)
        ]
        resolved_supersessions = [
            contradiction
            for contradiction in matching_contradictions
            if contradiction.get("resolution_status")
            == "resolved_by_temporal_supersession"
        ]
        normalized_text = _normalized_claim_text(text)
        superseded_by = next(
            (
                dict(contradiction.get("supersession") or {})
                for contradiction in resolved_supersessions
                if normalized_text
                == _normalized_claim_text(
                    str((contradiction.get("supersession") or {}).get("old_claim") or "")
                )
            ),
            {},
        )
        conflicting = bool(active_contradictions)
        support_relations = [item for item in relations if item["relation"] == "supports"]
        partial_relations = [item for item in relations if item["relation"] == "partial"]
        mismatch_relations = [
            item for item in relations if item["relation"] == "identity_mismatch"
        ]
        if superseded_by:
            status = "superseded"
        elif conflicting:
            status = "conflict"
        elif support_relations:
            status = "supported"
        elif partial_relations or mismatch_relations:
            status = "partial"
        else:
            status = "unsupported"
        relations.sort(
            key=lambda item: (
                {"supports": 0, "partial": 1, "identity_mismatch": 2}.get(
                    item["relation"], 3
                ),
                -item["score"],
                item["url"],
            )
        )
        support_domains = {
            item["domain"] for item in support_relations if item.get("domain")
        }
        support_roots = {
            str(item.get("evidence_root") or item.get("url") or "")
            for item in support_relations
            if item.get("evidence_root") or item.get("url")
        }
        primary_support_roots = {
            str(item.get("evidence_root") or item.get("url") or "")
            for item in support_relations
            if item.get("authority_role") == "primary"
        }
        false_corroboration = bool(
            len(support_domains) >= 2 and len(support_roots) < 2
        )
        root_authority: dict[str, float] = {}
        for item in support_relations:
            root = str(item.get("evidence_root") or item.get("url") or "")
            if root:
                root_authority[root] = max(
                    root_authority.get(root, 0.0),
                    float(item.get("authority_score") or 0.0),
                )
        temporal = evaluate_claim_temporal_evidence(
            text,
            [
                sources_by_url[item["url"]]
                for item in support_relations
                if item["url"] in sources_by_url
            ],
            now=now,
        )
        temporal_by_url = {
            str(item.get("source_url") or ""): item
            for item in temporal.get("assessments") or []
        }
        temporally_valid_roots: set[str] = set()
        for relation in support_relations:
            assessment = temporal_by_url.get(relation["url"], {})
            relation["source_date"] = str(assessment.get("source_date") or "")
            relation["source_age_days"] = assessment.get("age_days")
            relation["temporal_status"] = str(
                assessment.get("status") or "undated"
            )
            relation["temporally_valid"] = bool(
                assessment.get("temporally_valid", False)
            )
            if relation["temporally_valid"]:
                temporally_valid_roots.add(
                    str(relation.get("evidence_root") or relation["url"])
                )
        row = {
            "claim_id": _claim_id(index, text),
            "claim": text,
            "claim_class": _claim_class(claim),
            "status": status,
            "explicit_source_urls": sorted(explicit),
            "relations": relations[:8],
            "support_count": len(support_relations),
            "independent_support_domains": len(support_domains),
            "independent_support_roots": len(support_roots),
            "support_roots": sorted(support_roots),
            "primary_support_roots": len(primary_support_roots),
            "authority_weighted_support": round(sum(root_authority.values()), 3),
            "derived_support_count": sum(
                item.get("authority_role") in {"derived", "syndicated"}
                for item in support_relations
            ),
            "corroborated": len(support_roots) >= 2,
            "false_corroboration": false_corroboration,
            "provenance_status": (
                "false_corroboration"
                if false_corroboration
                else "independent"
                if len(support_roots) >= 2
                else "single_root"
                if support_roots
                else "no_support_root"
            ),
            "time_sensitive": bool(temporal.get("time_sensitive")),
            "volatility": str(temporal.get("volatility") or "low"),
            "freshness_window_days": temporal.get("ttl_days"),
            "temporal_status": str(temporal.get("status") or "not_applicable"),
            "temporal_ready": bool(temporal.get("temporal_ready")),
            "fresh_support_count": int(temporal.get("fresh_source_count") or 0),
            "dated_support_count": int(temporal.get("dated_source_count") or 0),
            "temporally_valid_support_roots": len(temporally_valid_roots),
            "temporal_signals": list(temporal.get("signals") or []),
            "historically_scoped": bool(temporal.get("historically_scoped")),
            "supersession_role": "old" if superseded_by else (
                "new"
                if any(
                    normalized_text
                    == _normalized_claim_text(
                        str((item.get("supersession") or {}).get("new_claim") or "")
                    )
                    for item in resolved_supersessions
                )
                else ""
            ),
            "superseded_by": superseded_by,
            "resolved_supersessions": [
                dict(item.get("supersession") or {})
                for item in resolved_supersessions
            ],
            "active_contradiction_count": len(active_contradictions),
        }
        repairs: list[str] = []
        if false_corroboration:
            repairs.append(_provenance_repair_query(text))
        if status not in {"supported", "superseded"}:
            repairs.append(_repair_query(text, status))
        if temporal.get("repair_query") and status != "superseded":
            repairs.append(str(temporal["repair_query"]))
        if repairs:
            row["repair_queries"] = list(dict.fromkeys(repairs))
            row["repair_query"] = row["repair_queries"][0]
        rows.append(row)

    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("supported", "partial", "unsupported", "conflict", "superseded")
    }
    total = len(rows)
    active_rows = [row for row in rows if row["status"] != "superseded"]
    active_total = len(active_rows)
    supported_rate = counts["supported"] / active_total if active_total else 1.0
    evidence_rate = (
        counts["supported"] + counts["partial"] + counts["conflict"]
    ) / active_total if active_total else 1.0
    corroborated = sum(bool(row["corroborated"]) for row in active_rows)
    false_corroborated = sum(bool(row["false_corroboration"]) for row in active_rows)
    claims_with_roots = sum(
        int(row["independent_support_roots"]) > 0 for row in active_rows
    )
    time_sensitive_claims = sum(bool(row["time_sensitive"]) for row in active_rows)
    temporally_ready_claims = sum(
        bool(row["time_sensitive"] and row["temporal_ready"]) for row in active_rows
    )
    stale_claims = sum(row["temporal_status"] == "stale" for row in active_rows)
    undated_temporal_claims = sum(
        bool(
            row["time_sensitive"]
            and row["temporal_status"] in {"undated", "future_dated"}
        )
        for row in active_rows
    )
    temporal_ready = all(bool(row["temporal_ready"]) for row in active_rows)
    resolved_temporal_conflicts = sum(
        item.get("resolution_status") == "resolved_by_temporal_supersession"
        for item in evaluated_contradictions
    )
    unresolved_contradictions = sum(
        _contradiction_is_active(item) for item in evaluated_contradictions
    )
    repair_queries = list(
        dict.fromkeys(
            str(query or "")
            for row in rows
            for query in (
                row.get("repair_queries")
                or ([row.get("repair_query")] if row.get("repair_query") else [])
            )
            if query
        )
    )
    return {
        "version": 4,
        "method": "local_claim_source_provenance_temporal_supersession",
        "claim_count": total,
        "active_claim_count": active_total,
        "source_count": len(normalized_sources),
        "supported_claims": counts["supported"],
        "partial_claims": counts["partial"],
        "unsupported_claims": counts["unsupported"],
        "conflicting_claims": counts["conflict"],
        "superseded_claims": counts["superseded"],
        "claim_coverage_rate": round(supported_rate, 4),
        "evidence_coverage_rate": round(evidence_rate, 4),
        "corroborated_claims": corroborated,
        "corroboration_rate": round(corroborated / total if total else 1.0, 4),
        "false_corroborated_claims": false_corroborated,
        "provenance_coverage_rate": round(
            claims_with_roots / total if total else 1.0, 4
        ),
        "provenance_ready": false_corroborated == 0,
        "time_sensitive_claims": time_sensitive_claims,
        "temporally_ready_claims": temporally_ready_claims,
        "stale_claims": stale_claims,
        "undated_temporal_claims": undated_temporal_claims,
        "temporal_coverage_rate": round(
            temporally_ready_claims / time_sensitive_claims
            if time_sensitive_claims
            else 1.0,
            4,
        ),
        "temporal_ready": temporal_ready,
        "resolved_temporal_conflicts": resolved_temporal_conflicts,
        "unresolved_contradictions": unresolved_contradictions,
        "contradiction_resolutions": evaluated_contradictions,
        "source_provenance": {
            key: provenance_graph.get(key)
            for key in (
                "node_count",
                "edge_count",
                "independent_root_count",
                "primary_root_count",
                "authority_weighted_roots",
            )
        },
        "publication_ready": bool(
            active_total
            and counts["supported"] == active_total
            and counts["conflict"] == 0
            and false_corroborated == 0
            and temporal_ready
        ),
        "repair_queries": repair_queries[:8],
        "rows": rows,
    }


def matrix_to_markdown(matrix: Mapping[str, Any], *, max_rows: int = 12) -> str:
    """Render a compact auditable matrix for research artifacts."""
    lines = [
        "## Claim–source matrix",
        "",
        "| Status | Claim | Independent roots | Temporal | Supersession | Supporting evidence |",
        "|---|---|---:|---|---|---|",
    ]
    for row in list(matrix.get("rows") or [])[:max_rows]:
        claim = str(row.get("claim") or "").replace("|", "\\|")[:180]
        relations = [
            relation
            for relation in row.get("relations") or []
            if relation.get("relation") == "supports"
        ]
        evidence = ", ".join(
            f"[{item.get('domain') or 'source'}]({item.get('url')})"
            for item in relations[:3]
        ) or "—"
        roots = int(row.get("independent_support_roots") or 0)
        root_label = (
            f"{roots} (shared origin)"
            if row.get("false_corroboration")
            else str(roots)
        )
        supersession = row.get("superseded_by") or {}
        supersession_label = "вЂ”"
        if row.get("status") == "superseded":
            newer = str(supersession.get("new_claim") or "newer claim").replace(
                "|", "\\|"
            )[:90]
            effective = str(
                supersession.get("effective_at")
                or supersession.get("new_source_date")
                or "unknown date"
            )[:10]
            supersession_label = f"→ {newer} ({effective}; history retained)"
        lines.append(
            f"| {row.get('status', 'unsupported')} | {claim} | "
            f"{root_label} | {row.get('temporal_status', 'not_applicable')} | "
            f"{supersession_label} | "
            f"{evidence} |"
        )
    return "\n".join(lines)


def evaluate_claim_source_matrix(
    matrix: Mapping[str, Any],
    expectations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate matrix rows against deterministic benchmark expectations."""
    rows = list(matrix.get("rows") or [])
    total = max(len(rows), len(expectations))
    status_correct = 0
    expected_support: set[tuple[int, str]] = set()
    predicted_support: set[tuple[int, str]] = set()
    unsupported_expected = unsupported_detected = 0
    conflict_expected = conflict_detected = 0
    negative_expected = false_supports = 0
    corroboration_expected = corroboration_correct = 0
    false_corroboration_expected = false_corroboration_detected = 0
    temporal_expected = temporal_correct = 0
    stale_expected = stale_detected = 0
    supersession_expected_values: list[int] = []

    for index in range(total):
        row = rows[index] if index < len(rows) else {}
        expected = expectations[index] if index < len(expectations) else {}
        wanted_status = str(expected.get("status") or "unsupported")
        actual_status = str(row.get("status") or "missing")
        status_correct += actual_status == wanted_status
        if wanted_status == "unsupported":
            unsupported_expected += 1
            unsupported_detected += actual_status == "unsupported"
        if wanted_status == "conflict":
            conflict_expected += 1
            conflict_detected += actual_status == "conflict"
        if wanted_status != "supported":
            negative_expected += 1
            false_supports += actual_status == "supported"
        if "corroborated" in expected:
            corroboration_expected += 1
            corroboration_correct += bool(row.get("corroborated")) is bool(
                expected.get("corroborated")
            )
        if expected.get("false_corroboration"):
            false_corroboration_expected += 1
            false_corroboration_detected += bool(row.get("false_corroboration"))
        if "temporal_status" in expected:
            temporal_expected += 1
            wanted_temporal = str(expected.get("temporal_status") or "")
            actual_temporal = str(row.get("temporal_status") or "")
            temporal_correct += actual_temporal == wanted_temporal
            if wanted_temporal == "stale":
                stale_expected += 1
                stale_detected += actual_temporal == "stale"
        if "resolved_temporal_conflicts" in expected:
            supersession_expected_values.append(
                int(expected.get("resolved_temporal_conflicts") or 0)
            )

        for raw_url in expected.get("supporting_urls") or []:
            url = canonicalize_url(str(raw_url))
            if url:
                expected_support.add((index, url))
        for relation in row.get("relations") or []:
            if relation.get("relation") != "supports":
                continue
            url = canonicalize_url(str(relation.get("url") or ""))
            if url:
                predicted_support.add((index, url))

    true_support = len(expected_support & predicted_support)
    support_precision = (
        true_support / len(predicted_support) if predicted_support else 1.0
    )
    support_recall = true_support / len(expected_support) if expected_support else 1.0
    ready_overrides = [
        bool(item.get("publication_ready"))
        for item in expectations
        if "publication_ready" in item
    ]
    expected_ready = (
        ready_overrides[0]
        if ready_overrides
        else bool(expectations)
        and all(str(item.get("status") or "") == "supported" for item in expectations)
    )
    expected_supersessions = (
        supersession_expected_values[0]
        if supersession_expected_values
        else int(matrix.get("resolved_temporal_conflicts") or 0)
    )
    actual_supersessions = int(matrix.get("resolved_temporal_conflicts") or 0)
    return {
        "row_count": len(rows),
        "status_accuracy": round(status_correct / total if total else 1.0, 4),
        "support_url_precision": round(support_precision, 4),
        "support_url_recall": round(support_recall, 4),
        "unsupported_recall": round(
            unsupported_detected / unsupported_expected if unsupported_expected else 1.0,
            4,
        ),
        "conflict_recall": round(
            conflict_detected / conflict_expected if conflict_expected else 1.0,
            4,
        ),
        "false_support_rate": round(
            false_supports / negative_expected if negative_expected else 0.0,
            4,
        ),
        "corroboration_accuracy": round(
            corroboration_correct / corroboration_expected
            if corroboration_expected
            else 1.0,
            4,
        ),
        "false_corroboration_recall": round(
            false_corroboration_detected / false_corroboration_expected
            if false_corroboration_expected
            else 1.0,
            4,
        ),
        "temporal_status_accuracy": round(
            temporal_correct / temporal_expected if temporal_expected else 1.0,
            4,
        ),
        "stale_claim_recall": round(
            stale_detected / stale_expected if stale_expected else 1.0,
            4,
        ),
        "supersession_accuracy": float(
            actual_supersessions == expected_supersessions
        ),
        "unsafe_supersession_rate": float(
            actual_supersessions > expected_supersessions
        ),
        "publication_ready_accuracy": float(
            bool(matrix.get("publication_ready")) == expected_ready
        ),
    }
