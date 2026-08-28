"""Local authority and provenance graph for fetched research evidence."""

from __future__ import annotations

from hashlib import sha1
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from remy.core.retrieval.source_filter import classify
from remy.core.search_gateway import canonicalize_url
from remy.core.source_credibility import credibility_scorer


_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
_ORIGIN_RE = re.compile(
    r"(?:originally\s+published(?:\s+at|\s+by)?|reprinted\s+from|adapted\s+from|"
    r"original\s+source|based\s+on|according\s+to|citing|source|via)"
    r"\s*[:\-]?\s*(https?://[^\s<>\]\[\"']+)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _fields(item: Mapping[str, Any]) -> tuple[str, str, str]:
    packet = item.get("evidence_packet") if isinstance(item.get("evidence_packet"), Mapping) else {}
    url = canonicalize_url(
        str(item.get("url") or item.get("uri") or packet.get("canonical_url") or "")
    )
    title = str(item.get("title") or packet.get("title") or "")
    content = str(item.get("content") or item.get("snippet") or "")
    return url, title, content


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    except ValueError:
        return ""


def _tokens(text: str) -> set[str]:
    return {
        token for token in _TOKEN_RE.findall((text or "")[:50_000].casefold())
        if len(token) >= 4
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _list_urls(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = _URL_RE.findall(value)
    elif isinstance(value, Sequence):
        raw = [str(item) for item in value]
    else:
        raw = []
    return list(
        dict.fromkeys(
            url for url in (canonicalize_url(item.rstrip(".,;)")) for item in raw) if url
        )
    )


def _explicit_origins(item: Mapping[str, Any], content: str) -> list[str]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    values = [
        item.get("original_url"),
        item.get("original_source_url"),
        item.get("canonical_source_url"),
        item.get("syndicated_from"),
        metadata.get("original_url"),
        metadata.get("original_source_url"),
        *_ORIGIN_RE.findall(content or ""),
    ]
    return list(
        dict.fromkeys(
            url
            for url in (canonicalize_url(str(value or "").rstrip(".,;)")) for value in values)
            if url
        )
    )


def _citations(item: Mapping[str, Any], content: str) -> list[str]:
    values = []
    for key in ("outbound_urls", "references", "citations", "links"):
        values.extend(_list_urls(item.get(key)))
    values.extend(_list_urls(content))
    return list(dict.fromkeys(values))[:24]


def _authority_role(
    url: str,
    source_class: str,
    content: str,
    *,
    has_explicit_origin: bool,
) -> str:
    host = _domain(url)
    lowered = content[:4_000].casefold()
    if any(
        phrase in lowered for phrase in ("originally published", "reprinted from", "syndicated from")
    ):
        return "syndicated"
    if has_explicit_origin:
        return "derived"
    if source_class in {"mirror", "seo"}:
        return "aggregator"
    if source_class == "forum":
        return "ugc"
    if (
        source_class in {"official_docs", "research", "github"}
        or host.endswith(".gov")
        or ".gov." in host
        or "press release" in lowered
        or "official announcement" in lowered
    ):
        return "primary"
    if source_class in {"news", "publisher"}:
        return "secondary"
    if any(phrase in lowered for phrase in ("according to", "reported by", "citing ")):
        return "secondary"
    return "unknown"


def _authority_score(url: str, role: str, packet: Mapping[str, Any]) -> float:
    score = float(credibility_scorer.get_score(url))
    score += {
        "primary": 0.18,
        "secondary": 0.05,
        "unknown": 0.0,
        "ugc": -0.15,
        "aggregator": -0.22,
        "syndicated": -0.08,
        "derived": -0.03,
    }.get(role, 0.0)
    if packet.get("has_mismatch"):
        score -= 0.35
    elif packet.get("ok") is True:
        score += 0.05
    return round(max(0.0, min(score, 1.0)), 3)


def build_source_provenance_graph(
    sources: Sequence[Mapping[str, Any]],
    *,
    duplicate_threshold: float = 0.82,
    minimum_content_chars: int = 120,
) -> dict[str, Any]:
    """Build source nodes, derivation edges, and independent evidence roots."""
    nodes: list[dict[str, Any]] = []
    by_url: dict[str, dict[str, Any]] = {}
    for item in sources:
        url, title, content = _fields(item)
        if (
            not url
            or url in by_url
            or len(re.sub(r"\s+", " ", content).strip())
            < max(1, int(minimum_content_chars))
        ):
            continue
        packet = item.get("evidence_packet") if isinstance(item.get("evidence_packet"), Mapping) else {}
        source_class = str(packet.get("source_class") or item.get("source_class") or "")
        if not source_class:
            source_class = classify({"uri": url, "title": title}).source_class
        origins = [origin for origin in _explicit_origins(item, content) if origin != url]
        citations = [citation for citation in _citations(item, content) if citation != url]
        role = _authority_role(
            url, source_class, content, has_explicit_origin=bool(origins)
        )
        node = {
            "node_id": "src-" + sha1(url.encode("utf-8")).hexdigest()[:10],
            "url": url,
            "domain": _domain(url),
            "title": title,
            "source_class": source_class,
            "authority_role": role,
            "authority_score": _authority_score(url, role, packet),
            "explicit_origins": origins,
            "citations": citations,
            "content_fingerprint": sha1(
                " ".join(sorted(_tokens(content))).encode("utf-8")
            ).hexdigest()[:16],
            "_tokens": _tokens(content),
        }
        nodes.append(node)
        by_url[url] = node

    edges: list[dict[str, Any]] = []
    parent: dict[str, str] = {}
    for node in nodes:
        if node["explicit_origins"]:
            origin = node["explicit_origins"][0]
            parent[node["url"]] = origin
            edges.append(
                {
                    "from": node["url"],
                    "to": origin,
                    "relation": "derived_from",
                    "confidence": 1.0,
                }
            )
        for citation in node["citations"]:
            edges.append(
                {
                    "from": node["url"],
                    "to": citation,
                    "relation": "cites",
                    "confidence": 0.8,
                }
            )

    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            similarity = _jaccard(left["_tokens"], right["_tokens"])
            if similarity < duplicate_threshold:
                continue
            if left["authority_score"] >= right["authority_score"]:
                origin, copy_node = left, right
            else:
                origin, copy_node = right, left
            inferred_duplicate_parent = copy_node["url"] not in parent
            if inferred_duplicate_parent:
                parent[copy_node["url"]] = origin["url"]
            if inferred_duplicate_parent and copy_node["authority_role"] not in {
                "syndicated", "aggregator"
            }:
                copy_node["authority_role"] = "syndicated"
                copy_node["authority_score"] = _authority_score(
                    copy_node["url"], "syndicated", {}
                )
            edges.append(
                {
                    "from": copy_node["url"],
                    "to": origin["url"],
                    "relation": "near_duplicate_of",
                    "confidence": round(similarity, 4),
                }
            )

    def root_for(url: str) -> str:
        current = url
        seen = set()
        while current in parent and current not in seen:
            seen.add(current)
            current = parent[current]
        return current

    roots: dict[str, dict[str, Any]] = {}
    for node in nodes:
        root_url = root_for(node["url"])
        node["evidence_root"] = root_url
        node["independent_root"] = root_url == node["url"]
        root = roots.setdefault(
            root_url,
            {
                "root_url": root_url,
                "root_domain": _domain(root_url),
                "member_urls": [],
                "max_authority_score": 0.0,
                "primary": False,
            },
        )
        root["member_urls"].append(node["url"])
        root["max_authority_score"] = max(
            float(root["max_authority_score"]), float(node["authority_score"])
        )
        root["primary"] = bool(root["primary"] or node["authority_role"] == "primary")

    public_nodes = []
    for node in nodes:
        public_nodes.append({key: value for key, value in node.items() if key != "_tokens"})
    return {
        "version": 1,
        "method": "local_source_provenance_graph",
        "node_count": len(public_nodes),
        "edge_count": len(edges),
        "independent_root_count": len(roots),
        "primary_root_count": sum(bool(root["primary"]) for root in roots.values()),
        "syndicated_source_count": sum(
            node["authority_role"] == "syndicated" for node in public_nodes
        ),
        "derived_source_count": sum(
            node["authority_role"] == "derived" for node in public_nodes
        ),
        "aggregator_source_count": sum(
            node["authority_role"] == "aggregator" for node in public_nodes
        ),
        "authority_weighted_roots": round(
            sum(float(root["max_authority_score"]) for root in roots.values()), 3
        ),
        "nodes": public_nodes,
        "edges": edges,
        "roots": sorted(roots.values(), key=lambda root: root["root_url"]),
    }


def apply_provenance_to_marginal_evidence(
    marginal: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    minimum_roots: int = 3,
) -> dict[str, Any]:
    """Require distinct provenance roots for evidence consensus."""
    updated = dict(marginal)
    accepted = set(str(url) for url in marginal.get("accepted_urls") or [])
    node_roots = {
        str(node.get("url") or ""): str(node.get("evidence_root") or "")
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping)
    }
    accepted_roots = sorted(
        {node_roots[url] for url in accepted if node_roots.get(url)}
    )
    root_target = max(1, int(minimum_roots))
    provenance_sufficient = len(accepted_roots) >= root_target
    updated["provenance"] = {
        "accepted_root_count": len(accepted_roots),
        "root_target": root_target,
        "accepted_roots": accepted_roots,
        "primary_root_count": int(graph.get("primary_root_count") or 0),
        "authority_weighted_roots": float(graph.get("authority_weighted_roots") or 0.0),
    }
    if updated.get("sufficient") and not provenance_sufficient:
        updated["sufficient"] = False
        updated["stop_recommended"] = False
        updated["decision"] = "continue_provenance_diversify"
        updated["saturated"] = False
        reasons = list(updated.get("reasons") or [])
        reasons.append("not_enough_independent_evidence_roots")
        updated["reasons"] = list(dict.fromkeys(reasons))
        repair = list(updated.get("repair_queries") or [])
        repair.append("independent primary source not derived from existing evidence roots")
        updated["repair_queries"] = list(dict.fromkeys(repair))[:8]
    return updated
