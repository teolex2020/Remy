"""Persisted research sessions for market_research and monitoring packs."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from remy.config.settings import settings
from remy.core.file_utils import atomic_write

SESSIONS_DIR = settings.DATA_DIR / "research_sessions"


@dataclass
class ResearchSession:
    session_id: str
    goal_id: str
    pack_id: str
    topic: str
    research_mode: str
    source_scope: str
    source_domains: list[str] = field(default_factory=list)
    citation_required: bool = True
    generated_queries: list[str] = field(default_factory=list)
    fetched_sources: list[dict] = field(default_factory=list)
    accepted_sources: list[dict] = field(default_factory=list)
    rejected_sources: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    final_artifact_id: str = ""
    status: str = "active"
    warnings: list[str] = field(default_factory=list)
    resumed_runs: int = 0
    # findings_per_run[i] = number of new findings added in run i (for saturation detection)
    findings_per_run: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def summary(self) -> dict:
        accepted = len(self.accepted_sources)
        rejected = len(self.rejected_sources)
        findings = len(self.findings)
        contradictions = len(self.contradictions)
        coverage = accepted / findings if findings else 0.0
        return {
            "session_id": self.session_id,
            "status": self.status,
            "pack_id": self.pack_id,
            "topic": self.topic,
            "research_mode": self.research_mode,
            "source_scope": self.source_scope,
            "source_domains": list(self.source_domains),
            "citation_required": self.citation_required,
            "generated_queries_count": len(self.generated_queries),
            "fetched_sources_count": len(self.fetched_sources),
            "accepted_sources_count": accepted,
            "rejected_sources_count": rejected,
            "findings_count": findings,
            "contradictions_count": contradictions,
            "citation_coverage_rate": round(coverage, 3),
            "final_artifact_id": self.final_artifact_id,
            "warnings": list(self.warnings),
            "resumed_runs": self.resumed_runs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _session_path(goal_id: str) -> str:
    return str(SESSIONS_DIR / f"{goal_id}.json")


def load_research_session(goal_id: str) -> ResearchSession | None:
    path = SESSIONS_DIR / f"{goal_id}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Forward-compat: strip unknown keys so old sessions load cleanly
        known = {f.name for f in ResearchSession.__dataclass_fields__.values()}
        raw = {k: v for k, v in raw.items() if k in known}
        return ResearchSession(**raw)
    except Exception:
        return None


def save_research_session(session: ResearchSession) -> ResearchSession:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session.updated_at = time.time()
    atomic_write(SESSIONS_DIR / f"{session.goal_id}.json", json.dumps(asdict(session), indent=2))
    return session


def get_or_create_research_session(
    goal_id: str,
    pack_id: str,
    topic: str,
    research_mode: str,
    source_scope: str,
    source_domains: list[str] | None = None,
    citation_required: bool = True,
    warnings: list[str] | None = None,
) -> tuple[ResearchSession, bool]:
    existing = load_research_session(goal_id)
    if existing:
        existing.pack_id = pack_id or existing.pack_id
        existing.topic = topic or existing.topic
        existing.research_mode = research_mode or existing.research_mode
        existing.source_scope = source_scope or existing.source_scope
        existing.source_domains = list(source_domains or existing.source_domains)
        existing.citation_required = citation_required
        if warnings:
            for warning in warnings:
                if warning not in existing.warnings:
                    existing.warnings.append(warning)
        existing.resumed_runs += 1
        # Append 0 placeholder — will be updated by research_worker after findings counted
        existing.findings_per_run.append(0)
        return save_research_session(existing), True

    session = ResearchSession(
        session_id=f"rs-{uuid.uuid4().hex[:12]}",
        goal_id=goal_id,
        pack_id=pack_id,
        topic=topic,
        research_mode=research_mode,
        source_scope=source_scope,
        source_domains=list(source_domains or []),
        citation_required=citation_required,
        warnings=list(warnings or []),
    )
    return save_research_session(session), False


def append_queries(goal_id: str, queries: list[str]) -> ResearchSession | None:
    session = load_research_session(goal_id)
    if not session:
        return None
    for query in queries:
        if query and query not in session.generated_queries:
            session.generated_queries.append(query)
    return save_research_session(session)


def record_source_fetch(goal_id: str, source: dict) -> ResearchSession | None:
    session = load_research_session(goal_id)
    if not session:
        return None
    url = str(source.get("url", "") or "")
    if url and not any(item.get("url") == url for item in session.fetched_sources):
        session.fetched_sources.append(source)
    return save_research_session(session)


def record_source_decision(
    goal_id: str, source: dict, accepted: bool, reason: str = ""
) -> ResearchSession | None:
    session = load_research_session(goal_id)
    if not session:
        return None
    payload = dict(source or {})
    if reason:
        payload["reason"] = reason
    url = str(payload.get("url", "") or "")
    bucket = session.accepted_sources if accepted else session.rejected_sources
    if url and not any(item.get("url") == url for item in bucket):
        bucket.append(payload)
    return save_research_session(session)


def record_finding(goal_id: str, finding: dict) -> ResearchSession | None:
    session = load_research_session(goal_id)
    if not session:
        return None
    session.findings.append(finding)
    return save_research_session(session)


def _infer_urls_from_text(text: str) -> list[str]:
    raw = text or ""
    urls = re.findall(r"https?://[^\s)]+", raw)
    handles = re.findall(r"@([A-Za-z0-9_]{2,32})", raw)
    inferred = [f"https://x.com/{handle}" for handle in handles]
    return list(dict.fromkeys([*(u.strip() for u in urls if u.strip()), *inferred]))


def record_run_findings_count(goal_id: str, count: int) -> None:
    """Update the findings count for the most recent run (used by saturation detection)."""
    session = load_research_session(goal_id)
    if not session:
        return
    if session.findings_per_run:
        session.findings_per_run[-1] = count
    else:
        session.findings_per_run.append(count)
    save_research_session(session)


def record_contradictions(goal_id: str, contradictions: list[dict]) -> ResearchSession | None:
    session = load_research_session(goal_id)
    if not session:
        return None
    for contradiction in contradictions:
        cid = contradiction.get("id") or json.dumps(contradiction, sort_keys=True)
        if not any(
            (item.get("id") or json.dumps(item, sort_keys=True)) == cid
            for item in session.contradictions
        ):
            session.contradictions.append(contradiction)
    return save_research_session(session)


def mark_session_completed(goal_id: str, final_artifact_id: str = "") -> ResearchSession | None:
    session = load_research_session(goal_id)
    if not session:
        return None
    session.status = "completed"
    if final_artifact_id:
        session.final_artifact_id = final_artifact_id
    return save_research_session(session)


def get_research_session_summary(goal_id: str) -> dict | None:
    session = load_research_session(goal_id)
    return session.summary() if session else None


def _session_source_domains(session: ResearchSession) -> list[dict]:
    counts: dict[str, int] = {}
    for bucket in (session.accepted_sources, session.fetched_sources):
        for item in bucket:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "") or "").strip()
            if not url:
                continue
            try:
                host = (urlparse(url).netloc or "").lower().strip()
            except Exception:
                host = ""
            if not host:
                continue
            counts[host] = counts.get(host, 0) + 1
    return [
        {"domain": domain, "count": count}
        for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _derive_research_gaps(session: ResearchSession) -> list[str]:
    summary = session.summary()
    gaps: list[str] = []
    if summary["generated_queries_count"] == 0:
        gaps.append("Generate the first focused search query.")
    if summary["accepted_sources_count"] == 0 and summary["fetched_sources_count"] > 0:
        gaps.append("Promote fetched sources into accepted evidence.")
    if summary["accepted_sources_count"] > summary["findings_count"]:
        gaps.append("Synthesize accepted sources into explicit findings.")
    if summary["citation_required"] and summary["findings_count"] and summary["citation_coverage_rate"] < 0.8:
        gaps.append("Improve citation coverage for current findings.")
    if summary["contradictions_count"] > 0:
        gaps.append("Resolve contradictory findings before closing the session.")
    for warning in session.warnings:
        warning_text = str(warning or "").strip()
        if warning_text:
            gaps.append(warning_text)
    return list(dict.fromkeys(gaps))[:4]


def _load_artifact_info(record_id: str) -> dict:
    target_id = str(record_id or "").strip()
    if not target_id:
        return {}
    try:
        from remy.core.agent_tools import brain, brain_lock

        with brain_lock:
            rec = brain.get(target_id)
        if not rec:
            return {}
        metadata = rec.metadata or {}
        markdown_body = str(metadata.get("markdown_body", "") or "")
        return {
            "record_id": rec.id,
            "artifact_format": metadata.get("artifact_format") or metadata.get("type") or "",
            "viewer_url": f"/api/autonomy/research-artifacts/{rec.id}/view",
            "markdown_url": f"/api/autonomy/research-artifacts/{rec.id}/markdown",
            "pdf_url": str(metadata.get("pdf_url", "") or ""),
            "pdf_filename": str(metadata.get("pdf_filename", "") or ""),
            "pdf_record_id": str(metadata.get("pdf_record_id", "") or ""),
            "markdown_available": bool(markdown_body),
            "markdown_preview": markdown_body[:280] if markdown_body else "",
        }
    except Exception:
        return {}


def get_research_session_trace(
    goal_id: str,
    *,
    query_limit: int = 3,
    source_limit: int = 3,
    warning_limit: int = 3,
) -> dict | None:
    session = load_research_session(goal_id)
    if not session:
        return None

    summary = session.summary()
    top_domains = _session_source_domains(session)
    accepted_preview = []
    for item in session.accepted_sources[:source_limit]:
        if not isinstance(item, dict):
            continue
        accepted_preview.append(
            {
                "title": str(item.get("title", "") or "")[:120],
                "url": str(item.get("url", "") or ""),
                "reason": str(item.get("reason", "") or "")[:120],
            }
        )

    return {
        **summary,
        "artifact": _load_artifact_info(session.final_artifact_id),
        "recent_queries": list(session.generated_queries[-query_limit:]),
        "top_source_domains": top_domains[:source_limit],
        "accepted_source_preview": accepted_preview,
        "knowledge_gaps": _derive_research_gaps(session),
        "warnings": list(session.warnings[:warning_limit]),
    }


def reconcile_session_sources(goal_id: str) -> ResearchSession | None:
    session = load_research_session(goal_id)
    if not session:
        return None

    fetched_by_url = {
        str(item.get("url", "") or ""): item
        for item in session.fetched_sources
        if str(item.get("url", "") or "")
    }
    accepted_by_url = {
        str(item.get("url", "") or ""): item
        for item in session.accepted_sources
        if str(item.get("url", "") or "")
    }

    changed = False
    for finding in session.findings:
        if not isinstance(finding, dict):
            continue
        summary = str(finding.get("summary", "") or finding.get("text", "") or "")
        source_url = str(finding.get("source_url", "") or "")
        urls = _infer_urls_from_text(summary)
        if not source_url and urls:
            source_url = urls[0]
            finding["source_url"] = source_url
            changed = True
        for url in urls:
            if url and url not in fetched_by_url:
                fetched_by_url[url] = {"url": url, "title": "", "snippet": summary[:180]}
                changed = True
            if url and url not in accepted_by_url:
                accepted_by_url[url] = {
                    "url": url,
                    "title": "",
                    "snippet": summary[:180],
                    "reason": "inferred_from_finding",
                }
                changed = True
        if source_url and source_url not in fetched_by_url:
            fetched_by_url[source_url] = {"url": source_url, "title": "", "snippet": summary[:180]}
            changed = True
        if source_url and source_url not in accepted_by_url:
            accepted_by_url[source_url] = {
                "url": source_url,
                "title": "",
                "snippet": summary[:180],
                "reason": "inferred_from_finding",
            }
            changed = True

    if changed:
        session.fetched_sources = list(fetched_by_url.values())
        session.accepted_sources = list(accepted_by_url.values())
        return save_research_session(session)
    return session
