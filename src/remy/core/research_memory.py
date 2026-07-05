"""
Research memory — reuse prior findings and track contradictions.

Builds on existing brain storage (research-finding, research-project tags)
to provide:
- Prior findings retrieval by topic for update-existing-analysis
- Contradiction index with resolution status
- Compact prompt context for research worker
- Metrics signals for research reuse tracking

Does NOT duplicate brain storage — queries and indexes existing records.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from remy.config.settings import settings

logger = logging.getLogger("ResearchMemory")

_LOCK = threading.Lock()
_MAX_CONTRADICTIONS = 200
_CONTRADICTION_FILE = "research_contradictions.json"


# ============== Contradiction Index ==============


def _contradiction_path() -> Path:
    return settings.DATA_DIR / _CONTRADICTION_FILE


def _load_contradictions() -> list[dict]:
    path = _contradiction_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_contradictions(records: list[dict]) -> None:
    path = _contradiction_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records[:_MAX_CONTRADICTIONS], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def record_contradiction(
    *,
    topic: str,
    finding_a: str,
    finding_b: str,
    source_a: str = "",
    source_b: str = "",
    finding_id_a: str = "",
    finding_id_b: str = "",
    project_id: str = "",
) -> None:
    """Record a contradiction between two findings on the same topic."""
    now = datetime.now().isoformat()
    key = tuple(sorted([finding_id_a, finding_id_b])) if finding_id_a and finding_id_b else None

    with _LOCK:
        records = _load_contradictions()

        # Dedup by finding IDs
        if key:
            for rec in records:
                existing_key = tuple(
                    sorted([rec.get("finding_id_a", ""), rec.get("finding_id_b", "")])
                )
                if existing_key == key:
                    rec["count"] = int(rec.get("count", 0)) + 1
                    rec["last_seen"] = now
                    _save_contradictions(
                        sorted(records, key=lambda r: r.get("last_seen", ""), reverse=True)
                    )
                    return

        records.append(
            {
                "topic": topic,
                "finding_a": finding_a[:300],
                "finding_b": finding_b[:300],
                "source_a": source_a,
                "source_b": source_b,
                "finding_id_a": finding_id_a,
                "finding_id_b": finding_id_b,
                "project_id": project_id,
                "status": "unresolved",
                "count": 1,
                "first_seen": now,
                "last_seen": now,
            }
        )
        _save_contradictions(sorted(records, key=lambda r: r.get("last_seen", ""), reverse=True))


def get_contradictions(topic: str = "", limit: int = 10) -> list[dict]:
    """Get contradictions, optionally filtered by topic."""
    with _LOCK:
        records = _load_contradictions()

    if topic:
        topic_lower = topic.lower()
        records = [r for r in records if topic_lower in r.get("topic", "").lower()]

    return records[:limit]


def get_unresolved_contradictions(topic: str = "", limit: int = 5) -> list[dict]:
    """Get only unresolved contradictions for a topic."""
    all_c = get_contradictions(topic=topic, limit=50)
    return [c for c in all_c if c.get("status") == "unresolved"][:limit]


def resolve_contradiction(
    finding_id_a: str, finding_id_b: str, resolution: str = "resolved"
) -> bool:
    """Mark a contradiction as resolved."""
    key = tuple(sorted([finding_id_a, finding_id_b]))
    with _LOCK:
        records = _load_contradictions()
        for rec in records:
            existing_key = tuple(sorted([rec.get("finding_id_a", ""), rec.get("finding_id_b", "")]))
            if existing_key == key:
                rec["status"] = resolution
                rec["resolved_at"] = datetime.now().isoformat()
                _save_contradictions(records)
                return True
    return False


# ============== Prior Findings Retrieval ==============


def get_prior_findings(topic: str, limit: int = 10) -> list[dict]:
    """Retrieve existing research findings from brain by topic.

    Returns compact dicts with content, source, confidence, timestamp.
    """
    try:
        from remy.core.tool_handlers.research import (
            _RESEARCH_FINDING_TAG,
            _get_brain,
            _get_brain_lock,
        )

        brain = _get_brain()
        brain_lock = _get_brain_lock()

        with brain_lock:
            records = brain.search(query=topic, tags=[_RESEARCH_FINDING_TAG], limit=limit * 2)

        findings = []
        for rec in records[:limit]:
            meta = rec.metadata or {}
            findings.append(
                {
                    "id": rec.id,
                    "content": rec.content[:300],
                    "source_url": meta.get("source_url", ""),
                    "confidence": meta.get("confidence", 0.5),
                    "project_id": meta.get("project_id", ""),
                    "timestamp": meta.get("timestamp", ""),
                }
            )
        return findings
    except Exception as e:
        logger.debug("Prior findings retrieval failed: %s", e)
        return []


def get_completed_reports(topic: str, limit: int = 3) -> list[dict]:
    """Retrieve completed research reports related to a topic."""
    try:
        from remy.core.tool_handlers.research import _get_brain, _get_brain_lock

        brain = _get_brain()
        brain_lock = _get_brain_lock()

        with brain_lock:
            records = brain.search(query=topic, tags=["research"], limit=limit * 2)

        reports = []
        for rec in records:
            meta = rec.metadata or {}
            if meta.get("type") != "research_report":
                continue
            reports.append(
                {
                    "id": rec.id,
                    "topic": meta.get("topic", ""),
                    "content": rec.content[:500],
                    "sources": meta.get("sources", []),
                    "findings_count": meta.get("findings_count", 0),
                    "confidence_avg": meta.get("confidence_avg", 0.0),
                    "project_id": meta.get("project_id", ""),
                    "timestamp": meta.get("timestamp", ""),
                }
            )
            if len(reports) >= limit:
                break
        return reports
    except Exception as e:
        logger.debug("Completed reports retrieval failed: %s", e)
        return []


def extract_known_companies(findings: list[dict]) -> list[str]:
    """Extract company/product names mentioned across findings.

    Simple heuristic: looks for capitalized multi-word phrases and known patterns.
    """
    companies = set()
    for f in findings:
        content = f.get("content", "")
        # Look for patterns like "Company Name" or "ProductName"
        import re

        # Match capitalized words that are likely company/product names
        matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content)
        for m in matches:
            # Filter out common non-company words
            if (
                m.lower()
                not in (
                    "research",
                    "finding",
                    "source",
                    "the",
                    "this",
                    "that",
                    "note",
                    "important",
                    "however",
                    "according",
                    "based",
                    "analysis",
                    "report",
                    "study",
                    "data",
                    "results",
                    "market",
                    "industry",
                    "sector",
                    "company",
                    "product",
                    "service",
                    "platform",
                    "solution",
                    "technology",
                )
                and len(m) > 2
            ):
                companies.add(m)
    return sorted(companies)[:20]


# ============== Prompt Context Formatting ==============


def format_existing_research_for_prompt(topic: str) -> str:
    """Build compact research context for the research worker prompt.

    Shows: prior findings, known companies, contradictions, stale areas.
    """
    findings = get_prior_findings(topic, limit=8)
    reports = get_completed_reports(topic, limit=2)
    contradictions = get_unresolved_contradictions(topic, limit=3)

    if not findings and not reports and not contradictions:
        return ""

    lines = ["\nEXISTING RESEARCH (from prior sessions):"]

    # Prior reports
    if reports:
        for r in reports:
            lines.append(
                f'- PRIOR REPORT: "{r["topic"]}" ({r["findings_count"]} findings, avg confidence {r["confidence_avg"]})'
            )

    # Known findings summary
    if findings:
        lines.append(f"- {len(findings)} prior findings on this topic:")
        for f in findings[:5]:
            conf = f.get("confidence", 0.5)
            src = f.get("source_url", "")
            snippet = f["content"][:120].replace("\n", " ")
            line = f"  * [{conf:.1f}] {snippet}"
            if src:
                line += f" ({src})"
            lines.append(line)
        if len(findings) > 5:
            lines.append(f"  * ... and {len(findings) - 5} more findings")

    # Known companies
    companies = extract_known_companies(findings)
    if companies:
        lines.append(f"- Known entities: {', '.join(companies[:10])}")

    # Unresolved contradictions
    if contradictions:
        lines.append("- UNRESOLVED CONTRADICTIONS:")
        for c in contradictions:
            lines.append(f'  * "{c["finding_a"][:80]}" vs "{c["finding_b"][:80]}"')

    lines.append(
        "- Build on existing findings. Do not duplicate. Resolve contradictions if possible."
    )
    return "\n".join(lines) + "\n"


# ============== Contradiction Detection ==============


def check_finding_contradictions(
    new_content: str,
    new_source: str,
    new_finding_id: str,
    topic: str,
    project_id: str = "",
) -> list[dict]:
    """Check if a new finding contradicts existing findings on the same topic.

    Uses simple heuristic: looks for negation patterns and opposing claims.
    Returns list of potential contradictions found.
    """
    findings = get_prior_findings(topic, limit=20)
    if not findings:
        return []

    new_lower = new_content.lower()
    contradictions_found = []

    # Negation markers that suggest contradiction
    negation_markers = (
        "not ",
        "no longer",
        "incorrect",
        "wrong",
        "false",
        "contrary to",
        "unlike",
        "however",
        "but ",
        "instead",
        "declined",
        "decreased",
        "dropped",
        "failed",
    )
    # Positive markers
    positive_markers = (
        "increased",
        "grew",
        "improved",
        "succeeded",
        "leading",
        "dominant",
        "top ",
        "best ",
        "largest",
    )

    for f in findings:
        if f["id"] == new_finding_id:
            continue

        existing_lower = f["content"].lower()

        # Check for opposing sentiment on same entities
        score = 0
        for marker in negation_markers:
            if marker in new_lower and marker not in existing_lower:
                score += 1
            elif marker in existing_lower and marker not in new_lower:
                score += 1
        for marker in positive_markers:
            if marker in new_lower and marker not in existing_lower:
                score += 0.5
            elif marker in existing_lower and marker not in new_lower:
                score += 0.5

        if score >= 2:
            contradictions_found.append(
                {
                    "finding_id": f["id"],
                    "content": f["content"][:200],
                    "source_url": f.get("source_url", ""),
                    "score": score,
                }
            )

    # Record any found contradictions
    for c in contradictions_found[:3]:
        record_contradiction(
            topic=topic,
            finding_a=new_content[:300],
            finding_b=c["content"][:300],
            source_a=new_source,
            source_b=c.get("source_url", ""),
            finding_id_a=new_finding_id,
            finding_id_b=c["finding_id"],
            project_id=project_id,
        )

    return contradictions_found
