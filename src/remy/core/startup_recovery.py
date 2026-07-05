from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from remy.core.agent_tools import Level, brain, brain_lock


def normalize_text(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"^\[\d{4}-\d{2}-\d{2} [^\]]+\]\s*", "", value)
    return value.lower()


def parse_timestamp(text: str) -> str:
    match = re.match(r"^\[(\d{4}-\d{2}-\d{2} [^\]]+)\]", str(text or "").strip())
    return match.group(1) if match else ""


def classify_record(text: str) -> tuple[object, str, str]:
    value = str(text or "").strip()
    lower = value.lower()

    if (
        lower.startswith("user profile:")
        or lower.startswith("[") and "user profile:" in lower
        or "brother, born" in lower
        or "mother, born" in lower
        or "grandmother" in lower
        or "family:" in lower
        or "birth date:" in lower
        or "you are remy" in lower
    ):
        return Level.IDENTITY, "identity", "preference"

    if (
        "goal [" in lower
        or lower.startswith("goal ")
        or "task [" in lower
        or lower.startswith("task ")
        or lower.startswith("scheduled:")
        or lower.startswith("plan:")
        or lower.startswith("action outcome")
        or lower.startswith("background insights")
        or lower.startswith("research project:")
        or lower.startswith("research finding")
    ):
        return Level.DECISIONS, "decision", "decision"

    if lower.startswith("research:"):
        return Level.DOMAIN, "research_report", "fact"
    if lower.startswith("web search:"):
        return Level.DOMAIN, "web_search", "fact"

    return Level.DOMAIN, "domain_fact", "fact"


def infer_tags(text: str, kind: str, *, include_recovery_tags: bool = False) -> list[str]:
    value = str(text or "")
    lower = value.lower()
    tags: set[str] = set()

    if include_recovery_tags:
        tags.update({"recovered", "startup-backup-recovery"})

    if "aurasdk" in lower or "aura-memory" in lower or "aura sdk" in lower:
        tags.add("aurasdk")
    if "mcp" in lower:
        tags.add("mcp")
    if any(token in lower for token in ["grant", "apollo", "0g", "nlnet", "solana", "near", "doe"]):
        tags.add("grants")
    if any(token in lower for token in ["contact", "linkedin", "github", "twitter", "x profile", "email", "lead"]):
        tags.add("contact")
    if any(token in lower for token in ["strategy", "roadmap", "mvp", "milestone", "execution plan"]):
        tags.add("strategy")
    if any(token in lower for token in ["revenue", "funding", "price", "pricing", "financial", "monetization"]):
        tags.add("financial")
    if "v10" in lower:
        tags.add("v10")
    if "v9" in lower:
        tags.add("v9")
    if "proto-self" in lower:
        tags.add("proto-self")
    if "conflict resolution" in lower:
        tags.add("conflict-resolution")
    if "telegram" in lower:
        tags.add("telegram")
    if "profile" in lower or kind == "identity":
        tags.add("profile")
    if "task [" in lower or lower.startswith("scheduled:") or lower.startswith("todo "):
        tags.add("task")
    if "goal [" in lower or lower.startswith("goal "):
        tags.add("goal")
    if lower.startswith("research:") or lower.startswith("research project:"):
        tags.add("research")
    if lower.startswith("web search:"):
        tags.add("web-search")

    return sorted(tags)


def _load_backup_payload(backup_path: Path) -> list[str]:
    payload = json.loads(backup_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {backup_path}")
    return [str(item) for item in payload]


def inspect_backup_recovery(backup_path: Path) -> dict:
    payload = _load_backup_payload(backup_path)
    with brain_lock:
        active_records = brain.search(query="", limit=5000) or []

    active_norm = {normalize_text(getattr(record, "content", "")) for record in active_records}
    missing = [item for item in payload if normalize_text(item) and normalize_text(item) not in active_norm]
    would_import = Counter()
    for item in missing:
        level, _kind, _semantic_type = classify_record(item)
        would_import[str(level)] += 1
    return {
        "backup_path": str(backup_path),
        "backup_records": len(payload),
        "active_records_before": len(active_records),
        "missing_records": len(missing),
        "would_import_by_level": dict(would_import),
        "sample_missing": missing[:8],
    }


def apply_backup_recovery(backup_path: Path) -> dict:
    payload = _load_backup_payload(backup_path)
    with brain_lock:
        active_records = brain.search(query="", limit=5000) or []

    active_norm = {normalize_text(getattr(record, "content", "")) for record in active_records}
    missing = [item for item in payload if normalize_text(item) and normalize_text(item) not in active_norm]
    summary = {
        "backup_path": str(backup_path),
        "backup_records": len(payload),
        "active_records_before": len(active_records),
        "missing_records": len(missing),
        "would_import_by_level": {},
        "imported_by_level": {},
        "sample_missing": missing[:8],
        "imported_count": 0,
    }
    if not missing:
        return summary

    would_import = Counter()
    imported = Counter()
    imported_ids: list[str] = []

    for item in missing:
        level, _kind, _semantic_type = classify_record(item)
        would_import[str(level)] += 1

    with brain_lock:
        for index, item in enumerate(missing, start=1):
            level, kind, semantic_type = classify_record(item)
            tags = infer_tags(item, kind)
            metadata = {
                "source": "startup_backup_recovery",
                "source_type": "inferred",
                "semantic_type": semantic_type,
                "recovery_kind": kind,
                "recovery_backup_path": str(backup_path),
                "recovery_backup_index": index,
                "recovered_at": parse_timestamp(item),
                "verified": False,
            }
            record = brain.store(
                content=item,
                level=level,
                tags=tags,
                metadata=metadata,
                source_type="inferred",
                semantic_type=semantic_type,
                deduplicate=False,
            )
            imported_ids.append(str(getattr(record, "id", "") or ""))
            imported[str(level)] += 1

        artifact = brain.store(
            content="\n".join(
                [
                    "Startup Backup Recovery Summary",
                    "",
                    f"Backup path: {backup_path}",
                    f"Recovered records: {len(imported_ids)}",
                    f"Identity: {imported.get(str(Level.IDENTITY), 0)}",
                    f"Decisions: {imported.get(str(Level.DECISIONS), 0)}",
                    f"Domain: {imported.get(str(Level.DOMAIN), 0)}",
                ]
            ).strip(),
            level=Level.DECISIONS,
            tags=["operator", "incident_snapshot", "review", "startup_backup_recovery"],
            metadata={
                "type": "startup_backup_recovery",
                "source": "startup",
                "backup_path": str(backup_path),
                "imported_record_ids": imported_ids[:200],
                "imported_count": len(imported_ids),
            },
            source_type="inferred",
            semantic_type="decision",
            deduplicate=False,
        )

    summary["recovery_artifact_id"] = str(getattr(artifact, "id", "") or "")
    summary["active_records_after"] = summary["active_records_before"] + len(imported_ids)
    summary["imported_count"] = len(imported_ids)
    summary["would_import_by_level"] = dict(would_import)
    summary["imported_by_level"] = dict(imported)
    return summary


def reconcile_recovered_records(*, apply: bool = False) -> dict:
    with brain_lock:
        records = brain.search(query="", limit=5000) or []

    recovered = [
        record
        for record in records
        if ((getattr(record, "metadata", None) or {}).get("source") == "startup_backup_recovery")
    ]

    changes = []
    for record in recovered:
        content = getattr(record, "content", "")
        expected_level, kind, semantic_type = classify_record(content)
        actual_level = str(getattr(record, "level", "") or "")
        metadata = dict(getattr(record, "metadata", None) or {})
        actual_semantic_type = str(metadata.get("semantic_type") or "")
        desired_tags = infer_tags(content, kind)
        actual_tags = sorted(list(getattr(record, "tags", []) or []))

        if (
            actual_level != str(expected_level)
            or actual_semantic_type != semantic_type
            or actual_tags != desired_tags
        ):
            changes.append(
                {
                    "id": getattr(record, "id", ""),
                    "content_preview": str(content)[:160],
                    "actual_level": actual_level,
                    "expected_level": str(expected_level),
                    "actual_semantic_type": actual_semantic_type,
                    "expected_semantic_type": semantic_type,
                    "actual_tags": actual_tags,
                    "expected_tags": desired_tags,
                }
            )

    if apply and changes:
        with brain_lock:
            for change in changes:
                record = brain.get(change["id"])
                if not record:
                    continue
                content = getattr(record, "content", "")
                expected_level, kind, semantic_type = classify_record(content)
                metadata = dict(getattr(record, "metadata", None) or {})
                metadata["semantic_type"] = semantic_type
                metadata["recovery_kind"] = kind
                metadata["recovery_reconciled"] = True
                brain.update(
                    change["id"],
                    level=expected_level,
                    tags=infer_tags(content, kind),
                    metadata=metadata,
                    source_type="inferred",
                )

    return {
        "recovered_records": len(recovered),
        "changes_needed": len(changes),
        "applied": bool(apply),
        "changes": changes[:100],
    }


def cleanup_recovered_records(*, apply: bool = False) -> dict:
    with brain_lock:
        records = brain.search(query="", limit=5000) or []

    recovered = [
        record
        for record in records
        if ((getattr(record, "metadata", None) or {}).get("source") == "startup_backup_recovery")
    ]

    changes = []
    for record in recovered:
        actual_tags = list(getattr(record, "tags", []) or [])
        cleaned_tags = [tag for tag in actual_tags if tag not in {"recovered", "startup-backup-recovery"}]
        if cleaned_tags != actual_tags:
            changes.append(
                {
                    "id": getattr(record, "id", ""),
                    "content_preview": str(getattr(record, "content", "") or "")[:160],
                    "actual_tags": actual_tags,
                    "cleaned_tags": cleaned_tags,
                }
            )

    if apply and changes:
        with brain_lock:
            for change in changes:
                record = brain.get(change["id"])
                if not record:
                    continue
                metadata = dict(getattr(record, "metadata", None) or {})
                metadata["recovery_cleanup_applied"] = True
                brain.update(
                    change["id"],
                    tags=change["cleaned_tags"],
                    metadata=metadata,
                    source_type=str(metadata.get("source_type") or "inferred"),
                )

    return {
        "recovered_records": len(recovered),
        "cleanup_candidates": len(changes),
        "applied": bool(apply),
        "changes": changes[:100],
    }
