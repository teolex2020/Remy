"""Response auditor — orchestrates detect → resolve → act.

Public entry point: audit_response(text, turn, *, mode).

Modes:
  "off"  — skip entirely (returns empty report)
  "warn" — detect, resolve, log violations; return report with rewritten_text=None
  "block"— detect, resolve, and rewrite the response for safe claim types

Violations are appended to data/epistemic_violations.jsonl for later analysis.
This file is NOT brain-memory. It's an operational audit log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from remy.core.retrieval.claim_spans import (
    AuditAction,
    AuditReport,
    ClaimSpan,
    EvidenceMatch,
)
from remy.core.retrieval.detectors import detect_all
from remy.core.retrieval.evidence_resolver import (
    TurnContext,
    find_supporting_evidence,
)


logger = logging.getLogger("ResponseAuditor")

AuditorMode = Literal["off", "warn", "block"]


def _default_log_path() -> Path:
    # <app>/data/epistemic_violations.jsonl
    root = Path(__file__).resolve().parents[4]
    return root / "data" / "epistemic_violations.jsonl"


_LOG_PATH = Path(os.environ.get("EPISTEMIC_AUDIT_LOG", _default_log_path()))


def _current_mode() -> AuditorMode:
    env = (os.environ.get("RESPONSE_AUDITOR_MODE") or "warn").lower()
    if env in ("off", "warn", "block"):
        return env  # type: ignore[return-value]
    return "warn"


def audit_response(
    text: str,
    turn: TurnContext | None = None,
    *,
    mode: AuditorMode | None = None,
    log_path: Path | None = None,
    turn_id: str | None = None,
) -> AuditReport:
    """Scan *text*, resolve each claim against *turn*, return an AuditReport."""
    effective_mode: AuditorMode = mode or _current_mode()
    if effective_mode == "off" or not text or not text.strip():
        return AuditReport(response_text=text, actions=[])

    turn = turn or TurnContext()
    spans = detect_all(text)
    actions: list[AuditAction] = []

    for span in spans:
        evidence = find_supporting_evidence(turn, span)
        actions.append(_decide(span, evidence, effective_mode))

    report = AuditReport(response_text=text, actions=actions)

    if effective_mode == "block" and report.has_violations:
        report.rewritten_text = _apply_rewrites(text, actions)

    if report.has_violations:
        try:
            _log_violations(report, turn, turn_id, log_path or _LOG_PATH)
        except Exception as e:  # logging must never break the response path
            logger.debug("audit log failed: %s", e)

    return report


# ============== decision policy ==============


def _decide(
    span: ClaimSpan, evidence: EvidenceMatch | None, mode: AuditorMode
) -> AuditAction:
    if evidence is not None:
        return AuditAction(mode="pass", claim=span, evidence=evidence)

    if mode == "warn":
        return AuditAction(
            mode="warn", claim=span, reason=f"no evidence for {span.claim_type}"
        )

    # Block mode: choose rewrite strategy per claim_type.
    if span.claim_type in ("arxiv_id", "doi", "url_authoritative"):
        return AuditAction(
            mode="redact", claim=span,
            reason=f"unverified {span.claim_type}",
            rewrite="[потребує перевірки]",
        )
    if span.claim_type == "live_metric":
        return AuditAction(
            mode="redact", claim=span,
            reason="fabricated internal state metric",
            rewrite="[метрика недоступна]",
        )
    if span.claim_type in ("record_count", "belief_count"):
        return AuditAction(
            mode="redact", claim=span,
            reason="count not backed by fresh tool call",
            rewrite="[число недоступне]",
        )
    if span.claim_type == "entitlement":
        return AuditAction(
            mode="downgrade", claim=span,
            reason="no tool call backing discovery claim",
            rewrite=_downgrade_entitlement(span.text),
        )
    return AuditAction(mode="warn", claim=span, reason="unknown claim type")


def _downgrade_entitlement(original: str) -> str:
    # "я знайшла" -> "я припускаю, що" etc.
    lower = original.lower()
    if "знайш" in lower or "вияв" in lower or "помітил" in lower:
        return "я припускаю, що"
    if "виділ" in lower:
        return "можливо, є"
    if "дослідил" in lower or "проаналізувал" in lower:
        return "варто було б перевірити"
    if "перевірил" in lower:
        return "потрібно перевірити"
    return "можливо,"


def _apply_rewrites(text: str, actions: list[AuditAction]) -> str:
    """Apply redact/downgrade rewrites in reverse span order to keep offsets valid."""
    rewrites = [a for a in actions if a.mode in ("redact", "downgrade") and a.rewrite]
    rewrites.sort(key=lambda a: a.claim.span[0], reverse=True)
    out = text
    for a in rewrites:
        s, e = a.claim.span
        out = out[:s] + (a.rewrite or "") + out[e:]
    return out


# ============== audit log ==============


def _log_violations(
    report: AuditReport,
    turn: TurnContext,
    turn_id: str | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    excerpt = report.response_text
    if len(excerpt) > 400:
        excerpt = excerpt[:400] + "…"
    record = {
        "ts": time.time(),
        "session_id": turn.session_id,
        "turn_id": turn_id,
        "response_excerpt": excerpt,
        "response_hash": "sha256:" + hashlib.sha256(
            report.response_text.encode("utf-8")
        ).hexdigest()[:16],
        "violations": [
            {
                "claim_type": a.claim.claim_type,
                "text": a.claim.text,
                "span": list(a.claim.span),
                "detector": a.claim.detector,
                "mode": a.mode,
                "reason": a.reason,
                "entity_hint": a.claim.entity_hint,
                "numeric_value": a.claim.numeric_value,
            }
            for a in report.violations
        ],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
