"""Persistent browser execution memory for hardening.

Tracks repeated browser failures and successes by domain/tool/signature so the
agent and UI can see where real-world flows are brittle and what already works.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from remy.config.settings import settings

_LOCK = threading.Lock()
_MAX_CLUSTERS = 200


def _failure_storage_path() -> Path:
    return settings.DATA_DIR / "browser_failure_memory.json"


def _success_storage_path() -> Path:
    return settings.DATA_DIR / "browser_success_playbooks.json"


def _normalize_domain(url: str) -> str:
    if not url:
        return "unknown"
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        return (parsed.hostname or "unknown").lower()
    except Exception:
        return "unknown"


def _signature_from_text(text: str) -> str:
    lower = (text or "").lower()
    if "captcha" in lower:
        return "captcha"
    if "email verification" in lower or "verify your email" in lower or "check your email" in lower:
        return "email_verification"
    if "sms" in lower or "phone verification" in lower or "verification code" in lower:
        return "phone_verification"
    if "kyc" in lower or "identity verification" in lower:
        return "kyc_verification"
    if "payment" in lower or "card declined" in lower:
        return "payment_block"
    if "validation" in lower or "required field" in lower or "please correct" in lower:
        return "validation_error"
    if "overlay" in lower or "popup" in lower or "cookie" in lower:
        return "blocking_overlay"
    if "timeout" in lower:
        return "timeout"
    if "not found" in lower or "no element" in lower:
        return "selector_not_found"
    if "detached" in lower or "navigat" in lower:
        return "navigation_detach"
    if "error" in lower:
        return "page_error"
    return "unknown"


def _flow_from_text(url: str, text: str) -> str:
    lower = f"{url} {text}".lower()
    if any(
        marker in lower
        for marker in ("signup", "sign up", "register", "create account", "dashboard", "account")
    ):
        return "signup"
    if any(
        marker in lower
        for marker in ("publish", "post is live", "view post", "/status/", "/posts/", "compose")
    ):
        return "publish"
    if any(marker in lower for marker in ("checkout", "payment", "billing")):
        return "checkout"
    if any(marker in lower for marker in ("search", "results", "pricing", "plans")):
        return "research"
    return "navigation"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records[:_MAX_CLUSTERS], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_browser_failure(
    *,
    tool: str,
    action: str = "",
    url: str = "",
    text: str = "",
    selector: str = "",
    status: str = "failed",
) -> None:
    """Upsert a browser failure cluster keyed by domain/tool/action/signature."""
    domain = _normalize_domain(url)
    signature = _signature_from_text(text)
    now = datetime.now().isoformat()
    key = (domain, tool, action or "", signature)
    path = _failure_storage_path()

    with _LOCK:
        records = _load(path)
        for rec in records:
            if (
                rec.get("domain"),
                rec.get("tool"),
                rec.get("action", ""),
                rec.get("signature"),
            ) == key:
                rec["count"] = int(rec.get("count", 0)) + 1
                rec["last_seen"] = now
                rec["last_url"] = url
                rec["last_text"] = text[:500]
                if selector:
                    rec["last_selector"] = selector
                    selector_counts = rec.setdefault("selector_counts", {})
                    selector_counts[selector] = int(selector_counts.get(selector, 0)) + 1
                rec["status"] = status
                _save(
                    path, sorted(records, key=lambda item: item.get("last_seen", ""), reverse=True)
                )
                return

        records.append(
            {
                "domain": domain,
                "tool": tool,
                "action": action or "",
                "signature": signature,
                "count": 1,
                "status": status,
                "last_seen": now,
                "last_url": url,
                "last_text": text[:500],
                "last_selector": selector,
                "selector_counts": {selector: 1} if selector else {},
            }
        )
        _save(path, sorted(records, key=lambda item: item.get("last_seen", ""), reverse=True))


def record_browser_success(
    *,
    tool: str,
    action: str = "",
    url: str = "",
    text: str = "",
    selector: str = "",
    status: str = "verified",
) -> None:
    """Upsert a reusable browser success playbook keyed by domain/tool/action/flow."""
    domain = _normalize_domain(url)
    flow = _flow_from_text(url, text)
    now = datetime.now().isoformat()
    key = (domain, tool, action or "", flow)
    path = _success_storage_path()

    with _LOCK:
        records = _load(path)
        for rec in records:
            if (
                rec.get("domain"),
                rec.get("tool"),
                rec.get("action", ""),
                rec.get("flow"),
            ) == key:
                rec["count"] = int(rec.get("count", 0)) + 1
                rec["last_seen"] = now
                rec["last_url"] = url
                rec["last_text"] = text[:500]
                if selector:
                    rec["last_selector"] = selector
                    selector_counts = rec.setdefault("selector_counts", {})
                    selector_counts[selector] = int(selector_counts.get(selector, 0)) + 1
                rec["status"] = status
                _save(
                    path,
                    sorted(
                        records,
                        key=lambda item: (item.get("count", 0), item.get("last_seen", "")),
                        reverse=True,
                    ),
                )
                return

        records.append(
            {
                "domain": domain,
                "tool": tool,
                "action": action or "",
                "flow": flow,
                "count": 1,
                "status": status,
                "last_seen": now,
                "last_url": url,
                "last_text": text[:500],
                "last_selector": selector,
                "selector_counts": {selector: 1} if selector else {},
            }
        )
        _save(
            path,
            sorted(
                records,
                key=lambda item: (item.get("count", 0), item.get("last_seen", "")),
                reverse=True,
            ),
        )


def get_browser_failure_report(limit: int = 10) -> dict:
    """Return summary + top clusters for diagnostics and prompt injection."""
    with _LOCK:
        records = sorted(
            _load(_failure_storage_path()),
            key=lambda item: (item.get("count", 0), item.get("last_seen", "")),
            reverse=True,
        )

    top = records[:limit]
    domains = {}
    for rec in records:
        domains[rec.get("domain", "unknown")] = domains.get(rec.get("domain", "unknown"), 0) + int(
            rec.get("count", 0)
        )

    hottest_domain = None
    if domains:
        hottest_domain = max(domains.items(), key=lambda item: item[1])[0]

    return {
        "total_clusters": len(records),
        "total_failures": sum(int(rec.get("count", 0)) for rec in records),
        "hottest_domain": hottest_domain,
        "top_clusters": top,
    }


def get_browser_success_report(limit: int = 10) -> dict:
    """Return summary + top success playbooks for diagnostics and prompt injection."""
    with _LOCK:
        records = sorted(
            _load(_success_storage_path()),
            key=lambda item: (item.get("count", 0), item.get("last_seen", "")),
            reverse=True,
        )

    top = records[:limit]
    domains = {}
    for rec in records:
        domains[rec.get("domain", "unknown")] = domains.get(rec.get("domain", "unknown"), 0) + int(
            rec.get("count", 0)
        )

    hottest_domain = None
    if domains:
        hottest_domain = max(domains.items(), key=lambda item: item[1])[0]

    return {
        "total_playbooks": len(records),
        "total_successes": sum(int(rec.get("count", 0)) for rec in records),
        "hottest_domain": hottest_domain,
        "top_playbooks": top,
    }


def _flow_sequence_storage_path() -> Path:
    return settings.DATA_DIR / "browser_flow_sequences.json"


_MAX_SEQUENCES = 100


def record_flow_sequence(
    *,
    domain: str,
    flow: str,
    steps: list[dict],
    status: str = "verified",
) -> None:
    """Store a multi-step browser flow that reached a verified outcome.

    Only for signup_operator and publisher flows — keeps last N sequences
    ranked by recency, keyed by domain+flow.
    """
    if flow not in ("signup", "publish"):
        return
    if not steps:
        return

    now = datetime.now().isoformat()
    path = _flow_sequence_storage_path()

    compact_steps = []
    for step in steps:
        compact_steps.append(
            {
                "tool": step.get("tool", ""),
                "action": step.get("action", ""),
                "selector": step.get("selector", ""),
                "url": step.get("url", "")[:200],
                "status": step.get("status", ""),
            }
        )

    key = (domain, flow)

    with _LOCK:
        records = _load(path)
        for rec in records:
            if (rec.get("domain"), rec.get("flow")) == key:
                rec["count"] = int(rec.get("count", 0)) + 1
                rec["last_seen"] = now
                rec["steps"] = compact_steps
                rec["status"] = status
                _save(
                    path,
                    sorted(records, key=lambda r: r.get("last_seen", ""), reverse=True)[
                        :_MAX_SEQUENCES
                    ],
                )
                return

        records.append(
            {
                "domain": domain,
                "flow": flow,
                "count": 1,
                "status": status,
                "last_seen": now,
                "steps": compact_steps,
            }
        )
        _save(
            path,
            sorted(records, key=lambda r: r.get("last_seen", ""), reverse=True)[:_MAX_SEQUENCES],
        )


def get_flow_sequence(domain: str, flow: str) -> dict | None:
    """Get the last successful flow sequence for a domain+flow pair."""
    with _LOCK:
        records = _load(_flow_sequence_storage_path())
    for rec in records:
        if rec.get("domain") == domain and rec.get("flow") == flow:
            return rec
    return None


def get_browser_execution_hints(
    *,
    url: str = "",
    text: str = "",
    action: str = "",
    limit: int = 3,
) -> dict:
    """Return domain/flow-specific browser hints for the next execution step."""
    domain = _normalize_domain(url)
    if domain == "unknown":
        return {
            "domain": None,
            "flow": _flow_from_text(url, text),
            "failure_hints": [],
            "success_hints": [],
        }

    flow = _flow_from_text(url, text)
    normalized_action = (action or "").strip().lower()

    def _score(record: dict) -> tuple[int, int, str]:
        score = 0
        if record.get("domain") == domain:
            score += 5
        if flow and record.get("flow") == flow:
            score += 3
        if normalized_action and record.get("action", "").lower() == normalized_action:
            score += 2
        return (
            score,
            int(record.get("count", 0)),
            str(record.get("last_seen", "")),
        )

    with _LOCK:
        failures = _load(_failure_storage_path())
        successes = _load(_success_storage_path())

    matching_failures = [
        rec
        for rec in failures
        if rec.get("domain") == domain
        and (not normalized_action or rec.get("action", "").lower() == normalized_action)
    ]
    matching_successes = [
        rec
        for rec in successes
        if rec.get("domain") == domain
        and (not normalized_action or rec.get("action", "").lower() == normalized_action)
    ]

    matching_failures.sort(key=_score, reverse=True)
    matching_successes.sort(key=_score, reverse=True)

    avoided_selectors = []
    preferred_selectors = []
    for rec in matching_failures:
        selector_counts = rec.get("selector_counts") or {}
        for selector, count in sorted(
            selector_counts.items(), key=lambda item: item[1], reverse=True
        ):
            avoided_selectors.append({"selector": selector, "count": int(count)})
    for rec in matching_successes:
        selector_counts = rec.get("selector_counts") or {}
        for selector, count in sorted(
            selector_counts.items(), key=lambda item: item[1], reverse=True
        ):
            preferred_selectors.append({"selector": selector, "count": int(count)})

    def _dedupe_selector_counts(items: list[dict], limit_value: int) -> list[dict]:
        merged = {}
        for item in items:
            selector = item.get("selector")
            if not selector:
                continue
            merged[selector] = merged.get(selector, 0) + int(item.get("count", 0))
        ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        return [{"selector": selector, "count": count} for selector, count in ranked[:limit_value]]

    return {
        "domain": domain,
        "flow": flow,
        "failure_hints": matching_failures[:limit],
        "success_hints": matching_successes[:limit],
        "avoided_selectors": _dedupe_selector_counts(avoided_selectors, limit),
        "preferred_selectors": _dedupe_selector_counts(preferred_selectors, limit),
    }
