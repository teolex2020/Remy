"""Browser worker wrapper with a short prompt and scoped tool access."""

from __future__ import annotations

import logging
from typing import Any

from remy.core.workers.contracts import WorkerExecutionResult

logger = logging.getLogger("BrowserWorker")

BROWSER_WORKER_CHANNEL = "browser-worker"


def _latest_browser_entry(session_log: list[dict]) -> dict[str, Any] | None:
    for entry in reversed(session_log or []):
        if entry.get("type") == "tool_call" and entry.get("tool") in ("browse_page", "browser_act"):
            return entry
    return None


def _compact_browser_context(session_log: list[dict], limit: int = 3) -> str:
    lines: list[str] = []
    recent = [
        entry
        for entry in (session_log or [])
        if entry.get("type") == "tool_call" and entry.get("tool") in ("browse_page", "browser_act")
    ][-limit:]
    for entry in recent:
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        page_url = evidence.get("page_url") or entry.get("url") or entry.get("requested_url") or ""
        page_state = entry.get("page_state") or "unknown"
        status = entry.get("status") or ("verified" if entry.get("verified") else "attempted")
        err = entry.get("visible_error_text") or entry.get("blocker_reason") or ""
        snippet = f"- {entry.get('tool')} status={status} page_state={page_state}"
        if page_url:
            snippet += f" url={page_url}"
        if err:
            snippet += f" error={err}"
        lines.append(snippet)
    return "\n".join(lines)


def _format_browser_memory_hints(goal: dict, session_log: list[dict] | None = None) -> str:
    """Build pre-action hints from browser failure/success memory for the worker prompt."""
    try:
        from remy.core.browser_failure_memory import get_browser_execution_hints
    except Exception:
        return ""

    # Extract target URL from goal or recent session log
    url = (goal or {}).get("target_url", "") or ""
    if not url and session_log:
        for entry in reversed(session_log):
            if entry.get("type") == "tool_call" and entry.get("tool") in (
                "browse_page",
                "browser_act",
            ):
                evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
                url = (
                    evidence.get("page_url") or entry.get("url") or entry.get("requested_url") or ""
                )
                if url:
                    break

    desc = (goal or {}).get("description", "") or ""
    action = (goal or {}).get("task_action", "") or ""
    text = f"{desc} {action}".strip()

    if not url and not text:
        return ""

    try:
        hints = get_browser_execution_hints(url=url, text=text, action="", limit=3)
    except Exception:
        return ""

    if not hints.get("failure_hints") and not hints.get("success_hints"):
        return ""

    lines = ["\nBROWSER MEMORY (from past executions):"]

    # Known blockers
    if hints.get("failure_hints"):
        for rec in hints["failure_hints"][:3]:
            sig = rec.get("signature", "unknown")
            count = rec.get("count", 0)
            lines.append(f"- KNOWN BLOCKER: {sig} x{count} on {hints.get('domain', '?')}")

    # Avoided selectors
    if hints.get("avoided_selectors"):
        bad = [
            f"'{item['selector']}' (failed x{item['count']})"
            for item in hints["avoided_selectors"][:3]
        ]
        lines.append(f"- AVOID selectors: {', '.join(bad)}")

    # Preferred selectors
    if hints.get("preferred_selectors"):
        good = [
            f"'{item['selector']}' (worked x{item['count']})"
            for item in hints["preferred_selectors"][:3]
        ]
        lines.append(f"- PREFER selectors: {', '.join(good)}")

    # Success playbook hints
    if hints.get("success_hints"):
        for rec in hints["success_hints"][:2]:
            flow = rec.get("flow", "navigation")
            count = rec.get("count", 0)
            tool_part = rec.get("tool", "browser")
            if rec.get("action"):
                tool_part += f"/{rec['action']}"
            lines.append(f"- KNOWN SUCCESS: {flow} x{count} via {tool_part}")

    # Flow sequence — show the last verified multi-step path
    try:
        from remy.core.browser_failure_memory import (
            _flow_from_text,
            _normalize_domain,
            get_flow_sequence,
        )

        domain = _normalize_domain(url)
        flow = _flow_from_text(url, text)
        if flow in ("signup", "publish"):
            seq = get_flow_sequence(domain, flow)
            if seq and seq.get("steps"):
                steps_str = " → ".join(
                    f"{s.get('tool', '?')}({s.get('action', '')}{': ' + s['selector'] if s.get('selector') else ''})"
                    for s in seq["steps"][:8]
                )
                lines.append(f"- VERIFIED FLOW ({flow} x{seq.get('count', 0)}): {steps_str}")
    except Exception:
        pass

    lines.append(
        "- Use preferred selectors first. If a known blocker appears, report blocked_external immediately."
    )
    return "\n".join(lines) + "\n"


def build_browser_worker_prompt(
    goal: dict,
    current_plan: object | None = None,
    session_log: list[dict] | None = None,
) -> str:
    """Build a compact operational prompt for signup/publish browser flows."""
    desc = (goal or {}).get("description", "") or ""
    template = (goal or {}).get("goal_template", "") or ""
    resume_context = (goal or {}).get("resume_context", "") or ""
    blocked_reason = (goal or {}).get("blocked_reason", "") or ""
    task_action = (goal or {}).get("task_action", "") or ""
    task_done_when = (goal or {}).get("task_done_when", "") or ""

    plan_line = ""
    if task_action:
        plan_line = f"\nCURRENT TASK ACTION: {task_action}"
        if task_done_when:
            plan_line += f"\nDONE WHEN: {task_done_when}"
    elif current_plan:
        try:
            plan_line = f"\nPLAN CONTEXT: {str(current_plan)[:400]}"
        except Exception:
            plan_line = ""

    browser_context = _compact_browser_context(session_log or [])
    if browser_context:
        browser_context = f"\nRECENT BROWSER STATE:\n{browser_context}\n"

    resume_text = ""
    if resume_context or blocked_reason:
        resume_text = "\nRESUME CONTEXT:\n"
        if blocked_reason:
            resume_text += f"- Previous blocker: {blocked_reason}\n"
        if resume_context:
            resume_text += f"- Continue from: {resume_context}\n"

    memory_hints = _format_browser_memory_hints(goal, session_log)

    return (
        "You are BROWSER_WORKER.\n"
        "Execute the next browser step for this task using only browser tools.\n"
        "Do not research broadly. Do not explain strategy. Do not ask follow-up questions.\n"
        "Return concise evidence-backed output only.\n"
        "\nTASK:\n"
        f"{desc}\n"
        f"JOB TEMPLATE: {template or 'browser'}\n"
        f"{plan_line}"
        f"{resume_text}"
        f"{browser_context}"
        f"{memory_hints}"
        "\nRULES:\n"
        "- Use browse_page/browser_act/browser_close only.\n"
        "- Prefer one verified step over a long explanation.\n"
        "- If the page shows an error, cite the exact visible error text.\n"
        "- If blocked by captcha, email verification, SMS, payment, or KYC, report blocked_external with evidence.\n"
        "- If a step is unverified, say attempted, not completed.\n"
        "- Keep the final response under 6 lines in operator format.\n"
        f"{_publisher_playbook(goal)}"
        f"{_pack_guardrails(goal)}"
    )


def _pack_guardrails(goal: dict) -> str:
    """Inject capability pack guardrails into the worker prompt."""
    try:
        from remy.core.capability_packs import format_guardrails_for_prompt, resolve_pack

        pack = resolve_pack(goal)
        return format_guardrails_for_prompt(pack)
    except Exception:
        return ""


def _publisher_playbook(goal: dict) -> str:
    """Inject publisher mode/channel playbook into publisher tasks only."""
    try:
        from remy.core.capability_packs import format_publisher_playbook_for_prompt, resolve_pack

        pack = resolve_pack(goal)
        if pack.id != "publisher":
            return ""
        return format_publisher_playbook_for_prompt(goal)
    except Exception:
        return ""


def _derive_browser_worker_status(
    session_log: list[dict], response_text: str
) -> tuple[str, dict[str, Any]]:
    latest = _latest_browser_entry(session_log)
    if not latest:
        return ("no_action", {})

    evidence = latest.get("evidence") if isinstance(latest.get("evidence"), dict) else {}
    compact = {
        "tool": latest.get("tool"),
        "status": latest.get("status"),
        "page_state": latest.get("page_state"),
        "current_url": evidence.get("page_url") or latest.get("url") or latest.get("requested_url"),
        "visible_error_text": latest.get("visible_error_text"),
        "verified": latest.get("verified"),
    }
    if latest.get("external_blocker_likely") or latest.get("blocker_reason"):
        return ("blocked_external", compact)
    if latest.get("verified") is True or latest.get("status") == "verified":
        return ("verified", compact)
    if response_text and "Status: blocked_external" in response_text:
        return ("blocked_external", compact)
    return ("attempted", compact)


async def run_browser_worker(
    goal: dict,
    session_id: str,
    session_log: list,
    history: list | None = None,
    current_plan: object | None = None,
) -> WorkerExecutionResult:
    """Run the specialized browser worker via the main agent graph with a scoped channel."""
    from remy.core.agent import invoke_agent

    prompt = build_browser_worker_prompt(goal, current_plan=current_plan, session_log=session_log)
    response_text, new_history, new_log = await invoke_agent(
        user_message=prompt,
        session_id=session_id,
        channel=BROWSER_WORKER_CHANNEL,
        session_log=session_log,
        history=history,
    )
    status, evidence = _derive_browser_worker_status(new_log, response_text)
    tool_calls = sum(1 for entry in new_log if entry.get("type") == "tool_call")

    try:
        from remy.core.capability_packs import (
            infer_publisher_channel,
            infer_publisher_mode,
            resolve_pack,
        )

        pack = resolve_pack(goal or {})
        evidence["capability_pack"] = pack.id
        evidence["approval_mode"] = pack.approval_mode
        if pack.id == "publisher":
            evidence["publisher_mode"] = infer_publisher_mode(goal or {})
            evidence["publisher_channel"] = infer_publisher_channel(goal or {})
    except Exception:
        logger.debug("Could not enrich browser worker evidence with capability pack", exc_info=True)

    # Record flow sequence on verified outcomes (signup/publish only)
    if status == "verified":
        _try_record_flow_sequence(goal, new_log)

    return WorkerExecutionResult(
        worker="browser_worker",
        status=status,
        response_text=response_text,
        history=new_history,
        session_log=new_log,
        evidence=evidence,
        tool_calls=tool_calls,
    )


def _try_record_flow_sequence(goal: dict, session_log: list[dict]) -> None:
    """Extract and persist the multi-step browser flow from a verified session."""
    try:
        from remy.core.browser_failure_memory import (
            _flow_from_text,
            _normalize_domain,
            record_flow_sequence,
        )

        # Extract steps from session log
        steps = []
        last_url = ""
        for entry in session_log:
            if entry.get("type") != "tool_call":
                continue
            if entry.get("tool") not in ("browse_page", "browser_act"):
                continue
            evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
            url = evidence.get("page_url") or entry.get("url") or entry.get("requested_url") or ""
            if url:
                last_url = url
            steps.append(
                {
                    "tool": entry.get("tool", ""),
                    "action": str(entry.get("action") or ""),
                    "selector": str(entry.get("selector") or evidence.get("selector") or ""),
                    "url": url,
                    "status": str(entry.get("status") or ""),
                }
            )

        if not steps or not last_url:
            return

        domain = _normalize_domain(last_url)
        desc = (goal or {}).get("description", "") or ""
        flow = _flow_from_text(last_url, desc)
        if flow not in ("signup", "publish"):
            return

        record_flow_sequence(domain=domain, flow=flow, steps=steps, status="verified")
    except Exception:
        logger.debug("Flow sequence recording failed", exc_info=True)
