"""
Orchestrator — worker selection, dispatch, and post-execution decisions.

Extracted from autonomy.py (P2.4) to separate:
- WHAT worker to invoke (selection)
- HOW to invoke it (dispatch)
- WHAT to do after (evaluate, block, complete, escalate)

autonomy.py keeps: cycle timing, budget, goal management, sleep, event emission.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Orchestrator")


# ============== Worker Selection (via Capability Packs) ==============


def resolve_pack_for_goal(goal: dict | None):
    """Resolve the capability pack for a goal. Returns CapabilityPack."""
    from remy.core.capability_packs import resolve_pack

    return resolve_pack(goal)


def select_worker(goal: dict | None) -> str:
    """Decide which worker handles this goal.

    Uses capability pack resolution: explicit goal_template first, keyword inference second.
    Returns one of: "browser_worker", "research_worker", "monitoring_worker", "generic".
    """
    return resolve_pack_for_goal(goal).worker


def _should_use_browser_worker(goal: dict | None) -> bool:
    """Backward compat — now delegates to capability pack resolution."""
    return resolve_pack_for_goal(goal).worker == "browser_worker"


def _should_use_research_worker(goal: dict | None) -> bool:
    """Backward compat — now delegates to capability pack resolution."""
    return resolve_pack_for_goal(goal).worker == "research_worker"


# ============== Goal Focus ==============


_focus_stale_tracker: dict[str, int] = {}  # mission_id → consecutive cycles without progress
_FOCUS_STALE_LIMIT = 10  # release focus after this many cycles without progress


def focus_execution_goals(goals: list[dict]) -> list[dict]:
    """Prioritize runnable tasks from the current mission over legacy goals.

    Includes stale-focus escape: if the focused mission makes no progress for
    _FOCUS_STALE_LIMIT consecutive cycles, release focus so other goals can run.
    """
    if not goals:
        return goals

    runnable_mission_tasks = [
        goal
        for goal in goals
        if goal.get("mission_id")
        and goal.get("mission_task_id")
        and goal.get("status") not in ("blocked_external", "blocked_by_user", "archived")
    ]
    if not runnable_mission_tasks:
        _focus_stale_tracker.clear()
        return goals

    focus_mission_id = runnable_mission_tasks[0].get("mission_id")

    # Stale-focus escape: check if the focus mission is making progress
    stale_count = _focus_stale_tracker.get(focus_mission_id, 0)
    if stale_count >= _FOCUS_STALE_LIMIT:
        logger.info(
            "Mission '%s' stale for %d cycles — releasing focus to allow other goals",
            focus_mission_id, stale_count,
        )
        # Reset counter so it gets another chance next round
        _focus_stale_tracker[focus_mission_id] = 0
        return goals  # No focus filtering — all goals visible

    # Track this cycle (caller should call mark_focus_progress() on success)
    _focus_stale_tracker[focus_mission_id] = stale_count + 1

    focused: list[dict] = []
    blocked: list[dict] = []
    for goal in goals:
        if goal.get("status") in ("blocked_external", "blocked_by_user"):
            blocked.append(goal)
        elif goal.get("mission_id") == focus_mission_id:
            focused.append(goal)
    return focused + blocked


def mark_focus_progress(mission_id: str):
    """Call after a mission task makes real progress — resets stale counter."""
    _focus_stale_tracker.pop(mission_id, None)


def get_focus_stale_cycles(mission_id: str) -> int:
    """Return consecutive no-progress focus cycles for a mission."""
    return int(_focus_stale_tracker.get(mission_id, 0) or 0)


# ============== Pre-run Budget Gate ==============


def _check_run_budget(goal: dict) -> str | None:
    """Check if mission budget_per_run allows this execution.

    Returns an error message string if budget exceeded, None if OK.
    """
    budget_limit = goal.get("_budget_per_run_usd")
    if not budget_limit:
        return None

    try:
        from remy.core.cost_tracker import get_cost_tracker
        status = get_cost_tracker().get_status()
        # Use hourly spend as proxy for "this run" (runs happen every 5min)
        hourly_spend = status["hourly"]["total_usd"]
        if hourly_spend >= budget_limit:
            return (
                f"Budget gate: mission budget_per_run=${budget_limit:.2f} exceeded "
                f"(spent ${hourly_spend:.4f} this hour). Skipping cycle."
            )
    except Exception as e:
        logger.debug("Budget check failed (allowing run): %s", e)

    return None


# ============== Worker Dispatch ==============


async def dispatch_worker(
    goal: dict | None,
    decision_prompt: str,
    session_id: str,
    session_log: list,
    history: list | None,
    current_plan: object | None = None,
) -> tuple[str, list, list, Any]:
    """Dispatch the appropriate worker and return (response_text, history, log, worker_result).

    This is the single entry point for all worker invocations from the autonomy loop.
    """
    pack = resolve_pack_for_goal(goal)
    worker = pack.worker
    logger.info(
        "Capability pack: %s → worker: %s (steps=%d, timeout=%ds)",
        pack.id,
        worker,
        pack.step_budget,
        pack.timeout_sec,
    )

    # Inject pack budgets into goal so workers can forward them
    goal_with_pack = dict(goal or {})
    goal_with_pack["_pack_step_budget"] = pack.step_budget
    goal_with_pack["_pack_timeout_sec"] = pack.timeout_sec
    goal_with_pack["_pack_id"] = pack.id
    goal_with_pack["_pack_label"] = pack.label
    goal_with_pack["_pack_approval_mode"] = pack.approval_mode

    # Apply mission manifest overrides (from missions.json pack fields)
    manifest = (goal or {}).get("pack_manifest") or {}
    if manifest.get("budget_per_run") is not None:
        goal_with_pack["_budget_per_run_usd"] = float(manifest["budget_per_run"])
    if manifest.get("tools"):
        goal_with_pack["_tools_whitelist"] = list(manifest["tools"])
    if manifest.get("approval_gates"):
        goal_with_pack["_approval_gates"] = list(manifest["approval_gates"])

    # Pre-run budget check (if budget_per_run is set)
    budget_check = _check_run_budget(goal_with_pack)
    if budget_check:
        logger.warning("Run budget exceeded for goal: %s", budget_check)
        return budget_check, history or [], session_log, None

    if worker == "browser_worker":
        from remy.core.workers.browser_worker import run_browser_worker

        result = await run_browser_worker(
            goal=goal_with_pack,
            session_id=session_id,
            session_log=session_log,
            history=history,
            current_plan=current_plan,
        )
        return result.response_text, result.history, result.session_log, result

    if worker == "research_worker":
        from remy.core.workers.research_worker import run_research_worker

        result = await run_research_worker(
            goal=goal_with_pack,
            session_id=session_id,
            session_log=session_log,
            history=history,
            current_plan=current_plan,
        )
        if result.status in ("completed", "findings_collected"):
            try:
                from remy.core.notification_router import notify
                topic = (goal_with_pack or {}).get("description", "Research")[:80]
                findings = result.evidence.get("findings_count") or result.evidence.get("accepted_sources_count", 0)
                artifact_id = result.evidence.get("final_artifact_id", "")
                msg = (
                    f"Research complete: {topic}\n\n"
                    f"Findings: {findings}\n"
                    + (f"Artifact: {artifact_id}" if artifact_id else "Report saved to brain")
                )
                notify(msg, level="info", event_type="research.complete")
            except Exception as _ne:
                logger.debug("Could not send research completion notify: %s", _ne)
        return result.response_text, result.history, result.session_log, result

    if worker == "monitoring_worker":
        from remy.core.workers.monitoring_worker import run_monitoring_worker

        result = await run_monitoring_worker(
            goal=goal_with_pack,
            session_id=session_id,
            session_log=session_log,
            history=history,
            current_plan=current_plan,
        )
        return result.response_text, result.history, result.session_log, result

    # Generic: use the main agent
    from remy.core.agent import invoke_agent

    response_text, new_history, new_log = await invoke_agent(
        user_message=decision_prompt,
        session_id=session_id,
        channel="autonomous",
        session_log=session_log,
        history=history,
    )
    return response_text, new_history, new_log, None


# ============== Post-Execution: Reporter ==============


def format_execution_report(worker_result, response_text: str) -> str:
    """Normalize execution-heavy flows into concise operator-facing output."""
    if not worker_result:
        return response_text

    from remy.core.workers.reporter import format_worker_report

    return format_worker_report(worker_result, fallback_text=response_text)


# ============== Post-Execution: Blocker Detection ==============


def _check_domain_blocker_history(goal: dict | None) -> dict | None:
    """Fast-path: if browser memory shows repeated hard blockers on target domain, escalate immediately."""
    if not goal:
        return None
    try:
        from remy.core.browser_failure_memory import get_browser_execution_hints

        url = goal.get("target_url", "") or goal.get("description", "") or ""
        hints = get_browser_execution_hints(url=url, text="", action="", limit=5)
        for rec in hints.get("failure_hints") or []:
            sig = rec.get("signature", "")
            count = int(rec.get("count", 0))
            # If a hard blocker has been seen 3+ times on this domain, escalate immediately
            if (
                sig
                in (
                    "captcha",
                    "email_verification",
                    "phone_verification",
                    "kyc_verification",
                    "payment_block",
                )
                and count >= 3
            ):
                domain = hints.get("domain", "unknown")
                return {
                    "reason": f"{sig} (seen {count} times on {domain})",
                    "evidence": f"Browser memory: {sig} x{count} on {domain}. Historical pattern — do not retry blind.",
                    "resume_context": f"Domain {domain} has a persistent {sig} blocker. Needs manual intervention or alternative approach.",
                }
    except Exception:
        pass
    return None


def detect_external_blocker(goal: dict | None, session_log: list[dict]) -> dict | None:
    """Detect hard external blockers for signup/publish flows from structured tool evidence."""
    if not goal or goal.get("goal_template") not in ("signup_operator", "publisher"):
        return None

    # Fast-path: check browser memory for historically blocked domains
    history_blocker = _check_domain_blocker_history(goal)
    if history_blocker:
        return history_blocker

    blocker_markers = [
        ("captcha", "captcha challenge"),
        ("verify your email", "email verification required"),
        ("check your email", "email verification required"),
        ("verification code", "verification code required"),
        ("sms", "sms verification required"),
        ("text message", "sms verification required"),
        ("phone verification", "phone verification required"),
        ("payment", "payment step required"),
        ("card", "payment step required"),
        ("kyc", "kyc/manual verification required"),
        ("identity verification", "kyc/manual verification required"),
    ]

    for entry in reversed(session_log):
        if entry.get("type") != "tool_call":
            continue
        if entry.get("tool") not in ("browse_page", "browser_act"):
            continue
        if entry.get("external_blocker_likely") and entry.get("blocker_reason"):
            evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
            page_url = (
                evidence.get("page_url") or entry.get("url") or entry.get("requested_url") or ""
            )
            blocker_count = entry.get("blocker_count")
            detail = f"{entry['blocker_reason']} at {page_url or 'observed page'}".strip()
            if blocker_count:
                detail += f" | repeated {blocker_count} times"
            return {
                "reason": str(entry["blocker_reason"]),
                "evidence": detail[:500],
                "resume_context": f"Resume from the page/blocker context near {page_url or 'the blocked step'}",
            }
        text_parts = [
            str(entry.get("page_state", "") or ""),
            str(entry.get("answer", "") or ""),
            str(entry.get("description", "") or ""),
            str(entry.get("result", "") or ""),
        ]
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        text_parts.append(str(evidence.get("page_text_snippet", "") or ""))
        haystack = " ".join(text_parts).lower()

        for marker, reason in blocker_markers:
            if marker in haystack:
                page_url = (
                    evidence.get("page_url") or entry.get("url") or entry.get("requested_url") or ""
                )
                detail = f"{reason} at {page_url or 'observed page'}".strip()
                if entry.get("answer"):
                    detail += f" | answer: {str(entry.get('answer'))[:160]}"
                return {
                    "reason": reason,
                    "evidence": detail[:500],
                    "resume_context": f"Resume from the page/blocker context near {page_url or 'the blocked step'}",
                }

    return None


# ============== Post-Execution: Zero-Tool Guard ==============


def check_zero_tool_cycle(session_log: list[dict], response_text: str) -> dict | None:
    """If agent talked but used no tools, return a pre-built failure evaluation."""
    tool_calls = [e for e in session_log if e.get("type") == "tool_call"]
    if not tool_calls and response_text:
        return {
            "success": False,
            "confidence": 0.95,
            "reason": "No tools called — agent produced text without action",
            "goal_completed": False,
        }
    return None


def check_obvious_failure(response_text: str) -> dict | None:
    """Detect obvious error responses without spending tokens on evaluation."""
    if (
        any(
            m in response_text.lower()
            for m in ("couldn't generate a response", "error", "failed to", "cannot")
        )
        and len(response_text) < 200
    ):
        return {
            "success": False,
            "confidence": 0.8,
            "reason": "Agent returned an error response",
            "goal_completed": False,
        }
    return None
