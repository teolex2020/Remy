"""
Monitoring Worker — fetches target pages, detects changes, stores findings.

Unlike research_worker which explores broadly, monitoring_worker:
- Takes a specific list of URLs from the goal
- Fetches current content for each
- Compares against stored snapshots
- Reports structured change events
- Stores significant changes into research memory
"""

from __future__ import annotations

import logging

from remy.core.workers.contracts import WorkerExecutionResult

logger = logging.getLogger("MonitoringWorker")

MONITORING_WORKER_CHANNEL = "autonomous"


def build_monitoring_prompt(goal: dict) -> str:
    """Build a compact operational prompt for monitoring tasks."""
    desc = (goal or {}).get("description", "") or ""
    task_action = (goal or {}).get("task_action", "") or ""
    task_done_when = (goal or {}).get("task_done_when", "") or ""

    action_line = ""
    if task_action:
        action_line = f"\nCURRENT TASK ACTION: {task_action}"
        if task_done_when:
            action_line += f"\nDONE WHEN: {task_done_when}"

    guardrails = ""
    try:
        from remy.core.capability_packs import format_guardrails_for_prompt, resolve_pack

        pack = resolve_pack(goal)
        guardrails = format_guardrails_for_prompt(pack)
    except Exception:
        pass

    return (
        "You are MONITORING_WORKER.\n"
        "Your job: fetch target URLs, extract their content, and report what changed.\n"
        "Do NOT research broadly. Do NOT visit unrelated pages.\n"
        "\nTASK:\n"
        f"{desc}\n"
        f"{action_line}\n"
        "\nSTEPS:\n"
        "1. Use extract_content or http_get to fetch each target URL.\n"
        "2. The system will automatically compare content against stored snapshots.\n"
        "3. Report the change detection results.\n"
        "\nRULES:\n"
        "- Use extract_content for web pages, http_get for APIs/JSON.\n"
        "- Do NOT use browse_page or browser_act.\n"
        "- If a URL is unreachable, note it and move on.\n"
        "- Keep the final response under 10 lines: list each URL + change status.\n"
        f"{guardrails}"
    )


def _extract_target_urls(goal: dict) -> list[str]:
    """Extract monitoring target URLs from goal metadata."""
    urls = []

    # Explicit target_urls list
    target_urls = goal.get("target_urls") or []
    if isinstance(target_urls, list):
        urls.extend(target_urls)

    # Single target_url
    target_url = goal.get("target_url") or ""
    if target_url and target_url not in urls:
        urls.append(target_url)

    # Parse URLs from description/task_action if no explicit targets
    if not urls:
        import re

        text = f"{goal.get('description', '')} {goal.get('task_action', '')}"
        found = re.findall(r'https?://[^\s<>"\']+', text)
        urls.extend(found)

    return urls


def _process_monitoring_results(
    session_log: list[dict],
    target_urls: list[str],
) -> tuple[str, dict]:
    """Process session log to detect changes for monitored URLs.

    Returns (status, evidence) tuple.
    """
    from remy.core.monitoring_store import monitoring_store

    changes: list[dict] = []
    errors: list[str] = []

    # Find extracted content in session log
    extracted: dict[str, str] = {}
    for entry in session_log or []:
        if entry.get("type") != "tool_call":
            continue
        tool = entry.get("tool", "")
        args = entry.get("args") or {}

        if tool == "extract_content":
            url = args.get("url", "")
            result = entry.get("result", "")
            if isinstance(result, str) and url:
                # Try to parse JSON result
                try:
                    import json

                    parsed = json.loads(result)
                    content = parsed.get("content") or parsed.get("text") or result
                except (json.JSONDecodeError, TypeError, AttributeError):
                    content = result
                extracted[url] = str(content)

        elif tool == "http_get":
            url = args.get("url", "")
            result = entry.get("result", "")
            if url:
                extracted[url] = str(result)[:5000]

    # Check each target URL for changes
    for url in target_urls:
        if url in extracted:
            content = extracted[url]
            if not content or content.startswith("Error") or len(content) < 10:
                event = monitoring_store.record_unreachable(url)
                errors.append(f"{url}: unreachable or empty")
            else:
                event = monitoring_store.check_for_changes(url, content)
                changes.append(
                    {
                        "url": url,
                        "change_type": event.change_type,
                        "similarity": event.similarity,
                        "diff_highlights": event.diff_highlights,
                    }
                )

                # Store significant changes in research memory
                if event.change_type in ("significant_change", "content_removed"):
                    _store_change_finding(url, event)
        else:
            event = monitoring_store.record_unreachable(url)
            errors.append(f"{url}: not fetched")

    # Derive status
    significant = [
        c for c in changes if c["change_type"] in ("significant_change", "content_removed")
    ]
    minor = [c for c in changes if c["change_type"] == "minor_update"]
    no_change = [c for c in changes if c["change_type"] in ("no_change", "new")]

    if significant:
        status = "significant_changes_detected"
    elif minor:
        status = "minor_updates_detected"
    elif errors and not changes:
        status = "all_unreachable"
    elif no_change:
        status = "no_changes"
    else:
        status = "attempted"

    evidence = {
        "targets_checked": len(target_urls),
        "changes": changes,
        "errors": errors,
        "significant_count": len(significant),
        "minor_count": len(minor),
        "no_change_count": len(no_change),
    }

    return status, evidence


def _store_change_finding(url: str, event) -> None:
    """Store a significant change as a research finding."""
    try:
        from remy.core.research_memory import add_finding

        highlights = (
            "; ".join(event.diff_highlights[:3])
            if event.diff_highlights
            else "content changed significantly"
        )
        content = (
            f"Change detected on {url} ({event.change_type}, "
            f"similarity: {event.similarity:.0%}): {highlights}"
        )
        add_finding(
            content=content,
            source_url=url,
            tags=["monitoring", "change-detection", event.change_type],
        )
    except Exception as e:
        logger.debug("Failed to store change finding: %s", e)


def _format_monitoring_report(status: str, evidence: dict) -> str:
    """Format monitoring results as a concise operator report."""
    lines = []

    if status == "no_changes":
        lines.append(f"No changes detected across {evidence['targets_checked']} targets.")
    elif status == "all_unreachable":
        lines.append(f"All {evidence['targets_checked']} targets unreachable.")
    else:
        if evidence["significant_count"]:
            lines.append(f"Significant changes: {evidence['significant_count']}")
        if evidence["minor_count"]:
            lines.append(f"Minor updates: {evidence['minor_count']}")
        if evidence["no_change_count"]:
            lines.append(f"No change: {evidence['no_change_count']}")

    # List changes with highlights
    for change in evidence.get("changes", []):
        if change["change_type"] == "no_change":
            continue
        emoji = {
            "significant_change": "!!",
            "minor_update": "~",
            "new": "+",
            "content_removed": "X",
        }.get(change["change_type"], "?")
        line = f"  [{emoji}] {change['url']}: {change['change_type']}"
        if change.get("diff_highlights"):
            line += f" — {change['diff_highlights'][0]}"
        lines.append(line)

    for err in evidence.get("errors", []):
        lines.append(f"  [ERR] {err}")

    return "\n".join(lines) if lines else "Monitoring check complete."


async def run_monitoring_worker(
    goal: dict,
    session_id: str,
    session_log: list,
    history: list | None = None,
    current_plan: object | None = None,
) -> WorkerExecutionResult:
    """Run the monitoring worker — fetch URLs, detect changes, report."""
    from remy.core.worker import WorkerTask, execute_single_worker

    target_urls = _extract_target_urls(goal)

    if not target_urls:
        return WorkerExecutionResult(
            worker="monitoring_worker",
            status="no_targets",
            response_text="No target URLs found in goal. Add target_urls or URLs in description.",
            history=history or [],
            session_log=session_log or [],
            evidence={"error": "no_targets"},
            tool_calls=0,
        )

    desc = (goal or {}).get("description", "") or ""
    task_action = (goal or {}).get("task_action", "") or ""
    instruction = task_action if task_action else desc

    # Add explicit URL list to instruction
    url_list = "\n".join(f"- {u}" for u in target_urls)
    instruction = f"{instruction}\n\nTARGET URLs to check:\n{url_list}"

    context_parts = []
    resume_context = (goal or {}).get("resume_context", "") or ""
    if resume_context:
        context_parts.append(f"Resume from: {resume_context}")

    # Add last check info
    from remy.core.monitoring_store import monitoring_store

    for url in target_urls:
        status = monitoring_store.get_target_status(url)
        if status.get("monitored"):
            context_parts.append(
                f"Previously checked {url}: {status['check_count']} times, "
                f"last hash: {status.get('last_hash', '?')}"
            )

    task = WorkerTask(
        role="osint",  # reuse osint role — it has web_search + extract_content tools
        instruction=instruction,
        context="\n".join(context_parts),
    )

    result = await execute_single_worker(
        task=task,
        session_id=session_id,
        channel=MONITORING_WORKER_CHANNEL,
        step_budget=goal.get("_pack_step_budget", 0),
        timeout_override=goal.get("_pack_timeout_sec", 0),
    )

    # Process results — compare fetched content against snapshots
    worker_session_log = list(result.session_log or [])
    status, evidence = _process_monitoring_results(worker_session_log, target_urls)

    # Override status from worker engine errors
    if result.status == "timeout":
        status = "timeout"
    elif result.status == "error":
        status = "error"

    report = _format_monitoring_report(status, evidence)

    return WorkerExecutionResult(
        worker="monitoring_worker",
        status=status,
        response_text=report,
        history=history or [],
        session_log=worker_session_log,
        evidence=evidence,
        tool_calls=result.tool_calls,
    )
