"""Reporter layer that converts worker results into concise operator summaries."""

from __future__ import annotations

from remy.core.workers.contracts import WorkerExecutionResult


def format_worker_report(result: WorkerExecutionResult | None, fallback_text: str = "") -> str:
    """Convert a worker result into a short operator-facing report."""
    if not result:
        return fallback_text

    if result.worker == "research_worker":
        return _format_research_report(result, fallback_text)

    return _format_browser_report(result, fallback_text)


def _format_browser_report(result: WorkerExecutionResult, fallback_text: str = "") -> str:
    """Format a browser worker result."""
    evidence = result.evidence if isinstance(result.evidence, dict) else {}
    lines = [f"Status: {result.status}"]

    pack_id = evidence.get("capability_pack") or ""
    publisher_mode = evidence.get("publisher_mode") or ""
    publisher_channel = evidence.get("publisher_channel") or ""
    approval_mode = evidence.get("approval_mode") or ""

    if pack_id == "publisher":
        mode_bits = []
        if publisher_mode:
            mode_bits.append(f"mode={publisher_mode}")
        if publisher_channel:
            mode_bits.append(f"channel={publisher_channel}")
        if mode_bits:
            lines.append(f"Draft target: {' | '.join(mode_bits)}")

    current_url = evidence.get("current_url") or evidence.get("url") or ""
    if current_url:
        lines.append(f"URL: {current_url}")

    page_state = evidence.get("page_state") or ""
    if page_state:
        lines.append(f"Page state: {page_state}")

    visible_error = evidence.get("visible_error_text") or ""
    if visible_error:
        lines.append(f"Evidence: {visible_error}")
    elif fallback_text:
        first_line = fallback_text.strip().splitlines()[0][:220]
        lines.append(f"Evidence: {first_line}")
    else:
        lines.append("Evidence: No explicit evidence captured.")

    if pack_id == "publisher":
        if result.status == "verified":
            lines.append(
                "Next step: review the saved draft or queued action before any publish approval."
            )
        elif result.status == "blocked_external":
            lines.append(
                "Next step: resolve the blocker, then resume at the draft/approval checkpoint."
            )
        elif approval_mode:
            lines.append(
                "Next step: keep this in draft mode or queue it for approval before any live publish."
            )
        else:
            lines.append("Next step: refine the draft path and avoid any live publish action.")
    elif result.status == "verified":
        lines.append("Next step: continue to the next task stage.")
    elif result.status == "blocked_external":
        lines.append(
            "Next step: resolve the external blocker, then resume from the current checkpoint."
        )
    elif result.status == "attempted":
        lines.append("Next step: retry with corrected input or a different browser path.")
    else:
        lines.append("Next step: inspect the latest browser state before retrying.")

    return "\n".join(lines)


def _format_research_report(result: WorkerExecutionResult, fallback_text: str = "") -> str:
    """Format a research worker result."""
    evidence = result.evidence if isinstance(result.evidence, dict) else {}
    lines = [f"Status: {result.status}"]

    findings_count = evidence.get("findings_count", 0)
    if findings_count:
        lines.append(f"Findings: {findings_count}")

    queries = evidence.get("queries", [])
    if queries:
        lines.append(f"Queries: {', '.join(queries[:5])}")

    sources = evidence.get("sources", [])
    if sources:
        lines.append(f"Sources: {', '.join(sources[:5])}")

    project_id = evidence.get("project_id", "")
    if project_id:
        lines.append(f"Project: {project_id}")

    if result.response_text:
        summary_lines = [
            ln.strip()
            for ln in result.response_text.strip().splitlines()
            if ln.strip() and not ln.strip().startswith("[")
        ][:3]
        if summary_lines:
            lines.append(f"Summary: {' '.join(summary_lines)[:300]}")

    if result.status == "completed":
        lines.append("Next step: review the research report artifact.")
    elif result.status in ("findings_collected", "partial_progress", "searching"):
        lines.append("Next step: continue research and synthesize the stored findings.")
    elif result.status == "timeout":
        lines.append("Next step: resume research from the last query.")
    else:
        lines.append("Next step: start or continue the research project.")

    return "\n".join(lines)
