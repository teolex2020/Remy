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

    matrix = evidence.get("claim_source_matrix") or {}
    if isinstance(matrix, dict) and matrix.get("claim_count"):
        lines.append(
            "Claim coverage: "
            f"{matrix.get('supported_claims', 0)}/{matrix.get('claim_count', 0)} supported, "
            f"{matrix.get('partial_claims', 0)} partial, "
            f"{matrix.get('unsupported_claims', 0)} unsupported, "
            f"{matrix.get('conflicting_claims', 0)} conflict"
        )
        lines.append(
            "Claim provenance: "
            f"{matrix.get('corroborated_claims', 0)} independently corroborated, "
            f"{matrix.get('false_corroborated_claims', 0)} false corroboration, "
            f"coverage={float(matrix.get('provenance_coverage_rate', 0.0)):.0%}"
        )
        if matrix.get("time_sensitive_claims"):
            lines.append(
                "Temporal evidence: "
                f"{matrix.get('temporally_ready_claims', 0)}/"
                f"{matrix.get('time_sensitive_claims', 0)} current, "
                f"{matrix.get('stale_claims', 0)} stale, "
                f"{matrix.get('undated_temporal_claims', 0)} undated"
            )
        if (
            matrix.get("resolved_temporal_conflicts")
            or matrix.get("superseded_claims")
            or matrix.get("unresolved_contradictions")
        ):
            lines.append(
                "Temporal supersession: "
                f"{matrix.get('resolved_temporal_conflicts', 0)} resolved, "
                f"{matrix.get('unresolved_contradictions', 0)} active conflicts, "
                f"{matrix.get('superseded_claims', 0)} historical claims retained"
            )

    lifecycle = evidence.get("claim_lifecycle") or {}
    lifecycle_summary = (
        lifecycle.get("summary") if isinstance(lifecycle, dict) else {}
    ) or {}
    if lifecycle_summary.get("tracked_subjects"):
        lines.append(
            "Claim lifecycle: "
            f"{lifecycle_summary.get('tracked_subjects', 0)} tracked subjects, "
            f"{lifecycle_summary.get('confirmed_transitions', 0)} confirmed transitions, "
            f"{lifecycle_summary.get('pending_changes', 0)} pending changes"
        )

    schedule = evidence.get("execution_schedule") or {}
    if isinstance(schedule, dict) and schedule.get("required_lanes"):
        lines.append(
            "Execution coverage: "
            f"{schedule.get('executed_lanes', 0)}/{schedule.get('required_lanes', 0)} lanes searched, "
            f"{schedule.get('fetched_lanes', 0)} fetched, "
            f"{schedule.get('distinct_domains', 0)}/{schedule.get('domain_target', 3)} independent domains"
        )

    recovery = evidence.get("same_run_recovery") or {}
    if isinstance(recovery, dict) and recovery.get("should_retry"):
        recovery_state = "resolved" if recovery.get("resolved") else "still incomplete"
        lines.append(
            "Same-run recovery: "
            f"{recovery_state}, {recovery.get('tool_calls', 0)} additional tool calls"
        )

    marginal = evidence.get("marginal_evidence") or {}
    if isinstance(marginal, dict) and marginal.get("rows"):
        lines.append(
            "Marginal evidence: "
            f"{marginal.get('accepted_source_count', 0)} high-gain sources, "
            f"{marginal.get('duplicate_rejected', 0)} duplicates rejected, "
            f"decision={marginal.get('decision', 'continue_fetch')}"
        )

    provenance = evidence.get("source_provenance_graph") or {}
    if isinstance(provenance, dict) and provenance.get("node_count"):
        lines.append(
            "Provenance: "
            f"{provenance.get('independent_root_count', 0)} independent roots, "
            f"{provenance.get('primary_root_count', 0)} primary, "
            f"{provenance.get('syndicated_source_count', 0)} syndicated, "
            f"{provenance.get('derived_source_count', 0)} derived"
        )

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
