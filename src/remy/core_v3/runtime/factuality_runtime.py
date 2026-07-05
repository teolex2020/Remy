"""Applies factuality/verification policy to v3 execution outputs."""

from __future__ import annotations

from remy.core.factuality import enforce_factuality


class FactualityRuntime:
    """Normalize execution responses so unsupported observation claims are visible and corrected."""

    def apply(self, exec_result):
        response, report = enforce_factuality(exec_result.response or "", exec_result.session_log or [])
        exec_result.response = response
        exec_result.unsupported_observed_claims = report.unsupported_observed_claims
        exec_result.had_external_evidence = report.had_external_evidence
        exec_result.factuality_modified = report.modified
        exec_result.factuality_report = report
        return exec_result
