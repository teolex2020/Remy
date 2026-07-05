"""
Result Runtime for Remy v3.

Builds normalized post-execution outcome results so apply-layer runtimes do
not manually assemble decision payloads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutcomeResult:
    decision: str = "pause"
    reason: str = ""
    next_action: str = ""


class ResultRuntime:
    """Factory for normalized outcome results."""

    @staticmethod
    def initial(reason: str = "") -> OutcomeResult:
        return OutcomeResult(reason=reason)

    @staticmethod
    def complete(reason: str = "all_steps_complete") -> OutcomeResult:
        return OutcomeResult(decision="complete", reason=reason)

    @staticmethod
    def execute_step(reason: str = "", next_action: str = "") -> OutcomeResult:
        return OutcomeResult(decision="execute_step", reason=reason, next_action=next_action)

    @staticmethod
    def pause(reason: str = "") -> OutcomeResult:
        return OutcomeResult(decision="pause", reason=reason)

    @staticmethod
    def escalate(reason: str = "") -> OutcomeResult:
        return OutcomeResult(decision="escalate", reason=reason)

    @staticmethod
    def abort(reason: str = "") -> OutcomeResult:
        return OutcomeResult(decision="abort", reason=reason)

    @staticmethod
    def replan(reason: str = "") -> OutcomeResult:
        return OutcomeResult(decision="replan", reason=reason)
