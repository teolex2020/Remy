"""
Evaluation Engine for Remy v3.

Evaluates execution results to determine success, failure,
and whether replanning is needed.

Phase 6: Enhanced with blocker classification, failure pattern detection,
specialist scoring, and outcome learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..execution.execution_runtime import ExecutionStatus

log = logging.getLogger(__name__)


class EvalVerdict(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    BLOCKED = "blocked"
    NEEDS_REPLAN = "needs_replan"
    INCONCLUSIVE = "inconclusive"


class BlockerType(str, Enum):
    """Classification of what blocked execution."""
    CAPTCHA = "captcha"
    EMAIL_VERIFY = "email_verification"
    PAYMENT_REQUIRED = "payment_required"
    KYC = "kyc"
    RATE_LIMIT = "rate_limit"
    AUTH_REQUIRED = "auth_required"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class EvalResult:
    """Result of evaluating an execution outcome."""
    verdict: EvalVerdict = EvalVerdict.INCONCLUSIVE
    confidence: float = 0.0
    reason: str = ""
    evidence_quality: str = "none"  # none, weak, moderate, strong
    criteria_met: int = 0
    criteria_total: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    should_replan: bool = False
    should_continue: bool = True
    blocker_type: BlockerType | None = None
    is_repeated_failure: bool = False
    unsupported_observed_claims: int = 0
    factuality_penalty: float = 0.0


# Blocker patterns: keyword → BlockerType
_BLOCKER_PATTERNS = {
    "captcha": BlockerType.CAPTCHA,
    "recaptcha": BlockerType.CAPTCHA,
    "verify your email": BlockerType.EMAIL_VERIFY,
    "email verification": BlockerType.EMAIL_VERIFY,
    "payment required": BlockerType.PAYMENT_REQUIRED,
    "upgrade your plan": BlockerType.PAYMENT_REQUIRED,
    "kyc": BlockerType.KYC,
    "identity verification": BlockerType.KYC,
    "rate limit": BlockerType.RATE_LIMIT,
    "too many requests": BlockerType.RATE_LIMIT,
    "429": BlockerType.RATE_LIMIT,
    "login required": BlockerType.AUTH_REQUIRED,
    "sign in": BlockerType.AUTH_REQUIRED,
    "unauthorized": BlockerType.AUTH_REQUIRED,
    "401": BlockerType.AUTH_REQUIRED,
    "not found": BlockerType.NOT_FOUND,
    "404": BlockerType.NOT_FOUND,
    "timeout": BlockerType.TIMEOUT,
    "timed out": BlockerType.TIMEOUT,
    "connection error": BlockerType.NETWORK,
    "network error": BlockerType.NETWORK,
}


class EvaluationEngine:
    """Evaluates execution outcomes against success criteria."""

    def __init__(self):
        self._v2_criteria = None
        self._failure_history: list[dict[str, Any]] = []
        self._specialist_scores: dict[str, dict[str, int]] = {}

    def _get_v2_criteria(self):
        if self._v2_criteria is None:
            try:
                from remy.core import success_criteria
                self._v2_criteria = success_criteria
            except ImportError:
                log.warning("v2 success_criteria not available")
        return self._v2_criteria

    def evaluate(
        self,
        execution_result,
        success_criteria: list[dict] | None = None,
        session_log: list[dict] | None = None,
        goal_id: str = "",
        specialist: str = "",
        unsupported_observed_claims: int = 0,
        blocker_history_summary: dict[str, Any] | None = None,
        approval_state: dict[str, Any] | None = None,
        source_link_completeness: float | None = None,
        budget_pressure_snapshot: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate an execution result."""
        status = getattr(execution_result, "status", None)
        response = getattr(execution_result, "response", "")
        tool_calls = getattr(execution_result, "tool_calls", 0)
        evidence = getattr(execution_result, "evidence", {}) or {}

        # Check for repeated failure pattern
        is_repeated = self._check_repeated_failure(goal_id)

        # Quick exits
        if status == ExecutionStatus.BLOCKED:
            blocker = self._classify_blocker(response)
            result = EvalResult(
                verdict=EvalVerdict.BLOCKED,
                confidence=0.9,
                reason="Execution blocked by external factor",
                should_replan=True,
                should_continue=False,
                blocker_type=blocker,
                is_repeated_failure=is_repeated,
            )
            result = self._apply_factuality_penalty(result, unsupported_observed_claims)
            result = self._apply_runtime_quality_signals(
                result,
                blocker_history_summary=blocker_history_summary,
                approval_state=approval_state,
                source_link_completeness=source_link_completeness,
                budget_pressure_snapshot=budget_pressure_snapshot,
            )
            self._record_outcome(goal_id, specialist, result)
            return result

        if status == ExecutionStatus.ERROR:
            result = EvalResult(
                verdict=EvalVerdict.FAILURE,
                confidence=0.9,
                reason=getattr(execution_result, "error", "Unknown error"),
                should_replan=True,
                is_repeated_failure=is_repeated,
            )
            result = self._apply_factuality_penalty(result, unsupported_observed_claims)
            result = self._apply_runtime_quality_signals(
                result,
                blocker_history_summary=blocker_history_summary,
                approval_state=approval_state,
                source_link_completeness=source_link_completeness,
                budget_pressure_snapshot=budget_pressure_snapshot,
            )
            self._record_outcome(goal_id, specialist, result)
            return result

        if status == ExecutionStatus.TIMEOUT:
            if tool_calls > 0:
                result = EvalResult(
                    verdict=EvalVerdict.PARTIAL,
                    confidence=0.5,
                    reason=f"Timed out but made {tool_calls} tool calls",
                    should_continue=True,
                    is_repeated_failure=is_repeated,
                )
            else:
                result = EvalResult(
                    verdict=EvalVerdict.FAILURE,
                    confidence=0.7,
                    reason="Timed out with no tool calls",
                    should_replan=True,
                    blocker_type=BlockerType.TIMEOUT,
                    is_repeated_failure=is_repeated,
                )
            result = self._apply_factuality_penalty(result, unsupported_observed_claims)
            result = self._apply_runtime_quality_signals(
                result,
                blocker_history_summary=blocker_history_summary,
                approval_state=approval_state,
                source_link_completeness=source_link_completeness,
                budget_pressure_snapshot=budget_pressure_snapshot,
            )
            self._record_outcome(goal_id, specialist, result)
            return result

        # If we have programmatic success criteria, verify them
        if success_criteria:
            result = self._verify_criteria(
                success_criteria, session_log, execution_result
            )
            result = self._apply_factuality_penalty(result, unsupported_observed_claims)
            result = self._apply_runtime_quality_signals(
                result,
                blocker_history_summary=blocker_history_summary,
                approval_state=approval_state,
                source_link_completeness=source_link_completeness,
                budget_pressure_snapshot=budget_pressure_snapshot,
            )
            result.is_repeated_failure = is_repeated
            self._record_outcome(goal_id, specialist, result)
            return result

        evidence_result = self._evaluate_evidence(status, tool_calls, evidence)
        if evidence_result is not None:
            evidence_result = self._apply_factuality_penalty(
                evidence_result,
                unsupported_observed_claims,
            )
            evidence_result = self._apply_runtime_quality_signals(
                evidence_result,
                blocker_history_summary=blocker_history_summary,
                approval_state=approval_state,
                source_link_completeness=source_link_completeness,
                budget_pressure_snapshot=budget_pressure_snapshot,
            )
            evidence_result.is_repeated_failure = is_repeated
            self._record_outcome(goal_id, specialist, evidence_result)
            return evidence_result

        # Check for blockers in response text
        blocker = self._classify_blocker(response)
        if blocker and blocker != BlockerType.UNKNOWN:
            result = EvalResult(
                verdict=EvalVerdict.BLOCKED,
                confidence=0.7,
                reason=f"Detected blocker: {blocker.value}",
                blocker_type=blocker,
                should_replan=True,
                should_continue=False,
                is_repeated_failure=is_repeated,
            )
            result = self._apply_factuality_penalty(result, unsupported_observed_claims)
            result = self._apply_runtime_quality_signals(
                result,
                blocker_history_summary=blocker_history_summary,
                approval_state=approval_state,
                source_link_completeness=source_link_completeness,
                budget_pressure_snapshot=budget_pressure_snapshot,
            )
            self._record_outcome(goal_id, specialist, result)
            return result

        # Heuristic evaluation
        if tool_calls > 0 and status == ExecutionStatus.SUCCESS:
            result = EvalResult(
                verdict=EvalVerdict.SUCCESS,
                confidence=0.7,
                reason=f"Completed with {tool_calls} tool calls",
                evidence_quality="moderate",
                should_continue=True,
            )
        elif tool_calls > 0:
            result = EvalResult(
                verdict=EvalVerdict.PARTIAL,
                confidence=0.5,
                reason=f"Partial progress: {tool_calls} tool calls",
                evidence_quality="weak",
                should_continue=True,
                is_repeated_failure=is_repeated,
            )
        else:
            result = EvalResult(
                verdict=EvalVerdict.FAILURE,
                confidence=0.6,
                reason="No tool calls executed",
                should_replan=True,
                is_repeated_failure=is_repeated,
            )

        result = self._apply_factuality_penalty(result, unsupported_observed_claims)
        result = self._apply_runtime_quality_signals(
            result,
            blocker_history_summary=blocker_history_summary,
            approval_state=approval_state,
            source_link_completeness=source_link_completeness,
            budget_pressure_snapshot=budget_pressure_snapshot,
        )
        self._record_outcome(goal_id, specialist, result)
        return result

    def _evaluate_evidence(
        self,
        status,
        tool_calls: int,
        evidence: dict[str, Any],
    ) -> EvalResult | None:
        """Deterministic evidence checks win before heuristic text reading."""
        from ..research.research_policy import assess_evidence_dict

        findings = evidence.get("findings") or []
        artifacts = evidence.get("artifacts") or []
        assessment = assess_evidence_dict(evidence)

        if assessment.is_success:
            return EvalResult(
                verdict=EvalVerdict.SUCCESS,
                confidence=0.9,
                reason=assessment.reason,
                evidence_quality="strong",
                should_continue=False,
            )

        if assessment.is_partial:
            return EvalResult(
                verdict=EvalVerdict.PARTIAL,
                confidence=0.7,
                reason=assessment.reason,
                evidence_quality="moderate",
                should_continue=True,
            )

        if tool_calls == 0 and status != ExecutionStatus.SUCCESS:
            return EvalResult(
                verdict=EvalVerdict.FAILURE,
                confidence=0.8,
                reason="No evidence, artifacts, or tool activity",
                should_replan=True,
            )
        return None

    def _apply_factuality_penalty(
        self,
        result: EvalResult,
        unsupported_observed_claims: int,
    ) -> EvalResult:
        if unsupported_observed_claims <= 0:
            return result

        penalty = min(0.3, unsupported_observed_claims * 0.12)
        result.unsupported_observed_claims = unsupported_observed_claims
        result.factuality_penalty = penalty
        result.confidence = max(0.0, result.confidence - penalty)
        result.details.append({
            "type": "factuality",
            "unsupported_observed_claims": unsupported_observed_claims,
            "penalty": round(penalty, 3),
        })
        if result.evidence_quality == "strong":
            result.evidence_quality = "moderate"
        elif result.evidence_quality == "moderate":
            result.evidence_quality = "weak"
        if result.reason:
            result.reason = (
                f"{result.reason}; factuality caution: "
                f"{unsupported_observed_claims} unsupported observed claim(s)"
            )
        return result

    def _apply_runtime_quality_signals(
        self,
        result: EvalResult,
        *,
        blocker_history_summary: dict[str, Any] | None,
        approval_state: dict[str, Any] | None,
        source_link_completeness: float | None,
        budget_pressure_snapshot: dict[str, Any] | None,
    ) -> EvalResult:
        penalty = 0.0

        recent_failures = int((blocker_history_summary or {}).get("recent_failures", 0) or 0)
        blocker_reason = (blocker_history_summary or {}).get("blocker_reason", "") or ""
        if recent_failures >= 2:
            applied = min(0.12, 0.03 * recent_failures)
            penalty += applied
            result.details.append({
                "type": "blocker_history",
                "recent_failures": recent_failures,
                "blocker_reason": blocker_reason,
            })
            if result.verdict in (EvalVerdict.PARTIAL, EvalVerdict.FAILURE, EvalVerdict.BLOCKED):
                result.should_replan = True

        pending_approvals = int((approval_state or {}).get("pending_approvals", 0) or 0)
        if pending_approvals:
            result.details.append({
                "type": "approval_state",
                "pending_approvals": pending_approvals,
                "task_requires_approval": bool((approval_state or {}).get("task_requires_approval", False)),
            })

        if source_link_completeness is not None and source_link_completeness < 1.0:
            applied = min(0.12, (1.0 - float(source_link_completeness)) * 0.12)
            penalty += applied
            result.details.append({
                "type": "source_links",
                "completeness": round(float(source_link_completeness), 3),
            })

        budget_status = (budget_pressure_snapshot or {}).get("status", "") or ""
        if budget_status in {"warning", "critical", "exhausted"}:
            applied = 0.04 if budget_status == "warning" else 0.08
            penalty += applied
            result.details.append({
                "type": "budget_pressure",
                "status": budget_status,
                "daily_remaining_usd": (budget_pressure_snapshot or {}).get("daily_remaining_usd", 0.0),
                "recommended_model": (budget_pressure_snapshot or {}).get("recommended_model", ""),
            })

        if penalty <= 0.0:
            return result

        result.confidence = max(0.0, result.confidence - min(0.2, penalty))
        if result.evidence_quality == "strong" and penalty >= 0.08:
            result.evidence_quality = "moderate"
        elif result.evidence_quality == "moderate" and penalty >= 0.08:
            result.evidence_quality = "weak"
        elif result.evidence_quality == "weak" and penalty >= 0.12:
            result.evidence_quality = "none"
        if result.reason:
            result.reason = f"{result.reason}; runtime quality pressure {min(0.2, penalty):.2f}"
        return result

    def _verify_criteria(
        self,
        criteria: list[dict],
        session_log: list[dict] | None,
        execution_result,
    ) -> EvalResult:
        """Verify programmatic success criteria via v2 engine."""
        v2 = self._get_v2_criteria()
        if v2 is None:
            return EvalResult(
                verdict=EvalVerdict.INCONCLUSIVE,
                reason="Criteria engine not available",
            )

        met, total, details = v2.verify_criteria(criteria, session_log)

        if met == total and total > 0:
            return EvalResult(
                verdict=EvalVerdict.SUCCESS,
                confidence=0.9,
                reason=f"All {total} criteria met",
                evidence_quality="strong",
                criteria_met=met,
                criteria_total=total,
                details=details,
                should_continue=False,
            )

        if met > 0:
            return EvalResult(
                verdict=EvalVerdict.PARTIAL,
                confidence=0.6,
                reason=f"{met}/{total} criteria met",
                evidence_quality="moderate",
                criteria_met=met,
                criteria_total=total,
                details=details,
                should_continue=True,
            )

        return EvalResult(
            verdict=EvalVerdict.FAILURE,
            confidence=0.7,
            reason=f"0/{total} criteria met",
            evidence_quality="none",
            criteria_met=0,
            criteria_total=total,
            details=details,
            should_replan=True,
        )

    # -------------------------------------------------------------------
    # Pattern detection
    # -------------------------------------------------------------------

    def _classify_blocker(self, text: str) -> BlockerType | None:
        """Classify what type of blocker was encountered."""
        if not text:
            return None
        text_lower = text.lower()
        for pattern, blocker_type in _BLOCKER_PATTERNS.items():
            if pattern in text_lower:
                return blocker_type
        return None

    def _check_repeated_failure(self, goal_id: str) -> bool:
        """Check if this goal has failed recently (last 3 attempts)."""
        if not goal_id:
            return False
        recent = [
            h for h in self._failure_history[-10:]
            if h.get("goal_id") == goal_id
        ]
        return len(recent) >= 2

    def _record_outcome(self, goal_id: str, specialist: str, result: EvalResult):
        """Track outcomes for pattern detection."""
        if goal_id and result.verdict in (EvalVerdict.FAILURE, EvalVerdict.BLOCKED):
            self._failure_history.append({
                "goal_id": goal_id,
                "specialist": specialist,
                "verdict": result.verdict.value,
                "blocker": result.blocker_type.value if result.blocker_type else None,
            })
            if len(self._failure_history) > 100:
                self._failure_history = self._failure_history[-100:]

        # Specialist scoring
        if specialist:
            if specialist not in self._specialist_scores:
                self._specialist_scores[specialist] = {
                    "success": 0,
                    "failure": 0,
                    "total": 0,
                    "unsupported_claims": 0,
                    "factuality_penalty": 0.0,
                }
            self._specialist_scores[specialist]["total"] += 1
            self._specialist_scores[specialist]["unsupported_claims"] += result.unsupported_observed_claims
            self._specialist_scores[specialist]["factuality_penalty"] += result.factuality_penalty
            if result.verdict == EvalVerdict.SUCCESS:
                self._specialist_scores[specialist]["success"] += 1
            elif result.verdict in (EvalVerdict.FAILURE, EvalVerdict.BLOCKED):
                self._specialist_scores[specialist]["failure"] += 1

    def specialist_success_rate(self, specialist: str) -> float:
        """Get success rate for a specialist."""
        scores = self._specialist_scores.get(specialist)
        if not scores or scores["total"] == 0:
            return 0.5  # No data — neutral
        base = scores["success"] / scores["total"]
        quality_penalty = min(0.35, scores.get("factuality_penalty", 0.0) / scores["total"])
        return max(0.0, base - quality_penalty)

    def failure_count_for_goal(self, goal_id: str) -> int:
        """Count recent failures for a specific goal."""
        return sum(
            1 for h in self._failure_history
            if h.get("goal_id") == goal_id
        )

    def summary(self) -> dict[str, Any]:
        """Evaluation summary for observability."""
        return {
            "failure_history_size": len(self._failure_history),
            "specialist_scores": {
                k: {
                    "success_rate": round(v["success"] / v["total"], 2) if v["total"] else 0,
                    "quality_adjusted_success_rate": round(
                        max(0.0, (v["success"] / v["total"]) - min(0.35, v.get("factuality_penalty", 0.0) / v["total"])),
                        2,
                    ) if v["total"] else 0,
                    **v,
                }
                for k, v in self._specialist_scores.items()
            },
        }
