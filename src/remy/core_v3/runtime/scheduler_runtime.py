"""
Scheduler runtime for Remy v3.

Owns mission discovery and prioritization so the autonomy loop does not manage
mission lists directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from ..missions.mission_models import Mission, MissionStatus


@dataclass
class MissionSelection:
    mission: Mission | None
    reason: str = ""
    runnable_count: int = 0
    score: float = 0.0
    details: dict | None = None


class SchedulerRuntime:
    """Deterministic mission selection for the loop."""

    def __init__(
        self,
        mission_query_runtime,
        projection_runtime,
        mission_state_runtime=None,
        loop_runtime=None,
        evaluator=None,
        ops_query_runtime=None,
    ):
        self.mission_query_runtime = mission_query_runtime
        self.projection_runtime = projection_runtime
        self.mission_state_runtime = mission_state_runtime
        self.loop_runtime = loop_runtime
        self.evaluator = evaluator
        self.ops_query_runtime = ops_query_runtime
        self._recent_decisions: list[dict] = []

    def recent_decisions(self, limit: int = 10) -> list[dict]:
        return self._recent_decisions[-limit:]

    def _record_decision(self, *, mission: Mission | None, reason: str, runnable_count: int, score: float, details: dict | None = None):
        self._recent_decisions.append({
            "mission_id": mission.id if mission else "",
            "reason": reason,
            "runnable_count": runnable_count,
            "score": round(score, 3),
            "details": details or {},
            "timestamp": time.time(),
        })
        if len(self._recent_decisions) > 20:
            self._recent_decisions = self._recent_decisions[-20:]

    def _budget_health(self, mission: Mission) -> float:
        budget = getattr(mission, "budget", None)
        if budget is None:
            return 0.0
        token_limit = max(float(getattr(budget, "tokens", 0) or 0), 1.0)
        expected = float(getattr(mission, "total_cost_usd", 0.0) or 0.0)
        if expected <= 0.0:
            return 0.0
        # Missions already burning cost get a slight penalty.
        return min(0.2, expected / max(token_limit / 1000.0, 1.0))

    def _specialist_quality_factor(self, summary: dict, mission: Mission) -> float:
        if self.evaluator is None:
            return 0.0
        task = summary.get("current_task") or {}
        text = " ".join(
            part for part in (
                mission.description,
                task.get("action", ""),
                task.get("blocker_reason", ""),
            )
            if part
        ).lower()
        if not any(keyword in text for keyword in (
            "verify",
            "research",
            "source",
            "profile",
            "identity",
            "github",
            "website",
            "browser",
            "comment",
            "social",
        )):
            return 0.0
        candidates = ("researcher", "analyst", "executor")
        return max(self.evaluator.specialist_success_rate(candidate) for candidate in candidates) - 0.5

    def _specialist_quality(self, specialist_id: str) -> dict[str, float | int]:
        if self.ops_query_runtime is not None and hasattr(self.ops_query_runtime, "specialist_quality"):
            quality = self.ops_query_runtime.specialist_quality(specialist_id) or {}
            return {
                "success_rate": float(quality.get("success_rate", 0.5) or 0.5),
                "quality_adjusted_success_rate": float(
                    quality.get("quality_adjusted_success_rate", quality.get("success_rate", 0.5)) or 0.5
                ),
                "unsupported_claims": int(quality.get("unsupported_claims", 0) or 0),
            }
        if self.evaluator is None:
            return {
                "success_rate": 0.5,
                "quality_adjusted_success_rate": 0.5,
                "unsupported_claims": 0,
            }
        raw = float(self.evaluator.specialist_success_rate(specialist_id))
        details = self.evaluator.summary().get("specialist_scores", {}).get(specialist_id, {})
        return {
            "success_rate": raw,
            "quality_adjusted_success_rate": float(details.get("quality_adjusted_success_rate", raw) or raw),
            "unsupported_claims": int(details.get("unsupported_claims", 0) or 0),
        }

    def _candidate_specialists(self, summary: dict, mission: Mission) -> list[str]:
        current_task = summary.get("current_task") or {}
        current_step = summary.get("current_plan_step") or {}
        text = " ".join(
            part for part in (
                mission.description,
                current_task.get("action", ""),
                current_task.get("blocker_reason", ""),
                current_step.get("instruction", ""),
            )
            if part
        ).lower()
        if any(keyword in text for keyword in (
            "research",
            "verify",
            "source",
            "profile",
            "identity",
            "evidence",
            "filing",
            "website",
            "github",
        )):
            return ["researcher", "analyst", "executor"]
        if any(keyword in text for keyword in (
            "browse",
            "signup",
            "register",
            "page",
            "browser",
            "click",
            "submit",
            "form",
        )):
            return ["executor", "researcher", "analyst"]
        if any(keyword in text for keyword in ("analyze", "compare", "summary", "synthesize")):
            return ["analyst", "researcher", "executor"]
        return ["researcher", "analyst", "executor"]

    def _routing_pressure_factor(self, summary: dict, mission: Mission) -> tuple[float, str]:
        candidates = self._candidate_specialists(summary, mission)
        if not candidates:
            return 0.0, ""
        preferred_id = candidates[0]
        preferred = self._specialist_quality(preferred_id)
        adjusted = float(preferred["quality_adjusted_success_rate"])
        raw = float(preferred["success_rate"])
        unsupported = int(preferred["unsupported_claims"])
        debt = max(0.0, raw - adjusted)
        degraded = debt >= 0.15 or unsupported >= 2
        if degraded:
            penalty = -0.35 - min(0.25, debt) - min(0.2, unsupported * 0.05)
            return penalty, f"routing_avoid={preferred_id}"
        if adjusted >= 0.7:
            bonus = min(0.35, (adjusted - 0.7) * 0.5 + 0.1)
            return bonus, f"routing_prefer={preferred_id}"
        return 0.0, ""

    def _mission_score(self, mission: Mission) -> tuple[float, str, dict]:
        summary = self.projection_runtime.mission_summary(mission)
        score = 0.0
        reasons: list[str] = []
        details: dict[str, object] = {
            "mission_id": mission.id,
            "mission_description": mission.description[:100],
            "priority": int(getattr(mission, "priority", 0) or 0),
        }

        if summary.get("current_task"):
            score += 2.5
            reasons.append("runnable_task")
            details["current_task"] = (summary.get("current_task") or {}).get("action", "")
        else:
            reasons.append("no_current_task")

        if mission.immortal:
            score += 1.0
            reasons.append("immortal_bias")

        priority = max(0, int(getattr(mission, "priority", 0) or 0))
        score += priority * 0.2
        if priority:
            reasons.append(f"priority={priority}")

        current_task = summary.get("current_task") or {}
        blocker = current_task.get("blocker_reason", "") or ""
        if blocker:
            score -= 0.6
            reasons.append("blocker_pressure")

        stuck_pressure = self.loop_runtime.pressure_for(mission.id) if self.loop_runtime is not None else 0
        if stuck_pressure:
            score -= min(2.0, stuck_pressure * 0.25)
            reasons.append(f"stuck={stuck_pressure}")

        budget_penalty = self._budget_health(mission)
        if budget_penalty:
            score -= budget_penalty
            reasons.append("budget_pressure")

        quality_factor = self._specialist_quality_factor(summary, mission)
        if quality_factor:
            score += quality_factor
            reasons.append(f"quality={quality_factor:.2f}")
            details["quality_factor"] = round(quality_factor, 3)

        routing_factor, routing_reason = self._routing_pressure_factor(summary, mission)
        if routing_factor:
            score += routing_factor
            reasons.append(f"{routing_reason}:{routing_factor:.2f}")
        details["routing_factor"] = round(routing_factor, 3)
        details["routing_reason"] = routing_reason
        details["candidate_specialists"] = self._candidate_specialists(summary, mission)

        updated_at = float(getattr(mission, "updated_at", 0.0) or 0.0)
        score += min(0.5, max(0.0, time.time() - updated_at) / 86400.0)
        details["final_score"] = round(score, 3)
        return score, ",".join(reasons), details

    def next_mission(self) -> MissionSelection:
        runnable: list[Mission] = []
        for mission in self.mission_query_runtime.all_missions():
            if mission.status == MissionStatus.ACTIVE:
                runnable.append(mission)
            elif self.mission_state_runtime and self.mission_state_runtime.is_schedulable(mission):
                runnable.append(self.mission_state_runtime.activate_for_execution(mission))
            elif mission.status == MissionStatus.PLANNING:
                runnable.append(mission)

        if not runnable:
            selection = MissionSelection(
                mission=None,
                reason="no_runnable_missions",
                runnable_count=0,
                details={},
            )
            self._record_decision(mission=None, reason=selection.reason, runnable_count=0, score=0.0, details={})
            return selection

        scored: list[tuple[float, str, dict, Mission]] = []
        for mission in runnable:
            score, reason, details = self._mission_score(mission)
            scored.append((score, reason, details, mission))

        scored.sort(key=lambda item: (
            -item[0],
            -(int(getattr(item[3], "priority", 0) or 0)),
            -(float(getattr(item[3], "updated_at", 0.0) or 0.0)),
            item[3].id,
        ))
        score, reason, details, mission = scored[0]
        selection = MissionSelection(
            mission=mission,
            reason=reason or "selected",
            runnable_count=len(runnable),
            score=score,
            details=details,
        )
        self._record_decision(
            mission=mission,
            reason=selection.reason,
            runnable_count=selection.runnable_count,
            score=selection.score,
            details=selection.details,
        )
        return selection
