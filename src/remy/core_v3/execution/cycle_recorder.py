"""
Cycle Recorder for Remy v3.

Records each execution cycle result to v2 execution_log
and stores outcomes in Aura memory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CycleRecord:
    """Complete record of a single execution cycle."""
    cycle_num: int = 0
    timestamp: float = field(default_factory=time.time)
    mission_id: str = ""
    goal_id: str = ""
    goal_description: str = ""
    plan_id: str = ""
    step_id: str = ""
    specialist: str = ""

    # Result
    status: str = ""               # success, partial, failure, timeout, blocked
    verdict: str = ""              # eval verdict
    confidence: float = 0.0
    reason: str = ""

    # Cost
    duration_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    model: str = ""
    fallback_used: bool = False

    # Flags
    verified: bool = False
    repeated_failure: bool = False
    memory_assisted: bool = False

    # Chief decision
    decision: str = ""             # execute_step, replan, pause, complete, abort
    planned_action: str = ""       # task/step action that was actually attempted
    unsupported_observed_claims: int = 0
    factuality_report: Any = None


class CycleRecorder:
    """Records cycle outcomes to both v2 execution_log and v3 memory."""

    def __init__(self):
        self._records: list[CycleRecord] = []
        self._v2_log = None

    def _get_v2_log(self):
        if self._v2_log is None:
            try:
                from remy.core.execution_log import (
                    record_cycle_execution, execution_log
                )
                self._v2_log = record_cycle_execution
            except ImportError:
                log.debug("v2 execution_log not available")
        return self._v2_log

    def record(self, rec: CycleRecord):
        """Record a cycle outcome."""
        self._records.append(rec)

        # Also write to v2 execution_log for dashboard compatibility
        self._write_v2(rec)

        # Store outcome in memory
        self._store_outcome(rec)

        log.info(
            "Cycle %d: %s [%s] specialist=%s cost=$%.4f decision=%s",
            rec.cycle_num, rec.status, rec.verdict,
            rec.specialist, rec.cost_usd, rec.decision,
        )

    def _write_v2(self, rec: CycleRecord):
        """Write to v2 execution_log for backwards compatibility."""
        v2_record = self._get_v2_log()
        if v2_record is None:
            return

        # Build a minimal goal dict for v2
        goal_dict = {
            "goal_id": rec.goal_id,
            "description": rec.goal_description,
            "metadata": {
                "status": "active",
                "mission_id": rec.mission_id,
            },
        } if rec.goal_id else None

        # Build minimal evaluation dict
        evaluation = {
            "success": rec.status in ("success", "partial_progress"),
            "confidence": rec.confidence,
            "reason": rec.reason,
            "goal_completed": rec.verified,
            "unsupported_observed_claims": rec.unsupported_observed_claims,
        }

        try:
            v2_record(
                cycle_num=rec.cycle_num,
                goal=goal_dict,
                worker_result=None,
                session_log=None,
                evaluation=evaluation,
                duration_ms=rec.duration_ms,
                tokens_used=rec.tokens_used,
                cost_usd=rec.cost_usd,
                turn_class="productive" if evaluation["success"] else "idle",
                verified=rec.verified,
                repeated_failure=rec.repeated_failure,
                memory_assisted=rec.memory_assisted,
            )
        except Exception as e:
            log.debug("Failed to write v2 execution_log: %s", e)

    def _store_outcome(self, rec: CycleRecord):
        """Store cycle outcome in Aura memory."""
        if not rec.goal_id:
            return
        try:
            from ..memory.memory_api import get_memory
            memory = get_memory()

            consequence = "INCONCLUSIVE"
            trust = 0
            if rec.status == "success":
                consequence = "SUPPORTS"
                trust = 1
            elif rec.status in ("failure", "blocked", "timeout"):
                consequence = "REFUTES"
                trust = -1

            action = rec.planned_action or rec.decision or rec.specialist or "cycle"

            memory.capture_consequence(
                situation=rec.goal_description[:240],
                action=action,
                consequence=consequence,
                trust=trust,
                scope=[
                    f"mission:{rec.mission_id}" if rec.mission_id else "mission:",
                    f"goal:{rec.goal_id}" if rec.goal_id else "goal:",
                    f"step:{rec.step_id}" if rec.step_id else "step:",
                    f"specialist:{rec.specialist}" if rec.specialist else "specialist:",
                    f"model:{rec.model}" if rec.model else "model:",
                    f"fallback:{str(rec.fallback_used).lower()}",
                    f"status:{rec.status}" if rec.status else "status:",
                    f"decision:{rec.decision}" if rec.decision else "decision:",
                ],
                provenance=[
                    "remy:cycle_recorder",
                    f"cycle:{rec.cycle_num}",
                    f"verdict:{rec.verdict}" if rec.verdict else "verdict:",
                ],
                links={
                    "mission": rec.mission_id,
                    "goal": rec.goal_id,
                    "step": rec.step_id,
                },
                namespace="remy",
            )
            if rec.model:
                memory.capture_consequence(
                    situation=self._model_situation(rec),
                    action=action,
                    consequence=consequence,
                    trust=trust,
                    scope=[
                        "model-outcome",
                        f"model:{rec.model}",
                        f"fallback:{str(rec.fallback_used).lower()}",
                        f"specialist:{rec.specialist}" if rec.specialist else "specialist:",
                        f"status:{rec.status}" if rec.status else "status:",
                        f"verdict:{rec.verdict}" if rec.verdict else "verdict:",
                    ],
                    provenance=[
                        "remy:model_outcome",
                        f"cycle:{rec.cycle_num}",
                        f"mission:{rec.mission_id}" if rec.mission_id else "mission:",
                    ],
                    links={
                        "mission": rec.mission_id,
                        "goal": rec.goal_id,
                        "step": rec.step_id,
                        "model": rec.model,
                    },
                    namespace="remy-models",
                )
                for situation in self._model_routing_situations(rec):
                    memory.capture_consequence(
                        situation=situation,
                        action=f"model:{rec.model}",
                        consequence=consequence,
                        trust=trust,
                        scope=[
                            "model-outcome",
                            "model-routing",
                            f"model:{rec.model}",
                            f"fallback:{str(rec.fallback_used).lower()}",
                            f"specialist:{rec.specialist}" if rec.specialist else "specialist:",
                            f"task_type:{self._model_task_type(rec)}",
                            f"status:{rec.status}" if rec.status else "status:",
                            f"verdict:{rec.verdict}" if rec.verdict else "verdict:",
                        ],
                        provenance=[
                            "remy:model_routing_outcome",
                            f"cycle:{rec.cycle_num}",
                            f"mission:{rec.mission_id}" if rec.mission_id else "mission:",
                        ],
                        links={
                            "mission": rec.mission_id,
                            "goal": rec.goal_id,
                            "step": rec.step_id,
                            "model": rec.model,
                        },
                        namespace="remy-models",
                    )
            if rec.unsupported_observed_claims > 0:
                memory.capture_consequence(
                    situation=rec.goal_description[:240],
                    action=action,
                    consequence="REFUTES",
                    trust=-1,
                    scope=[
                        "factuality-scar",
                        f"unsupported_observed_claims:{rec.unsupported_observed_claims}",
                        f"model:{rec.model}" if rec.model else "model:",
                        f"specialist:{rec.specialist}" if rec.specialist else "specialist:",
                        f"status:{rec.status}" if rec.status else "status:",
                    ],
                    provenance=[
                        "remy:factuality_runtime",
                        f"cycle:{rec.cycle_num}",
                        f"verdict:{rec.verdict}" if rec.verdict else "verdict:",
                    ],
                    links={
                        "mission": rec.mission_id,
                        "goal": rec.goal_id,
                        "step": rec.step_id,
                        "model": rec.model,
                    },
                    namespace="remy-factuality",
                )
                self._store_factuality_claim_type_scars(memory, rec)
        except Exception as e:
            log.debug("Failed to store outcome in memory: %s", e)

    def _store_factuality_claim_type_scars(self, memory, rec: CycleRecord) -> None:
        """Persist unsupported claim classes as reusable factuality policies."""
        report = rec.factuality_report
        counts: dict[str, int] = {}
        samples: dict[str, str] = {}
        for claim in list(getattr(report, "claim_details", None) or []):
            if bool(getattr(claim, "supported", False)):
                continue
            claim_class = str(getattr(claim, "claim_class", "") or "unsupported_claim")
            counts[claim_class] = counts.get(claim_class, 0) + 1
            samples.setdefault(claim_class, str(getattr(claim, "text", "") or "")[:160])
        for claim_class, count in sorted(counts.items()):
            memory.capture_consequence(
                situation=f"factuality-claim-type:{claim_class}|runtime:v3",
                action=f"answer_claim_type:{claim_class}:without_evidence",
                consequence="REFUTES",
                trust=-1,
                scope=[
                    "factuality-scar",
                    "claim-type-scar",
                    f"claim_class:{claim_class}",
                    f"unsupported_claims:{count}",
                    f"model:{rec.model}" if rec.model else "model:",
                    f"specialist:{rec.specialist}" if rec.specialist else "specialist:",
                    f"status:{rec.status}" if rec.status else "status:",
                    f"sample:{samples.get(claim_class, '')}",
                ],
                provenance=[
                    "remy:v3_factuality_claim_type",
                    f"cycle:{rec.cycle_num}",
                    f"verdict:{rec.verdict}" if rec.verdict else "verdict:",
                ],
                links={
                    "mission": rec.mission_id,
                    "goal": rec.goal_id,
                    "step": rec.step_id,
                    "model": rec.model,
                    "claim_class": claim_class,
                },
                namespace="remy-factuality",
            )

    def _model_situation(self, rec: CycleRecord) -> str:
        parts = [
            f"specialist:{rec.specialist or 'unknown'}",
            f"goal:{(rec.goal_description or '')[:160]}",
        ]
        if rec.tool_calls:
            parts.append(f"tool_calls:{rec.tool_calls}")
        if rec.fallback_used:
            parts.append("fallback:true")
        return "|".join(parts)

    def _model_task_type(self, rec: CycleRecord) -> str:
        text = " ".join(
            [
                rec.goal_description or "",
                rec.planned_action or "",
                rec.decision or "",
                rec.specialist or "",
            ]
        ).lower()
        specialist = (rec.specialist or "").lower()
        if specialist in {"researcher", "osint", "research_worker"} or any(
            token in text for token in ("research", "source", "evidence", "market", "competitor")
        ):
            return "research"
        if specialist in {"browser", "browser_worker"} or any(
            token in text for token in ("browser", "page", "selector", "website")
        ):
            return "browser"
        if specialist in {"executor", "coder", "engineer"} or any(
            token in text for token in ("code", "test", "bug", "python", "rust")
        ):
            return "coding"
        return "general"

    def _model_routing_situation(self, specialist: str = "", task_type: str = "") -> str:
        base = f"model-routing:runtime:v3|specialist:{specialist or 'unknown'}"
        if task_type:
            return f"{base}|task_type:{task_type}"
        return base

    def _model_routing_situations(self, rec: CycleRecord) -> list[str]:
        task_type = self._model_task_type(rec)
        situations = [
            self._model_routing_situation(rec.specialist, task_type),
            self._model_routing_situation(rec.specialist),
        ]
        return list(dict.fromkeys(situations))

    def _memory_model_routing_hint(self, specialist: str = "", task_type: str = "") -> dict[str, Any]:
        try:
            from remy.config.settings import settings
            from remy.core.consequence_gate import consult_policy_hint
            from ..memory.memory_api import get_memory

            models: list[str] = []
            for model in [
                settings.SUMMARY_MODEL,
                *list(settings.FALLBACK_MODELS or []),
                *[r.model for r in self._records if r.model],
            ]:
                model = str(model or "").strip()
                if model and model not in models:
                    models.append(model)
            if not models:
                return {}

            situations = [
                self._model_routing_situation(specialist, task_type),
                self._model_routing_situation(specialist),
            ]
            situations = list(dict.fromkeys(situations))
            memory = get_memory()
            avoid: list[str] = []
            preferred = ""
            preferred_score = 0
            ranked: list[dict[str, Any]] = []
            for model in models:
                context: dict[str, Any] = {}
                for situation in situations:
                    hint = consult_policy_hint(
                        memory,
                        situation=situation,
                        action=f"model:{model}",
                        namespace="remy-models",
                    )
                    context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
                    if int(context.get("supports", 0) or 0) or int(context.get("refutes", 0) or 0):
                        break
                supports = int(context.get("supports", 0) or 0)
                refutes = int(context.get("refutes", 0) or 0)
                if not supports and not refutes:
                    continue
                policy = str(context.get("hint") or "")
                score = supports - refutes
                ranked.append(
                    {
                        "model": model,
                        "cycles": supports + refutes,
                        "successes": supports,
                        "failures": refutes,
                        "unsupported_observed_claims": 0,
                        "fallback_uses": 0,
                        "score": float(score),
                    }
                )
                if policy == "avoid" and refutes > 0:
                    avoid.append(model)
                    continue
                if policy == "prefer" and supports > 0 and (not preferred or score > preferred_score):
                    preferred = model
                    preferred_score = score
            ranked.sort(key=lambda item: (-float(item["score"]), -int(item["successes"]), item["model"]))
            if not ranked:
                return {}
            return {
                "preferred_model": preferred,
                "avoid_models": tuple(avoid),
                "model_scores": ranked,
                "source": "memory",
            }
        except Exception as e:
            log.debug("model routing memory hint failed: %s", e)
            return {}

    def model_routing_hint(self, specialist: str = "") -> dict[str, Any]:
        """Suggest a soft model routing override from lived cycle outcomes.

        This is not a classifier and not a hard block. It summarizes observed
        consequences: prefer models that have supported recent work, and move
        models with failures or factuality debt to the end of the fallback chain.
        If there is no specialist-specific evidence, it falls back to global
        model history.
        """
        used_source = "specialist"
        records = [
            r for r in self._records
            if r.model and (not specialist or r.specialist == specialist)
        ]
        if not records and specialist:
            records = [r for r in self._records if r.model]
            used_source = "global"
        elif not specialist:
            used_source = "global"
        if not records:
            return self._memory_model_routing_hint(specialist) or {
                "preferred_model": "",
                "avoid_models": (),
                "model_scores": [],
                "source": "none",
            }

        scores: dict[str, dict[str, Any]] = {}
        for r in records:
            item = scores.setdefault(
                r.model,
                {
                    "model": r.model,
                    "cycles": 0,
                    "successes": 0,
                    "failures": 0,
                    "unsupported_observed_claims": 0,
                    "fallback_uses": 0,
                    "score": 0.0,
                },
            )
            item["cycles"] += 1
            if r.status == "success":
                item["successes"] += 1
                item["score"] += 1.0
            elif r.status in ("failure", "blocked", "timeout"):
                item["failures"] += 1
                item["score"] -= 1.25
            if r.fallback_used:
                item["fallback_uses"] += 1
                item["score"] -= 0.15
            unsupported = int(r.unsupported_observed_claims or 0)
            item["unsupported_observed_claims"] += unsupported
            item["score"] -= unsupported * 0.35

        ranked = sorted(
            scores.values(),
            key=lambda item: (-float(item["score"]), -int(item["successes"]), item["model"]),
        )
        preferred = ""
        if ranked and float(ranked[0]["score"]) > 0 and int(ranked[0]["successes"]) > 0:
            preferred = str(ranked[0]["model"])

        avoid = [
            str(item["model"])
            for item in ranked
            if int(item["failures"]) > int(item["successes"])
            or int(item["unsupported_observed_claims"]) >= 2
            or float(item["score"]) < 0
        ]

        return {
            "preferred_model": preferred,
            "avoid_models": tuple(avoid),
            "model_scores": ranked,
            "source": used_source,
        }

    # -------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------

    def recent(self, limit: int = 20) -> list[CycleRecord]:
        return list(reversed(self._records[-limit:]))

    def by_mission(self, mission_id: str, limit: int = 20) -> list[CycleRecord]:
        return [
            r for r in reversed(self._records)
            if r.mission_id == mission_id
        ][:limit]

    def stats(self) -> dict[str, Any]:
        if not self._records:
            return {"cycles": 0}

        successes = sum(1 for r in self._records if r.status == "success")
        total_cost = sum(r.cost_usd for r in self._records)
        total_unsupported_claims = sum(r.unsupported_observed_claims for r in self._records)
        model_outcomes: dict[str, dict[str, Any]] = {}
        for r in self._records:
            if not r.model:
                continue
            item = model_outcomes.setdefault(
                r.model,
                {
                    "model": r.model,
                    "cycles": 0,
                    "successes": 0,
                    "failures": 0,
                    "fallback_uses": 0,
                    "unsupported_observed_claims": 0,
                    "cost_usd": 0.0,
                },
            )
            item["cycles"] += 1
            if r.status == "success":
                item["successes"] += 1
            elif r.status in ("failure", "blocked", "timeout"):
                item["failures"] += 1
            if r.fallback_used:
                item["fallback_uses"] += 1
            item["unsupported_observed_claims"] += int(r.unsupported_observed_claims or 0)
            item["cost_usd"] += float(r.cost_usd or 0.0)
        model_items = []
        for item in model_outcomes.values():
            cycles = item["cycles"] or 1
            model_items.append({
                **item,
                "success_rate": round(item["successes"] / cycles, 2),
                "cost_usd": round(item["cost_usd"], 4),
            })
        model_items.sort(
            key=lambda item: (
                -item["unsupported_observed_claims"],
                -item["failures"],
                item["model"],
            )
        )
        return {
            "cycles": len(self._records),
            "successes": successes,
            "success_rate": round(successes / len(self._records), 2),
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_usd": round(total_cost / len(self._records), 4),
            "unsupported_observed_claims_total": total_unsupported_claims,
            "model_outcomes": model_items,
        }

    def recent_outcomes_summary(self, limit: int = 5) -> str:
        """Build a summary of recent outcomes for decision prompt context."""
        recent = self.recent(limit)
        if not recent:
            return "No recent execution history."

        lines = []
        for r in recent:
            status_icon = "✓" if r.status == "success" else "✗"
            lines.append(
                f"  {status_icon} Cycle {r.cycle_num}: {r.goal_description[:60]} "
                f"→ {r.status} ({r.specialist}, ${r.cost_usd:.3f})"
            )
        return "Recent outcomes:\n" + "\n".join(lines)

    def scar_check(self, goal_description: str, action: str):
        """Scar-protected pre-check for a (goal, action) the chief is about to run.

        Closes the consequence loop: a cycle that previously REFUTED this exact
        (situation, action) pair — stored via `_store_outcome` — now influences
        the *next* decision. The scar guard (in AuraSDK) means later SUPPORTS
        from frequency cannot bury that lived failure, so a repeatedly-tried-and-
        failed action stays flagged even if it sometimes looked fine.

        Returns a `ConsequenceVerdict` (see `remy.core.consequence_gate`).
        Always fail-soft: any error yields an inconclusive/abstain verdict so the
        cycle never breaks because of the check.
        """
        try:
            from ..memory.memory_api import get_memory
            from remy.core.consequence_gate import consult_consequence_memory

            memory = get_memory()
            return consult_consequence_memory(
                memory,
                situation=(goal_description or "")[:240],
                action=action or "cycle",
                namespace="remy",
            )
        except Exception as e:  # pragma: no cover - defensive
            log.debug("scar_check failed: %s", e)
            from remy.core.consequence_gate import ConsequenceVerdict
            return ConsequenceVerdict(
                situation=(goal_description or "")[:240],
                action=action or "cycle",
            )

    def policy_hint(self, goal_description: str, action: str):
        """Return runtime policy hint from consequence and factuality memory."""
        situation = (goal_description or "")[:240]
        action = action or "cycle"
        try:
            from ..memory.memory_api import get_memory
            from remy.core.consequence_gate import (
                ConsequencePolicyHint,
                consult_policy_hint,
                consult_consequence_memory,
            )

            memory = get_memory()
            hint = consult_policy_hint(
                memory,
                situation=situation,
                action=action,
                namespace="remy",
            )
            factuality = consult_consequence_memory(
                memory,
                situation=situation,
                action=action,
                namespace="remy-factuality",
            )
            if factuality.is_refuted and hint.hint != "avoid":
                return ConsequencePolicyHint(
                    situation=situation,
                    action=action,
                    hint="requires_evidence",
                    reason=(
                        "Prior factuality audit refuted this action's unsupported observed claims; "
                        "require evidence before trusting the output."
                    ),
                    verdict=factuality.verdict,
                    supports=factuality.supports,
                    refutes=factuality.refutes,
                    requires_evidence=True,
                    should_block=False,
                )
            return hint
        except Exception as e:  # pragma: no cover - defensive
            log.debug("policy_hint failed: %s", e)
            from remy.core.consequence_gate import ConsequencePolicyHint
            return ConsequencePolicyHint(situation=situation, action=action)

    def routing_policy_hint(self, specialist_id: str):
        """Return consequence-memory routing pressure for a specialist."""
        specialist_id = specialist_id or "unknown"
        try:
            from ..memory.memory_api import get_memory
            from remy.core.consequence_gate import consult_policy_hint

            memory = get_memory()
            return consult_policy_hint(
                memory,
                situation=f"specialist:{specialist_id}",
                action=f"route_to:{specialist_id}",
                namespace="remy-routing",
            )
        except Exception as e:  # pragma: no cover - defensive
            log.debug("routing_policy_hint failed: %s", e)
            from remy.core.consequence_gate import ConsequencePolicyHint
            return ConsequencePolicyHint(
                situation=f"specialist:{specialist_id}",
                action=f"route_to:{specialist_id}",
            )

    def recent_outcomes_summary_with_scars(
        self, goal_description: str, candidate_action: str, limit: int = 5
    ) -> str:
        """`recent_outcomes_summary` plus an explicit scar warning for the
        candidate (goal, action), so the decision prompt sees a refuted action
        as a hard signal rather than just one row among recent history."""
        base = self.recent_outcomes_summary(limit)
        verdict = self.scar_check(goal_description, candidate_action)
        if verdict.is_refuted:
            from remy.core.consequence_gate import render_scar_warning
            warning = render_scar_warning(verdict)
            if warning:
                return f"{base}\n\n{warning}"
        return base
