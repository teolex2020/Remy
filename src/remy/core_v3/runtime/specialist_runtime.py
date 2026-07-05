"""
Specialist resolution runtime for Remy v3.

Owns deterministic mapping from mission/task/step context to a specialist
profile, so routing policy does not live inside ChiefAgent.
"""

from __future__ import annotations

from ..missions.mission_models import Goal, Mission, MissionMode, Task
from ..planning.plan_models import PlanStep


class SpecialistRuntime:
    """Resolve the best specialist profile for the next unit of work."""

    def __init__(
        self,
        *,
        registry,
        mission_runtime,
        specialist_inference_runtime,
        goal_context_runtime,
        evaluator=None,
        ops_query_runtime=None,
        recorder=None,
    ):
        self.registry = registry
        self.mission_runtime = mission_runtime
        self.specialist_inference_runtime = specialist_inference_runtime
        self.goal_context_runtime = goal_context_runtime
        self.evaluator = evaluator
        self.ops_query_runtime = ops_query_runtime
        self.recorder = recorder
        self._last_resolution: dict[str, object] = {}

    def last_resolution(self) -> dict[str, object]:
        return dict(self._last_resolution)

    def resolve(
        self,
        *,
        step: PlanStep,
        task: Task | None,
        mission: Mission,
        goal: Goal | None,
    ):
        preferred = ""
        preferred_reason = ""
        candidates = []
        if task is not None:
            preferred = self.mission_runtime.task_specialist(task)
            preferred_reason = "task_specialist"
        elif step.specialist:
            preferred = step.specialist
            preferred_reason = "step_specialist"
        elif goal and goal.pack_template:
            preferred = goal.pack_template
            preferred_reason = "goal_pack_template"

        specialist = self.registry.get(preferred) if preferred else None
        if specialist is not None:
            candidates.append(specialist)

        specialist = self.registry.get(self.specialist_inference_runtime.infer(step.instruction))
        if specialist is not None:
            candidates.append(specialist)

        resolved = self.registry.resolve(
            self.goal_context_runtime.build_goal_dict(step=step, mission=mission)
        )
        if resolved is not None:
            candidates.append(resolved)

        deduped = []
        seen = set()
        for candidate in candidates:
            if candidate.id not in seen:
                seen.add(candidate.id)
                deduped.append(candidate)

        sensitive = self._is_sensitive_work(
            mission=mission,
            goal=goal,
            task=task,
            step=step,
        )

        if not deduped:
            self._remember_resolution(
                specialist=None,
                reason="no_candidate",
                sensitive=sensitive,
                quality_factor=0.0,
            )
            return resolved

        preferred_id = preferred if preferred else ""
        if preferred_id:
            chosen = next((candidate for candidate in deduped if candidate.id == preferred_id), None)
            if chosen is not None:
                rerouted = self._maybe_reroute_preferred(
                    preferred=chosen,
                    candidates=deduped,
                    preferred_reason=preferred_reason or "preferred_candidate",
                    sensitive=sensitive,
                )
                if rerouted is not None:
                    return rerouted
                quality = self._specialist_quality(chosen.id)
                self._remember_resolution(
                    specialist=chosen,
                    reason=preferred_reason or "preferred_candidate",
                    sensitive=sensitive,
                    quality_factor=quality["quality_adjusted_success_rate"],
                    routing_policy=str(quality.get("routing_policy") or ""),
                )
                return chosen

        if len(deduped) == 1 or self.evaluator is None or not sensitive:
            chosen = deduped[0]
            quality = self._specialist_quality(chosen.id)
            self._remember_resolution(
                specialist=chosen,
                reason="single_candidate" if len(deduped) == 1 else "non_sensitive",
                sensitive=sensitive,
                quality_factor=quality["quality_adjusted_success_rate"],
                routing_policy=str(quality.get("routing_policy") or ""),
            )
            return chosen

        scored = [
            (self._specialist_quality(candidate.id), candidate)
            for candidate in deduped
        ]
        scored.sort(
            key=lambda item: (
                -item[0]["quality_adjusted_success_rate"],
                item[0]["unsupported_claims"],
                item[1].id,
            )
        )
        quality, chosen = scored[0]
        self._remember_resolution(
            specialist=chosen,
            reason=f"sensitive_quality_tiebreak:{quality['quality_adjusted_success_rate']:.2f}",
            sensitive=sensitive,
            quality_factor=quality["quality_adjusted_success_rate"],
            routing_policy=str(quality.get("routing_policy") or ""),
        )
        return chosen

    def _specialist_quality(self, specialist_id: str) -> dict[str, object]:
        if self.ops_query_runtime is not None and hasattr(self.ops_query_runtime, "specialist_quality"):
            quality = self.ops_query_runtime.specialist_quality(specialist_id) or {}
            return self._with_routing_policy(specialist_id, {
                "success_rate": float(quality.get("success_rate", 0.5) or 0.5),
                "quality_adjusted_success_rate": float(
                    quality.get("quality_adjusted_success_rate", quality.get("success_rate", 0.5)) or 0.5
                ),
                "unsupported_claims": int(quality.get("unsupported_claims", 0) or 0),
                "factuality_penalty": float(quality.get("factuality_penalty", 0.0) or 0.0),
            })
        if self.evaluator is None:
            return self._with_routing_policy(specialist_id, {
                "success_rate": 0.5,
                "quality_adjusted_success_rate": 0.5,
                "unsupported_claims": 0,
                "factuality_penalty": 0.0,
            })
        raw = float(self.evaluator.specialist_success_rate(specialist_id))
        details = self.evaluator.summary().get("specialist_scores", {}).get(specialist_id, {})
        return self._with_routing_policy(specialist_id, {
            "success_rate": raw,
            "quality_adjusted_success_rate": float(details.get("quality_adjusted_success_rate", raw) or raw),
            "unsupported_claims": int(details.get("unsupported_claims", 0) or 0),
            "factuality_penalty": float(details.get("factuality_penalty", 0.0) or 0.0),
        })

    @staticmethod
    def _routing_causal_edge(supports: int, refutes: int) -> str:
        """Label a (route-to-specialist → SUCCESS) pair as a typed causal edge.

        Distinguishes a specialist that genuinely *causes* success (the win
        reliably depends on routing here) from one that merely *precedes* it
        (routing here just correlates with wins that would happen anyway). The
        effect is SUCCESS, so support for the (route→success) pattern is the
        number of SUPPORTS and counterevidence is the number of REFUTES.

        Returns "causes" | "enables" | "precedes" | "refutes". Fail-soft to a
        conservative label on older AuraSDK wheels.
        """
        try:
            from aura import classify_causal_edge

            lift = 2.0 if supports > refutes else 0.0
            return classify_causal_edge(
                support_count=int(supports),
                counterevidence=int(refutes),
                transition_lift=lift,
                positive_effect_signals=int(supports),
                negative_effect_signals=int(refutes),
            )
        except Exception:
            if supports > 0 and refutes == 0:
                return "causes"
            return "precedes"

    def _with_routing_policy(self, specialist_id: str, quality: dict[str, object]) -> dict[str, object]:
        hint = self._routing_policy_for_specialist(specialist_id)
        policy = str(hint.get("hint") or "")
        adjustment = 0.0
        unsupported_delta = 0
        if policy == "avoid":
            adjustment = -0.40
            unsupported_delta = 2
        elif policy == "prefer":
            adjustment = 0.20
        elif policy in ("verify_first", "requires_evidence"):
            adjustment = -0.10
            unsupported_delta = 1

        # Refine a PREFER signal by the typed causal edge: a specialist that
        # counterfactually *causes* success earns a stronger boost, while one
        # that merely *precedes* success (correlation) is dampened toward neutral
        # so it does not crowd out a genuinely causal alternative. `avoid` is left
        # untouched — it is already scar-protected upstream and must not be
        # weakened by this refinement.
        edge = ""
        if policy == "prefer":
            supports = int(hint.get("supports", 0) or 0)
            refutes = int(hint.get("refutes", 0) or 0)
            edge = self._routing_causal_edge(supports, refutes)
            if edge == "causes":
                adjustment = 0.30
            elif edge == "precedes":
                adjustment = 0.05  # mere correlation — near-neutral

        adjusted = float(quality.get("quality_adjusted_success_rate", 0.5) or 0.5)
        quality["quality_adjusted_success_rate"] = max(0.0, min(1.0, adjusted + adjustment))
        quality["unsupported_claims"] = int(quality.get("unsupported_claims", 0) or 0) + unsupported_delta
        quality["routing_policy"] = policy
        quality["routing_policy_reason"] = str(hint.get("reason") or "")
        quality["routing_policy_adjustment"] = adjustment
        if edge:
            quality["routing_causal_edge"] = edge
        return quality

    def _routing_policy_for_specialist(self, specialist_id: str) -> dict[str, object]:
        if self.recorder is None or not hasattr(self.recorder, "routing_policy_hint"):
            return {}
        try:
            hint = self.recorder.routing_policy_hint(specialist_id)
            if hasattr(hint, "to_context"):
                context = dict(hint.to_context())
                return self._neutralize_empty_routing_hint(context)
            if isinstance(hint, dict):
                return self._neutralize_empty_routing_hint(dict(hint))
        except Exception:
            return {}
        return {}

    def _neutralize_empty_routing_hint(self, hint: dict[str, object]) -> dict[str, object]:
        policy = str(hint.get("hint") or "")
        if policy != "verify_first":
            return hint
        supports = int(hint.get("supports", 0) or 0)
        refutes = int(hint.get("refutes", 0) or 0)
        reason = str(hint.get("reason") or "").lower()
        if supports == 0 and refutes == 0 and "no lived consequence" in reason:
            return {}
        return hint

    def _maybe_reroute_preferred(self, *, preferred, candidates, preferred_reason: str, sensitive: bool):
        if len(candidates) <= 1:
            return None

        preferred_quality = self._specialist_quality(preferred.id)
        alternatives = [candidate for candidate in candidates if candidate.id != preferred.id]
        if not alternatives:
            return None

        ranked = sorted(
            ((self._specialist_quality(candidate.id), candidate) for candidate in alternatives),
            key=lambda item: (
                -item[0]["quality_adjusted_success_rate"],
                item[0]["unsupported_claims"],
                item[1].id,
            ),
        )
        best_quality, best_candidate = ranked[0]

        preferred_adjusted = float(preferred_quality["quality_adjusted_success_rate"])
        best_adjusted = float(best_quality["quality_adjusted_success_rate"])
        preferred_unsupported = int(preferred_quality["unsupported_claims"])
        preferred_degraded = (
            (float(preferred_quality["success_rate"]) - preferred_adjusted) >= 0.15
            or preferred_unsupported >= 2
        )
        should_override = False
        if preferred_degraded and best_adjusted >= preferred_adjusted + 0.05:
            should_override = True
        elif sensitive and best_adjusted >= preferred_adjusted + 0.10:
            should_override = True

        if not should_override:
            return None

        self._remember_resolution(
            specialist=best_candidate,
            reason=f"routing_pressure_override:{preferred_reason}:{preferred.id}->{best_candidate.id}",
            sensitive=sensitive,
            quality_factor=best_adjusted,
            routing_policy=str(best_quality.get("routing_policy") or ""),
        )
        return best_candidate

    def _remember_resolution(
        self,
        *,
        specialist,
        reason: str,
        sensitive: bool,
        quality_factor: float,
        routing_policy: str = "",
    ):
        self._last_resolution = {
            "specialist_id": specialist.id if specialist is not None else "",
            "reason": reason,
            "sensitive": sensitive,
            "quality_factor": round(quality_factor, 3),
            "routing_policy": routing_policy,
        }

    def _is_sensitive_work(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        step: PlanStep,
    ) -> bool:
        if mission.mode in (MissionMode.DEEP_RESEARCH, MissionMode.CONTINUOUS_MONITORING):
            return True
        text = " ".join(filter(None, [
            mission.description,
            goal.description if goal else "",
            task.action if task else "",
            step.instruction,
        ])).lower()
        return any(keyword in text for keyword in (
            "verify",
            "verification",
            "research",
            "search",
            "source",
            "profile",
            "identity",
            "email",
            "phone",
            "birth",
            "location",
            "github",
            "repo",
            "website",
            "browser",
        ))
