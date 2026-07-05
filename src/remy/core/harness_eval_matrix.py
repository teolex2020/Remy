"""Load and summarize harness evaluation matrix scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class HarnessEvalScenario:
    scenario_id: str
    label: str
    status: str
    baseline_mode: str
    ablation_mode: str
    target_metrics: list[str]

    def to_summary(self) -> dict:
        return {
            "id": self.scenario_id,
            "label": self.label,
            "status": self.status,
            "baseline_mode": self.baseline_mode,
            "ablation_mode": self.ablation_mode,
            "target_metrics": list(self.target_metrics),
        }


@dataclass(slots=True)
class HarnessEvalMatrix:
    version: str
    name: str
    scenarios: list[HarnessEvalScenario]
    source_path: str

    def to_summary(self) -> dict:
        planned = [scenario for scenario in self.scenarios if scenario.status == "planned"]
        tracked_metrics = sorted(
            {
                metric
                for scenario in self.scenarios
                for metric in scenario.target_metrics
                if metric
            }
        )
        return {
            "version": self.version,
            "name": self.name,
            "count": len(self.scenarios),
            "planned_count": len(planned),
            "tracked_metrics": tracked_metrics,
            "scenarios": [scenario.to_summary() for scenario in self.scenarios],
            "source_path": self.source_path,
        }


def _matrix_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "harness_eval_matrix.yaml"


@lru_cache(maxsize=1)
def load_harness_eval_matrix() -> HarnessEvalMatrix:
    path = _matrix_path()
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(raw)
    except Exception:
        payload = json.loads(raw)

    payload = payload or {}
    scenarios = []
    for item in payload.get("scenarios", []) or []:
        if not isinstance(item, dict):
            continue
        scenarios.append(
            HarnessEvalScenario(
                scenario_id=str(item.get("id") or "").strip(),
                label=str(item.get("label") or item.get("id") or "").strip(),
                status=str(item.get("status") or "unknown").strip(),
                baseline_mode=str(item.get("baseline_mode") or "").strip(),
                ablation_mode=str(item.get("ablation_mode") or "").strip(),
                target_metrics=[str(x) for x in item.get("target_metrics", []) or []],
            )
        )
    return HarnessEvalMatrix(
        version=str(payload.get("version") or "unknown"),
        name=str(payload.get("name") or "harness-eval-matrix"),
        scenarios=scenarios,
        source_path=str(path),
    )


def get_harness_eval_matrix_summary() -> dict:
    return load_harness_eval_matrix().to_summary()


def run_verify_gate_ablation_eval(*, memory_verification: dict | None = None) -> dict:
    """Compare verify-gate baseline against a no-gate counterfactual."""
    memory_verification = memory_verification or {}
    verified_count = int(memory_verification.get("verified_count", 0) or 0)
    repair_required_count = int(memory_verification.get("repair_required_count", 0) or 0)
    total = verified_count + repair_required_count
    false_success_rate = (repair_required_count / total) if total > 0 else 0.0

    return {
        "id": "verify_gate_ablation",
        "label": "Verify Gate Ablation",
        "status": "completed" if total > 0 else "not_enough_data",
        "artifact_count": total,
        "baseline": {
            "mode": "with_verify_gate",
            "verified_artifacts": verified_count,
            "blocked_successes": repair_required_count,
            "false_success_rate": 0.0,
        },
        "ablation": {
            "mode": "without_verify_gate",
            "assumed_successes": total,
            "false_success_rate": round(false_success_rate, 4),
        },
        "delta": {
            "prevented_false_successes": repair_required_count,
            "false_success_rate_delta": round(false_success_rate, 4),
        },
        "summary": (
            f"Verify gate prevented {repair_required_count} false success claim(s) across {total} recent artifact check(s)."
            if total > 0
            else "Not enough recent verification data to run the compare."
        ),
    }


def run_recovery_replay_ablation_eval(
    *,
    history_review: dict | None = None,
    last_reconstruction: dict | None = None,
) -> dict:
    """Compare recovery/replay baseline against no-recovery counterfactual."""
    history_review = history_review or {}
    last_reconstruction = last_reconstruction or {}
    missing_candidates = int(history_review.get("missing_candidates_count", 0) or 0)
    review_candidates = int(history_review.get("review_candidates_count", 0) or 0)
    reconstruction_status = str(last_reconstruction.get("status") or "").strip()
    recovered_recently = 1 if reconstruction_status == "verified" else 0
    recovery_pressure = missing_candidates + review_candidates

    return {
        "id": "recovery_replay_ablation",
        "label": "Recovery Replay Ablation",
        "status": "completed" if (recovery_pressure > 0 or reconstruction_status) else "not_enough_data",
        "baseline": {
            "mode": "with_recovery_replay",
            "missing_candidates_detected": missing_candidates,
            "review_candidates_detected": review_candidates,
            "last_reconstruction_status": reconstruction_status or "none",
            "recent_recoveries": recovered_recently,
        },
        "ablation": {
            "mode": "without_recovery_replay",
            "assumed_unresolved_missing": missing_candidates,
            "assumed_unreviewed_candidates": review_candidates,
        },
        "delta": {
            "recovery_visibility_gain": recovery_pressure,
            "recent_recoveries": recovered_recently,
        },
        "summary": (
            f"Recovery/replay surfaced {missing_candidates} missing and {review_candidates} review candidate(s); "
            f"without it they would remain hidden in the runtime."
            if (recovery_pressure > 0 or reconstruction_status)
            else "Not enough recovery/replay data to run the compare."
        ),
    }


def run_correction_loop_ablation_eval(*, corrections: dict | None = None) -> dict:
    """Compare correction-loop baseline against no-correction counterfactual."""
    corrections = corrections or {}
    suggestions_count = int(corrections.get("suggestions_count", 0) or 0)
    review_queue_count = int(corrections.get("review_queue_count", 0) or 0)
    recently_corrected_count = int(corrections.get("recently_corrected_count", 0) or 0)
    correction_pressure = suggestions_count + review_queue_count + recently_corrected_count

    return {
        "id": "correction_loop_ablation",
        "label": "Correction Loop Ablation",
        "status": "completed" if correction_pressure > 0 else "not_enough_data",
        "baseline": {
            "mode": "with_correction_loop",
            "suggestions_detected": suggestions_count,
            "queued_reviews": review_queue_count,
            "recent_corrections": recently_corrected_count,
        },
        "ablation": {
            "mode": "without_correction_loop",
            "assumed_missed_corrections": suggestions_count + review_queue_count,
            "assumed_repeated_errors": recently_corrected_count,
        },
        "delta": {
            "correction_visibility_gain": suggestions_count + review_queue_count,
            "repair_events_captured": recently_corrected_count,
        },
        "summary": (
            f"Correction loop surfaced {suggestions_count} suggestion(s), {review_queue_count} review item(s), "
            f"and {recently_corrected_count} recent repair event(s)."
            if correction_pressure > 0
            else "Not enough correction-loop data to run the compare."
        ),
    }


def run_decision_dossier_ablation_eval(
    *,
    decision_snapshot_count: int = 0,
    pinned_snapshot_count: int = 0,
    active_goal_count: int = 0,
) -> dict:
    """Compare decision-dossier surface against a no-dossier counterfactual."""
    decision_snapshot_count = int(decision_snapshot_count or 0)
    pinned_snapshot_count = int(pinned_snapshot_count or 0)
    active_goal_count = int(active_goal_count or 0)
    traceability_gain = decision_snapshot_count + pinned_snapshot_count

    return {
        "id": "decision_dossier_ablation",
        "label": "Decision Dossier Ablation",
        "status": "completed" if (traceability_gain > 0 or active_goal_count > 0) else "not_enough_data",
        "baseline": {
            "mode": "with_decision_dossier",
            "decision_snapshots": decision_snapshot_count,
            "pinned_snapshots": pinned_snapshot_count,
            "active_goals": active_goal_count,
        },
        "ablation": {
            "mode": "without_decision_dossier",
            "assumed_untraceable_goals": active_goal_count,
            "assumed_missing_snapshots": traceability_gain,
        },
        "delta": {
            "traceability_gain": traceability_gain,
            "pinned_review_surfaces": pinned_snapshot_count,
        },
        "summary": (
            f"Decision dossier keeps {decision_snapshot_count} snapshot(s) and {pinned_snapshot_count} pinned review surface(s) "
            f"available across {active_goal_count} active goal(s)."
            if (traceability_gain > 0 or active_goal_count > 0)
            else "Not enough decision-dossier data to run the compare."
        ),
    }
