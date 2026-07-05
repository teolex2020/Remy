"""Summarize first-class harness modules for ablation-style inspection."""

from __future__ import annotations


def get_harness_ablation_summary(*, api=None, runtime_snapshot: dict | None = None) -> dict:
    runtime_snapshot = runtime_snapshot or {}
    auto = runtime_snapshot.get("autonomy", {}) or {}
    brain = getattr(api, "brain", None) if api is not None else None

    modules: list[dict] = []

    modules.append(
        {
            "id": "verify_gate",
            "label": "Verify gate",
            "enabled": True,
            "evidence": [
                "generate_report verification",
                "complete_research verification",
                "reconstruction verification",
            ],
            "description": "Execution must pass verification before success is finalized.",
        }
    )

    recovery_enabled = bool(brain and hasattr(brain, "search"))
    modules.append(
        {
            "id": "recovery_replay",
            "label": "Recovery replay",
            "enabled": recovery_enabled,
            "evidence": [
                "history gap analyzer",
                "selective reconstruction",
            ],
            "description": "History replay can reconstruct missing durable memory after incidents.",
        }
    )

    correction_enabled = bool(brain and hasattr(brain, "get_correction_review_queue"))
    modules.append(
        {
            "id": "correction_loop",
            "label": "Correction loop",
            "enabled": correction_enabled,
            "evidence": [
                "correction review queue",
                "negative feedback path",
            ],
            "description": "User corrections can push records into review and reduce trust in bad memory.",
        }
    )

    decision_dossier_enabled = bool(auto.get("running") or auto.get("current_role") or auto.get("goals"))
    modules.append(
        {
            "id": "decision_dossier_surface",
            "label": "Decision dossier surface",
            "enabled": decision_dossier_enabled,
            "evidence": [
                "decision snapshot surface",
                "activity dossier access",
            ],
            "description": "Runtime decisions are exposed as operator-reviewable artifacts.",
        }
    )

    return {
        "count": len(modules),
        "enabled_count": sum(1 for item in modules if item["enabled"]),
        "modules": modules,
    }
