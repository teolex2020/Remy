"""Safe runtime routing for durable Trajectory model-promotion canaries."""

from __future__ import annotations

import hashlib
from typing import Any


def resolve_model_promotion_routing(
    *,
    project_id: str,
    session_id: str,
    trajectory_store=None,
) -> dict[str, Any]:
    """Return a stable session-level canary assignment without changing settings."""
    try:
        if trajectory_store is None:
            from remy.core.trajectory_store import get_trajectory_store

            trajectory_store = get_trajectory_store()
        promotion = trajectory_store.get_active_model_promotion(project_id=project_id)
    except Exception:
        promotion = None
    if promotion and str(promotion.get("status") or "") == "rollback_pending":
        return {
            "preferred_model": str(promotion["previous_model"]),
            "routing_source": "trajectory_promotion_rollback_pending",
            "canary_applied": False,
            "bucket": -1,
            "canary_percent": 0,
            "promotion_id": str(promotion["promotion_id"]),
        }
    if not promotion or str(promotion.get("status") or "") not in {"canary", "ready"}:
        return {
            "preferred_model": "",
            "routing_source": "none",
            "canary_applied": False,
            "bucket": -1,
        }
    identity = f"{promotion['promotion_id']}|{session_id or '__anonymous__'}"
    bucket = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % 100
    canary_percent = max(5, min(int(promotion.get("canary_percent") or 0), 50))
    canary_applied = bucket < canary_percent
    return {
        "preferred_model": str(
            promotion["candidate_model"] if canary_applied else promotion["previous_model"]
        ),
        "routing_source": "trajectory_promotion_canary",
        "canary_applied": canary_applied,
        "bucket": bucket,
        "canary_percent": canary_percent,
        "ramp_stage": int(promotion.get("ramp_stage") or 0),
        "promotion_id": str(promotion["promotion_id"]),
    }


def evaluate_model_promotion_canary(
    *,
    project_id: str,
    trajectory_store=None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate production telemetry and stop a regressed pre-promotion canary."""
    if trajectory_store is None:
        from remy.core.trajectory_store import get_trajectory_store

        trajectory_store = get_trajectory_store()
    promotion = trajectory_store.get_active_model_promotion(project_id=project_id)
    if not promotion or str(promotion.get("status") or "") not in {"canary", "ready"}:
        return {"telemetry": None, "promotion": promotion, "auto_stopped": False}
    if records is None:
        records = trajectory_store.list_model_promotion_requests(
            project_id=project_id,
            promotion_id=promotion["promotion_id"],
            limit=5000,
        )
    from remy.core.trajectory_diagnostics import analyze_model_promotion_canary

    telemetry = analyze_model_promotion_canary(records, promotion=promotion)
    previous_stage = int(promotion.get("ramp_stage") or 0)
    previous_status = str(promotion.get("status") or "")
    promotion = trajectory_store.observe_model_promotion_telemetry(
        project_id=project_id,
        promotion_id=promotion["promotion_id"],
        telemetry=telemetry,
    )
    auto_stopped = (
        telemetry["status"] == "regressed"
        and previous_status in {"canary", "ready"}
        and promotion.get("status") == "rolled_back"
    )
    return {
        "telemetry": telemetry,
        "promotion": promotion,
        "auto_stopped": auto_stopped,
        "ramped": int(promotion.get("ramp_stage") or 0) > previous_stage,
        "ramp_complete": bool(promotion.get("ramp_complete")),
    }
