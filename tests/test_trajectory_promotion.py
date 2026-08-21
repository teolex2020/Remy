from types import SimpleNamespace

from remy.core.trajectory_promotion import (
    evaluate_model_promotion_canary,
    resolve_model_promotion_routing,
)


def test_model_promotion_canary_routing_is_stable_and_bounded():
    promotion = {
        "promotion_id": "promotion-1",
        "status": "canary",
        "candidate_model": "provider/candidate",
        "previous_model": "provider/current",
        "canary_percent": 25,
    }
    store = SimpleNamespace(
        get_active_model_promotion=lambda **kwargs: promotion,
    )

    first = resolve_model_promotion_routing(
        project_id="project-1", session_id="session-stable", trajectory_store=store
    )
    second = resolve_model_promotion_routing(
        project_id="project-1", session_id="session-stable", trajectory_store=store
    )

    assert first == second
    assert first["routing_source"] == "trajectory_promotion_canary"
    assert first["preferred_model"] in {"provider/candidate", "provider/current"}
    assert 0 <= first["bucket"] < 100

    assignments = {
        resolve_model_promotion_routing(
            project_id="project-1",
            session_id=f"session-{index}",
            trajectory_store=store,
        )["preferred_model"]
        for index in range(100)
    }
    assert assignments == {"provider/candidate", "provider/current"}


def test_model_promotion_routing_is_disabled_outside_canary_states():
    store = SimpleNamespace(
        get_active_model_promotion=lambda **kwargs: {
            "promotion_id": "promotion-1",
            "status": "promoted",
            "candidate_model": "provider/candidate",
            "previous_model": "provider/current",
            "canary_percent": 25,
        },
    )

    routing = resolve_model_promotion_routing(
        project_id="project-1", session_id="session-1", trajectory_store=store
    )

    assert routing["preferred_model"] == ""
    assert routing["canary_applied"] is False


def test_pending_rollback_pins_every_session_to_previous_model():
    store = SimpleNamespace(
        get_active_model_promotion=lambda **kwargs: {
            "promotion_id": "promotion-1",
            "status": "rollback_pending",
            "candidate_model": "provider/candidate",
            "previous_model": "provider/current",
            "canary_percent": 25,
        },
    )

    routings = [
        resolve_model_promotion_routing(
            project_id="project-1",
            session_id=f"session-{index}",
            trajectory_store=store,
        )
        for index in range(20)
    ]

    assert {row["preferred_model"] for row in routings} == {"provider/current"}
    assert {row["routing_source"] for row in routings} == {
        "trajectory_promotion_rollback_pending"
    }


def test_production_canary_regression_automatically_stops_workflow():
    promotion = {
        "promotion_id": "promotion-1",
        "project_id": "project-1",
        "status": "canary",
        "candidate_model": "provider/candidate",
        "previous_model": "provider/current",
    }
    records = []
    for candidate in (True, False):
        for index in range(10):
            failed = candidate and index < 3
            records.append({
                "event_id": f"request-{candidate}-{index}",
                "session_id": f"session-{candidate}-{index}",
                "kind": "REQUEST",
                "status": "failed" if failed else "completed",
                "error": "ProviderError" if failed else "",
                "duration_ms": 2500 if candidate else 500,
                "details": {
                    "options": {
                        "promotion_id": "promotion-1",
                        "canary_applied": candidate,
                    },
                    "usage": {"total_tokens": 1800 if candidate else 700},
                },
            })
    observations = []
    store = SimpleNamespace(
        get_active_model_promotion=lambda **kwargs: promotion,
        list_model_promotion_requests=lambda **kwargs: records,
        observe_model_promotion_telemetry=lambda **kwargs: (
            observations.append(kwargs)
            or {
                **promotion,
                "status": "rolled_back",
                "rollback_reason": "canary_telemetry_regression",
            }
        ),
    )

    result = evaluate_model_promotion_canary(
        project_id="project-1", trajectory_store=store
    )

    assert result["telemetry"]["status"] == "regressed"
    assert result["auto_stopped"] is True
    assert result["promotion"]["status"] == "rolled_back"
    assert observations[0]["telemetry"]["status"] == "regressed"
