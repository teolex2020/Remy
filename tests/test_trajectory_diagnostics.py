from types import SimpleNamespace

import pytest

from remy.core.trajectory_diagnostics import (
    analyze_project_trajectory,
    analyze_model_promotion_canary,
    analyze_trajectory,
    analyze_trajectory_slo,
    build_incident_dossier,
    build_model_promotion_recommendation,
    derive_incident_eval_criteria,
    evaluate_trajectory_policies,
    evaluate_trajectory_regression,
    render_incident_dossier_markdown,
)


def _record(event_id, kind, **changes):
    record = {
        "event_id": event_id,
        "turn_id": "turn-1",
        "kind": kind,
        "status": "completed",
        "error": "",
        "started_at": 100.0,
        "first_output_at": 100.1,
        "completed_at": 101.0,
        "duration_ms": 1000,
        "details": {"preview": event_id},
        "output": "ok",
    }
    record.update(changes)
    return record


@pytest.mark.asyncio
async def test_execution_trajectory_route_is_read_only_and_project_scoped(tmp_path, monkeypatch):
    from remy.core.trajectory_store import TrajectoryStore
    from remy.web.routes import trajectory_routes

    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    run_event_id = store.begin_execution_run(
        scope="experiment",
        project_id="project-1",
        source_id="exp-1",
        source_name="Test experiment",
        run_id="run-1",
        goal="Compare outcomes",
    )
    store.record_execution_event(
        parent_event_id=run_event_id,
        event_kind="EXPERIMENT_PHASE",
        name="Preparing",
        details={"phase": "preparing"},
    )
    store.complete_execution_run(event_id=run_event_id, status="completed", output="done")
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: store)

    payload = await trajectory_routes.get_execution_trajectory(
        "experiment", "exp-1", "run-1", limit=100
    )

    assert payload["execution"] == {
        "scope": "experiment",
        "source_id": "exp-1",
        "run_id": "run-1",
        "read_only": True,
    }
    assert payload["execution_session_id"] == "experiment:exp-1:run-1"
    assert [record["kind"] for record in payload["records"]] == [
        "EXPERIMENT_RUN", "EXPERIMENT_PHASE", "EXPERIMENT_RESULT"
    ]


def _model_comparison(index, winner="provider/candidate", include_current=True):
    models = [{
        "preferred_model": winner,
        "rank": 1,
        "gate_passed": True,
        "passed_count": 4,
        "failed_count": 0,
        "error_count": 0,
        "avg_score": 96,
        "avg_request_ms": 650,
        "avg_tokens_per_request": 850,
    }]
    if include_current and winner != "provider/current":
        models.append({
            "preferred_model": "provider/current",
            "rank": 2,
            "gate_passed": False,
            "passed_count": 3,
            "failed_count": 1,
            "error_count": 0,
            "avg_score": 88,
            "avg_request_ms": 500,
            "avg_tokens_per_request": 700,
        })
    return {
        "comparison_id": f"comparison-{index}",
        "status": "completed",
        "winner_model": winner,
        "completed_at": f"2026-08-{index:02d}T10:00:00+00:00",
        "models": models,
    }


def _canary_request(
    index, *, candidate, failed=False, fallback=False, duration_ms=500, tokens=800,
    ramp_stage=0,
):
    return _record(
        f"request-{candidate}-{index}",
        "REQUEST",
        session_id=f"session-{candidate}-{index}",
        status="failed" if failed else "completed",
        error="ProviderError" if failed else "",
        duration_ms=duration_ms,
        details={
            "options": {
                "promotion_id": "promotion-1",
                "canary_applied": candidate,
                "ramp_stage": ramp_stage,
            },
            "usage": {"total_tokens": tokens},
            "fallback_used": fallback,
        },
    )


def test_canary_telemetry_collects_then_gates_production_regression():
    promotion = {
        "promotion_id": "promotion-1",
        "candidate_model": "provider/candidate",
        "previous_model": "provider/current",
    }
    collecting = analyze_model_promotion_canary(
        [
            *[_canary_request(index, candidate=True) for index in range(4)],
            *[_canary_request(index, candidate=False) for index in range(10)],
        ],
        promotion=promotion,
    )
    assert collecting["status"] == "collecting-data"
    assert collecting["sample_complete"] is False

    records = [
        *[
            _canary_request(index, candidate=True, failed=True, duration_ms=2400, tokens=1800)
            for index in range(10)
        ],
        *[
            _canary_request(index, candidate=False, duration_ms=500, tokens=700)
            for index in range(10)
        ],
    ]
    regressed = analyze_model_promotion_canary(records, promotion=promotion)

    assert regressed["status"] == "regressed"
    assert regressed["candidate"]["requests"] == 10
    assert regressed["control"]["requests"] == 10
    assert {row["metric"] for row in regressed["reasons"]} == {
        "failure_rate", "avg_request_ms", "tokens_per_request",
    }
    assert regressed["confidence"]["decision"] == "rollback"
    assert regressed["confidence"]["familywise_confidence"] == 0.95
    assert regressed["privacy"]["raw_errors_included"] is False

    uncertain_records = [
        *[_canary_request(index, candidate=True) for index in range(10)],
        *[_canary_request(index, candidate=False) for index in range(10)],
    ]
    for index, row in enumerate(uncertain_records):
        row["started_at"] = 100 + index * 60
    uncertain = analyze_model_promotion_canary(uncertain_records, promotion=promotion)
    assert uncertain["status"] == "collecting-data"
    assert uncertain["confidence"]["decision"] == "continue"
    assert uncertain["sample_plan"]["status"] == "projected"
    assert uncertain["sample_plan"]["projected_decision"] == "advance"
    assert uncertain["sample_plan"]["limiting_metric"] == "failure_rate"
    assert 10 < uncertain["sample_plan"]["target_requests_per_arm"] <= 200
    assert uncertain["sample_plan"]["estimated_hours"] > 0
    assert uncertain["sample_plan"]["within_time_budget"] is True

    healthy = analyze_model_promotion_canary(
        [
            *[_canary_request(index, candidate=True) for index in range(100)],
            *[_canary_request(index, candidate=False) for index in range(100)],
        ],
        promotion=promotion,
    )
    assert healthy["status"] == "healthy"
    assert healthy["confidence"]["decision"] == "advance"
    assert healthy["sample_plan"]["status"] == "resolved"
    assert healthy["sample_plan"]["additional_candidate_requests"] == 0
    assert set(healthy["confidence"]["intervals"]) == {
        "failure_rate", "avg_request_ms", "tokens_per_request",
    }

    fallback_regression = analyze_model_promotion_canary(
        [
            *[
                _canary_request(index, candidate=True, fallback=index < 8)
                for index in range(10)
            ],
            *[_canary_request(index, candidate=False) for index in range(10)],
        ],
        promotion=promotion,
    )
    assert fallback_regression["status"] == "regressed"
    assert fallback_regression["candidate"]["fallbacks"] == 8

    stage_one = analyze_model_promotion_canary(
        [
            *[_canary_request(index, candidate=True, ramp_stage=1) for index in range(100)],
            *[_canary_request(index, candidate=False, ramp_stage=1) for index in range(100)],
            *[_canary_request(index + 20, candidate=True, ramp_stage=0) for index in range(10)],
        ],
        promotion={**promotion, "ramp_stage": 1, "canary_percent": 25},
    )
    assert stage_one["status"] == "healthy"
    assert stage_one["candidate"]["requests"] == 100
    assert stage_one["ramp_stage"] == 1


def test_canary_sequential_confidence_stops_as_inconclusive_at_sample_cap():
    promotion = {
        "promotion_id": "promotion-1",
        "candidate_model": "provider/candidate",
        "previous_model": "provider/current",
    }
    telemetry = analyze_model_promotion_canary(
        [
            *[
                _canary_request(index, candidate=True, failed=index < 30)
                for index in range(200)
            ],
            *[_canary_request(index, candidate=False) for index in range(200)],
        ],
        promotion=promotion,
    )

    assert telemetry["status"] == "inconclusive"
    assert telemetry["confidence"]["decision"] == "inconclusive"
    assert telemetry["sample_plan"]["status"] == "unresolved-within-budget"
    assert telemetry["sample_plan"]["projected_decision"] == "inconclusive"
    assert telemetry["sample_plan"]["within_request_budget"] is False
    failure_interval = telemetry["confidence"]["intervals"]["failure_rate"]
    assert failure_interval["lower"] <= failure_interval["margin"]
    assert failure_interval["upper"] > failure_interval["margin"]


def test_model_promotion_requires_three_safe_wins_against_current_model():
    comparisons = [_model_comparison(index) for index in (18, 19, 20)]

    recommendation = build_model_promotion_recommendation(
        comparisons, current_model="provider/current"
    )

    assert recommendation["status"] == "promote"
    assert recommendation["recommended_model"] == "provider/candidate"
    assert recommendation["consecutive_wins"] == 3
    assert recommendation["all_evidence_gates_passed"] is True
    assert recommendation["current_model_compared"] is True
    assert recommendation["automatic_change_applied"] is False
    assert recommendation["evidence_comparison_ids"] == [
        "comparison-20", "comparison-19", "comparison-18",
    ]
    assert recommendation["privacy"]["raw_outputs_included"] is False


def test_model_promotion_holds_without_current_model_evidence_or_stable_streak():
    missing_current = [
        _model_comparison(index, include_current=False) for index in (18, 19, 20)
    ]
    recommendation = build_model_promotion_recommendation(
        missing_current, current_model="provider/current"
    )
    assert recommendation["status"] == "hold"
    assert recommendation["recommended_model"] == ""
    assert recommendation["current_model_compared"] is False

    unstable = [
        _model_comparison(18, winner="provider/other"),
        _model_comparison(19),
        _model_comparison(20),
    ]
    recommendation = build_model_promotion_recommendation(
        unstable, current_model="provider/current"
    )
    assert recommendation["status"] == "hold"
    assert recommendation["consecutive_wins"] == 2


def test_fork_dialogue_replays_only_completed_pairs_before_boundary():
    from remy.web.routes.trajectory_routes import _fork_dialogue

    records = [
        _record("user-1", "USER", sequence=1, turn_id="turn-1", input="first"),
        _record("assistant-1", "ASSISTANT", sequence=2, turn_id="turn-1", output="done"),
        _record("user-2", "USER", sequence=3, turn_id="turn-2", input="retry me"),
        _record("tool-2", "TOOL", sequence=4, turn_id="turn-2", output="side effect"),
    ]

    dialogue, prompt, boundary = _fork_dialogue(records, "tool-2")

    assert [(item["role"], item["content"]) for item in dialogue] == [
        ("user", "first"),
        ("assistant", "done"),
    ]
    assert prompt == "retry me"
    assert boundary["event_id"] == "tool-2"


def test_fork_dialogue_includes_boundary_assistant_as_completed_turn():
    from remy.web.routes.trajectory_routes import _fork_dialogue

    records = [
        _record("user-1", "USER", sequence=1, input="first"),
        _record("assistant-1", "ASSISTANT", sequence=2, output="done"),
    ]

    dialogue, prompt, _ = _fork_dialogue(records, "assistant-1")

    assert [item["content"] for item in dialogue] == ["first", "done"]
    assert prompt == ""


def test_diagnostics_identifies_failure_bottleneck_and_successes():
    records = [
        _record("request-1", "REQUEST", duration_ms=65_000, completed_at=165.0),
        _record("assistant-1", "ASSISTANT"),
        _record("tool-1", "TOOL", duration_ms=40_000, completed_at=140.0),
        _record(
            "tool-2",
            "TOOL",
            status="failed",
            error="provider rejected payload",
            duration_ms=25,
        ),
    ]

    result = analyze_trajectory(records, now=170.0)

    assert result["health"] == "failed"
    assert result["error_count"] == 1
    assert result["warning_count"] == 2
    assert result["bottleneck"]["event_id"] == "request-1"
    assert result["successes"] == {
        "completed_requests": 1,
        "completed_tools": 1,
        "completed_verifications": 0,
    }
    assert records[3]["diagnostic"]["severity"] == "error"
    assert {cause["root_cause_id"] for cause in result["root_causes"]} == {
        "latency", "tool-execution"
    }


def test_diagnostics_classifies_interruption_as_warning():
    records = [
        _record(
            "request-1",
            "REQUEST",
            status="failed",
            error="interrupted by process restart",
        )
    ]

    result = analyze_trajectory(records)

    assert result["health"] == "degraded"
    assert result["findings"][0]["category"] == "interruption"
    assert result["findings"][0]["severity"] == "warning"


def test_diagnostics_detects_slow_ttft_empty_output_and_fallback():
    records = [
        _record(
            "request-1",
            "REQUEST",
            first_output_at=109.0,
            duration_ms=10_000,
            details={"preview": "request", "fallback_used": True},
        ),
        _record("assistant-1", "ASSISTANT", output={"content": ""}),
    ]

    result = analyze_trajectory(records)
    categories = {finding["category"] for finding in result["findings"]}

    assert categories == {"slow-ttft", "fallback", "empty-output"}
    assert result["health"] == "degraded"


def test_healthy_diagnostics_reports_completed_work():
    records = [
        _record("request-1", "REQUEST"),
        _record("assistant-1", "ASSISTANT"),
        _record("tool-1", "TOOL"),
        _record("verification-1", "VERIFICATION", output={"unsupported_claims_total": 0}),
    ]

    result = analyze_trajectory(records)

    assert result["health"] == "healthy"
    assert result["findings"] == []
    assert result["successes"]["completed_verifications"] == 1


def test_diagnostics_exposes_claim_level_false_corroboration():
    records = [
        _record(
            "verification-1",
            "VERIFICATION",
            output={
                "type": "claim_source_matrix",
                "false_corroborated_claims": 2,
            },
        )
    ]

    result = analyze_trajectory(records)

    assert result["health"] == "degraded"
    assert result["warning_count"] == 1
    assert result["findings"][0]["category"] == "false-corroboration"
    assert "same evidence root" in result["findings"][0]["title"]


def test_diagnostics_exposes_stale_and_undated_temporal_evidence():
    records = [
        _record(
            "verification-1",
            "VERIFICATION",
            output={
                "type": "claim_source_matrix",
                "stale_claims": 2,
                "undated_temporal_claims": 1,
            },
        )
    ]

    result = analyze_trajectory(records)

    assert result["health"] == "degraded"
    assert result["warning_count"] == 1
    assert result["findings"][0]["category"] == "temporal-evidence"
    assert "2 stale, 1 undated" in result["findings"][0]["title"]


def test_diagnostics_exposes_safe_temporal_supersession_as_information():
    records = [
        _record(
            "verification-1",
            "VERIFICATION",
            output={
                "type": "claim_source_matrix",
                "resolved_temporal_conflicts": 1,
                "superseded_claims": 1,
                "unresolved_contradictions": 0,
            },
        )
    ]

    result = analyze_trajectory(records)

    assert result["health"] == "healthy"
    assert len(result["findings"]) == 1
    assert result["findings"][0]["severity"] == "info"
    assert result["findings"][0]["category"] == "temporal-supersession"
    assert "history" in result["findings"][0]["explanation"]


def test_diagnostics_warns_when_contradiction_remains_active():
    records = [
        _record(
            "verification-1",
            "VERIFICATION",
            output={
                "type": "claim_source_matrix",
                "unresolved_contradictions": 2,
            },
        )
    ]

    result = analyze_trajectory(records)

    assert result["health"] == "degraded"
    assert result["warning_count"] == 1
    assert result["findings"][0]["category"] == "unresolved-contradictions"


def test_diagnostics_exposes_confirmed_claim_lifecycle_transition():
    result = analyze_trajectory([
        _record(
            "verification-lifecycle",
            "VERIFICATION",
            output={
                "type": "claim_lifecycle",
                "summary": {"tracked_subjects": 1},
                "last_run": {"confirmed_transitions": 1, "pending_changes": 0},
            },
        )
    ])

    assert result["health"] == "healthy"
    assert result["findings"][0]["severity"] == "info"
    assert result["findings"][0]["category"] == "claim-lifecycle-transition"


def test_diagnostics_warns_about_pending_claim_lifecycle_change():
    result = analyze_trajectory([
        _record(
            "verification-lifecycle",
            "VERIFICATION",
            output={
                "type": "claim_lifecycle",
                "summary": {"tracked_subjects": 1},
                "last_run": {"confirmed_transitions": 0, "pending_changes": 1},
            },
        )
    ])

    assert result["health"] == "degraded"
    assert result["findings"][0]["category"] == "claim-lifecycle-pending"


def test_failed_provider_attempt_is_informational_when_request_recovers():
    records = [
        _record("request-1", "REQUEST", request_id="request-1"),
        _record(
            "attempt-1",
            "ATTEMPT",
            request_id="request-1",
            status="failed",
            error="provider timeout",
            details={"retry_action": "fallback-model", "model": "primary"},
        ),
        _record(
            "attempt-2",
            "ATTEMPT",
            request_id="request-1",
            details={"model": "backup"},
        ),
        _record("assistant-1", "ASSISTANT", request_id="request-1"),
    ]

    result = analyze_trajectory(records)

    assert result["health"] == "healthy"
    assert result["error_count"] == 0
    assert result["warning_count"] == 0
    assert result["findings"][0]["category"] == "recovered-attempt"
    assert result["findings"][0]["severity"] == "info"
    assert result["recovery"]["recovered_requests"] == 1
    assert result["recovery"]["failed_attempts"] == 1
    assert result["recovery"]["paths"][0]["final_model"] == "backup"
    assert result["root_causes"][0]["root_cause_id"] == "provider-reliability"


def test_turn_summary_keeps_recovered_attempt_visible_without_failing_turn():
    from remy.web.routes.trajectory_routes import _turn_summaries

    records = [
        _record("user-1", "USER"),
        _record("request-1", "REQUEST", request_id="request-1"),
        _record(
            "attempt-1",
            "ATTEMPT",
            request_id="request-1",
            status="failed",
            error="timeout",
            details={"retry_action": "fallback-model", "model": "primary"},
        ),
        _record("attempt-2", "ATTEMPT", request_id="request-1"),
        _record("assistant-1", "ASSISTANT", request_id="request-1"),
    ]
    analyze_trajectory(records)

    turn = _turn_summaries(records)[0]

    assert turn["status"] == "completed"
    assert turn["attempt_count"] == 2
    assert turn["failure_count"] == 1
    assert turn["error_count"] == 0


def test_timing_breakdown_uses_leaf_spans_and_reports_idle_time():
    records = [
        _record(
            "request-1", "REQUEST", request_id="request-1",
            started_at=100.0, completed_at=110.0, duration_ms=10_000,
        ),
        _record(
            "attempt-1", "ATTEMPT", request_id="request-1",
            started_at=100.0, completed_at=104.0, duration_ms=4_000,
        ),
        _record(
            "attempt-2", "ATTEMPT", request_id="request-1",
            started_at=104.0, completed_at=107.0, duration_ms=3_000,
        ),
        _record(
            "tool-1", "TOOL", request_id="request-1",
            started_at=107.0, completed_at=109.0, duration_ms=2_000,
        ),
        _record(
            "assistant-1", "ASSISTANT", request_id="request-1",
            started_at=109.0, completed_at=110.0, duration_ms=1_000,
        ),
    ]

    timing = analyze_trajectory(records)["timing_breakdown"]

    assert timing == {
        "wall_clock_ms": 10_000,
        "busy_ms": 9_000,
        "idle_ms": 1_000,
        "overlap_ms": 0,
        "model_ms": 7_000,
        "tool_ms": 2_000,
        "operation_count": 3,
        "max_concurrency": 1,
    }


def test_trace_integrity_reports_complete_correlated_trace():
    source = {"kind": "test"}
    records = [
        _record(
            "request-1", "REQUEST", sequence=1, request_id="request-1",
            source=source,
        ),
        _record(
            "attempt-1", "ATTEMPT", sequence=2, request_id="request-1",
            parent_id="request-1", source=source,
        ),
        _record(
            "assistant-1", "ASSISTANT", sequence=3, request_id="request-1",
            parent_id="request-1", source=source,
        ),
    ]

    integrity = analyze_trajectory(records)["integrity"]

    assert integrity["status"] == "reliable"
    assert integrity["coverage"]["score"] == 100
    assert integrity["issues"] == []


def test_trace_integrity_detects_duplicate_orphan_and_invalid_clock():
    records = [
        _record("duplicate", "USER", sequence=1),
        _record(
            "duplicate", "TOOL", sequence=1, request_id="missing-request",
            parent_id="missing-parent", started_at=105.0, completed_at=100.0,
        ),
    ]

    integrity = analyze_trajectory(records)["integrity"]
    categories = {issue["category"] for issue in integrity["issues"]}

    assert integrity["status"] == "invalid"
    assert {"duplicate-event", "sequence-order", "orphan-parent", "orphan-request", "negative-duration"} <= categories


def test_error_fingerprints_cluster_repeated_recovered_failures():
    records = [
        _record("request-1", "REQUEST", request_id="request-1"),
        _record("request-2", "REQUEST", request_id="request-2"),
        _record(
            "attempt-1", "ATTEMPT", request_id="request-1", status="failed",
            error="Provider timeout after 30 seconds",
            details={"provider": "provider-a", "model": "model-a"},
            annotation={
                "label": "resolved", "note": "Transient incident", "bookmarked": True
            },
        ),
        _record(
            "attempt-2", "ATTEMPT", request_id="request-2", status="failed",
            error="Provider timeout after 60 seconds",
            details={"provider": "provider-a", "model": "model-a"},
        ),
    ]

    clusters = analyze_trajectory(records)["error_clusters"]

    assert len(clusters) == 1
    assert clusters[0]["occurrence_count"] == 2
    assert clusters[0]["recurring"] is True
    assert clusters[0]["fully_recovered"] is True
    assert clusters[0]["severity"] == "info"
    assert clusters[0]["resolution"] == "triaged"
    assert clusters[0]["resolved_count"] == 1


def test_project_analytics_detects_regressions_without_exposing_event_payloads():
    records = [
        _record(
            "request-a", "REQUEST", session_id="session-a", request_id="request-a",
            turn_id="turn-a", started_at=190_000.0, completed_at=190_000.1,
            duration_ms=100, input="PRIVATE PROMPT", output="PRIVATE OUTPUT",
            details={"usage": {"total_tokens": 100}},
        ),
        _record(
            "attempt-a", "ATTEMPT", session_id="session-a", request_id="request-a",
            turn_id="turn-a", started_at=190_000.0, completed_at=190_000.05,
            duration_ms=50, status="failed", error="SECRET provider timeout after 30 seconds",
            details={"provider": "provider-a", "model": "model-a"},
        ),
        _record(
            "request-b", "REQUEST", session_id="session-b", request_id="request-b",
            turn_id="turn-b", started_at=199_000.0, completed_at=199_003.0,
            duration_ms=3_000, input="ANOTHER SECRET", output="HIDDEN OUTPUT",
            details={"usage": {"total_tokens": 400}},
        ),
        _record(
            "attempt-b", "ATTEMPT", session_id="session-b", request_id="request-b",
            turn_id="turn-b", started_at=199_000.0, completed_at=199_002.0,
            duration_ms=2_000, status="failed", error="SECRET provider timeout after 60 seconds",
            details={"provider": "provider-a", "model": "model-a"},
        ),
        _record(
            "tool-b", "TOOL", session_id="session-b", request_id="request-b",
            turn_id="turn-b", started_at=199_002.0, completed_at=199_003.0,
            duration_ms=1_000, status="failed", error="SECRET tool result",
            details={"name": "web_search"},
        ),
    ]

    result = analyze_project_trajectory(
        records,
        conversation_titles={"session-a": "Baseline", "session-b": "Regressed"},
        days=1,
        now=200_000.0,
    )

    assert result["summary"]["sessions"] == 2
    assert result["summary"]["requests"] == 2
    assert result["summary"]["failures"] == 1
    assert result["summary"]["recovered_failures"] == 2
    assert result["providers"][0]["attempts"] == 2
    assert result["providers"][0]["recovered"] == 2
    assert result["tools"][0]["name"] == "web_search"
    assert result["regressions"][0]["conversation_id"] == "session-b"
    assert {reason["metric"] for reason in result["regressions"][0]["reasons"]} == {
        "failure_rate", "latency", "tokens"
    }
    recurring = next(cluster for cluster in result["error_clusters"] if cluster["recurring"])
    assert recurring["session_count"] == 2
    assert "sample_error" not in recurring
    assert "event_ids" not in recurring
    serialized = str(result)
    assert "PRIVATE PROMPT" not in serialized
    assert "PRIVATE OUTPUT" not in serialized
    assert "SECRET" not in serialized


def test_saved_baseline_can_flag_a_single_new_session():
    records = [
        _record(
            "request-new", "REQUEST", session_id="session-new",
            request_id="request-new", turn_id="turn-new",
            started_at=199_000.0, completed_at=199_003.0,
            duration_ms=3_000, details={"usage": {"total_tokens": 350}},
        ),
        _record(
            "tool-new", "TOOL", session_id="session-new",
            request_id="request-new", turn_id="turn-new",
            status="failed", error="private failure",
            started_at=199_002.0, completed_at=199_003.0,
            duration_ms=1_000, details={"name": "web_search"},
        ),
    ]

    result = analyze_project_trajectory(
        records,
        days=1,
        now=200_000.0,
        baseline_metrics={
            "failure_rate": 0,
            "avg_request_ms": 500,
            "tokens_per_request": 100,
        },
    )

    assert result["baseline_source"] == "saved"
    assert result["baseline"]["avg_request_ms"] == 500
    assert len(result["regressions"]) == 1
    assert {reason["metric"] for reason in result["regressions"][0]["reasons"]} == {
        "failure_rate", "latency", "tokens"
    }


def test_trajectory_slo_detects_fast_burn_and_reports_error_budget():
    now = 20_000.0
    records = [
        _record("request-ok", "REQUEST", started_at=now - 60),
        _record(
            "tool-failed", "TOOL", started_at=now - 30,
            status="failed", error="private failure details",
        ),
    ]

    result = analyze_trajectory_slo(
        records,
        target_success_rate=0.99,
        window_days=30,
        min_operations=1,
        now=now,
    )

    assert result["status"] == "critical"
    assert result["alert"]["reason"] == "fast-burn"
    assert result["operations"] == 2
    assert result["failures"] == 1
    assert result["budget_consumed"] == 50.0
    assert result["budget_remaining"] == -49.0
    assert "private failure details" not in str(result)


def test_trajectory_slo_is_healthy_when_observed_window_has_no_failures():
    now = 20_000.0
    result = analyze_trajectory_slo(
        [
            _record(f"request-{index}", "REQUEST", started_at=now - index)
            for index in range(6)
        ],
        target_success_rate=0.99,
        window_days=7,
        min_operations=5,
        now=now,
    )

    assert result["status"] == "healthy"
    assert result["success_rate"] == 1.0
    assert result["budget_remaining"] == 1.0
    assert result["projected_exhaustion_hours"] is None


def test_incident_dossier_contains_structured_evidence_without_raw_payloads():
    records = [
        _record(
            "request-1", "REQUEST", request_id="request-1", turn_id="turn-1",
            input="SECRET PROMPT", output="SECRET OUTPUT", status="failed",
            error="SECRET provider credential", duration_ms=4_000,
            source={"provider": "provider-a", "model": "model-a"},
        ),
        _record(
            "tool-1", "TOOL", request_id="request-1", turn_id="turn-1",
            parent_id="request-1", status="failed", error="SECRET tool output",
            details={"name": "web_search", "preview": "SECRET QUERY"},
        ),
    ]
    dossier = build_incident_dossier(
        records,
        incident={
            "alert_id": "alert-1", "source_type": "policy",
            "status": "open", "severity": "error",
            "metric": "tool.failure_rate", "observed": 1,
            "baseline": 0.2, "delta": 0.8,
            "conversation_id": "conversation-1", "event_id": "tool-1",
        },
        generated_at=20_000.0,
    )
    markdown = render_incident_dossier_markdown(dossier)

    assert dossier["incident"]["incident_id"] == "alert-1"
    assert dossier["selected_event"]["component"] == "web_search"
    assert {row["event_id"] for row in dossier["causal_chain"]} == {
        "request-1", "tool-1"
    }
    assert dossier["root_causes"]
    assert dossier["privacy"] == {
        "raw_inputs_included": False,
        "raw_outputs_included": False,
        "raw_errors_included": False,
        "annotation_notes_included": False,
    }
    assert "SECRET" not in str(dossier)
    assert "SECRET" not in markdown
    assert "Privacy boundary" in markdown


def test_incident_eval_criteria_and_candidate_evaluation_are_deterministic():
    failing_records = [
        _record(
            "tool-1", "TOOL", session_id="conversation-1",
            status="failed", error="SECRET failure", details={"name": "web_search"},
        )
    ]
    dossier = build_incident_dossier(
        failing_records,
        incident={
            "alert_id": "alert-1", "source_type": "policy",
            "metric": "tool.failure_rate", "baseline": 0.2,
            "observed": 1, "conversation_id": "conversation-1",
            "event_id": "tool-1", "severity": "error", "status": "open",
        },
        generated_at=200.0,
    )
    criteria = derive_incident_eval_criteria(dossier)
    baseline = evaluate_trajectory_regression(failing_records, criteria=criteria)
    passing_records = [
        _record(
            "request-2", "REQUEST", session_id="conversation-2",
            request_id="request-2", status="completed", error="",
        )
    ]
    candidate = evaluate_trajectory_regression(
        passing_records,
        criteria={**criteria, "min_trace_coverage": 0},
        baseline_snapshot=baseline,
    )

    assert criteria["max_failure_rate"] == 0.2
    assert criteria["max_error_count"] == 0
    assert criteria["blocked_fingerprints"]
    assert baseline["status"] == "failed"
    assert candidate["status"] == "passed"
    assert candidate["comparison"]["removed_fingerprints"] == criteria["blocked_fingerprints"]
    assert "SECRET" not in str(baseline)
    assert "SECRET" not in str(candidate)


def test_alert_policies_target_provider_model_and_tool_aggregates():
    analytics = {
        "sessions": [{
            "conversation_id": "session-1", "failure_rate": 0.2,
            "avg_request_ms": 2_000, "tokens_per_request": 500,
            "problem_event_id": "tool-1", "latest_event_id": "tool-1",
        }],
        "providers": [{
            "provider": "provider-a", "model": "model-a", "attempts": 4,
            "failures": 2, "p95_ms": 5_000,
        }],
        "tools": [{
            "name": "web_search", "calls": 3, "failures": 1,
            "p95_ms": 12_000,
        }],
    }
    policies = [
        {
            "policy_id": "policy-provider", "name": "Provider guard",
            "scope_type": "provider", "scope_value": "provider-a", "enabled": True,
            "thresholds": {
                "failure_rate_warning": 0.25, "failure_rate_critical": 0.5,
                "latency_warning_ms": 4_000, "latency_critical_ms": 10_000,
            },
        },
        {
            "policy_id": "policy-tool", "name": "Search guard",
            "scope_type": "tool", "scope_value": "web_search", "enabled": True,
            "thresholds": {
                "latency_warning_ms": 10_000, "latency_critical_ms": 11_000,
            },
        },
    ]

    signals = evaluate_trajectory_policies(analytics, policies)

    assert {row["policy_id"] for row in signals} == {"policy-provider", "policy-tool"}
    provider = next(row for row in signals if row["policy_id"] == "policy-provider")
    assert provider["regression"]["severity"] == "error"
    assert {reason["metric"] for reason in provider["regression"]["reasons"]} == {
        "provider.failure_rate", "provider.latency"
    }
    tool = next(row for row in signals if row["policy_id"] == "policy-tool")
    assert tool["regression"]["reasons"][0]["threshold_level"] == "critical"


@pytest.mark.asyncio
async def test_project_trajectory_analytics_route_is_scoped_and_paginated(monkeypatch):
    from remy.web.routes import trajectory_routes

    captured = {}
    conversations = [SimpleNamespace(conversation_id="conversation-1", title="One")]
    conversation_store = SimpleNamespace(
        list=lambda **kwargs: conversations,
    )
    project_records = [
        _record("request-1", "REQUEST", session_id="conversation-1")
    ]
    def list_project_events(**kwargs):
        captured["store"] = kwargs
        return project_records

    trajectory_store = SimpleNamespace(
        list_project_events=list_project_events,
        count_project_events=lambda **kwargs: 25_123,
        get_active_analytics_baseline=lambda **kwargs: None,
        list_analytics_baselines=lambda **kwargs: [],
        list_alert_policies=lambda **kwargs: [],
        list_regression_alerts=lambda **kwargs: [],
        list_slo_incidents=lambda **kwargs: [],
        list_eval_cases=lambda **kwargs: [],
        list_eval_matrices=lambda **kwargs: [],
        list_eval_comparisons=lambda **kwargs: [],
        list_model_promotions=lambda **kwargs: [],
        get_slo_config=lambda **kwargs: {
            "project_id": kwargs["project_id"],
            "target_success_rate": 0.99,
            "window_days": 30,
            "min_operations": 5,
            "updated_at": "",
        },
        list_alert_history=lambda **kwargs: [],
    )

    def analyze(rows, **kwargs):
        captured["rows"] = rows
        captured["analytics"] = kwargs
        return {"summary": {"sessions": 1}, "sessions": []}

    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(
        trajectory_routes, "get_conversation_store", lambda _: conversation_store
    )
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: trajectory_store)
    monkeypatch.setattr(trajectory_routes, "analyze_project_trajectory", analyze)

    payload = await trajectory_routes.get_project_trajectory_analytics(days=30, limit=20_000)

    assert captured["store"] == {"project_id": "project-1", "limit": 20_000}
    assert captured["analytics"] == {
        "conversation_titles": {"conversation-1": "One"},
        "days": 30,
        "baseline_metrics": None,
    }
    assert payload["project_id"] == "project-1"
    assert payload["pagination"] == {
        "limit": 20_000,
        "returned": 1,
        "estimated_total": 25_123,
        "window_truncated": True,
    }
    assert payload["active_baseline"] is None
    assert payload["baselines"] == []
    assert payload["policies"] == []
    assert payload["alerts"] == []
    assert payload["slo_incidents"] == []
    assert payload["eval_cases"] == []
    assert payload["eval_matrices"] == []
    assert payload["eval_comparisons"] == []
    assert payload["promotion_recommendation"]["status"] == "insufficient-data"
    assert payload["model_promotions"] == []
    assert payload["active_model_promotion"] is None
    assert payload["canary_telemetry"]["status"] == "inactive"
    assert payload["alert_history"] == []
    assert payload["slo_config"]["target_success_rate"] == 0.99
    assert payload["slo"]["status"] == "insufficient-data"


@pytest.mark.asyncio
async def test_trajectory_baseline_and_alert_routes_remain_project_scoped(monkeypatch):
    from remy.web.routes import trajectory_routes

    captured = {}

    def create_baseline(**kwargs):
        captured["baseline"] = kwargs
        return {"baseline_id": "baseline-1", "project_id": kwargs["project_id"]}

    def update_alert(**kwargs):
        captured["alert"] = kwargs
        return {"alert_id": kwargs["alert_id"], "status": kwargs["status"]}

    def create_policy(**kwargs):
        captured["policy"] = kwargs
        return {"policy_id": "policy-1", "project_id": kwargs["project_id"]}

    def update_slo(**kwargs):
        captured["slo"] = kwargs
        return dict(kwargs)

    def update_slo_incident(**kwargs):
        captured["slo_incident"] = kwargs
        return {"incident_id": kwargs["incident_id"], "status": kwargs["status"]}

    trajectory_store = SimpleNamespace(
        list_project_events=lambda **kwargs: [_record("request-1", "REQUEST")],
        create_analytics_baseline=create_baseline,
        update_regression_alert=update_alert,
        create_alert_policy=create_policy,
        update_slo_config=update_slo,
        list_alert_history=lambda **kwargs: [{
            "alert_id": kwargs.get("alert_id"), "project_id": kwargs["project_id"]
        }],
        update_slo_incident=update_slo_incident,
        list_slo_incidents=lambda **kwargs: [{
            "incident_id": "slo-1", "project_id": kwargs["project_id"]
        }],
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: trajectory_store)
    monkeypatch.setattr(
        trajectory_routes,
        "analyze_project_trajectory",
        lambda *args, **kwargs: {
            "summary": {"sessions": 2, "requests": 4},
            "window_baseline": {
                "failure_rate": 0,
                "avg_request_ms": 500,
                "tokens_per_request": 120,
            },
        },
    )

    baseline_payload = await trajectory_routes.create_project_trajectory_baseline(
        trajectory_routes.TrajectoryBaselinePayload(
            name="Stable", days=30, activate=True
        )
    )
    alert_payload = await trajectory_routes.update_project_trajectory_alert(
        "alert-1",
        trajectory_routes.TrajectoryAlertPayload(status="acknowledged"),
    )
    policy_payload = await trajectory_routes.create_project_trajectory_policy(
        trajectory_routes.TrajectoryAlertPolicyPayload(
            name="Provider guard",
            scope_type="provider",
            scope_value="provider-a",
            failure_rate_warning=0.1,
            failure_rate_critical=0.5,
            latency_warning_ms=5_000,
            latency_critical_ms=20_000,
        )
    )
    slo_payload = await trajectory_routes.update_project_trajectory_slo(
        trajectory_routes.TrajectorySloPayload(
            target_success_rate=0.995, window_days=14, min_operations=25
        )
    )
    history_payload = await trajectory_routes.list_project_trajectory_alert_history(
        alert_id="alert-1", limit=25
    )
    incident_payload = await trajectory_routes.update_project_trajectory_slo_incident(
        "slo-1", trajectory_routes.TrajectoryAlertPayload(status="acknowledged")
    )
    incidents_payload = await trajectory_routes.list_project_trajectory_slo_incidents(
        status="open", limit=25
    )

    assert baseline_payload["baseline"]["baseline_id"] == "baseline-1"
    assert captured["baseline"]["project_id"] == "project-1"
    assert captured["baseline"]["metrics"]["avg_request_ms"] == 500
    assert captured["baseline"]["summary"]["requests"] == 4
    assert alert_payload["alert"]["status"] == "acknowledged"
    assert captured["alert"] == {
        "project_id": "project-1",
        "alert_id": "alert-1",
        "status": "acknowledged",
    }
    assert policy_payload["policy"]["policy_id"] == "policy-1"
    assert captured["policy"]["project_id"] == "project-1"
    assert captured["policy"]["scope_type"] == "provider"
    assert captured["policy"]["thresholds"]["failure_rate_critical"] == 0.5
    assert slo_payload["slo_config"]["project_id"] == "project-1"
    assert captured["slo"] == {
        "project_id": "project-1",
        "target_success_rate": 0.995,
        "window_days": 14,
        "min_operations": 25,
    }
    assert history_payload["history"][0]["alert_id"] == "alert-1"
    assert incident_payload["incident"]["status"] == "acknowledged"
    assert captured["slo_incident"] == {
        "project_id": "project-1",
        "incident_id": "slo-1",
        "status": "acknowledged",
    }
    assert incidents_payload["incidents"][0]["project_id"] == "project-1"


@pytest.mark.asyncio
async def test_trajectory_incident_dossier_route_is_project_scoped(monkeypatch):
    from remy.web.routes import trajectory_routes

    alert = {
        "alert_id": "alert-1", "project_id": "project-1",
        "conversation_id": "conversation-1", "event_id": "tool-1",
        "source_type": "policy", "metric": "tool.failure_rate",
        "severity": "error", "status": "open", "observed": 1,
        "baseline": 0.2, "delta": 0.8, "detected_at": "now", "updated_at": "now",
    }
    trajectory_store = SimpleNamespace(
        list_regression_alerts=lambda **kwargs: [alert],
        list_slo_incidents=lambda **kwargs: [],
        list_events=lambda **kwargs: [
            _record(
                "tool-1", "TOOL", session_id="conversation-1",
                status="failed", error="SECRET raw failure",
                details={"name": "web_search"},
            )
        ],
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: trajectory_store)

    payload = await trajectory_routes.get_project_trajectory_incident_dossier("alert-1")

    assert payload["project_id"] == "project-1"
    assert payload["dossier"]["incident"]["incident_id"] == "alert-1"
    assert "SECRET" not in str(payload)


@pytest.mark.asyncio
async def test_trajectory_eval_case_create_and_run_routes_are_project_scoped(monkeypatch):
    from remy.web.routes import trajectory_routes

    captured = {}
    records = [_record("request-1", "REQUEST", session_id="conversation-1")]
    dossier = {
        "incident": {
            "incident_id": "alert-1", "metric": "failure_rate",
            "threshold": 0.2, "conversation_id": "conversation-1",
        },
        "selected_event": {"kind": "REQUEST"},
        "error_clusters": [],
    }
    case = {
        "case_id": "case-1", "project_id": "project-1",
        "incident_id": "alert-1",
        "source_conversation_id": "conversation-1",
        "criteria": {"max_failure_rate": 0.2, "max_error_count": 0},
        "baseline_snapshot": {"metrics": {}},
    }

    def create_case(**kwargs):
        captured["create"] = kwargs
        return {**case, **kwargs}

    def record_run(**kwargs):
        captured["run"] = kwargs
        return {
            "run_id": "run-1", "case_id": kwargs["case_id"],
            "candidate_conversation_id": kwargs["candidate_conversation_id"],
            "status": kwargs["evaluation"]["status"],
            "mode": kwargs.get("mode", "recorded"),
            "replay": kwargs.get("replay") or {},
        }

    trajectory_store = SimpleNamespace(
        get_slo_config=lambda **kwargs: {"target_success_rate": 0.99},
        create_eval_case=create_case,
        get_eval_case=lambda **kwargs: case,
        list_events=lambda **kwargs: records,
        record_eval_run=record_run,
    )
    conversation_store = SimpleNamespace(
        require=lambda conversation_id: SimpleNamespace(
            conversation_id=conversation_id, project_id="project-1"
        )
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: trajectory_store)
    monkeypatch.setattr(
        trajectory_routes, "get_conversation_store", lambda _: conversation_store
    )
    monkeypatch.setattr(
        trajectory_routes,
        "_incident_dossier_data",
        lambda **kwargs: (dossier, "markdown", records),
    )
    async def replay_case(**kwargs):
        return {
            "records": records,
            "replay": {
                "sandboxed": True, "status": "completed",
                "side_effects_executed": 0, "tool_calls": 0,
                "fixture_hits": 0, "blocked_calls": 0, "decisions": [],
            },
            "conversation": SimpleNamespace(
                conversation_id="sandbox-conversation",
                to_dict=lambda: {"conversation_id": "sandbox-conversation"},
            ),
        }
    monkeypatch.setattr(
        trajectory_routes, "run_trajectory_sandbox_replay", replay_case
    )

    created = await trajectory_routes.create_project_trajectory_eval_case(
        "alert-1", trajectory_routes.TrajectoryEvalCasePayload(name="Search regression")
    )
    run = await trajectory_routes.run_project_trajectory_eval_case(
        "case-1",
        trajectory_routes.TrajectoryEvalRunPayload(conversation_id="conversation-1"),
    )
    replay = await trajectory_routes.sandbox_replay_project_trajectory_eval_case(
        "case-1"
    )

    assert created["case"]["project_id"] == "project-1"
    assert captured["create"]["incident_id"] == "alert-1"
    assert captured["create"]["criteria"]["max_failure_rate"] == 0.2
    assert run["run"]["candidate_conversation_id"] == "conversation-1"
    assert captured["run"]["project_id"] == "project-1"
    assert replay["run"]["mode"] == "sandbox-replay"
    assert captured["run"]["replay"]["side_effects_executed"] == 0


@pytest.mark.asyncio
async def test_trajectory_eval_matrix_runs_all_cases_and_blocks_gate_on_error(monkeypatch):
    from remy.web.routes import trajectory_routes

    cases = [{
        "case_id": f"case-{index}", "incident_id": f"alert-{index}",
        "project_id": "project-1", "enabled": True,
        "name": f"Regression {index}",
        "source_conversation_id": f"source-{index}",
        "criteria": {"max_error_count": 0, "min_trace_coverage": 0},
        "baseline_snapshot": {"metrics": {}},
    } for index in (1, 2)]
    entries = []
    observed_models = []

    def record_entry(**kwargs):
        entries.append(kwargs)
        return kwargs

    def complete_matrix(**kwargs):
        return {
            "matrix_id": kwargs["matrix_id"], "status": "failed",
            "gate_passed": False, "case_count": 2,
            "passed_count": 1, "failed_count": 0, "error_count": 1,
            "entries": entries,
        }

    store = SimpleNamespace(
        list_eval_cases=lambda **kwargs: cases,
        create_eval_matrix=lambda **kwargs: {"matrix_id": "matrix-1", **kwargs},
        record_eval_run=lambda **kwargs: {
            "run_id": f"run-{kwargs['case_id']}", "status": kwargs["evaluation"]["status"]
        },
        record_eval_matrix_entry=record_entry,
        complete_eval_matrix=complete_matrix,
    )

    async def replay(**kwargs):
        observed_models.append(kwargs["preferred_model"])
        if kwargs["case"]["case_id"] == "case-2":
            raise RuntimeError("SECRET provider details")
        return {
            "records": [_record(
                "request-1", "REQUEST", session_id="sandbox-1",
                details={"usage": {}},
            )],
            "replay": {
                "sandboxed": True, "status": "completed", "tool_calls": 0,
                "fixture_hits": 0, "blocked_calls": 0, "decisions": [],
            },
            "conversation": SimpleNamespace(conversation_id="sandbox-1"),
        }

    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: store)
    monkeypatch.setattr(
        trajectory_routes, "get_conversation_store", lambda _: SimpleNamespace()
    )
    monkeypatch.setattr(
        trajectory_routes, "_incident_dossier_data",
        lambda **kwargs: (
            {"selected_event": {"event_id": "request-1", "turn_id": "turn-1"}},
            "markdown", [],
        ),
    )
    monkeypatch.setattr(trajectory_routes, "run_trajectory_sandbox_replay", replay)
    monkeypatch.setattr(trajectory_routes, "_runtime_agent_version", lambda: "0.9.0")

    payload = await trajectory_routes.run_project_trajectory_eval_matrix(
        trajectory_routes.TrajectoryEvalMatrixPayload(
            name="Release gate", preferred_model="provider/model-v2"
        )
    )

    assert payload["matrix"]["gate_passed"] is False
    assert observed_models == ["provider/model-v2", "provider/model-v2"]
    assert [entry["status"] for entry in entries] == ["passed", "error"]
    assert entries[1]["error_type"] == "RuntimeError"
    assert "SECRET" not in str(entries)


@pytest.mark.asyncio
async def test_trajectory_eval_comparison_runs_models_sequentially_and_returns_winner(
    monkeypatch,
):
    from remy.web.routes import trajectory_routes

    cases = [{
        "case_id": "case-1", "enabled": True, "incident_id": "alert-1",
    }]
    observed_models = []
    recorded = []

    async def run_matrix(**kwargs):
        model = kwargs["preferred_model"]
        observed_models.append(model)
        passed = model == "provider/safe"
        return {
            "matrix_id": f"matrix-{len(observed_models)}",
            "preferred_model": model,
            "status": "passed" if passed else "failed",
            "gate_passed": passed,
            "passed_count": int(passed),
            "failed_count": int(not passed),
            "error_count": 0,
            "avg_score": 100 if passed else 80,
            "avg_failure_rate": 0,
            "avg_request_ms": 800 if passed else 300,
            "avg_tokens_per_request": 900 if passed else 600,
        }

    def record_model(**kwargs):
        recorded.append(kwargs["matrix"])
        return kwargs["matrix"]

    store = SimpleNamespace(
        list_eval_cases=lambda **kwargs: cases,
        create_eval_comparison=lambda **kwargs: {
            "comparison_id": "comparison-1", **kwargs,
        },
        record_eval_comparison_model=record_model,
        complete_eval_comparison=lambda **kwargs: {
            "comparison_id": kwargs["comparison_id"],
            "status": "completed", "winner_model": "provider/safe",
            "models": recorded,
        },
        get_active_model_promotion=lambda **kwargs: None,
        observe_model_promotion_comparison=lambda **kwargs: None,
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: store)
    monkeypatch.setattr(trajectory_routes, "_run_eval_matrix", run_matrix)
    monkeypatch.setattr(trajectory_routes, "_runtime_agent_version", lambda: "0.10.0")

    payload = await trajectory_routes.run_project_trajectory_eval_comparison(
        trajectory_routes.TrajectoryEvalComparisonPayload(
            name="Models", models=["provider/fast", "provider/safe"]
        )
    )

    assert observed_models == ["provider/fast", "provider/safe"]
    assert [row["preferred_model"] for row in recorded] == observed_models
    assert payload["comparison"]["winner_model"] == "provider/safe"


@pytest.mark.asyncio
async def test_trajectory_model_promotion_routes_require_confirmation_and_restore_rollback(
    monkeypatch,
):
    from remy.web.routes import trajectory_routes

    active = {
        "promotion_id": "promotion-1",
        "project_id": "project-1",
        "previous_model": "provider/current",
        "candidate_model": "provider/candidate",
        "status": "ready",
        "ramp_complete": False,
    }
    created = {}
    runtime_updates = []

    def create_promotion(**kwargs):
        created.update(kwargs)
        return {"promotion_id": "promotion-1", "status": "canary", **kwargs}

    def set_runtime(key, value, *, target):
        runtime_updates.append((key, value))
        setattr(target, key, value)

    store = SimpleNamespace(
        list_eval_comparisons=lambda **kwargs: [{"comparison_id": "comparison-1"}],
        create_model_promotion=create_promotion,
        get_model_promotion=lambda **kwargs: dict(active),
        mark_model_promotion_promoted=lambda **kwargs: {**active, "status": "promoted"},
        rollback_model_promotion=lambda **kwargs: {**active, "status": "rolled_back"},
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: store)
    monkeypatch.setattr(trajectory_routes.settings, "SUMMARY_MODEL", "provider/current")
    monkeypatch.setattr(trajectory_routes, "set_runtime_setting", set_runtime)
    monkeypatch.setattr(
        trajectory_routes,
        "build_model_promotion_recommendation",
        lambda *args, **kwargs: {
            "status": "promote",
            "recommended_model": "provider/candidate",
            "evidence_comparison_ids": ["comparison-1", "comparison-2", "comparison-3"],
        },
    )

    started = await trajectory_routes.start_project_model_promotion(
        trajectory_routes.TrajectoryModelPromotionPayload(
            candidate_model="provider/candidate",
            confirm_model="provider/candidate",
            canary_percent=20,
        )
    )
    assert started["summary_model_changed"] is False
    assert created["previous_model"] == "provider/current"
    assert created["canary_percent"] == 10

    with pytest.raises(trajectory_routes.HTTPException) as blocked:
        await trajectory_routes.finalize_project_model_promotion(
            "promotion-1",
            trajectory_routes.TrajectoryModelPromotionActionPayload(
                confirm_model="provider/candidate"
            ),
        )
    assert blocked.value.status_code == 409
    assert runtime_updates == []
    active["ramp_complete"] = True

    promoted = await trajectory_routes.finalize_project_model_promotion(
        "promotion-1",
        trajectory_routes.TrajectoryModelPromotionActionPayload(
            confirm_model="provider/candidate"
        ),
    )
    assert promoted["summary_model"] == "provider/candidate"
    assert runtime_updates[-1] == ("SUMMARY_MODEL", "provider/candidate")

    active["status"] = "promoted"
    rolled_back = await trajectory_routes.rollback_project_model_promotion(
        "promotion-1",
        trajectory_routes.TrajectoryModelPromotionActionPayload(
            confirm_model="provider/current"
        ),
    )
    assert rolled_back["summary_model"] == "provider/current"
    assert runtime_updates[-1] == ("SUMMARY_MODEL", "provider/current")


@pytest.mark.asyncio
async def test_model_comparison_automatically_restores_promoted_model_on_regression(
    monkeypatch,
):
    from remy.web.routes import trajectory_routes

    updates = []
    active = {
        "promotion_id": "promotion-1",
        "status": "promoted",
        "previous_model": "provider/current",
        "candidate_model": "provider/candidate",
    }

    async def run_matrix(**kwargs):
        return {
            "matrix_id": f"matrix-{kwargs['preferred_model']}",
            "preferred_model": kwargs["preferred_model"],
        }

    store = SimpleNamespace(
        list_eval_cases=lambda **kwargs: [{"case_id": "case-1", "enabled": True}],
        create_eval_comparison=lambda **kwargs: {
            "comparison_id": "comparison-new", **kwargs,
        },
        record_eval_comparison_model=lambda **kwargs: kwargs["matrix"],
        complete_eval_comparison=lambda **kwargs: {
            "comparison_id": kwargs["comparison_id"],
            "status": "completed",
            "winner_model": "provider/current",
            "models": [],
        },
        get_active_model_promotion=lambda **kwargs: dict(active),
        observe_model_promotion_comparison=lambda **kwargs: {
            **active, "status": "rollback_pending",
            "rollback_reason": "candidate_gate_failed",
        },
        complete_model_promotion_auto_rollback=lambda **kwargs: {
            **active, "status": "rolled_back", "rollback_reason": "candidate_gate_failed",
        },
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: store)
    monkeypatch.setattr(trajectory_routes, "_run_eval_matrix", run_matrix)
    monkeypatch.setattr(trajectory_routes, "_runtime_agent_version", lambda: "0.10.0")
    monkeypatch.setattr(
        trajectory_routes,
        "set_runtime_setting",
        lambda key, value, *, target: updates.append((key, value)),
    )

    payload = await trajectory_routes.run_project_trajectory_eval_comparison(
        trajectory_routes.TrajectoryEvalComparisonPayload(
            models=["provider/current", "provider/candidate"]
        )
    )

    assert payload["auto_rollback"] is True
    assert updates == [("SUMMARY_MODEL", "provider/current")]


@pytest.mark.asyncio
async def test_trajectory_route_includes_diagnostics(monkeypatch):
    from remy.web.routes import trajectory_routes

    conversation = SimpleNamespace(
        conversation_id="conversation-1",
        project_id="project-1",
    )
    conversation_store = SimpleNamespace(require=lambda _: conversation)
    trajectory_store = SimpleNamespace(
        list_events=lambda **_: [_record("request-1", "REQUEST")],
        count_events=lambda **_: 2_000,
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(
        trajectory_routes,
        "get_conversation_store",
        lambda _: conversation_store,
    )
    monkeypatch.setattr(
        trajectory_routes,
        "get_trajectory_store",
        lambda: trajectory_store,
    )
    monkeypatch.setattr(trajectory_routes, "_legacy_projection", lambda *_: [])

    payload = await trajectory_routes.get_conversation_trajectory(
        "conversation-1",
        limit=100,
    )

    assert payload["diagnostics"]["health"] == "healthy"
    assert payload["records"][0]["diagnostic"]["severity"] == "ok"
    assert payload["pagination"] == {
        "limit": 100,
        "returned": 1,
        "estimated_total": 2_000,
        "has_more": True,
        "window_truncated": True,
        "next_limit": 600,
    }
    assert payload["diagnostics"]["integrity"]["window_truncated"] is True
    assert payload["event_log"]["status"] == "unavailable"
    assert payload["turns"] == [{
        "turn_id": "turn-1",
        "index": 1,
        "status": "completed",
        "started_at": 100.0,
        "completed_at": 101.0,
        "duration_ms": 1000,
        "record_count": 1,
        "request_count": 1,
        "tool_count": 0,
        "attempt_count": 0,
        "failure_count": 0,
        "error_count": 0,
        "warning_count": 0,
        "total_tokens": 0,
        "user_preview": "",
        "assistant_preview": "",
    }]


@pytest.mark.asyncio
async def test_trajectory_annotation_route_is_project_scoped(monkeypatch):
    from remy.web.routes import trajectory_routes

    conversation = SimpleNamespace(
        conversation_id="conversation-1",
        project_id="project-1",
    )
    conversation_store = SimpleNamespace(require=lambda _: conversation)
    calls = []
    trajectory_store = SimpleNamespace(
        set_annotation=lambda **kwargs: calls.append(kwargs) or {
            "label": kwargs["label"],
            "note": kwargs["note"],
            "bookmarked": kwargs["bookmarked"],
            "updated_at": "2026-08-20T00:00:00+00:00",
        }
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(
        trajectory_routes, "get_conversation_store", lambda _: conversation_store
    )
    monkeypatch.setattr(
        trajectory_routes, "get_trajectory_store", lambda: trajectory_store
    )

    payload = await trajectory_routes.update_trajectory_annotation(
        "conversation-1",
        "event-1",
        trajectory_routes.TrajectoryAnnotationPayload(
            label="bug", note="Unexpected fallback", bookmarked=True
        ),
    )

    assert payload["annotation"]["label"] == "bug"
    assert calls == [{
        "project_id": "project-1",
        "session_id": "conversation-1",
        "event_id": "event-1",
        "label": "bug",
        "note": "Unexpected fallback",
        "bookmarked": True,
    }]


@pytest.mark.asyncio
async def test_trajectory_fork_route_copies_history_without_executing_tools(monkeypatch):
    from remy.web.routes import trajectory_routes

    source = SimpleNamespace(
        conversation_id="source-conversation",
        project_id="project-1",
        brain_id="brain-1",
        title="Source chat",
    )
    fork = SimpleNamespace(
        conversation_id="fork-conversation",
        project_id="project-1",
        brain_id="brain-1",
        title="Forked chat",
        to_dict=lambda: {
            "conversation_id": "fork-conversation",
            "project_id": "project-1",
            "brain_id": "brain-1",
            "title": "Forked chat",
            "metadata": {},
        },
    )
    created = []
    conversation_store = SimpleNamespace(
        require=lambda _: source,
        create=lambda *args, **kwargs: created.append((args, kwargs)) or fork,
        get_active_id=lambda: source.conversation_id,
        archive=lambda _: None,
    )
    copied = []
    transcript_store = SimpleNamespace(append=lambda **kwargs: copied.append(kwargs))
    fork_events = []
    trajectory_store = SimpleNamespace(
        record_fork=lambda **kwargs: fork_events.append(kwargs)
    )

    async def trajectory_payload(*_, **__):
        return {
            "records": [
                _record("user-1", "USER", sequence=1, turn_id="turn-1", input="first"),
                _record("assistant-1", "ASSISTANT", sequence=2, turn_id="turn-1", output="done"),
                _record("user-2", "USER", sequence=3, turn_id="turn-2", input="retry me"),
                _record("tool-2", "TOOL", sequence=4, turn_id="turn-2", output="unsafe"),
            ]
        }

    class Manager:
        async def switch_conversation(self, project_id, conversation_id):
            assert (project_id, conversation_id) == ("project-1", "fork-conversation")
            return SimpleNamespace(session_id=conversation_id)

    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(
        trajectory_routes, "get_conversation_store", lambda _: conversation_store
    )
    monkeypatch.setattr(trajectory_routes, "get_transcript_store", lambda: transcript_store)
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: trajectory_store)
    monkeypatch.setattr(trajectory_routes, "get_conversation_trajectory", trajectory_payload)
    monkeypatch.setattr(
        trajectory_routes,
        "_get_api",
        lambda: SimpleNamespace(get_session_manager=lambda: Manager()),
    )

    result = await trajectory_routes.fork_conversation_from_trajectory(
        "source-conversation",
        "tool-2",
        trajectory_routes.TrajectoryForkPayload(
            title="Forked chat", preferred_model="test-model", activate=True
        ),
    )

    assert [item["role"] for item in copied] == ["user", "assistant"]
    assert [item["content"] for item in copied] == ["first", "done"]
    assert all("unsafe" not in str(item) for item in copied)
    assert created[0][1]["activate"] is False
    assert fork_events[0]["boundary_event_id"] == "tool-2"
    assert result["fork"]["next_prompt"] == "retry me"
    assert result["fork"]["auto_executed"] is False
    assert result["active_conversation_id"] == "fork-conversation"
