from datetime import datetime, timezone
from unittest.mock import patch

from remy.core.claim_lifecycle import (
    claim_subject_key,
    claim_value_signature,
    is_lifecycle_claim,
    load_claim_lifecycle,
    record_claim_lifecycle,
    lifecycle_scope_key,
    trajectory_lifecycle_view,
    update_claim_lifecycle,
)


NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def _row(claim, url, date, root, *, authority=0.8, valid=True, **extra):
    return {
        "claim_id": f"claim-{claim_value_signature(claim)}",
        "claim": claim,
        "status": "supported",
        "relations": [
            {
                "relation": "supports",
                "url": url,
                "source_date": date,
                "evidence_root": root,
                "authority_role": "primary",
                "authority_score": authority,
                "temporally_valid": valid,
            }
        ],
        **extra,
    }


def _matrix(*rows):
    return {"version": 4, "rows": list(rows)}


def test_subject_key_and_value_signature_separate_stable_subject_from_price():
    old = "The current service price is $20 per month."
    new = "The current service price is $25 per month."

    assert claim_subject_key(old) == claim_subject_key(new)
    assert claim_value_signature(old) != claim_value_signature(new)
    assert is_lifecycle_claim(old) is True


def test_scope_is_workspace_bound_and_does_not_split_on_topic_wording():
    assert lifecycle_scope_key("project-1", "service pricing") == lifecycle_scope_key(
        "project-1", "latest service cost"
    )
    assert lifecycle_scope_key("project-1", "pricing") != lifecycle_scope_key(
        "project-2", "pricing"
    )


def test_immutable_scientific_result_is_not_tracked():
    ledger = update_claim_lifecycle(
        None,
        _matrix(
            _row(
                "The experiment reports accuracy of 82 percent.",
                "https://paper.example/result",
                "2026-08-20T00:00:00+00:00",
                "https://paper.example/result",
            )
        ),
        topic="experiment",
        observed_at=NOW,
    )

    assert ledger["summary"]["tracked_subjects"] == 0


def test_first_observation_creates_current_state_and_history():
    ledger = update_claim_lifecycle(
        None,
        _matrix(
            _row(
                "The current service price is $20 per month.",
                "https://old.example/pricing",
                "2026-08-20T00:00:00+00:00",
                "https://old.example/pricing",
            )
        ),
        project_id="project-1",
        topic="service pricing",
        observed_at=NOW,
    )

    subject = next(iter(ledger["subjects"].values()))
    assert subject["current"]["value_signature"] == "$20"
    assert subject["history"][0]["event_type"] == "first_seen"
    assert ledger["last_run"]["new_subjects"] == 1


def test_same_value_reaffirms_without_duplicating_history():
    row = _row(
        "The current service price is $20 per month.",
        "https://old.example/pricing",
        "2026-08-20T00:00:00+00:00",
        "https://old.example/pricing",
    )
    first = update_claim_lifecycle(None, _matrix(row), topic="pricing", observed_at=NOW)
    second = update_claim_lifecycle(
        first,
        _matrix(row),
        topic="pricing",
        observed_at=datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc),
    )

    subject = next(iter(second["subjects"].values()))
    assert len(subject["history"]) == 1
    assert subject["current"]["observation_count"] == 2
    assert second["last_run"]["reaffirmed_claims"] == 1


def test_cumulative_transition_count_survives_later_reaffirmation():
    old = _row(
        "The current service price is $20 per month.",
        "https://old.example/pricing",
        "2026-08-15T00:00:00+00:00",
        "https://old.example/pricing",
    )
    new = _row(
        "The current service price is $25 per month.",
        "https://new.example/pricing",
        "2026-08-23T00:00:00+00:00",
        "https://new.example/pricing",
    )
    first = update_claim_lifecycle(None, _matrix(old), topic="pricing", observed_at=NOW)
    changed = update_claim_lifecycle(
        first,
        _matrix(new),
        topic="pricing",
        observed_at=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
    )
    reaffirmed = update_claim_lifecycle(
        changed,
        _matrix(new),
        topic="pricing",
        observed_at=datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert reaffirmed["summary"]["confirmed_transitions"] == 1
    assert reaffirmed["last_run"]["confirmed_transitions"] == 0


def test_newer_independent_fresh_authoritative_value_confirms_transition():
    first = update_claim_lifecycle(
        None,
        _matrix(
            _row(
                "The current service price is $20 per month.",
                "https://old.example/pricing",
                "2026-08-15T00:00:00+00:00",
                "https://old.example/pricing",
            )
        ),
        topic="pricing",
        observed_at=NOW,
    )
    second = update_claim_lifecycle(
        first,
        _matrix(
            _row(
                "The current service price is $25 per month.",
                "https://new.example/pricing",
                "2026-08-23T00:00:00+00:00",
                "https://new.example/pricing",
            )
        ),
        topic="pricing",
        observed_at=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
    )

    subject = next(iter(second["subjects"].values()))
    assert subject["current"]["value_signature"] == "$25"
    assert subject["history"][-1]["event_type"] == "confirmed_transition"
    assert subject["history"][-1]["history_preserved"] is True
    assert second["last_run"]["confirmed_transitions"] == 1


def test_unsafe_change_is_pending_and_does_not_replace_current_value():
    old_row = _row(
        "The current service price is $20 per month.",
        "https://vendor.example/pricing",
        "2026-08-15T00:00:00+00:00",
        "https://vendor.example/pricing",
    )
    first = update_claim_lifecycle(None, _matrix(old_row), topic="pricing", observed_at=NOW)
    candidate = _row(
        "The current service price is $25 per month.",
        "https://vendor.example/pricing",
        "2026-08-23T00:00:00+00:00",
        "https://vendor.example/pricing",
    )
    second = update_claim_lifecycle(
        first,
        _matrix(candidate),
        topic="pricing",
        observed_at=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
    )

    subject = next(iter(second["subjects"].values()))
    assert subject["current"]["value_signature"] == "$20"
    assert subject["pending_changes"][0]["event_type"] == "pending_change"
    assert "independent_evidence_root_required" in subject["pending_changes"][0][
        "decision_reasons"
    ]


def test_in_run_supersession_seeds_old_and_new_history():
    supersession = {
        "old_claim": "The current service price is $20 per month.",
        "new_claim": "The current service price is $25 per month.",
        "old_source_url": "https://old.example/pricing",
        "new_source_url": "https://new.example/pricing",
        "old_source_date": "2026-07-01T00:00:00+00:00",
        "new_source_date": "2026-08-21T00:00:00+00:00",
        "old_evidence_root": "https://old.example/pricing",
        "new_evidence_root": "https://new.example/pricing",
        "old_authority_score": 0.8,
        "new_authority_score": 0.8,
    }
    ledger = update_claim_lifecycle(
        None,
        _matrix(
            _row(
                supersession["new_claim"],
                supersession["new_source_url"],
                supersession["new_source_date"],
                supersession["new_evidence_root"],
                resolved_supersessions=[supersession],
            )
        ),
        topic="pricing",
        observed_at=NOW,
    )

    subject = next(iter(ledger["subjects"].values()))
    assert [item["event_type"] for item in subject["history"]] == [
        "first_seen",
        "confirmed_transition",
    ]
    assert subject["history"][0]["to_value"] == "$20"
    assert subject["current"]["value_signature"] == "$25"


def test_persistence_is_scoped_and_returns_bounded_trajectory_view(tmp_path):
    with patch("remy.core.claim_lifecycle.CLAIM_LIFECYCLE_DIR", tmp_path):
        view = record_claim_lifecycle(
            "project-1",
            "pricing",
            _matrix(
                _row(
                    "The current service price is $20 per month.",
                    "https://old.example/pricing",
                    "2026-08-20T00:00:00+00:00",
                    "https://old.example/pricing",
                )
            ),
            observed_at=NOW,
        )
        stored = load_claim_lifecycle("project-1", "pricing")

    assert stored is not None
    assert view["summary"]["tracked_subjects"] == 1
    assert isinstance(view["subjects"], list)
    assert view["scope_key"] == stored["scope_key"]


def test_trajectory_view_bounds_subject_and_history_windows():
    ledger = {
        "version": 1,
        "scope_key": "scope-1",
        "summary": {"tracked_subjects": 2},
        "subjects": {
            "subject-a": {
                "current": {"claim": "A", "last_seen": "2026-08-22"},
                "history": [{"event_id": f"a-{index}"} for index in range(5)],
                "pending_changes": [{"event_id": f"p-{index}"} for index in range(4)],
            },
            "subject-b": {
                "current": {"claim": "B", "last_seen": "2026-08-21"},
                "history": [{"event_id": "b-1"}],
                "pending_changes": [],
            },
        },
    }

    view = trajectory_lifecycle_view(
        ledger, subject_limit=1, history_limit=2, pending_limit=1
    )

    assert len(view["subjects"]) == 1
    assert len(view["subjects"][0]["history"]) == 2
    assert len(view["subjects"][0]["pending_changes"]) == 1
    assert view["summary"]["trajectory_window_truncated"] is True
