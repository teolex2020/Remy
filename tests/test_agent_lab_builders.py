import pytest

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_builders import (
    validate_builder_fanout_plan,
    validate_builder_shard_proposal,
)


def fanout_plan():
    return {
        "team_required": True,
        "reason": "Independent source modules",
        "members": [
            {
                "id": "entry",
                "instruction": "Own the entrypoint",
                "model": "model-a",
                "file_claims": ["src/main.py"],
            },
            {
                "id": "domain",
                "instruction": "Own domain logic",
                "model": "model-b",
                "file_claims": ["src/domain.py"],
            },
        ],
    }


def test_builder_fanout_rejects_overlap_unavailable_models_and_unsafe_paths():
    overlap = fanout_plan()
    overlap["members"][1]["file_claims"] = ["src/main.py"]
    with pytest.raises(ValueError, match="claim overlap"):
        validate_builder_fanout_plan(
            overlap, max_builders=3, available_models=["model-a", "model-b"]
        )

    unavailable = fanout_plan()
    unavailable["members"][1]["model"] = "unknown"
    with pytest.raises(ValueError, match="model is unavailable"):
        validate_builder_fanout_plan(
            unavailable, max_builders=3, available_models=["model-a", "model-b"]
        )

    escaped = fanout_plan()
    escaped["members"][1]["file_claims"] = ["../domain.py"]
    with pytest.raises(ValueError, match="exact Python file inside src"):
        validate_builder_fanout_plan(
            escaped, max_builders=3, available_models=["model-a", "model-b"]
        )


def test_builder_shard_must_return_exact_claim_set():
    accepted = validate_builder_shard_proposal(
        {
            "rationale": "bounded module",
            "files": [
                {"path": "src/main.py", "purpose": "entry", "content": "print('ok')\n"}
            ],
        },
        file_claims=["src/main.py"],
    )
    assert accepted["files"][0]["path"] == "src/main.py"

    with pytest.raises(ValueError, match="unclaimed file"):
        validate_builder_shard_proposal(
            {
                "files": [
                    {"path": "src/extra.py", "purpose": "escape", "content": "pass\n"}
                ]
            },
            file_claims=["src/main.py"],
        )


def test_file_claim_registration_is_atomic_and_tracks_lifecycle(tmp_path):
    store = AgentLabStore(tmp_path, owner_project_id="project-a")
    run = store.create(goal="Build independently claimed modules")
    store.prepare(run["run_id"])
    assignments = [
        {
            "builder_id": "entry",
            "model": "model-a",
            "workspace_id": "ws-entry",
            "file_claims": ["src/main.py"],
        },
        {
            "builder_id": "domain",
            "model": "model-b",
            "workspace_id": "ws-domain",
            "file_claims": ["src/domain.py"],
        },
    ]
    claims = store.register_file_claims(run["run_id"], assignments)
    assert [claim["status"] for claim in claims] == ["active", "active"]

    with pytest.raises(ValueError, match="claim overlap"):
        store.register_file_claims(
            run["run_id"],
            [{**assignments[0], "builder_id": "other", "workspace_id": "ws-other"}],
        )
    assert len(store.require(run["run_id"])["file_claims"]) == 2

    store.resolve_file_claims(
        run["run_id"], builder_id="entry", status="merged", merge_id="merge-entry"
    )
    entry = store.require(run["run_id"])["file_claims"][0]
    assert entry["status"] == "merged"
    assert entry["merge_id"] == "merge-entry"
