import asyncio
import threading
import time

import pytest

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_coordinator import (
    AgentLabCoordinator,
    _provider_runtime_available,
    _validate_workspace_proposal_sources,
    _validate_intake_response,
    validate_verifier_proposal,
    validate_workspace_proposal,
)


def prepared_store(tmp_path):
    store = AgentLabStore(tmp_path, owner_project_id="project-a")
    record = store.create(goal="Build a verified local result")
    store.prepare(record["run_id"])
    return store, record["run_id"]


def proposal(*, main: str, verify: str, rationale: str = "bounded design"):
    return {
        "rationale": rationale,
        "entrypoint": "src/main.py",
        "verification_entrypoint": "tests/verify.py",
        "files": [
            {"path": "src/main.py", "purpose": "implementation", "content": main},
            {"path": "tests/verify.py", "purpose": "acceptance", "content": verify},
        ],
    }


def verifier_proposal(*, verify: str, rationale: str = "independent evidence check"):
    return {
        "rationale": rationale,
        "verification_entrypoint": "tests/verify.py",
        "files": [
            {
                "path": "tests/verify.py",
                "purpose": "independent acceptance check",
                "content": verify,
            }
        ],
    }


async def single_agent_team_runner(*args, **kwargs):
    return {
        "status": "single_agent",
        "team_required": False,
        "plan": {"reason": "test keeps the legacy single-coordinator path"},
        "results": [],
        "context": "",
        "usage": {"members": 0, "tool_calls": 0, "elapsed_ms": 0},
    }


def test_workspace_proposal_rejects_permission_expansion_and_path_escape():
    raw = proposal(main="print('ok')\n", verify="assert True\n")
    raw["network_access"] = True
    with pytest.raises(ValueError, match="Unknown coordinator fields"):
        validate_workspace_proposal(raw)

    escaped = proposal(main="print('ok')\n", verify="assert True\n")
    escaped["files"][0]["path"] = "../host.py"
    with pytest.raises(ValueError, match="inside src/ or tests/"):
        validate_workspace_proposal(escaped)


def test_workspace_proposal_requires_declared_entrypoints():
    raw = proposal(main="print('ok')\n", verify="assert True\n")
    raw["files"][0]["path"] = "src/other.py"
    with pytest.raises(ValueError, match="include its entrypoint"):
        validate_workspace_proposal(raw)


def test_workspace_proposal_declares_narrow_execution_requirements():
    raw = proposal(main="print('ok')\n", verify="assert True\n")
    raw["execution_requirements"] = {
        "language": "python",
        "artifact_kinds": ["json"],
        "network_required": False,
        "gpu_required": False,
        "minimum_isolation": "hardened_container",
    }
    accepted = validate_workspace_proposal(raw)
    assert accepted["execution_requirements"]["minimum_isolation_rank"] == 20
    assert accepted["execution_requirements"]["artifact_kinds"] == ("json",)

    raw["execution_requirements"]["network_required"] = "false"
    with pytest.raises(ValueError, match="network_required must be a boolean"):
        validate_workspace_proposal(raw)


def test_workspace_proposal_allows_its_own_src_namespace_but_blocks_system_imports():
    local = validate_workspace_proposal(proposal(
        main="def answer():\n    return 42\n",
        verify="from src.main import answer\nassert answer() == 42\n",
    ))
    assert "src" in _validate_workspace_proposal_sources(local)

    unsafe = validate_workspace_proposal(proposal(
        main="import os\nprint(os.getcwd())\n",
        verify="assert True\n",
    ))
    with pytest.raises(ValueError, match="violates Agent Lab policy"):
        _validate_workspace_proposal_sources(unsafe)


def test_independent_verifier_contract_is_tests_only_and_strict():
    accepted = validate_verifier_proposal(
        verifier_proposal(verify="assert True\n")
    )
    assert accepted["verification_entrypoint"] == "tests/verify.py"

    escaped = verifier_proposal(verify="assert True\n")
    escaped["files"][0]["path"] = "src/verify.py"
    with pytest.raises(ValueError, match="inside tests"):
        validate_verifier_proposal(escaped)

    expanded = verifier_proposal(verify="assert True\n")
    expanded["network_access"] = True
    with pytest.raises(ValueError, match="Unknown verifier fields"):
        validate_verifier_proposal(expanded)


def test_interactive_intake_contract_requires_a_concrete_question():
    assert _validate_intake_response({
        "ready": False,
        "question": "Which output format do you need?",
        "missing_fields": ["output format"],
        "reason": "The deliverable changes materially.",
    })["missing_fields"] == ["output format"]

    with pytest.raises(ValueError, match="provide a question"):
        _validate_intake_response({"ready": False, "missing_fields": []})


@pytest.mark.asyncio
async def test_interactive_run_pauses_and_asks_before_building(tmp_path):
    store, run_id = prepared_store(tmp_path)
    store.mutate(run_id, lambda item: item.update({
        "interactive": {
            "enabled": True,
            "awaiting_input": False,
            "question": "",
            "answers": [],
        },
    }))
    team_called = False

    async def team_runner(*args, **kwargs):
        nonlocal team_called
        team_called = True
        return await single_agent_team_runner(*args, **kwargs)

    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: {
            "ready": False,
            "question": "Should the result be an HTML report or a JSON dataset?",
            "missing_fields": ["output format"],
            "reason": "The acceptance check depends on the chosen format.",
        },
        team_runner=team_runner,
    )
    coordinator.start(run_id, model="fake-model")
    task = coordinator._tasks[run_id]
    await asyncio.wait_for(task, timeout=3)

    record = store.require(run_id)
    assert record["status"] == "paused"
    assert record["phase"] == "awaiting_input"
    assert record["interactive"]["awaiting_input"] is True
    assert record["interactive"]["question"].startswith("Should the result")
    assert record["autonomous"]["model_calls"] == 1
    assert record["task_ledger"]["blockers"][-1]["kind"] == "user_input"
    assert team_called is False
    assert record["executions"] == []


def test_verifier_prefers_a_distinct_connected_model(tmp_path, monkeypatch):
    store, run_id = prepared_store(tmp_path)
    store.mutate(
        run_id,
        lambda item: item.update({
            "autonomous": {"served_by": "builder-model"}
        }),
    )
    monkeypatch.setattr(
        "remy.core.agent_lab_coordinator._agent_lab_model_catalog",
        lambda selected, maximum: [
            {"name": "builder-model"},
            {"name": "verifier-model"},
        ],
    )
    coordinator = AgentLabCoordinator(store, team_runner=single_agent_team_runner)

    selected, independence, _ = coordinator._select_verifier_model(
        store.require(run_id),
        coordinator_model="builder-model",
        requested_model="",
    )

    assert selected == "verifier-model"
    assert independence == "cross_model"


def test_provider_runtime_probe_skips_missing_optional_adapter(monkeypatch):
    monkeypatch.setattr(
        "remy.core.agent_lab_coordinator.importlib.util.find_spec",
        lambda module: None if module == "langchain_nvidia_ai_endpoints" else object(),
    )

    assert _provider_runtime_available("nvidia") is False
    assert _provider_runtime_available("google") is True


@pytest.mark.asyncio
async def test_automatic_verifier_falls_back_to_isolated_primary_model(tmp_path, monkeypatch):
    from remy.core.cancellation import CancellationToken

    store, run_id = prepared_store(tmp_path)
    store.mutate(
        run_id,
        lambda item: item.update({"autonomous": {"served_by": "builder-model", "model_calls": 0}}),
    )
    monkeypatch.setattr(
        "remy.core.agent_lab_coordinator._agent_lab_model_catalog",
        lambda selected, maximum: [
            {"name": "builder-model"},
            {"name": "verifier-model"},
        ],
    )
    calls = []

    def model_call(prompt, model):
        calls.append(model)
        if model == "verifier-model":
            raise RuntimeError("provider balance unavailable")
        return verifier_proposal(verify="assert True\n")

    coordinator = AgentLabCoordinator(
        store,
        model_call=model_call,
        team_runner=single_agent_team_runner,
    )
    result = await coordinator._prepare_independent_verifier(
        run_id,
        token=CancellationToken(),
        coordinator_model="builder-model",
        requested_model="",
        deadline=time.monotonic() + 10,
    )

    assert calls == ["verifier-model", "builder-model"]
    assert result["assigned_model"] == "builder-model"
    assert result["independence_level"] == "isolated_context_same_model"
    assert store.require(run_id)["autonomous"]["model_calls"] == 2


@pytest.mark.asyncio
async def test_retry_resumes_last_passed_checkpoint_without_rebuilding(tmp_path):
    store, run_id = prepared_store(tmp_path)
    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: verifier_proposal(
            verify=(
                'from pathlib import Path\n'
                'assert Path("artifacts/result.txt").read_text(encoding="utf-8") == "ready"\n'
            ),
        ),
        team_runner=lambda *args, **kwargs: pytest.fail("checkpoint retry must not rebuild"),
    )
    coordinator.executor.write_file(
        run_id,
        path="src/main.py",
        content=(
            'from pathlib import Path\n'
            'Path("artifacts/result.txt").write_text("ready", encoding="utf-8")\n'
        ),
    )
    first = coordinator.executor.execute(run_id, entrypoint="src/main.py")
    store.append_execution(run_id, first)
    store.mutate(
        run_id,
        lambda item: item.update({
            "autonomous": {
                "entrypoint": "src/main.py",
                "served_by": "builder-model",
                "backend_selection": {"requirements": {}},
            }
        }),
    )
    store.transition(run_id, "running", message="seed checkpoint")
    store.transition(run_id, "paused", message="verifier unavailable")

    coordinator.start(run_id, model="builder-model", max_repair_rounds=2)
    await asyncio.wait_for(coordinator._tasks[run_id], timeout=8)

    record = store.require(run_id)
    assert record["status"] == "completed"
    assert record["autonomous"]["resume_from_checkpoint"] is True
    assert record["autonomous"]["resumed_entrypoint"] == "src/main.py"
    assert len(record["executions"]) == 2
    assert record["verification"][-1]["world_fact"] == "supports"


def test_container_required_coordinator_preflights_before_changing_run(tmp_path, monkeypatch):
    store, run_id = prepared_store(tmp_path)
    monkeypatch.setattr(
        "remy.core.agent_lab_coordinator.probe_agent_lab_container_runtime",
        lambda: {"available": False, "reason": "local runtime image missing"},
    )
    coordinator = AgentLabCoordinator(store, team_runner=single_agent_team_runner)

    with pytest.raises(ValueError, match="cannot start.*image missing"):
        coordinator.start(run_id, isolation_mode="container_required")

    record = store.require(run_id)
    assert record["status"] == "prepared"
    assert record["phase"] == "ready_for_execution"
    assert not coordinator.is_active(run_id)


@pytest.mark.asyncio
async def test_autonomous_coordinator_builds_verifies_and_completes(tmp_path):
    store, run_id = prepared_store(tmp_path)
    calls = []

    def model_call(prompt, model):
        calls.append((prompt, model))
        if "independent Verifier" in prompt:
            return verifier_proposal(
                verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "ready"\nprint("verified")\n'
            )
        return proposal(
            main='from pathlib import Path\nPath("artifacts/result.txt").write_text("ready", encoding="utf-8")\nprint("built")\n',
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "ready"\nprint("verified")\n',
        )

    coordinator = AgentLabCoordinator(
        store, model_call=model_call, team_runner=single_agent_team_runner
    )
    coordinator.start(run_id, model="fake-model", max_repair_rounds=2)
    task = coordinator._tasks[run_id]
    await asyncio.wait_for(task, timeout=8)

    record = store.require(run_id)
    assert record["status"] == "completed"
    assert record["phase"] == "completed"
    assert record["autonomous"]["model_calls"] == 2
    assert record["autonomous"]["model"] == "fake-model"
    assert record["verification"][-1]["world_fact"] == "supports"
    assert {item["name"] for item in record["artifacts"]} >= {
        "result.txt", "proof-pack.json", "proof-pack.md"
    }
    assert record["verifier"]["independence_level"]
    assert record["verification"][-1]["read_only"] is True
    assert record["proof_pack"]["decision"] == "accepted"
    assert any(
        item["node_id"] == "build" and item["status"] == "merged"
        for item in record["workspace_branches"]
    )
    assert record["merge_receipts"][-1]["status"] in {"merged", "no_changes"}
    assert record["autonomous"]["last_workspace_id"].startswith("ws-")
    assert all(step["status"] == "completed" for step in record["plan"])
    assert len(calls) == 2
    assert "Use pathlib for file paths. Never import os, sys" in calls[1][0]


@pytest.mark.asyncio
async def test_central_model_decomposes_assigns_workers_then_synthesizes(tmp_path):
    from remy.core.team_planner import validate_team_plan

    store, run_id = prepared_store(tmp_path)
    prompts = []
    responses = [
        {
            "team_required": True,
            "reason": "Planning and analysis are independent slices",
            "members": [
                {
                    "id": "requirements",
                    "role": "planner",
                    "instruction": "Define bounded requirements and concrete success criteria.",
                    "model": "central-model",
                },
                {
                    "id": "risks",
                    "role": "analyst",
                    "instruction": "Identify failure modes and verification evidence for the build.",
                    "model": "central-model",
                },
            ],
        },
        {
            "team_required": False,
            "reason": "The central synthesis is one tightly coupled entrypoint",
            "members": [],
        },
        proposal(
            main='from pathlib import Path\nPath("artifacts/result.txt").write_text("team-built", encoding="utf-8")\n',
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "team-built"\n',
            rationale="Synthesized requirements and risk findings",
        ),
        verifier_proposal(
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "team-built"\n'
        ),
    ]

    def model_call(prompt, model):
        prompts.append(prompt)
        return responses.pop(0)

    async def team_runner(goal, **kwargs):
        raw = kwargs["planner"](goal, kwargs["mode"], kwargs["limits"])
        plan = validate_team_plan(
            raw,
            mode=kwargs["mode"],
            limits=kwargs["limits"],
            available_models=kwargs["available_models"],
        )
        for member in plan["members"]:
            allowed = set(kwargs["role_tool_ceilings"][member["role"]])
            member["allowed_tools"] = [
                name for name in member["allowed_tools"] if name in allowed
            ]
            member["capability_profile"] = kwargs["capability_profile"]
        return {
            "status": "completed",
            "team_required": True,
            "run_id": "team-run-1",
            "plan": plan,
            "results": [
                {
                    "kind": "team_member_result",
                    "member_id": "requirements",
                    "role": "planner",
                    "status": "success",
                    "output": "Use a deterministic text artifact and exact acceptance check.",
                    "tool_calls": 1,
                    "elapsed_sec": 0.1,
                    "assigned_model": "central-model",
                    "served_by": "central-model",
                },
                {
                    "kind": "team_member_result",
                    "member_id": "risks",
                    "role": "analyst",
                    "status": "success",
                    "output": "Verify artifact content rather than trusting implementation output.",
                    "tool_calls": 1,
                    "elapsed_sec": 0.1,
                    "assigned_model": "central-model",
                    "served_by": "central-model",
                },
            ],
            "context": "Requirements: deterministic artifact. Risks: verify observable content.",
            "usage": {"members": 2, "tool_calls": 2, "elapsed_ms": 100},
        }

    coordinator = AgentLabCoordinator(
        store,
        model_call=model_call,
        team_runner=team_runner,
    )
    coordinator.start(run_id, model="central-model", max_repair_rounds=0)
    await asyncio.wait_for(coordinator._tasks[run_id], timeout=8)

    record = store.require(run_id)
    assert record["status"] == "completed"
    assert record["autonomous"]["model_calls"] == 4
    assert [agent["agent_id"] for agent in record["team"]] == [
        "coordinator", "worker-requirements", "worker-risks"
    ]
    assert record["delegation"]["run_id"] == "team-run-1"
    assert record["delegation"]["usage"]["members"] == 2
    assert all(
        "web_search" not in agent.get("allowed_tools", [])
        and "http_get" not in agent.get("allowed_tools", [])
        for agent in record["team"]
    )
    assert [agent["model"] for agent in record["team"][1:]] == [
        "central-model", "central-model"
    ]
    assert record["delegation"]["model_catalog"][0]["name"] == "central-model"
    assert "internal Team Planner" in prompts[0]
    assert "central Builder scheduler" in prompts[1]
    assert "TEAM_FINDINGS_JSON=" in prompts[2]
    assert "deterministic artifact" in prompts[2]


@pytest.mark.asyncio
async def test_parallel_builders_use_private_claims_and_deterministic_fan_in(tmp_path):
    store, run_id = prepared_store(tmp_path)
    builder_starts = []
    start_lock = threading.Lock()

    def model_call(prompt, model):
        if "central Builder scheduler" in prompt:
            return {
                "team_required": True,
                "reason": "Entrypoint and domain logic have separate file ownership",
                "members": [
                    {
                        "id": "logic",
                        "instruction": "Implement deterministic domain output.",
                        "model": "central-model",
                        "file_claims": ["src/logic.py"],
                    },
                    {
                        "id": "entry",
                        "instruction": "Implement the artifact-writing entrypoint.",
                        "model": "central-model",
                        "file_claims": ["src/main.py"],
                    },
                ],
            }
        if "one isolated Builder" in prompt:
            with start_lock:
                builder_starts.append(time.monotonic())
            time.sleep(0.12)
            if '"id": "entry"' in prompt:
                return {
                    "rationale": "Thin entrypoint",
                    "files": [{
                        "path": "src/main.py",
                        "purpose": "write observable artifact",
                        "content": 'from pathlib import Path\nfrom logic import result\nPath("artifacts/result.txt").write_text(result(), encoding="utf-8")\n',
                    }],
                }
            return {
                "rationale": "Pure domain function",
                "files": [{
                    "path": "src/logic.py",
                    "purpose": "domain result",
                    "content": 'def result():\n    return "parallel-ready"\n',
                }],
            }
        if "independent Verifier" in prompt:
            return verifier_proposal(
                verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "parallel-ready"\n'
            )
        raise AssertionError("Unexpected model prompt")

    async def team_runner(goal, **kwargs):
        return {
            "status": "completed",
            "team_required": True,
            "run_id": "specialists-before-builders",
            "plan": {
                "members": [
                    {"id": "requirements", "role": "planner", "instruction": "Define interfaces", "model": "central-model"},
                    {"id": "risks", "role": "analyst", "instruction": "Define evidence", "model": "central-model"},
                ]
            },
            "results": [
                {"member_id": "requirements", "status": "success", "output": "Separate entrypoint and domain logic."},
                {"member_id": "risks", "status": "success", "output": "Verify the final artifact content."},
            ],
            "context": "Use a pure domain module behind a thin entrypoint.",
            "usage": {"members": 2, "tool_calls": 0, "elapsed_ms": 1},
        }

    coordinator = AgentLabCoordinator(
        store,
        model_call=model_call,
        team_runner=team_runner,
    )
    coordinator.start(run_id, model="central-model", max_repair_rounds=0)
    await asyncio.wait_for(coordinator._tasks[run_id], timeout=8)

    record = store.require(run_id)
    assert record["status"] == "completed"
    assert record["builder_fanout"]["status"] == "merged"
    assert record["builder_fanout"]["merge_order"] == ["entry", "logic"]
    assert {claim["path"] for claim in record["file_claims"]} == {
        "src/main.py", "src/logic.py"
    }
    assert {claim["status"] for claim in record["file_claims"]} == {"merged"}
    assert len({claim["workspace_id"] for claim in record["file_claims"]}) == 2
    assert len(builder_starts) == 2
    assert max(builder_starts) - min(builder_starts) < 0.08
    assert record["autonomous"]["model_calls"] == 4
    proof = __import__("json").loads(
        (store.workspace_path(run_id) / "artifacts" / "proof-pack.json").read_text(encoding="utf-8")
    )
    assert proof["model_assignments"]["builder_fanout"]["merge_order"] == ["entry", "logic"]
    assert len(proof["workspace_evidence"]["file_claims"]) == 2


@pytest.mark.asyncio
async def test_parallel_builder_contract_failure_preserves_snapshots_and_resolves_claims(tmp_path):
    store, run_id = prepared_store(tmp_path)

    def model_call(prompt, model):
        if "central Builder scheduler" in prompt:
            return {
                "team_required": True,
                "reason": "Two source shards",
                "members": [
                    {"id": "entry", "instruction": "Own entrypoint", "model": "central-model", "file_claims": ["src/main.py"]},
                    {"id": "domain", "instruction": "Own domain", "model": "central-model", "file_claims": ["src/domain.py"]},
                ],
            }
        if "one isolated Builder" in prompt and '"id": "entry"' in prompt:
            return {
                "rationale": "valid shard",
                "files": [{"path": "src/main.py", "purpose": "entry", "content": "print('ready')\n"}],
            }
        if "one isolated Builder" in prompt:
            return {
                "rationale": "attempted claim expansion",
                "files": [{"path": "src/extra.py", "purpose": "unclaimed", "content": "pass\n"}],
            }
        raise AssertionError("Unexpected model prompt")

    async def team_runner(goal, **kwargs):
        return {
            "status": "completed",
            "team_required": True,
            "run_id": "specialists-before-failure",
            "plan": {"members": [
                {"id": "requirements", "role": "planner", "instruction": "Define interface", "model": "central-model"},
                {"id": "risks", "role": "analyst", "instruction": "Define risks", "model": "central-model"},
            ]},
            "results": [],
            "context": "Independent modules are possible.",
            "usage": {"members": 2, "tool_calls": 0, "elapsed_ms": 1},
        }

    coordinator = AgentLabCoordinator(store, model_call=model_call, team_runner=team_runner)
    coordinator.start(run_id, model="central-model", max_repair_rounds=0)
    await asyncio.wait_for(coordinator._tasks[run_id], timeout=8)

    record = store.require(run_id)
    assert record["status"] == "paused"
    assert record["phase"] == "repair_needed"
    assert record["builder_fanout"]["status"] == "failed"
    assert {claim["status"] for claim in record["file_claims"]} == {"failed"}
    assert len(record["workspace_branches"]) == 2
    assert {branch["status"] for branch in record["workspace_branches"]} == {"open"}
    assert not record["merge_receipts"]
    assert "unclaimed file" in record["error"]


@pytest.mark.asyncio
async def test_autonomous_coordinator_repairs_failed_build(tmp_path):
    store, run_id = prepared_store(tmp_path)
    responses = [
        proposal(
            main='raise RuntimeError("broken")\n',
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").exists()\n',
            rationale="initial attempt",
        ),
        proposal(
            main='from pathlib import Path\nPath("artifacts/result.txt").write_text("fixed", encoding="utf-8")\n',
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "fixed"\n',
            rationale="repair observed RuntimeError",
        ),
        verifier_proposal(
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "fixed"\n'
        ),
    ]

    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: responses.pop(0),
        team_runner=single_agent_team_runner,
    )
    coordinator.start(run_id, max_repair_rounds=1)
    task = coordinator._tasks[run_id]
    await asyncio.wait_for(task, timeout=8)

    record = store.require(run_id)
    assert record["status"] == "completed"
    assert [item["status"] for item in record["executions"]] == ["failed", "passed"]
    assert record["autonomous"]["repair_round"] == 1
    assert record["autonomous"]["model_calls"] == 3
    assert record["autonomous"]["last_rationale"] == "repair observed RuntimeError"


@pytest.mark.asyncio
async def test_autonomous_coordinator_rejects_unsafe_generated_source(tmp_path):
    store, run_id = prepared_store(tmp_path)
    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: proposal(
            main="import os\nprint(os.getcwd())\n",
            verify="assert True\n",
        ),
        team_runner=single_agent_team_runner,
    )
    coordinator.start(run_id, max_repair_rounds=0)
    task = coordinator._tasks[run_id]
    await asyncio.wait_for(task, timeout=5)

    record = store.require(run_id)
    assert record["status"] == "paused"
    assert record["phase"] == "repair_needed"
    assert "violates Agent Lab policy" in record["error"]
    assert not record["executions"]


@pytest.mark.asyncio
async def test_autonomous_coordinator_repairs_rejected_source_without_user_input(tmp_path):
    store, run_id = prepared_store(tmp_path)
    responses = [
        proposal(main="import os\nprint(os.getcwd())\n", verify="assert True\n"),
        proposal(
            main='from pathlib import Path\nPath("artifacts/result.txt").write_text("safe", encoding="utf-8")\n',
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "safe"\n',
            rationale="removed rejected system import",
        ),
        verifier_proposal(
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "safe"\n'
        ),
    ]
    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: responses.pop(0),
        team_runner=single_agent_team_runner,
    )

    coordinator.start(run_id, max_repair_rounds=1)
    await asyncio.wait_for(coordinator._tasks[run_id], timeout=8)

    record = store.require(run_id)
    assert record["status"] == "completed"
    assert record["autonomous"]["repair_round"] == 1
    assert not (record.get("interactive") or {}).get("awaiting_input")
    assert len(record["workspace_branches"]) == 2
    assert record["workspace_branches"][0]["status"] == "merged"


@pytest.mark.asyncio
async def test_autonomous_coordinator_repairs_invalid_response_schema_without_user_input(tmp_path):
    store, run_id = prepared_store(tmp_path)
    store.mutate(run_id, lambda item: item.update({"error": "stale retry error"}))
    invalid = proposal(main="print('unused')\n", verify="assert True\n")
    invalid.update({"explanation": "extra prose", "plan": ["extra structure"]})
    fixed = proposal(
        main='from pathlib import Path\nPath("artifacts/result.txt").write_text("fixed", encoding="utf-8")\n',
        verify='from pathlib import Path\nassert Path("artifacts/result.txt").exists()\n',
        rationale="returned the required contract only",
    )
    responses = [
        invalid,
        fixed,
        verifier_proposal(
            verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "fixed"\n'
        ),
    ]
    prompts = []

    def model_call(prompt, model):
        prompts.append(prompt)
        return responses.pop(0)

    coordinator = AgentLabCoordinator(
        store,
        model_call=model_call,
        team_runner=single_agent_team_runner,
    )

    coordinator.start(run_id, max_repair_rounds=1)
    await asyncio.wait_for(coordinator._tasks[run_id], timeout=8)

    record = store.require(run_id)
    assert record["status"] == "completed"
    assert record["error"] is None
    assert record["autonomous"]["repair_round"] == 1
    assert not (record.get("interactive") or {}).get("awaiting_input")
    assert not responses
    assert "Top-level keys must be exactly" in prompts[1]
    assert "Do not return agents, coordinator, plan, explanation" in prompts[1]
    assert "Use pathlib for file paths. Never import os" in prompts[1]


@pytest.mark.asyncio
async def test_invalid_independent_verifier_pauses_the_verify_node(tmp_path):
    store, run_id = prepared_store(tmp_path)

    def model_call(prompt, model):
        if "independent Verifier" in prompt:
            invalid = verifier_proposal(verify="assert True\n")
            invalid["files"][0]["path"] = "src/not-independent.py"
            return invalid
        return proposal(
            main="print('built')\n",
            verify="assert True\n",
        )

    coordinator = AgentLabCoordinator(
        store,
        model_call=model_call,
        team_runner=single_agent_team_runner,
    )
    coordinator.start(run_id, max_repair_rounds=0)
    await asyncio.wait_for(coordinator._tasks[run_id], timeout=8)

    record = store.require(run_id)
    assert record["status"] == "paused"
    assert record["task_ledger"]["node_states"]["verify"]["status"] == "needs_repair"
    assert record["task_ledger"]["blockers"][-1]["node_id"] == "verify"
    assert not record["verification"]


@pytest.mark.asyncio
async def test_autonomous_coordinator_stops_after_repair_budget(tmp_path):
    store, run_id = prepared_store(tmp_path)
    response = proposal(
        main='print("runs")\n',
        verify='raise AssertionError("not accepted")\n',
    )
    verifier_response = verifier_proposal(
        verify='raise AssertionError("not accepted")\n'
    )
    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: (
            verifier_response if "independent Verifier" in prompt else response
        ),
        team_runner=single_agent_team_runner,
    )
    coordinator.start(run_id, max_repair_rounds=1)
    task = coordinator._tasks[run_id]
    await asyncio.wait_for(task, timeout=8)

    record = store.require(run_id)
    assert record["status"] == "paused"
    assert record["phase"] == "repair_needed"
    assert record["autonomous"]["model_calls"] == 4
    assert record["autonomous"]["repair_round"] == 1
    assert len(record["verification"]) == 2


@pytest.mark.asyncio
async def test_autonomous_coordinator_cancels_after_inflight_model_boundary(tmp_path):
    store, run_id = prepared_store(tmp_path)

    def slow_model(prompt, model):
        time.sleep(0.15)
        return proposal(main="print('late')\n", verify="assert True\n")

    coordinator = AgentLabCoordinator(
        store, model_call=slow_model, team_runner=single_agent_team_runner
    )
    coordinator.start(run_id, max_repair_rounds=0)
    task = coordinator._tasks[run_id]
    coordinator.cancel(run_id)
    await asyncio.wait_for(task, timeout=5)

    record = store.require(run_id)
    assert record["status"] == "paused"
    assert record["phase"] == "paused"
    assert not record["executions"]


@pytest.mark.asyncio
async def test_autonomous_coordinator_emits_decisions_artifacts_and_verification(tmp_path, monkeypatch):
    store, run_id = prepared_store(tmp_path)
    store.mutate(run_id, lambda item: item.update({"trajectory_run_event_id": "trajectory-root"}))

    class FakeTrajectory:
        def __init__(self):
            self.events = []
            self.completed = []

        def record_execution_event(self, **payload):
            self.events.append(payload)
            return f"event-{len(self.events)}"

        def complete_execution_run(self, **payload):
            self.completed.append(payload)
            return "trajectory-result"

    trajectory = FakeTrajectory()
    monkeypatch.setattr("remy.core.trajectory_store.get_trajectory_store", lambda: trajectory)
    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: (
            verifier_proposal(
                verify='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "ok"\n'
            )
            if "independent Verifier" in prompt
            else proposal(
                main='from pathlib import Path\nPath("artifacts/result.txt").write_text("ok", encoding="utf-8")\n',
                verify='assert False, "builder test must be replaced"\n',
            )
        ),
        team_runner=single_agent_team_runner,
    )
    coordinator.start(run_id, max_repair_rounds=0)
    task = coordinator._tasks[run_id]
    await asyncio.wait_for(task, timeout=8)

    kinds = [item["event_kind"] for item in trajectory.events]
    assert "AGENT_LAB_TEAM" in kinds
    assert "AGENT_LAB_DECISION" in kinds
    assert "AGENT_LAB_WORKSPACE" in kinds
    assert "AGENT_LAB_MERGE" in kinds
    assert "AGENT_LAB_ARTIFACT" in kinds
    assert "AGENT_LAB_VERIFIER_PLAN" in kinds
    assert "AGENT_LAB_VERIFICATION" in kinds
    assert "AGENT_LAB_PROOF" in kinds
    assert trajectory.completed[0]["status"] == "completed"
    assert store.require(run_id)["trajectory_result_event_id"] == "trajectory-result"


@pytest.mark.asyncio
async def test_autonomous_coordinator_enforces_overall_time_budget(tmp_path):
    store, run_id = prepared_store(tmp_path)
    store.mutate(run_id, lambda item: item["policy"].update({"time_budget_seconds": 0}))
    calls = []
    coordinator = AgentLabCoordinator(
        store,
        model_call=lambda prompt, model: calls.append(prompt) or proposal(
            main="print('late')\n", verify="assert True\n"
        ),
        team_runner=single_agent_team_runner,
    )
    coordinator.start(run_id, max_repair_rounds=0)
    task = coordinator._tasks[run_id]
    await asyncio.wait_for(task, timeout=3)

    record = store.require(run_id)
    assert record["status"] == "paused"
    assert record["phase"] == "budget_exhausted"
    assert not calls
