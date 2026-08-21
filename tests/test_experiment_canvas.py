import pytest

from remy.core.experiment_canvas import build_template, compile_canvas, role_catalog, validate_canvas


MODELS = {"model-a", "model-b", "model-c"}


def test_scientific_template_is_valid_and_compiles_roles():
    canvas = build_template("scientific-panel", sorted(MODELS))
    next(node for node in canvas["nodes"] if node["type"] == "problem")["data"]["problem"] = "Test a hypothesis"
    assert validate_canvas(canvas, connected_models=MODELS) == []
    plan = compile_canvas(canvas, connected_models=MODELS)
    assert [item["role"] for item in plan["participants"]] == ["investigator", "analyst", "skeptic"]
    assert [item["role_preset"] for item in plan["participants"]] == [
        "investigator", "evidence-analyst", "skeptic",
    ]
    assert all(item["role_instruction"] for item in plan["participants"])
    assert plan["peer_review"] is True
    assert plan["max_calls"] == 7


def test_role_catalog_explains_purpose_use_and_expected_output():
    roles = role_catalog()
    assert len(roles) >= 8
    assert {item["preset_id"] for item in roles} >= {
        "investigator", "evidence-analyst", "skeptic",
        "systems-designer", "safety-reviewer", "stakeholder-advocate",
    }
    for role in roles:
        assert role["label"]
        assert role["description"]
        assert role["best_for"]
        assert role["expected_output"]
        assert role["instruction"]


def test_canvas_rejects_missing_mandatory_node():
    canvas = build_template("blank", ["model-a"])
    canvas["nodes"] = [node for node in canvas["nodes"] if node["type"] != "success_gate"]
    canvas["edges"] = [edge for edge in canvas["edges"] if edge["source"] != "gate" and edge["target"] != "gate"]
    errors = validate_canvas(canvas, connected_models=MODELS)
    assert any("exactly one 'success_gate'" in error for error in errors)


def test_canvas_rejects_unconnected_role_and_unknown_model():
    canvas = build_template("blank", ["unknown"])
    canvas["edges"] = [edge for edge in canvas["edges"] if edge["source"] != "role-1"]
    errors = validate_canvas(canvas, connected_models=MODELS)
    assert any("unconnected model" in error for error in errors)
    assert any("must connect to the discussion group" in error for error in errors)


def test_canvas_rejects_cycles():
    canvas = build_template("blank", ["model-a"])
    canvas["edges"].append({"source": "synthesis", "target": "problem"})
    errors = validate_canvas(canvas, connected_models=MODELS)
    assert any("acyclic" in error for error in errors)
    assert any("final node" in error for error in errors)


def test_compiler_includes_context_prompt_search_and_embedded_data():
    canvas = build_template("blank", ["model-a"])
    next(node for node in canvas["nodes"] if node["type"] == "problem")["data"]["problem"] = "Analyze measurements"
    canvas["nodes"].extend([
        {"id": "ctx", "type": "context", "data": {"content": "Extra facts"}},
        {"id": "prompt", "type": "prompt", "data": {"prompt": "Compare alternatives"}},
        {"id": "search", "type": "web_search", "data": {"query": "latest {problem}", "num_results": 3}},
        {"id": "data", "type": "private_data", "data": {"name": "Measurements", "content": "x=4"}},
    ])
    for node_id in ("ctx", "prompt", "search", "data"):
        canvas["edges"].extend([{"source": "problem", "target": node_id}, {"source": node_id, "target": "discussion"}])
    assert validate_canvas(canvas, connected_models=MODELS) == []
    plan = compile_canvas(canvas, connected_models=MODELS)
    assert plan["global_context"] == "Extra facts"
    assert plan["global_prompt"] == "Compare alternatives"
    assert plan["web_searches"][0]["num_results"] == 3
    assert plan["embedded_data"][0]["content"] == "x=4"


def test_success_gate_rejects_invalid_confidence():
    canvas = build_template("blank", ["model-a"])
    next(node for node in canvas["nodes"] if node["type"] == "success_gate")["data"]["min_avg_confidence"] = 2
    assert any("between 0 and 1" in error for error in validate_canvas(canvas, connected_models=MODELS))


def test_canvas_rejects_information_block_after_discussion():
    canvas = build_template("blank", ["model-a"])
    next(node for node in canvas["nodes"] if node["type"] == "problem")["data"]["problem"] = "Test order"
    canvas["nodes"].append({"id": "late-context", "type": "context", "data": {"content": "Too late"}})
    canvas["edges"].extend([{"source": "board", "target": "late-context"}, {"source": "late-context", "target": "gate"}])
    errors = validate_canvas(canvas, connected_models=MODELS)
    assert any("before Discussion Group" in error for error in errors)


def test_scenario_template_compiles_world_interventions_and_replicas():
    canvas = build_template("scenario-simulation", sorted(MODELS))
    next(node for node in canvas["nodes"] if node["type"] == "problem")["data"]["problem"] = "Test policy response"
    next(node for node in canvas["nodes"] if node["type"] == "world_rules")["data"]["environment"] = "A small city with limited water."
    next(node for node in canvas["nodes"] if node["type"] == "intervention")["data"]["content"] = "A drought warning is issued."
    assert validate_canvas(canvas, connected_models=MODELS) == []
    plan = compile_canvas(canvas, connected_models=MODELS)
    assert plan["scenario"]["enabled"] is True
    assert plan["scenario"]["replicas"] == 3
    assert plan["scenario"]["interventions"][0]["round"] == 2
    assert plan["max_calls"] == 19


def test_scenario_rejects_too_many_replicas():
    canvas = build_template("scenario-simulation", ["model-a"])
    next(node for node in canvas["nodes"] if node["type"] == "problem")["data"]["problem"] = "Test"
    next(node for node in canvas["nodes"] if node["type"] == "replicas")["data"]["count"] = 10
    assert any("between 1 and 3" in error for error in validate_canvas(canvas, connected_models=MODELS))


def test_problem_attachments_compile_as_private_experiment_data():
    canvas = build_template("blank", ["model-a"])
    problem = next(node for node in canvas["nodes"] if node["type"] == "problem")
    problem["data"].update({
        "problem": "Analyze the attached observations",
        "attachments": [{"name": "observations.md", "content": "Measured value: 42"}],
    })
    plan = compile_canvas(canvas, connected_models=MODELS)
    assert plan["embedded_data"] == [{"name": "observations.md", "content": "Measured value: 42"}]
