"""Schema validation and compilation for visual Experiment Canvas graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

NODE_TYPES = {
    "problem", "role_model", "private_data", "context", "prompt", "web_search",
    "discussion_group", "shared_board", "peer_review", "success_gate", "synthesis",
    "world_rules", "intervention", "replicas",
}
MANDATORY_TYPES = {"problem", "discussion_group", "shared_board", "success_gate", "synthesis"}

ROLE_CATALOG: list[dict[str, Any]] = [
    {
        "preset_id": "investigator", "category": "Research", "label": "Investigator",
        "role": "investigator",
        "description": "Explores the problem, proposes mechanisms and testable hypotheses, and identifies missing evidence.",
        "best_for": "Open questions where the team first needs plausible explanations or research directions.",
        "expected_output": "Testable hypotheses, causal reasoning, evidence gaps, and bounded next investigations.",
        "instruction": "Develop testable explanations and research directions. Separate supplied evidence from inference, identify missing evidence, and propose discriminating tests.",
        "recommended_web_access": True,
    },
    {
        "preset_id": "evidence-analyst", "category": "Research", "label": "Evidence Analyst",
        "role": "analyst",
        "description": "Checks what the supplied data and sources actually support, contradict, or leave uncertain.",
        "best_for": "Experiments containing documents, measurements, comparisons, or competing claims.",
        "expected_output": "Evidence table, strength-of-support judgments, contradictions, uncertainty, and confidence estimates.",
        "instruction": "Evaluate the supplied evidence methodically. Quantify support where possible, expose uncertainty and contradictions, and never treat an unsupported inference as an observation.",
        "recommended_web_access": False,
    },
    {
        "preset_id": "skeptic", "category": "Quality", "label": "Skeptic / Red Team",
        "role": "skeptic",
        "description": "Tries to falsify the leading ideas and finds hidden assumptions, edge cases, and alternative explanations.",
        "best_for": "High-impact conclusions, architecture choices, or any result vulnerable to confirmation bias.",
        "expected_output": "Concrete failure cases, counterexamples, challenged assumptions, and tests that could disprove a claim.",
        "instruction": "Act as a constructive red team. Search for counterexamples, hidden assumptions, confounders, and failure modes. Every criticism should include a practical test or mitigation.",
        "recommended_web_access": False,
    },
    {
        "preset_id": "systems-designer", "category": "Engineering", "label": "Systems Designer",
        "role": "designer",
        "description": "Turns requirements and evidence into a coherent architecture, workflow, or intervention design.",
        "best_for": "Software architecture, product design, operating models, and implementation planning.",
        "expected_output": "Proposed design, component boundaries, interfaces, trade-offs, and an incremental implementation path.",
        "instruction": "Design a coherent solution from the stated constraints. Make component boundaries, interfaces, invariants, trade-offs, and incremental validation steps explicit.",
        "recommended_web_access": False,
    },
    {
        "preset_id": "domain-expert", "category": "Research", "label": "Domain Expert",
        "role": "domain_expert",
        "description": "Applies field-specific concepts, constraints, terminology, and standards to the shared problem.",
        "best_for": "Problems where generic reasoning is insufficient without specialist domain framing.",
        "expected_output": "Domain constraints, applicable principles, terminology corrections, and risks requiring specialist verification.",
        "instruction": "Apply relevant domain concepts and constraints while marking the boundary of your knowledge. Identify claims that require external standards or human specialist verification.",
        "recommended_web_access": True,
    },
    {
        "preset_id": "safety-reviewer", "category": "Quality", "label": "Safety & Ethics Reviewer",
        "role": "safety_reviewer",
        "description": "Examines harms, misuse, compliance, reversibility, and conditions that should stop deployment.",
        "best_for": "Medical, financial, security, autonomous, or otherwise high-impact experiments.",
        "expected_output": "Risk register, affected parties, safeguards, stop conditions, and unresolved compliance questions.",
        "instruction": "Assess safety, misuse, privacy, fairness, compliance, and reversibility. Define concrete safeguards and stop conditions; do not convert uncertainty into reassurance.",
        "recommended_web_access": True,
    },
    {
        "preset_id": "stakeholder-advocate", "category": "People", "label": "Stakeholder Advocate",
        "role": "stakeholder",
        "description": "Represents the people affected by the proposed result and tests whether it is understandable and useful.",
        "best_for": "Products, policies, workflows, and scenario simulations with distinct affected groups.",
        "expected_output": "User needs, adoption barriers, accessibility concerns, incentives, and practical acceptance criteria.",
        "instruction": "Evaluate the proposal from the affected stakeholder's perspective. Surface needs, incentives, accessibility barriers, trust concerns, and observable acceptance criteria.",
        "recommended_web_access": False,
    },
    {
        "preset_id": "decision-maker", "category": "People", "label": "Decision Maker",
        "role": "actor",
        "description": "Chooses actions under the scenario's constraints and explains the assumptions behind each choice.",
        "best_for": "Scenario simulations, policy choices, and resource-allocation exercises.",
        "expected_output": "Chosen action, decision criteria, expected consequences, and signals that would trigger a change of course.",
        "instruction": "Choose actions only within the stated scenario constraints. Explain decision criteria, expected consequences, uncertainty, and the signals that would cause you to revise the decision.",
        "recommended_web_access": False,
    },
    {
        "preset_id": "independent-observer", "category": "Quality", "label": "Independent Observer",
        "role": "observer",
        "description": "Compares participants neutrally and records agreements, contradictions, and unexplained changes.",
        "best_for": "Multi-role discussions and scenario replicas that need a neutral comparison layer.",
        "expected_output": "Neutral observations, disagreement map, causal inconsistencies, and unanswered questions.",
        "instruction": "Observe without advocating for a participant. Compare claims and outcomes, record agreements and contradictions, and flag causal gaps or unexplained state changes.",
        "recommended_web_access": False,
    },
]


def role_catalog() -> list[dict[str, Any]]:
    """Return copies so API consumers cannot mutate the canonical presets."""
    return [dict(item) for item in ROLE_CATALOG]


def _role_preset(preset_id: str) -> dict[str, Any]:
    return next(
        (dict(item) for item in ROLE_CATALOG if item["preset_id"] == preset_id),
        dict(ROLE_CATALOG[0]),
    )


def canvas_templates() -> list[dict[str, Any]]:
    return [
        {"template_id": "scientific-panel", "name": "Scientific Panel", "description": "Investigator, analyst, and skeptic with evidence review."},
        {"template_id": "drug-discovery", "name": "Drug Discovery Hypothesis", "description": "Preclinical hypothesis generation with strict medical safety framing."},
        {"template_id": "engineering-review", "name": "Engineering Design Review", "description": "Designer, analyst, and failure-mode critic."},
        {"template_id": "scenario-simulation", "name": "Scenario Simulation", "description": "Reproducible multi-agent world with timed interventions and independent replicas."},
        {"template_id": "blank", "name": "Blank Experiment", "description": "Mandatory research spine with one editable role."},
    ]


def build_template(template_id: str, models: list[str]) -> dict[str, Any]:
    selected = models or [""]
    domain = "medical" if template_id == "drug-discovery" else "engineering" if template_id == "engineering-review" else "general"
    role_specs = {
        "scientific-panel": [
            ("investigator", None), ("evidence-analyst", None), ("skeptic", None),
        ],
        "drug-discovery": [
            ("investigator", "Mechanism Researcher"),
            ("evidence-analyst", "Evidence Analyst"),
            ("safety-reviewer", "Safety Reviewer"),
        ],
        "engineering-review": [
            ("systems-designer", None),
            ("evidence-analyst", "Trade-off Analyst"),
            ("skeptic", "Failure-mode Critic"),
        ],
        "scenario-simulation": [
            ("decision-maker", None),
            ("stakeholder-advocate", "Affected Group"),
            ("independent-observer", None),
        ],
        "blank": [("investigator", "Researcher")],
    }.get(template_id, [("investigator", "Researcher")])
    nodes = [
        {"id": "problem", "type": "problem", "x": 70, "y": 220, "data": {"title": "New experiment", "problem": "", "success_criteria": "", "domain": domain, "attachments": []}},
        {"id": "discussion", "type": "discussion_group", "x": 650, "y": 220, "data": {"rounds": 2, "turn_policy": "adaptive"}},
        {"id": "board", "type": "shared_board", "x": 950, "y": 220, "data": {"evidence_required": True}},
        {"id": "gate", "type": "success_gate", "x": 1240, "y": 220, "data": {"min_rounds": 1, "min_avg_confidence": 0.65}},
        {"id": "synthesis", "type": "synthesis", "x": 1510, "y": 220, "data": {
            "model": selected[0] if selected else "",
            "require_approval": False,
        }},
    ]
    edges = [
        {"source": "problem", "target": "discussion"}, {"source": "discussion", "target": "board"},
        {"source": "board", "target": "gate"}, {"source": "gate", "target": "synthesis"},
    ]
    for index, (preset_id, label_override) in enumerate(role_specs):
        preset = _role_preset(preset_id)
        node_id = f"role-{index + 1}"
        nodes.append({"id": node_id, "type": "role_model", "x": 360, "y": 80 + index * 170, "data": {
            "role_preset": preset["preset_id"],
            "label": label_override or preset["label"],
            "role": preset["role"],
            "role_instruction": preset["instruction"],
            "model": selected[index % len(selected)] if selected else "",
            "custom_prompt": "", "visible_datasets": [],
            "web_access": bool(preset["recommended_web_access"]),
        }})
        edges.append({"source": "problem", "target": node_id})
        edges.append({"source": node_id, "target": "discussion"})
    if template_id != "blank":
        nodes.append({"id": "review", "type": "peer_review", "x": 950, "y": 430, "data": {"mode": "cross_review"}})
        edges.extend([{"source": "board", "target": "review"}, {"source": "review", "target": "gate"}])
    if template_id == "scenario-simulation":
        world = {
            "id": "world", "type": "world_rules", "x": 70, "y": 520,
            "data": {
                "environment": "Describe the initial state, institutions, resources, constraints, and permitted actions.",
                "time_step": "1 simulated day", "seed": 42,
            },
        }
        intervention = {
            "id": "intervention", "type": "intervention", "x": 360, "y": 590,
            "data": {"round": 2, "content": "Describe a planned what-if event or leave blank."},
        }
        replicas = {
            "id": "replicas", "type": "replicas", "x": 650, "y": 590,
            "data": {"count": 3},
        }
        nodes.extend([world, intervention, replicas])
        edges.extend([
            {"source": "problem", "target": "world"},
            {"source": "world", "target": "intervention"},
            {"source": "intervention", "target": "replicas"},
            {"source": "replicas", "target": "discussion"},
        ])
    return {"version": 1, "template_id": template_id, "nodes": nodes, "edges": edges}


def _reachable(adjacency: dict[str, list[str]], source: str, target: str) -> bool:
    queue, seen = deque([source]), {source}
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for nxt in adjacency.get(current, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def validate_canvas(canvas: dict[str, Any], *, connected_models: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    nodes = list(canvas.get("nodes") or [])
    edges = list(canvas.get("edges") or [])
    by_id: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        node_id, node_type = str(node.get("id") or ""), str(node.get("type") or "")
        if not node_id:
            errors.append("Every node needs an id.")
            continue
        if node_id in by_id:
            errors.append(f"Duplicate node id: {node_id}")
        by_id[node_id] = node
        counts[node_type] += 1
        if node_type not in NODE_TYPES:
            errors.append(f"Unknown node type: {node_type}")
    for required in sorted(MANDATORY_TYPES):
        if counts[required] != 1:
            errors.append(f"Canvas requires exactly one '{required}' node.")
    if counts["role_model"] < 1:
        errors.append("Canvas requires at least one role_model node.")

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node_id: 0 for node_id in by_id}
    for edge in edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source not in by_id or target not in by_id:
            errors.append(f"Invalid edge: {source} -> {target}")
            continue
        if source == target:
            errors.append(f"Self-connections are not allowed: {source}")
            continue
        adjacency[source].append(target)
        indegree[target] += 1
    queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
    visited = 0
    while queue:
        node_id = queue.popleft(); visited += 1
        for target in adjacency.get(node_id, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(by_id):
        errors.append("Experiment Canvas must be acyclic; use Discussion Round settings for repetition.")

    ids_by_type = {kind: next((node_id for node_id, node in by_id.items() if node.get("type") == kind), "") for kind in MANDATORY_TYPES}
    spine = ["problem", "discussion_group", "shared_board", "success_gate", "synthesis"]
    for left, right in zip(spine, spine[1:]):
        if ids_by_type.get(left) and ids_by_type.get(right) and not _reachable(adjacency, ids_by_type[left], ids_by_type[right]):
            errors.append(f"'{left}' must lead to '{right}'.")
    discussion_id = ids_by_type.get("discussion_group", "")
    problem_id = ids_by_type.get("problem", "")
    synthesis_id = ids_by_type.get("synthesis", "")
    for node_id, node in by_id.items():
        data = node.get("data") or {}
        if node.get("type") == "role_model":
            model = str(data.get("model") or "")
            if not model:
                errors.append(f"Role '{data.get('label') or node_id}' has no model.")
            elif connected_models is not None and model not in connected_models:
                errors.append(f"Role '{data.get('label') or node_id}' uses an unconnected model: {model}")
            if problem_id and not _reachable(adjacency, problem_id, node_id):
                errors.append(f"Problem must lead to role '{data.get('label') or node_id}'.")
            if discussion_id and not _reachable(adjacency, node_id, discussion_id):
                errors.append(f"Role '{data.get('label') or node_id}' must connect to the discussion group.")
        if node.get("type") == "problem" and not str(data.get("problem") or "").strip():
            errors.append("Problem node must contain a problem statement.")
        if node.get("type") == "discussion_group":
            try:
                rounds = int(data.get("rounds") or 2)
                if not 1 <= rounds <= 5:
                    errors.append("Discussion rounds must be between 1 and 5.")
            except (TypeError, ValueError):
                errors.append("Discussion rounds must be a number.")
        if node.get("type") == "success_gate":
            try:
                confidence = float(data.get("min_avg_confidence") or 0.65)
                if not 0 <= confidence <= 1:
                    errors.append("Success Gate confidence must be between 0 and 1.")
            except (TypeError, ValueError):
                errors.append("Success Gate confidence must be a number.")
        if node.get("type") in {"private_data", "context", "prompt", "web_search"}:
            if discussion_id and not _reachable(adjacency, node_id, discussion_id):
                errors.append(f"'{node.get('type')}' ({node_id}) must be placed before Discussion Group.")
        if node.get("type") in {"world_rules", "intervention", "replicas"}:
            if discussion_id and not _reachable(adjacency, node_id, discussion_id):
                errors.append(f"'{node.get('type')}' ({node_id}) must be placed before Discussion Group.")
        if node.get("type") == "replicas":
            try:
                count = int(data.get("count") or 1)
                if not 1 <= count <= 3:
                    errors.append("Scenario replicas must be between 1 and 3.")
            except (TypeError, ValueError):
                errors.append("Scenario replicas must be a number.")
        if node.get("type") == "intervention":
            try:
                intervention_round = int(data.get("round") or 1)
                if not 1 <= intervention_round <= 5:
                    errors.append("Intervention round must be between 1 and 5.")
            except (TypeError, ValueError):
                errors.append("Intervention round must be a number.")
        if node.get("type") == "peer_review":
            board_id = ids_by_type.get("shared_board", "")
            gate_id = ids_by_type.get("success_gate", "")
            if board_id and not _reachable(adjacency, board_id, node_id):
                errors.append(f"Peer Review ({node_id}) must be placed after Shared Board.")
            if gate_id and not _reachable(adjacency, node_id, gate_id):
                errors.append(f"Peer Review ({node_id}) must lead to Success Gate.")
    for node_id, node in by_id.items():
        if problem_id and node_id != problem_id and not _reachable(adjacency, problem_id, node_id):
            errors.append(f"Node '{node.get('type')}' ({node_id}) is not connected from Problem.")
        if synthesis_id and node_id != synthesis_id and not _reachable(adjacency, node_id, synthesis_id):
            errors.append(f"Node '{node.get('type')}' ({node_id}) does not lead to Synthesis.")
    if synthesis_id and adjacency.get(synthesis_id):
        errors.append("Synthesis must be the final node.")
    return list(dict.fromkeys(errors))


def compile_canvas(canvas: dict[str, Any], *, connected_models: set[str] | None = None) -> dict[str, Any]:
    errors = validate_canvas(canvas, connected_models=connected_models)
    if errors:
        raise ValueError("\n".join(errors))
    nodes = list(canvas["nodes"])
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_type[node["type"]].append(node)
    problem = by_type["problem"][0].get("data") or {}
    discussion = by_type["discussion_group"][0].get("data") or {}
    gate = by_type["success_gate"][0].get("data") or {}
    synthesis = by_type["synthesis"][0].get("data") or {}
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {str(node["id"]): 0 for node in nodes}
    for edge in canvas.get("edges") or []:
        source, target = str(edge["source"]), str(edge["target"])
        adjacency[source].append(target); indegree[target] += 1
    queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
    execution_order = []
    while queue:
        node_id = queue.popleft(); execution_order.append(node_id)
        for target in adjacency.get(node_id, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    participants = []
    for node in by_type["role_model"]:
        data = node.get("data") or {}
        participants.append({
            "model": str(data.get("model") or ""), "role": str(data.get("role") or "investigator"),
            "role_preset": str(data.get("role_preset") or "custom"),
            "label": str(data.get("label") or data.get("role") or "Researcher"),
            "role_instruction": str(data.get("role_instruction") or "Advance the experiment from this role."),
            "custom_prompt": str(data.get("custom_prompt") or "")[:5000],
            "visible_datasets": list(data.get("visible_datasets") or []),
            "web_access": bool(data.get("web_access", False)),
        })
    rounds = max(1, min(int(discussion.get("rounds") or 2), 5))
    contexts = [str((node.get("data") or {}).get("content") or "")[:20_000] for node in by_type["context"]]
    prompts = [str((node.get("data") or {}).get("prompt") or "")[:5000] for node in by_type["prompt"]]
    searches = [{
        "query": str((node.get("data") or {}).get("query") or "{problem}")[:1000],
        "num_results": max(1, min(int((node.get("data") or {}).get("num_results") or 5), 10)),
    } for node in by_type["web_search"]]
    embedded_data = [{
        "name": str((node.get("data") or {}).get("name") or "Canvas data")[:200],
        "content": str((node.get("data") or {}).get("content") or "")[:80_000],
    } for node in by_type["private_data"] if str((node.get("data") or {}).get("content") or "").strip()]
    for attachment in list(problem.get("attachments") or [])[:8]:
        if not isinstance(attachment, dict):
            continue
        content = str(attachment.get("content") or "").strip()[:80_000]
        if content:
            embedded_data.append({
                "name": str(attachment.get("name") or "Problem source")[:200],
                "content": content,
            })
    is_scenario = canvas.get("template_id") == "scenario-simulation" or bool(by_type["world_rules"])
    world = (by_type["world_rules"][0].get("data") or {}) if by_type["world_rules"] else {}
    replicas = (by_type["replicas"][0].get("data") or {}) if by_type["replicas"] else {}
    interventions = []
    for node in by_type["intervention"]:
        data = node.get("data") or {}
        content = str(data.get("content") or "").strip()[:10_000]
        if content:
            interventions.append({
                "intervention_id": f"planned-{len(interventions) + 1}",
                "round": max(1, min(int(data.get("round") or 1), rounds)),
                "content": content,
                "source": "canvas",
            })
    replica_count = max(1, min(int(replicas.get("count") or 1), 3))
    return {
        "title": str(problem.get("title") or "Canvas experiment")[:200],
        "problem": str(problem.get("problem") or "")[:20_000],
        "success_criteria": str(problem.get("success_criteria") or "")[:10_000],
        "domain": str(problem.get("domain") or "general")[:40],
        "participants": participants, "rounds": rounds,
        "turn_policy": "adaptive",
        "max_calls": len(participants) * rounds * replica_count + 1,
        "global_context": "\n\n".join(item for item in contexts if item)[:40_000],
        "global_prompt": "\n\n".join(item for item in prompts if item)[:10_000],
        "web_searches": searches,
        "embedded_data": embedded_data,
        "peer_review": bool(by_type["peer_review"]),
        "success_gate": {
            "min_rounds": max(1, min(int(gate.get("min_rounds") or 1), rounds)),
            "min_avg_confidence": max(0.0, min(float(gate.get("min_avg_confidence") or 0.65), 1.0)),
        },
        "synthesis_model": str(synthesis.get("model") or participants[0]["model"]),
        "require_synthesis_approval": bool(synthesis.get("require_approval", False)),
        "execution_order": execution_order,
        "scenario": {
            "enabled": is_scenario,
            "environment": str(world.get("environment") or "")[:20_000],
            "time_step": str(world.get("time_step") or "1 simulated step")[:200],
            "seed": int(world.get("seed") or 42),
            "replicas": replica_count,
            "interventions": interventions,
        },
    }
