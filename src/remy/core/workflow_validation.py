"""Validation helpers for visual workflow graphs."""

from __future__ import annotations


SUPPORTED_WORKFLOW_STEP_TYPES = {
    "llm_call",
    "web_search",
    "memory_search",
    "memory_save",
    "http_request",
    "page_scrape",
    "template",
    "router",
    "merge",
    "delay",
    "filter",
    "set_variable",
    "parse_json",
    "transform",
    "notification",
    "file_read",
    "file_write",
    "code",
    "error_handler",
    "condition",
    "loop",
}

PIPELINE_WORKFLOW_STEP_TYPES = set(SUPPORTED_WORKFLOW_STEP_TYPES)
AUTOMATION_WORKFLOW_STEP_TYPES = SUPPORTED_WORKFLOW_STEP_TYPES - {"condition", "loop"}

ROUTER_OPERATORS = {
    "contains",
    "not_contains",
    "equals",
    "not_equals",
    "starts_with",
    "ends_with",
    "regex",
    "always",
    "true",
    "never",
    "false",
    "fallback",
}

TRANSFORM_MODES = {
    "trim",
    "lower",
    "upper",
    "replace",
    "regex_replace",
    "extract_regex",
    "truncate",
    "join_lines",
}

CODE_MODES = {
    "safe_expression",
    "expression",
    "safe",
    "local_script",
    "local",
}

CODE_LANGUAGES = {
    "python",
    "py",
    "javascript",
    "js",
    "node",
}

PAGE_SCRAPE_MODES = {
    "text",
    "title",
    "links",
}


def _flow_nodes(drawflow_data: dict | None) -> dict:
    return (((drawflow_data or {}).get("drawflow") or {}).get("Home", {}) or {}).get("data", {}) or {}


def _node_by_name(nodes: dict, name: str) -> tuple[str, dict] | None:
    for node_id, node in nodes.items():
        if (node or {}).get("name") == name:
            return str(node_id), node
    return None


def _node(nodes: dict, node_id: str) -> dict:
    return nodes.get(node_id) or nodes.get(int(node_id)) or {}


def _output_connections(node: dict) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for output_name, output in sorted(((node or {}).get("outputs") or {}).items()):
        for conn in (output or {}).get("connections", []) or []:
            target = conn.get("node", "")
            if target != "":
                targets.append((str(target), str(output_name)))
    return targets


def _connected_inputs(node: dict) -> list[str]:
    return [
        name
        for name, input_meta in sorted(((node or {}).get("inputs") or {}).items())
        if (input_meta or {}).get("connections")
    ]


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_text(value) -> bool:
    return bool(str(value or "").strip())


def validate_workflow_step_configs(
    *,
    steps: list[dict],
    workflow_label: str,
    allowed_step_types: set[str] | None = None,
) -> list[str]:
    """Return blocking configuration errors for workflow action blocks."""
    allowed = allowed_step_types or SUPPORTED_WORKFLOW_STEP_TYPES
    errors: list[str] = []

    for index, step in enumerate(steps or [], start=1):
        step_type = step.get("type", "")
        config = step.get("config") or {}
        label = step.get("label") or step_type or f"Step {index}"
        prefix = f"{workflow_label} step {index} ({label})"

        if step_type not in allowed:
            errors.append(f"{prefix} has unsupported block type: {step_type or '<empty>'}.")
            continue

        if step_type == "template" and not _has_text(config.get("text")):
            errors.append(f"{prefix} Text / Template block cannot be empty.")
        elif step_type == "web_search":
            num_results = _as_int(config.get("num_results") or 5, 5)
            if num_results < 1 or num_results > 10:
                errors.append(f"{prefix} Web Search results must be between 1 and 10.")
        elif step_type == "memory_search":
            limit = _as_int(config.get("limit") or 5, 5)
            if limit < 1 or limit > 20:
                errors.append(f"{prefix} Memory Search results must be between 1 and 20.")
        elif step_type == "memory_save":
            if not _has_text(config.get("text") or config.get("input_source")):
                errors.append(f"{prefix} Save to Memory text is required.")
        elif step_type == "http_request":
            method = str(config.get("method") or "GET").strip().upper()
            if method not in {"GET", "POST"}:
                errors.append(f"{prefix} HTTP method must be GET or POST.")
        elif step_type == "page_scrape":
            mode = str(config.get("mode") or "text").strip().lower()
            if mode not in PAGE_SCRAPE_MODES:
                errors.append(f"{prefix} Page Scraper extract mode must be text, title, or links.")
            max_chars = _as_int(config.get("max_chars") or 12000, 12000)
            if max_chars < 500 or max_chars > 50000:
                errors.append(f"{prefix} Page Scraper max characters must be between 500 and 50000.")
        elif step_type == "delay":
            seconds = _as_float(config.get("seconds"), -1.0)
            if seconds < 0 or seconds > 300:
                errors.append(f"{prefix} Delay must be between 0 and 300 seconds.")
        elif step_type == "filter":
            operator = str(config.get("operator") or "").strip()
            if operator not in ROUTER_OPERATORS:
                errors.append(f"{prefix} Filter operator is not supported.")
            if operator not in {"always", "true", "never", "false", "fallback"} and not _has_text(config.get("value") or config.get("condition")):
                errors.append(f"{prefix} Filter value is required for this operator.")
        elif step_type == "set_variable":
            name = str(config.get("name") or "").strip()
            if not name:
                errors.append(f"{prefix} variable name is required.")
            elif not name.replace("_", "").isalnum() or name[0].isdigit():
                errors.append(f"{prefix} variable name must use letters, numbers, and underscores, and cannot start with a number.")
        elif step_type == "parse_json":
            if not _has_text(config.get("text")):
                errors.append(f"{prefix} JSON text is required.")
            if not _has_text(config.get("path")):
                errors.append(f"{prefix} JSON path is required.")
        elif step_type == "transform":
            mode = str(config.get("mode") or "trim").strip()
            if mode not in TRANSFORM_MODES:
                errors.append(f"{prefix} transform mode is not supported.")
            if mode in {"regex_replace", "extract_regex"} and not _has_text(config.get("pattern")):
                errors.append(f"{prefix} regex pattern is required.")
            if mode == "replace" and not _has_text(config.get("find")):
                errors.append(f"{prefix} replacement search text is required.")
            if mode == "truncate" and _as_int(config.get("limit"), -1) < 0:
                errors.append(f"{prefix} truncate limit must be 0 or greater.")
        elif step_type in {"file_read", "file_write"}:
            filename = str(config.get("filename") or "").strip()
            if not filename:
                errors.append(f"{prefix} file name is required.")
            if "/" in filename or "\\" in filename or filename in {".", ".."}:
                errors.append(f"{prefix} file name must be a simple file name inside workflow_files.")
            if step_type == "file_write" and not _has_text(config.get("text") or config.get("input_source")):
                errors.append(f"{prefix} File Write text is required.")
            if step_type == "file_write":
                mode = str(config.get("mode") or "overwrite").strip()
                if mode not in {"overwrite", "append"}:
                    errors.append(f"{prefix} File Write mode must be overwrite or append.")
        elif step_type == "code":
            mode = str(config.get("mode") or "safe_expression").strip()
            language = str(config.get("language") or "python").strip()
            if mode not in CODE_MODES:
                errors.append(f"{prefix} code mode is not supported.")
            if language not in CODE_LANGUAGES:
                errors.append(f"{prefix} code language is not supported.")
            if not _has_text(config.get("code")):
                errors.append(f"{prefix} Code / Script block cannot be empty.")
            timeout = _as_float(config.get("timeout_seconds") or 3, 3.0)
            if timeout <= 0 or timeout > 30:
                errors.append(f"{prefix} code timeout must be between 0 and 30 seconds.")
            max_output = _as_int(config.get("max_output_chars") or 12000, 12000)
            if max_output < 1 or max_output > 50000:
                errors.append(f"{prefix} code output limit must be between 1 and 50000 characters.")
        elif step_type == "router":
            routes = config.get("routes")
            if isinstance(routes, list):
                if len(routes) < 1:
                    errors.append(f"{prefix} Router needs at least one route.")
                for route_index, route in enumerate(routes, start=1):
                    if not isinstance(route, dict):
                        errors.append(f"{prefix} route {route_index} is invalid.")
                        continue
                    operator = str(route.get("operator") or "").strip()
                    if operator not in ROUTER_OPERATORS:
                        errors.append(f"{prefix} route {route_index} operator is not supported.")
                    if operator not in {"always", "true", "never", "false", "fallback"} and not _has_text(route.get("value") or route.get("condition")):
                        errors.append(f"{prefix} route {route_index} needs a value or condition.")
        elif step_type == "merge":
            input_count = _as_int(config.get("input_count"), 0)
            if input_count and input_count < 2:
                errors.append(f"{prefix} Merge needs at least two inputs.")
        elif step_type == "loop":
            max_iterations = _as_int(config.get("max_iterations") or 5, 5)
            if max_iterations < 1 or max_iterations > 20:
                errors.append(f"{prefix} Loop max iterations must be between 1 and 20.")

    return errors


def validate_visual_workflow_graph(
    *,
    steps: list[dict],
    drawflow_data: dict | None,
    entry_name: str,
    terminal_name: str,
    workflow_label: str,
) -> list[str]:
    """Return blocking graph errors for Drawflow-backed workflows.

    Workflows without Drawflow data are treated as legacy sequential workflows.
    """
    nodes = _flow_nodes(drawflow_data)
    if not nodes:
        return []

    errors: list[str] = []
    entry = _node_by_name(nodes, entry_name)
    terminal = _node_by_name(nodes, terminal_name)
    if not entry:
        errors.append(f"{workflow_label} is missing the {entry_name} node.")
    if not terminal:
        errors.append(f"{workflow_label} is missing the {terminal_name} node.")
    if errors:
        return errors

    entry_id, _entry_node = entry
    terminal_id, _terminal_node = terminal
    action_node_ids = {
        str(node_id)
        for node_id, node in nodes.items()
        if (node or {}).get("name") not in {entry_name, terminal_name}
    }
    step_node_ids = {
        str(step.get("_df_id") or str(step.get("id", "")).removeprefix("s"))
        for step in steps
        if step.get("id")
    }

    missing_steps = sorted(action_node_ids - step_node_ids)
    for node_id in missing_steps:
        node_name = (_node(nodes, node_id) or {}).get("name", "block")
        errors.append(f"{workflow_label} block {node_id} ({node_name}) is on the canvas but is not saved as a step.")

    reachable: set[str] = set()
    stack = [entry_id]
    while stack:
        current_id = stack.pop()
        if current_id in reachable:
            continue
        reachable.add(current_id)
        for target_id, _output_name in _output_connections(_node(nodes, current_id)):
            if target_id not in reachable:
                stack.append(target_id)

    unreachable = sorted(action_node_ids - reachable)
    for node_id in unreachable:
        node_name = (_node(nodes, node_id) or {}).get("name", "block")
        errors.append(f"{workflow_label} block {node_id} ({node_name}) is not reachable from {entry_name}.")

    if terminal_id not in reachable:
        errors.append(f"{workflow_label} has no connected path from {entry_name} to {terminal_name}.")

    for node_id in sorted(action_node_ids & reachable):
        node = _node(nodes, node_id)
        node_name = node.get("name", "block")
        if node_name == "merge" and len(_connected_inputs(node)) < 2:
            errors.append(f"{workflow_label} Merge block {node_id} needs at least two connected inputs.")
        if node_name == "router":
            for output_name, output in sorted(((node or {}).get("outputs") or {}).items()):
                if not ((output or {}).get("connections") or []):
                    errors.append(f"{workflow_label} Router block {node_id} has an unconnected {output_name}.")
        if not _output_connections(node):
            errors.append(f"{workflow_label} block {node_id} ({node_name}) has no outgoing connection.")

    return errors
