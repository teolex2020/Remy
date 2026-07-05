"""
Tool Registry Management — singleton registry, sandbox tool handlers.

Manages the ToolRegistry singleton, cache invalidation, and sandbox
meta-tool handlers (create, test, list).
"""

import ast
import json
import logging

from remy.core.tool_declarations import BRAIN_TOOLS
from remy.core.tool_registry import ToolRegistry
from remy.sandbox.runner import (
    install_dependencies,
    run_tests,
    run_tests_world_fact,
    validate_tool_file,
)

logger = logging.getLogger("BrainTools")


def _get_bt():
    """Lazy accessor for brain_tools module (mutable registry state lives there)."""
    import remy.core.brain_tools as _bt

    return _bt


def _get_brain():
    """Lazy accessor — reads brain from brain_tools (supports test patching)."""
    return _get_bt().brain


# ============== REGISTRY ==============
# Mutable singleton (_registry, _registry_lock) lives in brain_tools.py
# so tests can patch remy.core.brain_tools._registry.


def get_registry() -> ToolRegistry:
    bt = _get_bt()
    with bt._registry_lock:
        if bt._registry is None:
            bt._registry = ToolRegistry(BRAIN_TOOLS)
        return bt._registry


def invalidate_registry() -> None:
    """Force registry singleton to rebuild on next get_registry() call."""
    bt = _get_bt()
    with bt._registry_lock:
        bt._registry = None


def reload_tools() -> None:
    """Invalidate all tool caches so new/removed sandbox tools take effect immediately.

    Call after sandbox tool approval, rejection, or auto-approve.
    Resets: registry singleton, LangChain tool cache, compiled agent graphs.
    """
    invalidate_registry()

    from remy.core.langgraph_tools import invalidate_tool_cache

    invalidate_tool_cache()

    from remy.core.agent import invalidate_graph_cache

    invalidate_graph_cache()

    logger.info("Tool caches invalidated — new tools will load on next request")


# ============== SANDBOX META-TOOL HANDLERS ==============


def _sandbox_create_tool(args: dict) -> str:
    """Write a tool file, validate it, register in manifest."""
    from pathlib import Path

    bt_settings = _get_bt().settings  # tests patch brain_tools.settings

    name = args["name"].strip()
    code = args["code"]
    manifest = get_registry().manifest

    # Write tool file
    tools_dir = Path(bt_settings.SANDBOX_TOOLS_DIR)
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool_file = tools_dir / f"{name}.py"
    tool_file.write_text(code, encoding="utf-8")

    # Validate via AST (no execution)
    valid, msg = validate_tool_file(tool_file)
    if not valid:
        tool_file.unlink()
        return json.dumps({"created": False, "error": msg})

    # Parse tool metadata from code via AST
    tree = ast.parse(code)
    tool_meta = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    tool_meta[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass

    description = tool_meta.get("TOOL_DESCRIPTION", f"Sandbox tool: {name}")
    parameters = tool_meta.get("TOOL_PARAMETERS", {})
    required = tool_meta.get("TOOL_REQUIRED", [])
    dependencies = tool_meta.get("DEPENDENCIES", [])

    # Install dependencies if any
    if dependencies:
        ok, dep_msg = install_dependencies(dependencies)
        if not ok:
            return json.dumps({"created": False, "error": f"Dependency install failed: {dep_msg}"})

    # Register in manifest
    manifest.add_tool(
        name=name,
        file=f"{name}.py",
        description=description,
        parameters=parameters,
        required=required,
        dependencies=dependencies,
    )

    # Classify trust level (AUTON-9)
    from remy.core.tool_trust import classify_tool_source, format_trust_report

    classification = classify_tool_source(code)
    tool_entry = manifest.get_tool(name)
    if tool_entry:
        tool_entry["trust_level"] = classification.trust_level
        tool_entry["trust_report"] = format_trust_report(classification)
        manifest.save()

    # Store in brain for learning
    _get_brain().store(
        content=f"Created sandbox tool '{name}': {description}",
        tags=["sandbox", "tool-creation"],
    )

    return json.dumps(
        {
            "created": True,
            "name": name,
            "status": "draft",
            "trust_level": classification.trust_level,
            "message": f"Tool '{name}' created (trust: {classification.trust_level}). Run sandbox_test_tool to test it.",
        }
    )


def _sandbox_test_tool(args: dict) -> str:
    """Run tests for a sandbox tool in isolated subprocess."""
    from pathlib import Path

    bt_settings = _get_bt().settings  # tests patch brain_tools.settings

    name = args["name"].strip()
    manifest = get_registry().manifest
    tool = manifest.get_tool(name)
    if not tool:
        return json.dumps({"tested": False, "error": f"Tool '{name}' not found in manifest."})

    tool_path = Path(bt_settings.SANDBOX_TOOLS_DIR) / tool["file"]
    if not tool_path.exists():
        return json.dumps({"tested": False, "error": f"Tool file missing: {tool['file']}"})

    success, passed, failed, output, world_fact = run_tests_world_fact(tool_path)
    manifest.set_test_result(name, passed, failed, output)

    # Classify trust level (AUTON-9)
    from remy.core.tool_trust import should_auto_approve as trust_auto_approve

    # Executable-judge guard: only a `supports` world fact (tests genuinely ran
    # and passed) qualifies for auto-approval. An `inconclusive` fact means the
    # run produced no observable pass/fail (e.g. no tests collected, or a hidden
    # stderr error under a clean exit) — never auto-approve a tool on that.
    if success and passed > 0 and world_fact == "supports":
        # Auto-approve in autonomous mode if configured
        if bt_settings.AUTONOMY_AUTO_APPROVE_SANDBOX:
            # Check trust level before auto-approving
            trust_ok, trust_reason = trust_auto_approve(tool_path)
            if trust_ok:
                auto_approved = manifest.auto_approve_tested()
                if name in auto_approved:
                    # Store trust level on approved entry
                    tool_entry = manifest.get_tool(name)
                    if tool_entry and "trust_level" not in tool_entry:
                        from remy.core.tool_trust import classify_tool_source

                        cls = classify_tool_source(tool_path.read_text(encoding="utf-8"))
                        tool_entry["trust_level"] = cls.trust_level
                        manifest.save()
                    reload_tools()
                    _get_brain().store(
                        content=f"Tool '{name}' auto-approved ({passed} tests passed). Trust: {trust_reason}",
                        tags=["sandbox", "auto-approved"],
                    )
                    return json.dumps(
                        {
                            "tested": True,
                            "passed": passed,
                            "failed": failed,
                            "status": "approved",
                            "trust": trust_reason,
                            "message": "Tests passed! Auto-approved and loaded.",
                        }
                    )
            else:
                # Dangerous tool — tests passed but requires human approval
                manifest.submit_for_approval(name)
                _get_brain().store(
                    content=f"Tool '{name}' passed {passed} tests but trust blocked auto-approve: {trust_reason}",
                    tags=["sandbox", "trust-blocked"],
                )
                return json.dumps(
                    {
                        "tested": True,
                        "passed": passed,
                        "failed": failed,
                        "status": "pending",
                        "trust": trust_reason,
                        "message": f"Tests passed but {trust_reason}. Awaiting human approval.",
                    }
                )

        manifest.submit_for_approval(name)
        _get_brain().store(
            content=f"Tool '{name}' passed {passed} tests. Awaiting human approval.",
            tags=["sandbox", "test-success"],
        )
        return json.dumps(
            {
                "tested": True,
                "passed": passed,
                "failed": failed,
                "status": "pending",
                "message": "Tests passed! Awaiting human approval. Ask the user to run: remy --sandbox-approve",
            }
        )
    else:
        _get_brain().store(
            content=f"Tool '{name}' failed testing: {passed} passed, {failed} failed.\n{output[:300]}",
            tags=["sandbox", "test-failure"],
        )
        return json.dumps(
            {
                "tested": False,
                "passed": passed,
                "failed": failed,
                "output": output[:800],
                "hint": "Read the error above, fix the code in sandbox_create_tool, then test again.",
            }
        )


def _sandbox_list_tools() -> str:
    """Return summary of all sandbox tools."""
    manifest = get_registry().manifest
    tools = manifest.summary()
    if not tools:
        return "No sandbox tools created yet."
    return json.dumps(tools, ensure_ascii=False)
