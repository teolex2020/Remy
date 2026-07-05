"""
LangGraph Tools — dynamically generates LangChain StructuredTool wrappers
from existing BRAIN_TOOLS declarations.

Single source of truth: brain_tools.BRAIN_TOOLS defines tool schemas,
brain_tools.execute_tool() handles execution. This module just bridges
them into LangChain's tool interface for LangGraph.
"""

import contextvars
import logging
import threading
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from remy.core.brain_tools import BRAIN_TOOLS, execute_tool, get_registry

logger = logging.getLogger(__name__)

# Per-request session_id — thread/async-safe via ContextVar
_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)

# Per-request channel — tracks provenance (autonomous/desktop/telegram/voice)
_channel_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "channel", default=None
)


def set_session_id(sid: str | None) -> None:
    """Set session_id for tool execution context (thread/async-safe)."""
    _session_id_var.set(sid)


def get_session_id() -> str | None:
    return _session_id_var.get()


def set_channel(ch: str | None) -> None:
    """Set channel for provenance tracking (thread/async-safe)."""
    _channel_var.set(ch)


def get_channel() -> str | None:
    return _channel_var.get()


# ============== TYPE MAPPING ==============

_GEMINI_SIMPLE_TYPES = {
    "STRING": str,
    "INTEGER": int,
    "NUMBER": float,
    "BOOLEAN": bool,
}

# Counter for unique nested model names
_model_counter = 0


def _resolve_python_type(prop_schema, parent_name: str = "Nested") -> type:
    """Recursively resolve a Gemini Schema to a Python/Pydantic type.

    Handles ARRAY items and OBJECT properties so that
    langchain-google-genai can reconstruct proper Gemini schemas
    with items fields for arrays.
    """
    global _model_counter
    schema_type = str(prop_schema.type) if prop_schema.type else "STRING"
    # prop_schema.type may be enum like Type.STRING — normalise
    schema_type = schema_type.replace("Type.", "").upper()

    simple = _GEMINI_SIMPLE_TYPES.get(schema_type)
    if simple:
        return simple

    if schema_type == "ARRAY":
        if prop_schema.items:
            inner = _resolve_python_type(prop_schema.items, parent_name + "Item")
            return list[inner]
        return list[str]  # fallback: list of strings

    if schema_type == "OBJECT":
        if prop_schema.properties:
            _model_counter += 1
            nested_name = f"{parent_name}{_model_counter}"
            nested_fields = {}
            req = list(prop_schema.required) if prop_schema.required else []
            for pname, pschema in prop_schema.properties.items():
                inner_type = _resolve_python_type(pschema, nested_name + pname.capitalize())
                desc = pschema.description or ""
                if pname in req:
                    nested_fields[pname] = (inner_type, Field(description=desc))
                else:
                    nested_fields[pname] = (inner_type | None, Field(default=None, description=desc))
            return create_model(nested_name, **nested_fields)
        return dict  # OBJECT without properties → dict

    return str  # unknown → str


def _gemini_schema_to_pydantic_field(
    prop_name: str, prop_schema, required_fields: list[str]
) -> tuple[type, Any]:
    """Convert a Gemini Schema property to a Pydantic field (type, Field)."""
    python_type = _resolve_python_type(prop_schema, prop_name.capitalize())
    description = prop_schema.description or ""
    is_required = prop_name in required_fields

    if is_required:
        return (python_type, Field(description=description))
    else:
        return (python_type | None, Field(default=None, description=description))


def _build_pydantic_model(tool_name: str, gemini_decl) -> type[BaseModel]:
    """Build a Pydantic model from a Gemini FunctionDeclaration's parameters."""
    fields = {}
    params = gemini_decl.parameters
    if params and params.properties:
        required = list(params.required) if params.required else []
        for prop_name, prop_schema in params.properties.items():
            fields[prop_name] = _gemini_schema_to_pydantic_field(
                prop_name, prop_schema, required
            )

    model_name = "".join(word.capitalize() for word in tool_name.split("_")) + "Args"

    # aura_cognitive_ops: 'params' field must accept both str and dict.
    # The model sometimes passes {} (dict) even though schema says STRING.
    if tool_name == "aura_cognitive_ops" and "params" in fields:
        fields["params"] = (str | dict | None, Field(default=None, description="JSON string or dict of kwargs"))

    if not fields:
        # No-arg tool — create empty model (Pydantic forbids leading underscores)
        return create_model(model_name)

    return create_model(model_name, **fields)


# ============== TOOL GENERATION ==============


def _deep_to_dict(value):
    """Recursively convert Pydantic BaseModel instances to plain dicts.

    LangChain passes nested Pydantic models (created by _resolve_python_type)
    as BaseModel instances, but execute_tool handlers expect plain dicts/lists.
    """
    if isinstance(value, BaseModel):
        return {k: _deep_to_dict(v) for k, v in value.model_dump().items()}
    if isinstance(value, list):
        return [_deep_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {k: _deep_to_dict(v) for k, v in value.items()}
    return value


def _make_tool_func(tool_name: str):
    """Create a closure that calls execute_tool with the right name and session_id."""

    def tool_func(**kwargs) -> str:
        # Filter out None values for optional args not provided
        # Convert Pydantic models to dicts (from nested ARRAY/OBJECT schemas)
        clean_args = {k: _deep_to_dict(v) for k, v in kwargs.items() if v is not None}

        # aura_cognitive_ops: params must be a JSON string, but model sometimes passes a dict.
        # Coerce silently so pydantic never sees the dict after schema validation.
        if tool_name == "aura_cognitive_ops" and "params" in clean_args:
            p = clean_args["params"]
            if isinstance(p, dict):
                import json as _json
                clean_args["params"] = _json.dumps(p)

        return execute_tool(tool_name, clean_args, session_id=get_session_id(), channel=get_channel())

    tool_func.__name__ = tool_name
    return tool_func


def build_langchain_tools() -> list[StructuredTool]:
    """Convert BRAIN_TOOLS (Gemini FunctionDeclarations) into LangChain StructuredTools.

    Each tool delegates to execute_tool() — no logic duplication.
    """
    tools = []

    for decl in BRAIN_TOOLS:
        try:
            args_model = _build_pydantic_model(decl.name, decl)
            tool = StructuredTool(
                name=decl.name,
                description=decl.description,
                func=_make_tool_func(decl.name),
                args_schema=args_model,
            )
            tools.append(tool)
        except Exception as e:
            logger.error("Failed to convert tool '%s': %s", decl.name, e)

    logger.info("Built %d LangChain tools from BRAIN_TOOLS", len(tools))
    return tools


def build_sandbox_tools() -> list[StructuredTool]:
    """Convert approved sandbox tools into LangChain StructuredTools."""
    registry = get_registry()
    sandbox_decls = registry._load_sandbox_declarations()
    tools = []

    for decl in sandbox_decls:
        try:
            args_model = _build_pydantic_model(decl.name, decl)

            def _make_sandbox_func(name: str):
                def func(**kwargs):
                    clean_args = {k: v for k, v in kwargs.items() if v is not None}
                    return registry.execute_sandbox_tool(name, clean_args)
                func.__name__ = name
                return func

            tool = StructuredTool(
                name=decl.name,
                description=decl.description,
                func=_make_sandbox_func(decl.name),
                args_schema=args_model,
            )
            tools.append(tool)
        except Exception as e:
            logger.error("Failed to convert sandbox tool '%s': %s", decl.name, e)

    if tools:
        logger.info("Built %d LangChain sandbox tools", len(tools))
    return tools


_cached_tools: list[StructuredTool] | None = None
_cached_tools_lock = threading.Lock()


def get_all_tools() -> list[StructuredTool]:
    """Get all tools (brain + sandbox) as LangChain StructuredTools.

    Cached after first build. Call invalidate_tool_cache() to force rebuild
    (e.g. after sandbox tool approval).
    """
    global _cached_tools
    with _cached_tools_lock:
        if _cached_tools is None:
            _cached_tools = build_langchain_tools() + build_sandbox_tools()
        return _cached_tools


def get_tools_by_names(names: set[str]) -> list[StructuredTool]:
    """Return tools filtered by name set from the cached full list.

    Used by agent.py for selective tool loading (core vs extended).
    """
    return [t for t in get_all_tools() if t.name in names]


def build_worker_tools(allowed_names: set[str]) -> list[StructuredTool]:
    """Build filtered StructuredTool list for worker agents.

    Reuses cached tools from get_all_tools(), just filters by name.
    """
    return [t for t in get_all_tools() if t.name in allowed_names]


def invalidate_tool_cache() -> None:
    """Force tools to be rebuilt on next get_all_tools() call."""
    global _cached_tools
    with _cached_tools_lock:
        _cached_tools = None
