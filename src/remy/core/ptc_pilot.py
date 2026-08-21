"""Safe programmatic tool-calling pilot for bounded read-only workflows."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


PTC_SCHEMA = "remy.ptc-pilot"
PTC_VERSION = 1
MAX_PROGRAM_CHARS = 64_000
MAX_STEPS = 12
MAX_VALUE_DEPTH = 12
MAX_VALUE_NODES = 2_000
HARD_MAX_CALLS = 12
HARD_MAX_TIME_MS = 30_000
HARD_MAX_OUTPUT_CHARS = 128_000
_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_REF_RE = re.compile(r"^([a-z][a-z0-9_-]{0,63})(?:\.([A-Za-z0-9_.-]{1,300}))?$")


@dataclass(frozen=True, slots=True)
class PTCField:
    kind: str
    required: bool = False
    max_length: int = 4_000
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()

    def public(self) -> dict[str, Any]:
        item = {"type": self.kind, "required": self.required}
        if self.kind == "string":
            item["max_length"] = self.max_length
        if self.minimum is not None:
            item["minimum"] = self.minimum
        if self.maximum is not None:
            item["maximum"] = self.maximum
        if self.choices:
            item["enum"] = list(self.choices)
        return item


@dataclass(frozen=True, slots=True)
class PTCTool:
    name: str
    description: str
    fields: dict[str, PTCField]
    require_any: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "read_only": True,
            "arguments": {name: field.public() for name, field in self.fields.items()},
            "require_any": list(self.require_any),
        }


@dataclass(frozen=True, slots=True)
class PTCLimits:
    max_calls: int = 8
    time_budget_ms: int = 10_000
    output_chars: int = 32_000

    @classmethod
    def parse(cls, value: dict[str, Any] | None) -> "PTCLimits":
        raw = dict(value or {})
        unknown = set(raw) - {"max_calls", "time_budget_ms", "output_chars"}
        if unknown:
            raise ValueError("Unknown PTC limit fields: " + ", ".join(sorted(unknown)))

        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            candidate = raw.get(name, default)
            if isinstance(candidate, bool) or not isinstance(candidate, int):
                raise ValueError(f"PTC limit {name} must be an integer")
            if not minimum <= candidate <= maximum:
                raise ValueError(f"PTC limit {name} must be between {minimum} and {maximum}")
            return candidate

        return cls(
            max_calls=integer("max_calls", 8, 1, HARD_MAX_CALLS),
            time_budget_ms=integer("time_budget_ms", 10_000, 50, HARD_MAX_TIME_MS),
            output_chars=integer(
                "output_chars",
                32_000,
                256,
                HARD_MAX_OUTPUT_CHARS,
            ),
        )


def _s(
    *,
    required: bool = False,
    max_length: int = 4_000,
    choices: tuple[str, ...] = (),
) -> PTCField:
    return PTCField(
        "string",
        required=required,
        max_length=max_length,
        choices=choices,
    )


def _i(
    *,
    required: bool = False,
    minimum: int = 0,
    maximum: int = 10_000,
) -> PTCField:
    return PTCField(
        "integer",
        required=required,
        minimum=minimum,
        maximum=maximum,
    )


def _n(
    *,
    required: bool = False,
    minimum: float = 0,
    maximum: float = 1,
) -> PTCField:
    return PTCField(
        "number",
        required=required,
        minimum=minimum,
        maximum=maximum,
    )


PTC_TOOLS: dict[str, PTCTool] = {
    "get_current_datetime": PTCTool(
        "get_current_datetime",
        "Read the current date and time.",
        {},
    ),
    "recall": PTCTool(
        "recall",
        "Recall relevant local memories.",
        {"query": _s(required=True), "token_budget": _i(minimum=128, maximum=8_000)},
    ),
    "search": PTCTool(
        "search",
        "Search local memory records.",
        {"query": _s(), "tags": _s(max_length=1_000)},
        require_any=("query", "tags"),
    ),
    "get_connections": PTCTool(
        "get_connections",
        "Read connections for one memory record.",
        {"record_id": _s(required=True, max_length=300)},
    ),
    "insights": PTCTool("insights", "Read memory health statistics.", {}),
    "recall_memory_as_of": PTCTool(
        "recall_memory_as_of",
        "Recall memory valid at a historical time.",
        {
            "query": _s(required=True),
            "timestamp": _s(required=True, max_length=200),
            "top_k": _i(minimum=1, maximum=20),
            "namespace": _s(max_length=300),
        },
    ),
    "explain_memory_recall": PTCTool(
        "explain_memory_recall",
        "Read selection and rejection reasons for memory recall.",
        {
            "query": _s(required=True),
            "top_k": _i(minimum=1, maximum=20),
            "min_strength": _n(minimum=0, maximum=1),
            "namespace": _s(max_length=300),
        },
    ),
    "build_memory_context": PTCTool(
        "build_memory_context",
        "Build a deterministic read-only memory capsule.",
        {
            "purpose": _s(required=True),
            "token_budget": _i(minimum=128, maximum=8_000),
            "valid_at": _s(max_length=200),
            "namespace": _s(max_length=300),
        },
    ),
    "search_transcript_history": PTCTool(
        "search_transcript_history",
        "Search exact local conversation transcripts.",
        {
            "query": _s(required=True),
            "session_id": _s(max_length=300),
            "limit": _i(minimum=1, maximum=100),
        },
    ),
    "review_history_memory_gaps": PTCTool(
        "review_history_memory_gaps",
        "Read likely gaps between transcript history and memory.",
        {"sample_limit": _i(minimum=1, maximum=100)},
    ),
    "list_local_workspaces": PTCTool(
        "list_local_workspaces",
        "List connected workspace grants without changing them.",
        {},
    ),
    "read_file": PTCTool(
        "read_file",
        "Read a file through workspace capability checks.",
        {"path": _s(required=True, max_length=4_000)},
    ),
    "list_directory": PTCTool(
        "list_directory",
        "List a directory through workspace capability checks.",
        {"path": _s(required=True, max_length=4_000)},
    ),
    "list_todos": PTCTool(
        "list_todos",
        "Read todo items.",
        {
            "status": _s(choices=("pending", "in_progress", "done", "all")),
            "category": _s(max_length=200),
        },
    ),
    "metric_summary": PTCTool(
        "metric_summary",
        "Read recent metric summaries.",
        {"period": _s(required=True, choices=("week", "month", "year"))},
    ),
    "list_child_sessions": PTCTool(
        "list_child_sessions",
        "Read delegated child session state.",
        {
            "parent_session_id": _s(max_length=300),
            "limit": _i(minimum=1, maximum=100),
        },
    ),
    "get_child_report": PTCTool(
        "get_child_report",
        "Read a delegated child report and attempt history.",
        {"child_id": _s(required=True, max_length=200)},
    ),
    "list_pipeline_candidates": PTCTool(
        "list_pipeline_candidates",
        "Read approval-gated pipeline candidates.",
        {
            "status": _s(
                choices=("draft", "dry_run_passed", "activated", "rejected", "all")
            )
        },
    ),
}


def list_ptc_tools() -> list[dict[str, Any]]:
    return [PTC_TOOLS[name].public() for name in sorted(PTC_TOOLS)]


def _walk_value(value: Any, *, path: str, depth: int, counter: list[int]) -> list[str]:
    counter[0] += 1
    if counter[0] > MAX_VALUE_NODES:
        return [f"{path} exceeds the {MAX_VALUE_NODES}-node value limit"]
    if depth > MAX_VALUE_DEPTH:
        return [f"{path} exceeds the maximum nesting depth {MAX_VALUE_DEPTH}"]
    errors: list[str] = []
    if isinstance(value, dict):
        if "$ref" in value:
            if set(value) != {"$ref"} or not isinstance(value["$ref"], str):
                errors.append(f"{path} reference must be exactly {{'$ref': 'step.path'}}")
            return errors
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 200:
                errors.append(f"{path} contains an invalid object key")
                continue
            errors.extend(
                _walk_value(item, path=f"{path}.{key}", depth=depth + 1, counter=counter)
            )
    elif isinstance(value, list):
        if len(value) > 200:
            errors.append(f"{path} exceeds the 200-item list limit")
        for index, item in enumerate(value[:201]):
            errors.extend(
                _walk_value(item, path=f"{path}[{index}]", depth=depth + 1, counter=counter)
            )
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        errors.append(f"{path} has unsupported value type {type(value).__name__}")
    return errors


def _is_reference(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"$ref"} and isinstance(value["$ref"], str)


def _validate_field(value: Any, field: PTCField, path: str, *, allow_ref: bool) -> list[str]:
    if allow_ref and _is_reference(value):
        return []
    if field.kind == "string":
        if not isinstance(value, str):
            return [f"{path} must be a string"]
        if len(value) > field.max_length:
            return [f"{path} exceeds {field.max_length} characters"]
    elif field.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{path} must be an integer"]
    elif field.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path} must be a number"]
    if field.minimum is not None and isinstance(value, (int, float)) and value < field.minimum:
        return [f"{path} must be at least {field.minimum}"]
    if field.maximum is not None and isinstance(value, (int, float)) and value > field.maximum:
        return [f"{path} must be at most {field.maximum}"]
    if field.choices and value not in field.choices:
        return [f"{path} must be one of: {', '.join(map(str, field.choices))}"]
    return []


def _validate_args(
    spec: PTCTool,
    args: dict[str, Any],
    *,
    path: str,
    allow_refs: bool,
) -> list[str]:
    errors: list[str] = []
    unknown = set(args) - set(spec.fields)
    if unknown:
        errors.append(f"{path} has unknown arguments: {', '.join(sorted(unknown))}")
    for name, field in spec.fields.items():
        if field.required and name not in args:
            errors.append(f"{path}.{name} is required")
        elif name in args:
            errors.extend(
                _validate_field(args[name], field, f"{path}.{name}", allow_ref=allow_refs)
            )
    if spec.require_any and not any(name in args and args[name] not in (None, "") for name in spec.require_any):
        errors.append(f"{path} requires at least one of: {', '.join(spec.require_any)}")
    return errors


def _collect_references(value: Any) -> list[str]:
    if _is_reference(value):
        return [str(value["$ref"])]
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _collect_references(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in _collect_references(item)]
    return []


def validate_ptc_program(
    program: dict[str, Any],
    limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        encoded = json.dumps(program, ensure_ascii=False, sort_keys=True, default=str)
    except Exception as exc:
        return {"valid": False, "errors": [f"Program is not JSON serializable: {exc}"]}
    if len(encoded) > MAX_PROGRAM_CHARS:
        errors.append(f"Program exceeds {MAX_PROGRAM_CHARS} characters")
    if not isinstance(program, dict):
        return {"valid": False, "errors": ["Program must be an object"]}
    unknown_program_fields = set(program) - {"version", "steps"}
    if unknown_program_fields:
        errors.append(
            "Program has unknown fields: " + ", ".join(sorted(unknown_program_fields))
        )
    version = program.get("version", PTC_VERSION)
    if version != PTC_VERSION:
        errors.append(f"Unsupported PTC program version: {version}")
    steps = program.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("Program steps must be a non-empty list")
        steps = []
    if len(steps) > MAX_STEPS:
        errors.append(f"Program exceeds the {MAX_STEPS}-step pilot limit")
    try:
        parsed_limits = PTCLimits.parse(limits)
    except ValueError as exc:
        parsed_limits = PTCLimits()
        errors.append(str(exc))

    prior_ids: set[str] = set()
    normalized_steps = []
    for index, step in enumerate(steps[: MAX_STEPS + 1]):
        path = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{path} must be an object")
            continue
        unknown_step_fields = set(step) - {"id", "tool", "args"}
        if unknown_step_fields:
            errors.append(f"{path} has unknown fields: {', '.join(sorted(unknown_step_fields))}")
        step_id = str(step.get("id") or "")
        tool_name = str(step.get("tool") or "")
        args = step.get("args", {})
        if not _ID_RE.fullmatch(step_id):
            errors.append(f"{path}.id is invalid")
        elif step_id in prior_ids:
            errors.append(f"{path}.id duplicates {step_id!r}")
        if tool_name not in PTC_TOOLS:
            errors.append(f"{path}.tool {tool_name!r} is not in the read-only PTC allowlist")
        if not isinstance(args, dict):
            errors.append(f"{path}.args must be an object")
            args = {}
        errors.extend(_walk_value(args, path=f"{path}.args", depth=0, counter=[0]))
        for reference in _collect_references(args):
            match = _REF_RE.fullmatch(reference)
            if not match:
                errors.append(f"{path} has invalid reference {reference!r}")
            elif match.group(1) not in prior_ids:
                errors.append(f"{path} reference {reference!r} is not a prior step")
        if tool_name in PTC_TOOLS:
            errors.extend(
                _validate_args(
                    PTC_TOOLS[tool_name],
                    args,
                    path=f"{path}.args",
                    allow_refs=True,
                )
            )
        if _ID_RE.fullmatch(step_id) and step_id not in prior_ids:
            prior_ids.add(step_id)
        normalized_steps.append({"id": step_id, "tool": tool_name, "args": args})

    normalized = {"version": PTC_VERSION, "steps": normalized_steps}
    program_hash = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "valid": not errors,
        "errors": errors,
        "program": normalized,
        "program_hash": program_hash,
        "limits": asdict(parsed_limits),
        "tool_count": len(PTC_TOOLS),
        "step_count": len(normalized_steps),
        "read_only": True,
    }


def _resolve_reference(reference: str, results: dict[str, Any]) -> Any:
    match = _REF_RE.fullmatch(reference)
    if not match or match.group(1) not in results:
        raise ValueError(f"Unresolvable PTC reference: {reference}")
    value = results[match.group(1)]
    path = match.group(2) or ""
    if not path:
        return value
    for token in path.split("."):
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError(f"PTC reference path not found: {reference}")
    return value


def _resolve_value(value: Any, results: dict[str, Any]) -> Any:
    if _is_reference(value):
        return _resolve_reference(str(value["$ref"]), results)
    if isinstance(value, dict):
        return {key: _resolve_value(item, results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, results) for item in value]
    return value


def _parse_tool_result(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _tool_failed(value: Any, raw: str) -> str:
    if isinstance(value, dict) and value.get("error"):
        return str(value.get("error"))
    if raw.startswith("Error:") or raw.startswith("Unknown tool:"):
        return raw[:2_000]
    return ""


def _invoke_tool(
    tool_name: str,
    args: dict[str, Any],
    session_id: str,
    channel: str,
) -> tuple[str, dict[str, Any] | None]:
    from remy.core.tool_dispatch import execute_tool
    from remy.core.tool_pipeline import get_last_tool_pipeline_snapshot

    get_last_tool_pipeline_snapshot(clear=True)
    result = execute_tool(tool_name, args, session_id=session_id, channel=channel)
    snapshot = get_last_tool_pipeline_snapshot(clear=True)
    return str(result), snapshot


def _record_ptc_trajectory(
    *,
    project_id: str,
    session_id: str,
    run_id: str,
    program_hash: str,
    receipt: dict[str, Any],
) -> None:
    try:
        from remy.core.trajectory_store import get_trajectory_store

        get_trajectory_store().record_ptc_run(
            project_id=project_id,
            session_id=session_id,
            run_id=run_id,
            program_hash=program_hash,
            receipt=receipt,
        )
    except Exception:
        return


def execute_ptc_program(
    program: dict[str, Any],
    *,
    owner_project_id: str,
    session_id: str,
    channel: str = "ptc",
    limits: dict[str, Any] | None = None,
    tool_invoker: Callable[[str, dict[str, Any], str, str], tuple[str, dict | None]] | None = None,
) -> dict[str, Any]:
    validation = validate_ptc_program(program, limits)
    if not validation["valid"]:
        return {
            "schema": PTC_SCHEMA,
            "version": PTC_VERSION,
            "status": "rejected",
            "validation": validation,
            "read_only": True,
        }

    from remy.core.project_store import get_project_store
    from remy.core.run_envelope import RunLimits, finish_run, start_run

    project = get_project_store().require_project(owner_project_id)
    budget = PTCLimits(**validation["limits"])
    normalized = validation["program"]
    run = start_run(
        kind="ptc_pilot",
        source_id=session_id or "ptc",
        goal=f"Execute read-only PTC program {validation['program_hash'][:12]}",
        owner_project_id=project.project_id,
        brain_id=project.brain_id,
        conversation_id=session_id,
        channel=channel,
        limits=RunLimits(
            max_turns=budget.max_calls,
            token_budget=max(1_000, budget.output_chars),
            max_parallel_workers=1,
            loop_repeat_limit=max(2, min(budget.max_calls, 8)),
        ),
        idempotency_class="read_only",
        metadata={
            "ptc_schema": PTC_SCHEMA,
            "program_hash": validation["program_hash"],
            "limits": validation["limits"],
            "read_only": True,
        },
    )
    started = time.monotonic()
    outputs: dict[str, Any] = {}
    step_receipts: list[dict[str, Any]] = []
    output_used = 0
    status = "completed"
    stop_reason = ""
    error = ""
    invoke = tool_invoker or _invoke_tool

    for index, step in enumerate(normalized["steps"]):
        elapsed_ms = int((time.monotonic() - started) * 1_000)
        if len(step_receipts) >= budget.max_calls:
            status, stop_reason = "completed_with_limits", "call_budget"
            break
        remaining_ms = budget.time_budget_ms - elapsed_ms
        if remaining_ms <= 0:
            status, stop_reason = "completed_with_limits", "time_budget"
            break
        try:
            resolved_args = _resolve_value(step["args"], outputs)
            arg_errors = _validate_args(
                PTC_TOOLS[step["tool"]],
                resolved_args,
                path=f"steps[{index}].args",
                allow_refs=False,
            )
            if arg_errors:
                raise ValueError("; ".join(arg_errors))
        except ValueError as exc:
            status, stop_reason, error = "failed", "reference_or_type_error", str(exc)
            break

        canonical_args = json.dumps(
            resolved_args,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        step_started = time.monotonic()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="remy-ptc-readonly",
        )
        future = executor.submit(
            invoke,
            step["tool"],
            resolved_args,
            session_id,
            channel,
        )
        try:
            raw_result, pipeline = future.result(timeout=max(0.05, remaining_ms / 1_000))
        except concurrent.futures.TimeoutError:
            future.cancel()
            status, stop_reason, error = (
                "completed_with_limits",
                "time_budget",
                f"Step {step['id']} exceeded the remaining time budget",
            )
            step_receipts.append(
                {
                    "id": step["id"],
                    "tool": step["tool"],
                    "status": "timed_out",
                    "duration_ms": int((time.monotonic() - step_started) * 1_000),
                    "args_sha256": hashlib.sha256(canonical_args.encode("utf-8")).hexdigest(),
                    "argument_keys": sorted(resolved_args),
                }
            )
            executor.shutdown(wait=False, cancel_futures=True)
            break
        except Exception as exc:
            status, stop_reason, error = "failed", "tool_exception", str(exc)
            executor.shutdown(wait=False, cancel_futures=True)
            break
        executor.shutdown(wait=False, cancel_futures=True)

        parsed = _parse_tool_result(raw_result)
        result_chars = len(raw_result)
        remaining_output = max(0, budget.output_chars - output_used)
        clipped_result = raw_result[:remaining_output]
        output_used += min(result_chars, remaining_output)
        tool_error = _tool_failed(parsed, raw_result)
        receipt = {
            "id": step["id"],
            "tool": step["tool"],
            "status": "failed" if tool_error else "completed",
            "duration_ms": int((time.monotonic() - step_started) * 1_000),
            "args_sha256": hashlib.sha256(canonical_args.encode("utf-8")).hexdigest(),
            "argument_keys": sorted(resolved_args),
            "result": clipped_result,
            "result_chars": result_chars,
            "result_sha256": hashlib.sha256(raw_result.encode("utf-8")).hexdigest(),
            "truncated": result_chars > remaining_output,
            "pipeline": pipeline,
        }
        if tool_error:
            receipt["error"] = tool_error
        step_receipts.append(receipt)
        if result_chars > remaining_output:
            status, stop_reason, error = (
                "completed_with_limits",
                "output_budget",
                f"Step {step['id']} exceeded the remaining output budget",
            )
            break
        if tool_error:
            status, stop_reason, error = "failed", "tool_error", tool_error
            break
        outputs[step["id"]] = parsed

    elapsed_ms = int((time.monotonic() - started) * 1_000)
    receipt = {
        "schema": PTC_SCHEMA,
        "version": PTC_VERSION,
        "run_id": run["run_id"],
        "attempt_id": run["attempt_id"],
        "program_hash": validation["program_hash"],
        "status": status,
        "stop_reason": stop_reason,
        "error": error,
        "read_only": True,
        "limits": validation["limits"],
        "usage": {
            "calls": len(step_receipts),
            "elapsed_ms": elapsed_ms,
            "output_chars": output_used,
        },
        "steps": step_receipts,
    }
    final_run = finish_run(
        run["attempt_id"],
        status=status,
        stop_reason=stop_reason,
        error=error,
        output_ref=f"ptc:{validation['program_hash']}",
        artifacts=[{"kind": "ptc_receipt", "receipt": receipt}],
    )
    receipt["run_status"] = final_run["status"]
    _record_ptc_trajectory(
        project_id=project.project_id,
        session_id=session_id or f"ptc:{run['run_id']}",
        run_id=run["run_id"],
        program_hash=validation["program_hash"],
        receipt=receipt,
    )
    return receipt


def handle_ptc_tool(
    name: str,
    args: dict[str, Any],
    *,
    session_id: str | None = None,
    channel: str | None = None,
) -> str:
    """Agent-facing adapter; actual steps still use the canonical ToolPipeline."""
    payload = dict(args or {})
    if name == "list_ptc_tools":
        result = {
            "schema": PTC_SCHEMA,
            "version": PTC_VERSION,
            "read_only": True,
            "tools": list_ptc_tools(),
        }
    elif name == "validate_ptc_program":
        result = validate_ptc_program(
            payload.get("program"),
            payload.get("limits"),
        )
    elif name == "run_ptc_program":
        from remy.core.microbrain import current_project_id

        result = execute_ptc_program(
            payload.get("program"),
            owner_project_id=current_project_id(),
            session_id=str(session_id or ""),
            channel=str(channel or "ptc"),
            limits=payload.get("limits"),
        )
    else:
        result = {"error": f"Unknown PTC tool: {name}"}
    return json.dumps(result, ensure_ascii=False, default=str)
