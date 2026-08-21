"""Side-effect-free replay of a recorded Trajectory turn against the current agent."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterator


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _digest(value: Any) -> str:
    payload = json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) and "content" in value:
        return _text(value.get("content"))
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts).strip()
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"))


def build_replay_spec(
    records: list[dict[str, Any]],
    *,
    selected_event: dict[str, Any] | None = None,
    history_turn_limit: int = 12,
) -> dict[str, Any]:
    """Extract an ephemeral prompt, prior dialogue, and exact tool fixtures."""
    ordered = sorted(records, key=lambda row: int(row.get("sequence") or 0))
    selected = selected_event if isinstance(selected_event, dict) else {}
    target_turn_id = str(selected.get("turn_id") or "")
    if not target_turn_id:
        event_id = str(selected.get("event_id") or "")
        target = next(
            (row for row in ordered if str(row.get("event_id") or "") == event_id),
            None,
        )
        target_turn_id = str((target or {}).get("turn_id") or "")
    if not target_turn_id:
        failed = next(
            (
                row for row in reversed(ordered)
                if row.get("status") == "failed" or row.get("error")
            ),
            None,
        )
        target_turn_id = str((failed or {}).get("turn_id") or "")
    if not target_turn_id:
        user = next((row for row in reversed(ordered) if row.get("kind") == "USER"), None)
        target_turn_id = str((user or {}).get("turn_id") or "")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        turn_id = str(row.get("turn_id") or "")
        if turn_id:
            grouped.setdefault(turn_id, []).append(row)
    turn_ids = list(grouped)
    if target_turn_id not in grouped:
        raise ValueError("The incident does not identify a replayable trajectory turn")
    target_index = turn_ids.index(target_turn_id)
    target_rows = grouped[target_turn_id]
    user = next((row for row in target_rows if row.get("kind") == "USER"), None)
    prompt = _text((user or {}).get("input") or (user or {}).get("output"))
    if not prompt:
        raise ValueError("The incident turn has no replayable user input")

    history = []
    for turn_id in turn_ids[max(0, target_index - history_turn_limit):target_index]:
        rows = grouped[turn_id]
        prior_user = next((row for row in rows if row.get("kind") == "USER"), None)
        assistants = [row for row in rows if row.get("kind") == "ASSISTANT"]
        user_text = _text((prior_user or {}).get("input") or (prior_user or {}).get("output"))
        assistant_text = _text((assistants[-1] if assistants else {}).get("output"))
        if user_text and assistant_text:
            history.extend((
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ))

    fixtures = []
    for row in target_rows:
        if str(row.get("kind") or "") not in {"TOOL", "SUBTOOL"}:
            continue
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        tool_name = str(details.get("name") or source.get("name") or "")
        if not tool_name:
            continue
        args = row.get("input")
        fixtures.append({
            "tool": tool_name,
            "args_fingerprint": _digest(args if isinstance(args, dict) else {}),
            "result": row.get("output"),
            "event_id": str(row.get("event_id") or ""),
            "status": str(row.get("status") or "completed"),
        })
    return {
        "prompt": prompt,
        "history": history,
        "fixtures": fixtures,
        "source_turn_id": target_turn_id,
        "source_event_id": str(selected.get("event_id") or ""),
    }


@dataclass(slots=True)
class ReplaySandbox:
    fixtures: list[dict[str, Any]]
    max_tool_calls: int = 32
    decisions: list[dict[str, Any]] = field(default_factory=list)
    _used: set[int] = field(default_factory=set)

    def resolve(self, tool_name: str, args: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Return a recorded fixture or a synthetic block; never invoke the tool."""
        args_fingerprint = _digest(args)
        fixture_index = next(
            (
                index for index, fixture in enumerate(self.fixtures)
                if index not in self._used
                and fixture.get("tool") == tool_name
                and fixture.get("args_fingerprint") == args_fingerprint
            ),
            None,
        )
        if len(self.decisions) >= max(1, int(self.max_tool_calls)):
            fixture_index = None
            reason = "sandbox tool-call budget exhausted"
        elif fixture_index is None:
            reason = "no exact recorded fixture for this tool and argument fingerprint"
        else:
            reason = "exact recorded fixture reused"
            self._used.add(fixture_index)

        if fixture_index is None:
            action = "blocked"
            result: Any = {
                "sandbox_replay": {
                    "status": "blocked",
                    "reason": reason,
                    "side_effect_executed": False,
                },
                "tool": tool_name,
            }
            fixture_event_id = ""
            source_status = ""
        else:
            fixture = self.fixtures[fixture_index]
            action = "fixture-hit"
            result = fixture.get("result")
            fixture_event_id = str(fixture.get("event_id") or "")
            source_status = str(fixture.get("status") or "completed")

        decision = {
            "tool": str(tool_name)[:120],
            "action": action,
            "args_fingerprint": args_fingerprint,
            "fixture_event_id": fixture_event_id,
            "source_status": source_status,
            "side_effect_executed": False,
        }
        self.decisions.append(decision)
        return result, {"sandbox_replay": True, **decision}

    def summary(self) -> dict[str, Any]:
        fixture_hits = sum(row["action"] == "fixture-hit" for row in self.decisions)
        blocked = sum(row["action"] == "blocked" for row in self.decisions)
        return {
            "sandboxed": True,
            "side_effects_executed": 0,
            "tool_calls": len(self.decisions),
            "fixture_hits": fixture_hits,
            "blocked_calls": blocked,
            "decisions": list(self.decisions),
        }


_ACTIVE_REPLAY: ContextVar[ReplaySandbox | None] = ContextVar(
    "trajectory_replay_sandbox", default=None
)


def current_replay_sandbox() -> ReplaySandbox | None:
    return _ACTIVE_REPLAY.get()


@contextmanager
def activate_replay_sandbox(sandbox: ReplaySandbox) -> Iterator[ReplaySandbox]:
    token = _ACTIVE_REPLAY.set(sandbox)
    try:
        yield sandbox
    finally:
        _ACTIVE_REPLAY.reset(token)


async def invoke_current_agent_replay(
    *,
    prompt: str,
    history: list[dict[str, str]],
    session_id: str,
    preferred_model: str = "",
) -> tuple[str, list[Any], list[dict[str, Any]]]:
    """Run the current agent graph without its durable chat checkpoint or post-turn writes."""
    from langchain_core.messages import AIMessage, HumanMessage

    from remy.core.agent import AgentState, build_agent_graph

    messages = [
        HumanMessage(content=row["content"])
        if row.get("role") == "user"
        else AIMessage(content=row["content"])
        for row in history
        if row.get("role") in {"user", "assistant"} and row.get("content")
    ]
    messages.append(HumanMessage(content=prompt))
    state = AgentState(
        messages=messages,
        session_id=session_id,
        channel="trajectory-replay",
        session_log=[],
        enabled_tools=set(),
        enabled_bundles=set(),
        capability_profile="standard",
        _cached_session_ctx=(
            "SANDBOX REPLAY: tools are intercepted before execution. Exact recorded "
            "fixtures may be returned; all other calls are blocked. Never claim that "
            "a simulated action changed external state."
        ),
        _cached_scratchpad="",
        model_routing_enabled=False,
        replay_preferred_model=str(preferred_model or "")[:240],
    )
    graph = build_agent_graph("trajectory-replay", checkpointer=None)
    if callable(getattr(type(graph), "ainvoke", None)):
        result = await graph.ainvoke(state, {"recursion_limit": 24})
    else:
        result = graph.invoke(state, {"recursion_limit": 24})
    result_messages = list(result.get("messages") or [])
    response_text = ""
    for message in reversed(result_messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            response_text = _text(message.content)
            break
    if not response_text:
        response_text = "Sandbox replay ended without a final assistant response."
    return response_text, result_messages, list(result.get("session_log") or [])


async def run_trajectory_sandbox_replay(
    *,
    project_id: str,
    case: dict[str, Any],
    source_records: list[dict[str, Any]],
    selected_event: dict[str, Any],
    trajectory_store: Any,
    conversation_store: Any,
    preferred_model: str = "",
    invoke: Callable[..., Awaitable[tuple[str, list[Any], list[dict[str, Any]]]]] | None = None,
) -> dict[str, Any]:
    """Create an inspectable replay conversation and execute one isolated agent turn."""
    spec = build_replay_spec(source_records, selected_event=selected_event)
    conversation = conversation_store.create(
        f"Sandbox replay · {case.get('name') or 'Trajectory regression'}",
        activate=False,
        metadata={
            "trajectory_sandbox_replay": {
                "case_id": str(case.get("case_id") or ""),
                "source_conversation_id": str(case.get("source_conversation_id") or ""),
                "source_event_id": str(spec.get("source_event_id") or ""),
                "side_effects_allowed": False,
            }
        },
    )
    session_id = str(conversation.conversation_id)
    trajectory_store.register_replay_session(
        session_id=session_id,
        project_id=project_id,
        case_id=str(case.get("case_id") or ""),
    )
    trajectory_store.begin_turn(
        session_id=session_id,
        project_id=project_id,
        content=spec["prompt"],
        source={
            "kind": "sandbox-replay",
            "channel": "trajectory-replay",
            "trust_tier": "recorded-user-input",
        },
        metadata={
            "sandbox_replay": True,
            "case_id": str(case.get("case_id") or ""),
            "source_turn_id": str(spec.get("source_turn_id") or ""),
            "source_event_id": str(spec.get("source_event_id") or ""),
        },
    )
    sandbox = ReplaySandbox(list(spec.get("fixtures") or []))
    execute = invoke or invoke_current_agent_replay
    execution_error = ""
    try:
        with activate_replay_sandbox(sandbox):
            await execute(
                prompt=spec["prompt"],
                history=list(spec.get("history") or []),
                session_id=session_id,
                preferred_model=str(preferred_model or "")[:240],
            )
    except Exception as exc:
        execution_error = type(exc).__name__
        trajectory_store.finish_turn(
            session_id=session_id, error=exc, evaluate_regressions=False
        )
    else:
        trajectory_store.finish_turn(session_id=session_id, evaluate_regressions=False)
    candidate_records = trajectory_store.list_events(
        project_id=project_id, session_id=session_id, limit=5000
    )
    replay = sandbox.summary()
    replay.update({
        "status": "failed" if execution_error else "completed",
        "execution_error_type": execution_error,
        "source_turn_id": str(spec.get("source_turn_id") or ""),
        "source_event_id": str(spec.get("source_event_id") or ""),
        "candidate_conversation_id": session_id,
        "preferred_model": str(preferred_model or "")[:240],
    })
    return {"records": candidate_records, "replay": replay, "conversation": conversation}
