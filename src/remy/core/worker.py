"""
Multi-Agent Workers — isolated LangGraph sub-agents with scoped tools.

The orchestrator (main agent) delegates tasks to workers via `delegate_task`.
Each worker gets a filtered tool set based on its role, runs in its own thread,
and returns a concise result. Workers share the same brain instance (shared memory).
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from remy.config.settings import settings
from remy.core.autonomy import AGENT_ROLES, AgentRole

logger = logging.getLogger(__name__)

# ============== DATA TYPES ==============


@dataclass
class WorkerTask:
    """A task to delegate to a worker agent."""
    role: str               # "researcher" | "planner" | "executor" | "analyst"
    instruction: str        # What to do
    context: str = ""       # Optional context from orchestrator
    approval_mode: str = "none"  # "none" | "publish" | "financial" | "all_clicks"
    delegation_depth: int = 0    # 0 = top-level worker; 1 = sub-worker (max)


@dataclass
class WorkerResult:
    """Result from a completed worker agent."""
    role: str
    status: str         # "success" | "error" | "timeout"
    output: str
    tool_calls: int = 0
    elapsed_sec: float = 0.0
    session_log: list = field(default_factory=list)  # tool call log (survives timeout)


# ============== TOOL SCOPING ==============

_WORKER_BLOCKED_TOOLS = frozenset({
    "delegate_task",            # Workers never re-delegate; chief/orchestrator owns delegation
    "sandbox_create_tool",     # Workers can't create sandbox tools
    "sandbox_test_tool",       # Workers can't test sandbox tools
    "sandbox_approve_tool",    # Workers can't approve tools
})

_COMMON_TOOLS = frozenset({
    "recall", "store", "search", "get_current_datetime",
    "recall_knowledge", "store_knowledge",
})


def get_worker_tools(role: AgentRole) -> list:
    """Build a hard-filtered tool list for a worker role.

    Only includes role.priority_tools + common tools, minus blocked/avoided.
    """
    from remy.core.langgraph_tools import get_all_tools

    allowed = set(role.priority_tools) | _COMMON_TOOLS
    blocked = set(role.avoid_tools) | _WORKER_BLOCKED_TOOLS

    return [t for t in get_all_tools() if t.name in allowed and t.name not in blocked]


# ============== WORKER SYSTEM INSTRUCTION ==============


def _prefetch_brain_context(query: str, limit: int = 3) -> str:
    """Fetch relevant records from brain for a worker task query."""
    try:
        from remy.core.agent_tools import brain_lock
        from remy.core.autonomy import brain
        with brain_lock:
            records = brain.search(query=query, limit=limit)
        if not records:
            return ""
        lines = []
        for r in records:
            lines.append(f"- {r.content[:200]}")
        return "\n".join(lines)
    except Exception:
        return ""


def build_worker_system_instruction(role: AgentRole, task: WorkerTask) -> str:
    """Build a compact system instruction for a worker agent (~500 tokens)."""
    tool_names = [t.name for t in get_worker_tools(role)]

    brain_context = _prefetch_brain_context(task.instruction)

    return (
        f"You are a WORKER agent with role: {role.name.upper()} — {role.description}.\n\n"
        f"{role.instruction_suffix}\n"
        f"AVAILABLE TOOLS: {', '.join(tool_names)}\n\n"
        + (f"=== RELEVANT MEMORY (from shared brain) ===\n{brain_context}\n\n" if brain_context else "")
        + f"=== YOUR TASK ===\n"
        f"{task.instruction}\n"
        + (f"\n=== CONTEXT ===\n{task.context}\n" if task.context else "")
        + "\n=== RULES ===\n"
        "- Complete the task and return a CONCISE result (max 500 words).\n"
        "- Do NOT delegate or plan beyond your scope.\n"
        "- Do NOT greet or ask follow-up questions.\n"
        "- If you cannot complete the task, explain why.\n"
        "- Store important findings in brain memory for future reference.\n"
    )


# ============== WORKER GRAPH ==============


class WorkerState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    channel: str
    tool_call_count: int
    _live_tool_log: list  # shared mutable list for cross-thread tool tracking
    _max_iterations: int  # per-worker iteration cap (from step_budget or settings)


def _worker_call_model(state: WorkerState) -> dict:
    """Invoke LLM for worker agent with scoped tools."""
    messages = list(state["messages"])
    channel = state.get("channel", "worker")

    # Role is encoded in channel: "worker-researcher" → "researcher"
    role_name = channel.replace("worker-", "") if channel.startswith("worker-") else "researcher"
    role = AGENT_ROLES.get(role_name, AGENT_ROLES["researcher"])
    tools = get_worker_tools(role)

    # Strip text from intermediate AIMessages with tool_calls to prevent
    # duplicate response generation (model repeats its own intermediate text)
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls and msg.content:
            cleaned.append(AIMessage(content="", tool_calls=msg.tool_calls, id=msg.id))
        else:
            cleaned.append(msg)
    messages = cleaned

    from remy.core.llm import call_llm
    from remy.core.model_trace import model_call_event

    llm_start = time.time()
    response = call_llm(messages, tools=tools, purpose=f"worker-{role_name}")
    llm_duration_ms = int((time.time() - llm_start) * 1000)
    raw_response = response

    if not isinstance(response, AIMessage):
        content = ""
        if hasattr(response, "content"):
            content = str(response.content)
        elif hasattr(response, "text"):
            content = str(response.text)
        response = AIMessage(content=content or "Worker encountered an issue.")

    live_log = state.get("_live_tool_log")
    if live_log is not None:
        live_log.append(
            model_call_event(
                raw_response,
                purpose=f"worker-{role_name}",
                channel=channel,
                duration_ms=llm_duration_ms,
            )
        )

    return {"messages": [response]}


def _worker_call_tools(state: WorkerState) -> dict:
    """Execute tool calls for a worker agent."""
    from remy.core.langgraph_tools import get_all_tools, set_session_id, set_channel

    messages = state["messages"]
    session_id = state.get("session_id")
    channel = state.get("channel", "worker")

    set_session_id(session_id)
    set_channel(channel)

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": [], "tool_call_count": state.get("tool_call_count", 0)}

    # Build tool map from scoped tools
    role_name = channel.replace("worker-", "") if channel.startswith("worker-") else "researcher"
    role = AGENT_ROLES.get(role_name, AGENT_ROLES["researcher"])
    tool_map = {t.name: t for t in get_worker_tools(role)}

    tool_messages = []
    count = state.get("tool_call_count", 0)

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        logger.info("Worker[%s] tool: %s(%s)", role_name, tool_name, tool_args)

        # Depth guard: block recursive delegate_task beyond 1 level
        if tool_name == "delegate_task":
            depth = state.get("_delegation_depth", 0)
            if depth >= 1:
                tool_messages.append(ToolMessage(
                    content="Recursive delegation blocked: workers cannot delegate further. Complete the subtask directly.",
                    tool_call_id=tc["id"],
                ))
                count += 1
                continue
            # Inject depth+1 into sub-worker tasks via args
            raw_tasks = tool_args.get("tasks", [])
            if isinstance(raw_tasks, list):
                for t in raw_tasks:
                    if isinstance(t, dict):
                        t["_delegation_depth"] = depth + 1

        tool = tool_map.get(tool_name)
        if tool:
            # Approval gate — check pack approval_mode and URL-based rules
            try:
                from remy.core.approval_queue import needs_approval, approval_queue, build_approval_description
                url = tool_args.get("url") or tool_args.get("action_url") or ""
                approval_mode = state.get("_approval_mode", "none")
                # all_clicks: gate every browser_act regardless of URL
                force_approval = approval_mode == "all_clicks" and tool_name == "browser_act"
                if force_approval or needs_approval(tool_name, tool_args, url=url or None, channel=channel):
                    description = build_approval_description(tool_name, tool_args, url or None)
                    result = approval_queue.request_approval_sync(
                        description,
                        lambda t=tool, a=tool_args: t.invoke(a),
                        tool_name=tool_name,
                        tool_args=tool_args,
                        url=url or None,
                    )
                else:
                    result = tool.invoke(tool_args)
            except ImportError:
                result = tool.invoke(tool_args)
            except Exception as e:
                logger.error("Worker[%s] tool %s error: %s", role_name, tool_name, e)
                result = f"Error: {e}"
        else:
            result = f"Tool not available for worker role '{role_name}': {tool_name}"

        result_str = str(result)
        tool_messages.append(
            ToolMessage(content=result_str, tool_call_id=tc["id"])
        )
        count += 1

        # Append to shared live log (survives timeout)
        live_log = state.get("_live_tool_log")
        if live_log is not None:
            live_log.append({
                "type": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "result": result_str[:500],
            })

    return {"messages": tool_messages, "tool_call_count": count}


def _worker_should_continue(state: WorkerState) -> str:
    """Route worker: tools or END, with iteration limit."""
    messages = state["messages"]
    max_iterations = state.get("_max_iterations") or settings.WORKER_MAX_TOOL_ITERATIONS

    if not messages:
        return END

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        count = state.get("tool_call_count", 0)
        if count >= max_iterations:
            logger.warning("Worker max tool iterations (%d) reached", max_iterations)
            # Provide wrap-up messages for pending tool calls
            for tc in last_message.tool_calls:
                state["messages"].append(
                    ToolMessage(
                        content="[SYSTEM: Tool limit reached. Summarize your findings now.]",
                        tool_call_id=tc["id"],
                    )
                )
            return "model"
        return "tools"

    return END


def _build_worker_graph():
    """Build a worker LangGraph (not cached — workers are ephemeral)."""
    graph = StateGraph(WorkerState)

    graph.add_node("model", _worker_call_model)
    graph.add_node("tools", _worker_call_tools)

    graph.set_entry_point("model")

    graph.add_conditional_edges(
        "model",
        _worker_should_continue,
        {"tools": "tools", END: END, "model": "model"},
    )
    graph.add_edge("tools", "model")

    return graph.compile()


# ============== EXECUTION ==============


async def execute_single_worker(
    task: WorkerTask,
    session_id: str,
    channel: str,
    step_budget: int = 0,
    timeout_override: float | None = None,
) -> WorkerResult:
    """Execute a single worker task in a separate thread.

    Args:
        step_budget: if >0, overrides settings.WORKER_MAX_TOOL_ITERATIONS
        timeout_override: if set, overrides settings.WORKER_TIMEOUT_SEC
    """
    from remy.core.event_bus import event_bus

    def _append_worker_event(log: list[dict], event_name: str, **fields) -> None:
        log.append(
            {
                "type": "worker_event",
                "event": event_name,
                "role": task.role,
                **fields,
            }
        )

    role = AGENT_ROLES.get(task.role)
    if not role:
        logger.warning(
            "Worker role resolution failed: requested_role=%s session_id=%s channel=%s",
            task.role,
            session_id,
            channel,
        )
        event_bus.emit("worker_role_resolution", {
            "requested_role": task.role,
            "resolved_role": None,
            "session_id": session_id,
            "channel": channel,
            "status": "unknown_role",
        })
        session_log = [
            {
                "type": "worker_event",
                "event": "worker_role_resolution",
                "role": task.role,
                "requested_role": task.role,
                "resolved_role": None,
                "channel": channel,
                "session_id": session_id,
                "status": "unknown_role",
            }
        ]
        return WorkerResult(
            role=task.role, status="error",
            output=f"Unknown role: {task.role}", tool_calls=0, elapsed_sec=0,
            session_log=session_log,
        )

    worker_channel = f"worker-{task.role}"
    effective_timeout = timeout_override or settings.WORKER_TIMEOUT_SEC
    effective_iterations = step_budget if step_budget > 0 else settings.WORKER_MAX_TOOL_ITERATIONS

    logger.info(
        "Worker role resolved: requested_role=%s resolved_role=%s worker_channel=%s timeout=%ss step_budget=%s session_id=%s",
        task.role,
        role.name,
        worker_channel,
        effective_timeout,
        effective_iterations,
        session_id,
    )
    event_bus.emit("worker_role_resolution", {
        "requested_role": task.role,
        "resolved_role": role.name,
        "worker_channel": worker_channel,
        "timeout_sec": effective_timeout,
        "step_budget": effective_iterations,
        "session_id": session_id,
        "channel": channel,
        "status": "resolved",
    })
    event_bus.emit("worker_started", {
        "role": task.role,
        "worker_channel": worker_channel,
        "timeout_sec": effective_timeout,
        "step_budget": effective_iterations,
        "session_id": session_id,
        "channel": channel,
    })

    # Shared mutable list — survives asyncio.wait_for timeout
    live_tool_log: list[dict] = []
    _append_worker_event(
        live_tool_log,
        "worker_role_resolution",
        requested_role=task.role,
        resolved_role=role.name,
        worker_channel=worker_channel,
        timeout_sec=effective_timeout,
        step_budget=effective_iterations,
        channel=channel,
        session_id=session_id,
        status="resolved",
    )
    _append_worker_event(
        live_tool_log,
        "worker_started",
        worker_channel=worker_channel,
        timeout_sec=effective_timeout,
        step_budget=effective_iterations,
        channel=channel,
        session_id=session_id,
    )

    start_time = time.time()
    try:
        graph = _build_worker_graph()
        sys_instruction = build_worker_system_instruction(role, task)

        state = {
            "messages": [
                SystemMessage(content=sys_instruction),
                HumanMessage(content=task.instruction),
            ],
            "session_id": session_id,
            "channel": worker_channel,
            "tool_call_count": 0,
            "_live_tool_log": live_tool_log,
            "_max_iterations": effective_iterations,
            "_approval_mode": task.approval_mode,
            "_delegation_depth": task.delegation_depth,
        }

        config = {"recursion_limit": effective_iterations * 2 + 5}

        result_state = await asyncio.wait_for(
            asyncio.to_thread(graph.invoke, state, config),
            timeout=effective_timeout,
        )

        elapsed = time.time() - start_time

        # Extract response text from last AIMessage
        output = ""
        for msg in reversed(result_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    output = msg.content
                elif isinstance(msg.content, list):
                    output = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in msg.content
                    )
                break

        tool_calls = result_state.get("tool_call_count", 0)
        # Use live_tool_log as session_log (always accurate)
        _append_worker_event(
            live_tool_log,
            "worker_completed",
            status="success",
            tool_calls=tool_calls,
            elapsed_sec=round(elapsed, 1),
        )
        session_log = list(live_tool_log) if live_tool_log else []

        event_bus.emit("worker_completed", {
            "role": task.role,
            "status": "success",
            "tool_calls": tool_calls,
            "elapsed_sec": round(elapsed, 1),
        })

        return WorkerResult(
            role=task.role, status="success",
            output=output or "Worker completed without output.",
            tool_calls=tool_calls, elapsed_sec=round(elapsed, 1),
            session_log=session_log,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        # Snapshot tool calls made before timeout
        recovered_tool_count = sum(
            1 for entry in live_tool_log if isinstance(entry, dict) and entry.get("type") == "tool_call"
        )
        _append_worker_event(
            live_tool_log,
            "worker_completed",
            status="timeout",
            tool_calls=recovered_tool_count,
            elapsed_sec=round(elapsed, 1),
        )
        recovered_log: list[dict] = list(live_tool_log)
        logger.warning(
            "Worker[%s] timed out after %.1fs (limit=%.0fs), recovered %d tool calls",
            task.role, elapsed, effective_timeout, recovered_tool_count,
        )
        event_bus.emit("worker_completed", {
            "role": task.role, "status": "timeout",
            "tool_calls": recovered_tool_count,
            "elapsed_sec": round(elapsed, 1),
        })
        return WorkerResult(
            role=task.role, status="timeout",
            output=f"Worker timed out after {effective_timeout}s.",
            tool_calls=recovered_tool_count,
            elapsed_sec=round(elapsed, 1),
            session_log=recovered_log,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Worker[%s] error: %s", task.role, e)
        _append_worker_event(
            live_tool_log,
            "worker_completed",
            status="error",
            elapsed_sec=round(elapsed, 1),
            error=str(e)[:300],
        )
        event_bus.emit("worker_completed", {
            "role": task.role, "status": "error",
            "elapsed_sec": round(elapsed, 1),
        })
        return WorkerResult(
            role=task.role, status="error",
            output=str(e), elapsed_sec=round(elapsed, 1),
            session_log=list(live_tool_log),
        )


async def execute_workers(
    tasks: list[WorkerTask],
    session_id: str,
    channel: str,
) -> list[WorkerResult]:
    """Execute multiple worker tasks in parallel (fan-out/fan-in)."""
    from remy.core.event_bus import event_bus

    max_parallel = settings.WORKER_MAX_PARALLEL
    tasks = tasks[:max_parallel]

    if not tasks:
        return []

    event_bus.emit("workers_started", {
        "count": len(tasks),
        "roles": [t.role for t in tasks],
    })

    results = await asyncio.gather(
        *[execute_single_worker(t, session_id, channel) for t in tasks],
        return_exceptions=True,
    )

    final: list[WorkerResult] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            final.append(WorkerResult(
                role=tasks[i].role, status="error",
                output=str(r), tool_calls=0, elapsed_sec=0,
            ))
        else:
            final.append(r)

    event_bus.emit("workers_completed", {
        "results": [{"role": r.role, "status": r.status} for r in final],
    })

    return final
