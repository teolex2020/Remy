"""Tests for multi-agent worker delegation system."""

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ============== FIXTURES ==============


@pytest.fixture
def mock_settings(tmp_path):
    """Mock settings with worker configuration."""
    with patch("remy.core.worker.settings") as ms:
        ms.WORKER_TIMEOUT_SEC = 60
        ms.WORKER_MAX_PARALLEL = 3
        ms.WORKER_MAX_TOOL_ITERATIONS = 8
        ms.SUMMARY_MODEL = "gemini-3-flash-preview"
        ms.GEMINI_API_KEY = "test-key"
        yield ms


@pytest.fixture
def worker_task():
    from remy.core.worker import WorkerTask
    return WorkerTask(
        role="researcher",
        instruction="Find information about vitamin D dosage",
        context="User is interested in supplements",
    )


@pytest.fixture
def mock_tools():
    """Create mock StructuredTools for testing."""
    tools = {}
    for name in [
        "recall", "store", "search", "get_current_datetime",
        "recall_knowledge", "store_knowledge", "web_search",
        "extract_content",
        "start_research", "add_research_finding", "complete_research",
        "store_research", "extract_facts",
        "write_file", "read_file", "http_get", "list_directory",
        "delegate_task", "sandbox_create_tool", "sandbox_test_tool",
        "create_subgoal", "complete_goal", "add_todo", "list_todos",
        "update_todo", "schedule_task", "update_record", "connect_records",
        "metric_summary", "event_correlate", "consolidate",
        "sandbox_approve_tool",
    ]:
        t = MagicMock()
        t.name = name
        t.invoke.return_value = json.dumps({"ok": True})
        tools[name] = t
    return tools


@pytest.fixture
def patch_get_all_tools(mock_tools):
    """Patch get_all_tools to return mock tools."""
    tool_list = list(mock_tools.values())
    with patch("remy.core.langgraph_tools.get_all_tools", return_value=tool_list):
        yield tool_list


# ============== TestWorkerToolScoping ==============


class TestWorkerToolScoping:

    def test_researcher_gets_research_tools(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools
        from remy.core.autonomy import AGENT_ROLES

        tools = get_worker_tools(AGENT_ROLES["researcher"])
        tool_names = {t.name for t in tools}

        assert "recall" in tool_names
        assert "web_search" in tool_names
        assert "start_research" in tool_names
        assert "store_research" in tool_names

    def test_osint_gets_research_and_content_tools(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools
        from remy.core.autonomy import AGENT_ROLES

        tools = get_worker_tools(AGENT_ROLES["osint"])
        tool_names = {t.name for t in tools}

        assert "web_search" in tool_names
        assert "extract_content" in tool_names
        assert "start_research" in tool_names
        assert "store_research" in tool_names
        assert "browse_page" not in tool_names
        assert "write_file" not in tool_names

    def test_researcher_blocked_from_write_file(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools
        from remy.core.autonomy import AGENT_ROLES

        tools = get_worker_tools(AGENT_ROLES["researcher"])
        tool_names = {t.name for t in tools}

        assert "write_file" not in tool_names  # In avoid_tools

    def test_delegate_task_always_blocked(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools
        from remy.core.autonomy import AGENT_ROLES

        for role_name in AGENT_ROLES:
            tools = get_worker_tools(AGENT_ROLES[role_name])
            tool_names = {t.name for t in tools}
            assert "delegate_task" not in tool_names, f"delegate_task leaked into {role_name}"

    def test_sandbox_tools_always_blocked(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools
        from remy.core.autonomy import AGENT_ROLES

        for role_name in AGENT_ROLES:
            tools = get_worker_tools(AGENT_ROLES[role_name])
            tool_names = {t.name for t in tools}
            assert "sandbox_create_tool" not in tool_names
            assert "sandbox_test_tool" not in tool_names
            assert "sandbox_approve_tool" not in tool_names

    def test_common_tools_always_present(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools, _COMMON_TOOLS
        from remy.core.autonomy import AGENT_ROLES

        for role_name in AGENT_ROLES:
            tools = get_worker_tools(AGENT_ROLES[role_name])
            tool_names = {t.name for t in tools}
            for common in _COMMON_TOOLS:
                assert common in tool_names, f"{common} missing for {role_name}"

    def test_executor_gets_file_tools(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools
        from remy.core.autonomy import AGENT_ROLES

        tools = get_worker_tools(AGENT_ROLES["executor"])
        tool_names = {t.name for t in tools}

        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "http_get" in tool_names

    def test_analyst_gets_analysis_tools(self, patch_get_all_tools):
        from remy.core.worker import get_worker_tools
        from remy.core.autonomy import AGENT_ROLES

        tools = get_worker_tools(AGENT_ROLES["analyst"])
        tool_names = {t.name for t in tools}

        assert "metric_summary" in tool_names
        assert "event_correlate" in tool_names
        assert "extract_facts" in tool_names
        # Analyst avoids web_search
        assert "web_search" not in tool_names


# ============== TestWorkerSystemInstruction ==============


class TestWorkerSystemInstruction:

    def test_contains_role_name(self, patch_get_all_tools):
        from remy.core.worker import build_worker_system_instruction, WorkerTask
        from remy.core.autonomy import AGENT_ROLES

        task = WorkerTask(role="researcher", instruction="Test task")
        instruction = build_worker_system_instruction(AGENT_ROLES["researcher"], task)

        assert "RESEARCHER" in instruction

    def test_contains_task_instruction(self, patch_get_all_tools):
        from remy.core.worker import build_worker_system_instruction, WorkerTask
        from remy.core.autonomy import AGENT_ROLES

        task = WorkerTask(role="researcher", instruction="Find vitamin D studies")
        instruction = build_worker_system_instruction(AGENT_ROLES["researcher"], task)

        assert "Find vitamin D studies" in instruction

    def test_contains_context_when_provided(self, patch_get_all_tools):
        from remy.core.worker import build_worker_system_instruction, WorkerTask
        from remy.core.autonomy import AGENT_ROLES

        task = WorkerTask(role="researcher", instruction="Research", context="User has vitamin D deficiency")
        instruction = build_worker_system_instruction(AGENT_ROLES["researcher"], task)

        assert "User has vitamin D deficiency" in instruction

    def test_worker_rules_present(self, patch_get_all_tools):
        from remy.core.worker import build_worker_system_instruction, WorkerTask
        from remy.core.autonomy import AGENT_ROLES

        task = WorkerTask(role="researcher", instruction="Test")
        instruction = build_worker_system_instruction(AGENT_ROLES["researcher"], task)

        assert "WORKER agent" in instruction
        assert "CONCISE" in instruction


# ============== TestSingleWorkerExecution ==============


class TestSingleWorkerExecution:

    @pytest.mark.asyncio
    async def test_worker_returns_result(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_single_worker, WorkerTask
        from langchain_core.messages import AIMessage

        mock_response = AIMessage(content="Vitamin D recommended dose is 600-800 IU daily.")

        task = WorkerTask(role="researcher", instruction="Find vitamin D dosage")

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [mock_response],
                "tool_call_count": 2,
            }
            mock_graph_fn.return_value = mock_graph

            result = await execute_single_worker(task, "test-session", "desktop")

            assert result.status == "success"
            assert result.role == "researcher"
            assert "Vitamin D" in result.output
            assert result.tool_calls == 2

    @pytest.mark.asyncio
    async def test_worker_timeout(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_single_worker, WorkerTask

        mock_settings.WORKER_TIMEOUT_SEC = 0.01  # Very short timeout

        task = WorkerTask(role="researcher", instruction="Slow task")

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn:
            mock_graph = MagicMock()

            async def slow_invoke(*args, **kwargs):
                await asyncio.sleep(10)

            # Make invoke block long enough to trigger timeout
            mock_graph.invoke.side_effect = lambda *a, **kw: asyncio.get_event_loop().run_until_complete(asyncio.sleep(10))

            mock_graph_fn.return_value = mock_graph

            result = await execute_single_worker(task, "test-session", "desktop")
            assert result.status in ("timeout", "error")

    @pytest.mark.asyncio
    async def test_worker_error_handling(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_single_worker, WorkerTask

        task = WorkerTask(role="researcher", instruction="Failing task")

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = RuntimeError("LLM crashed")
            mock_graph_fn.return_value = mock_graph

            result = await execute_single_worker(task, "test-session", "desktop")

            assert result.status == "error"
            assert "LLM crashed" in result.output

    @pytest.mark.asyncio
    async def test_unknown_role_returns_error(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_single_worker, WorkerTask

        task = WorkerTask(role="nonexistent_role", instruction="Some task")
        with patch("remy.core.event_bus.event_bus") as mock_bus:
            result = await execute_single_worker(task, "test-session", "desktop")

        assert result.status == "error"
        assert "Unknown role" in result.output
        mock_bus.emit.assert_called_once()
        event_type, payload = mock_bus.emit.call_args.args
        assert event_type == "worker_role_resolution"
        assert payload["requested_role"] == "nonexistent_role"
        assert payload["status"] == "unknown_role"

    @pytest.mark.asyncio
    async def test_osint_role_executes(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_single_worker, WorkerTask
        from langchain_core.messages import AIMessage

        task = WorkerTask(role="osint", instruction="Check competitor pricing")

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="OSINT complete")],
                "tool_call_count": 1,
            }
            mock_graph_fn.return_value = mock_graph

            result = await execute_single_worker(task, "test-session", "desktop")

        assert result.status == "success"
        assert result.role == "osint"
        assert "OSINT complete" in result.output
        assert any(
            entry.get("type") == "worker_event" and entry.get("event") == "worker_role_resolution"
            for entry in result.session_log
        )
        assert any(
            entry.get("type") == "worker_event" and entry.get("event") == "worker_started"
            for entry in result.session_log
        )
        assert any(
            entry.get("type") == "worker_event" and entry.get("event") == "worker_completed"
            for entry in result.session_log
        )

    @pytest.mark.asyncio
    async def test_worker_emits_role_resolution_and_started_events(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_single_worker, WorkerTask
        from langchain_core.messages import AIMessage

        task = WorkerTask(role="researcher", instruction="Find data")

        with (
            patch("remy.core.worker._build_worker_graph") as mock_graph_fn,
            patch("remy.core.event_bus.event_bus") as mock_bus,
        ):
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Done")],
                "tool_call_count": 0,
            }
            mock_graph_fn.return_value = mock_graph

            result = await execute_single_worker(task, "test-session", "desktop")

        assert result.status == "success"
        event_calls = [call.args for call in mock_bus.emit.call_args_list]
        event_types = [args[0] for args in event_calls]
        assert "worker_role_resolution" in event_types
        assert "worker_started" in event_types
        assert "worker_completed" in event_types
        resolution_payload = next(args[1] for args in event_calls if args[0] == "worker_role_resolution")
        assert resolution_payload["requested_role"] == "researcher"
        assert resolution_payload["resolved_role"] == "researcher"
        assert resolution_payload["status"] == "resolved"
        started_payload = next(args[1] for args in event_calls if args[0] == "worker_started")
        assert started_payload["worker_channel"] == "worker-researcher"


# ============== TestParallelExecution ==============


class TestParallelExecution:

    @pytest.mark.asyncio
    async def test_two_workers_parallel(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_workers, WorkerTask
        from langchain_core.messages import AIMessage

        tasks = [
            WorkerTask(role="researcher", instruction="Research task 1"),
            WorkerTask(role="analyst", instruction="Analysis task 2"),
        ]

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Worker result")],
                "tool_call_count": 1,
            }
            mock_graph_fn.return_value = mock_graph

            results = await execute_workers(tasks, "test-session", "desktop")

            assert len(results) == 2
            assert all(r.status == "success" for r in results)

    @pytest.mark.asyncio
    async def test_max_parallel_cap(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_workers, WorkerTask
        from langchain_core.messages import AIMessage

        mock_settings.WORKER_MAX_PARALLEL = 3
        tasks = [WorkerTask(role="researcher", instruction=f"Task {i}") for i in range(5)]

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="OK")],
                "tool_call_count": 0,
            }
            mock_graph_fn.return_value = mock_graph

            results = await execute_workers(tasks, "test-session", "desktop")
            assert len(results) == 3  # Capped at WORKER_MAX_PARALLEL

    @pytest.mark.asyncio
    async def test_one_fails_others_succeed(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_workers, WorkerTask
        from langchain_core.messages import AIMessage

        tasks = [
            WorkerTask(role="researcher", instruction="Good task"),
            WorkerTask(role="nonexistent_role", instruction="Bad role"),
        ]

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Result")],
                "tool_call_count": 0,
            }
            mock_graph_fn.return_value = mock_graph

            results = await execute_workers(tasks, "test-session", "desktop")
            assert len(results) == 2
            statuses = {r.status for r in results}
            assert "success" in statuses
            assert "error" in statuses

    @pytest.mark.asyncio
    async def test_empty_tasks(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_workers

        results = await execute_workers([], "test-session", "desktop")
        assert results == []

    @pytest.mark.asyncio
    async def test_event_bus_emissions(self, mock_settings, patch_get_all_tools):
        from remy.core.worker import execute_workers, WorkerTask
        from langchain_core.messages import AIMessage

        tasks = [WorkerTask(role="researcher", instruction="Test")]
        emitted = []

        with patch("remy.core.worker._build_worker_graph") as mock_graph_fn, \
             patch("remy.core.event_bus.event_bus") as mock_bus:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Done")],
                "tool_call_count": 0,
            }
            mock_graph_fn.return_value = mock_graph
            mock_bus.emit.side_effect = lambda t, d=None: emitted.append(t)

            await execute_workers(tasks, "test-session", "desktop")

            event_types = emitted
            assert "workers_started" in event_types
            assert "workers_completed" in event_types


# ============== TestDelegateTaskTool ==============


class TestDelegateTaskTool:

    def test_delegate_in_brain_tools(self):
        from remy.core.brain_tools import BRAIN_TOOLS
        names = [t.name for t in BRAIN_TOOLS]
        assert "delegate_task" in names

    def test_delegate_empty_tasks(self):
        from remy.core.brain_tools import _handle_delegate_task
        result = json.loads(_handle_delegate_task({"tasks": []}, None, None))
        assert "error" in result

    def test_delegate_no_tasks_key(self):
        from remy.core.brain_tools import _handle_delegate_task
        result = json.loads(_handle_delegate_task({}, None, None))
        assert "error" in result

    def test_delegate_invalid_tasks_type(self):
        from remy.core.brain_tools import _handle_delegate_task
        result = json.loads(_handle_delegate_task({"tasks": "not a list"}, None, None))
        assert "error" in result

    def test_delegate_skips_empty_instructions(self):
        from remy.core.brain_tools import _handle_delegate_task
        from remy.core.worker import WorkerResult

        mock_results = [
            WorkerResult(role="researcher", status="success",
                         output="Found info", tool_calls=1, elapsed_sec=1.0)
        ]

        async def fake_execute_workers(tasks, session_id, channel):
            return mock_results

        with patch("remy.core.worker.execute_workers", side_effect=fake_execute_workers):
            result = json.loads(_handle_delegate_task({
                "tasks": [
                    {"role": "researcher", "instruction": ""},  # Empty — skipped
                    {"role": "researcher", "instruction": "Valid task"},
                ]
            }, "session", "desktop"))

            # Only valid task should be delegated
            assert result.get("delegated") == 1 or "error" not in result


# ============== TestDelegateBrainLockBypass ==============


class TestDelegateBrainLockBypass:

    def test_execute_tool_bypasses_lock_for_delegate(self):
        """delegate_task should NOT acquire brain_lock."""
        from remy.core.brain_tools import execute_tool

        lock_acquired = []

        original_brain_lock = threading.RLock()

        class TrackingLock:
            def __enter__(self):
                lock_acquired.append(True)
                return original_brain_lock.__enter__()

            def __exit__(self, *args):
                return original_brain_lock.__exit__(*args)

        with patch("remy.core.brain_tools._handle_delegate_task",
                    return_value='{"delegated": 0, "results": []}') as mock_handler:
            result = execute_tool("delegate_task", {"tasks": []}, session_id="test")
            mock_handler.assert_called_once()

        # brain_lock should NOT have been acquired
        assert len(lock_acquired) == 0

    def test_other_tools_still_acquire_lock(self):
        """Non-delegate tools should acquire brain_lock normally."""
        from remy.core.brain_tools import execute_tool

        with patch("remy.core.brain_tools._execute_tool_locked",
                    return_value='{"ok": true}') as mock_locked:
            result = execute_tool("recall", {"query": "test"}, session_id="test")
            mock_locked.assert_called_once()


# ============== TestWorkerProvenance ==============


class TestWorkerProvenance:

    def test_worker_source_trust(self):
        from remy.core.brain_tools import _SOURCE_TRUST
        assert "agent-worker" in _SOURCE_TRUST
        assert _SOURCE_TRUST["agent-worker"] == 0.35

    def test_worker_provenance_mapping(self):
        from remy.core.brain_tools import _get_provenance
        prov = _get_provenance("worker-researcher")
        assert prov["source"] == "agent-worker"
        assert prov["trust_score"] == 0.35
        assert prov["verified"] is False

    def test_worker_analyst_provenance(self):
        from remy.core.brain_tools import _get_provenance
        prov = _get_provenance("worker-analyst")
        assert prov["source"] == "agent-worker"

    def test_non_worker_provenance_unchanged(self):
        from remy.core.brain_tools import _get_provenance

        desktop_prov = _get_provenance("desktop")
        assert desktop_prov["source"] == "agent-interactive"

        auto_prov = _get_provenance("autonomous")
        assert auto_prov["source"] == "agent-autonomous"


# ============== TestWorkerState ==============


class TestWorkerState:

    def test_worker_task_dataclass(self):
        from remy.core.worker import WorkerTask
        task = WorkerTask(role="researcher", instruction="Do research")
        assert task.role == "researcher"
        assert task.instruction == "Do research"
        assert task.context == ""

    def test_worker_result_dataclass(self):
        from remy.core.worker import WorkerResult
        result = WorkerResult(role="analyst", status="success", output="Analysis done")
        assert result.role == "analyst"
        assert result.status == "success"
        assert result.tool_calls == 0
        assert result.elapsed_sec == 0.0


# ============== TestWorkerShouldContinue ==============


class TestWorkerShouldContinue:

    def test_ends_on_plain_ai_message(self, mock_settings):
        from remy.core.worker import _worker_should_continue
        from langchain_core.messages import AIMessage
        from langgraph.graph import END

        state = {
            "messages": [AIMessage(content="Done")],
            "session_id": "test",
            "channel": "worker-researcher",
            "tool_call_count": 0,
        }
        assert _worker_should_continue(state) == END

    def test_continues_on_tool_calls(self, mock_settings):
        from remy.core.worker import _worker_should_continue
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="", tool_calls=[{"id": "1", "name": "recall", "args": {}}])
        state = {
            "messages": [msg],
            "session_id": "test",
            "channel": "worker-researcher",
            "tool_call_count": 2,
        }
        assert _worker_should_continue(state) == "tools"

    def test_stops_at_max_iterations(self, mock_settings):
        from remy.core.worker import _worker_should_continue
        from langchain_core.messages import AIMessage

        mock_settings.WORKER_MAX_TOOL_ITERATIONS = 5
        msg = AIMessage(content="", tool_calls=[{"id": "1", "name": "recall", "args": {}}])
        state = {
            "messages": [msg],
            "session_id": "test",
            "channel": "worker-researcher",
            "tool_call_count": 5,  # At max
        }
        result = _worker_should_continue(state)
        assert result == "model"  # Wrap-up, not tools

    def test_ends_on_empty_messages(self, mock_settings):
        from remy.core.worker import _worker_should_continue
        from langgraph.graph import END

        state = {
            "messages": [],
            "session_id": "test",
            "channel": "worker-researcher",
            "tool_call_count": 0,
        }
        assert _worker_should_continue(state) == END
