"""Tests for ExecutionLog — structured per-cycle run logging."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from remy.core.execution_log import (
    ExecutionEntry,
    ExecutionLog,
    ToolCallEntry,
    derive_cycle_status,
    record_cycle_execution,
)


@pytest.fixture
def log_dir(tmp_path):
    """Isolated log directory."""
    return tmp_path


def _make_log(log_dir: Path) -> ExecutionLog:
    """Create an ExecutionLog writing to a temp directory."""
    log_file = log_dir / "execution_log.jsonl"
    log = ExecutionLog.__new__(ExecutionLog)
    log._path = log_file
    log._entries = []
    import threading

    log._lock = threading.Lock()
    return log


def _entry(
    cycle: int = 1,
    goal_id: str = "g1",
    pack_id: str = "market_research",
    pack_label: str = "Market Research",
    worker: str = "research",
    status: str = "success",
    duration_ms: int = 500,
    tokens: int = 100,
    cost: float = 0.001,
    tool_calls: list | None = None,
    step_budget: int = 10,
    steps_used: int = 5,
    confidence: float = 0.8,
) -> ExecutionEntry:
    return ExecutionEntry(
        timestamp=time.time(),
        cycle_num=cycle,
        goal_id=goal_id,
        goal_description="test goal",
        pack_id=pack_id,
        pack_label=pack_label,
        worker=worker,
        status=status,
        duration_ms=duration_ms,
        tokens_used=tokens,
        cost_usd=cost,
        tool_calls=tool_calls or [],
        step_budget=step_budget,
        steps_used=steps_used,
        evaluation_confidence=confidence,
    )


# ============== ToolCallEntry ==============


class TestToolCallEntry:
    def test_defaults(self):
        tc = ToolCallEntry(tool="web_search")
        assert tc.tool == "web_search"
        assert tc.success is True
        assert tc.duration_ms == 0

    def test_custom(self):
        tc = ToolCallEntry(tool="browser_navigate", success=False, duration_ms=1200)
        assert tc.success is False
        assert tc.duration_ms == 1200


# ============== ExecutionEntry ==============


class TestExecutionEntry:
    def test_defaults(self):
        e = ExecutionEntry(timestamp=1000.0)
        assert e.cycle_num == 0
        assert e.goal_id == ""
        assert e.status == ""
        assert e.tool_calls == []
        assert e.verified is False
        assert e.repeated_failure is False
        assert e.memory_assisted is False

    def test_with_tool_calls(self):
        calls = [ToolCallEntry("search", True, 100), ToolCallEntry("store", False, 50)]
        e = _entry(tool_calls=calls)
        assert len(e.tool_calls) == 2
        assert e.tool_calls[0].tool == "search"
        assert e.tool_calls[1].success is False


# ============== ExecutionLog.record ==============


class TestRecord:
    def test_record_appends_to_memory(self, log_dir):
        log = _make_log(log_dir)
        assert log.record(_entry(cycle=1)) is True
        assert len(log._entries) == 1

    def test_record_writes_to_file(self, log_dir):
        log = _make_log(log_dir)
        assert log.record(_entry(cycle=1)) is True
        assert log._path.exists()
        lines = log._path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["cycle_num"] == 1
        assert data["pack_id"] == "market_research"

    def test_record_flattens_tool_calls(self, log_dir):
        log = _make_log(log_dir)
        calls = [ToolCallEntry("search", True), ToolCallEntry("store", False)]
        log.record(_entry(tool_calls=calls))
        data = json.loads(log._path.read_text(encoding="utf-8").strip())
        assert data["tool_calls"] == [
            {"tool": "search", "ok": True},
            {"tool": "store", "ok": False},
        ]
        assert data["tool_count"] == 2

    def test_record_multiple(self, log_dir):
        log = _make_log(log_dir)
        for i in range(5):
            log.record(_entry(cycle=i))
        assert len(log._entries) == 5
        lines = log._path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_in_memory_cap(self, log_dir):
        """Memory buffer should not exceed MAX_IN_MEMORY."""
        log = _make_log(log_dir)
        for i in range(250):
            log.record(_entry(cycle=i))
        assert len(log._entries) <= 200
        # File should have all 250
        lines = log._path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 250


# ============== ExecutionLog.get_recent ==============


class TestGetRecent:
    def test_empty(self, log_dir):
        log = _make_log(log_dir)
        assert log.get_recent() == []

    def test_limit(self, log_dir):
        log = _make_log(log_dir)
        for i in range(10):
            log.record(_entry(cycle=i))
        recent = log.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[-1]["cycle_num"] == 9

    def test_default_limit(self, log_dir):
        log = _make_log(log_dir)
        for i in range(60):
            log.record(_entry(cycle=i))
        recent = log.get_recent()
        assert len(recent) == 50


# ============== ExecutionLog.get_by_pack ==============


class TestGetByPack:
    def test_filters_by_pack(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="market_research"))
        log.record(_entry(pack_id="publisher"))
        log.record(_entry(pack_id="market_research"))
        result = log.get_by_pack("market_research")
        assert len(result) == 2
        assert all(e["pack_id"] == "market_research" for e in result)

    def test_empty_when_no_match(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="publisher"))
        assert log.get_by_pack("monitoring") == []


# ============== ExecutionLog.get_by_goal ==============


class TestGetByGoal:
    def test_filters_by_goal(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(goal_id="g1"))
        log.record(_entry(goal_id="g2"))
        log.record(_entry(goal_id="g1"))
        result = log.get_by_goal("g1")
        assert len(result) == 2

    def test_limit(self, log_dir):
        log = _make_log(log_dir)
        for _ in range(10):
            log.record(_entry(goal_id="g1"))
        result = log.get_by_goal("g1", limit=3)
        assert len(result) == 3


# ============== ExecutionLog.get_pack_summary ==============


class TestGetPackSummary:
    def test_empty(self, log_dir):
        log = _make_log(log_dir)
        assert log.get_pack_summary() == {}

    def test_counts_statuses(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="pub", status="success"))
        log.record(_entry(pack_id="pub", status="success"))
        log.record(_entry(pack_id="pub", status="failure"))
        log.record(_entry(pack_id="pub", status="timeout"))
        log.record(_entry(pack_id="pub", status="blocked"))
        summary = log.get_pack_summary()
        pub = summary["pub"]
        assert pub["total_runs"] == 5
        assert pub["successes"] == 2
        assert pub["failures"] == 1
        assert pub["timeouts"] == 1
        assert pub["blocked"] == 1
        assert pub["completion_rate"] == 0.4

    def test_aggregates_cost_and_tokens(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="mr", tokens=100, cost=0.01))
        log.record(_entry(pack_id="mr", tokens=200, cost=0.02))
        summary = log.get_pack_summary()
        mr = summary["mr"]
        assert mr["total_tokens"] == 300
        assert mr["total_cost_usd"] == 0.03
        assert mr["avg_tokens"] == 150

    def test_step_utilization(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="mon", step_budget=10, steps_used=5))
        log.record(_entry(pack_id="mon", step_budget=10, steps_used=10))
        summary = log.get_pack_summary()
        assert summary["mon"]["avg_step_utilization"] == 0.75

    def test_partial_progress_counts_as_positive(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="mr", status="success"))
        log.record(_entry(pack_id="mr", status="partial_progress"))
        log.record(_entry(pack_id="mr", status="failure"))
        summary = log.get_pack_summary()
        mr = summary["mr"]
        assert mr["successes"] == 1
        assert mr["partial_progress"] == 1
        assert mr["failures"] == 1
        # completion_rate = (success + partial) / total = 2/3
        assert mr["completion_rate"] == round(2 / 3, 3)

    def test_multiple_packs(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="a", status="success"))
        log.record(_entry(pack_id="b", status="failure"))
        summary = log.get_pack_summary()
        assert "a" in summary
        assert "b" in summary
        assert summary["a"]["completion_rate"] == 1.0
        assert summary["b"]["completion_rate"] == 0.0


# ============== ExecutionLog.get_step_efficiency ==============


class TestGetStepEfficiency:
    def test_empty(self, log_dir):
        log = _make_log(log_dir)
        assert log.get_step_efficiency() == {}

    def test_skips_zero_budget(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(step_budget=0, steps_used=5))
        assert log.get_step_efficiency() == {}

    def test_calculates_utilization(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="mr", step_budget=10, steps_used=8))
        log.record(_entry(pack_id="mr", step_budget=10, steps_used=2))
        eff = log.get_step_efficiency()
        assert eff["mr"]["runs"] == 2
        assert eff["mr"]["avg_utilization"] == 0.5

    def test_maxed_out_detection(self, log_dir):
        log = _make_log(log_dir)
        log.record(_entry(pack_id="pub", step_budget=10, steps_used=10))
        log.record(_entry(pack_id="pub", step_budget=10, steps_used=5))
        log.record(_entry(pack_id="pub", step_budget=10, steps_used=9.5))
        eff = log.get_step_efficiency()
        assert eff["pub"]["maxed_out_runs"] == 2  # 10/10=1.0 and 9.5/10=0.95
        assert eff["pub"]["maxed_out_rate"] == round(2 / 3, 3)


# ============== Load tail from disk ==============


class TestLoadTail:
    def test_loads_existing_file(self, log_dir):
        log = _make_log(log_dir)
        # Write some entries first
        for i in range(5):
            log.record(_entry(cycle=i))
        assert len(log._entries) == 5

        # Create a new log from the same file — should load tail
        log2 = _make_log(log_dir)
        log2._path = log._path
        log2._load_tail()
        assert len(log2._entries) == 5

    def test_handles_missing_file(self, log_dir):
        log = _make_log(log_dir)
        log._path = log_dir / "nonexistent.jsonl"
        log._load_tail()  # Should not raise
        assert log._entries == []

    def test_handles_malformed_lines(self, log_dir):
        log_file = log_dir / "execution_log.jsonl"
        log_file.write_text('{"cycle_num": 1}\nnot json\n{"cycle_num": 2}\n')
        log = _make_log(log_dir)
        log._path = log_file
        log._load_tail()
        assert len(log._entries) == 2


class TestRecordCycleExecution:
    def test_records_with_pack_fallback(self, log_dir):
        log = _make_log(log_dir)
        with patch("remy.core.execution_log.execution_log", log):
            record_cycle_execution(
                cycle_num=3,
                goal={
                    "goal_id": "g-1",
                    "description": "Research competitors",
                    "goal_template": "market_research",
                },
                worker_result=None,
                session_log=[{"type": "tool_call", "tool": "web_search", "result": "ok"}],
                evaluation={"success": True, "confidence": 0.9, "reason": "done"},
                duration_ms=400,
                tokens_used=120,
                cost_usd=0.002,
            )

        recent = log.get_recent(limit=1)[0]
        assert recent["goal_id"] == "g-1"
        assert recent["pack_id"] == "market_research"
        assert recent["status"] == "success"
        assert recent["tool_count"] == 1

    def test_records_zero_tool_status(self, log_dir):
        log = _make_log(log_dir)
        with patch("remy.core.execution_log.execution_log", log):
            record_cycle_execution(
                cycle_num=4,
                goal={"goal_id": "g-2", "description": "Do something", "goal_template": "general"},
                worker_result=None,
                session_log=[],
                evaluation={"success": False, "confidence": 0.2, "reason": "no progress"},
                duration_ms=200,
                tokens_used=80,
                cost_usd=0.0,
            )

        recent = log.get_recent(limit=1)[0]
        assert recent["status"] == "zero_tool"

    def test_records_partial_progress_from_worker_status(self, log_dir):
        log = _make_log(log_dir)

        class WorkerResult:
            status = "partial_progress"
            worker = "research_worker"
            tool_calls = 4
            evidence = {"findings_count": 3}

        with patch("remy.core.execution_log.execution_log", log):
            record_cycle_execution(
                cycle_num=5,
                goal={
                    "goal_id": "g-3",
                    "description": "Research influencers",
                    "goal_template": "market_research",
                },
                worker_result=WorkerResult(),
                session_log=[
                    {
                        "type": "tool_call",
                        "tool": "store",
                        "args": {"tags": "influencer-research"},
                        "result": "ok",
                    }
                ],
                evaluation={"success": False, "confidence": 0.3, "reason": "timeout"},
                duration_ms=800,
                tokens_used=200,
                cost_usd=0.003,
            )

        recent = log.get_recent(limit=1)[0]
        assert recent["status"] == "partial_progress"  # worker said partial_progress

    def test_records_partial_progress_when_success_with_real_tools(self, log_dir):
        log = _make_log(log_dir)

        class WorkerResult:
            status = "searching"
            worker = "research_worker"
            tool_calls = 2
            evidence = {}

        with patch("remy.core.execution_log.execution_log", log):
            record_cycle_execution(
                cycle_num=6,
                goal={
                    "goal_id": "g-4",
                    "description": "Research influencers",
                    "goal_template": "market_research",
                },
                worker_result=WorkerResult(),
                session_log=[
                    {
                        "type": "tool_call",
                        "tool": "web_search",
                        "args": {"query": "ai agent memory influencers"},
                        "result": "ok",
                    }
                ],
                evaluation={"success": True, "confidence": 0.7, "reason": "progress"},
                duration_ms=600,
                tokens_used=120,
                cost_usd=0.002,
            )

        recent = log.get_recent(limit=1)[0]
        assert recent["status"] == "partial_progress"


# ============== derive_cycle_status ==============


class TestDeriveCycleStatus:
    def test_partial_progress_from_worker(self):
        assert (
            derive_cycle_status(
                worker_status="partial_progress",
                eval_success=False,
                has_real_tools=True,
            )
            == "partial_progress"
        )

    def test_findings_collected(self):
        assert (
            derive_cycle_status(
                worker_status="findings_collected",
                eval_success=False,
                has_real_tools=True,
            )
            == "partial_progress"
        )

    def test_timeout_with_tools(self):
        assert (
            derive_cycle_status(
                worker_status="timeout",
                eval_success=False,
                has_real_tools=True,
            )
            == "partial_progress"
        )

    def test_timeout_without_tools(self):
        assert (
            derive_cycle_status(
                worker_status="timeout",
                eval_success=False,
                has_real_tools=False,
            )
            == "timeout"
        )

    def test_blocked_external(self):
        assert (
            derive_cycle_status(
                worker_status="blocked_external",
                eval_success=False,
                has_real_tools=True,
            )
            == "blocked"
        )

    def test_zero_tool(self):
        assert (
            derive_cycle_status(
                worker_status="",
                eval_success=False,
                has_real_tools=False,
            )
            == "zero_tool"
        )

    def test_success(self):
        assert (
            derive_cycle_status(
                worker_status="",
                eval_success=True,
                has_real_tools=True,
            )
            == "success"
        )

    def test_failure(self):
        assert (
            derive_cycle_status(
                worker_status="",
                eval_success=False,
                has_real_tools=True,
            )
            == "failure"
        )

    def test_searching_with_tools_is_partial(self):
        assert (
            derive_cycle_status(
                worker_status="searching",
                eval_success=True,
                has_real_tools=True,
            )
            == "partial_progress"
        )

    def test_attempted_with_tools_is_partial(self):
        assert (
            derive_cycle_status(
                worker_status="attempted",
                eval_success=True,
                has_real_tools=True,
            )
            == "partial_progress"
        )
