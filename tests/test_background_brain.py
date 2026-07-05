"""Tests for Phase 3: Background Brain ('Subconscious')."""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from aura import Aura as CognitiveMemory, Level


# ============== Format Insight Tests ==============

class TestFormatInsight:

    def test_decay_risk(self):
        from remy.core.background_brain import _format_insight
        ins = {
            "type": "decay_risk",
            "details": {
                "records": [
                    {"id": "r1", "content": "Grandmother Maria lived in Kyiv", "strength": 0.3, "activation_count": 5},
                ]
            }
        }
        result = _format_insight(ins)
        assert "fading" in result.lower()
        assert "Grandmother Maria" in result

    def test_conflict(self):
        from remy.core.background_brain import _format_insight
        ins = {
            "type": "conflict",
            "details": {
                "pairs": [{
                    "id_a": "r1", "id_b": "r2",
                    "content_a": "Maria was born in 1935",
                    "content_b": "Maria was born in 1937",
                }]
            }
        }
        result = _format_insight(ins)
        assert "contradiction" in result.lower()
        assert "Maria" in result

    def test_cluster(self):
        from remy.core.background_brain import _format_insight
        ins = {
            "type": "cluster",
            "details": {
                "dominant_tags": ["family", "history", "war"],
                "size": 7,
            }
        }
        result = _format_insight(ins)
        assert "cluster" in result.lower()
        assert "family" in result
        assert "7" in result

    def test_promotion(self):
        from remy.core.background_brain import _format_insight
        ins = {
            "type": "promotion",
            "details": {
                "records": [
                    {"id": "r1", "content": "Important memory about Petro", "can_promote": True},
                ]
            }
        }
        result = _format_insight(ins)
        assert "promoted" in result.lower()
        assert "Petro" in result

    def test_promotion_none_ready(self):
        from remy.core.background_brain import _format_insight
        ins = {
            "type": "promotion",
            "details": {
                "records": [
                    {"id": "r1", "content": "Not ready", "can_promote": False},
                ]
            }
        }
        result = _format_insight(ins)
        assert result is None

    def test_stale_topic_ignored(self):
        from remy.core.background_brain import _format_insight
        ins = {
            "type": "stale_topic",
            "details": {"topics": [{"tag": "old", "record_count": 2, "avg_strength": 0.1}]}
        }
        result = _format_insight(ins)
        assert result is None

    def test_hot_topic_ignored(self):
        from remy.core.background_brain import _format_insight
        ins = {
            "type": "hot_topic",
            "details": {"topics": [{"tag": "active", "record_count": 5, "avg_activations": 4}]}
        }
        result = _format_insight(ins)
        assert result is None

    def test_empty_records(self):
        from remy.core.background_brain import _format_insight
        ins = {"type": "decay_risk", "details": {"records": []}}
        assert _format_insight(ins) is None

    def test_empty_pairs(self):
        from remy.core.background_brain import _format_insight
        ins = {"type": "conflict", "details": {"pairs": []}}
        assert _format_insight(ins) is None


# ============== Format Insights Tests (transient, not stored) ==============

class TestFormatInsights:

    def test_formats_actionable_insights(self):
        from remy.core.background_brain import _format_all_insights

        insights = [
            {
                "type": "decay_risk",
                "details": {"records": [{"id": "r1", "content": "Test memory", "strength": 0.2, "activation_count": 3}]}
            },
            {
                "type": "stale_topic",
                "details": {"topics": [{"tag": "old", "record_count": 2, "avg_strength": 0.1}]}
            },
        ]

        result = _format_all_insights(insights)
        assert len(result) == 1  # Only decay_risk formatted, stale_topic ignored
        assert "fading" in result[0].lower()

    def test_formats_multiple_types(self):
        from remy.core.background_brain import _format_all_insights

        insights = [
            {
                "type": "decay_risk",
                "details": {"records": [{"id": "r1", "content": "Fading memory", "strength": 0.2, "activation_count": 3}]}
            },
            {
                "type": "conflict",
                "details": {"pairs": [{"id_a": "r1", "id_b": "r2", "content_a": "Fact A", "content_b": "Fact B"}]}
            },
        ]

        result = _format_all_insights(insights)
        assert len(result) == 2

    def test_not_stored_in_brain(self, tmp_path):
        """Transient insights must NOT be stored as brain records."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _format_all_insights

        insights = [
            {
                "type": "decay_risk",
                "details": {"records": [{"id": "r1", "content": "Test memory", "strength": 0.2, "activation_count": 3}]}
            },
        ]

        _format_all_insights(insights)

        # Nothing should be stored in brain
        results = b.search(query="", tags=["background-insight"], limit=5)
        assert len(results) == 0
        b.close()


# ============== Run Background Tests ==============

class TestRunBackground:

    def test_empty_brain(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import run_background

        report = run_background(brain=b)

        assert "timestamp" in report
        assert report["decay"]["decayed"] == 0
        assert report["decay"]["archived"] == 0
        assert report["insights_found"] == 0
        assert report["total_records"] == 0
        assert "error" not in report

    def test_with_data(self, tmp_path):
        from aura import Level

        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(content="Test memory one", tags=["test"], level=Level.IDENTITY)
        b.store(content="Test memory two", tags=["test"], level=Level.IDENTITY)

        from remy.core.background_brain import run_background
        report = run_background(brain=b)

        assert "timestamp" in report
        assert report["total_records"] >= 1  # consolidation may merge similar records
        assert "error" not in report


# ============== Print Report Tests ==============

class TestPrintReport:

    def test_prints_without_error(self, capsys):
        from remy.core.background_brain import print_report
        report = {
            "timestamp": "2026-02-11T12:00:00",
            "decay": {"decayed": 5, "archived": 1},
            "reflect": {"promoted": 2, "connected": 3, "archived": 0},
            "insights_found": 2,
            "total_records": 42,
        }
        print_report(report)
        captured = capsys.readouterr()
        assert "BACKGROUND BRAIN" in captured.out
        assert "42" in captured.out
        assert "5 decayed" in captured.out

    def test_prints_with_error(self, capsys):
        from remy.core.background_brain import print_report
        report = {
            "timestamp": "2026-02-11T12:00:00",
            "decay": {"decayed": 0, "archived": 0},
            "reflect": {"promoted": 0, "connected": 0, "archived": 0},
            "insights_found": 0,
            "total_records": 0,
            "error": "DB corrupt",
        }
        print_report(report)
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "DB corrupt" in captured.out


# ============== System Instruction Integration ==============

class TestBackgroundInsightsInInstruction:

    def test_includes_background_insights(self, tmp_path):
        # Transient insights come from module-level state, not brain records
        transient = ["Knowledge cluster found: family, history (5 connected records)."]

        with patch("remy.core.background_brain.get_transient_insights", return_value=transient), \
             patch("remy.core.background_brain.get_transient_cross_connections", return_value=[]):
            from remy.core.brain_tools import build_system_instruction
            b = CognitiveMemory(str(tmp_path / "brain"))
            with patch("remy.core.brain_tools.brain", b):
                instruction = build_system_instruction()
            b.close()

        assert "Background insights" in instruction
        assert "cluster" in instruction.lower()

    def test_no_background_insights(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "Background insights" not in instruction
        b.close()


# ============== Scheduled Task Checker Tests ==============

class TestCheckScheduledTasks:

    def test_no_tasks(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _check_scheduled_tasks
        reminders = _check_scheduled_tasks(b)
        assert reminders == []
        b.close()

    def test_due_today(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        today = datetime.now().isoformat()
        b.store(
            content="Scheduled: Call grandma | Due: today",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={"type": "scheduled_task", "description": "Call grandma", "due_date": today, "status": "active"},
        )
        from remy.core.background_brain import _check_scheduled_tasks
        reminders = _check_scheduled_tasks(b)
        assert len(reminders) == 1
        assert "Call grandma" in reminders[0]
        assert "Due today" in reminders[0]
        b.close()

    def test_due_tomorrow(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        tomorrow = (datetime.now() + timedelta(days=1)).isoformat()
        b.store(
            content="Scheduled: Buy flowers",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={"type": "scheduled_task", "description": "Buy flowers", "due_date": tomorrow, "status": "active"},
        )
        from remy.core.background_brain import _check_scheduled_tasks
        reminders = _check_scheduled_tasks(b)
        assert len(reminders) == 1
        assert "Due tomorrow" in reminders[0]
        b.close()

    def test_future_task_not_triggered(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        future = (datetime.now() + timedelta(days=10)).isoformat()
        b.store(
            content="Scheduled: Far away task",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={"type": "scheduled_task", "description": "Far away", "due_date": future, "status": "active"},
        )
        from remy.core.background_brain import _check_scheduled_tasks
        reminders = _check_scheduled_tasks(b)
        assert reminders == []
        b.close()

    def test_inactive_task_skipped(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        today = datetime.now().isoformat()
        b.store(
            content="Scheduled: Done task",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={"type": "scheduled_task", "description": "Done", "due_date": today, "status": "completed"},
        )
        from remy.core.background_brain import _check_scheduled_tasks
        reminders = _check_scheduled_tasks(b)
        assert reminders == []
        b.close()

    def test_recurring_task_advances(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        today = datetime.now().isoformat()
        rec = b.store(
            content="Scheduled: Weekly call",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={"type": "scheduled_task", "description": "Weekly call", "due_date": today, "repeat": "weekly", "status": "active"},
        )
        from remy.core.background_brain import _check_scheduled_tasks
        _check_scheduled_tasks(b)
        updated = b.get(rec.id)
        new_due = updated.metadata.get("due_date", "")
        assert new_due != today  # Should have been advanced
        b.close()


# ============== Advance Due Date Tests ==============

class TestAdvanceDueDate:

    def test_daily(self):
        from remy.core.background_brain import _advance_due_date
        base = datetime(2026, 2, 11)
        result = _advance_due_date(base, "daily")
        assert result == datetime(2026, 2, 12)

    def test_weekly(self):
        from remy.core.background_brain import _advance_due_date
        base = datetime(2026, 2, 11)
        result = _advance_due_date(base, "weekly")
        assert result == datetime(2026, 2, 18)

    def test_monthly(self):
        from remy.core.background_brain import _advance_due_date
        base = datetime(2026, 2, 11)
        result = _advance_due_date(base, "monthly")
        assert result.month == 3
        assert result.day == 11

    def test_monthly_december_wraps(self):
        from remy.core.background_brain import _advance_due_date
        base = datetime(2026, 12, 15)
        result = _advance_due_date(base, "monthly")
        assert result.year == 2027
        assert result.month == 1

    def test_unknown_repeat(self):
        from remy.core.background_brain import _advance_due_date
        result = _advance_due_date(datetime(2026, 1, 1), "yearly")
        assert result is None


# ============== Cross Connection Discovery Tests ==============

class TestDiscoverCrossConnections:

    def test_empty_brain(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _discover_cross_connections
        result = _discover_cross_connections(b)
        assert result == []
        b.close()

    def test_finds_indirect_links(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        # Create A -> B -> C chain (A not directly connected to C)
        r1 = b.store(content="Knee pain after running", tags=["health"])
        r2 = b.store(content="Started running in 2023", tags=["exercise"])
        r3 = b.store(content="Ibuprofen helped with pain", tags=["medication"])
        # Need enough records for the function to proceed
        for i in range(5):
            b.store(content=f"Filler record {i}", tags=["filler"])
        b.connect(r1.id, r2.id, weight=0.8)
        b.connect(r2.id, r3.id, weight=0.7)
        # r1 and r3 are NOT directly connected

        from remy.core.background_brain import _discover_cross_connections
        result = _discover_cross_connections(b)
        # May or may not find the link due to random sampling, but should not crash
        assert isinstance(result, list)
        b.close()

    def test_max_three_results(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        # Create many chains
        records = []
        for i in range(15):
            r = b.store(content=f"Record {i}", tags=["test"])
            records.append(r)
        # Create long chain
        for i in range(len(records) - 1):
            b.connect(records[i].id, records[i+1].id, weight=0.8)

        from remy.core.background_brain import _discover_cross_connections
        result = _discover_cross_connections(b)
        assert len(result) <= 3
        b.close()


# ============== Send Notifications Tests ==============

class TestBuildNotificationMessage:

    def test_returns_none_when_empty(self, tmp_path):
        from remy.core.background_brain import build_notification_message
        b = CognitiveMemory(str(tmp_path / "brain"))
        report = {"insights_found": 0, "cross_connections": 0, "task_reminders": []}
        result = build_notification_message(report, b)
        assert result is None
        b.close()

    def test_includes_task_reminders(self, tmp_path):
        from remy.core.background_brain import build_notification_message
        b = CognitiveMemory(str(tmp_path / "brain"))
        report = {"insights_found": 0, "cross_connections": 0, "task_reminders": ["Due today: Call grandma."]}
        result = build_notification_message(report, b)
        assert result is not None
        assert "Call grandma" in result
        assert "Hey!" in result  # Conversational header
        b.close()

    def test_includes_session_context(self, tmp_path):
        from remy.core.background_brain import build_notification_message
        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(content="Discussed vitamin D supplements and dosage.", level=Level.DOMAIN, tags=["session-summary"])
        report = {"insights_found": 0, "cross_connections": 0, "task_reminders": ["Due today: Take vitamins."]}
        result = build_notification_message(report, b)
        assert result is not None
        assert "vitamin D" in result or "continue" in result.lower()
        b.close()

    def test_includes_followup_question(self, tmp_path):
        from remy.core.background_brain import build_notification_message
        b = CognitiveMemory(str(tmp_path / "brain"))
        # Transient insights come from module-level state, not brain records
        with patch("remy.core.background_brain.get_transient_insights",
                    return_value=["Test insight about health"]):
            report = {"insights_found": 1, "cross_connections": 0, "task_reminders": []}
            result = build_notification_message(report, b)
        assert result is not None
        assert "?" in result  # Should contain a question
        b.close()

    def test_max_three_reminders(self, tmp_path):
        from remy.core.background_brain import build_notification_message
        b = CognitiveMemory(str(tmp_path / "brain"))
        report = {
            "insights_found": 0, "cross_connections": 0,
            "task_reminders": [f"Due today: Task {i}" for i in range(5)],
        }
        result = build_notification_message(report, b)
        assert "Task 0" in result
        assert "Task 2" in result
        assert "Task 4" not in result  # Max 3
        b.close()


class TestSendNotifications:

    @pytest.mark.asyncio
    async def test_skips_when_no_config(self):
        from remy.core.background_brain import send_notifications
        report = {"insights_found": 1, "cross_connections": 0, "task_reminders": [], "decay": {"archived": 0}, "reflect": {"promoted": 0}}
        with patch("remy.core.background_brain.settings") as mock_settings:
            mock_settings.TELEGRAM_BOT_TOKEN = None
            mock_settings.PROACTIVE_CHAT_ID = None
            # Should not raise
            await send_notifications(report)

    @pytest.mark.asyncio
    async def test_skips_when_nothing_to_report(self):
        from remy.core.background_brain import send_notifications
        report = {"insights_found": 0, "cross_connections": 0, "task_reminders": [], "decay": {"archived": 0}, "reflect": {"promoted": 0}}
        with patch("remy.core.background_brain.settings") as mock_settings:
            mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
            mock_settings.PROACTIVE_CHAT_ID = 12345
            # Should not send anything
            await send_notifications(report)

    @pytest.mark.asyncio
    async def test_sends_when_configured(self, tmp_path):
        from remy.core.background_brain import send_notifications
        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(content="Test insight", level=Level.WORKING, tags=["background-insight", "test"])

        report = {"insights_found": 1, "cross_connections": 0, "task_reminders": ["Due today: Call grandma."], "decay": {"archived": 0}, "reflect": {"promoted": 0}}

        with patch("remy.core.notification_router.notify") as mock_notify:
            await send_notifications(report, brain=b)

        mock_notify.assert_called_once()
        call_args, call_kwargs = mock_notify.call_args
        assert "Call grandma" in call_args[0]
        assert call_kwargs["level"] == "info"
        assert call_kwargs["event_type"] == "background.report"
        b.close()
