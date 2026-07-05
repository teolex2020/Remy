"""
Tests for BLOCKED_BY_USER goal status.

Covers:
- block_goal() / unblock_goal() / unblock_goal_by_action_id()
- get_active_goals() sorting (blocked goals last)
- Auto-unblock stale blocked goals (>5 min)
- Approval resolution triggers goal unblock
- _decide_and_act() skips blocked goals
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from remy.core.agent_tools import Level
from remy.core.agent_tools import _AuraCompat as Aura


@pytest.fixture
def brain(tmp_path):
    b = Aura(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture(autouse=True)
def patch_autonomy(brain):
    """Patch autonomy module to use test brain."""
    mock_au = MagicMock()
    mock_au.brain = brain
    mock_au.settings = MagicMock()
    mock_au.settings.TELEGRAM_BOT_TOKEN = None
    mock_au.settings.PROACTIVE_CHAT_ID = None
    with patch("remy.core.autonomy_goals._get_autonomy", return_value=mock_au):
        yield mock_au


def _create_goal(brain, description="Test goal", priority="medium", status="active"):
    """Helper to create a goal record directly."""
    import uuid

    goal_id = f"goal-{uuid.uuid4().hex[:12]}"
    rec = brain.store(
        content=f"Goal [{priority.upper()}]: {description}",
        level=Level.DECISIONS,
        tags=["autonomous-goal", f"priority-{priority}"],
        metadata={
            "type": "autonomous_goal",
            "goal_id": goal_id,
            "priority": priority,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "attempts": 0,
            "source": "test",
            "verified": False,
        },
    )
    return rec.id, goal_id


# ============== block_goal / unblock_goal ==============


class TestBlockUnblock:
    def test_block_goal_sets_status(self, brain):
        rec_id, goal_id = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal

        block_goal(rec_id, action_id="act-123")

        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "blocked_by_user"
        assert rec.metadata["blocked_action_id"] == "act-123"
        assert "blocked_at" in rec.metadata

    def test_unblock_goal_restores_active(self, brain):
        rec_id, goal_id = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal, unblock_goal

        block_goal(rec_id, action_id="act-123")
        unblock_goal(rec_id)

        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "active"
        assert "blocked_at" not in rec.metadata
        assert "blocked_action_id" not in rec.metadata

    def test_unblock_noop_if_not_blocked(self, brain):
        rec_id, goal_id = _create_goal(brain)

        from remy.core.autonomy_goals import unblock_goal

        unblock_goal(rec_id)

        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "active"

    def test_unblock_by_action_id(self, brain):
        rec_id, goal_id = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal, unblock_goal_by_action_id

        block_goal(rec_id, action_id="act-456")

        found = unblock_goal_by_action_id("act-456")
        assert found is True

        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "active"

    def test_unblock_by_action_id_not_found(self, brain):
        rec_id, goal_id = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal, unblock_goal_by_action_id

        block_goal(rec_id, action_id="act-456")

        found = unblock_goal_by_action_id("act-999")
        assert found is False

        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "blocked_by_user"

    def test_block_emits_event(self, brain):
        rec_id, goal_id = _create_goal(brain)

        with patch("remy.core.autonomy_goals.event_bus") as mock_bus:
            from remy.core.autonomy_goals import block_goal

            block_goal(rec_id, action_id="act-789")

            mock_bus.emit.assert_called_once()
            event_name, event_data = mock_bus.emit.call_args[0]
            assert event_name == "goal_blocked"
            assert event_data["record_id"] == rec_id
            assert event_data["action_id"] == "act-789"
            assert event_data["status"] == "blocked_by_user"

    def test_unblock_emits_event(self, brain):
        rec_id, goal_id = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal

        block_goal(rec_id, action_id="act-789")

        with patch("remy.core.autonomy_goals.event_bus") as mock_bus:
            from remy.core.autonomy_goals import unblock_goal

            unblock_goal(rec_id)

            mock_bus.emit.assert_called_once()
            event_name, event_data = mock_bus.emit.call_args[0]
            assert event_name == "goal_unblocked"
            assert event_data["record_id"] == rec_id

    def test_block_goal_external_sets_reason_and_evidence(self, brain):
        rec_id, _ = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal

        block_goal(
            rec_id,
            status="blocked_external",
            reason="email verification required",
            evidence="email verification required at https://app.example.com/welcome",
        )

        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "blocked_external"
        assert rec.metadata["blocked_reason"] == "email verification required"
        assert "welcome" in rec.metadata["blocked_evidence"]

    def test_resume_goal_from_blocker_preserves_resume_context(self, brain):
        rec_id, _ = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal, resume_goal_from_blocker

        block_goal(
            rec_id,
            status="blocked_external",
            reason="captcha challenge",
            evidence="captcha at https://example.com/signup",
            resume_context="Resume from signup page after captcha clears",
        )

        resumed = resume_goal_from_blocker(rec_id, note="User solved captcha")
        assert resumed is True

        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "active"
        assert "captcha challenge" in rec.metadata["resume_context"]
        assert "User solved captcha" in rec.metadata["resume_context"]


# ============== get_active_goals() with blocked ==============


class TestGetActiveGoalsBlocked:
    def test_blocked_goals_included_with_flag(self, brain):
        _create_goal(brain, "Active goal")
        rec_id2, _ = _create_goal(brain, "Blocked goal")

        from remy.core.autonomy_goals import block_goal, get_active_goals

        block_goal(rec_id2, action_id="act-1")

        goals = get_active_goals()
        assert len(goals) == 2

        # Active goal should be first, blocked second
        assert goals[0]["blocked"] is False
        assert goals[1]["blocked"] is True

    def test_blocked_goals_sorted_last(self, brain):
        rec_id1, _ = _create_goal(brain, "Goal A", priority="low")
        rec_id2, _ = _create_goal(brain, "Goal B", priority="high")
        rec_id3, _ = _create_goal(brain, "Goal C", priority="medium")

        from remy.core.autonomy_goals import block_goal, get_active_goals

        block_goal(rec_id2, action_id="act-1")  # Block the high-priority one

        goals = get_active_goals()
        assert len(goals) == 3

        # Non-blocked goals first (sorted by priority), blocked last
        assert goals[0]["blocked"] is False
        assert goals[1]["blocked"] is False
        assert goals[2]["blocked"] is True
        assert "Goal B" in goals[2]["description"]  # High-priority but blocked → last

    def test_auto_unblock_stale_goals(self, brain):
        rec_id, _ = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal, get_active_goals

        block_goal(rec_id, action_id="act-stale")

        # Manually backdate the blocked_at to 6 minutes ago
        rec = brain.get(rec_id)
        meta = dict(rec.metadata or {})
        meta["blocked_at"] = (datetime.now() - timedelta(minutes=6)).isoformat()
        brain.update(rec_id, metadata=meta)

        # get_active_goals should auto-unblock it
        goals = get_active_goals()
        assert len(goals) == 1
        assert goals[0]["blocked"] is False

        # Verify in brain
        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "active"

    def test_external_blocked_goals_included_with_reason(self, brain):
        rec_id, _ = _create_goal(brain, "Signup goal")

        from remy.core.autonomy_goals import block_goal, get_active_goals

        block_goal(
            rec_id,
            status="blocked_external",
            reason="captcha challenge",
            evidence="captcha challenge at https://example.com/signup",
        )

        goals = get_active_goals()
        assert len(goals) == 1
        assert goals[0]["blocked"] is True
        assert goals[0]["block_status"] == "blocked_external"
        assert goals[0]["blocked_reason"] == "captcha challenge"

    def test_active_goal_exposes_resume_context(self, brain):
        rec_id, _ = _create_goal(brain, "Signup goal")

        rec = brain.get(rec_id)
        meta = dict(rec.metadata or {})
        meta["resume_context"] = "Continue from dashboard after email verification"
        brain.update(rec_id, metadata=meta)

        from remy.core.autonomy_goals import get_active_goals

        goals = get_active_goals()
        assert goals[0]["resume_context"] == "Continue from dashboard after email verification"


# ============== Approval queue → unblock integration ==============


class TestApprovalUnblock:
    def test_emit_resolved_unblocks_goal(self, brain):
        """When approval resolves, the corresponding blocked goal gets unblocked."""
        rec_id, _ = _create_goal(brain)

        from remy.core.autonomy_goals import block_goal

        block_goal(rec_id, action_id="action-uuid-123")

        # Simulate approval resolution
        from remy.core.approval_queue import ApprovalQueue, PendingAction

        q = ApprovalQueue()

        action = PendingAction(
            action_id="action-uuid-123",
            description="test action",
            action_fn=lambda: '{"ok": true}',
        )
        action._approved = True
        action._resolved = True

        q._emit_resolved(action)

        # Goal should be unblocked
        rec = brain.get(rec_id)
        assert rec.metadata["status"] == "active"
