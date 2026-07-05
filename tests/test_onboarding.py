"""Tests for user onboarding and profile system."""

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_brain(tmp_path):
    """Real CognitiveMemory for integration testing."""
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def execute_tool(mock_brain, tmp_path):
    """Provide execute_tool with mocked brain and registry."""
    with patch("remy.core.brain_tools.brain", mock_brain), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.tool_registry.settings") as mock_settings:
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

        from remy.core.brain_tools import execute_tool
        yield execute_tool


class TestProfileDetection:

    def test_first_time_gets_onboarding(self, mock_brain):
        """Empty brain → system instruction contains FIRST-TIME USER block."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "FIRST-TIME USER" in instruction
        assert "store_user_profile" in instruction

    def test_returning_user_gets_profile(self, mock_brain):
        """Brain with user-profile → system instruction contains USER IDENTITY block."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Taras; Location: Kyiv",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"type": "user_profile", "name": "Taras", "location": "Kyiv",
                       "source": "user-confirmed", "verified": True},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "USER IDENTITY" in instruction
        assert "Taras" in instruction
        assert "Kyiv" in instruction
        assert "FIRST-TIME USER" not in instruction

    def test_profile_works_all_channels(self, mock_brain):
        """Profile injection works for all channel types."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Olena",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"type": "user_profile", "name": "Olena",
                       "source": "user-confirmed", "verified": True},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            for ch in ("voice", "telegram", "desktop"):
                instruction = build_system_instruction(channel=ch)
                assert "Olena" in instruction, f"Profile missing for channel={ch}"


class TestStoreUserProfile:

    def test_create_new_profile(self, execute_tool):
        """First call creates a new profile."""
        result = execute_tool("store_user_profile", {"name": "Taras", "location": "Kyiv"})
        data = json.loads(result)
        assert data["created"] is True
        assert data["profile"]["name"] == "Taras"
        assert data["profile"]["location"] == "Kyiv"

    def test_update_existing_profile(self, execute_tool):
        """Second call merges new fields into existing profile."""
        execute_tool("store_user_profile", {"name": "Taras"})
        result = execute_tool("store_user_profile", {"occupation": "Engineer", "age": "30"})
        data = json.loads(result)
        assert data["updated"] is True
        assert data["profile"]["name"] == "Taras"  # preserved
        assert data["profile"]["occupation"] == "Engineer"  # new
        assert data["profile"]["age"] == "30"  # new

    def test_birth_date_in_age_field_is_normalized(self, execute_tool):
        """If age receives an ISO birth date, persist birth_date and compute age."""
        result = execute_tool("store_user_profile", {"name": "Taras", "age": "1983-05-19"})
        data = json.loads(result)

        assert data.get("created") is True or data.get("updated") is True
        assert data["profile"]["birth_date"] == "1983-05-19"
        assert data["profile"]["age"].isdigit()

    def test_overwrite_existing_field(self, execute_tool):
        """Updating a field overwrites the old value."""
        execute_tool("store_user_profile", {"name": "Taras", "location": "Kyiv"})
        result = execute_tool("store_user_profile", {"location": "Lviv"})
        data = json.loads(result)
        assert data["updated"] is True
        assert data["profile"]["location"] == "Lviv"
        assert data["profile"]["name"] == "Taras"  # preserved

    def test_empty_fields_ignored(self, execute_tool):
        """Calling with empty strings returns error."""
        result = execute_tool("store_user_profile", {"name": "", "age": ""})
        data = json.loads(result)
        assert "error" in data

    def test_profile_stored_at_identity_level(self, mock_brain, execute_tool):
        """Profile record is stored at IDENTITY level."""
        from aura import Level
        execute_tool("store_user_profile", {"name": "Taras"})
        profiles = mock_brain.search(query="", tags=["user-profile"], limit=1)
        assert len(profiles) == 1
        assert profiles[0].level == Level.IDENTITY

    def test_profile_has_correct_tags(self, mock_brain, execute_tool):
        """Profile record has user-profile and identity tags."""
        execute_tool("store_user_profile", {"name": "Maria"})
        profiles = mock_brain.search(query="", tags=["user-profile"], limit=1)
        assert len(profiles) == 1
        assert "user-profile" in profiles[0].tags
        assert "identity" in profiles[0].tags

    def test_profile_contact_data_is_protected(self, mock_brain, execute_tool):
        """Phone/email stay stored exactly but are not exposed in broad profile content."""
        result = execute_tool(
            "store_user_profile",
            {"name": "Taras", "phone": "+380000000000", "email": "taras@example.com"},
        )
        data = json.loads(result)
        assert data["profile"]["phone"] == "[protected]"
        assert data["profile"]["email"] == "[protected]"

        profiles = mock_brain.search(query="", tags=["user-profile"], limit=1)
        assert len(profiles) == 1
        assert "taras@example.com" not in profiles[0].content.lower()
        assert "+380000000000" not in profiles[0].content
        assert profiles[0].metadata["email"] == "taras@example.com"
        assert profiles[0].metadata["phone"] == "+380000000000"

    def test_get_full_record_redacts_protected_profile_fields(self, mock_brain, execute_tool):
        execute_tool(
            "store_user_profile",
            {"name": "Taras", "phone": "+380000000000", "email": "taras@example.com"},
        )
        profile = mock_brain.search(query="", tags=["user-profile"], limit=1)[0]

        result = execute_tool("get_full_record", {"record_id": profile.id})
        data = json.loads(result)

        assert "taras@example.com" not in data["content"].lower()
        assert "+380000000000" not in data["content"]
        assert data["metadata"]["email"] == "[protected]"
        assert data["metadata"]["phone"] == "[protected]"
        assert sorted(data["protected_fields_present"]) == ["email", "phone"]

    def test_get_protected_record_returns_exact_profile_fields(self, mock_brain, execute_tool):
        execute_tool(
            "store_user_profile",
            {"name": "Taras", "phone": "+380000000000", "email": "taras@example.com"},
        )
        profile = mock_brain.search(query="", tags=["user-profile"], limit=1)[0]

        result = execute_tool(
            "get_protected_record",
            {"record_id": profile.id, "fields": "email"},
        )
        data = json.loads(result)

        assert data["values"]["email"] == "taras@example.com"
        assert data["protected_fields"] == ["email"]
        assert data["verified"] is True


class TestUserIdentityBuilder:
    """Tests for _build_user_identity() — rich identity with verification status."""

    def test_returns_none_without_profile(self, mock_brain):
        """No user-profile → returns None."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _build_user_identity
            result = _build_user_identity()
        assert result is None

    def test_verified_facts_marked_with_checkmark(self, mock_brain):
        """Verified profile fields get ✓ marker."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Taras; Age: 30",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"type": "user_profile", "name": "Taras", "age": "30",
                       "source": "user-confirmed", "verified": True},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _build_user_identity
            result = _build_user_identity()
        assert result is not None
        assert "USER IDENTITY" in result
        assert "Taras" in result
        assert "✓" in result
        assert "CONFIRMED" in result or "CERTAIN" in result

    def test_unverified_facts_marked_with_question(self, mock_brain):
        """Unverified profile fields get ? marker."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Unknown",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"type": "user_profile", "name": "Unknown",
                       "source": "agent-autonomous", "verified": False},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _build_user_identity
            result = _build_user_identity()
        assert result is not None
        assert "?" in result
        assert "Unverified" in result

    def test_includes_person_records(self, mock_brain):
        """Person records (family members) appear in identity."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Oleksandr",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"name": "Oleksandr", "source": "user-confirmed", "verified": True},
        )
        mock_brain.store(
            content="Maksym Example, brother, born 01.01.1990",
            level=Level.IDENTITY,
            tags=["person"],
            metadata={"full_name": "Maksym Example", "role": "brother",
                       "source": "user-confirmed", "verified": True, "trust_score": 1.0},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _build_user_identity
            result = _build_user_identity()
        assert "Maksym" in result
        assert "✓" in result

    def test_low_trust_person_marked_unverified(self, mock_brain):
        """Low-trust person record gets ? marker."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Oleksandr",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"name": "Oleksandr", "source": "user-confirmed", "verified": True},
        )
        mock_brain.store(
            content="Ivan Petrenko, colleague",
            level=Level.DOMAIN,
            tags=["person"],
            metadata={"full_name": "Ivan Petrenko",
                       "source": "agent-autonomous", "verified": False, "trust_score": 0.3},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _build_user_identity
            result = _build_user_identity()
        assert "Ivan" in result
        assert "unverified" in result

    def test_identity_rules_present(self, mock_brain):
        """Identity block contains behavior rules for the agent."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Taras",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"name": "Taras", "source": "user-confirmed", "verified": True},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _build_user_identity
            result = _build_user_identity()
        assert "never say" in result.lower() or "Never say" in result
        assert "assume" in result.lower() or "guess" in result.lower()
        assert "Do NOT recite" in result

    def test_system_instruction_uses_identity_block(self, mock_brain):
        """build_system_instruction() uses _build_user_identity() for returning users."""
        from aura import Level
        mock_brain.store(
            content="User Profile: Name: Taras; Location: Kyiv",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={"name": "Taras", "location": "Kyiv",
                       "source": "user-confirmed", "verified": True},
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="desktop")
        assert "USER IDENTITY" in instruction
        assert "Verified facts" in instruction
        assert "✓" in instruction
        assert "Taras" in instruction


class TestProactiveSessionStart:
    """Tests for proactive session start — scheduled tasks via get_proactive_context()."""

    def setup_method(self):
        """Clear the proactive context cache before each test to prevent cross-test contamination."""
        import remy.core.brain_tools as bt
        bt._proactive_context_cache.clear()

    def test_today_task_in_instruction(self, mock_brain):
        """Scheduled task due today appears in system instruction."""
        from datetime import datetime
        from aura import Level
        today = datetime.now().strftime("%Y-%m-%d")
        mock_brain.store(
            content=f"Scheduled: Call grandma | Due: {today}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "type": "scheduled_task",
                "description": "Call grandma",
                "due_date": today,
                "status": "active",
            },
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "Call grandma" in instruction
        assert "TASKS FOR TODAY" in instruction or "URGENT" in instruction

    def test_tomorrow_task_in_instruction(self, mock_brain):
        """Scheduled task due tomorrow appears in system instruction."""
        from datetime import datetime, timedelta
        from aura import Level
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        mock_brain.store(
            content=f"Scheduled: Doctor appointment | Due: {tomorrow}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "type": "scheduled_task",
                "description": "Doctor appointment",
                "due_date": tomorrow,
                "status": "active",
            },
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "Doctor appointment" in instruction
        assert "TOMORROW" in instruction

    def test_future_task_not_in_instruction(self, mock_brain):
        """Scheduled task far in the future does NOT appear."""
        from datetime import datetime, timedelta
        from aura import Level
        future = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        mock_brain.store(
            content=f"Scheduled: Future thing | Due: {future}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "type": "scheduled_task",
                "description": "Future thing",
                "due_date": future,
                "status": "active",
            },
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "Future thing" not in instruction

    def test_completed_task_not_in_scheduled_section(self, mock_brain):
        """Completed (non-active) tasks are excluded from proactive context."""
        from datetime import datetime
        from aura import Level
        today = datetime.now().strftime("%Y-%m-%d")
        mock_brain.store(
            content=f"Scheduled: Done task | Due: {today}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "type": "scheduled_task",
                "description": "Done task",
                "due_date": today,
                "status": "completed",
            },
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        # Completed task should not appear in URGENT or UPCOMING sections
        assert "- Done task" not in instruction

    def test_task_reminder_prompt_present(self, mock_brain):
        """When tasks are present, proactive awakening context section is included."""
        from datetime import datetime
        from aura import Level
        today = datetime.now().strftime("%Y-%m-%d")
        mock_brain.store(
            content=f"Scheduled: Test task | Due: {today}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "type": "scheduled_task",
                "description": "Test task",
                "due_date": today,
                "status": "active",
            },
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "PROACTIVE AWAKENING CONTEXT" in instruction or "INSTRUCTION:" in instruction

    def test_personal_event_today_appears_in_proactive_context(self, mock_brain):
        """Today's birthdays should appear in proactive context for first-reply reminders."""
        from datetime import datetime
        from aura import Level

        today = datetime.now().strftime("%Y-%m-%d")
        mock_brain.store(
            content="Наталія, mother",
            level=Level.IDENTITY,
            tags=["person", "family"],
            metadata={
                "type": "person",
                "full_name": "Наталія",
                "role": "mother",
                "birth_date": today,
            },
        )

        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")

        assert "PERSONAL EVENTS TODAY" in instruction
        assert "Наталія birthday" in instruction

    def test_no_tasks_no_proactive_section(self, mock_brain):
        """No scheduled tasks → no proactive awakening section in instruction."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "PROACTIVE AWAKENING CONTEXT" not in instruction


class TestTemporalContext:
    """Tests for temporal context in system instruction."""

    def test_contains_current_day(self, mock_brain):
        """System instruction contains the current day of the week."""
        from datetime import datetime
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        day_name = datetime.now().strftime("%A")
        assert day_name in instruction

    def test_contains_time_period(self, mock_brain):
        """System instruction contains a time period hint."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        # Should contain one of the time periods
        time_periods = ["morning", "afternoon", "evening", "night", "late night"]
        assert any(p in instruction.lower() for p in time_periods)

    def test_contains_adapt_instruction(self, mock_brain):
        """System instruction tells agent to adapt to time of day."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")
        assert "Adapt" in instruction or "adapt" in instruction

    def test_temporal_in_all_channels(self, mock_brain):
        """Temporal context present in all channel types."""
        from datetime import datetime
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            for ch in ("voice", "telegram", "desktop"):
                instruction = build_system_instruction(channel=ch)
                day_name = datetime.now().strftime("%A")
                assert day_name in instruction, f"Day missing for channel={ch}"


class TestFormatProfileContent:

    def test_format_all_fields(self):
        from remy.core.brain_tools import _format_profile_content
        content = _format_profile_content({
            "name": "Taras",
            "age": "30",
            "location": "Kyiv",
        })
        assert "Taras" in content
        assert "30" in content
        assert "Kyiv" in content
        assert content.startswith("User Profile:")

    def test_format_single_field(self):
        from remy.core.brain_tools import _format_profile_content
        content = _format_profile_content({"name": "Olena"})
        assert "Olena" in content
        assert content.startswith("User Profile:")

    def test_format_ignores_empty(self):
        from remy.core.brain_tools import _format_profile_content
        content = _format_profile_content({"name": "Test", "age": "", "location": None})
        assert "Test" in content
        assert "Age" not in content

    def test_format_excludes_protected_contact_fields(self):
        from remy.core.brain_tools import _format_profile_content

        content = _format_profile_content({
            "name": "Test",
            "phone": "+380001112233",
            "email": "user@example.com",
        })

        assert "Test" in content
        assert "+380001112233" not in content
        assert "user@example.com" not in content
