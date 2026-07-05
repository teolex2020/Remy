"""Tests for Runtime Directives (AUTON-2) — runtime_directives.py."""

import json
import time
from unittest.mock import patch

import pytest

from remy.core.agent_tools import _AuraCompat as Aura


@pytest.fixture
def directives_env(tmp_path):
    """Isolated environment for directives tests."""
    brain = Aura(str(tmp_path / "dir_brain"))

    # Clear global session directives between tests
    from remy.core.runtime_directives import _directives_lock, _session_directives

    with _directives_lock:
        _session_directives.clear()

    with patch("remy.core.runtime_directives.brain", brain):
        yield {"brain": brain}

    brain.close()


# ============== Unit Tests: Session Directives ==============


class TestSessionDirectives:
    def test_add_session_directive(self, directives_env):
        from remy.core.runtime_directives import add_session_directive, get_active_directives

        did = add_session_directive("Always use Ukrainian", session_id="s1")
        assert did == "session-0"

        dirs = get_active_directives("s1")
        assert len(dirs) == 1
        assert dirs[0]["text"] == "Always use Ukrainian"
        assert dirs[0]["type"] == "session"

    def test_multiple_directives(self, directives_env):
        from remy.core.runtime_directives import add_session_directive, get_active_directives

        add_session_directive("Directive A", session_id="s1")
        add_session_directive("Directive B", session_id="s1")

        dirs = get_active_directives("s1")
        assert len(dirs) == 2

    def test_session_isolation(self, directives_env):
        from remy.core.runtime_directives import add_session_directive, get_active_directives

        add_session_directive("For session 1", session_id="s1")
        add_session_directive("For session 2", session_id="s2")

        assert len(get_active_directives("s1")) == 1
        assert len(get_active_directives("s2")) == 1
        assert get_active_directives("s1")[0]["text"] == "For session 1"

    def test_ttl_expiration(self, directives_env):
        from remy.core.runtime_directives import (
            _directives_lock,
            _session_directives,
            add_session_directive,
            get_active_directives,
        )

        add_session_directive("Short-lived", session_id="s1", ttl_seconds=1)

        # Manually set created_at to the past
        with _directives_lock:
            _session_directives["s1"][0]["created_at"] = time.time() - 10

        dirs = get_active_directives("s1")
        assert len(dirs) == 0  # Expired

    def test_remove_session_directive(self, directives_env):
        from remy.core.runtime_directives import (
            add_session_directive,
            get_active_directives,
            remove_session_directive,
        )

        add_session_directive("To remove", session_id="s1")
        assert len(get_active_directives("s1")) == 1

        ok = remove_session_directive("s1", 0)
        assert ok is True
        assert len(get_active_directives("s1")) == 0

    def test_remove_invalid_index(self, directives_env):
        from remy.core.runtime_directives import remove_session_directive

        ok = remove_session_directive("nonexistent", 0)
        assert ok is False

    def test_clear_session_directives(self, directives_env):
        from remy.core.runtime_directives import (
            add_session_directive,
            clear_session_directives,
            get_active_directives,
        )

        add_session_directive("A", session_id="s1")
        add_session_directive("B", session_id="s1")

        count = clear_session_directives("s1")
        assert count == 2
        assert len(get_active_directives("s1")) == 0

    def test_cleanup_expired(self, directives_env):
        from remy.core.runtime_directives import (
            _directives_lock,
            _session_directives,
            add_session_directive,
            cleanup_expired,
            get_active_directives,
        )

        add_session_directive("Permanent", session_id="s1")
        add_session_directive("Expires", session_id="s1", ttl_seconds=1)

        # Expire the second one
        with _directives_lock:
            _session_directives["s1"][1]["created_at"] = time.time() - 10

        removed = cleanup_expired("s1")
        assert removed == 1
        assert len(get_active_directives("s1")) == 1
        assert get_active_directives("s1")[0]["text"] == "Permanent"


# ============== Unit Tests: Persistent Directives ==============


class TestPersistentDirectives:
    def test_add_persistent_directive(self, directives_env):
        from remy.core.runtime_directives import add_persistent_directive, get_active_directives

        record_id = add_persistent_directive("Never use web_search for personal questions")
        assert record_id is not None

        dirs = get_active_directives("any-session")
        persistent = [d for d in dirs if d["type"] == "persistent"]
        assert len(persistent) == 1
        assert "web_search" in persistent[0]["text"]

    def test_deactivate_persistent_directive(self, directives_env):
        from remy.core.runtime_directives import (
            add_persistent_directive,
            deactivate_persistent_directive,
            get_active_directives,
        )

        record_id = add_persistent_directive("To deactivate")
        assert record_id is not None

        ok = deactivate_persistent_directive(record_id)
        assert ok is True

        dirs = get_active_directives("any-session")
        persistent = [d for d in dirs if d["type"] == "persistent"]
        assert len(persistent) == 0

    def test_persistent_survives_sessions(self, directives_env):
        from remy.core.runtime_directives import add_persistent_directive, get_active_directives

        add_persistent_directive("Cross-session rule")

        # Check from different sessions
        dirs_s1 = get_active_directives("session-1")
        dirs_s2 = get_active_directives("session-2")

        persistent_s1 = [d for d in dirs_s1 if d["type"] == "persistent"]
        persistent_s2 = [d for d in dirs_s2 if d["type"] == "persistent"]

        assert len(persistent_s1) == 1
        assert len(persistent_s2) == 1


# ============== Unit Tests: Format for Instruction ==============


class TestFormatDirectives:
    def test_empty_returns_empty(self, directives_env):
        from remy.core.runtime_directives import format_directives_for_instruction

        result = format_directives_for_instruction("empty-session")
        assert result == ""

    def test_formats_with_directives(self, directives_env):
        from remy.core.runtime_directives import (
            add_session_directive,
            format_directives_for_instruction,
        )

        add_session_directive("Be concise", session_id="s1")
        add_session_directive("Use formal tone", session_id="s1")

        result = format_directives_for_instruction("s1")
        assert "RUNTIME DIRECTIVES" in result
        assert "Be concise" in result
        assert "Use formal tone" in result

    def test_user_directives_first(self, directives_env):
        from remy.core.runtime_directives import (
            add_session_directive,
            format_directives_for_instruction,
        )

        add_session_directive("Agent directive", session_id="s1", source="agent")
        add_session_directive("User directive", session_id="s1", source="user")

        result = format_directives_for_instruction("s1")
        # User directive should appear before agent directive
        user_pos = result.index("User directive")
        agent_pos = result.index("Agent directive")
        assert user_pos < agent_pos


# ============== Integration: Directive in System Instruction ==============


class TestDirectiveInSystemInstruction:
    def test_directives_injected_into_system_instruction(self, directives_env):
        from remy.core.runtime_directives import add_session_directive

        add_session_directive("SPECIAL_TEST_DIRECTIVE_XYZ", session_id="test-session")

        with patch("remy.core.agent.build_system_instruction", return_value="Base instruction."):
            from remy.core.agent import (
                _get_cached_system_instruction,
                _sys_instruction_cache,
                _sys_instruction_cache_lock,
            )

            # Clear cache first
            with _sys_instruction_cache_lock:
                _sys_instruction_cache.clear()

            result = _get_cached_system_instruction("test-session", "desktop")

            assert "Base instruction." in result
            assert "SPECIAL_TEST_DIRECTIVE_XYZ" in result
            assert "RUNTIME DIRECTIVES" in result


# ============== Tool execution tests ==============


class TestDirectiveTools:
    def test_add_directive_tool(self, directives_env):
        from remy.core.tool_dispatch import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            result_str = _execute_tool_inner(
                "add_runtime_directive",
                {"text": "Test directive via tool", "persistent": False},
                session_id="tool-session",
                channel="desktop",
            )

        result = json.loads(result_str)
        assert result["added"] is True
        assert result["type"] == "session"

    def test_add_persistent_directive_tool(self, directives_env):
        from remy.core.tool_dispatch import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            result_str = _execute_tool_inner(
                "add_runtime_directive",
                {"text": "Persistent test", "persistent": True},
                session_id="tool-session",
                channel="desktop",
            )

        result = json.loads(result_str)
        assert result["added"] is True
        assert result["type"] == "persistent"
        assert result["record_id"] is not None

    def test_add_directive_empty_text_fails(self, directives_env):
        from remy.core.tool_dispatch import _execute_tool_inner

        result_str = _execute_tool_inner(
            "add_runtime_directive",
            {"text": ""},
            session_id="s1",
            channel="desktop",
        )

        result = json.loads(result_str)
        assert "error" in result
