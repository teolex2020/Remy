"""Tests for Tool Health Visibility & Adaptive Routing (AUTON-11) — tool_routing.py."""

import time
from unittest.mock import patch

import pytest

# ============== Unit Tests: get_alternatives ==============


class TestGetAlternatives:
    def test_web_search_has_alternatives(self):
        from remy.core.tool_routing import get_alternatives

        alts = get_alternatives("web_search")
        assert len(alts) >= 1
        assert any(a["tool"] == "http_get" for a in alts)

    def test_browse_page_has_alternatives(self):
        from remy.core.tool_routing import get_alternatives

        alts = get_alternatives("browse_page")
        assert len(alts) >= 1
        assert any(a["tool"] == "http_get" for a in alts)

    def test_unknown_tool_no_alternatives(self):
        from remy.core.tool_routing import get_alternatives

        alts = get_alternatives("recall")
        assert alts == []


# ============== Unit Tests: get_best_alternative ==============


class TestGetBestAlternative:
    def test_returns_available_alternative(self):
        from remy.core.tool_routing import get_best_alternative

        alt = get_best_alternative("web_search")
        assert alt is not None
        assert "tool" in alt
        assert "hint" in alt

    def test_returns_none_for_unknown(self):
        from remy.core.tool_routing import get_best_alternative

        alt = get_best_alternative("some_unknown_tool")
        assert alt is None

    def test_skips_unavailable_alternatives(self):
        from remy.core.tool_health import tool_health
        from remy.core.tool_routing import get_best_alternative

        # Make http_get unavailable
        tool_health._circuit_open_until["http_get"] = time.time() + 600
        try:
            alt = get_best_alternative("web_search")
            # Should still find recall as alternative
            if alt:
                assert alt["tool"] != "http_get" or alt["tool"] == "recall"
        finally:
            tool_health._circuit_open_until.pop("http_get", None)


# ============== Unit Tests: get_tool_status_report ==============


class TestToolStatusReport:
    def setup_method(self):
        from remy.core.tool_health import tool_health

        tool_health._failures.clear()
        tool_health._circuit_open_until.clear()

    def test_returns_expected_keys(self):
        from remy.core.tool_routing import get_tool_status_report

        report = get_tool_status_report()
        assert "healthy" in report
        assert "degraded" in report
        assert "unavailable" in report
        assert "alternatives" in report

    def test_all_healthy_by_default(self):
        from remy.core.tool_routing import get_tool_status_report

        report = get_tool_status_report()
        assert len(report["healthy"]) > 0
        assert len(report["degraded"]) == 0
        assert len(report["unavailable"]) == 0

    def test_unavailable_tool_shown(self):
        from remy.core.tool_health import tool_health
        from remy.core.tool_routing import get_tool_status_report

        tool_health._circuit_open_until["web_search"] = time.time() + 600
        tool_health._failures["web_search"] = [time.time()] * 3
        try:
            report = get_tool_status_report()
            assert any(item["tool"] == "web_search" for item in report["unavailable"])
            assert "web_search" not in report["healthy"]
            # Should have alternative suggested
            assert "web_search" in report["alternatives"]
        finally:
            tool_health._circuit_open_until.pop("web_search", None)
            tool_health._failures.pop("web_search", None)

    def test_degraded_tool_shown(self):
        from remy.core.tool_health import tool_health
        from remy.core.tool_routing import get_tool_status_report

        # Add failures but don't open circuit
        tool_health._failures["http_get"] = [time.time()]
        try:
            report = get_tool_status_report()
            assert any(item["tool"] == "http_get" for item in report["degraded"])
        finally:
            tool_health._failures.pop("http_get", None)


# ============== Unit Tests: format_tool_health_for_prompt ==============


class TestFormatToolHealthForPrompt:
    def test_empty_when_all_healthy(self):
        from remy.core.tool_routing import format_tool_health_for_prompt

        report = {
            "healthy": ["web_search", "http_get"],
            "degraded": [],
            "unavailable": [],
            "alternatives": {},
        }
        with (
            patch(
                "remy.core.browser_failure_memory.get_browser_failure_report",
                return_value={"top_clusters": []},
            ),
            patch(
                "remy.core.browser_failure_memory.get_browser_success_report",
                return_value={"top_playbooks": []},
            ),
        ):
            assert format_tool_health_for_prompt(report) == ""

    def test_shows_unavailable_with_alternative(self):
        from remy.core.tool_routing import format_tool_health_for_prompt

        report = {
            "healthy": ["http_get"],
            "degraded": [],
            "unavailable": [{"tool": "web_search", "status": "UNAVAILABLE (500s)"}],
            "alternatives": {"web_search": {"tool": "http_get", "hint": "Use http_get"}},
        }
        text = format_tool_health_for_prompt(report)
        assert "UNAVAILABLE" in text
        assert "web_search" in text
        assert "http_get" in text

    def test_shows_degraded(self):
        from remy.core.tool_routing import format_tool_health_for_prompt

        report = {
            "healthy": [],
            "degraded": [{"tool": "http_get", "status": "degraded (2 recent failures)"}],
            "unavailable": [],
            "alternatives": {},
        }
        text = format_tool_health_for_prompt(report)
        assert "DEGRADED" in text
        assert "http_get" in text


# ============== Unit Tests: test_tools_on_startup ==============


class TestStartupCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        from unittest.mock import patch

        from remy.core.tool_routing import test_tools_on_startup

        with patch(
            "remy.core.tool_dispatch.execute_tool", return_value='{"datetime": "2024-01-01"}'
        ):
            results = await test_tools_on_startup()
        assert isinstance(results, dict)
        assert "get_current_datetime" in results


# ============== Integration: tool_status in tool_dispatch ==============


class TestToolStatusDispatch:
    def test_tool_status_returns_json(self):
        import json

        from remy.core.tool_dispatch import execute_tool

        result = execute_tool("tool_status", {})
        parsed = json.loads(result)
        assert "healthy" in parsed
        assert "degraded" in parsed
        assert "unavailable" in parsed

    def test_direct_tool_dispatch_blocks_refuted_action(self, monkeypatch):
        import json

        from remy.core.tool_dispatch import execute_tool

        class FakeBrain:
            def policy_hint(self, situation, action, namespace=None):
                assert situation == "tool_call:autonomous:sess-1"
                assert action == "tool:tool_status"
                assert namespace == "remy-tools"
                return {
                    "hint": "avoid",
                    "reason": "prior direct tool failure",
                    "verdict": "refutes",
                    "refutes": 1,
                    "supports": 0,
                    "should_block": True,
                }

        monkeypatch.setattr("remy.core.brain_tools.brain", FakeBrain())

        result = execute_tool("tool_status", {}, session_id="sess-1", channel="autonomous")
        parsed = json.loads(result)

        assert "Blocked by consequence memory" in parsed["error"]
        assert parsed["consequence_gate"]["blocked"] is True
        assert parsed["consequence_gate"]["policy_hint"]["hint"] == "avoid"

    def test_direct_tool_dispatch_records_success_consequence(self, monkeypatch):
        import json

        from remy.core.tool_dispatch import execute_tool

        captures = []

        class FakeBrain:
            def policy_hint(self, situation, action, namespace=None):
                return {
                    "hint": "verify_first",
                    "verdict": "inconclusive",
                    "refutes": 0,
                    "supports": 0,
                    "should_block": False,
                }

            def capture_consequence(self, **kwargs):
                captures.append(kwargs)

        monkeypatch.setattr("remy.core.brain_tools.brain", FakeBrain())
        monkeypatch.setattr(
            "remy.core.brain_tools._execute_tool_locked",
            lambda name, args, session_id=None, channel=None: json.dumps({"ok": True}),
        )

        result = execute_tool("tool_status", {}, session_id="sess-2", channel="desktop")
        parsed = json.loads(result)

        assert parsed == {"ok": True}
        assert captures
        assert captures[0]["situation"] == "tool_call:desktop:sess-2"
        assert captures[0]["action"] == "tool:tool_status"
        assert captures[0]["consequence"] == "SUPPORTS"
        assert captures[0]["namespace"] == "remy-tools"
        assert "direct-tool-dispatch" in captures[0]["scope"]
