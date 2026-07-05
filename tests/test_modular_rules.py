"""Tests for Modular System Prompt Rules (v2.3).

Validates that:
1. Rule blocks are only injected for channels that need them
2. Voice/proactive channels get a leaner prompt (fewer tokens)
3. All channels still get core safety rules (financial, anti-hallucination)
4. Rule constants are well-formed strings
"""

from unittest.mock import MagicMock, patch

# ============== Helpers ==============


def _build_for_channel(channel: str, browser_enabled: bool = False) -> str:
    """Build system instruction for a given channel."""
    from remy.core.brain_tools import build_system_instruction

    mock_settings = MagicMock()
    mock_settings.BROWSER_ENABLED = browser_enabled

    with patch("remy.core.system_instruction._get_bt") as mock_bt:
        mock_bt.return_value.brain = MagicMock()
        mock_bt.return_value.brain.recall.return_value = ""
        mock_bt.return_value.brain.tier_stats.side_effect = Exception("skip")
        mock_bt.return_value.brain.search.return_value = []
        mock_bt.return_value.brain_lock = MagicMock()
        mock_bt.return_value.settings = mock_settings
        return build_system_instruction(channel=channel)


# ============== Tests: Channel-specific injection ==============


class TestModularRuleInjection:
    def test_voice_excludes_research_rules(self):
        """Voice channel should NOT get Research Orchestrator instructions.

        Note: we check for the rule block marker ("Research Orchestrator") and
        for *usage* of start_research (e.g. "call start_research", "start_research("),
        not the bare token — tool names can appear inside auto-generated tool
        listings that every channel shares, which is not a rule-injection leak.
        """
        instruction = _build_for_channel("voice")
        assert "Research Orchestrator" not in instruction
        assert "start_research(" not in instruction
        assert "call start_research" not in instruction
        assert "Use 'start_research'" not in instruction

    def test_voice_excludes_planning_rules(self):
        """Voice channel should NOT get failure-aware planning."""
        instruction = _build_for_channel("voice")
        assert "FAILURE-AWARE PLANNING" not in instruction
        assert "NEGATIVE KNOWLEDGE" not in instruction

    def test_voice_excludes_execution_guard(self):
        """Voice channel without browser should NOT get memory-gated execution."""
        instruction = _build_for_channel("voice", browser_enabled=False)
        assert "MEMORY-GATED EXECUTION" not in instruction

    def test_voice_excludes_delegation(self):
        """Voice channel should NOT get multi-agent delegation rules."""
        instruction = _build_for_channel("voice")
        assert "Multi-Agent Delegation" not in instruction
        assert "delegate_task" not in instruction

    def test_proactive_excludes_research(self):
        """Proactive channel should NOT get research rules."""
        instruction = _build_for_channel("proactive")
        assert "Research Orchestrator" not in instruction

    def test_proactive_excludes_delegation(self):
        """Proactive channel should NOT get delegation rules."""
        instruction = _build_for_channel("proactive")
        assert "Multi-Agent Delegation" not in instruction

    def test_desktop_includes_all(self):
        """Desktop channel should include all modular rules."""
        instruction = _build_for_channel("desktop", browser_enabled=True)
        assert "Research Orchestrator" in instruction
        assert "FAILURE-AWARE PLANNING" in instruction
        assert "NEGATIVE KNOWLEDGE" in instruction
        assert "MEMORY-GATED EXECUTION" in instruction
        assert "Multi-Agent Delegation" in instruction

    def test_telegram_includes_research_and_planning(self):
        """Telegram should get research + planning rules."""
        instruction = _build_for_channel("telegram")
        assert "Research Orchestrator" in instruction
        assert "FAILURE-AWARE PLANNING" in instruction

    def test_autonomous_includes_planning_and_execution(self):
        """Autonomous should get planning + execution guard rules."""
        instruction = _build_for_channel("autonomous")
        assert "FAILURE-AWARE PLANNING" in instruction
        assert "MEMORY-GATED EXECUTION" in instruction

    def test_autonomous_includes_research(self):
        """Autonomous should get research rules."""
        instruction = _build_for_channel("autonomous")
        assert "Research Orchestrator" in instruction

    def test_browser_enables_execution_guard(self):
        """Browser-enabled channels should get MEMORY-GATED EXECUTION."""
        instruction = _build_for_channel("telegram", browser_enabled=True)
        assert "MEMORY-GATED EXECUTION" in instruction

    def test_no_browser_no_execution_guard_for_telegram(self):
        """Telegram without browser should NOT get execution guard."""
        instruction = _build_for_channel("telegram", browser_enabled=False)
        assert "MEMORY-GATED EXECUTION" not in instruction


# ============== Tests: Core safety rules always present ==============


class TestCoreSafetyRules:
    def test_financial_safety_in_voice(self):
        """FINANCIAL DATA SAFETY must be present in all channels."""
        instruction = _build_for_channel("voice")
        assert "FINANCIAL DATA SAFETY" in instruction

    def test_anti_hallucination_in_voice(self):
        """ANTI-HALLUCINATION must be present in all channels."""
        instruction = _build_for_channel("voice")
        assert "ANTI-HALLUCINATION" in instruction

    def test_memory_first_in_autonomous(self):
        """MEMORY-FIRST rule must be in autonomous."""
        instruction = _build_for_channel("autonomous")
        assert "MEMORY-FIRST" in instruction

    def test_real_data_flag_in_proactive(self):
        """REAL DATA FLAG must be in proactive."""
        instruction = _build_for_channel("proactive")
        assert "REAL DATA FLAG" in instruction

    def test_tool_budget_in_all(self):
        """TOOL BUDGET PER TURN must be in all channels."""
        for ch in ("voice", "telegram", "desktop", "autonomous", "proactive"):
            instruction = _build_for_channel(ch)
            assert "TOOL BUDGET PER TURN" in instruction, f"Missing in {ch}"

    def test_stop_on_failures_in_all(self):
        """STOP ON REPEATED FAILURES must be in all channels."""
        for ch in ("voice", "telegram", "desktop", "autonomous", "proactive"):
            instruction = _build_for_channel(ch)
            assert "STOP ON REPEATED FAILURES" in instruction, f"Missing in {ch}"


# ============== Tests: Token savings measurement ==============


class TestTokenSavings:
    def test_voice_shorter_than_desktop(self):
        """Voice instruction should be significantly shorter than desktop."""
        voice = _build_for_channel("voice")
        desktop = _build_for_channel("desktop", browser_enabled=True)
        savings = len(desktop) - len(voice)
        # At least 1000 chars saved (research + planning + execution + delegation)
        assert savings > 1000, f"Only {savings} chars saved"

    def test_proactive_shorter_than_autonomous(self):
        """Proactive should be shorter than autonomous."""
        proactive = _build_for_channel("proactive")
        autonomous = _build_for_channel("autonomous")
        assert len(proactive) < len(autonomous)

    def test_voice_has_interactive_rules(self):
        """Voice should still get INTERACTIVE rules."""
        instruction = _build_for_channel("voice")
        assert "INTERACTIVE SESSION BEHAVIOR" in instruction

    def test_autonomous_no_interactive_rules(self):
        """Autonomous should NOT get interactive rules."""
        instruction = _build_for_channel("autonomous")
        assert "INTERACTIVE SESSION BEHAVIOR" not in instruction


# ============== Tests: Rule constants well-formed ==============


class TestRuleConstants:
    def test_all_rules_are_strings(self):
        from remy.core.system_instruction import (
            _BROWSER_RULES,
            _DELEGATION_RULES,
            _EXECUTION_GUARD_RULES,
            _INTERACTIVE_RULES,
            _PLANNING_RULES,
            _RESEARCH_RULES,
        )

        for name, rule in [
            ("INTERACTIVE", _INTERACTIVE_RULES),
            ("BROWSER", _BROWSER_RULES),
            ("RESEARCH", _RESEARCH_RULES),
            ("PLANNING", _PLANNING_RULES),
            ("EXECUTION_GUARD", _EXECUTION_GUARD_RULES),
            ("DELEGATION", _DELEGATION_RULES),
        ]:
            assert isinstance(rule, str), f"{name} is not a string"
            assert len(rule) > 50, f"{name} is suspiciously short"
            assert rule.endswith("\n"), f"{name} should end with newline"

    def test_no_duplicate_rules_between_base_and_modules(self):
        """Extracted rules should NOT also appear in the base string."""
        instruction = _build_for_channel("voice", browser_enabled=False)
        # These should NOT be in voice (extracted to modules)
        assert "FAILURE-AWARE PLANNING" not in instruction
        assert "MEMORY-GATED EXECUTION" not in instruction
        assert "Research Orchestrator" not in instruction
        assert "delegate_task" not in instruction
