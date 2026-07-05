"""Tests for Behavioral Rules Engine (AUTON-1) — autonomy_rules.py."""

import json
from unittest.mock import patch

import pytest

from remy.core.agent_tools import _AuraCompat as Aura


@pytest.fixture
def rules_env(tmp_path):
    """Isolated environment for rules tests."""
    brain = Aura(str(tmp_path / "rules_brain"))

    with patch("remy.core.autonomy_rules.brain", brain):
        yield {"brain": brain}

    brain.close()


# ============== Unit Tests: load_active_rules ==============


class TestLoadActiveRules:
    def test_empty_brain_returns_empty(self, rules_env):
        from remy.core.autonomy_rules import load_active_rules

        rules = load_active_rules()
        assert rules == []

    def test_loads_stored_rules(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, load_active_rules

        _create_rule(
            condition_type="goal_keyword",
            condition_value="research",
            action="Decompose research goals immediately",
            confidence=0.8,
        )

        rules = load_active_rules()
        assert len(rules) == 1
        assert rules[0]["condition_type"] == "goal_keyword"
        assert rules[0]["condition_value"] == "research"
        assert rules[0]["confidence"] == 0.8

    def test_skips_archived_rules(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, load_active_rules

        # Create a rule then archive it
        rule_id = _create_rule(
            condition_type="goal_keyword",
            condition_value="write",
            action="Be careful with write goals",
        )

        brain = rules_env["brain"]
        record = brain.get(rule_id)
        meta = getattr(record, "metadata", None) or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        meta["archived"] = True
        brain.update(rule_id, metadata=meta)

        rules = load_active_rules()
        assert len(rules) == 0


# ============== Unit Tests: match_rules ==============


class TestMatchRules:
    def test_no_rules_returns_empty(self):
        from remy.core.autonomy_rules import match_rules

        result = match_rules("Research AI safety", rules=[])
        assert result == []

    def test_goal_keyword_match(self):
        from remy.core.autonomy_rules import match_rules

        rules = [
            {
                "record_id": "abc123",
                "content": "test",
                "condition_type": "goal_keyword",
                "condition_value": "research",
                "action": "Decompose first",
                "confidence": 0.8,
                "applied_count": 0,
                "success_after_apply": 0,
                "created_at": "",
            }
        ]

        matched = match_rules("Research AI safety", rules=rules)
        assert len(matched) == 1
        assert matched[0]["action"] == "Decompose first"

    def test_goal_keyword_no_match(self):
        from remy.core.autonomy_rules import match_rules

        rules = [
            {
                "record_id": "abc123",
                "content": "test",
                "condition_type": "goal_keyword",
                "condition_value": "research",
                "action": "Decompose first",
                "confidence": 0.8,
                "applied_count": 0,
                "success_after_apply": 0,
                "created_at": "",
            }
        ]

        matched = match_rules("Organize my files", rules=rules)
        assert len(matched) == 0

    def test_tool_failure_always_matches(self):
        from remy.core.autonomy_rules import match_rules

        rules = [
            {
                "record_id": "def456",
                "content": "test",
                "condition_type": "tool_failure",
                "condition_value": "web_search",
                "action": "Try http_get instead of web_search",
                "confidence": 0.7,
                "applied_count": 2,
                "success_after_apply": 1,
                "created_at": "",
            }
        ]

        matched = match_rules("Any goal at all", rules=rules)
        assert len(matched) == 1

    def test_critique_pattern_always_matches(self):
        from remy.core.autonomy_rules import match_rules

        rules = [
            {
                "record_id": "ghi789",
                "content": "test",
                "condition_type": "critique_pattern",
                "condition_value": "hallucinated tool call",
                "action": "Verify tool was actually called before claiming success",
                "confidence": 0.9,
                "applied_count": 5,
                "success_after_apply": 3,
                "created_at": "",
            }
        ]

        matched = match_rules("Write a report", rules=rules)
        assert len(matched) == 1

    def test_sorted_by_confidence(self):
        from remy.core.autonomy_rules import match_rules

        rules = [
            {
                "record_id": "a",
                "content": "t",
                "condition_type": "tool_failure",
                "condition_value": "",
                "action": "Low confidence",
                "confidence": 0.3,
                "applied_count": 0,
                "success_after_apply": 0,
                "created_at": "",
            },
            {
                "record_id": "b",
                "content": "t",
                "condition_type": "tool_failure",
                "condition_value": "",
                "action": "High confidence",
                "confidence": 0.9,
                "applied_count": 0,
                "success_after_apply": 0,
                "created_at": "",
            },
        ]

        matched = match_rules("anything", rules=rules)
        assert matched[0]["confidence"] == 0.9
        assert matched[1]["confidence"] == 0.3

    def test_capped_at_10(self):
        from remy.core.autonomy_rules import match_rules

        rules = [
            {
                "record_id": f"r{i}",
                "content": "t",
                "condition_type": "critique_pattern",
                "condition_value": "",
                "action": f"rule-{i}",
                "confidence": 0.5,
                "applied_count": 0,
                "success_after_apply": 0,
                "created_at": "",
            }
            for i in range(15)
        ]

        matched = match_rules("anything", rules=rules)
        assert len(matched) == 10


# ============== Unit Tests: format_rules_for_prompt ==============


class TestFormatRulesForPrompt:
    def test_empty_rules(self):
        from remy.core.autonomy_rules import format_rules_for_prompt

        assert format_rules_for_prompt([]) == ""

    def test_formats_rules_text(self):
        from remy.core.autonomy_rules import format_rules_for_prompt

        rules = [
            {
                "condition_type": "goal_keyword",
                "action": "Decompose research goals immediately",
            }
        ]

        text = format_rules_for_prompt(rules)
        assert "BEHAVIORAL RULES" in text
        assert "Decompose research goals immediately" in text


# ============== Unit Tests: record_rule_applied ==============


class TestRecordRuleApplied:
    def test_updates_applied_count(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, load_active_rules, record_rule_applied

        rule_id = _create_rule(
            condition_type="goal_keyword",
            condition_value="test",
            action="test action",
        )

        rule = {"record_id": rule_id}
        record_rule_applied(rule, success=True)

        rules = load_active_rules()
        assert len(rules) == 1
        assert rules[0]["applied_count"] == 1
        assert rules[0]["success_after_apply"] == 1

    def test_failure_increments_applied_not_success(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, load_active_rules, record_rule_applied

        rule_id = _create_rule(
            condition_type="goal_keyword",
            condition_value="test",
            action="test action",
        )

        record_rule_applied({"record_id": rule_id}, success=False)

        rules = load_active_rules()
        assert rules[0]["applied_count"] == 1
        assert rules[0]["success_after_apply"] == 0


# ============== Unit Tests: _create_rule ==============


class TestCreateRule:
    def test_creates_rule_record(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, load_active_rules

        rule_id = _create_rule(
            condition_type="critique_pattern",
            condition_value="hallucination",
            action="Verify before claiming",
            confidence=0.85,
            source_info="5 critique occurrences",
        )

        assert rule_id is not None

        rules = load_active_rules()
        assert len(rules) == 1
        assert rules[0]["condition_type"] == "critique_pattern"
        assert rules[0]["confidence"] == 0.85


# ============== Unit Tests: _rule_exists ==============


class TestRuleExists:
    def test_returns_true_for_existing(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, _rule_exists

        _create_rule(
            condition_type="goal_keyword",
            condition_value="research",
            action="decompose",
        )

        assert _rule_exists("goal_keyword", "research") is True

    def test_returns_false_for_missing(self, rules_env):
        from remy.core.autonomy_rules import _rule_exists

        assert _rule_exists("goal_keyword", "nonexistent") is False

    def test_case_insensitive(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, _rule_exists

        _create_rule(
            condition_type="goal_keyword",
            condition_value="Research",
            action="decompose",
        )

        assert _rule_exists("goal_keyword", "research") is True


# ============== Unit Tests: check_and_generate_rules ==============


class TestCheckAndGenerateRules:
    def test_generates_rule_from_repeated_outcomes(self, rules_env):
        from remy.core.autonomy_rules import check_and_generate_rules, load_active_rules

        # Simulate 4 failed "research" outcomes
        outcomes = [
            {"success": False, "goal_type": "research", "content": "failed research"}
            for _ in range(4)
        ]

        new_ids = check_and_generate_rules(
            recent_critiques=[],
            recent_outcomes=outcomes,
        )

        assert len(new_ids) >= 1

        rules = load_active_rules()
        assert any(r["condition_type"] == "goal_keyword" for r in rules)

    def test_no_rule_for_few_failures(self, rules_env):
        from remy.core.autonomy_rules import check_and_generate_rules

        # Only 2 failures — below MIN_FAILURES_FOR_RULE
        outcomes = [
            {"success": False, "goal_type": "write", "content": "failed write"} for _ in range(2)
        ]

        new_ids = check_and_generate_rules(
            recent_critiques=[],
            recent_outcomes=outcomes,
        )

        assert len(new_ids) == 0

    def test_no_duplicate_rules(self, rules_env):
        from remy.core.autonomy_rules import check_and_generate_rules, load_active_rules

        outcomes = [
            {"success": False, "goal_type": "research", "content": "failed"} for _ in range(4)
        ]

        # Generate once
        check_and_generate_rules(recent_critiques=[], recent_outcomes=outcomes)
        count_after_first = len(load_active_rules())

        # Generate again — should not create duplicates
        check_and_generate_rules(recent_critiques=[], recent_outcomes=outcomes)
        count_after_second = len(load_active_rules())

        assert count_after_first == count_after_second


# ============== Unit Tests: decay_stale_rules ==============


class TestDecayStaleRules:
    def test_archives_old_unused_rule(self, rules_env):
        from datetime import datetime, timedelta

        from remy.core.autonomy_rules import _create_rule, decay_stale_rules, load_active_rules

        rule_id = _create_rule(
            condition_type="goal_keyword",
            condition_value="old_test",
            action="old rule",
        )

        # Manually set created_at to 10 days ago
        brain = rules_env["brain"]
        record = brain.get(rule_id)
        meta = getattr(record, "metadata", None) or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        meta["created_at"] = (datetime.now() - timedelta(days=10)).isoformat()
        brain.update(rule_id, metadata=meta)

        archived = decay_stale_rules()
        assert archived == 1

        rules = load_active_rules()
        assert len(rules) == 0

    def test_keeps_recent_rule(self, rules_env):
        from remy.core.autonomy_rules import _create_rule, decay_stale_rules, load_active_rules

        _create_rule(
            condition_type="goal_keyword",
            condition_value="recent_test",
            action="fresh rule",
        )

        archived = decay_stale_rules()
        assert archived == 0

        rules = load_active_rules()
        assert len(rules) == 1

    def test_keeps_effective_old_rule(self, rules_env):
        from datetime import datetime, timedelta

        from remy.core.autonomy_rules import (
            _create_rule,
            decay_stale_rules,
            load_active_rules,
            record_rule_applied,
        )

        rule_id = _create_rule(
            condition_type="goal_keyword",
            condition_value="effective_test",
            action="effective rule",
        )

        # Make it old
        brain = rules_env["brain"]
        record = brain.get(rule_id)
        meta = getattr(record, "metadata", None) or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        meta["created_at"] = (datetime.now() - timedelta(days=10)).isoformat()
        brain.update(rule_id, metadata=meta)

        # Apply it 5 times with high success
        for _ in range(5):
            record_rule_applied({"record_id": rule_id}, success=True)

        archived = decay_stale_rules()
        assert archived == 0  # Effective rule should be kept

        rules = load_active_rules()
        assert len(rules) == 1


# ============== Integration: rules in decision prompt ==============


class TestRulesInPrompt:
    def test_rules_appear_in_decision_prompt(self, rules_env):
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = True
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = rules_env["brain"]

            with patch("remy.core.autonomy.brain", rules_env["brain"]):
                from remy.core.autonomy import AutonomousLoop

                loop = AutonomousLoop()

                rules = [
                    {
                        "condition_type": "goal_keyword",
                        "action": "Always decompose research goals",
                        "confidence": 0.8,
                    }
                ]

                prompt = loop._build_decision_prompt(
                    goals=[
                        {
                            "goal_id": "g1",
                            "record_id": "r1",
                            "description": "Research something",
                            "priority": "high",
                            "attempts": 0,
                        }
                    ],
                    past_outcomes="",
                    budget={
                        "tokens_today": 0,
                        "daily_limit": 100000,
                        "tokens_this_hour": 0,
                        "hourly_limit": 20000,
                    },
                    behavioral_rules=rules,
                )

                assert "BEHAVIORAL RULES" in prompt
                assert "Always decompose research goals" in prompt
