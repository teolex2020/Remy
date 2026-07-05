"""Tests for RM-3: Autonomous Research Cycles — research-aware goals, decision prompt integration."""

import json
from unittest.mock import patch, MagicMock

import pytest
from aura import Aura as CognitiveMemory, Level


# ============== Research-Aware Goal Detection ==============

class TestIsResearchGoal:

    def _make_loop(self):
        from remy.core.autonomy import AutonomousLoop
        loop = AutonomousLoop.__new__(AutonomousLoop)
        return loop

    def test_detects_research_keyword(self):
        """Goal with 'research' keyword is detected as research."""
        loop = self._make_loop()
        assert loop._is_research_goal("Research the benefits of meditation") is True

    def test_detects_investigate_keyword(self):
        """Goal with 'investigate' keyword is detected."""
        loop = self._make_loop()
        assert loop._is_research_goal("Investigate sleep patterns for better health") is True

    def test_detects_find_out_keyword(self):
        """Goal with 'find out' keyword is detected."""
        loop = self._make_loop()
        assert loop._is_research_goal("Find out about vitamin D dosage recommendations") is True

    def test_detects_ukrainian_keyword(self):
        """Goal with Ukrainian keyword 'дослідж' is detected."""
        loop = self._make_loop()
        assert loop._is_research_goal("Дослідж вплив сну на здоров'я") is True

    def test_non_research_goal_not_detected(self):
        """Regular goal is not detected as research."""
        loop = self._make_loop()
        assert loop._is_research_goal("Organize stored knowledge") is False

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        loop = self._make_loop()
        assert loop._is_research_goal("RESEARCH optimal exercise routines") is True


# ============== _ensure_research_project ==============

class TestEnsureResearchProject:

    def test_creates_project_for_research_goal(self, tmp_path):
        """Research goal triggers project creation."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            from remy.core.autonomy import AutonomousLoop, ResourceBudget, create_goal

            # Create a research goal
            goal_rec_id = create_goal("Research the health benefits of green tea", priority="medium")
            goals = [{"record_id": goal_rec_id, "description": "Research the health benefits of green tea", "goal_id": "g1"}]

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.budget = ResourceBudget(daily_limit=100000, hourly_limit=20000, session_limit=500000)

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("skip LLM")

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                loop._ensure_research_project(goals[0])

            # Check goal metadata was updated
            rec = b.get(goal_rec_id)
            meta = rec.metadata or {}
            assert meta.get("is_research") is True
            assert meta.get("research_project_id") is not None
        b.close()

    def test_skips_non_research_goal(self, tmp_path):
        """Non-research goal is tagged is_research=False."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b):
            from remy.core.autonomy import AutonomousLoop, ResourceBudget, create_goal

            goal_rec_id = create_goal("Organize stored knowledge", priority="medium")
            goal = {"record_id": goal_rec_id, "description": "Organize stored knowledge", "goal_id": "g2"}

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.budget = ResourceBudget(daily_limit=100000, hourly_limit=20000, session_limit=500000)

            loop._ensure_research_project(goal)

            rec = b.get(goal_rec_id)
            assert rec.metadata.get("is_research") is False
            assert rec.metadata.get("research_project_id") is None
        b.close()

    def test_skips_if_project_already_exists(self, tmp_path):
        """Does not create duplicate project."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b):
            from remy.core.autonomy import AutonomousLoop, ResourceBudget, create_goal

            goal_rec_id = create_goal("Research sleep quality", priority="high")

            # Pre-tag as already having project
            rec = b.get(goal_rec_id)
            meta = dict(rec.metadata)
            meta["research_project_id"] = "rp-existing"
            meta["is_research"] = True
            b.update(goal_rec_id, metadata=meta)

            goal = {"record_id": goal_rec_id, "description": "Research sleep quality", "goal_id": "g3"}

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.budget = ResourceBudget(daily_limit=100000, hourly_limit=20000, session_limit=500000)

            # Should not attempt to create new project (no LLM call needed)
            loop._ensure_research_project(goal)

            rec = b.get(goal_rec_id)
            assert rec.metadata["research_project_id"] == "rp-existing"
        b.close()

    def test_skips_if_budget_insufficient(self, tmp_path):
        """Does not create project if budget is too low."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b):
            from remy.core.autonomy import AutonomousLoop, ResourceBudget, create_goal

            goal_rec_id = create_goal("Research meditation benefits", priority="low")
            goal = {"record_id": goal_rec_id, "description": "Research meditation benefits", "goal_id": "g4"}

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.budget = ResourceBudget(daily_limit=100, hourly_limit=50, session_limit=100)
            # Exhaust budget
            loop.budget.tokens_today = 99
            loop.budget.tokens_this_hour = 49

            loop._ensure_research_project(goal)

            rec = b.get(goal_rec_id)
            # Project should NOT have been created
            assert rec.metadata.get("research_project_id") is None
        b.close()


# ============== Decision Prompt with Research Context ==============

class TestDecisionPromptResearch:

    def test_includes_active_research(self, tmp_path):
        """Decision prompt includes ACTIVE RESEARCH section."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        # Create a research project
        b.store(
            content="Research Project: vitamin D",
            level=Level.DOMAIN,
            tags=["research-project"],
            metadata={
                "project_id": "rp-vitd",
                "topic": "vitamin D",
                "status": "researching",
                "depth": "standard",
                "query_plan": ["q1", "q2", "q3", "q4"],
                "queries_done": 2,
                "findings_count": 3,
            },
        )

        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[{"description": "Research vitamin D", "priority": "high", "attempts": 1}],
                past_outcomes="",
                budget={"tokens_today": 500, "daily_limit": 10000,
                        "tokens_this_hour": 100, "hourly_limit": 2000},
            )

            assert "ACTIVE RESEARCH:" in prompt
            assert "vitamin D" in prompt
            assert "2/4 queries done" in prompt
            assert "3 findings" in prompt
            assert "rp-vitd" in prompt
        b.close()

    def test_no_research_section_when_empty(self, tmp_path):
        """No ACTIVE RESEARCH section when no projects exist."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[], past_outcomes="",
                budget={"tokens_today": 0, "daily_limit": 10000,
                        "tokens_this_hour": 0, "hourly_limit": 2000},
            )

            assert "ACTIVE RESEARCH:" not in prompt
        b.close()

    def test_prompt_includes_research_tools_cost(self, tmp_path):
        """Prompt includes cost annotations for research tools."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[], past_outcomes="",
                budget={"tokens_today": 0, "daily_limit": 10000,
                        "tokens_this_hour": 0, "hourly_limit": 2000},
            )

            assert "start_research (~800)" in prompt
            assert "add_research_finding (~30)" in prompt
            assert "complete_research (~500)" in prompt
        b.close()

    def test_prompt_includes_research_instructions(self, tmp_path):
        """Prompt instructs to use research orchestrator and continue existing research."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[], past_outcomes="",
                budget={"tokens_today": 0, "daily_limit": 10000,
                        "tokens_this_hour": 0, "hourly_limit": 2000},
            )

            assert "start_research" in prompt
            assert "add_research_finding" in prompt
            assert "complete_research" in prompt
            assert "ACTIVE RESEARCH exists" in prompt
        b.close()
