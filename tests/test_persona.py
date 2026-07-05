"""Tests for AIEOS-style Agent Persona — machine-readable/writable persona system."""

import json
from unittest.mock import MagicMock, patch

import pytest

from remy.core.agent_tools import Level
from remy.core.agent_tools import _AuraCompat as Aura


@pytest.fixture
def mock_brain(tmp_path):
    """Real CognitiveMemory for integration testing."""
    b = Aura(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def patched_brain(mock_brain, tmp_path):
    """Patch brain_tools.brain and registry for tool execution."""
    with (
        patch("remy.core.brain_tools.brain", mock_brain),
        patch("remy.core.brain_tools._registry", None),
        patch("remy.core.tool_registry.settings") as ms,
        patch("remy.core.brain_tools.tool_health") as mh,
    ):
        ms.SANDBOX_DIR = tmp_path / "sandbox"
        ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
        mh.is_available.return_value = True
        yield mock_brain


# ============== Schema & Defaults ==============


class TestPersonaDefaults:
    def test_default_persona_when_no_record(self, mock_brain):
        """Empty brain returns _DEFAULT_PERSONA."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _DEFAULT_PERSONA, _get_agent_persona

            persona = _get_agent_persona()
            assert persona["name"] == _DEFAULT_PERSONA["name"]
            assert persona["role"] == _DEFAULT_PERSONA["role"]
            assert persona["tone"] == _DEFAULT_PERSONA["tone"]
            assert persona["traits"] == _DEFAULT_PERSONA["traits"]

    def test_default_persona_exception_safe(self):
        """Exception during brain.search returns defaults gracefully."""
        mock_brain = MagicMock()
        mock_brain.search.side_effect = Exception("DB error")
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _get_agent_persona

            persona = _get_agent_persona()
            assert persona["name"] == "Remy"


# ============== Persona to Instruction ==============


class TestPersonaToInstruction:
    def test_default_produces_remy_text(self):
        """Default persona produces instruction mentioning 'Remy'."""
        from remy.core.brain_tools import _DEFAULT_PERSONA, _persona_to_instruction

        text = _persona_to_instruction(_DEFAULT_PERSONA)
        assert "You are Remy" in text
        assert "warm and knowledgeable" in text

    def test_custom_name_and_tone(self):
        """Custom name and tone render correctly."""
        from remy.core.brain_tools import _persona_to_instruction

        persona = {
            "name": "Atlas",
            "role": "strict medical advisor",
            "scope": "health only",
            "tone": "formal and precise",
            "motivations": "provide accurate medical guidance",
            "catchphrases": ["Stay healthy!", "Prevention is key"],
            "avoid": ["slang", "jokes"],
            "traits": {"humor": 0.1, "formality": 0.9, "conciseness": 0.9},
        }
        text = _persona_to_instruction(persona)
        assert "You are Atlas" in text
        assert "strict medical advisor" in text
        assert "formal and precise" in text
        assert "Stay healthy!" in text
        assert "slang" in text
        assert "maintain a professional tone" in text
        assert "be especially concise" in text

    def test_empty_lists_no_extra_lines(self):
        """Empty catchphrases and avoid lists produce no extra text."""
        from remy.core.brain_tools import _persona_to_instruction

        persona = {
            "name": "Remy",
            "role": "assistant",
            "scope": "all",
            "tone": "warm",
            "motivations": "help",
            "catchphrases": [],
            "avoid": [],
            "traits": {"humor": 0.3, "formality": 0.3, "conciseness": 0.5},
        }
        text = _persona_to_instruction(persona)
        assert "Signature phrases" not in text
        assert "Avoid:" not in text

    def test_humor_trait_hint(self):
        """High humor trait adds humor hint."""
        from remy.core.brain_tools import _persona_to_instruction

        persona = {
            "name": "Fun",
            "role": "comedian",
            "scope": "all",
            "tone": "playful",
            "motivations": "entertain",
            "catchphrases": [],
            "avoid": [],
            "traits": {"humor": 0.8, "formality": 0.2, "conciseness": 0.5},
        }
        text = _persona_to_instruction(persona)
        assert "light humor" in text


# ============== read_persona Tool ==============


class TestReadPersonaTool:
    def test_read_persona_returns_json(self, patched_brain):
        """read_persona returns valid JSON with persona fields."""
        from remy.core.brain_tools import _execute_tool_inner

        result = json.loads(_execute_tool_inner("read_persona", {}))
        assert result["name"] == "Remy"
        assert "traits" in result
        assert result["traits"]["warmth"] == 0.8

    def test_read_persona_after_update(self, patched_brain):
        """read_persona returns updated values after update_persona."""
        from remy.core.brain_tools import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            _execute_tool_inner("update_persona", {"name": "Atlas"}, channel="desktop")
        result = json.loads(_execute_tool_inner("read_persona", {}))
        assert result["name"] == "Atlas"


# ============== update_persona Tool ==============


class TestUpdatePersonaTool:
    def test_update_basic_fields(self, patched_brain):
        """update_persona stores new values and returns confirmation."""
        from remy.core.brain_tools import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            result = json.loads(
                _execute_tool_inner(
                    "update_persona",
                    {
                        "name": "Nova",
                        "tone": "formal and precise",
                        "formality": "formal",
                    },
                    channel="desktop",
                )
            )
        assert result["updated"] is True
        assert result["persona"]["name"] == "Nova"
        assert result["persona"]["tone"] == "formal and precise"

    def test_update_traits(self, patched_brain):
        """update_persona updates individual traits."""
        from remy.core.brain_tools import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            result = json.loads(
                _execute_tool_inner(
                    "update_persona",
                    {
                        "humor": 0.9,
                        "warmth": 0.3,
                    },
                    channel="desktop",
                )
            )
        assert result["persona"]["traits"]["humor"] == 0.9
        assert result["persona"]["traits"]["warmth"] == 0.3
        # Other traits preserved
        assert result["persona"]["traits"]["curiosity"] == 0.7

    def test_update_traits_clamped(self, patched_brain):
        """Traits > 1.0 clamped to 1.0, < 0.0 clamped to 0.0."""
        from remy.core.brain_tools import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            result = json.loads(
                _execute_tool_inner(
                    "update_persona",
                    {
                        "humor": 5.0,
                        "warmth": -2.0,
                    },
                    channel="desktop",
                )
            )
        assert result["persona"]["traits"]["humor"] == 1.0
        assert result["persona"]["traits"]["warmth"] == 0.0

    def test_update_catchphrases_and_avoid(self, patched_brain):
        """Comma-separated catchphrases and avoid are parsed into lists."""
        from remy.core.brain_tools import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            result = json.loads(
                _execute_tool_inner(
                    "update_persona",
                    {
                        "catchphrases": "Stay healthy!, Keep moving!",
                        "avoid": "slang, profanity",
                    },
                    channel="desktop",
                )
            )
        assert result["persona"]["catchphrases"] == ["Stay healthy!", "Keep moving!"]
        assert result["persona"]["avoid"] == ["slang", "profanity"]

    def test_update_is_upsert(self, patched_brain):
        """Second update_persona modifies existing record, not creates new."""
        from remy.core.brain_tools import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache"):
            r1 = json.loads(
                _execute_tool_inner(
                    "update_persona",
                    {
                        "name": "V1",
                    },
                    channel="desktop",
                )
            )
            r2 = json.loads(
                _execute_tool_inner(
                    "update_persona",
                    {
                        "tone": "formal",
                    },
                    channel="desktop",
                )
            )
        # Same record id — upsert, not duplicate
        assert r1["id"] == r2["id"]
        # Name preserved from first update
        assert r2["persona"]["name"] == "V1"
        assert r2["persona"]["tone"] == "formal"

    def test_update_invalidates_system_cache(self, patched_brain):
        """update_persona calls invalidate_system_instruction_cache."""
        from remy.core.brain_tools import _execute_tool_inner

        with patch("remy.core.agent.invalidate_system_instruction_cache") as mock_inv:
            _execute_tool_inner("update_persona", {"name": "Test"}, channel="desktop")
        mock_inv.assert_called_once()


# ============== System Instruction Integration ==============


class TestSystemInstructionPersona:
    def test_instruction_uses_default_persona(self, mock_brain):
        """build_system_instruction with empty brain uses default persona 'Remy'."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction

            instruction = build_system_instruction(channel="telegram")
        assert "You are Remy" in instruction
        assert "Rules:" in instruction

    def test_instruction_uses_custom_persona(self, mock_brain):
        """build_system_instruction picks up stored persona record."""
        from remy.core.brain_tools import _PERSONA_TAG

        mock_brain.store(
            content="Agent Persona: Atlas — strict advisor. Tone: formal.",
            level=Level.IDENTITY,
            tags=[_PERSONA_TAG, "identity"],
            metadata={
                "type": "agent_persona",
                "name": "Atlas",
                "role": "strict medical advisor",
                "tone": "formal and precise",
                "scope": "health only",
                "motivations": "provide accurate medical guidance",
                "traits": {"warmth": 0.2, "humor": 0.1, "formality": 0.9, "conciseness": 0.8},
                "catchphrases": [],
                "avoid": [],
            },
        )
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction

            instruction = build_system_instruction(channel="telegram")
        assert "You are Atlas" in instruction
        assert "strict medical advisor" in instruction
        assert "formal and precise" in instruction
        assert "Rules:" in instruction

    def test_backward_compatible_no_persona_record(self, mock_brain):
        """No persona record → still produces valid instruction with 'Remy'."""
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction

            instruction = build_system_instruction(channel="voice")
        assert "You are Remy" in instruction
        assert "Rules:" in instruction
        # Safety rules still present
        assert "FINANCIAL DATA SAFETY" in instruction


# ============== Tool Registration ==============


class TestPersonaToolRegistration:
    def test_read_persona_in_brain_tools(self):
        """read_persona is declared in BRAIN_TOOLS."""
        from remy.core.brain_tools import BRAIN_TOOLS

        names = {t.name for t in BRAIN_TOOLS}
        assert "read_persona" in names

    def test_update_persona_in_brain_tools(self):
        """update_persona is declared in BRAIN_TOOLS."""
        from remy.core.brain_tools import BRAIN_TOOLS

        names = {t.name for t in BRAIN_TOOLS}
        assert "update_persona" in names

    def test_persona_tools_are_core(self):
        """read_persona and update_persona are in CORE_TOOL_NAMES (always available)."""
        from remy.core.brain_tools import CORE_TOOL_NAMES

        assert "read_persona" in CORE_TOOL_NAMES
        assert "update_persona" in CORE_TOOL_NAMES
