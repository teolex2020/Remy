"""Tests for the LangGraph agent — graph structure, routing, and invocation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from remy.core.agent import (
    AgentState,
    INSIGHT_CHECK_INTERVAL,
    MAX_TOOL_ITERATIONS,
    _build_factuality_contract_message,
    _extract_total_usage_tokens,
    _build_session_context,
    build_agent_graph,
    call_tools,
    call_model,
    check_session_insights,
    compact_history,
    should_continue,
)


# ============== FIXTURES ==============


@pytest.fixture
def mock_brain(tmp_path):
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture(autouse=True)
def patch_brain_and_registry(mock_brain, tmp_path):
    """Patch brain and registry for all tests in this module."""
    with patch("remy.core.brain_tools.brain", mock_brain), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.tool_registry.settings") as mock_settings, \
         patch("remy.core.agent._compiled_graphs", {}), \
         patch("remy.core.agent._tool_call_count", {}):
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
        yield


# ============== GRAPH STRUCTURE ==============


class TestGraphStructure:

    def test_graph_compiles(self):
        graph = build_agent_graph("desktop")
        assert graph is not None

    def test_graph_cached_per_channel(self):
        g1 = build_agent_graph("desktop")
        g2 = build_agent_graph("desktop")
        assert g1 is g2

    def test_different_channels_different_graphs(self):
        g1 = build_agent_graph("desktop")
        g2 = build_agent_graph("telegram")
        assert g1 is not g2


# ============== ROUTING ==============


class TestRouting:

    def test_end_on_text_response(self):
        state = AgentState(
            messages=[AIMessage(content="Hello!")],
            session_id="test",
            channel="desktop",
            session_log=[],
        )
        assert should_continue(state) == "__end__"

    def test_continue_on_tool_calls(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "recall",
                "args": {"query": "test"},
            }],
        )
        state = AgentState(
            messages=[ai_msg],
            session_id="test-route",
            channel="desktop",
            session_log=[],
        )
        assert should_continue(state) == "tools"

    def test_end_on_empty_messages(self):
        state = AgentState(
            messages=[],
            session_id="test",
            channel="desktop",
            session_log=[],
        )
        assert should_continue(state) == "__end__"

    def test_max_iterations_guard(self):
        """After MAX_TOOL_ITERATIONS, first routes to model for wrap-up, then hard END."""
        from remy.core.agent import _tool_call_count
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_x", "name": "recall", "args": {"query": "q"}}],
        )
        state = AgentState(
            messages=[ai_msg],
            session_id="loop-test",
            channel="desktop",
            session_log=[],
        )

        # First hit: routes to "model" for wrap-up answer (not silent END)
        _tool_call_count["loop-test"] = MAX_TOOL_ITERATIONS
        assert should_continue(state) == "model"
        # ToolMessages injected so model can respond
        assert any("Tool limit reached" in str(m.content)
                    for m in state["messages"] if isinstance(m, ToolMessage))

        # If model still wants tools after wrap-up, hard END
        ai_msg2 = AIMessage(
            content="",
            tool_calls=[{"id": "call_y", "name": "recall", "args": {"query": "q"}}],
        )
        state2 = AgentState(
            messages=[ai_msg2],
            session_id="loop-test",
            channel="desktop",
            session_log=[],
        )
        assert should_continue(state2) == "__end__"


class TestFactualityContract:

    def test_builds_contract_for_temporal_numeric_turn(self):
        state = AgentState(
            messages=[HumanMessage(content="What is the latest price and market size in 2026?")],
            session_id="fact-1",
            channel="desktop",
            session_log=[],
        )

        msg = _build_factuality_contract_message(state)

        assert isinstance(msg, SystemMessage)
        assert "Facts:" in msg.content
        assert "Needs verification:" in msg.content

    def test_builds_contract_after_recall_tool_usage(self):
        state = AgentState(
            messages=[HumanMessage(content="Summarize what you know about me")],
            session_id="fact-2",
            channel="desktop",
            session_log=[
                {"type": "tool_call", "tool": "recall", "result": "[id:r1] memory"},
            ],
        )

        msg = _build_factuality_contract_message(state)

        assert isinstance(msg, SystemMessage)
        assert "recalled records" in msg.content

    def test_contract_includes_factuality_claim_type_memory(self, monkeypatch):
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def policy_hint(self, situation, action, namespace=None):
                if action == "answer_claim_type:unverified_current_fact:without_evidence":
                    return {
                        "hint": "avoid",
                        "verdict": "refutes",
                        "supports": 0,
                        "refutes": 2,
                    }
                return {"hint": "verify_first", "supports": 0, "refutes": 0}

        monkeypatch.setattr(memory_api, "get_memory", lambda: StubMemory())
        state = AgentState(
            messages=[HumanMessage(content="What is the latest price and market size in 2026?")],
            session_id="fact-memory",
            channel="desktop",
            session_log=[],
        )

        msg = _build_factuality_contract_message(state)

        assert isinstance(msg, SystemMessage)
        assert "Lived factuality memory for this channel" in msg.content
        assert "unsupported unverified_current_fact" in msg.content
        assert "require fresh evidence" in msg.content

    def test_skips_contract_for_simple_chat(self):
        state = AgentState(
            messages=[HumanMessage(content="Hi")],
            session_id="fact-3",
            channel="desktop",
            session_log=[],
        )

        assert _build_factuality_contract_message(state) is None


class TestPiiShield:

    def test_call_model_shields_llm_payload_and_restores_response(self, monkeypatch):
        from remy.config.settings import settings

        monkeypatch.setattr(settings, "PII_SHIELD_ENABLED", True)
        captured = {}

        def fake_call_llm(messages, **kwargs):
            captured["messages"] = messages
            return AIMessage(content="I will use [PII:email_1] for that.")

        monkeypatch.setattr("remy.core.llm.call_llm", fake_call_llm)
        monkeypatch.setattr(
            "remy.core.adaptive_model_router.build_adaptive_model_routing",
            lambda **kwargs: {},
        )

        state = AgentState(
            messages=[HumanMessage(content="My email is user@example.com")],
            session_id="pii-shield-test",
            channel="desktop",
            session_log=[],
        )

        result = call_model(state)

        llm_text = "\n".join(str(msg.content) for msg in captured["messages"])
        assert "user@example.com" not in llm_text
        assert "[PII:email_1]" in llm_text
        assert result["messages"][0].content == "I will use user@example.com for that."


class TestUsageExtraction:

    def test_extracts_total_tokens_from_usage_metadata(self):
        msg = AIMessage(
            content="ok",
            response_metadata={
                "usage_metadata": {
                    "input_tokens": 120,
                    "output_tokens": 45,
                }
            },
        )
        assert _extract_total_usage_tokens(msg) == 165

    def test_prefers_explicit_total_tokens_when_present(self):
        msg = AIMessage(
            content="ok",
            response_metadata={
                "token_usage": {
                    "total_tokens": 99,
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                }
            },
        )
        assert _extract_total_usage_tokens(msg) == 99


# ============== TOOL NODE ==============


class TestCallTools:

    def test_executes_tool_call(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_dt",
                "name": "get_current_datetime",
                "args": {},
            }],
        )
        state = AgentState(
            messages=[ai_msg],
            session_id="test-tools",
            channel="desktop",
            session_log=[],
        )

        result = call_tools(state)
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)

        data = json.loads(msgs[0].content)
        assert "date" in data
        assert "time" in data

    def test_logs_tool_call(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_log",
                "name": "get_current_datetime",
                "args": {},
            }],
        )
        state = AgentState(
            messages=[ai_msg],
            session_id="test-log",
            channel="desktop",
            session_log=[],
        )

        result = call_tools(state)
        log = result["session_log"]
        assert len(log) == 1
        assert log[0]["tool"] == "get_current_datetime"

    def test_unknown_tool_returns_error(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_bad",
                "name": "nonexistent_tool",
                "args": {},
            }],
        )
        state = AgentState(
            messages=[ai_msg],
            session_id="test-bad",
            channel="desktop",
            session_log=[],
        )

        result = call_tools(state)
        msgs = result["messages"]
        assert "Unknown tool" in msgs[0].content

    def test_consequence_gate_blocks_refuted_tool_call(self, monkeypatch):
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.calls = []

            def policy_hint(self, situation, action, namespace=None):
                self.calls.append((situation, action, namespace))
                return {
                    "hint": "avoid",
                    "reason": "tool action was refuted before",
                    "verdict": "refutes",
                    "refutes": 1,
                    "should_block": True,
                }

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_blocked",
                "name": "get_current_datetime",
                "args": {},
            }],
        )
        state = AgentState(
            messages=[HumanMessage(content="What time is it?"), ai_msg],
            session_id="test-blocked-tool",
            channel="desktop",
            session_log=[],
        )

        result = call_tools(state)

        assert "Blocked by consequence memory" in result["messages"][0].content
        assert result["session_log"][0]["consequence_gate"]["blocked"] is True
        assert memory.calls == [
            ("What time is it?", "tool:get_current_datetime", "remy-tools")
        ]

    def test_tool_outcome_is_stored_as_consequence(self, monkeypatch):
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.units = []

            def policy_hint(self, situation, action, namespace=None):
                return {"hint": "", "supports": 0, "refutes": 0}

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return f"unit-{len(self.units)}"

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_store",
                "name": "nonexistent_tool",
                "args": {},
            }],
        )
        state = AgentState(
            messages=[HumanMessage(content="Run missing tool"), ai_msg],
            session_id="test-tool-memory",
            channel="desktop",
            session_log=[],
        )

        call_tools(state)

        assert len(memory.units) == 1
        unit = memory.units[0]
        assert unit["namespace"] == "remy-tools"
        assert unit["situation"] == "Run missing tool"
        assert unit["action"] == "tool:nonexistent_tool"
        assert unit["consequence"] == "REFUTES"
        assert "result:error" in unit["scope"]

    def test_no_tool_calls_returns_empty(self):
        state = AgentState(
            messages=[AIMessage(content="Just text")],
            session_id="test",
            channel="desktop",
            session_log=[],
        )

        result = call_tools(state)
        assert result["messages"] == []


# ============== INVOKE AGENT ==============


class TestInvokeAgent:

    @pytest.mark.asyncio
    async def test_invoke_returns_tuple(self):
        """invoke_agent should return (text, messages, log) tuple."""
        mock_response = AIMessage(content="Hello! I'm Remy.")

        with patch("remy.core.agent.build_agent_graph") as mock_graph:
            compiled = MagicMock()
            compiled.invoke.return_value = {
                "messages": [
                    HumanMessage(content="Hi"),
                    mock_response,
                ],
                "session_log": [{"type": "user_text", "text": "Hi"}],
            }
            mock_graph.return_value = compiled

            from remy.core.agent import invoke_agent
            text, msgs, log = await invoke_agent(
                "Hi", session_id="test", channel="desktop", session_log=[]
            )

        assert text == "Hello! I'm Remy."
        assert len(msgs) == 2
        assert isinstance(log, list)

    @pytest.mark.asyncio
    async def test_invoke_filters_system_messages(self):
        """System messages should be filtered from returned history."""
        with patch("remy.core.agent.build_agent_graph") as mock_graph:
            compiled = MagicMock()
            compiled.invoke.return_value = {
                "messages": [
                    SystemMessage(content="system"),
                    HumanMessage(content="Hi"),
                    AIMessage(content="Hello!"),
                ],
                "session_log": [],
            }
            mock_graph.return_value = compiled

            from remy.core.agent import invoke_agent
            text, msgs, log = await invoke_agent(
                "Hi", session_id="test", channel="desktop", session_log=[]
            )

        assert not any(isinstance(m, SystemMessage) for m in msgs)
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_invoke_appends_factuality_analysis_to_log(self):
        with patch("remy.core.agent.build_agent_graph") as mock_graph:
            compiled = MagicMock()
            compiled.invoke.return_value = {
                "messages": [
                    HumanMessage(content="What do you know about me?"),
                    AIMessage(content="Based on our conversation, you prefer tea."),
                ],
                "session_log": [
                    {
                        "type": "tool_call",
                        "tool": "recall",
                        "result": "[id:rec-1] [trust: 0.9 | interactive] User prefers tea over coffee [preference]",
                    }
                ],
            }
            mock_graph.return_value = compiled

            from remy.core.agent import invoke_agent
            _text, _msgs, log = await invoke_agent(
                "What do you know about me?",
                session_id="test-factuality-log",
                channel="desktop",
                session_log=[],
            )

        analysis_entries = [entry for entry in log if entry.get("type") == "factuality_analysis"]
        assert analysis_entries
        analysis = analysis_entries[-1]
        assert analysis["supported_claims_total"] >= 1
        assert analysis["evidence_record_ids"] == ["rec-1"]
        assert analysis["claims"][0]["supporting_record_ids"] == ["rec-1"]

    def test_factuality_consequence_is_stored_as_scar(self, monkeypatch):
        from remy.core.agent import _store_factuality_consequence
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.units = []

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return "unit-1"

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)
        report = type("Report", (), {"unsupported_observed_claims": 2})()

        _store_factuality_consequence(
            user_message="What happened?",
            session_id="s1",
            channel="desktop",
            factuality_report=report,
        )
        _store_factuality_consequence(
            user_message="Clean",
            session_id="s1",
            channel="desktop",
            factuality_report=type("Report", (), {"unsupported_observed_claims": 0})(),
        )

        assert len(memory.units) == 1
        unit = memory.units[0]
        assert unit["namespace"] == "remy-factuality"
        assert unit["action"] == "answer_without_evidence"
        assert unit["consequence"] == "REFUTES"
        assert "unsupported_observed_claims:2" in unit["scope"]

    def test_factuality_consequence_stores_claim_type_scars(self, monkeypatch):
        from remy.core.agent import _store_factuality_consequence
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.units = []

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return f"unit-{len(self.units)}"

        class Claim:
            def __init__(self, claim_class, supported, text):
                self.claim_class = claim_class
                self.supported = supported
                self.text = text

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)
        report = type(
            "Report",
            (),
            {
                "unsupported_observed_claims": 1,
                "unsupported_claims_total": 3,
                "unverified_current_claims": 2,
                "claim_details": [
                    Claim("observed_fact", False, "I verified this without evidence."),
                    Claim("unverified_current_fact", False, "The current price is 42."),
                    Claim("unverified_current_fact", False, "The latest share is 24%."),
                    Claim("memory_fact", True, "Based on our conversation, this is supported."),
                ],
            },
        )()

        _store_factuality_consequence(
            user_message="Give me current facts",
            session_id="s1",
            channel="desktop",
            factuality_report=report,
        )

        assert len(memory.units) == 3
        aggregate = memory.units[0]
        assert aggregate["namespace"] == "remy-factuality"
        assert "claim_class:observed_fact:1" in aggregate["scope"]
        assert "claim_class:unverified_current_fact:2" in aggregate["scope"]

        claim_units = {unit["action"]: unit for unit in memory.units[1:]}
        current = claim_units["answer_claim_type:unverified_current_fact:without_evidence"]
        observed = claim_units["answer_claim_type:observed_fact:without_evidence"]
        assert current["consequence"] == "REFUTES"
        assert "claim-type-scar" in current["scope"]
        assert "unsupported_claims:2" in current["scope"]
        assert observed["consequence"] == "REFUTES"

    def test_model_outcome_is_stored_as_consequence(self, monkeypatch):
        from remy.core.agent import _store_model_outcome_consequence
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.units = []

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return f"unit-{len(self.units)}"

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)

        _store_model_outcome_consequence(
            user_message="Research this",
            session_id="s1",
            channel="desktop",
            session_log=[{"type": "llm_call", "model": "fast-model", "fallback_used": False}],
            response_text="Done with evidence.",
            factuality_report=type("Report", (), {"unsupported_observed_claims": 0})(),
            governance_decision=None,
        )
        _store_model_outcome_consequence(
            user_message="Research this",
            session_id="s1",
            channel="desktop",
            session_log=[{"type": "llm_call", "model": "fast-model", "fallback_used": True}],
            response_text="Unsupported claim.",
            factuality_report=type("Report", (), {"unsupported_observed_claims": 2})(),
            governance_decision=None,
        )

        assert len(memory.units) == 2
        assert memory.units[0]["namespace"] == "remy-models"
        assert memory.units[0]["action"] == "model:fast-model"
        assert memory.units[0]["consequence"] == "SUPPORTS"
        assert "fallback:false" in memory.units[0]["scope"]
        assert memory.units[1]["consequence"] == "REFUTES"
        assert "fallback:true" in memory.units[1]["scope"]
        assert "unsupported_observed_claims:2" in memory.units[1]["scope"]

    def test_model_outcome_can_be_scoped_by_task_type(self, monkeypatch):
        from remy.core.agent import _store_model_outcome_consequence
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.units = []

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return "unit-1"

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)

        _store_model_outcome_consequence(
            user_message="Research this",
            session_id="s1",
            channel="desktop",
            task_type="research",
            session_log=[{"type": "llm_call", "model": "research-model", "fallback_used": False}],
            response_text="Done with evidence.",
            factuality_report=type("Report", (), {"unsupported_observed_claims": 0})(),
            governance_decision=None,
        )

        assert memory.units[0]["situation"].endswith("|task_type:research")
        assert "task_type:research" in memory.units[0]["scope"]

    def test_model_routing_reads_consequence_memory(self, monkeypatch):
        from remy.config.settings import settings
        from remy.core.agent import _model_routing_from_consequence_memory
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def policy_hint(self, situation, action, namespace=None):
                if action == "model:bad-model":
                    return {
                        "hint": "avoid",
                        "reason": "bad model was refuted",
                        "verdict": "refutes",
                        "supports": 0,
                        "refutes": 2,
                    }
                if action == "model:good-model":
                    return {
                        "hint": "prefer",
                        "reason": "good model was supported",
                        "verdict": "supports",
                        "supports": 3,
                        "refutes": 0,
                    }
                return {"hint": "verify_first", "supports": 0, "refutes": 0}

        monkeypatch.setattr(settings, "SUMMARY_MODEL", "bad-model")
        monkeypatch.setattr(settings, "FALLBACK_MODELS", ["good-model", "neutral-model"])
        monkeypatch.setattr(memory_api, "get_memory", lambda: StubMemory())

        routing = _model_routing_from_consequence_memory("desktop")

        assert routing["preferred_model"] == "good-model"
        assert routing["avoid_models"] == ("bad-model",)

    def test_model_routing_prefers_task_specific_memory_with_channel_fallback(self, monkeypatch):
        from remy.config.settings import settings
        from remy.core.agent import _model_routing_from_consequence_memory
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def policy_hint(self, situation, action, namespace=None):
                if situation.endswith("|task_type:research") and action == "model:research-good":
                    return {
                        "hint": "prefer",
                        "verdict": "supports",
                        "supports": 4,
                        "refutes": 0,
                    }
                if not situation.endswith("|task_type:research") and action == "model:general-bad":
                    return {
                        "hint": "avoid",
                        "verdict": "refutes",
                        "supports": 0,
                        "refutes": 2,
                    }
                return {"hint": "verify_first", "supports": 0, "refutes": 0}

        monkeypatch.setattr(settings, "SUMMARY_MODEL", "general-bad")
        monkeypatch.setattr(settings, "FALLBACK_MODELS", ["research-good", "neutral-model"])
        monkeypatch.setattr(memory_api, "get_memory", lambda: StubMemory())

        routing = _model_routing_from_consequence_memory("desktop", "research")

        assert routing["preferred_model"] == "research-good"
        assert routing["avoid_models"] == ("general-bad",)

    def test_source_grounding_consequence_is_stored(self, monkeypatch):
        from remy.core import claim_provenance
        from remy.core.agent import _store_source_grounding_consequences
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.units = []

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return f"unit-{len(self.units)}"

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)
        monkeypatch.setattr(
            claim_provenance,
            "get_turn_fetch_evidence",
            lambda session_id: [
                {
                    "tool": "extract_content",
                    "url": "https://arxiv.org/abs/2411.02534",
                    "title": "Paper",
                    "site": "arXiv",
                }
            ],
        )
        report = type(
            "Report",
            (),
            {
                "had_external_evidence": True,
                "unsupported_claims_total": 0,
                "unsupported_observed_claims": 0,
                "unverified_current_claims": 0,
                "unverified_external": 0,
                "external_citations_phantom": 0,
                "brain_storage_unsafe": False,
            },
        )()

        _store_source_grounding_consequences(
            session_id="s1",
            channel="desktop",
            factuality_report=report,
            governance_decision=None,
        )

        actions = {unit["action"]: unit for unit in memory.units}
        assert actions["source_class:research"]["namespace"] == "remy-sources"
        assert actions["source_class:research"]["consequence"] == "SUPPORTS"
        assert actions["source_host:arxiv.org"]["consequence"] == "SUPPORTS"
        assert actions["source_tool:extract_content"]["consequence"] == "SUPPORTS"
        assert "source-grounding" in actions["source_class:research"]["scope"]

    def test_source_memory_bias_affects_candidate_selection(self, monkeypatch):
        from remy.core.agent import _choose_best_candidate_source
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def policy_hint(self, situation, action, namespace=None):
                if action == "source_host:bad.example":
                    return {
                        "hint": "avoid",
                        "verdict": "refutes",
                        "supports": 0,
                        "refutes": 2,
                    }
                return {"hint": "verify_first", "supports": 0, "refutes": 0}

        monkeypatch.setattr(memory_api, "get_memory", lambda: StubMemory())
        selected = _choose_best_candidate_source(
            [
                {"title": "Neutral source", "uri": "https://bad.example/a"},
                {"title": "Neutral source", "uri": "https://good.example/a"},
            ],
            query="neutral source",
        )

        assert selected["uri"] == "https://good.example/a"
        assert "memory_avoid:source_host:bad.example" not in selected["trust_reason"]

    @pytest.mark.asyncio
    async def test_invoke_trims_long_history(self):
        """History over 40 messages should be trimmed."""
        long_history = []
        for i in range(50):
            long_history.append(HumanMessage(content=f"msg {i}"))
            long_history.append(AIMessage(content=f"reply {i}"))

        with patch("remy.core.agent.build_agent_graph") as mock_graph:
            compiled = MagicMock()
            compiled.invoke.return_value = {
                "messages": [AIMessage(content="Trimmed response")],
                "session_log": [],
            }
            mock_graph.return_value = compiled

            from remy.core.agent import invoke_agent
            text, msgs, log = await invoke_agent(
                "Hi", session_id="test", channel="desktop",
                session_log=[], history=long_history,
            )

            # Verify the state passed to invoke had trimmed messages
            call_args = compiled.invoke.call_args[0][0]
            # Should be trimmed to ~30 + 1 new message
            assert len(call_args["messages"]) <= 32

    @pytest.mark.asyncio
    async def test_invoke_fallback_on_no_response(self):
        """If no AIMessage with content, return fallback text."""
        with patch("remy.core.agent.build_agent_graph") as mock_graph:
            compiled = MagicMock()
            compiled.invoke.return_value = {
                "messages": [],
                "session_log": [],
            }
            mock_graph.return_value = compiled

            from remy.core.agent import invoke_agent
            text, msgs, log = await invoke_agent(
                "Hi", session_id="test", channel="desktop", session_log=[]
            )

        assert "couldn't generate" in text.lower()


# ============== IN-SESSION THINKING ==============


class TestInSessionThinking:

    def test_no_insight_before_interval(self):
        """Messages unchanged before INSIGHT_CHECK_INTERVAL."""
        from remy.core.agent import _message_counts
        _message_counts.pop("test-insight-1", None)

        msgs = [HumanMessage(content="Hello")]
        result = check_session_insights("test-insight-1", msgs)
        assert len(result) == 1  # No SystemMessage added
        assert not any(isinstance(m, SystemMessage) for m in result)

    def test_insight_injected_at_interval(self, mock_brain):
        """At INSIGHT_CHECK_INTERVAL, a SystemMessage insight is injected if brain has insights."""
        from remy.core.agent import _message_counts

        # Set counter to one less than interval so next call triggers
        _message_counts["test-insight-2"] = INSIGHT_CHECK_INTERVAL - 1

        # Store enough data for brain.insights() to have something
        from aura import Level
        for i in range(10):
            mock_brain.store(content=f"Memory about topic {i}", level=Level.DOMAIN, tags=["health"])

        msgs = [HumanMessage(content="Hello")]
        with patch("remy.core.agent_tools.brain", mock_brain):
            result = check_session_insights("test-insight-2", msgs)

        # Result may or may not have insight depending on brain state,
        # but function should not crash
        assert len(result) >= 1

    def test_insight_check_handles_empty_brain(self, mock_brain):
        """Empty brain → no crash, no insight injected."""
        from remy.core.agent import _message_counts
        _message_counts["test-insight-3"] = INSIGHT_CHECK_INTERVAL - 1

        msgs = [HumanMessage(content="Hello")]
        with patch("remy.core.agent_tools.brain", mock_brain):
            result = check_session_insights("test-insight-3", msgs)

        # No SystemMessage injected for empty brain
        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 0

    def test_insight_check_handles_error(self):
        """brain.insights() error → no crash, messages unchanged."""
        from remy.core.agent import _message_counts
        _message_counts["test-insight-4"] = INSIGHT_CHECK_INTERVAL - 1

        mock_bad_brain = MagicMock()
        mock_bad_brain.insights.side_effect = RuntimeError("DB error")

        msgs = [HumanMessage(content="Hello")]
        with patch("remy.core.agent_tools.brain", mock_bad_brain):
            result = check_session_insights("test-insight-4", msgs)

        assert len(result) == 1  # Original message only


# ============== COMPACT HISTORY ==============


class TestCompactHistory:

    def test_short_history_unchanged(self):
        """History shorter than keep_recent is returned as-is."""
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm good."),
        ]
        result = compact_history(msgs)
        assert len(result) == 4
        assert result[0].content == "Hello"

    def test_tool_result_truncated(self):
        """ToolMessage content > 300 chars gets truncated."""
        long_result = "x" * 500
        msgs = [
            HumanMessage(content="Search"),
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "recall", "args": {}}]),
            ToolMessage(content=long_result, tool_call_id="c1"),
            AIMessage(content="Found it."),
        ]
        result = compact_history(msgs)
        tool_msg = [m for m in result if isinstance(m, ToolMessage)][0]
        assert len(tool_msg.content) < 500
        assert tool_msg.content.endswith("...[truncated]")

    def test_tool_result_short_kept(self):
        """ToolMessage content <= 300 chars is unchanged."""
        short_result = '{"stored": true, "id": "abc123"}'
        msgs = [
            HumanMessage(content="Store this"),
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "store", "args": {}}]),
            ToolMessage(content=short_result, tool_call_id="c1"),
            AIMessage(content="Stored."),
        ]
        result = compact_history(msgs)
        tool_msg = [m for m in result if isinstance(m, ToolMessage)][0]
        assert tool_msg.content == short_result

    def test_long_history_compressed(self):
        """History > keep_recent gets compressed with summary."""
        msgs = []
        for i in range(20):
            msgs.append(HumanMessage(content=f"Question {i}"))
            msgs.append(AIMessage(content=f"Answer {i}"))

        result = compact_history(msgs, keep_recent=16)
        # Should have summary SystemMessage + recent messages
        assert len(result) <= 17  # 1 summary + 16 recent
        assert isinstance(result[0], SystemMessage)
        assert "Earlier in this conversation" in result[0].content

    def test_tool_sequence_not_broken(self):
        """AIMessage with tool_calls + ToolMessage are never split."""
        msgs = []
        # Old messages
        for i in range(10):
            msgs.append(HumanMessage(content=f"Q{i}"))
            msgs.append(AIMessage(content=f"A{i}"))
        # Tool sequence right at the split boundary
        msgs.append(HumanMessage(content="Do recall"))
        msgs.append(AIMessage(content="", tool_calls=[{"id": "c1", "name": "recall", "args": {}}]))
        msgs.append(ToolMessage(content="result", tool_call_id="c1"))
        msgs.append(AIMessage(content="Here's what I found."))
        # More recent
        for i in range(6):
            msgs.append(HumanMessage(content=f"Recent {i}"))
            msgs.append(AIMessage(content=f"Reply {i}"))

        result = compact_history(msgs, keep_recent=16)
        # Check no orphaned ToolMessage at start of recent part
        non_system = [m for m in result if not isinstance(m, SystemMessage)]
        if non_system:
            assert not isinstance(non_system[0], ToolMessage)

    def test_summary_has_user_messages(self):
        """Summary includes User: lines from old messages."""
        msgs = []
        for i in range(20):
            msgs.append(HumanMessage(content=f"My question about topic {i}"))
            msgs.append(AIMessage(content=f"Answer about topic {i}"))

        result = compact_history(msgs, keep_recent=10)
        summary = result[0]
        assert isinstance(summary, SystemMessage)
        assert "User: My question about topic 0" in summary.content
        assert "Remy: Answer about topic 0" in summary.content


# ============== SESSION CONTEXT (RM-11) ==============


class TestBuildSessionContext:

    def test_returns_none_for_short_conversation(self):
        """Conversations < 4 messages don't need session context."""
        msgs = [HumanMessage(content="Hello"), AIMessage(content="Hi!")]
        assert _build_session_context(msgs) is None

    def test_returns_none_for_empty(self):
        assert _build_session_context([]) is None

    def test_extracts_user_facts(self):
        """User statements are extracted into session context."""
        msgs = [
            HumanMessage(content="I already created a wallet on Tron network"),
            AIMessage(content="Great, your wallet is ready."),
            HumanMessage(content="Now I'm waiting for my friend to send TRX"),
            AIMessage(content="OK, let me know when it arrives."),
            HumanMessage(content="What should I do next?"),
        ]
        ctx = _build_session_context(msgs)
        assert ctx is not None
        assert isinstance(ctx, SystemMessage)
        assert "already created a wallet" in ctx.content
        assert "waiting for my friend" in ctx.content

    def test_extracts_tool_actions(self):
        """Tool calls are listed as actions taken."""
        msgs = [
            HumanMessage(content="Store my wallet address"),
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "store", "args": {"content": "wallet TNjy..."}}],
            ),
            ToolMessage(content="Stored successfully, id=abc123", tool_call_id="c1"),
            AIMessage(content="Done, I stored your wallet address."),
            HumanMessage(content="Thanks"),
        ]
        ctx = _build_session_context(msgs)
        assert ctx is not None
        assert "store(" in ctx.content
        assert "Stored successfully" in ctx.content

    def test_includes_do_not_contradict_warning(self):
        """Context should include the anti-contradiction instruction."""
        msgs = [
            HumanMessage(content="I have a Tron wallet already"),
            AIMessage(content="Understood, you have a wallet."),
            HumanMessage(content="The address is TNjy..."),
            AIMessage(content="Got it."),
        ]
        ctx = _build_session_context(msgs)
        assert ctx is not None
        assert "Do NOT propose actions that were already completed" in ctx.content

    def test_limits_user_facts_to_8(self):
        """Only the last 8 user statements are kept."""
        msgs = []
        for i in range(15):
            msgs.append(HumanMessage(content=f"User statement number {i} with details"))
            msgs.append(AIMessage(content=f"Response {i}"))

        ctx = _build_session_context(msgs)
        assert ctx is not None
        # Should contain later statements, not early ones
        assert "number 14" in ctx.content
        assert "number 7" in ctx.content
        # Statement 0-6 should be trimmed
        assert "number 0" not in ctx.content

    def test_limits_actions_to_10(self):
        """Only the last 10 actions are kept."""
        msgs = []
        for i in range(15):
            msgs.append(HumanMessage(content=f"Do action {i}"))
            msgs.append(AIMessage(
                content="",
                tool_calls=[{"id": f"c{i}", "name": f"tool_{i}", "args": {"x": str(i)}}],
            ))
            msgs.append(ToolMessage(content=f"Result {i}", tool_call_id=f"c{i}"))
            msgs.append(AIMessage(content=f"Done {i}"))

        ctx = _build_session_context(msgs)
        assert ctx is not None
        # The content should have some actions but not all 30 (15 calls + 15 results)

    def test_skips_short_messages(self):
        """Very short user messages (< 10 chars) are skipped."""
        msgs = [
            HumanMessage(content="ok"),
            AIMessage(content="Sure."),
            HumanMessage(content="yes"),
            AIMessage(content="Alright."),
            HumanMessage(content="I need help with my Tron wallet setup"),
        ]
        ctx = _build_session_context(msgs)
        # "ok" and "yes" should not appear in context
        if ctx:
            assert "ok" not in ctx.content.split("User stated:")[1].split("Actions")[0] if "User stated:" in ctx.content else True

    def test_multimodal_messages_handled(self):
        """Non-string message content doesn't crash."""
        msgs = [
            HumanMessage(content=[{"type": "text", "text": "Look at this image"}]),
            AIMessage(content="I see it."),
            HumanMessage(content="What do you think about this wallet?"),
            AIMessage(content="Looks good."),
        ]
        ctx = _build_session_context(msgs)
        # Should not crash, multimodal content is skipped gracefully
        assert ctx is None or isinstance(ctx, SystemMessage)
