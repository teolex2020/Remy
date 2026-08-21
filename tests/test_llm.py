"""Tests for REL-1: Multi-Model Fallback — core/llm.py"""

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

# ============== _is_transient_error ==============


class TestIsTransientError:

    def test_generic_provider_error_is_transient(self):
        from remy.core.llm import _is_transient_error

        assert _is_transient_error(RuntimeError("Provider returned error")) is True

    def test_nested_http_status_is_transient(self):
        from remy.core.llm import _is_transient_error

        cause = RuntimeError("upstream unavailable")
        cause.status_code = 503
        outer = RuntimeError("model adapter failed")
        outer.__cause__ = cause

        assert _is_transient_error(outer) is True

    def test_server_error_is_transient(self):
        """google.genai ServerError (500/503) should be transient."""
        from google.genai.errors import ServerError

        from remy.core.llm import _is_transient_error

        exc = ServerError.__new__(ServerError)
        Exception.__init__(exc, "server error")
        assert _is_transient_error(exc) is True

    def test_rate_limit_wrapped_is_transient(self):
        """ChatGoogleGenerativeAIError wrapping 429 ClientError → transient."""
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

        from remy.core.llm import _is_transient_error

        # Create a cause that looks like a 429 error
        cause = Exception("rate limited")
        cause.code = 429

        real_exc = ChatGoogleGenerativeAIError("Error: rate limited")
        real_exc.__cause__ = cause
        assert _is_transient_error(real_exc) is True

    def test_connection_error_is_transient(self):
        """ConnectionError and TimeoutError are transient."""
        from remy.core.llm import _is_transient_error

        assert _is_transient_error(ConnectionError("refused")) is True
        assert _is_transient_error(TimeoutError("timed out")) is True
        assert _is_transient_error(OSError("network unreachable")) is True

    def test_value_error_not_transient(self):
        """ValueError is NOT transient — should not retry."""
        from remy.core.llm import _is_transient_error

        assert _is_transient_error(ValueError("bad input")) is False

    def test_import_error_not_transient(self):
        """ImportError is NOT transient."""
        from remy.core.llm import _is_transient_error

        assert _is_transient_error(ImportError("missing module")) is False


# ============== get_llm ==============


class TestGetLlm:

    def test_google_model_returns_chat_google(self):
        """Gemini model name → ChatGoogleGenerativeAI instance."""
        from remy.core.llm import get_llm

        llm = get_llm("gemini-3-flash-preview")
        assert "Google" in type(llm).__name__ or "Generative" in type(llm).__name__

    def test_default_model_uses_summary_model(self):
        """No model_name → uses settings.SUMMARY_MODEL."""
        from remy.core.llm import get_llm

        llm = get_llm()
        assert llm is not None

    def test_openai_model_without_package_raises(self):
        """OpenAI model name without langchain-openai → ImportError."""
        from remy.core.llm import get_llm

        with patch.dict("sys.modules", {"langchain_openai": None}):
            with pytest.raises(ImportError, match="langchain-openai"):
                get_llm("gpt-4o")

    def test_openai_reasoning_model_uses_responses_api(self, monkeypatch):
        import sys
        import types

        from remy.core import model_registry
        from remy.core.llm import get_llm

        calls = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        fake_module = types.ModuleType("langchain_openai")
        fake_module.ChatOpenAI = FakeChatOpenAI
        monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
        monkeypatch.setattr(model_registry, "get_api_key_for_model", lambda model: "openai-key")

        get_llm("gpt-5.6-sol")

        assert calls["use_responses_api"] is True
        assert calls["output_version"] == "responses/v1"

    def test_openai_legacy_chat_model_keeps_chat_completions(self, monkeypatch):
        import sys
        import types

        from remy.core import model_registry
        from remy.core.llm import get_llm

        calls = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        fake_module = types.ModuleType("langchain_openai")
        fake_module.ChatOpenAI = FakeChatOpenAI
        monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
        monkeypatch.setattr(model_registry, "get_api_key_for_model", lambda model: "openai-key")

        get_llm("gpt-4o")

        assert "use_responses_api" not in calls


# ============== _get_fallback_chain ==============


class TestGetFallbackChain:

    def test_empty_by_default(self):
        """No FALLBACK_MODELS configured → empty list."""
        from remy.core.llm import _get_fallback_chain

        with patch("remy.core.llm.settings") as s:
            s.FALLBACK_MODELS = []
            assert _get_fallback_chain() == []

    def test_returns_configured_models(self):
        """Configured models returned in order."""
        from remy.core.llm import _get_fallback_chain

        with patch("remy.core.llm.settings") as s:
            s.FALLBACK_MODELS = ["gemini-2.5-flash", "gpt-4o-mini"]
            result = _get_fallback_chain()
            assert result == ["gemini-2.5-flash", "gpt-4o-mini"]


# ============== call_llm ==============


class TestCallLlm:

    def test_primary_model_success(self):
        """Primary model works → return result, no fallback."""
        from remy.core.llm import call_llm

        mock_response = MagicMock()
        mock_response.content = "Hello"
        mock_response.response_metadata = {}

        with patch("remy.core.llm.get_llm") as mock_get:
            mock_get.return_value.invoke.return_value = mock_response
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "gemini-3-flash-preview"
                s.FALLBACK_MODELS = []
                result = call_llm("test")

        assert result.content == "Hello"
        assert result.response_metadata["_fallback_used"] is False
        assert result.response_metadata["_served_by"] == "gemini-3-flash-preview"

    def test_fallback_on_transient_error(self):
        """Primary fails with transient error → fallback succeeds."""
        from remy.core.llm import call_llm

        primary_llm = MagicMock()
        primary_llm.invoke.side_effect = ConnectionError("connection refused")

        fallback_response = MagicMock()
        fallback_response.content = "Fallback OK"
        fallback_response.response_metadata = {}
        fallback_llm = MagicMock()
        fallback_llm.invoke.return_value = fallback_response

        # Primary retries 3 times before falling back, so provide 3 primary + 1 fallback
        with patch("remy.core.llm.get_llm") as mock_get, \
             patch("remy.core.llm.time.sleep"):
            mock_get.side_effect = [primary_llm, primary_llm, primary_llm, fallback_llm]
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "primary-model"
                s.FALLBACK_MODELS = ["fallback-model"]
                result = call_llm("test")

        assert result.content == "Fallback OK"
        assert result.response_metadata["_served_by"] == "fallback-model"
        assert result.response_metadata["_fallback_used"] is True

    def test_fallback_attempt_lifecycle_is_sent_to_trajectory(self):
        from remy.core.llm import call_llm
        from remy.core.logging_config import ctx_session_id

        primary_llm = MagicMock()
        primary_llm.invoke.side_effect = ConnectionError("connection refused")
        fallback_llm = MagicMock()
        fallback_llm.invoke.return_value = AIMessage(content="Recovered")
        trajectory = MagicMock()
        trajectory.begin_model_attempt.side_effect = [
            "attempt-1", "attempt-2", "attempt-3", "attempt-4"
        ]
        session_token = ctx_session_id.set("conversation-1")
        try:
            with patch("remy.core.llm.get_llm") as mock_get, \
                 patch("remy.core.llm.time.sleep"), \
                 patch(
                     "remy.core.trajectory_store.get_trajectory_store",
                     return_value=trajectory,
                 ):
                mock_get.side_effect = [
                    primary_llm, primary_llm, primary_llm, fallback_llm
                ]
                with patch("remy.core.llm.settings") as settings_mock:
                    settings_mock.SUMMARY_MODEL = "primary-model"
                    settings_mock.FALLBACK_MODELS = ["fallback-model"]
                    result = call_llm("test", purpose="agent")
        finally:
            ctx_session_id.reset(session_token)

        assert result.content == "Recovered"
        assert trajectory.begin_model_attempt.call_count == 4
        assert trajectory.complete_model_attempt.call_count == 4
        completed = [call.kwargs for call in trajectory.complete_model_attempt.call_args_list]
        assert completed[0]["retry_action"] == "retry-same-model"
        assert completed[2]["retry_action"] == "fallback-model"
        assert completed[3]["success"] is True

    def test_empty_completion_retries_once_then_uses_fallback(self):
        """A provider's empty AI message must not become a fake chat reply."""
        from remy.core.llm import call_llm

        empty = AIMessage(content="")
        primary_llm = MagicMock()
        primary_llm.invoke.return_value = empty
        fallback_llm = MagicMock()
        fallback_llm.invoke.return_value = AIMessage(content="Fallback answer")

        with patch("remy.core.llm.get_llm") as mock_get, \
             patch("remy.core.llm.time.sleep"):
            mock_get.side_effect = [primary_llm, primary_llm, fallback_llm]
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "empty-primary"
                s.FALLBACK_MODELS = ["working-fallback"]
                result = call_llm("hello")

        assert result.content == "Fallback answer"
        assert result.response_metadata["_served_by"] == "working-fallback"
        assert mock_get.call_count == 3

    def test_tool_bound_empty_completion_retries_as_plain_chat(self):
        """NVIDIA-style empty tool response gets one unbound chat retry."""
        from remy.core.llm import call_llm

        bound = MagicMock()
        bound.invoke.return_value = AIMessage(content="")
        primary_llm = MagicMock()
        primary_llm.bind_tools.return_value = bound
        primary_llm.invoke.return_value = AIMessage(content="Відповідь із наявного контексту")
        tools = [MagicMock()]

        with patch("remy.core.llm.get_llm", return_value=primary_llm) as mock_get, \
             patch("remy.core.llm.time.sleep"):
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "nvidia-model"
                s.FALLBACK_MODELS = []
                result = call_llm("question", tools=tools)

        assert result.content == "Відповідь із наявного контексту"
        primary_llm.bind_tools.assert_called_once_with(tools)
        bound.invoke.assert_called_once()
        primary_llm.invoke.assert_called_once()
        assert mock_get.call_count == 2

    def test_llamacpp_template_error_retries_with_alternating_plain_chat(self):
        """Strict Gemma templates should fall back without exposing HTTP 400."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from remy.core.llm import _LOCAL_MODELS_WITHOUT_NATIVE_TOOLS, call_llm

        bound = MagicMock()
        bound.invoke.side_effect = RuntimeError(
            "Error code: 400 - Unable to generate parser for this template. "
            "Jinja Exception: Conversation roles must alternate user/assistant"
        )
        local_llm = MagicMock()
        local_llm.bind_tools.return_value = bound
        local_llm.invoke.return_value = AIMessage(content="Local answer")
        prompt = [
            SystemMessage(content="Core instruction"),
            SystemMessage(content="Memory context"),
            HumanMessage(content="Hello"),
        ]
        _LOCAL_MODELS_WITHOUT_NATIVE_TOOLS.discard("llamacpp:gemma-local")

        with patch("remy.core.llm.get_llm", return_value=local_llm) as mock_get:
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "llamacpp:gemma-local"
                s.FALLBACK_MODELS = []
                result = call_llm(prompt, tools=[MagicMock()])
                second = call_llm(prompt, tools=[MagicMock()])

        assert result.content == "Local answer"
        local_llm.bind_tools.assert_called_once()
        bound.invoke.assert_called_once()
        plain_prompt = local_llm.invoke.call_args.args[0]
        assert [message.type for message in plain_prompt] == ["system", "human"]
        assert "Core instruction" in plain_prompt[0].content
        assert "Memory context" in plain_prompt[0].content
        assert mock_get.call_count == 3
        assert second.content == "Local answer"
        assert local_llm.bind_tools.call_count == 1
        assert local_llm.invoke.call_count == 2

    def test_llamacpp_plain_chat_flattens_tool_history_and_keeps_role_order(self):
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        from remy.core.llm import _prepare_llamacpp_prompt

        prompt = [
            SystemMessage(content="Core"),
            SystemMessage(content="Context"),
            HumanMessage(content="Research this"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call-1"}],
            ),
            ToolMessage(content="Result one", tool_call_id="call-1", name="search"),
            ToolMessage(content="Result two", tool_call_id="call-2", name="search"),
            AIMessage(content="Interim conclusion"),
            HumanMessage(content="Continue"),
        ]

        prepared = _prepare_llamacpp_prompt(prompt, plain_chat=True)

        assert [message.type for message in prepared] == [
            "system",
            "human",
            "ai",
            "human",
            "ai",
            "human",
        ]
        assert prepared[0].content == "Core\n\nContext"
        assert "Result one" in prepared[3].content
        assert "Result two" in prepared[3].content

    def test_no_fallback_on_non_transient(self):
        """Non-transient error (ValueError) raises immediately, no fallback."""
        from remy.core.llm import call_llm

        with patch("remy.core.llm.get_llm") as mock_get:
            mock_get.return_value.invoke.side_effect = ValueError("bad prompt")
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "primary"
                s.FALLBACK_MODELS = ["backup"]
                with pytest.raises(ValueError, match="bad prompt"):
                    call_llm("test")

        # Fallback model should NOT have been tried
        assert mock_get.call_count == 1

    def test_all_models_fail_raises_last(self):
        """All models fail with transient errors → raise last exception."""
        from remy.core.llm import call_llm

        with patch("remy.core.llm.get_llm") as mock_get, \
             patch("remy.core.llm.time.sleep"):
            mock_get.return_value.invoke.side_effect = ConnectionError("down")
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "primary"
                s.FALLBACK_MODELS = ["backup1", "backup2"]
                with pytest.raises(ConnectionError):
                    call_llm("test")

        # 3 models × 3 retries each = 9 get_llm calls
        assert mock_get.call_count == 9

    def test_tool_binding_passed_to_fallback(self):
        """When tools provided, fallback model also gets bind_tools."""
        from remy.core.llm import call_llm

        primary_llm = MagicMock()
        primary_llm.bind_tools.return_value.invoke.side_effect = ConnectionError("fail")

        fallback_response = MagicMock()
        fallback_response.content = "OK"
        fallback_response.response_metadata = {}
        fallback_llm = MagicMock()
        fallback_llm.bind_tools.return_value.invoke.return_value = fallback_response

        tools = [MagicMock()]
        # Primary retries 3 times, then fallback succeeds on first try
        with patch("remy.core.llm.get_llm") as mock_get, \
             patch("remy.core.llm.time.sleep"):
            mock_get.side_effect = [primary_llm, primary_llm, primary_llm, fallback_llm]
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "primary"
                s.FALLBACK_MODELS = ["backup"]
                call_llm("test", tools=tools)

        primary_llm.bind_tools.assert_called_with(tools)
        fallback_llm.bind_tools.assert_called_once_with(tools)

    def test_no_fallback_configured(self):
        """Empty FALLBACK_MODELS → only tries primary, works fine."""
        from remy.core.llm import call_llm

        mock_response = MagicMock()
        mock_response.content = "OK"
        mock_response.response_metadata = {}

        with patch("remy.core.llm.get_llm") as mock_get:
            mock_get.return_value.invoke.return_value = mock_response
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "gemini-3-flash-preview"
                s.FALLBACK_MODELS = []
                result = call_llm("test")

        assert result.content == "OK"
        assert mock_get.call_count == 1

    def test_deduplicates_models(self):
        """If primary is also in FALLBACK_MODELS, don't try it twice."""
        from remy.core.llm import call_llm

        with patch("remy.core.llm.get_llm") as mock_get, \
             patch("remy.core.llm.time.sleep"):
            mock_get.return_value.invoke.side_effect = ConnectionError("down")
            with patch("remy.core.llm.settings") as s:
                s.SUMMARY_MODEL = "gemini-3-flash-preview"
                s.FALLBACK_MODELS = ["gemini-3-flash-preview", "backup"]
                with pytest.raises(ConnectionError):
                    call_llm("test")

        # 2 unique models × 3 retries each = 6 get_llm calls
        assert mock_get.call_count == 6


# ============== call_llm_async ==============


class TestCallLlmAsync:

    def test_async_delegates_to_sync(self):
        """call_llm_async wraps call_llm via asyncio.to_thread."""
        from remy.core.llm import call_llm_async

        mock_response = MagicMock()
        mock_response.content = "async OK"
        mock_response.response_metadata = {}

        with patch("remy.core.llm.call_llm") as mock_call:
            mock_call.return_value = mock_response
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    call_llm_async("test", purpose="test")
                )
            finally:
                loop.close()

        assert result.content == "async OK"
        mock_call.assert_called_once()


# ============== _get_model_provider ==============


class TestGetModelProvider:

    def test_gpt_is_openai(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("gpt-4o") == "openai"
        assert _get_model_provider("gpt-3.5-turbo") == "openai"

    def test_o1_is_openai(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("o1-mini") == "openai"
        assert _get_model_provider("o3-mini") == "openai"

    def test_gemini_is_google(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("gemini-3-flash-preview") == "google"
        assert _get_model_provider("gemini-2.5-flash") == "google"

    def test_unknown_defaults_to_google(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("some-custom-model") == "google"

    def test_llamacpp_prefix_is_local_provider(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("llamacpp:qwen-gguf") == "llamacpp"

    def test_openrouter_slash_model_is_openrouter(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("anthropic/claude-sonnet-4-5") == "openrouter"


class TestProviderDispatch:

    def test_nvidia_registry_provider_uses_chat_nvidia(self, monkeypatch):
        from remy.core import model_registry
        from remy.core.llm import get_llm

        calls = {}

        class FakeChatNVIDIA:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        fake_module = types.ModuleType("langchain_nvidia_ai_endpoints")
        fake_module.ChatNVIDIA = FakeChatNVIDIA
        monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", fake_module)
        monkeypatch.setattr(model_registry, "get_api_key_for_model", lambda model: "nvapi-test")
        monkeypatch.setattr(model_registry, "get_provider_for_model", lambda model: "nvidia")

        llm = get_llm("deepseek-ai/deepseek-v4-flash")

        assert isinstance(llm, FakeChatNVIDIA)
        assert calls == {
            "model": "deepseek-ai/deepseek-v4-flash",
            "api_key": "nvapi-test",
        }

    def test_llamacpp_uses_local_openai_server_without_api_key(self, monkeypatch):
        from remy.config.settings import settings
        from remy.core.llama_cpp_service import llama_cpp_service
        from remy.core.llm import get_llm

        calls = {}
        started = []

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        fake_module = types.ModuleType("langchain_openai")
        fake_module.ChatOpenAI = FakeChatOpenAI
        monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
        monkeypatch.setattr(settings, "LLAMA_CPP_BASE_URL", "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(llama_cpp_service, "start_model", started.append)

        llm = get_llm("llamacpp:qwen-gguf")

        assert isinstance(llm, FakeChatOpenAI)
        assert started == ["qwen-gguf"]
        assert calls == {
            "model": "qwen-gguf",
            "api_key": "local-llama-cpp",
            "base_url": "http://127.0.0.1:11435/v1",
        }

    def test_deepseek_uses_openai_compatible_base_url(self, monkeypatch):
        from remy.core import model_registry
        from remy.core.llm import get_llm

        calls = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        fake_module = types.ModuleType("langchain_openai")
        fake_module.ChatOpenAI = FakeChatOpenAI
        monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
        monkeypatch.setattr(model_registry, "get_api_key_for_model", lambda model: "deepseek-key")

        llm = get_llm("deepseek-chat")

        assert isinstance(llm, FakeChatOpenAI)
        assert calls["model"] == "deepseek-chat"
        assert calls["api_key"] == "deepseek-key"
        assert calls["base_url"] == "https://api.deepseek.com"

    def test_xai_uses_openai_compatible_base_url(self, monkeypatch):
        from remy.core import model_registry
        from remy.core.llm import get_llm

        calls = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        fake_module = types.ModuleType("langchain_openai")
        fake_module.ChatOpenAI = FakeChatOpenAI
        monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
        monkeypatch.setattr(model_registry, "get_api_key_for_model", lambda model: "xai-key")

        llm = get_llm("grok-3")

        assert isinstance(llm, FakeChatOpenAI)
        assert calls["model"] == "grok-3"
        assert calls["api_key"] == "xai-key"
        assert calls["base_url"] == "https://api.x.ai/v1"


# ============== Settings Validator ==============


class TestFallbackModelsValidator:

    def test_parse_comma_separated(self):
        """Comma-separated string → list of model names."""
        from remy.config.settings import Settings

        s = Settings(FALLBACK_MODELS="gemini-2.5-flash, gpt-4o-mini")
        assert s.FALLBACK_MODELS == ["gemini-2.5-flash", "gpt-4o-mini"]

    def test_empty_string(self):
        """Empty string → empty list."""
        from remy.config.settings import Settings

        s = Settings(FALLBACK_MODELS="")
        assert s.FALLBACK_MODELS == []

    def test_list_input(self):
        """List input passes through unchanged."""
        from remy.config.settings import Settings

        s = Settings(FALLBACK_MODELS=["model-a", "model-b"])
        assert s.FALLBACK_MODELS == ["model-a", "model-b"]
