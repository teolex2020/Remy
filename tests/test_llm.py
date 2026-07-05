"""Tests for REL-1: Multi-Model Fallback — core/llm.py"""

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ============== _is_transient_error ==============


class TestIsTransientError:

    def test_server_error_is_transient(self):
        """google.genai ServerError (500/503) should be transient."""
        from google.genai.errors import ServerError
        from remy.core.llm import _is_transient_error

        exc = ServerError.__new__(ServerError)
        Exception.__init__(exc, "server error")
        assert _is_transient_error(exc) is True

    def test_rate_limit_wrapped_is_transient(self):
        """ChatGoogleGenerativeAIError wrapping 429 ClientError → transient."""
        from remy.core.llm import _is_transient_error
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

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

    def test_ollama_prefix_is_local_provider(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("ollama:llama3.1") == "ollama"

    def test_openrouter_slash_model_is_openrouter(self):
        from remy.core.llm import _get_model_provider

        assert _get_model_provider("anthropic/claude-sonnet-4-5") == "openrouter"


class TestProviderDispatch:

    def test_ollama_uses_chat_ollama_without_api_key(self, monkeypatch):
        from remy.config.settings import settings
        from remy.core.llm import get_llm

        calls = {}

        class FakeChatOllama:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        fake_module = types.ModuleType("langchain_ollama")
        fake_module.ChatOllama = FakeChatOllama
        monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)
        monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")

        llm = get_llm("ollama:llama3.1")

        assert isinstance(llm, FakeChatOllama)
        assert calls == {"model": "llama3.1", "base_url": "http://127.0.0.1:11434"}

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
