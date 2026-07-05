"""Tests for Multi-Model Review Gate — second-opinion LLM safety check."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from remy.core.approval_queue import (
    ApprovalQueue,
    review_action,
)


# ============== REVIEW_ACTION ==============


class TestReviewAction:

    @pytest.fixture(autouse=True)
    def patch_settings(self):
        """Patch settings at the import location used by review_action."""
        mock_s = MagicMock()
        mock_s.REVIEW_ENABLED = True
        mock_s.REVIEW_MODEL = "test-model"
        with patch("remy.config.settings.settings", mock_s):
            self._settings = mock_s
            yield

    def test_returns_safe_true(self):
        """review_action returns {safe: True, concerns: ''} for benign actions."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"safe": true, "concerns": ""}'
        )

        with patch("remy.core.llm.get_llm", return_value=mock_llm):
            result = review_action("recall", {"query": "hello"}, "Recall memory")

        assert result is not None
        assert result["safe"] is True
        assert result["concerns"] == ""

    def test_returns_safe_false_with_concerns(self):
        """review_action returns {safe: False} with concerns for risky actions."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"safe": false, "concerns": "Suspicious wallet address detected"}'
        )

        with patch("remy.core.llm.get_llm", return_value=mock_llm):
            result = review_action(
                "browse_page",
                {"url": "https://crypto-scam.com"},
                "Navigate to crypto site",
                url="https://crypto-scam.com",
            )

        assert result is not None
        assert result["safe"] is False
        assert "wallet" in result["concerns"].lower()

    def test_handles_markdown_wrapped_json(self):
        """review_action strips ```json ... ``` wrapper."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n{"safe": true, "concerns": ""}\n```'
        )

        with patch("remy.core.llm.get_llm", return_value=mock_llm):
            result = review_action("store", {"content": "hi"}, "Store data")

        assert result is not None
        assert result["safe"] is True

    def test_returns_none_when_disabled(self):
        """review_action returns None when REVIEW_ENABLED=False."""
        self._settings.REVIEW_ENABLED = False
        result = review_action("recall", {"query": "test"}, "Test")
        assert result is None

    def test_returns_none_on_llm_error(self):
        """review_action returns None (fail-open) when LLM call fails."""
        with patch("remy.core.llm.get_llm", side_effect=Exception("API down")):
            result = review_action("store", {"content": "data"}, "Store")

        assert result is None

    def test_returns_none_on_invalid_json(self):
        """review_action returns None when LLM returns non-JSON."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="I cannot help with that.")

        with patch("remy.core.llm.get_llm", return_value=mock_llm):
            result = review_action("store", {"content": "data"}, "Store")

        assert result is None


# ============== INTEGRATION WITH APPROVAL QUEUE ==============


class TestReviewIntegration:

    @pytest.fixture
    def queue(self):
        q = ApprovalQueue()
        q._enabled = True
        return q

    def _run_with_review(self, queue, mock_s, tool_name=None, tool_args=None, url=None):
        """Helper: run request_approval_sync and capture the description seen by request_approval."""
        description_seen = []

        async def capture_request(desc, action_fn):
            description_seen.append(desc)
            return '{"error": "timed out"}'

        # Ensure numeric attributes needed by request_approval_sync
        mock_s.APPROVAL_TIMEOUT_SEC = 5
        mock_s.APPROVAL_QUEUE_ENABLED = True
        mock_s.TELEGRAM_BOT_TOKEN = "tok"
        mock_s.PROACTIVE_CHAT_ID = 123

        with patch("remy.config.settings.settings", mock_s), \
             patch.object(queue, "request_approval", side_effect=capture_request), \
             patch.object(ApprovalQueue, "_telegram_configured", new_callable=PropertyMock, return_value=True):
            kwargs = {}
            if tool_name is not None:
                kwargs["tool_name"] = tool_name
            if tool_args is not None:
                kwargs["tool_args"] = tool_args
            if url is not None:
                kwargs["url"] = url

            queue.request_approval_sync(
                "Navigate to bank site",
                lambda: '{"ok": true}',
                **kwargs,
            )

        return description_seen

    def test_description_includes_warning_when_unsafe(self, queue):
        """Approval description is annotated with AI review warning."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"safe": false, "concerns": "Suspicious activity"}'
        )

        mock_s = MagicMock()
        mock_s.REVIEW_ENABLED = True
        mock_s.REVIEW_MODEL = "test-model"

        with patch("remy.core.llm.get_llm", return_value=mock_llm):
            seen = self._run_with_review(
                queue, mock_s,
                tool_name="browse_page",
                tool_args={"url": "https://bank.com"},
                url="https://bank.com",
            )

        assert len(seen) == 1
        assert "AI Review" in seen[0]
        assert "Suspicious activity" in seen[0]

    def test_description_includes_ok_when_safe(self, queue):
        """Approval description is annotated with AI Review OK."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"safe": true, "concerns": ""}'
        )

        mock_s = MagicMock()
        mock_s.REVIEW_ENABLED = True
        mock_s.REVIEW_MODEL = "test-model"

        with patch("remy.core.llm.get_llm", return_value=mock_llm):
            seen = self._run_with_review(
                queue, mock_s,
                tool_name="browse_page",
                tool_args={"url": "https://bank.com"},
                url="https://bank.com",
            )

        assert len(seen) == 1
        assert "AI Review: OK" in seen[0]

    def test_no_annotation_when_review_fails(self, queue):
        """No AI review annotation when review returns None (error)."""
        mock_s = MagicMock()
        mock_s.REVIEW_ENABLED = True
        mock_s.REVIEW_MODEL = "test-model"

        with patch("remy.core.llm.get_llm", side_effect=Exception("fail")):
            seen = self._run_with_review(
                queue, mock_s,
                tool_name="browse_page",
                tool_args={"url": "https://bank.com"},
                url="https://bank.com",
            )

        assert len(seen) == 1
        assert "AI Review" not in seen[0]

    def test_no_review_without_tool_name(self, queue):
        """No review when tool_name is not passed (backward compat)."""
        mock_s = MagicMock()

        seen = self._run_with_review(queue, mock_s)

        assert len(seen) == 1
        assert "AI Review" not in seen[0]
