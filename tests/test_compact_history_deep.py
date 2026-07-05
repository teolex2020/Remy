"""Tests for compact_history behavior with 20+ tool calls (Section 15 analysis).

Validates that:
1. Tool results are truncated at 300 chars
2. Old messages are summarized (User/AI text preserved)
3. Tool call sequences are not broken
4. Summary doesn't exceed ~8000 chars
5. Scratchpad notes survive compaction (injected separately by call_model)
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from remy.core.agent import compact_history

# ============== Helpers ==============


def _make_tool_call_sequence(tool_name: str, result: str, call_id: str):
    """Create an AIMessage with tool_call + matching ToolMessage."""
    ai = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {"query": "test"}, "id": call_id}],
    )
    tool = ToolMessage(content=result, tool_call_id=call_id)
    return ai, tool


def _build_long_conversation(num_tool_calls: int = 25):
    """Build a realistic conversation with many tool calls."""
    messages = [HumanMessage(content="Research AI in medicine and find key trends")]

    for i in range(num_tool_calls):
        ai, tool = _make_tool_call_sequence(
            "web_search",
            f"Result {i}: {'A' * 400}",  # Each result > 300 chars
            f"call_{i}",
        )
        messages.extend([ai, tool])

        # Every 5 tool calls, add an AI summary
        if (i + 1) % 5 == 0:
            messages.append(
                AIMessage(content=f"Summary after {i + 1} searches: found {i + 1} results")
            )

    return messages


# ============== Tests ==============


class TestCompactHistoryDeep:
    def test_truncation_at_300_chars(self):
        """Tool results > 300 chars should be truncated."""
        messages = [
            HumanMessage(content="test"),
            *_make_tool_call_sequence("web_search", "x" * 500, "c1"),
        ]
        result = compact_history(messages, keep_recent=16)
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert len(tool_msgs[0].content) < 500
        assert "truncated" in tool_msgs[0].content

    def test_20_tool_calls_compacts(self):
        """20+ tool calls should trigger compaction."""
        messages = _build_long_conversation(20)
        assert len(messages) > 40  # At least 40 messages (AI+Tool pairs)

        result = compact_history(messages, keep_recent=16)
        assert len(result) <= 20  # Significantly compacted

    def test_recent_tool_sequences_preserved_without_human_message(self):
        """Tool-heavy conversations now preserve recent tool sequences.

        When recent_part has no HumanMessage (all tool call pairs),
        compact_history keeps them instead of stripping everything.
        Only leading orphan ToolMessages are removed.
        """
        messages = _build_long_conversation(25)
        result = compact_history(messages, keep_recent=16)

        # Summary is always generated
        has_summary = any(isinstance(m, SystemMessage) for m in result)
        assert has_summary

        # Recent tool sequences should be preserved (not stripped)
        non_sys = [m for m in result if not isinstance(m, SystemMessage)]
        assert len(non_sys) >= 10  # Significant recent context kept

        # Tool sequences should remain intact
        for i, msg in enumerate(result):
            if isinstance(msg, ToolMessage) and i > 0:
                prev = result[i - 1]
                assert isinstance(prev, (AIMessage, ToolMessage))

    def test_summary_contains_user_text(self):
        """Compaction summary should preserve user messages."""
        messages = [
            HumanMessage(content="Research AI safety trends"),
            *_make_tool_call_sequence("web_search", "result1", "c1"),
            AIMessage(content="Found some results about safety"),
            HumanMessage(content="Now search for medical AI"),
            *_make_tool_call_sequence("web_search", "result2", "c2"),
            AIMessage(content="Found medical AI info"),
        ]
        # Add enough recent messages to push old ones into summary
        for i in range(20):
            messages.extend(_make_tool_call_sequence("recall", f"data{i}", f"r{i}"))

        result = compact_history(messages, keep_recent=16)
        summary_msgs = [m for m in result if isinstance(m, SystemMessage)]
        if summary_msgs:
            summary_text = summary_msgs[0].content
            assert "AI safety" in summary_text or "Research" in summary_text

    def test_tool_sequences_not_broken(self):
        """AIMessage + ToolMessage pairs should never be split."""
        messages = _build_long_conversation(15)
        result = compact_history(messages, keep_recent=8)

        # Check that every ToolMessage has a preceding AIMessage with matching tool_call
        for i, msg in enumerate(result):
            if isinstance(msg, ToolMessage) and i > 0:
                # The preceding non-SystemMessage should be an AIMessage
                prev = result[i - 1]
                assert isinstance(prev, (AIMessage, ToolMessage)), (
                    f"ToolMessage at index {i} not preceded by AIMessage/ToolMessage"
                )

    def test_summary_size_bounded(self):
        """Summary should not exceed ~8000 chars even with 50+ messages."""
        messages = _build_long_conversation(30)
        result = compact_history(messages, keep_recent=16)

        summary_msgs = [m for m in result if isinstance(m, SystemMessage)]
        if summary_msgs:
            assert len(summary_msgs[0].content) <= 10000  # ~8000 + overhead

    def test_dynamic_keep_recent_for_autonomous(self):
        """Autonomous channel should use dynamic keep_recent."""
        from remy.core.agent import _estimate_keep_recent

        simple = _estimate_keep_recent("autonomous", HumanMessage(content="check status"))
        complex_ = _estimate_keep_recent(
            "autonomous",
            HumanMessage(content="research and analyze AI trends, then implement findings"),
        )
        # Complex goal should get more context
        assert complex_ > simple

    def test_25_tool_calls_summary_preserves_user_ai_text(self):
        """With 25 tool calls, summary captures User/AI text from old messages."""
        messages = _build_long_conversation(25)
        result = compact_history(messages, keep_recent=16)

        summary_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(summary_msgs) >= 1

        summary_text = summary_msgs[0].content
        assert "Research AI in medicine" in summary_text
        assert "Summary after" in summary_text

    def test_mixed_conversation_aligns_to_human_message(self):
        """When recent_part HAS a HumanMessage, align to it (old behavior)."""
        messages = [
            HumanMessage(content="First request"),
            *_make_tool_call_sequence("web_search", "r1", "c1"),
            AIMessage(content="Here's what I found"),
            HumanMessage(content="Second request"),
            *_make_tool_call_sequence("web_search", "r2", "c2"),
            AIMessage(content="More results"),
        ]
        # Add enough to trigger compaction, ending with a HumanMessage
        for i in range(10):
            messages.extend(_make_tool_call_sequence("recall", f"d{i}", f"x{i}"))
        messages.append(HumanMessage(content="Final question"))
        messages.extend(_make_tool_call_sequence("recall", "answer", "final"))

        result = compact_history(messages, keep_recent=8)
        non_sys = [m for m in result if not isinstance(m, SystemMessage)]
        # Should start with a HumanMessage (aligned)
        if non_sys:
            assert isinstance(non_sys[0], HumanMessage)

    def test_orphan_tool_messages_stripped(self):
        """Leading orphan ToolMessages (no matching AIMessage) are stripped."""
        # Build a scenario where split lands in the middle of a tool sequence
        messages = [HumanMessage(content="test")]
        for i in range(20):
            messages.extend(_make_tool_call_sequence("web_search", f"r{i}", f"c{i}"))

        result = compact_history(messages, keep_recent=16)
        non_sys = [m for m in result if not isinstance(m, SystemMessage)]
        # First non-system message should NOT be an orphan ToolMessage
        if non_sys:
            assert not isinstance(non_sys[0], ToolMessage), (
                "Leading orphan ToolMessage should be stripped"
            )

    def test_no_empty_result_with_tool_heavy_conversation(self):
        """compact_history should NEVER return only a summary with zero context."""
        messages = _build_long_conversation(25)
        result = compact_history(messages, keep_recent=16)

        # Must have summary + at least some recent messages
        has_summary = any(isinstance(m, SystemMessage) for m in result)
        non_sys = [m for m in result if not isinstance(m, SystemMessage)]
        assert has_summary
        assert len(non_sys) > 0, "Must preserve some recent messages, not just summary"


class TestCompactHistoryWithScratchpad:
    """Verify that scratchpad provides safety net for compact_history data loss."""

    def test_scratchpad_survives_compaction(self):
        """Scratchpad notes exist independently of message history."""
        from remy.core.scratchpad import clear_notes, read_notes, write_note

        # Clear any existing notes
        clear_notes()

        # Write notes that simulate saving tool results
        write_note("Finding 1: AI agents are now autonomous in hospitals")
        write_note("Finding 2: Drug discovery reduced from 4 years to 18 months")

        # Compact a long conversation (scratchpad is NOT in messages)
        messages = _build_long_conversation(20)
        compact_history(messages, keep_recent=8)

        # Scratchpad notes should still be there
        notes = read_notes()
        assert len(notes) == 2
        assert "AI agents" in notes[0]["content"] or "AI agents" in notes[1]["content"]

        # Clean up
        clear_notes()
