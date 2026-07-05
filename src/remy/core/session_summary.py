"""
Session Summary — LLM-generated session summaries stored in brain.

Shared by all channels (voice, telegram, desktop, autonomous).
"""

import asyncio
import logging

logger = logging.getLogger("BrainTools")


def _get_bt():
    """Lazy accessor for brain_tools module (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt


async def generate_session_summary(client, session_log: list[dict], session_id: str) -> str | None:
    """Generate a concise session summary and store in brain.

    Shared by all channels (voice, telegram, etc.).
    """
    bt = _get_bt()
    brain = bt.brain
    settings = bt.settings

    if not session_log:
        return None

    entries = []
    for entry in session_log:
        if entry["type"] == "tool_call":
            entries.append(
                f"- Tool '{entry['tool']}': args={entry['args']}, result={str(entry.get('result', ''))[:200]}"
            )
        elif entry["type"] == "user_text":
            entries.append(f'- User said: "{entry["text"]}"')

    if not entries:
        return None

    # Cap at ~4000 tokens (~16000 chars) to prevent LLM overflow on long sessions.
    # Keep first 20% (session start context) + last 80% (most recent activity).
    _MAX_LOG_CHARS = 16000
    log_text = "\n".join(entries)
    if len(log_text) > _MAX_LOG_CHARS:
        head_budget = _MAX_LOG_CHARS // 5
        tail_budget = _MAX_LOG_CHARS - head_budget - 50
        log_text = log_text[:head_budget] + "\n...[truncated]...\n" + log_text[-tail_budget:]

    prompt = (
        "You are summarizing a session with a memory-equipped AI assistant. "
        "Based on the activity log below, write a 2-3 sentence summary of what was discussed. "
        "Focus on: topics discussed, information stored, questions answered. "
        "Write in the same language the user used (Ukrainian or English). "
        "Be concise and natural.\n\n"
        f"Activity log:\n{log_text}"
    )

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.SUMMARY_MODEL,
            contents=prompt,
        )
        summary = response.text.strip() if response.text else None

        if summary:
            from remy.core.agent_tools import Level, brain_lock
            from remy.core.proactive_context import _proactive_context_cache
            from remy.core.provenance import _stamp_provenance

            with brain_lock:
                brain.store(
                    content=summary,
                    level=Level.DOMAIN,
                    tags=["session-summary"],
                    metadata=_stamp_provenance(
                        {"session_id": session_id, "type": "session_summary"},
                        "system",
                        tags=["session-summary"],
                    ),
                    auto_promote=False,
                )
            logger.info(f"Session summary stored: {summary[:100]}")
            # Invalidate proactive context cache so next session sees this summary immediately
            _proactive_context_cache.clear()

        return summary
    except Exception as e:
        logger.warning(f"Session summary generation failed: {e}")
        return None
