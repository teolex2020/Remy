"""
Self-Critique Loop (AUTON-13) — evaluates agent actions BEFORE marking them complete.

A separate LLM call that critiques the agent's response, detects hallucinations,
identifies tool failures, and suggests retries. Runs between invoke_agent() and
_evaluate_outcome() in the autonomous cycle.

Key design decisions:
- Skip critique for trivial actions (recall, store, get_current_datetime)
- Budget: ~100-200 tokens per critique call
- Store critique history as brain records for meta-learning (AUTON-1)
- Fail-open: if critique LLM fails, proceed without blocking evaluation
"""

import json
import logging
from datetime import datetime

from remy.core.agent_tools import brain
from remy.core.autonomy_models import _llm_content_to_str
from remy.core.brain_tools import estimate_tokens, parse_llm_json

logger = logging.getLogger("Autonomy.Critique")

# Tools that are trivial / read-only — skip critique to save budget
SKIP_CRITIQUE_TOOLS = frozenset(
    {
        "recall",
        "recall_full",
        "get_full_record",
        "get_connections",
        "get_current_datetime",
        "list_records",
        "search_records",
        "metric_summary",
        "get_active_research_projects",
        "read_persona",
        "tool_status",
    }
)

# Tools that ALWAYS require critique — high-impact actions
ALWAYS_CRITIQUE_TOOLS = frozenset(
    {
        "write_file",
        "sandbox_create_tool",
        "sandbox_test_tool",
        "browse_page",
        "browser_act",
        "complete_research",
        "store",
        "store_research",
        "update_record",
        "delete_record",
        "delegate_task",
        "http_get",
    }
)

# Maximum retries driven by critique
MAX_CRITIQUE_RETRIES = 2

# Tags for storing critique records in brain
CRITIQUE_TAGS = ["self-critique", "autonomous-critique"]


def should_critique(session_log: list[dict]) -> bool:
    """Decide whether to run critique based on tool calls in the session log.

    Rules:
    - If ANY tool in ALWAYS_CRITIQUE_TOOLS was called → critique
    - If ALL tools are in SKIP_CRITIQUE_TOOLS → skip
    - If no tool calls → skip (pure chat response)
    - Otherwise → critique
    """
    tool_calls = [entry for entry in session_log if entry.get("type") == "tool_call"]

    if not tool_calls:
        return False

    tool_names = {entry.get("tool", "") for entry in tool_calls}

    # If any high-impact tool was used, always critique
    if tool_names & ALWAYS_CRITIQUE_TOOLS:
        return True

    # If all tools are trivial, skip
    if tool_names <= SKIP_CRITIQUE_TOOLS:
        return False

    # Mixed or unknown tools — critique
    return True


def check_action_accountability(
    response_text: str,
    session_log: list[dict],
) -> dict:
    """Deterministic (zero-LLM) accountability check.

    Detects action claims in response text that have no corresponding tool call.
    Returns a critique-compatible dict so callers can use it without an LLM call.

    Always runs — no feature flag needed. Zero token cost.
    """
    from remy.core.factuality import check_action_claims

    violations = check_action_claims(response_text, session_log)
    if not violations:
        return {"quality": 1.0, "issues": [], "suggestions": [], "should_retry": False,
                "critique_text": "action_accountability: OK", "deterministic": True}

    # violations is at most 1 entry (whole-turn analysis)
    v = violations[0]
    called_str = ", ".join(v.tools_called) if v.tools_called else "none"
    issues = [
        f"Agent produced a substantive response but called only read tools ({called_str}) — "
        "any action claims (store/update/deprecate/...) are unsubstantiated."
    ]

    return {
        "quality": 0.3,
        "issues": issues,
        "suggestions": [
            "Call the write tool first, then report what the result was.",
            "If no write action was needed, do not describe completing one.",
        ],
        "should_retry": True,
        "critique_text": f"action_accountability: read-only turn with action claims (called: {called_str})",
        "deterministic": True,
    }


def _extract_tool_summary(session_log: list[dict], max_tools: int = 10) -> str:
    """Extract a compact summary of tool calls from session log."""
    tool_calls = [entry for entry in session_log if entry.get("type") == "tool_call"][-max_tools:]

    if not tool_calls:
        return "(no tool calls)"

    lines = []
    for tc in tool_calls:
        tool = tc.get("tool", "?")
        result_snippet = str(tc.get("result", ""))[:100]
        args_snippet = str(tc.get("args", ""))[:80]
        lines.append(f"  - {tool}({args_snippet}) → {result_snippet}")

    return "\n".join(lines)


async def critique_response(
    goal_description: str,
    decision_prompt: str,
    response_text: str,
    session_log: list[dict],
) -> dict:
    """Run self-critique on agent's response. Returns critique analysis.

    Returns:
        {
            "quality": float (0.0-1.0),
            "issues": list[str],
            "suggestions": list[str],
            "should_retry": bool,
            "critique_text": str,
        }
    """
    tool_summary = _extract_tool_summary(session_log)

    critique_prompt = (
        "You are a STRICT quality reviewer of an AI agent's autonomous action. "
        "Your job is to find problems BEFORE the action is marked as complete. "
        "Output ONLY a JSON object.\n\n"
        "CRITIQUE PROTOCOL:\n"
        "- Compare what the agent SAID it did vs what tools actually RAN\n"
        "- If agent claims 'stored data' but no store tool was called → issue\n"
        "- If a tool returned an error or empty result → issue\n"
        "- If the response is vague filler ('I will do this...') with no concrete result → quality <= 0.2\n"
        "- If NO tools were called at all → quality = 0.0, should_retry = true\n"
        "- If worker timed out BUT tools were called and data was stored → quality >= 0.4, should_retry = false (partial progress counts)\n"
        "- If agent stated numbers/facts without calling a tool to verify → issue (fabricated data)\n"
        "- If response is mostly 'planning' or 'suggesting' without execution → quality <= 0.3\n"
        "- If the approach seems wrong for the goal → suggest alternative\n"
        "- should_retry: true ONLY if a different approach could succeed (not for impossible goals)\n\n"
        f"GOAL: {goal_description}\n\n"
        f"TASK CONTEXT (first 300 chars):\n{decision_prompt[:300]}\n\n"
        f"AGENT RESPONSE (first 400 chars):\n{response_text[:400]}\n\n"
        f"TOOL CALLS MADE:\n{tool_summary}\n\n"
        "Output exactly this JSON format (no extra text):\n"
        '{"quality": 0.7, "issues": ["issue1"], "suggestions": ["try X instead"], "should_retry": false}\n\n'
        "Fields:\n"
        "- quality: 0.0-1.0 (0.0 = complete failure, 1.0 = perfect execution)\n"
        "- issues: list of specific problems found (empty if none)\n"
        "- suggestions: concrete improvement suggestions (empty if none)\n"
        "- should_retry: true only if retrying with a DIFFERENT approach could help"
    )

    try:
        from remy.core.llm import call_llm_async

        result = await call_llm_async(
            critique_prompt,
            purpose="self_critique",
            channel="autonomous",
        )
        raw = _llm_content_to_str(result.content).strip()

        if not raw or len(raw) < 5:
            logger.warning("Critique LLM returned empty/short response")
            return _default_critique("LLM returned empty response")

        try:
            parsed = parse_llm_json(raw)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Critique JSON parse failed, using raw: %s", raw[:200])
            return _default_critique(raw[:200])

        # Validate and normalize
        quality = max(0.0, min(1.0, float(parsed.get("quality", 0.5))))
        issues = parsed.get("issues", [])
        if isinstance(issues, str):
            issues = [issues]
        suggestions = parsed.get("suggestions", [])
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        should_retry = bool(parsed.get("should_retry", False))

        critique = {
            "quality": quality,
            "issues": issues[:5],  # Cap at 5 issues
            "suggestions": suggestions[:3],  # Cap at 3 suggestions
            "should_retry": should_retry,
            "critique_text": raw[:300],
        }

        logger.info(
            "Critique: quality=%.2f issues=%d retry=%s",
            quality,
            len(issues),
            should_retry,
        )

        return critique

    except Exception as e:
        logger.warning("Self-critique failed (non-blocking): %s", e)
        return _default_critique(f"Critique unavailable: {e}")


def _default_critique(reason: str) -> dict:
    """Fallback critique when LLM fails — optimistic pass-through."""
    return {
        "quality": 0.5,
        "issues": [],
        "suggestions": [],
        "should_retry": False,
        "critique_text": reason,
    }


def store_critique(
    critique: dict,
    goal_description: str,
    action_id: str,
    session_id: str | None = None,
) -> str | None:
    """Store critique result in brain for meta-learning (AUTON-1 integration).

    Only stores critiques with issues (quality < 0.7) to avoid noise.
    Returns record ID if stored, None if skipped.
    """
    if critique["quality"] >= 0.7 and not critique["issues"]:
        return None  # Good quality, no issues — don't store noise

    try:
        from remy.core.agent_tools import Level

        content = (
            f"Self-critique for action {action_id}: "
            f"quality={critique['quality']:.2f}. "
            f"Issues: {'; '.join(critique['issues']) if critique['issues'] else 'none'}. "
            f"Suggestions: {'; '.join(critique['suggestions']) if critique['suggestions'] else 'none'}."
        )

        record = brain.store(
            content=content,
            level=Level.WORKING,
            tags=CRITIQUE_TAGS,
            metadata={
                "action_id": action_id,
                "quality": critique["quality"],
                "issues_count": len(critique["issues"]),
                "should_retry": critique["should_retry"],
                "goal": goal_description[:100],
                "source": "self-critique",
                "verified": False,
                "session_id": session_id or "",
                "timestamp": datetime.now().isoformat(),
            },
        )

        record_id = record.id if hasattr(record, "id") else str(record)
        logger.debug("Stored critique record: %s", record_id)
        return record_id

    except Exception as e:
        logger.warning("Failed to store critique: %s", e)
        return None


def estimate_critique_tokens(
    goal_description: str,
    decision_prompt: str,
    response_text: str,
    session_log: list[dict],
) -> int:
    """Estimate token cost of a critique call."""
    tool_summary = _extract_tool_summary(session_log)
    prompt_text = goal_description + decision_prompt[:300] + response_text[:400] + tool_summary
    return estimate_tokens(prompt_text) + 80  # +80 for template overhead + response
