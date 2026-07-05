"""
Brain-native mouth — renders structured governance decisions into final
user-facing text.

Architecture (Phase A.7 Step 4):
  - Brain emits GovernanceOutput (structured, language-agnostic).
  - This module converts GovernanceOutput + user message into a final
    reply in the turn's surface language, using the SLM as the rendering
    organ and a deterministic universal surface as hard fallback.

No hardcoded per-language sentence tables. No _HONEST_BLOCK_TEXT_*.
If the SLM cannot render, the fallback is an enum/symbol surface that
works across any script.

Only the block path is currently owned by the mouth renderer. soft and
aggressive modes are still handled by the legacy banner flow (A.7 Step 3
will clean that up after streaming/non-streaming are unified in Step 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MouthRenderResult:
    text: str
    source: str  # "slm" | "fallback_universal"
    error: str = ""


_BLOCK_RENDER_SYSTEM = (
    "You are a honest-uncertainty renderer. You are given:\n"
    "  - the user's original message\n"
    "  - a structured epistemic governance decision from the brain\n"
    "The draft answer that was going to be sent contained external references\n"
    "that failed structural verification. You MUST replace that answer with an\n"
    "honest statement of uncertainty.\n\n"
    "HARD RULES:\n"
    "  1. Reply in the EXACT SAME language the user used. Not a related\n"
    "     language. If the user wrote Ukrainian, reply in Ukrainian, not\n"
    "     Russian. If the user wrote transliterated Ukrainian (Latin\n"
    "     letters), reply in Ukrainian using those same Latin letters.\n"
    "     Match the user's script and language literally.\n"
    "  2. Do NOT repeat, restate, paraphrase, or allude to any unverified\n"
    "     external fact, citation, paper title, URL, DOI, or author name.\n"
    "  3. Do NOT invent citations, sources, or numeric details.\n"
    "  4. Explicitly state that you cannot confirm the references you were\n"
    "     about to give.\n"
    "  5. Optionally suggest a safer next action (a narrower question or\n"
    "     a grounded re-search).\n"
    "  6. Keep it short (2-4 sentences). No lists, no headers.\n"
    "  7. No emojis unless the user used them.\n"
)


def _build_block_prompt(user_message: str, output: Any) -> str:
    counts = f"{output.phantom_count}/{output.external_total}" if output.external_total else f"{output.phantom_count}/?"
    hint = getattr(output, "operator_hint", "none") or "none"
    return (
        f"USER MESSAGE:\n{user_message}\n\n"
        f"GOVERNANCE DECISION:\n"
        f"  mode: block\n"
        f"  reason_code: {output.reason_code}\n"
        f"  failed_external_references: {counts}\n"
        f"  operator_hint: {hint}\n\n"
        f"Write the honest-uncertainty reply now. Reply in the user's language."
    )


def _universal_fallback(output: Any) -> str:
    """Deterministic language-neutral surface used when SLM rendering fails.

    Enum + symbol only. Any human language wrapping is the mouth's job;
    if the mouth cannot speak, we surface the structure, not locale tables.
    """
    counts = (
        f"{output.phantom_count}/{output.external_total}"
        if output.external_total else f"{output.phantom_count}/?"
    )
    hint = getattr(output, "operator_hint", "none") or "none"
    return (
        "🛑 epistemic_block\n"
        f"reason: {output.reason_code}\n"
        f"unverified_external_refs: {counts}\n"
        f"suggested_action: {hint}"
    )


def render_block_response(
    user_message: str,
    governance_output: Any,
) -> MouthRenderResult:
    """Render a hard-block final response.

    Primary path: SLM renders honest uncertainty in the turn's language,
    constrained by a strict system prompt that forbids fabricated refs.

    Fallback path: deterministic universal enum/symbol surface. Used when
    the SLM call fails, returns empty, or is otherwise unusable.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from remy.core.llm import call_llm

        messages = [
            SystemMessage(content=_BLOCK_RENDER_SYSTEM),
            HumanMessage(content=_build_block_prompt(user_message, governance_output)),
        ]
        ai = call_llm(messages, purpose="mouth_block_render")
        raw = getattr(ai, "content", "") or ""
        if isinstance(raw, list):
            # Gemini and some providers return a list of content parts.
            parts = []
            for p in raw:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
            raw = "".join(parts)
        text = str(raw).strip()
        if not text:
            return MouthRenderResult(
                text=_universal_fallback(governance_output),
                source="fallback_universal",
                error="empty_slm_output",
            )
        return MouthRenderResult(text=text, source="slm")
    except Exception as exc:  # noqa: BLE001
        return MouthRenderResult(
            text=_universal_fallback(governance_output),
            source="fallback_universal",
            error=repr(exc),
        )
