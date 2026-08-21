"""Shadow-mode runner for the append-only session state.

Runs the append-only + date-guard compressor ALONGSIDE the real chat: after each
completed exchange it folds the exchange into the frozen state in a background
task and logs what the optimized prompt WOULD have cost versus the raw history.
Answers are never affected — this collects real-traffic numbers so the measured
94% retention / 60% saving (LoCoMo) can be confirmed or refuted on this app's
own sessions before the mode is ever switched on for real.

Design constraints:
- zero impact on answer latency: fired as a background asyncio task AFTER the
  response is finished;
- failure-safe: any error is swallowed and logged as a metric, never raised
  into the chat path;
- kill switch: set env SESSION_STATE_SHADOW=0 to disable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

_CHEAP_MODEL = "gemini-flash-lite-latest"


def shadow_enabled() -> bool:
    return os.environ.get("SESSION_STATE_SHADOW", "1") != "0"


def _cheap_llm_func(prompt: str):
    """One small compression call on the cheap model.

    Returns (text, {"usage_metadata": {...}}) — the tuple shape
    ``fold_exchange_append_only`` already understands.
    """
    from google import genai

    from remy.config.settings import settings

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(model=_CHEAP_MODEL, contents=prompt)
    usage = getattr(response, "usage_metadata", None)
    meta = {
        "usage_metadata": {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
        }
    }
    return (getattr(response, "text", "") or "").strip(), meta


async def _shadow_fold(session_id: str, user_text: str, response_text: str,
                       session_log: list | None) -> None:
    from remy.core.session_state_wrapper import (
        build_raw_prompt,
        estimate_tokens,
        fold_exchange_append_only,
        load_state,
        log_state_metric,
    )

    try:
        state = load_state(f"shadow_{session_id}")
        state.session_id = f"shadow_{session_id}"

        exchange = f"user: {user_text}\nassistant: {response_text}"
        session_date = time.strftime("%Y-%m-%d")

        # The compression call itself runs in a thread so the event loop is
        # never blocked by the HTTP request to the provider.
        loop = asyncio.get_running_loop()

        def _sync_llm(prompt: str):
            return _cheap_llm_func(prompt)

        async def _threaded_llm(prompt: str):
            return await loop.run_in_executor(None, _sync_llm, prompt)

        await fold_exchange_append_only(
            state,
            exchange,
            _threaded_llm,
            session_date=session_date,
            persist=True,
        )

        # Shadow economics: what would the next request cost, raw vs optimized?
        raw_prompt = build_raw_prompt(session_log, "(next request)")
        raw_tokens = estimate_tokens(raw_prompt)
        frozen_tokens = estimate_tokens(state.append_only_text())
        log_state_metric(
            {
                "event": "shadow_exchange",
                "session_id": state.session_id,
                "turns_folded": state.turns_folded,
                "raw_history_tokens_estimate": raw_tokens,
                "frozen_state_tokens_estimate": frozen_tokens,
                "shadow_saving_pct_estimate": round(
                    100.0 * (raw_tokens - frozen_tokens) / max(1, raw_tokens), 2
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 — shadow must never break chat
        logger.debug("session-state shadow fold failed: %s", exc)
        try:
            from remy.core.session_state_wrapper import log_state_metric

            log_state_metric(
                {
                    "event": "shadow_fold_error",
                    "session_id": f"shadow_{session_id}",
                    "error": str(exc)[:200],
                }
            )
        except Exception:
            pass


def schedule_shadow_fold(session_id: str, user_text: str, response_text: str,
                         session_log: list | None = None) -> None:
    """Fire-and-forget shadow fold. Safe to call from the chat path."""
    if not shadow_enabled():
        return
    if not user_text.strip() or not response_text.strip():
        return
    try:
        asyncio.get_running_loop().create_task(
            _shadow_fold(session_id, user_text, response_text, session_log)
        )
    except RuntimeError:
        # No running loop (sync context) — skip silently; shadow is best-effort.
        pass
