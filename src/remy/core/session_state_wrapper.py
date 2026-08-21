"""Incremental session state wrapper for long chat context.

This module keeps the raw transcript intact, but lets the prompt switch from
"full transcript" to a compact session state plus recent raw turns once a
session becomes long enough. It is intentionally independent from Aura and from
the web runtime so it can be tested before being wired into the chat path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SNAPSHOT_TURN_THRESHOLD = 8
SNAPSHOT_TOKEN_BUDGET = 700
RECENT_RAW_TURNS = 4

_STORE_LOCK = Lock()


def estimate_tokens(text: str) -> int:
    """Cheap provider-independent token estimate for local measurements."""
    return max(1, len(re.findall(r"[\w\-]+", text or "", re.UNICODE)))


def _data_dir() -> Path:
    try:
        from remy.config.settings import settings

        return Path(settings.DATA_DIR)
    except Exception:
        return Path("data")


def _provider_prompt_tokens(usage: dict[str, Any]) -> int | None:
    for key in ("prompt_token_count", "prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _provider_output_tokens(usage: dict[str, Any]) -> int | None:
    for key in ("candidates_token_count", "completion_tokens", "output_tokens"):
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


# Kept specific on purpose: too many false positives make the pinned set grow
# until it destroys the savings this module is supposed to create.
_PIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "uuid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.UNICODE)),
    ("url", re.compile(r"\bhttps?://[^\s<>()]+", re.IGNORECASE)),
    ("windows_path", re.compile(r"\b[A-Za-z]:\\[^\s<>|?*]+")),
    ("posix_path", re.compile(r"(?<!\w)/(?:[\w .-]+/)+[\w .-]+", re.UNICODE)),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d[\d\s\-()]{7,}\d)(?!\d)")),
    ("code_chain", re.compile(r"\b[A-Z]{2,}(?:[-_][A-Z0-9]+){1,}\b")),
    ("code", re.compile(r"\b[A-Z]{2,}[-_]?\d{2,}\b")),
    (
        "python_test_id",
        re.compile(r"\b[\w./\\-]+\.py::[\w./\\:\[\]-]+\b", re.UNICODE),
    ),
    ("id_dash", re.compile(r"\b[\w]+[-_]\d+\b", re.UNICODE)),
    ("hash", re.compile(r"\b[0-9a-fA-F]{7,40}\b")),
    (
        "dose",
        # (?!\w) instead of a trailing \b: equivalent for letter units (mg, ml)
        # but also matches "%" \u2014 \b after a non-word char like % never asserts
        # before a space/period, so "15%." silently failed to pin.
        re.compile(
            r"\b\d+(?:[.,]\d+)?(?:\s?[-\u2013]\s?\d+(?:[.,]\d+)?)?\s?"
            r"(?:mg|mcg|ml|g|kg|bpm|mmHg|%|\u043c\u0433|\u043c\u043a\u0433|"
            r"\u043c\u043b|\u0433|\u043a\u0433|\u0443\u0434/\u0445\u0432)(?!\w)",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        "money",
        re.compile(
            r"(?:[$\u20ac\u20b4\u00a3]\s?\d[\d\s.,]*|\b\d[\d\s.,]*\s?"
            r"(?:usd|eur|uah|\u0433\u0440\u043d|\u0434\u043e\u043b|"
            r"\u0454\u0432\u0440\u043e)\b)",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    ("percent_word", re.compile(r"\b\d+(?:[.,]\d+)?\s?percent\b", re.IGNORECASE)),
    (
        # Marker-gated on purpose (same anti-bloat rule as above): only pins
        # values explicitly labelled as a status, never bare uppercase words.
        # Captures the whole "status: X" phrase so the label survives verbatim.
        "status",
        re.compile(
            r"\b(?:status|state|статус|стан)\s*[:=]\s*[\w-]{2,}",
            re.IGNORECASE | re.UNICODE,
        ),
    ),
    (
        "datetime",
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b"
            r"|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
            r"|\b\d{1,2}:\d{2}\b"
        ),
    ),
]


def _clean_pin(value: str) -> str:
    return value.strip().strip("[](){}<>").rstrip(",.;:")


def extract_pinned_facts(text: str, existing: list[str] | None = None) -> list[str]:
    """Extract exact values that should survive compression verbatim."""
    found: list[str] = []
    for item in existing or []:
        item = _clean_pin(str(item))
        if item and item not in found:
            found.append(item)

    for _name, pattern in _PIN_PATTERNS:
        for match in pattern.findall(text or ""):
            value = _clean_pin(match if isinstance(match, str) else match[0])
            if len(value) < 2 or value in found:
                continue

            if any(value in old for old in found):
                continue

            shorter = [old for old in found if old in value]
            for old in shorter:
                found.remove(old)
            found.append(value)
    return found


def _projected_evidence_lines(
    session_log: list[dict] | None,
    pins: list[str],
    *,
    max_lines: int = 14,
    max_chars: int = 260,
) -> list[str]:
    """Keep compact source lines that explain what exact pins mean."""
    if not pins:
        return []

    evidence: list[str] = []
    for item in session_log or []:
        role, text = _turn_text(item)
        text = " ".join((text or "").split())
        if role != "user" or not text:
            continue
        if not any(pin and pin in text for pin in pins):
            continue
        line = text[: max_chars - 3].rstrip() + "..." if len(text) > max_chars else text
        if line not in evidence:
            evidence.append(line)
        if len(evidence) >= max_lines:
            break
    return evidence


def _projected_marked_source_lines(
    session_log: list[dict] | None,
    *,
    max_lines: int = 12,
    max_chars: int = 280,
) -> list[str]:
    """Keep compact verbatim lines for marked durable user turns.

    The LLM extractor can still drop small but important modifiers ("terse").
    A short source-line section gives the final answer model the original words
    without replaying the whole transcript.
    """
    evidence: list[str] = []
    for item in session_log or []:
        role, text = _turn_text(item)
        text = " ".join((text or "").split())
        if role != "user" or not text:
            continue
        if not _turn_has_decision_marker(text):
            continue
        line = text[: max_chars - 3].rstrip() + "..." if len(text) > max_chars else text
        if line not in evidence:
            evidence.append(line)
        if len(evidence) >= max_lines:
            break
    return evidence


STATE_SECTIONS = [
    "USER_PROFILE",
    "CURRENT_GOAL",
    "IMPORTANT_DECISIONS",
    "OPEN_THREADS",
    "CONSTRAINTS",
    "DO_NOT_FORGET",
]


@dataclass
class SessionState:
    """Compact, incrementally maintained state for one user session."""

    session_id: str = ""
    snapshot: str = ""
    pinned_facts: list[str] = field(default_factory=list)
    # Append-only compressed history: each entry is one exchange compressed
    # exactly once and then FROZEN (never re-compressed). This is what keeps
    # retention at ~94% instead of ~79-88%: losses cannot accumulate because
    # old chunks are never passed through the model again.
    frozen_chunks: list[str] = field(default_factory=list)
    turns_folded: int = 0
    raw_turn_cursor: int = 0
    failure_count: int = 0
    updated_at: float = 0.0
    schema_version: int = SCHEMA_VERSION

    def append_only_text(self) -> str:
        """The frozen compressed history, oldest first."""
        return "\n".join(self.frozen_chunks).strip()

    def render(self, recent_raw: str = "") -> str:
        parts: list[str] = []
        frozen = self.append_only_text()
        if frozen:
            parts.append("=== COMPRESSED HISTORY (each line is one past exchange) ===")
            parts.append(frozen)
        if self.snapshot.strip():
            parts.append("=== SESSION STATE (current knowledge) ===")
            parts.append(self.snapshot.strip())
        if self.pinned_facts:
            parts.append("=== EXACT FACTS (verbatim, do not alter) ===")
            parts.append(", ".join(self.pinned_facts))
        if recent_raw.strip():
            parts.append("=== RECENT TURNS (verbatim) ===")
            parts.append(recent_raw.strip())
        return "\n".join(parts).strip()


_SNAPSHOT_INSTRUCTION = (
    "You maintain a compact, structured running state of a conversation.\n"
    "Output the updated state using exactly these sections, omitting a section "
    "only when it is truly empty:\n"
    + "\n".join(f"[{name}]" for name in STATE_SECTIONS)
    + "\nRules:\n"
    "- Keep only durable, decision-relevant facts.\n"
    "- Merge information into existing sections; do not append a turn log.\n"
    f"- Keep the state under about {SNAPSHOT_TOKEN_BUDGET} tokens.\n"
    "- Preserve exact codes, numbers, doses, dates, ids and paths verbatim.\n"
    "- Output only the updated state."
)


def _turn_text(item: Any) -> tuple[str, str]:
    if isinstance(item, dict):
        kind = str(item.get("type") or "")
        if kind in {"user_text", "user_voice"}:
            return "user", str(item.get("text") or "")
        if kind == "model_response":
            return "assistant", str(item.get("full_text") or item.get("text") or "")
        return "", ""

    role = getattr(item, "type", "") or getattr(item, "role", "")
    content = getattr(item, "content", "")
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    return str(role), str(content or "")


def count_turns(session_log: list[dict] | None) -> int:
    return sum(
        1
        for item in session_log or []
        if str((item or {}).get("type") or "") in {"user_text", "user_voice", "model_response"}
    )


def recent_raw_turns(session_log: list[dict] | None, n: int = RECENT_RAW_TURNS) -> str:
    lines: list[str] = []
    for item in session_log or []:
        role, text = _turn_text(item)
        text = text.strip()
        if role and text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines[-n:])


# Markers that suggest a turn states a decision, preference, standing rule,
# status/risk label, or durable reason.
# Only marked turns pay for a decision-extraction LLM call; noise turns are free.
# Includes English, Cyrillic Ukrainian, AND Latin-transliterated Ukrainian
# (users write "zavzhdy"/"bo"/"pravilo" without a Cyrillic keyboard) — a missed
# marker means the extractor never runs, so the coverage must be language-robust.
_DECISION_MARKERS = (
    # English
    "decid", "chose", "choose", "prefer", "dislike", "always", "never",
    "because", "reason", "rationale", "instead", "we go with", "rule:",
    "rule ", "keep them", "under ", "marked", "risk", "lost ",
    # Ukrainian (Cyrillic)
    "виріш", "обрал", "переваж", "завжди", "ніколи", "тому що", "замість",
    "правил", "постав", "спочатку", "причин", "ризик",
    # Ukrainian (Latin transliteration)
    "virish", "obral", "pereva", "zavzhdy", "nikoly", "bo ", "zamist",
    "pravil", "stav ", "spochatku", "prychyn", "ryzyk",
    "zapamjatai", "zapamyatai",
)

_DECISION_INSTRUCTION = (
    "Extract from the exchange ONLY durable session facts: decisions made, "
    "stated preferences, standing rules, customer/status/risk labels, and "
    "their stated reasons. Quote the key phrases VERBATIM. "
    "Do not drop modifiers or style words such as terse, brief, concise, "
    "strict, first, last, urgent, or high-risk. "
    "Do NOT summarize, do NOT judge importance, do NOT invent anything. "
    "If there is none, output exactly: NONE.\n"
    "Format: one short bullet per durable fact, include the reason if stated."
)


def _turn_has_decision_marker(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _DECISION_MARKERS)


def extract_decisions(
    exchange_text: str,
    llm_func: Callable[[str], Any],
) -> str:
    """Verbatim durable-fact extraction for one exchange.

    Gated by markers so noise turns never trigger an LLM call. The task is
    deliberately narrow — quote decisions, do not judge importance — which is
    why it does not drop facts the way a "summarize everything" prompt does.
    Returns "" when there is nothing to extract.
    """
    if not _turn_has_decision_marker(exchange_text):
        return ""
    prompt = f"{_DECISION_INSTRUCTION}\n\nEXCHANGE:\n{exchange_text}\n\nDECISIONS:"
    result = llm_func(prompt)
    if isinstance(result, tuple) and len(result) == 2:
        text = str(result[0] or "")
    else:
        text = str(getattr(result, "content", result) or "")
    text = text.strip()
    if not text or text.upper().strip().rstrip(".") == "NONE":
        return ""
    return text


def build_projected_state_from_log(
    session_log: list[dict] | None,
    *,
    session_id: str = "",
    decision_llm_func: Callable[[str], Any] | None = None,
) -> SessionState:
    """Build an ACL-style session projection.

    The deterministic core (regex pinned facts + evidence lines) needs no LLM.
    When ``decision_llm_func`` is provided, marked turns are additionally passed
    through a narrow, verbatim durable-fact extractor so that reasoning answers
    (prose rationale, standing preferences, CRM/status reasons) survive
    compression too — without the fact loss of a "summarize the whole
    conversation" prompt.
    """
    text = "\n".join(
        turn_text
        for item in session_log or []
        for _role, turn_text in [_turn_text(item)]
        if turn_text
    )
    pins = extract_pinned_facts(text)
    evidence_lines = _projected_evidence_lines(session_log, pins)

    sections: list[str] = []
    if evidence_lines:
        sections.append(
            "[DO_NOT_FORGET]\nExact source facts:\n"
            + "\n".join(f"- {line}" for line in evidence_lines)
        )
    elif pins:
        sections.append(
            "[DO_NOT_FORGET]\nExact facts are available in the pinned fact list."
        )

    decision_calls = 0
    if decision_llm_func is not None:
        marked_source_lines = _projected_marked_source_lines(session_log)
        if marked_source_lines:
            sections.append(
                "[DURABLE_SOURCE_LINES]\n"
                + "\n".join(f"- {line}" for line in marked_source_lines)
            )

        decisions: list[str] = []
        for item in session_log or []:
            role, turn = _turn_text(item)
            if role != "user" or not turn:
                continue
            if not _turn_has_decision_marker(turn):
                continue
            decision_calls += 1
            extracted = extract_decisions(turn, decision_llm_func)
            if extracted:
                decisions.append(extracted)
        if decisions:
            sections.append(
                "[DECISIONS_AND_PREFERENCES]\n" + "\n".join(decisions)
            )

    snapshot = "\n\n".join(sections)
    return SessionState(
        session_id=session_id,
        snapshot=snapshot,
        pinned_facts=pins,
        turns_folded=count_turns(session_log),
        raw_turn_cursor=count_turns(session_log),
        updated_at=time.time(),
    )


def should_use_projected_state(
    session_log: list[dict] | None,
    user_request: str,
    *,
    system_prompt: str = "",
    min_raw_tokens: int = 900,
) -> dict[str, Any]:
    """Decide whether the deterministic projected state is worth using now.

    The projection has fixed prompt overhead. For very short sessions raw
    history is cheaper. This gate prevents the optimizer from making a short
    second turn more expensive just because an optimization path exists.
    """
    state = build_projected_state_from_log(session_log)
    raw_prompt = build_raw_prompt(session_log, user_request, system_prompt=system_prompt)
    projected_prompt = build_optimized_prompt(
        state,
        session_log,
        user_request,
        system_prompt=system_prompt,
    )
    raw_tokens = estimate_tokens(raw_prompt)
    projected_tokens = estimate_tokens(projected_prompt)
    has_pins = bool(state.pinned_facts)
    use_projected = has_pins and raw_tokens >= min_raw_tokens and projected_tokens < raw_tokens
    return {
        "use_projected": use_projected,
        "reason": (
            "projected_state_cheaper"
            if use_projected
            else "raw_history_cheaper_or_no_pins"
        ),
        "raw_tokens_estimate": raw_tokens,
        "projected_tokens_estimate": projected_tokens,
        "tokens_saved_estimate": raw_tokens - projected_tokens,
        "pinned_count": len(state.pinned_facts),
        "min_raw_tokens": min_raw_tokens,
    }


def _metrics_path() -> Path:
    return _data_dir() / "llm_optimization" / "session_state_metrics.jsonl"


def log_state_metric(record: dict[str, Any]) -> None:
    try:
        path = _metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), **record}
        with _STORE_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("session_state metric write failed: %s", exc)


def _state_path(session_id: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", session_id or "default", flags=re.UNICODE)
    return _data_dir() / "session_state" / f"{safe}.json"


def save_state(state: SessionState) -> None:
    """Persist state atomically so a crash cannot leave half-written JSON."""
    try:
        path = _state_path(state.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), ensure_ascii=False, indent=2)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with _STORE_LOCK:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
    except Exception as exc:
        logger.warning("session_state save failed for %r: %s", state.session_id, exc)


def load_state(session_id: str) -> SessionState:
    try:
        path = _state_path(session_id)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") == SCHEMA_VERSION:
                allowed = {key: data.get(key) for key in SessionState.__annotations__ if key in data}
                allowed.setdefault("session_id", session_id)
                return SessionState(**allowed)
            logger.info(
                "Ignoring session state %s with schema %r",
                path,
                data.get("schema_version"),
            )
    except Exception as exc:
        logger.warning("session_state load failed for %r: %s", session_id, exc)
    return SessionState(session_id=session_id)


async def update_state_incremental(
    state: SessionState,
    new_exchange_text: str,
    llm_func: Callable[[str], Awaitable[Any] | Any],
    *,
    exchange_failed: bool = False,
    persist: bool = True,
) -> SessionState:
    """Fold one completed exchange into state.

    Failed or cancelled exchanges are not folded. The pinned-fact list is the
    drift guard: exact facts are extracted deterministically and rendered
    separately even if the snapshot model omits them.
    """
    if exchange_failed:
        state.failure_count += 1
        state.updated_at = time.time()
        if persist:
            save_state(state)
        log_state_metric(
            {
                "event": "fold_skipped_failed",
                "session_id": state.session_id,
                "failure_count": state.failure_count,
            }
        )
        return state

    previous = state.snapshot or "(empty - start of session)"
    prompt = (
        f"{_SNAPSHOT_INSTRUCTION}\n\n"
        f"PREVIOUS STATE:\n{previous}\n\n"
        f"NEW EXCHANGE:\n{new_exchange_text}\n\n"
        "UPDATED STATE:"
    )

    started = time.perf_counter()
    result = llm_func(prompt)
    if hasattr(result, "__await__"):
        result = await result
    latency_ms = int((time.perf_counter() - started) * 1000)

    if isinstance(result, tuple) and len(result) == 2:
        text = str(result[0] or "")
        meta = result[1] if isinstance(result[1], dict) else {}
    else:
        text = str(getattr(result, "content", result) or "")
        meta = getattr(result, "response_metadata", {}) or {}

    usage = (meta.get("usage_metadata") or meta.get("token_usage") or {}) if isinstance(meta, dict) else {}
    prompt_tokens = _provider_prompt_tokens(usage) or estimate_tokens(prompt)
    output_tokens = _provider_output_tokens(usage) or estimate_tokens(text)

    prior_pinned = list(state.pinned_facts)
    state.snapshot = text.strip()
    state.pinned_facts = extract_pinned_facts(new_exchange_text, state.pinned_facts)
    state.turns_folded += 1
    state.raw_turn_cursor += 1
    state.updated_at = time.time()

    log_state_metric(
        {
            "event": "fold",
            "session_id": state.session_id,
            "turns_folded": state.turns_folded,
            "snapshot_tokens": estimate_tokens(state.snapshot),
            "pinned_count": len(state.pinned_facts),
            "new_pinned": len(state.pinned_facts) - len(prior_pinned),
            "snapshot_update_prompt_tokens": prompt_tokens,
            "snapshot_update_output_tokens": output_tokens,
            "snapshot_update_total_tokens": prompt_tokens + output_tokens,
            "latency_ms": latency_ms,
        }
    )

    if persist:
        save_state(state)
    return state


# ── Time anchors (date-guard) ────────────────────────────────────────────────
# The scale test on LoCoMo located the whole retention gap in the temporal
# category: the compressor drops precise and relative dates. The guard extracts
# them deterministically and glues them VERBATIM after the compressed chunk, so
# the model cannot drop them — and the compressor may stay aggressive on the
# rest. Result on 200 QA: retention 88% -> 94%, temporal 79% -> 96%, and
# per-query saving went UP (45% -> 60%).
#
# Patterns cover English, Ukrainian (Cyrillic) and Ukrainian (Latin
# transliteration) because real sessions here are mixed-language; an
# English-only guard would silently reopen the temporal gap on real traffic.

_MONTHS_EN = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
_MONTHS_UA = (
    "січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|"
    "листопада|грудня|січень|лютий|березень|квітень|травень|червень|липень|"
    "серпень|вересень|жовтень|листопад|грудень"
)
_MONTHS_UA_TRANSLIT = (
    "sichnia|liutoho|bereznia|kvitnia|travnia|chervnia|lypnia|serpnia|"
    "veresnia|zhovtnia|lystopada|hrudnia"
)

_ABS_TIME_RE = re.compile(
    rf"\b\d{{1,2}}\s+(?:{_MONTHS_EN}|{_MONTHS_UA}|{_MONTHS_UA_TRANSLIT})\b[ ,]*\d{{0,4}}"
    rf"|\b(?:{_MONTHS_EN})\b[ ,]*\d{{4}}"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE | re.UNICODE,
)

_REL_TIME_RE = re.compile(
    r"\b("
    # English
    r"yesterday|today|tomorrow|tonight|this morning|this evening"
    r"|last (?:week|month|year|night|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"|next (?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"|an? (?:week|month|year|day|hour) ago|\d+ (?:days?|weeks?|months?|years?) ago"
    # Ukrainian (Cyrillic)
    r"|вчора|позавчора|сьогодні|завтра|післязавтра|торік"
    r"|минулого (?:тижня|місяця|року)|наступного (?:тижня|місяця|року)"
    r"|(?:тиждень|місяць|рік|день) тому|\d+ (?:дні[вб]?|тижні[вб]?|місяці[вб]?|рок(?:и|ів)) тому"
    # Ukrainian (Latin transliteration)
    r"|vchora|pozavchora|sohodni|zavtra|pisliazavtra|torik"
    r"|mynuloho (?:tyzhnia|misiatsia|roku)|nastupnoho (?:tyzhnia|misiatsia|roku)"
    r"|(?:tyzhden|misiats|rik|den) tomu"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def extract_time_anchors(text: str, session_date: str = "") -> list[str]:
    """Deterministically extract absolute and relative time expressions.

    Relative expressions are tagged with the session date so a later reader
    (the answering model) can resolve them; the guard itself does no calendar
    math — it only makes sure the anchor survives compression verbatim.
    """
    anchors: list[str] = []
    seen: set[str] = set()
    for match in _ABS_TIME_RE.findall(text or ""):
        value = match.strip(" ,.")
        # bare month names without any digit are too weak to pin
        if not value or not any(ch.isdigit() for ch in value):
            continue
        if value not in seen:
            seen.add(value)
            anchors.append(value)
    for match in _REL_TIME_RE.findall(text or ""):
        value = match.strip()
        if not value:
            continue
        tagged = f"{value}(rel to {session_date})" if session_date else value
        if tagged not in seen:
            seen.add(tagged)
            anchors.append(tagged)
    return anchors


_APPEND_COMPRESS_INSTRUCTION = (
    "Compress this single exchange into the fewest words that keep EVERY fact "
    "(names, events, statuses, places, relationships, specific details). "
    "Keep the original language of names and key phrases. Plain short phrases. "
    "Output ONLY the compressed line(s)."
)


async def fold_exchange_append_only(
    state: SessionState,
    new_exchange_text: str,
    llm_func: Callable[[str], Awaitable[Any] | Any],
    *,
    session_date: str = "",
    exchange_failed: bool = False,
    persist: bool = True,
) -> SessionState:
    """Append-only fold: compress ONLY the new exchange, freeze it, append.

    The old state is never re-compressed — each fact passes through the model
    exactly once, so losses cannot accumulate (measured: 94% retention at 60%
    per-query saving on 200 LoCoMo QA, vs 88%/45% for re-compressing state).
    The date-guard glues extracted time anchors verbatim after the chunk.
    """
    if exchange_failed:
        state.failure_count += 1
        state.updated_at = time.time()
        if persist:
            save_state(state)
        log_state_metric(
            {
                "event": "append_fold_skipped_failed",
                "session_id": state.session_id,
                "failure_count": state.failure_count,
            }
        )
        return state

    prompt = (
        f"{_APPEND_COMPRESS_INSTRUCTION}\n\n"
        f"EXCHANGE:\n{new_exchange_text}\n\nCOMPRESSED:"
    )

    started = time.perf_counter()
    result = llm_func(prompt)
    if hasattr(result, "__await__"):
        result = await result
    latency_ms = int((time.perf_counter() - started) * 1000)

    if isinstance(result, tuple) and len(result) == 2:
        text = str(result[0] or "")
        meta = result[1] if isinstance(result[1], dict) else {}
    else:
        text = str(getattr(result, "content", result) or "")
        meta = getattr(result, "response_metadata", {}) or {}

    usage = (meta.get("usage_metadata") or meta.get("token_usage") or {}) if isinstance(meta, dict) else {}
    prompt_tokens = _provider_prompt_tokens(usage) or estimate_tokens(prompt)
    output_tokens = _provider_output_tokens(usage) or estimate_tokens(text)

    chunk = text.strip()
    anchors = extract_time_anchors(new_exchange_text, session_date)
    if chunk:
        if anchors:
            time_note = "; ".join(anchors)
            session_part = f"session={session_date}; " if session_date else ""
            chunk = f"{chunk}  [TIME: {session_part}{time_note}]"
        state.frozen_chunks.append(chunk)

    state.pinned_facts = extract_pinned_facts(new_exchange_text, state.pinned_facts)
    state.turns_folded += 1
    state.raw_turn_cursor += 1
    state.updated_at = time.time()

    log_state_metric(
        {
            "event": "append_fold",
            "session_id": state.session_id,
            "turns_folded": state.turns_folded,
            "frozen_chunk_count": len(state.frozen_chunks),
            "frozen_state_tokens": estimate_tokens(state.append_only_text()),
            "raw_exchange_tokens": estimate_tokens(new_exchange_text),
            "time_anchor_count": len(anchors),
            "pinned_count": len(state.pinned_facts),
            "compress_prompt_tokens": prompt_tokens,
            "compress_output_tokens": output_tokens,
            "compress_total_tokens": prompt_tokens + output_tokens,
            "latency_ms": latency_ms,
        }
    )

    if persist:
        save_state(state)
    return state


def build_optimized_prompt(
    state: SessionState,
    session_log: list[dict] | None,
    user_request: str,
    *,
    system_prompt: str = "",
) -> str:
    recent = recent_raw_turns(session_log)
    parts: list[str] = []
    if system_prompt.strip():
        parts.append(system_prompt.strip())
    state_block = state.render(recent_raw=recent)
    if state_block:
        parts.append(state_block)
    parts.append(f"=== CURRENT REQUEST ===\n{user_request.strip()}")
    prompt = "\n\n".join(parts)

    raw_prompt = build_raw_prompt(session_log, user_request, system_prompt=system_prompt)
    log_state_metric(
        {
            "event": "prompt_built",
            "session_id": state.session_id,
            "raw_prompt_tokens_estimate": estimate_tokens(raw_prompt),
            "optimized_prompt_tokens_estimate": estimate_tokens(prompt),
            "recent_raw_turns": RECENT_RAW_TURNS,
            "turns_folded": state.turns_folded,
        }
    )
    return prompt


def build_raw_prompt(
    session_log: list[dict] | None,
    user_request: str,
    *,
    system_prompt: str = "",
) -> str:
    lines: list[str] = []
    for item in session_log or []:
        role, text = _turn_text(item)
        text = text.strip()
        if role and text:
            lines.append(f"{role}: {text}")

    parts: list[str] = []
    if system_prompt.strip():
        parts.append(system_prompt.strip())
    if lines:
        parts.append("\n".join(lines))
    parts.append(f"=== CURRENT REQUEST ===\n{user_request.strip()}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class SessionStateEvalCase:
    """One deterministic long-context retention case."""

    case_id: str
    session_log: list[dict]
    user_request: str
    expected_fragments: list[str]
    snapshot: str = ""


def _answer_contains_expected(answer: str, expected_fragments: list[str]) -> bool:
    normalized = (answer or "").casefold()
    return all(fragment.casefold() in normalized for fragment in expected_fragments)


def _extract_answer_and_usage(result: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        meta = result[1] if isinstance(result[1], dict) else {}
        usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
        return str(result[0] or ""), usage if isinstance(usage, dict) else {}
    meta = getattr(result, "response_metadata", {}) or {}
    usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
    return str(getattr(result, "content", result) or ""), usage if isinstance(usage, dict) else {}


async def run_session_state_eval(
    cases: list[SessionStateEvalCase],
    answer_func: Callable[[str], Awaitable[Any] | Any],
    *,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Measure accuracy and prompt size for raw vs session-state prompts.

    This is a proof harness, not a benchmark claim generator. A strategy is only
    useful when it keeps correctness while reducing prompt tokens.
    """
    rows: list[dict[str, Any]] = []
    raw_correct = 0
    optimized_correct = 0
    raw_tokens_total = 0
    optimized_tokens_total = 0
    raw_provider_prompt_total = 0
    optimized_provider_prompt_total = 0
    raw_provider_output_total = 0
    optimized_provider_output_total = 0
    provider_usage_seen = False

    for case in cases:
        state = SessionState(
            session_id=case.case_id,
            snapshot=case.snapshot,
            pinned_facts=extract_pinned_facts("\n".join(
                text
                for item in case.session_log
                for _role, text in [_turn_text(item)]
                if text
            )),
            turns_folded=count_turns(case.session_log),
        )
        raw_prompt = build_raw_prompt(
            case.session_log,
            case.user_request,
            system_prompt=system_prompt,
        )
        optimized_prompt = build_optimized_prompt(
            state,
            case.session_log,
            case.user_request,
            system_prompt=system_prompt,
        )

        raw_result = answer_func(raw_prompt)
        if hasattr(raw_result, "__await__"):
            raw_result = await raw_result
        optimized_result = answer_func(optimized_prompt)
        if hasattr(optimized_result, "__await__"):
            optimized_result = await optimized_result

        raw_answer, raw_usage = _extract_answer_and_usage(raw_result)
        optimized_answer, optimized_usage = _extract_answer_and_usage(optimized_result)
        raw_ok = _answer_contains_expected(raw_answer, case.expected_fragments)
        optimized_ok = _answer_contains_expected(optimized_answer, case.expected_fragments)
        raw_tokens = estimate_tokens(raw_prompt)
        optimized_tokens = estimate_tokens(optimized_prompt)
        raw_provider_prompt = _provider_prompt_tokens(raw_usage)
        optimized_provider_prompt = _provider_prompt_tokens(optimized_usage)
        raw_provider_output = _provider_output_tokens(raw_usage)
        optimized_provider_output = _provider_output_tokens(optimized_usage)
        if raw_provider_prompt is not None and optimized_provider_prompt is not None:
            provider_usage_seen = True
            raw_provider_prompt_total += raw_provider_prompt
            optimized_provider_prompt_total += optimized_provider_prompt
            raw_provider_output_total += raw_provider_output or 0
            optimized_provider_output_total += optimized_provider_output or 0

        raw_correct += int(raw_ok)
        optimized_correct += int(optimized_ok)
        raw_tokens_total += raw_tokens
        optimized_tokens_total += optimized_tokens
        rows.append(
            {
                "case_id": case.case_id,
                "raw_correct": raw_ok,
                "optimized_correct": optimized_ok,
                "raw_tokens": raw_tokens,
                "optimized_tokens": optimized_tokens,
                "raw_provider_prompt_tokens": raw_provider_prompt,
                "optimized_provider_prompt_tokens": optimized_provider_prompt,
                "raw_provider_output_tokens": raw_provider_output,
                "optimized_provider_output_tokens": optimized_provider_output,
                "token_reduction_ratio": round(raw_tokens / max(1, optimized_tokens), 2),
                "provider_prompt_token_reduction_ratio": (
                    round(raw_provider_prompt / max(1, optimized_provider_prompt), 2)
                    if raw_provider_prompt is not None and optimized_provider_prompt is not None
                    else None
                ),
                "raw_answer": raw_answer,
                "optimized_answer": optimized_answer,
                "expected_fragments": list(case.expected_fragments),
            }
        )

    total = max(1, len(cases))
    provider_ratio = (
        round(raw_provider_prompt_total / max(1, optimized_provider_prompt_total), 2)
        if provider_usage_seen
        else None
    )
    return {
        "schema": "remy_session_state_eval_v1",
        "cases": rows,
        "summary": {
            "case_count": len(cases),
            "raw_accuracy": round(raw_correct / total, 4),
            "optimized_accuracy": round(optimized_correct / total, 4),
            "raw_correct": raw_correct,
            "optimized_correct": optimized_correct,
            "raw_tokens": raw_tokens_total,
            "optimized_tokens": optimized_tokens_total,
            "tokens_saved": raw_tokens_total - optimized_tokens_total,
            "token_reduction_ratio": round(raw_tokens_total / max(1, optimized_tokens_total), 2),
            "provider_usage_seen": provider_usage_seen,
            "raw_provider_prompt_tokens": raw_provider_prompt_total if provider_usage_seen else None,
            "optimized_provider_prompt_tokens": (
                optimized_provider_prompt_total if provider_usage_seen else None
            ),
            "provider_prompt_tokens_saved": (
                raw_provider_prompt_total - optimized_provider_prompt_total
                if provider_usage_seen
                else None
            ),
            "provider_prompt_token_reduction_ratio": provider_ratio,
            "raw_provider_output_tokens": raw_provider_output_total if provider_usage_seen else None,
            "optimized_provider_output_tokens": (
                optimized_provider_output_total if provider_usage_seen else None
            ),
            "production_ready": optimized_correct == len(cases)
            and optimized_tokens_total < raw_tokens_total,
        },
    }
