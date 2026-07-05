"""Runtime factuality and verification guard for user-facing responses.

The guard stays lightweight and deterministic. It uses session-log evidence
first, then applies small textual heuristics to classify risky claims in the
final answer before the answer is shown to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re


# Tools whose successful call = fetched external evidence was gathered this turn.
# Candidate discovery alone (web_search/search_web) is not enough.
_EXTERNAL_EVIDENCE_TOOLS = frozenset(
    {
        "browse_page",
        "browser_act",
        "http_get",
        "extract_content",
        "fetch_url",
    }
)

# Tools whose successful execution proves a write action happened.
_ACTION_EVIDENCE_TOOLS = frozenset(
    {
        "store",
        "update_record",
        "delete_record",
        "mark_stale",
        "store_person",
        "store_research",
        "store_story",
        "verify_record",
        "schedule_task",
        "browse_page",
        "browser_act",
        "http_get",
        "write_file",
        "sandbox_create_tool",
        "sandbox_test_tool",
        "delegate_task",
        "memory_feedback",
        "aura_cognitive_ops",
        "get_corrections",
        "deprecate_belief",
        "deprecate_belief_with_reason",
    }
)

_URL_RE = re.compile(r"https?://[^\s<>\]\)\"',]+", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CURRENT_RE = re.compile(
    r"\b(latest|current|currently|right now|today|this week|this month|202[0-9])\b",
    re.IGNORECASE,
)
_OBSERVED_RE = re.compile(
    r"\b(i\s+(?:just\s+)?(?:checked|reviewed|looked at|verified|confirmed|inspected|found))\b",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"\b(based on (?:our|this) conversation|from our conversation|you (?:prefer|usually|tend to))\b",
    re.IGNORECASE,
)
_INFERENCE_RE = re.compile(
    r"\b(it seems|it looks like|likely|probably|i suspect|i infer)\b",
    re.IGNORECASE,
)
_RISKY_RESEARCH_RE = re.compile(
    r"\b("
    r"market|tam|sam|som|cagr|pricing|price|valuation|benchmark|accuracy|contract|"
    r"revenue|profit|roi|yield|market research|deep research|verified in this turn|"
    r"according to research|competitor|2026"
    r")\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?:\$|€|£)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|b|bn|million|billion))?|\b\d+(?:\.\d+)?\s?%|\b\d[\d,]*(?:\.\d+)?\b",
    re.IGNORECASE,
)
_SOURCE_NOTE_RE = re.compile(
    r"(?:\n\s*)*Source note:\s*I used external tools in this turn, but no stable source link "
    r"was captured in the tool output, so treat specific external facts as provisional "
    r"until I provide a direct link\.\s*",
    re.IGNORECASE,
)
_SOURCE_NOTE_TEXT = (
    "Source note: I used external tools in this turn, but no stable source link "
    "was captured in the tool output, so treat specific external facts as provisional "
    "until I provide a direct link."
)
_MEMORY_DOWNGRADE_TEXT = (
    "I have not directly inspected that artifact in this turn. "
    "Based on the stored context and this conversation, treat that statement as an inference, not a verified observation."
)
_UNVERIFIED_CURRENT_NOTE = (
    "I have not verified the current external state in this turn, so treat any latest/current "
    "details here as provisional until I check a live source."
)
_UNVERIFIED_RESEARCH_NOTE = (
    "This answer includes market/commercial or numeric claims without live verification in this turn. "
    "Treat those claims as hypotheses, not facts, until I cite sources."
)
_UNSUPPORTED_BINDING_NOTE = (
    "Parts of this answer are not backed by the recalled evidence from this turn. "
    "Treat unsupported statements as hypotheses until I cite or recall stronger evidence."
)
_UNGROUNDED_ANSWER_NOTE = (
    "I do not have an evidence anchor for this answer in the current turn. "
    "Treat it as an unverified draft, not as fact, until I attach source, tool, memory, or calculation evidence."
)
_RECALL_ID_RE = re.compile(r"\[id:([^\]]+)\]")
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]{3,}")


@dataclass
class ResponseClaim:
    text: str
    claim_class: str
    supported: bool
    supporting_record_ids: list[str] = field(default_factory=list)


@dataclass
class FactualityReport:
    unsupported_observed_claims: int = 0
    unverified_current_claims: int = 0
    unsupported_claims_total: int = 0
    supported_claims_total: int = 0
    had_external_evidence: bool = False
    modified: bool = False
    citations_added: int = 0
    missing_source_links: bool = False
    claim_counts: dict[str, int] = field(default_factory=dict)
    evidence_record_ids: list[str] = field(default_factory=list)
    claim_details: list[ResponseClaim] = field(default_factory=list)
    # External-claim verification (set by enforce_factuality via
    # external_claim_verifier). Keeps the structural truth separate from the
    # token-overlap `supported` used by memory/inference bucketing.
    external_citations_total: int = 0
    external_citations_grounded: int = 0
    external_citations_phantom: int = 0          # unverified + placeholder + dead
    external_citations_details: list[dict] = field(default_factory=list)
    external_citations_banner: str = ""
    brain_storage_unsafe: bool = False           # set when phantom ratio too high
    # Split evidence counts — never collapse into one "supported" number.
    # supported_internal:        memory_fact/inference bound to recalled records
    # supported_external_verified: external claims structurally grounded in this turn's tools
    # unverified_external:       external claims that failed structural verification (phantom)
    # unsupported:               claims that neither recall nor tools back
    supported_internal: int = 0
    supported_external_verified: int = 0
    unverified_external: int = 0
    unsupported: int = 0
    ungrounded_answer_claims: int = 0


@dataclass
class ActionClaimViolation:
    tools_called: list[str]
    tools_missing: list[str]
    zero_action_tools: bool


def _has_external_evidence(session_log: list) -> bool:
    for entry in session_log or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "tool_call":
            continue
        if entry.get("tool") in _EXTERNAL_EVIDENCE_TOOLS:
            result = str(entry.get("result_full") or entry.get("result", "") or "")
            if result and "error" not in result[:120].lower():
                return True
    return False


def _extract_source_links(session_log: list) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for entry in session_log or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "tool_call":
            continue
        if entry.get("tool") not in _EXTERNAL_EVIDENCE_TOOLS:
            continue
        for key in ("args_full", "result_full", "args", "result"):
            raw = entry.get(key)
            if raw is None:
                continue
            text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
            for match in _URL_RE.findall(text):
                if match not in seen:
                    seen.add(match)
                    links.append(match)
    return links


def _iter_sentences(response_text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(response_text or "")]
    return [part for part in parts if part]


def _tokenize(text: str) -> set[str]:
    normalized: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        tok = raw.lower()
        normalized.add(tok)
        if len(tok) > 4 and tok.endswith("s"):
            normalized.add(tok[:-1])
    return normalized


def _extract_recall_evidence(session_log: list) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in session_log or []:
        if not isinstance(entry, dict) or entry.get("type") != "tool_call":
            continue
        if entry.get("tool") not in {"recall", "recall_full"}:
            continue
        result = str(entry.get("result", "") or "")
        for raw_line in result.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = _RECALL_ID_RE.search(line)
            if not match:
                continue
            record_id = match.group(1).strip()
            content = _RECALL_ID_RE.sub("", line)
            content = re.sub(r"\[[^\]]+\]", "", content).strip()
            if not record_id or not content:
                continue
            key = f"{record_id}:{content[:80].lower()}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append({"record_id": record_id, "content": content})
    return evidence


def _bind_claim_to_evidence(text: str, evidence: list[dict[str, str]]) -> list[str]:
    claim_tokens = _tokenize(text)
    if not claim_tokens:
        return []

    supporting: list[str] = []
    for item in evidence:
        overlap = claim_tokens & _tokenize(item.get("content", ""))
        if len(overlap) >= 2:
            supporting.append(item["record_id"])
            continue
        if len(overlap) == 1:
            token = next(iter(overlap))
            if len(token) >= 7:
                supporting.append(item["record_id"])
    return supporting[:5]


def _has_turn_evidence(session_log: list) -> bool:
    return _has_external_evidence(session_log) or bool(_extract_recall_evidence(session_log))


def _is_substantive_ungrounded_output(response_text: str) -> bool:
    """Structural fallback: large no-evidence output is still a claim surface.

    Avoid domain words and synonym rings here. This only checks whether the
    mouth emitted enough text that it must carry an evidence anchor or be
    downgraded to draft/hypothesis.
    """
    text = (response_text or "").strip()
    if len(text) < 120:
        return False
    token_count = len(_TOKEN_RE.findall(text))
    substantive = [
        sentence
        for sentence in _iter_sentences(text)
        if len(_TOKEN_RE.findall(sentence)) >= 5
    ]
    return token_count >= 24 and len(substantive) >= 2


def _classify_sentence(sentence: str, had_external_evidence: bool) -> ResponseClaim | None:
    if not sentence or len(sentence) < 8:
        return None

    text = sentence.strip()
    lower = text.lower()

    if _OBSERVED_RE.search(text):
        return ResponseClaim(
            text=text,
            claim_class="observed_fact",
            supported=had_external_evidence,
        )

    if _MEMORY_RE.search(text):
        return ResponseClaim(text=text, claim_class="memory_fact", supported=True)

    if _INFERENCE_RE.search(text):
        return ResponseClaim(text=text, claim_class="inference", supported=True)

    risky_numeric = bool(_NUMBER_RE.search(text)) and bool(_CURRENT_RE.search(text) or _RISKY_RESEARCH_RE.search(text))
    risky_research = bool(_RISKY_RESEARCH_RE.search(text)) and bool(_NUMBER_RE.search(text) or "research" in lower)

    if risky_numeric or risky_research:
        return ResponseClaim(
            text=text,
            claim_class="unverified_current_fact",
            supported=had_external_evidence,
        )

    if _CURRENT_RE.search(text):
        return ResponseClaim(
            text=text,
            claim_class="unverified_current_fact",
            supported=had_external_evidence,
        )

    return None


def classify_response_claims(response_text: str, session_log: list) -> list[ResponseClaim]:
    """Classify the answer into a few risk-oriented claim buckets."""
    had_external_evidence = _has_external_evidence(session_log)
    claims: list[ResponseClaim] = []
    for sentence in _iter_sentences(response_text):
        claim = _classify_sentence(sentence, had_external_evidence)
        if claim is not None:
            claims.append(claim)
    return claims


def detect_unsupported_observed_claims(response_text: str, session_log: list) -> FactualityReport:
    """Inspect final answer claims against turn-level evidence."""
    report = FactualityReport(had_external_evidence=_has_external_evidence(session_log))
    if not response_text or len(response_text) < 20:
        return report

    claims = classify_response_claims(response_text, session_log)
    recall_evidence = _extract_recall_evidence(session_log)
    report.evidence_record_ids = [item["record_id"] for item in recall_evidence]
    for claim in claims:
        if claim.claim_class in {"memory_fact", "inference"}:
            claim.supporting_record_ids = _bind_claim_to_evidence(claim.text, recall_evidence)
            claim.supported = bool(claim.supporting_record_ids)
        report.claim_counts[claim.claim_class] = report.claim_counts.get(claim.claim_class, 0) + 1
        if claim.claim_class == "observed_fact" and not claim.supported:
            report.unsupported_observed_claims += 1
        if claim.claim_class == "unverified_current_fact" and not claim.supported:
            report.unverified_current_claims += 1
        if claim.supported:
            report.supported_claims_total += 1
            # memory_fact + inference bound to recall = supported_internal.
            # observed_fact/unverified_current_fact are "supported" only via
            # had_external_evidence, which is a turn-level signal; those are
            # counted in supported_external_verified after the verifier runs.
            if claim.claim_class in {"memory_fact", "inference"}:
                report.supported_internal += 1
        else:
            report.unsupported_claims_total += 1
            report.unsupported += 1
    report.claim_details = claims

    if (
        not claims
        and len(response_text) >= 60
        and not report.had_external_evidence
        and (_CURRENT_RE.search(response_text) or _RISKY_RESEARCH_RE.search(response_text))
    ):
        report.unverified_current_claims = 1
        report.claim_counts["unverified_current_fact"] = 1

    if not claims and not _has_turn_evidence(session_log) and _is_substantive_ungrounded_output(response_text):
        claim = ResponseClaim(
            text=response_text.strip()[:240],
            claim_class="ungrounded_answer",
            supported=False,
        )
        report.claim_details = [claim]
        report.claim_counts["ungrounded_answer"] = 1
        report.unsupported_claims_total = 1
        report.unsupported = 1
        report.ungrounded_answer_claims = 1

    return report


def _has_citations(response_text: str) -> bool:
    return bool(response_text and _URL_RE.search(response_text))


def _append_sources(response_text: str, source_links: list[str], report: FactualityReport) -> str:
    limited = source_links[:5]
    if not limited:
        return response_text
    if any(url in response_text for url in limited):
        return response_text
    sources_block = "\n\nSources:\n" + "\n".join(f"- {url}" for url in limited)
    report.citations_added = len(limited)
    return response_text.rstrip() + sources_block


def _dedupe_source_notes(response_text: str) -> str:
    if not response_text:
        return response_text
    matches = list(_SOURCE_NOTE_RE.finditer(response_text))
    if not matches:
        return response_text
    cleaned = _SOURCE_NOTE_RE.sub("", response_text).rstrip()
    return cleaned + "\n\n" + _SOURCE_NOTE_TEXT


def _rewrite_unsupported_observed_claims(response_text: str) -> str:
    stripped = response_text.strip()
    if _MEMORY_DOWNGRADE_TEXT in stripped:
        return stripped
    return _MEMORY_DOWNGRADE_TEXT


def _rewrite_unverified_current_claims(response_text: str) -> str:
    stripped = response_text.strip()
    note = _UNVERIFIED_RESEARCH_NOTE if _RISKY_RESEARCH_RE.search(stripped) or _NUMBER_RE.search(stripped) else _UNVERIFIED_CURRENT_NOTE
    if note in stripped:
        return stripped
    return f"{note}\n\nUnverified draft:\n{stripped}"


def _rewrite_unsupported_binding_claims(response_text: str) -> str:
    stripped = response_text.strip()
    if _UNSUPPORTED_BINDING_NOTE in stripped:
        return stripped
    return f"{_UNSUPPORTED_BINDING_NOTE}\n\nDraft:\n{stripped}"


def _rewrite_ungrounded_answer(response_text: str) -> str:
    stripped = response_text.strip()
    if _UNGROUNDED_ANSWER_NOTE in stripped:
        return stripped
    return f"{_UNGROUNDED_ANSWER_NOTE}\n\nUnverified draft:\n{stripped}"


def _format_claim_line(text: str, record_ids: list[str] | None = None) -> str:
    suffix = ""
    if record_ids:
        suffix = f" [evidence: {', '.join(record_ids[:3])}]"
    return f"- {text}{suffix}"


def _rewrite_structured_claims(response_text: str, report: FactualityReport) -> str:
    facts: list[str] = []
    inferences: list[str] = []
    unknowns: list[str] = []
    needs_verification: list[str] = []

    for claim in report.claim_details:
        if claim.claim_class == "unverified_current_fact":
            needs_verification.append(_format_claim_line(claim.text))
            continue
        if claim.supported:
            if claim.claim_class == "memory_fact":
                facts.append(_format_claim_line(claim.text, claim.supporting_record_ids))
            else:
                inferences.append(_format_claim_line(claim.text, claim.supporting_record_ids))
            continue
        if claim.claim_class == "inference":
            unknowns.append(_format_claim_line(claim.text))
        elif claim.claim_class == "observed_fact":
            needs_verification.append(_format_claim_line(claim.text))
        else:
            unknowns.append(_format_claim_line(claim.text))

    if not (facts or inferences or unknowns or needs_verification):
        return response_text

    parts = ["Structured answer:"]
    if facts:
        parts.append("Facts:")
        parts.extend(facts)
    if inferences:
        parts.append("Inferences:")
        parts.extend(inferences)
    if unknowns:
        parts.append("Unknowns:")
        parts.extend(unknowns)
    if needs_verification:
        parts.append("Needs verification:")
        parts.extend(needs_verification)
    return "\n".join(parts)


def summarize_claim_details(report: FactualityReport | None) -> list[dict]:
    """Return JSON-friendly claim summaries for logs, UI, and evaluation."""
    if report is None:
        return []
    return [
        {
            "text": claim.text,
            "claim_class": claim.claim_class,
            "supported": claim.supported,
            "supporting_record_ids": list(claim.supporting_record_ids),
        }
        for claim in report.claim_details
    ]


def enforce_factuality(
    response_text: str,
    session_log: list,
    *,
    locale: str = "en",
    live_citation_check: bool = False,
    session_id: str = "",
) -> tuple[str, FactualityReport]:
    """Apply factuality corrections to response based on session-log evidence."""
    report = detect_unsupported_observed_claims(response_text, session_log)
    if not response_text:
        return response_text, report

    # ── External-claim verification (phantom URL/DOI/arxiv detection) ──
    # Runs *before* rewriting so the structural banner lands at the bottom
    # of the final answer and so downstream brain.store() can consult
    # report.brain_storage_unsafe.
    try:
        from remy.core.external_claim_verifier import (
            verify_external_claims,
            render_banner,
        )
        ext = verify_external_claims(
            response_text,
            session_log,
            live_check=live_citation_check,
        )
        report.external_citations_total = ext.total
        report.external_citations_grounded = ext.grounded_count
        report.external_citations_phantom = ext.phantom_count
        report.external_citations_details = [c.to_dict() for c in ext.citations]
        # Feed the split evidence-count axis. These are independent of the
        # recall-bound supported_internal bucket: the same response can have
        # N internal supports *and* M verified external citations.
        report.supported_external_verified += ext.grounded_count
        report.unverified_external += ext.phantom_count
        banner = render_banner(ext, locale=locale)
        if banner:
            report.external_citations_banner = banner
        # Block auto-store when half+ of citations are phantom, or the
        # response itself contains placeholder markers like "умовне посилання".
        if ext.total > 0 and ext.phantom_count / ext.total >= 0.5:
            report.brain_storage_unsafe = True
        elif ext.phantom_text_markers:
            report.brain_storage_unsafe = True
        # Publish the signal so the brain.store() gate can see it during
        # subsequent tool calls in the same turn.
        if session_id:
            try:
                from remy.core.claim_provenance import record_turn_factuality_signal
                record_turn_factuality_signal(
                    session_id,
                    phantom_count=ext.phantom_count,
                    external_total=ext.total,
                    phantom_text_markers=bool(ext.phantom_text_markers),
                    brain_storage_unsafe=report.brain_storage_unsafe,
                )
            except Exception:
                pass
    except Exception:
        # Never let verifier errors mask the assistant reply.
        pass

    corrected = response_text
    source_links = _extract_source_links(session_log)
    has_current_claims = any(
        claim.claim_class == "unverified_current_fact" for claim in report.claim_details
    )

    if report.unsupported_observed_claims > 0:
        corrected = _rewrite_unsupported_observed_claims(corrected)

    if report.unverified_current_claims > 0 and not report.had_external_evidence:
        corrected = _rewrite_unverified_current_claims(corrected)
    elif report.ungrounded_answer_claims > 0 and not report.had_external_evidence:
        corrected = _rewrite_ungrounded_answer(corrected)
    elif report.supported_claims_total + report.unsupported_claims_total > 0:
        total = report.supported_claims_total + report.unsupported_claims_total
        unsupported_ratio = report.unsupported_claims_total / total
        has_mixed_claims = report.supported_claims_total > 0 and report.unsupported_claims_total > 0
        has_verification_bucket = report.unverified_current_claims > 0
        if has_mixed_claims or has_verification_bucket:
            corrected = _rewrite_structured_claims(corrected, report)
        if report.unsupported_claims_total >= 2 and unsupported_ratio >= 0.5:
            corrected = _rewrite_unsupported_binding_claims(corrected)

    if report.had_external_evidence:
        if has_current_claims and not source_links:
            corrected = _rewrite_structured_claims(corrected, report)
        if not _has_citations(corrected):
            if source_links:
                corrected = _append_sources(corrected, source_links, report)
            else:
                report.missing_source_links = True
                if "Source note:" not in corrected:
                    corrected = corrected.rstrip() + "\n\n" + _SOURCE_NOTE_TEXT
        corrected = _dedupe_source_notes(corrected)
        report.modified = corrected != response_text
        return corrected, report

    corrected = _dedupe_source_notes(corrected)
    report.modified = corrected != response_text
    return corrected, report


def check_action_claims(response_text: str, session_log: list) -> list[ActionClaimViolation]:
    """Detect turns where agent likely claimed actions without performing them."""
    if not response_text or len(response_text) < 80:
        return []

    tool_calls = [
        e for e in (session_log or [])
        if isinstance(e, dict) and e.get("type") == "tool_call"
    ]
    if not tool_calls:
        return []

    called: set[str] = set()
    successful_writes: set[str] = set()

    for entry in tool_calls:
        tool = entry.get("tool", "")
        if not tool:
            continue
        called.add(tool)
        result = str(entry.get("result", "") or "")
        is_error = "error" in result[:100].lower() or result.startswith("Error:")
        if tool in _ACTION_EVIDENCE_TOOLS and not is_error:
            successful_writes.add(tool)

    if successful_writes:
        return []

    read_only_called = called - _ACTION_EVIDENCE_TOOLS
    if not read_only_called:
        return []

    return [
        ActionClaimViolation(
            tools_called=sorted(called),
            tools_missing=["store", "update_record", "aura_cognitive_ops", "..."],
            zero_action_tools=True,
        )
    ]
