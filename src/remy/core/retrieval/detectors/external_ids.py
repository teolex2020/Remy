"""Detect external identifiers: arXiv IDs, DOIs, authoritative URLs."""

from __future__ import annotations

import re

from remy.core.retrieval.claim_spans import ClaimSpan, EvidenceRequirement


_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>)]+", re.I)
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.I)


def detect_external_ids(text: str) -> list[ClaimSpan]:
    spans: list[ClaimSpan] = []

    for m in _ARXIV_RE.finditer(text):
        arxiv_id = m.group(1)
        spans.append(
            ClaimSpan(
                text=m.group(0),
                span=m.span(),
                claim_type="arxiv_id",
                requires_evidence=EvidenceRequirement.TOOL_CALL_WITH_ID,
                detector="external_ids",
                entity_hint=arxiv_id,
                context_window=_window(text, m.start(), m.end()),
            )
        )

    for m in _DOI_RE.finditer(text):
        spans.append(
            ClaimSpan(
                text=m.group(0),
                span=m.span(),
                claim_type="doi",
                requires_evidence=EvidenceRequirement.TOOL_CALL_WITH_ID,
                detector="external_ids",
                entity_hint=m.group(0).lower().rstrip(".,;)"),
                context_window=_window(text, m.start(), m.end()),
            )
        )

    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;)]")
        spans.append(
            ClaimSpan(
                text=url,
                span=(m.start(), m.start() + len(url)),
                claim_type="url_authoritative",
                requires_evidence=EvidenceRequirement.TOOL_CALL_WITH_ID,
                detector="external_ids",
                entity_hint=url,
                context_window=_window(text, m.start(), m.end()),
            )
        )

    return spans


def _window(text: str, start: int, end: int, radius: int = 40) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b]
