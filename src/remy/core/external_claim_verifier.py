"""
External claim verifier — structural fact-check between LLM output and
real tool evidence.

Problem this solves: the LLM can fabricate citations (URLs, DOIs, arXiv
IDs, paper titles with dates). The existing factuality guard treats any
sentence as "supported" if ANY external tool was used in the turn, even
if the specific URL in the text never appeared in any tool result. This
lets hallucinated papers flow into the brain as facts.

Approach: structural cross-check, no LLM involved.

For every external reference found in the response_text we ask: did
this exact citation (or its canonical form) appear in a tool_call.args
or tool_call.result anywhere in the session log?

- Yes → grounded.
- No but it is a well-formed URL/DOI/arxiv id → unverified (phantom).
- Matches a known placeholder pattern → placeholder.
- Optionally: HEAD-ping live to catch dead links (off by default —
  network I/O is too expensive per reply; operator can enable).

The caller decides the policy: downgrade "supported" counts, attach a
banner, tag the brain record, refuse to auto-store, etc.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional


_FETCH_EVIDENCE_TOOLS = frozenset({
    "browse_page",
    "browser_act",
    "http_get",
    "extract_content",
    "fetch_url",
})
_URL_RE = re.compile(r"https?://[^\s<>\]\)\"',]+", re.IGNORECASE)
# DOI: 10.<registrant>/<suffix>
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
# arXiv legacy: arXiv:1234.5678  and new: 2101.01234
_ARXIV_RE = re.compile(
    r"\b(?:arxiv[:\s]+)?(\d{4}\.\d{4,5})(?:v\d+)?\b",
    re.IGNORECASE,
)

# Placeholder / weasel-link patterns
_PLACEHOLDER_URL_HOSTS = {
    "example.com", "example.org", "placeholder.com", "link.example",
}
_PLACEHOLDER_TEXT_PATTERNS = [
    re.compile(r"умовн\w+\s+посиланн", re.IGNORECASE),       # "умовне посилання"
    re.compile(r"conditional\s+link", re.IGNORECASE),
    re.compile(r"placeholder\s+(?:url|link)", re.IGNORECASE),
    re.compile(r"\[link\s+here\]", re.IGNORECASE),
    re.compile(r"\[insert\s+(?:url|link)", re.IGNORECASE),
]

# Domains whose bare landing page (no paper id) counts as placeholder
_BARE_LANDING_HOSTS = {
    "arxiv.org", "sciencedirect.com", "elsevier.com",
    "aclanthology.org", "springer.com", "nature.com", "ieeexplore.ieee.org",
}

# ── Bibliographic-claim extraction patterns ────────────────────────────────
#
# The most dangerous class. LLMs routinely invent plausible paper titles
# complete with conference + year. We extract candidate biblio mentions and
# demand a verified identifier (URL/DOI/arXiv) within a short window, or we
# mark the biblio claim as unverified.

_VENUE_TOKENS = (
    r"arXiv|arxiv|ACL|NAACL|EMNLP|NeurIPS|ICLR|ICML|CVPR|ECCV|ICCV|AAAI|IJCAI|"
    r"SIGIR|SIGKDD|KDD|COLT|UAI|ISWC|ECIR|WWW|EACL|COLING|SIGMOD|VLDB|OSDI|"
    r"SOSP|NSDI|USENIX|CCS|Oakland|ScienceDirect|Elsevier|Springer|Nature|"
    r"Science|IEEE|Elsevier|PLOS|ACM|TACL"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Explicit "Title: X" / "Назва: X"
_EXPLICIT_TITLE_RE = re.compile(
    r"(?:title|назва|название)\s*[:—–-]\s*[\"“«\*_]*"
    r"(?P<title>[^\n\r*_”»\"\[]{10,200}?)"
    r"[\"”»\*_]*\s*(?=[\n\r\.\,\(]|$)",
    re.IGNORECASE,
)

# Markdown bold/italic phrase followed by year somewhere close
_BOLD_TITLE_RE = re.compile(
    r"(?:\*\*|__)(?P<title>[^*_\n]{10,200}?)(?:\*\*|__)"
    r"(?P<tail>[^\n]{0,120})",
)

# "Venue YYYY" e.g. "ACL 2025", "NeurIPS 2024"
_VENUE_YEAR_RE = re.compile(
    rf"\b(?P<venue>{_VENUE_TOKENS})\s+(?P<year>(?:19|20)\d{{2}})\b",
)

# In-text citation "(Smith et al., 2025)" / "(Smith and Jones, 2024)"
_AUTHOR_YEAR_RE = re.compile(
    r"\(\s*(?P<authors>[A-ZА-ЯЇІЄҐ][A-Za-zА-Яа-яЇїІіЄєҐґ\-']+(?:\s+(?:et\s+al\.?|and\s+[A-ZА-ЯЇІЄҐ][A-Za-zА-Яа-яЇїІіЄєҐґ\-']+))?)"
    r"[,;\s]+(?P<year>(?:19|20)\d{2})\s*\)",
)

_IDENTIFIER_NEAR_WINDOW = 160   # chars on each side considered "nearby"


# ── Model ──────────────────────────────────────────────────────────────────

CitationKind = Literal["url", "doi", "arxiv", "bibliographic"]
CitationStatus = Literal["grounded", "unverified", "placeholder", "dead", "reference_identity_mismatch"]


@dataclass
class Citation:
    raw: str
    canonical: str
    kind: CitationKind
    status: CitationStatus = "unverified"
    reason: str = ""
    # For bibliographic claims only: the component pieces we extracted.
    title: str = ""
    venue: str = ""
    year: str = ""
    has_identifier_nearby: bool = False   # did an URL/DOI/arxiv sit right next to this?

    def to_dict(self) -> dict:
        d = {
            "raw": self.raw,
            "canonical": self.canonical,
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
        }
        if self.kind == "bibliographic":
            d.update({
                "title": self.title,
                "venue": self.venue,
                "year": self.year,
                "has_identifier_nearby": self.has_identifier_nearby,
            })
        return d


@dataclass
class ExternalClaimReport:
    citations: list[Citation] = field(default_factory=list)
    grounded_count: int = 0
    unverified_count: int = 0
    placeholder_count: int = 0
    dead_count: int = 0
    reference_identity_mismatch_count: int = 0
    phantom_text_markers: list[str] = field(default_factory=list)  # placeholder phrases in prose
    bibliographic_unverified_count: int = 0                        # biblio claims with no verified id

    @property
    def total(self) -> int:
        return len(self.citations)

    @property
    def phantom_count(self) -> int:
        return self.unverified_count + self.placeholder_count + self.dead_count + self.reference_identity_mismatch_count

    @property
    def has_problems(self) -> bool:
        return (
            self.phantom_count > 0
            or bool(self.phantom_text_markers)
            or self.bibliographic_unverified_count > 0
        )

    def to_dict(self) -> dict:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "grounded_count": self.grounded_count,
            "unverified_count": self.unverified_count,
            "placeholder_count": self.placeholder_count,
            "dead_count": self.dead_count,
            "reference_identity_mismatch_count": self.reference_identity_mismatch_count,
            "phantom_count": self.phantom_count,
            "bibliographic_unverified_count": self.bibliographic_unverified_count,
            "total": self.total,
            "phantom_text_markers": list(self.phantom_text_markers),
            "has_problems": self.has_problems,
        }


# ── Canonicalization ───────────────────────────────────────────────────────


def _canonicalize_url(url: str) -> str:
    url = url.strip().strip(".,;:)")
    try:
        parsed = urllib.parse.urlsplit(url)
        netloc = (parsed.netloc or "").lower()
        # Strip common tracking params
        q = [
            (k, v) for k, v in urllib.parse.parse_qsl(parsed.query or "")
            if not k.lower().startswith("utm_")
        ]
        query = urllib.parse.urlencode(q)
        path = parsed.path or ""
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return urllib.parse.urlunsplit((parsed.scheme.lower(), netloc, path, query, ""))
    except Exception:
        return url


def _canonicalize_doi(doi: str) -> str:
    return doi.strip().rstrip(".,;:)").lower()


def _canonicalize_arxiv(aid: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})", aid)
    return m.group(1) if m else aid.strip().lower()


# ── Extraction ─────────────────────────────────────────────────────────────


def _identifier_nearby(text: str, start: int, end: int) -> bool:
    """Check whether a URL/DOI/arxiv id sits within the nearby window."""
    left = max(0, start - _IDENTIFIER_NEAR_WINDOW)
    right = min(len(text), end + _IDENTIFIER_NEAR_WINDOW)
    window = text[left:right]
    if _URL_RE.search(window):
        return True
    if _DOI_RE.search(window):
        return True
    for m in _ARXIV_RE.finditer(window):
        if "." in m.group(1):
            return True
    return False


def _clean_title(raw: str) -> str:
    t = raw.strip().strip("*_`\"“”»«").strip()
    # Drop trailing venue/year parentheticals
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t[:200]


def _biblio_canonical(title: str, venue: str, year: str) -> str:
    key = re.sub(r"[^a-z0-9а-яїієґ]+", "", (title or "").lower())[:80]
    return f"biblio::{key}|{venue.lower()}|{year}"


def extract_bibliographic_claims(text: str) -> list[Citation]:
    """Extract candidate paper/article references that lack a verified identifier.

    Covers four common LLM-fabrication shapes:
      1. "Title: <X>" / "Назва: <X>"
      2. Markdown-bold phrase followed by a year/venue tail
      3. "Venue YYYY" tokens (anchor a biblio mention)
      4. In-text author-year citations "(Smith et al., 2025)"
    """
    if not text:
        return []

    out: list[Citation] = []
    seen_keys: set[str] = set()

    def _add(title: str, venue: str, year: str, raw: str, start: int, end: int) -> None:
        title = _clean_title(title)
        if len(title) < 8:
            return
        canon = _biblio_canonical(title, venue, year)
        if canon in seen_keys:
            return
        seen_keys.add(canon)
        out.append(Citation(
            raw=raw.strip()[:240],
            canonical=canon,
            kind="bibliographic",
            title=title,
            venue=venue,
            year=year,
            has_identifier_nearby=_identifier_nearby(text, start, end),
        ))

    # 1. Explicit "Title: X"
    for m in _EXPLICIT_TITLE_RE.finditer(text):
        title = m.group("title")
        tail = text[m.end(): m.end() + _IDENTIFIER_NEAR_WINDOW]
        venue_m = _VENUE_YEAR_RE.search(tail)
        year_m = _YEAR_RE.search(tail)
        venue = venue_m.group("venue") if venue_m else ""
        year = (venue_m.group("year") if venue_m else (year_m.group(0) if year_m else ""))
        _add(title, venue, year, m.group(0), m.start(), m.end())

    # 2. Bold/italic title + nearby year or venue
    for m in _BOLD_TITLE_RE.finditer(text):
        title = m.group("title")
        tail = m.group("tail") or ""
        # Require at least a year OR a venue in the tail, else too noisy.
        venue_m = _VENUE_YEAR_RE.search(tail)
        year_m = _YEAR_RE.search(tail)
        if not (venue_m or year_m):
            continue
        venue = venue_m.group("venue") if venue_m else ""
        year = (venue_m.group("year") if venue_m else (year_m.group(0) if year_m else ""))
        _add(title, venue, year, m.group(0), m.start(), m.end())

    # 3. Standalone "Venue YYYY" not already covered — anchor to the
    #    preceding ~120 chars as an implicit title.
    for m in _VENUE_YEAR_RE.finditer(text):
        left = max(0, m.start() - 120)
        snippet = text[left: m.start()]
        # Best-guess title = last sentence-ish fragment
        title_guess = re.split(r"(?:[.!?]|\n)\s+", snippet.strip())[-1]
        title_guess = re.sub(r"^[-*\d\.\)\s]+", "", title_guess)
        if len(title_guess) >= 12:
            _add(
                title_guess, m.group("venue"), m.group("year"),
                m.group(0), m.start(), m.end(),
            )

    # 4. In-text author-year citations
    for m in _AUTHOR_YEAR_RE.finditer(text):
        authors = m.group("authors")
        year = m.group("year")
        _add(f"{authors} {year}", "", year, m.group(0), m.start(), m.end())

    return out


def extract_citations(text: str) -> list[Citation]:
    """Pull every external reference from text, deduped by canonical form."""
    if not text:
        return []

    seen: set[tuple[CitationKind, str]] = set()
    out: list[Citation] = []

    for raw in _URL_RE.findall(text):
        canon = _canonicalize_url(raw)
        key = ("url", canon)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=raw, canonical=canon, kind="url"))

    for raw in _DOI_RE.findall(text):
        canon = _canonicalize_doi(raw)
        key = ("doi", canon)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=raw, canonical=canon, kind="doi"))

    for m in _ARXIV_RE.finditer(text):
        canon = _canonicalize_arxiv(m.group(1))
        # Skip "2026" / "2025" that triggered arxiv regex (require a dot)
        if "." not in canon:
            continue
        key = ("arxiv", canon)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=m.group(0), canonical=canon, kind="arxiv"))

    # Bibliographic claims — LLM-fabricated paper titles are the most
    # dangerous leak path, since regex for URLs never catches them.
    for bib in extract_bibliographic_claims(text):
        key = ("bibliographic", bib.canonical)
        if key in seen:
            continue
        seen.add(key)
        out.append(bib)

    return out




def _fetch_evidence_entries(session_log: Iterable[dict]) -> list[dict]:
    entries: list[dict] = []
    for entry in session_log or []:
        if not isinstance(entry, dict) or entry.get("type") != "tool_call":
            continue
        if entry.get("tool") not in _FETCH_EVIDENCE_TOOLS:
            continue
        parts: list[str] = []
        payloads: list[dict] = []
        for key in ("args_full", "result_full", "args", "result"):
            val = entry.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                parts.append(val)
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict):
                        payloads.append(parsed)
                except Exception:
                    pass
            else:
                dumped = json.dumps(val, ensure_ascii=False, default=str) if not isinstance(val, str) else val
                parts.append(dumped)
                if isinstance(val, dict):
                    payloads.append(val)
        url = ""
        title = ""
        author = ""
        site = ""
        for payload in payloads:
            url = url or str(payload.get("url") or "").strip()
            title = title or str(payload.get("title") or "").strip()
            author = author or str(payload.get("author") or payload.get("authors") or "").strip()
            site = site or str(payload.get("site") or payload.get("source_name") or "").strip()
        entries.append({"tool": str(entry.get("tool") or ""), "text": "\n".join(parts), "url": url, "title": title, "author": author, "site": site})
    return entries


def _tool_evidence_blob(evidence_entries: Iterable[dict]) -> str:
    return "\n".join(str(item.get("text") or "") for item in evidence_entries if item.get("text"))


def _title_token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z?-??-?????????0-9]{4,}", (text or "").lower()) if len(tok) >= 4}


_QUOTED_TITLE_RE = re.compile(r'["??](?P<title>[^"??\n]{8,200})["??]')


def _best_title_match(claimed_title: str, fetched_titles: list[str]) -> tuple[float, str]:
    claimed_tokens = _title_token_set(claimed_title)
    if not claimed_tokens:
        return 0.0, ""
    best_ratio = 0.0
    best_title = ""
    for fetched in fetched_titles:
        fetched_tokens = _title_token_set(fetched)
        if not fetched_tokens:
            continue
        overlap = claimed_tokens & fetched_tokens
        ratio = len(overlap) / max(1, len(claimed_tokens))
        if ratio > best_ratio:
            best_ratio = ratio
            best_title = fetched
    return best_ratio, best_title


def _extract_nearby_claimed_title(response_text: str, citation: Citation) -> str:
    if not response_text or not citation.raw:
        return ""
    idx = response_text.lower().find(str(citation.raw).lower())
    if idx < 0 and citation.canonical:
        idx = response_text.lower().find(str(citation.canonical).lower())
    if idx < 0:
        return ""
    window = response_text[max(0, idx - 220): min(len(response_text), idx + len(str(citation.raw)) + 220)]
    for match in _EXPLICIT_TITLE_RE.finditer(window):
        title = _clean_title(match.group("title"))
        if len(title) >= 8:
            return title
    for match in _BOLD_TITLE_RE.finditer(window):
        title = _clean_title(match.group("title"))
        if len(title) >= 8:
            return title
    for match in _QUOTED_TITLE_RE.finditer(window):
        title = _clean_title(match.group("title"))
        if len(title) >= 8:
            return title
    return ""


def _matching_fetched_titles(citation: Citation, evidence_entries: list[dict]) -> list[str]:
    matches: list[str] = []
    canon = citation.canonical.lower()
    bare = re.sub(r"^arxiv:\s*", "", canon)
    for entry in evidence_entries:
        hay = (str(entry.get("text") or "") + "\n" + str(entry.get("url") or "")).lower()
        matched = False
        if citation.kind == "url" and canon and canon in hay:
            matched = True
        elif citation.kind in {"arxiv", "doi"} and canon and (canon in hay or bare in hay):
            matched = True
        elif citation.kind == "bibliographic" and citation.has_identifier_nearby:
            matched = True
        if matched:
            title = str(entry.get("title") or "").strip()
            if title and title not in matches:
                matches.append(title)
    if matches:
        return matches
    return [title for title in [str(entry.get("title") or "").strip() for entry in evidence_entries] if title]




def _classify_citation(citation: Citation, tool_blob: str, response_text: str, evidence_entries: list[dict]) -> None:
    lower_blob = tool_blob.lower() if tool_blob else ""
    canon = citation.canonical.lower()
    matching_titles = _matching_fetched_titles(citation, evidence_entries)
    if citation.kind == "bibliographic":
        title_tokens = _title_token_set(citation.title)
        blob_tokens = _title_token_set(lower_blob)
        overlap = title_tokens & blob_tokens
        strong_overlap = len(overlap) >= 3 or (title_tokens and len(overlap) >= max(2, int(len(title_tokens) * 0.6)))
        if strong_overlap and citation.has_identifier_nearby:
            citation.status = "grounded"
            citation.reason = "title tokens present in tool output + id nearby"
            return
        if citation.has_identifier_nearby and matching_titles:
            best_ratio, best_title = _best_title_match(citation.title, matching_titles)
            if best_ratio < 0.6:
                citation.status = "reference_identity_mismatch"
                citation.reason = "identifier nearby but claimed title mismatches fetched title" + (f": {best_title[:120]}" if best_title else "")
                return
        citation.status = "unverified"
        citation.reason = "title absent from tool output - likely fabricated" if not strong_overlap else "title tokens present but no identifier anchors this mention"
        return
    if citation.kind == "url":
        try:
            host = urllib.parse.urlsplit(citation.canonical).netloc.lower()
        except Exception:
            host = ""
        if host in _PLACEHOLDER_URL_HOSTS:
            citation.status = "placeholder"
            citation.reason = f"placeholder host: {host}"
            return
        if host in _BARE_LANDING_HOSTS:
            path_bits = urllib.parse.urlsplit(citation.canonical).path or ""
            if not bool(re.search(r"\d", path_bits)):
                citation.status = "placeholder"
                citation.reason = f"bare landing on {host} (no paper id)"
                return
    grounded_match = False
    if lower_blob and canon and canon in lower_blob:
        grounded_match = True
    elif citation.kind in ("arxiv", "doi") and lower_blob:
        bare = re.sub(r"^arxiv:\s*", "", canon)
        grounded_match = bare in lower_blob
    if grounded_match:
        claimed_title = _extract_nearby_claimed_title(response_text, citation)
        if claimed_title and matching_titles:
            best_ratio, best_title = _best_title_match(claimed_title, matching_titles)
            if best_ratio < 0.6:
                citation.status = "reference_identity_mismatch"
                citation.reason = "identifier exists but nearby title mismatches fetched title" + (f": {best_title[:120]}" if best_title else "")
                return
        citation.status = "grounded"
        citation.reason = "match in tool output"
        return
    citation.status = "unverified"
    citation.reason = "absent from tool output"


def _detect_placeholder_text_markers(response_text: str) -> list[str]:
    if not response_text:
        return []
    hits: list[str] = []
    for pat in _PLACEHOLDER_TEXT_PATTERNS:
        for m in pat.finditer(response_text):
            snippet = m.group(0)
            if snippet not in hits:
                hits.append(snippet)
    return hits[:5]


# ── Optional live HEAD check ───────────────────────────────────────────────


def _live_ping_url(url: str, timeout: float = 3.0) -> Optional[int]:
    """Return HTTP status code or None if unreachable. Never raises."""
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD", headers={
            "User-Agent": "Remi-CitationVerifier/1.0",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: B310 (operator opt-in)
            return int(getattr(resp, "status", 200))
    except Exception:
        return None


def _apply_live_checks(report: ExternalClaimReport, timeout: float) -> None:
    for c in report.citations:
        if c.kind != "url":
            continue
        if c.status != "unverified":
            continue
        code = _live_ping_url(c.canonical, timeout=timeout)
        if code is None:
            c.status = "dead"
            c.reason = "HEAD request failed"
        elif 400 <= code < 600:
            c.status = "dead"
            c.reason = f"HEAD returned {code}"
        # else: still unverified (reachable but not grounded in tools)


# ── Public API ─────────────────────────────────────────────────────────────


def verify_external_claims(
    response_text: str,
    session_log: Iterable[dict],
    *,
    live_check: bool = False,
    live_timeout: float = 3.0,
) -> ExternalClaimReport:
    """Structural verification of citations in response_text.

    Parameters
    ----------
    response_text: assistant's answer (post-LLM, pre-user)
    session_log:   turn session log containing tool_call dicts
    live_check:    if True, HEAD-ping still-unverified URLs to detect dead links
    """
    report = ExternalClaimReport()
    citations = extract_citations(response_text)
    if not citations:
        report.phantom_text_markers = _detect_placeholder_text_markers(response_text)
        return report

    evidence_entries = _fetch_evidence_entries(session_log or [])
    tool_blob = _tool_evidence_blob(evidence_entries)
    for c in citations:
        _classify_citation(c, tool_blob, response_text, evidence_entries)
    report.citations = citations

    if live_check:
        _apply_live_checks(report, timeout=live_timeout)

    # Tally
    for c in citations:
        if c.status == "grounded":
            report.grounded_count += 1
        elif c.status == "unverified":
            report.unverified_count += 1
            if c.kind == "bibliographic":
                report.bibliographic_unverified_count += 1
        elif c.status == "placeholder":
            report.placeholder_count += 1
        elif c.status == "dead":
            report.dead_count += 1
        elif c.status == "reference_identity_mismatch":
            report.reference_identity_mismatch_count += 1

    report.phantom_text_markers = _detect_placeholder_text_markers(response_text)
    return report


# ── Response banner ────────────────────────────────────────────────────────


_BANNER_EN = (
    "⚠ Citation check: {total} reference(s) cited, but {phantom} could not "
    "be verified against tools used this turn"
    "{breakdown}"
    ". Treat unverified sources as claims, not evidence."
)
_BANNER_UA = (
    "⚠ Перевірка посилань: процитовано {total}, але {phantom} не підтверджено "
    "інструментами цього ходу"
    "{breakdown}"
    ". Невідповідні джерела — припущення, не доказ."
)


def _breakdown_suffix(report: ExternalClaimReport, locale: str) -> str:
    bits = []
    if locale == "ua":
        if report.unverified_count:
            bits.append(f"{report.unverified_count} фантомних")
        if report.placeholder_count:
            bits.append(f"{report.placeholder_count} плейсхолдерів")
        if report.dead_count:
            bits.append(f"{report.dead_count} мертвих")
    else:
        if report.unverified_count:
            bits.append(f"{report.unverified_count} phantom")
        if report.placeholder_count:
            bits.append(f"{report.placeholder_count} placeholder")
        if report.dead_count:
            bits.append(f"{report.dead_count} dead")
        if report.reference_identity_mismatch_count:
            bits.append(f"{report.reference_identity_mismatch_count} identity-mismatch")
    if not bits:
        return ""
    return " (" + ", ".join(bits) + ")"


def render_banner(report: ExternalClaimReport, locale: str = "en") -> Optional[str]:
    if not report.has_problems or report.total == 0:
        return None
    tpl = _BANNER_UA if str(locale).lower().startswith(("ua", "uk")) else _BANNER_EN
    return tpl.format(
        total=report.total,
        phantom=report.phantom_count,
        breakdown=_breakdown_suffix(report, "ua" if tpl is _BANNER_UA else "en"),
    )
