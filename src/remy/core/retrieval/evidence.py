"""Evidence packet and identity discipline.

Phase 3 of the brain-native retrieval roadmap:
    D:\\AuraSDK-verify\\private\\roadmaps\\BRAIN_NATIVE_RETRIEVAL_ROADMAP_2026-04-13.md

Motivation: a real URL is not enough. A page that returns HTTP 200 at
`https://arxiv.org/abs/2412.15803` may be a paper about WebLLM while the
agent is about to claim it supports a thesis on "thermodynamic epistemic
governance". Phase 2 classified candidates; Phase 3 enforces identity on
what was actually fetched.

Flow:
    Candidate   -> web_search discovery result (title/uri/snippet + source_class)
    Artifact    -> extract_content fetched page (url/title/content + host/class)
    EvidencePacket -> Artifact + identity_checks (result of cross-checks)

Identity checks performed here are deterministic, cheap, and local:
  - host matches canonical URL (defense against mirrors/redirects)
  - source_class preserved from candidate through fetch
  - claimed_title (if supplied by caller) matches fetched title above threshold
  - expected identifier (arxiv_id / doi) present in fetched content

Non-goals:
  - no LLM calls
  - no network I/O (the fetch happened upstream in extract_content)
  - no full NLP — token overlap is good enough at this layer
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Literal

from remy.core.retrieval.source_filter import classify


IdentityStatus = Literal["ok", "mismatch", "unknown"]


@dataclass
class Candidate:
    """Thin typed view over a web_search discovery dict.

    Accepts dicts produced by the annotate() pipeline; fields beyond these
    are preserved on the underlying dict if the caller holds it.
    """

    title: str
    uri: str
    snippet: str = ""
    source_class: str = "unknown"
    source_score: int = 0
    source_signals: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Candidate:
        return cls(
            title=str(d.get("title") or ""),
            uri=str(d.get("uri") or d.get("url") or ""),
            snippet=str(d.get("snippet") or ""),
            source_class=str(d.get("source_class") or "unknown"),
            source_score=int(d.get("source_score") or 0),
            source_signals=list(d.get("source_signals") or []),
        )


@dataclass
class Artifact:
    """Typed view over an extract_content result.

    Only the fields we need for identity checks. `content` is kept short here;
    callers that need the full body should keep it alongside the packet.
    """

    url: str
    title: str = ""
    content: str = ""
    author: str = ""
    date: str = ""
    site: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Artifact:
        return cls(
            url=str(d.get("url") or ""),
            title=str(d.get("title") or ""),
            content=str(d.get("content") or ""),
            author=str(d.get("author") or ""),
            date=str(d.get("date") or ""),
            site=str(d.get("site") or ""),
        )


@dataclass
class IdentityCheck:
    name: str
    status: IdentityStatus
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class EvidencePacket:
    """Artifact + structural identity checks.

    This is the contract the agent and downstream verifier should consume.
    An agent asserting a claim citing a URL MUST have an EvidencePacket with
    status == "ok" on the identity checks relevant to that claim; otherwise
    the citation should be treated as unverified.

    `claim_span` optionally records the specific excerpt of artifact content
    the claim rests on. When present, it makes claim-to-evidence binding
    explicit and auditable.
    """

    artifact: Artifact
    canonical_url: str
    host: str
    source_class: str
    source_score: int
    identity_checks: list[IdentityCheck] = field(default_factory=list)
    claim_span: str = ""

    @property
    def ok(self) -> bool:
        return all(c.status != "mismatch" for c in self.identity_checks)

    @property
    def has_mismatch(self) -> bool:
        return any(c.status == "mismatch" for c in self.identity_checks)

    def to_dict(self) -> dict:
        return {
            "canonical_url": self.canonical_url,
            "host": self.host,
            "source_class": self.source_class,
            "source_score": self.source_score,
            "title": self.artifact.title,
            "site": self.artifact.site,
            "identity_checks": [c.to_dict() for c in self.identity_checks],
            "ok": self.ok,
            "has_mismatch": self.has_mismatch,
            "claim_span": self.claim_span,
        }


# ── Canonicalization ──────────────────────────────────────────────────────


def _canonical_url(url: str) -> str:
    try:
        p = urllib.parse.urlsplit(url.strip())
        netloc = p.netloc.lower().removeprefix("www.")
        path = p.path or ""
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        q = [
            (k, v) for k, v in urllib.parse.parse_qsl(p.query or "")
            if not k.lower().startswith("utm_")
        ]
        return urllib.parse.urlunsplit((p.scheme.lower(), netloc, path, urllib.parse.urlencode(q), ""))
    except Exception:
        return url.strip()


def _host(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url or "").netloc.lower().removeprefix("www.")
    except Exception:
        return ""


# ── Identity checks ───────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0400-\u04FF]{3,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 4}


def _title_overlap(claimed: str, fetched: str) -> float:
    a = _tokens(claimed)
    b = _tokens(fetched)
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def _check_host_matches(requested_url: str, artifact_url: str) -> IdentityCheck:
    req = _host(requested_url)
    got = _host(artifact_url)
    if not req or not got:
        return IdentityCheck("host_match", "unknown", "missing host on either side")
    if req == got or got.endswith("." + req) or req.endswith("." + got):
        return IdentityCheck("host_match", "ok", f"{req} == {got}")
    return IdentityCheck("host_match", "mismatch", f"{req} != {got}")


def _check_title_matches(claimed_title: str, fetched_title: str, threshold: float = 0.5) -> IdentityCheck:
    if not claimed_title:
        return IdentityCheck("title_match", "unknown", "no claimed title supplied")
    if not fetched_title:
        return IdentityCheck("title_match", "unknown", "no title in fetched artifact")
    ratio = _title_overlap(claimed_title, fetched_title)
    if ratio >= threshold:
        return IdentityCheck("title_match", "ok", f"overlap={ratio:.2f}")
    return IdentityCheck(
        "title_match",
        "mismatch",
        f"overlap={ratio:.2f} < {threshold}; claimed={claimed_title[:80]!r} fetched={fetched_title[:80]!r}",
    )


def _check_identifier_present(expected_id: str, haystack: str) -> IdentityCheck:
    eid = (expected_id or "").strip()
    if not eid:
        return IdentityCheck("identifier_present", "unknown", "no expected identifier supplied")
    hay = (haystack or "").lower()
    if eid.lower() in hay:
        return IdentityCheck("identifier_present", "ok", f"found {eid}")
    # arXiv bare form
    m = _ARXIV_RE.search(eid)
    if m and m.group(1).lower() in hay:
        return IdentityCheck("identifier_present", "ok", f"found arxiv id {m.group(1)}")
    return IdentityCheck("identifier_present", "mismatch", f"expected {eid!r} not found in artifact")


def _check_author_matches(claimed_authors: str, fetched_author: str) -> IdentityCheck:
    """Soft author cross-check.

    Passes if any claimed author surname appears in the fetched author
    metadata. Authors are notoriously noisy (initials, formatting, order),
    so we accept partial surname overlap rather than demanding exact match.
    """
    claimed = (claimed_authors or "").strip()
    fetched = (fetched_author or "").strip()
    if not claimed:
        return IdentityCheck("author_match", "unknown", "no claimed authors supplied")
    if not fetched:
        return IdentityCheck("author_match", "unknown", "no author in fetched artifact")
    claimed_surnames = {
        t.lower() for t in re.findall(r"[A-Z][a-z]{2,}", claimed)
    }
    fetched_lower = fetched.lower()
    if not claimed_surnames:
        return IdentityCheck("author_match", "unknown", "no parseable surnames in claim")
    if any(s in fetched_lower for s in claimed_surnames):
        return IdentityCheck("author_match", "ok", f"surname overlap with {fetched[:60]!r}")
    return IdentityCheck(
        "author_match",
        "mismatch",
        f"no claimed surname in fetched author={fetched[:80]!r}",
    )


def _check_source_class_preserved(candidate_class: str, artifact_url: str) -> IdentityCheck:
    """Artifact host should classify to the same or a compatible class.

    Redirects from a research host to a mirror/SEO host are a mismatch.
    """
    if not candidate_class or candidate_class == "unknown":
        return IdentityCheck("source_class_preserved", "unknown", "candidate had no source_class")
    scored = classify({"title": "", "uri": artifact_url, "snippet": ""})
    if scored.source_class == candidate_class:
        return IdentityCheck("source_class_preserved", "ok", f"{candidate_class} preserved")
    # Soft-compatible pairs: publisher↔official_docs, github↔official_docs for docs sites.
    compatible = {
        ("official_docs", "publisher"),
        ("publisher", "official_docs"),
        ("research", "publisher"),
    }
    if (candidate_class, scored.source_class) in compatible:
        return IdentityCheck(
            "source_class_preserved",
            "ok",
            f"{candidate_class} -> {scored.source_class} (compatible)",
        )
    return IdentityCheck(
        "source_class_preserved",
        "mismatch",
        f"{candidate_class} -> {scored.source_class}",
    )


# ── Packet construction ───────────────────────────────────────────────────


def build_packet(
    artifact_dict: dict,
    *,
    requested_url: str = "",
    candidate: dict | None = None,
    expected_title: str = "",
    expected_identifier: str = "",
    expected_authors: str = "",
    claim_span: str = "",
) -> EvidencePacket:
    """Build an EvidencePacket from an extract_content result.

    Parameters
    ----------
    artifact_dict:       raw extract_content JSON (dict after json.loads)
    requested_url:       URL the agent asked for (for host-match check)
    candidate:           annotated candidate dict from web_search
    expected_title:      title the agent believes this URL is about
    expected_identifier: arXiv id / DOI / string that MUST appear in content
    expected_authors:    author names the claim asserts (any surname overlap ok)
    claim_span:          excerpt of artifact content the claim rests on
    """
    art = Artifact.from_dict(artifact_dict or {})
    canonical = _canonical_url(art.url or requested_url)
    host = _host(canonical)

    if candidate:
        cand_class = str(candidate.get("source_class") or "unknown")
        cand_score = int(candidate.get("source_score") or 0)
    else:
        scored = classify({"title": art.title, "uri": canonical, "snippet": ""})
        cand_class = scored.source_class
        cand_score = scored.source_score

    checks: list[IdentityCheck] = []
    if requested_url:
        checks.append(_check_host_matches(requested_url, art.url or canonical))
    if expected_title:
        checks.append(_check_title_matches(expected_title, art.title))
    if expected_identifier:
        checks.append(_check_identifier_present(expected_identifier, f"{art.title}\n{art.content}"))
    if expected_authors:
        checks.append(_check_author_matches(expected_authors, art.author))
    if candidate and cand_class != "unknown":
        checks.append(_check_source_class_preserved(cand_class, canonical))

    return EvidencePacket(
        artifact=art,
        canonical_url=canonical,
        host=host,
        source_class=cand_class,
        source_score=cand_score,
        identity_checks=checks,
        claim_span=claim_span,
    )


# ── Claim → Packet binding ────────────────────────────────────────────────
#
# Downstream callers (agent verifier, research gate) need to answer: given
# this claim text and this packet, is the claim structurally grounded? The
# answer is a discrete verdict, not a float — we surface exactly the classes
# the roadmap names so the gate logic stays auditable.


VerdictStatus = Literal[
    "grounded",                       # packet ok + claim identifiers all present
    "reference_identity_mismatch",    # identifier/title/author says wrong object
    "no_evidence",                    # no packet provided
    "unverifiable",                   # packet present but checks inconclusive
]


@dataclass
class ClaimVerdict:
    status: VerdictStatus
    reasons: list[str] = field(default_factory=list)
    packet: EvidencePacket | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "packet": self.packet.to_dict() if self.packet else None,
        }


def extract_claim_identifiers(claim_text: str) -> dict:
    """Pull structured identifiers out of a claim string.

    Returns {"arxiv_ids": [...], "dois": [...]} — bare strings, no canonicalisation.
    Empty lists if nothing found. Used to auto-derive `expected_identifier`
    from the agent's own claim so we don't trust the agent to pass it in.
    """
    text = claim_text or ""
    arxiv = [m.group(1) for m in _ARXIV_RE.finditer(text)]
    dois = [m.group(0) for m in _DOI_RE.finditer(text)]
    return {
        "arxiv_ids": list(dict.fromkeys(arxiv)),
        "dois": list(dict.fromkeys(dois)),
    }


def verify_claim_against_packet(
    claim_text: str,
    packet: EvidencePacket | None,
) -> ClaimVerdict:
    """Bind a claim to a packet and emit a discrete verdict.

    Rules:
      - no packet                          -> no_evidence
      - packet has any identity mismatch   -> reference_identity_mismatch
      - claim mentions arxiv/DOI identifiers missing from artifact -> reference_identity_mismatch
      - otherwise                          -> grounded

    The 'unverifiable' status is reserved for cases where checks exist but
    all report 'unknown' (e.g., missing title on fetched page) — we surface
    that distinction so the gate can refuse rather than silently accept.
    """
    if packet is None:
        return ClaimVerdict(status="no_evidence", reasons=["no evidence packet supplied"])

    reasons: list[str] = []
    if packet.has_mismatch:
        for c in packet.identity_checks:
            if c.status == "mismatch":
                reasons.append(f"{c.name}: {c.detail}")
        return ClaimVerdict(status="reference_identity_mismatch", reasons=reasons, packet=packet)

    ids = extract_claim_identifiers(claim_text)
    haystack = f"{packet.artifact.title}\n{packet.artifact.content}".lower()
    missing: list[str] = []
    for aid in ids["arxiv_ids"]:
        if aid.lower() not in haystack:
            missing.append(f"arxiv:{aid}")
    for doi in ids["dois"]:
        if doi.lower() not in haystack:
            missing.append(f"doi:{doi}")
    if missing:
        reasons.append("claim identifiers not present in fetched content: " + ", ".join(missing))
        return ClaimVerdict(
            status="reference_identity_mismatch",
            reasons=reasons,
            packet=packet,
        )

    if packet.identity_checks and all(c.status == "unknown" for c in packet.identity_checks):
        return ClaimVerdict(
            status="unverifiable",
            reasons=["all identity checks returned unknown"],
            packet=packet,
        )

    return ClaimVerdict(status="grounded", reasons=[], packet=packet)
