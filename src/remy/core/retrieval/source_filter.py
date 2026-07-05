"""Deterministic source classification and candidate scoring.

Phase 2 of the brain-native retrieval roadmap:
    D:\\AuraSDK-verify\\private\\roadmaps\\BRAIN_NATIVE_RETRIEVAL_ROADMAP_2026-04-13.md

Responsibilities:
  - classify a candidate source into a coarse class (official_docs, github,
    research, publisher, news, forum, seo, mirror, unknown)
  - produce a per-candidate score in roughly [-3, +3] from signal sums
  - rerank a candidate list and emit a filtered top-N

Non-goals:
  - no machine learning
  - no network calls; pure string/host inspection
  - no per-user domain memory yet (deferred to later phase)

Contract:
  Input candidate shape (minimum):
      {"title": str, "uri": str, "snippet": str}
  Output candidate shape (additive, original fields preserved):
      {..., "source_class": str, "source_score": int, "source_signals": [str]}
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass


_RESEARCH_HOSTS = {
    "arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "nature.com",
    "science.org",
    "biorxiv.org",
    "medrxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "semanticscholar.org",
}

_OFFICIAL_DOCS_HOSTS = {
    "docs.python.org",
    "docs.rs",
    "doc.rust-lang.org",
    "go.dev",
    "pkg.go.dev",
    "nodejs.org",
    "developer.mozilla.org",
    "kubernetes.io",
    "cloud.google.com",
    "learn.microsoft.com",
    "docs.aws.amazon.com",
    "docs.djangoproject.com",
    "fastapi.tiangolo.com",
    "docs.pytest.org",
    "docs.python-requests.org",
    "reactjs.org",
    "react.dev",
    "typescriptlang.org",
    "www.typescriptlang.org",
    "docs.oracle.com",
}

_GITHUB_HOSTS = {"github.com", "gist.github.com", "raw.githubusercontent.com"}

_NEWS_HOSTS = {
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "bloomberg.com",
    "apnews.com",
    "theguardian.com",
    "wsj.com",
    "ft.com",
}

_FORUM_HOSTS = {
    "stackoverflow.com",
    "stackexchange.com",
    "serverfault.com",
    "superuser.com",
    "reddit.com",
    "news.ycombinator.com",
    "discuss.python.org",
    "forum.rust-lang.org",
    "community.openai.com",
}

# Publisher blogs / legitimate long-form but not primary source.
_PUBLISHER_HOSTS = {
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "blog.google",
    "ai.meta.com",
    "research.google",
    "microsoft.com",
    "huggingface.co",
    "pytorch.org",
}

# Known paper aggregators / mirrors that republish arXiv/conference PDFs.
# These often show up with titles that look authoritative but are scraped.
_MIRROR_HOSTS = {
    "alphaxiv.org",
    "arxiv-sanity.com",
    "arxiv-sanity-lite.com",
    "deeplearn.org",
    "paperswithcode.com",
    "scholar.google.com",
    "researchgate.net",
    "academia.edu",
    "x-mol.com",
    "scinapse.io",
}

# Listicle / SEO signals in URL path or title.
_SEO_PATH_PATTERNS = (
    re.compile(r"/best-"),
    re.compile(r"/the-best-"),
    re.compile(r"/top-\d+"),
    re.compile(r"/top\d+"),
    re.compile(r"/\d{4}/best"),
    re.compile(r"/ultimate-guide"),
    re.compile(r"/leaderboard"),
    re.compile(r"/compare-"),
    re.compile(r"/vs-"),
    re.compile(r"/\d+-best-"),
    re.compile(r"/blog/best-"),
    re.compile(r"/blog/the-best-"),
    re.compile(r"/blogs/best-"),
    re.compile(r"/blogs/the-best-"),
    re.compile(r"/resources/blog/.*best-"),
    re.compile(r"/best-\w+-(models?|llms?|ais?)"),
    re.compile(r"/top-(models?|llms?|ais?)"),
    re.compile(r"/top-\w+-(models?|llms?|ais?)"),
    re.compile(r"-(models?|llms?|ais?)-right-now"),
    re.compile(r"/right-now"),
)

_SEO_TITLE_PATTERNS = (
    re.compile(r"\btop\s+\d+\b", re.I),
    re.compile(r"\bbest\s+\w+\s+(of\s+)?20\d{2}\b", re.I),
    re.compile(r"\bbest\s+(llms?|ai|models?|ai\s+models?)\b", re.I),
    re.compile(r"\bthe\s+best\s+\w+", re.I),
    re.compile(r"\bultimate\s+guide\b", re.I),
    re.compile(r"\bleaderboard\b", re.I),
    re.compile(r"\b\d+\s+best\b", re.I),
)

# Hosts whose primary business model is SEO content farms.
# Intentionally conservative; we lean on path/title signals rather than
# blanket-banning domains.
_SEO_HOST_HINTS = (
    "medium.com",
    "towardsdatascience.com",
    "hackernoon.com",
    "dev.to",
    "geeksforgeeks.org",
    "analyticsvidhya.com",
    "kdnuggets.com",
)


SourceClass = str  # one of the constants below


CLASS_OFFICIAL_DOCS = "official_docs"
CLASS_GITHUB = "github"
CLASS_RESEARCH = "research"
CLASS_PUBLISHER = "publisher"
CLASS_NEWS = "news"
CLASS_FORUM = "forum"
CLASS_SEO = "seo"
CLASS_MIRROR = "mirror"
CLASS_UNKNOWN = "unknown"


@dataclass
class Scored:
    source_class: SourceClass
    source_score: int
    source_signals: list[str]


def _host(uri: str) -> str:
    try:
        return urllib.parse.urlsplit(uri or "").netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _path(uri: str) -> str:
    try:
        return urllib.parse.urlsplit(uri or "").path.lower()
    except Exception:
        return ""


def _host_matches(host: str, known: set[str]) -> bool:
    if not host:
        return False
    if host in known:
        return True
    for k in known:
        if host.endswith("." + k):
            return True
    return False


def classify(candidate: dict) -> Scored:
    uri = candidate.get("uri") or candidate.get("url") or ""
    title = (candidate.get("title") or "").strip()
    path = _path(uri)
    host = _host(uri)

    signals: list[str] = []
    score = 0

    # Class detection (ordered — first match wins, with mirror/SEO override).
    cls: SourceClass = CLASS_UNKNOWN

    if _host_matches(host, _RESEARCH_HOSTS):
        cls = CLASS_RESEARCH
        signals.append("research_host")
        score += 2
    elif _host_matches(host, _OFFICIAL_DOCS_HOSTS):
        cls = CLASS_OFFICIAL_DOCS
        signals.append("official_docs_host")
        score += 2
    elif _host_matches(host, _GITHUB_HOSTS):
        cls = CLASS_GITHUB
        signals.append("github_host")
        score += 1
    elif _host_matches(host, _NEWS_HOSTS):
        cls = CLASS_NEWS
        signals.append("news_host")
        score += 1
    elif _host_matches(host, _FORUM_HOSTS):
        cls = CLASS_FORUM
        signals.append("forum_host")
    elif _host_matches(host, _PUBLISHER_HOSTS):
        cls = CLASS_PUBLISHER
        signals.append("publisher_host")
        score += 1
    elif _host_matches(host, _MIRROR_HOSTS):
        cls = CLASS_MIRROR
        signals.append("mirror_host")
        score -= 2

    # SEO override: even on otherwise-unknown hosts, SEO patterns demote hard.
    seo_hit = False
    for pat in _SEO_PATH_PATTERNS:
        if pat.search(path):
            signals.append(f"seo_path:{pat.pattern}")
            seo_hit = True
            break
    if not seo_hit:
        for pat in _SEO_TITLE_PATTERNS:
            if pat.search(title):
                signals.append(f"seo_title:{pat.pattern}")
                seo_hit = True
                break
    for hint in _SEO_HOST_HINTS:
        if hint in host:
            signals.append(f"seo_host_hint:{hint}")
            seo_hit = True
            break

    if seo_hit:
        # Only downgrade if we haven't already classed as a high-trust source.
        # A BBC listicle is still BBC. But an unknown host with listicle patterns
        # is SEO until proven otherwise.
        if cls in (CLASS_UNKNOWN, CLASS_PUBLISHER, CLASS_FORUM, CLASS_MIRROR):
            cls = CLASS_SEO
        score -= 3

    # Paths that look like real content (docs pages, abstracts, release notes)
    # get a mild boost. Cheap heuristic.
    if any(seg in path for seg in ("/docs/", "/reference/", "/guide/", "/tutorial/")):
        signals.append("doc_like_path")
        score += 1
    if any(seg in path for seg in ("/abs/", "/pdf/", "/html/")) and cls == CLASS_RESEARCH:
        signals.append("paper_ref_path")
        score += 1

    return Scored(source_class=cls, source_score=score, source_signals=signals)


# ── Site-constraint enforcement ─────────────────────────────────────────────
#
# ddgs backends (particularly Yahoo) frequently honor `site:` operators only
# weakly — the raw result set can contain off-domain spillover even on
# strict site-constrained queries. Phase 2 enforces the constraint ourselves
# post-search: if the agent asked for `site:arxiv.org`, non-arxiv.org hosts
# are dropped outright, not merely demoted.

_SITE_RE = re.compile(r"\bsite:([a-z0-9][a-z0-9\-._]*[a-z0-9])\b", re.I)


def extract_site_constraint(query: str) -> str | None:
    """Return the domain from a `site:<domain>` operator in the query, or
    None if no such operator is present. Strips leading www. for matching.
    """
    if not query:
        return None
    m = _SITE_RE.search(query)
    if not m:
        return None
    return m.group(1).lower().removeprefix("www.")


def _domain_matches(host: str, allowed: str) -> bool:
    host = (host or "").lower().removeprefix("www.")
    allowed = allowed.lower().removeprefix("www.")
    if not host or not allowed:
        return False
    return host == allowed or host.endswith("." + allowed)


def enforce_site_constraint(candidates: list[dict], domain: str | None) -> list[dict]:
    """Drop candidates whose host doesn't match the site: constraint.

    When `domain` is None, returns candidates unchanged. When given, only
    candidates on that exact domain or a subdomain survive. This is a hard
    filter, not a rerank — off-domain candidates are removed from the list.
    """
    if not domain:
        return list(candidates or [])
    out = []
    for c in candidates or []:
        host = _host(c.get("uri") or c.get("url") or "")
        if _domain_matches(host, domain):
            out.append(c)
    return out


def annotate(candidates: list[dict]) -> list[dict]:
    """Return new candidate dicts with classification fields attached."""
    out: list[dict] = []
    for c in candidates or []:
        s = classify(c)
        merged = dict(c)
        merged["source_class"] = s.source_class
        merged["source_score"] = s.source_score
        merged["source_signals"] = s.source_signals
        out.append(merged)
    return out


def rerank(candidates: list[dict], *, drop_classes: set[str] | None = None) -> list[dict]:
    """Rerank annotated candidates by score descending, preserving original
    order on ties. Optionally drop candidates in given classes.

    Does NOT re-annotate; caller should pass already-annotated list.
    """
    drop_classes = drop_classes or set()
    ranked = [c for c in candidates if c.get("source_class") not in drop_classes]
    ranked.sort(
        key=lambda c: (-(c.get("source_score") or 0), candidates.index(c)),
    )
    return ranked
