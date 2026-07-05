"""
Source Ranking for Remy v3 Research Engine.

Scores and ranks sources by credibility, relevance, and freshness.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from .research_models import Source, SourceCredibility

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain trust tiers
# ---------------------------------------------------------------------------

HIGH_TRUST_DOMAINS = frozenset({
    "github.com", "docs.python.org", "developer.mozilla.org",
    "arxiv.org", "scholar.google.com", "ieee.org", "acm.org",
    "stackoverflow.com", "en.wikipedia.org",
    "reuters.com", "bloomberg.com", "techcrunch.com",
    "docs.google.com", "microsoft.com", "apple.com",
})

MEDIUM_TRUST_PATTERNS = (
    r"\.gov$", r"\.edu$", r"\.ac\.\w+$",
    r"medium\.com", r"dev\.to", r"hackernoon\.com",
    r"reddit\.com", r"news\.ycombinator\.com",
)

LOW_TRUST_PATTERNS = (
    r"blogspot\.", r"wordpress\.com", r"tumblr\.com",
    r"quora\.com",
)


class SourceRanker:
    """Ranks sources by composite score: credibility + relevance + freshness."""

    def __init__(
        self,
        credibility_weight: float = 0.4,
        relevance_weight: float = 0.4,
        freshness_weight: float = 0.2,
    ):
        self.w_cred = credibility_weight
        self.w_rel = relevance_weight
        self.w_fresh = freshness_weight

    def rank(self, sources: list[Source]) -> list[Source]:
        """Rank sources by composite score (descending)."""
        for src in sources:
            src.credibility = self.assess_credibility(src)
            src.relevance_score = self._composite_score(src)
        return sorted(sources, key=lambda s: s.relevance_score, reverse=True)

    def assess_credibility(self, source: Source) -> SourceCredibility:
        """Assess source credibility from domain and metadata."""
        domain = self._extract_domain(source.url)
        source.domain = domain

        if domain in HIGH_TRUST_DOMAINS:
            return SourceCredibility.HIGH

        for pattern in MEDIUM_TRUST_PATTERNS:
            if re.search(pattern, domain):
                return SourceCredibility.MEDIUM

        for pattern in LOW_TRUST_PATTERNS:
            if re.search(pattern, domain):
                return SourceCredibility.LOW

        # Default: medium for recognized TLDs, unknown otherwise
        if any(domain.endswith(tld) for tld in (".com", ".org", ".net", ".io")):
            return SourceCredibility.MEDIUM

        return SourceCredibility.UNKNOWN

    def _composite_score(self, source: Source) -> float:
        """Calculate composite ranking score 0.0–1.0."""
        cred_score = {
            SourceCredibility.HIGH: 1.0,
            SourceCredibility.MEDIUM: 0.6,
            SourceCredibility.LOW: 0.3,
            SourceCredibility.UNKNOWN: 0.4,
        }.get(source.credibility, 0.4)

        # Relevance from search rank (lower = better)
        rank = max(1, source.search_rank)
        rel_score = max(0.0, 1.0 - (rank - 1) * 0.1)

        # Freshness (newer = better)
        if source.freshness_days < 0:
            fresh_score = 0.5  # Unknown age
        elif source.freshness_days <= 7:
            fresh_score = 1.0
        elif source.freshness_days <= 30:
            fresh_score = 0.8
        elif source.freshness_days <= 90:
            fresh_score = 0.6
        elif source.freshness_days <= 365:
            fresh_score = 0.4
        else:
            fresh_score = 0.2

        return (
            self.w_cred * cred_score
            + self.w_rel * rel_score
            + self.w_fresh * fresh_score
        )

    def _extract_domain(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    def top_n(self, sources: list[Source], n: int = 5) -> list[Source]:
        """Return top N ranked sources."""
        return self.rank(sources)[:n]

    def filter_usable(self, sources: list[Source]) -> list[Source]:
        """Filter to only usable (fetched, no error) sources."""
        return [s for s in sources if s.is_usable()]
