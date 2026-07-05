"""
Source Credibility Module (RM-2)

Provides domain reputation scoring for web research.
Distinguishes high-quality institutional/scientific sources from lower-quality ones.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger("SourceCredibility")

# Default credibility scores (0.0 - 1.0)
DEFAULT_CREDIBILITY = {
    # Scientific / institutional sources (high trust)
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    "ncbi.nlm.nih.gov": 0.95,
    "who.int": 0.95,
    "cdc.gov": 0.95,
    "mayoclinic.org": 0.90,
    "clevelandclinic.org": 0.90,
    "hopkinsmedicine.org": 0.90,
    "medlineplus.gov": 0.90,
    "sciencedirect.com": 0.85,
    "nature.com": 0.90,
    "bmj.com": 0.90,
    "thelancet.com": 0.90,
    "webmd.com": 0.70,
    "healthline.com": 0.65,
    "medicalnewstoday.com": 0.65,

    # General Knowledge (Medium-High Trust)
    "wikipedia.org": 0.75,
    "britannica.com": 0.85,
    "scholar.google.com": 0.85,

    # News (Medium-High Trust)
    "bbc.com": 0.80,
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "npr.org": 0.80,
    "nytimes.com": 0.80,
    "wsj.com": 0.80,
    "economist.com": 0.85,
    "bloomberg.com": 0.80,

    # Social / UGC (Lower Trust)
    "reddit.com": 0.40,
    "quora.com": 0.35,
    "twitter.com": 0.30,
    "x.com": 0.30,
    "facebook.com": 0.25,
    "instagram.com": 0.25,
    "tiktok.com": 0.20,
    "youtube.com": 0.40, # Content varies wildly
    "medium.com": 0.50,  # Mixed quality
    "linkedin.com": 0.50,

    # Tech (Medium Trust)
    "stackoverflow.com": 0.75,
    "github.com": 0.70,
}

DEFAULT_SCORE = 0.50

class SourceCredibility:
    def __init__(self):
        self._cache = DEFAULT_CREDIBILITY.copy()
        self._user_overrides = {}

    def get_score(self, url: str) -> float:
        """Get credibility score for a URL."""
        if not url:
            return DEFAULT_SCORE

        try:
            domain = self._extract_domain(url)
            
            # Check user overrides first
            if domain in self._user_overrides:
                return self._user_overrides[domain]
            
            # Check exact match
            if domain in self._cache:
                return self._cache[domain]
            
            # Check parent domains (e.g., mail.google.com -> google.com)
            parts = domain.split('.')
            if len(parts) > 2:
                parent = ".".join(parts[-2:])
                if parent in self._cache:
                    return self._cache[parent]
                    
            return DEFAULT_SCORE
        except Exception as e:
            logger.warning(f"Error scoring URL {url}: {e}")
            return DEFAULT_SCORE

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    def load_overrides(self, brain_records: list):
        """Load user overrides from brain records (tags=["source-credibility"])."""
        count = 0
        for rec in brain_records:
            try:
                meta = rec.metadata or {}
                domain = meta.get("domain", "").lower().strip()
                score = float(meta.get("score", 0.5))
                if domain:
                    self._user_overrides[domain] = max(0.0, min(1.0, score))
                    count += 1
            except Exception:
                pass
        if count:
            logger.info(f"Loaded {count} source credibility overrides")

# Singleton instance
credibility_scorer = SourceCredibility()
