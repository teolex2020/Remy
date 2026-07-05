"""
Research models for Remy v3.

Structured data types for the research pipeline:
project → questions → sources → findings → contradictions → synthesis.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResearchStatus(str, Enum):
    PLANNING = "planning"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceCredibility(str, Enum):
    HIGH = "high"            # Official docs, academic papers, verified data
    MEDIUM = "medium"        # News articles, known blogs, product pages
    LOW = "low"              # Forums, social media, unverified
    UNKNOWN = "unknown"


class FindingConfidence(str, Enum):
    HIGH = "high"            # Multiple corroborating sources
    MEDIUM = "medium"        # Single credible source
    LOW = "low"              # Inferred or unverified
    CONTRADICTED = "contradicted"


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """A single information source discovered during research."""
    id: str = field(default_factory=lambda: f"src_{uuid.uuid4().hex[:8]}")
    url: str = ""
    title: str = ""
    domain: str = ""
    snippet: str = ""           # First 300 chars of extracted content
    full_text: str = ""         # Full extracted content (if available)

    credibility: SourceCredibility = SourceCredibility.UNKNOWN
    relevance_score: float = 0.0   # 0.0–1.0
    freshness_days: int = -1       # Days since publication (-1 = unknown)

    fetched: bool = False
    fetch_error: str = ""
    fetched_at: float = 0.0

    # Provenance
    query: str = ""             # Search query that found this source
    search_rank: int = 0        # Position in search results

    def is_usable(self) -> bool:
        return self.fetched and not self.fetch_error and bool(self.full_text or self.snippet)


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A discrete piece of information extracted from sources."""
    id: str = field(default_factory=lambda: f"find_{uuid.uuid4().hex[:8]}")
    content: str = ""
    category: str = ""          # pricing, feature, competitor, strategy, ...
    confidence: FindingConfidence = FindingConfidence.MEDIUM

    source_ids: list[str] = field(default_factory=list)
    supporting_quotes: list[str] = field(default_factory=list)

    # For structured data (pricing, metrics, etc.)
    structured_data: dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------

@dataclass
class Contradiction:
    """A detected contradiction between findings."""
    id: str = field(default_factory=lambda: f"contra_{uuid.uuid4().hex[:8]}")
    finding_a_id: str = ""
    finding_b_id: str = ""
    description: str = ""
    resolution: str = ""        # How it was resolved (if at all)
    resolved: bool = False


# ---------------------------------------------------------------------------
# Research Question
# ---------------------------------------------------------------------------

@dataclass
class ResearchQuestion:
    """A specific question the research aims to answer."""
    id: str = field(default_factory=lambda: f"rq_{uuid.uuid4().hex[:8]}")
    question: str = ""
    priority: int = 5          # 1 = highest
    answered: bool = False
    answer_summary: str = ""
    finding_ids: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

@dataclass
class Synthesis:
    """Final synthesis of research findings."""
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 0.0     # Overall confidence 0.0–1.0
    unanswered_questions: list[str] = field(default_factory=list)
    contradictions_unresolved: int = 0
    source_count: int = 0
    finding_count: int = 0
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Research Project
# ---------------------------------------------------------------------------

@dataclass
class ResearchProject:
    """Top-level research project — the unit of structured research.

    Lifecycle:
    planning → collecting → analyzing → synthesizing → completed
    """
    id: str = field(default_factory=lambda: f"rp_{uuid.uuid4().hex[:12]}")
    objective: str = ""         # What we're trying to learn
    mission_id: str = ""
    goal_id: str = ""
    status: ResearchStatus = ResearchStatus.PLANNING

    # Research plan
    questions: list[ResearchQuestion] = field(default_factory=list)

    # Collection
    sources: list[Source] = field(default_factory=list)
    queries_executed: list[str] = field(default_factory=list)
    max_sources: int = 10
    max_queries: int = 5

    # Analysis
    findings: list[Finding] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)

    # Output
    synthesis: Synthesis | None = None
    prior_context: list[str] = field(default_factory=list)
    strategy_hints: list[str] = field(default_factory=list)
    reused_playbook_id: str = ""

    # Budget
    tokens_used: int = 0
    cost_usd: float = 0.0
    time_limit_sec: int = 300

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def is_active(self) -> bool:
        return self.status not in (ResearchStatus.COMPLETED, ResearchStatus.FAILED)

    @property
    def progress(self) -> float:
        """0.0–1.0 research completion estimate."""
        if not self.questions:
            return 0.0
        answered = sum(1 for q in self.questions if q.answered)
        return answered / len(self.questions)

    @property
    def source_coverage(self) -> float:
        """Fraction of sources successfully fetched."""
        if not self.sources:
            return 0.0
        usable = sum(1 for s in self.sources if s.is_usable())
        return usable / len(self.sources)

    def add_source(self, source: Source):
        self.sources.append(source)
        self.updated_at = time.time()

    def add_finding(self, finding: Finding):
        self.findings.append(finding)
        self.updated_at = time.time()

    def add_contradiction(self, c: Contradiction):
        self.contradictions.append(c)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective[:100],
            "status": self.status.value,
            "questions": len(self.questions),
            "sources": len(self.sources),
            "usable_sources": sum(1 for s in self.sources if s.is_usable()),
            "findings": len(self.findings),
            "contradictions": len(self.contradictions),
            "progress": round(self.progress, 2),
            "cost_usd": round(self.cost_usd, 4),
        }
