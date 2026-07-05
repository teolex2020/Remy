"""
Synthesis Engine for Remy v3 Research Engine.

Handles contradiction detection, cross-source validation,
confidence scoring, and final synthesis generation.
"""

from __future__ import annotations

import logging
from typing import Any

from .research_models import (
    ResearchProject, Finding, FindingConfidence,
    Contradiction, Synthesis, ResearchQuestion,
)

log = logging.getLogger(__name__)


class SynthesisEngine:
    """Synthesizes research findings into structured output."""

    def __init__(self, min_corroboration: int = 2):
        self.min_corroboration = min_corroboration

    # -------------------------------------------------------------------
    # Contradiction detection
    # -------------------------------------------------------------------

    def detect_contradictions(
        self, findings: list[Finding]
    ) -> list[Contradiction]:
        """Detect contradictions between findings in the same category."""
        contradictions = []
        by_category: dict[str, list[Finding]] = {}
        for f in findings:
            cat = f.category or "general"
            by_category.setdefault(cat, []).append(f)

        for cat, cat_findings in by_category.items():
            if len(cat_findings) < 2:
                continue
            # Check for conflicting structured data (e.g. different prices)
            for i, fa in enumerate(cat_findings):
                for fb in cat_findings[i + 1:]:
                    conflict = self._check_conflict(fa, fb)
                    if conflict:
                        contradictions.append(Contradiction(
                            finding_a_id=fa.id,
                            finding_b_id=fb.id,
                            description=conflict,
                        ))
        return contradictions

    def _check_conflict(self, a: Finding, b: Finding) -> str:
        """Check if two findings conflict. Returns description or empty string."""
        # Structured data conflicts (numbers differ by >20%)
        if a.structured_data and b.structured_data:
            for key in a.structured_data:
                if key in b.structured_data:
                    va, vb = a.structured_data[key], b.structured_data[key]
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        if va > 0 and vb > 0:
                            ratio = max(va, vb) / min(va, vb)
                            if ratio > 1.2:
                                return (
                                    f"Conflicting {key}: {va} vs {vb} "
                                    f"(ratio {ratio:.1f}x)"
                                )

        # Confidence-based: high vs low on same topic
        if (a.confidence == FindingConfidence.HIGH
                and b.confidence == FindingConfidence.LOW
                and a.category == b.category):
            if a.content[:50] != b.content[:50]:
                return f"High vs low confidence findings in '{a.category}'"

        return ""

    # -------------------------------------------------------------------
    # Confidence scoring
    # -------------------------------------------------------------------

    def score_findings(self, findings: list[Finding]) -> list[Finding]:
        """Update finding confidence based on source corroboration."""
        for finding in findings:
            n_sources = len(finding.source_ids)
            if n_sources >= self.min_corroboration:
                finding.confidence = FindingConfidence.HIGH
            elif n_sources == 1:
                finding.confidence = FindingConfidence.MEDIUM
            # Contradicted ones stay as-is
        return findings

    # -------------------------------------------------------------------
    # Synthesis
    # -------------------------------------------------------------------

    def synthesize(self, project: ResearchProject) -> Synthesis:
        """Generate final synthesis from a research project."""
        # Score findings
        self.score_findings(project.findings)

        # Detect contradictions
        new_contradictions = self.detect_contradictions(project.findings)
        for c in new_contradictions:
            if not any(
                ec.finding_a_id == c.finding_a_id and ec.finding_b_id == c.finding_b_id
                for ec in project.contradictions
            ):
                project.add_contradiction(c)

        # Count high-confidence findings
        high_conf = [
            f for f in project.findings
            if f.confidence in (FindingConfidence.HIGH, FindingConfidence.MEDIUM)
        ]

        # Build key findings list
        key_findings = []
        for f in sorted(high_conf, key=lambda x: (
            0 if x.confidence == FindingConfidence.HIGH else 1,
            -len(x.source_ids),
        ))[:10]:
            key_findings.append(f.content[:200])

        # Unanswered questions
        unanswered = [
            q.question for q in project.questions if not q.answered
        ]

        # Overall confidence
        if not project.findings:
            confidence = 0.0
        else:
            conf_scores = {
                FindingConfidence.HIGH: 1.0,
                FindingConfidence.MEDIUM: 0.6,
                FindingConfidence.LOW: 0.3,
                FindingConfidence.CONTRADICTED: 0.1,
            }
            total = sum(
                conf_scores.get(f.confidence, 0.5)
                for f in project.findings
            )
            confidence = total / len(project.findings)

        # Penalty for unresolved contradictions
        unresolved = sum(1 for c in project.contradictions if not c.resolved)
        if unresolved:
            confidence *= max(0.5, 1.0 - unresolved * 0.1)

        # Build summary
        summary_parts = []
        summary_parts.append(
            f"Research on '{project.objective[:80]}' "
            f"analyzed {len(project.sources)} sources, "
            f"produced {len(project.findings)} findings."
        )
        if project.contradictions:
            summary_parts.append(
                f"{len(project.contradictions)} contradictions detected "
                f"({unresolved} unresolved)."
            )
        if unanswered:
            summary_parts.append(
                f"{len(unanswered)} questions remain unanswered."
            )

        synthesis = Synthesis(
            summary=" ".join(summary_parts),
            key_findings=key_findings,
            recommendations=self._generate_recommendations(project),
            confidence=round(confidence, 2),
            unanswered_questions=unanswered,
            contradictions_unresolved=unresolved,
            source_count=len(project.sources),
            finding_count=len(project.findings),
        )

        project.synthesis = synthesis
        return synthesis

    def _generate_recommendations(self, project: ResearchProject) -> list[str]:
        """Generate actionable recommendations based on findings."""
        recs = []

        # If contradictions exist
        unresolved = [c for c in project.contradictions if not c.resolved]
        if unresolved:
            recs.append(
                f"Resolve {len(unresolved)} contradictions before acting on findings"
            )

        # If unanswered questions
        unanswered = [q for q in project.questions if not q.answered]
        if unanswered:
            recs.append(
                f"Investigate {len(unanswered)} unanswered questions: "
                + "; ".join(q.question[:50] for q in unanswered[:3])
            )

        # If low source coverage
        if project.source_coverage < 0.5:
            recs.append(
                "Source coverage is low — expand search with different queries"
            )

        # If few findings
        if len(project.findings) < 3:
            recs.append(
                "Few findings extracted — consider deeper content analysis"
            )

        return recs

    # -------------------------------------------------------------------
    # Answer questions
    # -------------------------------------------------------------------

    def try_answer_questions(self, project: ResearchProject):
        """Attempt to answer research questions from findings."""
        for question in project.questions:
            if question.answered:
                continue

            # Simple: check if any finding mentions the question keywords
            q_words = set(question.question.lower().split())
            best_finding = None
            best_overlap = 0

            for finding in project.findings:
                f_words = set(finding.content.lower().split())
                overlap = len(q_words & f_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_finding = finding

            if best_finding and best_overlap >= 2:
                question.answered = True
                question.answer_summary = best_finding.content[:200]
                question.finding_ids.append(best_finding.id)
