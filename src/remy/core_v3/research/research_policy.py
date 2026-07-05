"""
Research policy for Remy v3.

Centralizes evidence thresholds so Research Ops completion is determined by
one deterministic contract rather than scattered heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchThresholds:
    min_sources: int = 2
    min_findings: int = 3
    min_confidence: float = 0.4
    require_contradiction_check: bool = True


@dataclass
class ResearchEvidenceAssessment:
    verdict: str = "failure"
    reason: str = ""
    usable_sources: int = 0
    finding_count: int = 0
    confidence: float = 0.0
    contradictions_checked: bool = False

    @property
    def is_success(self) -> bool:
        return self.verdict == "success"

    @property
    def is_partial(self) -> bool:
        return self.verdict == "partial"


def assess_project(project, thresholds: ResearchThresholds | None = None) -> ResearchEvidenceAssessment:
    thresholds = thresholds or ResearchThresholds()
    synthesis = project.synthesis
    usable_sources = sum(1 for source in project.sources if source.is_usable())
    finding_count = len(project.findings)
    confidence = float(getattr(synthesis, "confidence", 0.0) or 0.0)
    contradictions_checked = bool(project.contradictions or not thresholds.require_contradiction_check)
    if thresholds.require_contradiction_check:
        contradictions_checked = True

    return assess_evidence(
        usable_sources=usable_sources,
        finding_count=finding_count,
        confidence=confidence,
        contradictions_checked=contradictions_checked,
        thresholds=thresholds,
    )


def assess_evidence(
    *,
    usable_sources: int,
    finding_count: int,
    confidence: float,
    contradictions_checked: bool,
    thresholds: ResearchThresholds | None = None,
) -> ResearchEvidenceAssessment:
    thresholds = thresholds or ResearchThresholds()

    if (
        usable_sources >= thresholds.min_sources
        and finding_count >= thresholds.min_findings
        and confidence >= thresholds.min_confidence
        and (contradictions_checked or not thresholds.require_contradiction_check)
    ):
        return ResearchEvidenceAssessment(
            verdict="success",
            reason=(
                f"Research evidence threshold met ({finding_count} findings, "
                f"{usable_sources} sources, confidence {confidence:.2f})"
            ),
            usable_sources=usable_sources,
            finding_count=finding_count,
            confidence=confidence,
            contradictions_checked=contradictions_checked,
        )

    if finding_count > 0 or usable_sources > 0:
        return ResearchEvidenceAssessment(
            verdict="partial",
            reason=(
                f"Research evidence incomplete ({finding_count} findings, "
                f"{usable_sources} sources, confidence {confidence:.2f})"
            ),
            usable_sources=usable_sources,
            finding_count=finding_count,
            confidence=confidence,
            contradictions_checked=contradictions_checked,
        )

    return ResearchEvidenceAssessment(
        verdict="failure",
        reason="Research produced no usable evidence",
        usable_sources=usable_sources,
        finding_count=finding_count,
        confidence=confidence,
        contradictions_checked=contradictions_checked,
    )


def assess_evidence_dict(evidence: dict[str, Any], thresholds: ResearchThresholds | None = None) -> ResearchEvidenceAssessment:
    synthesis = evidence.get("synthesis") or {}
    findings = evidence.get("findings") or []
    artifacts = evidence.get("artifacts") or []
    usable_sources = int(synthesis.get("source_count") or len(artifacts))
    finding_count = int(synthesis.get("finding_count") or len(findings))
    confidence_raw = synthesis.get("confidence")
    confidence = float(confidence_raw or 0.0)
    if confidence_raw in (None, "") and usable_sources >= 2 and finding_count >= 3:
        confidence = 0.5
    contradictions_checked = bool(evidence.get("contradictions_checked"))

    assessment = assess_evidence(
        usable_sources=usable_sources,
        finding_count=finding_count,
        confidence=confidence,
        contradictions_checked=contradictions_checked,
        thresholds=thresholds,
    )
    if assessment.verdict == "partial" and not findings and not artifacts:
        assessment.verdict = "failure"
        assessment.reason = "No evidence, artifacts, or tool activity"
    return assessment
