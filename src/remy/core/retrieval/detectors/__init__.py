"""Claim detectors — emit ClaimSpans from a response text.

Each detector is a pure function: text -> list[ClaimSpan]. Detectors do NOT
resolve evidence. The orchestrator hands each span to evidence_resolver.
"""

from remy.core.retrieval.detectors.entitlement import detect_entitlement
from remy.core.retrieval.detectors.external_ids import detect_external_ids
from remy.core.retrieval.detectors.live_telemetry import detect_live_telemetry


ALL_DETECTORS = (
    detect_external_ids,
    detect_live_telemetry,
    detect_entitlement,
)


def detect_all(text: str) -> list:
    """Run all detectors over *text* and return the flattened span list."""
    spans = []
    for det in ALL_DETECTORS:
        spans.extend(det(text))
    return spans


__all__ = [
    "detect_external_ids",
    "detect_live_telemetry",
    "detect_entitlement",
    "detect_all",
    "ALL_DETECTORS",
]
