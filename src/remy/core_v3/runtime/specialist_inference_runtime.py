"""
Specialist inference runtime for Remy v3.

Owns heuristic specialist inference so this policy no longer lives on
ChiefAgent.
"""

from __future__ import annotations


class SpecialistInferenceRuntime:
    """Infer a specialist id from free-form task or step text."""

    def infer(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ("research", "analyze", "find", "search", "study")):
            return "researcher"
        if any(w in text_lower for w in ("browse", "signup", "register", "navigate", "publish")):
            return "executor"
        return "analyst"
