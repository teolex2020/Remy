"""
Playbook Engine for Remy v3 Self-Improvement.

Generates and stores reusable playbooks from successful executions.
A playbook is a proven sequence of steps that worked for a type of goal.

Future: LLM-assisted playbook refinement and goal-to-playbook matching.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class PlaybookStep:
    """A single step in a playbook."""
    order: int = 0
    action: str = ""
    specialist: str = ""
    tools: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    notes: str = ""


@dataclass
class Playbook:
    """A reusable execution playbook."""
    id: str = field(default_factory=lambda: f"pb_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    goal_pattern: str = ""     # What type of goal this applies to
    domain: str = ""           # research, signup, publishing, etc.
    steps: list[PlaybookStep] = field(default_factory=list)

    # Track record
    times_used: int = 0
    times_succeeded: int = 0
    avg_cost_usd: float = 0.0
    avg_duration_ms: int = 0

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.times_succeeded / self.times_used if self.times_used else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "goal_pattern": self.goal_pattern,
            "steps": [
                {"order": s.order, "action": s.action, "specialist": s.specialist}
                for s in self.steps
            ],
            "success_rate": round(self.success_rate, 2),
            "times_used": self.times_used,
        }


class PlaybookEngine:
    """Generates, stores, and matches playbooks."""

    def __init__(self):
        self._playbooks: list[Playbook] = []
        self._stored_playbook_ids: set[str] = set()

    def create_from_execution(
        self,
        name: str,
        goal_description: str,
        domain: str,
        steps: list[dict[str, Any]],
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ) -> Playbook:
        """Create a playbook from a successful execution."""
        pb_steps = []
        for i, step in enumerate(steps):
            pb_steps.append(PlaybookStep(
                order=i + 1,
                action=step.get("action", step.get("description", "")),
                specialist=step.get("specialist", ""),
                tools=step.get("tools", []),
                expected_outcome=step.get("outcome", ""),
            ))

        pb = Playbook(
            name=name,
            description=f"Playbook for: {goal_description[:100]}",
            goal_pattern=self._extract_pattern(goal_description),
            domain=domain,
            steps=pb_steps,
            times_used=1,
            times_succeeded=1,
            avg_cost_usd=cost_usd,
            avg_duration_ms=duration_ms,
        )

        # Check for existing similar playbook
        existing = self.match(goal_description, domain)
        if existing:
            # Update existing rather than create duplicate
            existing.times_used += 1
            existing.times_succeeded += 1
            existing.avg_cost_usd = (existing.avg_cost_usd + cost_usd) / 2
            existing.avg_duration_ms = (existing.avg_duration_ms + duration_ms) // 2
            existing.updated_at = time.time()
            return existing

        self._playbooks.append(pb)
        log.info("Playbook created: %s (%s)", pb.name, pb.id)
        return pb

    def record_usage(self, playbook_id: str, success: bool, cost_usd: float = 0.0):
        """Record a playbook usage attempt."""
        pb = self.get(playbook_id)
        if pb:
            pb.times_used += 1
            if success:
                pb.times_succeeded += 1
            pb.avg_cost_usd = (pb.avg_cost_usd * (pb.times_used - 1) + cost_usd) / pb.times_used
            pb.updated_at = time.time()

    def match(self, goal_description: str, domain: str = "") -> Playbook | None:
        """Find the best matching playbook for a goal."""
        candidates = self._playbooks
        if domain:
            domain_matches = [pb for pb in candidates if pb.domain == domain]
            if domain_matches:
                candidates = domain_matches

        if not candidates:
            return None

        # Simple keyword overlap scoring
        goal_words = set(goal_description.lower().split())
        best_pb = None
        best_score = 0

        for pb in candidates:
            pattern_words = set(pb.goal_pattern.lower().split())
            overlap = len(goal_words & pattern_words)
            # Weight by success rate
            score = overlap * (0.5 + pb.success_rate * 0.5)
            if score > best_score:
                best_score = score
                best_pb = pb

        return best_pb if best_score >= 2 else None

    def get(self, playbook_id: str) -> Playbook | None:
        for pb in self._playbooks:
            if pb.id == playbook_id:
                return pb
        return None

    def list_playbooks(self, domain: str = "") -> list[Playbook]:
        if domain:
            return [pb for pb in self._playbooks if pb.domain == domain]
        return list(self._playbooks)

    def top_playbooks(self, limit: int = 5) -> list[Playbook]:
        """Get most successful playbooks."""
        return sorted(
            self._playbooks,
            key=lambda pb: (pb.success_rate, pb.times_used),
            reverse=True,
        )[:limit]

    def store_to_memory(self):
        """Persist playbooks to Aura memory."""
        try:
            from ..memory.memory_api import get_memory, MemoryClass
            from ..memory.record_models import playbook_record

            memory = get_memory()
            for pb in self._playbooks:
                if pb.times_used >= 2 and pb.success_rate >= 0.5 and pb.id not in self._stored_playbook_ids:
                    steps_str = " → ".join(s.action[:30] for s in pb.steps[:5])
                    memory.store(
                        content=(
                            f"[PLAYBOOK] {pb.name}: {steps_str} "
                            f"(success: {pb.success_rate:.0%}, used {pb.times_used}x)"
                        ),
                        tags=["playbook", pb.domain],
                        metadata={
                            "playbook_id": pb.id,
                            "domain": pb.domain,
                            "success_rate": pb.success_rate,
                        },
                        memory_class=MemoryClass.STRATEGIC,
                    )
                    self._stored_playbook_ids.add(pb.id)
        except Exception as e:
            log.debug("Failed to store playbooks to memory: %s", e)

    def _extract_pattern(self, description: str) -> str:
        """Extract a reusable pattern from a goal description."""
        import re
        # Remove URLs
        pattern = re.sub(r'https?://\S+', '', description)
        # Remove hex-like IDs (8+ hex chars, not regular words)
        pattern = re.sub(r'\b[0-9a-f]{8,}\b', '', pattern)
        # Remove UUIDs
        pattern = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '', pattern)
        return pattern.strip()[:100]

    def summary(self) -> dict[str, Any]:
        return {
            "total_playbooks": len(self._playbooks),
            "by_domain": {
                d: len([p for p in self._playbooks if p.domain == d])
                for d in set(p.domain for p in self._playbooks)
            } if self._playbooks else {},
            "top_3": [pb.to_dict() for pb in self.top_playbooks(3)],
        }


_playbook_engine: PlaybookEngine | None = None


def get_playbook_engine() -> PlaybookEngine:
    global _playbook_engine
    if _playbook_engine is None:
        _playbook_engine = PlaybookEngine()
    return _playbook_engine
