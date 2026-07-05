"""
Memory API for Remy v3.

Thin, typed interface over Aura SDK. Replaces direct _AuraCompat usage
with explicit memory classes (Identity, Working, Task, Outcome, Strategic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory class taxonomy (maps to Aura tiers)
# ---------------------------------------------------------------------------

class MemoryClass(str, Enum):
    """Logical memory classes from ROADMAP § Memory Architecture."""
    IDENTITY = "identity"       # → Aura Level.Identity
    WORKING = "working"         # → Aura Level.Working
    TASK = "task"               # → Aura Level.Decisions
    OUTCOME = "outcome"         # → Aura Level.Decisions
    STRATEGIC = "strategic"     # → Aura Level.Domain


# Mapping from v3 MemoryClass to Aura Level names
_CLASS_TO_LEVEL = {
    MemoryClass.IDENTITY: "Identity",
    MemoryClass.WORKING: "Working",
    MemoryClass.TASK: "Decisions",
    MemoryClass.OUTCOME: "Decisions",
    MemoryClass.STRATEGIC: "Domain",
}


# ---------------------------------------------------------------------------
# Record wrapper
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """Typed wrapper around an Aura record."""
    id: str = ""
    content: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_class: MemoryClass = MemoryClass.WORKING
    semantic_type: str = "fact"
    score: float = 0.0             # Recall relevance score
    level: str = ""                # Raw Aura level name
    timestamp: float = 0.0

    @property
    def record_type(self) -> str:
        """Extract structured record type from tags (mission, goal, finding, ...)."""
        for tag in self.tags:
            if tag in _RECORD_TYPES:
                return tag
        return "unknown"


_RECORD_TYPES = frozenset({
    "mission", "goal", "plan", "task", "finding", "outcome",
    "failure", "hypothesis", "playbook", "improvement-suggestion",
    "tool-health", "budget-event", "approval-event",
})


# ---------------------------------------------------------------------------
# Memory API Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol that any memory backend must satisfy.

    The default implementation wraps _AuraCompat from v2.
    """

    def store(
        self,
        content: str,
        tags: list[str],
        metadata: dict[str, Any] | None = None,
        memory_class: MemoryClass = MemoryClass.WORKING,
        deduplicate: bool = False,
    ) -> str:
        """Store a record. Returns record ID.

        ``deduplicate`` opts a write into the backend's content-similarity
        dedup (reactivate+merge an existing near-identical record instead of
        inserting a new one). Off by default so factual/admission writes are
        never silently merged; enable it only for append-log records
        (outcome/failure/goal/research) where identical content is a true
        repeat, not a distinct fact.
        """
        ...

    def capture_consequence(
        self,
        *,
        situation: str,
        action: str,
        consequence: str,
        trust: int = 0,
        scope: list[str] | None = None,
        provenance: list[str] | None = None,
        links: dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Store a lived situation -> action -> consequence unit. Returns record ID."""
        ...

    def policy_hint(
        self,
        situation: str,
        action: str,
        namespace: str | None = None,
    ) -> dict:
        """Runtime policy hint for a proposed (situation, action)."""
        ...

    def recall(
        self,
        query: str,
        memory_class: MemoryClass | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Recall records by semantic query."""
        ...

    def get(self, record_id: str) -> MemoryRecord | None:
        """Get a specific record by ID."""
        ...

    def update(
        self,
        record_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing record."""
        ...

    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        ...

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """Search records (tag-filtered, broad)."""
        ...

    def list_records(
        self,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """List records by tags."""
        ...

    def run_maintenance(self) -> dict[str, Any]:
        """Run Aura maintenance cycle. Returns report dict."""
        ...


# ---------------------------------------------------------------------------
# Aura-backed implementation
# ---------------------------------------------------------------------------

class AuraMemoryBackend:
    """Memory backend backed by _AuraCompat from v2 agent_tools.

    This is the adapter that bridges v2 Aura singleton → v3 MemoryBackend.
    """

    def __init__(self, aura_compat):
        """
        Args:
            aura_compat: An _AuraCompat instance (from agent_tools.brain).
        """
        self._aura = aura_compat
        self._level_map = self._build_level_map()

    def _build_level_map(self) -> dict[MemoryClass, Any]:
        """Map MemoryClass → Aura Level enum values."""
        try:
            from aura import Level
            return {
                MemoryClass.IDENTITY: Level.Identity,
                MemoryClass.WORKING: Level.Working,
                MemoryClass.TASK: Level.Decisions,
                MemoryClass.OUTCOME: Level.Decisions,
                MemoryClass.STRATEGIC: Level.Domain,
            }
        except ImportError:
            log.warning("aura SDK not available, using string level names")
            return {}

    def _to_level(self, mc: MemoryClass):
        """Convert MemoryClass to Aura Level."""
        if self._level_map:
            return self._level_map.get(mc)
        return _CLASS_TO_LEVEL.get(mc, "Working")

    def _wrap_record(self, record, score: float = 0.0) -> MemoryRecord:
        """Convert Aura record to MemoryRecord."""
        from remy.core.memory_policy import infer_semantic_type

        content = getattr(record, "content", "") or ""
        tags = list(getattr(record, "tags", []) or [])
        metadata = dict(getattr(record, "metadata", {}) or {})
        level_name = str(getattr(record, "level", ""))
        rec_id = getattr(record, "id", "") or str(getattr(record, "record_id", ""))

        # Infer memory class from level
        mc = MemoryClass.WORKING
        level_lower = level_name.lower()
        if "identity" in level_lower:
            mc = MemoryClass.IDENTITY
        elif "domain" in level_lower:
            mc = MemoryClass.STRATEGIC
        elif "decision" in level_lower:
            mc = MemoryClass.TASK

        semantic_type = infer_semantic_type(
            explicit=metadata.get("semantic_type"),
            tags=tags,
            level=getattr(record, "level", None),
            content=content,
        )
        metadata.setdefault("semantic_type", semantic_type)

        return MemoryRecord(
            id=rec_id,
            content=content,
            tags=tags,
            metadata=metadata,
            memory_class=mc,
            semantic_type=semantic_type,
            score=score,
            level=level_name,
            timestamp=float(metadata.get("created_at", 0)),
        )

    # --- MemoryBackend implementation ---

    def store(
        self,
        content: str,
        tags: list[str],
        metadata: dict[str, Any] | None = None,
        memory_class: MemoryClass = MemoryClass.WORKING,
        deduplicate: bool = False,
    ) -> str:
        from remy.core.memory_policy import infer_semantic_type

        meta = dict(metadata or {})
        meta["memory_class"] = memory_class.value
        semantic_type = infer_semantic_type(
            explicit=meta.get("semantic_type"),
            tags=tags,
            level=self._to_level(memory_class),
            content=content,
        )
        meta["semantic_type"] = semantic_type
        try:
            result = self._aura.store(
                content,
                tags=tags,
                metadata=meta,
                semantic_type=semantic_type,
                deduplicate=deduplicate,
            )
        except TypeError:
            # Older Aura surface without the deduplicate kwarg — fail safe to a
            # plain insert rather than dropping the write.
            result = self._aura.store(
                content,
                tags=tags,
                metadata=meta,
                semantic_type=semantic_type,
            )
        return getattr(result, "id", str(result))

    def _resolve_existing_record_link(self, relation: str, logical_id: str) -> str:
        """Resolve a Remy runtime id to an existing Aura record id when possible."""
        logical_id = (logical_id or "").strip()
        if not logical_id:
            return ""

        try:
            existing = self._aura.get(logical_id)
            if existing is not None:
                return logical_id
        except Exception:
            pass

        relation_keys = {
            "mission": ("mission_id",),
            "goal": ("goal_id",),
            "task": ("task_id",),
            "step": ("step_id", "task_id"),
        }.get(relation, ())
        if not relation_keys:
            return ""

        try:
            candidates = self._aura.search(query="", limit=500)
        except Exception:
            candidates = []

        best_id = ""
        best_score = -1
        for raw in candidates or []:
            record = raw[1] if isinstance(raw, tuple) and len(raw) == 2 else raw
            metadata = dict(getattr(record, "metadata", {}) or {})
            if not any(str(metadata.get(key, "")).strip() == logical_id for key in relation_keys):
                continue

            tags = set(getattr(record, "tags", []) or [])
            score = 1
            if relation in tags:
                score += 4
            if f"{relation}_outcome" in tags:
                score += 2
            if "outcome" in tags or "failure" in tags:
                score += 1

            record_id = getattr(record, "id", "") or str(getattr(record, "record_id", ""))
            if record_id and score > best_score:
                best_id = record_id
                best_score = score

        return best_id

    def _resolve_consequence_links(self, links: dict[str, str]) -> dict[str, str]:
        """Preserve logical links and add concrete Aura record links when found."""
        resolved = {
            str(key): str(value)
            for key, value in (links or {}).items()
            if str(key).strip() and str(value).strip()
        }

        for relation in ("mission", "goal", "task", "step"):
            logical_id = resolved.get(relation, "")
            if not logical_id:
                continue
            record_id = self._resolve_existing_record_link(relation, logical_id)
            if record_id and record_id != logical_id:
                resolved.setdefault(f"{relation}_record", record_id)

        return resolved

    def capture_consequence(
        self,
        *,
        situation: str,
        action: str,
        consequence: str,
        trust: int = 0,
        scope: list[str] | None = None,
        provenance: list[str] | None = None,
        links: dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Persist a first-class consequence unit when AuraSDK supports it.

        Older bundled Aura wheels do not expose `capture_consequence`, so this
        keeps Remy compatible by falling back to a structured outcome record.
        """
        scope = list(scope or [])
        provenance = list(provenance or ["remy:memory_api"])
        links = self._resolve_consequence_links(dict(links or {}))
        namespace = namespace or "default"

        if hasattr(self._aura, "capture_consequence"):
            unit = self._aura.capture_consequence(
                situation=situation,
                action=action,
                consequence=consequence,
                trust=int(trust),
                scope=scope,
                provenance=provenance,
                links=links,
                namespace=namespace,
            )
            if isinstance(unit, dict):
                return str(unit.get("record_id") or unit.get("id") or "")
            return str(getattr(unit, "record_id", None) or getattr(unit, "id", None) or unit)

        polarity_tag = "consequence-inconclusive"
        if trust > 0 or consequence.upper() == "SUPPORTS":
            polarity_tag = "consequence-support"
        elif trust < 0 or consequence.upper() == "REFUTES":
            polarity_tag = "consequence-refute"

        return self.store(
            content=(
                f"[CONSEQUENCE] state={situation}; action={action}; "
                f"consequence={consequence}; trust={trust}"
            ),
            tags=["outcome", "consequence-unit", polarity_tag],
            metadata={
                "kind": "consequence_unit",
                "cu_situation": situation,
                "cu_action": action,
                "cu_consequence": consequence,
                "cu_trust": trust,
                "cu_scope": scope,
                "cu_provenance": provenance,
                "cu_links": links,
                "namespace": namespace,
                "semantic_type": "outcome",
            },
            memory_class=MemoryClass.OUTCOME,
        )

    @staticmethod
    def _consequence_key(value: str) -> str:
        return " ".join((value or "").strip().casefold().split())

    def _fallback_consequence_verdict(
        self, situation: str, action: str, namespace: str | None = None
    ) -> dict:
        """Compute a scar-protected verdict from fallback consequence records.

        This keeps Remy useful with older AuraSDK wheels that can store records
        but do not yet expose the native `consequence_verdict` API.
        """
        situation_key = self._consequence_key(situation)
        action_key = self._consequence_key(action)
        namespace_key = namespace or "default"
        if not situation_key or not action_key:
            return {
                "verdict": "inconclusive",
                "supports": 0,
                "refutes": 0,
                "inconclusive": 0,
                "abstain": True,
            }

        try:
            records = self.search(query="", tags=["consequence-unit"], limit=1000)
        except Exception:
            records = []

        supports = 0
        refutes = 0
        inconclusive = 0
        for record in records:
            meta = dict(getattr(record, "metadata", {}) or {})
            if meta.get("kind") != "consequence_unit":
                continue
            rec_namespace = str(meta.get("namespace") or "default")
            if namespace is not None and rec_namespace != namespace_key:
                continue
            if self._consequence_key(str(meta.get("cu_situation") or "")) != situation_key:
                continue
            if self._consequence_key(str(meta.get("cu_action") or "")) != action_key:
                continue

            consequence = str(meta.get("cu_consequence") or "").upper()
            trust = int(meta.get("cu_trust") or 0)
            if consequence == "REFUTES" or trust < 0:
                refutes += 1
            elif consequence == "SUPPORTS" or trust > 0:
                supports += 1
            else:
                inconclusive += 1

        if refutes > 0:
            verdict = "refutes"
        elif supports > 0:
            verdict = "supports"
        else:
            verdict = "inconclusive"
        return {
            "verdict": verdict,
            "supports": supports,
            "refutes": refutes,
            "inconclusive": inconclusive,
            "abstain": verdict == "inconclusive",
        }

    def consequence_verdict(
        self, situation: str, action: str, namespace: str | None = None
    ) -> dict:
        """Scar-protected verdict for a (situation, action) pair.

        Passes through to AuraSDK's `Aura.consequence_verdict`, where a lived
        REFUTES outranks any amount of later supporting frequency. Older bundled
        wheels without this surface return an inconclusive/abstain dict so callers
        (e.g. the cycle scar-check) degrade safely instead of crashing.
        """
        if hasattr(self._aura, "consequence_verdict"):
            try:
                return self._aura.consequence_verdict(
                    situation, action, namespace=namespace
                )
            except TypeError:
                return self._aura.consequence_verdict(situation, action)
        return self._fallback_consequence_verdict(situation, action, namespace)

    def policy_hint(
        self, situation: str, action: str, namespace: str | None = None
    ) -> dict:
        """Return AuraSDK policy hint, or derive one from consequence verdict."""
        if hasattr(self._aura, "policy_hint"):
            try:
                raw = self._aura.policy_hint(situation, action, namespace=namespace)
                if isinstance(raw, dict):
                    return raw
            except TypeError:
                raw = self._aura.policy_hint(situation, action)
                if isinstance(raw, dict):
                    return raw
            except Exception:
                pass

        verdict = self.consequence_verdict(situation, action, namespace=namespace)
        if str(verdict.get("verdict") or "") == "refutes":
            hint = "avoid"
        elif str(verdict.get("verdict") or "") == "supports":
            hint = "prefer"
        else:
            hint = "verify_first"
        return {
            "hint": hint,
            "verdict": verdict.get("verdict", "inconclusive"),
            "supports": int(verdict.get("supports", 0) or 0),
            "refutes": int(verdict.get("refutes", 0) or 0),
            "requires_evidence": hint in {"verify_first", "requires_evidence"},
            "should_block": hint == "avoid",
        }

    def recall(
        self,
        query: str,
        memory_class: MemoryClass | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        level = self._to_level(memory_class) if memory_class else None
        raw = self._aura.search(query=query, tags=tags, limit=limit, level=level)

        results = []
        for item in raw:
            if isinstance(item, tuple) and len(item) == 2:
                score, record = item
                results.append(self._wrap_record(record, score=float(score)))
            else:
                results.append(self._wrap_record(item))
        return results[:limit]

    def get(self, record_id: str) -> MemoryRecord | None:
        record = self._aura.get(record_id)
        if record is None:
            return None
        return self._wrap_record(record)

    def update(
        self,
        record_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return self._aura.update(record_id, metadata=metadata, content=content, tags=tags)

    def delete(self, record_id: str) -> bool:
        if hasattr(self._aura, "delete"):
            return self._aura.delete(record_id)
        return False

    def search(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        raw = self._aura.search(query, tags=tags, limit=limit)
        return [self._wrap_record(r) for r in raw][:limit]

    def list_records(
        self,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        raw = self._aura.list_records(tags=tags, limit=limit)
        return [self._wrap_record(r) for r in raw][:limit]

    def run_maintenance(self) -> dict[str, Any]:
        if hasattr(self._aura, "_aura") and hasattr(self._aura._aura, "run_maintenance"):
            report = self._aura._aura.run_maintenance()
            reflect = getattr(report, "reflect", None)
            decay = getattr(report, "decay", None)
            consolidation = getattr(report, "consolidation", None)
            return {
                "promoted": getattr(reflect, "promoted", 0) if reflect else 0,
                "decayed": getattr(decay, "decayed", 0) if decay else 0,
                "pruned": getattr(decay, "pruned", 0) if decay else 0,
                "consolidated": getattr(consolidation, "consolidated", 0) if consolidation else 0,
            }
        return {}


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_memory: AuraMemoryBackend | None = None


def get_memory() -> AuraMemoryBackend:
    """Get or create the global v3 memory backend.

    Lazily wraps the v2 brain singleton.
    """
    global _memory
    if _memory is None:
        from remy.core.agent_tools import brain
        _memory = AuraMemoryBackend(brain)
    return _memory
