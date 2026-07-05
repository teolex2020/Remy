"""
Remy — Memory instances.

Two memory systems:
  - brain (Aura Cognitive): Episodic memory — conversations, decisions, context. Has decay.
  - knowledge (Aura Memory): Semantic knowledge base — facts, documents, anchors. No decay.
"""

import atexit
import inspect
import logging
import os
import sys
import threading
import json
import shutil
import subprocess
import time
from pathlib import Path
from collections import Counter
from aura import Aura
from aura import Level as _AuraLevel
from aura import AgentPersona, PersonaTraits, TagTaxonomy, TrustConfig, ArchivalRule, MaintenanceConfig
from remy.config.settings import settings
from remy.core.history_replay import replay_history


logger = logging.getLogger(__name__)
_brain_close_lock = threading.Lock()
_brain_init_lock = threading.Lock()
_brain_closed = False
_brain_shutdown_started = False
_brain_close_error = ""
_brain_instance = None
_brain_initialized = False
_brain_quarantined_at_startup = False
_brain_quarantine_reason = ""
_brain_startup_blocked = False
_brain_startup_incident = ""
_brain_quarantine_path = ""
_brain_backup_path = ""
_brain_recovery_stats: dict = {}
_brain_startup_artifact_id = ""


_STRING_METADATA_KEYS = frozenset({
    "id", "timestamp", "source", "volatility", "query", "task",
    "created_at", "updated_at", "verified_at", "verified_by",
    "source_type", "channel", "goal_id", "parent_goal_id",
    "record_id", "report_id", "project_id", "todo_id", "session_id",
    "type", "status", "due_date", "last_attempt", "repeat",
    "category", "full_name", "role", "note", "topic_slug",
    "semantic_type",
})


class _CompatRecord:
    def __init__(self, rec, metadata):
        self.id = getattr(rec, "id", None)
        self.content = getattr(rec, "content", "")
        self.tags = list(getattr(rec, "tags", []) or [])
        self.level = getattr(rec, "level", None)
        self.strength = getattr(rec, "strength", 0.0)
        self.activation_count = getattr(rec, "activation_count", 0)
        self.connections = dict(getattr(rec, "connections", {}) or {})
        self.importance = getattr(rec, "importance", None)
        self.metadata = metadata
        self._record = rec

    def __getattr__(self, name):
        return getattr(self._record, name)

    def __getitem__(self, key):
        if key == "metadata":
            return self.metadata
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key == "metadata" or hasattr(self, key)

    def keys(self):
        return ("id", "content", "tags", "level", "strength", "activation_count", "connections", "importance", "metadata")

    def items(self):
        return [(key, self.get(key)) for key in self.keys()]

    def to_dict(self):
        return {key: self.get(key) for key in self.keys()}


class _StoreResult:
    """Minimal wrapper so older code can keep using `rec.id` after store()."""

    __slots__ = ("id",)

    def __init__(self, record_id: str):
        self.id = record_id

    def __str__(self):
        return self.id

    def __repr__(self):
        return f"_StoreResult(id={self.id!r})"


class _LevelCompat:
    """Compatibility shim for Aura enum naming drift.

    Older code expects uppercase enum attrs (`WORKING`), while the installed
    Aura build exposes capitalized attrs (`Working`).
    """

    WORKING = getattr(_AuraLevel, "WORKING", getattr(_AuraLevel, "Working"))
    DECISIONS = getattr(_AuraLevel, "DECISIONS", getattr(_AuraLevel, "Decisions"))
    DOMAIN = getattr(_AuraLevel, "DOMAIN", getattr(_AuraLevel, "Domain"))
    IDENTITY = getattr(_AuraLevel, "IDENTITY", getattr(_AuraLevel, "Identity"))

    Working = WORKING
    Decisions = DECISIONS
    Domain = DOMAIN
    Identity = IDENTITY


Level = _LevelCompat

COGNITIVE_LEVELS = (Level.WORKING, Level.DECISIONS)
CORE_LEVELS = (Level.DOMAIN, Level.IDENTITY)

_LEVEL_NAMES = {
    int(Level.WORKING): "WORKING",
    int(Level.DECISIONS): "DECISIONS",
    int(Level.DOMAIN): "DOMAIN",
    int(Level.IDENTITY): "IDENTITY",
}

def level_name(level) -> str:
    """Return a human-readable level name like 'IDENTITY'."""
    if isinstance(level, str):
        return level.upper().replace("LEVEL.", "")
    try:
        return _LEVEL_NAMES.get(int(level), str(level))
    except (TypeError, ValueError):
        return str(level)


def tier_of(level) -> str:
    """Return 'cognitive' or 'core' based on the requested level."""
    if isinstance(level, str):
        level_str = level.upper().replace('LEVEL.', '')
        if level_str in ('WORKING', 'DECISIONS'):
            return "cognitive"
        if level_str in ('DOMAIN', 'IDENTITY'):
            return "core"
        return "core"
    
    if level in COGNITIVE_LEVELS:
        return "cognitive"
    return "core"


def _apply_factual_recall_filter(items):
    """Phase 3 Step 2: apply promotion/conflict/supersession gate to recall output.

    Delegates to hybrid_search._is_factual_forbidden so tag/admission-class/
    promotion-flag/truth-state semantics stay in one place. Best-effort — on
    import error, pass records through unfiltered rather than blocking recall.

    Step 4: each blocked record emits a structured promotion_audit event so
    we can see *why* something didn't reach the LLM, not just that it didn't.
    """
    try:
        from remy.core.hybrid_search import _is_factual_forbidden
    except Exception:
        return items
    if not items:
        return []
    kept = []
    blocked = []
    for item in items:
        if _is_factual_forbidden(item):
            blocked.append(item)
        else:
            kept.append(item)
    if blocked:
        try:
            from remy.core.promotion_audit import (
                SURFACE_RECALL,
                block_reason,
                record_block,
            )
            for item in blocked:
                reason = block_reason(item) or "unknown"
                rid = item.get("id") if isinstance(item, dict) else None
                record_block(SURFACE_RECALL, rid, reason)
        except Exception:
            pass
    return kept


# ── Phase 3 Step 3: brain.connect() promotion gate ──────────────────────────


def _record_to_check_item(rec) -> dict:
    """Shape a brain record (object or dict) into the dict `_is_factual_forbidden`
    expects. Keeps the gate's rule-set in one place."""
    if rec is None:
        return {}
    if isinstance(rec, dict):
        return {
            "id": rec.get("id"),
            "tags": list(rec.get("tags") or []),
            "metadata": dict(rec.get("metadata") or {}),
        }
    return {
        "id": getattr(rec, "id", None),
        "tags": list(getattr(rec, "tags", []) or []),
        "metadata": dict(getattr(rec, "metadata", {}) or {}),
    }


def promotion_allowed(rec) -> bool:
    """Return True if this record is eligible to participate in promoted
    structure (concept / causal / policy) via `brain.connect()`.

    Mirrors `_is_factual_forbidden` inverted: a record is allowed when it
    does not carry any admission/promotion/conflict/supersession/freshness
    signal that would block it from factual recall. Admitted-but-not-promoted
    records stay in memory but do not form graph edges.

    Best-effort: on import/shape error, allow (never block promotion on
    an infrastructure failure — matches the existing pattern for
    `_is_factual_forbidden`).
    """
    try:
        from remy.core.hybrid_search import _is_factual_forbidden
    except Exception:
        return True
    item = _record_to_check_item(rec)
    if not item:
        return True
    try:
        return not _is_factual_forbidden(item)
    except Exception:
        return True


def gated_connect(brain_handle, id_a, id_b, weight: float = 0.0) -> bool:
    """Promotion-gated wrapper around `brain.connect()`.

    Reads both endpoints via `brain_handle.get(id)`, evaluates
    `promotion_allowed()` on each, and only calls `brain_handle.connect(...)`
    when BOTH endpoints are eligible. Blocking one endpoint blocks the edge
    (strictest option — matches audit intent: promotion-forbidden records
    must not silently form concept/causal/policy edges).

    Returns True if the edge was created, False if blocked. Missing
    endpoints (brain.get returns None) default to False — an edge to a
    vanished record is never safe.

    Step 4: a blocked edge emits one promotion_audit event naming the
    offending endpoint and the first signal that blocked it.
    """
    def _emit(record_id, reason, partner_id):
        try:
            from remy.core.promotion_audit import (
                SURFACE_CONNECT,
                record_block,
            )
            record_block(
                SURFACE_CONNECT,
                record_id,
                reason,
                extra={"partner_id": partner_id, "weight": weight},
            )
        except Exception:
            pass

    try:
        rec_a = brain_handle.get(id_a)
        rec_b = brain_handle.get(id_b)
    except Exception:
        try:
            from remy.core.promotion_audit import (
                REASON_SDK_FAILURE,
                SURFACE_CONNECT,
                record_block,
            )
            record_block(
                SURFACE_CONNECT,
                id_a,
                REASON_SDK_FAILURE,
                extra={"partner_id": id_b, "weight": weight},
            )
        except Exception:
            pass
        return False
    if rec_a is None:
        try:
            from remy.core.promotion_audit import REASON_MISSING_ENDPOINT
            _emit(id_a, REASON_MISSING_ENDPOINT, id_b)
        except Exception:
            pass
        return False
    if rec_b is None:
        try:
            from remy.core.promotion_audit import REASON_MISSING_ENDPOINT
            _emit(id_b, REASON_MISSING_ENDPOINT, id_a)
        except Exception:
            pass
        return False
    try:
        from remy.core.promotion_audit import block_reason
        reason_a = block_reason(_record_to_check_item(rec_a))
        reason_b = block_reason(_record_to_check_item(rec_b))
    except Exception:
        reason_a = None if promotion_allowed(rec_a) else "unknown"
        reason_b = None if promotion_allowed(rec_b) else "unknown"
    if reason_a is not None:
        _emit(id_a, reason_a, id_b)
    if reason_b is not None:
        _emit(id_b, reason_b, id_a)
    if reason_a is not None or reason_b is not None:
        return False
    try:
        brain_handle.connect(id_a, id_b, weight=weight)
    except TypeError:
        # Some callers historically passed weight positionally.
        brain_handle.connect(id_a, id_b, weight)
    return True


class _AuraCompat:
    """
    Compatibility layer — adapts the new Aura SDK API to the old CognitiveMemory API.
    Uses composition (not inheritance) because the Rust Aura type cannot be subclassed.
    Handles metadata serialization and translates method names like recall_core.
    """
    def __init__(self, path: str):
        self._aura = Aura(path)
        store_sig = inspect.signature(self._aura.store)
        self._has_auto_promote = "auto_promote" in store_sig.parameters
        # Check if source_type/semantic_type are explicit params OR accepted via **kwargs
        _has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in store_sig.parameters.values()
        )
        self._has_source_type = "source_type" in store_sig.parameters or _has_var_keyword
        self._has_semantic_type = "semantic_type" in store_sig.parameters or _has_var_keyword
        self._has_recall_full = hasattr(self._aura, "recall_full")
        self._has_promotion_candidates = hasattr(self._aura, "promotion_candidates")
        self._has_tier_stats = hasattr(self._aura, "tier_stats")

    def __getattr__(self, name):
        # Proxy all unknown attributes to the underlying Aura instance
        return getattr(self._aura, name)

    def _deserialize_metadata(self, metadata: dict) -> dict:
        if not metadata:
            return {}
        res = {}
        for k, v in metadata.items():
            if k in _STRING_METADATA_KEYS:
                res[k] = v
            elif str(v).lower() == 'true': res[k] = True
            elif str(v).lower() == 'false': res[k] = False
            elif v == "":
                res[k] = None
            elif isinstance(v, str) and k not in _STRING_METADATA_KEYS and v.lstrip('-').replace('.', '', 1).isdigit():
                res[k] = float(v) if '.' in v else int(v)
            elif isinstance(v, str) and (v.startswith('[') or v.startswith('{')):
                try:
                    res[k] = json.loads(v)
                except Exception:
                    res[k] = v
            else:
                res[k] = v
        return res

    def _stringify_metadata(self, metadata: dict) -> dict:
        if not metadata: return {}
        res = {}
        for k, v in metadata.items():
            if v is None:
                res[k] = ""
            elif isinstance(v, bool): res[k] = "true" if v else "false"
            elif isinstance(v, (int, float)): res[k] = str(v)
            elif isinstance(v, (dict, list)): res[k] = json.dumps(v, ensure_ascii=False)
            else: res[k] = str(v)
        return res

    def _deserialize_record(self, rec):
        if rec is None:
            return None
        metadata = self._deserialize_metadata(getattr(rec, "metadata", None) or {})
        return _CompatRecord(rec, metadata)

    def _deserialize_records(self, records):
        final = []
        for item in records or []:
            if isinstance(item, tuple) and len(item) == 2:
                score, rec = item
                final.append((score, self._deserialize_record(rec)))
            else:
                final.append(self._deserialize_record(item))
        return final

    def _normalize_runtime_payload(self, value):
        if isinstance(value, dict):
            return {k: self._normalize_runtime_payload(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize_runtime_payload(v) for v in value]
        if hasattr(value, 'keys') and callable(getattr(value, 'keys')):
            try:
                return {k: self._normalize_runtime_payload(value[k]) for k in value.keys()}
            except Exception:
                pass
        if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
            try:
                return self._normalize_runtime_payload(value.to_dict())
            except Exception:
                pass
        attrs = [n for n in dir(value) if not n.startswith('_')]
        if attrs and not isinstance(value, (str, int, float, bool)):
            out = {}
            for attr in attrs:
                try:
                    attr_value = getattr(value, attr)
                except Exception:
                    continue
                if callable(attr_value):
                    continue
                out[attr] = self._normalize_runtime_payload(attr_value)
            if out:
                return out
        return value

    @property
    def records(self):
        wrapped = self._deserialize_records(self._aura.search(limit=5000))
        return {rec.id: rec for rec in wrapped if rec is not None}

    def store(self, content, level=Level.WORKING, tags=None, deduplicate=False, metadata=None, **kwargs):
        if tags is None: tags = []
        if metadata is None: metadata = {}
        if content and isinstance(content, str) and not content.startswith("[20"):
            from datetime import datetime as _dt
            content = f"[{_dt.now().strftime('%Y-%m-%d %A')}] {content}"

        source_type = kwargs.get("source_type")
        meta_str = self._stringify_metadata(metadata)
        if 'channel' in kwargs:
            meta_str['channel'] = str(kwargs['channel'])
        if source_type is None and "source_type" in meta_str:
            source_type = meta_str.pop("source_type")
        semantic_type = kwargs.get("semantic_type")
        if semantic_type is None and "semantic_type" in meta_str:
            semantic_type = meta_str.pop("semantic_type")

        store_kwargs = {
            "level": level,
            "tags": tags,
            "deduplicate": deduplicate,
            "metadata": meta_str,
        }
        if source_type is not None:
            if self._has_source_type:
                store_kwargs["source_type"] = source_type
            else:
                store_kwargs["metadata"]["source_type"] = str(source_type)
        if semantic_type is not None:
            if self._has_semantic_type:
                store_kwargs["semantic_type"] = str(semantic_type)
            else:
                store_kwargs["metadata"]["semantic_type"] = str(semantic_type)
        if "pin" in kwargs:
            store_kwargs["pin"] = kwargs["pin"]
        if "content_type" in kwargs:
            store_kwargs["content_type"] = kwargs["content_type"]
        if "caused_by_id" in kwargs:
            store_kwargs["caused_by_id"] = kwargs["caused_by_id"]
        if "channel" in kwargs:
            store_kwargs["channel"] = kwargs["channel"]
        if kwargs.get("auto_promote") is not None and self._has_auto_promote:
            store_kwargs["auto_promote"] = kwargs["auto_promote"]

        result = self._aura.store(content, **store_kwargs)
        if isinstance(result, str):
            return _StoreResult(result)
        rec = self._deserialize_record(result)
        return rec if rec is not None else _StoreResult("")

    def list_records(self, tags=None, min_strength=0.0, limit=5000):
        records = self.search(query="", tags=tags, limit=limit)
        if min_strength > 0:
            records = [rec for rec in records if getattr(rec, "strength", 0.0) >= min_strength]
        return records

    def get(self, record_id):
        return self._deserialize_record(self._aura.get(record_id))

    def search(self, query="", tags=None, limit=20, **kwargs):
        params = {"query": query or "", "tags": tags, "limit": limit}
        if kwargs.get("level") is not None:
            params["level"] = kwargs["level"]
        if kwargs.get("content_type") is not None:
            params["content_type"] = kwargs["content_type"]
        if kwargs.get("source_type") is not None and self._has_source_type:
            params["source_type"] = kwargs["source_type"]
        return self._deserialize_records(self._aura.search(**params))

    def update(self, record_id, content=None, metadata=None, **kwargs):
        meta_str = None if metadata is None else self._stringify_metadata(metadata)
        return self._aura.update(
            record_id,
            content=content,
            level=kwargs.get("level"),
            tags=kwargs.get("tags"),
            strength=kwargs.get("strength"),
            metadata=meta_str,
            source_type=kwargs.get("source_type"),
        )

    def recall_core(self, query=None, limit=5000):
        if query is not None:
            records = self.search(query=query, limit=limit, level=Level.DOMAIN) + self.search(query=query, limit=limit, level=Level.IDENTITY)
            seen = set()
            deduped = []
            for rec in records:
                if rec and rec.id not in seen:
                    seen.add(rec.id)
                    deduped.append(rec)
            return deduped[:limit]
        records = self._deserialize_records(self._aura.recall_core_tier())
        return records[:limit]

    def recall_cognitive(self, query=None, limit=5000):
        if query is not None:
            records = self.search(query=query, limit=limit, level=Level.WORKING) + self.search(query=query, limit=limit, level=Level.DECISIONS)
            seen = set()
            deduped = []
            for rec in records:
                if rec and rec.id not in seen:
                    seen.add(rec.id)
                    deduped.append(rec)
            return deduped[:limit]
        records = self._deserialize_records(self._aura.recall_cognitive())
        return records[:limit]

    def _normalize_recall_results(self, results):
        """Normalize recall results to list[dict].

        Aura SDK >=1.4.0 returns list[dict] from recall_structured/recall_full.
        Older versions returned list[(score, Record)].  We normalize both to
        list[dict] which is what downstream code (brain_tools, scratchpad,
        background_brain) already expects.
        """
        final = []
        for item in results or []:
            if isinstance(item, dict):
                # New format — already a dict with id/content/score/etc.
                if "metadata" in item and isinstance(item["metadata"], dict):
                    item["metadata"] = self._deserialize_metadata(item["metadata"])
                elif "metadata" not in item:
                    # Build metadata dict from top-level keys for downstream compat.
                    item["metadata"] = {}
                    for mk in ("trust", "source", "source_type", "verified",
                                "trust_score", "semantic_type"):
                        if mk in item:
                            item["metadata"][mk] = item[mk]
                final.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                score, rec = item
                rec = self._deserialize_record(rec)
                final.append({
                    "id": rec.id,
                    "content": rec.content,
                    "score": float(score),
                    "level": str(getattr(rec, "level", "")),
                    "strength": getattr(rec, "strength", 0.0),
                    "tags": list(getattr(rec, "tags", []) or []),
                    "metadata": rec.metadata,
                    "source_type": getattr(rec, "source_type", ""),
                })
            else:
                rec = self._deserialize_record(item)
                if rec is not None:
                    final.append({
                        "id": rec.id,
                        "content": rec.content,
                        "score": 0.0,
                        "level": str(getattr(rec, "level", "")),
                        "strength": getattr(rec, "strength", 0.0),
                        "tags": list(getattr(rec, "tags", []) or []),
                        "metadata": rec.metadata,
                    })
        return final

    def recall_structured(self, query, top_k=15, min_strength=None, session_id=None):
        rs_kwargs = {"top_k": top_k}
        if min_strength is not None:
            rs_kwargs["min_strength"] = float(min_strength)
        if session_id:
            rs_kwargs["session_id"] = session_id
        results = self._aura.recall_structured(query, **rs_kwargs)
        return self._normalize_recall_results(results)

    def recall_full(self, query, top_k=20, include_failures=True, **kwargs):
        if not self._has_recall_full:
            raise AttributeError("recall_full not available on this Aura core version")
        session_id = kwargs.get('session_id', "")
        results = self._aura.recall_full(
            query, top_k=top_k, include_failures=include_failures,
            min_strength=0.1, expand_connections=True, session_id=session_id,
        )
        return self._normalize_recall_results(results)

    def tier_stats(self):
        if self._has_tier_stats:
            raw = self._aura.tier_stats()
            # Normalize flat keys (cognitive_working, core_domain, etc.)
            # into nested format expected by downstream code.
            if isinstance(raw, dict) and "cognitive" not in raw:
                working = int(raw.get("cognitive_working", 0) or 0)
                decisions = int(raw.get("cognitive_decisions", 0) or 0)
                domain = int(raw.get("core_domain", 0) or 0)
                identity = int(raw.get("core_identity", 0) or 0)
                cog_total = int(raw.get("cognitive_total", working + decisions) or 0)
                core_total = int(raw.get("core_total", domain + identity) or 0)
                total = int(raw.get("total", cog_total + core_total) or 0)
                return {
                    "cognitive": {"total": cog_total, "working": working, "decisions": decisions},
                    "core": {"total": core_total, "domain": domain, "identity": identity},
                    "total": total,
                }
            return raw
        stats = {
            "cognitive": {"total": 0, "working": 0, "decisions": 0},
            "core": {"total": 0, "domain": 0, "identity": 0},
            "total": 0,
        }
        for rec in self.records.values():
            if rec.level == Level.WORKING:
                stats["cognitive"]["working"] += 1
                stats["cognitive"]["total"] += 1
            elif rec.level == Level.DECISIONS:
                stats["cognitive"]["decisions"] += 1
                stats["cognitive"]["total"] += 1
            elif rec.level == Level.DOMAIN:
                stats["core"]["domain"] += 1
                stats["core"]["total"] += 1
            elif rec.level == Level.IDENTITY:
                stats["core"]["identity"] += 1
                stats["core"]["total"] += 1
        stats["total"] = stats["cognitive"]["total"] + stats["core"]["total"]
        return stats

    def promotion_candidates(self, min_activations=5, min_strength=0.7):
        if self._has_promotion_candidates:
            return self._deserialize_records(
                self._aura.promotion_candidates(
                    min_activations=min_activations,
                    min_strength=min_strength,
                )
            )
        candidates = []
        for rec in self.recall_cognitive(limit=5000):
            if rec.activation_count >= min_activations and rec.strength >= min_strength:
                candidates.append(rec)
        return candidates

    def end_session(self, session_id):
        return self._aura.end_session(session_id)

    def run_maintenance(self):
        return self._aura.run_maintenance()

    def cognitive_insights(self):
        """Return native Aura cognitive insights (cluster, causal, trending, etc.)."""
        return self._aura.insights()

    def cognitive_maintenance_report(self):
        """Run maintenance and return the full MaintenanceReport with cognitive phases.

        The report includes:
          .belief   — BeliefPhaseReport (beliefs_created, resolved, churn_rate, ...)
          .concept  — ConceptPhaseReport (candidates_found, stable_count, ...)
          .policy   — PolicyPhaseReport (hints_found, stable_hints, ...)
          .causal   — CausalPhaseReport (candidates_found, stable_count, ...)
          .epistemic — EpistemicPhaseReport (support_links, conflict_links, ...)
          .stability — LayerStability (belief_churn, concept_churn, ...)
          .timings  — PhaseTimings (belief_ms, concept_ms, total_ms, ...)
        """
        return self._aura.run_maintenance()

    def explain_record(self, record_id):
        if not hasattr(self._aura, 'explain_record'):
            return None
        return self._normalize_runtime_payload(self._aura.explain_record(record_id))

    def explain_recall(self, query, top_k=10):
        if not hasattr(self._aura, 'explain_recall'):
            return {"query": query, "items": []}
        return self._normalize_runtime_payload(self._aura.explain_recall(query, top_k))

    def explainability_bundle(self, record_id):
        if not hasattr(self._aura, 'explainability_bundle'):
            return None
        return self._normalize_runtime_payload(self._aura.explainability_bundle(record_id))

    def memory_health_digest(self, limit=10):
        if not hasattr(self._aura, 'get_memory_health_digest'):
            return {}
        return self._normalize_runtime_payload(self._aura.get_memory_health_digest(limit))

    def belief_instability_summary(self):
        if not hasattr(self._aura, 'get_belief_instability_summary'):
            return {}
        return self._normalize_runtime_payload(self._aura.get_belief_instability_summary())

    def salience_summary(self):
        if not hasattr(self._aura, 'get_salience_summary'):
            return {}
        return self._normalize_runtime_payload(self._aura.get_salience_summary())

    def high_salience_records(self, limit=20):
        if not hasattr(self._aura, 'get_high_salience_records'):
            return []
        return self._normalize_runtime_payload(self._aura.get_high_salience_records(limit))

    def mark_record_salience(self, record_id, salience, reason=None):
        if not hasattr(self._aura, 'mark_record_salience'):
            return False
        self._aura.mark_record_salience(record_id, salience, reason)
        return True

    def latest_reflection_digest(self):
        if not hasattr(self._aura, 'get_latest_reflection_digest'):
            return None
        return self._normalize_runtime_payload(self._aura.get_latest_reflection_digest())

    def reflection_digest(self, limit=10):
        if not hasattr(self._aura, 'get_reflection_digest'):
            return {}
        return self._normalize_runtime_payload(self._aura.get_reflection_digest(limit))

    def reflection_summaries(self, limit=20):
        if not hasattr(self._aura, 'get_reflection_summaries'):
            return []
        return self._normalize_runtime_payload(self._aura.get_reflection_summaries(limit))

    def contradiction_clusters(self, limit=20, namespace=None):
        if not hasattr(self._aura, 'get_contradiction_clusters'):
            return []
        return self._normalize_runtime_payload(
            self._aura.get_contradiction_clusters(namespace=namespace, limit=limit)
        )

    def contradiction_review_queue(self, limit=20, namespace=None):
        if not hasattr(self._aura, 'get_contradiction_review_queue'):
            return []
        return self._normalize_runtime_payload(
            self._aura.get_contradiction_review_queue(namespace=namespace, limit=limit)
        )

    def policy_lifecycle_summary(self):
        if not hasattr(self._aura, 'get_policy_lifecycle_summary'):
            return {}
        return self._normalize_runtime_payload(self._aura.get_policy_lifecycle_summary())

    def policy_pressure_report(self, limit=20, namespace=None):
        if not hasattr(self._aura, 'get_policy_pressure_report'):
            return []
        return self._normalize_runtime_payload(
            self._aura.get_policy_pressure_report(namespace=namespace, limit=limit)
        )

    def namespace_governance_status(self, namespaces=None):
        if not hasattr(self._aura, 'get_namespace_governance_status'):
            return []
        if namespaces and hasattr(self._aura, 'get_namespace_governance_status_filtered'):
            return self._normalize_runtime_payload(self._aura.get_namespace_governance_status_filtered(namespaces))
        return self._normalize_runtime_payload(self._aura.get_namespace_governance_status())

    def rejected_policy_hints(self, limit=20):
        if not hasattr(self._aura, 'get_rejected_policy_hints'):
            return []
        return self._normalize_runtime_payload(self._aura.get_rejected_policy_hints(limit))

    def suppressed_policy_hints(self, limit=20):
        if not hasattr(self._aura, 'get_suppressed_policy_hints'):
            return []
        return self._normalize_runtime_payload(self._aura.get_suppressed_policy_hints(limit))

    def get_epistemic_state(self, record_id):
        """Get epistemic state for a specific record.

        Returns dict with confidence, support_mass, conflict_mass, volatility,
        epistemic_health if the record exists, None otherwise.
        """
        rec = self._aura.get(record_id)
        if rec is None:
            return None
        return {
            "confidence": getattr(rec, "confidence", None),
            "support_mass": getattr(rec, "support_mass", None),
            "conflict_mass": getattr(rec, "conflict_mass", None),
            "volatility": getattr(rec, "volatility", None),
            "epistemic_health": getattr(rec, "epistemic_health", None),
        }

    def get_surfaced_concepts(self, limit=20, namespace=None):
        """Return surfaced concept groups from the cognitive layer.

        Each item is a SurfacedConcept with attributes like label, record_ids,
        abstraction_score, etc.  Returns empty list if the layer has not yet
        accumulated enough data.
        """
        if namespace is not None and hasattr(self._aura, "get_surfaced_concepts_for_namespace"):
            return self._aura.get_surfaced_concepts_for_namespace(namespace, limit)
        if hasattr(self._aura, "get_surfaced_concepts"):
            return self._aura.get_surfaced_concepts(limit)
        return []

    def get_surfaced_policy_hints(self, limit=20, namespace=None):
        """Return surfaced policy hints from the cognitive layer.

        Each item is a SurfacedPolicyHint with attributes like hint, strength,
        provenance, etc.  Returns empty list if the layer has not yet
        accumulated enough data.
        """
        if namespace is not None and hasattr(self._aura, "get_surfaced_policy_hints_for_namespace"):
            return self._aura.get_surfaced_policy_hints_for_namespace(namespace, limit)
        if hasattr(self._aura, "get_surfaced_policy_hints"):
            return self._aura.get_surfaced_policy_hints(limit)
        return []

    def feedback(self, record_id: str, useful: bool) -> tuple:
        """Signal to AuraSDK whether a record was useful (True) or harmful (False).

        Returns (positive_count, negative_count, net_score).
        AuraSDK uses this to adjust belief confidence and suggest corrections.
        """
        if hasattr(self._aura, "feedback"):
            return self._aura.feedback(record_id, useful)
        return (0, 0, 0)

    def feedback_stats(self, record_id: str) -> tuple:
        """Return (positive, negative, net_score) feedback stats for a record."""
        if hasattr(self._aura, "feedback_stats"):
            return self._aura.feedback_stats(record_id)
        return (0, 0, 0)

    def get_suggested_corrections(self, limit: int = 10) -> list:
        """Return list of corrections AuraSDK suggests based on feedback and belief analysis.

        Each entry: {target_kind, target_id, suggested_action, reason_detail, priority_score, severity}
        """
        if hasattr(self._aura, "get_suggested_corrections"):
            return self._normalize_runtime_payload(self._aura.get_suggested_corrections(limit))
        return []

    def get_suggested_corrections_report(self, limit: int = 10) -> dict:
        """Return full corrections report with scan_latency_ms and entries list."""
        if hasattr(self._aura, "get_suggested_corrections_report"):
            return self._normalize_runtime_payload(self._aura.get_suggested_corrections_report(limit))
        return {"entries": []}

    def get_correction_review_queue(self, limit: int = 20) -> list:
        """Return records queued for human/agent correction review."""
        if hasattr(self._aura, "get_correction_review_queue"):
            return self._normalize_runtime_payload(self._aura.get_correction_review_queue(limit))
        return []

    def get_correction_log(self) -> list:
        """Return log of all corrections applied so far."""
        if hasattr(self._aura, "get_correction_log"):
            return self._normalize_runtime_payload(self._aura.get_correction_log())
        return []

    def get_recently_corrected_beliefs(self, limit: int = 10) -> list:
        """Return beliefs that were recently corrected — useful for auditing."""
        if hasattr(self._aura, "get_recently_corrected_beliefs"):
            return self._normalize_runtime_payload(self._aura.get_recently_corrected_beliefs(limit))
        return []

    def deprecate_belief(self, record_id: str) -> bool:
        """Penalize/downvote a belief — reduces its confidence score in AuraSDK.

        Use when a belief is confirmed wrong but should be kept for audit trail
        rather than deleted. Returns True if successful.
        """
        if hasattr(self._aura, "deprecate_belief"):
            return self._aura.deprecate_belief(record_id)
        return False

    def deprecate_belief_with_reason(self, record_id: str, reason: str) -> bool:
        """Penalize a belief with an explicit reason stored alongside it.

        Stronger than deprecate_belief — attaches human-readable explanation
        so AuraSDK can use the reason for future correction suggestions.
        """
        if hasattr(self._aura, "deprecate_belief_with_reason"):
            return self._aura.deprecate_belief_with_reason(record_id, reason)
        return False

    def invalidate_causal_pattern(self, pattern_id: str) -> bool:
        """Reject a learned causal pattern — marks it as invalid so AuraSDK stops using it."""
        if hasattr(self._aura, "invalidate_causal_pattern"):
            return self._aura.invalidate_causal_pattern(pattern_id)
        return False

    def retract_causal_pattern(self, pattern_id: str) -> bool:
        """Undo/retract a causal pattern — removes it from active reasoning entirely."""
        if hasattr(self._aura, "retract_causal_pattern"):
            return self._aura.retract_causal_pattern(pattern_id)
        return False

    def retract_policy_hint(self, hint_id: str) -> bool:
        """Reject a surfaced policy hint — tells AuraSDK this hint was wrong or irrelevant."""
        if hasattr(self._aura, "retract_policy_hint"):
            return self._aura.retract_policy_hint(hint_id)
        return False

    def get_high_volatility_beliefs(self, limit: int = 10) -> list:
        """Return beliefs with high volatility — frequently changing or contradicted."""
        if hasattr(self._aura, "get_high_volatility_beliefs"):
            return self._aura.get_high_volatility_beliefs(limit)
        return []

    def get_low_stability_beliefs(self, limit: int = 10) -> list:
        """Return beliefs with low stability score — weak or poorly supported."""
        if hasattr(self._aura, "get_low_stability_beliefs"):
            return self._aura.get_low_stability_beliefs(limit)
        return []

    def provenance_chain(self, record_id: str) -> dict:
        """Return provenance chain for a record — why it was surfaced and how it was derived."""
        if hasattr(self._aura, "provenance_chain"):
            return self._aura.provenance_chain(record_id)
        return {"record_id": record_id, "narrative": "provenance_chain not available", "steps": []}

    def record_tool_success(self, tool_name: str) -> None:
        """Inform AuraSDK that a tool call succeeded — used for tool health scoring."""
        if hasattr(self._aura, "record_tool_success"):
            self._aura.record_tool_success(tool_name)

    def record_tool_failure(self, tool_name: str) -> None:
        """Inform AuraSDK that a tool call failed — used for tool health scoring."""
        if hasattr(self._aura, "record_tool_failure"):
            self._aura.record_tool_failure(tool_name)

    def tool_health(self) -> dict:
        """Return tool health report from AuraSDK — which tools are degraded."""
        if hasattr(self._aura, "tool_health"):
            return self._aura.tool_health()
        return {}

    def get_person_digest(self, record_id: str):
        """Return person digest for a person record — full context about this person."""
        if hasattr(self._aura, "get_person_digest"):
            return self._aura.get_person_digest(record_id)
        return None

    def get_project_digest(self, project_id: str):
        """Return project digest — status, timeline, tasks for a project."""
        if hasattr(self._aura, "get_project_digest"):
            try:
                return self._aura.get_project_digest(project_id)
            except Exception:
                return None
        return None

    def get_entity_digest(self, entity_id: str):
        """Return entity digest — all relations and context for an entity."""
        if hasattr(self._aura, "get_entity_digest"):
            try:
                return self._aura.get_entity_digest(entity_id)
            except Exception:
                return None
        return None

    def insights(self):
        """Cheap runtime insights for compatibility with older Remy logic."""
        insights = []
        try:
            records = self.list_records(min_strength=0.0, limit=500)
        except Exception:
            return []

        # 1. Weak non-identity/domain memories that are likely to decay away.
        decay_records = []
        for rec in records:
            level_name = str(getattr(rec, "level", "")).upper()
            if "WORKING" in level_name or "DECISIONS" in level_name:
                if getattr(rec, "strength", 0.0) <= 0.35:
                    decay_records.append(
                        {
                            "id": rec.id,
                            "content": rec.content,
                            "strength": round(getattr(rec, "strength", 0.0), 3),
                        }
                    )
        if decay_records:
            decay_records.sort(key=lambda r: r["strength"])
            insights.append({"type": "decay_risk", "details": {"records": decay_records[:5]}})

        # 2. Repeated meaningful tags -> hot topics.
        ignore_tags = {
            "autonomous-goal", "autonomous-outcome", "session-summary", "todo-item",
            "research-project", "research-finding", "web-search-cache", "completed",
            "active", "pending", "scheduled-task",
        }
        tag_counts = Counter()
        for rec in records:
            for tag in getattr(rec, "tags", []) or []:
                if tag and tag not in ignore_tags:
                    tag_counts[tag] += 1
        hot_topics = [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(5) if count >= 2]
        if hot_topics:
            insights.append({"type": "hot_topic", "details": {"topics": hot_topics}})

        # 3. Simple contradiction heuristic on records sharing tags with negation.
        conflict_pairs = []
        lowered = [(rec, (rec.content or "").lower()) for rec in records if getattr(rec, "content", None)]
        for idx, (a, a_text) in enumerate(lowered[:100]):
            a_tags = set(getattr(a, "tags", []) or [])
            if not a_tags:
                continue
            for b, b_text in lowered[idx + 1 : idx + 40]:
                if not a_tags.intersection(getattr(b, "tags", []) or []):
                    continue
                if (
                    (" not " in a_text or " never " in a_text or " no " in a_text)
                    != (" not " in b_text or " never " in b_text or " no " in b_text)
                ):
                    conflict_pairs.append(
                        {
                            "id_a": a.id,
                            "id_b": b.id,
                            "content_a": a.content,
                            "content_b": b.content,
                        }
                    )
                    if len(conflict_pairs) >= 3:
                        break
            if len(conflict_pairs) >= 3:
                break
        if conflict_pairs:
            insights.append({"type": "conflict", "details": {"pairs": conflict_pairs}})

        return insights

# Global lock for CognitiveMemory access
# Protects against concurrent access to non-thread-safe Rust backend (Aura Memory)
# from Web (FastAPI), Telegram, and Background threads.
brain_lock = threading.RLock()


class _BrainLockTimeout:
    """Context manager: acquire brain_lock with a timeout for read-only web requests.

    If the lock cannot be acquired within `timeout` seconds (e.g. autonomy loop
    is holding it during an LLM call), raises RuntimeError so the caller can
    return a graceful empty/stale response instead of blocking indefinitely.

    Usage:
        try:
            with brain_lock_read():
                data = brain.list_records(...)
        except RuntimeError:
            data = []   # return stale/empty while busy
    """
    def __init__(self, timeout: float = 2.0):
        self._timeout = timeout

    def __enter__(self):
        acquired = brain_lock.acquire(timeout=self._timeout)
        if not acquired:
            raise RuntimeError("brain_lock timeout — autonomy loop is busy")
        return self

    def __exit__(self, *_):
        brain_lock.release()


def brain_lock_read(timeout: float = 2.0) -> _BrainLockTimeout:
    """Acquire brain_lock for a read-only web request with a 2s timeout."""
    return _BrainLockTimeout(timeout)


async def brain_run(fn, *args, timeout: float = 5.0, **kwargs):
    """Run a brain operation in a thread pool so it never blocks the async event loop.

    All brain operations use a sync RLock internally. Calling them directly in an
    async FastAPI handler blocks the entire event loop — no other request can be
    served while the lock is held.  This helper offloads the call to a thread,
    releasing the event loop to process other requests concurrently.

    Usage:
        records = await brain_run(brain.list_records, tags=tag_list)
        stats   = await brain_run(brain.stats)

    Raises asyncio.TimeoutError if the call takes longer than `timeout` seconds.
    """
    import asyncio

    def _call():
        with brain_lock:
            return fn(*args, **kwargs)

    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _call),
        timeout=timeout,
    )


_AURA_VERSION_STAMP = ".aura_version"


def _get_aura_wheel_version() -> str:
    """Return the installed aura module version, or 'unknown' if not accessible."""
    try:
        import aura as _aura_mod
        return str(getattr(_aura_mod, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _brain_version_stamp_path(brain_path: Path) -> Path:
    return brain_path / _AURA_VERSION_STAMP


def _check_and_backup_on_version_change(brain_path: Path) -> str | None:
    """If aura wheel version changed since last run, backup brain before opening.

    Returns the backup path string if a backup was made, None otherwise.
    Writes a new version stamp after backup (or on first run).
    Failures are logged but never raise — brain startup must not be blocked here.
    """
    stamp_path = _brain_version_stamp_path(brain_path)
    current_ver = _get_aura_wheel_version()

    # Read previous version from stamp file
    previous_ver: str | None = None
    if stamp_path.exists():
        try:
            previous_ver = stamp_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Brain version stamp unreadable: %s", exc)

    # First run — no stamp yet, just write it
    if previous_ver is None:
        try:
            stamp_path.write_text(current_ver, encoding="utf-8")
            logger.info("Brain version stamp created: aura %s", current_ver)
        except Exception as exc:
            logger.warning("Could not write brain version stamp: %s", exc)
        return None

    # Version unchanged — nothing to do
    if previous_ver == current_ver:
        return None

    # Version changed — make a full directory backup before opening
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = brain_path.parent / f"{brain_path.name}_backup_{timestamp}_aura{previous_ver}"
    logger.warning(
        "aura-memory version changed: %s → %s. Backing up brain to %s before opening.",
        previous_ver, current_ver, backup_dir,
    )
    try:
        shutil.copytree(str(brain_path), str(backup_dir))
        logger.info("Brain backup complete: %s (%d files)", backup_dir,
                    sum(1 for _ in backup_dir.rglob("*") if _.is_file()))
    except Exception as exc:
        logger.error("Brain backup FAILED on version change (%s → %s): %s", previous_ver, current_ver, exc)
        return None

    # Update stamp to current version
    try:
        stamp_path.write_text(current_ver, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not update brain version stamp after backup: %s", exc)

    return str(backup_dir)


def _probe_aura_store(path: Path) -> tuple[bool, str]:
    """Open an Aura store in a subprocess so Rust aborts cannot kill the main process."""
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from aura import Aura; "
                    f"Aura(r'{str(path)}'); "
                    "print('ok')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if probe.returncode == 0:
            return True, ""
        stderr = (probe.stderr or "").strip()
        stdout = (probe.stdout or "").strip()
        return False, stderr or stdout or f"exit={probe.returncode}"
    except Exception as exc:
        return False, str(exc)


def _probe_aura_store_with_retries(
    path: Path,
    *,
    attempts: int = 5,
    delay_sec: float = 0.75,
) -> tuple[bool, str]:
    """Retry startup probe to avoid quarantining healthy stores on fast restarts."""
    last_reason = ""
    for attempt in range(1, attempts + 1):
        ok, reason = _probe_aura_store(path)
        if ok:
            return True, ""
        last_reason = reason
        if attempt < attempts:
            time.sleep(delay_sec)
    return False, last_reason


def _try_export_brain_backup(path: Path, timestamp: str) -> Path | None:
    """Best-effort JSON export of brain records before quarantine.

    Tries to open the store and dump all records to a human-readable JSON file
    so data can be recovered even if the binary format becomes unreadable.
    """
    backup_file = path.parent / f"{path.name}_backup_{timestamp}.json"
    try:
        tmp_brain = Aura(str(path))
        data = tmp_brain.export_json()
        tmp_brain.close()
        backup_file.write_text(data, encoding="utf-8")
        logger.info("Brain JSON backup saved to %s before quarantine", backup_file)
        return backup_file
    except Exception as exc:
        logger.warning("Could not export brain JSON backup before quarantine: %s", exc)
        # Fallback: extract raw content strings from brain.cog binary
        try:
            import re as _re
            cog_path = path / "brain.cog"
            if cog_path.exists():
                raw = cog_path.read_bytes()
                pattern = rb'"content":"((?:[^"\\]|\\.)*)"'
                matches = _re.findall(pattern, raw)
                texts = []
                seen: set = set()
                for m in matches:
                    try:
                        text = m.decode("utf-8", errors="replace")
                        text = text.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t")
                        if len(text) >= 30 and text[:80] not in seen:
                            seen.add(text[:80])
                            texts.append(text)
                    except Exception:
                        pass
                backup_file.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("Brain raw-content backup saved (%d records) to %s", len(texts), backup_file)
                return backup_file
        except Exception as exc2:
            logger.warning("Raw brain backup also failed: %s", exc2)
    return None


def _quarantine_incompatible_store(path: Path, reason: str = "") -> Path:
    """Move an incompatible Aura store aside without deleting user data.

    Returns the path that should be used for fresh runtime initialization.
    """
    global _brain_quarantine_path, _brain_backup_path
    try:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Save a JSON backup before moving — last resort data preservation.
        backup_file = _try_export_brain_backup(path, timestamp)
        _brain_backup_path = str(backup_file) if backup_file else ""
        target = path.parent / f"{path.name}_incompatible_{timestamp}"
        counter = 1
        while target.exists():
            counter += 1
            target = path.parent / f"{path.name}_incompatible_{timestamp}_{counter}"
        shutil.move(str(path), str(target))
        _brain_quarantine_path = str(target)
        path.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Aura store at %s was quarantined to %s due to incompatible startup probe%s",
            path,
            target,
            f": {reason}" if reason else "",
        )
        return path
    except Exception as exc:
        logger.exception("Failed to quarantine incompatible Aura store at %s", path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fallback = path.parent / f"{path.name}_fresh_{timestamp}"
        fallback.mkdir(parents=True, exist_ok=True)
        settings.AURA_BRAIN_PATH = fallback
        logger.warning(
            "Falling back to fresh Aura store at %s because incompatible store could not be moved%s",
            fallback,
            f": {exc}" if exc else "",
        )
        return fallback


def _allow_automatic_brain_quarantine() -> bool:
    """Explicit opt-in for risky startup behavior that can mask data-loss incidents."""
    value = os.environ.get("REMY_ALLOW_AURA_AUTO_QUARANTINE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_brain_startup_status() -> dict:
    return {
        "initialized": _brain_initialized,
        "quarantined_at_startup": _brain_quarantined_at_startup,
        "quarantine_reason": _brain_quarantine_reason,
        "startup_blocked": _brain_startup_blocked,
        "startup_incident": _brain_startup_incident,
        "quarantine_path": _brain_quarantine_path,
        "backup_path": _brain_backup_path,
        "recovery": dict(_brain_recovery_stats),
        "startup_artifact_id": _brain_startup_artifact_id,
        "auto_quarantine_enabled": _allow_automatic_brain_quarantine(),
        "shutdown_started": _brain_shutdown_started,
        "closed": _brain_closed,
        "close_error": _brain_close_error,
    }


def brain_runtime_allows_access() -> bool:
    """Whether the shared brain is still safe for live reads/writes."""
    return not _brain_shutdown_started and not _brain_closed


def brain_is_initialized() -> bool:
    """Whether the shared Aura brain has been instantiated in this process."""
    return _brain_initialized and _brain_instance is not None


def _is_expected_brain_close_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    return (
        "os error 2" in text
        or "cannot find the file specified" in text
        or "already closed" in text
    )


def _persist_startup_recovery_artifact(recovery_stats: dict | None = None) -> str:
    """Persist a reviewable startup incident artifact after quarantine/recovery."""
    global _brain_startup_artifact_id, _brain_recovery_stats
    if not _brain_quarantined_at_startup:
        return ""
    if _brain_startup_artifact_id:
        return _brain_startup_artifact_id

    recovery_stats = dict(recovery_stats or {})
    _brain_recovery_stats = recovery_stats
    recovery_status = str(recovery_stats.get("status") or "").strip()
    existing_before = int(recovery_stats.get("existing_records_before", 0) or 0)
    replayed = int(recovery_stats.get("tool_calls_replayed", 0) or 0)
    skipped = int(recovery_stats.get("tool_calls_skipped", 0) or 0)
    files = int(recovery_stats.get("files", 0) or 0)
    entries = int(recovery_stats.get("entries", 0) or 0)
    tool_errors = list(recovery_stats.get("tool_errors") or [])

    content = "\n".join(
        [
            "Startup Incident Snapshot",
            "",
            "Aura store integrity recovery was applied during startup.",
            "",
            f"Quarantine reason: {_brain_quarantine_reason}",
            f"Quarantined store: {_brain_quarantine_path}",
            f"Backup path: {_brain_backup_path}",
            f"Recovery status: {recovery_status}",
            f"Existing records before replay: {existing_before}",
            f"History files scanned: {files}",
            f"History entries scanned: {entries}",
            f"Tool calls replayed: {replayed}",
            f"Tool calls skipped: {skipped}",
            f"Tool errors: {len(tool_errors)}",
        ]
    ).strip()

    with brain_lock:
        rec = brain.store(
            content=content,
            level=Level.DECISIONS,
            tags=["operator", "incident_snapshot", "review", "startup_incident"],
            metadata={
                "type": "startup_incident",
                "source": "startup",
                "failure_code": "memory_recovery_applied",
                "quarantine_reason": _brain_quarantine_reason,
                "quarantine_path": _brain_quarantine_path,
                "backup_path": _brain_backup_path,
                "recovery": recovery_stats,
            },
        )

    _brain_startup_artifact_id = str(getattr(rec, "id", "") or "")

    try:
        from remy.core.notification_router import notify

        notify(
            "Startup recovery was applied after an Aura store integrity incident.",
            level="warning",
            event_type="operator_alert",
            event_data={
                "source": "startup",
                "action_target": "open_memory_verification",
                "artifact_ids": [_brain_startup_artifact_id] if _brain_startup_artifact_id else [],
                "failure_code": "memory_recovery_applied",
                "verification_status": "recovered_after_quarantine",
                "verification_reason": _brain_quarantine_reason,
                "dedupe_key": "startup|brain_quarantine",
            },
            parse_mode="",
        )
    except Exception:
        logger.exception("Failed to emit startup recovery operator alert")

    return _brain_startup_artifact_id


def _maybe_restore_cognitive_snapshot(brain_path: Path, base_id: str, version: str) -> bool:
    """
    Restore a sealed .cog snapshot into brain_path before Aura.open().
    Returns True if snapshot was restored, False if none exists.
    Called before brain is initialized — uses filesystem ops only.
    """
    snap_dir = brain_path / "snapshots" / f"{base_id}-{version}"
    manifest_file = snap_dir / "manifest.json"
    if not manifest_file.exists():
        return False
    try:
        import json as _json
        manifest = _json.loads(manifest_file.read_text(encoding="utf-8"))
        for fname in manifest.get("files", []):
            src = snap_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(brain_path / fname))
        logger.info(
            "Cognitive snapshot restored for %s@%s (%d files) — skipping JSON load",
            base_id, version, len(manifest.get("files", []))
        )
        return True
    except Exception as e:
        logger.warning("Failed to restore cognitive snapshot for %s@%s: %s", base_id, version, e)
        return False


def _load_specialist_base(brain_compat: "_AuraCompat", base_pack_path: str, base_id: str, version: str) -> None:
    """
    Load a BasePack JSON into the brain, run maintenance, then seal a .cog snapshot.
    Called on first run when no snapshot exists yet.
    """
    try:
        import json as _json
        from aura import BasePack
        pack_data = _json.loads(Path(base_pack_path).read_text(encoding="utf-8"))
        pack = BasePack(**pack_data)
        report = brain_compat._aura.load_base_pack(pack)
        logger.info(
            "Specialist base loaded: %s@%s — %d records, %d concepts, %d cautions",
            report.base_id, report.version,
            report.records_loaded, report.concepts_seeded, report.cautions_loaded,
        )
        # Run one maintenance cycle so beliefs/concepts/causal patterns form
        brain_compat._aura.run_maintenance()
        # Seal native .cog snapshot — future startups skip JSON entirely
        seal = brain_compat._aura.seal_cognitive_snapshot(base_id, version)
        if seal.already_existed:
            logger.info("Cognitive snapshot already existed for %s@%s", base_id, version)
        else:
            logger.info(
                "Cognitive snapshot sealed for %s@%s — %d files, %d bytes at %s",
                base_id, version, seal.files_sealed, seal.total_bytes, seal.snapshot_path,
            )
    except ImportError:
        logger.warning("BasePack not available in this Aura version — specialist base not loaded")
    except Exception as e:
        logger.warning("Specialist base load failed (%s) — continuing without base", e)


def _init_brain() -> "_AuraCompat":
    """Initialize the brain safely across Aura storage-format changes."""
    global _brain_quarantined_at_startup, _brain_quarantine_reason
    global _brain_startup_blocked, _brain_startup_incident
    brain_path = Path(settings.AURA_BRAIN_PATH)
    brain_path.mkdir(parents=True, exist_ok=True)

    # ── Cognitive snapshot restore (before Aura.open) ──────────────────────────
    # If a specialist base is configured and a sealed snapshot exists,
    # restore .cog files directly — no JSON parsing needed.
    base_pack_path = settings.REMI_BASE_PACK
    base_id = settings.REMI_BASE_PACK_ID
    base_version = settings.REMI_BASE_PACK_VERSION
    _snapshot_restored = False
    if base_pack_path and base_id and base_version:
        _snapshot_restored = _maybe_restore_cognitive_snapshot(brain_path, base_id, base_version)

    has_existing_files = any(brain_path.iterdir())
    if has_existing_files:
        # Backup brain if aura-memory wheel version changed since last run
        _check_and_backup_on_version_change(brain_path)
        ok, reason = _probe_aura_store_with_retries(brain_path)
        if not ok:
            _brain_quarantine_reason = reason
            if _allow_automatic_brain_quarantine():
                _brain_quarantined_at_startup = True
                brain_path = _quarantine_incompatible_store(brain_path, reason)
            else:
                _brain_startup_blocked = True
                _brain_startup_incident = (
                    f"Aura startup probe failed for existing store at {brain_path}. "
                    "Automatic quarantine/fresh-store fallback is disabled to avoid silent empty restarts. "
                    f"Reason: {reason or 'unknown startup probe failure'}. "
                    "Set REMY_ALLOW_AURA_AUTO_QUARANTINE=1 only for explicit manual recovery."
                )
                logger.error(_brain_startup_incident)
                raise RuntimeError(_brain_startup_incident)

    brain_compat = _AuraCompat(str(brain_path))

    # ── First-time specialist base load ───────────────────────────────────────
    # If snapshot was not restored (first run) and base pack is configured,
    # load the JSON now and seal a snapshot for next time.
    if base_pack_path and base_id and base_version and not _snapshot_restored:
        if Path(base_pack_path).exists():
            _load_specialist_base(brain_compat, base_pack_path, base_id, base_version)
        else:
            logger.warning("REMI_BASE_PACK path not found: %s — skipping specialist base load", base_pack_path)

    return brain_compat


def _brain_record_count(limit: int = 20) -> int:
    try:
        return len(brain.search(query="", limit=limit))
    except Exception:
        return 0


def _maybe_recover_brain_from_history() -> None:
    global _brain_recovery_stats
    if _brain_startup_blocked:
        return
    if not _brain_quarantined_at_startup:
        return
    current_records = _brain_record_count()
    if current_records:
        _brain_recovery_stats = {
            "status": "already_present",
            "existing_records_before": current_records,
        }
        logger.info(
            "Skipping history replay because active brain already has %d records after startup quarantine.",
            current_records,
        )
        _persist_startup_recovery_artifact(_brain_recovery_stats)
        return
    history_dir = settings.DATA_DIR / "history"
    if not history_dir.exists():
        _brain_recovery_stats = {
            "status": "history_missing",
            "existing_records_before": current_records,
        }
        logger.warning(
            "Startup quarantine occurred but no history directory exists for recovery%s",
            f": {_brain_quarantine_reason}" if _brain_quarantine_reason else "",
        )
        _persist_startup_recovery_artifact(_brain_recovery_stats)
        return

    def _execute(tool: str, tool_args: dict):
        from remy.core.tool_dispatch import execute_tool

        return execute_tool(tool, tool_args, session_id="startup-recovery", channel="system")

    try:
        stats = replay_history(
            _execute,
            history_dir=history_dir,
            dry_run=False,
            force=True,
            count_records_fn=_brain_record_count,
        )
        _brain_recovery_stats = {
            **dict(stats or {}),
            "status": "history_replayed",
            "existing_records_before": current_records,
        }
        logger.warning(
            "Recovered active brain from history after startup quarantine%s: %s",
            f" ({_brain_quarantine_reason})" if _brain_quarantine_reason else "",
            stats,
        )
        _persist_startup_recovery_artifact(_brain_recovery_stats)
    except Exception:
        _brain_recovery_stats = {
            "status": "replay_failed",
            "existing_records_before": current_records,
        }
        logger.exception("Automatic history replay failed after startup quarantine")
        _persist_startup_recovery_artifact(_brain_recovery_stats)


# Episodic memory — remembers what happened
def get_brain():
    """Return the shared Aura brain, initializing it lazily on first real use."""
    global _brain_instance, _brain_initialized
    if _brain_instance is not None:
        return _brain_instance
    if _brain_closed:
        raise RuntimeError("Brain has already been closed for this process.")
    with _brain_init_lock:
        if _brain_instance is None:
            instance = _init_brain()
            _brain_instance = instance
            _brain_initialized = True
            _maybe_recover_brain_from_history()
    return _brain_instance


class _LazyBrainProxy:
    """Attribute-forwarding proxy that delays Aura initialization until first use."""

    def __getattr__(self, name: str):
        return getattr(get_brain(), name)

    def __repr__(self) -> str:
        if brain_is_initialized():
            return repr(_brain_instance)
        return "<LazyBrainProxy uninitialized>"


brain = _LazyBrainProxy()


def _safe_shutdown_log(level: str, message: str, *args) -> None:
    """Log during interpreter shutdown only while handlers are still usable."""
    handlers = list(logger.handlers) + list(logging.getLogger().handlers)
    for handler in handlers:
        stream = getattr(handler, "stream", None)
        if getattr(stream, "closed", False):
            return
    getattr(logger, level)(message, *args)


def close_brain() -> None:
    """Flush and close the shared Aura brain once per process."""
    global _brain_closed, _brain_shutdown_started, _brain_close_error
    with _brain_close_lock:
        if _brain_closed:
            return
        _brain_shutdown_started = True
        if _brain_instance is None:
            _brain_closed = True
            _safe_shutdown_log("info", "Brain close skipped because Aura brain was never initialized")
            return
        try:
            with brain_lock:
                _brain_instance.close()
            _safe_shutdown_log("info", "Brain closed cleanly")
        except Exception as exc:
            _brain_close_error = str(exc)
            if _is_expected_brain_close_error(exc):
                _safe_shutdown_log("info", "Brain close finished with expected late-close condition: %s", exc)
            else:
                _safe_shutdown_log("warning", "Failed to close Aura brain cleanly: %s", exc)
        finally:
            _brain_closed = True


atexit.register(close_brain)

# knowledge unified into brain (Aura SDK v1.0.5+) — kept as None for backward compat
knowledge = None
knowledge_lock = threading.Lock()


def init_brain_policy() -> None:
    """Apply AgentPersona, TagTaxonomy, TrustConfig and ArchivalRules to the brain.

    Called once at agent startup. Reads missions.json for role/motivations.
    Safe to call multiple times — overwrites previous config.
    """
    try:
        # --- Persona ---
        persona = AgentPersona()
        persona.name = "Remy"
        persona.role = "autonomous AI research and promotion agent"
        persona.tone = "professional"

        traits = PersonaTraits()
        traits.curiosity = 0.9
        traits.formality = 0.6
        traits.warmth = 0.7
        traits.conciseness = 0.8
        persona.traits = traits

        brain.set_persona(persona)

        # --- Tag Taxonomy ---
        tax = TagTaxonomy()
        tax.identity_tags = {"remy", "mission", "aurasdk", "survival", "wallet"}
        tax.stable_tags = {"architecture", "python", "fastapi", "research", "aura-sdk"}
        tax.volatile_tags = {"todo", "temp", "debug", "draft"}
        tax.sensitive_tags = {"wallet", "api-key", "credentials", "private-key"}
        tax.archive_protected_tags = {"mission", "aurasdk", "survival"}
        brain.set_taxonomy(tax)

        # --- Trust Config ---
        trust = TrustConfig()
        trust.source_trust = {
            "user": 1.0,
            "agent": 0.85,
            "web_search": 0.7,
            "extract_content": 0.75,
            "add_research_finding": 0.8,
        }
        trust.recency_half_life_days = 30.0
        trust.recency_boost_max = 0.2
        brain.set_trust_config(trust)

        # --- Maintenance / Archival Rules ---
        if hasattr(brain, "configure_maintenance"):
            mc = MaintenanceConfig()
            mc.decay_enabled = True
            mc.consolidation_enabled = True
            mc.insights_enabled = True
            mc.archival_enabled = True
            mc.archival_rules = [
                ArchivalRule("debug", max_age_days=3, keep_recent=0),
                ArchivalRule("temp", max_age_days=1, keep_recent=0),
                ArchivalRule("todo", max_age_days=14, keep_recent=5),
            ]
            brain.configure_maintenance(mc)

        logger.info("Brain policy initialised (persona=Remy, taxonomy=%d tags, trust=%d sources)",
                    len(tax.identity_tags) + len(tax.stable_tags),
                    len(trust.source_trust))

    except Exception as exc:  # never crash startup
        logger.warning("init_brain_policy failed (non-fatal): %s", exc)
