"""
Remy Package
"""
__version__ = "0.1.0"

import json
from enum import Enum


def _stringify_metadata(metadata):
    if not metadata:
        return {}
    result = {}
    for key, value in metadata.items():
        if value is None:
            result[key] = ""
        elif isinstance(value, bool):
            result[key] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            result[key] = str(value)
        elif isinstance(value, (dict, list)):
            result[key] = json.dumps(value, ensure_ascii=False)
        else:
            result[key] = str(value)
    return result


def _deserialize_metadata(metadata):
    if not metadata:
        return {}
    result = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            lowered = value.lower()
            if lowered == "true":
                result[key] = True
                continue
            if lowered == "false":
                result[key] = False
                continue
            if value == "":
                result[key] = None
                continue
            if value.lstrip("-").replace(".", "", 1).isdigit():
                result[key] = float(value) if "." in value else int(value)
                continue
            if value.startswith("[") or value.startswith("{"):
                try:
                    result[key] = json.loads(value)
                    continue
                except Exception:
                    pass
        result[key] = value
    return result


_POSITIVE_OUTCOME_WORDS = {
    "pass",
    "passed",
    "passes",
    "success",
    "successful",
    "succeeded",
    "verified",
    "approved",
    "complete",
    "completed",
}
_NEGATIVE_OUTCOME_WORDS = {
    "fail",
    "failed",
    "fails",
    "error",
    "errors",
    "rejected",
    "blocked",
    "broken",
    "denied",
}
_DEPLOYMENT_WORDS = {"deploy", "deployment", "staging", "production", "release"}
_VERIFICATION_WORDS = {"audit", "verify", "verification", "scan", "review"}
_SUBJECT_STOP_WORDS = {"a", "an", "the", "to", "for", "with", "and", "or"}


def _record_words(content, tags):
    text = " ".join([str(content or ""), *(str(tag) for tag in (tags or []))]).lower()
    return {part.strip(".,:;!?()[]{}\"'") for part in text.split()}


def _infer_outcome_value(content, tags):
    words = _record_words(content, tags)
    if words & _POSITIVE_OUTCOME_WORDS:
        return "positive"
    if words & _NEGATIVE_OUTCOME_WORDS:
        return "negative"
    return None


def _infer_state_slot(content, tags, state_value):
    if state_value in {"positive", "negative"}:
        return "outcome"
    words = _record_words(content, tags)
    if words & _DEPLOYMENT_WORDS:
        return "deployment_status"
    if words & _VERIFICATION_WORDS:
        return "verification_status"
    return None


def _infer_subject(content, tags, state_value, state_slot, metadata):
    explicit = (metadata or {}).get("subject")
    if explicit:
        return explicit
    if state_slot is None:
        return None

    words_to_strip = _POSITIVE_OUTCOME_WORDS | _NEGATIVE_OUTCOME_WORDS | _SUBJECT_STOP_WORDS
    pieces = []
    for raw in str(content or "").lower().split():
        word = raw.strip(".,:;!?()[]{}\"'")
        if word and word not in words_to_strip:
            pieces.append(word)
    if pieces:
        return " ".join(pieces)
    if tags:
        return ":".join(str(tag).lower() for tag in tags if str(tag).lower() not in words_to_strip) or None
    return state_value


def _coerce_outcome_polarity(state_value):
    if state_value not in {"positive", "negative"}:
        return None
    try:
        from aura import OutcomePolarity

        return OutcomePolarity.Positive if state_value == "positive" else OutcomePolarity.Negative
    except Exception:
        return state_value


class _AuraRecordProxy:
    def __init__(self, record):
        self._record = record
        self.id = getattr(record, "id", None)
        self.content = getattr(record, "content", "")
        self.tags = list(getattr(record, "tags", []) or [])
        self.level = getattr(record, "level", None)
        self.strength = getattr(record, "strength", 0.0)
        self.activation_count = getattr(record, "activation_count", 0)
        self.connections = dict(getattr(record, "connections", {}) or {})
        self.created_at = getattr(record, "created_at", None)
        self.metadata = _deserialize_metadata(getattr(record, "metadata", None) or {})
        inferred_value = _infer_outcome_value(self.content, self.tags)
        self.state_value = getattr(record, "state_value", None) or inferred_value
        self.state_slot = getattr(record, "state_slot", None) or _infer_state_slot(
            self.content,
            self.tags,
            self.state_value,
        )
        self.subject = getattr(record, "subject", None) or _infer_subject(
            self.content,
            self.tags,
            self.state_value,
            self.state_slot,
            self.metadata,
        )
        self.outcome_domain = list(getattr(record, "outcome_domain", None) or self.tags or [])
        self.outcome_polarity = getattr(record, "outcome_polarity", None) or _coerce_outcome_polarity(
            self.state_value
        )
        self.conflict_mass = getattr(record, "conflict_mass", 0.0)
        self.volatility = getattr(record, "volatility", 0.0)

    def __getattr__(self, name):
        return getattr(self._record, name)

    def is_alive(self) -> bool:
        return self.strength >= 0.05

    def can_promote(self) -> bool:
        return self.activation_count >= 1 and self.strength >= 0.7

    def apply_decay(self) -> None:
        level = self.level
        if level is not None:
            rate = float(getattr(level, "decay_rate", 0.95))
        else:
            rate = 0.95
        self.strength = max(0.0, self.strength * rate)
        # sync back to underlying record if mutable
        try:
            object.__setattr__(self._record, "strength", self.strength)
        except Exception:
            pass


class _AuraStoreResult(str):
    def __new__(cls, record_id, *, content="", level=None, tags=None, metadata=None):
        obj = str.__new__(cls, record_id or "")
        obj.id = record_id or ""
        obj.content = content or ""
        obj.level = level
        obj.tags = list(tags or [])
        obj.metadata = dict(metadata or {})
        obj.strength = 1.0
        obj.activation_count = 0
        obj.connections = {}
        return obj

    def is_alive(self) -> bool:
        return self.strength >= 0.05

    def can_promote(self) -> bool:
        return self.activation_count >= 1 and self.strength >= 0.7

    def apply_decay(self) -> None:
        level = self.level
        if level is not None:
            rate = float(getattr(level, "decay_rate", 0.95))
        else:
            rate = 0.95
        self.strength = max(0.0, self.strength * rate)


try:
    from aura import Aura as _Aura
    from aura import Level as _AuraLevel
    import aura as _aura_module

    if not hasattr(_aura_module, "OutcomePolarity"):
        class _OutcomePolarity(Enum):
            Positive = "positive"
            Negative = "negative"
            Neutral = "neutral"

        _aura_module.OutcomePolarity = _OutcomePolarity

    for _upper, _camel in (
        ("WORKING", "Working"),
        ("DECISIONS", "Decisions"),
        ("DOMAIN", "Domain"),
        ("IDENTITY", "Identity"),
    ):
        if not hasattr(_AuraLevel, _upper) and hasattr(_AuraLevel, _camel):
            setattr(_AuraLevel, _upper, getattr(_AuraLevel, _camel))

    # Patch Level to support comparison operators (ordered by decay_rate)
    if not hasattr(_AuraLevel, "_remy_cmp_patched"):
        try:
            def _level_ge(self, other):
                return float(getattr(self, "decay_rate", 0)) >= float(getattr(other, "decay_rate", 0))
            def _level_le(self, other):
                return float(getattr(self, "decay_rate", 0)) <= float(getattr(other, "decay_rate", 0))
            def _level_gt(self, other):
                return float(getattr(self, "decay_rate", 0)) > float(getattr(other, "decay_rate", 0))
            def _level_lt(self, other):
                return float(getattr(self, "decay_rate", 0)) < float(getattr(other, "decay_rate", 0))
            _AuraLevel.__ge__ = _level_ge
            _AuraLevel.__le__ = _level_le
            _AuraLevel.__gt__ = _level_gt
            _AuraLevel.__lt__ = _level_lt
            _AuraLevel._remy_cmp_patched = True
        except Exception:
            pass

    # Module-level activation tracker: id(brain_instance) → {record_id: count}
    _remy_activation_store: dict = {}
    _remy_connection_store: dict = {}

    if not getattr(_Aura, "_remy_compat_patched", False):
        _orig_store = _Aura.store
        _orig_update = getattr(_Aura, "update", None)
        _orig_search = getattr(_Aura, "search", None)
        _orig_get = getattr(_Aura, "get", None)
        _orig_recall = getattr(_Aura, "recall", None)
        _orig_reflect = getattr(_Aura, "reflect", None)
        _orig_connect = getattr(_Aura, "connect", None)
        _orig_decay = getattr(_Aura, "decay", None)

        def _wrap_record(record, instance=None):
            if record is None or isinstance(record, _AuraRecordProxy):
                return record
            proxy = _AuraRecordProxy(record)
            # Overlay compat-tracked activation count if higher
            if instance is not None:
                tracker = _remy_activation_store.get(id(instance), {})
                rid = proxy.id
                if rid and rid in tracker:
                    proxy.activation_count = max(proxy.activation_count, tracker[rid])
                connections = _remy_connection_store.get(id(instance), {}).get(rid, {})
                if connections:
                    proxy.connections.update(connections)
            return proxy

        def _wrap_records(records, instance=None):
            wrapped = []
            for item in records or []:
                if isinstance(item, tuple) and len(item) == 2:
                    score, record = item
                    wrapped.append((score, _wrap_record(record, instance)))
                else:
                    wrapped.append(_wrap_record(item, instance))
            return wrapped

        def _compat_store(self, content, *args, metadata=None, **kwargs):
            metadata = _stringify_metadata(metadata)
            result = _orig_store(self, content, *args, metadata=metadata, **kwargs)
            if isinstance(result, str):
                return _AuraStoreResult(
                    result,
                    content=content,
                    level=kwargs.get("level"),
                    tags=kwargs.get("tags"),
                    metadata=metadata,
                )
            return result

        def _compat_update(self, record_id, *args, metadata=None, **kwargs):
            metadata = None if metadata is None else _stringify_metadata(metadata)
            return _orig_update(self, record_id, *args, metadata=metadata, **kwargs)

        def _compat_search(self, *args, **kwargs):
            if _orig_search is None:
                return []
            return _wrap_records(_orig_search(self, *args, **kwargs), self)

        def _compat_get(self, record_id):
            if _orig_get is None:
                return None
            return _wrap_record(_orig_get(self, record_id), self)

        def _compat_list_records(self, tags=None, min_strength=0.0, limit=5000):
            records = _compat_search(self, query="", tags=tags, limit=limit)
            if min_strength > 0:
                records = [
                    rec for rec in records
                    if getattr(rec, "strength", 0.0) >= min_strength
                ]
            return list(records)

        def _compat_recall(self, query, *args, **kwargs):
            result = _orig_recall(self, query, *args, **kwargs)
            # Track activations per record using module-level store
            try:
                inst_id = id(self)
                if inst_id not in _remy_activation_store:
                    _remy_activation_store[inst_id] = {}
                tracker = _remy_activation_store[inst_id]
                hits = _orig_search(self, query=query, limit=5) if _orig_search else []
                for item in hits or []:
                    if isinstance(item, tuple):
                        item = item[1]
                    rid = getattr(item, "id", None)
                    if rid:
                        tracker[rid] = tracker.get(rid, 0) + 1
            except Exception:
                pass
            return result

        def _compat_connect(self, id_a, id_b, *args, weight=0.0, **kwargs):
            result = None
            if _orig_connect is not None:
                try:
                    result = _orig_connect(self, id_a, id_b, *args, weight=weight, **kwargs)
                except TypeError:
                    result = _orig_connect(self, id_a, id_b, weight, *args, **kwargs)
            store = _remy_connection_store.setdefault(id(self), {})
            store.setdefault(id_a, {})[id_b] = weight
            store.setdefault(id_b, {})[id_a] = weight
            return result

        def _compat_decay(self, *args, **kwargs):
            result = None
            if _orig_decay is not None:
                result = _orig_decay(self, *args, **kwargs)
            store = _remy_connection_store.get(id(self), {})
            for source_id in list(store.keys()):
                for target_id, weight in list(store[source_id].items()):
                    decayed = float(weight) * 0.99
                    if decayed < 0.05:
                        store[source_id].pop(target_id, None)
                    else:
                        store[source_id][target_id] = decayed
                if not store[source_id]:
                    store.pop(source_id, None)
            return result

        def _compat_reflect(self, *args, **kwargs):
            if _orig_reflect is None:
                return {"archived": 0, "promoted": 0, "connected": 0}
            result = _orig_reflect(self, *args, **kwargs)
            if isinstance(result, dict):
                result = dict(result)
                if "connected" not in result:
                    result["connected"] = 0
                # Compat promotion: promote records that meet can_promote() criteria
                try:
                    if _orig_search is not None:
                        all_recs = _orig_search(self, query="", limit=500) or []
                        promoted = result.get("promoted", 0)
                        for item in all_recs:
                            rec = item[1] if isinstance(item, tuple) else item
                            proxy = _wrap_record(rec, self)
                            if proxy and proxy.can_promote():
                                lvl = proxy.level
                                lvl_name = str(lvl) if lvl else "WORKING"
                                if "WORKING" in lvl_name:
                                    # Promote to DECISIONS level
                                    try:
                                        from aura import Level as _L
                                        _orig_update(self, proxy.id, level=_L.Decisions)
                                        promoted += 1
                                    except Exception:
                                        pass
                        result["promoted"] = promoted
                except Exception:
                    pass
            return result

        _Aura.store = _compat_store
        if _orig_update is not None:
            _Aura.update = _compat_update
        if _orig_search is not None:
            _Aura.search = _compat_search
        if _orig_get is not None:
            _Aura.get = _compat_get
        if not hasattr(_Aura, "list_records"):
            _Aura.list_records = _compat_list_records
        if _orig_recall is not None:
            _Aura.recall = _compat_recall
        if _orig_connect is not None:
            _Aura.connect = _compat_connect
        if _orig_decay is not None:
            _Aura.decay = _compat_decay
        if _orig_reflect is not None:
            _Aura.reflect = _compat_reflect
        _Aura._remy_compat_patched = True
except Exception:
    pass
