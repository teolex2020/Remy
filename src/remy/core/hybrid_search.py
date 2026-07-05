"""Hybrid memory search helpers."""

from __future__ import annotations

from collections.abc import Iterable

from remy.core.memory_policy import sanitize_memory_content, sanitize_memory_metadata


_EXCLUDED_RECALL_TAGS = frozenset({"web-search-cache"})

# Phase A.8 — tags that must never be the primary factual substrate.
# Records carrying any of these tags can exist in memory but must not be
# returned as authoritative evidence on a factual/citation/verify path.
_FACTUAL_FORBIDDEN_TAGS = frozenset({
    "generated-report",    # LLM-synthesised reports stored back into brain
    "research-project",    # scaffolding records for in-progress research runs
    "research-finding",    # intermediate findings from tool-loop; not verified
    "research",            # general research summaries / narrative blobs
    "research-summary",    # explicit summary records
    "session-summary",     # compressed conversation memory, not factual substrate
    "scratchpad",          # ephemeral working notes
    "quarantine-unverified",  # explicitly quarantined by admission guard
    "claim:llm-unverified",   # LLM-generated claim without tool grounding
})


def _is_excluded_item(item: dict) -> bool:
    return bool(_EXCLUDED_RECALL_TAGS.intersection(set(item.get("tags") or [])))


_PROMOTION_BLOCKED_TRUTH_STATUSES = frozenset({
    "stale_hard",
    "conflict_unresolved",
    "superseded",
})


def _is_factual_forbidden(item: dict) -> bool:
    """Return True if this item must not be primary input on a factual path.

    Checks four signals:
      1. Tag-based (A.8): any of _FACTUAL_FORBIDDEN_TAGS present.
      2. Admission-class-based (A.9): metadata.admission_class in
         FACTUAL_FORBIDDEN_ADMISSION_CLASSES or derive_admission_class()
         returns a forbidden class.
      3. Promotion-flag-based (Phase 3): requires_promotion=True and not yet
         promoted — the writer stamped the record as needing promotion and no
         explicit promotion event has flipped it.
      4. Truth-state-based (Phase 4): record is conflict_unresolved, superseded,
         or stale_hard — structural freshness/conflict flags must block the
         factual surface even when admission_class alone would allow it.

    A record is forbidden if ANY signal fires.
    """
    # Tag check (A.8 gate — fast path)
    if _FACTUAL_FORBIDDEN_TAGS.intersection(set(item.get("tags") or [])):
        return True

    meta = item.get("metadata") or {}
    tags = item.get("tags") or []

    # Promotion-flag check (Phase 3): writer said "needs promotion", nothing
    # promoted it. Admitted != promoted.
    if meta.get("requires_promotion") and not meta.get("promoted"):
        return True

    # Truth-state check (Phase 4): structural freshness/conflict flags.
    # Fast path on explicit metadata first; only fall back to recomputing
    # truth_status if explicit flags are absent.
    if meta.get("superseded_by"):
        return True
    if meta.get("unresolved_conflict"):
        return True
    try:
        from remy.core.retrieval.freshness import truth_status
        status = truth_status(meta)
        if status in _PROMOTION_BLOCKED_TRUTH_STATUSES:
            return True
    except Exception:
        pass  # best-effort; never block recall on import error

    # Admission class check (A.9 gate)
    try:
        from remy.core.memory_policy import (
            FACTUAL_FORBIDDEN_ADMISSION_CLASSES,
            derive_admission_class,
        )
        cls = derive_admission_class(meta, tags)
        return cls in FACTUAL_FORBIDDEN_ADMISSION_CLASSES
    except Exception:
        return False  # best-effort; never block recall on import error


def _normalize_level(level) -> str | None:
    if level is None:
        return None
    if isinstance(level, str):
        return level.upper().replace("LEVEL.", "")
    try:
        name = getattr(level, "name", None)
        if name:
            return str(name).upper()
    except Exception:
        pass
    return str(level).upper().replace("LEVEL.", "")


def _record_to_item(rec, score: float) -> dict:
    metadata = getattr(rec, "metadata", None) or {}
    tags = list(getattr(rec, "tags", []) or [])
    item = {
        "id": getattr(rec, "id", None),
        "content": sanitize_memory_content(
            getattr(rec, "content", "") or "",
            metadata=metadata,
            tags=tags,
        ),
        "tags": tags,
        "level": _normalize_level(getattr(rec, "level", None)),
        "strength": float(getattr(rec, "strength", 0.0) or 0.0),
        "activation_count": int(getattr(rec, "activation_count", 0) or 0),
        "metadata": sanitize_memory_metadata(metadata, tags=tags),
        "source": metadata.get("source"),
        "importance": getattr(rec, "importance", None),
        "score": float(score),
    }
    # Phase 3: cognitive fields — used by _render_hits for epistemic annotations
    confidence = getattr(rec, "confidence", None)
    if confidence is not None:
        item["confidence"] = float(confidence)
    conflict = getattr(rec, "conflict_mass", None)
    if conflict is not None and int(conflict) > 0:
        item["conflict_mass"] = int(conflict)
    subject = getattr(rec, "subject", None)
    if subject:
        item["subject"] = str(subject)
    polarity = getattr(rec, "outcome_polarity", None)
    if polarity is not None:
        item["outcome_polarity"] = str(polarity).split(".")[-1].lower()
    return item


def _iter_semantic_items(results: Iterable | None) -> Iterable[dict]:
    for item in results or []:
        if isinstance(item, tuple) and len(item) == 2:
            score, rec = item
            yield _record_to_item(rec, float(score or 0.0))
        else:
            yield _record_to_item(item, 1.0)


def _merge_item(existing: dict, incoming: dict) -> dict:
    existing["score"] = max(float(existing.get("score", 0.0) or 0.0), float(incoming.get("score", 0.0) or 0.0))
    existing["strength"] = max(float(existing.get("strength", 0.0) or 0.0), float(incoming.get("strength", 0.0) or 0.0))
    existing["activation_count"] = max(int(existing.get("activation_count", 0) or 0), int(incoming.get("activation_count", 0) or 0))
    if incoming.get("metadata") and not existing.get("metadata"):
        existing["metadata"] = incoming["metadata"]
    if incoming.get("source") and not existing.get("source"):
        existing["source"] = incoming["source"]
    if incoming.get("level") and not existing.get("level"):
        existing["level"] = incoming["level"]
    if incoming.get("importance") is not None and existing.get("importance") is None:
        existing["importance"] = incoming["importance"]
    return existing


def _merge_ranked_items(*groups: Iterable[dict], top_k: int, tags: list[str] | None = None) -> list[dict]:
    combined: dict[str, dict] = {}
    for group in groups:
        for item in group or []:
            item_id = item.get("id")
            if not item_id:
                continue
            item_tags = item.get("tags") or []
            if tags and not any(tag in item_tags for tag in tags):
                continue
            if item_id in combined:
                combined[item_id] = _merge_item(combined[item_id], item)
            else:
                combined[item_id] = item

    results = list(combined.values())
    results.sort(
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            float(item.get("strength", 0.0) or 0.0),
            int(item.get("activation_count", 0) or 0),
        ),
        reverse=True,
    )
    return results[:top_k]


def search_exact_structured(
    brain,
    query: str,
    *,
    tags: list[str] | None = None,
    top_k: int = 10,
    lexical_limit: int = 10,
) -> list[dict]:
    """Structured exact/core retrieval over IDENTITY and DOMAIN levels."""
    tags = [tag for tag in (tags or []) if tag]
    lexical_limit = max(top_k, lexical_limit)
    lexical = []
    try:
        for level in ("IDENTITY", "DOMAIN"):
            lexical.extend(
                item
                for item in (
                    _record_to_item(rec, 1.0 if not query else 0.68)
                    for rec in brain.search(query=query or "", tags=tags or None, level=level, limit=lexical_limit)
                )
                if not _is_excluded_item(item)
            )
    except Exception:
        lexical = []
    return _merge_ranked_items(lexical, top_k=top_k, tags=tags)


def recall_cognitive_structured(
    brain,
    query: str,
    *,
    tags: list[str] | None = None,
    top_k: int = 10,
    min_strength: float | None = 0.05,
    session_id: str | None = None,
) -> list[dict]:
    """Structured cognitive recall over DECISIONS and WORKING levels.

    Provenance-aware: when the installed AuraSDK exposes `recall_provenance_ranked`,
    ordering is driven by born-from-collision provenance so a memory from a lived
    consequence outranks an equally-relevant model-generated description. Falls
    back transparently to plain `recall_structured` otherwise.
    """
    tags = [tag for tag in (tags or []) if tag]
    semantic = []
    # Preferred: provenance-ranked recall (lived-consequence first).
    if hasattr(brain, "recall_provenance_ranked"):
        try:
            raw = brain.recall_provenance_ranked(
                query,
                top_k=max(top_k * 2, 8),
                min_strength=min_strength,
                session_id=session_id,
            )
            semantic = [
                _provenance_dict_to_item(d) for d in (raw or []) if isinstance(d, dict)
            ]
        except Exception:
            semantic = []
    # Fallback (or empty result): ordinary structured recall.
    if not semantic:
        try:
            semantic = list(
                _iter_semantic_items(
                    brain.recall_structured(
                        query,
                        top_k=max(top_k * 2, 8),
                        min_strength=min_strength,
                        session_id=session_id,
                    )
                )
            )
        except Exception:
            semantic = []
    semantic = [
        item for item in semantic
        if item.get("level") in {"DECISIONS", "WORKING"} and not _is_excluded_item(item)
    ]
    return _merge_ranked_items(semantic, top_k=top_k, tags=tags)


def _provenance_dict_to_item(d: dict) -> dict:
    """Map a `recall_provenance_ranked` result dict to the internal item shape.

    The SDK already applied the born-from-collision multiplier, so we carry
    `effective_score` as the item's `score` — a lived-consequence memory thereby
    ranks above an equally-relevant model-generated one. `provenance` is kept for
    epistemic annotation.
    """
    tags = list(d.get("tags") or [])
    item = {
        "id": d.get("id"),
        "content": sanitize_memory_content(d.get("content") or "", metadata={}, tags=tags),
        "tags": tags,
        "level": _normalize_level(d.get("level")),
        "strength": float(d.get("strength") or 0.0),
        "metadata": {},
        # effective_score = relevance × provenance multiplier (born-from-collision)
        "score": float(d.get("effective_score", d.get("score", 0.0)) or 0.0),
        "provenance": d.get("provenance"),
    }
    return item


def recall_provenance_ranked(
    brain,
    query: str,
    *,
    tags: list[str] | None = None,
    top_k: int = 10,
    min_strength: float | None = 0.05,
    session_id: str | None = None,
) -> list[dict]:
    """Provenance-ranked cognitive recall.

    Same contract as `recall_cognitive_structured`, but ordering is driven by the
    SDK's `recall_provenance_ranked` so memories born from a lived consequence
    outrank model-generated descriptions of equal surface relevance. Fail-soft:
    if the installed AuraSDK lacks the native method, falls back to the ordinary
    cognitive recall, so behavior is never worse than before.
    """
    tags = [tag for tag in (tags or []) if tag]
    if not hasattr(brain, "recall_provenance_ranked"):
        return recall_cognitive_structured(
            brain, query, tags=tags, top_k=top_k,
            min_strength=min_strength, session_id=session_id,
        )
    try:
        raw = brain.recall_provenance_ranked(
            query,
            top_k=max(top_k * 2, 8),
            min_strength=min_strength,
            session_id=session_id,
        )
    except Exception:
        return recall_cognitive_structured(
            brain, query, tags=tags, top_k=top_k,
            min_strength=min_strength, session_id=session_id,
        )
    semantic = [
        _provenance_dict_to_item(d)
        for d in (raw or [])
        if isinstance(d, dict)
    ]
    semantic = [
        item for item in semantic
        if item.get("level") in {"DECISIONS", "WORKING"} and not _is_excluded_item(item)
    ]
    return _merge_ranked_items(semantic, top_k=top_k, tags=tags)


def hybrid_search_structured(
    brain,
    query: str,
    *,
    tags: list[str] | None = None,
    top_k: int = 10,
    min_strength: float | None = 0.05,
    session_id: str | None = None,
    lexical_limit: int = 10,
) -> list[dict]:
    """Combine exact/core retrieval and cognitive recall into one ranked result list."""
    query = (query or "").strip()
    tags = [tag for tag in (tags or []) if tag]

    exact = search_exact_structured(
        brain,
        query,
        tags=tags,
        top_k=top_k,
        lexical_limit=lexical_limit,
    )
    cognitive = []
    if query:
        cognitive = recall_cognitive_structured(
            brain,
            query,
            tags=tags,
            top_k=top_k,
            min_strength=min_strength,
            session_id=session_id,
        )
    return _merge_ranked_items(exact, cognitive, top_k=top_k, tags=tags)


def build_cognitive_context(brain) -> str | None:
    """Build a cognitive awareness block for LLM context injection.

    Surfaces epistemic state: what the system is confident about,
    where contradictions exist, and what patterns it has detected.
    This gives the LLM situational awareness beyond raw recall.

    Returns a compact text block or None if nothing notable.
    """
    sections: list[str] = []

    # 1. Active contradictions — things the system is uncertain about
    try:
        clusters = brain.get_contradiction_clusters(limit=3)
        if clusters:
            lines = []
            for c in clusters:
                tags = getattr(c, "shared_tags", None) or []
                keys = getattr(c, "belief_keys", None) or []
                n_beliefs = len(getattr(c, "belief_ids", []) or [])
                conflict = getattr(c, "max_conflict_mass", 0)
                if n_beliefs < 2 or conflict == 0:
                    continue  # skip clusters without real conflict signal
                topic = ", ".join(tags[:4]) if tags else (
                    ", ".join(k.split(":")[-1][:30] for k in keys[:3]) if keys else "unknown"
                )
                lines.append(
                    f"  - {topic} ({n_beliefs} beliefs, conflict={conflict:.1f})"
                )
            if not lines:
                lines = None
            if lines:
                sections.append(
                    "Contradictions (uncertain — treat with caution):\n" + "\n".join(lines)
                )
    except Exception:
        pass

    # 2. Surfaced concepts — stable knowledge patterns
    try:
        concepts = brain.get_surfaced_concepts(limit=5)
        if concepts:
            lines = []
            for c in concepts:
                score = getattr(c, "abstraction_score", 0) or 0
                core = getattr(c, "core_terms", []) or []
                shell = getattr(c, "shell_terms", []) or []
                n_beliefs = len(getattr(c, "belief_ids", []) or [])
                # Filter out noise: skip terms that look like base64/binary blobs
                clean_shell = [t for t in shell if len(t) < 40 and t.isascii()]
                core_str = ", ".join(core[:5]) if core else (
                    ", ".join(clean_shell[:5]) if clean_shell else "unnamed"
                )
                detail = f" (also: {', '.join(clean_shell[:3])})" if clean_shell and core else ""
                lines.append(
                    f"  - {core_str}{detail} — {n_beliefs} beliefs, score={score:.2f}"
                )
            sections.append(
                "Stable knowledge patterns:\n" + "\n".join(lines)
            )
    except Exception:
        pass

    # 3. Surfaced policy hints — behavioral guidelines from experience
    try:
        policies = brain.get_surfaced_policy_hints(limit=3)
        if policies:
            lines = []
            for p in policies:
                rec = getattr(p, "recommendation", "") or ""
                action = getattr(p, "action_kind", "") or ""
                strength = getattr(p, "strength", 0) or 0
                lines.append(f"  - [{action}] {rec[:120]} (strength={strength:.2f})")
            sections.append(
                "Learned guidelines:\n" + "\n".join(lines)
            )
    except Exception:
        pass

    # 4. Belief instability — topics where understanding is shifting
    try:
        instability = brain.get_belief_instability_summary()
        unresolved = 0
        high_vol = 0
        if isinstance(instability, dict):
            unresolved = int(instability.get("unresolved", 0) or 0)
            high_vol = int(instability.get("high_volatility_count", 0) or 0)
        else:
            unresolved = int(getattr(instability, "unresolved", 0) or 0)
            high_vol = int(getattr(instability, "high_volatility_count", 0) or 0)
        total = 0
        if isinstance(instability, dict):
            total = int(instability.get("total_beliefs", 0) or 0)
        else:
            total = int(getattr(instability, "total_beliefs", 0) or 0)
        # Only show if there's a meaningful signal (>10% unresolved or >50% volatile)
        show_unresolved = unresolved > 0 and total > 0 and (unresolved / total) > 0.10
        show_volatile = high_vol > 0 and total > 0 and (high_vol / total) > 0.50
        if show_unresolved or show_volatile:
            parts = []
            if show_unresolved:
                parts.append(f"{unresolved}/{total} unresolved beliefs")
            if show_volatile:
                parts.append(f"{high_vol}/{total} volatile beliefs")
            sections.append(f"Epistemic caution: {', '.join(parts)}.")
    except Exception:
        pass

    if not sections:
        return None

    return "[Cognitive State]\n" + "\n".join(sections)


_FACTUAL_QUERY_KEYWORDS = frozenset({
    # citation / paper lookup
    "arxiv", "doi", "isbn", "paper", "publication", "citation", "cite",
    "стаття", "публікац", "посилання",
    # verification
    "verify", "check", "confirm", "true", "false", "right", "correct",
    "правда", "перевір", "підтверди", "вірно", "точно",
    # exact lookup
    "who", "when", "where", "what year", "how many", "which",
    "хто", "коли", "де", "скільки", "який рік",
    # recent factual lists
    "list", "find papers", "show papers", "recent papers", "top papers",
    "recent research", "статті", "знайди статті",
})


def is_factual_query(query: str) -> bool:
    """Return True if the query looks like a factual/citation/verify request.

    Used by the direct recall tool handler to decide whether to apply
    the A.8 factual boundary filter.  Intentionally conservative — false
    negatives (missing a factual query) are safer than false positives
    (blocking legitimate general memory recall).
    """
    if not query:
        return False
    q = query.lower()
    return any(kw in q for kw in _FACTUAL_QUERY_KEYWORDS)


def build_evidence_packet(
    brain,
    query: str,
    *,
    top_k: int = 5,
    min_strength: float = 0.05,
    session_id: str | None = None,
) -> list[dict]:
    """Phase A.8 — factual-path bounded evidence retrieval.

    Returns only records that are safe as primary factual substrate:
    - no generated-report / research-project / research-finding / research blobs
    - no session-summary / scratchpad / quarantine / claim:llm-unverified
    - exact + semantic recall combined, forbidden classes silently excluded
    - each returned item carries verification_state and provenance fields

    The returned list is the EvidencePacket for a factual/citation/verify turn.
    It intentionally contains FEWER items than general recall — quality over quantity.
    """
    query = (query or "").strip()
    candidates: list[dict] = []

    # Exact recall: IDENTITY + DOMAIN levels
    try:
        for level in ("IDENTITY", "DOMAIN"):
            candidates.extend(
                item
                for item in (
                    _record_to_item(rec, 0.68)
                    for rec in brain.search(
                        query=query or "",
                        tags=None,
                        level=level,
                        limit=top_k * 3,
                    )
                )
                if not _is_excluded_item(item) and not _is_factual_forbidden(item)
            )
    except Exception:
        pass

    # Semantic recall: DECISIONS + WORKING levels (filtered)
    try:
        semantic_raw = list(
            _iter_semantic_items(
                brain.recall_structured(
                    query,
                    top_k=top_k * 3,
                    min_strength=min_strength,
                    session_id=session_id,
                )
            )
        )
        candidates.extend(
            item for item in semantic_raw
            if item.get("level") in {"DECISIONS", "WORKING"}
            and not _is_excluded_item(item)
            and not _is_factual_forbidden(item)
        )
    except Exception:
        pass

    # Merge, rank, cap
    evidence = _merge_ranked_items(candidates, top_k=top_k)

    # Annotate each item with structured provenance fields for mouth/LLM
    for item in evidence:
        meta = item.get("metadata") or {}
        # verification_state: derive from metadata or default to "unverified"
        verified = meta.get("verified", False)
        source = meta.get("source", "") or item.get("source", "") or ""
        item["verification_state"] = "verified" if verified else "unverified"
        item["provenance"] = source
        # claim_status: pull from metadata or infer
        item["claim_status"] = meta.get("claim_status", "stated")
        # allowed_use: grounded verified records may be cited; others — context only
        item["allowed_use"] = "cite" if verified else "context_only"

    return evidence
