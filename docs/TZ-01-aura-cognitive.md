# TZ-01: Aura Cognitive Engine Improvements

**Version:** 1.0
**Date:** 2026-02-13
**Scope:** Changes to Aura Cognitive library (D:\aura-cognitive) — the memory engine
**Goal:** Strengthen recall quality, graph intelligence, and consolidation — making Aura a better standalone product

---

## Current State (v0.9.0)

- **Recall:** N-gram MinHash (128 hash functions) + optional Rust SDR engine
- **Graph:** Bidirectional weighted edges, MAX_CONNECTIONS=50, 1-hop walk (damping 0.6), causal walk (3 hops, 0.8 damping)
- **Decay:** Fixed 4-tier rates (0.80/0.90/0.95/0.99), consolidated memories get 0.999
- **Insights:** 9 zero-LLM pattern detectors (decay_risk, clusters, conflicts, hot_topic, etc.)
- **Session:** Co-activation tracking, 30-min auto-consolidate, diminishing returns edge strengthening
- **Storage:** Binary append-only log with compaction, ~O(N) recall

---

## AC-1: Enhanced Graph Walk (Multi-Hop Expansion)

**Problem:** Current recall does only 1-hop associative walk (damping 0.6). This misses transitive knowledge: A→B→C where C is relevant but not directly connected to query match A.

**Current behavior (memory.py `_recall_core`):**
```
Query matches A (score 0.8)
  A.connections = {B: 0.7}  → include B with score 0.8 × 0.7 × 0.6 = 0.336
  B.connections = {C: 0.6}  → NOT included (1-hop limit)
```

**Proposed:**
- Add configurable `max_hops` parameter (default 2, max 3) to `recall()` and `recall_structured()`
- Per-hop damping: hop_1=0.6, hop_2=0.35, hop_3=0.2 (steep falloff prevents noise)
- Visited set to prevent cycles
- Cap expanded results per hop (top 5 per hop to prevent explosion)

**API change:**
```python
recall(query, token_budget=2048, max_hops=2, ...)
recall_structured(query, top_k=20, max_hops=2, ...)
```

**Expected impact:** Better transitive recall without embeddings. "Diabetes" → "blood sugar" → "insulin resistance" becomes reachable.

**Estimated complexity:** ~50 lines in `memory.py`, backward compatible (default 2 doesn't change existing 1-hop behavior for hop_1 damping).

**Tests:** 5-7 new tests (2-hop walk, 3-hop walk, cycle prevention, hop limit, damping correctness)

---

## AC-2: Weighted Tag Similarity in Recall

**Problem:** Tag matching gives flat 0.7x penalty regardless of how many tags overlap. Records sharing 3/4 tags should rank higher than records sharing 1/4.

**Current behavior (ngram.py):**
- Tags indexed separately with fixed `0.7` weight penalty
- No gradual scoring based on tag overlap count

**Proposed:**
- Score tag matches proportionally: `tag_score = shared_tags / max(query_tags, record_tags) * 0.8`
- Blend with content score: `final = max(content_score, tag_score)` (not additive, to prevent double-counting)
- Add tag boost when tags match exactly (all query tags present): `+0.15` bonus

**Estimated complexity:** ~30 lines in `ngram.py`

**Tests:** 4 tests (partial tag match, full tag match, no tag overlap, tag-only recall)

---

## AC-3: Adaptive Decay Rates

**Problem:** Decay rates are hardcoded per level (0.80/0.90/0.95/0.99). A DOMAIN record about "daily medication" (recalled every day) decays at the same rate as a DOMAIN record about "one-time vacation trip".

**Proposed:**
- **Activation-aware decay:** Records with high activation_count get slower decay
- Formula: `effective_rate = base_rate + (1 - base_rate) * min(activation_count / 10, 1.0) * 0.5`
  - Example L1 (base 0.80): 0 activations → 0.80, 5 activations → 0.90, 10+ → 0.90
  - Example L3 (base 0.95): 0 activations → 0.95, 5 activations → 0.975, 10+ → 0.975
- This replaces the binary `activation_count >= 5 → 0.999` hack with a smooth curve
- **Tag-aware boost:** If record has a tag present in 5+ other active records (hot topic), apply extra `*1.02` boost to strength during decay (prevents popular topics from fading)

**API:** No API change. Behavior change in `record.apply_decay()`.

**Backward compat:** Slightly different decay curves. Existing records will decay slightly differently. Acceptable.

**Estimated complexity:** ~40 lines in `record.py` + `memory.py`

**Tests:** 6 tests (activation-based slowdown, tag hotness boost, edge cases, backward compat check)

---

## AC-4: Native Consolidation Support

**Problem:** Consolidation logic currently lives in Remy's `background_brain.py`. This should be an Aura Cognitive primitive so any agent using Aura gets consolidation.

**Current Remy implementation:**
1. `_find_consolidation_clusters()` — groups by tags
2. `_verify_cluster_similarity()` — recall_structured check
3. `_merge_cluster()` — LLM summarize + connect + mark
4. `_generate_consolidation_summary()` — LLM call

**Proposed Aura API:**
```python
# New method on CognitiveMemory
brain.consolidate(
    summarize_fn: Callable[[str, list[dict]], str | None],
    skip_tags: set[str] = DEFAULT_SKIP_TAGS,
    min_cluster_size: int = 3,
    max_clusters: int = 3,
) -> ConsolidationResult

@dataclass
class ConsolidationResult:
    clusters_found: int
    records_merged: int
    meta_records_created: int
    meta_record_ids: list[str]
```

**Key design:**
- Aura handles clustering, similarity verification, connection, marking
- **LLM call is external** (passed as `summarize_fn`) — keeps Aura model-agnostic
- Default `skip_tags` includes common system tags
- Meta-records stored with `content_type="consolidation"` (new content type)
- Original records get `consolidated_into` in metadata
- Connections automatically created (0.8 weight)

**Migration:** Remy's `_consolidate_records()` becomes a thin wrapper calling `brain.consolidate()` with LLM fn.

**Estimated complexity:** ~150 lines in new `consolidation.py` module + ~30 lines in `memory.py`

**Tests:** 10 tests (cluster finding, similarity check, merge, skip tags, already consolidated, max limit, summarize_fn failure, min size, meta-record structure, connection verification)

---

## AC-5: Record Importance Scoring

**Problem:** No unified "importance" metric. Recall relies on `overlap × strength × recency` which doesn't account for how central a record is to the knowledge graph.

**Proposed:**
- Add computed `importance` property to CognitiveRecord:
  ```python
  @property
  def importance(self) -> float:
      """0.0-1.0 importance based on level, connections, activations, strength."""
      level_weight = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.9}[self.level]
      conn_score = min(len(self.connections) / 20, 1.0)
      act_score = min(self.activation_count / 10, 1.0)
      return (level_weight * 0.3 + conn_score * 0.3 + act_score * 0.2 + self.strength * 0.2)
  ```
- Use in recall ranking: `score = overlap * importance * recency` (replaces `overlap * strength * recency`)
- Use in consolidation: higher importance records become cluster anchors
- Expose via `recall_structured` results

**Estimated complexity:** ~25 lines in `record.py`, ~10 lines in `memory.py`

**Tests:** 5 tests (low importance, high importance, boundary cases, ranking impact, recall_structured inclusion)

---

## AC-6: Synonym Ring Expansion for Ukrainian/Russian

**Problem:** Built-in `synonyms.toml` is English-only. Agent primarily communicates in Ukrainian/Russian. "Здоров'я" and "health" don't match. "Бiль" and "pain" don't connect.

**Proposed:**
- Expand `synonyms.toml` with multilingual groups:
  ```toml
  [groups]
  health = ["health", "здоров'я", "здоровье", "хелс"]
  pain = ["pain", "біль", "боль"]
  sleep = ["sleep", "сон", "спати", "спать"]
  exercise = ["exercise", "вправа", "упражнение", "спорт"]
  medication = ["medication", "ліки", "лекарство", "medicine", "drugs"]
  doctor = ["doctor", "лікар", "врач", "доктор"]
  food = ["food", "їжа", "еда", "харчування", "питание"]
  stress = ["stress", "стрес", "стресс"]
  ```
- Add `SynonymRing.load_toml_dir(dir_path)` — load multiple .toml files (user can add custom)
- Add `SynonymRing.add_group_from_list(words)` — runtime expansion

**Estimated complexity:** ~50 lines in `synonym.py`, ~100 lines in new `synonyms_multilingual.toml`

**Tests:** 4 tests (cross-language recall, Ukrainian query → English match, custom group loading, runtime expansion)

---

## AC-7: Causal Chain Metadata

**Problem:** `caused_by_id` is single-parent only. Real reasoning is multi-causal: "I recommend exercise because [sleep study] AND [your stress level] AND [doctor's advice]".

**Proposed:**
- Add `caused_by_ids: list[str]` field to CognitiveRecord (alongside existing `caused_by_id` for backward compat)
- Causal walk in recall follows ALL causes (breadth-first, max 3 hops)
- Store operation accepts `caused_by_ids` parameter
- Preamble formatting shows multi-causal chains:
  ```
  - Exercise recommendation [health]
      ^ because: Sleep study shows poor quality
      ^ because: Your stress level is elevated
  ```

**Backward compat:** Existing `caused_by_id` continues to work. `caused_by_ids` adds to it.

**Estimated complexity:** ~60 lines in `record.py` + `memory.py` + `store.py`

**Tests:** 5 tests (multi-causal store, walk breadth-first, backward compat, preamble format, deep chain)

---

## Priority Ranking

| ID | Feature | Impact on Remy | Standalone Value | Complexity | Priority |
|----|---------|---------------|-----------------|------------|----------|
| AC-1 | Multi-hop graph walk | HIGH — compensates for no embeddings | HIGH | Medium | 1 |
| AC-4 | Native consolidation | HIGH — deduplicates Remy code | HIGH | Medium | 2 |
| AC-5 | Importance scoring | MEDIUM — better recall ranking | HIGH | Low | 3 |
| AC-3 | Adaptive decay | MEDIUM — longer life for important records | HIGH | Low | 4 |
| AC-6 | Multilingual synonyms | HIGH — Ukrainian/Russian support | MEDIUM | Low | 5 |
| AC-2 | Weighted tag similarity | MEDIUM — better tag-based recall | MEDIUM | Low | 6 |
| AC-7 | Multi-causal chains | LOW — nice-to-have reasoning | MEDIUM | Medium | 7 |

---

## Implementation Order (Suggested Sprints)

**Sprint 1 (Engine Core):** AC-1 + AC-5 + AC-3
- Multi-hop walk + importance scoring + adaptive decay
- All modify recall/ranking pipeline — natural batch
- ~115 lines of code, ~16 tests

**Sprint 2 (Consolidation):** AC-4 + AC-2
- Native consolidation + tag scoring
- Moves consolidation from Remy into engine
- ~180 lines of code, ~14 tests

**Sprint 3 (Localization):** AC-6 + AC-7
- Multilingual synonyms + multi-causal chains
- Less urgent but high user impact for Ukrainian speakers
- ~210 lines of code, ~9 tests

---

## Version Target

After all 7 tasks: **Aura Cognitive v1.0.0** — production-ready memory engine with multi-hop recall, adaptive decay, native consolidation, importance scoring, multilingual support.
