# TZ-02: Remy Agent Logic Improvements

**Version:** 1.0
**Date:** 2026-02-13
**Scope:** Changes to Remy agent code (E:\remy\app) — the "driver" that uses Aura Cognitive
**Goal:** Reach Tier 2 (Research Assistant) capabilities — structured research, source tracking, proactive delivery, autonomous reliability

---

## Current State (v1.2.0)

- **Autonomous loop:** Goal-driven cycles, action plans, self-evaluation, budget management
- **Tools:** 25+ tools (recall, store, web_search, store_research, schedule_task, etc.)
- **Research mode:** Documented in system prompt, not scaffolded by code
- **Proactive sessions:** Scheduled tasks, decay risk, inactivity triggers via Telegram
- **Background brain:** 6-phase pipeline (decay, reflect, insights, consolidation, synthesis, tasks)
- **Channels:** Voice + Telegram + Desktop + Autonomous — same brain

---

## RM-1: Research Orchestrator

**Problem:** Research is currently prompt-instructed ("plan 2-3 queries, search, synthesize"). The agent doesn't track research projects, can't resume multi-session investigations, and has no structured research pipeline.

**Proposed architecture:**
```
ResearchProject (brain record, tags=["research-project"])
├── status: planning | researching | synthesizing | complete
├── topic: str
├── research_plan: list[SearchQuery]
├── findings: list[Finding]
├── sources: list[Source]
├── confidence: float
└── created_at / updated_at
```

**New tools (3):**

1. **`start_research`** — Create a research project
   ```
   Args: topic (str), depth ("quick"|"standard"|"deep"), context (str, optional)
   Returns: project_id, generated research plan (3-7 search queries)
   ```
   - Depth controls query count: quick=2, standard=4, deep=7
   - LLM generates search plan from topic + existing knowledge (via recall)
   - Stores as brain record with tags `["research-project"]`
   - Metadata: `{status, depth, query_plan, findings_count, started_at}`

2. **`add_research_finding`** — Record a finding from any source
   ```
   Args: project_id, content (str), source_url (str, optional), confidence (0.0-1.0), contradicts_finding_id (str, optional)
   Returns: finding_id, duplicate_warning (if similar exists)
   ```
   - Stores as brain record with tags `["research-finding", project_topic_slug]`
   - Auto-connects to project record (0.8 weight)
   - Checks for duplicates via recall before storing
   - If `contradicts_finding_id` provided, creates conflict connection (weight -0.5)

3. **`complete_research`** — Synthesize findings into final report
   ```
   Args: project_id
   Returns: report (str), source_count, confidence_avg
   ```
   - Gathers all findings connected to project
   - LLM synthesizes into structured report
   - Updates project status to "complete"
   - Auto-stores as `store_research` for persistence

**Decision prompt integration:**
- `_build_decision_prompt()` injects active research projects
- Agent sees: "ACTIVE RESEARCH: [topic] — 3/5 queries done, 4 findings so far"
- Naturally continues research across cycles

**Estimated complexity:** ~200 lines in `brain_tools.py`, ~30 lines in `autonomy.py`

**Tests:** 12 tests (create project, add finding, duplicate detection, contradiction, complete synthesis, depth levels, resume across sessions, decision prompt injection)

---

## RM-2: Source Credibility Scoring

**Problem:** All web search results treated equally. A Wikipedia article, a random blog, and a medical journal carry the same weight.

**Proposed:**

- **Domain reputation table** (hardcoded initial, expandable):
  ```python
  SOURCE_CREDIBILITY = {
      # Medical
      "pubmed.ncbi.nlm.nih.gov": 0.95,
      "who.int": 0.95,
      "mayoclinic.org": 0.90,
      "webmd.com": 0.70,
      "healthline.com": 0.65,
      # General
      "wikipedia.org": 0.75,
      "britannica.com": 0.85,
      # News
      "bbc.com": 0.80,
      "reuters.com": 0.85,
      # Low trust
      "reddit.com": 0.40,
      "quora.com": 0.35,
      # Default
      "_default": 0.50,
  }
  ```

- **Integration points:**
  - `web_search` tool result includes `credibility` score extracted from URL domain
  - `add_research_finding` automatically inherits source credibility
  - `complete_research` weights findings by source credibility in synthesis
  - Research report shows source quality: "Based on 3 high-credibility and 2 medium-credibility sources"

- **User-expandable:** Brain record with tag `["source-credibility"]` allows user/agent to add new domain scores

**Estimated complexity:** ~60 lines (new `source_credibility.py` module + integration in `brain_tools.py`)

**Tests:** 6 tests (known domain, unknown domain, user override, credibility in search result, weighted synthesis, persistence)

---

## RM-3: Autonomous Research Cycles

**Problem:** Autonomous mode treats research like any other goal. No specialized behavior for multi-step investigations requiring multiple web searches across cycles.

**Proposed changes to autonomy.py:**

1. **Research-aware goal detection:**
   - If active goal contains keywords ("research", "investigate", "find out", "learn about"), auto-create ResearchProject
   - Tag goal with `is_research: true` in metadata

2. **Research cycle behavior:**
   - When goal has active research project:
     - Decision prompt includes project status, remaining queries, current findings
     - Agent is instructed to execute ONE search query per cycle (token efficiency)
     - After all queries done, next cycle does synthesis
   - This spreads research across cycles instead of cramming into one

3. **Research budget awareness:**
   - Each web_search costs ~500-1000 tokens
   - Budget check before research cycle: `can_spend(estimated_research_cost)`
   - If budget tight: skip research, do zero-cost recall/organize instead
   - Token estimation: `depth * 800` for full research project

4. **Cross-session persistence:**
   - Research project state stored in brain (already handled by RM-1 tools)
   - On autonomous restart, `_decide_and_act()` checks for incomplete projects
   - Continues from last checkpoint

**Estimated complexity:** ~80 lines in `autonomy.py`, ~20 lines in `brain_tools.py`

**Tests:** 8 tests (research goal detection, cycle-by-cycle execution, budget-aware skip, cross-session resume, synthesis trigger, budget estimation, research priority vs normal goal)

---

## RM-4: Structured Fact Extraction

**Problem:** Web search results stored as free-text. No structured extraction of discrete facts that can be individually verified, connected, or contradicted.

**Proposed:**

- **Fact extraction tool:**
  ```
  extract_facts(text: str, topic: str) -> list[Fact]

  @dataclass
  class Fact:
      claim: str          # "Vitamin D deficiency affects 40% of adults"
      source: str         # URL
      confidence: float   # 0.0-1.0 based on source credibility
      category: str       # "statistic" | "recommendation" | "definition" | "opinion"
      contradicts: list[str]  # IDs of contradicting facts
  ```

- **LLM-powered extraction:** One call per web result (~100 tokens)
  ```
  Extract 2-5 discrete factual claims from this text.
  For each fact, classify as: statistic, recommendation, definition, or opinion.
  ```

- **Storage:** Each fact as separate WORKING-level record
  - Tags: `["fact", category, topic_slug]`
  - Metadata: `{source_url, confidence, category, project_id}`
  - Auto-connects facts from same source (weight 0.4)
  - Auto-connects to research project (weight 0.7)

- **Contradiction detection:**
  - Before storing new fact: `recall(claim, top_k=5)`
  - If similar fact exists with different value → mark contradiction
  - Agent notified: "New fact contradicts existing finding from [source]"

**Estimated complexity:** ~120 lines in `brain_tools.py`

**Tests:** 8 tests (extraction, categorization, dedup, contradiction detection, connection to project, confidence scoring)

---

## RM-5: Proactive Research Delivery

**Problem:** Research completes silently during autonomous cycles. User doesn't know results are ready unless they ask.

**Proposed:**

- **New proactive trigger** in `_should_start_proactive_session()`:
  ```python
  # Priority: HIGH (same as scheduled_task_due)
  "research_complete": {
      "check": lambda: brain.search(query="", tags=["research-project"], limit=5)
                        .filter(status="complete", delivered=False),
      "message_template": "I finished researching '{topic}'. Here's what I found: {summary}"
  }
  ```

- **Delivery flow:**
  1. Research completes in autonomous cycle → project status = "complete"
  2. Next proactive session check detects undelivered research
  3. Triggers proactive Telegram message with summary
  4. Marks project `delivered: true`
  5. User can ask follow-up questions (flows through normal chat)

- **Delivery format (Telegram):**
  ```
  Research complete: [Topic]

  Key findings:
  - Finding 1 (source, confidence)
  - Finding 2 (source, confidence)

  Sources: 3 high-credibility, 1 medium
  Overall confidence: 0.78

  Reply to ask follow-up questions.
  ```

**Estimated complexity:** ~40 lines in `autonomy.py`, ~20 lines in `brain_tools.py`

**Tests:** 5 tests (trigger detection, delivery format, mark delivered, no duplicate delivery, follow-up handling)

---

## RM-6: Agent Error Recovery & Reliability

**Problem:** When tools fail (API timeout, rate limit, network error), agent doesn't have explicit recovery strategy. Consecutive failures trigger circuit breaker (10 min pause) but don't adapt behavior.

**Proposed:**

1. **Retry with backoff** for transient failures:
   ```python
   # In execute_tool()
   RETRYABLE_TOOLS = {"web_search", "http_get"}
   MAX_RETRIES = 2
   RETRY_DELAY = [2, 5]  # seconds
   ```
   - Only retry tools known to have transient failures
   - Log retry attempts

2. **Graceful degradation** in autonomous mode:
   - If `web_search` fails 2x → switch to recall-only mode for this cycle
   - If `store` fails → queue writes for next cycle (in-memory buffer, max 5)
   - If all tools fail → emit "degraded_mode" event, skip cycle

3. **Smart circuit breaker:**
   - Current: 3 consecutive failures → 10 min pause
   - Proposed: Track failure by tool type, not globally
   - `web_search` fails 3x → pause web_search only, keep recall/store working
   - Full circuit break only if 3+ different tools fail

4. **Error context in decision prompt:**
   - Add "TOOL HEALTH" section to `_build_decision_prompt()`:
     ```
     TOOL HEALTH:
     - web_search: degraded (2 failures, last: timeout)
     - recall: healthy
     - store: healthy
     ```
   - Agent can avoid broken tools and adapt strategy

**Estimated complexity:** ~100 lines in `autonomy.py` + `brain_tools.py`

**Tests:** 10 tests (retry success, retry exhaustion, degraded mode, per-tool circuit breaker, tool health reporting, write queue, recovery)

---

## RM-7: Conversation Memory Injection

**Problem:** When user asks about a topic, agent calls `recall()` reactively. It doesn't proactively inject relevant context into the conversation.

**Proposed:**

- **Pre-query context injection** in `agent.py`:
  - Before sending user message to LLM, run `recall_structured(user_message, top_k=3)`
  - If results found with `score > 0.5`:
    - Inject as SystemMessage: "CONTEXT FROM MEMORY: [brief facts]"
    - Agent naturally references them in response
  - If no results: skip injection (zero overhead)

- **Implementation in `call_model()` node:**
  ```python
  # Before LLM call
  if channel != "autonomous":  # Autonomous already has context in prompt
      context = brain.recall_structured(last_user_msg, top_k=3, min_strength=0.3)
      high_relevance = [r for r in context if r["score"] > 0.5]
      if high_relevance:
          inject_context_message(state, high_relevance)
  ```

- **Smart injection:** Only inject if user's message is a question or new topic (not continuation). Detect via simple heuristics: contains "?", first message in >5 min, topic shift.

**Estimated complexity:** ~40 lines in `agent.py`

**Tests:** 5 tests (injection on relevant query, skip on irrelevant, skip on autonomous, topic shift detection, score threshold)

---

## RM-8: Knowledge Dashboard API

**Problem:** Web GUI shows raw records but doesn't visualize knowledge structure — clusters, research projects, health trends, topic coverage.

**Proposed new API endpoints:**

1. **`GET /api/knowledge/topics`** — Topic overview
   ```json
   {
     "topics": [
       {"tag": "health", "record_count": 15, "avg_strength": 0.72, "trend": "stable"},
       {"tag": "cooking", "record_count": 8, "avg_strength": 0.45, "trend": "decaying"},
       {"tag": "work", "record_count": 22, "avg_strength": 0.81, "trend": "growing"}
     ]
   }
   ```

2. **`GET /api/knowledge/research`** — Research projects
   ```json
   {
     "projects": [
       {"id": "...", "topic": "Vitamin D", "status": "complete", "findings": 5, "confidence": 0.78},
       {"id": "...", "topic": "Sleep patterns", "status": "researching", "progress": "3/5 queries"}
     ]
   }
   ```

3. **`GET /api/knowledge/metrics`** — tracked metric timeline
   ```json
   {
     "entries": [
       {"date": "2026-02-10", "type": "event", "content": "Client feedback received", "connected_to": ["release"]},
       {"date": "2026-02-08", "type": "measurement", "content": "Focus minutes: 75"}
     ]
   }
   ```

4. **`GET /api/knowledge/gaps`** — What agent doesn't know but should
   ```json
   {
     "gaps": [
       {"topic": "project contacts", "reason": "User mentioned stakeholders but no contact list is stored"},
       {"topic": "recurring tasks", "reason": "No current reminder list despite workflow focus"}
     ]
   }
   ```

**Estimated complexity:** ~150 lines in `api.py`, ~80 lines frontend

**Tests:** 8 tests (topics endpoint, research endpoint, health timeline, gaps detection, empty states, filtering)

---

## RM-9: Metric/Event Intelligence

**Problem:** Agent is generic. For Tier 1 workflow focus, it should understand metric/event patterns: event tracking, recurring task reminders, tracked metrics trends.

**Proposed new tools (3):**

1. **`track_metric`** — Store a measurable tracked metric data point
   ```
   Args: metric_type ("weight"|"blood_pressure"|"sleep_hours"|"pain_level"|"mood"|custom),
         value (float), unit (str), notes (str, optional)
   Returns: record_id, trend (str: "improving"|"stable"|"worsening"|"insufficient_data")
   ```
   - Stores with tags `["metric", metric_type]`
   - Calculates trend from last 5 entries of same metric_type
   - Auto-schedules follow-up if trend is "worsening"

2. **`metric_summary`** — Generate metric overview
   ```
   Args: period ("week"|"month"|"all")
   Returns: summary text with metrics, events, recurring tasks
   ```
   - Aggregates metric records by type
   - Shows trends per metric
   - Lists active events and recurring tasks
   - Zero-LLM aggregation (pure data processing)

3. **`event_correlate`** — Find potential event-cause connections
   ```
   Args: event (str)
   Returns: potential_causes with confidence
   ```
   - Searches metric/event records around the event date (+/- 3 days)
   - Cross-references with workflow knowledge base (web_search if needed)
   - Returns: "Possible correlation: headaches started after [new recurring task / poor sleep / stress event]"

**Estimated complexity:** ~180 lines in `brain_tools.py`

**Tests:** 10 tests (metric storage, trend calculation, health summary, correlation detection, insufficient data handling, multi-metric tracking)

---

## RM-10: Token Optimization for Autonomous Mode

**Problem:** Autonomous cycles are expensive. Web search alone costs ~500-1000 tokens. Budget burns through quickly.

**Proposed optimizations:**

1. **Query deduplication:**
   - Before `web_search`: check if similar query was run in last 24h
   - `brain.search(query=query, tags=["web-search-cache"], limit=1)`
   - If exists and fresh: return cached result (zero tokens)
   - Store search results with TTL metadata

2. **Recall-first strategy enforcement:**
   - In `_build_decision_prompt()`: "ALWAYS recall before web_search. If recall answers the question, skip web_search."
   - Currently just suggested; make it a hard rule with tool ordering

3. **Summarize-on-store:**
   - Web search results often verbose (2000+ chars)
   - Before storing: compress to key facts (~200 chars)
   - Saves tokens on future recall (smaller records = cheaper context)

4. **Budget-aware tool selection:**
   - Pass `remaining_budget` to agent in tool descriptions
   - Tools annotated with estimated cost: `web_search (~800 tokens)`, `recall (~50 tokens)`
   - Agent can make informed decisions

**Estimated complexity:** ~70 lines across `brain_tools.py` + `autonomy.py`

**Tests:** 6 tests (query dedup, cache hit, cache miss, summarize on store, budget annotation, recall-first enforcement)

---

## Priority Ranking

| ID | Feature | Tier Target | Impact | Complexity | Priority |
|----|---------|-------------|--------|------------|----------|
| RM-1 | Research Orchestrator | Tier 2 | HIGH | High | 1 |
| RM-3 | Autonomous Research Cycles | Tier 2 | HIGH | Medium | 2 |
| RM-10 | Token Optimization | Tier 1-2 | HIGH | Low | 3 |
| RM-6 | Error Recovery & Reliability | Tier 1-2 | HIGH | Medium | 4 |
| RM-9 | Metric Intelligence | Tier 1 | HIGH | Medium | 5 |
| RM-2 | Source Credibility | Tier 2 | MEDIUM | Low | 6 |
| RM-4 | Structured Fact Extraction | Tier 2 | MEDIUM | Medium | 7 |
| RM-7 | Conversation Memory Injection | Tier 1-2 | MEDIUM | Low | 8 |
| RM-5 | Proactive Research Delivery | Tier 2 | MEDIUM | Low | 9 |
| RM-8 | Knowledge Dashboard | Tier 1 | LOW | Medium | 10 |

---

## Implementation Order (Suggested Sprints)

**Sprint 1 (Reliability + Optimization):** RM-10 + RM-6
- Token optimization + error recovery
- Makes autonomous mode cheaper and more reliable BEFORE adding research features
- ~170 lines of code, ~16 tests

**Sprint 2 (Research Core):** RM-1 + RM-3
- Research orchestrator + autonomous research cycles
- Core Tier 2 feature: structured multi-session research
- ~280 lines of code, ~20 tests

**Sprint 3 (Health + Credibility):** RM-9 + RM-2
- Health-specific tools + source credibility
- Tier 1 workflow differentiation + Tier 2 source quality
- ~240 lines of code, ~16 tests

**Sprint 4 (Intelligence):** RM-4 + RM-7 + RM-5
- Fact extraction + memory injection + proactive delivery
- Full research pipeline completion
- ~180 lines of code, ~18 tests

**Sprint 5 (Polish):** RM-8
- Knowledge dashboard
- Visualization layer for all accumulated data
- ~230 lines of code, ~8 tests

---

## Dependencies Between TZ-01 and TZ-02

| Remy Feature | Depends on Aura Feature | Why |
|---|---|---|
| RM-1 Research Orchestrator | AC-1 Multi-hop walk | Research findings connected via graph; deeper walk = better context retrieval |
| RM-3 Autonomous Research | AC-3 Adaptive decay | Research findings should persist longer (high activation = slow decay) |
| RM-4 Fact Extraction | AC-5 Importance scoring | Facts need importance ranking in recall |
| RM-7 Memory Injection | AC-1 Multi-hop walk | Transitive recall for richer context |
| RM-9 Metric Intelligence | AC-6 Multilingual synonyms | Ukrainian workflow metric terms must match English records |
| Background consolidation | AC-4 Native consolidation | Move consolidation logic from Remy to Aura |

**Recommendation:** Start Aura Sprint 1 (AC-1, AC-5, AC-3) in parallel with Remy Sprint 1 (RM-10, RM-6). No cross-dependencies in first sprints.

---

## Version Target

After all 10 tasks: **Remy v2.0.0** — Tier 2 Research Assistant with structured research, health intelligence, source credibility, proactive delivery, and reliable autonomous operation.
