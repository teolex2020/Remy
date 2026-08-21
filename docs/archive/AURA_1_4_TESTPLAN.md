# Aura 1.4.0 Test Plan

## Goal

Validate `aura-memory 1.4.0` in Remy after the recent autonomy/runtime refactor, with focus on:

- semantic search quality
- working-memory hygiene
- autonomy memory usage
- regression safety for the web Memory UI

Current baseline:

- brain was effectively reset and now contains a small post-refactor dataset
- obvious `perf-test` records were removed
- this is a good point to treat the current brain as a fresh validation baseline

## 1. Smoke Checks

### 1.1 Brain path and count

Verify that Remy uses the expected brain path:

- `settings.AURA_BRAIN_PATH`
- current record count in UI and direct runtime should match closely

Expected:

- web Memory page count matches direct `brain.list_records(...)`
- no accidental secondary brain path is active

### 1.2 Tier stats

Verify:

- `IDENTITY`
- `DOMAIN`
- `DECISIONS`
- `WORKING`

Expected:

- `/api/memory/tier-stats` matches direct `brain.tier_stats()`
- no zero-count UI bug unless the tier is truly empty

## 2. Semantic Search Validation

## 2.1 Identity recall

Store or confirm a few identity/domain facts:

- user name
- location
- one stable preference
- one ongoing project fact

Queries:

- exact: `Олександр`
- paraphrase: `де я живу`
- semantic: `моє місто`, `мій населений пункт`, `хто я`

Expected:

- semantic variants retrieve the same underlying record or clearly related records
- recall is not dependent only on exact tag matching

## 2.2 Domain recall

Create 3-5 domain records around one topic, for example:

- AI agent memory
- OpenClaw
- Aura SDK

Queries:

- exact: `agent memory`
- semantic: `memory for ai agents`
- related wording: `cognitive memory layer`, `persistent agent memory`

Expected:

- relevant domain records rank above unrelated working/outcome noise
- recall should still work when wording changes substantially

## 2.3 Failure recall

Use or store at least 2 failure/outcome records:

- browser/login failure
- no viable path

Queries:

- `what failed before`
- `login blocker`
- `research failure`

Expected:

- failure-aware recall surfaces these records early
- autonomy can reuse them to avoid repeating the same dead path

## 2.4 False positive control

Test semantically broad queries:

- `project`
- `important`
- `research`

Expected:

- not every generic working/outcome note floods the result set
- obvious unrelated records stay out of the top results

## 3. Scratchpad / Working Memory Hygiene

## 3.1 Working note write/read

Create 5-10 scratchpad notes during one session.

Expected:

- notes appear in working memory
- scratchpad context inject works

## 3.2 Summarization

Call:

- `scratchpad(action="summarize")`

Expected:

- old raw notes compress into a `scratchpad-summary`
- summary record contains `compression_ratio` in metadata
- key facts survive the summary

## 3.3 Filter working memory

Call:

- `filter_working(query="current research topic")`

Expected:

- relevant working notes remain active
- irrelevant working notes are demoted or excluded from current context
- no important current note disappears unexpectedly

## 3.4 Over-filter regression

Test with one clearly relevant and one borderline note.

Expected:

- relevant note survives
- borderline note behavior is explainable and stable
- `delete_irrelevant=True` should only be used after confidence is established

## 4. Autonomy Memory Behavior

## 4.1 Mission start on fresh memory

Run one small Research Ops mission with `AUTONOMY_V3=True`.

Expected:

- mission creates structured state records
- outcomes and failures are stored once per cycle
- no uncontrolled duplication

## 4.2 Repeated failure avoidance

Force a known failing mission twice.

Expected:

- second run should surface prior failure context
- planner/runtime should choose a different path, wait, or escalate
- not blindly repeat the exact failed attempt

## 4.3 Playbook reuse

Run one successful research mission, then a similar one.

Expected:

- prior playbook or strategy hints appear in context
- second run starts from reused knowledge, not pure zero-state exploration

## 5. Web UI Validation

## 5.1 Memory page

Verify:

- total record count
- tier counters
- list rendering
- search
- tag filters

Expected:

- UI reflects direct runtime state
- no stale counters after reload

## 5.2 Profile / Settings memory-adjacent behavior

Verify profile and people data still save and appear correctly after the new memory layer changes.

Expected:

- identity writes land in memory
- profile views read back correctly

## 6. Metrics to Watch

Key metrics after Aura 1.4.0 rollout:

- `recall_hit_rate`
- `duplicate_store_rate`
- `scratchpad_compression_ratio`
- `working_memory_total`
- `working_memory_active`
- `working_memory_bloat`

Interpretation:

- `recall_hit_rate` should trend upward with semantic search
- `duplicate_store_rate` should not increase sharply
- `working_memory_bloat` should stay bounded if summarize/filter behavior is healthy

## 7. Pass Criteria

Aura 1.4.0 rollout is acceptable if:

- semantic paraphrase recall works better than exact-only matching
- identity/domain recall remains stable
- failure recall is useful, not noisy
- working-memory summarization does not lose key facts
- autonomy uses prior memory to avoid obvious repetition
- Memory UI reflects real counts and tiers

## 8. Immediate Recommended Sequence

1. Seed 5-10 clean identity/domain records.
2. Run semantic recall checks manually.
3. Run scratchpad summarize/filter checks.
4. Run one small `AUTONOMY_V3=True` research mission.
5. Inspect memory growth and UI counters.
6. Only then start rebuilding richer long-term memory.
