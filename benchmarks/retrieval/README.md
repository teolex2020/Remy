# Search quality benchmarks

Remy has seven complementary retrieval suites:

- `benchmark_v1.yaml` + `run_benchmark.py` is the legacy live smoke-test. It
  checks provider behaviour, safety boundaries, and conflict handling against
  the changing public web.
- `benchmark_v2.yaml` + `run_quality_benchmark.py` is the reproducible ranking
  contract. Fixed provider snapshots pass through the production
  `SearchGateway`, so CI can detect a real code regression without confusing it
  with a search-engine or network change.
- `claim_matrix_v1.yaml` + `run_claim_matrix_benchmark.py` verifies that factual
  claims are bound to the right evidence. It catches missing citations,
  identity mismatches, unsupported claims, conflicts, and weak corroboration.
- `query_planner_v1.yaml` + `run_query_planner_benchmark.py` verifies that a
  research task becomes a budgeted mix of primary, corroboration,
  counterevidence, freshness, domain-scoped, and repair queries.
- `claim_lifecycle_v1.yaml` + `run_claim_lifecycle_benchmark.py` verifies that
  mutable facts persist across runs, safe changes advance the current value,
  unsafe candidates remain pending, and prior states stay auditable.
- `execution_scheduler_v1.yaml` + `run_execution_scheduler_benchmark.py`
  verifies that planned lanes were actually searched, readable pages were
  fetched, and evidence comes from enough independent publishers.
- `marginal_evidence_v1.yaml` + `run_marginal_evidence_benchmark.py` verifies
  pre-fetch duplicate suppression and post-fetch marginal-value decisions.
- `provenance_graph_v1.yaml` + `run_provenance_graph_benchmark.py` verifies
  authority roles, derivation links, independent evidence roots, and rejection
  of false cross-domain consensus built from copies or syndicated material.

## Run the deterministic quality gate

```powershell
.venv\Scripts\python.exe benchmarks\retrieval\run_quality_benchmark.py
.venv\Scripts\python.exe benchmarks\retrieval\run_claim_matrix_benchmark.py
.venv\Scripts\python.exe benchmarks\retrieval\run_claim_lifecycle_benchmark.py
.venv\Scripts\python.exe benchmarks\retrieval\run_query_planner_benchmark.py
.venv\Scripts\python.exe benchmarks\retrieval\run_execution_scheduler_benchmark.py
.venv\Scripts\python.exe benchmarks\retrieval\run_marginal_evidence_benchmark.py
.venv\Scripts\python.exe benchmarks\retrieval\run_provenance_graph_benchmark.py
```

The command compares the current report with
`search_quality_baseline_v2.json` and exits non-zero when a metric crosses its
allowed tolerance or a scenario fails its minimum contract.

Useful variants:

```powershell
# Inspect one fixed case
.venv\Scripts\python.exe benchmarks\retrieval\run_quality_benchmark.py --case q08_live_freshness_forces_web

# Save an internet health report; live proxy results never replace the CI baseline
.venv\Scripts\python.exe benchmarks\retrieval\run_quality_benchmark.py --live --out live_search_report.json

# Measure the local multilingual reranker's contribution against the same snapshots
.venv\Scripts\python.exe benchmarks\retrieval\run_quality_benchmark.py --compare-reranker

# Intentionally accept a reviewed ranking change as the new baseline
.venv\Scripts\python.exe benchmarks\retrieval\run_quality_benchmark.py --write-baseline

# Inspect one claim-source binding case
.venv\Scripts\python.exe benchmarks\retrieval\run_claim_matrix_benchmark.py --case m04_identity_mismatch

# Accept a reviewed matrix change as the new baseline
.venv\Scripts\python.exe benchmarks\retrieval\run_claim_matrix_benchmark.py --write-baseline

# Inspect one planner scenario
.venv\Scripts\python.exe benchmarks\retrieval\run_query_planner_benchmark.py --case p08_conflict_repair_priority

# Accept a reviewed planner change as the new baseline
.venv\Scripts\python.exe benchmarks\retrieval\run_query_planner_benchmark.py --write-baseline

# Inspect one execution failure/recovery scenario
.venv\Scripts\python.exe benchmarks\retrieval\run_execution_scheduler_benchmark.py --case s04_same_domain

# Accept a reviewed scheduler change as the new baseline
.venv\Scripts\python.exe benchmarks\retrieval\run_execution_scheduler_benchmark.py --write-baseline

# Inspect one marginal-value scenario
.venv\Scripts\python.exe benchmarks\retrieval\run_marginal_evidence_benchmark.py --case g02_cross_domain_content_copies

# Accept a reviewed evidence-gain change as the new baseline
.venv\Scripts\python.exe benchmarks\retrieval\run_marginal_evidence_benchmark.py --write-baseline

# Inspect one false-consensus scenario
.venv\Scripts\python.exe benchmarks\retrieval\run_provenance_graph_benchmark.py --case p02_explicit_syndication

# Accept a reviewed provenance change as the new baseline
.venv\Scripts\python.exe benchmarks\retrieval\run_provenance_graph_benchmark.py --write-baseline
```

## Metrics

- `Recall@10`: how many known relevant results survived filtering and ranking.
- `nDCG@10`: whether highly relevant results appear before weaker evidence.
- `MRR`: how early the first relevant result appears.
- target-domain hit rate: whether an expected primary/official source appears.
- independent-domain rate: whether evidence is distributed across enough
  independent hosts.
- duplicate rate and suppression rate: whether duplicates leak to users and
  whether provider overlap is fused.
- local-cache hit and external-search rates: visibility into local-first routing.
- p50/p95 latency: reported for diagnostics, but not used as a deterministic
  snapshot gate because local timing varies by machine.
- claim status accuracy: whether each claim is correctly classified as
  `supported`, `partial`, `unsupported`, or `conflict`.
- support URL precision/recall: whether a claim points to the exact evidence
  expected by the fixture rather than merely any citation.
- false-support rate: the safety metric that catches unsupported claims being
  presented as verified.
- corroboration and publication readiness: whether important claims have enough
  independent evidence and no unresolved conflict.

The v2 corpus currently contains 18 high-signal cases across official docs,
research, Ukrainian and English queries, search operators, SEO noise,
deduplication, freshness, local-first routing, typo correction, source
diversity, no-result behaviour, and resilience. Add new cases whenever a real
failure is discovered; the benchmark is intended to grow into the 100–200 case
product corpus rather than remain a one-time score.

## Local multilingual reranking

`LocalMultilingualReranker` is enabled by default. It performs a second ranking
stage after provider fusion using Unicode tokenization, query-local IDF weights,
field-aware coverage, ordered-term proximity, conservative fuzzy typo matching,
and Cyrillic word-form matching. It has no account, API key, model download, or
new runtime dependency. Every candidate gets an explainable `local_rerank`
object with exact matches, fuzzy matches, missing terms, and component scores.

Set `LOCAL_SEARCH_RERANKER_ENABLED=False` to disable it for diagnosis or A/B
testing. The benchmark's `--compare-reranker` mode does this automatically and
reports overall and hard multilingual deltas.

## Claim-source matrix

The matrix is generated locally after evidence collection. Explicit citations
are checked against fetched content, implicit bindings use the same explainable
multilingual reranker, and an identity mismatch can never count as support. A
URL without extracted material is only `partial`; a citation by itself is not
proof.

Research sessions persist the matrix and expose it in Trajectory. The report
artifact includes per-claim evidence, conflicts, coverage, corroboration, and
repair queries. `publication_ready` is true only when every tracked claim is
supported, no contradiction remains, and no cross-domain corroboration collapses
to a shared evidence root. Each supporting relation includes its authority role,
authority score, and provenance root. A single direct source can support a
claim, but derivative articles cannot masquerade as independent corroboration.

Time-sensitive claims receive an additional local temporal gate. Current
prices, latest versions, live status, and similar claims use the existing
volatility windows (`high=7 days`, `medium=90 days`). Structured publication or
modification dates are carried from the fetch result into every supporting
relation. Stale, future-dated, or undated evidence cannot make a current claim
publication-ready and produces a dated-source repair query. Explicitly
historical claims such as `As of 2024` remain valid historical assertions rather
than being incorrectly treated as expired current facts.

Mutable facts can be resolved by temporal supersession, but only through a
deterministic safety gate. Remy requires the same stable subject, a changed
mutable value (for example price, version, status, availability, or office
holder), dates on both sources, a fresh newer source, independent evidence
roots, and no material authority downgrade. The older claim is marked
`superseded` and retained with its source, effective date, and resolution
history. Immutable scientific disagreements never auto-resolve merely because
one paper is newer. Trajectory reports resolved supersessions separately from
active contradictions and exposes the old/new evidence chain for inspection.

Across separate research runs, mutable supported claims enter a project-scoped
claim lifecycle ledger. Repeated values increase the observation count without
duplicating history. A changed value becomes current only when it is newer,
fresh, independently rooted, and not materially weaker in authority; otherwise
it remains a visible pending candidate. The durable ledger keeps up to 100
events per subject, while Trajectory receives only the 50 most recently observed
subjects with bounded history windows so long-lived projects stay responsive.

## Evidence-aware query planning

`ResearchQueryPlanner` is deterministic and local. In balanced mode it reserves
three complementary lanes: a primary or official source, independent
corroboration, and counterevidence. Deep mode can also add freshness and broader
scope-specific lanes; speed mode stays within a two-query budget. Explicit
operator queries are preserved, domain restrictions are applied first, and
claim-matrix repair queries receive the highest priority on the next cycle.

Every planned query includes a stable ID, intent, rationale, expected source
types, and origin. The structured plan is stored in worker evidence and, unlike
the legacy query list, is injected into the actual worker instruction.

## Research execution scheduler

`ResearchExecutionScheduler` reconciles the structured plan with the actual
worker tool log. A lane counts as executed only after `web_search`; it counts as
fetched only when a corresponding fetch returns at least 120 readable
characters. Citation-bound research requires every scheduled lane, at least
three readable sources, and at least three independent domains before it can be
marked complete.

Incomplete schedules produce deterministic next actions and repair queries:
execute a missing lane, fetch a discovered page, retry unreadable content, or
diversify publishers. Those queries are persisted into the next research cycle.

Citation-bound runs also get one bounded same-run recovery pass. It executes at
most four focused actions with at most four extra tool steps and a 30-second
timeout, then reconciles the combined tool log again. Timeout, error,
cancellation, sufficient evidence, or an exhausted retry limit never triggers
another pass. Set `same_run_recovery: false` on a research goal to opt out.

## Marginal evidence control

Before a recovery fetch, Remy builds a local allow-list using canonical URLs,
query relevance, content/snippet similarity, publisher diversity, and a
one-page-per-publisher default. Duplicate URLs, near-identical candidates,
irrelevant pages, and excess pages from one publisher are suppressed before
another extraction is attempted.

After fetch, each readable page receives an explainable marginal-gain score
from relevance, novelty, content quality, and independent-domain value. Only
high-gain evidence counts toward completion. Two consecutive duplicate or
low-gain pages mark the same-run search as saturated, preventing another costly
retry while preserving a diversification query for the next durable cycle.

## Source authority and provenance

Every readable source becomes a node in a local provenance graph. Remy assigns
an explainable authority role (`primary`, `secondary`, `derived`, `syndicated`,
`aggregator`, or `ugc`), records citations and explicit origin links, and uses
near-duplicate content to infer copied material. Identity mismatch evidence
reduces authority instead of silently inheriting trust from a URL.

Completion now requires three accepted provenance roots, not merely three URLs
or domains. Three publishers repeating one original report therefore trigger a
focused diversification query and remain incomplete. The graph and its counts
are stored in worker evidence and emitted as a `VERIFICATION` Trajectory event.
