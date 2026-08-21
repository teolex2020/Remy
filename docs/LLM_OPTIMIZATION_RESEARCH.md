# LLM Optimization Research Notes

Date: 2026-07-01

This document captures the current state of the Remy context-window optimization research so the work can be resumed without relying on chat history.

## Goal

Evaluate whether Remy can reduce LLM prompt/context usage without losing answer correctness.

The product hypothesis:

- long-running agent sessions waste tokens by resending stale transcript;
- a compact session state can preserve important facts while reducing prompt size;
- this may become a separate API-layer product only if correctness and savings are repeatable.

## Current Implementation

Added research/eval infrastructure:

- `src/remy/core/session_state_wrapper.py`
  - deterministic projected session state;
  - exact pinned fact extraction;
  - evidence lines so exact values keep their meaning;
  - incremental state update support;
  - raw vs optimized prompt builders;
  - local metrics logging.

- `data/evals/llm_optimization/cases.jsonl`
  - synthetic eval corpus;
  - covers exact facts, support, health notes, CRM, coding sessions, RAG-like context, mixed language, and short-context negative cases.
  - current size: 30 cases, including 9+ reasoning/decision/preference cases and 3+ negative short-context cases.

- `data/evals/llm_optimization/README.md`
  - corpus schema and run instructions.

- `tools/validation/llm_optimization_corpus_eval.py`
  - repeatable eval runner;
  - supports `raw`, `projected_facts`, `projected_hybrid`, `incremental`;
  - keeps `projected` as a backward-compatible alias for old reports/runs;
  - supports dry-run and real Gemini runs;
  - supports long-session simulation via `--append-noise-turns`;
  - writes JSON reports to `data/llm_optimization/`.

- `tools/validation/llm_optimization_matrix.py`
  - runs a model/noise eval matrix;
  - writes individual `corpus_eval_*.json` reports plus one `matrix_eval_*.json`;
  - gates every run using `llm_optimization_gate.py`;
  - exits non-zero if any matrix run fails the configured quality/commercial
    thresholds.

- `tools/validation/llm_optimization_locomo.py`
  - runs the external LoCoMo conversational-memory benchmark;
  - supports `raw`, `projected_facts`, `projected_hybrid`, and
    `evidence_oracle`;
  - keeps session timestamps in every dialogue line so relative dates
    (`yesterday`, `last week`, `next month`) can be resolved;
  - writes `locomo_eval_*.json` reports.

- `tests/test_llm_optimization_corpus_eval.py`
  - regression tests for corpus loading, schema validation, context savings, effectiveness metrics, and incremental setup cost.

- `tests/test_session_state_wrapper.py`
  - regression tests for fact extraction, state persistence, projected state, prompt building, and eval correctness.

## Current Product Metrics

The runner now reports explicit product-oriented metrics:

- `accuracy`
  - fraction of cases answered correctly.

- `context_window_saved_tokens_estimate`
  - estimated prompt/context tokens saved versus raw transcript.

- `context_window_saved_pct_vs_raw_estimate`
  - percentage of context-window usage saved versus raw transcript.

- `context_window_freed_pct_points_estimate`
  - absolute percentage-point reduction relative to configured context-window size.

- `optimization_effective`
  - true only when the answer is correct and context tokens are actually saved.

- `effective_cases`
  - number of cases where optimization was genuinely effective.

- `effectiveness_rate`
  - effective cases divided by total cases.

- `accuracy_weighted_context_saving_pct`
  - savings adjusted by correctness; this is the most useful product score.

- `mean_efficiency_score_estimate`
  - average per-case effective saving score.

- `break_even_request_count_estimate`
  - first final-answer request where optimized cumulative cost wins.

## Latest Local Test Status

Last focused regression run:

```text
python -m pytest tests\test_llm_optimization_corpus_eval.py tests\test_session_state_wrapper.py

28 passed
```

This means the research infrastructure is stable enough to continue experiments, but it is not production proof yet.

After adding the quality-gate test suite:

```text
python -m pytest tests\test_llm_optimization_gate.py tests\test_llm_optimization_corpus_eval.py tests\test_session_state_wrapper.py

39 passed
```

## Observed Results

### Short Sessions

Projected optimization is not useful on very short sessions.

Observed pattern:

```text
0 added noise exchanges: projected is worse than raw
1 added noise exchange: projected is still usually worse
```

Reason:

- projected state has fixed prompt overhead;
- raw history is still small;
- optimization can cost more than it saves.

### Medium Sessions

With simulated extra conversation history:

```text
3 added noise exchanges:
accuracy: 1.0
effectiveness_rate: 1.0
context_window_saved_pct_vs_raw_estimate: about 17-24%
```

Representative latest dry-run summary:

```text
projected:
accuracy: 1.0
cases: 30
effective_cases: 29
effectiveness_rate: 0.9667
context_window_saved_tokens_estimate: 849
context_window_saved_pct_vs_raw_estimate: 22.36
accuracy_weighted_context_saving_pct: 22.36
mean_efficiency_score_estimate: 21.73
```

### Longer Sessions

Earlier dry-run observations:

```text
5 added noise exchanges: about 35% context saving
10 added noise exchanges: about 58% context saving
40 added noise exchanges: about 86% context saving
```

Conclusion:

- the approach becomes more valuable as transcript length grows;
- the product is potentially useful for long-running agent sessions, support threads, CRM workflows, coding agents, and long medical/admin note sessions;
- it is not useful for one-shot or very short chats.

### Real Model Run

One real provider run on `gemini-flash-lite-latest` at `--append-noise-turns 3`:

```text
raw:
accuracy: 1.0
provider_prompt_tokens: 1801

projected:
accuracy: 1.0
provider_prompt_tokens: 1670
```

This confirmed a small real provider-token saving at the medium-session point.

Smoke run on stronger model `gemini-3.5-flash`, limited to 10 cases at
`--append-noise-turns 3`:

```text
raw:
accuracy: 1.0
provider_prompt_tokens: 2278

projected:
accuracy: 1.0
effectiveness_rate: 1.0
context_window_saved_pct_vs_raw_estimate: 18.48
provider_prompt_tokens: 2383
provider_calls: 13
```

Interpretation:

- the model is available and passes the smoke eval;
- final projected prompt is smaller by context-window estimate;
- total provider prompt tokens are higher in this short smoke because projected
  also paid for 3 decision-extraction calls;
- this confirms that cost savings and context-window savings must be tracked as
  separate product metrics.

Quality-gate result for this same `gemini-3.5-flash` smoke:

```text
accuracy 1.000 OK
effectiveness_rate 1.000 OK
no false savings
FAIL - context savings 18.48% < 20.0% (long session, noise=3)
RESULT: FAIL
```

Interpretation:

- the gate is working, not rubber-stamping results;
- the stronger-model smoke is correct but does not meet the current 20% context
  savings threshold at this small case limit;
- this should be rerun on larger/longer sessions before treating it as a model
  regression failure.

After adding provider total-token metrics, a fresh 5-case `gemini-3.5-flash`
smoke showed the key product distinction:

```text
projected:
accuracy: 1.0
context_window_saved_pct_vs_raw_estimate: 14.61
provider_total_saved_pct_vs_raw: -2.83
provider_calls: 6 vs raw 5
```

Interpretation:

- final projected prompt used less context window;
- total provider-token cost was worse because decision extraction added a call;
- context-window savings and API-cost savings must stay separate in the gate.

### Real history validation (not synthetic noise)

Until here, all long-session savings used `--append-noise-turns` — synthetic
filler. The open risk was that real, *relevant* conversation history compresses
worse than pure noise. Ran facts-mode over the app's own stored sessions
(`data/history/*.json`, 140 sessions, 29 with >= 6 user turns — real agentic
threads: arXiv search, tool loops, multi-turn Q&A).

```text
REAL HISTORY, facts-mode, dry token estimate:
  min = 68.3%   median = 78.4%   max = 90.6%   mean = 78.9%
```

Synthetic +40 noise gave ~83%; real relevant history gives ~78% median. Almost
the same — real agentic transcripts carry enough repeated context (tool output,
restated context, follow-ups) that the state snapshot folds them nearly as well
as pure noise.

This moves "useful for agents" from hypothesis to fact on real data: the app's
own long agent sessions (18 user turns + 46 tool calls, etc.) compress 68-90%.

Open (not yet done on real history): provider-token confirmation of the estimate
on these specific sessions; answer-correctness check (real sessions have no
expected_fragments, so a different correctness proxy is needed); hybrid mode for
any reasoning questions inside real sessions.

### Correctness on real sessions: why fact-comparison fails (methodology)

Tried to measure correctness on real sessions by running the same question with
raw history vs projected, and comparing. This is methodologically broken for a
generative model: the same prompt gives different wordings, so answer difference
does not prove information loss — it may just be sampling noise.

Attempted fix: compare extracted FACTS (regex codes/dates/ids), which are
invariant to wording, and use raw as the ground-truth oracle. Also measured a
raw-vs-raw baseline (the generative noise floor).

Result on real sessions (`data/history`, flash-lite):

```text
session A: raw-vs-raw = 1.00 | raw-vs-projected = 0.50
session B: raw-vs-raw = 0.00 | raw-vs-projected = 0.00
```

`raw-vs-raw = 0.00` is the key finding: on real sessions the ORACLE itself is
unstable — raw answers the same question with different facts twice. Fact
comparison cannot measure compression correctness when the reference is not
reproducible.

Root cause — the real sessions are the wrong shape for fact tests. Inspection
shows they are conversational/consulting threads ("here is my project, research
it", "focus on monetization", "make a PDF"), not fact-retrieval. There is no
single correct fact to compare; answers are advice/prose/artifacts.

Consequences:
- economy is proven on real data (78% median) — that stands;
- correctness via fact-match is impossible on reasoning/consulting sessions
  (no invariant answer, unstable oracle);
- the real traffic here is reasoning-heavy, so `projected_facts` would lose the
  substance; it needs `projected_hybrid`;
- measuring reasoning-compression correctness needs a different judge than fact
  comparison. Options: public benchmark with labeled answers (in progress),
  LLM-as-judge on usefulness (raw vs projected — "did projected keep everything
  the user needs?"), or targeted fact-questions injected into real sessions.

### LoCoMo external benchmark — the decisive negative (CRITICAL)

All prior correctness numbers used OUR OWN synthetic corpus, whose cases were
written around our mechanism (regex facts, explicit decisions). To get an
independent oracle we ran LoCoMo (`snap-research/locomo`, the field-standard
conversational long-term-memory benchmark: real multi-session dialogues, short
labeled answers, evidence pointers).

One 419-turn conversation, 15 non-adversarial QA, `gemini-flash-lite-latest`:

```text
raw     : 8/15 = 0.53
facts   : 2/15 = 0.13   context saved 99%
hybrid  : 2/15 = 0.13   context saved 94%
```

Both projected modes retain only ~25% of raw's correct answers. The 94-99%
"savings" are meaningless because the state dropped the needed information.

Root cause — architectural, not tuning:
- LoCoMo asks about EPISODIC memory: when did X happen, how many years, where
  does Y live, what date. That is who-did-what-when across the dialogue.
- our extractors capture regex facts (codes/dates in strict format) and
  decisions/preferences — NEITHER captures episodic events. LoCoMo dates are
  prose ("the sunday before 25 May"), not regex-matchable; the questions are not
  about decisions.
- so both modes are structurally blind to the memory type LoCoMo tests.

Honest consequences:
- our synthetic 30/30 was self-confirming: we wrote cases our mechanism could
  answer. An independent benchmark exposes 0.13.
- as a general "conversation memory" product the current approach does NOT work
  — proven on an external oracle, not speculated.
- the valid niche narrows hard: exact fact-retrieval where the answer IS a
  regex-extractable value (id/code/amount). Not "remember our conversation and
  answer about events".
- against the proxy end-goal (ship only if profitable AND stable): it is NOT
  stable for conversational memory, which is the broadest market. Stable only
  for narrow fact-retrieval.

This is the most important falsification in the whole effort. Without an
external labeled benchmark, "30/30, 99% savings" would have failed on a
customer's first real test. The open question is whether an episodic-event
extractor (a third strategy beyond facts/decisions) can recover LoCoMo accuracy
while keeping meaningful savings — untested.

### LoCoMo external runner added (2026-07-01)

The one-off LoCoMo script was replaced with a versioned runner:

```powershell
python tools\validation\llm_optimization_locomo.py --model gemini-flash-lite-latest --mode raw,projected_hybrid,evidence_oracle --qa-limit 10 --delay-sec 0.2 --min-accuracy 0
```

Modes:

```text
raw               full LoCoMo conversation
projected_facts   deterministic fact projection
projected_hybrid  fact + durable extraction projection
evidence_oracle   only LoCoMo evidence lines from the dataset
```

The adapter initially missed `session_N_date_time`, which made relative-date
questions unfairly hard. After prefixing each dialogue line with its session
timestamp, raw accuracy improved sharply:

```text
LoCoMo sample 0, first 10 non-adversarial QA, gemini-flash-lite-latest

raw:
accuracy: 0.80
provider_total_tokens: 232014

projected_hybrid:
accuracy: 0.20
retained_vs_raw_correct: 0.25
context_window_saved_pct_vs_raw_estimate: 91.14
provider_total_saved_pct_vs_raw: 70.20

evidence_oracle:
accuracy: 0.30
provider_total_saved_pct_vs_raw: 99.09
```

Important caveat: the current matcher is strict substring/token overlap, so it
undercounts some semantically correct date answers. Example: LoCoMo gold says
`The sunday before 25 May 2023`, while the model may answer `20 May 2023`.
That is likely correct but currently fails the strict checker. The raw vs
projected gap is still too large to dismiss: raw gets 8/10, projected gets 2/10.

Updated conclusion:

- LoCoMo confirms the current compressor is not a general conversational memory
  system.
- `projected_hybrid` can preserve explicit durable facts/rules/preferences, but
  it does not yet preserve broad episodic memory.
- The next product-relevant research direction is an `episodic_events` strategy:
  write compact event records with speaker, action, object, time anchor, and
  source dialogue id, then retrieve those records for questions.

### Code-agent memory smoke (2026-07-01)

Because LoCoMo tests broad episodic human conversation, a separate check was run
on code-agent style memory: failing tests, file paths, release commands, and
patch boundaries. This is closer to Remy's agent-wrapper use case.

Current corpus has only 3 explicit `coding_agent_session` cases, so this is a
smoke test, not benchmark proof.

```text
cases:
  - coding_failure_path
  - coding_patch_boundary
  - coding_release_command
noise: 40

gemini-flash-lite-latest:
raw accuracy: 1.0
projected_facts accuracy: 0.6667
projected_hybrid accuracy: 1.0
projected_hybrid provider_total_saved_pct_vs_raw: 75.44
commercial gate: PASS

gemini-3.5-flash:
raw accuracy: 1.0
projected_facts accuracy: 0.6667
projected_hybrid accuracy: 1.0
projected_hybrid provider_total_saved_pct_vs_raw: 73.69
commercial gate: PASS
```

Interpretation:

- Code-agent memory is a more promising niche than broad human conversational
  memory. It contains exact artifacts (paths, tests, commands, tags) and durable
  constraints (scope, no-refactor rules) that `projected_hybrid` can preserve.
- `projected_facts` is still too narrow: it keeps exact paths/tags but drops
  non-regex scope rules such as "do not refactor the scheduler".
- This does not prove a product yet because 3 synthetic code cases are too few.
  The next external code-oriented candidates are:
  - RepoBench-R / RepoBench-P for repository-level retrieval/completion context;
  - LongBench-v2's code/repository subset as a long-context complement;
  - SWE-bench later for end-to-end agent editing, but it is not the first choice
    for isolating memory-compression quality because success depends on editing,
    test execution, environment setup, and planning, not only retained context.

### "Compress vs retrieve" — the prior project's MinHash mechanism

The user asked directly: are we building a binary imprint, or just cleaning the
prompt? Answer: what we built is text compression (regex + LLM restate). It is
NOT the prior project's mechanism. `AuraSDK-verify` uses
`crates/substrate/src/ngram.rs` — byte-trigram MinHash + LSH: each turn is signed
into a MinHash signature and LSH-bucketed; a query is signed the same way and the
most-overlapping turns are RETRIEVED. It keeps everything and fetches per query,
rather than deciding up front what to drop.

Reproduced MinHash+LSH faithfully in Python (64 hashes, byte-trigram, xxh3),
indexed all 419 turns, retrieved top-12 per question, same LoCoMo QA:

```text
raw (all history)          : 8/15 = 0.53
text compression (ours)    : 2/15 = 0.13
MinHash retrieval (prior)  : 3/15 = 0.20
evidence_oracle (perfect)  :        0.30   (from the versioned runner)
```

Checked whether MinHash even fetches the labeled evidence turn: only ~42% of the
time (5/12). Root cause: byte-trigram MinHash is a LEXICAL match, not semantic.
"What is Caroline's identity?" never shares words with "I'm a transgender
woman", so it misses. LoCoMo paraphrases, so lexical retrieval has a ceiling.

Key insight from combining with `evidence_oracle`: even PERFECT retrieval (feed
only the exact evidence lines) scores just 0.30 under the strict checker. So two
separate ceilings are in play:
1. retrieval quality — MinHash is lexical, misses paraphrase (~42% evidence hit);
2. the strict substring checker — undercounts semantically-correct answers
   (`20 May 2023` vs gold `the sunday before 25 May 2023`).

Consequences:
- the user's instinct "imprint beats compression" is confirmed (0.20 > 0.13);
- but byte-trigram MinHash alone is lexical → below raw; the cheap no-embeddings
  edge carries a real accuracy cost;
- to clear LoCoMo you need semantic retrieval (embeddings) or the `episodic_events`
  strategy above, AND a semantic answer checker (LLM-judge) to stop the strict
  matcher from hiding correct answers.

### Code domain — hypothesis "structured identifiers survive better" REFUTED

Hypothesis: code is more structured (identifiers, paths), so lexical MinHash /
compression should do better on code than on paraphrase-heavy conversation.
Tested on RepoBench v1.1 (`tianyang/repobench_python_v1.1`, cross_file_first),
12 long examples (>4000 tokens, 6+ cross-file snippets each). Cross-file context
= "history", `cropped_code` = query, `next_line` = ground truth. MinHash
retrieves top-4 relevant snippets vs full context.

```text
raw (all cross-file context) : 5/12 = 0.42
MinHash retrieval            : 2/12 = 0.17   saved 30%
```

Same collapse as conversation (LoCoMo: 0.53 → 0.20). Code did NOT rescue the
lexical approach. Why: code completion needs the snippet that DEFINES a symbol
the next line introduces (`PILtoTorch`, `ConcatDataset`) — a symbol NOT yet in
the query. MinHash matches "similar to what's already there", but the needed
info is "what will be needed next" — the same semantic gap as paraphrase.

### Two-benchmark verdict (conversation + code)

| domain | raw | retrieval/compress | ceiling cause |
|--------|-----|--------------------|---------------|
| conversation (LoCoMo) | 0.53 | 0.20 | paraphrase, no shared words |
| code (RepoBench)      | 0.42 | 0.17 | needed symbol not in query |

The ceiling is in the MECHANISM, not the domain. byte-trigram MinHash is lexical
and text compression drops info up front; both lose ~half of raw's accuracy on
two independent field benchmarks. The "cheap, no-embeddings" edge costs about
half the accuracy, proven twice.

Surviving niche (passed both falsifications): only where the QUESTION already
contains the exact answer token (retrieve an already-mentioned code/id/amount).
That is narrow fact-retrieval, not "recall/predict what will be needed". For
general QA over long context — conversation or code — this approach does not
replace full context, and closing the gap requires semantic retrieval
(embeddings) or a structured `episodic_events` index plus an LLM-judge checker.

### RECODE-TO-NOTATION — the first approach that keeps quality (BREAKTHROUGH)

Reframing that broke the deadlock: the LLM's native "language" is vectors, not
English — English is just the input codec. Compression and retrieval both failed
because they operate on the wrong layer (bytes / dropped text). Insight: don't
drop information and don't search — RECODE the context into a dense, regular
NOTATION that tokenizes cheaper and whose explicit structure the LLM reads
without losing links. The LLM then supplies the semantics from its own vectors
for free.

Notation form (dense, regular):

```text
Caroline| identity=transgender_woman; goal=continue_education+careers
Caroline.event| attended=LGBTQ_support_group @6May2023; reaction=inspired
Melanie.event| painted=sunrise @2022
```

Hand-encoded proof (LoCoMo session 1, 4 QA): raw 3/4, notation 4/4, 68% fewer
tokens — no quality loss.

Auto-encoded (one LLM call recodes a whole conversation, amortized over many
questions), LoCoMo, 2 conversations, 20 QA. First measured with a strict
substring checker (misleading), then with an LLM judge (fair — handles date
equivalence like "20 May" == "the sunday before 25 May"):

```text
substring checker : raw 0.80  notation 0.40   (checker lied — see below)
LLM judge (fair)  : raw 0.75  notation 0.70   88% token saving/query
```

Diagnosis of the substring gap: 9/10 answer facts WERE present in the notation;
the strict checker just failed date-equivalent phrasings. The auto-encoder
preserved the facts; the notation was not lossy. Fair judge → notation retains
93% of raw accuracy at 88% token saving.

Why this works where compress/retrieve failed:

| approach | retention vs raw | saving |
|----------|------------------|--------|
| text compression | ~25% (0.13) | 90% |
| MinHash retrieval | ~38% (0.20) | 90% |
| **recode-to-notation** | **93% (0.70)** | **88%** |

It neither drops nor searches — it restates all facts in ~8x fewer tokens; the
LLM reads the explicit structure losslessly and adds meaning from its own
vectors. This is the first approach that keeps quality while saving tokens.

Open (not yet proven — needs hundreds of tests):
- only 20 QA / 2 conversations so far — signal, not product;
- the LLM judge is itself generative and may be lenient — needs manual spot-check;
- encoding costs ~3000 tokens once per conversation — need the break-even
  question count where amortized encoding beats raw;
- the encoder is an LLM call — still a per-conversation model call, though
  amortized (not per-turn). A rule-based encoder (grammar patterns → notation)
  is the cheaper ideal but untested.

### LoCoMo notation runner added (2026-07-01)

The temporary notation experiment is now a versioned runner:

```powershell
python tools\validation\llm_optimization_locomo_notation.py --conversation-limit 1 --qa-limit-per-conversation 3 --judge llm --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0 --max-accuracy-drop 1 --min-product-saving-pct -100
```

What it measures separately:

- raw answer quality from full conversation;
- notation answer quality from one recoded conversation snapshot;
- one-time notation encode cost;
- per-query notation answer cost;
- product cost: `encode once + notation answers` versus raw answers;
- eval-only LLM judge cost, excluded from product economics;
- amortized savings for repeated query multipliers `1x`, `2x`, `3x`, `5x`, `10x`.

Smoke result, LoCoMo conversation 1, first 3 answerable QA,
`gemini-flash-lite-latest`, LLM judge:

```text
raw accuracy:       3/3 = 1.00
notation accuracy:  3/3 = 1.00
retention vs raw:   100%

raw provider total tokens:               66,801
notation answer provider total tokens:    1,936
notation setup provider total tokens:     22,953
notation product total tokens:            24,889

notation query saved vs raw:              97.10%
product saved vs raw including encode:    62.74%
estimated context-window saved:           98.74%

amortized product saving:
1x:  62.74%
2x:  79.92%
3x:  85.65%
5x:  90.23%
10x: 93.67%
```

Interpretation:

- this is the strongest real-token signal so far because it includes one-time
  encode cost and still clears the commercial `>=40%` saving threshold;
- sample size is still too small for a product claim;
- next required run is all 10 LoCoMo conversations with semantic judge, plus
  manual spot-check of judge decisions on a random subset.

### Recode-to-notation production decision: reject as core (2026-07-01)

Follow-up run on the same LoCoMo shape but with 2 conversations / 20 QA showed
that the optimistic smoke was not stable enough:

```text
raw accuracy:       14/20 = 0.70
notation accuracy:   7/20 = 0.35
retention vs raw:    50%

raw provider total tokens:              406,992
notation answer provider total tokens:   16,828
notation setup provider total tokens:    42,498
notation product total tokens:           59,326
product saving including encode:         85.42%
```

The economics are excellent, but the product fails because the recoder drops or
normalizes details that later questions need. Examples observed:

- `researched adoption agencies` was lost while `applied adoption agencies`
  remained;
- `transgender woman` became too vague or inaccessible for the answer step;
- relative dates such as `yesterday` were normalized incorrectly;
- shared attributes such as both people using dance to destress were missed.

Fallback-on-UNKNOWN on the earlier 20-QA report would recover some quality:

```text
fallback UNKNOWN cases:          8/20
fallback accuracy:              13/20 = 0.65
raw accuracy:                   14/20 = 0.70
fallback product saving:        44.47%
```

This is useful as a research artifact, but still not product-core: it adds an
extra model pass, slows responses, depends on another model's context window,
and fails for the same fundamental reason as summarization: a model must read
the whole text and decide what matters before it knows future questions.

Decision:

- `LLM -> compact notation -> LLM` is rejected as the main commercial
  architecture.
- The old binary/ACL direction is also not treated as a ready solution. Its
  useful lesson is architectural: state must be incremental and machine-native,
  not a one-shot rewrite of a large transcript.
- The next viable direction is an incremental **turn codec / typed state
  layer**:
  raw stream -> compact tokenizer-efficient state format -> query planner ->
  minimal context/evidence packet -> target LLM.
- This is not "another model rewrites the chat". The core is a transformation
  into a format that is cheaper for existing tokenizers and still readable by
  current LLMs. A helper model may be tested later only as an optional extractor
  on the latest small chunk, not as the primary architecture.
- No component should need to reread the full chat to update memory.

### Target architecture: incremental state codec, not transcript compression

The viable product shape is a **state layer** between the user and the target
model. More precisely, it is a **turn codec**: every new request/answer is
immediately transformed into a compact format that is cheaper for existing
tokenizers and still understandable for current models. It must transform each
turn as it arrives, rather than rereading and rewriting the full conversation.

Desired flow:

```text
turn 1 user request
  -> encode immediately into compact tokenizer-efficient state format
  -> send minimal packet to target model
  -> receive answer
  -> encode answer / consequences into state

turn 2 user request
  -> send: compact state from turn 1 + last answer + current request
  -> do NOT send the whole raw chat
  -> update state incrementally again
```

So the model repeatedly reads only:

- the current user request;
- the previous answer or very recent exchange;
- the compact accumulated state.

It should not read the whole transcript again. That is where real context-window
saving can appear. The key product requirement is that the accumulated state
format must be cheaper for existing tokenizers than natural chat text while
remaining model-readable and preserving the recoverable facts, decisions,
entities, events, constraints, and temporal anchors needed for future turns.

This differs from rejected notation-recode:

```text
Rejected:
large raw chat -> model rewrites all chat -> target model

Target:
new small turn -> turn codec/state encoder -> compact model-readable state -> target model
```

Implications:

- first request may not save much because the state does not exist yet;
- savings should begin from the second request and grow as raw transcript length
  grows;
- encoding latency must be bounded per new turn, not proportional to total chat
  length;
- the encoder can use deterministic parsing, typed records, binary/indexed
  structures, or a custom compact grammar designed for current tokenizers;
- a helper model is optional research only and, if used, must operate on the
  latest small chunk, not the full chat;
- product metrics must report "saving from request N" and "context-window freed
  over time", not only aggregate compression ratio.

### APPEND-ONLY incremental state — the one approach that keeps 100% quality

The user's exact design, distinct from everything above: compress ONLY the new
exchange into a small frozen chunk, then APPEND it to the state. The old state is
never re-compressed. Each fact passes through compression exactly once, so losses
cannot accumulate over many turns (the flaw of re-summarising the whole state
every step).

Contrast of the two incremental variants (both fold one exchange at a time):

```text
re-compress whole state each step : retention 88%  (old facts re-processed 40x)
append-only (compress new, freeze) : retention 103% (each fact compressed once)
```

Measured on LoCoMo, 60 QA / 6 conversations, LLM judge:

```text
raw          : 39/60 = 0.65
append-only  : 40/60 = 0.67   retention 103%  (no quality loss vs raw)
per-query saving : 47%
```

Same design on CODE (RepoBench cross-file sessions, 30 QA / 6 sessions):

```text
raw          : 30/30 = 1.00
append-only  : 30/30 = 1.00   retention 100%
per-query saving : 27%   (code is already dense — less to compress than prose)
```

Full economics INCLUDING one-time compression overhead (this is the real
commercial number, not per-query saving):

| domain | retention | per-query saving | break-even | net @200 queries |
|--------|-----------|------------------|------------|------------------|
| conversation | 103% | 47% | ~5 queries | ~46% |
| code | 100% | 27% | ~7 queries | ~30% |

Net saving by conversation length (conversation domain):

```text
  5 queries:  -2%   (compression not yet amortised — LOSS)
 10 queries:  22%
 50 queries:  42%
200 queries:  46%   (compression cost negligible; approaches per-query saving)
```

Verdict — this is the first and only approach that survived every falsification:

- QUALITY: 100-103% of raw on two independent domains — genuinely no loss,
  because append-only compresses each fact exactly once and freezes it.
- SAVING is real but bounded: ~46% conversation, ~30% code — NOT the 90% dream,
  because keeping every fact (for quality) means a larger state (less saving).
  Aggressive 90% saving always cost quality; this trades saving for correctness.
- COMMERCIAL only for LONG sessions: break-even ~5-7 queries; loss below that.
  Fits long agent/support/consulting chats where the same history is queried many
  times. Not universal, not for one-shot.
- Code saves LESS than conversation (30% vs 46%): code is already dense, prose
  has more filler to drop. Long code sessions help amortisation, not per-query
  saving.

Bottom line: "45% token saving with zero answer loss on long conversational
sessions" is a real, honest, defensible pitch. The "cheap 90% magic" is not —
proven impossible across the whole day. This is the honest ceiling.

### Append-only runner added (2026-07-01)

The append-only idea now has a versioned runner:

```powershell
python tools\validation\llm_optimization_append_only.py --conversation-limit 6 --session-limit 5 --qa-limit-per-conversation 10 --mode append_full,append_retrieved --target-mode append_full --judge llm --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0 --max-accuracy-drop 0.02 --min-product-saving-pct 20
```

Modes:

```text
append_full       send all frozen compressed chunks
append_retrieved  locally select relevant frozen chunks before sending
```

Metrics reported:

- raw accuracy vs append-only accuracy;
- retained quality vs raw;
- one-time encode/setup cost;
- query cost after encoding;
- full product cost: `setup + optimized queries`;
- break-even query count;
- net savings at 5, 10, 20, 50, 100, and 200 queries.

Commercial improvement levers:

1. **Cheaper turn codec**
   Current experiments used an LLM call to encode each exchange. This is the
   main cost drag. A deterministic/tokenizer-aware codec or typed parser can
   lower setup cost and make the same quality profile profitable earlier.

2. **Frozen-chunk retrieval**
   Append-only quality is strong, but sending all frozen chunks caps savings.
   `append_retrieved` tests a local selector that sends only relevant chunks.
   This is the main path from ~47% per-query savings toward a stronger
   commercial number without reprocessing old state.

3. **Lossless dedup**
   Do not re-encode or resend repeated facts. If `user.name=Oleksandr` already
   exists, later turns should reference or update it, not append another copy.
   This is not summarization; it is record-level normalization.

4. **Length/workload router**
   Append-only should not run for short sessions. It becomes commercial after
   break-even (~5-7 queries in current tests). The product should enable it only
   when the session is likely to continue or when the user opts into persistent
   memory/state.

Next falsification:

- run `append_full` vs `append_retrieved` on the same 60 LoCoMo QA;
- require quality drop <= 2 percentage points;
- compare product saving at 10/50/200 queries;
- if retrieval keeps quality and raises net savings, it becomes the primary
  commercial path.

### Append-only runner result: retrieval is not yet safe (2026-07-02)

Real run:

```powershell
python tools\validation\llm_optimization_append_only.py --conversation-limit 6 --session-limit 5 --qa-limit-per-conversation 10 --mode append_full,append_retrieved --target-mode append_retrieved --judge llm --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0 --max-accuracy-drop 1 --min-product-saving-pct -100
```

Report:

```text
data\llm_optimization\append_only_eval_20260702_000452_1782939892641785300.json
```

Results, 6 LoCoMo conversations / 60 QA:

| mode | raw acc | optimized acc | retention | query saving | product saving | break-even | net @200 |
|------|---------|---------------|-----------|--------------|----------------|------------|----------|
| append_full | 30/60 = 0.50 | 32/60 = 0.533 | 106.7% | 45.89% | 23.92% | 28.72 queries | 39.30% |
| append_retrieved | 30/60 = 0.50 | 25/60 = 0.417 | 83.3% | 90.73% | 68.76% | 14.53 queries | 84.14% |

Interpretation:

- `append_full` is quality-safe in this run: it slightly beats raw and saves
  real product tokens, but only `23.92%` at this 60-QA workload because setup
  cost is high (`77,513` tokens).
- `append_retrieved` is economically excellent but quality-unsafe. It sends only
  ~8.5 chunks on average and saves `68.76%` product tokens, but retention falls
  to `83.3%` of raw. That is not acceptable for a quality-preserving product.

Observed retrieval failures:

- `What did Caroline research?` -> retrieved state missed `adoption agencies`;
- `How do Jon and Gina both like to destress?` -> missed the shared `dance`
  evidence;
- `What martial arts has John done?` -> missed `Kickboxing, Taekwondo`;
- multi-hop / shared-interest / relative-memory questions are especially weak.

Current commercial verdict:

- viable candidate: `append_full` for long conversational sessions, honest
  pitch around `~24%` net at 60 QA and `~39%` at 200 QA with zero quality loss;
- not yet viable: `append_retrieved` with the current lexical selector;
- next improvement should target selector quality, not more compression:
  hybrid selector, entity/time-aware selector, include evidence-neighbor chunks,
  or fallback to `append_full` when retrieval confidence is low.

### Deterministic codec v0 falsification (2026-07-02)

Added `--encoder deterministic` to the append-only runner. This removes LLM
setup cost entirely and tests the user's core commercial hypothesis: if the
turn codec is nearly free, append-only should become much more attractive.

Real run:

```powershell
python tools\validation\llm_optimization_append_only.py --encoder deterministic --conversation-limit 6 --session-limit 5 --qa-limit-per-conversation 10 --mode append_full --target-mode append_full --judge llm --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0 --max-accuracy-drop 1 --min-product-saving-pct -100
```

Report:

```text
data\llm_optimization\append_only_eval_20260702_001840_1782940720344402600.json
```

Result:

```text
raw accuracy:        30/60 = 0.50
optimized accuracy:  32/60 = 0.533
retention:           106.7%

setup provider tokens:      0
raw provider total:         352,940
optimized provider total:   366,966
product saving:             -3.97%
```

Interpretation:

- quality stayed safe;
- removing setup cost alone is not enough;
- deterministic v0 is too conservative and does not produce a tokenizer-cheaper
  representation. It mostly rewrites the transcript with ids/dates preserved,
  so the target prompt is slightly larger than raw.

Conclusion:

- a free codec only helps if it actually shortens the state;
- the next codec must be a **typed compact grammar**, not light text cleanup;
- target examples: `@u n=Oleksandr c=UA`, `@ev p=Caroline a=researched o=adoption_agencies t=2023-...`;
- keep the append-only/frozen rule, but redesign the deterministic encoder
  around structured fields and aliases.

### Abbreviated codec v0 falsification (2026-07-02)

Added `--encoder abbreviated`: a zero-provider-cost codec that shortens common
words (`researched -> rsch`, `adoption -> adpt`, `support -> sup`, etc.), removes
stopwords, and preserves names/dates/dialogue ids.

Real run:

```powershell
python tools\validation\llm_optimization_append_only.py --encoder abbreviated --conversation-limit 6 --session-limit 5 --qa-limit-per-conversation 10 --mode append_full --target-mode append_full --judge llm --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0 --max-accuracy-drop 1 --min-product-saving-pct -100
```

Report:

```text
data\llm_optimization\append_only_eval_20260702_003315_1782941595500320300.json
```

Result:

```text
raw accuracy:        31/60 = 0.517
optimized accuracy:  30/60 = 0.500
retention:           96.8%

setup provider tokens:      0
raw provider total:         352,950
optimized provider total:   321,061
product saving:             9.03%
```

Conversation state sizes stayed too close to raw:

```text
conv-26 raw 3439 -> abbreviated 3178
conv-30 raw 3550 -> abbreviated 3340
conv-41 raw 3571 -> abbreviated 3320
conv-42 raw 3435 -> abbreviated 3247
conv-43 raw 3479 -> abbreviated 3240
conv-44 raw 4026 -> abbreviated 3768
```

Failure pattern:

- almost all losses are temporal questions;
- abbreviated text often preserves `last week` / `yesterday` but weakens the
  date anchor needed to resolve it;
- examples: `19 Jan` became `20 Jan`, `21 Dec` became `22 Dec`, `next month`
  stayed unresolved.

Conclusion:

- LLMs can read abbreviated surface forms, but naive abbreviation is not enough;
- it saves only `9%`, below commercial interest;
- it slightly harms quality on date-heavy memory;
- next codec must explicitly type temporal anchors instead of relying on
  shortened prose, e.g. `@ev p=Jon a=lost_job o=banker t=2023-01-19 src=d1.4`.

### SCALE truth + DATE-GUARD — the honest, defensible result (200 QA)

The 100-103% retention from 6 conversations was a small-sample artifact. Scaled
to ALL 10 LoCoMo conversations, 200 QA, LLM judge, per-category:

```text
raw          : 126/200 = 0.63
append-only  : 111/200 = 0.56   retention 88%   (NOT 100%)
per-query saving : 45%

by category (raw / append):
  multi-hop   : 46 / 45   (facts+links preserved — 98%)
  temporal    : 58 / 46   (dates LOST — 79%)   <- the whole gap is here
  open-domain : 18 / 16
  single-hop  :  4 /  4
```

The loss is localised: append-only keeps facts and cross-session links almost
perfectly, but the compressor drops precise/relative dates ("yesterday",
"7 May", "a week ago"). Diagnosis, not vague degradation.

Fix — DATE-GUARD (deterministic, no extra model call): regex-extract absolute +
relative time expressions from each exchange, resolve relatives against the
session date, and append them VERBATIM after the compressed chunk
(`... [TIME: session=...; 7 May 2023; yesterday(rel to ...)]`). The model cannot
drop what is glued on after it.

Result of append-only + date-guard, same 200 QA:

```text
                 without guard   with date-guard
retention        88%             94%      (1 lost answer in 16, not in 8)
temporal cat     79%             96%      (the gap closed)
per-query saving 45%             60%      (went UP)
```

Savings went UP, not down: with dates protected separately, the prose compressor
can be more aggressive on the rest (no fear of dropping a date), so chunks are
shorter. Deterministic protection improved BOTH quality and saving.

Final honest numbers (200 QA, 10 conversations, fair judge):

```text
append-only + date-guard : 94% retention, 60% per-query saving
```

Architecture that emerged: COMPRESS prose with the model (good at facts/links)
+ DETERMINISTIC GUARD for exact fields (regex-extract, glue verbatim, let the
compressor be aggressive on the rest). Date-guard is the first guarded field;
the same pattern extends to numbers, amounts, codes, names, statuses — each
guarded field should push retention further up.

Why no one shipped this (honest answer): it is not one trick but a COMBINATION —
append-only-freeze (never re-summarise old state) + per-field deterministic
guards. Naive summary-memory (re-summarise everything) gives ~79%; this gives
94%. The result is "good, not wow" (94%/60%, not 100%/90%), and it cuts the
provider's own bill — so providers are not motivated to build it and hype-driven
research skips modest-but-honest numbers. A real but narrow gap: commercial only
for long sessions (break-even ~5-7 queries), best for conversational agent /
support / consulting threads.

### SHADOW INTEGRATION into the live agent (2026-07-02)

The lab reached diminishing returns; the next falsification is real traffic.
Integrated append-only + date-guard into the live chat as a SHADOW mode:

- `src/remy/core/session_state_wrapper.py`:
  - `SessionState.frozen_chunks` — append-only compressed history (each
    exchange compressed exactly once, then frozen);
  - `extract_time_anchors()` — date-guard covering English, Ukrainian
    (Cyrillic) AND Ukrainian (Latin transliteration). Real sessions here are
    mixed-language; an English-only guard would silently reopen the temporal
    gap that LoCoMo never shows;
  - `fold_exchange_append_only()` — the production fold: compress only the new
    exchange on the cheap model, glue `[TIME: ...]` anchors verbatim, append,
    persist, log metrics.
- `src/remy/core/session_state_shadow.py` — shadow runner: fires as a
  background task AFTER each successfully completed exchange (cancelled/failed
  exchanges are skipped, so the state is never poisoned). Zero effect on
  answers or latency; kill switch `SESSION_STATE_SHADOW=0`.
- `src/remy/web/routes/websocket.py` — `_do_generation` schedules the shadow
  fold in its `finally` only when generation succeeded.

Each real exchange now logs `shadow_exchange` metrics: raw-history tokens vs
frozen-state tokens = the saving the optimized mode WOULD deliver on this
app's own traffic. First live record honestly showed -58% (one-exchange
session — state with TIME guard is longer than a tiny raw history), exactly
matching the measured short-session loss; savings emerge as history grows.

Verified end-to-end on a real Ukrainian exchange: chunk compressed in
Ukrainian, date-guard caught `15 липня 2026` (absolute) + `вчора` (relative,
tagged with session date), pinned caught `PX-7702`. 43 tests pass.

Decision rule for enabling the mode for real: collect shadow metrics over
real long sessions; if retention-relevant signals and savings match the lab
numbers (94%/60%), flip "Use Optimization" to use `build_optimized_prompt`
with the frozen state. Until then, shadow only.

### facts vs hybrid across session length (decisive, provider tokens)

Running both cost-first modes at short and long session length settles the
product shape. Numbers are real provider tokens, `gemini-flash-lite-latest`.

Short session (`--append-noise-turns 3`, 5 cases):

```text
projected_facts:  provider_total_saved_pct_vs_raw = +6.56%  (cheaper)  gate PASS
projected_hybrid: provider_total_saved_pct_vs_raw = -3.92%  (dearer)   gate FAIL
```

Long session (`--append-noise-turns 40`, 8 cases):

```text
mode              accuracy  ctx_saved%  provider_total_saved%  calls
projected_facts   0.80      83.78%      83.31%                 10
projected_hybrid  1.00      82.90%      78.83%                 14
```

Strict per-case check of the two `projected_facts` failures at +40: both are
reasoning cases (`relational`, `three bullet points`, `risks`) genuinely
missing — a REAL loss, not a checker artifact. facts mode never runs decision
extraction, so it is structurally blind to prose decisions/preferences.

### Conclusion — the product is a router, not a single mode

- `projected_facts`: cheap at any length, but blind to reasoning. Correct only
  for fact-retrieval sessions (codes, ids, amounts). On a long mixed corpus it
  drops to 0.80 accuracy because it loses decision/preference answers.
- `projected_hybrid`: pays extra decision-extraction calls. On SHORT sessions
  that makes it dearer than raw (do not sell as savings there). On LONG sessions
  the one-time decision cost amortizes — ~79% real provider-token savings AND
  1.00 accuracy.

So the shipping design is a length/-type router:

```text
short session  OR  fact-only workload  -> projected_facts
long session   AND reasoning present   -> projected_hybrid
```

The exact switch-over length (where hybrid stops being dearer) is between +3 and
+40 and is not yet measured. `--target-mode` lets the gate validate each mode
independently against provider-total cost, not context-window size.

## Reasoning Boundary and Hybrid Fix (assistant session, 2026-07-01)

This section records a falsification pass and the fix that came out of it.

### What was falsified

Two adversarial checks were run against the deterministic projected state.

1. Estimate-vs-provider at long session (`--append-noise-turns 40`):

   ```text
   raw provider prompt tokens:       12345  (8/8 correct)
   projected provider prompt tokens:  1702  (8/8 correct)  -> 86.2% real
   estimate said:                            86.59%
   ```

   The token estimate does NOT lie on Cyrillic / long transcripts -- provider and
   estimate agreed within 0.5%. Fact-retrieval savings are real.

2. Added 2 reasoning cases (decision rationale, standing preference) whose
   answers are prose, not regex-extractable facts:

   ```text
   projected: 8/10 correct
   fact cases:      8/8  correct
   reasoning cases: 0/2  correct   <- FAILED
   ```

   Root cause (verified, not guessed): the projected state carries only regex
   pins plus the lines where a pin appears. A reasoning answer like "relational"
   or "three bullet points" has no regex anchor, so `pinned_facts == []`, the
   phrase never enters the prompt, and the model cannot answer.

### The fix -- two independent extractors (hybrid)

Instead of narrowing the product ("do not sell reasoning"), the projection was
extended. The earlier "summarize everything" LLM compression was rejected -- it
hallucinated by dropping medical facts and keeping step counts, because it was
asked to *judge importance*. The hybrid avoids that:

- **Fact extractor** (unchanged): regex pins -- deterministic, zero LLM cost.
- **Decision extractor** (new): a narrow LLM call that extracts ONLY decisions,
  preferences and standing rules, quoting phrases VERBATIM, explicitly told not
  to summarize or judge importance (returns `NONE` when there is nothing). It is
  **marker-gated** (`decid/prefer/because/virish/zavzhdy/...`) so noise turns never
  trigger a call. Output goes to a `[DECISIONS_AND_PREFERENCES]` section.
- Facts stay protected separately by regex (defense-in-depth): even if the
  decision extractor misses, exact values survive.

Wiring: `build_projected_state_from_log(..., decision_llm_func=...)`. In local
dry-run there are no external model calls; live provider runs pass the provider
function as `decision_llm_func`, so marked decision/preference turns can be
extracted into the projection.

### Result (provider `gemini-flash-lite-latest`, `--append-noise-turns 40`)

```text
before hybrid:  projected 8/10  (reasoning 0/2)
after  hybrid:  projected 10/10 (reasoning 2/2)

accuracy:                 1.0
effectiveness_rate:       1.0
context_window_saved_pct: 86.48%   (was 86.59% -- savings preserved)
provider prompt tokens:   15458 -> 2429  (7.4x)
decision-extract cost:    +3 LLM calls over 10 cases (13 vs 10) -- gate works
regression:               focused regression currently passes
```

### Niche, revised by measurement

The reasoning boundary was not a wall -- it was a missing extractor. The niche is
now broader and still evidence-backed:

- **Sell as:** durable memory for **facts AND decisions** in long threads --
  codes/dates/amounts (regex) plus decisions/preferences/agreements (decision
  extractor). Best for support/CRM/coding-agent/medical-admin threads; breaks
  even around the ~5th request; not for short one-shot chats.

## Current Interpretation

The direction is promising, but still research/prototype.

Strong signals:

- projected mode can preserve correctness on the current corpus;
- context-window savings become meaningful after the conversation grows;
- metrics now distinguish real savings from false savings.

Weaknesses:

- corpus is still small;
- no broad real-model regression suite yet;
- incremental mode needs more long-session testing;
- projected mode must not be enabled blindly;
- savings are workload-dependent.

## Required Gate Before Production

Optimization should only activate when it is expected to help.

Suggested rule:

```text
if raw_tokens < threshold:
    use raw
elif projected_tokens >= raw_tokens:
    use raw
else:
    use projected
```

Suggested initial threshold:

```text
raw_tokens >= 900
```

This threshold should be tuned with more data.

## Suggested Production Quality Gates

A release/build should fail if real-model evals do not meet minimum quality.

Suggested first thresholds:

```text
accuracy >= 0.95
effectiveness_rate >= 0.80
context_window_saved_pct_vs_raw_estimate >= 20% on long-session evals
0 false savings on short-session negative cases
```

Meaning:

- do not ship a system that saves tokens but answers incorrectly;
- do not enable projected mode where raw is cheaper;
- do not claim product value unless savings hold on realistic long sessions.

Current gate implementation:

- `tools/validation/llm_optimization_gate.py`
- tests: `tests/test_llm_optimization_gate.py`
- optional provider total-token gate:
  `--min-provider-total-savings-pct`
- target mode selector:
  `--target-mode projected_facts` or `--target-mode projected_hybrid`

Confirmed gate behavior:

```text
30-case dry-run baseline at --append-noise-turns 3:
RESULT: PASS

10-case gemini-3.5-flash smoke at --append-noise-turns 3:
RESULT: FAIL because context savings were 18.48% < 20.0%

5-case gemini-3.5-flash smoke with relaxed context threshold and provider
total-token gate:
RESULT: FAIL because provider total-token savings were -2.83% < 0.0%

5-case gemini-flash-lite-latest split-mode smoke at --append-noise-turns 3:
projected_facts RESULT: PASS with provider total-token savings 6.56%
projected_hybrid RESULT: FAIL with provider total-token savings -3.92%
```

## Cost-First Mode Split (2026-07-01)

The old `projected` mode mixed two different products:

- deterministic fact projection, which is cheap and has no setup LLM calls;
- hybrid projection, which can preserve decisions/preferences but pays extra
  LLM calls before the final answer.

That made context-window savings look useful even when total provider cost was
worse. The runner now separates the modes:

```text
raw                full transcript baseline
projected_facts    deterministic pinned facts only; cost-first baseline
projected_hybrid   pinned facts + LLM decision/preference extraction
incremental        rolling state update experiment
projected          backward-compatible alias for projected_hybrid
```

Real 5-case smoke on `gemini-flash-lite-latest`, `--append-noise-turns 3`:

```text
raw:
accuracy: 1.0
provider_total_tokens: 1478

projected_facts:
accuracy: 1.0
context_window_saved_pct_vs_raw_estimate: 15.68
provider_total_tokens: 1381
provider_total_saved_pct_vs_raw: 6.56
provider_calls: 5
gate with --target-mode projected_facts --min-provider-total-savings-pct 0: PASS

projected_hybrid:
accuracy: 1.0
context_window_saved_pct_vs_raw_estimate: 13.30
provider_total_tokens: 1536
provider_total_saved_pct_vs_raw: -3.92
provider_calls: 6
gate with --target-mode projected_hybrid --min-provider-total-savings-pct 0: FAIL
```

Product interpretation:

- `projected_facts` is currently the only mode showing positive real provider
  cost savings, but the observed 6.56% smoke result is only a proof signal, not
  a sellable result;
- `projected_hybrid` may still be useful for quality on reasoning-heavy long
  sessions, but it must prove amortized savings on longer sessions before it is
  marketed as cheaper;
- future reports should always compare both modes, not only `projected`.

Commercial threshold:

```text
provider_total_saved_pct_vs_raw < 10%   = not interesting commercially
10-20%                                  = maybe useful internally
20-30%                                  = minimum product-grade target
30%+                                    = strong sellable value if accuracy holds
```

For a standalone API-layer product, the gate should not be treated as passed
just because savings are positive. The product gate should require at least
20% real provider total-token savings on realistic long sessions, preferably
30%+ before positioning it as a serious cost-saving product.

## Commercial Long-Session Check (2026-07-01)

After setting the commercial bar, longer real-model runs were executed on
`gemini-flash-lite-latest`.

### 10 cases, medium-long session (`--append-noise-turns 10`)

```text
raw:
accuracy: 1.0
provider_total_tokens: 5366

projected_facts:
accuracy: 0.8
provider_total_saved_pct_vs_raw: 50.30
commercial gate: FAIL
reason: 2 wrong reasoning/preference answers despite high savings

projected_hybrid:
accuracy: 1.0
provider_total_saved_pct_vs_raw: 36.79
provider_calls: 14 vs raw 10
commercial gate with --min-provider-total-savings-pct 20: PASS
```

### 10 cases, long session (`--append-noise-turns 40`)

```text
raw:
accuracy: 1.0
provider_total_tokens: 16133

projected_facts:
accuracy: 0.8
provider_total_saved_pct_vs_raw: 83.31
commercial gate: FAIL
failed cases:
  - reasoning_decision_rationale: lost "relational"
  - reasoning_stated_preference: lost "three bullet" + "risks"

projected_hybrid:
accuracy: 1.0
provider_total_saved_pct_vs_raw: 78.83
commercial gate with --min-provider-total-savings-pct 20: PASS
```

### 30 cases, long session (`--append-noise-turns 40`)

This is the strongest result so far because it uses the full current corpus,
not just a smoke subset.

```text
raw:
accuracy: 1.0
provider_total_tokens: 47967
provider_calls: 30

projected_hybrid:
accuracy: 1.0
effectiveness_rate: 1.0
context_window_saved_pct_vs_raw_estimate: 82.93
provider_total_tokens: 10771
provider_calls: 48
provider_total_saved_pct_vs_raw: 77.54
commercial gate with --min-provider-total-savings-pct 20: PASS
```

Interpretation:

- `projected_facts` is too lossy for product use when sessions include
  decisions, preferences, or rationale. It can be an internal fast path only for
  exact-fact workloads.
- `projected_hybrid` is the current product candidate for long sessions: it pays
  extra extraction calls, but on long transcripts the final prompt savings
  dominate the setup cost.
- The sellable claim is not "saves a few percent"; the current long-session
  evidence is ~77-79% real provider total-token savings at 100% accuracy on the
  current 30-case corpus.
- This still needs broader corpus coverage and model-matrix repetition before
  it becomes a production claim.

## Stronger-Model Matrix Attempt and Fixes (2026-07-01)

A full long-session run was executed on `gemini-3.5-flash`:

```text
30 cases, --append-noise-turns 40

raw:
accuracy: 0.9667
provider_total_tokens: 48296

projected_hybrid:
accuracy: 0.9333
provider_total_tokens: 10892
provider_total_saved_pct_vs_raw: 77.45
commercial cost gate: PASS on savings, FAIL on accuracy
```

Failure analysis:

```text
preference_tone_style:
  raw and projected both preserved the rule but did not always use the exact
  word "terse". This was a checker/corpus issue for a semantic preference.

crm_churn_risk_reason:
  projected_hybrid lost the churn-risk reason because the extractor instruction
  only asked for decisions/preferences/rules. A CRM status reason is durable
  session context, but not a "decision".
```

Fixes applied:

- extraction instruction now includes durable customer/status/risk labels and
  their stated reasons, not only decisions/preferences;
- marker list now includes narrower durable-reason markers such as `marked`,
  `risk`, `reason`, `because`, and `lost`;
- a too-broad `status` marker was removed after it caused synthetic `status
  notes` noise to trigger 84 extraction calls and destroy savings;
- corpus checker now supports `expected_any` synonym groups for semantic
  wording checks, while exact facts remain strict in `expected_fragments`;
- added regression tests for durable CRM reasons and for avoiding extraction on
  routine `status notes` noise.

Targeted regression after the fix on `gemini-flash-lite-latest`:

```text
cases:
  - preference_tone_style
  - crm_churn_risk_reason
noise: 40

projected_hybrid:
accuracy: 1.0
provider_calls: 4
provider_total_saved_pct_vs_raw: 74.69
commercial gate with --min-provider-total-savings-pct 20: PASS
```

After adding marked source lines and the final-answer preference modifier rule,
the targeted `gemini-3.5-flash` regression passed:

```text
cases:
  - preference_tone_style
  - crm_churn_risk_reason
noise: 40

projected_hybrid:
accuracy: 1.0
provider_calls: 4
provider_total_saved_pct_vs_raw: 68.84
commercial gate with --min-provider-total-savings-pct 20: PASS
```

The full `gemini-3.5-flash` matrix was then repeated:

```text
30 cases, --append-noise-turns 40

raw:
accuracy: 1.0
provider_total_tokens: 49556
provider_calls: 30

projected_hybrid:
accuracy: 1.0
effectiveness_rate: 1.0
context_window_saved_pct_vs_raw_estimate: 78.47
provider_total_tokens: 13667
provider_calls: 48
provider_total_saved_pct_vs_raw: 72.42
commercial gate with --min-provider-total-savings-pct 20: PASS
```

Updated interpretation:

- the stronger-model failure was actionable, not a fatal flaw in the approach;
- preserving compact verbatim marked source lines is important because even a
  narrow extractor can drop small modifiers that matter;
- `projected_hybrid` now has two long-session provider confirmations:
  `gemini-flash-lite-latest` at 77.54% savings and `gemini-3.5-flash` at 72.42%
  savings, both at 30/30 accuracy on the current corpus.

## Commands

Dry-run, no model calls:

```powershell
python tools\validation\llm_optimization_corpus_eval.py --dry-run --mode raw,projected_facts,projected_hybrid
```

Dry-run with longer session simulation:

```powershell
python tools\validation\llm_optimization_corpus_eval.py --dry-run --mode raw,projected_facts,projected_hybrid --append-noise-turns 3
```

Real model run:

```powershell
python tools\validation\llm_optimization_corpus_eval.py --mode raw,projected_facts,projected_hybrid --model gemini-flash-lite-latest --append-noise-turns 3
```

Stronger model smoke run:

```powershell
python tools\validation\llm_optimization_corpus_eval.py --mode raw,projected_facts,projected_hybrid --model gemini-3.5-flash --case-limit 10 --append-noise-turns 3
```

Full local regression:

```powershell
python -m pytest tests\test_llm_optimization_corpus_eval.py tests\test_session_state_wrapper.py
```

Quality gate on a specific report:

```powershell
python tools\validation\llm_optimization_gate.py data\llm_optimization\corpus_eval_20260701_182526_1782919526815303200.json
```

Quality gate on latest report:

```powershell
python tools\validation\llm_optimization_gate.py
```

Quality gate requiring total provider-token savings:

```powershell
python tools\validation\llm_optimization_gate.py <report.json> --target-mode projected_facts --min-provider-total-savings-pct 0
```

Commercial cost gate:

```powershell
python tools\validation\llm_optimization_gate.py <report.json> --target-mode projected_facts --min-provider-total-savings-pct 20
```

Hybrid quality/cost gate:

```powershell
python tools\validation\llm_optimization_gate.py <report.json> --target-mode projected_hybrid --min-provider-total-savings-pct 0
```

Full focused regression including gate:

```powershell
python -m pytest tests\test_llm_optimization_gate.py tests\test_llm_optimization_corpus_eval.py tests\test_session_state_wrapper.py
```

Dry-run matrix smoke:

```powershell
python tools\validation\llm_optimization_matrix.py --dry-run --models dry-a,dry-b --noise-levels 3 --case-limit 5 --min-savings-pct 0 --min-provider-total-savings-pct 0
```

Commercial long-session matrix:

```powershell
python tools\validation\llm_optimization_matrix.py --models gemini-flash-lite-latest,gemini-3.5-flash --mode raw,projected_hybrid --noise-levels 40 --target-mode projected_hybrid --min-provider-total-savings-pct 20 --delay-sec 0.2
```

External LoCoMo smoke:

```powershell
python tools\validation\llm_optimization_locomo.py --model gemini-flash-lite-latest --mode raw,projected_hybrid,evidence_oracle --qa-limit 10 --delay-sec 0.2 --min-accuracy 0
```

## Next Steps

1. Harden the quality-gate script.
   - Done: basic gate exists and is tested.
   - Done: optional separate gate for total provider-token savings exists.
   - Done: model matrix runner evaluates and gates several models/noise levels
     in one command.

2. Expand corpus from 30 cases to 50+.
   - More exact facts.
   - More long support/CRM/coding cases.
   - More negative short-context cases.
   - More cases where meaning matters, not just exact IDs.
   - (started) reasoning cases added for decision rationale, standing
     preference, policy rules, source priority, and support rationale. Need
     more variants to confirm the decision extractor generalizes.

3. Run real-model regression matrix.
   - `gemini-flash-lite-latest`
   - `gemini-3.5-flash`
   - possibly other providers later

4. Stress-test incremental mode.
   - 50, 100, 300+ exchanges.
   - Track setup cost and break-even.

5. Wire gate into actual chat behavior.
   - Raw mode for short sessions.
   - Projected mode only after threshold.
   - Log when optimization is skipped and why.

6. Re-evaluate product direction.
   - If repeated real-model runs keep accuracy high and savings above threshold, this can become a hidden API-layer optimization product.
   - If value only appears in narrow cases, keep it as an internal Remy feature.

## Quality Gate + 30-Case Provider Proof (2026-07-01)

The automated quality gate (`tools/validation/llm_optimization_gate.py`) now
turns every eval run into a pass/fail verdict instead of hand-read numbers.

Gate checks (against projected mode): accuracy >= 0.95, effectiveness >= 0.80,
context savings >= 20% (enforced only on long sessions, noise >= 3), no false
savings, and no short negative case marked effective. Exit 0=pass, 1=fail,
2=error — CI/build usable.

### What the gate exposed (and why it matters)

A dry-run reported `accuracy 1.0` — but dry-run does not call the model, so
answers equal the expected fragments by construction. The **real provider run**
told the truth: 25/30 (0.833). The gate correctly FAILED it. Strict per-case
analysis then split the 5 failures:

```text
4 = TEST ARTIFACT: must_not_contain matched a forbidden phrase inside a correct
    answer that was *prohibiting* or *contrasting* it
    ("never use marketing", "throughput more than perfect prose").
1 = REAL PRODUCT GAP: a Latin-transliterated Ukrainian preference
    ("zavzhdy stav ryzyky pershymy") — the decision marker gate never fired,
    so the extractor never ran.
```

### Fixes applied

- Product: decision markers extended to English + Cyrillic Ukrainian + Latin-
  transliterated Ukrainian; answer system prompt now requires quoting the
  original phrase verbatim (translation only in parentheses) for other-language
  facts. This fixed the mixed-language case on the cheap model.
- Test checker: `check_answer` forbidden logic is now negation-aware
  (`_forbidden_used`) — a `must_not_contain` phrase is only a violation when used
  affirmatively, not when the answer negates or contrasts it. Real affirmative
  violations are still caught (regression-tested).

### Result — first full provider PASS

```text
model:        gemini-flash-lite-latest (cheap)
corpus:       30 cases (facts, reasoning, CRM, coding, health, RAG,
              negative, security, mixed-language)
noise:        +40 (long session)

accuracy:                 1.000  (30/30)
effectiveness_rate:       1.000
context_window_saved_pct: 82.86%  (real provider tokens)
false savings:            0
GATE:                     PASS
regression:               28 passed
```

Note on model tiers: `gemini-flash-lite-latest` translates mixed-language
answers by default (loses the verbatim quote); `gemini-3.5-flash` keeps the
original in parentheses. With the verbatim system-prompt rule, the cheap model
now also passes — but for heavy mixed-language workloads the stronger model is
the safer default.

## Current Bottom Line

Not yet a shipped standalone product, but the evidence is now real, not
estimated:

- correctness is preserved on a broad 30-case corpus at long session length
  (30/30 on a real provider), not just dry-run;
- ~83% real context savings on long sessions, break-even around the 5th request;
- an automated gate makes every future run pass/fail, so corpus growth, model
  matrices and long-session stress tests no longer need manual reading;
- clearest value for long-running agent / support / CRM / coding / medical-admin
  threads; not for short one-shot chats.

Next milestones: 50+ case corpus; real-model matrix (add a stronger model);
incremental-mode stress test at 100+ exchanges; then wire the gate into live
chat behaviour (raw for short sessions, projected past the threshold).

## External Code Benchmark: RepoBench Smoke (2026-07-01)

RepoBench was added as the first external code-oriented benchmark because it
tests repository-level Python context without requiring a full SWE-bench
execution environment.

Added:

- `tools/validation/llm_optimization_repobench.py`
  - local parquet/json/jsonl loader for RepoBench-style rows;
  - modes: `raw`, `cropped_only`, `gold_snippet_oracle`,
    `retrieved_snippets`, `projected_facts`, `projected_hybrid`;
  - exact normalized next-line scoring;
  - provider token accounting and context-window savings.
- `tests/test_llm_optimization_repobench.py`
  - miniature RepoBench fixture;
  - markdown/code-fence normalization regression.
- external data shard:
  - `data/evals/external/repobench_python_cross_file_first.parquet`
  - source dataset: `tianyang/repobench_python_v1.1`
  - downloaded only `cross_file_first-00000-of-00002` (~32 MB), not the whole
    dataset.

Validation:

```powershell
python -m pytest tests\test_llm_optimization_repobench.py tests\test_llm_optimization_locomo.py tests\test_llm_optimization_corpus_eval.py tests\test_session_state_wrapper.py
```

Result:

```text
35 passed
```

Dry-run sanity on 5 cases:

```powershell
python tools\validation\llm_optimization_repobench.py --dry-run --case-limit 5 --mode raw,cropped_only,gold_snippet_oracle,projected_facts --target-mode projected_facts --min-accuracy 0
```

Important dry-run signal:

```text
raw accuracy:              1.0 (dry-run, fake answer)
cropped_only context save: 36.59%
gold oracle context save:  22.19%
projected_facts save:     -7.24%  (larger than raw)
```

This shows that the current chat/session projector is not a code compressor.
For code it can increase prompt size.

Real model smoke, `gemini-flash-lite-latest`, 5 RepoBench cases:

```powershell
python tools\validation\llm_optimization_repobench.py --case-limit 5 --mode raw,cropped_only,gold_snippet_oracle,projected_facts --target-mode gold_snippet_oracle --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:                 accuracy 0.40, provider saved 0.00%
cropped_only:        accuracy 0.60, provider saved 37.29%
gold_snippet_oracle: accuracy 0.40, provider saved 23.67%
projected_facts:     accuracy 0.60, provider saved -5.34%
```

Real model smoke, `gemini-flash-lite-latest`, 3 RepoBench cases:

```powershell
python tools\validation\llm_optimization_repobench.py --case-limit 3 --mode raw,projected_hybrid --target-mode projected_hybrid --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:              accuracy 0.333, provider saved 0.00%
projected_hybrid: accuracy 0.667, provider saved -10.04%
```

Interpretation:

- External code benchmark does not validate the current projector as a
  standalone code-token saving product.
- Quality can improve in tiny samples because the projector changes prompt
  shape, but cost is worse than raw.
- `cropped_only` beating `raw` on 5 cases suggests that too much cross-file
  context can distract the model.
- The commercial path for code is likely **code-aware retrieval/reranking**:
  choose the right snippets before the LLM call, then measure accuracy and
  provider-token savings.

### First code-aware retrieval result

Added `retrieved_snippets`: a cheap non-LLM lexical selector that scores
cross-file snippets by identifier/import/path overlap and sends only the best
snippet with the target file context. This is not oracle: it does not use the
gold answer.

Dry-run, 10 RepoBench cases:

```powershell
python tools\validation\llm_optimization_repobench.py --dry-run --case-limit 10 --mode raw,cropped_only,retrieved_snippets,gold_snippet_oracle --target-mode retrieved_snippets --min-accuracy 0
```

```text
raw:                 context saved 0.00%
cropped_only:        context saved 42.85%
retrieved_snippets:  context saved 22.34%
gold_snippet_oracle: context saved 28.35%
```

Real model, `gemini-flash-lite-latest`, 10 RepoBench cases:

```text
raw:                 accuracy 0.20, provider saved 0.00%
cropped_only:        accuracy 0.30, provider saved 43.67%
retrieved_snippets:  accuracy 0.20, provider saved 22.61%
gold_snippet_oracle: accuracy 0.40, provider saved 29.21%
```

Real model, `gemini-flash-lite-latest`, 20 RepoBench cases:

```powershell
python tools\validation\llm_optimization_repobench.py --case-limit 20 --mode raw,retrieved_snippets --target-mode retrieved_snippets --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

```text
raw:                accuracy 0.25, provider tokens 29,979, saved 0.00%
retrieved_snippets: accuracy 0.35, provider tokens 22,204, saved 25.93%
retrieved hit rate: 5/20 = 25%
```

This is the first external code result that looks commercially interesting:
the code-aware selector improved accuracy over raw on this small sample while
reducing provider tokens by ~26%. The retrieval hit rate is still weak, so the
result is not production proof. It is a signal that the right product direction
is adaptive context selection, not universal context cleanup.

Next code-benchmark steps:

1. Run 50-100 RepoBench cases for `raw`, `cropped_only`, and
   `gold_snippet_oracle` to establish baseline and upper-bound savings.
2. Add a non-oracle `retrieved_snippets` mode:
   - lexical identifier/path overlap,
   - import/name overlap,
   - optionally embeddings later.
3. Gate code optimization separately from chat-memory optimization:
   - target accuracy should retain at least raw accuracy;
   - provider-token savings should be >= 20%;
   - no mode can pass if it is cheaper only because it drops necessary context.

## AuraSDK-verify Direction Audit: Binary Snapshot / Segment Store (2026-07-01)

User hypothesis: compression and cleanup are the wrong product primitive. The
older closed project `D:\AuraSDK-verify` used a binary snapshot/readout lane.

Audit result: this is a materially different architecture from summarization.
It does not try to make a long transcript shorter. It moves stable state into a
deterministic runtime readout substrate and sends only a scoped lookup key at
request time.

### What existed in AuraSDK-verify

Relevant files inspected:

- `scripts/aura_native_segment_store.py`
- `scripts/aura_native_segment_store_source_scope_context_key_v0.ps1`
- `scripts/aura_native_segment_store_source_scope_context_contract_v0.ps1`
- `scripts/aura_native_segment_store_duplicate_key_answer_guard_v0.ps1`
- `scripts/aura_native_segment_store_integrity_preflight_v0.ps1`
- `scripts/topological_crystallization_acl_brief_guided_batch_ingestion_v0.ps1`
- `scripts/topological_crystallization_acl_brief_night_batch_cost_benchmark_v0.ps1`
- `src/acl_brief.rs`
- `src/acl_brief_cli.rs`
- `AURA_SPECIALIST_ORGANISM_ARCHITECTURE_MAP_UA_V3.md`

Two distinct ideas were present:

1. **ACL-Brief as search warrant**
   - deterministic bounded state snapshot;
   - no LLM dependency while rendering;
   - fixed byte budget, structural refusal on overflow;
   - used to tell a later model what to search for, not to answer directly.

2. **Native binary segment store**
   - `route_lookup.bin` + `route_strings.seg`;
   - lookup by composite hash of `query + context_key`;
   - `context_key = scope + domain + source`;
   - exact/single match required;
   - duplicate keys and corrupt stores fail closed;
   - runtime readout is allowed only through answer gate.

### Key architecture shift

Old failed framing:

```text
long transcript -> summarize/compress -> ask LLM
```

AuraSDK framing:

```text
offline/build phase:
  source/events/transcript -> governed state -> binary/readout snapshot

runtime phase:
  request + scope/domain/source -> exact key lookup -> guarded readout
```

This means commercial savings come from **not asking the model to re-derive
state**, not from asking the model to read a shorter transcript.

### Safety/product invariants from AuraSDK

The old system had strong fail-closed contracts:

```text
retrieved text != evidence
segment record != truth
runtime segment readout != claim mutation
candidate != answer permission
operator feedback != truth
```

Readout could answer only if:

```text
SourceScope policy passes
context key is exact
match count is exactly 1
duplicate-key ambiguity is absent
record status allows answer
answer gate opens
no truth/evidence/route/crystallization/admission mutation happens
```

This is much stronger than a generic RAG/cache layer.

### Reported proof signals in AuraSDK docs

From `AURA_SPECIALIST_ORGANISM_ARCHITECTURE_MAP_UA_V3.md`:

```text
PASS_ACL_BRIEF_GUIDED_BATCH_INGESTION_V0
candidate_count_reduction_ratio = 0.9167
output_token_budget_saving_ratio = 0.75
outside_acl_search_warrant_admitted_count = 0
answer_permission_granted = false
truth_asserted = false
admits_evidence = false

PASS_ACL_BRIEF_GUIDED_LIVE_REQUEST_SMOKE_V0
candidate_count_reduction_ratio = 0.8
guided_llm_candidate_count = 1
unguided_llm_candidate_count = 5
```

These results are not the same as final-answer token savings. They show that a
bounded state snapshot can sharply reduce candidate/search space while keeping
truth boundaries closed.

### Product implication

The likely commercial primitive is not:

```text
context compression API
```

It is closer to:

```text
adaptive context/state gateway
```

Possible product API shape:

```text
POST /v1/state/index
  input: transcript/source/docs/events
  output: state_snapshot_id, binary index, audit report

POST /v1/resolve
  input: query, scope, domain, source, state_snapshot_id
  output:
    mode: direct_readout | compact_prompt | raw_fallback | blocked
    answer_context or final answer packet
    token_saved_estimate
    audit: why this mode was selected
```

For exact or recurring workflows, the expensive LLM call can be skipped or
replaced with a tiny guarded readout. For ambiguous/novel workflows, the system
falls back to retrieval or raw context.

### Why this may beat compression

Compression fails when the needed fact is not represented in the summary.
Binary/state readout fails differently: it either finds an exact governed
packet or blocks. That is safer commercially:

- less hallucination risk;
- predictable cost;
- auditable permission chain;
- no hidden broad context leakage;
- reusable across repeated requests;
- can be sold as infrastructure, not a prompt trick.

### Proposed next experiment in this repo

Build a small Python prototype, not a full port:

```text
tools/validation/llm_optimization_binary_snapshot.py
```

Experiment design:

1. Take 30-50 existing synthetic + code/support cases.
2. Build an offline `state_snapshot`:
   - key: `scope|domain|source|normalized_query`;
   - payload: answer packet or compact context packet;
   - metadata: status, source id, provenance, permissions.
3. Runtime modes:
   - `raw`;
   - `retrieved_snippets`;
   - `binary_snapshot_readout`;
   - `binary_snapshot_or_fallback`.
4. Gate:
   - direct readout accuracy >= raw on exact recurring queries;
   - provider-token saving >= 80% when readout hits;
   - blocked/unknown queries must not invent;
   - duplicate key must fail closed;
   - wrong scope/domain/source must fail closed.

If this passes, the product story becomes stronger:

```text
We do not merely reduce prompts.
We convert stable state into an auditable readout layer and call the LLM only
when the readout cannot safely answer.
```

### Prototype implemented: binary snapshot readout

Added:

- `tools/validation/llm_optimization_binary_snapshot.py`
- `tests/test_llm_optimization_binary_snapshot.py`

Prototype behavior:

- Builds a minimal binary snapshot:
  - `lookup.bin`
  - `strings.seg`
- Key contract:
  - `query + scope + domain + source`
- Runtime modes:
  - `raw`
  - `binary_snapshot_readout`
  - `binary_snapshot_or_fallback`
- Safety checks:
  - wrong scope blocks;
  - wrong domain blocks;
  - wrong source blocks;
  - duplicate composite key fails preflight;
  - preflight never grants answer permission;
  - readout never asserts truth/evidence mutation.

Local tests:

```powershell
python -m pytest tests\test_llm_optimization_binary_snapshot.py
python -m py_compile tools\validation\llm_optimization_binary_snapshot.py
```

Result:

```text
2 passed
py_compile ok
```

Dry-run, 30 synthetic cases:

```powershell
python tools\validation\llm_optimization_binary_snapshot.py --dry-run --case-limit 30 --mode raw,binary_snapshot_readout,binary_snapshot_or_fallback --target-mode binary_snapshot_readout --min-accuracy 0.95
```

```text
snapshot bytes: 8,716
preflight: Pass
negative probes: Pass

raw:
  accuracy 1.00
  provider_total_tokens estimate 4,127

binary_snapshot_readout:
  accuracy 1.00
  snapshot_hit_rate 1.00
  provider_total_tokens 0
  provider_total_saved_pct_vs_raw 100.00%
```

Real provider smoke, 10 synthetic cases, `gemini-flash-lite-latest`:

```powershell
python tools\validation\llm_optimization_binary_snapshot.py --case-limit 10 --mode raw,binary_snapshot_readout --target-mode binary_snapshot_readout --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0.95
```

```text
snapshot bytes: 2,958
preflight: Pass
negative probes: Pass

raw:
  accuracy 1.00
  provider_total_tokens 2,309

binary_snapshot_readout:
  accuracy 1.00
  snapshot_hit_rate 1.00
  provider_total_tokens 0
  provider_total_saved_pct_vs_raw 100.00%
```

Interpretation:

- This confirms the architecture-level economics for recurring/exact state:
  when a governed snapshot hit exists, provider cost drops to zero.
- This is not yet proof for open-ended tasks. It is an exact/readout lane.
- The commercial product should combine:
  - binary snapshot readout for stable recurring state;
  - retrieval/reranking for code/doc context;
  - raw fallback for novel or unsafe requests.
- This direction is stronger than compression because misses fail closed or
  fall back instead of silently losing information.

### Mixed traffic / partial snapshot economics

The runner now supports:

```text
--traffic-profile exact|paraphrase|mixed
--snapshot-coverage-pct <0..100>
```

This turns the perfect exact-hit proof into a more realistic blended traffic
test.

Dry-run, 30 cases, exact traffic with only 50% of cases indexed:

```powershell
python tools\validation\llm_optimization_binary_snapshot.py --dry-run --case-limit 30 --mode raw,binary_snapshot_readout,binary_snapshot_or_fallback --target-mode binary_snapshot_or_fallback --traffic-profile exact --snapshot-coverage-pct 50 --min-accuracy 0.95
```

```text
binary_snapshot_readout:
  accuracy 0.50
  snapshot_hit_rate 0.50
  blocked_rate 0.50
  provider_total_saved_pct_vs_raw 100.00%

binary_snapshot_or_fallback:
  accuracy 1.00
  snapshot_hit_rate 0.50
  fallback_rate 0.50
  provider_total_saved_pct_vs_raw 51.97%
```

Dry-run, 30 cases, mixed traffic (half exact, half paraphrase), full snapshot:

```powershell
python tools\validation\llm_optimization_binary_snapshot.py --dry-run --case-limit 30 --mode raw,binary_snapshot_readout,binary_snapshot_or_fallback --target-mode binary_snapshot_or_fallback --traffic-profile mixed --snapshot-coverage-pct 100 --min-accuracy 0.95
```

```text
binary_snapshot_readout:
  accuracy 0.50
  snapshot_hit_rate 0.50
  blocked_rate 0.50

binary_snapshot_or_fallback:
  accuracy 1.00
  snapshot_hit_rate 0.50
  fallback_rate 0.50
  provider_total_saved_pct_vs_raw 49.01%
```

Dry-run, 30 cases, paraphrase-only traffic:

```text
binary_snapshot_readout:
  accuracy 0.00
  snapshot_hit_rate 0.00
  blocked_rate 1.00

binary_snapshot_or_fallback:
  accuracy 1.00
  fallback_rate 1.00
  provider_total_saved_pct_vs_raw 0.00%
```

Real provider smoke, 10 cases, mixed traffic, `gemini-flash-lite-latest`:

```powershell
python tools\validation\llm_optimization_binary_snapshot.py --case-limit 10 --mode raw,binary_snapshot_or_fallback --target-mode binary_snapshot_or_fallback --traffic-profile mixed --snapshot-coverage-pct 100 --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0.95
```

```text
raw:
  accuracy 1.00
  provider_total_tokens 2,369

binary_snapshot_or_fallback:
  accuracy 1.00
  snapshot_hit_rate 0.50
  fallback_rate 0.50
  provider_total_tokens 1,164
  provider_total_saved_pct_vs_raw 50.87%
```

Product interpretation:

- Exact-hit lane is economically excellent.
- Pure readout is not user-facing enough by itself because misses block.
- `binary_snapshot_or_fallback` is the product mode: readout when safe, raw or
  retrieval fallback when not.
- Blended savings roughly track hit rate. A 50% hit rate produced ~49-52%
  savings in dry-run and 50.87% savings on a real provider smoke.
- If hit rate is 0%, the system correctly falls back and saves nothing instead
  of producing wrong answers. This is a useful production property.

## External benchmark rerun

Date: 2026-07-01

Reason: the synthetic and binary snapshot tests are controlled. They prove the
economics of known-state readout, but they do not prove that the optimizer can
handle open-ended memory or code tasks. We reran external benchmarks to check
whether the product idea collapses outside controlled traffic.

### LoCoMo episodic memory

Command:

```powershell
python tools\validation\llm_optimization_locomo.py --qa-limit 10 --mode raw,projected_hybrid,evidence_oracle --target-mode projected_hybrid --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:
  accuracy 0.60
  provider_total_tokens 231,959

projected_hybrid:
  accuracy 0.30
  retained_vs_raw_correct 0.50
  context_window_saved_pct_vs_raw_estimate 91.18%
  provider_total_tokens 68,930
  provider_total_saved_pct_vs_raw 70.28%

evidence_oracle:
  accuracy 0.30
  retained_vs_raw_correct 0.50
  context_window_saved_pct_vs_raw_estimate 99.01%
  provider_total_tokens 2,110
  provider_total_saved_pct_vs_raw 99.09%
```

Interpretation:

- This fails the quality bar. Saving 70-99% is not useful when accuracy drops
  from 60% to 30%.
- The projected/compressed memory lane is not production-ready for episodic
  memory.
- Binary snapshot should not be evaluated here by indexing benchmark gold
  answers; that would be an oracle, not a real optimization.
- For this class of traffic the only acceptable product mode is fallback-first:
  use raw context or high-recall retrieval until a state object is proven safe.

### RepoBench code context

Command:

```powershell
python tools\validation\llm_optimization_repobench.py --case-limit 20 --mode raw,retrieved_snippets,gold_snippet_oracle --target-mode retrieved_snippets --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:
  accuracy 0.35
  provider_total_tokens 29,983

retrieved_snippets:
  accuracy 0.35
  retrieved_hit_rate 0.25
  context_window_saved_pct_vs_raw_estimate 24.98%
  provider_total_tokens 22,218
  provider_total_saved_pct_vs_raw 25.90%

gold_snippet_oracle:
  accuracy 0.35
  context_window_saved_pct_vs_raw_estimate 33.77%
  provider_total_tokens 19,242
  provider_total_saved_pct_vs_raw 35.82%
```

Interpretation:

- Retrieval did not reduce accuracy in this run and saved 25.90% provider
  tokens, but the absolute accuracy is low and the hit rate is only 25%.
- This is directionally useful for code/document context, but not commercially
  strong yet.
- The oracle upper bound is only 35.82% savings on this sample, so this specific
  RepoBench setup does not prove a huge-cost-saving product.
- The next work should improve retrieval quality and routing, not generic
  compression.

### Current product thesis after external rerun

The universal "compress the chat and save tokens" product is still not proven.
External LoCoMo shows that aggressive projection can destroy quality.

The stronger commercial architecture is a gated optimization gateway:

1. `binary_snapshot_readout` for exact, stable, governed state where a safe hit
   can return without a provider call.
2. `retrieved_snippets` for code/docs where context can be selected without
   losing the task.
3. raw fallback for novel, paraphrased, ambiguous, or low-confidence requests.

Quality gates should fail CI if:

- external benchmark accuracy drops below raw by more than a small tolerance;
- provider savings are below a meaningful commercial threshold;
- projected/compressed mode activates on short sessions where it is more
  expensive than raw;
- binary snapshot readout answers outside verified scope/domain/source.

## Aura-clean transfer candidates

Date: 2026-07-01

Source reviewed: `D:\Aura-clean`

Useful patterns found:

1. Native segment store discipline:
   - composite key = query + source/scope/domain context;
   - duplicate composite keys fail closed;
   - partial/corrupt segment files fail closed;
   - preflight cannot grant answer permission or assert truth.

2. Query-anchored candidate guard:
   - candidate spans are filtered by structural conditions;
   - a found candidate is not automatically a public answer;
   - guard output keeps explicit flags such as `candidate_only`,
     `answer_permission_granted=false`, `truth_asserted=false`.

3. Bounded evidence assembler:
   - retrieval should produce bounded candidate material;
   - source text is not blindly replayed as answer memory;
   - answer authority remains a downstream decision.

4. Stop/Go review:
   - observability surfaces are allowed to expose useful material;
   - product answer use is blocked until a separate authority protocol says Go.

Transferred now:

- `llm_optimization_binary_snapshot.py` now has
  `review_snapshot_answer_authority`.
- Snapshot resolution now separates:
  - `candidate_found`;
  - `answer_permission_granted`;
  - `product_answer_use_allowed`;
  - final authority `Go` / `Stop`.
- A new test confirms that candidate-only snapshot material cannot be counted
  as a free answer hit.
- `llm_optimization_repobench.py` now has
  `review_retrieved_snippet_authority`.
- RepoBench now supports `retrieved_snippets_or_fallback`, which uses retrieved
  context only when the candidate passes the gate and otherwise uses raw
  context.

Next transfer candidates:

- Tune retrieval authority thresholds on real-model evals instead of dry-run
  only.
- Add duplicate/ambiguous candidate accounting to other retrieval-like runners:
  `candidate_found`, `candidate_ambiguous`, `candidate_authorized`,
  `fallback_reason`.
- Add commercial metrics:
  `eligible_request_rate`, `authorized_savings_rate`, `fallback_rate`,
  `wrong_cheap_answer_rate`.

Product interpretation:

The strongest Aura-clean lesson is that cheap state must be governed. The
business product should not be "always compress"; it should be "only use the
cheap lane when the cheap lane has answer authority, otherwise fall back."

### RepoBench retrieval authority dry-run

Command:

```powershell
python tools\validation\llm_optimization_repobench.py --dry-run --case-limit 20 --mode raw,retrieved_snippets,retrieved_snippets_or_fallback,gold_snippet_oracle --target-mode retrieved_snippets_or_fallback --min-accuracy 0
```

Result:

```text
retrieved_snippets:
  context_window_saved_pct_vs_raw_estimate 24.98%
  retrieved_hit_rate 0.25
  candidate_found_rate 1.00
  candidate_authorized_rate 0.90
  candidate_ambiguous_count 2
  fallback_rate 0.00

retrieved_snippets_or_fallback:
  context_window_saved_pct_vs_raw_estimate 19.93%
  retrieved_hit_rate 0.25
  candidate_found_rate 1.00
  candidate_authorized_rate 0.90
  candidate_ambiguous_count 2
  fallback_rate 0.10
```

Interpretation:

- The gate identified 2 ambiguous retrieval candidates out of 20.
- Fallback mode reduced estimated context-window savings from 24.98% to 19.93%,
  but it made the risk visible and avoided forcing cheap context when candidate
  authority was weak.
- This is the correct product direction: some savings are intentionally traded
  for quality preservation.

### RepoBench retrieval authority real-model smoke

Command:

```powershell
python tools\validation\llm_optimization_repobench.py --case-limit 10 --mode raw,retrieved_snippets,retrieved_snippets_or_fallback --target-mode retrieved_snippets_or_fallback --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:
  accuracy 0.20
  provider_total_tokens 15,396

retrieved_snippets:
  accuracy 0.20
  provider_total_tokens 11,742
  provider_total_saved_pct_vs_raw 23.73%
  candidate_authorized_rate 0.90
  fallback_rate 0.00

retrieved_snippets_or_fallback:
  accuracy 0.40
  provider_total_tokens 12,553
  provider_total_saved_pct_vs_raw 18.47%
  candidate_authorized_rate 0.90
  fallback_rate 0.10
```

Interpretation:

- On this small real-model slice, the gated fallback mode saved fewer tokens
  than raw retrieval but improved accuracy from 0.20 to 0.40.
- This is the desired product tradeoff: spend a little more when the cheap
  candidate is weak or ambiguous, and preserve quality.
- The sample is too small for a product claim, but it supports the direction of
  a state/gateway optimizer with explicit authority gates.

### Gate falsification on 25 RepoBench cases

After the 10-case smoke, a 25-case real-model run showed a more honest picture:

```text
raw:
  accuracy 0.28
  provider_total_tokens 36,970

retrieved_snippets:
  accuracy 0.24
  provider_total_saved_pct_vs_raw 26.62%

retrieved_snippets_or_fallback:
  accuracy 0.32
  provider_total_saved_pct_vs_raw 22.62%
  fallback_rate 0.08
  candidate_authorized_rate 0.92
```

The key falsification is not accuracy. It is whether the gate stops bad
retrieval:

```text
paired_cases: 25
wrong_retrieval_count: 19
wrong_retrieval_stopped_count: 1
wrong_retrieval_go_count: 18
wrong_retrieval_stop_rate: 0.0526

correct_retrieval_count: 6
correct_retrieval_stopped_count: 1
false_stop_rate_on_correct_retrieval: 0.1667

stop_count: 2
stop_precision_wrong_retrieval: 0.5
gate_discriminates: false
```

Interpretation:

- The current Go/Stop gate is mostly decorative for RepoBench retrieval.
- It authorized 18 of 19 wrong retrieval cases.
- It stopped only 1 wrong retrieval case and also stopped 1 correct retrieval
  case.
- The apparent `retrieved_snippets_or_fallback` improvement over raw/retrieval
  should not be credited to a strong gate. On this sample it is likely small-N
  generation noise plus a couple of fallback/call-variance effects.
- The commercial thesis remains valid, but the current retrieval authority
  criterion does not yet prove it.

New required gate metric:

```text
wrong_retrieval_stop_rate must be materially higher than false_stop_rate
```

For a product-grade retrieval gate, a reasonable first bar is:

```text
wrong_retrieval_stop_rate >= 0.50
false_stop_rate_on_correct_retrieval <= 0.10
provider_total_saved_pct_vs_raw >= 0.30
accuracy >= raw - 0.02
```

Current result fails the first and most important bar.

## State Language / LLM Intermediate Representation

Date: 2026-07-01

Hypothesis:

Natural language may not be the most economical format for passing session
state to an LLM. Instead of compressing a transcript, render durable state into
a deterministic intermediate representation that is shorter, more structured,
and still readable by current tokenizers/models.

Tested formats:

- `state_verbose`: current human-readable state render.
- `state_json`: compact JSON with no whitespace.
- `state_kv`: compact key-value lines.
- `state_symbolic`: short symbolic labels such as `@g`, `@d`, `@x`, `@r`.

### Dry-run: short sessions

Command:

```powershell
python tools\validation\llm_optimization_state_language.py --dry-run --case-limit 10 --mode raw,state_verbose,state_json,state_kv,state_symbolic --target-mode state_kv --min-accuracy 0
```

Result:

```text
raw:
  prompt_tokens_estimate 1,441

state_verbose:
  context_window_saved_pct_vs_raw_estimate -14.30%

state_json:
  context_window_saved_pct_vs_raw_estimate -8.74%

state_kv:
  context_window_saved_pct_vs_raw_estimate -8.47%

state_symbolic:
  context_window_saved_pct_vs_raw_estimate -2.50%
```

Interpretation:

- State language is not useful on short sessions. The wrapper/instruction
  overhead is larger than the saved transcript.
- Production routing must not activate state rendering on short chats.

### Dry-run: long/noisy sessions

Command:

```powershell
python tools\validation\llm_optimization_state_language.py --dry-run --case-limit 10 --append-noise-turns 20 --mode raw,state_verbose,state_json,state_kv,state_symbolic --target-mode state_kv --min-accuracy 0
```

Result:

```text
raw:
  prompt_tokens_estimate 5,041

state_verbose:
  context_window_saved_pct_vs_raw_estimate 65.86%

state_json:
  context_window_saved_pct_vs_raw_estimate 67.45%

state_kv:
  context_window_saved_pct_vs_raw_estimate 67.53%

state_symbolic:
  context_window_saved_pct_vs_raw_estimate 70.11%
```

Interpretation:

- On long/noisy sessions, deterministic state language crosses the commercial
  40% savings threshold in dry-run.
- `state_symbolic` is the most compact tested format.

### Real-model smoke: long/noisy sessions

Command:

```powershell
python tools\validation\llm_optimization_state_language.py --case-limit 5 --append-noise-turns 20 --mode raw,state_kv,state_symbolic --target-mode state_symbolic --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:
  accuracy 1.00
  provider_total_tokens 4,677

state_kv:
  accuracy 1.00
  provider_total_tokens 1,510
  provider_total_saved_pct_vs_raw 67.71%

state_symbolic:
  accuracy 1.00
  provider_total_tokens 1,364
  provider_total_saved_pct_vs_raw 70.84%
```

Interpretation:

- This is the strongest commercial signal so far: `state_symbolic` crossed
  70% provider-token savings without losing accuracy on a 5-case smoke.
- The sample is too small for a product claim.
- Unlike the RepoBench retrieval gate, this result is not about guessing
  whether context is safe. It is about rendering already-selected durable state
  in a cheaper LLM-readable format.
- Next falsification: run 20-30 real-model long/noisy cases and inspect failures
  by category. The bar should be:

```text
accuracy >= raw - 0.02
provider_total_saved_pct_vs_raw >= 0.40
short-session activation = false
exact facts preserved = true
```

### Real-model falsification: facts-only symbolic fails on decisions

Command:

```powershell
python tools\validation\llm_optimization_state_language.py --case-limit 20 --append-noise-turns 20 --mode raw,state_symbolic --target-mode state_symbolic --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:
  accuracy 1.00
  provider_total_tokens 18,558

state_symbolic:
  accuracy 0.55
  provider_total_tokens 5,100
  provider_total_saved_pct_vs_raw 72.52%
```

Failure analysis:

- Facts-only symbolic state saved tokens but dropped durable decisions and
  preferences.
- Failed cases included database rationale, report formatting rules, support
  policy reasons, model choice rationale, security rules, and CRM churn-risk
  reasons.
- In those cases the symbolic state often contained only the latest noisy
  recent turn: `@r:assistant: Acknowledged routine note 19.`

Interpretation:

- The compact language itself is not enough. The state builder must extract
  decision/reason/preference material before rendering.
- The commercial product must separate two layers:
  - state extraction/admission;
  - state rendering/language.

### Real-model smoke: hybrid symbolic state

Command:

```powershell
python tools\validation\llm_optimization_state_language.py --case-limit 10 --append-noise-turns 20 --mode raw,state_symbolic,state_symbolic_hybrid --target-mode state_symbolic_hybrid --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0
```

Result:

```text
raw:
  accuracy 1.00
  provider_total_tokens 9,353
  final_answer_provider_prompt_tokens 9,078

state_symbolic:
  accuracy 0.80
  provider_total_tokens 2,687
  provider_total_saved_pct_vs_raw 71.27%

state_symbolic_hybrid:
  accuracy 1.00
  provider_total_tokens 3,680
  provider_setup_tokens 714
  final_answer_provider_prompt_tokens 2,643
  final_answer_provider_prompt_saved_pct_vs_raw 70.89%
  provider_total_saved_pct_vs_raw 60.65%
```

Interpretation:

- Hybrid symbolic state recovered raw accuracy on this 10-case smoke.
- It still saved 60.65% total provider tokens even after counting extraction
  setup cost.
- Final answer prompt cost dropped by 70.89%.
- This is now the strongest product direction:

```text
long transcript -> admitted durable state -> symbolic state language -> LLM
```

Required next falsification:

```text
20-30 real-model cases
accuracy >= raw - 0.02
provider_total_saved_pct_vs_raw >= 0.40
provider_setup_tokens amortized over repeated requests
short-session activation disabled
```

### Real-model quality gate: hybrid symbolic state, 20 cases

Command:

```powershell
python tools\validation\llm_optimization_state_language.py --case-limit 20 --append-noise-turns 20 --mode raw,state_symbolic_hybrid --target-mode state_symbolic_hybrid --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0 --max-accuracy-drop 0.02 --min-provider-saving-pct 40
```

Result:

```text
raw:
  accuracy 1.00
  provider_total_tokens 18,537
  final_answer_provider_prompt_tokens 18,003

state_symbolic_hybrid:
  accuracy 1.00
  accuracy_delta_vs_raw 0.00
  provider_total_tokens 7,952
  provider_setup_tokens 2,067
  final_answer_provider_prompt_tokens 5,263
  final_answer_provider_prompt_saved_pct_vs_raw 70.77%
  provider_total_saved_pct_vs_raw 57.10%
  provider_calls 32

quality_gate:
  passed true
  max_accuracy_drop 0.02
  min_provider_saving_pct 40.0
```

Interpretation:

- This is the first result that crosses a plausible commercial threshold:
  no observed quality loss versus raw and 57.10% total provider-token savings.
- Setup extraction is counted in total provider tokens.
- Final-answer prompt cost is reduced by 70.77%; this matters for repeated
  follow-up requests after the state is built.
- The mode made 32 provider calls for 20 cases because hybrid extraction is
  triggered only for marked durable decision/preference turns.

Current product hypothesis:

```text
Do not sell generic prompt compression.
Sell durable-state admission + symbolic state rendering for long sessions.
```

Remaining falsifications before calling it product-grade:

- Run 30-50 cases and at least one stronger model.
- Test non-synthetic long sessions or customer-like logs.
- Add an activation policy: never use this path on short sessions where raw is
  cheaper.
- Add a repeated-request benchmark where setup cost is paid once and several
  follow-up answers reuse the same state.

### Repeated-request economics

Command:

```powershell
python tools\validation\llm_optimization_state_language.py --case-limit 10 --append-noise-turns 20 --mode raw,state_symbolic_hybrid --target-mode state_symbolic_hybrid --model gemini-flash-lite-latest --delay-sec 0.2 --min-accuracy 0 --max-accuracy-drop 0.02 --min-provider-saving-pct 40
```

Result:

```text
state_symbolic_hybrid:
  accuracy 1.00
  provider_total_saved_pct_vs_raw 60.81%
  provider_setup_tokens 711
  final_answer_provider_prompt_saved_pct_vs_raw 70.92%

amortized_provider_total_saved_pct_vs_raw_at_2x: 64.60%
amortized_provider_total_saved_pct_vs_raw_at_3x: 65.86%
amortized_provider_total_saved_pct_vs_raw_at_5x: 66.87%
amortized_provider_total_saved_pct_vs_raw_at_10x: 67.63%
```

Interpretation:

- Setup cost is not a blocker in repeated-use sessions.
- With two follow-up answers from the same admitted state, savings rises above
  64%.
- This supports a business positioning around long-running assistants,
  customer-support threads, coding-agent sessions, and workflow agents where
  state is reused.
