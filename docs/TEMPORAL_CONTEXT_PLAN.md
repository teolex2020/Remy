# Temporal Context Plan

Date: 2026-04-14
Status: planned

This note captures the temporal-context work we want to add to Remy so the
agent can reason about when information was said, learned, verified, or
superseded.

The immediate target is the agent layer, not the brain schema itself.

That means:

- first teach the agent to read and render time correctly
- later teach the memory substrate to model time as a first-class semantic axis

---

## Problem

Right now the agent tends to treat retrieved text as if it all exists "now".

That creates failures like:

- old plans sounding current
- stale facts being quoted as live truth
- "today / yesterday / two days ago" being handled poorly
- user statements from weeks ago being mixed with fresh updates

The issue is not only missing timestamps.

The issue is missing temporal interpretation:

- when something happened
- when it was said
- when it was observed by the agent
- when it was last confirmed
- whether it should still count as current

---

## Principle

Do not start with a deep brain migration.

Start with an agent-side temporal layer:

- current time anchor
- relative-time rendering
- stale/current framing in retrieved snippets

This is cheap, reversible, and likely to produce visible improvements fast.

Only after living with that should we add structural temporal semantics to the
brain itself.

---

## Phase A: Agent-Side Temporal Layer

### Goal

Make the agent understand time better during prompt construction and response
generation without changing core brain semantics yet.

### A1. Relative Time Translation

Before injecting records, beliefs, or prior messages into context, render
time-sensitive items with a relative label.

Examples:

- `[5 minutes ago | 2026-04-14 20:13] User said the project still fails to compile.`
- `[2 days ago | 2026-04-12] User asked to revise the architecture.`
- `[12 March 2026] Research plan for GitHub leads.`

Why:

- LLMs are weak at date arithmetic
- relative labels are much easier to reason over than raw ISO timestamps

### A2. Time Anchor In System Prompt

Inject a dynamic clock into the system prompt or equivalent turn scaffold.

Minimum:

- current date and time
- user timezone
- optionally: time since previous active session

Example framing:

- `Current time: Monday, 14 April 2026, 20:13 Europe/Kiev.`
- `Previous active session ended 4 hours ago.`

Why:

- the model needs to know where "now" is
- relative memory tags work much better when paired with a clear present anchor

### A3. Temporal Rendering Rules

Use human-friendly formatting instead of raw timestamps.

Recommended defaults:

- `< 2 minutes`: `just now`
- `< 1 hour`: `N minutes ago`
- `< 24 hours`: `N hours ago`
- `< 30 days`: `N days ago`
- `>= 30 days`: absolute date
- `>= 1 year`: month + year

Keep absolute dates alongside relative phrasing where useful.

Do not rely on relative phrasing alone for auditability.

### A4. Timezone Discipline

All temporal rendering must use one clear policy:

- store canonical timestamps in UTC
- render for the user in the user timezone

Without this, "just now" and "3 hours ago" will drift and break trust.

---

## Phase B: Agent Retrieval Policy

### Goal

Improve time awareness in recall without requiring full temporal decay logic in
the brain yet.

### B1. Retrieval Modes

The agent should distinguish between:

- current truth recall
- historical recall

Examples:

- `What is the current plan?` -> recent, verified, latest-confirmed items win
- `What were we discussing in March?` -> old March records should still be retrievable

This means recency should affect default ranking, not erase history.

### B2. Freshness Hints In Context

Retrieved items should carry one of:

- `current`
- `recent`
- `aging`
- `stale`

These should be used as prompt annotations, not yet as destructive memory edits.

### B3. Stale-Sensitive Response Policy

If the agent cites a time-sensitive claim, it should be able to say:

- when it was last confirmed
- or that it may be stale

Example:

- `This was last confirmed 12 days ago and may no longer be current.`

---

## Later: Brain-Side Temporal Semantics

This should not be the first implementation step.

It becomes appropriate once the agent-side temporal layer is working and we
have examples of where it is still insufficient.

### Desired Brain Fields

Longer term, records should distinguish:

- `first_seen`
- `last_confirmed`
- `source_time`
- `observed_at`
- `created_at`
- `updated_at`
- `expires_at` or `valid_until`

### Desired Brain Behaviors

- decay should affect default recall ranking, not hard deletion
- decay should be calibrated per type/tag, not globally
- repeated confirmation should refresh `last_confirmed`
- historical queries should bypass normal recency demotion when appropriate

### Important Caveat

`Decay != deletion`

If old records simply fade out of all recall, the agent loses legitimate
historical memory.

This is unacceptable for explicit historical questions.

---

## Memory Type Caveat

The semantic distinction we eventually want is something like:

- episodic memory
- semantic memory
- policy memory
- operational status memory

But we should not assume we can classify these perfectly at ingest time.

A more practical path is:

- default many things to episodic
- promote repeated/confirmed patterns later
- let semantic promotion be a background shaping process

Not an expensive LLM classifier on every write.

---

## Recommended Implementation Order

1. Agent-side relative-time rendering
2. System prompt current-time anchor
3. Time-aware retrieval formatting for snippets/messages
4. Observe real usage for 1-2 days
5. Add retrieval freshness modes if needed
6. Only then plan brain-side temporal schema/decay work

---

## Success Criteria

This work is successful if:

- the agent stops flattening old and new context into the same "now"
- stale plans stop sounding current by default
- the agent answers temporal questions more reliably
- users can see whether recalled information is fresh or old
- later brain-side temporal semantics can be added without rewriting the first layer

---

## Explicit Non-Goals For Now

Do not do these first:

- full temporal decay in the brain graph
- destructive expiry logic
- complex semantic/episodic auto-classification at ingest
- broad schema migration before the agent-side layer proves useful

The near-term win is in agent-side temporal interpretation.
