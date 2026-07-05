# Remy Memory Architecture

## Purpose

Remy uses Aura as an agent memory substrate, not as a replacement for a
traditional database.

Aura in Remy serves four goals:

- preserve stable long-term facts
- provide adaptive cognitive context for reasoning
- learn from repeated experience, feedback, and failures
- control how memory is used for autonomous action

This document defines how memory should be written, recalled, protected, and
used by the agent.

## Core Model

Remy memory has four levels:

- `IDENTITY`
- `DOMAIN`
- `DECISIONS`
- `WORKING`

And three behavior axes:

- `level`
- `semantic_type`
- `trust/provenance`

Together these define:

- how long a record should live
- how it should rank in recall
- whether it can be used for reasoning
- whether it can be used for external action

## Layer Semantics

### `IDENTITY`

Use for stable user or agent facts.

Examples:

- name
- birth date
- age
- location
- contact information
- language preferences
- stable user preferences
- verified credentials or identity-linked exact facts

Properties:

- highest persistence
- highest priority for profile continuity
- should contain mostly verified or high-trust data

### `DOMAIN`

Use for long-lived facts and reusable knowledge.

Examples:

- validated research findings
- people and pet facts
- domain knowledge
- durable notes
- reusable playbooks
- proven strategies

Properties:

- persistent but less strict than `IDENTITY`
- main long-term knowledge layer
- good target for promoted findings from research or autonomy

### `DECISIONS`

Use for agent or user choices that should persist for some time.

Examples:

- chosen plans
- approved next steps
- constraints
- blockers
- accepted mission directions
- committed actions

Properties:

- more persistent than `WORKING`
- should influence near-term execution
- good place for mission and plan state summaries

### `WORKING`

Use for short-lived cognitive context.

Examples:

- scratchpad notes
- current session context
- temporary hypotheses
- intermediate findings
- failed attempts
- recent observations

Properties:

- shortest-lived layer
- optimized for current reasoning, not archival truth
- must be regularly filtered, summarized, promoted, or discarded

## Exact vs Cognitive Memory

Remy memory is not split into "Aura vs database".
The split is inside the memory model itself.

### Exact Memory

Primarily:

- `IDENTITY`
- `DOMAIN`

Characteristics:

- stable
- structured
- trusted
- suitable for exact retrieval

Examples:

- phone numbers
- emails
- birth date
- location
- profile facts
- validated domain records

### Cognitive Memory

Primarily:

- `DECISIONS`
- `WORKING`

Characteristics:

- adaptive
- contextual
- reasoning-oriented
- less strict, more dynamic

Examples:

- current plan notes
- recent failures
- mission traces
- temporary research leads
- scratchpad context

## Semantic Types

`semantic_type` is a required behavioral modifier, not optional decoration.

Supported core types:

- `fact`
- `decision`
- `preference`
- `contradiction`
- `trend`
- `serendipity`

### `fact`

Default knowledge or exact record.

Use for:

- profile facts
- neutral stored knowledge
- validated research conclusions

### `decision`

Use for chosen or binding actions.

Use for:

- approved plans
- selected strategy
- mission choices
- resolved next steps

### `preference`

Use for user or agent preferences.

Use for:

- communication preferences
- style choices
- repeated behavioral preferences

### `contradiction`

Use when conflicting facts must be preserved for later resolution.

Use for:

- conflicting user statements
- conflicting research findings
- data disputes

### `trend`

Use for repeated patterns over time.

Use for:

- recurring failures
- usage patterns
- recurrent topics
- long-term changes in behavior

### `serendipity`

Use for unexpected but useful cross-domain connections.

Use for:

- novel associations
- unexpected opportunities
- emergent strategic ideas

## Trust and Provenance

Every stored record should carry provenance where possible.

Recommended metadata:

- `source`
- `verified`
- `trust_score`
- `timestamp`
- `volatility`
- `actionable`

This metadata determines:

- recall ranking
- action safety
- whether a record may be used in external operations

### Reasoning Rule

The agent may reason over broad memory context.

### Action Rule

The agent must not use memory for external action unless the record is:

- sufficiently trusted
- allowed by policy
- verified when the action is sensitive

## Retrieval Model

Remy uses three retrieval modes.

### `search_exact`

Purpose:

- precise lookup
- tag filtering
- structured retrieval
- exact field retrieval

Best for:

- profile facts
- contact data
- credentials
- exact names
- tags
- explicit filters

Priority:

- `IDENTITY`
- `DOMAIN`
- then `DECISIONS`
- then `WORKING`

### `recall_cognitive`

Purpose:

- semantic recollection
- contextual memory retrieval
- experience-based lookup

Best for:

- paraphrase queries
- mission context
- repeated failure lookup
- prior attempts
- LLM prompt injection

Priority depends on task, but usually includes:

- relevant `WORKING`
- relevant `DECISIONS`
- supporting `DOMAIN`

### `search_hybrid`

Purpose:

- combine exact and semantic retrieval
- deduplicate results
- rank by usefulness and trust

Best for:

- user-facing memory search
- agent memory browsing
- exploratory recall where exact and semantic signals both matter

Default hybrid order:

1. exact search
2. semantic recall
3. merge
4. dedupe
5. trust-aware ranking

## Protected Data Policy

Sensitive exact records require a stricter access path.

Protected categories include:

- passwords
- API keys
- wallet addresses
- phone numbers
- emails
- financial identifiers
- credentials

Rules:

- store as exact records with protection metadata
- do not expose broadly in normal recall
- do not inject into prompts unless explicitly necessary
- do not use for autonomous external actions without policy approval
- prefer explicit secure retrieval tools over general search

## Write Policy

### Store in `IDENTITY` when:

- the fact defines who the user or agent is
- the fact is stable over long periods
- the fact is exact and likely to remain true

### Store in `DOMAIN` when:

- the record is reusable knowledge
- the finding should survive beyond the current task
- the result has been validated or is likely to remain useful

### Store in `DECISIONS` when:

- the record captures a chosen action or commitment
- the agent must remember it for execution continuity
- the record may need to promote later

### Store in `WORKING` when:

- the record is temporary
- the record supports immediate reasoning
- the record may be summarized, promoted, or discarded later

## Promotion and Demotion Rules

### Promote `WORKING -> DECISIONS`

When:

- a note becomes an accepted plan
- a hypothesis becomes a chosen next step
- a temporary observation affects execution continuity

### Promote `WORKING/DECISIONS -> DOMAIN`

When:

- a finding becomes validated knowledge
- a pattern has repeated enough to matter
- a lesson becomes reusable

### Promote `DOMAIN -> IDENTITY`

Only when:

- the fact is truly stable
- it defines the user or agent profile
- it has long-term continuity value

### Demote or discard from `WORKING`

When:

- the record is stale
- the record is irrelevant to active work
- it can be summarized without losing important facts

## Usage by Subsystem

### Chat / Interaction

Use:

- `IDENTITY` for personal continuity
- `DOMAIN` for persistent background knowledge
- `WORKING` for immediate recent context

### Autonomy

Use:

- `DECISIONS` and `WORKING` for mission state
- `DOMAIN` for reusable strategies and validated findings
- `IDENTITY` for user constraints and long-term preferences

### Research

Use:

- `WORKING` for raw leads and active notes
- `DOMAIN` for validated findings
- `contradiction` semantic type for conflicting evidence
- `DOMAIN` for reusable research playbooks

## Operational Rules

### Reason on cognitive memory

The agent should reason using:

- `WORKING`
- `DECISIONS`
- relevant `DOMAIN`

### Act on exact trusted memory

The agent should act externally using:

- verified `IDENTITY`
- trusted `DOMAIN`
- protected records only through safe policy paths

### Never treat all memory as equal

The agent must distinguish:

- exact vs cognitive
- trusted vs low-trust
- protected vs safe-to-surface
- current vs archival

## Implementation Guidance

Recommended retrieval paths in Remy:

- `search_exact`
- `recall_cognitive`
- `search_hybrid`
- protected secret retrieval path

Recommended write matrix:

- `record type -> level -> semantic_type -> protection policy`

Recommended future improvements:

- require `semantic_type` for all important new records
- add explicit protected retrieval for sensitive exact data
- keep `WORKING` memory filtered and summarized
- use trust-aware hybrid retrieval for user-facing search

## Short Version

Remy memory should work like this:

- `IDENTITY` and `DOMAIN` preserve what is true and stable
- `DECISIONS` and `WORKING` preserve what the agent is doing and learning
- `search` finds exact things
- `recall` remembers context
- `hybrid` combines both
- `trust` and `policy` decide what may be used for action

Aura is therefore not just memory storage.
It is the cognitive and exact memory substrate that gives Remy continuity,
learning, and safer autonomous behavior.
