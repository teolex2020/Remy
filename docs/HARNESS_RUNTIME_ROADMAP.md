# Harness Runtime Roadmap

Date: 2026-03-30
Status: proposed

This roadmap captures what Remy and AuraSDK should borrow from the paper
`Natural-Language Agent Harnesses`.

The point is not to copy "natural language files" as a fashion choice.
The point is to make the harness layer explicit, testable, portable, and easier
to reason about.

For this repo, the most important lesson is:

- cognition is not enough
- memory is not enough
- the runtime harness itself must become a first-class system artifact

---

## Problem Statement

Remy already has many harness-like behaviors:

- tool routing
- specialist selection
- approval gates
- decision dossiers
- factuality checks
- recovery and replay
- background maintenance

But much of this still lives in a mixed form:

- prompt rules
- runtime heuristics
- scattered controller logic
- UI-only assumptions

That makes the system harder to:

- audit
- compare
- ablate
- port
- harden

The next maturity step is to make the harness layer more explicit.

---

## Core Thesis

Treat the harness as its own runtime object.

Not just:

- model prompt
- controller code
- hidden workflow assumptions

But as an explicit layer that defines:

- contracts
- stages
- role boundaries
- failure classes
- state semantics
- recovery semantics

---

## Critical

### 1. Runtime Contract Artifact

Goal:

- define the live agent loop as an explicit contract

Add a versioned artifact such as:

- `runtime_contract.md`
- `runtime_contract.yaml`

It should describe:

- expected inputs
- expected outputs
- allowed transitions
- validation gates
- retry rules
- stop conditions
- escalation conditions

Why it matters:

- easier to audit harness behavior
- easier to compare runtime changes
- easier to test regressions in orchestration

AuraSDK:

- no major direct change required

Remy:

- first-class contract surface for current runtime loop

### 2. Unified Failure Taxonomy

Goal:

- stop treating failures as ad hoc strings

Add canonical categories such as:

- `format_error`
- `tool_error`
- `validation_error`
- `verification_failed`
- `evidence_conflict`
- `memory_recovery_applied`
- `store_integrity_incident`
- `timeout`
- `approval_blocked`

Why it matters:

- cleaner operator visibility
- better retry and repair logic
- easier metrics and learning

AuraSDK:

- expose clearer store/integrity error classes

Remy:

- use shared failure codes in logs, UI, and runtime decisions

### 3. Explicit State Semantics

Goal:

- clearly define which state lives where

Split state into explicit classes:

- active runtime state
- durable cognitive memory
- replay/recovery journal
- operator artifacts
- temporary reasoning state

Why it matters:

- fewer hidden assumptions
- safer restart behavior
- easier recovery logic

AuraSDK:

- durable substrate and provenance primitives

Remy:

- explicit runtime charter for state ownership and lifecycle

### 4. Path-Addressable Runtime Artifacts

Goal:

- make important harness outputs inspectable and stable

Treat these as first-class artifacts:

- decision dossier
- research report
- reconstruction review
- startup incident report
- maintenance digest
- correction review summary

Why it matters:

- more operator trust
- easier debugging
- easier replay and verification

AuraSDK:

- stable record/provenance ids

Remy:

- artifact-first workflow instead of transient-only UI output

### 5. Verify / Repair Gates

Goal:

- no silent jump from execution to success claim

Add explicit gates between:

- plan -> execute
- execute -> verify
- verify -> repair
- repair -> finalize

Why it matters:

- prevents false success claims
- better fit for research, documents, recovery, and memory correction flows

AuraSDK:

- evidence and correction surfaces

Remy:

- gate logic in runtime and operator surfaces

---

## High ROI

### 6. Role Contracts Instead of Role Labels

Goal:

- roles should mean something operationally

For each runtime role define:

- allowed tools
- required output schema
- validation rule
- escalation path

Example roles:

- planner
- researcher
- verifier
- reviewer
- operator-facing summarizer

Why it matters:

- better multi-role orchestration without role theater

### 7. Harness-Level Ablation and Testing

Goal:

- test harness modules as modules

Examples:

- with vs without verify gate
- with vs without recovery replay
- with vs without correction loop
- with vs without decision dossier surface

Why it matters:

- lets you measure harness value directly

### 8. Operator-Visible Incident Layer

Goal:

- runtime incidents should be visible without reading raw logs

Show:

- quarantine happened
- recovery ran
- how many records restored
- what remains missing
- what failure class triggered the incident

Why it matters:

- converts hidden runtime failure into reviewable operator state

---

## Medium Priority

### 9. Harness Charter

Goal:

- separate task-specific harness logic from shared runtime policy

The charter should define:

- lifecycle semantics
- approval semantics
- retry semantics
- child-agent semantics
- shutdown semantics
- artifact persistence expectations

Why it matters:

- cleaner portability across task types

### 10. Script and Adapter Registry

Goal:

- distinguish deterministic actions from model-selected reasoning steps

Maintain a cleaner registry of:

- adapters
- parsers
- validators
- system actions

Why it matters:

- less hidden glue
- easier debugging and reuse

### 11. Harness Diff and Migration Discipline

Goal:

- treat harness changes as reviewable system changes

Track:

- contract version
- runtime behavior change summary
- migration notes

Why it matters:

- avoids silent orchestration drift

---

## Later

### 12. Natural-Language Harness Files

Goal:

- externalize more orchestration logic into explicit human-readable artifacts

This should be done only where it improves:

- clarity
- portability
- reviewability

It should not become:

- text for the sake of text
- duplication of deterministic code

### 13. Composable Harness Modules

Goal:

- compose runtime modules intentionally

Examples:

- planning module
- verification module
- recovery module
- correction module
- research module

Why it matters:

- cleaner experimentation
- cleaner task specialization

---

## Split of Responsibility

### Must Be Solved In AuraSDK

- stable evidence ids and provenance
- explicit integrity and compatibility signals
- durable record and snapshot primitives
- correction and contradiction surfaces
- inspection APIs for runtime validation

### Must Be Solved In Remy

- runtime contract
- failure taxonomy wiring
- harness charter
- operator-visible artifacts
- verify/repair gates
- incident visibility
- recovery workflow

### Must Be Solved In Both

- recovery validation
- artifact portability
- compatibility handling
- test scenarios for failure and restart

---

## Recommended Order

1. runtime contract artifact
2. unified failure taxonomy
3. explicit state semantics
4. verify / repair gates
5. operator-visible incident layer
6. role contracts
7. harness-level ablation tests
8. charter and adapter registry
9. migration/version discipline
10. later natural-language harness externalization

---

## Success Criteria

This roadmap is successful if:

- the harness becomes easier to inspect than the code alone
- runtime failures are explainable in contract terms
- restart, recovery, and correction flows become operator-visible
- role boundaries become operational instead of decorative
- major orchestration changes can be reviewed and tested as harness changes

---

## Canonical Framing

Do not frame this as:

- "we need more prompts"
- "we should rewrite everything in text"
- "the harness is just workflow glue"

Frame it as:

- harness is a first-class runtime layer
- harness quality determines agent reliability
- explicit contracts and state semantics are part of production readiness
