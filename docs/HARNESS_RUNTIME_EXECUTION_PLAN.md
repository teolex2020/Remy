# Harness Runtime Execution Plan

Date: 2026-03-30
Status: first slice complete; retained as implementation history

This document turns the harness roadmap into an implementation-first plan.

It answers two questions:

1. what to implement first
2. what exact files and interfaces should appear in the repo first

This is not the full long-term architecture.
It is the shortest path to making the runtime harness explicit and operational.

Implementation note (2026-08-25): the first slice is operational. Runtime state
semantics live in `state_semantics.py` plus `contracts/state_semantics.yaml` rather
than the originally proposed `runtime_state_schema.py`; System already exposes the
contract, state registry, incident taxonomy, verification gates, and eval summaries.

---

## Phase 1

These are the first changes worth making because they create the runtime spine.

### 1. Runtime Contract Artifact

Implement first:

- one canonical runtime contract file
- one runtime loader/parser
- one place in `System` that exposes the active contract version

Expected result:

- the agent loop is no longer defined only by scattered code and prompts

### 2. Unified Failure Taxonomy

Implement first:

- one shared enum-like failure code set
- one mapper from current exceptions/incidents into canonical codes
- one operator-visible failure summary in `System`

Expected result:

- incidents become classifiable, measurable, and reviewable

### 3. Explicit State Semantics

Implement first:

- one shared runtime state schema
- one documented split between:
  - runtime state
  - durable memory
  - replay journal
  - operator artifacts

Expected result:

- fewer hidden assumptions around what is persisted and why

### 4. Verify / Repair Gate

Implement first:

- one minimal verify gate before success/finalize paths
- one repair decision path if verification fails

Expected result:

- less false-success behavior
- cleaner operator trust

---

## Phase 2

These should follow immediately after the runtime spine exists.

### 5. Artifact-Backed Verification

Implement:

- evidence-bearing task artifacts
- explicit `verification_result`
- explicit `repair_reason`

Expected result:

- completion becomes inspectable, not only implied

### 6. Operator Incident Layer

Implement:

- startup incident banner
- recovery summary banner
- failure class + recovery action summary

Expected result:

- operator sees incidents without log-diving

### 7. Role Contracts

Implement:

- role definition files or runtime constants
- per-role obligations:
  - allowed tools
  - expected artifact
  - required evidence
  - verification owner
  - failure ownership

Expected result:

- roles become operational, not decorative

---

## Files To Add First

These are the first concrete repo artifacts I would add.

### 1. Contract File

Add:

- [`src/remy/contracts/runtime_contract.yaml`](E:/remy/app/src/remy/contracts/runtime_contract.yaml)

Purpose:

- canonical definition of current runtime loop

Initial sections:

- version
- stages
- transitions
- stop conditions
- retry rules
- escalation rules
- required artifacts
- failure classes

### 2. Failure Taxonomy Module

Add:

- [`src/remy/core/failure_taxonomy.py`](E:/remy/app/src/remy/core/failure_taxonomy.py)

Purpose:

- shared failure codes and normalization helpers

Initial interfaces:

- `FailureCode`
- `FailureSeverity`
- `RuntimeIncident`
- `normalize_failure(...)`
- `classify_exception(...)`

### 3. Runtime Contract Loader

Add:

- [`src/remy/core/runtime_contract.py`](E:/remy/app/src/remy/core/runtime_contract.py)

Purpose:

- load and validate active runtime contract

Initial interfaces:

- `RuntimeContract`
- `load_runtime_contract()`
- `get_runtime_contract_summary()`

### 4. State Semantics Module

Add:

- [`src/remy/core/runtime_state_schema.py`](E:/remy/app/src/remy/core/runtime_state_schema.py)

Purpose:

- explicit schema for runtime state classes

Initial interfaces:

- `RuntimeStateClass`
- `RuntimeStateDescriptor`
- `get_runtime_state_model()`

### 5. Verification Gate Module

Add:

- [`src/remy/core/verification_gate.py`](E:/remy/app/src/remy/core/verification_gate.py)

Purpose:

- minimal shared verify/repair decision logic

Initial interfaces:

- `VerificationResult`
- `VerificationOutcome`
- `run_verification_gate(...)`
- `should_repair(...)`

---

## Files To Update First

### 1. [`src/remy/core/agent.py`](E:/remy/app/src/remy/core/agent.py)

Use it to:

- attach contract version to session/runtime log
- normalize runtime incidents into failure taxonomy
- run verify gate before final success paths

### 2. [`src/remy/core/combined_runner.py`](E:/remy/app/src/remy/core/combined_runner.py)

Use it to:

- expose contract summary in combined runtime status
- surface startup/shutdown incidents with canonical failure codes

### 3. [`src/remy/web/routes/system_routes.py`](E:/remy/app/src/remy/web/routes/system_routes.py)

Use it to:

- return:
  - active contract version
  - current incident state
  - last recovery action
  - failure taxonomy summary

### 4. [`src/remy/web/static/js/system.js`](E:/remy/app/src/remy/web/static/js/system.js)

Use it to:

- render:
  - active runtime contract
  - runtime incident banner
  - failure class summary
  - verify/repair status

### 5. [`src/remy/web/static/js/activity.js`](E:/remy/app/src/remy/web/static/js/activity.js)

Use it to:

- include:
  - current stage
  - verification status
  - repair status
  - failure ownership

---

## Interfaces To Expose First

These are the minimal interfaces worth stabilizing early.

### Contract

- `get_runtime_contract_summary() -> dict`

Return:

- `version`
- `stages`
- `failure_classes`
- `required_artifacts`

### Incident

- `get_runtime_incident_status() -> dict`

Return:

- `active_incident`
- `failure_code`
- `severity`
- `recovery_applied`
- `operator_message`

### Verification

- `run_verification_gate(context) -> VerificationResult`

Return:

- `status`
- `failure_code`
- `artifact_ids`
- `repair_required`
- `reason`

### State

- `get_runtime_state_model() -> list[dict]`

Return:

- state class
- persistence expectation
- visibility expectation
- owner

---

## Metrics To Add Early

Do not wait until later to measure this.

Add at least:

- `% incidents mapped to canonical failure codes`
- `% completions passing explicit verify gate`
- `% recovery events surfaced to operator`
- `restart success rate after interruption`
- `% task artifacts carrying stable ids`

---

## Shortest Safe Order

If implementation time is limited, do it in this order:

1. `failure_taxonomy.py`
2. `runtime_contract.yaml`
3. `runtime_contract.py`
4. `system_routes.py` contract + incident summary
5. `system.js` operator-visible banner
6. `verification_gate.py`
7. `agent.py` verify-before-success path

That is the minimum slice that starts turning the harness into a real runtime layer.

---

## Definition Of Done For The First Slice

The first slice is done when:

- the active runtime contract has a versioned artifact
- startup/recovery incidents have canonical failure codes
- `System` shows contract version and incident status
- at least one success path runs through an explicit verify gate
- repair-required outcomes are visible, not hidden

---

## Canonical Framing

This plan should be read as:

- first make the harness explicit
- then make it observable
- then make it enforceable

Do not invert that order.
