# Memory Reliability Roadmap

Date: 2026-03-23
Status: proposed

This roadmap is about durability, recovery, and operational trust in the cognitive memory stack.

It is not about making the system "smarter".
It is about making sure the memory layer survives restarts, interruptions, upgrades, and operator mistakes without silent loss.

## Problem Statement

The current memory stack is already strong in cognition:

- multi-level memory
- salience
- reflection
- contradiction review
- explainability

But it is still weak in persistence reliability:

- abrupt shutdown can damage the active store
- startup may quarantine a store and reopen on an empty one
- recovery is possible, but still reactive rather than guaranteed
- operator visibility around incidents is still limited

So the next maturity step is not more cognition first.
It is reliability engineering.

---

## Critical

These are production-blocking items.

### 1. Crash-Safe Store Finalization

Goal:

- an interrupted process should not easily leave the active store unreadable

AuraSDK:

- make writes more crash-resilient
- ensure close/flush semantics are explicit and testable
- document what guarantees exist after partial writes

Remy:

- never bypass normal shutdown with hard exit in routine operator flow
- keep explicit `brain.close()` and safe shutdown messaging
- ensure all stop paths use the same graceful lifecycle

### 2. Startup Integrity Classification

Goal:

- distinguish:
  - healthy store
  - temporarily locked store
  - corrupted store
  - format mismatch

AuraSDK:

- expose clearer error categories than generic read failures

Remy:

- stop treating all startup failures as one class
- log and surface exact incident reason

### 3. Automatic Recovery Guarantees

Goal:

- startup after quarantine should not come up empty without recovery

AuraSDK:

- support safe import/rebuild paths from external sources

Remy:

- keep history-based recovery
- add post-recovery verification
- show operator alert when recovery ran

### 4. Snapshot / Rollback Discipline

Goal:

- every significant runtime period should be recoverable

AuraSDK:

- stable snapshot listing and rollback guarantees
- snapshot integrity checks

Remy:

- automatic pre-upgrade and pre-maintenance snapshots
- operator-visible snapshot status
- recovery playbook in UI/docs

### 5. Reliability Test Harness

Goal:

- reproduce failures before users do

AuraSDK:

- tests for interrupted writes
- reopen after abrupt termination
- snapshot rollback after damage

Remy:

- integration tests for:
  - first Ctrl+C
  - second Ctrl+C
  - restart after interrupted shutdown
  - quarantine + replay + UI recovery

---

## Medium Priority

These are important, but they come after the critical durability path.

### 6. Operator Incident Visibility

Goal:

- operator should immediately know:
  - quarantine happened
  - recovery ran
  - how many records were restored
  - what still looks missing

AuraSDK:

- not much required beyond better diagnostics

Remy:

- incident banner in `System`
- recovery summary in `Memory`
- logs linked to operator-facing status

### 7. Durable Export Layer

Goal:

- preserve critical facts independently of one binary store

AuraSDK:

- stable export/import surface for records, provenance, and snapshots

Remy:

- scheduled backup/export job
- export critical user/profile/task state before risky upgrades

### 8. Recovery Quality Scoring

Goal:

- recovery should be measurable, not just "it replayed something"

AuraSDK:

- support count/stat inspection needed for verification

Remy:

- compare:
  - records before incident
  - restored records
  - missing candidates after replay
- show confidence of recovery quality

### 9. Safer Upgrade Path

Goal:

- wheel/core upgrades should not silently jeopardize stored memory

AuraSDK:

- explicit compatibility/version metadata in store
- migration guidance or migration helpers

Remy:

- preflight compatibility check before startup
- warn before opening store with new core if migration is needed

---

## Later

These matter, but should not block the main reliability track.

### 10. Background Re-Reading / Reconstruction Intelligence

Goal:

- use history not only for recovery, but for secondary reflection and memory repair

AuraSDK:

- support richer review/promote flows if needed

Remy:

- let agent re-read conversation history
- surface missing durable facts
- propose promotions and merges

### 11. Advanced Multi-Store Strategy

Goal:

- split active runtime state from long-term durable substrate more explicitly

AuraSDK:

- optional future work if architecture evolves there

Remy:

- maybe later separate:
  - active working/cognitive store
  - durable exported facts
  - audit/replay journal

### 12. Self-Healing Maintenance Policies

Goal:

- bounded repair routines for minor inconsistencies

AuraSDK:

- integrity-aware maintenance hooks

Remy:

- maintenance jobs that detect and repair low-risk issues

---

## Split of Responsibility

### Must Be Solved In AuraSDK

- crash resilience of the underlying store
- clearer store error classes
- snapshot integrity
- compatibility/version introspection
- stable export/import primitives

### Must Be Solved In Remy

- safe shutdown orchestration
- startup recovery flow
- incident visibility
- replay/reconstruction workflow
- operator tooling and verification
- upgrade preflight discipline

### Must Be Solved In Both

- recovery validation
- integration tests for failure scenarios
- compatibility handling across versions

---

## Recommended Order

1. crash-safe shutdown and close guarantees
2. startup integrity classification
3. auto-recovery + operator-visible recovery status
4. snapshot policy
5. reliability test harness
6. export layer
7. upgrade compatibility workflow
8. secondary reconstruction intelligence

---

## Success Criteria

This track is successful if:

- abrupt shutdown no longer commonly produces empty restarts
- startup quarantine does not silently erase operator-visible memory state
- recovery is visible, bounded, and measurable
- upgrades become safer and more predictable
- operator trust improves because incidents are explainable and recoverable

---

## Canonical Framing

Do not frame this as:

- "memory is fragile so cognition is a failure"
- "the brain randomly forgets"

Frame it as:

- cognition is ahead of durability
- reliability engineering is now the priority layer
- the next maturity step is persistence guarantees, not just more features
