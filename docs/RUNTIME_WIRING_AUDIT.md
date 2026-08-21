# Runtime Wiring Audit

Date: 2026-06-29

Goal: keep the desktop production app as one coherent local system. Files that are
not wired into the runtime should be either integrated, moved to dev-only
validation, or removed with their tests.

## Confirmed Runtime Wiring

- Desktop entry point: `remy-app = remy.desktop_entry:main`
- CLI entry point: `remy = remy.main:main`
- Desktop/web app factory: `remy.core.desktop_gui.create_app`
- Split web routes are now declared in `remy.core.desktop_gui.ROUTE_MODULES`.
- `tests/test_local_desktop_app_contract.py` checks that every public
  `src/remy/web/routes/*.py` module with a `router` is included in the desktop app.

## Keep In Runtime

These modules are wired into the app, tool runtime, or web routes:

- `remy.core.agent_tools`
- `remy.core.brain_tools`
- `remy.core.tool_dispatch`
- `remy.core.tool_declarations`
- `remy.core.tool_registry_mgmt`
- `remy.core.pii_vault`
- `remy.core.bulk_ingestion`
- `remy.core.corpus_preprocessor`
- `remy.core.autonomy_benchmarks`
- `remy.core.autonomy_live_validation`
- `remy.web.routes.*`

## Tested But Not Runtime-Wired

These modules have tests or scripts, but no production importer in `src/remy`.
They are not necessarily wrong, but they are not part of the current desktop
agent path.

- `remy.core.preflight`: integrated on 2026-06-29 as a non-blocking autonomy
  prompt/status layer. `AutonomousLoop._decide_and_act` runs zero-LLM
  preflight before worker dispatch, injects formatted warnings into the
  decision prompt, and exposes the latest snapshot via autonomy status.
- `remy.core.loop_detection`: integrated on 2026-06-29 as a non-blocking
  autonomy warning/status layer. `AutonomousLoop._decide_and_act` fingerprints
  actual tool calls after worker dispatch, stores the latest detection in
  autonomy status, emits `loop_detection` events for non-empty warnings, and
  injects the prior warning into the next decision prompt. Skip/backoff
  enforcement remains a separate policy step.
- `remy.core.plan_invalidation`: integrated on 2026-06-29 as a plan
  health/confidence layer after each autonomy cycle. It now works structurally
  with runtime plan classes, exposes latest plan health in autonomy status,
  emits `plan_health` events for non-continue recommendations, inserts
  prerequisite steps for explicit setup/dependency blockers, and abandons plans
  that cross the failure threshold. Existing `advance_plan` still owns normal
  step advancement and LLM replanning.
- `remy.core.confidence_autonomy`: integrated on 2026-06-29 as a non-blocking
  autonomy policy/status layer. `AutonomousLoop._decide_and_act` records domain
  outcomes after evaluation, assesses confidence from domain familiarity,
  budget health, tool health, goal clarity, and recent outcomes, exposes the
  latest recommendation in autonomy status, and emits `confidence_policy`
  events for non-silent recommendations. It does not enforce guidance/skip in
  the local desktop product.
- `remy.core.turn_classification`: integrated on 2026-06-29 as a zero-LLM
  observability layer. `AutonomousLoop._decide_and_act` classifies each cycle
  from actual tool calls, stores it on `ActionRecord`, and passes it to
  `record_cycle_execution`.

Decision needed: decide whether any currently non-blocking autonomy warnings
should become hard runtime gates after manual product testing.

## Dev/Validation Code Outside Runtime Package

These manual validation, authoring, and standalone maintenance runners were
moved out of `src/remy/core` on 2026-06-29 so the packaged executable does not
include them as runtime modules:

- `tools.validation.thermal_accelerated_soak`
- `tools.validation.thermal_high_signal_tests`
- `tools.validation.thermal_longitudinal_soak`
- `tools.validation.thermal_runtime_validation`
- `tools.maintenance.maintenance_cron`
- `tools.authoring.knowledge_harvester` (used by `scripts/harvest_security_ops.py`)

Recommended cleanup: keep them out of the executable build unless an operator
explicitly packages a maintenance/authoring edition.

## Duplicate Or Orphan Candidates

- `remy.core.session_summary`: consolidated on 2026-06-29. Runtime still
  imports `remy.core.brain_tools.generate_session_summary` for compatibility,
  but that function now delegates to the shared module.
- `remy.core.autonomy_outcomes`: consolidated on 2026-06-29. `remy.core.autonomy`
  now exposes `ActionRecord`, `record_outcome`, and `recall_similar_outcomes`
  from this module.
- `remy.core.tool_handlers.delegate`: consolidated on 2026-06-29. Runtime still
  calls `remy.core.brain_tools._handle_delegate_task` for compatibility, but
  that function now delegates to the modular handler.
- `remy.core.tool_handlers.browser_dispatch`: consolidated on 2026-06-29.
  Runtime still calls `remy.core.brain_tools._handle_browser_tool` for
  compatibility, but browser dispatch, browser actions, close handling, and
  browser error analysis now delegate to this modular handler. Mutable circuit
  breaker state remains in `brain_tools` because tests and existing callers
  patch it there.
- `remy.core.pii_vault`: integrated on 2026-06-29 as the default LLM-boundary
  privacy layer. `remy.core.agent.call_model` shields LangChain messages before
  the LLM call and restores AI response text/tool-call args locally before
  runtime execution.

Recommended cleanup: either replace the runtime copies with these extracted
modules, or delete the orphan modules and their direct tests. Do not keep both
paths in production.

## Aura-clean Cross-Check

Checked against `D:\Aura-clean` on 2026-06-29.

`D:\Aura-clean` is a Rust-heavy Aura laboratory/runtime, not a Python Remy
package. The suspicious Python modules in this repo are therefore not direct
file copies: exact stem matching found only `feedback`
(`remy.core.tool_handlers.feedback` vs `src/cognition/feedback.rs`).

The transfer is architectural rather than literal. Strong semantic matches:

- `remy.core.consequence_gate` maps to Aura consequence/runtime gates such as
  `src/runtime/consequence_vsa_memory_gate.rs`.
- `remy.core_v3.runtime.loop_runtime` maps to
  `src/runtime/brain_runtime_loop.rs` and
  `src/runtime/brain_runtime_loop_receipt_store.rs`.
- `remy.core_v3.runtime.recovery_runtime` maps to
  `src/runtime/runtime_blocker_recovery_read.rs` and brain recovery receipt
  modules.
- `remy.core_v3.runtime.lifecycle_runtime` maps to
  `src/runtime/runtime_session_lifecycle_read.rs`.
- `remy.core_v3.runtime.learning_runtime` maps to
  `src/runtime/learning_runtime_cycle.rs`.
- `remy.core_v3.runtime.specialist_runtime` and
  `specialist_inference_runtime` map to Aura `src/specialist/*runtime*`
  modules.
- `remy.core_v3.runtime.evidence_debt_runtime` maps to Aura claim evidence
  debt modules.
- `remy.core.tool_registry*` maps conceptually to
  `src/runtime/tool_affordance_registry.rs`.

Conclusion: do not treat all unmatched modules as junk. Some are ports of Aura
organs into the Python agent wrapper. For production, the criterion should be:
is the organ wired into `remy-app`/desktop runtime, or is it still an
experimental import with tests only?

## Next Cleanup Order

1. Decide whether `loop_detection` should move from prompt/status warning to
   hard skip/backoff enforcement in production.
2. Run the full suite after each group.
