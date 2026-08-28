# DeepSeek Harness Adaptation Plan

Status: active
Updated: 2026-08-22

This is the canonical backlog for adapting the useful DeepSeek Harness runtime
contracts to Remy. Trajectory observability is part of the work, but it does not
replace the runtime foundation below.

## Ordered backlog

1. **Durable SessionEventStore and projections — complete**
   - Completed slice: append-only, hash-addressed event revisions are written in
     the same SQLite transaction as each Trajectory mutation.
   - Completed slice: existing trajectory rows are adopted through an idempotent
     `event.imported` backfill.
   - Completed slice: transcript, tool-run, metrics, recovery, and model-history
     projections plus materialized-projection drift verification.
   - Completed slice: transcript recovery can reconstruct user/assistant history
     from the journal when the legacy transcript database is unavailable.
   - Completed slice: chat-launched pipelines now emit correlated `PIPELINE_RUN`,
     `PIPELINE_STEP`, `PIPELINE_ROUTE`, and `PIPELINE_RESULT` records with redacted
     inputs/outputs, definition hashes, Run Envelope provenance, and direct UI links.
   - Completed slice: bounded checkpoint observations, execution attempts with
     receipt history, and project-scoped critical-action audits now dual-write
     through the same hash-verified append contract.
   - Checkpoint diagnostics, scoped audit APIs, and project execution inspection
     read projections first with explicit legacy fallback. LangGraph checkpoints
     and execution coordination tables remain operational materializations for
     resume, locking, and atomic claims rather than competing history sources.
   - Parity tests cover projection-first reads after legacy drift, receipt replay,
     secret redaction, checkpoint recovery metadata, and both JSONL/event-log integrity.

2. **Formal ToolPipeline over the current dispatcher — complete**
   - `resolve -> validate -> provenance -> pre-policy -> monotonic guards ->`
     `approval -> execute -> post-policy -> artifact spill -> durable observation`
   - Both legacy `brain_tools` and direct dispatcher calls now use the same
     ordered entrypoint while preserving handler lock boundaries.
   - Every run produces a privacy-safe receipt, persists large raw results as
     project artifacts, and appends its stages to the session event stream.
   - Interactive approval is classified and paused once by the pipeline before
     handler execution. An approved invocation receives a context-local grant bound
     to the exact tool and argument hash, so compatibility handlers cannot prompt
     twice or reuse approval after any argument change. Missing approval execution,
     denial, timeout, and classifier failure all fail closed; receipts expose only
     outcome and target/description hashes.

3. **Provenance-aware monotonic guards — complete**
   - A deny from permissions, security, or consequence memory can never be
     weakened by a later plugin or policy stage.
   - Consequence memory, circuit availability, sensitive-action provenance, and
     workspace capabilities now deny before handler execution.
   - Workspace handlers revalidate capabilities immediately before I/O as a
     deliberate defense against grant revocation and path changes after preflight.
   - Guard receipts contain decisions and hashes, not raw tool arguments or
     sensitive values.

4. **Scoped PluginContext with guaranteed dispose — complete**
   - Track tools, listeners, services, and prompt sections registered by a plugin.
   - Unload in reverse order and prove cleanup with failure-injection tests.
   - Plugin setup is atomic: partial registrations roll back before the plugin is
     published, and resource-name collisions cannot replace another owner.
   - Cleanup failures are reported but do not stop later disposers; runtime
     shutdown unloads all plugins in reverse load order.

5. **Profiles and bundle overlays — complete**
   - Complete: dynamic, validated overlays compose channel defaults, capability
     packs, skills, and session-enabled tools in a deterministic order.
   - Built-in `standard`, `research`, `read_only`, and `operator` profiles bind
     the same resolved tool set and prompt sections to each model request.
   - Profile ceilings and explicit denies run after every additive layer, so a
     skill or session tool cannot widen `read_only` or bypass ToolPipeline policy.
   - Profile and bundle identity remain explicit in agent state and Trajectory
     context provenance; switching profiles does not leak profile tools into the
     session-enabled set.

6. **Continuable child sessions — complete**
   - Stable `child_id` now spans multiple run attempts and remains bound to its
     owner project, MicroBrain, and parent conversation.
   - A durable bidirectional inbox preserves initial instructions, queued
     follow-ups, interrupts, child reports, and consumption state.
   - Follow-ups received during active work continue automatically at the next
     safe attempt boundary; interrupt is cooperative and preserves cold-resume state.
   - Every attempt produces an idempotent settlement, attempt history, parent
     continuation, and `CHILD`/`SETTLEMENT` record in the Trajectory event stream.
   - Resuming executor children requires explicit side-effect confirmation; an
     interrupted process is settled without replay and remains explicitly resumable.

7. **Safe PTC/workflow pilot — complete**
   - Added a deliberately small JSON program format: sequential typed steps and
     `$ref` links to prior results only, with deterministic hashes and static validation.
   - The explicit local allowlist contains read-only memory, transcript, workspace,
     todo/metric summary, child-report, and pipeline-candidate inspection tools.
   - Browser/network actions, finance, shell/code execution, writes, updates,
     deletes, arbitrary Python, loops, branches, and sensitive sinks are rejected
     before a run starts.
   - Hard call-count, wall-clock, result-size, structure-depth, node-count, and
     program-size limits constrain the pilot; every accepted step still executes
     through the canonical `ToolPipeline`.
   - Agent tools and `/api/ptc/*` routes expose catalog, validation, and execution.
     Each run receives a read-only Run Envelope plus step-level hashes, bounded
     results, pipeline receipts, and a directly inspectable `PTC` Trajectory event.

8. **User-activated Dynamic Agent Teams v1 — complete**
   - Chat exposes `off | adaptive | force`: `off` is the default, `adaptive`
     persists only by explicit user choice, and `force` applies to one task then resets.
   - A dedicated internal planner prompt emits a typed team proposal. A deterministic
     gate validates roles, member count, instructions, budgets, and immutable tool ceilings
     before any worker is created; enabling team mode grants permission, not an obligation.
   - Adaptive mode can select the single-agent path for simple work. Accepted teams contain
     2-3 parallel researcher/analyst/planner/OSINT workers with six tool steps each, bounded
     time/output, delegation depth zero, and no executor, browser action, finance, shell,
     code execution, memory/file writes, self-tooling, or model-supplied capabilities.
   - Results fan in as explicitly untrusted internal evidence for the main agent's final
     synthesis. `TEAM_PLAN`, `TEAM_GATE`, and `TEAM_RESULT` events expose the entire decision
     path in Trajectory, and each team receives a read-only Run Envelope.

9. **Safe Self-modification Lab foundation — complete**
   - Agent Teams still cannot mutate runtime prompts, policy, tools, code, approval,
     or sandbox behavior. The Lab accepts only a bounded additive `agent.guidance`
     overlay; base instructions and capability ceilings remain immutable.
   - Every candidate is immutable and hash-addressed against the currently active
     baseline. A green matrix is accepted only when sandbox replay actually ran
     with `agent_version=self-mod:<candidate_hash>` across at least three cases.
   - Promotion requires exact candidate-hash confirmation by an operator, then a
     stable 5-25% session canary. Failure, unsupported-claim, request-count, and
     latency gates automatically roll back a regressed canary.
   - Promotion is project-scoped and reversible. Rolling back an active overlay
     restores its previous version; all lifecycle transitions appear as redacted
     `SELF_MOD_*` records in Trajectory.
   - Experiments now exposes the lifecycle as a lazy-loaded Self-improvement Lab:
     proposal/version hashes, durable eval-case selection, explicit approval,
     bounded canary metrics, promotion/rollback, and the shared Trajectory inspector.
   - Canary metrics are derived automatically from production Trajectory events:
     stable candidate/control cohorts, request failures, latency, and factuality
     verification coverage. Raw prompts and responses never enter the aggregate,
     and a ready regressed gate rolls back automatically at turn completion.
   - Promotion now requires at least 20 terminal requests per cohort, a five-minute
     observation window, at least 80% verification coverage, and 95% confidence
     intervals that satisfy the non-inferiority gates. An explicit hard regression
     can still roll back after the five-request safety floor; statistically
     inconclusive results remain in canary and continue collecting evidence.
   - Those defaults are now a validated project-scoped Canary policy. Operators can
     tune sample floors, observation and stale-inconclusive windows, verification
     coverage, confidence, quality margins, latency tolerance, and alert delivery
     without changing other projects. Regression and long-running inconclusive
     signals are durably deduplicated in Incident Center, resolve on recovery, and
     open the exact Self-improvement Lab proposal.
   - Automatic source-code/worktree mutation remains out of scope. It would need
     a separate disposable-worktree executor, security test suite, signed review,
     and artifact promotion contract before becoming eligible for the Lab.

10. **Cross-surface execution Trajectory — complete**
   - Chat-attached pipelines retain their causal link to the originating turn.
   - Experiments and Automations use project-scoped synthetic execution sessions,
     so observability does not create fake conversations or pollute chat history.
   - Experiment runs expose phases/checkpoints, model contributions, synthesis
     decisions, scenario interventions, and the terminal result.
   - Automation runs expose the trigger, ordered steps and retry recovery, output
     delivery, and the terminal result; scheduled and manual runs share the contract.
   - Experiment details, Automation run results, and Automation history open the
     shared read-only Trajectory timeline and right-side inspector for an exact run.
   - Execution payloads reuse the bounded redaction layer, and telemetry failures
     remain isolated from the workflow being observed.

11. **Virtualized Trajectory and performance guardrails — complete**
   - The ledger switches to bounded DOM windowing after 100 visible records with
     twelve-row overscan, stable top/bottom spacers, and animation-frame scroll updates.
   - Semantic event identity, selection, replay navigation, inspector opening, and
     causal filters remain independent of whether a row is currently mounted.
   - Loading an older server window preserves the former first-record anchor instead
     of jumping the user to a different part of the execution history.
   - Browser regressions cover 5,000-record DOM bounds and mount budget, distant
     selection, tail reachability, and anchor preservation after older-page prepend.

12. **Token-aware context compaction and overflow recovery — complete**
   - Prompt size is estimated from the complete structured message payload and
     checked against the concrete model's learned, configured, or conservative
     provider budget; the normal trigger is 80% of usable context and the target
     is 55%, with output capacity reserved separately.
   - A bounded recent suffix and intact assistant/tool-result pairs survive the
     reduction. Artifact-backed tool results remain references; very large
     unbacked results retain their head and tail with an explicit omission marker.
   - The legacy message-count contract remains only as a secondary state-growth
     ceiling for extremely fragmented conversations, not the primary budget signal.
   - Each fallback model receives a fresh budget check. Provider context-overflow
     errors trigger one immediate stricter compaction retry instead of generic
     backoff, using a 35% target by default.
   - Every actual reduction emits a privacy-safe `COMPACTED` Trajectory record
     with model, reason, estimated tokens before/after, limits, retained counts,
     and overflow-retry state, never raw prompt or tool-result text.

13. **Category-based Settings workspace — complete**
   - The former single scrolling page is organized into Overview, AI & Models,
     Personalization, Connections, and Workspace & Data panels while preserving
     every existing form, element identity, API call, and asynchronous status view.
   - Only one category is visible at a time. The active category is durable across
     reloads, keyboard tabs support arrows/Home/End, and narrow screens switch to
     a horizontally scrollable category strip without overflowing the workspace.

14. **Resizable application sidebar — complete**
   - Desktop users can drag the navigation boundary between 184 and 420 pixels;
     the chosen width persists across reloads and double-click restores the
     compact 224-pixel default.
   - The separator exposes its current value to assistive technology and supports
     Arrow keys, Shift-modified steps, Home, and End. On mobile the resize affordance
     is hidden and the remembered width is capped so the close area stays reachable.

15. **Non-blocking chat startup and data loading — complete**
   - Incident-center badges use a lightweight alerts endpoint instead of eagerly
     requesting the full control-plane status during every application startup.
   - Full system status has one canonical route, runs the runtime snapshot outside
     the event loop, and serves a short-lived stale-while-revalidate cache.
   - Transcript reads reuse an initialized SQLite store, avoid repeated WAL setup,
     run outside the event loop, and expose a bounded browser deadline instead of
     leaving the conversation surface in an infinite loading state.

16. **Agent-owned LabWorkflowPlan v2 and cognitive ledgers — foundation complete**
   - Every prepared Agent Lab run now receives a validated JSON DAG with explicit
     dependencies, role and model slots, workspace authority, artifact inputs,
     output schemas, deterministic success gates, retry ownership, and policy-bounded
     parallel/total-agent caps. Cycles, unknown fields, unavailable models, disallowed
     tools, path escapes, and permission expansion fail before execution.
   - The original list-shaped plan remains a synchronized compatibility projection;
     existing prepared checkpoints are upgraded lazily without losing node status.
   - A durable Task Ledger owns node states, success criteria, blockers, facts,
     assumptions, and artifact hashes. The bounded Progress Ledger records phase
     snapshots, completed/active/blocked/next nodes, explicit replan reasons, and
     deterministic state fingerprints for future loop detection.
   - Manual and autonomous execution update the same ledgers, while Agent Lab exposes
     both ledgers and v2 node contracts in its observer UI.
   - Private builder snapshots and the conflict/merge gate are now connected:
     autonomous proposals stage only under run-owned `src/` and `tests/` copies,
     and enter the canonical workspace only after baseline-hash, current-hash,
     candidate-hash, source-policy, path, file-count, and disk-budget checks pass.
     Stale snapshots fail closed with a durable blocker and merge receipt.
   - Verification is now authored in a separate model call with a clean prompt that
     contains observable source/artifact evidence but no Builder reasoning or tests.
     Automatic routing prefers a different connected model and records `cross_model`;
     a one-model installation remains explicit as `isolated_context_same_model`.
     Verifier files pass through their own private snapshot and merge receipt, then run
     with filesystem mutation blocked at both `open` and `os` boundaries.
   - Accepted, rejected, and inconclusive outcomes export `proof-pack.json` and
     `proof-pack.md`: plan/model assignments, merge diffs and tree hashes, bounded
     stdout/stderr receipts, read-only verification evidence, artifact hashes, and the
     final decision. Prompts, hidden reasoning, and verifier rationale are excluded.
   - Complex specialist-backed runs can now activate strict Builder fan-out. The
     central scheduler assigns two or three connected models exact, non-overlapping
     `src/*.py` claims before any Builder call. All builders work concurrently from
     the same canonical baseline in separate snapshots; invalid, missing, extra, or
     overlapping files fail closed and preserve every snapshot for inspection.
   - Fan-in is deterministic by Builder id and each claim moves through an explicit
     `active` to `merged`, `conflict`, `failed`, or `cancelled` lifecycle. Local imports
     are allowed only for claimed and independently source-validated workspace modules.
     Trajectory, Agent Lab, and Proof Pack expose assignments, snapshots, exact claims,
     served models, merge order, and receipts without source content or hidden reasoning.
   - Snapshot retention is now operator-visible and policy-bounded: Agent Lab reports
     per-snapshot and aggregate disk use, protected/open/merged/conflict counts, and the
     configured aggregate budget without reading source content into the UI. Closed
     merged snapshots can be cleaned in one action; conflict snapshots require a
     separate explicit confirmation, while running runs, open workspaces, active claims,
     malformed metadata, and boundary-invalid targets always fail closed.
   - Cleanup never removes canonical files, claims, merge receipts, or branch history.
     Every attempt records recovered bytes and exact workspace ids in the run ledger and
     Trajectory; subsequent Proof Packs include the bounded cleanup receipts.
   - Agent Lab now offers an explicit `container_required` runtime alongside the
     compatible bounded-process mode. Container-required runs preflight before any
     model call or workspace mutation and never silently downgrade when Docker/Podman,
     a local engine, or the configured local image is unavailable.
   - The container command denies network access and image pulls, uses an immutable
     preflighted `sha256:` image ID, rejects remote Docker contexts, drops every Linux
     capability, enables no-new-privileges, runs as a non-root UID, limits PID/RAM/CPU,
     provides a bounded no-exec tmpfs, exposes a read-only root filesystem, and mounts
     only the run-owned workspace. Independent verification mounts even that workspace
     read-only. Timeout and cancellation perform a separate idempotent container teardown.
   - Isolation mode, engine, image tag and immutable image ID, plus the enforced
     security contract are recorded in execution receipts, Trajectory, Agent Lab, and
     Proof Pack. The runtime image recipe is local and dependency-free; Remy never
     downloads it or contacts a registry without an explicit operator action.
   - LabWorkflowPlan v2 implementation phases are complete. Future hardening should be
     driven by measured escape tests and platform-specific container availability.

## Safety constraints carried through every phase

- Preserve provenance and trust tier for every external result.
- Sensitive sinks require independent authorization and approvals.
- Keep consequence memory in the decision path.
- Maintain regression matrices for web, documents, skills, metadata, hidden
  Unicode, and indirect prompt injection.
- Do not rewrite Remy in TypeScript or adopt experimental Agent Teams as the
  primary runtime.

## Current implementation slice

- `src/remy/core/tool_contracts.py` compiles the five-section Tool Contract v2
  selection contract into both runtime declaration catalogs. Its synthetic
  routing suite measures exact-tool accuracy and forbidden-tool gravity; the
  optional live adapter permits model selection calls but never executes tools.
- `src/remy/core/tool_middleware.py` owns the ordered synchronous control chain
  around tool calls. Trusted handlers may pass, patch arguments before policy,
  block monotonically before execution, or transform results after execution.
  Before-handler failures default to fail-closed; after-handler failures preserve
  the real result so an already-completed side effect is not retried. Receipts
  contain handler decisions and modified key names, never argument/result values.
- `src/remy/core/event_bus.py` remains a lossy, fire-and-forget UI observation
  surface and is deliberately excluded from control-flow and authorization.
- `src/remy/core/tool_pipeline.py` owns the ordered execution contract and
  monotonic decision primitive, including middleware-before and middleware-after
  stages around policy, approval, execution, and durable observation.
- `src/remy/core/tool_dispatch.py` adapts existing handlers to the pipeline and
  appends privacy-safe policy observations.
- `src/remy/core/brain_tools.py` now delegates its compatibility entrypoint to
  the canonical dispatcher.
- `src/remy/core/agent.py` attaches pipeline receipts and artifact references to
  the corresponding Trajectory `TOOL` event.
- `src/remy/core/workspace_permissions.py` exposes a non-executing capability
  preflight while retaining defense-in-depth checks at the I/O boundary.
- `src/remy/core_v3/integrations/plugin_context.py` owns scoped plugin tools,
  listeners, services, prompt sections, and custom cleanup callbacks.
- `src/remy/core_v3/integrations/registry.py` performs atomic setup, rollback,
  reverse-order unload, and read-only execution-context binding.
- `src/remy/core/capability_overlays.py` validates profiles and pack/skill
  bundles, resolves deterministic tool ceilings, and renders bounded prompt
  sections without weakening downstream safety policy.
- `src/remy/core/agent.py` uses the same overlay for prompt context and actual
  tool binding, and records its hash as Trajectory policy provenance.
- `src/remy/core/child_sessions.py` owns stable child identity, atomic attempt
  claims, durable inbox messages, reports, and idempotent settlements.
- `src/remy/core/worker_tasks.py` maps child generations onto existing run
  envelopes, performs safe-boundary follow-up, interrupt, recovery, and delivery.
- `src/remy/web/routes/run_routes.py` and the agent tool catalog expose project-
  scoped child list, report, follow-up, interrupt, and cold-resume operations.
- `src/remy/core/trajectory_store.py` owns redacted Experiment and Automation
  run/event/result records alongside the existing chat and Pipeline records.
- `src/remy/core/session_event_store.py` owns the generic append contract plus
  checkpoint, execution-ledger, and critical-audit projections.
- `src/remy/core/execution_ledger.py`, `src/remy/core/durable_graph_runtime.py`,
  and `src/remy/core/audit_trail.py` retain their coordination/materialized stores
  while publishing projection-safe revisions and preferring the shared journal
  for project-scoped inspection.
- `src/remy/web/routes/trajectory_routes.py` exposes project-scoped read-only
  execution projections consumed by the shared timeline and inspector.
- `src/remy/web/static/js/trajectory.js` keeps large causal ledgers scrollable while
  mounting only the viewport plus overscan; the browser regression suite enforces
  DOM, mount-time, semantic-selection, tail, and prepend-anchor contracts.
- `src/remy/core/context_compaction.py` owns model-budget resolution, deterministic
  token estimation, recent-window preservation, tool-result pruning, overflow
  classification, and the metadata contract consumed by Trajectory.
- `src/remy/core/llm.py` rechecks the prompt for every primary/fallback model and
  performs the bounded context-overflow recovery before ordinary error handling.
- `src/remy/core/self_modification_lab.py` owns immutable additive-prompt proposals,
  candidate-bound sandbox evals, exact-hash approval, deterministic canary routing,
  promotion, automatic regression rollback, and previous-version restoration.
- `src/remy/web/routes/experiment_routes.py` exposes the project-scoped Lab lifecycle
  inside Experiments; `src/remy/core/agent.py` resolves only approved canary/active
  overlays and labels their cohort in model telemetry.
- `src/remy/core/agent_lab_workflow.py` owns the non-executable DAG compiler plus
  Task/Progress Ledger contracts; `src/remy/core/agent_lab.py` persists their synchronized
  run projection and migration, and the coordinator/routes advance shared node state.
- `src/remy/core/agent_lab_workspace.py` owns run-local builder snapshots, aggregate
  retention bounds, three-way hash conflict detection, rollback-safe promotion, and
  durable merge receipts consumed by Agent Lab and Trajectory.
- `src/remy/core/agent_lab_builders.py` validates central Builder fan-out, connected
  model assignments, exact non-overlapping source claims, and claim-complete shard
  responses before parallel execution or deterministic fan-in can proceed.
- `src/remy/core/agent_lab_proof.py` creates the privacy-bounded JSON/Markdown evidence
  packet and stable hashes used to audit an acceptance or non-acceptance decision.
- `src/remy/core/agent_lab_container.py` owns local-only runtime discovery, remote-context
  rejection, immutable image identity, the hardened Docker/Podman argument contract, and
  idempotent teardown. `packaging/agent-lab-runtime/` contains the minimal local image recipe.
- `src/remy/core/agent_lab_backends.py` defines the shared launch/teardown protocol and
  provenance handle used by bounded-process and container-required execution. The executor
  now owns one validation, monitoring, cancellation, receipt, and artifact lifecycle while
  isolation-specific command construction remains replaceable and fail-closed.
- `src/remy/core/agent_lab_backend_registry.py` is the single backend catalog for execution,
  coordinator preflight, preparation, API discovery, and the dynamic UI selector. New modes
  register a descriptor, factory, probe, and optional explicit preparation action; host paths
  and environments are removed from public status receipts.
  Each descriptor now declares language, artifact, network, GPU, read-only verification, and
  isolation capabilities. Automatic selection consumes a strictly validated requirement set,
  chooses the least-privileged available match deterministically, and records every rejected
  candidate; an explicitly requested backend never falls back.
