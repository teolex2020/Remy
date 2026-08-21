# Project MicroBrain Architecture

Status: foundation, conversations, research and generic worker execution,
project artifacts, active MicroBrain visualization, reliability, project
analytics isolation, and the durable fail-closed ownership boundary implemented.

## Decision

Remy has no global cognitive brain. Every project owns exactly one isolated
Aura store called a `MicroBrain`.

In product terms, a project is a complete closed workspace, not a chat. It can
represent any durable area of work, such as product marketing, biological
research, legal analysis, or a software product. A project has its own profile
(name, area, and purpose), and every chat inside it collaborates through the
same project memory and resources.

AuraSDK is an independent cognitive-memory product. It may evolve through
general-purpose memory improvements, but it does not own Remy users, projects,
tenancy, or MicroBrain multiplication; those responsibilities stay in Remy.

```text
Project
  -> Profile (name, area, purpose)
  -> MicroBrain
  -> Conversations (many chats, one shared project memory)
  -> Runs
  -> Documents
  -> Experiments
  -> Pipelines and automations
  -> Artifacts
```

Remy's personality, tools, safety rules, and cognitive algorithms are runtime
capabilities. They are not copied memory records.

## Identity

- `project_id` identifies the product workspace.
- `brain_id` identifies its Aura MicroBrain.
- `brain_provider` is an internal Remy deployment detail. Users do not select
  it when creating projects.
- `brain_locator` is an opaque provider-owned handle. It is a canonical path
  only for `local-aura`; it must never contain an API secret.
- `conversation_id` identifies one chat inside the project.
- `run_id` identifies one execution inside the project.
- `workspace_id` remains a local filesystem capability grant.

Every background receipt and persisted artifact must eventually carry all
applicable identifiers. Missing project identity must fail closed once the
migration is complete.

## Storage

The existing `data/brain` directory is registered as:

```text
project_id = legacy-workspace
brain_id   = brain-legacy-workspace
name       = Legacy Workspace
```

It is not moved, copied, split, or rewritten.

New projects are stored under:

```text
data/projects/<project_id>/brain/
```

The project artifact boundary is `data/projects/<project_id>/`. In desktop
Remy these paths are local; in hosted Remy the same project-to-path mapping
lives on Remy's server. In both deployments Remy creates and owns the separate
Aura instance for every project.

The catalog is stored atomically in `data/projects/index.json`. The currently
selected local project is stored in `data/runtime/active_project.json`.

## Isolation invariants

1. Two projects never share a `brain_id`.
2. A non-legacy brain path must remain inside its project directory.
3. A LangGraph conversation cannot resume under a different project.
4. Switching projects closes the active chat before mounting the next brain.
5. Closing Remy flushes every initialized MicroBrain.
6. Archiving a project does not delete its brain.
7. No external memory framework is introduced.
8. A background continuation is deliverable only to its owning `project_id`
   and `brain_id`, even when the original browser session no longer exists.
9. A missing, unreadable, or stale active-project marker is a hard error. It
   never silently mounts `legacy-workspace`.
10. New execution attempts, completion continuations, and transcript messages
    cannot be written or consumed without both project and MicroBrain identity.

## Project lifecycle

The sidebar project manager exposes one reversible lifecycle:

```text
create -> activate -> rename -> archive -> restore -> activate
```

Archiving the active project first unloads and flushes its live conversation
inside the owning MicroBrain, then activates Legacy Workspace, and only then
marks the project archived. Restore clears the archive marker without changing
`project_id`, `brain_id`, `brain_path`, or any project artifact. There is no
destructive delete operation in the user interface.

## MicroBrain provider contract

`MicroBrainRegistry` resolves the owning project first and then chooses an
explicitly registered opener by `brain_provider`. Normal project creation does
not accept `brain_provider` or `brain_locator`; Remy creates an isolated Aura
store inside the deployment's own data boundary. Extensions can add internal
providers through `register_microbrain_provider()` without exposing storage
selection to product users or coupling AuraSDK to Remy's project model.

If a project names an unavailable provider, memory access fails closed with
`MicroBrain provider is not configured`; it never opens Legacy memory. Features
that require direct `.cog` filesystem access use `local_brain_path()` and fail
explicitly for remote providers. Provider-neutral agent code continues to use
the same `brain` interface returned by the registered adapter.

### Optional Aura HTTP compatibility

Remy retains an advanced adapter for Aura's published v2 HTTP routes:

```text
GET  /health
GET  /stats
GET  /memories
POST /process
POST /retrieve
POST /update
POST /delete
GET  /explain-recall
GET  /memory-health
```

Bearer authentication can come from the deployment-only
`AURA_SERVER_API_KEY` setting. This adapter is not shown in project creation
and is not the mechanism used to multiply project memory.

This upstream profile has two current limitations:

1. One Aura server process owns one storage path.
2. `/process` cannot represent tags, metadata, namespaces, or a selected Aura
   level. Remy rejects such writes instead of silently discarding structure.

Consequently it is only a compatibility integration for Aura's existing core
HTTP feature set. Remy does not require AuraSDK to implement project routing,
tenant routing, or Remy-specific endpoints. Full project memory remains a
Remy-owned responsibility built from ordinary isolated Aura instances.

## MicroBrain host lifecycle

Remy mounts a project's Aura instance lazily on first memory access. The host
keeps a bounded least-recently-used set of mounted brains instead of opening
every project for the lifetime of the process.

- `MICROBRAIN_MAX_OPEN` controls the normal mounted-brain capacity (default 8).
- The active project and every project with an executing bound scope are
  pinned and cannot be evicted.
- An idle least-recently-used brain is flushed and closed before its registry
  entry is removed.
- If all mounted brains are pinned, the host temporarily exceeds its capacity
  rather than interrupting work. It trims again when the bound scope exits.
- If the active-project marker is unreadable, ownership is unknown and
  eviction stops fail-closed.
- A close failure retains the instance and is exposed through
  `microbrain_host.eviction_errors`; Remy never pretends that a failed close
  released the brain.

This lifecycle is implemented entirely inside Remy. It does not require
AuraSDK to understand projects, tenants, or server routing.

## Implemented execution boundary

Research records, execution attempts, completion continuations, and runtime
research notifications carry:

```text
owner_project_id
brain_id
session_id
research project_id
```

`owner_project_id` is deliberately separate from the research subsystem's
historical `project_id`. Existing SQLite ledgers are migrated in place; old
unscoped rows remain available only to `legacy-workspace`.

The research supervisor and scheduler scan every registered MicroBrain, but
bind the owning project before discovery, evidence storage, synthesis,
scheduled automation, pipeline execution, maintenance, or report delivery.
Switching the visible project therefore cannot redirect an active background
job.

Experiment Lab artifacts, pipeline definitions, and workflow run histories use
the same project filesystem boundary. Legacy artifacts remain at their
original `data/experiments`, `data/pipelines`, and `data/workflow_runs` paths.
New projects store them beside their MicroBrain under
`data/projects/<project_id>/`.

Documents, PDF reports, presentations, generated images, browser screenshots,
and session history now use that same project root. Legacy Workspace keeps its
original `data/<artifact-kind>/` locations. Media routes resolve a filename
only inside the currently active project, so an identical filename in another
project cannot be read accidentally.

Glass Brain thermal, plasticity, routing, observation, pruning, and graph views
resolve the active project's `brain_path`; they no longer read the legacy Aura
path unconditionally.

Task metrics, execution logs, and response-evaluation metrics are also owned by
the active or explicitly bound project. New projects store them under:

```text
data/projects/<project_id>/.meta/metrics/
```

Their long-lived Python trackers change storage scope lazily when project
context changes, so background work bound to one project cannot be counted in
another project's dashboard. The Legacy Workspace retains the old metric file
locations. Reliability is derived only from the active project's pipelines and
automations. Provider token accounting remains intentionally server-wide and
is labeled as such in the UI instead of being presented as MicroBrain data.

## Migration stages

1. **Foundation** — project catalog, MicroBrain registry, legacy registration,
   project API, project-bound web session.
2. **Conversations** — multiple chats per project and `conversation_id` as the
   LangGraph thread identity.
3. **Execution** — research, continuation receipts, experiments, scheduler
   jobs, pipelines, automations, and workflow histories are project-scoped.
   Generic workers and both autonomy runtimes retain their creating
   `owner_project_id`/`brain_id` and rebind that project for every cycle.
4. **Data surfaces** — documents, history, artifacts, Glass Brain,
   reliability, task/run metrics, and response-evaluation statistics are
   scoped to the active project. Server-wide resource counters are explicitly
   labeled rather than mixed into project ownership.
5. **Fail closed** — the active-project marker, durable execution ledger,
   continuation inbox, and transcript store reject missing ownership instead
   of falling back to Legacy. Existing pre-migration unscoped rows are
   readable only through the explicit Legacy compatibility path.
6. **Deployment boundary** — desktop and hosted Remy both multiply isolated
   Aura instances inside Remy. Public project creation cannot override the
   memory provider or locator. The optional Aura HTTP adapter remains an
   advanced compatibility surface and does not define product architecture.
