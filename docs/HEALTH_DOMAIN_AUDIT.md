# Health Domain Decommission Audit

Date: 2026-07-02

## Verdict

Remy no longer exposes the old health/medical layer as the primary product surface. The useful parts have been generalized into neutral workflow primitives, while old names remain only for compatibility and legacy-data visibility.

Remaining cleanup is mostly test/docs hygiene: old medical examples should be neutralized unless they explicitly exercise safety, epistemic governance, or causal-risk behavior.

## Migration Status

First cleanup pass completed on 2026-07-02:

- Public PWA name changed from `Health Secretary` to `Remy`.
- Service worker cache renamed from `health-sec-*` to `remy-*`.
- Public knowledge API moved from `/api/knowledge/health` to `/api/knowledge/metrics`.
- UI knowledge/dashboard labels moved from health cards to metrics.
- New generic runtime tools added: `track_metric`, `metric_summary`, `event_correlate`.
- Autonomy analyst roles now prefer metrics/events instead of health/symptom tools.
- Old tool names remain only as internal compatibility aliases for saved calls and legacy memory migration.
- Fact extraction moved from `tool_handlers.health` to neutral `tool_handlers.facts`; `health.py` is now a thin deprecated wrapper.
- User profile field `health_focus` migrated to `personal_focus`; old metadata is read as a legacy fallback.

Target product frame:

```text
Remy is a local-first AI workflow studio for private automations, documents,
memory, research, reminders, and scheduled workflows on the user's PC.
```

## Initial Findings

These findings were captured before the first cleanup pass. Items marked completed have already been migrated.

### 1. Public Product Surface

These are visible to users or package reviewers and should be cleaned first:

- `src/remy/web/static/manifest.json`
  - Completed: now uses `Remy` as the app name.
- `src/remy/web/static/sw.js`
  - Completed: now uses a `remy-*` cache name.
- `src/remy/web/static/js/chat.js`
  - Completed: visible labels now use metrics/events wording.
- `src/remy/web/static/index.html`
  - Completed: dashboard identifiers now use metrics wording.
- `docs/TZ-02-remy-agent.md`
  - Positions health as a tier-1 feature and describes Health Intelligence.
- `docs/TZ-01-aura-cognitive.md`
  - Uses health/medication/doctor examples in architecture planning.
- `docs/LLM_OPTIMIZATION_RESEARCH.md`
  - Mentions medical/admin note sessions as a target use case.

### 2. Runtime Tools

Before cleanup, the medical layer was wired into executable tooling:

- `src/remy/core/tool_handlers/health.py`
  - Completed: now a thin deprecated compatibility wrapper.
  - Neutral implementations live in `tool_handlers/metrics.py` and `tool_handlers/facts.py`.
  - Legacy tags `health-metric` and `symptom` are read so older user data remains visible.
- `src/remy/core/tool_declarations.py`
  - Completed: declares neutral `track_metric`, `metric_summary`, `event_correlate`.
- `src/remy/core/tool_dispatch.py`
  - Completed: fact extraction imports `tool_handlers.facts`.
  - Deprecated health tool names still dispatch as compatibility aliases.
- `src/remy/core/brain_tools.py`
  - Completed for model-facing declarations; legacy handlers remain for compatibility.
- `src/remy/core/preflight.py`
  - Completed: generic metrics/events group.
- `src/remy/core/turn_classification.py`
  - Completed: generic metrics/events classifications.
- `src/remy/core/autonomy.py`, `src/remy/core/autonomy_models.py`, `src/remy/core/autonomy_critique.py`
  - Completed: autonomy tool selection/prompting uses metrics/events.

The remaining old names are compatibility shims, not default registry/planning surface.

### 3. Web/API Surface

The health-specific API surface was present before cleanup:

- `src/remy/web/api.py`
  - Completed: `/api/knowledge/metrics` is now the public endpoint.
- `src/remy/web/routes/knowledge_routes.py`
  - Completed: `/api/knowledge/metrics` is now the public endpoint.
- `src/remy/web/static/js/api-client.js`
  - Completed: calls `/api/knowledge/metrics?limit=...`.

These should become neutral metrics/event endpoints before public packaging.

### 4. Tests And Examples

Several tests validate health-specific behavior or use medical examples:

- `tests/test_health_intelligence.py`
  - Direct tests for `_track_health_metric`, `_health_summary`, `_symptom_correlate`.
- `tests/test_storage_guards.py`
  - Guard tests for `track_health_metric`.
- `tests/test_workers.py`
  - Expects `health_summary` and `symptom_correlate` in worker tools.
- `tests/test_d01_extract_facts_boundary.py`
  - Completed: imports `_extract_facts` from `remy.core.tool_handlers.facts`.
- Medical scenario examples appear across tests:
  - `patient`, `doctor`, `medication`, `symptom`, `blood pressure`, `warfarin`, `metformin`, `ibuprofen`.

Not all such examples are harmful. Some are stress tests for factuality and causal reasoning. But production-facing examples should use neutral domains unless the test specifically needs a high-risk domain.

### 5. Keep These Uses Of "Health"

Do not remove every `health` occurrence. These are normal engineering terms:

- runtime health
- system health
- tool health
- plan health
- memory health
- health check
- edge health
- `/api/health` for operational status, if used only for system diagnostics

The removal target is the medical/health domain, not infrastructure health terminology.

## Recommended Migration

### Phase 1: Public Cleanup

Rename visible product labels:

- `Health Secretary` -> `Remy`
- `health-sec-*` cache names -> `remy-*`
- Health-specific UI labels -> neutral metrics/events labels.

Archive or mark obsolete:

- `docs/TZ-02-remy-agent.md` Health Intelligence sections.
- Health-specific roadmap claims in old docs.
- Medical/admin positioning in optimization research docs.

### Phase 2: Add Generic Tools

Create a generic handler module:

```text
src/remy/core/tool_handlers/metrics.py
```

Add generic tool names:

```text
track_health_metric  -> track_metric
health_summary       -> metric_summary
symptom_correlate    -> event_correlate
```

Generic semantics:

```text
health-metric tag    -> metric
health data          -> tracked metrics
symptom              -> event
Health Summary       -> Metric Summary
```

The feature remains useful for finance, productivity, fitness, project status, habits, expenses, and any user-defined metric without making medical claims.

### Phase 3: Compatibility Aliases

Keep old tool names as deprecated aliases for one release:

```text
track_health_metric -> track_metric
health_summary -> metric_summary
symptom_correlate -> event_correlate
```

The aliases should:

- emit neutral output;
- preserve existing user data;
- be hidden from default tool lists;
- remain blocked in non-interactive/autonomous channels where the current guard applies.

### Phase 4: API Migration

Add neutral endpoints:

```text
GET /api/knowledge/metrics
GET /api/knowledge/events
```

Remove `/api/knowledge/health` after UI migration.

Status: completed for the public web/API path. Legacy memory records tagged `health-metric` are still read by the metrics endpoint so older user data remains visible.

### Phase 5: Tests

Rename tests to neutral domains:

```text
test_health_intelligence.py -> test_metric_intelligence.py
health metrics             -> project metrics
doctor appointment         -> client meeting
medical note               -> admin note
symptom                    -> event
patient facts              -> user facts
medication reminder        -> recurring task reminder
```

Keep a small number of high-risk-domain tests only where they explicitly verify epistemic governance, causal direction, or safety boundaries.

## Priority

1. Clean PWA/product labels: `manifest.json`, `sw.js`, UI labels.
2. Add `metrics.py` with generic implementations and compatibility wrappers.
3. Update tool declarations, dispatch, preflight, turn classification, autonomy prompts.
4. Add `/api/knowledge/metrics` and migrate frontend calls.
5. Convert tests from medical examples to neutral workflow examples.
6. Archive old health-specific docs.

## Production Risk

Current risk is low-to-medium:

- The health tools are guarded against autonomous fabrication, which is good.
- Public tool names and API endpoints now use neutral metrics/events wording.
- Remaining medical examples in tests/docs may still confuse reviewers browsing the repository.

The right fix is a staged migration, not deletion.
