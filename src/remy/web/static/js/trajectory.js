/** Conversation Trajectory — causal ledger, timeline, and local record inspector. */

const chatTab = document.getElementById("chat-surface-chat");
const trajectoryTab = document.getElementById("chat-surface-trajectory");
const chatPanel = document.getElementById("chat-surface-chat-panel");
const trajectoryPanel = document.getElementById("chat-surface-trajectory-panel");
const ledger = document.getElementById("trajectory-ledger");
const overview = document.getElementById("trajectory-overview");
const status = document.getElementById("trajectory-status");
const windowStatus = document.getElementById("trajectory-window-status");
const loadOlderButton = document.getElementById("trajectory-load-older");
const search = document.getElementById("trajectory-search");
const kindFilter = document.getElementById("trajectory-kind-filter");
const problemsOnly = document.getElementById("trajectory-problems-only");
const bookmarksOnly = document.getElementById("trajectory-bookmarks-only");
const integrityButton = document.getElementById("trajectory-integrity-health");
const turnFilter = document.getElementById("trajectory-turn-filter");
const replayPlay = document.getElementById("trajectory-replay-play");
const replayPosition = document.getElementById("trajectory-replay-position");
const turnSummary = document.getElementById("trajectory-turn-summary");
const chainSummary = document.getElementById("trajectory-chain-summary");
const chainClear = document.getElementById("trajectory-chain-clear");
const comparePanel = document.getElementById("trajectory-compare-panel");
const healthButton = document.getElementById("trajectory-health");
const diagnosticsPanel = document.getElementById("trajectory-diagnostics");
const inspector = document.getElementById("trajectory-inspector");
const inspectorKind = document.getElementById("trajectory-inspector-kind");
const inspectorTitle = document.getElementById("trajectory-inspector-title");
const inspectorTabs = document.getElementById("trajectory-inspector-tabs");
const inspectorContent = document.getElementById("trajectory-inspector-content");
const liveStatus = document.getElementById("trajectory-live");
const analyticsButton = document.getElementById("trajectory-analytics-open");
const analyticsPanel = document.getElementById("trajectory-analytics");
const analyticsContent = document.getElementById("trajectory-analytics-content");
const analyticsDays = document.getElementById("trajectory-analytics-days");
const baselineSelect = document.getElementById("trajectory-baseline-select");
const baselineDelete = document.getElementById("trajectory-baseline-delete");
const alertCount = document.getElementById("trajectory-alert-count");

let activeConversationId = "";
let externalTrajectoryUrl = "";
let externalTrajectorySessionId = "";
let externalTrajectoryTitle = "";
let trajectoryPayload = null;
let records = [];
let selectedId = "";
let selectedTab = "Summary";
let loading = false;
let refreshQueued = false;
let refreshTimer = null;
let liveConnectionState = "connecting";
let liveRefreshBlockedBy = null;
let focusedTurnId = "";
let replayIndex = -1;
let replayTimer = null;
let replayPlaying = false;
let causalRootId = "";
let compareOpen = false;
let compareBaseId = "";
let compareTargetId = "";
let timelineMode = "duration";
let trajectoryLimit = 750;
let pendingScrollAnchor = "";
let timelineTotalCount = 0;
let timelineRenderedCount = 0;
let analyticsPayload = null;
let analyticsLoading = false;
let analyticsRefreshTimer = null;
let pendingAnalyticsEventId = "";
let ledgerViewCache = null;
let ledgerViewGeneration = 0;
let ledgerRenderSignature = "";
let ledgerScrollFrame = 0;
const LEDGER_VIRTUALIZATION_THRESHOLD = 100;
const LEDGER_VIRTUAL_OVERSCAN = 12;
const LEDGER_RECORD_HEIGHT = 42;
const LEDGER_DIVIDER_HEIGHT = 25;
const TIMELINE_BAR_LIMIT = 1200;
const INSPECTOR_WIDTH_KEY = "remy_trajectory_inspector_width_v1";

const KIND_GROUPS = {
    SYSTEM: "input", USER: "input", CONTEXT: "input", FORK: "input",
    REQUEST: "model", ATTEMPT: "model", ASSISTANT: "model", COMPACTED: "model",
    TOOL: "tools", SUBTOOL: "tools", MEMORY: "tools", POLICY: "tools",
    APPROVAL: "tools", VERIFICATION: "tools", SUBAGENT: "tools",
    CHILD: "tools", SETTLEMENT: "tools",
    PIPELINE_RUN: "pipeline", PIPELINE_STEP: "pipeline",
    PIPELINE_ROUTE: "pipeline", PIPELINE_RESULT: "pipeline",
    EXPERIMENT_RUN: "experiment", EXPERIMENT_PHASE: "experiment",
    EXPERIMENT_MODEL: "experiment", EXPERIMENT_DECISION: "experiment",
    EXPERIMENT_INTERVENTION: "experiment", EXPERIMENT_RESULT: "experiment",
    AUTOMATION_RUN: "automation", AUTOMATION_TRIGGER: "automation",
    AUTOMATION_STEP: "automation", AUTOMATION_DELIVERY: "automation",
    AUTOMATION_RESULT: "automation",
};

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function textValue(value) {
    if (value == null) return "";
    if (typeof value === "string") return value;
    if (value && typeof value === "object" && "content" in value) {
        return textValue(value.content);
    }
    return JSON.stringify(value, null, 2);
}

function formatDuration(ms) {
    const value = Number(ms || 0);
    if (value < 1000) return `${Math.round(value)} ms`;
    if (value < 60000) return `${(value / 1000).toFixed(value < 10000 ? 2 : 1)} s`;
    return `${Math.floor(value / 60000)}m ${Math.round((value % 60000) / 1000)}s`;
}

function formatTime(value) {
    if (value == null || value === "") return "—";
    const date = new Date(Number(value) * 1000);
    if (Number.isNaN(date.valueOf())) return "—";
    return date.toLocaleString();
}

function formatCompactNumber(value) {
    return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 })
        .format(Number(value || 0));
}

function formatPercent(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatIsoDate(value) {
    if (!value) return "-";
    const date = new Date(value);
    return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString();
}

function analyticsReason(reason) {
    if (reason.metric === "failure_rate") {
        return `Failure rate +${Math.round(Number(reason.delta || 0) * 100)} pp`;
    }
    if (reason.metric === "latency") return `Latency +${formatDuration(reason.delta)}`;
    if (reason.metric === "tokens") return `Tokens/request +${formatCompactNumber(reason.delta)}`;
    return String(reason.metric || "Regression");
}

function formatAlertValue(metric, value) {
    if (String(metric).includes("failure_rate")) return formatPercent(value);
    if (String(metric).includes("latency")) return formatDuration(value);
    if (String(metric).includes("tokens")) return `${formatCompactNumber(value)} tok`;
    return formatCompactNumber(value);
}

async function openAnalyticsRecord(conversationId, eventId) {
    if (!conversationId) return;
    const result = await window.apiClient.activateConversation(conversationId);
    pendingAnalyticsEventId = String(eventId || "");
    document.dispatchEvent(new CustomEvent("conversation-changed", {
        detail: { conversation: result.conversation },
    }));
    document.dispatchEvent(new CustomEvent("conversation-list-refresh"));
    if (analyticsPanel) analyticsPanel.hidden = true;
    analyticsButton?.setAttribute("aria-expanded", "false");
    setSurface("trajectory");
}

function wireAnalyticsLinks() {
    analyticsContent?.querySelectorAll("[data-analytics-conversation]").forEach((button) => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            try {
                await openAnalyticsRecord(
                    button.dataset.analyticsConversation,
                    button.dataset.analyticsEvent,
                );
            } catch (error) {
                button.disabled = false;
                button.title = error.message || "Could not open this conversation";
            }
        });
    });
    analyticsContent?.querySelectorAll("[data-analytics-alert-action]").forEach((button) => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            try {
                await window.apiClient.updateTrajectoryAlert(
                    button.dataset.analyticsAlert,
                    button.dataset.analyticsAlertAction,
                );
                analyticsPayload = null;
                await loadProjectAnalytics(true);
            } catch (error) {
                button.disabled = false;
                button.title = error.message || "Could not update alert";
            }
        });
    });
    analyticsContent?.querySelectorAll("[data-slo-incident-action]").forEach((button) => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            try {
                await window.apiClient.updateTrajectorySloIncident(
                    button.dataset.sloIncident,
                    button.dataset.sloIncidentAction,
                );
                analyticsPayload = null;
                await loadProjectAnalytics(true);
            } catch (error) {
                button.disabled = false;
                button.title = error.message || "Could not update SLO incident";
            }
        });
    });
    analyticsContent?.querySelectorAll("[data-incident-dossier]").forEach((button) => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            try {
                await showIncidentDossier(button.dataset.incidentDossier);
            } catch (error) {
                button.disabled = false;
                button.title = error.message || "Could not build incident dossier";
            }
        });
    });
    analyticsContent?.querySelectorAll("[data-eval-run]").forEach((button) => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            button.textContent = "Running...";
            try {
                await window.apiClient.runTrajectoryEvalCase(
                    button.dataset.evalRun,
                    activeConversationId || "",
                );
                analyticsPayload = null;
                await loadProjectAnalytics(true);
            } catch (error) {
                button.disabled = false;
                button.textContent = "Run current";
                button.title = error.message || "Could not run regression eval";
            }
        });
    });
    analyticsContent?.querySelectorAll("[data-eval-replay]").forEach((button) => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            button.textContent = "Replaying...";
            try {
                await window.apiClient.sandboxReplayTrajectoryEvalCase(button.dataset.evalReplay);
                analyticsPayload = null;
                await loadProjectAnalytics(true);
            } catch (error) {
                button.disabled = false;
                button.textContent = "Sandbox replay";
                button.title = error.message || "Could not run sandbox replay";
            }
        });
    });
    analyticsContent?.querySelector("#trajectory-eval-matrix-run")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        const caseCount = (analyticsPayload?.eval_cases || []).length;
        if (!caseCount || !window.confirm(`Run ${caseCount} regression cases in the side-effect-free sandbox?`)) return;
        const preferredModel = window.prompt(
            "Model for this release gate (leave blank for the current configured model)",
            "",
        );
        if (preferredModel == null) return;
        const name = window.prompt("Release gate name", `Release gate · ${new Date().toLocaleString()}`);
        if (name == null) return;
        button.disabled = true;
        button.textContent = "Running matrix...";
        try {
            await window.apiClient.runTrajectoryEvalMatrix({
                name: name.trim(),
                preferredModel: preferredModel.trim(),
            });
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            button.disabled = false;
            button.textContent = "Run release gate";
            button.title = error.message || "Could not run replay matrix";
        }
    });
    analyticsContent?.querySelector("#trajectory-eval-compare-run")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        const caseCount = (analyticsPayload?.eval_cases || []).length;
        const rawModels = window.prompt(
            "Models to compare (2–4, comma-separated)",
            "",
        );
        if (rawModels == null) return;
        const models = [...new Set(rawModels.split(",").map((value) => value.trim()).filter(Boolean))];
        if (models.length < 2 || models.length > 4) {
            button.title = "Enter 2 to 4 unique model identifiers";
            return;
        }
        const replayCount = models.length * caseCount;
        if (replayCount > 50) {
            button.title = "This comparison exceeds the 50-replay safety limit";
            return;
        }
        if (!window.confirm(
            `Run ${replayCount} side-effect-free replays (${models.length} models × ${caseCount} cases)?`,
        )) return;
        const name = window.prompt(
            "Comparison name",
            `Model comparison · ${new Date().toLocaleString()}`,
        );
        if (name == null) return;
        button.disabled = true;
        button.textContent = "Comparing...";
        try {
            await window.apiClient.runTrajectoryEvalComparison({
                name: name.trim(),
                models,
            });
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            button.disabled = false;
            button.textContent = "Compare models";
            button.title = error.message || "Could not compare models";
        }
    });
    analyticsContent?.querySelector("#trajectory-promotion-start")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        const candidate = button.dataset.candidateModel || "";
        const canaryPercent = 10;
        const confirmation = window.prompt(`Type the candidate model ID to start the 10% progressive canary:\n${candidate}`, "");
        if (confirmation !== candidate) {
            button.title = "Model confirmation did not match";
            return;
        }
        button.disabled = true;
        button.textContent = "Starting canary...";
        try {
            await window.apiClient.startTrajectoryModelPromotion({
                candidateModel: candidate,
                confirmModel: confirmation,
                canaryPercent,
            });
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            button.disabled = false;
            button.textContent = "Start canary";
            button.title = error.message || "Could not start canary";
        }
    });
    analyticsContent?.querySelector("#trajectory-promotion-finalize")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        const candidate = button.dataset.candidateModel || "";
        const confirmation = window.prompt(`Type the candidate model ID to promote globally:\n${candidate}`, "");
        if (confirmation !== candidate) {
            button.title = "Model confirmation did not match";
            return;
        }
        button.disabled = true;
        button.textContent = "Promoting...";
        try {
            await window.apiClient.finalizeTrajectoryModelPromotion(
                button.dataset.promotionId,
                confirmation,
            );
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            button.disabled = false;
            button.textContent = "Promote model";
            button.title = error.message || "Could not promote model";
        }
    });
    analyticsContent?.querySelector("#trajectory-promotion-rollback")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        const previous = button.dataset.previousModel || "";
        const confirmation = window.prompt(`Type the rollback model ID to confirm:\n${previous}`, "");
        if (confirmation !== previous) {
            button.title = "Rollback model confirmation did not match";
            return;
        }
        button.disabled = true;
        button.textContent = "Rolling back...";
        try {
            await window.apiClient.rollbackTrajectoryModelPromotion(
                button.dataset.promotionId,
                confirmation,
            );
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            button.disabled = false;
            button.textContent = "Roll back";
            button.title = error.message || "Could not roll back model";
        }
    });
    analyticsContent?.querySelectorAll("[data-eval-delete]").forEach((button) => {
        button.addEventListener("click", async () => {
            if (!window.confirm("Delete this regression eval and its run history?")) return;
            button.disabled = true;
            try {
                await window.apiClient.deleteTrajectoryEvalCase(button.dataset.evalDelete);
                analyticsPayload = null;
                await loadProjectAnalytics(true);
            } catch (error) {
                button.disabled = false;
                button.title = error.message || "Could not delete regression eval";
            }
        });
    });
    wirePolicyEditor();
    wireSloEditor();
}

function wireSloEditor() {
    const form = analyticsContent?.querySelector("#trajectory-slo-form");
    if (!form) return;
    const submit = form.querySelector("button[type='submit']");
    const status = form.querySelector(".trajectory-slo-form-status");
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        submit.disabled = true;
        status.textContent = "Saving...";
        try {
            await window.apiClient.updateTrajectorySlo({
                target_success_rate: Number(form.elements.target.value || 99) / 100,
                window_days: Number(form.elements.window_days.value || 30),
                min_operations: Number(form.elements.min_operations.value || 5),
            });
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            status.textContent = error.message || "Could not update SLO";
            submit.disabled = false;
        }
    });
}

async function showIncidentDossier(incidentId) {
    if (!incidentId || !analyticsContent) return;
    analyticsContent.querySelector("#trajectory-incident-dossier")?.remove();
    analyticsContent.insertAdjacentHTML(
        "afterbegin",
        '<section id="trajectory-incident-dossier" class="trajectory-incident-dossier"><div class="trajectory-analytics-empty">Building privacy-safe incident dossier...</div></section>',
    );
    const payload = await window.apiClient.getTrajectoryIncidentDossier(incidentId);
    const dossier = payload.dossier || {};
    const incident = dossier.incident || {};
    const summary = dossier.summary || {};
    const selected = dossier.selected_event || {};
    const causes = dossier.root_causes || [];
    const chain = dossier.causal_chain || [];
    const recovery = dossier.recovery_paths || [];
    const recommendations = dossier.recommendations || [];
    const element = analyticsContent.querySelector("#trajectory-incident-dossier");
    if (!element) return;
    element.innerHTML = `
        <header>
            <div><strong>Incident dossier</strong><span>${escapeHtml(incident.source_type)} · ${escapeHtml(incident.metric)} · ${escapeHtml(incident.status)}</span></div>
            <div><button type="button" id="trajectory-dossier-create-eval">Create regression eval</button><button type="button" id="trajectory-dossier-download">Download Markdown</button><button type="button" id="trajectory-dossier-close" aria-label="Close dossier">×</button></div>
        </header>
        <div class="trajectory-dossier-metrics">
            <article><span>Severity</span><strong>${escapeHtml(incident.severity)}</strong><small>${escapeHtml(incident.reason || incident.metric)}</small></article>
            <article><span>Observed</span><strong>${formatAlertValue(incident.metric, incident.observed)}</strong><small>threshold ${formatAlertValue(incident.metric, incident.threshold)}</small></article>
            <article><span>Trajectory</span><strong>${escapeHtml(summary.health)}</strong><small>${summary.errors || 0} errors · ${summary.warnings || 0} warnings</small></article>
            <article><span>Trace integrity</span><strong>${escapeHtml(summary.trace_integrity)}</strong><small>${summary.trace_coverage || 0}% coverage</small></article>
        </div>
        <div class="trajectory-dossier-grid">
            <section>
                <h4>Evidence event</h4>
                <button type="button" class="trajectory-dossier-event" data-analytics-conversation="${escapeHtml(incident.conversation_id)}" data-analytics-event="${escapeHtml(incident.event_id)}">
                    <strong>${escapeHtml(selected.kind)} · ${escapeHtml(selected.status)}</strong>
                    <span>${escapeHtml(selected.component || selected.provider || selected.model || "Recorded event")}</span>
                    <small>${formatDuration(selected.duration_ms)} · ${escapeHtml(selected.event_id)}</small>
                </button>
            </section>
            <section>
                <h4>Root causes</h4>
                <div class="trajectory-dossier-causes">${causes.length ? causes.map((row) => `
                    <article class="severity-${escapeHtml(row.severity)}"><strong>${escapeHtml(row.title)}</strong><span>${escapeHtml(row.confidence)} · score ${row.score || 0}</span><small>${(row.evidence_event_ids || []).length} evidence events</small></article>`).join("") : '<p>No structured root cause identified.</p>'}</div>
            </section>
            <section>
                <h4>Causal chain</h4>
                <div class="trajectory-dossier-chain">${chain.length ? chain.slice(0, 30).map((row) => `
                    <button type="button" data-analytics-conversation="${escapeHtml(incident.conversation_id)}" data-analytics-event="${escapeHtml(row.event_id)}"><strong>${escapeHtml(row.kind)}</strong><span>${escapeHtml(row.status)}</span><small>${formatDuration(row.duration_ms)}</small></button>`).join("") : '<p>No related records found.</p>'}</div>
            </section>
            <section>
                <h4>Recovery</h4>
                <div class="trajectory-dossier-recovery">${recovery.length ? recovery.map((row) => `
                    <article class="${row.recovered ? "recovered" : "failed"}"><strong>${row.recovered ? "Recovered" : escapeHtml(row.outcome)}</strong><span>${row.failed_attempt_count || 0}/${row.attempt_count || 0} failed attempts</span><small>${escapeHtml(row.final_model || row.request_id)}</small></article>`).join("") : '<p>No retry or fallback path recorded.</p>'}</div>
            </section>
            <section class="trajectory-dossier-recommendations">
                <h4>Recommended checks</h4>
                <ol>${recommendations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
            </section>
        </div>
        <footer>Privacy boundary: raw inputs, outputs, errors, and annotation notes are excluded.</footer>`;
    element.querySelector("#trajectory-dossier-close")?.addEventListener("click", () => element.remove());
    element.querySelector("#trajectory-dossier-create-eval")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        const defaultName = `Regression: ${incident.metric || incident.reason || "trajectory incident"}`;
        const name = window.prompt("Regression eval name", defaultName);
        if (name == null) return;
        button.disabled = true;
        button.textContent = "Creating...";
        try {
            await window.apiClient.createTrajectoryEvalCase(incidentId, name.trim() || defaultName);
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            button.disabled = false;
            button.textContent = "Create regression eval";
            button.title = error.message || "Could not create regression eval";
        }
    });
    element.querySelector("#trajectory-dossier-download")?.addEventListener("click", () => {
        const blob = new Blob([payload.markdown || ""], { type: "text/markdown;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `trajectory-incident-${incidentId}.md`;
        anchor.click();
        URL.revokeObjectURL(url);
    });
    element.querySelectorAll("[data-analytics-conversation]").forEach((button) => {
        button.addEventListener("click", () => openAnalyticsRecord(
            button.dataset.analyticsConversation,
            button.dataset.analyticsEvent,
        ));
    });
    element.scrollIntoView({ block: "start", behavior: "smooth" });
}

function policyApiPayload(policy, changes = {}) {
    const thresholds = policy?.thresholds || {};
    return {
        name: policy?.name || "Alert policy",
        scope_type: policy?.scope_type || "project",
        scope_value: policy?.scope_value || "*",
        failure_rate_warning: Number(thresholds.failure_rate_warning || 0),
        failure_rate_critical: Number(thresholds.failure_rate_critical || 0),
        latency_warning_ms: Number(thresholds.latency_warning_ms || 0),
        latency_critical_ms: Number(thresholds.latency_critical_ms || 0),
        tokens_warning: Number(thresholds.tokens_warning || 0),
        tokens_critical: Number(thresholds.tokens_critical || 0),
        enabled: Boolean(policy?.enabled),
        ...changes,
    };
}

function wirePolicyEditor() {
    const form = analyticsContent?.querySelector("#trajectory-policy-form");
    if (!form) return;
    const scope = form.querySelector("[name='scope_type']");
    const scopeValue = form.querySelector("[name='scope_value']");
    const submit = form.querySelector("button[type='submit']");
    const cancel = form.querySelector("#trajectory-policy-cancel");
    const policies = analyticsPayload?.policies || [];
    const syncScope = () => {
        scopeValue.disabled = scope.value === "project";
        scopeValue.placeholder = scope.value === "project" ? "All components" : `Exact ${scope.value} name`;
        if (scope.value === "project") scopeValue.value = "*";
        else if (scopeValue.value === "*") scopeValue.value = "";
    };
    scope?.addEventListener("change", syncScope);
    syncScope();
    const reset = () => {
        form.reset();
        form.dataset.policyId = "";
        scope.value = "project";
        scopeValue.value = "*";
        submit.textContent = "Create policy";
        cancel.hidden = true;
        syncScope();
    };
    cancel?.addEventListener("click", reset);
    analyticsContent.querySelectorAll("[data-policy-edit]").forEach((button) => {
        button.addEventListener("click", () => {
            const policy = policies.find((row) => row.policy_id === button.dataset.policyEdit);
            if (!policy) return;
            const thresholds = policy.thresholds || {};
            form.dataset.policyId = policy.policy_id;
            form.elements.name.value = policy.name;
            form.elements.scope_type.value = policy.scope_type;
            form.elements.scope_value.value = policy.scope_value;
            form.elements.failure_warning.value = Number(thresholds.failure_rate_warning || 0) * 100;
            form.elements.failure_critical.value = Number(thresholds.failure_rate_critical || 0) * 100;
            form.elements.latency_warning.value = Number(thresholds.latency_warning_ms || 0);
            form.elements.latency_critical.value = Number(thresholds.latency_critical_ms || 0);
            form.elements.tokens_warning.value = Number(thresholds.tokens_warning || 0);
            form.elements.tokens_critical.value = Number(thresholds.tokens_critical || 0);
            submit.textContent = "Update policy";
            cancel.hidden = false;
            syncScope();
            form.scrollIntoView({ block: "nearest" });
        });
    });
    analyticsContent.querySelectorAll("[data-policy-toggle]").forEach((button) => {
        button.addEventListener("click", async () => {
            const policy = policies.find((row) => row.policy_id === button.dataset.policyToggle);
            if (!policy) return;
            button.disabled = true;
            await window.apiClient.updateTrajectoryAlertPolicy(
                policy.policy_id,
                policyApiPayload(policy, { enabled: !policy.enabled }),
            );
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        });
    });
    analyticsContent.querySelectorAll("[data-policy-delete]").forEach((button) => {
        button.addEventListener("click", async () => {
            const policy = policies.find((row) => row.policy_id === button.dataset.policyDelete);
            if (!policy || !window.confirm(`Delete alert policy “${policy.name}”?`)) return;
            button.disabled = true;
            await window.apiClient.deleteTrajectoryAlertPolicy(policy.policy_id);
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        });
    });
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        submit.disabled = true;
        const fields = form.elements;
        const payload = {
            name: fields.name.value.trim(),
            scope_type: fields.scope_type.value,
            scope_value: fields.scope_value.value.trim() || "*",
            failure_rate_warning: Number(fields.failure_warning.value || 0) / 100,
            failure_rate_critical: Number(fields.failure_critical.value || 0) / 100,
            latency_warning_ms: Number(fields.latency_warning.value || 0),
            latency_critical_ms: Number(fields.latency_critical.value || 0),
            tokens_warning: Number(fields.tokens_warning.value || 0),
            tokens_critical: Number(fields.tokens_critical.value || 0),
            enabled: true,
        };
        try {
            if (form.dataset.policyId) {
                await window.apiClient.updateTrajectoryAlertPolicy(form.dataset.policyId, payload);
            } else {
                await window.apiClient.createTrajectoryAlertPolicy(payload);
            }
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        } catch (error) {
            form.querySelector(".trajectory-policy-status").textContent = error.message || "Could not save policy";
            submit.disabled = false;
        }
    });
}

function renderProjectAnalytics() {
    if (!analyticsContent || !analyticsPayload) return;
    const payload = analyticsPayload;
    const summary = payload.summary || {};
    const trends = (payload.trends || []).slice(-30);
    const regressions = payload.regressions || [];
    const providers = (payload.providers || []).slice(0, 8);
    const tools = (payload.tools || []).slice(0, 8);
    const clusters = (payload.error_clusters || []).slice(0, 8);
    const sessions = (payload.sessions || []).slice(0, 10);
    const alerts = payload.alerts || [];
    const openAlerts = alerts.filter((row) => row.status === "open");
    const activeBaseline = payload.active_baseline || null;
    const baselines = payload.baselines || [];
    const policies = payload.policies || [];
    const alertHistory = (payload.alert_history || []).slice(0, 20);
    const slo = payload.slo || {};
    const sloConfig = payload.slo_config || {};
    const sloWindows = slo.windows || [];
    const sloStatus = String(slo.status || "insufficient-data");
    const sloIncidents = payload.slo_incidents || [];
    const openSloIncidents = sloIncidents.filter((row) => row.status !== "resolved");
    const evalCases = payload.eval_cases || [];
    const evalMatrices = payload.eval_matrices || [];
    const evalComparisons = payload.eval_comparisons || [];
    const promotion = payload.promotion_recommendation || {};
    const modelPromotions = payload.model_promotions || [];
    const activePromotion = payload.active_model_promotion || null;
    const canaryTelemetry = payload.canary_telemetry || { status: "inactive" };
    const canaryConfidence = canaryTelemetry.confidence || {};
    const canaryIntervals = Object.values(canaryConfidence.intervals || {});
    const canarySamplePlan = canaryTelemetry.sample_plan || {};
    const formatConfidenceValue = (metric, value) => {
        if (metric === "failure_rate") return formatPercent(Number(value || 0));
        if (metric === "avg_request_ms") return formatDuration(Number(value || 0));
        return formatCompactNumber(Number(value || 0));
    };
    if (baselineSelect) {
        baselineSelect.innerHTML = '<option value="">Window median</option>'
            + baselines.map((row) => `<option value="${escapeHtml(row.baseline_id)}">${escapeHtml(row.name)}${row.active ? " В· active" : ""}</option>`).join("");
        baselineSelect.value = activeBaseline?.baseline_id || "";
    }
    if (baselineDelete) baselineDelete.hidden = !activeBaseline;
    if (alertCount) {
        const totalOpenAlerts = openAlerts.length + openSloIncidents.length;
        alertCount.hidden = !totalOpenAlerts;
        alertCount.textContent = String(totalOpenAlerts);
    }
    const maxTrend = Math.max(1, ...trends.map((row) => Number(row.requests || 0) + Number(row.tool_calls || 0)));
    const truncated = payload.pagination?.window_truncated;
    analyticsContent.innerHTML = `
        <div class="trajectory-analytics-metrics">
            <article><span>Sessions</span><strong>${formatCompactNumber(summary.sessions)}</strong><small>${formatCompactNumber(summary.turns)} turns</small></article>
            <article class="slo-${escapeHtml(sloStatus)}"><span>SLO</span><strong>${formatPercent(slo.success_rate)}</strong><small>${escapeHtml(sloStatus)} · target ${formatPercent(slo.target_success_rate)}</small></article>
            <article><span>Recovery</span><strong>${formatPercent(summary.recovery_rate)}</strong><small>${summary.recovered_failures || 0} recovered attempts</small></article>
            <article><span>Latency</span><strong>${formatDuration(summary.avg_request_ms)}</strong><small>p95 ${formatDuration(summary.p95_request_ms)}</small></article>
            <article><span>Tokens</span><strong>${formatCompactNumber(summary.total_tokens)}</strong><small>${formatCompactNumber(summary.requests)} requests</small></article>
            <article class="${regressions.length ? "has-regression" : ""}"><span>Regressions</span><strong>${regressions.length}</strong><small>${truncated ? "partial event window" : activeBaseline ? `vs ${escapeHtml(activeBaseline.name)}` : "against project median"}</small></article>
        </div>
        <div class="trajectory-analytics-grid">
            <section class="trajectory-analytics-card trajectory-slo-card slo-${escapeHtml(sloStatus)}">
                <header><strong>Reliability SLO</strong><span>${escapeHtml(sloStatus)} · ${formatCompactNumber(slo.operations)} operations</span></header>
                <div class="trajectory-slo-summary">
                    <article><span>Error budget remaining</span><strong>${formatPercent(slo.budget_remaining)}</strong><small>${formatCompactNumber(slo.failures)} failures / ${Number(slo.allowed_failures || 0).toFixed(1)} allowed</small></article>
                    <article><span>Projected exhaustion</span><strong>${slo.projected_exhaustion_hours == null ? "No burn" : `${formatCompactNumber(slo.projected_exhaustion_hours)} h`}</strong><small>${escapeHtml(slo.alert?.reason || "minimum-operations")}</small></article>
                </div>
                <div class="trajectory-slo-incidents">${sloIncidents.length ? sloIncidents.slice(0, 6).map((row) => `
                    <article class="status-${escapeHtml(row.status)} severity-${escapeHtml(row.severity)}">
                        <button type="button" data-analytics-conversation="${escapeHtml(row.conversation_id)}" data-analytics-event="${escapeHtml(row.event_id)}">
                            <strong>${escapeHtml(row.reason)} · ${Number(row.burn_rate || 0).toFixed(1)}x</strong>
                            <span>${row.failures}/${row.operations} failed · ${escapeHtml(row.status)}</span>
                            <small>${escapeHtml(formatIsoDate(row.updated_at))}</small>
                        </button>
                        <div>
                            <button type="button" data-incident-dossier="${escapeHtml(row.incident_id)}">Dossier</button>
                            ${row.status === "open" ? `<button type="button" data-slo-incident="${escapeHtml(row.incident_id)}" data-slo-incident-action="acknowledged">Acknowledge</button>` : ""}
                            ${row.status !== "resolved" ? `<button type="button" data-slo-incident="${escapeHtml(row.incident_id)}" data-slo-incident-action="resolved">Resolve</button>` : `<button type="button" data-slo-incident="${escapeHtml(row.incident_id)}" data-slo-incident-action="open">Reopen</button>`}
                        </div>
                    </article>`).join("") : '<p class="trajectory-analytics-empty good">No SLO incidents.</p>'}</div>
                <div class="trajectory-slo-windows">${sloWindows.length ? sloWindows.map((row) => {
                    const burn = Number(row.burn_rate || 0);
                    const width = Math.min(100, burn / 14.4 * 100);
                    return `<div class="${row.sufficient_data ? "" : "insufficient"}">
                        <span>${escapeHtml(row.label)}</span><i><em style="width:${width}%"></em></i>
                        <strong>${row.burn_rate == null ? "-" : `${burn.toFixed(1)}x`}</strong>
                        <small>${row.failures}/${row.operations}</small>
                    </div>`;
                }).join("") : '<p class="trajectory-analytics-empty">No SLO window data.</p>'}</div>
                <form id="trajectory-slo-form" class="trajectory-slo-form">
                    <label>Target %<input name="target" type="number" min="50" max="99.999" step="0.001" value="${Number(sloConfig.target_success_rate || 0.99) * 100}"></label>
                    <label>Window days<input name="window_days" type="number" min="1" max="365" value="${Number(sloConfig.window_days || 30)}"></label>
                    <label>Minimum operations<input name="min_operations" type="number" min="1" max="100000" value="${Number(sloConfig.min_operations || 5)}"></label>
                    <button type="submit" class="btn btn-outline btn-sm">Save SLO</button>
                    <span class="trajectory-slo-form-status"></span>
                </form>
            </section>
            <section class="trajectory-analytics-card trajectory-analytics-alerts">
                <header><strong>Regression alerts</strong><span>${openAlerts.length} open В· automatic after each turn</span></header>
                <div>${alerts.length ? alerts.slice(0, 12).map((row) => `
                    <article class="status-${escapeHtml(row.status)} severity-${escapeHtml(row.severity)}">
                        <button type="button" class="trajectory-alert-open"
                                data-analytics-conversation="${escapeHtml(row.conversation_id)}"
                                data-analytics-event="${escapeHtml(row.event_id)}">
                            <strong>${escapeHtml(row.title)} В· ${escapeHtml(row.policy?.name || row.metric)}</strong>
                            <span>${formatAlertValue(row.metric, row.baseline)} в†’ ${formatAlertValue(row.metric, row.observed)}</span>
                            <small>${escapeHtml(row.status)} В· ${escapeHtml(row.updated_at)}</small>
                        </button>
                        <div>
                            <button type="button" data-incident-dossier="${escapeHtml(row.alert_id)}">Dossier</button>
                            ${row.status === "open" ? `<button type="button" data-analytics-alert="${escapeHtml(row.alert_id)}" data-analytics-alert-action="acknowledged">Acknowledge</button>` : ""}
                            ${row.status !== "resolved" ? `<button type="button" data-analytics-alert="${escapeHtml(row.alert_id)}" data-analytics-alert-action="resolved">Resolve</button>` : `<button type="button" data-analytics-alert="${escapeHtml(row.alert_id)}" data-analytics-alert-action="open">Reopen</button>`}
                        </div>
                    </article>`).join("") : '<p class="trajectory-analytics-empty good">No regression alerts. Save a baseline or create a component policy to enable detection.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card trajectory-alert-history-card">
                <header><strong>Alert lifecycle</strong><span>Durable audit trail · newest first</span></header>
                <div class="trajectory-alert-history">${alertHistory.length ? alertHistory.map((row) => {
                    const details = row.details || {};
                    const content = `<strong>${escapeHtml(row.action)}</strong><span>${escapeHtml(details.metric || row.alert_id)}</span><small>${escapeHtml(row.actor)} · ${escapeHtml(formatIsoDate(row.created_at))}</small>`;
                    return details.conversation_id
                        ? `<button type="button" class="status-${escapeHtml(row.status)}" data-analytics-conversation="${escapeHtml(details.conversation_id)}" data-analytics-event="${escapeHtml(details.event_id)}">${content}</button>`
                        : `<article class="status-${escapeHtml(row.status)}">${content}</article>`;
                }).join("") : '<p class="trajectory-analytics-empty good">No alert transitions recorded.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card trajectory-eval-card">
                <header><div><strong>Regression evals</strong><span>${evalCases.length} durable cases · deterministic checks</span></div><div class="trajectory-eval-actions"><button type="button" id="trajectory-eval-matrix-run" ${evalCases.length ? "" : "disabled"}>Run release gate</button><button type="button" id="trajectory-eval-compare-run" ${evalCases.length ? "" : "disabled"}>Compare models</button></div></header>
                <article class="trajectory-promotion status-${escapeHtml(activePromotion?.status || promotion.status || "insufficient-data")}">
                    <div class="trajectory-promotion-heading">
                        <div>
                            <strong>${activePromotion?.status === "canary"
                                ? `Canary active · ${escapeHtml(activePromotion.candidate_model)}`
                                : activePromotion?.status === "ready"
                                    ? `Canary ready · ${escapeHtml(activePromotion.candidate_model)}`
                                    : activePromotion?.status === "promoted"
                                        ? `Promoted · ${escapeHtml(activePromotion.candidate_model)}`
                                        : activePromotion?.status === "rollback_pending"
                                            ? `Rollback pending · ${escapeHtml(activePromotion.previous_model)}`
                                        : promotion.status === "promote"
                                ? `Promotion candidate · ${escapeHtml(promotion.recommended_model)}`
                                : promotion.status === "current-model-leading"
                                    ? `Current model remains leader · ${escapeHtml(promotion.recommended_model)}`
                                    : promotion.status === "hold"
                                        ? "Model promotion on hold"
                                        : "Building promotion evidence"}</strong>
                            <span>${activePromotion
                                ? activePromotion.status === "rollback_pending"
                                    ? `Traffic pinned to rollback model · ${escapeHtml(activePromotion.rollback_reason || "regression detected")}`
                                    : `${activePromotion.canary_percent}% session-stable canary · rollback to ${escapeHtml(activePromotion.previous_model)}`
                                : escapeHtml(promotion.reason || "Run multi-model comparisons to build evidence.")}</span>
                        </div>
                        <b>${activePromotion
                            ? `${activePromotion.canary_percent || 10}% · ${activePromotion.healthy_windows || 0}/${activePromotion.required_healthy_windows || 2} windows`
                            : `${promotion.consecutive_wins || 0}/${promotion.required_wins || 3} wins`}</b>
                    </div>
                    <div class="trajectory-promotion-metrics">
                        <span><b>${Number(promotion.avg_score || 0).toFixed(1)}%</b> avg score</span>
                        <span><b>${formatDuration(promotion.avg_request_ms || 0)}</b> latency</span>
                        <span><b>${formatCompactNumber(promotion.avg_tokens_per_request || 0)}</b> tok/request</span>
                        <span><b>${promotion.comparisons_analyzed || 0}</b> runs analyzed</span>
                    </div>
                    <div class="trajectory-promotion-actions">
                        ${!activePromotion && promotion.status === "promote" ? `<button type="button" id="trajectory-promotion-start" data-candidate-model="${escapeHtml(promotion.recommended_model)}">Start canary</button>` : ""}
                        ${activePromotion?.status === "ready" && activePromotion.ramp_complete ? `<button type="button" id="trajectory-promotion-finalize" data-promotion-id="${escapeHtml(activePromotion.promotion_id)}" data-candidate-model="${escapeHtml(activePromotion.candidate_model)}">Promote model</button>` : ""}
                        ${activePromotion?.status === "ready" && !activePromotion.ramp_complete ? '<span>Comparison gate passed · complete the traffic ramp to promote</span>' : ""}
                        ${activePromotion ? `<button type="button" id="trajectory-promotion-rollback" data-promotion-id="${escapeHtml(activePromotion.promotion_id)}" data-previous-model="${escapeHtml(activePromotion.previous_model)}">Roll back</button>` : ""}
                    </div>
                    ${activePromotion && ["canary", "ready"].includes(activePromotion.status) ? `<div class="trajectory-ramp-stages">${[10, 25, 50].map((percent, index) => `<span class="${activePromotion.ramp_complete || index < (activePromotion.ramp_stage || 0) ? "complete" : index === (activePromotion.ramp_stage || 0) ? "active" : "pending"}"><b>${percent}%</b><small>${index === (activePromotion.ramp_stage || 0) && !activePromotion.ramp_complete ? `${activePromotion.healthy_windows || 0}/${activePromotion.required_healthy_windows || 2} healthy windows` : activePromotion.ramp_complete || index < (activePromotion.ramp_stage || 0) ? "complete" : "pending"}</small></span>`).join("")}</div>` : ""}
                    ${canaryTelemetry.status !== "inactive" ? `<section class="trajectory-canary-telemetry status-${escapeHtml(canaryTelemetry.status)}">
                        <header><strong>Production canary health · stage ${(canaryTelemetry.ramp_stage || 0) + 1}/3 <b class="decision-${escapeHtml(canaryConfidence.decision || "continue")}">${escapeHtml(canaryConfidence.decision || "continue")}</b></strong><span>Sequential ${formatPercent(canaryConfidence.familywise_confidence || 0.95)} confidence · look ${canaryConfidence.look || 0}</span></header>
                        <div class="trajectory-canary-arms">${[canaryTelemetry.candidate || {}, canaryTelemetry.control || {}].map((arm, index) => `<article>
                            <strong>${index === 0 ? "Candidate" : "Control"} · ${escapeHtml(arm.model || "unknown")}</strong>
                            <span>${arm.requests || 0} requests · ${arm.sessions || 0} sessions · ${arm.fallbacks || 0} fallbacks</span>
                            <small>${formatPercent(arm.failure_rate || 0)} failures · ${formatDuration(arm.avg_request_ms || 0)} · ${formatCompactNumber(arm.tokens_per_request || 0)} tok/request</small>
                        </article>`).join("")}</div>
                        ${canaryIntervals.length ? `<div class="trajectory-confidence-intervals">${canaryIntervals.map((interval) => `<article class="${interval.harm_confirmed ? "harm" : interval.non_inferior ? "safe" : "open"}">
                            <strong>${escapeHtml(interval.metric)}</strong>
                            <span>Δ ${formatConfidenceValue(interval.metric, interval.difference)} · CI ${formatConfidenceValue(interval.metric, interval.lower)} to ${formatConfidenceValue(interval.metric, interval.upper)}</span>
                            <small>non-inferiority margin ${formatConfidenceValue(interval.metric, interval.margin)}</small>
                        </article>`).join("")}</div>` : `<div class="trajectory-confidence-waiting">Minimum ${canaryTelemetry.min_requests_per_arm || 10} requests in each arm before the first decision.</div>`}
                        <section class="trajectory-sample-plan status-${escapeHtml(canarySamplePlan.status || "collecting-minimum")}">
                            <header><strong>Adaptive sample plan</strong><span>Projected ${escapeHtml(canarySamplePlan.projected_decision || "continue")} · ${canarySamplePlan.planning_horizon_hours || 24}h horizon</span></header>
                            <div class="trajectory-sample-plan-summary">
                                <article><span>Target / arm</span><b>${canarySamplePlan.target_requests_per_arm == null ? `>${canarySamplePlan.max_requests_per_arm || 200}` : formatCompactNumber(canarySamplePlan.target_requests_per_arm)}</b></article>
                                <article><span>Remaining</span><b>${formatCompactNumber(canarySamplePlan.additional_candidate_requests || 0)} cand · ${formatCompactNumber(canarySamplePlan.additional_control_requests || 0)} ctrl</b></article>
                                <article><span>Blocking metric</span><b>${escapeHtml(canarySamplePlan.limiting_metric || "minimum sample")}</b></article>
                                <article><span>ETA</span><b>${canarySamplePlan.estimated_hours == null ? "learning rate" : `${Number(canarySamplePlan.estimated_hours).toFixed(1)} h`}</b></article>
                            </div>
                            ${(canarySamplePlan.metric_plans || []).length ? `<div class="trajectory-sample-metric-plans">${canarySamplePlan.metric_plans.map((row) => `<span class="decision-${escapeHtml(row.projected_decision || "inconclusive")}"><b>${escapeHtml(row.metric)}</b>${escapeHtml(row.projected_decision || "inconclusive")} · ${row.target_requests_per_arm == null ? "over budget" : `${formatCompactNumber(row.target_requests_per_arm)}/arm`}</span>`).join("")}</div>` : ""}
                            <small>${escapeHtml(canarySamplePlan.assumption || "Waiting for enough production observations to estimate power.")}</small>
                        </section>
                        ${(canaryTelemetry.reasons || []).length ? `<div class="trajectory-canary-reasons">${canaryTelemetry.reasons.map((reason) => `<span>${escapeHtml(reason.metric)} · candidate ${escapeHtml(String(reason.candidate))} · gate ${escapeHtml(String(reason.threshold))}</span>`).join("")}</div>` : ""}
                        <small>Alpha-spending repeated-look guardrail · production aggregates only · raw prompts, outputs, and errors are excluded</small>
                    </section>` : ""}
                    <small>${activePromotion?.status === "rollback_pending"
                        ? "Durable settings rollback needs retry · runtime routing already protects all sessions"
                        : activePromotion?.status === "promoted"
                        ? "Global model changed explicitly · future regression comparisons trigger automatic rollback"
                        : activePromotion
                            ? "Global model unchanged · canary routing is deterministic per session"
                            : "Recommendation only · no model setting was changed automatically · aggregate evidence only"}</small>
                </article>
                <div class="trajectory-promotion-history">${modelPromotions.slice(0, 5).map((row) => `<span class="status-${escapeHtml(row.status)}"><b>${escapeHtml(row.status)}</b> ${escapeHtml(row.previous_model)} → ${escapeHtml(row.candidate_model)} · ${row.canary_percent || 10}% stage ${(row.ramp_stage || 0) + 1}/3 · ${row.healthy_comparisons || 0}/${row.required_healthy_comparisons || 2} comparisons${row.rollback_reason ? ` · ${escapeHtml(row.rollback_reason)}` : ""}</span>`).join("")}</div>
                <div class="trajectory-eval-list">${evalCases.length ? evalCases.map((row) => {
                    const criteria = row.criteria || {};
                    const latest = row.latest_run || null;
                    const criteriaLabels = [
                        criteria.max_failure_rate != null ? `fail ≤ ${formatPercent(criteria.max_failure_rate)}` : "",
                        criteria.max_avg_request_ms != null ? `latency ≤ ${formatDuration(criteria.max_avg_request_ms)}` : "",
                        criteria.max_tokens_per_request != null ? `tokens ≤ ${formatCompactNumber(criteria.max_tokens_per_request)}` : "",
                        criteria.max_error_count != null ? `errors ≤ ${criteria.max_error_count}` : "",
                        (criteria.blocked_fingerprints || []).length ? `${criteria.blocked_fingerprints.length} blocked fingerprints` : "",
                    ].filter(Boolean).join(" · ");
                    const failedChecks = (latest?.checks || []).filter((check) => !check.passed);
                    const replay = latest?.replay || {};
                    const runLabel = latest?.mode === "sandbox-replay"
                        ? `sandbox · ${replay.fixture_hits || 0} fixtures · ${replay.blocked_calls || 0} blocked`
                        : "recorded trajectory";
                    return `<article class="${latest ? `status-${escapeHtml(latest.status)}` : "status-not-run"}">
                        <div>
                            <strong>${escapeHtml(row.name)}</strong>
                            <span>${escapeHtml(criteriaLabels)}</span>
                            <small>${latest ? `${escapeHtml(latest.status)} · ${Number(latest.score || 0).toFixed(0)}% · ${failedChecks.length} failed checks · ${escapeHtml(runLabel)}` : "Not run yet"}</small>
                        </div>
                        <div>
                            <button type="button" data-eval-run="${escapeHtml(row.case_id)}">Run current</button>
                            <button type="button" data-eval-replay="${escapeHtml(row.case_id)}">Sandbox replay</button>
                            ${latest?.mode === "sandbox-replay" ? `<button type="button" data-analytics-conversation="${escapeHtml(latest.candidate_conversation_id)}">Open trace</button>` : ""}
                            <button type="button" data-incident-dossier="${escapeHtml(row.incident_id)}">Dossier</button>
                            <button type="button" data-eval-delete="${escapeHtml(row.case_id)}">Delete</button>
                        </div>
                        ${latest ? `<div class="trajectory-eval-checks">${(latest.checks || []).map((check) => `<span class="${check.passed ? "passed" : "failed"}">${check.passed ? "✓" : "×"} ${escapeHtml(check.label)}</span>`).join("")}</div>` : ""}
                    </article>`;
                }).join("") : '<p class="trajectory-analytics-empty">Open an incident dossier and create a regression eval.</p>'}</div>
                <div class="trajectory-comparison-list">${evalComparisons.length ? evalComparisons.slice(0, 6).map((comparison) => `
                    <article class="status-${escapeHtml(comparison.status)}">
                        <header>
                            <div><strong>${comparison.winner_model ? `Winner · ${escapeHtml(comparison.winner_model)}` : "No viable winner"}</strong><span>${escapeHtml(comparison.name)}</span></div>
                            <small>${comparison.model_count || 0} models × ${comparison.case_count || 0} cases · ${escapeHtml(formatIsoDate(comparison.completed_at || comparison.created_at))}</small>
                        </header>
                        <div class="trajectory-comparison-ranking">${(comparison.models || []).map((model) => `
                            <article class="${model.rank === 1 && comparison.winner_model ? "winner" : ""}">
                                <strong><b>#${model.rank || "–"}</b> ${escapeHtml(model.preferred_model)}</strong>
                                <span>${model.passed_count || 0} passed · ${model.failed_count || 0} failed · ${model.error_count || 0} errors</span>
                                <small>${Number(model.avg_score || 0).toFixed(1)}% score · ${formatDuration(model.avg_request_ms || 0)} · ${formatCompactNumber(model.avg_tokens_per_request || 0)} tok/request</small>
                            </article>`).join("")}</div>
                    </article>`).join("") : '<p class="trajectory-analytics-empty">No multi-model comparison has been run yet.</p>'}</div>
                <div class="trajectory-matrix-list">${evalMatrices.length ? evalMatrices.slice(0, 8).map((matrix) => `
                    <article class="status-${escapeHtml(matrix.status)}">
                        <div>
                            <strong>${matrix.gate_passed ? "Release allowed" : "Release blocked"} · ${escapeHtml(matrix.name)}</strong>
                            <span>${escapeHtml(matrix.preferred_model || "current model")} · agent ${escapeHtml(matrix.agent_version || "unknown")}</span>
                            <small>${matrix.passed_count || 0}/${matrix.case_count || 0} passed · ${matrix.failed_count || 0} failed · ${matrix.error_count || 0} errors · ${escapeHtml(formatIsoDate(matrix.completed_at || matrix.created_at))}</small>
                        </div>
                        <div class="trajectory-matrix-entries">${(matrix.entries || []).map((entry) => {
                            const evalCase = evalCases.find((item) => item.case_id === entry.case_id);
                            const content = `<strong>${escapeHtml(entry.status)}</strong><span>${escapeHtml(evalCase?.name || entry.case_id)}</span><small>${Number(entry.score || 0).toFixed(0)}%</small>`;
                            return entry.candidate_conversation_id
                                ? `<button type="button" class="status-${escapeHtml(entry.status)}" data-analytics-conversation="${escapeHtml(entry.candidate_conversation_id)}">${content}</button>`
                                : `<article class="status-${escapeHtml(entry.status)}">${content}</article>`;
                        }).join("")}</div>
                    </article>`).join("") : '<p class="trajectory-analytics-empty">No release gate has been run yet.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card trajectory-policy-card">
                <header><strong>Alert policies</strong><span>${policies.filter((row) => row.enabled).length} enabled В· project/provider/model/tool</span></header>
                <div class="trajectory-policy-list">${policies.length ? policies.map((row) => {
                    const t = row.thresholds || {};
                    const limits = [
                        t.failure_rate_warning ? `fail ${Math.round(t.failure_rate_warning * 100)}%` : "",
                        t.latency_warning_ms ? `latency ${formatDuration(t.latency_warning_ms)}` : "",
                        t.tokens_warning ? `tokens ${formatCompactNumber(t.tokens_warning)}` : "",
                    ].filter(Boolean).join(" В· ");
                    return `<article class="${row.enabled ? "enabled" : "disabled"}">
                        <div><strong>${escapeHtml(row.name)}</strong><span>${escapeHtml(row.scope_type)} В· ${escapeHtml(row.scope_value)}</span><small>${escapeHtml(limits)}</small></div>
                        <button type="button" data-policy-toggle="${escapeHtml(row.policy_id)}">${row.enabled ? "Disable" : "Enable"}</button>
                        <button type="button" data-policy-edit="${escapeHtml(row.policy_id)}">Edit</button>
                        <button type="button" data-policy-delete="${escapeHtml(row.policy_id)}">Delete</button>
                    </article>`;
                }).join("") : '<p class="trajectory-analytics-empty">No explicit component policy yet.</p>'}</div>
                <form id="trajectory-policy-form" class="trajectory-policy-form">
                    <input name="name" type="text" maxlength="120" placeholder="Policy name" required>
                    <select name="scope_type" aria-label="Policy scope">
                        <option value="project">Project</option><option value="provider">Provider</option>
                        <option value="model">Model</option><option value="tool">Tool</option>
                    </select>
                    <input name="scope_value" type="text" maxlength="240" value="*" placeholder="Exact component name">
                    <label>Failure warn %<input name="failure_warning" type="number" min="0" max="100" step="0.1" value="10"></label>
                    <label>Failure critical %<input name="failure_critical" type="number" min="0" max="100" step="0.1" value="30"></label>
                    <label>Latency warn ms<input name="latency_warning" type="number" min="0" value="10000"></label>
                    <label>Latency critical ms<input name="latency_critical" type="number" min="0" value="30000"></label>
                    <label>Tokens warn<input name="tokens_warning" type="number" min="0" value="0"></label>
                    <label>Tokens critical<input name="tokens_critical" type="number" min="0" value="0"></label>
                    <div><button type="submit" class="btn btn-outline btn-sm">Create policy</button><button id="trajectory-policy-cancel" type="button" class="btn btn-outline btn-sm" hidden>Cancel</button></div>
                    <span class="trajectory-policy-status"></span>
                </form>
            </section>
            <section class="trajectory-analytics-card trajectory-analytics-trend">
                <header><strong>Execution trend</strong><span>Requests + tool calls</span></header>
                <div>${trends.length ? trends.map((row) => {
                    const activity = Number(row.requests || 0) + Number(row.tool_calls || 0);
                    return `<div title="${escapeHtml(row.date)}: ${activity} operations, ${row.failures || 0} failures">
                        <span>${escapeHtml(row.date.slice(5))}</span>
                        <i><em style="width:${Math.max(2, activity / maxTrend * 100)}%"></em></i>
                        <strong>${activity}</strong>
                        <small class="${row.failures ? "failed" : ""}">${row.failures || 0}</small>
                    </div>`;
                }).join("") : '<p class="trajectory-analytics-empty">No recorded execution in this window.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card trajectory-analytics-regressions">
                <header><strong>Regression signals</strong><span>Session vs project median</span></header>
                <div>${regressions.length ? regressions.map((row) => `
                    <button type="button" class="severity-${escapeHtml(row.severity)}"
                            data-analytics-conversation="${escapeHtml(row.conversation_id)}"
                            data-analytics-event="${escapeHtml(row.event_id)}">
                        <strong>${escapeHtml(row.title)}</strong>
                        <span>${(row.reasons || []).map(analyticsReason).map(escapeHtml).join(" В· ")}</span>
                        <small>Open causal record в†’</small>
                    </button>`).join("") : '<p class="trajectory-analytics-empty good">No cross-session regression detected.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card">
                <header><strong>Providers & models</strong><span>Attempts / failures / p95</span></header>
                <div class="trajectory-analytics-table">${providers.length ? providers.map((row) => `
                    <div><strong>${escapeHtml(row.provider)} <span>${escapeHtml(row.model)}</span></strong><span>${row.attempts}</span><em class="${row.failures ? "failed" : ""}">${row.failures}</em><small>${formatDuration(row.p95_ms)}</small></div>`).join("") : '<p class="trajectory-analytics-empty">No provider attempts recorded.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card">
                <header><strong>Tool reliability</strong><span>Calls / failures / total</span></header>
                <div class="trajectory-analytics-table">${tools.length ? tools.map((row) => `
                    <div><strong>${escapeHtml(row.name)}</strong><span>${row.calls}</span><em class="${row.failures ? "failed" : ""}">${row.failures}</em><small>${formatDuration(row.total_ms)}</small></div>`).join("") : '<p class="trajectory-analytics-empty">No tools recorded.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card">
                <header><strong>Error fingerprints</strong><span>Occurrences / sessions</span></header>
                <div class="trajectory-analytics-fingerprints">${clusters.length ? clusters.map((row) => `
                    <button type="button" data-analytics-conversation="${escapeHtml(row.conversation_id)}"
                            data-analytics-event="${escapeHtml(row.event_id)}">
                        <strong>${escapeHtml(row.component)}</strong><span>${escapeHtml(row.kind)} В· ${escapeHtml(row.fingerprint)}</span>
                        <small>${row.occurrence_count}Г— in ${row.session_count} sessions В· ${row.recovered_count} recovered</small>
                    </button>`).join("") : '<p class="trajectory-analytics-empty good">No error fingerprints in this window.</p>'}</div>
            </section>
            <section class="trajectory-analytics-card">
                <header><strong>Recent sessions</strong><span>Failure rate / latency / tokens</span></header>
                <div class="trajectory-analytics-sessions">${sessions.length ? sessions.map((row) => `
                    <button type="button" data-analytics-conversation="${escapeHtml(row.conversation_id)}"
                            data-analytics-event="${escapeHtml(row.problem_event_id || row.latest_event_id)}">
                        <strong>${escapeHtml(row.title)}</strong>
                        <span class="${row.failures ? "failed" : ""}">${formatPercent(row.failure_rate)} fail</span>
                        <small>${formatDuration(row.avg_request_ms)} В· ${formatCompactNumber(row.total_tokens)} tok</small>
                    </button>`).join("") : '<p class="trajectory-analytics-empty">No sessions in this window.</p>'}</div>
            </section>
        </div>`;
    wireAnalyticsLinks();
}

async function loadProjectAnalytics(force = false) {
    if (!analyticsPanel || analyticsLoading || (analyticsPayload && !force)) {
        if (analyticsPayload) renderProjectAnalytics();
        return;
    }
    analyticsLoading = true;
    analyticsContent.innerHTML = '<div class="trajectory-analytics-empty">Aggregating project trajectoryвЂ¦</div>';
    try {
        const days = Number(analyticsDays?.value || 30);
        analyticsPayload = await window.apiClient.getTrajectoryAnalytics(days, 20000);
        const windowLabel = document.getElementById("trajectory-analytics-window");
        if (windowLabel) {
            const baselineName = analyticsPayload.active_baseline?.name;
            windowLabel.textContent = `Last ${days} days В· ${baselineName ? `baseline ${baselineName}` : "window median"} В· aggregate metadata only`;
        }
        renderProjectAnalytics();
    } catch (error) {
        analyticsContent.innerHTML = `<div class="trajectory-analytics-empty error">${escapeHtml(error.message || "Could not load project analytics")}</div>`;
    } finally {
        analyticsLoading = false;
    }
}

function scheduleAnalyticsRefresh() {
    analyticsPayload = null;
    if (analyticsPanel?.hidden) return;
    if (analyticsRefreshTimer) clearTimeout(analyticsRefreshTimer);
    analyticsRefreshTimer = setTimeout(() => loadProjectAnalytics(true), 500);
}

function previewFor(record) {
    const explicit = record?.details?.preview;
    if (explicit) return String(explicit);
    const value = record?.kind === "USER" ? record.input : record.output;
    return textValue(value).replace(/\s+/g, " ").trim().slice(0, 300);
}

function setSurface(name) {
    const isTrajectory = name === "trajectory";
    chatTab?.classList.toggle("active", !isTrajectory);
    trajectoryTab?.classList.toggle("active", isTrajectory);
    chatTab?.setAttribute("aria-selected", String(!isTrajectory));
    trajectoryTab?.setAttribute("aria-selected", String(isTrajectory));
    if (chatPanel) chatPanel.hidden = isTrajectory;
    if (trajectoryPanel) trajectoryPanel.hidden = !isTrajectory;
    if (isTrajectory) loadTrajectory();
    else stopReplay(false);
}

function setLiveStatus(state, detail = "") {
    liveConnectionState = ["connected", "reconnecting", "disconnected", "failed"]
        .includes(state) ? state : liveConnectionState;
    if (!liveStatus) return;
    const labels = {
        connected: "Live",
        updating: "Updating",
        paused: "Live paused",
        connecting: "Connecting",
        reconnecting: "Reconnecting",
        disconnected: "Offline",
        failed: "Offline",
    };
    liveStatus.className = `trajectory-live state-${state}`;
    const label = liveStatus.querySelector("span");
    if (label) label.textContent = labels[state] || "Live";
    liveStatus.title = detail || {
        connected: "Trajectory updates arrive live from the runtime stream",
        updating: "Applying new trajectory events",
        paused: "Live refresh is paused while you edit or replay",
        connecting: "Connecting to the live trajectory stream",
        reconnecting: "Reconnecting to the live trajectory stream",
        disconnected: "Live stream disconnected; manual refresh is still available",
        failed: "Live stream unavailable; use Refresh to update",
    }[state] || "Trajectory live status";
}

function liveDraftElement() {
    const active = document.activeElement;
    return inspectorContent?.contains(active)
        && active?.matches?.("input, textarea, select, [contenteditable='true']")
        ? active : null;
}

function scheduleLiveRefresh(delay = 180) {
    trajectoryPayload = null;
    if (trajectoryPanel?.hidden) return;
    refreshQueued = true;
    const draft = liveDraftElement();
    if (replayPlaying || draft) {
        setLiveStatus("paused");
        if (draft && liveRefreshBlockedBy !== draft) {
            liveRefreshBlockedBy = draft;
            draft.addEventListener("blur", () => {
                liveRefreshBlockedBy = null;
                if (refreshQueued) scheduleLiveRefresh(80);
            }, { once: true });
        }
        return;
    }
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(async () => {
        refreshTimer = null;
        if (loading) {
            scheduleLiveRefresh(120);
            return;
        }
        refreshQueued = false;
        setLiveStatus("updating");
        await loadTrajectory(true);
        setLiveStatus(liveConnectionState === "connected" ? "connected" : liveConnectionState);
        if (refreshQueued) scheduleLiveRefresh(120);
    }, delay);
}

function currentRecord() {
    return records.find((item) => item.event_id === selectedId) || null;
}

async function loadTrajectory(force = false) {
    if ((!activeConversationId && !externalTrajectoryUrl) || loading) return;
    if (trajectoryPayload && !force) {
        renderAll();
        return;
    }
    loading = true;
    const ledgerWrap = ledger?.closest(".trajectory-ledger-wrap");
    const priorScrollTop = Number(ledgerWrap?.scrollTop || 0);
    const pinnedToLatest = Boolean(ledgerWrap)
        && ledgerWrap.scrollHeight - ledgerWrap.clientHeight - ledgerWrap.scrollTop < 48;
    if (status) {
        status.hidden = false;
        status.textContent = "Loading causal history…";
    }
    try {
        if (externalTrajectoryUrl) {
            const separator = externalTrajectoryUrl.includes("?") ? "&" : "?";
            const response = await fetch(`${externalTrajectoryUrl}${separator}limit=${trajectoryLimit}`);
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(payload.detail || "Could not load execution trajectory.");
            trajectoryPayload = payload;
            externalTrajectorySessionId = payload.execution_session_id || externalTrajectorySessionId;
        } else {
            trajectoryPayload = await window.apiClient.getConversationTrajectory(activeConversationId, trajectoryLimit);
        }
        records = Array.isArray(trajectoryPayload.records) ? trajectoryPayload.records : [];
        invalidateLedgerView();
        if (selectedId && !records.some((item) => item.event_id === selectedId)) {
            selectedId = "";
        }
        renderAll();
        if (pendingAnalyticsEventId) {
            const target = pendingAnalyticsEventId;
            pendingAnalyticsEventId = "";
            if (records.some((record) => record.event_id === target)) {
                requestAnimationFrame(() => {
                    selectRecord(target);
                    ledger?.querySelector(`[data-event-id="${CSS.escape(target)}"]`)
                        ?.scrollIntoView({ block: "center" });
                });
            }
        }
        if (pendingScrollAnchor) {
            const anchor = pendingScrollAnchor;
            pendingScrollAnchor = "";
            requestAnimationFrame(() => {
                revealLedgerRecord(anchor, "start", false);
            });
        } else if (ledgerWrap) {
            requestAnimationFrame(() => {
                ledgerWrap.scrollTop = pinnedToLatest
                    ? ledgerWrap.scrollHeight
                    : Math.min(priorScrollTop, ledgerWrap.scrollHeight);
            });
        }
    } catch (error) {
        if (status) {
            status.hidden = false;
            status.textContent = error.message || "Could not load trajectory.";
            status.classList.add("error");
        }
    } finally {
        loading = false;
    }
}

function renderAll() {
    renderMetrics();
    renderPagination();
    renderReplayControls();
    renderOverview();
    renderTimingBreakdown();
    renderComparison();
    renderDiagnostics();
    renderLedger();
    if (selectedId) renderInspector(currentRecord());
}

function renderPagination() {
    if (!windowStatus || !loadOlderButton) return;
    const pagination = trajectoryPayload?.pagination || {};
    const returned = Number(pagination.returned ?? records.length);
    const total = Number(pagination.estimated_total ?? returned);
    const truncated = Boolean(pagination.window_truncated);
    windowStatus.hidden = !truncated && total <= returned;
    const label = windowStatus.querySelector("span");
    if (label) {
        label.textContent = truncated
            ? `Showing latest ${returned} of about ${Math.max(total, returned)} records`
            : `Showing all ${returned} records`;
    }
    loadOlderButton.hidden = !pagination.has_more;
    if (truncated && !pagination.has_more && label) {
        label.textContent += " · maximum diagnostic window reached";
    }
}

async function loadOlderTrajectory() {
    const pagination = trajectoryPayload?.pagination || {};
    if (!pagination.has_more || loading) return;
    pendingScrollAnchor = records[0]?.event_id || "";
    trajectoryLimit = Number(pagination.next_limit || Math.min(5000, trajectoryLimit * 2));
    loadOlderButton.disabled = true;
    loadOlderButton.textContent = "Loading…";
    try {
        await loadTrajectory(true);
    } finally {
        loadOlderButton.disabled = false;
        loadOlderButton.textContent = "Load earlier records";
    }
}

function renderMetrics() {
    const summary = trajectoryPayload?.summary || {};
    const duration = document.getElementById("trajectory-duration");
    const turns = document.getElementById("trajectory-turns");
    const calls = document.getElementById("trajectory-calls");
    if (duration) duration.textContent = `Duration ${formatDuration(summary.duration_ms)}`;
    if (turns) turns.textContent = `Turns ${summary.turns || 0}`;
    if (calls) calls.textContent = `Calls ${summary.tool_calls || 0}`;
    const diagnostics = trajectoryPayload?.diagnostics || {};
    if (healthButton) {
        const health = diagnostics.health || "healthy";
        const problemCount = Number(diagnostics.error_count || 0) + Number(diagnostics.warning_count || 0);
        healthButton.className = `trajectory-health health-${health}`;
        healthButton.textContent = `${health.charAt(0).toUpperCase()}${health.slice(1)}${problemCount ? ` · ${problemCount}` : ""}`;
    }
    if (integrityButton) {
        const integrity = diagnostics.integrity || {};
        const integrityStatus = integrity.status || "reliable";
        const score = Number(integrity.coverage?.score ?? 100);
        integrityButton.className = `trajectory-integrity-badge integrity-${integrityStatus}`;
        integrityButton.textContent = `Trace ${score}%${integrityStatus !== "reliable" ? ` · ${integrityStatus}` : ""}`;
    }
}

function visibleRecords() {
    const query = String(search?.value || "").trim().toLowerCase();
    const kind = String(kindFilter?.value || "");
    const causalIds = causalRecordIds(causalRootId);
    return records.filter((record) => {
        if (focusedTurnId && record.turn_id !== focusedTurnId) return false;
        if (causalRootId && !causalIds.has(record.event_id)) return false;
        if (kind === "TOOL" && !["TOOL", "SUBTOOL"].includes(record.kind)) return false;
        if (kind === "PIPELINE" && !record.kind.startsWith("PIPELINE_")) return false;
        if (kind === "EXPERIMENT" && !record.kind.startsWith("EXPERIMENT_")) return false;
        if (kind === "AUTOMATION" && !record.kind.startsWith("AUTOMATION_")) return false;
        if (kind && !["TOOL", "PIPELINE", "EXPERIMENT", "AUTOMATION"].includes(kind) && record.kind !== kind) return false;
        if (problemsOnly?.checked && !["error", "warning"].includes(record.diagnostic?.severity)) return false;
        if (bookmarksOnly?.checked && !record.annotation?.bookmarked) return false;
        if (!query) return true;
        const haystack = [
            record.kind, record.status, previewFor(record), record.error,
            textValue(record.source), textValue(record.input), textValue(record.output),
            textValue(record.annotation),
        ].join(" ").toLowerCase();
        return haystack.includes(query);
    });
}

function causalRecordIds(eventId) {
    if (!eventId) return new Set();
    const selected = records.find((record) => record.event_id === eventId);
    if (!selected) return new Set();
    let requestId = selected.kind === "REQUEST" ? selected.event_id : selected.request_id;
    if (!requestId && selected.turn_id) {
        requestId = records.find((record) =>
            record.turn_id === selected.turn_id && record.kind === "REQUEST"
        )?.event_id || "";
    }
    const ids = new Set(requestId ? [requestId] : [selected.event_id]);
    if (selected.turn_id) {
        for (const record of records) {
            if (record.turn_id === selected.turn_id && record.kind === "USER") ids.add(record.event_id);
        }
    }
    let changed = true;
    while (changed) {
        changed = false;
        for (const record of records) {
            if (
                ids.has(record.event_id)
                || (requestId && record.request_id === requestId)
                || (record.parent_id && ids.has(record.parent_id))
            ) {
                if (!ids.has(record.event_id)) {
                    ids.add(record.event_id);
                    changed = true;
                }
            }
        }
    }
    return ids;
}

function turnRows() {
    return Array.isArray(trajectoryPayload?.turns) ? trajectoryPayload.turns : [];
}

function stopReplay(reset = false) {
    if (replayTimer) clearInterval(replayTimer);
    replayTimer = null;
    replayPlaying = false;
    if (reset) replayIndex = -1;
    renderReplayControls();
    if (refreshQueued && !replayPlaying) scheduleLiveRefresh(80);
}

function renderReplayControls() {
    if (turnFilter) {
        const currentOptions = Array.from(turnFilter.options).map((option) => option.value).join("|");
        const nextOptions = ["", ...turnRows().map((turn) => turn.turn_id)].join("|");
        if (currentOptions !== nextOptions) {
            turnFilter.innerHTML = '<option value="">All turns</option>';
            for (const turn of turnRows()) {
                const option = document.createElement("option");
                option.value = turn.turn_id;
                option.textContent = `Turn ${turn.index} · ${turn.status} · ${formatDuration(turn.duration_ms)}`;
                turnFilter.appendChild(option);
            }
        }
        turnFilter.value = focusedTurnId;
    }
    const rows = visibleRecords();
    if (replayIndex >= rows.length) replayIndex = rows.length - 1;
    if (replayPlay) replayPlay.textContent = replayPlaying ? "Pause" : "Replay";
    if (replayPosition) {
        replayPosition.textContent = replayIndex >= 0 && rows.length
            ? `${replayIndex + 1} / ${rows.length}` : `All ${rows.length} records`;
    }
    if (turnSummary) {
        const turn = turnRows().find((item) => item.turn_id === focusedTurnId);
        turnSummary.textContent = turn
            ? `${turn.request_count} requests · ${turn.attempt_count} attempts · ${turn.tool_count} tools · ${turn.total_tokens} tokens`
            : "";
    }
    if (chainSummary && chainClear) {
        chainSummary.hidden = !causalRootId;
        chainClear.hidden = !causalRootId;
        chainSummary.textContent = causalRootId
            ? `Causal chain · ${causalRecordIds(causalRootId).size} records`
            : "";
    }
}

function focusCausalChain(eventId) {
    const record = records.find((item) => item.event_id === eventId);
    if (!record) return;
    causalRootId = record.kind === "REQUEST"
        ? record.event_id
        : record.request_id || record.event_id;
    focusedTurnId = "";
    invalidateLedgerView();
    stopReplay(true);
    renderAll();
    if (!causalRecordIds(causalRootId).has(selectedId)) {
        selectedId = causalRootId;
        renderLedger();
        renderInspector(currentRecord());
    }
}

function showReplayRecord(index) {
    const rows = visibleRecords();
    if (!rows.length) return;
    replayIndex = Math.max(0, Math.min(index, rows.length - 1));
    selectedId = rows[replayIndex].event_id;
    ensureLedgerIndex(replayIndex);
    selectedTab = "Summary";
    renderReplayControls();
    renderOverview();
    renderLedger();
    renderInspector(currentRecord());
    requestAnimationFrame(() => {
        ledger?.querySelector(`[data-event-id="${CSS.escape(selectedId)}"]`)
            ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
}

function stepReplay(delta) {
    const rows = visibleRecords();
    if (!rows.length) return;
    const current = replayIndex >= 0
        ? replayIndex : Math.max(-1, rows.findIndex((row) => row.event_id === selectedId));
    showReplayRecord(current + delta);
}

function toggleReplay() {
    if (replayPlaying) {
        stopReplay(false);
        return;
    }
    if (!visibleRecords().length) return;
    if (replayIndex < 0 || replayIndex >= visibleRecords().length - 1) replayIndex = -1;
    replayPlaying = true;
    renderReplayControls();
    stepReplay(1);
    replayTimer = setInterval(() => {
        if (replayIndex >= visibleRecords().length - 1) {
            stopReplay(false);
            return;
        }
        stepReplay(1);
    }, 850);
}

function signedDelta(value, suffix = "") {
    const number = Number(value || 0);
    if (!number) return `0${suffix}`;
    return `${number > 0 ? "+" : ""}${number}${suffix}`;
}

function renderComparison() {
    if (!comparePanel) return;
    comparePanel.hidden = !compareOpen;
    if (!compareOpen) return;
    const turns = turnRows();
    if (turns.length < 2) {
        comparePanel.innerHTML = `
            <div class="trajectory-compare-empty">
                At least two recorded turns are required for comparison.
                <button id="trajectory-compare-close" class="btn btn-outline btn-sm" type="button">Close</button>
            </div>`;
        comparePanel.querySelector("#trajectory-compare-close")?.addEventListener("click", () => {
            compareOpen = false;
            renderComparison();
        });
        return;
    }
    if (!turns.some((turn) => turn.turn_id === compareBaseId)) {
        compareBaseId = turns[Math.max(0, turns.length - 2)].turn_id;
    }
    if (!turns.some((turn) => turn.turn_id === compareTargetId) || compareTargetId === compareBaseId) {
        compareTargetId = [...turns].reverse()
            .find((turn) => turn.turn_id !== compareBaseId)?.turn_id || "";
    }
    const base = turns.find((turn) => turn.turn_id === compareBaseId);
    const target = turns.find((turn) => turn.turn_id === compareTargetId);
    const options = turns.map((turn) =>
        `<option value="${escapeHtml(turn.turn_id)}">Turn ${turn.index} · ${escapeHtml(turn.status)}</option>`
    ).join("");
    const durationDelta = Number(target.duration_ms || 0) - Number(base.duration_ms || 0);
    const tokenDelta = Number(target.total_tokens || 0) - Number(base.total_tokens || 0);
    const attemptDelta = Number(target.attempt_count || 0) - Number(base.attempt_count || 0);
    const failureDelta = Number(target.failure_count || 0) - Number(base.failure_count || 0);
    const models = (turn) => Array.from(new Set(records
        .filter((record) => record.turn_id === turn.turn_id && ["REQUEST", "ATTEMPT"].includes(record.kind))
        .map((record) => record.details?.model)
        .filter(Boolean))).join(", ") || "—";
    comparePanel.innerHTML = `
        <div class="trajectory-compare-header">
            <strong>Turn comparison</strong>
            <div>
                <select id="trajectory-compare-base" class="trajectory-filter" aria-label="Baseline turn">${options}</select>
                <span>versus</span>
                <select id="trajectory-compare-target" class="trajectory-filter" aria-label="Target turn">${options}</select>
                <button id="trajectory-compare-close" class="btn-icon" type="button" aria-label="Close comparison">&times;</button>
            </div>
        </div>
        <div class="trajectory-compare-metrics">
            <article><span>Status</span><strong>${escapeHtml(base.status)} → ${escapeHtml(target.status)}</strong></article>
            <article><span>Duration</span><strong>${formatDuration(base.duration_ms)} → ${formatDuration(target.duration_ms)}</strong><small>${signedDelta(durationDelta, " ms")}</small></article>
            <article><span>Tokens</span><strong>${base.total_tokens} → ${target.total_tokens}</strong><small>${signedDelta(tokenDelta)}</small></article>
            <article><span>Attempts</span><strong>${base.attempt_count} → ${target.attempt_count}</strong><small>${signedDelta(attemptDelta)}</small></article>
            <article><span>Failures</span><strong>${base.failure_count} → ${target.failure_count}</strong><small>${signedDelta(failureDelta)}</small></article>
            <article><span>Tools</span><strong>${base.tool_count} → ${target.tool_count}</strong><small>${signedDelta(target.tool_count - base.tool_count)}</small></article>
        </div>
        <div class="trajectory-compare-columns">
            <article>
                <span>Turn ${base.index} · baseline</span>
                <strong>${escapeHtml(models(base))}</strong>
                <p>${escapeHtml(base.user_preview || "No user preview")}</p>
                <div>${escapeHtml(base.assistant_preview || "No assistant output")}</div>
            </article>
            <article>
                <span>Turn ${target.index} · target</span>
                <strong>${escapeHtml(models(target))}</strong>
                <p>${escapeHtml(target.user_preview || "No user preview")}</p>
                <div>${escapeHtml(target.assistant_preview || "No assistant output")}</div>
            </article>
        </div>`;
    const baseSelect = comparePanel.querySelector("#trajectory-compare-base");
    const targetSelect = comparePanel.querySelector("#trajectory-compare-target");
    baseSelect.value = compareBaseId;
    targetSelect.value = compareTargetId;
    baseSelect.addEventListener("change", () => {
        compareBaseId = baseSelect.value;
        renderComparison();
    });
    targetSelect.addEventListener("change", () => {
        compareTargetId = targetSelect.value;
        renderComparison();
    });
    comparePanel.querySelector("#trajectory-compare-close")?.addEventListener("click", () => {
        compareOpen = false;
        renderComparison();
    });
}

function renderDiagnostics() {
    const diagnostics = trajectoryPayload?.diagnostics || {};
    const findingsEl = document.getElementById("trajectory-findings");
    const bottleneckEl = document.getElementById("trajectory-bottleneck");
    const rootCausesEl = document.getElementById("trajectory-root-causes");
    const recoveryEl = document.getElementById("trajectory-recovery-paths");
    const integrityEl = document.getElementById("trajectory-integrity");
    const clustersEl = document.getElementById("trajectory-error-clusters");
    const titleEl = document.getElementById("trajectory-diagnostics-title");
    const successEl = document.getElementById("trajectory-success-summary");
    if (!findingsEl || !bottleneckEl || !titleEl || !successEl || !rootCausesEl || !recoveryEl || !integrityEl || !clustersEl) return;
    const health = diagnostics.health || "healthy";
    titleEl.textContent = `Trajectory ${health}`;
    const successes = diagnostics.successes || {};
    successEl.textContent = [
        `${successes.completed_requests || 0} model requests completed`,
        `${successes.completed_tools || 0} tools succeeded`,
        `${successes.completed_verifications || 0} verifications completed`,
    ].join(" · ");
    const bottleneck = diagnostics.bottleneck;
    if (bottleneck) {
        bottleneckEl.innerHTML = `
            <span>Bottleneck</span>
            <button type="button" data-trajectory-ref="${escapeHtml(bottleneck.event_id)}">
                ${escapeHtml(bottleneck.kind)} · ${formatDuration(bottleneck.duration_ms)} · ${escapeHtml(bottleneck.preview || "Open record")}
            </button>`;
    } else {
        bottleneckEl.innerHTML = '<span>Bottleneck</span><em>No timed operation yet</em>';
    }
    const integrity = diagnostics.integrity || {};
    const coverage = integrity.coverage || {};
    const integrityIssues = integrity.issues || [];
    integrityEl.innerHTML = `
        <div class="trajectory-integrity-header">
            <div>
                <span>Trace integrity${integrity.window_truncated ? " · loaded window" : ""}</span>
                <strong class="integrity-${escapeHtml(integrity.status || "reliable")}">${escapeHtml(integrity.status || "reliable")}</strong>
            </div>
            <b>${Number(coverage.score ?? 100)}%</b>
        </div>
        <div class="trajectory-coverage-grid">
            ${[["Source", coverage.source_percent], ["Timing", coverage.timing_percent],
                ["Correlation", coverage.correlation_percent], ["Attempts", coverage.attempt_percent]]
                .map(([label, value]) => `
                    <div title="${escapeHtml(label)} coverage ${Number(value ?? 100)}%">
                        <span>${escapeHtml(label)}</span>
                        <i><em style="width:${Math.max(0, Math.min(100, Number(value ?? 100)))}%"></em></i>
                        <strong>${Number(value ?? 100)}%</strong>
                    </div>`).join("")}
        </div>
        ${integrity.window_truncated ? '<small class="trajectory-integrity-window">Load earlier records for full-session coverage.</small>' : ""}
        ${integrityIssues.length ? `<div class="trajectory-integrity-issues">${integrityIssues.map((issue) => `
            <button type="button" class="severity-${escapeHtml(issue.severity)}"
                    data-trajectory-ref="${escapeHtml(issue.event_id)}">
                <span>${escapeHtml(issue.category)}</span>
                <strong>${escapeHtml(issue.title)}</strong>
                <small>${escapeHtml(issue.explanation)}</small>
            </button>`).join("")}</div>` : '<small class="trajectory-integrity-ok">Causal links, ordering, and clocks are consistent.</small>'}`;
    const rootCauses = diagnostics.root_causes || [];
    rootCausesEl.innerHTML = rootCauses.length ? `
        <h4>Root cause candidates</h4>
        <div>${rootCauses.map((cause, index) => `
            <button type="button" class="trajectory-root-cause severity-${escapeHtml(cause.severity)}"
                    data-trajectory-ref="${escapeHtml(cause.evidence_event_ids?.[0] || "")}">
                <span>#${index + 1} · ${escapeHtml(cause.confidence)}</span>
                <strong>${escapeHtml(cause.title)}</strong>
                <small>${escapeHtml(cause.explanation)}</small>
            </button>`).join("")}</div>` : "";
    const recovery = diagnostics.recovery || {};
    const paths = recovery.paths || [];
    recoveryEl.innerHTML = paths.length ? `
        <h4>Recovery paths · ${Number(recovery.recovered_requests || 0)} recovered</h4>
        <div>${paths.map((path) => `
            <article class="trajectory-recovery-path ${path.recovered ? "recovered" : "unrecovered"}">
                <header>
                    <strong>${path.recovered ? "Recovered" : escapeHtml(path.outcome)}</strong>
                    <span>${path.attempt_count} attempts · final ${escapeHtml(path.final_model || "none")}</span>
                </header>
                <div>${path.steps.map((step, index) => `
                    <button type="button" class="trajectory-recovery-step status-${escapeHtml(step.status)}"
                            data-trajectory-ref="${escapeHtml(step.event_id)}"
                            title="${escapeHtml(step.error || step.retry_action || step.status)}">
                        <span>${index + 1}</span>
                        <strong>${escapeHtml(step.provider || "provider")} · ${escapeHtml(step.model || "model")}</strong>
                        <small>${escapeHtml(step.status)} · ${formatDuration(step.duration_ms)}</small>
                    </button>${index < path.steps.length - 1 ? '<i>→</i>' : ""}`
                ).join("")}</div>
            </article>`).join("")}</div>` : "";
    const errorClusters = diagnostics.error_clusters || [];
    clustersEl.innerHTML = errorClusters.length ? `
        <h4>Error fingerprints · ${errorClusters.filter((cluster) => cluster.recurring).length} recurring</h4>
        <div>${errorClusters.map((cluster) => `
            <article class="trajectory-error-cluster ${cluster.recurring ? "recurring" : "single"} severity-${escapeHtml(cluster.severity)}">
                <header>
                    <span>${cluster.occurrence_count}× · ${escapeHtml(cluster.kind)} · ${escapeHtml(cluster.fingerprint)}</span>
                    <strong>${escapeHtml(cluster.component)}</strong>
                    <em class="resolution-${escapeHtml(cluster.resolution)}">${escapeHtml(cluster.resolution)}</em>
                </header>
                <p>${escapeHtml(cluster.sample_error)}</p>
                <div class="trajectory-error-cluster-meta">
                    <span>${cluster.recovered_count}/${cluster.occurrence_count} recovered</span>
                    <span>${cluster.turn_ids?.length || 0} turns</span>
                    <span>${formatDuration(cluster.total_duration_ms)} total</span>
                </div>
                <div class="trajectory-error-events">${(cluster.event_ids || []).slice(0, 8).map((eventId, index) => `
                    <button type="button" data-trajectory-ref="${escapeHtml(eventId)}">#${index + 1}</button>`
                ).join("")}${cluster.event_ids?.length > 8 ? `<span>+${cluster.event_ids.length - 8}</span>` : ""}</div>
            </article>`).join("")}</div>` : "";
    const findings = diagnostics.findings || [];
    findingsEl.innerHTML = findings.length
        ? findings.map((finding) => `
            <button type="button" class="trajectory-finding severity-${escapeHtml(finding.severity)}"
                    data-trajectory-ref="${escapeHtml(finding.event_id)}">
                <span>${escapeHtml(finding.severity)}</span>
                <strong>${escapeHtml(finding.title)}</strong>
                <small>${escapeHtml(finding.explanation)}</small>
            </button>`).join("")
        : '<div class="trajectory-no-findings">No failures or latency warnings detected in the recorded path.</div>';
    diagnosticsPanel?.querySelectorAll("[data-trajectory-ref]").forEach((button) => {
        button.addEventListener("click", () => selectRecord(button.dataset.trajectoryRef));
    });
}

function renderOverview() {
    if (!overview) return;
    overview.innerHTML = "";
    const replayRows = visibleRecords();
    const replayOrder = new Map(replayRows.map((record, index) => [record.event_id, index]));
    let timed = replayRows.filter((record) => record.started_at != null);
    if (timelineMode === "calls") {
        timed = timed.filter((record) =>
            ["REQUEST", "ATTEMPT", "TOOL", "SUBTOOL", "ASSISTANT"].includes(record.kind)
        );
    }
    const allTimed = timed;
    timelineTotalCount = allTimed.length;
    if (allTimed.length > TIMELINE_BAR_LIMIT) {
        const critical = new Set(trajectoryPayload?.diagnostics?.critical_path || []);
        const pinned = allTimed.filter((record) =>
            record.event_id === selectedId
            || critical.has(record.event_id)
            || ["error", "warning"].includes(record.diagnostic?.severity)
        );
        const prioritizedPinned = [...pinned].sort((left, right) => {
            const priority = (record) =>
                record.event_id === selectedId ? 0
                : record.diagnostic?.severity === "error" ? 1
                : record.diagnostic?.severity === "warning" ? 2
                : critical.has(record.event_id) ? 3 : 4;
            return priority(left) - priority(right);
        }).slice(0, TIMELINE_BAR_LIMIT);
        const pinnedIds = new Set(prioritizedPinned.map((record) => record.event_id));
        const candidates = allTimed.filter((record) => !pinnedIds.has(record.event_id));
        const available = Math.max(0, TIMELINE_BAR_LIMIT - prioritizedPinned.length);
        const sampled = [];
        if (available > 0 && candidates.length) {
            const stride = candidates.length / available;
            for (let index = 0; index < available; index += 1) {
                sampled.push(candidates[Math.min(candidates.length - 1, Math.floor(index * stride))]);
            }
        }
        const keepIds = new Set([...prioritizedPinned, ...sampled].map((record) => record.event_id));
        timed = allTimed.filter((record) => keepIds.has(record.event_id)).slice(0, TIMELINE_BAR_LIMIT);
    }
    timelineRenderedCount = timed.length;
    if (!allTimed.length) {
        overview.innerHTML = '<div class="trajectory-overview-empty">No timing data yet</div>';
        return;
    }
    const starts = allTimed.map((record) => Number(record.started_at));
    const ends = allTimed.map((record) => Number(record.completed_at ?? record.started_at));
    const min = Math.min(...starts);
    const max = Math.max(...ends, min + 0.001);
    const span = max - min;
    const orderedTurns = Array.from(new Set(allTimed.map((record) => record.turn_id || "__none__")));
    const turnBounds = new Map(orderedTurns.map((turnId) => {
        const rows = allTimed.filter((record) => (record.turn_id || "__none__") === turnId);
        const turnMin = Math.min(...rows.map((record) => Number(record.started_at)));
        const turnMax = Math.max(
            ...rows.map((record) => Number(record.completed_at ?? record.started_at)),
            turnMin + 0.001,
        );
        return [turnId, { min: turnMin, span: turnMax - turnMin }];
    }));
    const callOrder = new Map(allTimed.map((record, index) => [record.event_id, index]));

    function placement(record) {
        const start = Number(record.started_at);
        const end = Number(record.completed_at ?? start + Math.max(0.001, Number(record.duration_ms || 0) / 1000));
        if (timelineMode === "calls") {
            const slot = 100 / Math.max(1, allTimed.length);
            return { left: callOrder.get(record.event_id) * slot, width: Math.max(0.45, slot * 0.78) };
        }
        if (timelineMode === "turns") {
            const turnId = record.turn_id || "__none__";
            const turnIndex = orderedTurns.indexOf(turnId);
            const segment = 100 / Math.max(1, orderedTurns.length);
            const bounds = turnBounds.get(turnId);
            return {
                left: turnIndex * segment + ((start - bounds.min) / bounds.span) * segment,
                width: Math.max(0.45, ((Math.max(end, start + 0.001) - start) / bounds.span) * segment),
            };
        }
        return {
            left: Math.max(0, ((start - min) / span) * 100),
            width: Math.max(0.45, ((Math.max(end, start + 0.001) - start) / span) * 100),
        };
    }

    const groups = [
        ["input", "Input"], ["model", "Model"],
        ["pipeline", "Pipeline"], ["experiment", "Experiment"],
        ["automation", "Automation"], ["tools", "Tools"],
    ]
        .filter(([group]) => timed.some((item) => (KIND_GROUPS[item.kind] || "tools") === group));
    for (const [group, label] of groups) {
        const row = document.createElement("div");
        row.className = "trajectory-overview-row";
        row.innerHTML = `<span class="trajectory-overview-label">${label}</span><div class="trajectory-overview-track"></div>`;
        const track = row.querySelector(".trajectory-overview-track");
        for (const record of timed.filter((item) => (KIND_GROUPS[item.kind] || "tools") === group)) {
            const position = placement(record);
            const bar = document.createElement("button");
            bar.type = "button";
            bar.className = `trajectory-overview-bar kind-${record.kind.toLowerCase()}`;
            bar.classList.toggle("has-problem", ["error", "warning"].includes(record.diagnostic?.severity));
            bar.classList.toggle("critical-path", (trajectoryPayload?.diagnostics?.critical_path || []).includes(record.event_id));
            bar.classList.toggle("replay-future", replayIndex >= 0 && replayOrder.get(record.event_id) > replayIndex);
            bar.classList.toggle("selected", record.event_id === selectedId);
            bar.style.left = `${position.left}%`;
            bar.style.width = `${position.width}%`;
            bar.title = `${record.kind} · ${previewFor(record)}`;
            bar.addEventListener("click", () => selectRecord(record.event_id));
            track.appendChild(bar);
        }
        overview.appendChild(row);
    }
}

function renderTimingBreakdown() {
    const element = document.getElementById("trajectory-timing-breakdown");
    if (!element) return;
    document.querySelectorAll("[data-trajectory-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.trajectoryMode === timelineMode);
        button.setAttribute("aria-pressed", String(button.dataset.trajectoryMode === timelineMode));
    });
    const timing = trajectoryPayload?.diagnostics?.timing_breakdown || {};
    if (!Number(timing.wall_clock_ms || 0) && !Number(timing.operation_count || 0)) {
        element.innerHTML = "";
        return;
    }
    element.innerHTML = `
        <span>Session timing</span>
        <strong>Wall ${formatDuration(timing.wall_clock_ms)}</strong>
        <strong>Busy ${formatDuration(timing.busy_ms)}</strong>
        <strong>Idle ${formatDuration(timing.idle_ms)}</strong>
        <strong>Model ${formatDuration(timing.model_ms)}</strong>
        <strong>Tools ${formatDuration(timing.tool_ms)}</strong>
        <strong>Overlap ${formatDuration(timing.overlap_ms)}</strong>
        <strong>Max parallel ${Number(timing.max_concurrency || 0)}</strong>
        ${timelineRenderedCount < timelineTotalCount
            ? `<strong class="trajectory-timeline-sampled">Timeline ${timelineRenderedCount}/${timelineTotalCount} · issues preserved</strong>`
            : ""}`;
}

function turnNumbers() {
    const map = new Map();
    let next = 1;
    for (const record of records) {
        if (record.turn_id && !map.has(record.turn_id)) map.set(record.turn_id, next++);
    }
    return map;
}

function invalidateLedgerView() {
    ledgerViewCache = null;
    ledgerViewGeneration += 1;
    ledgerRenderSignature = "";
}

function ledgerView() {
    if (ledgerViewCache) return ledgerViewCache;
    const rows = visibleRecords();
    const turns = turnNumbers();
    const turnById = new Map(turnRows().map((turn) => [turn.turn_id, turn]));
    const entries = [];
    const offsets = [0];
    const recordEntryIndex = new Map();
    let priorTurn = "";
    for (let recordIndex = 0; recordIndex < rows.length; recordIndex += 1) {
        const record = rows[recordIndex];
        if (record.turn_id && record.turn_id !== priorTurn) {
            entries.push({
                type: "divider",
                key: `turn:${record.turn_id}`,
                turnId: record.turn_id,
                turnNumber: turns.get(record.turn_id) || "",
                turn: turnById.get(record.turn_id) || null,
                height: LEDGER_DIVIDER_HEIGHT,
            });
            offsets.push(offsets[offsets.length - 1] + LEDGER_DIVIDER_HEIGHT);
            priorTurn = record.turn_id;
        }
        recordEntryIndex.set(record.event_id, entries.length);
        entries.push({
            type: "record",
            key: `event:${record.event_id}`,
            record,
            recordIndex,
            height: LEDGER_RECORD_HEIGHT,
        });
        offsets.push(offsets[offsets.length - 1] + LEDGER_RECORD_HEIGHT);
    }
    ledgerViewCache = { rows, entries, offsets, recordEntryIndex };
    return ledgerViewCache;
}

function offsetEntryIndex(offsets, value) {
    let low = 0;
    let high = Math.max(0, offsets.length - 2);
    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (offsets[middle + 1] <= value) low = middle + 1;
        else high = middle;
    }
    return low;
}

function ledgerVirtualRange(view) {
    if (view.rows.length <= LEDGER_VIRTUALIZATION_THRESHOLD) {
        return { start: 0, end: view.entries.length, virtualized: false };
    }
    const wrap = ledger.closest(".trajectory-ledger-wrap");
    const viewportHeight = Math.max(LEDGER_RECORD_HEIGHT, Number(wrap?.clientHeight || 640));
    const localScrollTop = Math.max(0, Number(wrap?.scrollTop || 0) - Number(ledger.offsetTop || 0));
    const first = offsetEntryIndex(view.offsets, localScrollTop);
    const last = offsetEntryIndex(view.offsets, localScrollTop + viewportHeight);
    return {
        start: Math.max(0, first - LEDGER_VIRTUAL_OVERSCAN),
        end: Math.min(view.entries.length, last + LEDGER_VIRTUAL_OVERSCAN + 1),
        virtualized: true,
    };
}

function appendLedgerSpacer(height, position) {
    if (height <= 0) return;
    const spacer = document.createElement("div");
    spacer.className = `trajectory-ledger-spacer trajectory-ledger-spacer-${position}`;
    spacer.style.height = `${height}px`;
    spacer.setAttribute("aria-hidden", "true");
    ledger.appendChild(spacer);
}

function appendLedgerEntry(entry, totalRecords) {
    if (entry.type === "divider") {
        const divider = document.createElement("button");
        divider.type = "button";
        divider.className = "trajectory-turn-divider";
        divider.dataset.virtualKey = entry.key;
        divider.textContent = `Turn ${entry.turnNumber}${entry.turn ? ` · ${entry.turn.status} · ${formatDuration(entry.turn.duration_ms)}` : ""}`;
        divider.title = "Focus this turn";
        divider.addEventListener("click", () => {
            focusedTurnId = entry.turnId;
            causalRootId = "";
            invalidateLedgerView();
            stopReplay(true);
            renderAll();
        });
        ledger.appendChild(divider);
        return;
    }
    const record = entry.record;
    const row = document.createElement("button");
    row.type = "button";
    row.className = `trajectory-record kind-${record.kind.toLowerCase()}`;
    row.classList.toggle("selected", record.event_id === selectedId);
    row.classList.toggle("failed", record.status === "failed" || Boolean(record.error));
    row.classList.toggle("has-warning", record.diagnostic?.severity === "warning");
    row.classList.toggle("replay-future", replayIndex >= 0 && entry.recordIndex > replayIndex);
    row.dataset.eventId = record.event_id;
    row.dataset.virtualKey = entry.key;
    row.setAttribute("aria-setsize", String(totalRecords));
    row.setAttribute("aria-posinset", String(entry.recordIndex + 1));
    const name = record.kind === "TOOL" || record.kind === "SUBTOOL"
        ? record.details?.name || record.source?.name || record.kind
        : ["PIPELINE_", "EXPERIMENT_", "AUTOMATION_"].some((prefix) => record.kind.startsWith(prefix))
            ? record.details?.name || record.kind
        : record.kind;
    const diagnosticMarker = ["error", "warning"].includes(record.diagnostic?.severity)
        ? `<span class="trajectory-diagnostic-marker severity-${record.diagnostic.severity}" title="Diagnostic finding"></span>`
        : "";
    const annotationMarker = record.annotation
        ? `<span class="trajectory-annotation-marker label-${escapeHtml(record.annotation.label || "note")}" title="${escapeHtml(record.annotation.note || record.annotation.label || "Annotated")}">${record.annotation.bookmarked ? "★" : "●"}</span>`
        : "";
    row.innerHTML = `
        <span class="trajectory-record-kind">${escapeHtml(record.kind)}</span>
        <span class="trajectory-record-copy">
            <strong>${diagnosticMarker}${annotationMarker}${escapeHtml(name)}${record.annotation?.label ? ` <em class="trajectory-record-label">${escapeHtml(record.annotation.label)}</em>` : ""}</strong>
            <span>${escapeHtml(previewFor(record) || "No preview")}</span>
        </span>
        <span class="trajectory-record-meta">${record.status === "failed" ? "Failed · " : ""}${formatDuration(record.duration_ms)}</span>`;
    row.addEventListener("click", () => selectRecord(record.event_id));
    ledger.appendChild(row);
}

function renderLedger(force = false) {
    if (!ledger || !status) return;
    const view = ledgerView();
    const allRows = view.rows;
    status.classList.remove("error");
    status.hidden = allRows.length > 0;
    if (!allRows.length) {
        ledger.innerHTML = "";
        ledger.classList.remove("virtualized");
        status.textContent = records.length
            ? "No records match this search."
            : "No trajectory has been recorded yet. Send a message to create the first causal trace.";
        return;
    }
    const range = ledgerVirtualRange(view);
    const signature = [
        ledgerViewGeneration, range.start, range.end, selectedId, replayIndex,
        range.virtualized ? "virtual" : "full",
    ].join(":");
    if (!force && signature === ledgerRenderSignature) return;
    ledgerRenderSignature = signature;
    ledger.innerHTML = "";
    ledger.classList.toggle("virtualized", range.virtualized);
    ledger.setAttribute("aria-rowcount", String(allRows.length));
    appendLedgerSpacer(view.offsets[range.start], "top");
    for (let index = range.start; index < range.end; index += 1) {
        appendLedgerEntry(view.entries[index], allRows.length);
    }
    appendLedgerSpacer(
        view.offsets[view.entries.length] - view.offsets[range.end],
        "bottom",
    );
}

function ensureLedgerIndex(index) {
    const view = ledgerView();
    const record = view.rows[index];
    if (!record) return;
    revealLedgerRecord(record.event_id, "nearest", false);
}

function revealLedgerRecord(eventId, block = "nearest", smooth = true) {
    const view = ledgerView();
    const entryIndex = view.recordEntryIndex.get(eventId);
    const wrap = ledger?.closest(".trajectory-ledger-wrap");
    if (entryIndex == null || !wrap) return;
    const entryTop = Number(ledger.offsetTop || 0) + view.offsets[entryIndex];
    const entryBottom = entryTop + view.entries[entryIndex].height;
    const viewportTop = wrap.scrollTop;
    const viewportBottom = viewportTop + wrap.clientHeight;
    let target = viewportTop;
    if (block === "start") target = entryTop;
    else if (block === "center") target = entryTop - (wrap.clientHeight - view.entries[entryIndex].height) / 2;
    else if (entryTop < viewportTop) target = entryTop;
    else if (entryBottom > viewportBottom) target = entryBottom - wrap.clientHeight;
    wrap.scrollTo({ top: Math.max(0, target), behavior: smooth ? "smooth" : "auto" });
    renderLedger(true);
    requestAnimationFrame(() => {
        ledger.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`)
            ?.scrollIntoView({ block, behavior: smooth ? "smooth" : "auto" });
    });
}

function selectRecord(eventId) {
    selectedId = eventId;
    selectedTab = "Summary";
    const index = visibleRecords().findIndex((record) => record.event_id === eventId);
    ensureLedgerIndex(index);
    if (replayIndex >= 0) {
        if (index >= 0) replayIndex = index;
    }
    renderReplayControls();
    renderOverview();
    renderLedger();
    renderInspector(currentRecord());
}

function tabsFor(record) {
    const kind = record.kind;
    const diagnosis = (record.diagnostic?.findings || []).length ? ["Diagnosis"] : [];
    const withAnnotation = (tabs) => externalTrajectoryUrl
        ? tabs
        : record.details?.legacy_projection
        ? [...tabs, "Fork"]
        : [tabs[0], "Annotation", ...tabs.slice(1), "Fork"];
    if (kind === "SYSTEM") {
        return withAnnotation(record.details?.change_kind === "initial"
            ? [...diagnosis, "System Prompt", "Tools", "Policy", "Raw"]
            : [...diagnosis, "Diff", "System Prompt", "Tools", "Policy", "Raw"]);
    }
    if (kind === "USER" || kind === "CONTEXT") {
        return withAnnotation(["Summary", ...diagnosis, "Preview", "Raw", "Source", "Trust"]);
    }
    if (kind === "ASSISTANT") {
        return withAnnotation(["Summary", ...diagnosis, "Preview", "Raw", "Request", "Usage", "Timing", "Evidence"]);
    }
    if (kind === "TOOL" || kind === "SUBTOOL") {
        return withAnnotation(["Summary", ...diagnosis, "Payload", "Result", "Schema", "Policy", "Timing", "Artifacts", "Raw"]);
    }
    if (kind === "REQUEST") {
        return withAnnotation(["Summary", ...diagnosis, "Options", "Usage", "Timing", "Input", "Tools", "Raw"]);
    }
    if (kind === "ATTEMPT") {
        return withAnnotation(["Summary", ...diagnosis, record.error ? "Error" : "Result", "Timing", "Input", "Source", "Raw"]);
    }
    if (kind === "COMPACTED") return withAnnotation(["Summary", ...diagnosis, "Preview", "Raw", "Timing"]);
    if (kind === "MEMORY") return withAnnotation(["Summary", ...diagnosis, "Candidates", "Selected", "Provenance", "Raw"]);
    if (kind === "VERIFICATION") return withAnnotation(["Summary", ...diagnosis, "Claims", "Evidence", "Verdict", "Raw"]);
    if (kind === "POLICY" || kind === "APPROVAL") return withAnnotation(["Summary", ...diagnosis, "Policy", "Decision", "Raw"]);
    if (kind === "SUBAGENT") return withAnnotation(["Summary", ...diagnosis, "Task", "Context", "Tools", "Result", "Raw"]);
    if (kind === "CHILD" || kind === "SETTLEMENT") {
        return withAnnotation(["Summary", ...diagnosis, "Result", "Source", "Timing", "Raw"]);
    }
    if (kind === "PIPELINE_RUN") {
        return withAnnotation(["Summary", ...diagnosis, "Input", "Definition", "Result", "Timing", "Source", "Raw"]);
    }
    if (kind === "PIPELINE_STEP") {
        return withAnnotation(["Summary", ...diagnosis, "Input", "Result", "Route", "Timing", "Source", "Raw"]);
    }
    if (kind === "PIPELINE_ROUTE") {
        return withAnnotation(["Summary", ...diagnosis, "Decision", "Source", "Raw"]);
    }
    if (kind === "PIPELINE_RESULT") {
        return withAnnotation(["Summary", ...diagnosis, "Result", "Timing", "Source", "Raw"]);
    }
    if (kind === "EXPERIMENT_RUN" || kind === "AUTOMATION_RUN") {
        return withAnnotation(["Summary", ...diagnosis, "Input", "Definition", "Result", "Timing", "Source", "Raw"]);
    }
    if (kind === "EXPERIMENT_DECISION") {
        return withAnnotation(["Summary", ...diagnosis, "Decision", "Input", "Result", "Source", "Raw"]);
    }
    if (kind === "EXPERIMENT_MODEL") {
        return withAnnotation(["Summary", ...diagnosis, "Input", "Result", "Evidence", "Timing", "Source", "Raw"]);
    }
    if (kind === "EXPERIMENT_RESULT" || kind === "AUTOMATION_RESULT") {
        return withAnnotation(["Summary", ...diagnosis, "Result", "Timing", "Source", "Raw"]);
    }
    if (kind.startsWith("EXPERIMENT_") || kind.startsWith("AUTOMATION_")) {
        return withAnnotation(["Summary", ...diagnosis, "Input", record.error ? "Error" : "Result", "Timing", "Source", "Raw"]);
    }
    return withAnnotation(["Summary", ...diagnosis, "Raw", "Source", "Timing"]);
}

function renderInspector(record) {
    if (!record || !inspector) {
        if (inspector) inspector.hidden = true;
        return;
    }
    inspector.hidden = false;
    inspectorKind.textContent = record.kind;
    inspectorKind.className = `trajectory-kind kind-${record.kind.toLowerCase()}`;
    const turn = turnNumbers().get(record.turn_id);
    inspectorTitle.textContent = [externalTrajectoryTitle, turn ? `Turn ${turn}` : "", record.details?.name || "Record"]
        .filter(Boolean).join(" · ");
    const tabs = tabsFor(record);
    if (!tabs.includes(selectedTab)) selectedTab = tabs[0];
    inspectorTabs.innerHTML = "";
    for (const tab of tabs) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "trajectory-inspector-tab";
        button.classList.toggle("active", tab === selectedTab);
        button.textContent = tab;
        button.setAttribute("role", "tab");
        button.setAttribute("aria-selected", String(tab === selectedTab));
        button.addEventListener("click", () => {
            selectedTab = tab;
            renderInspector(record);
        });
        inspectorTabs.appendChild(button);
    }
    inspectorContent.innerHTML = contentForTab(record, selectedTab);
    inspectorContent.querySelectorAll("[data-trajectory-ref]").forEach((button) => {
        button.addEventListener("click", () => selectRecord(button.dataset.trajectoryRef));
    });
    inspectorContent.querySelectorAll("[data-trajectory-chain]").forEach((button) => {
        button.addEventListener("click", () => focusCausalChain(button.dataset.trajectoryChain));
    });
    wireAnnotationEditor(record);
    wireForkEditor(record);
}

function jsonBlock(value, empty = "No data recorded") {
    if (value == null || value === "" || (Array.isArray(value) && !value.length)) {
        return `<div class="trajectory-empty-detail">${escapeHtml(empty)}</div>`;
    }
    const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    return `<pre class="trajectory-code">${escapeHtml(text)}</pre>`;
}

function field(label, value, className = "") {
    const text = value == null || value === "" ? "—" : value;
    return `<div class="trajectory-field ${className}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(text)}</strong></div>`;
}

function causalLinks(record) {
    const ids = [];
    if (record.request_id && record.event_id !== record.request_id) ids.push(["Request", record.request_id]);
    if (record.parent_id && record.parent_id !== record.request_id) ids.push(["Parent", record.parent_id]);
    if (record.kind === "REQUEST") {
        const result = records.find((item) => item.request_id === record.request_id && item.kind === "ASSISTANT");
        if (result) ids.push(["Assistant", result.event_id]);
    }
    return `<div class="trajectory-links">${ids.map(([label, id]) =>
        `<button type="button" data-trajectory-ref="${escapeHtml(id)}">${escapeHtml(label)} ↗</button>`
    ).join("")}<button type="button" data-trajectory-chain="${escapeHtml(record.event_id)}">Focus causal chain</button></div>`;
}

function summaryContent(record) {
    const sourceLabel = record.source?.name || record.source?.model || record.source?.kind || "—";
    const usage = record.details?.usage || {};
    const primaryFinding = record.diagnostic?.findings?.[0];
    const diagnosis = primaryFinding ? `
        <div class="trajectory-diagnosis-callout severity-${escapeHtml(primaryFinding.severity)}">
            <strong>${escapeHtml(primaryFinding.title)}</strong>
            <span>${escapeHtml(primaryFinding.explanation)}</span>
        </div>` : "";
    return `
        ${diagnosis}
        <div class="trajectory-summary-grid">
            ${field("Source", sourceLabel)}
            ${field("Status", record.status, record.status === "failed" ? "danger" : "")}
            ${field("Duration", formatDuration(record.duration_ms))}
            ${field("Started", formatTime(record.started_at))}
            ${record.details?.model ? field("Model", record.details.model) : ""}
            ${record.details?.provider ? field("Provider", record.details.provider) : ""}
            ${record.details?.attempt ? field("Attempt", record.details.attempt) : ""}
            ${record.details?.retry_action ? field("Retry action", record.details.retry_action) : ""}
            ${record.details?.attempt_count ? field("Provider attempts", record.details.attempt_count) : ""}
            ${record.details?.pipeline_id ? field("Pipeline ID", record.details.pipeline_id) : ""}
            ${record.details?.source_id ? field("Source ID", record.details.source_id) : ""}
            ${record.details?.run_id ? field("Run ID", record.details.run_id) : ""}
            ${record.details?.phase ? field("Phase", record.details.phase) : ""}
            ${record.details?.step ? field("Step", record.details.step) : ""}
            ${record.details?.step_type ? field("Step type", record.details.step_type) : ""}
            ${record.details?.role ? field("Role", record.details.role) : ""}
            ${record.details?.round ? field("Round", record.details.round) : ""}
            ${record.details?.replica ? field("Replica", record.details.replica) : ""}
            ${record.details?.run_id ? field("Run ID", record.details.run_id) : ""}
            ${record.details?.step_type ? field("Block type", record.details.step_type) : ""}
            ${record.details?.step_id ? field("Block ID", record.details.step_id) : ""}
            ${record.details?.definition_hash ? field("Definition", record.details.definition_hash) : ""}
            ${record.details?.steps_run != null ? field("Steps run", record.details.steps_run) : ""}
            ${record.call_id ? field("Call ID", record.call_id) : ""}
            ${usage.total_tokens != null ? field("Tokens", usage.total_tokens) : ""}
            ${record.error ? field("Error", record.error, "danger") : ""}
        </div>
        ${causalLinks(record)}
        <h4>Preview</h4>
        <div class="trajectory-preview">${escapeHtml(previewFor(record) || "No preview")}</div>`;
}

function timingContent(record) {
    const ttft = record.first_output_at != null && record.started_at != null
        ? Math.max(0, (Number(record.first_output_at) - Number(record.started_at)) * 1000)
        : null;
    return `<div class="trajectory-summary-grid">
        ${field("Started", formatTime(record.started_at))}
        ${field("First output", formatTime(record.first_output_at))}
        ${field("Completed", formatTime(record.completed_at))}
        ${field("Duration", formatDuration(record.duration_ms))}
        ${ttft != null ? field("TTFT", formatDuration(ttft)) : ""}
    </div>`;
}

function annotationContent(record) {
    const annotation = record.annotation || {};
    const labels = ["", "investigate", "bug", "expected", "resolved"];
    return `
        <div class="trajectory-annotation-editor">
            <label>
                <span>Label</span>
                <select id="trajectory-annotation-label">
                    ${labels.map((label) => `<option value="${label}" ${annotation.label === label ? "selected" : ""}>${label || "No label"}</option>`).join("")}
                </select>
            </label>
            <label class="trajectory-annotation-bookmark">
                <input id="trajectory-annotation-bookmarked" type="checkbox" ${annotation.bookmarked ? "checked" : ""}>
                Bookmark this record
            </label>
            <label>
                <span>Investigation note</span>
                <textarea id="trajectory-annotation-note" maxlength="4000" rows="8"
                          placeholder="What happened, why it matters, and what should be checked next…">${escapeHtml(annotation.note || "")}</textarea>
            </label>
            <div class="trajectory-annotation-actions">
                <button id="trajectory-annotation-save" class="btn btn-primary btn-sm" type="button">Save annotation</button>
                <button id="trajectory-annotation-delete" class="btn btn-outline btn-sm" type="button" ${record.annotation ? "" : "hidden"}>Remove</button>
                <span id="trajectory-annotation-status" role="status"></span>
            </div>
            ${annotation.updated_at ? `<small>Updated ${escapeHtml(annotation.updated_at)}</small>` : ""}
        </div>`;
}

function replayPromptFor(record) {
    if (record.kind === "ASSISTANT" || record.kind === "FORK") return "";
    const user = records.find((item) => item.turn_id === record.turn_id && item.kind === "USER");
    return textValue(user?.input || user?.output || "").trim();
}

function forkContent(record) {
    const currentModel = document.getElementById("chat-model-select")?.value || "";
    const modelOptions = Array.from(
        document.getElementById("chat-model-select")?.options || [],
    ).map((option) => `
        <option value="${escapeHtml(option.value)}" ${option.value === currentModel ? "selected" : ""}>
            ${escapeHtml(option.textContent || option.value)}
        </option>`).join("");
    return `
        <div class="trajectory-fork-editor">
            <div class="trajectory-fork-boundary">
                <span>Branch boundary</span>
                <strong>${escapeHtml(record.kind)} #${Number(record.sequence || 0)}</strong>
                <small>${escapeHtml(previewFor(record) || record.event_id)}</small>
            </div>
            <label>
                <span>Branch title</span>
                <input id="trajectory-fork-title" maxlength="120"
                       value="${escapeHtml(`Fork at ${record.kind} #${Number(record.sequence || 0)}`)}">
            </label>
            <label>
                <span>Preferred model</span>
                <select id="trajectory-fork-model">
                    <option value="">Use current model</option>
                    ${modelOptions}
                </select>
            </label>
            <label>
                <span>Editable replay prompt</span>
                <textarea id="trajectory-fork-prompt" maxlength="20000" rows="7"
                          placeholder="Continue from this boundary or rerun the unfinished turnвЂ¦">${escapeHtml(replayPromptFor(record))}</textarea>
            </label>
            <div class="trajectory-fork-safety">
                Completed chat turns are copied. Tool calls and other side effects are never replayed automatically.
            </div>
            <div class="trajectory-fork-actions">
                <button id="trajectory-fork-create" class="btn btn-primary btn-sm" type="button">Create branch</button>
                <span id="trajectory-fork-status" role="status"></span>
            </div>
        </div>`;
}

function wireAnnotationEditor(record) {
    if (selectedTab !== "Annotation" || !inspectorContent) return;
    const save = inspectorContent.querySelector("#trajectory-annotation-save");
    const remove = inspectorContent.querySelector("#trajectory-annotation-delete");
    const annotationStatus = inspectorContent.querySelector("#trajectory-annotation-status");
    save?.addEventListener("click", async () => {
        save.disabled = true;
        if (annotationStatus) annotationStatus.textContent = "Saving…";
        try {
            const payload = await window.apiClient.updateTrajectoryAnnotation(
                activeConversationId,
                record.event_id,
                {
                    label: inspectorContent.querySelector("#trajectory-annotation-label")?.value || "",
                    note: inspectorContent.querySelector("#trajectory-annotation-note")?.value || "",
                    bookmarked: Boolean(inspectorContent.querySelector("#trajectory-annotation-bookmarked")?.checked),
                },
            );
            record.annotation = payload.annotation;
            invalidateLedgerView();
            renderLedger();
            renderInspector(record);
        } catch (error) {
            if (annotationStatus) annotationStatus.textContent = error.message || "Could not save annotation";
            save.disabled = false;
        }
    });
    remove?.addEventListener("click", async () => {
        remove.disabled = true;
        if (annotationStatus) annotationStatus.textContent = "Removing…";
        try {
            await window.apiClient.deleteTrajectoryAnnotation(activeConversationId, record.event_id);
            record.annotation = null;
            invalidateLedgerView();
            renderLedger();
            renderInspector(record);
        } catch (error) {
            if (annotationStatus) annotationStatus.textContent = error.message || "Could not remove annotation";
            remove.disabled = false;
        }
    });
}

function wireForkEditor(record) {
    if (selectedTab !== "Fork" || !inspectorContent) return;
    const create = inspectorContent.querySelector("#trajectory-fork-create");
    const forkStatus = inspectorContent.querySelector("#trajectory-fork-status");
    create?.addEventListener("click", async () => {
        create.disabled = true;
        if (forkStatus) forkStatus.textContent = "Creating safe branchвЂ¦";
        try {
            const result = await window.apiClient.forkConversationTrajectory(
                activeConversationId,
                record.event_id,
                {
                    title: inspectorContent.querySelector("#trajectory-fork-title")?.value || "",
                    preferred_model: inspectorContent.querySelector("#trajectory-fork-model")?.value || "",
                    next_prompt: inspectorContent.querySelector("#trajectory-fork-prompt")?.value || "",
                    activate: true,
                },
            );
            document.dispatchEvent(new CustomEvent("conversation-changed", {
                detail: { conversation: result.conversation },
            }));
            document.dispatchEvent(new CustomEvent("conversation-list-refresh"));
            document.dispatchEvent(new CustomEvent("fork-replay-ready", {
                detail: result.fork || {},
            }));
            setSurface("chat");
        } catch (error) {
            if (forkStatus) forkStatus.textContent = error.message || "Could not create branch";
            create.disabled = false;
        }
    });
}

function contentForTab(record, tab) {
    const detail = record.details || {};
    const output = record.output;
    const outputContent = output && typeof output === "object" && "content" in output
        ? output.content : output;
    const values = {
        Summary: () => summaryContent(record),
        Annotation: () => annotationContent(record),
        Fork: () => forkContent(record),
        Diagnosis: () => {
            const findings = record.diagnostic?.findings || [];
            return findings.length ? findings.map((finding) => `
                <article class="trajectory-diagnosis-card severity-${escapeHtml(finding.severity)}">
                    <span>${escapeHtml(finding.category)}</span>
                    <h4>${escapeHtml(finding.title)}</h4>
                    <p>${escapeHtml(finding.explanation)}</p>
                    <strong>Next check</strong>
                    <p>${escapeHtml(finding.next_check)}</p>
                </article>`).join("") : jsonBlock(null, "No diagnostic finding for this record");
        },
        Preview: () => `<div class="trajectory-preview trajectory-preview-large">${escapeHtml(textValue(outputContent || record.input)).replaceAll("\n", "<br>")}</div>`,
        Raw: () => jsonBlock(record),
        Source: () => jsonBlock(record.source),
        Trust: () => jsonBlock({
            trust_tier: record.source?.trust_tier,
            admission_reason: record.source?.admission_reason,
            provenance: record.source,
        }),
        "System Prompt": () => jsonBlock(record.input),
        Tools: () => jsonBlock(record.schema),
        Diff: () => `<h4>Previous</h4>${jsonBlock(detail.previous_prompt)}<h4>Current</h4>${jsonBlock(record.input)}`,
        Policy: () => jsonBlock(detail.policy || (record.kind === "POLICY" ? output : null)),
        Payload: () => jsonBlock(record.input),
        Result: () => jsonBlock(output),
        Error: () => jsonBlock({
            error: record.error,
            retry_action: detail.retry_action,
            retry_delay_seconds: detail.retry_delay_seconds,
        }),
        Schema: () => jsonBlock(record.schema),
        Timing: () => timingContent(record),
        Artifacts: () => jsonBlock(detail.artifacts),
        Options: () => jsonBlock(detail.options),
        Usage: () => {
            let requestDetail = detail;
            if (record.kind === "ASSISTANT" && record.request_id) {
                requestDetail = records.find((item) => item.event_id === record.request_id)?.details || detail;
            }
            return jsonBlock({
                this_request: requestDetail.usage || {},
                session_cumulative: requestDetail.session_cumulative_usage || {},
            });
        },
        Input: () => jsonBlock(record.input),
        Definition: () => jsonBlock(record.schema),
        Route: () => jsonBlock({
            selected_outputs: detail.route_outputs || detail.selected_outputs || [],
        }),
        Request: () => {
            const request = records.find((item) => item.event_id === record.request_id);
            return request ? `${causalLinks(record)}${jsonBlock(request)}` : jsonBlock(null, "Request record unavailable");
        },
        Evidence: () => jsonBlock(detail.evidence || output?.evidence || output),
        Candidates: () => jsonBlock(detail.candidates || record.input),
        Selected: () => jsonBlock(detail.selected || output),
        Provenance: () => jsonBlock(record.source),
        Claims: () => jsonBlock(output?.claims || record.input?.claims || output),
        Verdict: () => jsonBlock(detail.verdict || output),
        Decision: () => jsonBlock(detail.decision || output),
        Task: () => jsonBlock(detail.task || record.input),
        Context: () => jsonBlock(detail.context || record.input),
    };
    return (values[tab] || values.Raw)();
}

function exportTrajectory() {
    if (!trajectoryPayload) return;
    const blob = new Blob([JSON.stringify(trajectoryPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `trajectory-${externalTrajectorySessionId || activeConversationId || "conversation"}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

function buildDebugReport() {
    if (!trajectoryPayload) return "";
    const summary = trajectoryPayload.summary || {};
    const diagnostics = trajectoryPayload.diagnostics || {};
    const lines = [
        `# Remy Trajectory debug report`,
        "",
        `- Session: \`${externalTrajectorySessionId || activeConversationId || "unknown"}\``,
        `- Generated: ${new Date().toISOString()}`,
        `- Health: **${diagnostics.health || "unknown"}**`,
        `- Trace integrity: **${diagnostics.integrity?.status || "unknown"}** (${Number(diagnostics.integrity?.coverage?.score ?? 0)}%)`,
        `- Duration: ${formatDuration(summary.duration_ms)}`,
        `- Turns: ${summary.turns || 0}`,
        `- Requests: ${summary.requests || 0}`,
        `- Tool calls: ${summary.tool_calls || 0}`,
        `- Recorded failures: ${summary.failures || 0}`,
        `- Trace window: ${trajectoryPayload.pagination?.returned ?? records.length} / approximately ${trajectoryPayload.pagination?.estimated_total ?? records.length} records${trajectoryPayload.pagination?.window_truncated ? " (truncated)" : ""}`,
        "",
        "## Trace integrity",
        "",
        `- Source coverage: ${Number(diagnostics.integrity?.coverage?.source_percent ?? 0)}%`,
        `- Timing coverage: ${Number(diagnostics.integrity?.coverage?.timing_percent ?? 0)}%`,
        `- Correlation coverage: ${Number(diagnostics.integrity?.coverage?.correlation_percent ?? 0)}%`,
        `- Provider-attempt coverage: ${Number(diagnostics.integrity?.coverage?.attempt_percent ?? 0)}%`,
        ...((diagnostics.integrity?.issues || []).map((issue) =>
            `- **${String(issue.severity).toUpperCase()} · ${issue.title}** — ${issue.explanation} (\`${issue.event_id || "trace"}\`)`
        )),
        "",
        "## Root cause candidates",
        "",
    ];
    const rootCauses = diagnostics.root_causes || [];
    if (!rootCauses.length) lines.push("No root cause signal detected.");
    for (const [index, cause] of rootCauses.entries()) {
        lines.push(
            `${index + 1}. **${cause.title}** (${cause.severity}, ${cause.confidence})`,
            `   - ${cause.explanation}`,
            `   - Evidence: ${(cause.evidence_event_ids || []).map((id) => `\`${id}\``).join(", ") || "none"}`,
        );
    }
    lines.push("", "## Recovery paths", "");
    const paths = diagnostics.recovery?.paths || [];
    if (!paths.length) lines.push("No retry or fallback path was recorded.");
    for (const path of paths) {
        lines.push(`### Request \`${path.request_id}\` — ${path.recovered ? "recovered" : path.outcome}`);
        for (const [index, step] of path.steps.entries()) {
            lines.push(
                `${index + 1}. ${step.provider || "provider"} / ${step.model || "model"}: `
                + `**${step.status}** in ${formatDuration(step.duration_ms)}`
                + `${step.retry_action ? ` → ${step.retry_action}` : ""}`
                + `${step.error ? ` — ${String(step.error).replaceAll("\n", " ")}` : ""}`
            );
        }
        lines.push("");
    }
    lines.push("## Error fingerprints", "");
    const errorClusters = diagnostics.error_clusters || [];
    if (!errorClusters.length) lines.push("No failed event fingerprint was recorded.");
    for (const cluster of errorClusters) {
        lines.push(
            `- **${cluster.occurrence_count}× · ${cluster.component}** — ${cluster.sample_error}`,
            `  - Fingerprint: \`${cluster.fingerprint}\`; ${cluster.recovered_count}/${cluster.occurrence_count} recovered; ${cluster.resolution}.`,
            `  - Events: ${(cluster.event_ids || []).map((id) => `\`${id}\``).join(", ")}`,
        );
    }
    lines.push("");
    lines.push("## Turn summary", "");
    for (const turn of turnRows()) {
        lines.push(
            `- Turn ${turn.index}: **${turn.status}**, ${formatDuration(turn.duration_ms)}, `
            + `${turn.total_tokens} tokens, ${turn.attempt_count} attempts, ${turn.tool_count} tools, `
            + `${turn.failure_count} recorded failures.`
        );
    }
    lines.push("", "## Investigation annotations", "");
    const annotatedRecords = records.filter((record) => record.annotation);
    if (!annotatedRecords.length) lines.push("No investigation annotations.");
    for (const record of annotatedRecords) {
        lines.push(
            `- ${record.annotation.bookmarked ? "★ " : ""}**${record.annotation.label || "note"} · ${record.kind}** (\`${record.event_id}\`)`,
            `  - ${record.annotation.note || "No note text"}`,
        );
    }
    lines.push("", "## Diagnostic findings", "");
    const findings = diagnostics.findings || [];
    if (!findings.length) lines.push("No diagnostic findings.");
    for (const finding of findings) {
        lines.push(
            `- **${finding.severity.toUpperCase()} · ${finding.title}**`,
            `  - ${finding.explanation}`,
            `  - Next check: ${finding.next_check}`,
            `  - Event: \`${finding.event_id}\``,
        );
    }
    return `${lines.join("\n")}\n`;
}

function downloadDebugReport() {
    const report = buildDebugReport();
    if (!report) return;
    const blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `trajectory-debug-${externalTrajectorySessionId || activeConversationId || "conversation"}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

chatTab?.addEventListener("click", () => setSurface("chat"));
trajectoryTab?.addEventListener("click", () => setSurface("trajectory"));
analyticsButton?.addEventListener("click", () => {
    if (!analyticsPanel) return;
    analyticsPanel.hidden = !analyticsPanel.hidden;
    analyticsButton.setAttribute("aria-expanded", String(!analyticsPanel.hidden));
    if (!analyticsPanel.hidden) loadProjectAnalytics();
});
document.getElementById("trajectory-analytics-close")?.addEventListener("click", () => {
    if (analyticsPanel) analyticsPanel.hidden = true;
    analyticsButton?.setAttribute("aria-expanded", "false");
});
document.getElementById("trajectory-analytics-refresh")?.addEventListener("click", () => {
    analyticsPayload = null;
    loadProjectAnalytics(true);
});
analyticsDays?.addEventListener("change", () => {
    analyticsPayload = null;
    loadProjectAnalytics(true);
});
document.addEventListener("trajectory-open-alert", async (event) => {
    const detail = event.detail || {};
    try {
        if (detail.conversationId) {
            await openAnalyticsRecord(detail.conversationId, detail.eventId);
        } else {
            setSurface("trajectory");
        }
        if (detail.analytics) {
            if (analyticsPanel) analyticsPanel.hidden = false;
            analyticsButton?.setAttribute("aria-expanded", "true");
            analyticsPayload = null;
            await loadProjectAnalytics(true);
        }
        if (detail.alertId) {
            if (analyticsPanel) analyticsPanel.hidden = false;
            analyticsButton?.setAttribute("aria-expanded", "true");
            if (!analyticsPayload) await loadProjectAnalytics(true);
            await showIncidentDossier(detail.alertId);
        }
    } catch (error) {
        if (status) status.textContent = error.message || "Could not open trajectory incident";
    }
});
baselineSelect?.addEventListener("change", async () => {
    baselineSelect.disabled = true;
    try {
        if (baselineSelect.value) {
            await window.apiClient.activateTrajectoryBaseline(baselineSelect.value);
        } else {
            await window.apiClient.useTrajectoryWindowMedian();
        }
        analyticsPayload = null;
        await loadProjectAnalytics(true);
    } finally {
        baselineSelect.disabled = false;
    }
});
document.getElementById("trajectory-baseline-create")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const suggested = `Stable ${new Date().toLocaleDateString()}`;
    const name = window.prompt("Name this aggregate trajectory baseline", suggested);
    if (!name?.trim()) return;
    button.disabled = true;
    try {
        await window.apiClient.createTrajectoryBaseline(
            name.trim(),
            Number(analyticsDays?.value || 30),
            true,
        );
        analyticsPayload = null;
        await loadProjectAnalytics(true);
    } catch (error) {
        button.title = error.message || "Could not save baseline";
    } finally {
        button.disabled = false;
    }
});
baselineDelete?.addEventListener("click", async () => {
    const baselineId = analyticsPayload?.active_baseline?.baseline_id;
    if (!baselineId || !window.confirm("Delete the active trajectory baseline? Existing alerts will be resolved.")) return;
    baselineDelete.disabled = true;
    try {
        await window.apiClient.deleteTrajectoryBaseline(baselineId);
        analyticsPayload = null;
        await loadProjectAnalytics(true);
    } finally {
        baselineDelete.disabled = false;
    }
});
document.getElementById("trajectory-refresh")?.addEventListener("click", () => loadTrajectory(true));
loadOlderButton?.addEventListener("click", loadOlderTrajectory);
ledger?.closest(".trajectory-ledger-wrap")?.addEventListener("scroll", () => {
    if (ledgerScrollFrame || records.length <= LEDGER_VIRTUALIZATION_THRESHOLD) return;
    ledgerScrollFrame = requestAnimationFrame(() => {
        ledgerScrollFrame = 0;
        renderLedger();
    });
}, { passive: true });
document.getElementById("trajectory-export")?.addEventListener("click", exportTrajectory);
document.getElementById("trajectory-report")?.addEventListener("click", downloadDebugReport);
document.querySelectorAll("[data-trajectory-mode]").forEach((button) => {
    button.addEventListener("click", () => {
        timelineMode = button.dataset.trajectoryMode || "duration";
        renderOverview();
        renderTimingBreakdown();
    });
});
document.getElementById("trajectory-inspector-close")?.addEventListener("click", () => {
    selectedId = "";
    inspector.hidden = true;
    renderLedger();
});
search?.addEventListener("input", () => {
    invalidateLedgerView();
    stopReplay(true);
    renderOverview();
    renderLedger();
});
kindFilter?.addEventListener("change", () => {
    invalidateLedgerView();
    stopReplay(true);
    renderOverview();
    renderLedger();
});
problemsOnly?.addEventListener("change", () => {
    invalidateLedgerView();
    stopReplay(true);
    renderOverview();
    renderLedger();
});
bookmarksOnly?.addEventListener("change", () => {
    invalidateLedgerView();
    stopReplay(true);
    renderOverview();
    renderLedger();
});
turnFilter?.addEventListener("change", () => {
    focusedTurnId = String(turnFilter.value || "");
    causalRootId = "";
    selectedId = "";
    invalidateLedgerView();
    if (inspector) inspector.hidden = true;
    stopReplay(true);
    renderAll();
});
replayPlay?.addEventListener("click", toggleReplay);
document.getElementById("trajectory-replay-prev")?.addEventListener("click", () => {
    stopReplay(false);
    stepReplay(-1);
});
document.getElementById("trajectory-replay-next")?.addEventListener("click", () => {
    stopReplay(false);
    stepReplay(1);
});
document.getElementById("trajectory-replay-reset")?.addEventListener("click", () => {
    focusedTurnId = "";
    causalRootId = "";
    selectedId = "";
    invalidateLedgerView();
    if (inspector) inspector.hidden = true;
    stopReplay(true);
    renderAll();
});
chainClear?.addEventListener("click", () => {
    causalRootId = "";
    invalidateLedgerView();
    stopReplay(true);
    renderAll();
});
document.getElementById("trajectory-compare-open")?.addEventListener("click", () => {
    compareOpen = !compareOpen;
    renderComparison();
});
healthButton?.addEventListener("click", () => {
    if (!diagnosticsPanel) return;
    diagnosticsPanel.hidden = !diagnosticsPanel.hidden;
    healthButton.setAttribute("aria-expanded", String(!diagnosticsPanel.hidden));
});
integrityButton?.addEventListener("click", () => {
    if (!diagnosticsPanel) return;
    diagnosticsPanel.hidden = false;
    healthButton?.setAttribute("aria-expanded", "true");
    requestAnimationFrame(() => {
        document.getElementById("trajectory-integrity")?.scrollIntoView({ block: "nearest" });
    });
});
document.getElementById("trajectory-diagnostics-close")?.addEventListener("click", () => {
    if (diagnosticsPanel) diagnosticsPanel.hidden = true;
    healthButton?.setAttribute("aria-expanded", "false");
});

const inspectorResize = document.getElementById("trajectory-inspector-resize");
const savedInspectorWidth = Number(localStorage.getItem(INSPECTOR_WIDTH_KEY) || 0);
if (inspector && savedInspectorWidth >= 320) {
    inspector.style.setProperty("--trajectory-inspector-width", `${savedInspectorWidth}px`);
}
inspectorResize?.addEventListener("pointerdown", (event) => {
    if (!inspector || window.matchMedia("(max-width: 900px)").matches) return;
    event.preventDefault();
    inspectorResize.setPointerCapture(event.pointerId);
    inspector.classList.add("resizing");
    const startX = event.clientX;
    const startWidth = inspector.getBoundingClientRect().width;
    const onMove = (moveEvent) => {
        const next = Math.max(320, Math.min(720, startWidth + startX - moveEvent.clientX));
        inspector.style.setProperty("--trajectory-inspector-width", `${Math.round(next)}px`);
    };
    const onEnd = () => {
        inspector.classList.remove("resizing");
        const width = Math.round(inspector.getBoundingClientRect().width);
        localStorage.setItem(INSPECTOR_WIDTH_KEY, String(width));
        inspectorResize.removeEventListener("pointermove", onMove);
        inspectorResize.removeEventListener("pointerup", onEnd);
        inspectorResize.removeEventListener("pointercancel", onEnd);
    };
    inspectorResize.addEventListener("pointermove", onMove);
    inspectorResize.addEventListener("pointerup", onEnd);
    inspectorResize.addEventListener("pointercancel", onEnd);
});

document.addEventListener("trajectory-open-execution", async (event) => {
    const detail = event.detail || {};
    externalTrajectoryUrl = String(detail.url || "");
    externalTrajectorySessionId = String(detail.sessionId || "");
    externalTrajectoryTitle = String(detail.title || "Execution");
    if (!externalTrajectoryUrl) return;
    trajectoryPayload = null;
    records = [];
    selectedId = "";
    selectedTab = "Summary";
    pendingAnalyticsEventId = String(detail.eventId || "");
    focusedTurnId = "";
    causalRootId = "";
    compareOpen = false;
    compareBaseId = "";
    compareTargetId = "";
    trajectoryLimit = 750;
    pendingScrollAnchor = "";
    invalidateLedgerView();
    stopReplay(true);
    setSurface("trajectory");
    await loadTrajectory(true);
});

document.addEventListener("conversation-changed", (event) => {
    activeConversationId = event.detail?.conversation?.conversation_id || "";
    externalTrajectoryUrl = "";
    externalTrajectorySessionId = "";
    externalTrajectoryTitle = "";
    trajectoryPayload = null;
    records = [];
    selectedId = "";
    focusedTurnId = "";
    causalRootId = "";
    compareOpen = false;
    compareBaseId = "";
    compareTargetId = "";
    trajectoryLimit = pendingAnalyticsEventId ? 5000 : 750;
    pendingScrollAnchor = "";
    invalidateLedgerView();
    stopReplay(true);
    if (!trajectoryPanel?.hidden) loadTrajectory(true);
});

document.addEventListener("keydown", (event) => {
    if (trajectoryPanel?.hidden || event.altKey || event.ctrlKey || event.metaKey) return;
    const tag = String(event.target?.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag) || event.target?.isContentEditable) return;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault();
        stopReplay(false);
        stepReplay(1);
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        stopReplay(false);
        stepReplay(-1);
    } else if (event.key === "Escape") {
        event.preventDefault();
        focusedTurnId = "";
        causalRootId = "";
        selectedId = "";
        invalidateLedgerView();
        if (inspector) inspector.hidden = true;
        stopReplay(true);
        renderAll();
    }
});

document.addEventListener("trajectory-refresh", () => {
    scheduleLiveRefresh();
});

window.apiClient?.onRuntimeEvent?.((event) => {
    if (event?.type === "trajectory.analytics.changed") {
        analyticsPayload = null;
        if (analyticsPanel?.hidden) {
            if (alertCount) {
                alertCount.hidden = false;
                alertCount.textContent = "!";
            }
        } else {
            scheduleAnalyticsRefresh();
        }
        return;
    }
    if (event?.type !== "trajectory.changed") return;
    const payload = event.payload && typeof event.payload === "object"
        ? event.payload : event;
    if (!analyticsPanel?.hidden) scheduleAnalyticsRefresh();
    const conversationId = payload.conversation_id || event.conversation_id || "";
    const activeSessionId = externalTrajectorySessionId || activeConversationId;
    if (!conversationId || conversationId !== activeSessionId) return;
    setLiveStatus("updating", `${payload.change || "event"} · ${payload.kind || "trajectory"}`);
    scheduleLiveRefresh(payload.status === "running" ? 100 : 180);
});

window.apiClient?.onRuntimeStatus?.((nextStatus) => {
    setLiveStatus(nextStatus || "disconnected");
});
window.apiClient?.connectRuntimeStream?.();

window.addEventListener("DOMContentLoaded", async () => {
    if (activeConversationId || !window.apiClient) return;
    try {
        const payload = await window.apiClient.getConversations();
        activeConversationId = payload.active_conversation_id || "";
    } catch (_) {
        // app.js will emit conversation-changed after its own initialization.
    }
});
