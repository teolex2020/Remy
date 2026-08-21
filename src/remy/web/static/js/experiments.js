/** Experiment Lab — sequential, evidence-oriented multi-model collaboration. */

import { showConfirm } from "./ui.js?v=1.21";

const root = document.getElementById("experiments-content");
let models = [];
let rolePresets = [];
let experiments = [];
let selectedId = "";
let pollTimer = null;
let canvasEditor = null;
let canvasSelectedNode = null;
let experimentSection = "experiments";
let selfModProposals = [];
let selfModEvalCases = [];
let selfModConstraints = {};
let selfModPolicy = {};
let selfModSelectedId = "";
let selfModCreating = false;
let selfModTelemetryById = {};

const CANVAS_BLOCKS = [
    { type: "problem", label: "Problem", icon: "?", color: "#ef4444", mandatory: true, description: "Defines the question, source documents, constraints, and the result that would count as success.", use: "Start every experiment here.", input: "Your question, criteria, and uploaded sources.", output: "A shared research brief for all connected branches.", data: { title: "New experiment", problem: "", success_criteria: "", domain: "general", attachments: [] } },
    { type: "role_model", label: "Model Role", icon: "AI", color: "#6366f1", description: "Assigns one connected model a clear responsibility, working method, and expected contribution.", use: "Choose a role preset first; use Custom role only when the existing responsibilities do not fit.", input: "Problem plus only the datasets allowed for this role.", output: "One evidence-labelled contribution per round.", data: { role_preset: "investigator", label: "Investigator", role: "investigator", role_instruction: "", model: "", custom_prompt: "", visible_datasets: [], web_access: false } },
    { type: "private_data", label: "Private Data", icon: "D", color: "#14b8a6", description: "Stores experiment-only evidence that is not written to long-term memory.", use: "Use for measurements, source excerpts, tables, or confidential notes.", input: "Pasted text or extracted document content.", output: "A named dataset available to permitted roles.", data: { name: "Canvas data", content: "" } },
    { type: "context", label: "Additional Context", icon: "C", color: "#0ea5e9", description: "Adds background facts or assumptions shared by the whole group.", use: "Use for definitions and known constraints that are not the main problem.", input: "Trusted explanatory text.", output: "Shared context included in every participant prompt.", data: { content: "" } },
    { type: "prompt", label: "Prompt", icon: "P", color: "#a855f7", description: "Adds an explicit instruction about how the group should approach the work.", use: "Use for method, format, tone, or comparison requirements.", input: "A bounded instruction.", output: "A shared instruction applied to every role.", data: { prompt: "" } },
    { type: "web_search", label: "Web Search", icon: "W", color: "#0284c7", description: "Retrieves fresh web results before the discussion starts.", use: "Use only when current external information is necessary.", input: "A query; {problem} inserts the problem text.", output: "A timestamped runtime source with URLs and extracted results.", data: { query: "{problem}", num_results: 5 } },
    { type: "world_rules", label: "World Rules", icon: "O", color: "#38bdf8", description: "Defines the simulated environment, permitted actions, constraints, and time represented by each round.", use: "Use in scenario simulations.", input: "Initial world state and a time step.", output: "The same starting rules for every independent replica.", data: { environment: "", time_step: "1 simulated day", seed: 42 } },
    { type: "intervention", label: "Intervention", icon: "!", color: "#fb7185", description: "Injects a planned what-if event at a selected simulation round.", use: "Use to test policy changes, shocks, discoveries, or failures.", input: "Event text and target round.", output: "A visible event applied to every replica at that round.", data: { round: 2, content: "" } },
    { type: "replicas", label: "Independent Replicas", icon: "R", color: "#c084fc", description: "Repeats the scenario with isolated evidence boards so one run cannot influence another.", use: "Use to distinguish stable trajectories from one-off model variation.", input: "Number of runs, from 1 to 3.", output: "Separate trajectories compared only by the final chair.", data: { count: 3 } },
    { type: "discussion_group", label: "Discussion Group", icon: "G", color: "#f59e0b", mandatory: true, description: "Controls turn order and the maximum number of collaborative rounds.", use: "This is the execution hub for model roles.", input: "Contributions from all connected roles.", output: "Committed turns sent to the shared board.", data: { rounds: 2, turn_policy: "adaptive" } },
    { type: "shared_board", label: "Shared Board", icon: "B", color: "#10b981", mandatory: true, description: "Stores only completed contributions, hypotheses, critiques, and evidence provenance.", use: "Use as the single source of truth for the experiment.", input: "Committed role outputs.", output: "A durable evidence ledger for review and synthesis.", data: { evidence_required: true } },
    { type: "peer_review", label: "Peer Review", icon: "P", color: "#f97316", description: "Requires participants to challenge concrete earlier claims and propose discriminating tests.", use: "Use when error detection matters more than speed.", input: "Claims already committed to the board.", output: "Critiques, counterexamples, and resolution tests.", data: { mode: "cross_review" } },
    { type: "success_gate", label: "Success Gate", icon: "Y", color: "#22c55e", mandatory: true, description: "Checks whether minimum rounds and confidence requirements are met before finishing.", use: "Use to prevent premature synthesis.", input: "Round count and confidence values from committed turns.", output: "Continue another round or allow synthesis.", data: { min_rounds: 1, min_avg_confidence: 0.65 } },
    { type: "synthesis", label: "Synthesis", icon: "S", color: "#ec4899", mandatory: true, description: "Lets one chair model merge results while preserving disagreements and limitations.", use: "This must be the final block.", input: "The complete shared board or all scenario replicas.", output: "A bounded report with consensus, disagreements, tests, and confidence.", data: { model: "", require_approval: false } },
];

const esc = (value) => {
    const node = document.createElement("div");
    node.textContent = String(value ?? "");
    return node.innerHTML;
};

async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const detail = Array.isArray(data.detail)
            ? data.detail.map((item) => item?.msg || String(item)).join(" · ")
            : data.detail;
        throw new Error(typeof detail === "string" ? detail : "Experiment request failed.");
    }
    return data;
}

export async function loadExperiments() {
    clearTimeout(pollTimer);
    if (!root) return;
    if (experimentSection === "self-modification") {
        await loadSelfModificationLab();
        return;
    }
    root.innerHTML = `<div class="skeleton-card" style="height:180px"></div>`;
    try {
        const [modelData, roleData, experimentData] = await Promise.all([
            api("/api/experiments/models"),
            api("/api/experiments/roles").catch(() => ({ roles: [] })),
            api("/api/experiments"),
        ]);
        models = modelData.models || [];
        rolePresets = roleData.roles || [];
        experiments = experimentData.experiments || [];
        renderShell();
        if (selectedId || experiments.length) await openExperiment(selectedId || experiments[0].experiment_id);
    } catch (error) {
        root.innerHTML = `<div class="settings-status" style="color:var(--red)">${esc(error.message)}</div>`;
    }
}

function statusColor(status) {
    if (status === "completed") return "var(--green)";
    if (["failed", "cancelled"].includes(status)) return "var(--red)";
    if (["running", "queued", "pausing"].includes(status)) return "var(--yellow)";
    if (["paused", "waiting_approval"].includes(status)) return "var(--blue)";
    return "var(--text-muted)";
}

function renderShell() {
    root.innerHTML = `
        ${renderExperimentTabs("experiments")}
        <div class="experiment-main-grid">
            <aside class="settings-section" style="margin:0;align-self:start">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:12px"><button id="exp-new" class="btn btn-primary">+ Quick</button><button id="exp-new-canvas" class="btn btn-outline">Canvas</button></div>
                <div id="exp-create" class="hidden"></div>
                <div id="exp-list">${renderList()}</div>
            </aside>
            <section id="exp-detail" class="settings-section" style="margin:0;min-width:0">
                <div class="settings-hint">Select an experiment or create a new one.</div>
            </section>
        </div>`;
    document.getElementById("exp-new")?.addEventListener("click", renderCreate);
    document.getElementById("exp-new-canvas")?.addEventListener("click", renderCanvasDesigner);
    bindExperimentTabs();
    bindList();
}

function renderExperimentTabs(active) {
    return `
        <div class="experiment-mode-tabs" role="tablist" aria-label="Experiment Lab mode">
            <button type="button" class="experiment-mode-tab ${active === "experiments" ? "active" : ""}" data-experiment-section="experiments" role="tab" aria-selected="${active === "experiments"}">Research experiments</button>
            <button type="button" class="experiment-mode-tab ${active === "self-modification" ? "active" : ""}" data-experiment-section="self-modification" role="tab" aria-selected="${active === "self-modification"}">Self-improvement Lab</button>
            <span class="experiment-mode-note">Prompt-only · approval gated</span>
        </div>`;
}

function bindExperimentTabs() {
    document.querySelectorAll("[data-experiment-section]").forEach((button) => {
        button.addEventListener("click", async () => {
            const next = button.dataset.experimentSection || "experiments";
            if (next === experimentSection) return;
            experimentSection = next;
            if (next === "self-modification") await loadSelfModificationLab();
            else await loadExperiments();
        });
    });
}

const SELF_MOD_STAGES = [
    { key: "proposal", label: "Proposal" },
    { key: "eval", label: "Eval" },
    { key: "approval", label: "Approve" },
    { key: "canary", label: "Canary" },
    { key: "release", label: "Release" },
];

function selfModStageIndex(status) {
    if (["draft", "eval_failed"].includes(status)) return status === "eval_failed" ? 1 : 0;
    if (status === "eval_passed") return 2;
    if (status === "approved") return 3;
    if (status === "canary") return 3;
    if (status === "canary_passed") return 4;
    if (["active", "superseded"].includes(status)) return 5;
    if (status === "rolled_back") return 4;
    return 0;
}

function selfModStatusTone(status) {
    if (["active", "eval_passed", "canary_passed"].includes(status)) return "success";
    if (["eval_failed", "rolled_back"].includes(status)) return "danger";
    if (["approved", "canary"].includes(status)) return "warning";
    return "neutral";
}

function formatSelfModStatus(status) {
    return String(status || "draft").replaceAll("_", " ");
}

function shortHash(value) {
    const hash = String(value || "");
    return hash ? `${hash.slice(0, 10)}…${hash.slice(-8)}` : "—";
}

function formatSelfModTime(value) {
    if (!value) return "";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
}

async function loadSelfModificationLab() {
    clearTimeout(pollTimer);
    if (!root) return;
    experimentSection = "self-modification";
    root.innerHTML = `${renderExperimentTabs("self-modification")}<div class="self-mod-loading"><div class="skeleton-card"></div><div class="skeleton-card"></div></div>`;
    bindExperimentTabs();
    try {
        const [proposalData, caseData, modelData] = await Promise.all([
            api("/api/experiments/self-modifications"),
            api("/api/trajectory/analytics/eval-cases?limit=100").catch(() => ({ cases: [] })),
            models.length ? Promise.resolve({ models }) : api("/api/experiments/models").catch(() => ({ models: [] })),
        ]);
        selfModProposals = proposalData.proposals || [];
        selfModConstraints = proposalData.constraints || {};
        selfModPolicy = proposalData.policy || {};
        selfModEvalCases = caseData.cases || [];
        models = modelData.models || models;
        if (selfModSelectedId && !selfModProposals.some((item) => item.proposal_id === selfModSelectedId)) selfModSelectedId = "";
        if (!selfModSelectedId && selfModProposals.length) selfModSelectedId = selfModProposals[0].proposal_id;
        const selected = selfModProposals.find((item) => item.proposal_id === selfModSelectedId);
        if (selected?.status === "canary") {
            const telemetryData = await api(`/api/experiments/self-modifications/${encodeURIComponent(selected.proposal_id)}/canary-telemetry`).catch(() => ({ telemetry: null }));
            if (telemetryData.telemetry) selfModTelemetryById[selected.proposal_id] = telemetryData.telemetry;
        }
        renderSelfModificationLab();
    } catch (error) {
        root.innerHTML = `${renderExperimentTabs("self-modification")}<div class="settings-status self-mod-error">${esc(error.message)}</div>`;
        bindExperimentTabs();
    }
}

function renderSelfModificationLab() {
    const selected = selfModProposals.find((item) => item.proposal_id === selfModSelectedId) || null;
    const immutable = selfModConstraints.immutable || ["code", "base_prompt", "tools", "policy", "approval", "sandbox"];
    root.innerHTML = `
        ${renderExperimentTabs("self-modification")}
        <section class="self-mod-safety" aria-label="Self-improvement safety boundary">
            <div class="self-mod-safety-icon">◆</div>
            <div><strong>Safe self-improvement boundary</strong><p>Only additive <code>${esc(selfModConstraints.target || "agent.guidance")}</code> guidance can change. ${immutable.map(esc).join(", ")} remain immutable.</p></div>
            <span>Human approval required</span>
        </section>
        ${renderSelfModPolicy()}
        <div class="self-mod-layout">
            <aside class="self-mod-sidebar">
                <div class="self-mod-sidebar-header"><div><strong>Proposals</strong><span>${selfModProposals.length} versions</span></div><button id="self-mod-new" class="btn btn-primary" type="button">+ Proposal</button></div>
                <div class="self-mod-proposal-list">
                    ${selfModProposals.length ? selfModProposals.map(renderSelfModListItem).join("") : `<div class="self-mod-empty"><strong>No proposals yet</strong><span>Create a bounded guidance experiment. Nothing changes at runtime until eval, approval and canary gates pass.</span></div>`}
                </div>
            </aside>
            <main id="self-mod-detail" class="self-mod-detail">
                ${selfModCreating || !selected ? renderSelfModCreate() : renderSelfModProposal(selected)}
            </main>
        </div>`;
    bindExperimentTabs();
    bindSelfModificationLab(selected);
}

function renderSelfModPolicy() {
    const policy = selfModPolicy || {};
    const source = policy.source === "project" ? "Project override" : "Safe defaults";
    return `<details class="self-mod-policy">
        <summary><div><strong>Canary policy</strong><span>Project-scoped thresholds, confidence and alert timing</span></div><i>${esc(source)}</i></summary>
        <form id="self-mod-policy-form">
            <div class="self-mod-policy-grid">
                <label><span>Safety floor / cohort</span><input class="input" name="minimum_requests_per_cohort" type="number" min="5" max="100" value="${Number(policy.minimum_requests_per_cohort || 5)}" required><small>Early regression detection</small></label>
                <label><span>Promotion target / cohort</span><input class="input" name="target_requests_per_cohort" type="number" min="5" max="10000" value="${Number(policy.target_requests_per_cohort || 20)}" required><small>Must be at least the safety floor</small></label>
                <label><span>Observation window (seconds)</span><input class="input" name="minimum_observation_seconds" type="number" min="0" max="604800" value="${Number(policy.minimum_observation_seconds ?? 300)}" required><small>Minimum time before promotion</small></label>
                <label><span>Stale inconclusive alert (seconds)</span><input class="input" name="inconclusive_alert_seconds" type="number" min="0" max="2592000" value="${Number(policy.inconclusive_alert_seconds ?? 1800)}" required><small>Must not be shorter than observation</small></label>
                <label><span>Verification coverage (%)</span><input class="input" name="minimum_verification_coverage" type="number" min="50" max="100" step="1" value="${Number(policy.minimum_verification_coverage || 0.8) * 100}" required><small>Audited completed responses</small></label>
                <label><span>Confidence level (%)</span><input class="input" name="confidence_level" type="number" min="80" max="99.9" step="0.1" value="${Number(policy.confidence_level || 0.95) * 100}" required><small>Two-sided intervals</small></label>
                <label><span>Failure margin (%)</span><input class="input" name="failure_rate_margin" type="number" min="0" max="25" step="0.1" value="${Number(policy.failure_rate_margin ?? 0.02) * 100}" required><small>Maximum non-inferiority delta</small></label>
                <label><span>Unsupported margin (%)</span><input class="input" name="unsupported_rate_margin" type="number" min="0" max="25" step="0.1" value="${Number(policy.unsupported_rate_margin ?? 0.02) * 100}" required><small>Maximum factuality delta</small></label>
                <label><span>Latency multiplier</span><input class="input" name="latency_multiplier" type="number" min="1" max="5" step="0.1" value="${Number(policy.latency_multiplier || 1.5)}" required><small>Candidate versus baseline</small></label>
            </div>
            <div class="self-mod-policy-actions"><label class="self-mod-policy-toggle"><input name="alerts_enabled" type="checkbox" ${policy.alerts_enabled !== false ? "checked" : ""}><span>Send deduplicated incidents for regression and stale inconclusive canaries</span></label><button id="self-mod-policy-save" class="btn btn-primary" type="submit">Save project policy</button><span id="self-mod-policy-status" role="status"></span></div>
        </form>
    </details>`;
}

function renderSelfModListItem(item) {
    const active = item.proposal_id === selfModSelectedId && !selfModCreating;
    return `
        <button type="button" class="self-mod-proposal-item ${active ? "active" : ""}" data-self-mod-proposal="${esc(item.proposal_id)}">
            <span class="self-mod-proposal-top"><b>${esc(item.rationale || "Guidance proposal")}</b><i class="self-mod-status ${selfModStatusTone(item.status)}">${esc(formatSelfModStatus(item.status))}</i></span>
            <span class="self-mod-proposal-copy">${esc(item.candidate_text)}</span>
            <span class="self-mod-proposal-meta"><code>${esc(shortHash(item.candidate_hash))}</code><time>${esc(formatSelfModTime(item.updated_at))}</time></span>
        </button>`;
}

function renderSelfModCreate() {
    return `
        <form id="self-mod-create-form" class="self-mod-create-form">
            <header><div><span class="self-mod-eyebrow">Immutable candidate</span><h3>Propose additive agent guidance</h3></div></header>
            <label><span>What behavior should improve?</span><textarea id="self-mod-candidate" class="input" rows="9" minlength="20" maxlength="4000" required placeholder="Example: Before finalizing a research answer, explicitly distinguish verified facts from inferences and list unresolved evidence gaps."></textarea><small>20–4,000 characters. This cannot override system instructions, safety, approvals, tools, or sandbox rules.</small></label>
            <label><span>Why should this improve the agent?</span><textarea id="self-mod-rationale" class="input" rows="4" maxlength="4000" placeholder="Expected benefit, observed failure mode, and how the eval cases demonstrate improvement."></textarea></label>
            <div class="self-mod-form-actions"><button id="self-mod-create-submit" type="submit" class="btn btn-primary">Create immutable proposal</button>${selfModProposals.length ? `<button id="self-mod-create-cancel" type="button" class="btn btn-outline">Cancel</button>` : ""}<span id="self-mod-action-status" role="status"></span></div>
        </form>`;
}

function renderSelfModProgress(item) {
    const current = selfModStageIndex(item.status);
    const failed = ["eval_failed", "rolled_back"].includes(item.status);
    return `<ol class="self-mod-progress">${SELF_MOD_STAGES.map((stage, index) => {
        const state = index < current ? "complete" : index === current ? (failed ? "failed" : "current") : "pending";
        return `<li class="${state}"><span>${index < current ? "✓" : index + 1}</span><b>${stage.label}</b></li>`;
    }).join("")}</ol>`;
}

function renderSelfModProposal(item) {
    const evaluation = item.evaluation || {};
    const canary = item.canary_evaluation || {};
    return `
        <article class="self-mod-proposal-detail">
            <header class="self-mod-detail-header">
                <div><span class="self-mod-eyebrow">${esc(item.target)} · ${esc(item.source || "operator")}</span><h3>${esc(item.rationale || "Guidance proposal")}</h3><span class="self-mod-status ${selfModStatusTone(item.status)}">${esc(formatSelfModStatus(item.status))}</span></div>
                <div class="self-mod-header-actions"><button id="self-mod-open-trajectory" type="button" class="btn btn-outline">Trajectory</button><button id="self-mod-copy-hash" type="button" class="btn btn-outline">Copy hash</button></div>
            </header>
            ${renderSelfModProgress(item)}
            <section class="self-mod-version-card">
                <div><span>Candidate SHA-256</span><code title="${esc(item.candidate_hash)}">${esc(item.candidate_hash)}</code></div>
                <div><span>Baseline</span><code title="${esc(item.baseline_hash)}">${esc(shortHash(item.baseline_hash))}</code></div>
                <pre>${esc(item.candidate_text)}</pre>
            </section>
            ${renderSelfModEvidence(item, evaluation, canary)}
            ${renderSelfModActions(item)}
            <div id="self-mod-action-status" class="self-mod-action-status" role="status"></div>
        </article>`;
}

function renderSelfModEvidence(item, evaluation, canary) {
    const evalState = evaluation.matrix_id
        ? `<div class="self-mod-metric"><span>Eval matrix</span><b>${evaluation.gate_passed ? "Passed" : "Blocked"}</b><small>${evaluation.passed_count || 0}/${evaluation.case_count || 0} cases · score ${Number(evaluation.avg_score || 0).toFixed(0)}</small></div>`
        : `<div class="self-mod-metric"><span>Eval matrix</span><b>Not run</b><small>At least ${selfModConstraints.min_eval_cases || 3} durable cases required</small></div>`;
    const canaryState = item.canary_percent
        ? `<div class="self-mod-metric"><span>Canary</span><b>${item.canary_percent}% cohort</b><small>${canary.gate_passed === true ? "Gate passed" : canary.gate_passed === false ? "Gate failed · auto-rolled back" : "Collecting evidence"}</small></div>`
        : `<div class="self-mod-metric"><span>Canary</span><b>Not started</b><small>Stable per-session routing</small></div>`;
    return `<section class="self-mod-evidence"><h4>Release evidence</h4><div class="self-mod-metrics">${evalState}${canaryState}<div class="self-mod-metric"><span>Operator</span><b>${esc(item.approved_by || "Not approved")}</b><small>${esc(formatSelfModTime(item.approved_at)) || "Exact hash confirmation required"}</small></div></div></section>`;
}

function renderSelfModActions(item) {
    if (["draft", "eval_failed"].includes(item.status)) return renderSelfModEvalAction(item);
    if (item.status === "eval_passed") return `<section class="self-mod-gate"><div><span>Gate 2</span><strong>Human approval</strong><p>Confirm the exact immutable candidate hash before any live traffic can see it.</p></div><button id="self-mod-approve" class="btn btn-primary" type="button">Approve exact hash</button></section>`;
    if (item.status === "approved") return `<section class="self-mod-gate"><div><span>Gate 3</span><strong>Start bounded canary</strong><p>Route a stable cohort to the candidate while all other sessions keep the baseline.</p></div><label class="self-mod-percent"><span>Traffic</span><input id="self-mod-canary-percent" class="input" type="number" min="5" max="25" value="10"><b>%</b></label><button id="self-mod-canary-start" class="btn btn-primary" type="button">Start canary</button></section>`;
    if (item.status === "canary") return `${renderSelfModAutomaticTelemetry(item)}${renderSelfModManualCanaryFallback()}<section class="self-mod-danger-zone"><div><strong>Manual rollback</strong><span>Immediately stop candidate routing.</span></div><button id="self-mod-rollback" class="btn btn-danger" type="button">Rollback</button></section>`;
    if (item.status === "canary_passed") return `<section class="self-mod-gate success"><div><span>Gate 4 passed</span><strong>Promote candidate</strong><p>The canary met request, failure, unsupported-output and latency thresholds.</p></div><button id="self-mod-promote" class="btn btn-primary" type="button">Promote to active</button></section><section class="self-mod-danger-zone"><div><strong>Do not promote</strong><span>Return all traffic to the baseline.</span></div><button id="self-mod-rollback" class="btn btn-danger" type="button">Rollback</button></section>`;
    if (item.status === "active") return `<section class="self-mod-gate success"><div><span>Active guidance</span><strong>Serving 100% of new sessions</strong><p>The previous active version can be restored with one explicit rollback.</p></div></section><section class="self-mod-danger-zone"><div><strong>Rollback active guidance</strong><span>Restore the preceding approved version when available.</span></div><button id="self-mod-rollback" class="btn btn-danger" type="button">Rollback</button></section>`;
    return `<section class="self-mod-gate muted"><div><span>Lifecycle closed</span><strong>${esc(formatSelfModStatus(item.status))}</strong><p>This immutable version remains available for audit in Trajectory.</p></div></section>`;
}

function renderSelfModEvalAction(item) {
    const minimum = Number(selfModConstraints.min_eval_cases || 3);
    return `<section class="self-mod-eval-gate"><header><div><span>Gate 1</span><strong>Sandbox regression evaluation</strong><p>Select at least ${minimum} durable Trajectory cases. Tool side effects remain disabled during replay.</p></div><span>${selfModEvalCases.length} available</span></header>
        <div class="self-mod-case-list">${selfModEvalCases.length ? selfModEvalCases.map((evalCase, index) => `<label><input type="checkbox" data-self-mod-case value="${esc(evalCase.case_id)}" ${index < minimum ? "checked" : ""}><span><b>${esc(evalCase.name || "Regression case")}</b><small>${esc(evalCase.latest_run ? `${evalCase.latest_run.status} · ${Number(evalCase.latest_run.score || 0).toFixed(0)}% latest run` : "Not replayed yet")}</small></span></label>`).join("") : `<div class="self-mod-empty"><strong>No durable eval cases</strong><span>Create cases from a Trajectory incident dossier first.</span></div>`}</div>
        <div class="self-mod-eval-controls"><label><span>Replay model</span><select id="self-mod-eval-model" class="input"><option value="">Current default</option>${models.map((model) => `<option value="${esc(model.name)}">${esc(model.name)}</option>`).join("")}</select></label><button id="self-mod-run-eval" class="btn btn-primary" type="button" ${selfModEvalCases.length < minimum ? "disabled" : ""}>Run sandbox gate</button></div>
        ${item.status === "eval_failed" ? `<div class="self-mod-gate-failure">Previous gate failed. Review the matrix in Trajectory, revise by creating a new immutable proposal, or rerun after fixing the eval environment.</div>` : ""}
    </section>`;
}

function formatTelemetryPercent(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatObservationSeconds(value) {
    const seconds = Math.max(0, Number(value || 0));
    if (seconds < 60) return `${Math.round(seconds)}s`;
    return `${(seconds / 60).toFixed(seconds < 600 ? 1 : 0)}m`;
}

function renderSelfModAutomaticTelemetry(item) {
    const telemetry = selfModTelemetryById[item.proposal_id] || {};
    const candidate = telemetry.candidate || {};
    const baseline = telemetry.baseline || {};
    const readiness = telemetry.readiness_checks || {};
    const promotionReadiness = telemetry.promotion_readiness_checks || {};
    const requirements = telemetry.requirements || {};
    const observation = telemetry.observation || {};
    const statistics = telemetry.statistics || {};
    const activeAlerts = (telemetry.alerts || []).filter((alert) => alert.status === "open");
    const statisticalChecks = statistics.checks || {};
    const targetRequests = Number(requirements.target_requests_per_cohort || 20);
    const minimumRequests = Number(requirements.minimum_requests_per_cohort || 5);
    const minimumObservation = Number(requirements.minimum_observation_seconds || 300);
    const confidenceLabel = `${(Number(statistics.confidence_level || requirements.confidence_level || 0.95) * 100).toFixed(1).replace(".0", "")}%`;
    const checks = [
        ["Candidate safety floor", readiness.enough_candidate_requests, `${candidate.requests || 0}/${minimumRequests}`],
        ["Baseline safety floor", readiness.enough_baseline_requests, `${baseline.requests || 0}/${minimumRequests}`],
        ["Candidate sample target", promotionReadiness.candidate_sample_target, `${candidate.requests || 0}/${targetRequests}`],
        ["Baseline sample target", promotionReadiness.baseline_sample_target, `${baseline.requests || 0}/${targetRequests}`],
        ["Observation window", promotionReadiness.minimum_observation_window, `${formatObservationSeconds(observation.seconds)}/${formatObservationSeconds(minimumObservation)}`],
        ["Candidate audit coverage", readiness.candidate_verification_coverage, formatTelemetryPercent(candidate.verification_coverage)],
        ["Baseline audit coverage", readiness.baseline_verification_coverage, formatTelemetryPercent(baseline.verification_coverage)],
    ];
    const statisticalRows = [
        ["Failure-rate interval", statisticalChecks.failure_rate_non_inferior, statistics.failure_rate],
        ["Unsupported-rate interval", statisticalChecks.unsupported_rate_non_inferior, statistics.unsupported_rate],
        ["Latency interval", statisticalChecks.latency_non_inferior, statistics.latency_ms],
    ];
    const telemetryHeadline = telemetry.status === "inconclusive"
        ? "Evidence is statistically inconclusive"
        : telemetry.status === "regressed"
            ? "Regression detected"
            : telemetry.ready && telemetry.gate_passed
                ? "Release gate is healthy"
                : "Collecting production evidence";
    const cohort = (name, values) => `<article class="self-mod-cohort-card"><header><strong>${name}</strong><span>${values.requests || 0} requests</span></header><div><span><b>${formatTelemetryPercent(values.failure_rate)}</b><small>Failures</small></span><span><b>${formatTelemetryPercent(values.unsupported_rate)}</b><small>Unsupported</small></span><span><b>${Math.round(Number(values.avg_request_ms || 0))} ms</b><small>Avg latency</small></span><span><b>${values.verified_requests || 0}/${values.completed_requests || 0}</b><small>Audited</small></span></div></article>`;
    return `<section class="self-mod-telemetry status-${esc(telemetry.status || "collecting")}">
        <header><div><span>Live Trajectory telemetry</span><strong>${telemetryHeadline}</strong><p>Candidate and baseline are assigned by stable session bucket, so one chat never switches cohorts and both groups require multiple conversations. Only aggregate status, latency and factuality outcomes are read; prompt and response text are excluded.</p></div><i>${esc(telemetry.status || "collecting")}</i></header>
        <div class="self-mod-cohort-grid">${cohort("Candidate", candidate)}${cohort("Baseline", baseline)}</div>
        ${activeAlerts.length ? `<div class="self-mod-live-alerts">${activeAlerts.map((alert) => `<span class="severity-${esc(alert.severity || "warning")}"><b>${alert.alert_code === "canary_regressed" ? "Regression incident" : "Long-running inconclusive canary"}</b><small>Deduplicated in Incident Center · ${esc(formatSelfModTime(alert.opened_at))}</small></span>`).join("")}</div>` : ""}
        <div class="self-mod-readiness">${checks.map(([label, passed, value]) => `<span class="${passed ? "passed" : "pending"}">${passed ? "✓" : "○"} ${esc(label)} <b>${esc(value)}</b></span>`).join("")}</div>
        <div class="self-mod-confidence">
            <header><div><strong>${confidenceLabel} statistical confidence</strong><span>Non-inferiority intervals must pass before promotion.</span></div><b>${formatTelemetryPercent(statistics.confidence_score)}</b></header>
            <div class="self-mod-confidence-track"><i style="width:${Math.max(0, Math.min(100, Number(statistics.confidence_score || 0) * 100))}%"></i></div>
            <div class="self-mod-statistical-checks">${statisticalRows.map(([label, passed, values]) => {
                const candidateUpper = Number(values?.candidate?.upper || 0);
                const baselineLower = Number(values?.baseline?.lower || 0);
                const unit = label.includes("Latency") ? "ms" : "%";
                const shownCandidate = unit === "%" ? (candidateUpper * 100).toFixed(1) : Math.round(candidateUpper);
                const shownBaseline = unit === "%" ? (baselineLower * 100).toFixed(1) : Math.round(baselineLower);
                return `<span class="${passed ? "passed" : "pending"}">${passed ? "✓" : "○"} ${esc(label)} <small>candidate upper ${shownCandidate}${unit} · baseline lower ${shownBaseline}${unit}</small></span>`;
            }).join("")}</div>
        </div>
        <div class="self-mod-telemetry-actions"><small>${esc(telemetry.privacy || "aggregate-only; prompts and responses excluded")}</small><button id="self-mod-observe-canary" class="btn btn-primary" type="button">Refresh & enforce gate</button></div>
    </section>`;
}

function renderSelfModManualCanaryFallback() {
    const fields = [
        ["candidate_requests", "Candidate requests", "5", "1"], ["baseline_requests", "Baseline requests", "5", "1"],
        ["candidate_failure_rate", "Candidate failure rate", "0", "0.001"], ["baseline_failure_rate", "Baseline failure rate", "0", "0.001"],
        ["candidate_unsupported_rate", "Candidate unsupported rate", "0", "0.001"], ["baseline_unsupported_rate", "Baseline unsupported rate", "0", "0.001"],
        ["candidate_avg_request_ms", "Candidate avg latency (ms)", "0", "1"], ["baseline_avg_request_ms", "Baseline avg latency (ms)", "0", "1"],
    ];
    return `<details class="self-mod-canary-eval"><summary>Diagnostic fallback · enter external aggregate metrics</summary><p>Use only when Trajectory telemetry is unavailable and you have independently measured every value. A failed gate immediately rolls the candidate back.</p><div class="self-mod-canary-fields">${fields.map(([id, label, min, step]) => `<label><span>${label}</span><input id="self-mod-${id}" class="input" type="number" min="${min}" ${id.includes("rate") ? 'max="1"' : ""} step="${step}" required placeholder="Measured value"></label>`).join("")}</div><button id="self-mod-evaluate-canary" class="btn btn-outline" type="button">Enforce external metrics</button></details>`;
}

function bindSelfModificationLab(selected) {
    document.getElementById("self-mod-policy-form")?.addEventListener("submit", saveSelfModPolicy);
    document.getElementById("self-mod-new")?.addEventListener("click", () => { selfModCreating = true; renderSelfModificationLab(); });
    document.getElementById("self-mod-create-cancel")?.addEventListener("click", () => { selfModCreating = false; renderSelfModificationLab(); });
    document.querySelectorAll("[data-self-mod-proposal]").forEach((button) => button.addEventListener("click", () => {
        selfModSelectedId = button.dataset.selfModProposal || "";
        selfModCreating = false;
        renderSelfModificationLab();
    }));
    document.getElementById("self-mod-create-form")?.addEventListener("submit", createSelfModificationProposal);
    if (!selected || selfModCreating) return;
    document.getElementById("self-mod-copy-hash")?.addEventListener("click", () => copySelfModHash(selected));
    document.getElementById("self-mod-open-trajectory")?.addEventListener("click", () => openSelfModTrajectory(selected));
    document.getElementById("self-mod-run-eval")?.addEventListener("click", () => runSelfModificationEval(selected));
    document.getElementById("self-mod-approve")?.addEventListener("click", () => approveSelfModification(selected));
    document.getElementById("self-mod-canary-start")?.addEventListener("click", () => startSelfModificationCanary(selected));
    document.getElementById("self-mod-observe-canary")?.addEventListener("click", () => observeSelfModificationCanary(selected));
    document.getElementById("self-mod-evaluate-canary")?.addEventListener("click", () => evaluateSelfModificationCanary(selected));
    document.getElementById("self-mod-promote")?.addEventListener("click", () => promoteSelfModification(selected));
    document.getElementById("self-mod-rollback")?.addEventListener("click", () => rollbackSelfModification(selected));
}

async function saveSelfModPolicy(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = document.getElementById("self-mod-policy-save");
    const status = document.getElementById("self-mod-policy-status");
    const data = new FormData(form);
    const number = (name) => Number(data.get(name));
    const payload = {
        minimum_requests_per_cohort: number("minimum_requests_per_cohort"),
        target_requests_per_cohort: number("target_requests_per_cohort"),
        minimum_observation_seconds: number("minimum_observation_seconds"),
        inconclusive_alert_seconds: number("inconclusive_alert_seconds"),
        minimum_verification_coverage: number("minimum_verification_coverage") / 100,
        confidence_level: number("confidence_level") / 100,
        failure_rate_margin: number("failure_rate_margin") / 100,
        unsupported_rate_margin: number("unsupported_rate_margin") / 100,
        latency_multiplier: number("latency_multiplier"),
        alerts_enabled: Boolean(form.elements.alerts_enabled?.checked),
    };
    button.disabled = true;
    if (status) status.textContent = "Saving project policy...";
    try {
        const response = await api("/api/experiments/self-modifications/policy", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        selfModPolicy = response.policy || payload;
        Object.assign(selfModConstraints, {
            canary_min_requests_per_cohort: selfModPolicy.minimum_requests_per_cohort,
            canary_target_requests_per_cohort: selfModPolicy.target_requests_per_cohort,
            canary_min_observation_seconds: selfModPolicy.minimum_observation_seconds,
            canary_min_verification_coverage: selfModPolicy.minimum_verification_coverage,
            canary_confidence_level: selfModPolicy.confidence_level,
        });
        if (status) status.textContent = "Saved for this project.";
        window.setTimeout(() => renderSelfModificationLab(), 600);
    } catch (error) {
        if (status) status.textContent = error.message;
        button.disabled = false;
    }
}

function setSelfModStatus(message, tone = "") {
    const node = document.getElementById("self-mod-action-status");
    if (!node) return;
    node.textContent = message;
    node.dataset.tone = tone;
}

async function withSelfModAction(buttonId, action) {
    const button = document.getElementById(buttonId);
    if (button) button.disabled = true;
    setSelfModStatus("Working…");
    try {
        await action();
        await loadSelfModificationLab();
    } catch (error) {
        if (button) button.disabled = false;
        setSelfModStatus(error.message, "danger");
    }
}

async function createSelfModificationProposal(event) {
    event.preventDefault();
    const candidate = document.getElementById("self-mod-candidate")?.value.trim() || "";
    const rationale = document.getElementById("self-mod-rationale")?.value.trim() || "";
    await withSelfModAction("self-mod-create-submit", async () => {
        const data = await api("/api/experiments/self-modifications", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ candidate_text: candidate, rationale, source: "operator" }),
        });
        selfModSelectedId = data.proposal.proposal_id;
        selfModCreating = false;
    });
}

async function copySelfModHash(item) {
    try {
        await navigator.clipboard.writeText(item.candidate_hash);
        setSelfModStatus("Candidate hash copied.", "success");
    } catch (_) {
        setSelfModStatus(item.candidate_hash);
    }
}

function openSelfModTrajectory(item) {
    document.dispatchEvent(new CustomEvent("execution-trajectory-open", { detail: {
        title: `Self-improvement · ${item.rationale || shortHash(item.candidate_hash)}`,
        url: `/api/trajectory/self-modifications/${encodeURIComponent(item.proposal_id)}`,
        sessionId: item.trajectory_session_id || "",
    } }));
}

async function runSelfModificationEval(item) {
    const caseIds = [...document.querySelectorAll("[data-self-mod-case]:checked")].map((node) => node.value);
    const minimum = Number(selfModConstraints.min_eval_cases || 3);
    if (caseIds.length < minimum) { setSelfModStatus(`Select at least ${minimum} eval cases.`, "danger"); return; }
    const preferredModel = document.getElementById("self-mod-eval-model")?.value || "";
    await withSelfModAction("self-mod-run-eval", () => api(`/api/experiments/self-modifications/${encodeURIComponent(item.proposal_id)}/run-evaluation`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_ids: caseIds, preferred_model: preferredModel, name: `Self-improvement ${shortHash(item.candidate_hash)}` }),
    }));
}

async function approveSelfModification(item) {
    const confirmed = await showConfirm("Approve immutable guidance", `Approve the exact SHA-256 candidate ${item.candidate_hash}? This authorizes a canary only; it does not promote the guidance.`);
    if (!confirmed) return;
    await withSelfModAction("self-mod-approve", () => api(`/api/experiments/self-modifications/${encodeURIComponent(item.proposal_id)}/approve`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_candidate_hash: item.candidate_hash, approved_by: "local-operator" }),
    }));
}

async function startSelfModificationCanary(item) {
    const percent = Number(document.getElementById("self-mod-canary-percent")?.value || 10);
    const confirmed = await showConfirm("Start guidance canary", `Route ${percent}% of stable session buckets to candidate ${shortHash(item.candidate_hash)}?`);
    if (!confirmed) return;
    await withSelfModAction("self-mod-canary-start", () => api(`/api/experiments/self-modifications/${encodeURIComponent(item.proposal_id)}/canary`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_candidate_hash: item.candidate_hash, canary_percent: percent }),
    }));
}

async function evaluateSelfModificationCanary(item) {
    const ids = [
        "candidate_requests", "baseline_requests", "candidate_failure_rate", "baseline_failure_rate",
        "candidate_unsupported_rate", "baseline_unsupported_rate", "candidate_avg_request_ms", "baseline_avg_request_ms",
    ];
    const fields = Object.fromEntries(ids.map((id) => [id, document.getElementById(`self-mod-${id}`)]));
    if (ids.some((id) => !String(fields[id]?.value || "").trim() || !fields[id]?.checkValidity())) {
        setSelfModStatus("Enter valid measured values for every canary metric before enforcing the gate.", "danger");
        ids.forEach((id) => fields[id]?.reportValidity());
        return;
    }
    const numeric = (id) => Number(fields[id].value);
    const metrics = {
        candidate_requests: numeric("candidate_requests"), baseline_requests: numeric("baseline_requests"),
        candidate_failure_rate: numeric("candidate_failure_rate"), baseline_failure_rate: numeric("baseline_failure_rate"),
        candidate_unsupported_rate: numeric("candidate_unsupported_rate"), baseline_unsupported_rate: numeric("baseline_unsupported_rate"),
        candidate_avg_request_ms: numeric("candidate_avg_request_ms"), baseline_avg_request_ms: numeric("baseline_avg_request_ms"),
    };
    const confirmed = await showConfirm("Enforce canary gate", "Evaluate these aggregate metrics now? Any failed threshold will immediately roll back the candidate.");
    if (!confirmed) return;
    await withSelfModAction("self-mod-evaluate-canary", () => api(`/api/experiments/self-modifications/${encodeURIComponent(item.proposal_id)}/canary-evaluation`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(metrics),
    }));
}

async function observeSelfModificationCanary(item) {
    await withSelfModAction("self-mod-observe-canary", () => api(`/api/experiments/self-modifications/${encodeURIComponent(item.proposal_id)}/canary-telemetry/observe`, {
        method: "POST",
    }));
}

async function promoteSelfModification(item) {
    const confirmed = await showConfirm("Promote guidance", `Make candidate ${item.candidate_hash} active for 100% of new sessions? The previous active version remains recoverable.`);
    if (!confirmed) return;
    await withSelfModAction("self-mod-promote", () => api(`/api/experiments/self-modifications/${encodeURIComponent(item.proposal_id)}/promote`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_candidate_hash: item.candidate_hash, canary_percent: 25 }),
    }));
}

async function rollbackSelfModification(item) {
    const confirmed = await showConfirm("Rollback guidance", `Stop candidate ${shortHash(item.candidate_hash)} and restore the previous active guidance when available?`);
    if (!confirmed) return;
    await withSelfModAction("self-mod-rollback", () => api(`/api/experiments/self-modifications/${encodeURIComponent(item.proposal_id)}/rollback`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: "operator UI rollback" }),
    }));
}

function renderList() {
    if (!experiments.length) return `<p class="settings-hint">No experiments yet.</p>`;
    return experiments.map((item) => `
        <button class="exp-list-item" data-id="${esc(item.experiment_id)}" style="display:block;width:100%;text-align:left;background:${item.experiment_id === selectedId ? "var(--surface-2)" : "transparent"};border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:8px;color:var(--text);cursor:pointer">
            <div style="font-weight:600">${esc(item.title)}</div>
            <div style="display:flex;justify-content:space-between;margin-top:5px;font-size:11px">
                <span style="color:${statusColor(item.status)}">${esc(item.status)}</span>
                <span class="settings-hint">${item.contribution_count} turns · ${item.calls_used}/${item.max_calls} calls</span>
            </div>
        </button>`).join("");
}

function bindList() {
    document.querySelectorAll(".exp-list-item").forEach((button) => button.addEventListener("click", () => openExperiment(button.dataset.id)));
}

function renderCreate() {
    const panel = document.getElementById("exp-create");
    panel.classList.remove("hidden");
    panel.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px">
            <input id="exp-title" class="input" placeholder="Experiment title">
            <textarea id="exp-problem" class="input" rows="5" placeholder="Problem to solve"></textarea>
            <textarea id="exp-success" class="input" rows="3" placeholder="Success criteria and constraints"></textarea>
            <select id="exp-domain" class="input">
                <option value="general">General research</option>
                <option value="engineering">Engineering</option>
                <option value="data-science">Data science</option>
                <option value="medical">Medical / drug discovery</option>
            </select>
            <div class="settings-hint">Connected models</div>
            <div style="max-height:150px;overflow:auto;border:1px solid var(--border);border-radius:7px;padding:8px">
                ${models.length ? models.map((model, index) => `<label style="display:block;margin:5px 0"><input class="exp-model" type="checkbox" value="${esc(model.name)}" ${index < Math.min(3, models.length) ? "checked" : ""}> ${esc(model.name)} <span class="settings-hint">${esc(model.provider)}</span></label>`).join("") : `<span class="settings-hint">Connect models in Settings first.</span>`}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <label class="settings-hint">Rounds<input id="exp-rounds" type="number" class="input" min="1" max="5" value="2" style="width:100%"></label>
                <label class="settings-hint">Maximum calls<input id="exp-budget" type="number" class="input" min="0" max="100" value="0" style="width:100%" title="0 = turns plus one synthesis call"></label>
            </div>
            <div style="display:flex;gap:7px"><button id="exp-create-save" class="btn btn-primary">Create draft</button><button id="exp-create-cancel" class="btn btn-outline">Cancel</button></div>
            <div id="exp-create-status" class="settings-hint"></div>
        </div>`;
    document.getElementById("exp-create-cancel").onclick = () => panel.classList.add("hidden");
    document.getElementById("exp-create-save").onclick = createExperiment;
}

async function createExperiment() {
    const selectedModels = [...document.querySelectorAll(".exp-model:checked")].map((node) => node.value);
    const status = document.getElementById("exp-create-status");
    try {
        const data = await api("/api/experiments", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                title: document.getElementById("exp-title").value.trim(),
                problem: document.getElementById("exp-problem").value.trim(),
                success_criteria: document.getElementById("exp-success").value.trim(),
                domain: document.getElementById("exp-domain").value,
                models: selectedModels,
                rounds: Number(document.getElementById("exp-rounds").value || 2),
                max_calls: Number(document.getElementById("exp-budget").value || 0),
            }),
        });
        selectedId = data.experiment.experiment_id;
        await loadExperiments();
    } catch (error) {
        status.textContent = error.message;
        status.style.color = "var(--red)";
    }
}

async function renderCanvasDesigner(existingExperiment = null) {
    clearTimeout(pollTimer);
    root.innerHTML = `
        <div style="display:flex;flex-direction:column;height:calc(100vh - 125px);min-height:650px;border:1px solid var(--border);border-radius:10px;overflow:hidden">
            <div style="display:flex;gap:8px;align-items:center;padding:10px;border-bottom:1px solid var(--border);background:var(--surface-2)">
                <button id="exp-canvas-back" class="btn btn-outline">← Experiments</button>
                <select id="exp-canvas-template" class="input"><option value="scientific-panel">Scientific Panel</option><option value="scenario-simulation">Scenario Simulation</option><option value="drug-discovery">Drug Discovery</option><option value="engineering-review">Engineering Review</option><option value="blank">Blank</option></select>
                <button id="exp-canvas-load" class="btn btn-outline">Load template</button>
                <span style="flex:1"></span><span id="exp-canvas-status" class="settings-hint"></span>
                <button id="exp-canvas-validate" class="btn btn-outline">Validate</button>
                <button id="exp-canvas-create" class="btn btn-primary">${existingExperiment ? "Save canvas" : "Create draft"}</button>
            </div>
            <div id="exp-canvas-workspace" style="display:grid;grid-template-columns:220px minmax(0,1fr) 0px;min-height:0;flex:1;transition:grid-template-columns .18s ease">
                <aside style="padding:10px;border-right:1px solid var(--border);overflow:auto"><b>Blocks</b><p class="settings-hint">Click to add a block, then connect its ports.</p><div id="exp-canvas-palette"></div></aside>
                <div id="exp-canvas-drawflow" class="pf-drawflow-container"></div>
                <aside id="exp-canvas-config" style="display:none;min-width:0;overflow:hidden"></aside>
            </div>
        </div>`;
    const palette = document.getElementById("exp-canvas-palette");
    palette.innerHTML = CANVAS_BLOCKS.map((block) => `<button class="exp-canvas-add" data-type="${block.type}" title="${esc(block.description)}" style="display:flex;width:100%;align-items:center;gap:8px;margin:7px 0;padding:9px;border:1px solid var(--border);border-left:3px solid ${block.color};border-radius:7px;background:var(--surface-2);color:var(--text);cursor:pointer;text-align:left"><b style="width:22px;text-align:center">${block.icon}</b><span>${block.label}${block.mandatory ? " *" : ""}</span></button>`).join("");
    if (typeof Drawflow === "undefined") {
        document.getElementById("exp-canvas-status").textContent = "Drawflow is unavailable.";
        return;
    }
    canvasEditor = new Drawflow(document.getElementById("exp-canvas-drawflow"));
    canvasEditor.reroute = true;
    canvasEditor.start();
    canvasEditor.on("nodeSelected", (id) => { canvasSelectedNode = String(id); renderCanvasConfig(String(id)); });
    canvasEditor.on("nodeUnselected", () => { canvasSelectedNode = null; closeCanvasConfigPanel(); });
    document.querySelectorAll(".exp-canvas-add").forEach((button) => button.addEventListener("click", () => {
        const id = addCanvasNode(button.dataset.type);
        if (id === null || id === undefined) return;
        canvasSelectedNode = String(id);
        renderCanvasConfig(String(id));
    }));
    document.getElementById("exp-canvas-back").onclick = () => loadExperiments();
    document.getElementById("exp-canvas-load").onclick = loadCanvasTemplate;
    document.getElementById("exp-canvas-validate").onclick = validateCanvas;
    document.getElementById("exp-canvas-create").onclick = () => createCanvasExperiment(existingExperiment?.experiment_id || "");
    if (existingExperiment?.canvas) {
        importCanvasSchema(existingExperiment.canvas);
        document.getElementById("exp-canvas-status").textContent = "Canvas draft loaded.";
    } else {
        await loadCanvasTemplate();
    }
}

function canvasBlock(type) {
    return CANVAS_BLOCKS.find((item) => item.type === type) || CANVAS_BLOCKS[1];
}

function canvasNodeHtml(type, data) {
    const block = canvasBlock(type);
    const subtitle = type === "role_model" ? (data.label || data.role || "Role") : type === "web_search" ? (data.query || "Search") : type === "world_rules" ? (data.time_step || "World") : type === "intervention" ? `Round ${data.round || 1}` : type === "replicas" ? `${data.count || 1} runs` : "Experiment block";
    return `<div style="border-left:4px solid ${block.color};padding:10px"><div style="font-weight:700">${block.icon} ${block.label}</div><div style="font-size:11px;color:var(--text-muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(subtitle)}</div></div>`;
}

function addCanvasNode(type, x = null, y = null, data = null) {
    if (!canvasEditor) return null;
    const block = canvasBlock(type);
    const currentCount = Object.keys(canvasEditor.export()?.drawflow?.Home?.data || {}).length;
    const nodeData = { ...block.data, ...(data || {}) };
    if (type === "role_model" && !data && rolePresets.length) {
        const preset = rolePresets[0];
        Object.assign(nodeData, {
            role_preset: preset.preset_id,
            label: preset.label,
            role: preset.role,
            role_instruction: preset.instruction,
            web_access: Boolean(preset.recommended_web_access),
        });
    }
    const inputs = type === "problem" ? 0 : 1;
    const outputs = type === "synthesis" ? 0 : 1;
    return canvasEditor.addNode(type, inputs, outputs, x ?? 90 + (currentCount % 4) * 250, y ?? 80 + Math.floor(currentCount / 4) * 170, `exp-node exp-node-${type}`, nodeData, canvasNodeHtml(type, nodeData), false);
}

async function loadCanvasTemplate() {
    const status = document.getElementById("exp-canvas-status");
    try {
        const selectedModels = models.slice(0, Math.min(3, models.length)).map((item) => item.name);
        const data = await api("/api/experiments/canvas/template", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ template_id: document.getElementById("exp-canvas-template").value, models: selectedModels }) });
        importCanvasSchema(data.canvas);
        status.textContent = "Template loaded. Configure nodes and connections.";
        status.style.color = "var(--text-muted)";
    } catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; }
}

function importCanvasSchema(schema) {
    closeCanvasConfigPanel();
    canvasEditor.clear();
    const idMap = {};
    for (const node of schema.nodes || []) idMap[String(node.id)] = addCanvasNode(node.type, node.x, node.y, node.data);
    for (const edge of schema.edges || []) {
        const source = idMap[String(edge.source)], target = idMap[String(edge.target)];
        if (source && target) canvasEditor.addConnection(source, target, "output_1", "input_1");
    }
}

function modelOptions(current) {
    return `<option value="">Select model</option>${models.map((item) => `<option value="${esc(item.name)}" ${item.name === current ? "selected" : ""}>${esc(item.name)}</option>`).join("")}`;
}

function rolePresetFor(data) {
    const presetId = String(data.role_preset || "");
    if (presetId) {
        const exact = rolePresets.find((item) => item.preset_id === presetId);
        if (exact) return exact;
    }
    return rolePresets.find((item) => item.role === data.role) || null;
}

function rolePresetOptions(currentId) {
    const categories = new Map();
    for (const preset of rolePresets) {
        const category = preset.category || "Other";
        if (!categories.has(category)) categories.set(category, []);
        categories.get(category).push(preset);
    }
    const groups = [...categories.entries()].map(([category, presets]) => `
        <optgroup label="${esc(category)}">
            ${presets.map((preset) => `<option value="${esc(preset.preset_id)}" ${preset.preset_id === currentId ? "selected" : ""}>${esc(preset.label)}</option>`).join("")}
        </optgroup>`).join("");
    return `${groups}<option value="custom" ${currentId === "custom" ? "selected" : ""}>Custom role…</option>`;
}

function roleHelpHtml(preset) {
    if (!preset) {
        return `<div class="pf-block-help-body">Define a responsibility that is genuinely different from the other participants. State what this role must examine and what it must return.</div>
            <div class="pf-block-help-tip">Use Custom role only when none of the presets describe the required responsibility.</div>`;
    }
    return `<div style="font-weight:700;margin-bottom:5px">${esc(preset.label)}</div>
        <div>${esc(preset.description)}</div>
        <div style="margin-top:7px"><b>Use when:</b> ${esc(preset.best_for)}</div>
        <div style="margin-top:7px"><b>Expected contribution:</b> ${esc(preset.expected_output)}</div>`;
}

function roleConfigFields(data) {
    const preset = rolePresetFor(data);
    const presetId = preset?.preset_id || "custom";
    return `
        ${canvasField("cn-role-preset", "Choose responsibility", presetId, "select", rolePresetOptions(presetId))}
        <div id="cn-role-help" class="pf-block-help" style="margin:0">
            <div class="pf-block-help-title">What this role contributes</div>
            ${roleHelpHtml(preset)}
        </div>
        ${canvasField("cn-model", "Model assigned to this role", data.model, "select", modelOptions(data.model))}
        ${canvasField("cn-custom", "Additional instructions (optional)", data.custom_prompt, "textarea")}
        <details ${presetId === "custom" ? "open" : ""} style="border:1px solid var(--border);border-radius:8px;padding:9px">
            <summary style="cursor:pointer;font-weight:600">Advanced role settings</summary>
            <div style="display:flex;flex-direction:column;gap:10px;margin-top:10px">
                ${canvasField("cn-label", "Display name", data.label)}
                ${canvasField("cn-role", "Internal role key", data.role)}
                ${canvasField("cn-role-instruction", "Role mission", data.role_instruction || preset?.instruction || "", "textarea")}
                ${canvasField("cn-visible", "Visible dataset names (comma-separated; empty = all)", (data.visible_datasets || []).join(", "))}
                ${canvasField("cn-web", "Allow web-derived context", data.web_access, "checkbox")}
            </div>
        </details>`;
}

function bindRolePresetFields() {
    const selector = document.getElementById("cn-role-preset");
    const help = document.getElementById("cn-role-help");
    selector?.addEventListener("change", () => {
        const preset = rolePresets.find((item) => item.preset_id === selector.value) || null;
        if (help) {
            help.innerHTML = `<div class="pf-block-help-title">What this role contributes</div>${roleHelpHtml(preset)}`;
        }
        if (!preset) return;
        const label = document.getElementById("cn-label");
        const role = document.getElementById("cn-role");
        const instruction = document.getElementById("cn-role-instruction");
        const web = document.getElementById("cn-web");
        if (label) label.value = preset.label;
        if (role) role.value = preset.role;
        if (instruction) instruction.value = preset.instruction;
        if (web) web.checked = Boolean(preset.recommended_web_access);
    });
}

function canvasField(id, label, value, kind = "text", options = "") {
    if (kind === "textarea") return `<label class="settings-hint">${label}<textarea id="${id}" class="input exp-node-field" rows="5" style="width:100%;margin-top:4px">${esc(value)}</textarea></label>`;
    if (kind === "select") return `<label class="settings-hint">${label}<select id="${id}" class="input exp-node-field" style="width:100%;margin-top:4px">${options}</select></label>`;
    if (kind === "checkbox") return `<label><input id="${id}" class="exp-node-field" type="checkbox" ${value ? "checked" : ""}> ${label}</label>`;
    return `<label class="settings-hint">${label}<input id="${id}" class="input exp-node-field" type="${kind}" value="${esc(value)}" style="width:100%;margin-top:4px"></label>`;
}

function openCanvasConfigPanel() {
    const workspace = document.getElementById("exp-canvas-workspace");
    const panel = document.getElementById("exp-canvas-config");
    if (!workspace || !panel) return;
    workspace.style.gridTemplateColumns = "220px minmax(0,1fr) 340px";
    Object.assign(panel.style, { display: "block", padding: "12px", borderLeft: "1px solid var(--border)", overflow: "auto" });
}

function closeCanvasConfigPanel() {
    const workspace = document.getElementById("exp-canvas-workspace");
    const panel = document.getElementById("exp-canvas-config");
    if (!workspace || !panel) return;
    workspace.style.gridTemplateColumns = "220px minmax(0,1fr) 0px";
    Object.assign(panel.style, { display: "none", padding: "0", borderLeft: "0", overflow: "hidden" });
    panel.innerHTML = "";
}

function renderProblemSources(data) {
    const attachments = data.attachments || [];
    return `<div style="border-top:1px solid var(--border);padding-top:10px">
        <b>Source material</b>
        <p class="settings-hint" style="margin:5px 0 8px">Attach evidence now. It becomes private experiment data and is not written to long-term memory.</p>
        ${attachments.length ? `<div style="display:grid;gap:6px;margin-bottom:9px">${attachments.map((item, index) => `<div style="display:flex;justify-content:space-between;gap:7px;align-items:center;border:1px solid var(--border);border-radius:7px;padding:7px"><span style="min-width:0;overflow:hidden;text-overflow:ellipsis">📎 ${esc(item.name || "Source")} <small class="settings-hint">· ${Number(item.characters || String(item.content || "").length).toLocaleString()} chars</small></span><button type="button" class="btn btn-danger cn-source-remove" data-index="${index}" style="padding:3px 7px">Remove</button></div>`).join("")}</div>` : `<div class="settings-hint" style="margin-bottom:8px">No source documents attached.</div>`}
        <input id="cn-source-name" class="input" placeholder="Source name (optional)" style="width:100%">
        <textarea id="cn-source-text" class="input" rows="4" placeholder="Paste source text or notes" style="width:100%;margin-top:7px"></textarea>
        <div style="display:flex;gap:7px;flex-wrap:wrap;margin-top:7px">
            <button id="cn-source-add-text" type="button" class="btn btn-outline">Add text</button>
            <button id="cn-source-upload" type="button" class="btn btn-outline">Upload document</button>
            <input id="cn-source-file" type="file" accept=".txt,.md,.csv,.json,.jsonl,.yaml,.yml,.pdf,.docx,.xlsx,.html,.htm,.xml" hidden>
        </div>
        <div id="cn-source-status" class="settings-hint" style="margin-top:6px"></div>
    </div>`;
}

function canvasBlockHelpHtml(type) {
    const block = canvasBlock(type);
    return `<div class="pf-block-help">
        <div class="pf-block-help-title">What this block does</div>
        <div class="pf-block-help-body">${esc(block.description)}</div>
        <div class="pf-block-help-tip">Tip: ${esc(block.use)}</div>
    </div>`;
}

function renderCanvasConfig(id) {
    const node = canvasEditor?.getNodeFromId(id);
    if (!node) return;
    const type = node.name;
    const data = node.data || {};
    let fields = "";
    if (type === "problem") fields = canvasField("cn-title", "Title", data.title) + canvasField("cn-problem", "Problem", data.problem, "textarea") + canvasField("cn-success", "Success criteria", data.success_criteria, "textarea") + canvasField("cn-domain", "Domain", data.domain, "select", `<option value="general" ${data.domain === "general" ? "selected" : ""}>General</option><option value="engineering" ${data.domain === "engineering" ? "selected" : ""}>Engineering</option><option value="medical" ${data.domain === "medical" ? "selected" : ""}>Medical / drug discovery</option>`) + renderProblemSources(data);
    else if (type === "role_model") fields = roleConfigFields(data);
    else if (type === "private_data") fields = canvasField("cn-name", "Dataset name", data.name) + canvasField("cn-content", "Data", data.content, "textarea");
    else if (type === "context") fields = canvasField("cn-content", "Additional information", data.content, "textarea");
    else if (type === "prompt") fields = canvasField("cn-prompt", "Instruction for the group", data.prompt, "textarea");
    else if (type === "web_search") fields = canvasField("cn-query", "Search query ({problem} supported)", data.query) + canvasField("cn-results", "Number of results", data.num_results, "number");
    else if (type === "world_rules") fields = canvasField("cn-environment", "Initial world, constraints and permitted actions", data.environment, "textarea") + canvasField("cn-time-step", "Time represented by one round", data.time_step) + canvasField("cn-seed", "Base seed label", data.seed, "number");
    else if (type === "intervention") fields = canvasField("cn-intervention-round", "Apply at round", data.round, "number") + canvasField("cn-intervention-content", "What-if event", data.content, "textarea");
    else if (type === "replicas") fields = canvasField("cn-replicas", "Independent runs (1–3)", data.count, "number") + `<p class="settings-hint">Each replica gets an isolated board. The chair compares replicas only during synthesis.</p>`;
    else if (type === "discussion_group") fields = canvasField("cn-rounds", "Maximum rounds", data.rounds, "number") + `<p class="settings-hint">Adaptive topology: independent research can start in parallel; dependent work stays centrally sequenced.</p>`;
    else if (type === "shared_board") fields = canvasField("cn-evidence", "Require evidence provenance", data.evidence_required, "checkbox");
    else if (type === "peer_review") fields = `<p class="settings-hint">Participants explicitly critique committed claims from earlier turns.</p>`;
    else if (type === "success_gate") fields = canvasField("cn-min-rounds", "Minimum rounds", data.min_rounds, "number") + canvasField("cn-confidence", "Minimum average confidence", data.min_avg_confidence, "number");
    else if (type === "synthesis") fields = canvasField("cn-model", "Chair model", data.model, "select", modelOptions(data.model)) + canvasField("cn-synthesis-approval", "Require my approval before final synthesis", data.require_approval, "checkbox");
    const panel = document.getElementById("exp-canvas-config");
    const block = canvasBlock(type);
    openCanvasConfigPanel();
    panel.innerHTML = `<h3 style="margin:0 0 12px">${esc(block.icon)} ${esc(block.label)}</h3>
        <div style="display:flex;flex-direction:column;gap:10px">${fields}<button id="cn-save" class="btn btn-primary">Apply</button><button id="cn-delete" class="btn btn-danger">Delete node</button>${canvasBlockHelpHtml(type)}</div>`;
    document.getElementById("cn-save").onclick = () => saveCanvasNode(id, type);
    document.getElementById("cn-delete").onclick = () => { canvasEditor.removeNodeId(`node-${id}`); canvasSelectedNode = null; closeCanvasConfigPanel(); };
    if (type === "problem") bindProblemSources(id);
    if (type === "role_model") bindRolePresetFields();
}

function updateProblemAttachments(id, attachments) {
    const node = canvasEditor.getNodeFromId(id);
    canvasEditor.updateNodeDataFromId(id, {
        ...(node.data || {}),
        title: document.getElementById("cn-title")?.value ?? node.data?.title,
        problem: document.getElementById("cn-problem")?.value ?? node.data?.problem,
        success_criteria: document.getElementById("cn-success")?.value ?? node.data?.success_criteria,
        domain: document.getElementById("cn-domain")?.value ?? node.data?.domain,
        attachments,
    });
    renderCanvasConfig(id);
}

function bindProblemSources(id) {
    document.querySelectorAll(".cn-source-remove").forEach((button) => button.addEventListener("click", () => {
        const node = canvasEditor.getNodeFromId(id);
        const attachments = [...(node.data?.attachments || [])];
        attachments.splice(Number(button.dataset.index), 1);
        updateProblemAttachments(id, attachments);
    }));
    document.getElementById("cn-source-add-text")?.addEventListener("click", () => {
        const content = document.getElementById("cn-source-text")?.value.trim() || "";
        const status = document.getElementById("cn-source-status");
        if (!content) { status.textContent = "Paste source text first."; status.style.color = "var(--red)"; return; }
        const node = canvasEditor.getNodeFromId(id);
        const attachments = [...(node.data?.attachments || [])];
        if (attachments.length >= 8) { status.textContent = "A problem can contain up to 8 sources."; status.style.color = "var(--red)"; return; }
        attachments.push({
            name: document.getElementById("cn-source-name")?.value.trim() || `Problem notes ${attachments.length + 1}`,
            content: content.slice(0, 80000), characters: Math.min(content.length, 80000), source_type: "pasted_text",
        });
        updateProblemAttachments(id, attachments);
    });
    const fileInput = document.getElementById("cn-source-file");
    document.getElementById("cn-source-upload")?.addEventListener("click", () => fileInput?.click());
    fileInput?.addEventListener("change", async () => {
        const file = fileInput.files?.[0];
        if (!file) return;
        const status = document.getElementById("cn-source-status");
        const node = canvasEditor.getNodeFromId(id);
        const attachments = [...(node.data?.attachments || [])];
        if (attachments.length >= 8) { status.textContent = "A problem can contain up to 8 sources."; status.style.color = "var(--red)"; return; }
        status.textContent = "Reading document…";
        try {
            const body = new FormData(); body.append("file", file);
            const extracted = await api("/api/experiments/canvas/source-file", { method: "POST", body });
            attachments.push({ ...extracted, source_type: "uploaded_document" });
            updateProblemAttachments(id, attachments);
        } catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; }
    });
}

function saveCanvasNode(id, type) {
    const node = canvasEditor.getNodeFromId(id);
    const data = { ...(node.data || {}) };
    const value = (field) => document.getElementById(field)?.value ?? "";
    if (type === "problem") Object.assign(data, { title: value("cn-title"), problem: value("cn-problem"), success_criteria: value("cn-success"), domain: value("cn-domain") });
    else if (type === "role_model") Object.assign(data, {
        role_preset: value("cn-role-preset") || "custom",
        label: value("cn-label"),
        role: value("cn-role"),
        role_instruction: value("cn-role-instruction"),
        model: value("cn-model"),
        custom_prompt: value("cn-custom"),
        visible_datasets: value("cn-visible").split(",").map((item) => item.trim()).filter(Boolean),
        web_access: document.getElementById("cn-web")?.checked,
    });
    else if (type === "private_data") Object.assign(data, { name: value("cn-name"), content: value("cn-content") });
    else if (type === "context") data.content = value("cn-content");
    else if (type === "prompt") data.prompt = value("cn-prompt");
    else if (type === "web_search") Object.assign(data, { query: value("cn-query"), num_results: Number(value("cn-results") || 5) });
    else if (type === "world_rules") Object.assign(data, { environment: value("cn-environment"), time_step: value("cn-time-step"), seed: Number(value("cn-seed") || 42) });
    else if (type === "intervention") Object.assign(data, { round: Number(value("cn-intervention-round") || 1), content: value("cn-intervention-content") });
    else if (type === "replicas") data.count = Number(value("cn-replicas") || 1);
    else if (type === "discussion_group") data.rounds = Number(value("cn-rounds") || 2);
    else if (type === "shared_board") data.evidence_required = document.getElementById("cn-evidence")?.checked;
    else if (type === "success_gate") Object.assign(data, { min_rounds: Number(value("cn-min-rounds") || 1), min_avg_confidence: Number(value("cn-confidence") || 0.65) });
    else if (type === "synthesis") Object.assign(data, { model: value("cn-model"), require_approval: Boolean(document.getElementById("cn-synthesis-approval")?.checked) });
    canvasEditor.updateNodeDataFromId(id, data);
    const content = document.querySelector(`#node-${id} .drawflow_content_node`);
    if (content) content.innerHTML = canvasNodeHtml(type, data);
    document.getElementById("exp-canvas-status").textContent = "Node updated.";
}

function exportCanvasSchema() {
    const flow = canvasEditor.export()?.drawflow?.Home?.data || {};
    const nodes = Object.values(flow).map((node) => ({ id: String(node.id), type: node.name, x: node.pos_x, y: node.pos_y, data: node.data || {} }));
    const edges = [];
    for (const node of Object.values(flow)) for (const output of Object.values(node.outputs || {})) for (const connection of output.connections || []) edges.push({ source: String(node.id), target: String(connection.node) });
    return { version: 1, nodes, edges };
}

async function validateCanvas() {
    const status = document.getElementById("exp-canvas-status");
    try {
        const data = await api("/api/experiments/canvas/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ canvas: exportCanvasSchema() }) });
        status.textContent = data.valid ? `Valid · ${data.plan.participants.length} roles · ${data.plan.rounds} rounds · ${data.plan.max_calls} calls` : data.errors.join(" | ");
        status.style.color = data.valid ? "var(--green)" : "var(--red)";
        return data.valid;
    } catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; return false; }
}

async function createCanvasExperiment(existingId = "") {
    if (!await validateCanvas()) return;
    const status = document.getElementById("exp-canvas-status");
    try {
        const data = await api(existingId ? `/api/experiments/${existingId}/canvas` : "/api/experiments/canvas", { method: existingId ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ canvas: exportCanvasSchema() }) });
        selectedId = data.experiment.experiment_id;
        await loadExperiments();
    } catch (error) { status.textContent = Array.isArray(error.message) ? error.message.join(" | ") : error.message; status.style.color = "var(--red)"; }
}

async function openExperiment(id) {
    if (!id) return;
    selectedId = id;
    const list = document.getElementById("exp-list");
    if (list) { list.innerHTML = renderList(); bindList(); }
    const detail = document.getElementById("exp-detail");
    detail.innerHTML = `<div class="settings-hint">Loading experiment…</div>`;
    try {
        const data = await api(`/api/experiments/${encodeURIComponent(id)}`);
        renderDetail(data.experiment, data.active);
        if (["queued", "running", "pausing"].includes(data.experiment.status)) {
            pollTimer = setTimeout(() => openExperiment(id), 1800);
        }
    } catch (error) {
        detail.innerHTML = `<div style="color:var(--red)">${esc(error.message)}</div>`;
    }
}

function renderDetail(exp, active) {
    const detail = document.getElementById("exp-detail");
    const running = active || ["queued", "running", "pausing"].includes(exp.status);
    let controls = "";
    if (exp.status === "waiting_approval") {
        controls = `<button id="exp-approve-synthesis" class="btn btn-primary">Approve synthesis</button><button id="exp-reject-synthesis" class="btn btn-outline">Reject & pause</button><button id="exp-stop" class="btn btn-danger">Stop</button>`;
    } else if (["paused", "pausing"].includes(exp.status)) {
        controls = `<button id="exp-resume" class="btn btn-primary">Resume</button><button id="exp-stop" class="btn btn-danger">Stop</button>`;
    } else if (running) {
        controls = `<button id="exp-pause" class="btn btn-outline">Pause safely</button><button id="exp-stop" class="btn btn-danger">Stop</button>`;
    } else {
        controls = `${exp.status === "draft" ? `<button id="exp-edit" class="btn btn-outline">Edit</button><button id="exp-start" class="btn btn-primary">Start experiment</button>` : ""}<button id="exp-delete" class="btn btn-danger">Delete</button>`;
    }
    if (exp.run_id && exp.trajectory_run_event_id) {
        controls = `<button id="exp-open-trajectory" class="btn btn-outline execution-trajectory-link">Trajectory</button>${controls}`;
    }
    detail.innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
            <div><h3 style="margin:0 0 5px">${esc(exp.title)}</h3><span style="color:${statusColor(exp.status)}">${esc(exp.status)}</span> <span class="settings-hint">${exp.total_replicas > 1 ? `replica ${exp.current_replica || 0}/${exp.total_replicas} · ` : ""}round ${exp.current_round || 0}/${exp.rounds} · ${exp.calls_used}/${exp.max_calls} calls</span></div>
            <div style="display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end">${controls}</div>
        </div>
        ${renderRunEnvelope(exp.run_envelope || exp.run || {})}
        ${renderDurableCheckpoint(exp)}
        ${renderTopology(exp)}
        ${renderScenario(exp)}
        ${(exp.current_participants || []).length ? `<div style="margin:12px 0;padding:10px;border-left:3px solid var(--blue);background:var(--surface-2)"><b>${exp.current_participants.map(esc).join(", ")}</b> are working independently. Their outputs stay isolated until the coordinator commits the round.</div>` : ""}
        ${exp.current_participant ? `<div style="margin:12px 0;padding:10px;border-left:3px solid var(--yellow);background:var(--surface-2)"><b>${esc(exp.current_participant)}</b> is working on the next centrally managed turn.</div>` : ""}
        ${exp.domain === "medical" ? `<div style="margin:12px 0;padding:10px;border-left:3px solid var(--red);background:var(--surface-2)">Research hypothesis mode only — no diagnosis, prescribing, human dosing, or claims of clinical efficacy.</div>` : ""}
        ${exp.error ? `<div style="margin:10px 0;color:var(--red)">${esc(exp.error)}</div>` : ""}
        <details open style="margin-top:14px"><summary style="cursor:pointer;font-weight:600">Problem and success criteria</summary><p style="white-space:pre-wrap">${esc(exp.problem)}</p><p class="settings-hint" style="white-space:pre-wrap">${esc(exp.success_criteria || "No explicit success criteria")}</p></details>
        <div style="margin-top:15px"><b>Research team</b><div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:7px">${exp.participants.map((p) => `<span style="border:1px solid var(--border);border-radius:999px;padding:4px 8px;font-size:12px">${esc(p.model)} · ${esc(p.role)}</span>`).join("")}</div></div>
        ${exp.status === "draft" ? renderDataInput(exp) : renderDatasets(exp)}
        ${renderResult(exp)}
        ${renderFollowUp(exp)}
        ${renderTasks(exp)}
        ${renderBoard(exp)}
        <details style="margin-top:16px"><summary style="cursor:pointer">Run ledger (${exp.events.length})</summary>${exp.events.slice().reverse().map((event) => `<div class="settings-hint" style="margin-top:5px">${esc(event.at)} · ${esc(event.type)} · ${esc(event.message)}</div>`).join("")}</details>`;
    document.getElementById("exp-start")?.addEventListener("click", () => startExperiment(exp));
    document.getElementById("exp-stop")?.addEventListener("click", () => stopExperiment(exp));
    document.getElementById("exp-pause")?.addEventListener("click", () => pauseExperiment(exp));
    document.getElementById("exp-resume")?.addEventListener("click", () => resumeExperiment(exp));
    document.getElementById("exp-approve-synthesis")?.addEventListener("click", () => decideSynthesis(exp, true));
    document.getElementById("exp-reject-synthesis")?.addEventListener("click", () => decideSynthesis(exp, false));
    document.getElementById("exp-edit")?.addEventListener("click", () => exp.mode === "canvas" ? renderCanvasDesigner(exp) : renderQuickEdit(exp));
    document.getElementById("exp-delete")?.addEventListener("click", () => deleteExperiment(exp));
    document.getElementById("exp-followup-start")?.addEventListener("click", () => continueExperiment(exp));
    document.getElementById("exp-intervention-add")?.addEventListener("click", () => addIntervention(exp));
    document.getElementById("exp-open-trajectory")?.addEventListener("click", () => {
        document.dispatchEvent(new CustomEvent("execution-trajectory-open", {
            detail: {
                title: `Experiment · ${exp.title}`,
                url: `/api/trajectory/executions/experiment/${encodeURIComponent(exp.experiment_id)}/${encodeURIComponent(exp.run_id)}`,
                sessionId: exp.trajectory_session_id || "",
                eventId: exp.trajectory_result_event_id || exp.trajectory_run_event_id,
            },
        }));
    });
    bindData(exp);
}

function renderRunEnvelope(run) {
    if (!run?.run_id) return "";
    const usage = run.usage || {};
    const limits = run.limits || {};
    const terminal = ["completed", "completed_with_limits", "failed", "cancelled", "interrupted", "blocked"].includes(run.status);
    const accent = terminal && run.status !== "completed" ? "var(--yellow)" : "var(--blue)";
    return `<div style="margin-top:12px;padding:10px;border:1px solid var(--border);border-left:3px solid ${accent};border-radius:8px;background:var(--surface-2)">
        <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap"><b>${esc((run.current_step || run.phase || run.status || "Run").replaceAll("_", " "))}</b><span class="settings-hint">${esc(run.status || "")} · ${usage.turns || 0}/${limits.max_turns || "∞"} steps · ${usage.total_tokens || 0}/${limits.token_budget || "∞"} tokens</span></div>
        <div class="settings-hint" style="margin-top:5px">Workers ${usage.active_workers || 0}/${limits.max_parallel_workers || 1}${run.stop_reason ? ` · ${esc(run.stop_reason.replaceAll("_", " "))}` : ""}</div>
    </div>`;
}

function renderDurableCheckpoint(exp) {
    const checkpoint = exp.durable_checkpoint || {};
    if (!checkpoint.node && !(exp.checkpoint_history || []).length) return "";
    const history = (exp.checkpoint_history || []).slice(-8);
    const label = checkpoint.node ? checkpoint.node.replaceAll(":", " › ") : "No checkpoint yet";
    return `<details ${["paused", "pausing", "waiting_approval"].includes(exp.status) ? "open" : ""} style="margin-top:12px;padding:10px;border:1px solid var(--border);border-left:3px solid ${statusColor(exp.status)};border-radius:8px;background:var(--surface-2)">
        <summary style="cursor:pointer"><b>Durable checkpoint: ${esc(label)}</b> <span class="settings-hint">${esc(checkpoint.status || "")}</span></summary>
        ${checkpoint.detail ? `<p class="settings-hint" style="margin:8px 0">${esc(checkpoint.detail)}</p>` : ""}
        <div class="settings-hint">Saved after ${checkpoint.calls_used || 0} model call(s) · round ${checkpoint.round || 0} · replica ${checkpoint.replica || 0}</div>
        ${history.length ? `<div style="display:grid;gap:5px;margin-top:9px">${history.slice().reverse().map((item) => `<div class="settings-hint">${esc(item.updated_at || "")} · ${esc(String(item.node || "").replaceAll(":", " › "))}</div>`).join("")}</div>` : ""}
    </details>`;
}

async function pauseExperiment(exp) {
    await api(`/api/experiments/${exp.experiment_id}/pause`, { method: "POST" });
    await openExperiment(exp.experiment_id);
}

async function resumeExperiment(exp) {
    await api(`/api/experiments/${exp.experiment_id}/resume`, { method: "POST" });
    await openExperiment(exp.experiment_id);
}

async function decideSynthesis(exp, approved) {
    await api(`/api/experiments/${exp.experiment_id}/synthesis/${approved ? "approve" : "reject"}`, { method: "POST" });
    await openExperiment(exp.experiment_id);
}

function renderScenario(exp) {
    const scenario = exp.experiment_plan?.scenario || {};
    if (!scenario.enabled) return "";
    const interventions = [...(scenario.interventions || []), ...(exp.scenario_runtime_interventions || [])];
    return `<div style="margin-top:12px;padding:12px;border:1px solid #38bdf8;border-radius:9px;background:var(--surface-2)">
        <div style="display:flex;justify-content:space-between;gap:10px"><b>Scenario world</b><span class="settings-hint">${esc(scenario.time_step || "1 simulated step")} · ${scenario.replicas || 1} independent replica(s)</span></div>
        <p style="white-space:pre-wrap;margin:8px 0">${esc(scenario.environment || "World rules were not specified.")}</p>
        ${interventions.length ? `<div class="settings-hint"><b>Interventions:</b> ${interventions.map((item) => `round ${item.round}: ${esc(item.content)}`).join(" · ")}</div>` : `<div class="settings-hint">No interventions scheduled.</div>`}
        ${["draft", "queued", "running", "pausing", "paused", "waiting_approval"].includes(exp.status) ? `<div style="display:grid;grid-template-columns:1fr 100px auto;gap:7px;margin-top:10px"><input id="exp-intervention-content" class="input" placeholder="Inject a what-if event"><input id="exp-intervention-round" class="input" type="number" min="1" max="${exp.rounds}" value="${Math.min(exp.rounds, Math.max(1, (exp.current_round || 0) + 1))}"><button id="exp-intervention-add" class="btn btn-outline">Schedule</button></div><div id="exp-intervention-status" class="settings-hint"></div>` : ""}
        <div class="settings-hint" style="margin-top:8px">Outputs are scenario hypotheses, not factual predictions.</div>
    </div>`;
}

async function addIntervention(exp) {
    const content = document.getElementById("exp-intervention-content")?.value.trim() || "";
    const status = document.getElementById("exp-intervention-status");
    if (!content) { status.textContent = "Enter a what-if event."; status.style.color = "var(--red)"; return; }
    try {
        await api(`/api/experiments/${exp.experiment_id}/interventions`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content, round: Number(document.getElementById("exp-intervention-round")?.value || 1) }),
        });
        await openExperiment(exp.experiment_id);
    } catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; }
}

function renderTopology(exp) {
    const topology = exp.topology || {};
    if (!topology.mode) return "";
    const labels = {
        single: "Single model",
        centralized_parallel: "Centralized parallel",
        centralized_sequential: "Centralized sequential",
    };
    const reasons = (topology.reasons || []).map((reason) => `<li>${esc(reason)}</li>`).join("");
    return `<details style="margin-top:12px;padding:10px;border:1px solid var(--border);border-radius:8px;background:var(--surface-2)">
        <summary style="cursor:pointer"><b>Execution topology: ${esc(labels[topology.mode] || topology.mode)}</b></summary>
        ${reasons ? `<ul class="settings-hint" style="margin:8px 0 0;padding-left:20px">${reasons}</ul>` : ""}
    </details>`;
}

async function continueExperiment(exp) {
    const question = document.getElementById("exp-followup-question")?.value.trim() || "";
    const status = document.getElementById("exp-followup-status");
    if (!question) { status.textContent = "Enter a follow-up question."; status.style.color = "var(--red)"; return; }
    const button = document.getElementById("exp-followup-start");
    button.disabled = true;
    status.textContent = "Adding context and starting the next round…";
    try {
        const dataText = document.getElementById("exp-followup-data")?.value.trim() || "";
        if (dataText) await api(`/api/experiments/${exp.experiment_id}/data`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: document.getElementById("exp-followup-data-name")?.value.trim() || "Follow-up context", content: dataText }) });
        const file = document.getElementById("exp-followup-file")?.files?.[0];
        if (file) { const body = new FormData(); body.append("file", file); await api(`/api/experiments/${exp.experiment_id}/files`, { method: "POST", body }); }
        await api(`/api/experiments/${exp.experiment_id}/continue`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, rounds: Number(document.getElementById("exp-followup-rounds")?.value || 1) }) });
        await openExperiment(exp.experiment_id);
    } catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; button.disabled = false; }
}

function renderQuickEdit(exp) {
    const detail = document.getElementById("exp-detail");
    const selected = new Set(exp.participants.map((item) => item.model));
    detail.innerHTML = `<h3>Edit draft</h3><div style="display:flex;flex-direction:column;gap:9px;max-width:760px">
        <input id="exp-edit-title" class="input" value="${esc(exp.title)}" placeholder="Title">
        <textarea id="exp-edit-problem" class="input" rows="6" placeholder="Problem">${esc(exp.problem)}</textarea>
        <textarea id="exp-edit-success" class="input" rows="4" placeholder="Success criteria">${esc(exp.success_criteria || "")}</textarea>
        <select id="exp-edit-domain" class="input"><option value="general" ${exp.domain === "general" ? "selected" : ""}>General</option><option value="engineering" ${exp.domain === "engineering" ? "selected" : ""}>Engineering</option><option value="medical" ${exp.domain === "medical" ? "selected" : ""}>Medical / drug discovery</option></select>
        <div><b>Models</b><div style="margin-top:6px;max-height:180px;overflow:auto;border:1px solid var(--border);border-radius:7px;padding:8px">${models.map((model) => `<label style="display:block;margin:5px"><input class="exp-edit-model" type="checkbox" value="${esc(model.name)}" ${selected.has(model.name) ? "checked" : ""}> ${esc(model.name)}</label>`).join("")}</div></div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px"><label class="settings-hint">Rounds<input id="exp-edit-rounds" class="input" type="number" min="1" max="5" value="${exp.rounds}" style="width:100%"></label><label class="settings-hint">Maximum calls<input id="exp-edit-budget" class="input" type="number" min="0" max="100" value="${exp.max_calls}" style="width:100%"></label></div>
        <div style="display:flex;gap:7px"><button id="exp-edit-save" class="btn btn-primary">Save changes</button><button id="exp-edit-cancel" class="btn btn-outline">Cancel</button></div><div id="exp-edit-status" class="settings-hint"></div>
    </div>`;
    document.getElementById("exp-edit-cancel").onclick = () => openExperiment(exp.experiment_id);
    document.getElementById("exp-edit-save").onclick = async () => {
        const status = document.getElementById("exp-edit-status");
        try {
            await api(`/api/experiments/${exp.experiment_id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
                title: document.getElementById("exp-edit-title").value.trim(), problem: document.getElementById("exp-edit-problem").value.trim(),
                success_criteria: document.getElementById("exp-edit-success").value.trim(), domain: document.getElementById("exp-edit-domain").value,
                models: [...document.querySelectorAll(".exp-edit-model:checked")].map((item) => item.value),
                rounds: Number(document.getElementById("exp-edit-rounds").value || 2), max_calls: Number(document.getElementById("exp-edit-budget").value || 0),
            }) });
            selectedId = exp.experiment_id;
            await loadExperiments();
        } catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; }
    };
}

async function deleteExperiment(exp) {
    const confirmed = await showConfirm("Delete experiment", `Delete “${exp.title}”? It will be moved to the local recoverable trash.`);
    if (!confirmed) return;
    try {
        await api(`/api/experiments/${exp.experiment_id}`, { method: "DELETE" });
        selectedId = "";
        await loadExperiments();
    } catch (error) {
        const detail = document.getElementById("exp-detail");
        detail.insertAdjacentHTML("afterbegin", `<div style="color:var(--red);margin-bottom:8px">${esc(error.message)}</div>`);
    }
}

function renderDataInput(exp) {
    return `<div style="margin-top:18px;padding:12px;background:var(--surface-2);border-radius:8px">
        <b>Private experiment data</b><p class="settings-hint">Stored only under this experiment and not added to long-term memory. Selected model providers receive this content when their turn runs.</p>
        ${renderDatasets(exp)}
        <input id="exp-data-name" class="input" placeholder="Dataset name" style="width:100%;margin-top:8px">
        <textarea id="exp-data-content" class="input" rows="5" placeholder="Paste observations, measurements, constraints, or source excerpts" style="width:100%;margin-top:7px"></textarea>
        <div style="display:flex;gap:7px;margin-top:7px"><button id="exp-add-data" class="btn btn-outline">Add text data</button><button id="exp-upload-button" class="btn btn-outline">Upload text file</button><input id="exp-file" type="file" accept=".txt,.md,.csv,.json,.yaml,.yml,.log" hidden></div>
        <div id="exp-data-status" class="settings-hint" style="margin-top:6px"></div>
    </div>`;
}

function renderDatasets(exp) {
    if (!exp.datasets.length) return `<div class="settings-hint" style="margin-top:8px">No datasets attached.</div>`;
    return `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px">${exp.datasets.map((item) => `<span style="border:1px solid var(--border);border-radius:6px;padding:4px 7px;font-size:12px">${esc(item.name)} · ${item.characters} chars</span>`).join("")}</div>`;
}

function bindData(exp) {
    const status = document.getElementById("exp-data-status");
    document.getElementById("exp-add-data")?.addEventListener("click", async () => {
        try {
            await api(`/api/experiments/${exp.experiment_id}/data`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: document.getElementById("exp-data-name").value || "Notes", content: document.getElementById("exp-data-content").value }) });
            await openExperiment(exp.experiment_id);
        } catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; }
    });
    const file = document.getElementById("exp-file");
    document.getElementById("exp-upload-button")?.addEventListener("click", () => file?.click());
    file?.addEventListener("change", async () => {
        if (!file.files[0]) return;
        const body = new FormData(); body.append("file", file.files[0]);
        try { await api(`/api/experiments/${exp.experiment_id}/files`, { method: "POST", body }); await openExperiment(exp.experiment_id); }
        catch (error) { status.textContent = error.message; status.style.color = "var(--red)"; }
    });
}

async function startExperiment(exp) {
    const confirmed = await showConfirm("Start experiment", `Run ${exp.participants.length} models for ${exp.rounds} round(s), with a limit of ${exp.max_calls} model calls?`);
    if (!confirmed) return;
    try { await api(`/api/experiments/${exp.experiment_id}/start`, { method: "POST" }); await openExperiment(exp.experiment_id); }
    catch (error) { alert(error.message); }
}

async function stopExperiment(exp) {
    const confirmed = await showConfirm("Stop experiment", "Stop after the current model call returns? All committed contributions will be preserved.");
    if (!confirmed) return;
    await api(`/api/experiments/${exp.experiment_id}/stop`, { method: "POST" });
    await openExperiment(exp.experiment_id);
}

function renderResult(exp) {
    const result = exp.result || {};
    if (!Object.keys(result).length) return "";
    return `<div style="margin-top:18px;border:1px solid var(--green);border-radius:9px;padding:13px"><h4 style="margin:0 0 8px">${exp.experiment_plan?.scenario?.enabled ? "Scenario Report" : "Synthesis"}</h4><p style="white-space:pre-wrap">${esc(result.conclusion || result.summary || "")}</p>
        ${result.consensus?.length ? `<b>Consensus</b><ul>${result.consensus.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}
        ${result.disagreements?.length ? `<b>Unresolved disagreements</b><ul>${result.disagreements.map((item) => `<li>${esc(item.issue || item)}</li>`).join("")}</ul>` : ""}
        ${result.recommended_experiments?.length ? `<b>Recommended next experiments</b><ul>${result.recommended_experiments.map((item) => `<li><b>${esc(item.title || "Test")}</b>: ${esc(item.method || "")}</li>`).join("")}</ul>` : ""}
        <div class="settings-hint">Confidence: ${Number(result.confidence || 0).toFixed(2)}${exp.experiment_plan?.scenario?.enabled ? " · Scenario hypothesis, not a prediction" : ""}</div></div>`;
}

function renderFollowUp(exp) {
    if (exp.status !== "completed") return "";
    return `<div style="margin-top:18px;border:1px solid var(--border);border-radius:9px;padding:13px;background:var(--surface-2)">
        <h4 style="margin:0 0 5px">Continue this experiment</h4>
        <p class="settings-hint" style="margin:0 0 10px">Ask the same research team a follow-up question. Previous turns, evidence, datasets, and synthesis stay in context.</p>
        <textarea id="exp-followup-question" class="input" rows="4" style="width:100%" placeholder="What should the team investigate next?"></textarea>
        <details style="margin-top:9px"><summary style="cursor:pointer">Add new data before continuing (optional)</summary>
            <div style="display:flex;flex-direction:column;gap:8px;margin-top:8px"><input id="exp-followup-data-name" class="input" placeholder="Data name"><textarea id="exp-followup-data" class="input" rows="5" placeholder="Paste additional evidence or context"></textarea><input id="exp-followup-file" type="file" accept=".txt,.md,.csv,.json,.yaml,.yml,.log"></div>
        </details>
        <div style="display:flex;gap:8px;align-items:center;margin-top:10px"><label class="settings-hint">Additional rounds <select id="exp-followup-rounds" class="input"><option value="1">1</option><option value="2">2</option><option value="3">3</option></select></label><button id="exp-followup-start" class="btn btn-primary">Continue research</button><span id="exp-followup-status" class="settings-hint"></span></div>
    </div>`;
}

function renderTasks(exp) {
    const tasks = exp.tasks || [];
    if (!tasks.length) return "";
    const terminal = ["completed", "failed", "cancelled"].includes(exp.status);
    return `<details style="margin-top:18px"><summary style="cursor:pointer;font-weight:600">Collaborative task board (${tasks.length})</summary><p class="settings-hint">Pending items are model-proposed next steps, not background work.</p><div style="display:grid;gap:6px">${tasks.map((task) => { const label = terminal && task.status === "pending" ? "proposed next task · not executed" : task.status; return `<div style="display:flex;justify-content:space-between;gap:10px;border:1px solid var(--border);border-radius:7px;padding:7px 9px"><span>${esc(task.title)}</span><span class="settings-hint">${esc(label)}${task.assigned_to ? ` · ${esc(task.assigned_to)}` : ""}</span></div>`; }).join("")}</div></details>`;
}

function renderBoard(exp) {
    return `<div style="margin-top:18px"><h4>Shared evidence board (${exp.contributions.length} committed turns)</h4>
        ${exp.contributions.length ? exp.contributions.map((turn) => `<details style="border:1px solid var(--border);border-radius:8px;padding:9px;margin-bottom:7px"><summary style="cursor:pointer">${exp.total_replicas > 1 ? `<b>Replica ${turn.replica || 1}</b> · ` : ""}<b>Round ${turn.round}</b> · ${esc(turn.model)} · ${esc(turn.role)} · confidence ${Number(turn.confidence || 0).toFixed(2)}</summary><p style="white-space:pre-wrap">${esc(turn.summary)}</p>${turn.hypotheses?.length ? `<b>Hypotheses</b><ul>${turn.hypotheses.map((item) => `<li>${esc(item.claim || item)}</li>`).join("")}</ul>` : ""}${turn.evidence?.length ? `<b>Evidence claims</b><ul>${turn.evidence.map((item) => `<li>${esc(item.detail || item.claim || item)} <span class="settings-hint">· ${esc(item.source || "model inference")} · ${esc(item.provenance_type || "unverified")}</span></li>`).join("")}</ul>` : ""}${turn.critiques?.length ? `<b>Critiques</b><ul>${turn.critiques.map((item) => `<li>${esc(item.issue || item)}</li>`).join("")}</ul>` : ""}</details>`).join("") : `<p class="settings-hint">The board is empty. Contributions appear only after a participant finishes its turn.</p>`}
    </div>`;
}

document.addEventListener("self-modification-open", async (event) => {
    experimentSection = "self-modification";
    selfModSelectedId = String(event.detail?.proposalId || selfModSelectedId || "");
    selfModCreating = false;
    await loadSelfModificationLab();
});
