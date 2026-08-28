/** Agent-owned autonomous laboratory observer. */

import { showConfirm } from "./ui.js?v=1.21";

const root = document.getElementById("agent-lab-content");
let runs = [];
let selectedId = "";
let selectedRun = null;
let coordinatorModels = [];
let defaultCoordinatorModel = "";
let pollTimer = null;
let launchNotice = "";
let pollDelayMs = 4000;

const POLL_BASE_MS = 4000;
const POLL_MAX_MS = 30000;

const esc = (value) => {
    const node = document.createElement("div");
    node.textContent = String(value ?? "");
    return node.innerHTML;
};

async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const error = new Error(data.detail || "Agent Lab request failed");
        error.status = response.status;
        error.retryAfterMs = Math.max(0, Number(response.headers.get("Retry-After") || 0) * 1000);
        throw error;
    }
    return data;
}

function formatTime(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function statusTone(status) {
    if (status === "completed") return "success";
    if (["failed", "cancelled"].includes(status)) return "danger";
    if (["running", "prepared"].includes(status)) return "active";
    if (status === "paused") return "warning";
    return "neutral";
}

function isAwaitingUser(run) {
    return Boolean(
        run?.status === "paused"
        && run?.interactive?.awaiting_input
        && String(run?.interactive?.question || "").trim(),
    );
}

function statusLabel(run, active = false) {
    if (active || run.status === "running") return "Remy is working";
    if (run.status === "completed") return "Ready";
    if (run.status === "paused") return isAwaitingUser(run) ? "Needs your input" : "Paused safely";
    if (run.status === "failed") return "Could not finish";
    if (run.status === "cancelled") return "Stopped";
    if (run.status === "prepared") return "Ready to start";
    return "Not started";
}

function phaseLabel(run, active = false) {
    if (run.status === "completed") return "Result checked and ready";
    if (run.status === "paused") return isAwaitingUser(run) ? "Waiting for your clarification" : "A technical retry is available";
    if (run.status === "failed") return "Stopped safely";
    if (!active && run.status === "prepared") return "Everything is prepared";
    const phase = String(run.phase || "");
    if (phase.includes("verif")) return "Checking the result";
    if (phase.includes("repair")) return "Improving the result";
    if (phase.includes("execution") || phase.includes("build")) return "Creating the result";
    if (phase.includes("planning") || phase.includes("evidence")) return "Understanding and planning";
    if (run.status === "draft") return "Saved as a draft";
    return active ? "Working" : "Waiting";
}

function progressStep(run, active = false) {
    if (run.status === "completed") return 4;
    if (run.status === "paused" || run.status === "failed") return 3;
    const phase = String(run.phase || "");
    if (phase.includes("verif") || phase.includes("repair")) return 3;
    if (phase.includes("execution") || phase.includes("build")) return 2;
    if (active || run.status === "running" || run.status === "prepared") return 1;
    return 0;
}

export async function loadAgentLab() {
    if (!root) return;
    clearTimeout(pollTimer);
    root.innerHTML = '<div class="skeleton-card" style="height:180px"></div>';
    try {
        const [data, modelData] = await Promise.all([
            api("/api/agent-lab/runs"),
            api("/api/agent-lab/models").catch(() => ({ models: [], default: "" })),
        ]);
        runs = data.runs || [];
        coordinatorModels = modelData.models || [];
        defaultCoordinatorModel = modelData.default || "";
        renderShell();
        if (selectedId || runs.length) await openRun(selectedId || runs[0].run_id);
    } catch (error) {
        if (selectedRun && document.getElementById("agent-lab-detail")) {
            showRefreshNotice(error.status === 429
                ? "Updates are temporarily slowed. Remy has not stopped."
                : `Could not refresh right now: ${error.message}`);
            scheduleRunPoll(error.retryAfterMs || POLL_MAX_MS);
        } else {
            root.innerHTML = `<div class="agent-lab-error">${esc(error.message)}</div>`;
        }
    }
}

function renderShell() {
    root.innerHTML = `
        <div class="agent-lab-chat-shell">
            <aside class="agent-lab-sidebar">
                <button id="agent-lab-new" class="btn btn-primary">+ New task</button>
                <h4>Laboratory history</h4>
                <div id="agent-lab-list">${renderList()}</div>
            </aside>
            <section class="agent-lab-chat-main">
                <div id="agent-lab-detail" class="agent-lab-detail">
                    <div class="agent-lab-welcome"><span>◈</span><h3>Closed Agent Laboratory</h3><p>Describe the outcome you need. Remy will work inside an isolated environment, ask when a decision is required, verify the result and return the files here.</p></div>
                </div>
                <form id="agent-lab-form" class="agent-lab-chat-composer">
                    <textarea id="agent-lab-goal" rows="3" maxlength="20000" required placeholder="Describe what the laboratory should create, analyse or test…"></textarea>
                    <div><span id="agent-lab-launch-status" class="settings-hint">${esc(launchNotice)}</span><button class="btn btn-primary" type="submit">Start</button></div>
                </form>
            </section>
        </div>`;
    document.getElementById("agent-lab-new")?.addEventListener("click", beginNewTask);
    document.getElementById("agent-lab-form")?.addEventListener("submit", handleComposerSubmit);
    bindList();
}

function beginNewTask() {
    selectedId = "";
    selectedRun = null;
    launchNotice = "";
    renderShell();
    document.getElementById("agent-lab-goal")?.focus();
}

function updateComposer(run) {
    const input = document.getElementById("agent-lab-goal");
    const button = document.querySelector("#agent-lab-form button[type=submit]");
    if (!input || !button) return;
    const waiting = isAwaitingUser(run);
    const working = run?.status === "running";
    input.disabled = working || Boolean(run && !waiting);
    button.disabled = working || Boolean(run && !waiting);
    if (waiting) {
        input.placeholder = "Answer Remy's question so the laboratory can continue…";
        button.textContent = "Continue";
    } else if (working) {
        input.placeholder = "The laboratory is working. You can safely leave this page.";
        button.textContent = "Working…";
    } else if (run) {
        input.placeholder = "Choose New task to start another laboratory run.";
        button.textContent = "Start";
    } else {
        input.placeholder = "Describe what the laboratory should create, analyse or test…";
        button.textContent = "Start";
    }
}

async function handleComposerSubmit(event) {
    if (isAwaitingUser(selectedRun)) {
        return submitClarification(event, selectedRun);
    }
    return createRun(event);
}

function renderList() {
    if (!runs.length) return '<div class="empty-state">No autonomous runs yet.</div>';
    return runs.map((run) => `
        <button class="agent-lab-run ${run.run_id === selectedId ? "active" : ""}" data-run-id="${esc(run.run_id)}">
            <span class="agent-lab-run-head"><b>${esc(run.title)}</b><i class="agent-lab-status ${statusTone(run.status)}">${esc(statusLabel(run))}</i></span>
            <span>${esc(run.goal)}</span>
            <small>${run.artifact_count || 0} result file(s) · ${esc(formatTime(run.updated_at))}</small>
        </button>`).join("");
}

function bindList() {
    document.querySelectorAll("[data-run-id]").forEach((button) => {
        button.addEventListener("click", () => openRun(button.dataset.runId));
    });
}

async function createRun(event) {
    event.preventDefault();
    const button = event.currentTarget.querySelector("button[type=submit]");
    const status = document.getElementById("agent-lab-launch-status");
    const goal = document.getElementById("agent-lab-goal").value.trim();
    if (!goal) return;
    button.disabled = true;
    button.textContent = "Starting…";
    if (status) status.textContent = "Creating a safe workspace…";
    try {
        const data = await api("/api/agent-lab/runs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                goal,
                title: "",
                interactive: true,
                policy: {
                    max_agents: 4,
                    max_models: 4,
                    time_budget_seconds: 900,
                },
            }),
        });
        selectedId = data.run.run_id;
        if (status) status.textContent = "Remy is preparing the plan…";
        await api(`/api/agent-lab/runs/${encodeURIComponent(selectedId)}/prepare`, { method: "POST" });
        if (status) status.textContent = "Starting the team…";
        await api(`/api/agent-lab/runs/${encodeURIComponent(selectedId)}/autonomous`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: "", verifier_model: "", max_repair_rounds: 2, isolation_mode: "automatic" }),
        });
        launchNotice = "Task started. You can leave this page and return later.";
        await loadAgentLab();
    } catch (error) {
        launchNotice = error.message;
        button.disabled = false;
        button.textContent = "Start";
        if (status) { status.textContent = error.message; status.classList.add("agent-lab-error"); }
        if (selectedId) await openRun(selectedId);
    }
}

function syncRunSummary(run) {
    const index = runs.findIndex((item) => item.run_id === run.run_id);
    const summary = {
        ...(index >= 0 ? runs[index] : {}),
        ...run,
        artifact_count: (run.artifacts || []).length,
    };
    if (index >= 0) runs[index] = summary;
    else runs.unshift(summary);
    const list = document.getElementById("agent-lab-list");
    if (list) { list.innerHTML = renderList(); bindList(); }
}

function showRefreshNotice(message) {
    const detail = document.getElementById("agent-lab-detail");
    if (!detail || !selectedRun) return;
    detail.querySelector(".agent-lab-refresh-notice")?.remove();
    detail.insertAdjacentHTML(
        "afterbegin",
        `<div class="agent-lab-refresh-notice">${esc(message)}</div>`,
    );
}

function scheduleRunPoll(delay = pollDelayMs) {
    clearTimeout(pollTimer);
    if (!selectedId || selectedRun?.status !== "running") return;
    pollTimer = setTimeout(refreshSelectedRun, Math.max(POLL_BASE_MS, delay));
}

async function refreshSelectedRun() {
    const runId = selectedId;
    if (!runId) return;
    try {
        const data = await api(`/api/agent-lab/runs/${encodeURIComponent(runId)}`);
        if (selectedId !== runId) return;
        selectedRun = data.run;
        pollDelayMs = POLL_BASE_MS;
        syncRunSummary(data.run);
        renderDetail(data.run, data.active);
        updateComposer(data.run);
        if (data.active || data.run.status === "running") scheduleRunPoll();
    } catch (error) {
        if (selectedId !== runId) return;
        if (error.status === 429) {
            pollDelayMs = Math.min(
                POLL_MAX_MS,
                Math.max(error.retryAfterMs || 0, Math.round(pollDelayMs * 1.8)),
            );
            showRefreshNotice("Updates are temporarily slowed. Remy is still working and this page will recover automatically.");
            scheduleRunPoll(pollDelayMs);
            return;
        }
        showRefreshNotice(`Could not refresh right now: ${error.message}`);
        pollDelayMs = Math.min(POLL_MAX_MS, Math.round(pollDelayMs * 1.8));
        scheduleRunPoll(pollDelayMs);
    }
}

async function openRun(runId) {
    if (!runId) return;
    clearTimeout(pollTimer);
    selectedId = runId;
    const list = document.getElementById("agent-lab-list");
    if (list) { list.innerHTML = renderList(); bindList(); }
    const detail = document.getElementById("agent-lab-detail");
    if (detail && selectedRun?.run_id !== runId) detail.innerHTML = '<div class="empty-state">Loading Agent Lab run…</div>';
    try {
        const data = await api(`/api/agent-lab/runs/${encodeURIComponent(runId)}`);
        if (selectedId !== runId) return;
        selectedRun = data.run;
        pollDelayMs = POLL_BASE_MS;
        syncRunSummary(data.run);
        renderDetail(data.run, data.active);
        updateComposer(data.run);
        if (data.active || data.run.status === "running") {
            scheduleRunPoll();
        }
    } catch (error) {
        if (selectedRun?.run_id === runId) {
            showRefreshNotice(error.status === 429
                ? "Updates are temporarily slowed. Remy has not stopped. This page will retry automatically."
                : `Could not refresh right now: ${error.message}`);
            if (selectedRun.status === "running") {
                pollDelayMs = Math.min(POLL_MAX_MS, error.retryAfterMs || POLL_BASE_MS * 2);
                scheduleRunPoll(pollDelayMs);
            }
        } else if (detail) {
            detail.innerHTML = `<div class="agent-lab-error">${esc(error.message)}</div>`;
        }
    }
}

function controls(run) {
    const trajectory = run.trajectory_run_event_id
        ? '<button id="agent-lab-trajectory" class="btn btn-outline">Details</button>' : "";
    if (["draft", "prepared"].includes(run.status)) return `<button data-lab-quick-start class="btn btn-primary">Start</button>${trajectory}<button data-lab-archive class="btn btn-outline">Remove from history</button>`;
    if (run.status === "running") return `${trajectory}<button data-lab-action="pause" class="btn btn-outline">Pause</button>`;
    if (run.status === "paused" && isAwaitingUser(run)) return `${trajectory}<button data-lab-action="cancel" class="btn btn-outline">Stop</button><button data-lab-archive class="btn btn-outline">Remove from history</button>`;
    if (run.status === "paused") return `<button data-lab-quick-start class="btn btn-primary">Retry automatically</button>${trajectory}<button data-lab-archive class="btn btn-outline">Remove from history</button>`;
    return `${trajectory}<button data-lab-archive class="btn btn-outline">Remove from history</button>`;
}

function renderProgress(run, active) {
    const current = progressStep(run, active);
    const labels = ["Understands", "Plans", "Creates", "Checks", "Ready"];
    return `<div class="agent-lab-progress">${labels.map((label, index) => `<div class="${index < current ? "done" : index === current ? "current" : ""}"><i>${index < current ? "✓" : index + 1}</i><span>${label}</span></div>`).join("")}</div>`;
}

function latestBlocker(run) {
    const blockers = run.task_ledger?.blockers || [];
    return blockers.length ? blockers[blockers.length - 1] : null;
}

function renderUserMessages(run) {
    const messages = (run.messages || []).filter((item) => item.role === "user");
    const values = messages.length ? messages : [{ content: run.goal }];
    return values.map((item) => `<article class="agent-lab-message user"><div><p>${esc(item.content)}</p></div><div class="agent-lab-message-avatar">You</div></article>`).join("");
}

function renderInputRequest(run) {
    if (!isAwaitingUser(run)) return "";
    return `<article class="agent-lab-message assistant question"><div class="agent-lab-message-avatar">R</div><div><b>Remy needs your help</b><p>${esc(run.interactive.question)}</p><span>Answer in the field below and the laboratory will continue automatically.</span></div></article>`;
}

function renderTechnicalPause(run) {
    if (run.status !== "paused" || isAwaitingUser(run)) return "";
    const blocker = latestBlocker(run);
    const reason = run.error || blocker?.message || "The generated workspace did not pass an internal check.";
    return `<article class="agent-lab-message assistant question"><div class="agent-lab-message-avatar">R</div><div><b>Remy stopped safely</b><p>This is an internal laboratory issue—not a problem with your request.</p><span>Choose Retry automatically. Technical detail: ${esc(reason)}</span></div></article>`;
}

function renderResult(run) {
    if (run.status !== "completed") return "";
    const artifacts = run.artifacts || [];
    const verification = (run.verification || []).slice(-1)[0] || {};
    return `<article class="agent-lab-message assistant result"><div class="agent-lab-message-avatar">R</div><div class="agent-lab-result"><header><div><b>Result is ready</b><span>${verification.world_fact === "supports" ? "Independently checked" : "Completed"}</span></div></header>${artifacts.length ? artifacts.map((item) => `<a class="agent-lab-artifact" href="/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/artifacts/${encodeURIComponent(item.artifact_id)}" download><b>${esc(item.name || item.path)}</b><span>Download · ${Number(item.size || 0).toLocaleString()} bytes</span></a>`).join("") : '<p>The work is complete. Open Details to inspect the execution record.</p>'}</div></article>`;
}

function renderDetail(run, active = false) {
    const detail = document.getElementById("agent-lab-detail");
    if (!detail) return;
    detail.innerHTML = `
        <div class="agent-lab-conversation">
        ${renderUserMessages(run)}
        <article class="agent-lab-message assistant status"><div class="agent-lab-message-avatar">R</div><div>
            <header class="agent-lab-detail-head"><div><h3>${esc(statusLabel(run, active))}</h3><span class="settings-hint">${esc(phaseLabel(run, active))}</span></div><div class="agent-lab-actions">${controls(run)}</div></header>
            ${renderProgress(run, active)}
        </div></article>
        ${renderInputRequest(run)}
        ${renderTechnicalPause(run)}
        ${renderResult(run)}
        ${renderAutonomousState(run, active)}
        <details class="agent-lab-technical"><summary>Technical details</summary>
            <section class="agent-lab-boundaries"><b>Safety boundaries</b><div class="agent-lab-policy-row"><span>${run.policy.max_agents} agents max</span><span>${run.policy.max_models} models max</span><span>${run.policy.time_budget_seconds}s</span><span>Project workspace only</span></div></section>
            <div class="agent-lab-observer-grid">
                <section><h4>Central assignment & specialist fan-in</h4>${renderTeam(run.team, run.delegation)}</section>
                <section><h4>Builder fan-out & file claims</h4>${renderBuilderFanout(run.builder_fanout, run.file_claims)}</section>
                <section><h4>Agent-owned plan · v2 DAG</h4>${renderPlan(run.workflow_plan, run.plan)}</section>
                <section><h4>Workspace & artifacts</h4>${renderWorkspace(run)}${renderWorkspaceBranches(run)}</section>
                <section><h4>Snapshot retention</h4>${renderSnapshotRetention(run)}</section>
                <section><h4>Verification</h4>${renderVerification(run.verification)}</section>
            </div>
            ${renderExecutions(run.executions)}
            ${renderCognitiveLedgers(run.task_ledger, run.progress_ledger)}
            <details class="agent-lab-ledger"><summary>Run ledger (${run.events.length})</summary>${run.events.slice().reverse().map((item) => `<div><time>${esc(formatTime(item.at))}</time><b>${esc(item.type)}</b><span>${esc(item.message)}</span></div>`).join("")}</details>
        </details>
        </div>`;
    document.querySelectorAll("[data-lab-action]").forEach((button) => {
        button.addEventListener("click", () => runAction(run, button.dataset.labAction));
    });
    document.querySelector("[data-lab-quick-start]")?.addEventListener("click", (event) => quickStartExisting(event, run));
    document.querySelector("[data-lab-archive]")?.addEventListener("click", () => archiveRun(run));
    document.querySelector("[data-lab-retention]")?.addEventListener("click", () => loadSnapshotRetention(run));
    document.getElementById("agent-lab-trajectory")?.addEventListener("click", () => {
        document.dispatchEvent(new CustomEvent("execution-trajectory-open", { detail: {
            title: `Agent Lab · ${run.title}`,
            url: `/api/trajectory/executions/agent_lab/${encodeURIComponent(run.run_id)}/${encodeURIComponent(run.run_id)}`,
            sessionId: run.trajectory_session_id || "",
            eventId: run.trajectory_result_event_id || run.trajectory_run_event_id,
        }}));
    });
}

async function archiveRun(run) {
    const approved = await showConfirm(
        "Remove this task from history?",
        "It will disappear from the laboratory list. Its files and audit evidence remain preserved locally.",
    );
    if (!approved) return;
    try {
        await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/archive`, { method: "POST" });
        selectedId = "";
        selectedRun = null;
        launchNotice = "Task removed from history. Its evidence remains in the local archive.";
        await loadAgentLab();
    } catch (error) {
        document.getElementById("agent-lab-detail")?.insertAdjacentHTML("afterbegin", `<div class="agent-lab-error">${esc(error.message)}</div>`);
    }
}

async function quickStartExisting(event, run) {
    const button = event.currentTarget;
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Starting…";
    try {
        if (run.status === "draft") {
            await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/prepare`, { method: "POST" });
        }
        await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/autonomous`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: "", verifier_model: "", max_repair_rounds: 2, isolation_mode: "automatic" }),
        });
        await loadAgentLab();
    } catch (error) {
        button.disabled = false;
        button.textContent = originalLabel;
        document.getElementById("agent-lab-detail")?.insertAdjacentHTML("afterbegin", `<div class="agent-lab-error">${esc(error.message)}</div>`);
    }
}

async function submitClarification(event, run) {
    event.preventDefault();
    const button = event.currentTarget.querySelector("button[type=submit]");
    const status = document.getElementById("agent-lab-launch-status");
    const message = document.getElementById("agent-lab-goal").value.trim();
    if (!message) return;
    button.disabled = true;
    if (status) status.textContent = "Continuing…";
    try {
        await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/clarify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        await loadAgentLab();
    } catch (error) {
        button.disabled = false;
        if (status) { status.textContent = error.message; status.classList.add("agent-lab-error"); }
    }
}

function renderAutonomousState(run, active) {
    const state = run.autonomous || {};
    if (!state.enabled) return "";
    const model = state.served_by || state.model || "automatic routing";
    return `<section class="agent-lab-autonomous-state ${active ? "active" : ""}">
        <div><b>${active ? "Coordinator is working" : "Autonomous coordinator"}</b><span>${esc((run.phase || "").replaceAll("_", " "))}</span></div>
        <div class="agent-lab-policy-row"><span>${esc(model)}</span><span>${esc(state.isolation_mode || "bounded_process")}${state.resolved_isolation_mode ? ` → ${esc(state.resolved_isolation_mode)}` : ""}</span><span>${Number(state.model_calls || 0)} model call(s)</span><span>${Number((run.delegation?.usage || {}).members || 0)} specialist(s)</span><span>repair ${Number(state.repair_round || 0)}/${Number(state.max_repair_rounds || 0)}</span></div>
        ${run.verifier?.prepared_at ? `<div class="agent-lab-policy-row"><span>Verifier: ${esc(run.verifier.served_by || run.verifier.assigned_model || "automatic")}</span><span>${esc(run.verifier.independence_level || "isolated context")}</span><span>read-only runtime</span></div>` : ""}
        ${state.last_rationale ? `<p>${esc(state.last_rationale)}</p>` : ""}
    </section>`;
}

function renderAutonomousConfig(run) {
    const host = document.getElementById("agent-lab-autonomous-config");
    if (!host) return;
    const options = [
        `<option value="">Automatic routing${defaultCoordinatorModel ? ` · ${esc(defaultCoordinatorModel)}` : ""}</option>`,
        ...coordinatorModels.map((item) => `<option value="${esc(item.name)}">${esc(item.name)}${item.provider ? ` · ${esc(item.provider)}` : ""}</option>`),
    ].join("");
    host.innerHTML = `<form id="agent-lab-autonomous-form" class="agent-lab-autonomous-config">
        <header><div><b>Autonomous team contract</b><span>The central model decomposes the goal, assigns local specialists, then owns build synthesis and up to two evidence-driven repairs.</span></div><button id="agent-lab-autonomous-close" type="button" class="btn btn-outline">Close</button></header>
        <label>Coordinator model<select id="agent-lab-autonomous-model">${options}</select></label>
        <label>Verifier model<select id="agent-lab-verifier-model"><option value="">Automatic · prefer a different connected model</option>${coordinatorModels.map((item) => `<option value="${esc(item.name)}">${esc(item.name)}${item.provider ? ` · ${esc(item.provider)}` : ""}</option>`).join("")}</select></label>
        <label>Execution isolation<select id="agent-lab-isolation-mode"><option value="bounded_process">Bounded process · compatible</option></select><span id="agent-lab-isolation-status" class="settings-hint">Checking execution environments…</span></label>
        <div id="agent-lab-container-assistant" class="agent-lab-container-assistant" hidden></div>
        <label>Repair budget<select id="agent-lab-autonomous-repairs"><option value="0">0 · stop after first failure</option><option value="1">1 repair</option><option value="2" selected>2 repairs</option></select></label>
        <p class="settings-hint">The model can propose files only. Every proposal is revalidated by the executor; the model cannot change permissions, install packages, run shell commands, or mark its own work verified.</p>
        <button type="submit" class="btn btn-primary">Start bounded autonomous run</button><span id="agent-lab-autonomous-status" class="settings-hint"></span>
    </form>`;
    document.getElementById("agent-lab-autonomous-close")?.addEventListener("click", () => { host.innerHTML = ""; });
    document.getElementById("agent-lab-autonomous-form")?.addEventListener("submit", (event) => startAutonomous(event, run));
    loadIsolationStatus();
}

async function loadIsolationStatus() {
    const status = document.getElementById("agent-lab-isolation-status");
    const select = document.getElementById("agent-lab-isolation-mode");
    const assistant = document.getElementById("agent-lab-container-assistant");
    if (!status || !select || !assistant) return;
    const renderAssistant = (container) => {
        if (container.available) {
            assistant.hidden = true;
            assistant.innerHTML = "";
            return;
        }
        const reasonCode = container.reason_code || "unknown";
        const engine = container.engine || "Docker";
        const actions = [`<button type="button" class="btn btn-outline" data-agent-lab-container-recheck>Check again</button>`];
        let title = "Secure container mode is unavailable";
        let message = container.reason || "Docker/Podman or the local Remy runtime is unavailable.";
        if (reasonCode === "cli_missing") {
            title = "Docker is not installed";
            message = "Install Docker Desktop, start it once, then return here and check again. You can keep using Bounded process meanwhile.";
            actions.unshift(`<a class="btn btn-primary" href="https://docs.docker.com/get-started/get-docker/" target="_blank" rel="noopener noreferrer">Install Docker Desktop</a>`);
        } else if (reasonCode === "runtime_unavailable" || reasonCode === "context_unavailable") {
            title = `${engine} is installed but not running`;
            message = "Start Docker Desktop or Podman, wait until its engine is ready, then check again.";
        } else if (reasonCode === "image_missing") {
            title = "Remy secure runtime is not prepared";
            message = "Docker is ready. Prepare Remy’s pinned local runtime now. Docker may download the pinned Python base; no account or registry sign-in is required.";
            actions.unshift(`<button type="button" class="btn btn-primary" data-agent-lab-container-prepare>Prepare secure runtime</button>`);
        } else if (reasonCode === "remote_context") {
            title = "A local Docker context is required";
            message = "Agent Lab rejects remote Docker engines because they can expose workspace files outside this computer.";
        }
        assistant.hidden = false;
        assistant.innerHTML = `<div><b>${esc(title)}</b><span>${esc(message)}</span></div><div class="agent-lab-container-actions">${actions.join("")}</div>`;
        assistant.querySelector("[data-agent-lab-container-recheck]")?.addEventListener("click", loadIsolationStatus);
        assistant.querySelector("[data-agent-lab-container-prepare]")?.addEventListener("click", prepareContainerRuntime);
    };
    try {
        status.textContent = "Checking execution environments…";
        const data = await api("/api/agent-lab/isolation");
        const selected = select.value || data.default || "bounded_process";
        const backends = Array.isArray(data.backends) ? data.backends : [];
        if (backends.length) {
            const automatic = data.automatic_supported
                ? [{ mode: "automatic", label: "Automatic selection", security_tier: "requirements-based", available: backends.some((item) => item.available), description: "Remy selects the least-privileged available environment that satisfies the declared task requirements." }]
                : [];
            const choices = [...automatic, ...backends];
            select.innerHTML = choices.map((item) => `<option value="${esc(item.mode)}" ${item.available ? "" : "disabled"}>${esc(item.label || item.mode)} · ${esc(item.security_tier || "registered")}</option>`).join("");
            const preferred = choices.find((item) => item.mode === selected && item.available)
                || backends.find((item) => item.mode === data.default && item.available)
                || backends.find((item) => item.available);
            if (preferred) select.value = preferred.mode;
        }
        const container = backends.find((item) => item.mode === "container_required") || data.container || {};
        const active = select.value === "automatic"
            ? { available: true, label: "Automatic selection", engine: "requirements-based", description: "The coordinator declares needs; policy selects the least-privileged matching backend." }
            : backends.find((item) => item.mode === select.value) || container;
        status.textContent = active.available
            ? `${active.label || active.mode} · ${active.engine || "ready"}`
            : `Unavailable · ${active.reason || "Execution backend is not ready"}`;
        status.title = active.description || "";
        if (container.mode) renderAssistant(container);
        else {
            assistant.hidden = true;
            assistant.innerHTML = "";
        }
    } catch (error) {
        status.textContent = `Unavailable · ${error.message}`;
        renderAssistant({ reason: error.message, reason_code: "unknown" });
    }
}

async function prepareContainerRuntime(event) {
    const button = event.currentTarget;
    const status = document.getElementById("agent-lab-isolation-status");
    const assistant = document.getElementById("agent-lab-container-assistant");
    button.disabled = true;
    button.textContent = "Preparing…";
    if (status) status.textContent = "Preparing the pinned local runtime…";
    try {
        await api("/api/agent-lab/isolation/prepare", { method: "POST" });
        await loadIsolationStatus();
    } catch (error) {
        button.disabled = false;
        button.textContent = "Try again";
        if (status) status.textContent = `Preparation failed · ${error.message}`;
        assistant?.classList.add("agent-lab-error");
    }
}

async function startAutonomous(event, run) {
    event.preventDefault();
    const button = event.currentTarget.querySelector("button[type=submit]");
    const status = document.getElementById("agent-lab-autonomous-status");
    button.disabled = true;
    try {
        await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/autonomous`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: document.getElementById("agent-lab-autonomous-model").value,
                verifier_model: document.getElementById("agent-lab-verifier-model").value,
                max_repair_rounds: Number(document.getElementById("agent-lab-autonomous-repairs").value),
                isolation_mode: document.getElementById("agent-lab-isolation-mode").value,
            }),
        });
        await loadAgentLab();
    } catch (error) {
        if (status) { status.textContent = error.message; status.classList.add("agent-lab-error"); }
        button.disabled = false;
    }
}

function renderTeam(team = [], delegation = {}) {
    if (!team.length) return '<div class="empty-state">Remy selects roles during preparation.</div>';
    const members = `<div class="agent-lab-team">${team.map((agent) => `<article>
        <span>${esc(agent.agent_id)}</span>
        <b>${esc(agent.name)}</b>
        ${agent.status ? `<i class="agent-lab-status ${agent.status === "success" ? "success" : agent.status === "error" ? "danger" : "neutral"}">${esc(agent.status)}</i>` : ""}
        <p>${esc(agent.responsibility)}</p>
        ${agent.model ? `<small><b>Model:</b> ${esc(agent.model)}${agent.served_by && agent.served_by !== agent.model ? ` → ${esc(agent.served_by)}` : ""}</small>` : ""}
        ${agent.capability_profile ? `<small>${esc(agent.capability_profile)}${(agent.allowed_tools || []).length ? ` · ${(agent.allowed_tools || []).map(esc).join(", ")}` : ""}</small>` : ""}
    </article>`).join("")}</div>`;
    const results = delegation.results || [];
    if (!results.length) return members;
    return `${members}<details class="agent-lab-team-results"><summary>Specialist results (${results.length})</summary>${results.map((item) => `<article>
        <header><b>${esc(item.member_id || item.role)}</b><i class="agent-lab-status ${item.status === "success" ? "success" : "danger"}">${esc(item.status)}</i><span>${esc(item.served_by || item.assigned_model || "automatic")} · ${Number(item.tool_calls || 0)} tool calls · ${Number(item.elapsed_sec || 0).toFixed(1)}s</span></header>
        <pre>${esc(item.output)}</pre>
    </article>`).join("")}</details>`;
}

function renderBuilderFanout(fanout = {}, claims = []) {
    if (!fanout.status && !claims.length) {
        return '<div class="empty-state">Complex builds can be split into isolated, non-overlapping source claims.</div>';
    }
    const members = fanout.members || [];
    const status = fanout.status || "pending";
    const order = (fanout.merge_order || []).length
        ? `<div class="agent-lab-merge merged"><b>Deterministic fan-in</b><span>${(fanout.merge_order || []).map(esc).join(" → ")}</span></div>`
        : "";
    const memberCards = members.length ? `<div class="agent-lab-team">${members.map((member) => {
        const id = member.builder_id || member.id || "builder";
        const ownClaims = claims.filter((claim) => claim.builder_id === id);
        const claimText = ownClaims.length
            ? ownClaims.map((claim) => `${claim.path} · ${claim.status}`).join("\n")
            : (member.file_claims || []).join("\n");
        return `<article>
            <span>${esc(id)}</span><b>${esc(member.model || "automatic")}</b>
            <i class="agent-lab-status ${member.status === "merged" ? "success" : member.status === "failed" || member.status === "conflict" ? "danger" : "neutral"}">${esc(member.status || status)}</i>
            <p>${esc(member.instruction || fanout.reason || "Bounded source shard")}</p>
            <small>${esc(member.workspace_id || "snapshot pending")}${member.merge_id ? ` · ${esc(member.merge_id)}` : ""}</small>
            ${claimText ? `<pre>${esc(claimText)}</pre>` : ""}
        </article>`;
    }).join("")}</div>` : "";
    return `<div class="agent-lab-policy-row"><span>${esc(status)}</span><span>${members.length} builder(s)</span><span>${claims.length} exact claim(s)</span></div>${memberCards}${order}`;
}

function renderPlan(workflow = {}, fallback = []) {
    const nodes = workflow.nodes || [];
    if (!nodes.length) {
        if (!fallback.length) return '<div class="empty-state">The plan is agent-owned and appears after preparation.</div>';
        return `<ol class="agent-lab-plan">${fallback.map((step) => `<li><i>${esc(step.status)}</i><b>${esc(step.title)}</b><span>Owner: ${esc(step.owner)}</span></li>`).join("")}</ol>`;
    }
    return `<div class="agent-lab-policy-row"><span>plan ${esc(workflow.plan_id)}</span><span>${Number(workflow.max_parallel || 1)} parallel max</span><span>${Number(workflow.max_total_agents || 1)} agents max</span></div>
        <ol class="agent-lab-plan">${nodes.map((node) => `<li>
            <i>${esc(node.status)}</i><b>${esc(node.title)}</b>
            <span>${esc(node.role)} · ${esc(node.owner)} · ${esc(node.model || "automatic")}</span>
            <small>${esc(node.workspace_mode)}${(node.depends_on || []).length ? ` · after ${(node.depends_on || []).map(esc).join(", ")}` : " · root"} · gate ${esc(node.success_gate?.kind || "node_output")}</small>
        </li>`).join("")}</ol>`;
}

function renderCognitiveLedgers(task = {}, progress = []) {
    if (!task.version) return "";
    const criteria = task.success_criteria || [];
    const blockers = (task.blockers || []).filter((item) => !item.resolved);
    const latest = progress.length ? progress[progress.length - 1] : null;
    return `<div class="agent-lab-observer-grid agent-lab-cognitive-ledgers">
        <details class="agent-lab-ledger" open><summary>Task Ledger · revision ${Number(task.plan_revision || 1)}</summary>
            <div><b>Facts</b><span>${Number((task.facts || []).length)}</span></div>
            <div><b>Assumptions</b><span>${Number((task.assumptions || []).length)}</span></div>
            <div><b>Success gates</b><span>${criteria.filter((item) => item.status === "satisfied").length}/${criteria.length} satisfied</span></div>
            <div><b>Open blockers</b><span>${blockers.length}</span></div>
            ${blockers.slice().reverse().slice(0, 5).map((item) => `<div><time>${esc(formatTime(item.at))}</time><b>${esc(item.node_id || item.kind)}</b><span>${esc(item.message)}</span></div>`).join("")}
        </details>
        <details class="agent-lab-ledger" open><summary>Progress Ledger (${progress.length})</summary>
            ${latest ? `<div><time>${esc(formatTime(latest.at))}</time><b>${esc(latest.phase)}</b><span>${latest.progress_made ? "progress" : "checkpoint"} · ${esc(latest.state_fingerprint)}</span></div>` : ""}
            ${progress.slice().reverse().slice(0, 8).map((item) => `<div><time>#${Number(item.sequence || 0)}</time><b>${esc(item.reason || item.phase)}</b><span>${(item.completed_nodes || []).length} done · ${(item.active_nodes || []).join(", ") || "no active node"}</span></div>`).join("")}
        </details>
    </div>`;
}

function renderWorkspace(run) {
    const artifacts = run.artifacts || [];
    return `<div class="agent-lab-workspace"><span>${run.workspace.ready ? "Ready" : "Not prepared"}</span><code>${esc(run.workspace.relative_path)}</code></div>${artifacts.length ? artifacts.map((item) => `<a class="agent-lab-artifact" href="/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/artifacts/${encodeURIComponent(item.artifact_id)}" download><b>${esc(item.name || item.path)}</b><span>${Number(item.size || 0).toLocaleString()} bytes · ${esc(item.mime_type)}</span></a>`).join("") : '<div class="empty-state">No produced artifacts yet.</div>'}`;
}

function renderWorkspaceBranches(run) {
    const branches = run.workspace_branches || [];
    const merges = run.merge_receipts || [];
    const latestBranch = branches.length ? branches[branches.length - 1] : null;
    const latestMerge = merges.length ? merges[merges.length - 1] : null;
    const proof = run.proof_pack || {};
    if (!latestBranch && !latestMerge && !proof.sha256) return "";
    return `${latestBranch ? `<div class="agent-lab-merge ${esc(latestBranch.status)}"><b>Private snapshot · ${esc(latestBranch.status)}</b><span>${esc(latestBranch.workspace_id)} · ${esc(String(latestBranch.baseline_root_hash || "").slice(0, 12))}</span></div>` : ""}
        ${latestMerge ? `<div class="agent-lab-merge ${esc(latestMerge.status)}"><b>Merge gate · ${esc(latestMerge.status)}</b><span>${(latestMerge.applied_files || []).length} applied · ${(latestMerge.conflicts || []).length} conflicts · ${esc(String(latestMerge.canonical_after_hash || latestMerge.canonical_before_hash || "").slice(0, 12))}</span></div>` : ""}
        ${proof.sha256 ? `<div class="agent-lab-merge merged"><b>Proof Pack · ${esc(proof.decision)}</b><span>${esc(String(proof.sha256).slice(0, 16))} · plan ${esc(String(proof.plan_sha256 || "").slice(0, 12))}</span></div>` : ""}
        ${merges.length ? `<details class="agent-lab-team-results"><summary>Merge receipts (${merges.length})</summary>${merges.slice().reverse().map((item) => `<article><header><b>${esc(item.merge_id)}</b><i class="agent-lab-status ${item.status === "merged" || item.status === "no_changes" ? "success" : "danger"}">${esc(item.status)}</i><span>${esc(formatTime(item.created_at))}</span></header><pre>${esc((item.changes || []).map((change) => `${change.action} ${change.path}`).join("\n") || (item.conflicts || []).map((conflict) => `${conflict.reason} ${conflict.path || ""}`).join("\n") || "No changes")}</pre></article>`).join("")}</details>` : ""}`;
}

function renderSnapshotRetention(run) {
    const cleanups = run.snapshot_cleanup_receipts || [];
    const latest = cleanups.length ? cleanups[cleanups.length - 1] : null;
    return `${latest ? `<div class="agent-lab-merge merged"><b>Last cleanup · ${esc(latest.status)}</b><span>${Number(latest.removed_count || 0)} removed · ${formatBytes(latest.recovered_bytes)}</span></div>` : ""}
        <div id="agent-lab-retention" class="agent-lab-retention"><button type="button" data-lab-retention class="btn btn-outline">Inspect snapshot storage</button><span>Active snapshots are always protected.</span></div>`;
}

function formatBytes(value) {
    const bytes = Math.max(0, Number(value || 0));
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function loadSnapshotRetention(run) {
    const host = document.getElementById("agent-lab-retention");
    if (!host) return;
    host.innerHTML = '<span>Inspecting retained snapshots…</span>';
    try {
        const data = await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/snapshots/retention`);
        const retention = data.retention || {};
        const items = retention.items || [];
        const conflicts = items.filter((item) => item.conflict_cleanup_eligible).map((item) => item.workspace_id);
        host.innerHTML = `<div class="agent-lab-retention-summary">
            <b>${Number(retention.snapshot_count || 0)} retained snapshot(s)</b>
            <span>${formatBytes(retention.used_bytes)} / ${formatBytes(retention.budget_bytes)}</span>
            <span>${Number(retention.protected_count || 0)} protected · ${Number(retention.cleanup_eligible_count || 0)} merged eligible · ${Number(retention.conflict_count || 0)} conflict</span>
        </div>
        ${items.length ? `<details class="agent-lab-team-results"><summary>Retention inventory (${items.length})</summary>${items.map((item) => `<article><header><b>${esc(item.workspace_id)}</b><i class="agent-lab-status ${item.status === "merged" ? "success" : item.status === "conflict" ? "danger" : "neutral"}">${esc(item.status)}</i><span>${formatBytes(item.size_bytes)}</span></header><pre>${esc(item.node_id || "unknown node")} · ${item.active_claim ? "active claim · protected" : item.cleanup_eligible ? "cleanup eligible" : item.conflict_cleanup_eligible ? "explicit confirmation required" : "protected"}</pre></article>`).join("")}</details>` : '<div class="empty-state">No private snapshots are retained.</div>'}
        <div class="agent-lab-actions">
            ${retention.cleanup_eligible_count && run.status !== "running" ? '<button type="button" id="agent-lab-clean-merged" class="btn btn-outline">Clean merged snapshots</button>' : ""}
            ${conflicts.length && run.status !== "running" ? '<button type="button" id="agent-lab-clean-conflicts" class="btn btn-danger">Clean conflict snapshots…</button>' : ""}
        </div>`;
        document.getElementById("agent-lab-clean-merged")?.addEventListener("click", () => cleanupSnapshots(run, [], false));
        document.getElementById("agent-lab-clean-conflicts")?.addEventListener("click", () => cleanupSnapshots(run, conflicts, true));
    } catch (error) {
        host.innerHTML = `<div class="agent-lab-error">${esc(error.message)}</div>`;
    }
}

async function cleanupSnapshots(run, workspaceIds, includeConflicts) {
    const approved = await showConfirm(
        includeConflicts ? "Delete conflict snapshots?" : "Clean merged snapshots?",
        includeConflicts
            ? "Conflict source copies will be permanently removed. Canonical files, merge receipts, claims, and audit history remain."
            : "Only closed merged copies will be removed. Canonical files and audit receipts remain.",
    );
    if (!approved) return;
    const host = document.getElementById("agent-lab-retention");
    if (host) host.innerHTML = '<span>Cleaning approved snapshots…</span>';
    try {
        await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/snapshots/cleanup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ workspace_ids: workspaceIds, include_conflicts: includeConflicts }),
        });
        await loadAgentLab();
    } catch (error) {
        if (host) host.innerHTML = `<div class="agent-lab-error">${esc(error.message)}</div>`;
    }
}

function renderVerification(items = []) {
    if (!items.length) return '<div class="empty-state">Independent checks will appear here with evidence.</div>';
    return items.slice().reverse().map((item) => `<div class="agent-lab-verification ${esc(item.world_fact)}"><b>${esc(item.world_fact)}</b><span>${esc(item.entrypoint)} · exit ${esc(item.exit_code)}</span></div>`).join("");
}

function renderExecutions(items = []) {
    if (!items.length) return "";
    return `<details class="agent-lab-console" open><summary>Execution console (${items.length})</summary>${items.slice().reverse().map((item) => `
        <article>
            <header><b>${esc(item.entrypoint)}</b><i class="agent-lab-status ${item.status === "passed" ? "success" : "danger"}">${esc(item.status)}</i><span>${esc(item.isolation_mode || "bounded_process")} · ${Number(item.duration_ms || 0)}ms · ${Number(item.peak_memory_mb || 0).toFixed(1)} MB</span></header>
            ${item.stdout ? `<label>stdout</label><pre>${esc(item.stdout)}</pre>` : ""}
            ${item.stderr ? `<label>stderr</label><pre class="stderr">${esc(item.stderr)}</pre>` : ""}
        </article>`).join("")}</details>`;
}

function renderStageEditor(run, path) {
    const host = document.getElementById("agent-lab-stage-editor");
    if (!host) return;
    const verification = path.startsWith("tests/");
    const sample = verification
        ? `from pathlib import Path\n\nartifact = Path("artifacts/result.txt")\nassert artifact.exists(), "Expected artifact is missing"\nassert artifact.read_text(encoding="utf-8").strip(), "Artifact is empty"\nprint("verification passed")\n`
        : `from pathlib import Path\n\nout = Path("artifacts/result.txt")\nout.write_text("Agent Lab produced this verified artifact.\\n", encoding="utf-8")\nprint(f"created {out}")\n`;
    host.innerHTML = `<form id="agent-lab-stage-form" class="agent-lab-stage-form">
        <header><b>${verification ? "Independent verification file" : "Sandboxed source file"}</b><button id="agent-lab-stage-close" type="button" class="btn btn-outline">Close</button></header>
        <label>Workspace path<input id="agent-lab-stage-path" value="${esc(path)}" readonly></label>
        <label>Python source<textarea id="agent-lab-stage-content" rows="12" spellcheck="false">${esc(sample)}</textarea></label>
        <p class="settings-hint">Only allowlisted standard-library imports are accepted. Network, shell, system modules, path escapes, dynamic code, and inherited API keys are blocked.</p>
        <button type="submit" class="btn btn-primary">Stage in isolated workspace</button>
        <span id="agent-lab-stage-status" class="settings-hint"></span>
    </form>`;
    document.getElementById("agent-lab-stage-close")?.addEventListener("click", () => { host.innerHTML = ""; });
    document.getElementById("agent-lab-stage-form")?.addEventListener("submit", (event) => stageFile(event, run));
}

async function stageFile(event, run) {
    event.preventDefault();
    const button = event.currentTarget.querySelector("button[type=submit]");
    const status = document.getElementById("agent-lab-stage-status");
    button.disabled = true;
    try {
        const data = await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/files`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                path: document.getElementById("agent-lab-stage-path").value,
                content: document.getElementById("agent-lab-stage-content").value,
            }),
        });
        if (status) status.textContent = `Staged ${data.file.path} · ${data.file.size} bytes`;
    } catch (error) {
        if (status) { status.textContent = error.message; status.classList.add("agent-lab-error"); }
    } finally {
        button.disabled = false;
    }
}

async function runAction(run, action) {
    if (action === "cancel") {
        const approved = await showConfirm("Cancel Agent Lab run", "Its workspace and audit history will be preserved.");
        if (!approved) return;
    }
    document.querySelectorAll("[data-lab-action]").forEach((button) => { button.disabled = true; });
    try {
        await api(`/api/agent-lab/runs/${encodeURIComponent(run.run_id)}/${action}`, { method: "POST" });
        await loadAgentLab();
    } catch (error) {
        await loadAgentLab();
        document.getElementById("agent-lab-detail")?.insertAdjacentHTML(
            "afterbegin",
            `<div class="agent-lab-error" style="margin-bottom:10px">${esc(error.message)}</div>`,
        );
    }
}
