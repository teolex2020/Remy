/**
 * Activity View — autonomous agent activity dashboard.
 * Live mode: real-time thought stream via WebSocket.
 * History mode: goals, outcomes, reflections, proactive sessions with summary stats.
 */

import { skeletonCards, EMPTY, errorState } from "./ui.js";

import { eventCssName, eventDomain, eventField, eventName } from "./runtime-events.js";
import { loadMemoryStats, loadRecords, openRecordDetail } from "./memory.js";

const summaryEl = document.getElementById("activity-summary");
const listEl = document.getElementById("activity-list");
const filterEl = document.getElementById("activity-filter");

// Live stream elements
const livePanelEl = document.getElementById("activity-live-panel");
const historyPanelEl = document.getElementById("activity-history-panel");
const streamEl = document.getElementById("activity-stream");
const liveStatusEl = document.getElementById("activity-live-status");
const liveCountEl = document.getElementById("activity-live-count");
const clearBtn = document.getElementById("btn-activity-clear");
const stopServerBtn = document.getElementById("btn-activity-stop-server");
const btnLive = document.getElementById("btn-activity-live");
const btnHistory = document.getElementById("btn-activity-history");
const autonomyToggleEl = document.getElementById("activity-autonomy-enabled");
const autonomyLabelEl = document.getElementById("activity-autonomy-label");
const plansEl = document.getElementById("activity-plans");
const doingEl = document.getElementById("activity-doing");
const decisionDossierBtn = document.getElementById("btn-activity-decision-dossier");
const queueEl = document.getElementById("activity-queue");
const approvalsEl = document.getElementById("activity-approvals");
const plansCountEl = document.getElementById("activity-plans-count");
const doingCountEl = document.getElementById("activity-doing-count");
const queueCountEl = document.getElementById("activity-queue-count");
const approvalsCountEl = document.getElementById("activity-approvals-count");

queueEl?.addEventListener("click", (event) => {
    const card = event.target.closest(".activity-queue-item[data-goal-id]");
    if (!card) return;
    const goalId = card.dataset.goalId || "";
    if (!goalId) return;
    openMissionQueueDetail(goalId).catch((err) => {
        console.error("Mission queue detail failed:", err);
    });
});

let _cachedData = null;
let _liveMode = true;
let _eventCount = 0;
const MAX_STREAM_EVENTS = 200;
const MAX_PANEL_EVENTS = 8;
const SNAPSHOT_REFRESH_MS = 15000;
const _panelState = {
    plans: [],
    doing: [],
    approvals: [],
    transitions: [],
};
const _snapshotState = {
    autonomyRunning: false,
    transportConnected: false,
    currentGoal: null,
    currentMission: null,
    currentTask: null,
    currentStep: null,
    lastCycleResult: null,
    currentRole: "",
    lastAgentResponse: null,
    lastResearchActivity: null,
    researchSession: null,
    schedulerReason: "",
    schedulerSelection: null,
    specialistResolution: null,
    stuckMissions: [],
    approvalQueue: [],
    qualityDebt: [],
    decisionFlow: [],
    activeHarnessModule: null,
};
let _snapshotTimer = null;
let _decisionDossierRefreshTimer = null;
let _decisionDossierRefreshInFlight = false;
let _lastSavedDecisionSnapshotId = "";
let _lastSavedDecisionSnapshotAt = 0;
let _lastSavedDecisionSnapshotContent = "";
let _lastSavedDecisionSnapshotSections = null;
let _showLastSavedDecisionSnapshot = false;
let _lastSavedDecisionSnapshotPinned = false;
let _autonomyPendingDesired = null;
let _autonomyPendingUntil = 0;

function _setAutonomyPending(desired, timeoutMs = 5000) {
    _autonomyPendingDesired = typeof desired === "boolean" ? desired : null;
    _autonomyPendingUntil = _autonomyPendingDesired === null ? 0 : (Date.now() + timeoutMs);
}

function _clearAutonomyPending() {
    _autonomyPendingDesired = null;
    _autonomyPendingUntil = 0;
}

function _hasAutonomyPending(desired) {
    return _autonomyPendingDesired === desired && Date.now() < _autonomyPendingUntil;
}

function isDecisionDossierOpen() {
    const app = document.querySelector(".app");
    const titleEl = document.getElementById("panel-title");
    return Boolean(
        _liveMode &&
        app?.classList.contains("panel-open") &&
        titleEl?.textContent === "Decision Dossier",
    );
}

function scheduleDecisionDossierRefresh() {
    if (!isDecisionDossierOpen() || _decisionDossierRefreshInFlight) return;
    if (_decisionDossierRefreshTimer) {
        window.clearTimeout(_decisionDossierRefreshTimer);
    }
    _decisionDossierRefreshTimer = window.setTimeout(async () => {
        _decisionDossierRefreshTimer = null;
        if (!isDecisionDossierOpen() || _decisionDossierRefreshInFlight) return;
        _decisionDossierRefreshInFlight = true;
        try {
            await openDecisionDossier();
        } catch (err) {
            console.error("Failed to refresh decision dossier", err);
        } finally {
            _decisionDossierRefreshInFlight = false;
        }
    }, 250);
}

async function openDecisionSnapshotInMemory(recordId = "", tags = "decision_dossier") {
    const memoryNav = document.querySelector('.nav-item[data-view="memory"]');
    memoryNav?.click();
    const tagsInput = document.getElementById("memory-tags");
    const searchInput = document.getElementById("memory-search");
    const tierSelect = document.getElementById("memory-tier");
    if (searchInput) searchInput.value = "";
    if (tagsInput) tagsInput.value = tags;
    if (tierSelect) tierSelect.value = "all";
    await loadRecords(false);
    await loadMemoryStats();
    if (recordId) {
        await openRecordDetail(recordId);
    }
}

// ============== Mode Toggle ==============

btnLive.addEventListener("click", () => switchMode("live"));
btnHistory.addEventListener("click", () => switchMode("history"));
autonomyToggleEl?.addEventListener("change", async () => {
    try {
        autonomyToggleEl.disabled = true;
        _setAutonomyPending(Boolean(autonomyToggleEl.checked));
        updateAutonomyToggle(Boolean(autonomyToggleEl.checked));
        const status = await window.apiClient.toggleAutonomy();
        updateAutonomyToggle(Boolean(status.running));
    } catch (err) {
        console.error("Failed to toggle autonomy", err);
        _clearAutonomyPending();
        await refreshAutonomyStatus();
    } finally {
        autonomyToggleEl.disabled = false;
    }
});
stopServerBtn?.addEventListener("click", async () => {
    const confirmed = window.confirm("Stop the Remy server?");
    if (!confirmed) return;
    try {
        stopServerBtn.disabled = true;
        document.dispatchEvent(new CustomEvent("server-shutdown-started"));
        await window.apiClient.shutdownServer();
    } catch (err) {
        console.error("Failed to stop server", err);
        document.dispatchEvent(new CustomEvent("server-shutdown-failed"));
        stopServerBtn.disabled = false;
    }
});
decisionDossierBtn?.addEventListener("click", () => {
    openDecisionDossier().catch((err) => {
        console.error("Decision dossier failed:", err);
    });
});
document.addEventListener("open-decision-dossier", () => {
    openDecisionDossier().catch((err) => {
        console.error("Decision dossier failed:", err);
    });
});

function switchMode(mode) {
    _liveMode = mode === "live";
    btnLive.classList.toggle("active", _liveMode);
    btnHistory.classList.toggle("active", !_liveMode);
    livePanelEl.style.display = _liveMode ? "" : "none";
    historyPanelEl.style.display = _liveMode ? "none" : "";
    filterEl.style.display = _liveMode ? "none" : "";

    if (_liveMode) {
        window.apiClient.connectActivity();
        refreshAutonomyStatus();
        startSnapshotPolling();
    } else {
        window.apiClient.disconnectActivity();
        stopSnapshotPolling();
        if (_decisionDossierRefreshTimer) {
            window.clearTimeout(_decisionDossierRefreshTimer);
            _decisionDossierRefreshTimer = null;
        }
        loadHistoryData();
    }
}

async function refreshAutonomyStatus() {
    if (!autonomyToggleEl) return;
    try {
        const status = await window.apiClient.getAutonomyStatus();
        updateAutonomyToggle(Boolean(status.running));
        if (status.budget) updateBudgetBar(status.budget);
        applyAutonomySnapshot(status);
    } catch (err) {
        console.error("Failed to load autonomy status", err);
    }
}

function startSnapshotPolling() {
    stopSnapshotPolling();
    _snapshotTimer = window.setInterval(() => {
        if (_liveMode && !_snapshotState.transportConnected) {
            refreshAutonomyStatus();
        }
    }, SNAPSHOT_REFRESH_MS);
}

function stopSnapshotPolling() {
    if (_snapshotTimer) {
        window.clearInterval(_snapshotTimer);
        _snapshotTimer = null;
    }
}

function updateAutonomyToggle(running) {
    if (!autonomyToggleEl || !autonomyLabelEl) return;
    if (!running && _hasAutonomyPending(true)) {
        autonomyToggleEl.checked = true;
        autonomyLabelEl.textContent = "Starting...";
        return;
    }
    if (running && _hasAutonomyPending(false)) {
        autonomyToggleEl.checked = false;
        autonomyLabelEl.textContent = "Stopping...";
        return;
    }
    if (running === _autonomyPendingDesired) {
        _clearAutonomyPending();
    } else if (Date.now() >= _autonomyPendingUntil) {
        _clearAutonomyPending();
    }
    autonomyToggleEl.checked = !!running;
    autonomyLabelEl.textContent = running ? "On" : "Off";
}

// ============== Live Stream ==============

window.apiClient.onActivityStatus((status) => {
    liveStatusEl.classList.remove("connected", "disconnected", "reconnecting");
    const textEl = liveStatusEl.querySelector(".status-text");
    if (status === "connected") {
        liveStatusEl.classList.add("connected");
        textEl.textContent = "Stream Live";
        _snapshotState.transportConnected = true;
    } else if (status === "reconnecting") {
        liveStatusEl.classList.add("reconnecting");
        textEl.textContent = "Stream Reconnecting...";
        _snapshotState.transportConnected = false;
    } else {
        liveStatusEl.classList.add("disconnected");
        textEl.textContent = "Stream Disconnected";
        _snapshotState.transportConnected = false;
    }
});

clearBtn.addEventListener("click", () => {
    streamEl.innerHTML = "";
    _eventCount = 0;
    liveCountEl.textContent = "0 events";
    _panelState.plans = [];
    _panelState.doing = [];
    _panelState.approvals = [];
    _panelState.transitions = [];
    renderPanels();
});

window.apiClient.onActivityEvent((event) => {
    if (eventName(event) === "activity.snapshot" && event?.payload && typeof event.payload === "object") {
        const snapshot = event.payload;
        updateAutonomyToggle(Boolean(snapshot.running));
        if (snapshot.budget) {
            updateBudgetBar(snapshot.budget);
        }
        applyAutonomySnapshot(snapshot);
        scheduleDecisionDossierRefresh();
        return;
    }
    if (eventName(event) === "activity.delta" && event?.payload && typeof event.payload === "object") {
        if (_applyActivityDelta(event.payload)) {
            updateAutonomyToggle(Boolean(_snapshotState.autonomyRunning));
            renderPanels();
            scheduleDecisionDossierRefresh();
        }
        return;
    }

    // Update budget display on cycle events or init
    const budget = eventField(event, "budget", null);
    if (budget) {
        updateBudgetBar(budget);
    }

    // budget_init is a silent event — don't render in stream
    if (eventName(event) === "budget_init") return;

    _eventCount++;
    liveCountEl.textContent = `${_eventCount} events`;

    const el = renderStreamEvent(event);
    streamEl.appendChild(el);
    pushDecisionTransitionEvent(event);
    pushPanelEvent(event);

    while (streamEl.children.length > MAX_STREAM_EVENTS) {
        streamEl.removeChild(streamEl.firstChild);
    }

    streamEl.scrollTop = streamEl.scrollHeight;
});

function _applyActivityDelta(delta) {
    if (!delta || typeof delta !== "object") {
        return false;
    }
    let changed = false;
    if (delta.running !== undefined) {
        _snapshotState.autonomyRunning = Boolean(delta.running);
        changed = true;
    }
    if (delta.transport_connected !== undefined) {
        _snapshotState.transportConnected = Boolean(delta.transport_connected);
        changed = true;
    }
    if (delta.scheduler_reason !== undefined) {
        _snapshotState.schedulerReason = String(delta.scheduler_reason || "");
        changed = true;
    }
    if (delta.current_mission !== undefined) {
        _snapshotState.currentMission = delta.current_mission || null;
        changed = true;
    }
    if (delta.current_goal !== undefined) {
        _snapshotState.currentGoal = delta.current_goal || null;
        changed = true;
    }
    if (delta.current_task !== undefined) {
        _snapshotState.currentTask = delta.current_task || null;
        changed = true;
    }
    if (delta.current_step !== undefined) {
        _snapshotState.currentStep = delta.current_step || null;
        changed = true;
    }
    if (delta.last_cycle_result !== undefined) {
        _snapshotState.lastCycleResult = delta.last_cycle_result || null;
        changed = true;
    }
    if (delta.current_role !== undefined) {
        _snapshotState.currentRole = String(delta.current_role || "");
        changed = true;
    }
    if (delta.last_agent_response !== undefined) {
        _snapshotState.lastAgentResponse = delta.last_agent_response || null;
        changed = true;
    }
    if (delta.last_research_activity !== undefined) {
        _snapshotState.lastResearchActivity = delta.last_research_activity || null;
        changed = true;
    }
    if (delta.research_session !== undefined) {
        _snapshotState.researchSession = delta.research_session || null;
        changed = true;
    }
    if (delta.scheduler_selection !== undefined) {
        const previousMissionId = _snapshotState.schedulerSelection?.mission_id || "";
        _snapshotState.schedulerSelection = delta.scheduler_selection || null;
        const nextMissionId = _snapshotState.schedulerSelection?.mission_id || "";
        if (nextMissionId && nextMissionId !== previousMissionId) {
            pushDecisionTransition({
                type: "scheduler_selection",
                timestamp: Date.now() / 1000,
                text: `Mission selected: ${nextMissionId}`,
                detail: Number.isFinite(Number(_snapshotState.schedulerSelection?.score))
                    ? `score ${Number(_snapshotState.schedulerSelection.score).toFixed(2)}`
                    : "",
            });
        }
        changed = true;
    }
    if (delta.specialist_resolution !== undefined) {
        const previousSpecialist = _snapshotState.specialistResolution?.specialist_id || "";
        _snapshotState.specialistResolution = delta.specialist_resolution || null;
        const nextSpecialist = _snapshotState.specialistResolution?.specialist_id || "";
        if (nextSpecialist && nextSpecialist !== previousSpecialist) {
            pushDecisionTransition({
                type: "specialist_resolution",
                timestamp: Date.now() / 1000,
                text: `Specialist chosen: ${nextSpecialist}`,
                detail: [
                    _snapshotState.specialistResolution?.reason || "",
                    _snapshotState.specialistResolution?.degraded_specialist
                        ? `degraded ${_snapshotState.specialistResolution.degraded_specialist}`
                        : "",
                ].filter(Boolean).join(" | "),
            });
        }
        changed = true;
    }
    if (delta.decision_flow !== undefined) {
        _snapshotState.decisionFlow = Array.isArray(delta.decision_flow) ? delta.decision_flow : [];
        changed = true;
    }
    if (delta.budget) {
        updateBudgetBar(delta.budget);
        changed = true;
    }
    if (delta.approval_queue?.upsert) {
        const item = delta.approval_queue.upsert;
        const queue = Array.isArray(_snapshotState.approvalQueue) ? _snapshotState.approvalQueue.slice() : [];
        const filtered = queue.filter((entry) => entry.id !== item.id);
        filtered.unshift(item);
        _snapshotState.approvalQueue = filtered.slice(0, MAX_PANEL_EVENTS);
        pushDecisionTransition({
            type: "approval.pending",
            timestamp: Date.now() / 1000,
            text: item.description || item.action || "Pending approval",
            detail: [
                item.specialist ? `specialist ${item.specialist}` : "",
                item.routing_pressure ? "routing pressure" : "",
                item.context?.target || "",
            ].filter(Boolean).join(" | "),
        });
        changed = true;
    }
    if (delta.approval_queue?.remove_id) {
        pushDecisionTransition({
            type: "approval.resolved",
            timestamp: Date.now() / 1000,
            text: `Approval resolved: ${delta.approval_queue.remove_id}`,
            detail: "",
        });
        _snapshotState.approvalQueue = (Array.isArray(_snapshotState.approvalQueue) ? _snapshotState.approvalQueue : [])
            .filter((entry) => entry.id !== delta.approval_queue.remove_id);
        changed = true;
    }
    return changed;
}

function pushDecisionTransition(item) {
    if (!item) return;
    _panelState.transitions.push(item);
    while (_panelState.transitions.length > 12) {
        _panelState.transitions.shift();
    }
    scheduleDecisionDossierRefresh();
}

function pushDecisionTransitionEvent(event) {
    const currentEventName = eventName(event);
    const allowed = new Set([
        "goal_selected",
        "thinking",
        "plan_step",
        "cycle_start",
        "cycle_end",
        "approval.pending",
        "approval.resolved",
        "approval_pending",
        "approval_resolved",
        "goal_blocked",
        "goal_unblocked",
        "goal_resumed",
        "goal_failed",
        "goal_archived",
        "mission.task_active",
        "mission.task_completed",
        "mission.task_failed",
        "outcome",
        "evaluation",
    ]);
    if (!allowed.has(currentEventName)) return;
    pushDecisionTransition({
        type: currentEventName,
        timestamp: eventField(event, "timestamp", Date.now() / 1000),
        text: panelText(event),
        detail: panelTransitionDetail(event),
    });
}

function panelTransitionDetail(event) {
    const currentEventName = eventName(event);
    switch (currentEventName) {
        case "goal_selected":
            return [
                eventField(event, "priority", "") ? `priority ${eventField(event, "priority", "")}` : "",
                eventField(event, "mission_id", "") || "",
            ].filter(Boolean).join(" | ");
        case "cycle_start":
        case "cycle_end":
            return [
                eventField(event, "decision", ""),
                eventField(event, "reason", ""),
            ].filter(Boolean).join(" | ");
        case "approval.pending":
        case "approval_pending":
        case "approval.resolved":
        case "approval_resolved":
            return [
                eventField(event, "specialist", "") ? `specialist ${eventField(event, "specialist", "")}` : "",
                eventField(event, "routing_pressure", false) ? "routing pressure" : "",
                eventField(event, "decision", ""),
            ].filter(Boolean).join(" | ");
        case "mission.task_active":
        case "mission.task_completed":
        case "mission.task_failed":
            return [
                eventField(event, "mission_id", ""),
                eventField(event, "status", ""),
            ].filter(Boolean).join(" | ");
        default:
            return eventField(event, "reason", "") || eventField(event, "summary", "") || "";
    }
}

function pushPanelEvent(event) {
    const target = classifyPanel(event);
    if (!target) return;
    _panelState[target].push(event);
    while (_panelState[target].length > MAX_PANEL_EVENTS) {
        _panelState[target].shift();
    }
    renderPanels();
    scheduleDecisionDossierRefresh();
}

function classifyPanel(event) {
    const currentEventName = eventName(event);
    if (currentEventName === "goal_selected" || currentEventName === "thinking" || currentEventName === "plan_step") {
        return "plans";
    }
    if (currentEventName === "tool_call" || currentEventName === "tool_result" || currentEventName === "evaluation" || currentEventName === "outcome" || currentEventName === "cycle_start" || currentEventName === "cycle_end") {
        return "doing";
    }
    return null;
}

function renderPanels() {
    if (!plansEl || !doingEl || !queueEl || !approvalsEl) return;
    const planItems = buildPlanItems();
    const doingItems = buildDoingItems();
    const queueItems = buildMissionQueueItems();
    const approvalItems = buildApprovalItems();

    plansEl.innerHTML = renderPanelItems(planItems);
    doingEl.innerHTML = renderPanelItems(doingItems);
    queueEl.innerHTML = renderMissionQueueItems(queueItems);
    approvalsEl.innerHTML = renderApprovalItems(approvalItems);
    plansCountEl.textContent = String(planItems.length);
    doingCountEl.textContent = String(doingItems.length);
    queueCountEl.textContent = String(queueItems.length);
    approvalsCountEl.textContent = String(approvalItems.length);

    renderFocusPanel();
    _updateAgentStatusText();
}

function renderPanelItems(events) {
    if (!events.length) {
        return `<div class="activity-live-empty">No recent events</div>`;
    }
    return events.slice().reverse().map(renderPanelItem).join("");
}

function renderPanelItem(event) {
    const rawTs = eventField(event, "timestamp", event.created_at || event.updated_at || Date.now() / 1000);
    const ts = typeof rawTs === "number" ? rawTs : Date.parse(rawTs) / 1000;
    const time = Number.isFinite(ts) ? new Date(ts * 1000).toLocaleTimeString() : "";
    const text = event.text || panelText(event);
    const detail = event.detail || "";
    const currentEventName = eventCssName(event) || "event";
    return `
        <article class="activity-panel-item activity-panel-${esc(currentEventName)}">
            <div class="activity-panel-time">${time}</div>
            <div class="activity-panel-body">${esc(text)}</div>
            ${detail ? `<div class="activity-queue-detail">${esc(detail)}</div>` : ""}
        </article>
    `;
}

function renderApprovalItems(items) {
    if (!items.length) {
        return `<div class="activity-live-empty">No pending approvals</div>`;
    }
    return items.map((item) => `
        <article class="activity-panel-item activity-panel-approval">
            <div class="activity-panel-time">${esc(eventField(item, "risk_category", ""))}</div>
            <div class="activity-panel-body">${esc(eventField(item, "description", eventField(item, "action", "Pending approval")))}</div>
            ${renderApprovalMeta(item)}
        </article>
    `).join("");
}

function renderApprovalMeta(item) {
    const detailParts = [];
    if (item.specialist) {
        detailParts.push(`specialist ${item.specialist}`);
    }
    if (item.routing_pressure) {
        detailParts.push("routing pressure");
    }
    if (item.context?.target) {
        detailParts.push(item.context.target);
    }
    if (!detailParts.length) {
        return "";
    }
    return `<div class="activity-queue-detail">${esc(detailParts.join(" • "))}</div>`;
}

function renderMissionQueueItems(items) {
    if (!items.length) {
        return `<div class="activity-live-empty">No queued mission tasks</div>`;
    }
    return items.map((item) => `
        <article class="activity-panel-item activity-panel-mission_queue activity-queue-item${item.goalId ? " activity-queue-item-clickable" : ""}"${item.goalId ? ` data-goal-id="${esc(item.goalId)}"` : ""}>
            <div class="activity-panel-time"><span class="activity-queue-status activity-queue-status-${esc(item.status || "pending")}">${esc(item.status || "pending")}</span></div>
            <div class="activity-panel-body">${esc(item.text || "")}</div>
            ${item.detail ? `<div class="activity-queue-detail">${esc(item.detail)}</div>` : ""}
        </article>
    `).join("");
}

function buildPlanItems() {
    const items = [];
    if (_snapshotState.currentRole) {
        items.push({
            type: "snapshot_role",
            timestamp: Date.now() / 1000,
            text: `Role: ${_snapshotState.currentRole}`,
        });
    }
    if (_snapshotState.activeHarnessModule?.id) {
        items.push({
            type: "snapshot_harness_module",
            timestamp: Date.now() / 1000,
            text: `Harness: ${_snapshotState.activeHarnessModule.label || _snapshotState.activeHarnessModule.id}`,
            detail: _snapshotState.activeHarnessModule.reason || "",
        });
    }
    if (_snapshotState.currentGoal) {
        items.push({
            type: "snapshot_goal",
            timestamp: Date.now() / 1000,
            text: `Goal${_snapshotState.currentGoal.priority ? ` [${_snapshotState.currentGoal.priority}]` : ""}: ${_snapshotState.currentGoal.description || _snapshotState.currentGoal.id}`,
        });
    }
    if (_snapshotState.currentMission) {
        const missionMeta = [];
        if (
            Number.isFinite(Number(_snapshotState.currentMission.completed_tasks))
            && Number.isFinite(Number(_snapshotState.currentMission.total_tasks))
            && Number(_snapshotState.currentMission.total_tasks) > 0
        ) {
            missionMeta.push(`${_snapshotState.currentMission.completed_tasks}/${_snapshotState.currentMission.total_tasks} done`);
        }
        if (Number.isFinite(Number(_snapshotState.currentMission.pending_tasks)) && Number(_snapshotState.currentMission.pending_tasks) > 0) {
            missionMeta.push(`${_snapshotState.currentMission.pending_tasks} pending`);
        }
        if (Number.isFinite(Number(_snapshotState.currentMission.blocked_tasks)) && Number(_snapshotState.currentMission.blocked_tasks) > 0) {
            missionMeta.push(`${_snapshotState.currentMission.blocked_tasks} blocked`);
        }
        if (Number.isFinite(Number(_snapshotState.currentMission.failed_tasks)) && Number(_snapshotState.currentMission.failed_tasks) > 0) {
            missionMeta.push(`${_snapshotState.currentMission.failed_tasks} failed`);
        }
        if (Number.isFinite(Number(_snapshotState.currentMission.active_tasks)) && Number(_snapshotState.currentMission.active_tasks) > 0) {
            missionMeta.push(`${_snapshotState.currentMission.active_tasks} active tasks`);
        }
        if (Number.isFinite(Number(_snapshotState.currentMission.focus_stale_cycles)) && Number(_snapshotState.currentMission.focus_stale_cycles) > 0) {
            missionMeta.push(`stale ${_snapshotState.currentMission.focus_stale_cycles} cycles`);
        }
        items.push({
            type: "snapshot_plan",
            timestamp: Date.now() / 1000,
            text: `Mission: ${_snapshotState.currentMission.description || _snapshotState.currentMission.id}${missionMeta.length ? ` (${missionMeta.join(", ")})` : ""}`,
            detail: [
                _snapshotState.currentGoal?.priority ? `priority ${_snapshotState.currentGoal.priority}` : "",
                _snapshotState.currentMission.id ? `id ${_snapshotState.currentMission.id}` : "",
            ].filter(Boolean).join(" | "),
        });
        if (Array.isArray(_snapshotState.currentMission.pending_task_labels) && _snapshotState.currentMission.pending_task_labels.length) {
            items.push({
                type: "snapshot_mission_queue",
                timestamp: Date.now() / 1000,
                text: `Next tasks: ${_snapshotState.currentMission.pending_task_labels.join(" | ")}`,
            });
        }
    }
    if (_snapshotState.schedulerReason) {
        items.push({
            type: "scheduler_reason",
            timestamp: Date.now() / 1000,
            text: `Scheduler: ${_snapshotState.schedulerReason}`,
        });
    }
    if (_snapshotState.schedulerSelection?.mission_id) {
        const sel = _snapshotState.schedulerSelection;
        const detail = sel.details || {};
        const routingReason = detail.routing_reason ? `, ${detail.routing_reason}` : "";
        const score = Number.isFinite(Number(sel.score)) ? Number(sel.score).toFixed(2) : "";
        items.push({
            type: "scheduler_selection",
            timestamp: Date.now() / 1000,
            text: `Selected mission ${sel.mission_id}${score ? ` [score ${score}]` : ""}${routingReason}`,
            detail: [
                detail.preferred_specialist ? `preferred ${detail.preferred_specialist}` : "",
                detail.degraded_specialist ? `degraded ${detail.degraded_specialist}` : "",
                detail.candidate_count ? `${detail.candidate_count} candidates` : "",
            ].filter(Boolean).join(" | "),
        });
    }
    if (_snapshotState.stuckMissions.length) {
        items.push({
            type: "stuck_missions",
            timestamp: Date.now() / 1000,
            text: `Stuck missions: ${_snapshotState.stuckMissions.length}`,
        });
    }
    return items.concat(_panelState.plans);
}

function buildDoingItems() {
    const items = [];
    if (_snapshotState.currentTask?.action) {
        items.push({
            type: "snapshot_task",
            timestamp: Date.now() / 1000,
            text: `Task: ${_snapshotState.currentTask.action}`,
        });
    }
    if (_snapshotState.currentStep?.instruction) {
        items.push({
            type: "snapshot_step",
            timestamp: Date.now() / 1000,
            text: `Step: ${_snapshotState.currentStep.instruction}`,
        });
    }
    if (_snapshotState.lastCycleResult?.decision) {
        items.push({
            type: "snapshot_result",
            timestamp: Date.now() / 1000,
            text: `Last result: ${_snapshotState.lastCycleResult.decision}${_snapshotState.lastCycleResult.reason ? ` - ${_snapshotState.lastCycleResult.reason}` : ""}`,
        });
    }
    if (_snapshotState.lastAgentResponse?.response) {
        items.push({
            type: "snapshot_response",
            timestamp: Date.now() / 1000,
            text: `Response: ${_snapshotState.lastAgentResponse.response}`,
        });
    }
    if (_snapshotState.lastResearchActivity?.summary) {
        items.push({
            type: "snapshot_research",
            timestamp: Date.now() / 1000,
            text: _snapshotState.lastResearchActivity.summary,
        });
    }
    const researchSession = _snapshotState.researchSession;
    if (researchSession?.topic) {
        const coverage = Number.isFinite(Number(researchSession.citation_coverage_rate))
            ? `${Math.round(Number(researchSession.citation_coverage_rate) * 100)}%`
            : "";
        const summaryBits = [
            Number.isFinite(Number(researchSession.generated_queries_count))
                ? `${researchSession.generated_queries_count} queries`
                : "",
            Number.isFinite(Number(researchSession.accepted_sources_count))
                ? `${researchSession.accepted_sources_count} accepted`
                : "",
            Number.isFinite(Number(researchSession.findings_count))
                ? `${researchSession.findings_count} findings`
                : "",
            Number.isFinite(Number(researchSession.contradictions_count))
                ? `${researchSession.contradictions_count} contradictions`
                : "",
            coverage ? `${coverage} cited` : "",
        ].filter(Boolean);
        items.push({
            type: "snapshot_research_session",
            timestamp: Date.now() / 1000,
            text: `Research: ${researchSession.topic}${summaryBits.length ? ` [${summaryBits.join(", ")}]` : ""}`,
        });
        if (Array.isArray(researchSession.knowledge_gaps) && researchSession.knowledge_gaps.length) {
        items.push({
            type: "snapshot_research_gap",
            timestamp: Date.now() / 1000,
            text: `Next gap: ${researchSession.knowledge_gaps[0]}`,
            detail: Array.isArray(researchSession.top_source_domains) && researchSession.top_source_domains.length
                ? `domains ${researchSession.top_source_domains.join(", ")}`
                : "",
        });
        }
        if (Array.isArray(researchSession.recent_queries) && researchSession.recent_queries.length) {
            items.push({
                type: "snapshot_research_queries",
                timestamp: Date.now() / 1000,
                text: `Queries: ${researchSession.recent_queries.join(" | ")}`.slice(0, 220),
            });
        }
        const artifact = researchSession.artifact || {};
        if (artifact.markdown_available || artifact.pdf_url) {
            items.push({
                type: "snapshot_research_artifact",
                timestamp: Date.now() / 1000,
                text: `Artifacts: ${artifact.markdown_available ? "markdown ready" : "markdown missing"}${artifact.pdf_url ? ", pdf ready" : ""}`,
            });
        }
    }
    if (_snapshotState.specialistResolution?.specialist_id) {
        const resolution = _snapshotState.specialistResolution;
        const quality = Number.isFinite(Number(resolution.quality_factor))
            ? Number(resolution.quality_factor).toFixed(2)
            : "";
        items.push({
            type: "snapshot_specialist_resolution",
            timestamp: Date.now() / 1000,
            text: `Routing: ${resolution.specialist_id}${resolution.reason ? ` (${resolution.reason})` : ""}${quality ? ` [${quality}]` : ""}`,
            detail: [
                resolution.preferred_specialist ? `preferred ${resolution.preferred_specialist}` : "",
                resolution.degraded_specialist ? `degraded ${resolution.degraded_specialist}` : "",
                resolution.override ? "override applied" : "",
            ].filter(Boolean).join(" | "),
        });
    }
    if (Array.isArray(_snapshotState.decisionFlow) && _snapshotState.decisionFlow.length) {
        _snapshotState.decisionFlow.slice(0, 4).forEach((item) => {
            items.push({
                type: `decision_flow_${item.stage || "stage"}`,
                timestamp: Date.now() / 1000,
                text: `${item.title || "Decision"}: ${item.summary || ""}`.trim(),
                detail: item.detail || "",
            });
        });
    }
    return items.concat(_panelState.doing);
}

function buildApprovalItems() {
    return (_snapshotState.approvalQueue || []).slice(0, MAX_PANEL_EVENTS);
}

function buildMissionQueueItems() {
    const items = Array.isArray(_snapshotState.currentMission?.pending_task_items)
        ? _snapshotState.currentMission.pending_task_items
        : [];
    if (items.length) {
        return items.slice(0, MAX_PANEL_EVENTS).map((item, index) => ({
            type: "mission_queue",
            text: `${index + 1}. ${item.label || ""}`,
            status: item.status || "pending",
            detail: item.detail || "",
            goalId: item.goal_id || "",
        }));
    }
    const labels = Array.isArray(_snapshotState.currentMission?.pending_task_labels)
        ? _snapshotState.currentMission.pending_task_labels
        : [];
    return labels.slice(0, MAX_PANEL_EVENTS).map((label, index) => ({
        type: "mission_queue",
        text: `${index + 1}. ${label}`,
        status: "pending",
        detail: "",
        goalId: "",
    }));
}

async function openMissionQueueDetail(goalId) {
    const panel = document.getElementById("context-panel");
    const titleEl = document.getElementById("panel-title");
    const contentEl = document.getElementById("panel-content");
    if (!panel || !titleEl || !contentEl) return;

    document.querySelector(".app")?.classList.add("panel-open");
    titleEl.textContent = "Mission Task";
    contentEl.innerHTML = '<p style="color:var(--text-muted)">Loading...</p>';

    const [todosData, historyData, systemData] = await Promise.all([
        window.apiClient.getTodos("all", null, 250, 30),
        window.apiClient.getGoalHistory(goalId),
        window.apiClient.getSystemStatus(),
    ]);
    const goal = (todosData?.todos || []).find((item) => item.id === goalId);
    if (!goal) {
        contentEl.innerHTML = '<div class="empty-state">Task detail not found.</div>';
        return;
    }

    const summary = historyData?.summary || {};
    const attempts = historyData?.attempts || [];
    const linkedApproval = (systemData?.approvals?.pending || []).find(
        (item) => item.id === goal.blocked_action_id,
    ) || null;
    const canArchive = goal.status !== "archived";
    const canResume = goal.status === "blocked_external" || goal.status === "blocked_by_user";
    const canUnblock = canResume;
    const contextRows = [
        ["Priority", goal.priority || ""],
        ["Pack", goal.capability_pack?.name || goal.goal_template || ""],
        ["Approval", goal.approval_mode || ""],
        ["Mission", goal.mission_id || ""],
        ["Task Action", goal.task_action || ""],
        ["Depends On", goal.task_depends_on || ""],
        ["Due", goal.due_date || ""],
    ].filter(([, value]) => value);
    const researchRows = [
        ["Mode", goal.research_mode || ""],
        ["Scope", goal.source_scope || ""],
        ["Accepted Sources", Number.isFinite(Number(goal.accepted_sources_count)) ? String(goal.accepted_sources_count) : ""],
        ["Rejected Sources", Number.isFinite(Number(goal.rejected_sources_count)) ? String(goal.rejected_sources_count) : ""],
        ["Contradictions", Number.isFinite(Number(goal.contradictions_count)) ? String(goal.contradictions_count) : ""],
        [
            "Citation Coverage",
            Number.isFinite(Number(goal.citation_coverage_rate))
                ? `${Math.round(Number(goal.citation_coverage_rate) * 100)}%`
                : "",
        ],
    ].filter(([, value]) => value && value !== "0" && value !== "0%");
    const researchArtifact = goal.research_session?.artifact || null;
    contentEl.innerHTML = `
        <div class="goal-detail">
            <div class="goal-detail-header">
                <div class="goal-detail-title">${esc(goal.title || goal.content || goalId)}</div>
                <div class="goal-detail-badges">
                    <span class="task-pill task-pill-${goal.block_status ? "blocked" : goal.status === "done" ? "success" : goal.status === "in_progress" ? "active" : "pending"}">${esc(goal.block_status || goal.status || "pending")}</span>
                </div>
            </div>
            <div class="goal-detail-grid">
                <div class="goal-detail-metric"><div class="goal-detail-metric-label">Attempts</div><div class="goal-detail-metric-value">${esc(String(summary.total_attempts || goal.attempts || 0))}</div></div>
                <div class="goal-detail-metric"><div class="goal-detail-metric-label">Success Rate</div><div class="goal-detail-metric-value">${esc(`${Math.round((summary.success_rate || 0) * 100)}%`)}</div></div>
                <div class="goal-detail-metric"><div class="goal-detail-metric-label">Goal ID</div><div class="goal-detail-metric-value">${esc(goalId.slice(0, 12))}</div></div>
                <div class="goal-detail-metric"><div class="goal-detail-metric-label">Updated</div><div class="goal-detail-metric-value">${esc(goal.updated_at || goal.timestamp || "")}</div></div>
            </div>
            ${goal.blocked_reason ? `<div class="goal-detail-alert goal-detail-alert-danger">${esc(goal.blocked_reason)}</div>` : ""}
            ${goal.resume_context ? `<div class="goal-detail-alert goal-detail-alert-info">Resume: ${esc(goal.resume_context)}</div>` : ""}
            ${goal.blocked_evidence ? `<div class="goal-detail-alert goal-detail-alert-info">Evidence: ${esc(goal.blocked_evidence)}</div>` : ""}
            ${linkedApproval ? `
                <div class="goal-detail-section">
                    <h4>Pending Approval</h4>
                    <div class="goal-detail-alert goal-detail-alert-info">
                        ${esc(linkedApproval.description || "Approval required")}
                        ${linkedApproval.age_sec ? ` (${esc(String(linkedApproval.age_sec))}s old)` : ""}
                    </div>
                    <div class="panel-actions goal-detail-actions">
                        <button class="btn-approve" data-goal-action="approve-linked" data-action-id="${esc(linkedApproval.id)}">Approve</button>
                        <button class="btn-deny" data-goal-action="deny-linked" data-action-id="${esc(linkedApproval.id)}">Deny</button>
                    </div>
                </div>
            ` : ""}
            <div class="panel-actions goal-detail-actions">
                ${canArchive ? '<button class="btn-outline" data-goal-action="archive">Archive</button>' : ""}
                ${canUnblock ? '<button class="btn-outline" data-goal-action="unblock">Unblock</button>' : ""}
                ${canResume ? '<button class="btn-primary" data-goal-action="resume">Resume</button>' : ""}
            </div>
            ${contextRows.length ? `
                <div class="goal-detail-section">
                    <h4>Execution Context</h4>
                    <div class="goal-detail-grid">
                        ${contextRows.map(([label, value]) => `
                            <div class="goal-detail-metric">
                                <div class="goal-detail-metric-label">${esc(label)}</div>
                                <div class="goal-detail-metric-value">${esc(value)}</div>
                            </div>
                        `).join("")}
                    </div>
                </div>
            ` : ""}
            ${researchRows.length ? `
                <div class="goal-detail-section">
                    <h4>Research Context</h4>
                    <div class="goal-detail-grid">
                        ${researchRows.map(([label, value]) => `
                            <div class="goal-detail-metric">
                                <div class="goal-detail-metric-label">${esc(label)}</div>
                                <div class="goal-detail-metric-value">${esc(value)}</div>
                            </div>
                        `).join("")}
                    </div>
                </div>
            ` : ""}
            ${researchArtifact && (researchArtifact.markdown_available || researchArtifact.pdf_url) ? `
                <div class="goal-detail-section">
                    <h4>Research Artifacts</h4>
                    <div class="goal-detail-grid">
                        <div class="goal-detail-metric">
                            <div class="goal-detail-metric-label">Markdown</div>
                            <div class="goal-detail-metric-value">
                                ${researchArtifact.viewer_url ? `<a href="${esc(researchArtifact.viewer_url)}" target="_blank" rel="noreferrer">open report</a>` : (researchArtifact.markdown_url ? `<a href="${esc(researchArtifact.markdown_url)}" target="_blank" rel="noreferrer">open report.md</a>` : (researchArtifact.markdown_available ? "ready" : "missing"))}
                            </div>
                        </div>
                        ${researchArtifact.pdf_url ? `
                            <div class="goal-detail-metric">
                                <div class="goal-detail-metric-label">PDF</div>
                                <div class="goal-detail-metric-value"><a href="${esc(researchArtifact.pdf_url)}" target="_blank" rel="noreferrer">${esc(researchArtifact.pdf_filename || "report.pdf")}</a></div>
                            </div>
                        ` : ""}
                        ${researchArtifact.record_id ? `
                            <div class="goal-detail-metric">
                                <div class="goal-detail-metric-label">Artifact ID</div>
                                <div class="goal-detail-metric-value">${esc(researchArtifact.record_id)}</div>
                            </div>
                        ` : ""}
                    </div>
                    ${researchArtifact.markdown_preview ? `<div class="goal-detail-alert goal-detail-alert-info">${esc(researchArtifact.markdown_preview)}</div>` : ""}
                </div>
            ` : ""}
            <div class="goal-detail-section">
                <h4>Recent Attempts</h4>
                ${attempts.length ? attempts.slice().reverse().slice(0, 5).map((attempt) => `
                    <div class="timeline-attempt ${attempt.success ? "success" : "failure"}">
                        <div class="timeline-attempt-header">
                            <div class="timeline-attempt-title">${esc(attempt.worker || "agent")} · ${esc(attempt.status || "unknown")}</div>
                            <div class="timeline-attempt-meta">${esc(attempt.timestamp || "")}</div>
                        </div>
                        ${attempt.reason ? `<div class="timeline-attempt-evidence">${esc(attempt.reason)}</div>` : ""}
                    </div>
                `).join("") : '<div class="empty-state">No attempts recorded yet.</div>'}
            </div>
        </div>
    `;

    const actionButtons = contentEl.querySelectorAll("[data-goal-action]");
    actionButtons.forEach((button) => {
        button.addEventListener("click", async () => {
            const action = button.dataset.goalAction || "";
            const actionId = button.dataset.actionId || "";
            const originalText = button.textContent || "";
            const buttons = Array.from(actionButtons);
            buttons.forEach((entry) => { entry.disabled = true; });
            button.textContent = "Working...";
            try {
                if (action === "archive") {
                    await window.apiClient.archiveAutonomyGoal(goalId, "archived_from_activity_queue");
                } else if (action === "unblock") {
                    await window.apiClient.unblockAutonomyGoal(goalId);
                } else if (action === "resume") {
                    const note = window.prompt("Resume note", goal.resume_context || "") || "";
                    await window.apiClient.resumeAutonomyGoal(goalId, note);
                } else if (action === "approve-linked" && actionId) {
                    await window.apiClient.approveAction(actionId);
                } else if (action === "deny-linked" && actionId) {
                    await window.apiClient.rejectAction(actionId);
                }
                await refreshAutonomyStatus();
                await openMissionQueueDetail(goalId);
            } catch (err) {
                console.error(`Failed to ${action} goal`, err);
                window.alert(`Failed to ${action} task.`);
                button.textContent = originalText;
                buttons.forEach((entry) => { entry.disabled = false; });
            }
        });
    });
}

async function openDecisionDossier() {
    const panel = document.getElementById("context-panel");
    const titleEl = document.getElementById("panel-title");
    const contentEl = document.getElementById("panel-content");
    if (!panel || !titleEl || !contentEl) return;

    document.querySelector(".app")?.classList.add("panel-open");
    titleEl.textContent = "Decision Dossier";
    contentEl.innerHTML = '<p style="color:var(--text-muted)">Loading...</p>';

    const [status, systemData, todosData] = await Promise.all([
        window.apiClient.getAutonomyStatus(),
        window.apiClient.getSystemStatus(),
        window.apiClient.getTodos("all", null, 250, 30),
    ]);

    const mission = status?.current_mission || null;
    const goal = status?.current_goal || null;
    const goalId = String(goal?.id || goal?.goal_id || "").trim();
    const fullGoal = goalId
        ? ((todosData?.todos || []).find((item) => item.id === goalId) || null)
        : null;
    const specialist = status?.specialist_resolution || null;
    const scheduler = status?.scheduler_selection || null;
    const approvalQueue = Array.isArray(status?.approval_queue) ? status.approval_queue : [];
    const pendingApproval = fullGoal?.blocked_action_id
        ? (approvalQueue.find((item) => item.id === fullGoal.blocked_action_id) || null)
        : (approvalQueue[0] || null);
    const research = status?.research_session || null;
    const currentTask = status?.current_task || null;
    const currentStep = status?.current_step || null;
    const lastCycle = status?.last_cycle_result || null;
    const decisionFlow = Array.isArray(systemData?.autonomy?.decision_flow) ? systemData.autonomy.decision_flow : [];
    const recentTransitions = Array.isArray(_panelState.transitions)
        ? _panelState.transitions.slice(-5).reverse()
        : [];
    const canArchive = Boolean(goalId) && fullGoal?.status !== "archived";
    const canResume = fullGoal?.status === "blocked_external" || fullGoal?.status === "blocked_by_user";
    const canUnblock = canResume;

    const schedulerDetails = scheduler?.details || {};
    const specialistBits = [
        specialist?.specialist_id || "",
        specialist?.reason || "",
        specialist?.quality_factor != null ? `quality ${Number(specialist.quality_factor).toFixed(2)}` : "",
    ].filter(Boolean);
    const schedulerBits = [
        scheduler?.mission_id || mission?.id || "",
        scheduler?.score != null ? `score ${Number(scheduler.score).toFixed(2)}` : "",
        schedulerDetails.routing_reason || status?.scheduler_reason || "",
    ].filter(Boolean);
    const riskBits = [
        pendingApproval?.risk_category ? `risk ${pendingApproval.risk_category}` : "",
        pendingApproval?.routing_pressure ? "routing pressure" : "",
        pendingApproval?.specialist ? `specialist ${pendingApproval.specialist}` : "",
        pendingApproval?.context?.target ? `target ${pendingApproval.context.target}` : "",
        pendingApproval?.context?.reason || pendingApproval?.reason || "",
    ].filter(Boolean);
    const evidenceBits = [
        research?.topic || "",
        Number.isFinite(Number(research?.accepted_sources_count)) ? `${research.accepted_sources_count} accepted sources` : "",
        Number.isFinite(Number(research?.findings_count)) ? `${research.findings_count} findings` : "",
        Number.isFinite(Number(research?.citation_coverage_rate)) ? `${Math.round(Number(research.citation_coverage_rate) * 100)}% cited` : "",
        Array.isArray(research?.knowledge_gaps) && research.knowledge_gaps.length ? `next gap ${research.knowledge_gaps[0]}` : "",
    ].filter(Boolean);
    const evidenceQualityRows = [
        [
            "Citation Coverage",
            Number.isFinite(Number(research?.citation_coverage_rate))
                ? `${Math.round(Number(research.citation_coverage_rate) * 100)}%`
                : "",
        ],
        [
            "Contradictions",
            Number.isFinite(Number(research?.contradictions_count))
                ? String(research.contradictions_count)
                : "",
        ],
        [
            "Accepted Sources",
            Number.isFinite(Number(research?.accepted_sources_count))
                ? String(research.accepted_sources_count)
                : "",
        ],
        [
            "Findings",
            Number.isFinite(Number(research?.findings_count))
                ? String(research.findings_count)
                : "",
        ],
        [
            "Routing Pressure",
            pendingApproval?.routing_pressure ? "active" : "",
        ],
        [
            "Specialist Quality",
            specialist?.quality_factor != null ? Number(specialist.quality_factor).toFixed(2) : "",
        ],
    ].filter(([, value]) => value !== "" && value !== null && value !== undefined);
    const evidenceBadges = [
        research?.citation_complete === false ? "partial citations" : "",
        Number(research?.contradictions_count || 0) > 0 ? `${research.contradictions_count} contradictions` : "",
        pendingApproval?.routing_pressure ? "routing pressure" : "",
        Array.isArray(research?.warnings) && research.warnings.length ? "review warnings" : "",
    ].filter(Boolean);
    const evidenceNote = String(research?.evidence_note || "").trim();
    const sourceDomainText = Array.isArray(research?.top_source_domains) && research.top_source_domains.length
        ? research.top_source_domains.join(", ")
        : "";
    const outcomeRows = [
        ["Expected Artifact", currentStep?.expected_artifact || ""],
        ["Current Task Output", currentTask?.expected_output || currentTask?.output || ""],
        ["Cycle Decision", lastCycle?.decision || ""],
        ["Cycle Reason", lastCycle?.reason || ""],
        ["Research Artifact", research?.artifact?.record_id || research?.final_artifact_id || ""],
    ].filter(([, value]) => String(value || "").trim());
    const producedArtifactLinks = [
        research?.artifact?.viewer_url
            ? `<a href="${esc(research.artifact.viewer_url)}" target="_blank" rel="noreferrer">Open report</a>`
            : "",
        research?.artifact?.pdf_url
            ? `<a href="${esc(research.artifact.pdf_url)}" target="_blank" rel="noreferrer">${esc(research.artifact.pdf_filename || "report.pdf")}</a>`
            : "",
        research?.artifact?.markdown_url && !research?.artifact?.viewer_url
            ? `<a href="${esc(research.artifact.markdown_url)}" target="_blank" rel="noreferrer">report.md</a>`
            : "",
    ].filter(Boolean);
    const blockerRows = [
        ["Status", fullGoal?.block_status || fullGoal?.status || ""],
        ["Blocked Reason", fullGoal?.blocked_reason || ""],
        ["Blocked Evidence", fullGoal?.blocked_evidence || ""],
        ["Resume Context", fullGoal?.resume_context || ""],
        ["Status Notes", fullGoal?.status_notes || ""],
    ].filter(([, value]) => String(value || "").trim());
    const executionBits = [
        status?.current_role || "",
        currentTask?.action || "",
        currentTask?.tool ? `tool ${currentTask.tool}` : "",
        currentStep?.instruction || "",
        currentStep?.expected_artifact ? `artifact ${currentStep.expected_artifact}` : "",
    ].filter(Boolean);
    const shareLines = [
        `Decision Dossier`,
        mission?.description || mission?.id ? `Mission: ${mission?.description || mission?.id}` : "",
        goalId ? `Goal ID: ${goalId}` : "",
        goal?.priority ? `Priority: ${goal.priority}` : "",
        specialistBits.length ? `Specialist: ${specialistBits.join(" | ")}` : "",
        schedulerBits.length ? `Selection: ${schedulerBits.join(" | ")}` : "",
        riskBits.length ? `Risk Gate: ${riskBits.join(" | ")}` : "",
        evidenceBits.length ? `Evidence: ${evidenceBits.join(" | ")}` : "",
        evidenceQualityRows.length
            ? `Evidence Quality: ${evidenceQualityRows.map(([label, value]) => `${label}=${value}`).join(" | ")}`
            : "",
        outcomeRows.length
            ? `Outcome: ${outcomeRows.map(([label, value]) => `${label}=${value}`).join(" | ")}`
            : "",
        blockerRows.length
            ? `Diagnostics: ${blockerRows.map(([label, value]) => `${label}=${value}`).join(" | ")}`
            : "",
        executionBits.length ? `Execution: ${executionBits.join(" | ")}` : "",
        recentTransitions.length
            ? `Recent Transitions: ${recentTransitions.map((item) => item.text || "transition").join(" -> ")}`
            : "",
    ].filter(Boolean);
    const currentSnapshotSections = {
        mission: [mission?.description || mission?.id || "", goalId, goal?.priority || "", schedulerBits.join(" | ")].join(" || "),
        specialist: specialistBits.join(" | "),
        risk_gate: riskBits.join(" | "),
        evidence: evidenceBits.join(" | "),
        evidence_quality: [
            evidenceQualityRows.map(([label, value]) => `${label}=${value}`).join(" | "),
            evidenceBadges.join(" | "),
            sourceDomainText,
            evidenceNote,
        ].filter(Boolean).join(" || "),
        outcome: [
            outcomeRows.map(([label, value]) => `${label}=${value}`).join(" | "),
            producedArtifactLinks.join(" | "),
        ].filter(Boolean).join(" || "),
        diagnostics: blockerRows.map(([label, value]) => `${label}=${value}`).join(" | "),
        execution: executionBits.join(" | "),
        transitions: recentTransitions.map((item) => `${item.text || ""}${item.detail ? ` | ${item.detail}` : ""}`).join(" -> "),
    };
    const shareSnapshot = shareLines.join("\n");
    const snapshotChangedSinceSave = Boolean(
        _lastSavedDecisionSnapshotContent
        && _lastSavedDecisionSnapshotContent !== shareSnapshot,
    );
    const changedSections = _lastSavedDecisionSnapshotSections
        ? Object.entries(currentSnapshotSections)
            .filter(([key, value]) => String(_lastSavedDecisionSnapshotSections?.[key] || "") !== String(value || ""))
            .map(([key]) => key.replace(/_/g, " "))
        : [];
    const changedSectionEntries = _lastSavedDecisionSnapshotSections
        ? Object.entries(currentSnapshotSections)
            .filter(([key, value]) => String(_lastSavedDecisionSnapshotSections?.[key] || "") !== String(value || ""))
            .map(([key, value]) => ({
                key,
                label: ({
                    mission: "Mission",
                    specialist: "Specialist",
                    risk_gate: "Risk Gate",
                    evidence: "Evidence",
                    evidence_quality: "Evidence Quality",
                    outcome: "Outcome",
                    diagnostics: "Diagnostics",
                    execution: "Execution",
                    transitions: "Recent Transitions",
                })[key] || key.replace(/_/g, " "),
                current: String(value || "").trim(),
                saved: String(_lastSavedDecisionSnapshotSections?.[key] || "").trim(),
            }))
        : [];
    const savedSnapshotRows = [
        ["Last Saved ID", _lastSavedDecisionSnapshotId || ""],
        [
            "Saved At",
            _lastSavedDecisionSnapshotAt
                ? new Date(_lastSavedDecisionSnapshotAt).toLocaleString()
                : "",
        ],
        [
            "Save Status",
            _lastSavedDecisionSnapshotId
                ? (snapshotChangedSinceSave ? "changed since save" : "up to date")
                : "",
        ],
        [
            "Changed Sections",
            changedSections.length ? changedSections.join(", ") : "",
        ],
        [
            "Pinned",
            _lastSavedDecisionSnapshotId ? (_lastSavedDecisionSnapshotPinned ? "yes" : "no") : "",
        ],
    ].filter(([, value]) => String(value || "").trim());

    contentEl.innerHTML = `
        <div class="goal-detail">
            ${(goalId || pendingApproval) ? `
            <div class="panel-actions goal-detail-actions">
                <button class="btn-outline" data-dossier-action="copy-snapshot">Copy Snapshot</button>
                <button class="btn-outline" data-dossier-action="save-snapshot">${_lastSavedDecisionSnapshotId ? (snapshotChangedSinceSave ? "Save Updated" : "Saved") : "Save Snapshot"}</button>
                ${_lastSavedDecisionSnapshotId ? '<button class="btn-outline" data-dossier-action="open-saved">Open Saved</button>' : ""}
                ${_lastSavedDecisionSnapshotId ? `<button class="btn-outline" data-dossier-action="toggle-pin-saved">${_lastSavedDecisionSnapshotPinned ? "Unpin Saved" : "Pin Saved"}</button>` : ""}
                ${_lastSavedDecisionSnapshotContent ? `<button class="btn-outline" data-dossier-action="toggle-last-saved">${_showLastSavedDecisionSnapshot ? "Hide Last Saved" : "View Last Saved"}</button>` : ""}
                <button class="btn-outline" data-dossier-action="view-saved-snapshots">View Saved Snapshots</button>
                <button class="btn-outline" data-dossier-action="view-pinned-snapshots">View Pinned Snapshots</button>
                ${goalId ? '<button class="btn-outline" data-dossier-action="open-task">Open task detail</button>' : ""}
                ${canArchive ? '<button class="btn-outline" data-dossier-action="archive">Archive</button>' : ""}
                ${canUnblock ? '<button class="btn-outline" data-dossier-action="unblock">Unblock</button>' : ""}
                ${canResume ? '<button class="btn-primary" data-dossier-action="resume">Resume</button>' : ""}
                ${pendingApproval ? '<button class="btn-approve" data-dossier-action="approve-linked">Approve</button>' : ""}
                ${pendingApproval ? '<button class="btn-deny" data-dossier-action="deny-linked">Deny</button>' : ""}
            </div>
            ` : ""}
            <div class="goal-detail-section">
                <h4>Selected Mission</h4>
                <div class="goal-detail-meta">
                    <span>${esc(mission?.description || mission?.id || "No active mission")}</span>
                    ${goal?.priority ? `<span>${esc(`priority ${goal.priority}`)}</span>` : ""}
                    ${mission?.active_tasks != null ? `<span>${esc(`${mission.active_tasks} active tasks`)}</span>` : ""}
                </div>
                ${schedulerBits.length ? `<div class="activity-queue-detail">${esc(schedulerBits.join(" | "))}</div>` : ""}
                ${goalId ? `<div class="activity-queue-detail">${esc(goalId)}</div>` : ""}
            </div>

            <div class="goal-detail-section">
                <h4>Chosen Specialist</h4>
                <div class="goal-detail-body">${esc(specialistBits.join(" | ") || "No specialist selected")}</div>
                ${schedulerDetails.preferred_specialist || schedulerDetails.degraded_specialist
                    ? `<div class="activity-queue-detail">${esc([
                        schedulerDetails.preferred_specialist ? `preferred ${schedulerDetails.preferred_specialist}` : "",
                        schedulerDetails.degraded_specialist ? `degraded ${schedulerDetails.degraded_specialist}` : "",
                    ].filter(Boolean).join(" | "))}</div>`
                    : ""}
            </div>

            <div class="goal-detail-section">
                <h4>Risk Gate</h4>
                <div class="goal-detail-body">${esc(riskBits.join(" | ") || (lastCycle?.decision ? `${lastCycle.decision}${lastCycle.reason ? ` | ${lastCycle.reason}` : ""}` : "No active risk gate"))}</div>
            </div>

            <div class="goal-detail-section">
                <h4>Approval Context</h4>
                <div class="goal-detail-body">${esc(pendingApproval?.description || pendingApproval?.action || "No pending approval")}</div>
                ${pendingApproval ? `<div class="activity-queue-detail">${esc(riskBits.join(" | "))}</div>` : ""}
            </div>

            <div class="goal-detail-section">
                <h4>Evidence / Research Summary</h4>
                <div class="goal-detail-body">${esc(evidenceBits.join(" | ") || "No active research evidence")}</div>
                ${research?.artifact?.viewer_url || research?.artifact?.pdf_url
                    ? `<div class="goal-detail-meta">
                        ${research?.artifact?.viewer_url ? `<a href="${esc(research.artifact.viewer_url)}" target="_blank" rel="noreferrer">Open report</a>` : ""}
                        ${research?.artifact?.pdf_url ? `<a href="${esc(research.artifact.pdf_url)}" target="_blank" rel="noreferrer">${esc(research.artifact.pdf_filename || "report.pdf")}</a>` : ""}
                    </div>`
                    : ""}
            </div>

            ${(evidenceQualityRows.length || evidenceBadges.length || evidenceNote || sourceDomainText) ? `
            <div class="goal-detail-section">
                <h4>Evidence Quality</h4>
                ${evidenceQualityRows.length ? `
                    <div class="goal-detail-grid">
                        ${evidenceQualityRows.map(([label, value]) => `
                            <div class="goal-detail-metric">
                                <div class="goal-detail-metric-label">${esc(label)}</div>
                                <div class="goal-detail-metric-value">${esc(value)}</div>
                            </div>
                        `).join("")}
                    </div>
                ` : ""}
                ${evidenceBadges.length ? `
                    <div class="goal-detail-meta">
                        ${evidenceBadges.map((badge) => `<span>${esc(badge)}</span>`).join("")}
                    </div>
                ` : ""}
                ${sourceDomainText ? `<div class="activity-queue-detail">domains ${esc(sourceDomainText)}</div>` : ""}
                ${evidenceNote ? `<div class="goal-detail-alert goal-detail-alert-info">${esc(evidenceNote)}</div>` : ""}
            </div>
            ` : ""}

            ${savedSnapshotRows.length ? `
            <div class="goal-detail-section">
                <h4>Saved Snapshot</h4>
                <div class="goal-detail-grid">
                    ${savedSnapshotRows.map(([label, value]) => `
                        <div class="goal-detail-metric">
                            <div class="goal-detail-metric-label">${esc(label)}</div>
                            <div class="goal-detail-metric-value">${esc(value)}</div>
                        </div>
                    `).join("")}
                </div>
            </div>
            ` : ""}

            ${_showLastSavedDecisionSnapshot && _lastSavedDecisionSnapshotContent ? `
            <div class="goal-detail-section">
                <h4>Last Saved Compare</h4>
                ${changedSectionEntries.length ? `
                    <div class="system-eval-list">
                        ${changedSectionEntries.map((section) => `
                            <div class="system-eval-item">
                                <strong>${esc(section.label)}</strong>
                                <span>Current: ${esc(section.current || "—")}</span>
                                <span>Last saved: ${esc(section.saved || "—")}</span>
                            </div>
                        `).join("")}
                    </div>
                ` : `
                    <div class="goal-detail-alert goal-detail-alert-info">Current dossier matches the last saved snapshot.</div>
                `}
            </div>
            ` : ""}

            ${(outcomeRows.length || producedArtifactLinks.length) ? `
            <div class="goal-detail-section">
                <h4>Outcome / Expected Artifact</h4>
                ${outcomeRows.length ? `
                    <div class="goal-detail-grid">
                        ${outcomeRows.map(([label, value]) => `
                            <div class="goal-detail-metric">
                                <div class="goal-detail-metric-label">${esc(label)}</div>
                                <div class="goal-detail-metric-value">${esc(value)}</div>
                            </div>
                        `).join("")}
                    </div>
                ` : ""}
                ${producedArtifactLinks.length ? `
                    <div class="goal-detail-meta">
                        ${producedArtifactLinks.join("")}
                    </div>
                ` : ""}
            </div>
            ` : ""}

            ${blockerRows.length ? `
            <div class="goal-detail-section">
                <h4>Failure / Blocker Diagnostics</h4>
                <div class="goal-detail-grid">
                    ${blockerRows.map(([label, value]) => `
                        <div class="goal-detail-metric">
                            <div class="goal-detail-metric-label">${esc(label)}</div>
                            <div class="goal-detail-metric-value">${esc(value)}</div>
                        </div>
                    `).join("")}
                </div>
                ${fullGoal?.blocked_reason ? `<div class="goal-detail-alert goal-detail-alert-danger">${esc(fullGoal.blocked_reason)}</div>` : ""}
                ${fullGoal?.resume_context ? `<div class="goal-detail-alert goal-detail-alert-info">Resume: ${esc(fullGoal.resume_context)}</div>` : ""}
                ${fullGoal?.blocked_evidence ? `<div class="goal-detail-alert goal-detail-alert-info">Evidence: ${esc(fullGoal.blocked_evidence)}</div>` : ""}
            </div>
            ` : ""}

            <div class="goal-detail-section">
                <h4>Current Execution Step</h4>
                <div class="goal-detail-body">${esc(executionBits.join(" | ") || "No active execution step")}</div>
            </div>

            ${recentTransitions.length ? `
            <div class="goal-detail-section">
                <h4>Recent Transitions</h4>
                <div class="system-eval-list">
                    ${recentTransitions.map((item) => {
                        const rawTs = item.timestamp || Date.now() / 1000;
                        const ts = typeof rawTs === "number" ? rawTs : Date.parse(rawTs) / 1000;
                        const time = Number.isFinite(ts) ? new Date(ts * 1000).toLocaleTimeString() : "";
                        return `
                            <div class="system-eval-item">
                                <strong>${esc(item.text || "Transition")}</strong>
                                <span>${esc(time)}${item.detail ? ` | ${esc(item.detail)}` : ""}</span>
                            </div>
                        `;
                    }).join("")}
                </div>
            </div>` : ""}

            ${decisionFlow.length ? `
            <div class="goal-detail-section">
                <h4>Decision Flow</h4>
                <div class="system-eval-list">
                    ${decisionFlow.map((item) => `
                        <div class="system-eval-item">
                            <strong>${esc(item.title || item.stage || "Decision")}</strong>
                            <span>${esc(item.summary || "")}${item.detail ? ` | ${esc(item.detail)}` : ""}</span>
                        </div>
                    `).join("")}
                </div>
            </div>` : ""}
        </div>
    `;

    const actionButtons = contentEl.querySelectorAll("[data-dossier-action]");
    actionButtons.forEach((button) => {
        button.addEventListener("click", async () => {
            const action = button.dataset.dossierAction || "";
            const originalText = button.textContent || "";
            const buttons = Array.from(actionButtons);
            buttons.forEach((entry) => { entry.disabled = true; });
            button.textContent = "Working...";
            try {
                if (action === "copy-snapshot") {
                    if (navigator.clipboard?.writeText) {
                        await navigator.clipboard.writeText(shareSnapshot);
                    } else {
                        const textArea = document.createElement("textarea");
                        textArea.value = shareSnapshot;
                        textArea.style.position = "fixed";
                        textArea.style.opacity = "0";
                        document.body.appendChild(textArea);
                        textArea.focus();
                        textArea.select();
                        document.execCommand("copy");
                        textArea.remove();
                    }
                    button.textContent = "Copied";
                    window.setTimeout(() => {
                        button.textContent = originalText;
                        buttons.forEach((entry) => { entry.disabled = false; });
                    }, 800);
                    return;
                }
                if (action === "save-snapshot") {
                    if (_lastSavedDecisionSnapshotContent && _lastSavedDecisionSnapshotContent === shareSnapshot) {
                        button.textContent = "Up to date";
                        window.setTimeout(() => {
                            button.textContent = originalText;
                            buttons.forEach((entry) => { entry.disabled = false; });
                        }, 800);
                        return;
                    }
                    const created = await window.apiClient.createRecord({
                        content: shareSnapshot,
                        tags: [
                            "operator",
                            "decision_dossier",
                            "review",
                            goalId ? "active_goal" : "",
                            pendingApproval ? "approval_context" : "",
                            pendingApproval?.routing_pressure ? "routing_pressure" : "",
                        ].filter(Boolean).join(", "),
                        level: "decisions",
                    });
                    _lastSavedDecisionSnapshotId = String(created?.id || created?.record_id || "").trim();
                    _lastSavedDecisionSnapshotAt = Date.now();
                    _lastSavedDecisionSnapshotContent = shareSnapshot;
                    _lastSavedDecisionSnapshotSections = { ...currentSnapshotSections };
                    _lastSavedDecisionSnapshotPinned = false;
                    button.textContent = "Saved";
                    window.setTimeout(() => {
                        button.textContent = originalText;
                        buttons.forEach((entry) => { entry.disabled = false; });
                    }, 900);
                    return;
                }
                if (action === "open-saved" && _lastSavedDecisionSnapshotId) {
                    await openDecisionSnapshotInMemory(_lastSavedDecisionSnapshotId);
                    return;
                }
                if (action === "toggle-pin-saved" && _lastSavedDecisionSnapshotId) {
                    const record = await window.apiClient.getRecord(_lastSavedDecisionSnapshotId);
                    const tags = Array.isArray(record?.tags) ? record.tags.slice() : [];
                    const nextTags = _lastSavedDecisionSnapshotPinned
                        ? tags.filter((tag) => tag !== "pinned_snapshot")
                        : Array.from(new Set(tags.concat(["pinned_snapshot"])));
                    await window.apiClient.updateRecord(_lastSavedDecisionSnapshotId, {
                        tags: nextTags.join(", "),
                    });
                    _lastSavedDecisionSnapshotPinned = !_lastSavedDecisionSnapshotPinned;
                    await openDecisionDossier();
                    return;
                }
                if (action === "toggle-last-saved") {
                    _showLastSavedDecisionSnapshot = !_showLastSavedDecisionSnapshot;
                    await openDecisionDossier();
                    return;
                }
                if (action === "view-saved-snapshots") {
                    await openDecisionSnapshotInMemory("");
                    return;
                }
                if (action === "view-pinned-snapshots") {
                    await openDecisionSnapshotInMemory("", "decision_dossier,pinned_snapshot");
                    return;
                }
                if (action === "open-task" && goalId) {
                    await openMissionQueueDetail(goalId);
                    return;
                }
                if (action === "archive" && goalId) {
                    await window.apiClient.archiveAutonomyGoal(goalId, "archived_from_decision_dossier");
                } else if (action === "unblock" && goalId) {
                    await window.apiClient.unblockAutonomyGoal(goalId);
                } else if (action === "resume" && goalId) {
                    const note = window.prompt("Resume note", fullGoal?.resume_context || "") || "";
                    await window.apiClient.resumeAutonomyGoal(goalId, note);
                } else if (action === "approve-linked" && pendingApproval?.id) {
                    await window.apiClient.approveAction(pendingApproval.id);
                } else if (action === "deny-linked" && pendingApproval?.id) {
                    await window.apiClient.rejectAction(pendingApproval.id);
                }
                await refreshAutonomyStatus();
                await openDecisionDossier();
            } catch (err) {
                console.error(`Failed to ${action} from decision dossier`, err);
                window.alert(`Failed to ${action.replace("-", " ")}.`);
                button.textContent = originalText;
                buttons.forEach((entry) => { entry.disabled = false; });
            }
        });
    });
}

function applyAutonomySnapshot(status) {
    _snapshotState.autonomyRunning = Boolean(status.running);
    _snapshotState.transportConnected = Boolean(status.transport_connected);
    _snapshotState.currentGoal = status.current_goal || null;
    _snapshotState.currentMission = status.current_mission || null;
    _snapshotState.currentTask = status.current_task || null;
    _snapshotState.currentStep = status.current_step || null;
    _snapshotState.lastCycleResult = status.last_cycle_result || null;
    _snapshotState.currentRole = status.current_role || "";
    _snapshotState.lastAgentResponse = status.last_agent_response || null;
    _snapshotState.lastResearchActivity = status.last_research_activity || null;
    _snapshotState.researchSession = status.research_session || null;
    _snapshotState.schedulerReason = status.scheduler_reason || "";
    _snapshotState.schedulerSelection = status.scheduler_selection || null;
    _snapshotState.specialistResolution = status.specialist_resolution || null;
    _snapshotState.stuckMissions = status.stuck_missions || [];
    _snapshotState.approvalQueue = status.approval_queue || [];
    _snapshotState.qualityDebt = status.quality_debt_by_specialist || [];
    _snapshotState.decisionFlow = Array.isArray(status.decision_flow) ? status.decision_flow : [];
    _snapshotState.activeHarnessModule = status.active_harness_module || null;

    if (liveStatusEl) {
        liveStatusEl.classList.remove("connected", "disconnected", "reconnecting");
        const textEl = liveStatusEl.querySelector(".status-text");
        if (status.transport_connected) {
            liveStatusEl.classList.add("connected");
            textEl.textContent = "Stream Live";
        } else {
            liveStatusEl.classList.add("disconnected");
            textEl.textContent = "Stream Disconnected";
        }
    }

    renderPanels();
}

function panelText(event) {
    const currentEventName = eventName(event);
    switch (currentEventName) {
        case "goal_selected":
            return `Goal [${eventField(event, "priority", "")}] ${eventField(event, "description", "")}`.trim();
        case "thinking":
            return eventField(event, "summary", "Thinking");
        case "plan_step":
            return `Step ${eventField(event, "step_num", "?")}/${eventField(event, "total_steps", "?")}: ${eventField(event, "step_description", "")}`;
        case "tool_call":
            return `${eventField(event, "tool", "tool")}(${eventField(event, "args_summary", "")})`;
        case "worker_role_resolution": {
            const requested = eventField(event, "requested_role", "");
            const resolved = eventField(event, "resolved_role", "");
            const status = eventField(event, "status", "");
            if (status === "unknown_role") {
                return `Worker role failed: ${requested}`;
            }
            return `Worker role: ${requested}${resolved && resolved !== requested ? ` -> ${resolved}` : ""}`;
        }
        case "worker_started":
            return `Worker started: ${eventField(event, "worker_channel", eventField(event, "role", "worker"))}`;
        case "worker_completed":
            return `Worker ${eventField(event, "role", "worker")} ${eventField(event, "status", "completed")}`;
        case "tool_result":
            return eventField(event, "is_error", false) ? `Error: ${eventField(event, "result", "")}` : (eventField(event, "result", "Tool completed"));
        case "evaluation":
            return eventField(event, "reason", "Evaluation complete");
        case "outcome":
            return eventField(event, "description", "Action complete");
        case "cycle_start":
            return `Cycle ${eventField(event, "cycle", "?")}`;
        case "cycle_end":
            return `Cycle ${eventField(event, "cycle", "?")} complete`;
        case "approval.pending":
        case "approval_pending":
            return eventField(event, "description", eventField(event, "action", "Pending approval"));
        case "approval.resolved":
        case "approval_resolved":
            return `Approval ${eventField(event, "decision", "updated")}: ${eventField(event, "description", eventField(event, "action", ""))}`.trim();
        default:
            return currentEventName || "event";
    }
}

// ============== Budget Bar ==============

function updateBudgetBar(b) {
    if (!b) return;

    // Normalize v2 (tokens-based) and v3 (USD-based) budget formats into one view.
    // v2 keys: tokens_today, daily_limit, tokens_this_hour, hourly_limit,
    //          tokens_this_session, session_limit, total_tokens_lifetime
    // v3 keys: daily_tokens, daily_limit_usd, daily_spent_usd, cycle_spent_usd,
    //          cycle_limit_usd, wallet_usd, status, seed_period, ...
    const isV3 = ("daily_tokens" in b || "daily_spent_usd" in b) && !("tokens_today" in b);

    let dailyUsed, dailyMax, hourlyUsed, hourlyMax, sessionUsed, sessionMax, lifetimeText;

    if (isV3) {
        // Show USD-based values for v3 budget
        dailyUsed  = b.daily_spent_usd  != null ? b.daily_spent_usd  : 0;
        dailyMax   = b.daily_limit_usd  != null ? b.daily_limit_usd  : 1;
        hourlyUsed = b.cycle_spent_usd  != null ? b.cycle_spent_usd  : 0;
        hourlyMax  = b.cycle_limit_usd  != null ? b.cycle_limit_usd  : 1;
        sessionUsed = dailyUsed;
        sessionMax  = dailyMax;
        const wallet = b.wallet_usd != null ? `$${b.wallet_usd.toFixed(2)}` : "--";
        const runway = b.runway_days != null ? ` ${b.runway_days}d runway` : "";
        lifetimeText = wallet + runway;
        // Override text elements to show USD instead of tokens
        document.getElementById("budget-daily-text").textContent =
            `$${dailyUsed.toFixed(4)} / $${dailyMax.toFixed(2)}`;
        document.getElementById("budget-hourly-text").textContent =
            `$${hourlyUsed.toFixed(4)} / $${hourlyMax.toFixed(3)}`;
        document.getElementById("budget-session-text").textContent =
            `${b.daily_tokens != null ? formatTokens(b.daily_tokens) : "--"} tokens today`;
        document.getElementById("budget-lifetime-text").textContent = lifetimeText;
    } else {
        // v2 token-based
        dailyUsed   = b.tokens_today        ?? 0;
        dailyMax    = b.daily_limit         ?? 1;
        hourlyUsed  = b.tokens_this_hour    ?? 0;
        hourlyMax   = b.hourly_limit        ?? 1;
        sessionUsed = b.tokens_this_session ?? 0;
        sessionMax  = b.session_limit       ?? 1;
        document.getElementById("budget-daily-text").textContent =
            `${formatTokens(dailyUsed)} / ${formatTokens(dailyMax)}`;
        document.getElementById("budget-hourly-text").textContent =
            `${formatTokens(hourlyUsed)} / ${formatTokens(hourlyMax)}`;
        document.getElementById("budget-session-text").textContent =
            `${formatTokens(sessionUsed)} / ${formatTokens(sessionMax)}`;
        document.getElementById("budget-lifetime-text").textContent =
            formatTokens(b.total_tokens_lifetime ?? 0) + " total";
    }

    const dailyPct   = Math.min(100, dailyMax   > 0 ? (dailyUsed   / dailyMax)   * 100 : 0);
    const hourlyPct  = Math.min(100, hourlyMax  > 0 ? (hourlyUsed  / hourlyMax)  * 100 : 0);
    const sessionPct = Math.min(100, sessionMax > 0 ? (sessionUsed / sessionMax) * 100 : 0);

    const dailyFill   = document.getElementById("budget-daily-fill");
    const hourlyFill  = document.getElementById("budget-hourly-fill");
    const sessionFill = document.getElementById("budget-session-fill");

    dailyFill.style.width   = dailyPct   + "%";
    dailyFill.className   = "budget-fill" + (dailyPct   > 80 ? " budget-danger" : dailyPct   > 50 ? " budget-warn" : "");
    hourlyFill.style.width  = hourlyPct  + "%";
    hourlyFill.className  = "budget-fill" + (hourlyPct  > 80 ? " budget-danger" : hourlyPct  > 50 ? " budget-warn" : "");
    sessionFill.style.width = sessionPct + "%";
    sessionFill.className = "budget-fill" + (sessionPct > 80 ? " budget-danger" : sessionPct > 50 ? " budget-warn" : "");
}

function renderStreamEvent(event) {
    const div = document.createElement("div");
    const currentEventName = eventName(event) || "event";
    div.className = `stream-event stream-event-${eventCssName(event) || "event"}`;

    const rawTimestamp = eventField(event, "timestamp", Date.now() / 1000);
    const timestamp = typeof rawTimestamp === "number" ? rawTimestamp : Date.parse(rawTimestamp) / 1000;
    const time = Number.isFinite(timestamp) ? new Date(timestamp * 1000).toLocaleTimeString() : "";

    switch (currentEventName) {
        case "cycle_start":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#9654;</span>
                <span class="stream-label">Cycle ${eventField(event, "cycle", "?")}</span>`;
            break;
        case "goal_selected":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#127919;</span>
                <span class="stream-label">Goal [${esc(eventField(event, "priority", ""))}]: ${esc(eventField(event, "description", ""))}</span>`;
            break;
        case "plan_step":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#128221;</span>
                <span class="stream-label">Step ${eventField(event, "step_num", "?")}/${eventField(event, "total_steps", "?")}: ${esc(eventField(event, "step_description", ""))}</span>`;
            break;
        case "thinking":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#129504;</span>
                <span class="stream-label">${esc(eventField(event, "summary", ""))}</span>`;
            break;
        case "agent_response":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#128172;</span>
                <span class="stream-label">${esc(eventField(event, "response", ""))}</span>
                <span class="stream-meta">${eventField(event, "duration_ms", 0)}ms | ~${eventField(event, "tokens_estimated", 0)} tok</span>`;
            break;
        case "evaluation": {
            const success = Boolean(eventField(event, "success", false));
            const evalIcon = success ? "&#10003;" : "&#10007;";
            const evalCls = success ? "stream-success" : "stream-failure";
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon ${evalCls}">${evalIcon}</span>
                <span class="stream-label">${esc(eventField(event, "reason", ""))}</span>
                <span class="stream-meta">${Math.round(Number(eventField(event, "confidence", 0)) * 100)}% confidence</span>`;
            break;
        }
        case "outcome": {
            const success = Boolean(eventField(event, "success", false));
            const outIcon = success ? "&#10003;" : "&#10007;";
            const outCls = success ? "stream-success" : "stream-failure";
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon ${outCls}">${outIcon}</span>
                <span class="stream-label">${esc(eventField(event, "description", ""))}</span>
                <span class="stream-meta">${eventField(event, "tokens_used", 0)} tok | ${eventField(event, "duration_ms", 0)}ms</span>`;
            break;
        }
        case "cycle_end":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#9632;</span>
                <span class="stream-label">Cycle ${eventField(event, "cycle", "?")} complete</span>`;
            break;
        case "reflection":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#128161;</span>
                <span class="stream-label">${esc(eventField(event, "content", ""))}</span>`;
            break;
        case "quiet_hours":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#127769;</span>
                <span class="stream-label">Quiet hours (${eventField(event, "start", "?")}:00\u2013${eventField(event, "end", "?")}:00)</span>`;
            break;
        case "budget_warning":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#9888;</span>
                <span class="stream-label">Budget: ${esc(eventField(event, "reason", ""))}</span>`;
            break;
        case "consolidation_start":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#128230;</span>
                <span class="stream-label">Consolidation: ${eventField(event, "clusters_found", 0)} clusters found</span>`;
            break;
        case "consolidation_merged":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#128279;</span>
                <span class="stream-label">Merged ${eventField(event, "records_merged", 0)} records [${esc(eventField(event, "tags", ""))}]</span>`;
            break;
        case "consolidation_end":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#9989;</span>
                <span class="stream-label">Consolidation done: ${eventField(event, "clusters_processed", 0)} clusters, ${eventField(event, "total_records_merged", 0)} records merged</span>`;
            break;
        case "goal_failed": {
            const who = eventField(event, "created_by", "") === "user" ? " (user)" : "";
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon stream-failure">&#10060;</span>
                <span class="stream-label">Goal failed${who} after ${eventField(event, "attempts", 0)} attempts: ${esc(eventField(event, "description", ""))}</span>
                <span class="stream-meta">${esc(eventField(event, "reason", ""))}</span>`;
            break;
        }
        case "tool_call":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon stream-tool">&#9881;</span>
                <span class="stream-label"><strong>${esc(eventField(event, "tool", ""))}</strong>(${esc(eventField(event, "args_summary", ""))})</span>`;
            break;
        case "worker_role_resolution": {
            const requested = esc(eventField(event, "requested_role", ""));
            const resolved = esc(eventField(event, "resolved_role", ""));
            const status = eventField(event, "status", "");
            const workerChannel = esc(eventField(event, "worker_channel", ""));
            const timeoutSec = esc(String(eventField(event, "timeout_sec", "")));
            const stepBudget = esc(String(eventField(event, "step_budget", "")));
            if (status === "unknown_role") {
                div.innerHTML = `<span class="stream-time">${time}</span>
                    <span class="stream-icon stream-failure">&#9888;</span>
                    <span class="stream-label">Worker role resolution failed: <strong>${requested}</strong></span>`;
            } else {
                const meta = [workerChannel, timeoutSec ? `${timeoutSec}s` : "", stepBudget ? `${stepBudget} steps` : ""]
                    .filter(Boolean)
                    .join(" | ");
                div.innerHTML = `<span class="stream-time">${time}</span>
                    <span class="stream-icon stream-worker">&#129302;</span>
                    <span class="stream-label">Worker role: <strong>${requested}</strong>${resolved && resolved !== requested ? ` &rarr; <strong>${resolved}</strong>` : ""}</span>
                    <span class="stream-meta">${meta}</span>`;
            }
            break;
        }
        case "worker_started":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon stream-worker">&#9654;</span>
                <span class="stream-label">Started <strong>${esc(eventField(event, "worker_channel", eventField(event, "role", "worker")))}</strong></span>
                <span class="stream-meta">${esc(String(eventField(event, "timeout_sec", "")))}s | ${esc(String(eventField(event, "step_budget", "")))} steps</span>`;
            break;
        case "worker_completed": {
            const status = esc(eventField(event, "status", "completed"));
            const role = esc(eventField(event, "role", "worker"));
            const elapsed = esc(String(eventField(event, "elapsed_sec", "")));
            const toolCalls = esc(String(eventField(event, "tool_calls", "")));
            const doneCls = status === "success" ? "stream-success" : status === "error" ? "stream-failure" : "stream-tool";
            const doneIcon = status === "success" ? "&#10003;" : status === "error" ? "&#10007;" : "&#9203;";
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon ${doneCls}">${doneIcon}</span>
                <span class="stream-label">Worker <strong>${role}</strong> ${status}</span>
                <span class="stream-meta">${elapsed}s | ${toolCalls} calls</span>`;
            break;
        }
        case "tool_result": {
            const isError = Boolean(eventField(event, "is_error", false));
            const trIcon = isError ? "&#10007;" : "&#10003;";
            const trCls = isError ? "stream-failure" : "stream-tool-ok";
            // Try to format JSON results nicely
            let resultText = eventField(event, "result", "");
            try {
                const parsed = JSON.parse(resultText);
                // Show key fields for common tool results
                if (parsed.findings) resultText = parsed.findings.substring(0, 600);
                else if (parsed.answer) resultText = parsed.answer.substring(0, 600);
                else if (parsed.content) resultText = parsed.content.substring(0, 600);
                else if (parsed.report) resultText = typeof parsed.report === "string" ? parsed.report.substring(0, 600) : resultText.substring(0, 400);
                else resultText = resultText.substring(0, 400);
            } catch { resultText = resultText.substring(0, 400); }
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon ${trCls}">${trIcon}</span>
                <span class="stream-label stream-result-text">${esc(resultText)}</span>`;
            break;
        }
        case "approval.pending":
        case "approval_pending":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#9203;</span>
                <span class="stream-label">Approval needed: ${esc(eventField(event, "description", eventField(event, "action", "Pending approval")))}</span>`;
            break;
        case "approval.resolved":
        case "approval_resolved":
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#9989;</span>
                <span class="stream-label">Approval ${esc(eventField(event, "decision", "updated"))}: ${esc(eventField(event, "description", eventField(event, "action", "")))}</span>`;
            break;
        default:
            div.innerHTML = `<span class="stream-time">${time}</span>
                <span class="stream-icon">&#8226;</span>
                <span class="stream-label">${esc(currentEventName)}</span>`;
    }

    return div;
}

// ============== Public Entry Point ==============

export async function loadActivity() {
    await refreshAutonomyStatus();
    renderPanels();
    if (_liveMode) {
        window.apiClient.connectActivity();
        startSnapshotPolling();
        return;
    }
    loadHistoryData();
}

// ============== History Mode ==============

async function loadHistoryData() {
    summaryEl.innerHTML = "";
    listEl.innerHTML = skeletonCards(4);
    try {
        const res = await fetch("/api/activity");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        _cachedData = await res.json();
        renderSummary(_cachedData.summary);
        renderTimeline(_cachedData, filterEl.value);
    } catch (e) {
        summaryEl.innerHTML = "";
        listEl.innerHTML = errorState(e.message);
    }
}

// ============== Summary Cards ==============

function renderSummary(s) {
    if (!s) { summaryEl.innerHTML = ""; return; }

    summaryEl.innerHTML = `
        <div class="stat-card">
            <div class="stat-card-title">Total Actions</div>
            <div class="stat-card-value">${s.total_actions}</div>
            <div class="stat-card-detail">${s.success} success / ${s.failure} failed</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Success Rate</div>
            <div class="stat-card-value">${s.success_rate}%</div>
            <div class="stat-card-detail">${s.total_actions} actions total</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Tokens Used</div>
            <div class="stat-card-value">${formatTokens(s.total_tokens)}</div>
            <div class="stat-card-detail">across all autonomous actions</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Goals</div>
            <div class="stat-card-value">${s.active_goals}</div>
            <div class="stat-card-detail">${s.completed_goals} completed</div>
        </div>
    `;
}

function formatTokens(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1) + "K";
    return String(n);
}

// ============== Timeline ==============

function renderTimeline(data, filter) {
    let items = [];

    if (filter === "all" || filter === "outcomes") {
        items.push(...(data.outcomes || []).map((o) => ({ ...o, _sort: o.timestamp })));
    }
    if (filter === "all" || filter === "goals") {
        items.push(...(data.goals || []).map((g) => ({ ...g, _sort: g.timestamp || g.created_at })));
    }
    if (filter === "all" || filter === "reflections") {
        items.push(...(data.reflections || []).map((r) => ({ ...r, _sort: r.timestamp })));
    }
    if (filter === "all" || filter === "proactive") {
        items.push(...(data.proactive || []).map((p) => ({ ...p, _sort: p.timestamp })));
    }

    items.sort((a, b) => (b._sort || "").localeCompare(a._sort || ""));

    if (items.length === 0) {
        listEl.innerHTML = EMPTY.activity();
        return;
    }

    listEl.innerHTML = items.map(renderItem).join("");

    listEl.querySelectorAll(".activity-item").forEach((el) => {
        el.addEventListener("click", () => {
            el.classList.toggle("expanded");
        });
    });
}

function renderItem(item) {
    switch (item.type) {
        case "outcome": return renderOutcome(item);
        case "goal": return renderGoal(item);
        case "reflection": return renderReflection(item);
        case "proactive": return renderProactive(item);
        default: return "";
    }
}

function renderOutcome(o) {
    const icon = o.success ? "&#10003;" : "&#10007;";
    const cls = o.success ? "activity-success" : "activity-failure";
    const dur = o.duration_ms ? `${(o.duration_ms / 1000).toFixed(1)}s` : "";
    const tok = o.tokens_used ? `${o.tokens_used} tok` : "";

    return `
        <div class="memory-card activity-item" data-id="${esc(o.id)}">
            <div class="memory-card-header">
                <span>
                    <span class="${cls}" style="font-weight:700;margin-right:6px;">${icon}</span>
                    <span class="activity-badge activity-badge-outcome">action</span>
                    <span class="activity-badge">${esc(o.action_type)}</span>
                </span>
                <span class="activity-ts">${formatTime(o.timestamp)}</span>
            </div>
            <div class="memory-card-content">${esc(o.content)}</div>
            <div class="memory-card-meta">
                ${tok ? `<span class="tag">${tok}</span>` : ""}
                ${dur ? `<span class="tag">${dur}</span>` : ""}
                ${o.goal_id ? `<span class="tag">goal: ${esc(o.goal_id.slice(0, 12))}</span>` : ""}
            </div>
        </div>
    `;
}

function renderGoal(g) {
    const statusCls = g.status === "completed" ? "activity-success"
        : g.status === "failed" ? "activity-failure" : "";
    const priorityMap = { critical: "!!!", high: "!!", medium: "!", low: "" };

    return `
        <div class="memory-card activity-item" data-id="${esc(g.id)}">
            <div class="memory-card-header">
                <span>
                    <span class="activity-badge activity-badge-goal">goal</span>
                    <span class="activity-badge">${esc(g.priority)}</span>
                    <span class="${statusCls}" style="margin-left:6px;">${esc(g.status)}</span>
                </span>
                <span class="activity-ts">${formatTime(g.timestamp || g.created_at)}</span>
            </div>
            <div class="memory-card-content">${esc(g.content)}</div>
            <div class="memory-card-meta">
                <span class="tag">${g.attempts} attempts</span>
                ${priorityMap[g.priority] ? `<span class="tag">${priorityMap[g.priority]} ${esc(g.priority)}</span>` : ""}
            </div>
        </div>
    `;
}

function renderReflection(r) {
    return `
        <div class="memory-card activity-item" data-id="${esc(r.id)}">
            <div class="memory-card-header">
                <span>
                    <span class="activity-badge activity-badge-reflection">reflection</span>
                    <span class="tag">${r.action_count} actions</span>
                </span>
                <span class="activity-ts">${formatTime(r.timestamp)}</span>
            </div>
            <div class="memory-card-content">${esc(r.content)}</div>
            <div class="memory-card-meta">
                <span class="tag">session: ${esc(r.session_id.slice(0, 16))}</span>
            </div>
        </div>
    `;
}

function renderProactive(p) {
    return `
        <div class="memory-card activity-item" data-id="${esc(p.id)}">
            <div class="memory-card-header">
                <span>
                    <span class="activity-badge activity-badge-proactive">proactive</span>
                    <span class="tag">${esc(p.trigger_reason)}</span>
                </span>
                <span class="activity-ts">${formatTime(p.timestamp)}</span>
            </div>
            <div class="memory-card-content">${esc(p.content)}</div>
            <div class="memory-card-meta">
                <span class="tag">${esc(p.trigger_context.slice(0, 80))}</span>
            </div>
        </div>
    `;
}

// ============== Helpers ==============

function formatTime(iso) {
    if (!iso) return "";
    try {
        const d = new Date(iso);
        const now = new Date();
        const diffH = (now - d) / 3600000;

        if (diffH < 1) return `${Math.round(diffH * 60)}m ago`;
        if (diffH < 24) return `${Math.round(diffH)}h ago`;
        if (diffH < 48) return "yesterday";
        return d.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
    } catch {
        return iso.slice(0, 16);
    }
}

function esc(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
}

// ============== Filter ==============

filterEl.addEventListener("change", () => {
    if (_cachedData) renderTimeline(_cachedData, filterEl.value);
});

// ============== Stream toggle ==============

const streamToggleBtn = document.getElementById("btn-activity-stream-toggle");
const streamChevron = streamToggleBtn?.querySelector(".activity-stream-chevron");
let _streamExpanded = false;

streamToggleBtn?.addEventListener("click", () => {
    _streamExpanded = !_streamExpanded;
    streamEl?.classList.toggle("activity-stream-collapsed", !_streamExpanded);
    if (streamChevron) streamChevron.textContent = _streamExpanded ? "▴" : "▾";
});

// ============== Focus Panel (human-readable "what Remy is doing") ==============

const focusEl = document.getElementById("activity-focus");
const agentStatusEl = document.getElementById("activity-agent-status");

function _updateAgentStatusText() {
    if (!agentStatusEl) return;
    const s = _snapshotState;
    if (!s.autonomyRunning) {
        agentStatusEl.textContent = "Remy is idle";
        return;
    }
    if (s.currentTask?.action) {
        agentStatusEl.textContent = `Running: ${s.currentTask.action}`;
        return;
    }
    if (s.currentGoal?.description) {
        agentStatusEl.textContent = `Goal: ${s.currentGoal.description}`;
        return;
    }
    if (s.schedulerReason) {
        agentStatusEl.textContent = s.schedulerReason;
        return;
    }
    agentStatusEl.textContent = "Remy is active, waiting for a task";
}

function renderFocusPanel() {
    if (!focusEl) return;
    const s = _snapshotState;

    if (!s.autonomyRunning) {
        focusEl.innerHTML = `<div class="focus-idle">
            <div class="focus-idle-icon">&#128564;</div>
            <div class="focus-idle-text">Autonomous mode is off</div>
            <div class="focus-idle-hint">Enable the toggle above to let Remy work autonomously</div>
        </div>`;
        return;
    }

    const sections = [];

    // Current goal
    if (s.currentGoal) {
        const g = s.currentGoal;
        const desc = g.description || g.id || "";
        const prio = g.priority ? `<span class="focus-badge focus-badge-prio">${esc(g.priority)}</span>` : "";
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Current goal ${prio}</div>
                <div class="focus-section-value">${esc(desc)}</div>
            </div>
        `);
    }

    // Current mission (plan)
    if (s.currentMission) {
        const m = s.currentMission;
        const desc = m.description || m.id || "";
        const done = Number(m.completed_tasks || 0);
        const total = Number(m.total_tasks || 0);
        const progress = total > 0
            ? `<div class="focus-progress-wrap"><div class="focus-progress-bar" style="width:${Math.round(done/total*100)}%"></div></div><span class="focus-progress-text">${done} of ${total} steps</span>`
            : "";
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Plan</div>
                <div class="focus-section-value">${esc(desc)}</div>
                ${progress}
            </div>
        `);
    }

    // Current task
    if (s.currentTask?.action) {
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Currently running</div>
                <div class="focus-section-value focus-active">${esc(s.currentTask.action)}</div>
            </div>
        `);
    }

    // Current step
    if (s.currentStep?.instruction) {
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Step</div>
                <div class="focus-section-value">${esc(s.currentStep.instruction)}</div>
            </div>
        `);
    }

    // Research session
    if (s.researchSession?.topic) {
        const rs = s.researchSession;
        const bits = [
            rs.generated_queries_count ? `${rs.generated_queries_count} queries` : "",
            rs.findings_count ? `${rs.findings_count} findings` : "",
            rs.accepted_sources_count ? `${rs.accepted_sources_count} sources` : "",
        ].filter(Boolean).join(" · ");
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Research</div>
                <div class="focus-section-value">${esc(rs.topic)}</div>
                ${bits ? `<div class="focus-section-meta">${esc(bits)}</div>` : ""}
            </div>
        `);
    }

    // Last agent response summary
    if (s.lastAgentResponse?.response && !s.currentTask) {
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Last response</div>
                <div class="focus-section-value focus-muted">${esc(String(s.lastAgentResponse.response).slice(0, 200))}</div>
            </div>
        `);
    }

    // Last cycle result
    if (s.lastCycleResult?.decision && !s.currentTask) {
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Result</div>
                <div class="focus-section-value focus-muted">${esc(s.lastCycleResult.decision)}${s.lastCycleResult.reason ? ` — ${esc(s.lastCycleResult.reason.slice(0, 120))}` : ""}</div>
            </div>
        `);
    }

    // Pending approvals warning
    if (s.approvalQueue?.length) {
        const item = s.approvalQueue[0];
        const desc = item.description || item.action || "Approval required";
        sections.push(`
            <div class="focus-section focus-section-alert">
                <div class="focus-section-label">&#9888; Awaiting approval</div>
                <div class="focus-section-value">${esc(desc)}</div>
                <button class="btn btn-outline btn-mini" onclick="document.getElementById('btn-activity-decision-dossier').click()">Review</button>
            </div>
        `);
    }

    // Stuck missions
    if (s.stuckMissions?.length) {
        sections.push(`
            <div class="focus-section focus-section-warn">
                <div class="focus-section-label">&#9888; Stuck missions: ${s.stuckMissions.length}</div>
            </div>
        `);
    }

    // Scheduler reason (if nothing else meaningful to show)
    if (!sections.length && s.schedulerReason) {
        sections.push(`
            <div class="focus-section">
                <div class="focus-section-label">Status</div>
                <div class="focus-section-value focus-muted">${esc(s.schedulerReason)}</div>
            </div>
        `);
    }

    // Idle with autonomy on
    if (!sections.length) {
        focusEl.innerHTML = `<div class="focus-idle">
            <div class="focus-idle-icon">&#128336;</div>
            <div class="focus-idle-text">Remy is waiting for the next task</div>
        </div>`;
        return;
    }

    // Show "Details" button only when there's active work
    const hasDossier = Boolean(s.currentGoal || s.currentMission);
    const dossierBtn = document.getElementById("btn-activity-decision-dossier");
    if (dossierBtn) dossierBtn.style.display = hasDossier ? "" : "none";

    focusEl.innerHTML = sections.join("");
}
