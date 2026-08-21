/**
 * App Controller - navigation, init, event wiring.
 */

import { showConfirm, skeletonCards, skeletonGrid } from "./ui.js?v=1.21";

const _moduleCache = new Map();
window.__remyBootCompleted = false;

async function _loadModule(key, importer) {
    if (!_moduleCache.has(key)) {
        _moduleCache.set(key, importer());
    }
    return _moduleCache.get(key);
}

function _setHtml(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
}

function _showViewSkeleton(viewName) {
    if (viewName === "memory") {
        _setHtml("memory-list", skeletonCards(5));
        _setHtml("memory-stats-bar", `<div class="skeleton skeleton-line skeleton-short"></div>`);
    } else if (viewName === "tasks") {
        _setHtml("tasks-content", skeletonCards(4));
    } else if (viewName === "profile") {
        _setHtml("profile-content", skeletonCards(2));
    } else if (viewName === "stats") {
        _setHtml("stats-cards", skeletonGrid(6));
    } else if (viewName === "settings") {
        _setHtml("settings-content", skeletonCards(3));
    } else if (viewName === "history") {
        _setHtml("history-list", skeletonCards(5));
    } else if (viewName === "activity") {
        _setHtml("activity-list", skeletonCards(3));
    } else if (viewName === "reliability") {
        _setHtml("reliability-content", skeletonCards(4));
    } else if (viewName === "documents") {
        _setHtml("docs-list", skeletonCards(4));
        _setHtml("reports-list", skeletonCards(3));
    } else if (viewName === "pipelines") {
        _setHtml("pipelines-content", `<div class="pl-loading">${skeletonCards(3)}</div>`);
    } else if (viewName === "experiments") {
        _setHtml("experiments-content", skeletonCards(3));
    } else if (viewName === "automations") {
        _setHtml("automations-content", `<div class="pf-loading">${skeletonCards(3)}</div>`);
    } else if (viewName === "glass-brain") {
        _setHtml("view-glass-brain", `<div style="padding:20px">${skeletonCards(3)}</div>`);
    }
}

async function _loadMemoryView() {
    const mod = await _loadModule("memory", () => import("./memory.js?v=1.22"));
    await mod.loadRecords();
    await mod.loadMemoryStats();
}

async function _loadTasksView() {
    const mod = await _loadModule("tasks", () => import("./tasks.js?v=1.22"));
    await mod.loadTasks();
}

async function _loadProfileView() {
    const mod = await _loadModule("profile", () => import("./profile.js?v=1.22"));
    await mod.loadProfile();
}

async function _loadStatsView() {
    const mod = await _loadModule("stats", () => import("./stats.js?v=1.23"));
    await mod.loadStats();
}

async function _stopStatsRefresh() {
    if (!_moduleCache.has("stats")) return;
    const mod = await _moduleCache.get("stats");
    mod.stopHealthRefresh?.();
}

async function _loadSettingsView() {
    const mod = await _loadModule("settings", () => import("./settings.js?v=1.33"));
    await mod.loadSettings();
}

async function _loadHistoryView() {
    const mod = await _loadModule("history", () => import("./history.js?v=1.22"));
    await mod.loadHistory();
}

async function _loadActivityView() {
    const mod = await _loadModule("activity", () => import("./activity.js?v=1.23"));
    await mod.loadActivity();
}

async function _loadReliabilityView() {
    const mod = await _loadModule("reliability", () => import("./reliability.js?v=1.0"));
    await mod.loadReliability();
}


async function loadGraph() {
    const mod = await _loadModule("graph", () => import("./graph.js?v=1.23"));
    await mod.loadGraph();
}

async function loadDocuments() {
    const mod = await _loadModule("documents", () => import("./documents.js?v=1.23"));
    mod.initDocuments?.();
    await mod.loadDocuments();
}

async function loadCalendar() {
    await _loadModule("calendar", () => import("./calendar.js?v=1.19"));
    await window.loadCalendar?.();
}

async function loadPipelines() {
    const mod = await _loadModule("pipelines", () => import("./pipelines.js?v=3.0"));
    await mod.loadPipelines?.();
}

async function _loadExperimentsView() {
    const mod = await _loadModule("experiments", () => import("./experiments.js?v=1.15"));
    await mod.loadExperiments?.();
}

async function _loadAutomationsView() {
    const mod = await _loadModule("automations", () => import("./automations.js?v=2.9"));
    await mod.loadAutomations?.();
}

let _glassBrainMod = null;
async function _loadGlassBrainView() {
    if (!_glassBrainMod) {
        _glassBrainMod = await import("./glass_brain.js?v=1.6");
    }
    await _glassBrainMod.loadGlassBrain();
}

function _stopGlassBrainRefresh() {
    _glassBrainMod?.stopGlassBrainRefresh?.();
}


async function _initHumanLoopSurfaces() {
    const [approvalMod, guidanceMod] = await Promise.all([
        _loadModule("approval", () => import("./approval.js?v=1.22")),
        _loadModule("guidance", () => import("./guidance.js?v=1.22")),
    ]);
    approvalMod.initApprovals?.();
    guidanceMod.initGuidance?.();
}

const navItems = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");
const newSessionBtn = document.getElementById("btn-new-session");
const stopRemyBtn = document.getElementById("btn-stop-remy");
const projectSelect = document.getElementById("project-select");
const newProjectBtn = document.getElementById("btn-new-project");
const manageProjectsBtn = document.getElementById("btn-manage-projects");
const operatorAlertsButton = document.getElementById("btn-operator-alerts");
const operatorAlertCount = document.getElementById("operator-alert-count");
const projectBrainLabel = document.getElementById("project-brain-label");
const projectManagerModal = document.getElementById("project-manager-modal");
const closeProjectManagerBtn = document.getElementById("btn-close-project-manager");
const projectCreateForm = document.getElementById("project-create-form");
const projectCreateName = document.getElementById("project-create-name");
const projectCreateDomain = document.getElementById("project-create-domain");
const projectCreateDescription = document.getElementById("project-create-description");
const projectCreateSubmit = document.getElementById("btn-create-project-submit");
const projectManagerStatus = document.getElementById("project-manager-status");
const projectManagerActiveList = document.getElementById("project-manager-active-list");
const projectManagerArchivedList = document.getElementById("project-manager-archived-list");
const projectManagerArchiveSection = document.getElementById("project-manager-archive-section");
const projectManagerActiveSection = document.getElementById("project-manager-active-section");
const projectAgentPanel = document.getElementById("project-agent-panel");
const closeProjectAgentBtn = document.getElementById("btn-close-project-agent");
const projectAgentTitle = document.getElementById("project-agent-title");
const projectAgentForm = document.getElementById("project-agent-form");
const projectAgentInstruction = document.getElementById("project-agent-instruction");
const projectAgentStatus = document.getElementById("project-agent-status");

let _operatorAlerts = [];
let _operatorAlertsRefreshTimer = null;

function _operatorEsc(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function _operatorAlertTime(value) {
    if (!value) return "";
    const timestamp = Number(value);
    const date = Number.isFinite(timestamp) ? new Date(timestamp * 1000) : new Date(value);
    return Number.isNaN(date.valueOf()) ? "" : date.toLocaleString();
}

function _syncOperatorAlertBadge() {
    const count = _operatorAlerts.filter((row) => !row.acknowledged && !row.resolved).length;
    if (operatorAlertCount) {
        operatorAlertCount.hidden = count === 0;
        operatorAlertCount.textContent = String(count);
    }
    operatorAlertsButton?.classList.toggle("has-alerts", count > 0);
}

function _renderOperatorAlerts() {
    const content = document.getElementById("panel-content");
    if (!content) return;
    content.innerHTML = `<div class="operator-alert-center">
        <div class="operator-alert-center-summary">
            <strong>${_operatorAlerts.filter((row) => !row.acknowledged && !row.resolved).length} active</strong>
            <span>Incidents are coalesced and recovery resolves the matching alert.</span>
        </div>
        <div class="operator-alert-center-list">${_operatorAlerts.length ? _operatorAlerts.map((row) => `
            <article class="level-${_operatorEsc(row.level || "info")} ${row.resolved ? "resolved" : ""}">
                <div>
                    <strong>${_operatorEsc(row.message || "Operator incident")}</strong>
                    <span>${_operatorEsc(row.source || "system")} · ${_operatorEsc(row.failure_code || row.level || "info")}</span>
                    <small>${_operatorEsc(_operatorAlertTime(row.timestamp))}${Number(row.repeat_count || 1) > 1 ? ` · repeated ${Number(row.repeat_count)}x` : ""}</small>
                </div>
                <div class="operator-alert-center-actions">
                    ${row.action_target ? `<button type="button" data-operator-open="${_operatorEsc(row.id)}">Open</button>` : ""}
                    ${!row.acknowledged && !row.resolved ? `<button type="button" data-operator-ack="${_operatorEsc(row.id)}">Acknowledge</button>` : ""}
                    ${row.resolved ? '<span class="operator-alert-resolved">Resolved</span>' : ""}
                </div>
            </article>`).join("") : '<div class="empty-state"><div class="empty-state-title">No operator incidents</div><div class="empty-state-hint">Critical runtime and Trajectory signals will appear here.</div></div>'}</div>
    </div>`;
    content.querySelectorAll("[data-operator-ack]").forEach((button) => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            try {
                await window.apiClient.acknowledgeOperatorAlert(button.dataset.operatorAck);
                await _refreshOperatorAlerts(true);
            } catch (error) {
                button.disabled = false;
                button.title = error.message || "Could not acknowledge incident";
            }
        });
    });
    content.querySelectorAll("[data-operator-open]").forEach((button) => {
        button.addEventListener("click", async () => {
            const alert = _operatorAlerts.find((row) => row.id === button.dataset.operatorOpen);
            if (!alert) return;
            const artifacts = Array.isArray(alert.artifact_ids) ? alert.artifact_ids : [];
            if (String(alert.action_target || "").startsWith("open_trajectory_")) {
                const opensAnalytics = alert.action_target === "open_trajectory_analytics";
                await switchView("chat");
                document.querySelector(".app")?.classList.remove("panel-open");
                document.dispatchEvent(new CustomEvent("trajectory-open-alert", {
                    detail: {
                        conversationId: opensAnalytics ? "" : (artifacts[0] || ""),
                        eventId: opensAnalytics ? "" : (artifacts[1] || ""),
                        alertId: artifacts[2] || artifacts[0] || "",
                        analytics: opensAnalytics,
                    },
                }));
            } else if (alert.action_target === "open_self_modification_lab") {
                await switchView("experiments");
                document.querySelector(".app")?.classList.remove("panel-open");
                document.dispatchEvent(new CustomEvent("self-modification-open", {
                    detail: { proposalId: artifacts[0] || alert.proposal_id || "" },
                }));
            } else if (String(alert.action_target || "").includes("memory")) {
                await switchView("memory");
                document.dispatchEvent(new CustomEvent("operator-alert-action", { detail: alert }));
            }
        });
    });
}

async function _refreshOperatorAlerts(render = false) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 4000);
    try {
        const data = await window.apiClient.getOperatorAlerts(8, {
            signal: controller.signal,
        });
        _operatorAlerts = Array.isArray(data?.operator_alerts?.items)
            ? data.operator_alerts.items
            : [];
        _syncOperatorAlertBadge();
        if (render) _renderOperatorAlerts();
    } catch (error) {
        if (render) {
            const content = document.getElementById("panel-content");
            if (content) content.innerHTML = `<div class="empty-state">${_operatorEsc(error.message || "Could not load incidents")}</div>`;
        }
    } finally {
        window.clearTimeout(timeout);
    }
}

async function _openOperatorAlertCenter() {
    const title = document.getElementById("panel-title");
    const content = document.getElementById("panel-content");
    document.querySelector(".app")?.classList.add("panel-open");
    if (title) title.textContent = "Incident center";
    if (content) content.innerHTML = '<div class="empty-state">Loading incidents...</div>';
    await _refreshOperatorAlerts(true);
}

document.addEventListener("execution-trajectory-open", async (event) => {
    await switchView("chat");
    document.querySelector(".app")?.classList.remove("panel-open");
    document.dispatchEvent(new CustomEvent("trajectory-open-execution", {
        detail: event.detail || {},
    }));
});

function _initOperatorAlertCenter() {
    operatorAlertsButton?.addEventListener("click", _openOperatorAlertCenter);
    window.apiClient.onRuntimeEvent((event) => {
        if (event?.type !== "operator_alert" && event?.event_name !== "operator_alert") return;
        if (_operatorAlertsRefreshTimer) clearTimeout(_operatorAlertsRefreshTimer);
        _operatorAlertsRefreshTimer = setTimeout(() => {
            const open = document.querySelector(".app")?.classList.contains("panel-open")
                && document.getElementById("panel-title")?.textContent === "Incident center";
            _refreshOperatorAlerts(open);
        }, 150);
    });
    _refreshOperatorAlerts(false);
}
const knowledgePackCreateForm = document.getElementById("knowledge-pack-create-form");
const knowledgePackName = document.getElementById("knowledge-pack-name");
const knowledgePackFiles = document.getElementById("knowledge-pack-files");
const knowledgePackList = document.getElementById("knowledge-pack-list");
const conversationList = document.getElementById("conversation-list");
const chatConversationTitle = document.getElementById("chat-conversation-title");
const sidebarInsights = document.getElementById("sidebar-insights");
const closePanelBtn = document.getElementById("btn-close-panel");
const statusEl = document.getElementById("connection-status");
const statusText = statusEl?.querySelector(".status-text");
const startupSplash = document.getElementById("startup-splash");
const startupSplashStatus = document.getElementById("startup-splash-status");
const FIRST_RUN_DONE_KEY = "remy_first_run_done_v1";
const SIDEBAR_INSIGHTS_OPEN_KEY = "remy_sidebar_insights_open_v1";
const SIDEBAR_WIDTH_KEY = "remy_sidebar_width_v1";
const SIDEBAR_DEFAULT_WIDTH = 224;
const SIDEBAR_MIN_WIDTH = 184;
const SIDEBAR_MAX_WIDTH = 420;
const SIDEBAR_DESKTOP_QUERY = "(min-width: 769px)";
let _activeProjectId = "";
let _activeConversationId = "";
let _conversationSwitchSequence = 0;
let _conversationsById = new Map();

if (sidebarInsights) {
    sidebarInsights.open =
        window.localStorage.getItem(SIDEBAR_INSIGHTS_OPEN_KEY) === "true";
    sidebarInsights.addEventListener("toggle", () => {
        window.localStorage.setItem(
            SIDEBAR_INSIGHTS_OPEN_KEY,
            sidebarInsights.open ? "true" : "false",
        );
    });
}

function _emitConversationChanged(conversation, detail = {}) {
    if (!conversation?.conversation_id) return;
    _activeConversationId = conversation.conversation_id;
    if (chatConversationTitle) {
        chatConversationTitle.textContent = conversation.title || "Conversation";
    }
    document.dispatchEvent(new CustomEvent("conversation-changed", {
        detail: { conversation, ...detail },
    }));
}

function _markConversationActive(conversationId) {
    _activeConversationId = conversationId || "";
    conversationList?.querySelectorAll(".conversation-row").forEach((row) => {
        row.classList.toggle(
            "active",
            row.dataset.conversationId === _activeConversationId,
        );
    });
}

function _conversationActionButton(label, title, className = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `conversation-action ${className}`.trim();
    button.textContent = label;
    button.title = title;
    return button;
}

function _renderConversationList(payload) {
    if (!conversationList) return;
    conversationList.innerHTML = "";
    _activeConversationId = payload.active_conversation_id || "";
    const conversations = payload.conversations || [];
    _conversationsById = new Map(
        conversations.map((conversation) => [conversation.conversation_id, conversation]),
    );
    if (!conversations.length) {
        const empty = document.createElement("span");
        empty.className = "conversation-list-status";
        empty.textContent = "No chats yet";
        conversationList.appendChild(empty);
        return;
    }

    for (const conversation of conversations) {
        const row = document.createElement("div");
        row.className = "conversation-row";
        row.dataset.conversationId = conversation.conversation_id;
        row.classList.toggle(
            "active",
            conversation.conversation_id === _activeConversationId,
        );

        const open = document.createElement("button");
        open.type = "button";
        open.className = "conversation-open";
        open.textContent = conversation.title || "Untitled chat";
        open.title = conversation.title || "Untitled chat";
        open.addEventListener("click", async () => {
            if (conversation.conversation_id === _activeConversationId) {
                await switchView("chat");
                _emitConversationChanged(conversation, { preferCachedTranscript: true });
                return;
            }
            const sequence = ++_conversationSwitchSequence;
            const previousId = _activeConversationId;
            const previousConversation = _conversationsById.get(previousId);
            _markConversationActive(conversation.conversation_id);
            row.classList.add("loading");
            const activation = window.apiClient.activateConversation(
                conversation.conversation_id,
            );
            try {
                await switchView("chat");
                // Transcript loading is independent of mounting the agent
                // session, so begin it immediately and in parallel.
                _emitConversationChanged(conversation, { activationPending: true });
                const result = await activation;
                if (sequence !== _conversationSwitchSequence) return;
                _markConversationActive(result.conversation.conversation_id);
                _emitConversationChanged(result.conversation, {
                    skipTranscriptReload: true,
                });
            } catch (err) {
                console.error("Failed to switch conversation", err);
                if (sequence === _conversationSwitchSequence) {
                    _markConversationActive(previousId);
                    if (previousConversation) {
                        _emitConversationChanged(previousConversation);
                    }
                }
            } finally {
                row.classList.remove("loading");
            }
        });

        const rename = _conversationActionButton("✎", "Rename chat");
        rename.addEventListener("click", async () => {
            const title = window.prompt("Rename this chat", conversation.title || "");
            if (!title?.trim()) return;
            try {
                const result = await window.apiClient.renameConversation(
                    conversation.conversation_id,
                    title.trim(),
                );
                if (conversation.conversation_id === _activeConversationId) {
                    _emitConversationChanged(result.conversation);
                }
                await loadConversations({ emitActive: false });
            } catch (err) {
                console.error("Failed to rename conversation", err);
            }
        });

        const archive = _conversationActionButton("×", "Archive chat", "danger");
        archive.addEventListener("click", async () => {
            const confirmed = await showConfirm(
                "Archive chat",
                `Archive “${conversation.title || "Untitled chat"}”? Its history will be kept.`,
            );
            if (!confirmed) return;
            try {
                const result = await window.apiClient.archiveConversation(
                    conversation.conversation_id,
                );
                const refreshed = await loadConversations({ emitActive: false });
                if (result.chat_reset) {
                    const active = (refreshed?.conversations || []).find(
                        (item) => item.conversation_id === result.active_conversation_id,
                    );
                    if (active) _emitConversationChanged(active);
                }
            } catch (err) {
                console.error("Failed to archive conversation", err);
            }
        });

        row.append(open, rename, archive);
        conversationList.appendChild(row);
    }
}

async function loadConversations({ emitActive = true } = {}) {
    const apiClient = await _waitForApiClient();
    if (!apiClient || !conversationList) return null;
    try {
        const payload = await apiClient.getConversations();
        _renderConversationList(payload);
        if (emitActive) {
            const active = (payload.conversations || []).find(
                (item) => item.conversation_id === payload.active_conversation_id,
            );
            if (active) _emitConversationChanged(active);
        }
        return payload;
    } catch (err) {
        console.error("Failed to load conversations", err);
        conversationList.innerHTML =
            `<span class="conversation-list-status error">Chats unavailable</span>`;
        return null;
    }
}

document.addEventListener("conversation-list-refresh", () => {
    loadConversations({ emitActive: false });
});

function _setProjectManagerStatus(message = "", isError = false) {
    if (!projectManagerStatus) return;
    projectManagerStatus.textContent = message;
    projectManagerStatus.classList.toggle("error", Boolean(isError));
}

let _projectAgentProject = null;

function _setProjectAgentStatus(message = "", isError = false) {
    if (!projectAgentStatus) return;
    projectAgentStatus.textContent = message;
    projectAgentStatus.classList.toggle("error", Boolean(isError));
}

function _toggleProjectAgentPanel(show) {
    projectAgentPanel?.classList.toggle("hidden", !show);
    projectCreateForm?.classList.toggle("hidden", show);
    projectManagerActiveSection?.classList.toggle("hidden", show);
    if (show) {
        projectManagerArchiveSection?.classList.add("hidden");
    } else if (projectManagerArchivedList?.children.length) {
        projectManagerArchiveSection?.classList.remove("hidden");
    }
}

function _knowledgePackSourceRow(project, pack, source) {
    const row = document.createElement("div");
    row.className = "knowledge-pack-source";
    const name = document.createElement("span");
    name.textContent = source.name;
    name.title = `${source.name} - ${Math.max(1, Math.round((source.size || 0) / 1024))} KB`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn-icon";
    remove.textContent = "x";
    remove.title = "Remove source";
    remove.addEventListener("click", async () => {
        remove.disabled = true;
        try {
            await window.apiClient.deleteKnowledgePackSource(
                project.project_id, pack.pack_id, source.source_id,
            );
            await _openProjectAgent(project);
        } catch (err) {
            remove.disabled = false;
            _setProjectAgentStatus(err.message || "Could not remove source", true);
        }
    });
    row.append(name, remove);
    return row;
}

function _knowledgePackCard(project, pack) {
    const card = document.createElement("article");
    card.className = "knowledge-pack-card";
    card.classList.toggle("disabled", !pack.enabled);

    const header = document.createElement("div");
    header.className = "knowledge-pack-card-header";
    const copy = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = pack.name;
    const description = document.createElement("div");
    description.className = "project-manager-hint";
    description.textContent = pack.description || "Curated project reference sources";
    copy.append(title, description);

    const controls = document.createElement("div");
    controls.className = "knowledge-pack-controls";
    const enabledLabel = document.createElement("label");
    enabledLabel.className = "knowledge-pack-toggle";
    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.checked = Boolean(pack.enabled);
    enabled.addEventListener("change", async () => {
        enabled.disabled = true;
        try {
            await window.apiClient.updateKnowledgePack(
                project.project_id, pack.pack_id, { enabled: enabled.checked },
            );
            await _openProjectAgent(project);
        } catch (err) {
            enabled.checked = !enabled.checked;
            enabled.disabled = false;
            _setProjectAgentStatus(err.message || "Could not update pack", true);
        }
    });
    enabledLabel.append(enabled, document.createTextNode(" Enabled"));
    const removePack = _projectAction("Delete", "project-archive");
    removePack.addEventListener("click", async () => {
        const confirmed = await showConfirm(
            "Delete Knowledge Pack",
            `Delete вЂњ${pack.name}вЂќ and its uploaded sources? Project memory is not affected.`,
        );
        if (!confirmed) return;
        try {
            await window.apiClient.deleteKnowledgePack(project.project_id, pack.pack_id);
            await _openProjectAgent(project);
        } catch (err) {
            _setProjectAgentStatus(err.message || "Could not delete pack", true);
        }
    });
    controls.append(enabledLabel, removePack);
    header.append(copy, controls);

    const sources = document.createElement("div");
    sources.className = "knowledge-pack-sources";
    for (const source of pack.sources || []) {
        sources.appendChild(_knowledgePackSourceRow(project, pack, source));
    }
    if (!(pack.sources || []).length) {
        const empty = document.createElement("span");
        empty.className = "project-manager-hint";
        empty.textContent = "No sources yet.";
        sources.appendChild(empty);
    }

    const upload = document.createElement("div");
    upload.className = "knowledge-pack-upload";
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".txt,.md,.csv,.json,.jsonl,.yaml,.yml,.pdf,.docx,.xlsx,.html,.htm,.xml";
    input.className = "hidden";
    const add = _projectAction("+ Add source");
    const hint = document.createElement("span");
    hint.className = "project-manager-hint";
    hint.textContent = "Text, PDF, Word, or Excel - up to 5 MB";
    add.addEventListener("click", () => input.click());
    input.addEventListener("change", async () => {
        const file = input.files?.[0];
        if (!file) return;
        add.disabled = true;
        _setProjectAgentStatus(`Adding ${file.name}...`);
        try {
            await window.apiClient.uploadKnowledgePackSource(
                project.project_id, pack.pack_id, file,
            );
            await _openProjectAgent(project);
            _setProjectAgentStatus(`${file.name} added.`);
        } catch (err) {
            add.disabled = false;
            _setProjectAgentStatus(err.message || "Could not add source", true);
        }
    });
    upload.append(add, hint, input);
    card.append(header, sources, upload);
    return card;
}

function _renderProjectAgent(project, payload) {
    _projectAgentProject = project;
    const profile = payload.profile || {};
    if (projectAgentTitle) projectAgentTitle.textContent = `${project.name} Agent`;
    if (projectAgentInstruction) projectAgentInstruction.value = profile.instruction || "";
    knowledgePackList?.replaceChildren();
    for (const pack of payload.knowledge_packs || []) {
        knowledgePackList?.appendChild(_knowledgePackCard(project, pack));
    }
    if (!(payload.knowledge_packs || []).length && knowledgePackList) {
        const empty = document.createElement("div");
        empty.className = "empty-state knowledge-pack-empty";
        empty.textContent = "No Knowledge Packs yet. Create one for curated domain references.";
        knowledgePackList.appendChild(empty);
    }
}

async function _openProjectAgent(project) {
    _toggleProjectAgentPanel(true);
    _setProjectAgentStatus("Loading Project Agent...");
    try {
        const payload = await window.apiClient.getProjectAgent(project.project_id);
        _renderProjectAgent(project, payload);
        _setProjectAgentStatus();
    } catch (err) {
        _setProjectAgentStatus(err.message || "Could not load Project Agent", true);
    }
}

function _closeProjectManager() {
    projectManagerModal?.classList.remove("active");
    _toggleProjectAgentPanel(false);
    _projectAgentProject = null;
    _setProjectManagerStatus();
}

function _projectAction(label, className = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `btn btn-outline ${className}`.trim();
    button.textContent = label;
    return button;
}

function _projectMemoryLabel(project) {
    const area = project.domain || "General";
    if (project.archived_at) return `${area} · workspace preserved`;
    return `${area} · shared memory across project chats`;
}

function _renderProjectManagerItem(project) {
    const row = document.createElement("div");
    row.className = "project-manager-item";
    row.classList.toggle("active", Boolean(project.active));
    row.dataset.projectId = project.project_id;

    const copy = document.createElement("div");
    copy.className = "project-manager-copy";
    const nameRow = document.createElement("div");
    nameRow.className = "project-manager-name-row";
    const name = document.createElement("span");
    name.className = "project-manager-name";
    name.textContent = project.name;
    nameRow.appendChild(name);
    if (project.active) {
        const badge = document.createElement("span");
        badge.className = "project-manager-badge";
        badge.textContent = "Active";
        nameRow.appendChild(badge);
    }
    const meta = document.createElement("div");
    meta.className = "project-manager-meta";
    meta.textContent = _projectMemoryLabel(project);
    copy.append(nameRow, meta);
    if (project.description) {
        const description = document.createElement("div");
        description.className = "project-manager-description";
        description.textContent = project.description;
        copy.appendChild(description);
    }

    const actions = document.createElement("div");
    actions.className = "project-manager-actions";

    if (!project.archived_at && !project.active) {
        const open = _projectAction("Open");
        open.addEventListener("click", async () => {
            _setProjectManagerStatus(`Opening “${project.name}”…`);
            [...actions.querySelectorAll("button")].forEach((button) => {
                button.disabled = true;
            });
            try {
                await window.apiClient.activateProject(project.project_id);
                window.location.reload();
            } catch (err) {
                _setProjectManagerStatus(err.message || "Could not open project", true);
                [...actions.querySelectorAll("button")].forEach((button) => {
                    button.disabled = false;
                });
            }
        });
        actions.appendChild(open);
    }

    if (!project.archived_at) {
        const agent = _projectAction("Agent");
        agent.title = "Agent instruction and Knowledge Packs";
        agent.addEventListener("click", () => _openProjectAgent(project));
        actions.appendChild(agent);

        const rename = _projectAction("Edit");
        rename.addEventListener("click", () => {
            const input = document.createElement("input");
            input.className = "input project-rename-input";
            input.value = project.name;
            input.maxLength = 120;
            input.placeholder = "Project name";

            const domainInput = document.createElement("input");
            domainInput.className = "input";
            domainInput.value = project.domain || "";
            domainInput.maxLength = 80;
            domainInput.placeholder = "Area: Marketing, Biology, Legal...";

            const descriptionInput = document.createElement("textarea");
            descriptionInput.className = "input";
            descriptionInput.value = project.description || "";
            descriptionInput.maxLength = 2000;
            descriptionInput.rows = 3;
            descriptionInput.placeholder = "What is this project for?";

            copy.replaceChildren(input, domainInput, descriptionInput);
            copy.classList.add("project-manager-edit-fields");

            const save = _projectAction("Save");
            save.classList.remove("btn-outline");
            save.classList.add("btn-primary");
            const cancel = _projectAction("Cancel");
            actions.replaceChildren(save, cancel);
            input.focus();
            input.select();

            const commit = async () => {
                const cleanName = input.value.trim();
                if (!cleanName) {
                    input.focus();
                    return;
                }
                save.disabled = true;
                cancel.disabled = true;
                _setProjectManagerStatus("Saving project profile…");
                try {
                    await window.apiClient.updateProject(project.project_id, {
                        name: cleanName,
                        domain: domainInput.value.trim(),
                        description: descriptionInput.value.trim(),
                    });
                    await initProjectSwitcher();
                    await loadProjectManager();
                } catch (err) {
                    _setProjectManagerStatus(err.message || "Could not update project", true);
                    save.disabled = false;
                    cancel.disabled = false;
                }
            };
            save.addEventListener("click", commit);
            cancel.addEventListener("click", () => loadProjectManager());
            input.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    event.preventDefault();
                    commit();
                } else if (event.key === "Escape") {
                    loadProjectManager();
                }
            });
        });
        actions.appendChild(rename);
    }

    if (!project.archived_at && !project.legacy) {
        const archive = _projectAction("Archive", "project-archive");
        archive.addEventListener("click", async () => {
            _closeProjectManager();
            const confirmed = await showConfirm(
                "Archive project",
                `Archive “${project.name}”? Its MicroBrain, chats, files, and workflows will be kept.`,
            );
            if (!confirmed) {
                await openProjectManager();
                return;
            }
            try {
                const result = await window.apiClient.archiveProject(project.project_id);
                if (result.chat_reset) {
                    window.location.reload();
                    return;
                }
                await initProjectSwitcher();
                await openProjectManager();
                _setProjectManagerStatus(`“${project.name}” was archived.`);
            } catch (err) {
                await openProjectManager();
                _setProjectManagerStatus(err.message || "Could not archive project", true);
            }
        });
        actions.appendChild(archive);
    }

    if (project.archived_at) {
        const restore = _projectAction("Restore");
        restore.addEventListener("click", async () => {
            restore.disabled = true;
            _setProjectManagerStatus(`Restoring “${project.name}”…`);
            try {
                await window.apiClient.restoreProject(project.project_id);
                await initProjectSwitcher();
                await loadProjectManager();
                _setProjectManagerStatus(`“${project.name}” is available again.`);
            } catch (err) {
                restore.disabled = false;
                _setProjectManagerStatus(err.message || "Could not restore project", true);
            }
        });
        actions.appendChild(restore);
    }

    row.append(copy, actions);
    return row;
}

function _renderProjectManager(payload) {
    if (!projectManagerActiveList || !projectManagerArchivedList) return;
    projectManagerActiveList.replaceChildren();
    projectManagerArchivedList.replaceChildren();
    const projects = payload.projects || [];
    const available = projects.filter((project) => !project.archived_at);
    const archived = projects.filter((project) => project.archived_at);

    available.forEach((project) => {
        projectManagerActiveList.appendChild(_renderProjectManagerItem(project));
    });
    archived.forEach((project) => {
        projectManagerArchivedList.appendChild(_renderProjectManagerItem(project));
    });
    projectManagerArchiveSection?.classList.toggle("hidden", archived.length === 0);
}

async function loadProjectManager() {
    const apiClient = await _waitForApiClient();
    if (!apiClient) return null;
    try {
        const payload = await apiClient.getProjects(true);
        _renderProjectManager(payload);
        _setProjectManagerStatus();
        return payload;
    } catch (err) {
        _setProjectManagerStatus(err.message || "Could not load projects", true);
        return null;
    }
}

async function openProjectManager({ focusCreate = false } = {}) {
    if (!projectManagerModal) return;
    projectManagerModal.classList.add("active");
    _setProjectManagerStatus("Loading projects…");
    await loadProjectManager();
    if (focusCreate) projectCreateName?.focus();
}

async function initProjectSwitcher() {
    const apiClient = await _waitForApiClient();
    if (!apiClient || !projectSelect) return;
    try {
        const payload = await apiClient.getProjects();
        _activeProjectId = payload.active_project_id || "";
        projectSelect.disabled = false;
        if (newProjectBtn) newProjectBtn.disabled = false;
        if (manageProjectsBtn) manageProjectsBtn.disabled = false;
        projectSelect.innerHTML = "";
        for (const project of payload.projects || []) {
            const option = document.createElement("option");
            option.value = project.project_id;
            option.textContent = project.name;
            option.selected = project.project_id === _activeProjectId;
            projectSelect.appendChild(option);
        }
        const active = (payload.projects || []).find(
            (project) => project.project_id === _activeProjectId
        );
        if (projectBrainLabel && active) {
            const area = active.domain || "General workspace";
            projectBrainLabel.textContent = `${area} · shared project memory`;
            projectBrainLabel.title = active.description ||
                "All chats in this project use the same private memory and project resources.";
        }
    } catch (err) {
        console.error("Failed to load projects", err);
        projectSelect.innerHTML = `<option value="">Projects unavailable</option>`;
        projectSelect.disabled = true;
        if (newProjectBtn) newProjectBtn.disabled = true;
        if (manageProjectsBtn) manageProjectsBtn.disabled = true;
    }
}

projectSelect?.addEventListener("change", async () => {
    const nextProjectId = projectSelect.value;
    if (!nextProjectId || nextProjectId === _activeProjectId) return;
    const previousProjectId = _activeProjectId;
    projectSelect.disabled = true;
    if (projectBrainLabel) projectBrainLabel.textContent = "Switching project workspace...";
    try {
        await window.apiClient.activateProject(nextProjectId);
        window.location.reload();
    } catch (err) {
        console.error("Failed to switch project", err);
        projectSelect.value = previousProjectId;
        projectSelect.disabled = false;
        if (projectBrainLabel) projectBrainLabel.textContent = "Could not switch project";
    }
});

newProjectBtn?.addEventListener("click", () => {
    openProjectManager({ focusCreate: true });
});

manageProjectsBtn?.addEventListener("click", () => {
    openProjectManager();
});

closeProjectManagerBtn?.addEventListener("click", _closeProjectManager);

closeProjectAgentBtn?.addEventListener("click", async () => {
    _projectAgentProject = null;
    _toggleProjectAgentPanel(false);
    _setProjectAgentStatus();
    await loadProjectManager();
});

projectAgentForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!_projectAgentProject) return;
    const save = document.getElementById("btn-save-project-agent");
    if (save) save.disabled = true;
    _setProjectAgentStatus("Saving instruction...");
    try {
        const payload = await window.apiClient.updateProjectAgent(
            _projectAgentProject.project_id,
            {
                instruction: projectAgentInstruction?.value.trim() || "",
            },
        );
        _renderProjectAgent(_projectAgentProject, payload);
        _setProjectAgentStatus("Instruction saved.");
    } catch (err) {
        _setProjectAgentStatus(err.message || "Could not save specialization", true);
    } finally {
        if (save) save.disabled = false;
    }
});

knowledgePackCreateForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!_projectAgentProject) return;
    const name = knowledgePackName?.value.trim() || "";
    if (!name) {
        knowledgePackName?.focus();
        return;
    }
    const files = Array.from(knowledgePackFiles?.files || []);
    if (!files.length) {
        knowledgePackFiles?.focus();
        _setProjectAgentStatus("Select at least one knowledge file.", true);
        return;
    }
    const submit = knowledgePackCreateForm.querySelector("button[type='submit']");
    if (submit) submit.disabled = true;
    _setProjectAgentStatus("Creating Knowledge Pack...");
    try {
        const created = await window.apiClient.createKnowledgePack(
            _projectAgentProject.project_id,
            name,
            "",
        );
        const packId = created?.pack?.pack_id;
        if (!packId) throw new Error("Knowledge Pack was created without an identifier");

        let uploaded = 0;
        const failures = [];
        for (const file of files) {
            _setProjectAgentStatus(`Uploading ${uploaded + 1}/${files.length}: ${file.name}...`);
            try {
                await window.apiClient.uploadKnowledgePackSource(
                    _projectAgentProject.project_id, packId, file,
                );
                uploaded += 1;
            } catch (err) {
                failures.push(`${file.name}: ${err.message || "upload failed"}`);
            }
        }
        if (knowledgePackName) knowledgePackName.value = "";
        if (knowledgePackFiles) knowledgePackFiles.value = "";
        await _openProjectAgent(_projectAgentProject);
        if (failures.length) {
            _setProjectAgentStatus(
                `Pack created. Uploaded ${uploaded}/${files.length}. ${failures.join("; ")}`,
                true,
            );
        } else {
            _setProjectAgentStatus(`Knowledge Pack created with ${uploaded} file${uploaded === 1 ? "" : "s"}.`);
        }
    } catch (err) {
        _setProjectAgentStatus(err.message || "Could not create Knowledge Pack", true);
    } finally {
        if (submit) submit.disabled = false;
    }
});

projectManagerModal?.addEventListener("click", (event) => {
    if (event.target === projectManagerModal) _closeProjectManager();
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && projectManagerModal?.classList.contains("active")) {
        _closeProjectManager();
    }
});

projectCreateForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = projectCreateName?.value.trim() || "";
    if (!name) {
        projectCreateName?.focus();
        return;
    }
    if (projectCreateSubmit) projectCreateSubmit.disabled = true;
    if (projectCreateName) projectCreateName.disabled = true;
    if (projectCreateDomain) projectCreateDomain.disabled = true;
    if (projectCreateDescription) projectCreateDescription.disabled = true;
    _setProjectManagerStatus("Creating a closed project workspace…");
    try {
        const created = await window.apiClient.createProject(name, true, {
            domain: projectCreateDomain?.value.trim() || "",
            description: projectCreateDescription?.value.trim() || "",
        });
        if (created.compatibility_warning) {
            window.sessionStorage.setItem(
                "remy_project_compatibility_warning",
                created.compatibility_warning,
            );
        }
        window.location.reload();
    } catch (err) {
        console.error("Failed to create project", err);
        if (projectCreateSubmit) projectCreateSubmit.disabled = false;
        if (projectCreateName) projectCreateName.disabled = false;
        if (projectCreateDomain) projectCreateDomain.disabled = false;
        if (projectCreateDescription) projectCreateDescription.disabled = false;
        _setProjectManagerStatus(err.message || "Could not create project", true);
        projectCreateName?.focus();
    }
});
let startupSplashHidden = false;
let firstRunStep = 0;

function _sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function _waitForApiClient(timeoutMs = 8000, pollMs = 100) {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
        if (window.apiClient) {
            return window.apiClient;
        }
        await _sleep(pollMs);
    }
    return null;
}

function setStartupStatus(text) {
    if (startupSplashStatus) {
        startupSplashStatus.textContent = text;
    }
}

function hideStartupSplash() {
    if (!startupSplash || startupSplashHidden) return;
    startupSplashHidden = true;
    startupSplash.classList.add("is-hidden");
    window.__remyBootCompleted = true;
}

function _setFirstRunStep(step) {
    firstRunStep = step;
    document.querySelectorAll("[data-first-run-step]").forEach((el) => {
        el.classList.toggle("hidden", Number(el.dataset.firstRunStep) !== step);
    });
    document.querySelectorAll("[data-first-run-dot]").forEach((el) => {
        el.classList.toggle("active", Number(el.dataset.firstRunDot) <= step);
    });
}

function _closeFirstRunWizard(done = true) {
    if (done) window.localStorage.setItem(FIRST_RUN_DONE_KEY, "1");
    document.getElementById("first-run-wizard")?.classList.add("hidden");
}

async function _maybeShowFirstRunWizard() {
    if (window.localStorage.getItem(FIRST_RUN_DONE_KEY) === "1") return;
    const wizard = document.getElementById("first-run-wizard");
    if (!wizard) return;
    let settings = null;
    try {
        settings = await fetch("/api/settings").then((r) => r.json());
    } catch {
        settings = null;
    }
    if (settings?.has_api_key || settings?.has_openrouter_key) {
        window.localStorage.setItem(FIRST_RUN_DONE_KEY, "1");
        return;
    }
    _setFirstRunStep(0);
    wizard.classList.remove("hidden");
}

async function _saveFirstRunKey() {
    const provider = document.getElementById("first-run-provider")?.value || "gemini";
    const key = document.getElementById("first-run-key")?.value?.trim() || "";
    const status = document.getElementById("first-run-status");
    if (!key) {
        if (status) status.textContent = "Paste a key or skip this step.";
        return;
    }
    if (status) status.textContent = "Saving...";
    const body = provider === "openrouter"
        ? { openrouter_api_key: key }
        : { gemini_api_key: key };
    const res = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    }).catch(() => null);
    if (!res?.ok) {
        if (status) status.textContent = "Could not save key. You can add it later in Settings.";
        return;
    }
    const input = document.getElementById("first-run-key");
    if (input) input.value = "";
    if (status) status.textContent = "Saved locally.";
    _setFirstRunStep(2);
}

function _initFirstRunWizard() {
    document.getElementById("first-run-start")?.addEventListener("click", () => _setFirstRunStep(1));
    document.getElementById("first-run-later")?.addEventListener("click", () => _closeFirstRunWizard(true));
    document.getElementById("first-run-skip-key")?.addEventListener("click", () => _setFirstRunStep(2));
    document.getElementById("first-run-save-key")?.addEventListener("click", _saveFirstRunKey);
    document.getElementById("first-run-finish")?.addEventListener("click", () => {
        _closeFirstRunWizard(true);
        switchView("chat").catch((err) => console.error("Failed to open Chat", err));
    });
    document.getElementById("first-run-open-settings")?.addEventListener("click", () => {
        _closeFirstRunWizard(true);
        switchView("settings").catch((err) => console.error("Failed to open Settings", err));
    });
}

async function switchView(viewName) {
    navItems.forEach((item) => {
        const isActive = item.dataset.view === viewName;
        item.classList.toggle("active", isActive);
        if (isActive) {
            item.setAttribute("aria-current", "page");
            const parentDetails = item.closest("details");
            if (parentDetails) parentDetails.open = true;
        } else {
            item.removeAttribute("aria-current");
        }
    });
    views.forEach((view) => {
        view.classList.toggle("active", view.id === `view-${viewName}`);
    });

    document.querySelector(".app")?.classList.remove("panel-open");
    if (!_moduleCache.has(viewName)) {
        _showViewSkeleton(viewName);
    }

    if (viewName === "memory") await _loadMemoryView();
    if (viewName === "tasks") await _loadTasksView();
    if (viewName === "profile") await _loadProfileView();
    if (viewName === "stats") await _loadStatsView();
    if (viewName === "graph") await loadGraph();
    if (viewName === "settings") await _loadSettingsView();
    if (viewName === "history") await _loadHistoryView();
    if (viewName === "activity") await _loadActivityView();
    if (viewName === "reliability") await _loadReliabilityView();


    if (viewName === "documents") await loadDocuments();
    if (viewName === "calendar") await loadCalendar();
    if (viewName === "pipelines")    await loadPipelines();
    if (viewName === "experiments")  await _loadExperimentsView();
    if (viewName === "automations")  await _loadAutomationsView();
    if (viewName === "glass-brain") await _loadGlassBrainView();

    if (viewName !== "glass-brain") _stopGlassBrainRefresh();

    if (viewName !== "activity") {
        window.apiClient?.disconnectActivity?.();
    }
    if (viewName !== "stats") {
        _stopStatsRefresh();
    }


}

navItems.forEach((item) => {
    item.setAttribute("role", "button");
    item.setAttribute("tabindex", "0");
    item.addEventListener("click", () => {
        switchView(item.dataset.view).catch((err) => {
            console.error(`Failed to switch view '${item.dataset.view}'`, err);
        });
    });
    item.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        item.click();
    });
});

newSessionBtn?.addEventListener("click", async () => {
    newSessionBtn.disabled = true;
    try {
        const result = await window.apiClient.createConversation();
        _emitConversationChanged(result.conversation);
        await loadConversations({ emitActive: false });
        await switchView("chat");
    } catch (err) {
        console.error("Failed to create conversation", err);
    } finally {
        newSessionBtn.disabled = false;
    }
});

stopRemyBtn?.addEventListener("click", async () => {
    const confirmed = await showConfirm(
        "Stop Remy",
        "Finish active work, save memory, and gracefully stop the local Remy server?"
    );
    if (!confirmed) return;

    try {
        stopRemyBtn.disabled = true;
        stopRemyBtn.textContent = "Stopping...";
        document.dispatchEvent(new CustomEvent("server-shutdown-started"));
        const result = await window.apiClient.shutdownServer();
        if (!result?.ok) {
            throw new Error(result?.message || "Graceful shutdown was not accepted");
        }
    } catch (err) {
        console.error("Failed to stop Remy", err);
        document.dispatchEvent(new CustomEvent("server-shutdown-failed"));
        stopRemyBtn.disabled = false;
        stopRemyBtn.textContent = "Stop Remy";
    }
});

closePanelBtn?.addEventListener("click", () => {
    document.querySelector(".app")?.classList.remove("panel-open");
});

const banner = document.getElementById("connection-banner");
const bannerText = document.getElementById("banner-text");
const reconnectBtn = document.getElementById("btn-reconnect");

function _handleTransportStatus(status) {
    statusEl?.classList.remove("connected", "disconnected", "reconnecting");

    if (status === "connected") {
        statusEl?.classList.add("connected");
        if (statusText) statusText.textContent = "Connected";
        banner?.classList.add("hidden");
        setStartupStatus("Connection established. Opening workspace...");
        window.setTimeout(hideStartupSplash, 250);
    } else if (status === "reconnecting") {
        statusEl?.classList.add("reconnecting");
        if (statusText) statusText.textContent = "Reconnecting...";
        if (bannerText) bannerText.textContent = "Disconnected. Reconnecting...";
        banner?.classList.remove("hidden");
        if (!startupSplashHidden) {
            setStartupStatus("Backend is waking up. Retrying connection...");
        }
    } else if (status === "failed") {
        statusEl?.classList.add("disconnected");
        if (statusText) statusText.textContent = "Disconnected";
        if (bannerText) bannerText.textContent = "Connection lost. Could not reconnect.";
        banner?.classList.remove("hidden");
        if (!startupSplashHidden) {
            setStartupStatus("Still waiting for the server response...");
        }
    } else {
        statusEl?.classList.add("disconnected");
        if (statusText) statusText.textContent = "Disconnected";
        if (!startupSplashHidden) {
            setStartupStatus("Starting chat transport...");
        }
    }
}

async function _bootstrapTransport() {
    const apiClient = await _waitForApiClient();
    if (!apiClient) {
        console.error("apiClient did not initialize before app boot timeout");
        if (!startupSplashHidden) {
            setStartupStatus("Chat transport failed to initialize. Opening workspace in degraded mode...");
        }
        return;
    }

    apiClient.onStatus(_handleTransportStatus);

    reconnectBtn?.addEventListener("click", () => {
        if (bannerText) bannerText.textContent = "Reconnecting...";
        apiClient.manualReconnect();
    });

    apiClient.connectChat();
}

document.addEventListener("server-shutdown-started", () => {
    statusEl?.classList.remove("connected", "disconnected", "reconnecting");
    statusEl?.classList.add("disconnected");
    if (statusText) statusText.textContent = "Shutting down...";
    if (bannerText) {
        bannerText.textContent = "Shutting down gracefully... Do not press Ctrl+C again or close the terminal until the process fully exits.";
    }
    banner?.classList.remove("hidden");
    hideStartupSplash();
});

document.addEventListener("server-shutdown-failed", () => {
    if (bannerText) bannerText.textContent = "Failed to request graceful shutdown.";
    banner?.classList.remove("hidden");
});

const llmBanner = document.getElementById("llm-banner");
const llmBannerText = document.getElementById("llm-banner-text");
const llmCountdown = document.getElementById("llm-banner-countdown");
const llmDismissBtn = document.getElementById("btn-llm-dismiss");

let _llmCountdownInterval = null;
let _llmRecoveryTarget = null;
let _llmCurrentEstimate = 0;

function _formatCountdown(sec) {
    if (sec < 60) return `~${sec}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `~${m}m ${s}s`;
}

function _startLlmCountdown(estimateSeconds) {
    _stopLlmCountdown();
    _llmCurrentEstimate = estimateSeconds;
    _llmRecoveryTarget = Date.now() + estimateSeconds * 1000;

    _llmCountdownInterval = setInterval(() => {
        const remaining = Math.max(0, Math.ceil((_llmRecoveryTarget - Date.now()) / 1000));
        if (llmCountdown) {
            if (remaining > 0) {
                llmCountdown.textContent = `Retry in ${_formatCountdown(remaining)}`;
            } else {
                llmCountdown.textContent = "Checking...";
                _stopLlmCountdown();
                document.dispatchEvent(new CustomEvent("llm-probe-request"));
            }
        }
    }, 1000);
}

function _stopLlmCountdown() {
    if (_llmCountdownInterval) {
        clearInterval(_llmCountdownInterval);
        _llmCountdownInterval = null;
    }
}

document.addEventListener("llm-status-change", (e) => {
    if (!llmBanner) return;
    if (!e.detail.available) {
        llmBanner.classList.remove("hidden");
        const errorClass = e.detail.errorClass || "unknown";
        const estimateSeconds = e.detail.estimateSeconds || 60;

        if (errorClass === "auth") {
            if (llmBannerText) {
                llmBannerText.textContent = "API authentication error. Check your API key in Settings.";
            }
            if (llmCountdown) llmCountdown.textContent = "";
            _stopLlmCountdown();
        } else {
            if (llmBannerText) {
                llmBannerText.textContent = "LLM unavailable. You can still browse memory, tasks, and knowledge.";
            }
            const nextEstimate = _llmCurrentEstimate > 0
                ? Math.min(_llmCurrentEstimate * 2, 300)
                : estimateSeconds;
            _startLlmCountdown(nextEstimate);
        }
    } else {
        llmBanner.classList.add("hidden");
        _stopLlmCountdown();
        _llmCurrentEstimate = 0;
        if (llmCountdown) llmCountdown.textContent = "";
    }
});

llmDismissBtn?.addEventListener("click", () => llmBanner?.classList.add("hidden"));

document.addEventListener("graph-node-selected", (e) => {
    const id = e.detail.id;
    if (!id) return;
    _loadModule("memory", () => import("./memory.js?v=1.22"))
        .then((mod) => mod.openRecordDetail?.(id))
        .catch((err) => console.error("Failed to open graph node detail", err));
});

const menuBtn = document.getElementById("btn-menu");
const appRoot = document.querySelector(".app");
const sidebar = document.querySelector(".sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const sidebarResizeHandle = document.getElementById("sidebar-resize-handle");

let _sidebarWidth = SIDEBAR_DEFAULT_WIDTH;
let _sidebarResizePointerId = null;
let _sidebarResizeFrame = 0;
let _sidebarResizePendingWidth = null;

function _clampSidebarWidth(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return SIDEBAR_DEFAULT_WIDTH;
    return Math.round(Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, parsed)));
}

function _setSidebarWidth(value, { persist = false, announce = false } = {}) {
    _sidebarWidth = _clampSidebarWidth(value);
    document.documentElement.style.setProperty("--sidebar-width", `${_sidebarWidth}px`);
    sidebarResizeHandle?.setAttribute("aria-valuenow", String(_sidebarWidth));
    sidebarResizeHandle?.setAttribute("aria-valuetext", `${_sidebarWidth} pixels`);
    if (persist) {
        try {
            window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(_sidebarWidth));
        } catch (err) {
            console.debug("Could not save sidebar width", err);
        }
    }
    if (announce) {
        document.dispatchEvent(new CustomEvent("remy-sidebar-resized", {
            detail: { width: _sidebarWidth },
        }));
    }
    return _sidebarWidth;
}

function _savedSidebarWidth() {
    try {
        const saved = window.localStorage.getItem(SIDEBAR_WIDTH_KEY);
        return saved === null || saved.trim() === ""
            ? SIDEBAR_DEFAULT_WIDTH
            : _clampSidebarWidth(saved);
    } catch (err) {
        console.debug("Could not read sidebar width", err);
        return SIDEBAR_DEFAULT_WIDTH;
    }
}

function _flushSidebarResizeFrame() {
    _sidebarResizeFrame = 0;
    if (_sidebarResizePendingWidth === null) return;
    _setSidebarWidth(_sidebarResizePendingWidth);
    _sidebarResizePendingWidth = null;
}

function _queueSidebarResize(width) {
    _sidebarResizePendingWidth = width;
    if (_sidebarResizeFrame) return;
    _sidebarResizeFrame = window.requestAnimationFrame(_flushSidebarResizeFrame);
}

function _finishSidebarResize(event) {
    if (_sidebarResizePointerId === null) return;
    if (event?.pointerId !== undefined && event.pointerId !== _sidebarResizePointerId) return;
    if (_sidebarResizeFrame) {
        window.cancelAnimationFrame(_sidebarResizeFrame);
        _flushSidebarResizeFrame();
    }
    const completedPointerId = _sidebarResizePointerId;
    _sidebarResizePointerId = null;
    if (sidebarResizeHandle?.hasPointerCapture?.(completedPointerId)) {
        sidebarResizeHandle.releasePointerCapture(completedPointerId);
    }
    appRoot?.classList.remove("sidebar-resizing");
    _setSidebarWidth(_sidebarWidth, { persist: true, announce: true });
}

function _initSidebarResize() {
    _setSidebarWidth(_savedSidebarWidth());
    if (!sidebarResizeHandle) return;

    sidebarResizeHandle.addEventListener("pointerdown", (event) => {
        if (!window.matchMedia(SIDEBAR_DESKTOP_QUERY).matches) return;
        if (event.button !== 0) return;
        event.preventDefault();
        _sidebarResizePointerId = event.pointerId;
        sidebarResizeHandle.setPointerCapture?.(event.pointerId);
        appRoot?.classList.add("sidebar-resizing");
    });
    sidebarResizeHandle.addEventListener("pointermove", (event) => {
        if (event.pointerId !== _sidebarResizePointerId) return;
        const appLeft = appRoot?.getBoundingClientRect().left || 0;
        _queueSidebarResize(event.clientX - appLeft);
    });
    sidebarResizeHandle.addEventListener("pointerup", _finishSidebarResize);
    sidebarResizeHandle.addEventListener("pointercancel", _finishSidebarResize);
    sidebarResizeHandle.addEventListener("lostpointercapture", _finishSidebarResize);
    sidebarResizeHandle.addEventListener("dblclick", () => {
        _setSidebarWidth(SIDEBAR_DEFAULT_WIDTH, { persist: true, announce: true });
    });
    sidebarResizeHandle.addEventListener("keydown", (event) => {
        const step = event.shiftKey ? 24 : 8;
        let nextWidth = null;
        if (event.key === "ArrowLeft") nextWidth = _sidebarWidth - step;
        if (event.key === "ArrowRight") nextWidth = _sidebarWidth + step;
        if (event.key === "Home") nextWidth = SIDEBAR_MIN_WIDTH;
        if (event.key === "End") nextWidth = SIDEBAR_MAX_WIDTH;
        if (nextWidth === null) return;
        event.preventDefault();
        _setSidebarWidth(nextWidth, { persist: true, announce: true });
    });
}

_initSidebarResize();

function openSidebar() {
    sidebar?.classList.add("open");
    sidebarOverlay?.classList.add("open");
}

function closeSidebar() {
    sidebar?.classList.remove("open");
    sidebarOverlay?.classList.remove("open");
}

menuBtn?.addEventListener("click", openSidebar);
sidebarOverlay?.addEventListener("click", closeSidebar);

navItems.forEach((item) => {
    item.addEventListener("click", closeSidebar);
});

_bootstrapTransport().catch((err) => {
    console.error("Failed to bootstrap chat transport", err);
    if (!startupSplashHidden) {
        setStartupStatus("Chat transport failed to start.");
    }
});

function _initDeferredSurfaces() {
    _initHumanLoopSurfaces().catch((err) => {
        console.error("Failed to initialize approval/guidance surfaces", err);
    });
}

switchView("chat").catch((err) => console.error("Failed to open Chat", err));
initProjectSwitcher().then(async () => {
    const warning = window.sessionStorage.getItem("remy_project_compatibility_warning");
    if (!warning) return;
    window.sessionStorage.removeItem("remy_project_compatibility_warning");
    await openProjectManager();
    _setProjectManagerStatus(warning, true);
});
loadConversations();
_initFirstRunWizard();
_initOperatorAlertCenter();
_maybeShowFirstRunWizard().catch((err) => console.error("Failed to open first-run wizard", err));

if ("requestIdleCallback" in window) {
    window.requestIdleCallback(_initDeferredSurfaces, { timeout: 3000 });
} else {
    window.setTimeout(_initDeferredSurfaces, 1500);
}

setInterval(async () => {
    try {
        await fetch("/api/ping");
    } catch (e) {
        console.debug("Heartbeat failed", e);
    }
}, 60000);

setStartupStatus("Checking server readiness...");
fetch("/api/ping")
    .then(() => {
        setStartupStatus("Server is online. Finalizing interface...");
        window.setTimeout(hideStartupSplash, 400);
    })
    .catch((e) => {
        console.debug(e);
        if (!startupSplashHidden) {
            setStartupStatus("Server is still starting. Waiting for connection...");
        }
    });

window.setTimeout(() => {
    if (!startupSplashHidden) {
        setStartupStatus("Almost ready...");
    }
}, 4000);

window.setTimeout(() => {
    if (!startupSplashHidden) {
        console.warn("Splash timeout - forcing hide after 6s");
        hideStartupSplash();
    }
}, 6000);
