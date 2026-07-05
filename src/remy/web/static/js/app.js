/**
 * App Controller - navigation, init, event wiring.
 */

import { skeletonCards, skeletonGrid } from "./ui.js?v=1.21";

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
    const mod = await _loadModule("settings", () => import("./settings.js?v=1.22"));
    await mod.loadSettings();
}

async function _loadHistoryView() {
    const mod = await _loadModule("history", () => import("./history.js?v=1.22"));
    await mod.loadHistory();
}

async function _loadActivityView() {
    const mod = await _loadModule("activity", () => import("./activity.js?v=1.22"));
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
    const mod = await _loadModule("documents", () => import("./documents.js?v=1.22"));
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

async function _loadAutomationsView() {
    const mod = await _loadModule("automations", () => import("./automations.js?v=2.8"));
    await mod.loadAutomations?.();
}

let _glassBrainMod = null;
async function _loadGlassBrainView() {
    if (!_glassBrainMod) {
        _glassBrainMod = await import("./glass_brain.js?v=1.2");
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
const closePanelBtn = document.getElementById("btn-close-panel");
const statusEl = document.getElementById("connection-status");
const statusText = statusEl?.querySelector(".status-text");
const startupSplash = document.getElementById("startup-splash");
const startupSplashStatus = document.getElementById("startup-splash-status");
const HOME_HISTORY_STORAGE_KEY = "remy_home_workflow_history_v1";
const FIRST_RUN_DONE_KEY = "remy_first_run_done_v1";
const DEFAULT_HOME_TEMPLATES = [
    {
        id: "summarize-document",
        title: "Summarize Document",
        pack: "Document Pack",
        icon: "DOC",
        description: "Import a local file, summarize it, extract action items, and keep a report.",
        fields: [
            { id: "source", label: "Document name", placeholder: "meeting-notes.md" },
            { id: "goal", label: "Focus", placeholder: "decisions, risks, deadlines" },
        ],
        steps: ["Document intake", "Evidence-bounded summary", "Action items", "Report preview"],
        targetView: "documents",
    },
    {
        id: "daily-brief",
        title: "Create Daily Brief",
        pack: "Personal Admin Pack",
        icon: "DAY",
        description: "Collect tasks, memory, and recent notes into a local morning brief.",
        fields: [
            { id: "time", label: "Run time", placeholder: "09:00" },
            { id: "scope", label: "Brief scope", placeholder: "tasks, reminders, decisions" },
        ],
        steps: ["Search tasks", "Search memory", "Draft brief", "Require approval before schedule"],
        targetView: "automations",
    },
    {
        id: "extract-deadlines",
        title: "Extract Deadlines",
        pack: "Document Pack",
        icon: "DATE",
        description: "Find dates, commitments, and reminder candidates from pasted or imported text.",
        fields: [
            { id: "source", label: "Source name", placeholder: "contract.txt" },
            { id: "notify", label: "Output", placeholder: "Run history, task list, Telegram later" },
        ],
        steps: ["Read source", "Detect dates", "Mark unverified reminders", "Dry-run notification"],
        targetView: "documents",
    },
    {
        id: "research-topic",
        title: "Research Topic",
        pack: "Research Pack",
        icon: "SRC",
        description: "Turn a research question into a source-backed finding set and saved notes.",
        fields: [
            { id: "topic", label: "Topic", placeholder: "local-first AI automation" },
            { id: "question", label: "Decision question", placeholder: "what should we build first?" },
        ],
        steps: ["Plan queries", "Collect sources", "Synthesize findings", "Save reviewable memory"],
        targetView: "chat",
    },
    {
        id: "monitor-website",
        title: "Monitor Website",
        pack: "Research Pack",
        icon: "URL",
        description: "Check a URL on a schedule, compare changes, and pause on failures.",
        fields: [
            { id: "url", label: "URL", placeholder: "https://example.com/changelog" },
            { id: "cadence", label: "Cadence", placeholder: "daily" },
        ],
        steps: ["Fetch page", "Compare with previous run", "Summarize change", "Auto-pause on failure"],
        targetView: "automations",
    },
    {
        id: "save-memory",
        title: "Save Memory",
        pack: "Memory Pack",
        icon: "MEM",
        description: "Save a note as a reviewable memory candidate without treating generated text as fact.",
        fields: [
            { id: "memory", label: "Memory candidate", placeholder: "Project fact, preference, decision" },
            { id: "source", label: "Source", placeholder: "operator note" },
        ],
        steps: ["Create candidate", "Mark source class", "Keep generated text unverified", "Queue for admission"],
        targetView: "memory",
    },
];

let HOME_TEMPLATES = DEFAULT_HOME_TEMPLATES;
let startupSplashHidden = false;
let selectedHomeTemplateId = HOME_TEMPLATES[0].id;
let homeRunHistory = [];
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

function _homeTemplate() {
    return HOME_TEMPLATES.find((template) => template.id === selectedHomeTemplateId) || HOME_TEMPLATES[0];
}

async function _loadHomeTemplates() {
    try {
        const res = await fetch("/api/pipelines/home-templates/list");
        if (!res.ok) return;
        const data = await res.json();
        const templates = data.templates || [];
        if (Array.isArray(templates) && templates.length) {
            HOME_TEMPLATES = templates;
            if (!HOME_TEMPLATES.some((template) => template.id === selectedHomeTemplateId)) {
                selectedHomeTemplateId = HOME_TEMPLATES[0].id;
            }
        }
    } catch {
        HOME_TEMPLATES = DEFAULT_HOME_TEMPLATES;
    }
}

function _escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

async function _loadHomeHistory() {
    try {
        const res = await fetch("/api/pipelines/home-templates/runs?limit=20");
        if (res.ok) {
            const data = await res.json();
            homeRunHistory = (data.runs || []).map((run) => ({
                id: run.run_id,
                templateId: run.template_id,
                title: run.title || run.workflow_name || "Template run",
                pack: run.pack || "",
                mode: run.mode || run.trigger || "",
                status: run.status || "",
                durationMs: run.duration_ms ?? 0,
                cost: run.cost || "$0.00 local",
                retryCount: run.retry_count ?? 0,
                autoPaused: !!run.auto_paused,
                createdAt: run.created_at || run.started_at || "",
                preview: run.preview || run.output_preview || "",
                inputs: run.inputs || {},
            }));
            return;
        }
    } catch {
        // Fall back to local history below.
    }
    try {
        const raw = window.localStorage.getItem(HOME_HISTORY_STORAGE_KEY);
        homeRunHistory = raw ? JSON.parse(raw) : [];
        if (!Array.isArray(homeRunHistory)) homeRunHistory = [];
    } catch {
        homeRunHistory = [];
    }
}

function _saveHomeHistory() {
    window.localStorage.setItem(HOME_HISTORY_STORAGE_KEY, JSON.stringify(homeRunHistory.slice(0, 20)));
}

function renderHomeTemplates() {
    const grid = document.getElementById("home-template-grid");
    if (!grid) return;
    grid.innerHTML = HOME_TEMPLATES.map((template) => `
        <button class="home-template-card ${template.id === selectedHomeTemplateId ? "active" : ""}" type="button" data-home-template="${_escapeHtml(template.id)}">
            <span class="home-template-pack">${_escapeHtml(template.pack)}</span>
            <span class="home-template-icon">${_escapeHtml(template.icon)}</span>
            <strong>${_escapeHtml(template.title)}</strong>
            <span>${_escapeHtml(template.description)}</span>
        </button>
    `).join("");
    grid.querySelectorAll("[data-home-template]").forEach((button) => {
        button.addEventListener("click", () => {
            selectedHomeTemplateId = button.dataset.homeTemplate;
            renderHomeTemplates();
            renderSelectedHomeTemplate();
        });
    });
}

function renderSelectedHomeTemplate() {
    const template = _homeTemplate();
    const selected = document.getElementById("home-selected-template");
    const fields = document.getElementById("home-template-fields");
    if (!selected || !fields || !template) return;
    selected.innerHTML = `
        <div class="home-selected-pack">${_escapeHtml(template.pack)}</div>
        <h4>${_escapeHtml(template.title)}</h4>
        <p>${_escapeHtml(template.description)}</p>
        <ol>${template.steps.map((step) => `<li>${_escapeHtml(step)}</li>`).join("")}</ol>
    `;
    fields.innerHTML = template.fields.map((field) => `
        <label class="home-template-field">
            <span>${_escapeHtml(field.label)}</span>
            <input type="text" data-home-field="${_escapeHtml(field.id)}" placeholder="${_escapeHtml(field.placeholder)}">
        </label>
    `).join("");
}

function _homeTemplateInputs() {
    const values = {};
    document.querySelectorAll("[data-home-field]").forEach((input) => {
        values[input.dataset.homeField] = input.value.trim();
    });
    return values;
}

function _localHomeRunRecord(template, inputs, mode) {
    return {
        id: `${template.id}-${Date.now()}`,
        templateId: template.id,
        title: template.title,
        pack: template.pack,
        mode,
        status: mode === "dry_run" ? "dry_run_ready" : "queued_for_human_approval",
        durationMs: 0,
        cost: "$0.00 local",
        retryCount: 0,
        autoPaused: false,
        createdAt: new Date().toISOString(),
        preview: template.steps.join(" -> "),
        inputs,
    };
}

async function runHomeTemplate(mode) {
    const template = _homeTemplate();
    const inputs = _homeTemplateInputs();
    const missing = template.fields.find((field) => !inputs[field.id]);
    if (missing) {
        alert(`Fill required field: ${missing.label}`);
        return;
    }
    let record = _localHomeRunRecord(template, inputs, mode);
    try {
        const res = await fetch("/api/pipelines/home-templates/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                template_id: template.id,
                title: template.title,
                pack: template.pack,
                mode,
                inputs,
                steps: template.steps,
            }),
        });
        if (res.ok) {
            const data = await res.json();
            const run = data.run || {};
            record = {
                id: run.run_id,
                templateId: run.template_id,
                title: run.title || template.title,
                pack: run.pack || template.pack,
                mode: run.mode || mode,
                status: run.status || record.status,
                durationMs: run.duration_ms ?? 0,
                cost: run.cost || "$0.00 local",
                retryCount: run.retry_count ?? 0,
                autoPaused: !!run.auto_paused,
                createdAt: run.created_at || new Date().toISOString(),
                preview: run.preview || record.preview,
                inputs: run.inputs || inputs,
            };
        }
    } catch {
        // Keep local fallback record.
    }
    homeRunHistory = [record, ...homeRunHistory].slice(0, 20);
    _saveHomeHistory();
    renderHomeHistory();
    if (mode === "run") {
        const targetView = await _instantiateHomeTemplate(template, inputs);
        switchView(targetView).catch((err) => console.error("Failed to open template target", err));
    }
}

async function _instantiateHomeTemplate(template, inputs = {}) {
    const isAutomation = template.targetView === "automations";
    const url = isAutomation
        ? `/api/automations/templates/${encodeURIComponent(template.id)}/instantiate`
        : `/api/pipelines/templates/${encodeURIComponent(template.id)}/instantiate`;
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            name: template.title,
            description: template.description || "",
            enabled: false,
            inputs,
        }),
    }).catch(() => null);
    if (!res?.ok) {
        alert("Template was recorded, but the workflow could not be created. Open Templates and try again.");
        return template.targetView || "pipelines";
    }
    const data = await res.json().catch(() => ({}));
    if (isAutomation && data.automation?.id) {
        window.sessionStorage.setItem("remy_pending_automation_open", data.automation.id);
    } else if (!isAutomation && data.pipeline?.id) {
        window.sessionStorage.setItem("remy_pending_pipeline_open", data.pipeline.id);
    }
    return isAutomation ? "automations" : "pipelines";
}

function renderHomeHistory() {
    const list = document.getElementById("home-run-history");
    if (!list) return;
    if (homeRunHistory.length === 0) {
        list.innerHTML = `
            <div class="home-empty-history">
                No workflow runs yet. Dry run a template before enabling scheduled automation.
            </div>
        `;
        return;
    }
    list.innerHTML = homeRunHistory.map((run) => `
        <div class="home-history-item">
            <div class="home-history-topline">
                <strong>${_escapeHtml(run.title)}</strong>
                <span class="home-run-status">${_escapeHtml(run.status)}</span>
            </div>
            <div class="home-history-meta">
                <span>${_escapeHtml(new Date(run.createdAt).toLocaleString())}</span>
                <span>${_escapeHtml(run.durationMs)}ms</span>
                <span>${_escapeHtml(run.cost)}</span>
                <span>retries=${_escapeHtml(run.retryCount)}</span>
            </div>
            <p>${_escapeHtml(run.preview)}</p>
        </div>
    `).join("");
}

async function initHomeSurface() {
    await _loadHomeTemplates();
    await _loadHomeHistory();
    renderHomeTemplates();
    renderSelectedHomeTemplate();
    renderHomeHistory();
    document.getElementById("home-template-dry-run")?.addEventListener("click", () => runHomeTemplate("dry_run"));
    document.getElementById("home-template-run")?.addEventListener("click", () => runHomeTemplate("run"));
    document.getElementById("home-clear-history")?.addEventListener("click", async () => {
        try {
            await fetch("/api/pipelines/home-templates/runs", { method: "DELETE" });
        } catch {
            // Local fallback below still clears the visible history.
        }
        homeRunHistory = [];
        _saveHomeHistory();
        renderHomeHistory();
    });
    document.getElementById("home-open-pipelines")?.addEventListener("click", () => {
        switchView("pipelines").catch((err) => console.error("Failed to open pipelines", err));
    });
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
        switchView("home").catch((err) => console.error("Failed to open Home", err));
    });
    document.getElementById("first-run-open-settings")?.addEventListener("click", () => {
        _closeFirstRunWizard(true);
        switchView("settings").catch((err) => console.error("Failed to open Settings", err));
    });
}

async function switchView(viewName) {
    navItems.forEach((item) => {
        item.classList.toggle("active", item.dataset.view === viewName);
    });
    views.forEach((view) => {
        view.classList.toggle("active", view.id === `view-${viewName}`);
    });

    document.querySelector(".app")?.classList.remove("panel-open");
    if (!_moduleCache.has(viewName)) {
        _showViewSkeleton(viewName);
    }

    if (viewName === "home") renderHomeHistory();
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
    item.addEventListener("click", () => {
        switchView(item.dataset.view).catch((err) => {
            console.error(`Failed to switch view '${item.dataset.view}'`, err);
        });
    });
});

newSessionBtn?.addEventListener("click", () => {
    window.apiClient?.sendNewSession?.();
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
const sidebar = document.querySelector(".sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");

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

initHomeSurface();
_initFirstRunWizard();
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
