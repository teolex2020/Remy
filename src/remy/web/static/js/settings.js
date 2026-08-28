/**
 * Settings View — configuration, model registry, system prompt, diagnostics.
 */

import { showConfirm } from "./ui.js";

const contentEl = document.getElementById("settings-content");

// Cached registered models — used to auto-fill API key when adding a new model
let _cachedRegisteredModels = [];
let _activeSettingsCategory = localStorage.getItem("remy.settings.category") || "overview";

const SETTINGS_CATEGORIES = [
    {
        id: "overview",
        icon: "◎",
        label: "Overview",
        description: "Runtime health and installed memory version",
        sections: ["System Status", "Aura Memory"],
    },
    {
        id: "models",
        icon: "AI",
        label: "AI & Models",
        description: "Providers, cloud models and local GGUF runtime",
        sections: ["Local Secrets", "Models", "Local Models (llama.cpp)"],
    },
    {
        id: "personalization",
        icon: "✦",
        label: "Personalization",
        description: "Instructions, theme and voice",
        sections: ["Custom Instructions", "Appearance & Voice"],
    },
    {
        id: "connections",
        icon: "↗",
        label: "Connections",
        description: "Telegram, email and push delivery",
        sections: ["Integrations"],
    },
    {
        id: "workspace",
        icon: "▣",
        label: "Workspace & Data",
        description: "Folder permissions, import and export",
        sections: ["Local Workspaces", "Data"],
    },
];

function setActiveSettingsCategory(categoryId, { focus = false } = {}) {
    const category = SETTINGS_CATEGORIES.find((item) => item.id === categoryId)
        || SETTINGS_CATEGORIES[0];
    _activeSettingsCategory = category.id;
    localStorage.setItem("remy.settings.category", category.id);
    contentEl.querySelectorAll("[data-settings-category]").forEach((button) => {
        const active = button.dataset.settingsCategory === category.id;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
        button.tabIndex = active ? 0 : -1;
        if (active && focus) button.focus();
    });
    contentEl.querySelectorAll("[data-settings-panel]").forEach((panel) => {
        panel.hidden = panel.dataset.settingsPanel !== category.id;
    });
    const title = contentEl.querySelector("#settings-panel-title");
    const description = contentEl.querySelector("#settings-panel-description");
    if (title) title.textContent = category.label;
    if (description) description.textContent = category.description;
    contentEl.querySelector(".settings-hub-main")?.scrollTo({ top: 0, behavior: "auto" });
}

function buildSettingsNavigation() {
    const sections = [...contentEl.querySelectorAll(":scope > .settings-section")];
    if (!sections.length) return;
    const byTitle = new Map(sections.map((section) => [
        section.querySelector(".settings-section-title")?.textContent.trim() || "",
        section,
    ]));
    const hub = document.createElement("div");
    hub.className = "settings-hub";
    hub.innerHTML = `
        <nav class="settings-hub-nav" aria-label="Settings categories">
            <div class="settings-hub-nav-heading"><strong>Settings</strong><span>Choose one area to configure</span></div>
            <div class="settings-hub-tabs" role="tablist" aria-orientation="vertical">
                ${SETTINGS_CATEGORIES.map((category) => `<button type="button" role="tab" data-settings-category="${category.id}" aria-controls="settings-panel-${category.id}"><i>${category.icon}</i><span><b>${category.label}</b><small>${category.description}</small></span></button>`).join("")}
            </div>
        </nav>
        <main class="settings-hub-main">
            <header class="settings-hub-panel-header"><div><span>Settings</span><h3 id="settings-panel-title"></h3><p id="settings-panel-description"></p></div></header>
            <div class="settings-hub-panels"></div>
        </main>`;
    const panels = hub.querySelector(".settings-hub-panels");
    SETTINGS_CATEGORIES.forEach((category) => {
        const panel = document.createElement("section");
        panel.id = `settings-panel-${category.id}`;
        panel.className = "settings-hub-panel";
        panel.dataset.settingsPanel = category.id;
        panel.setAttribute("role", "tabpanel");
        panel.setAttribute("aria-label", category.label);
        category.sections.forEach((title) => {
            const section = byTitle.get(title);
            if (section) panel.append(section);
        });
        panels.append(panel);
    });
    contentEl.prepend(hub);
    const valid = SETTINGS_CATEGORIES.some((item) => item.id === _activeSettingsCategory);
    setActiveSettingsCategory(valid ? _activeSettingsCategory : "overview");
    const tabs = [...hub.querySelectorAll("[data-settings-category]")];
    tabs.forEach((button, index) => {
        button.addEventListener("click", () => {
            setActiveSettingsCategory(button.dataset.settingsCategory);
        });
        button.addEventListener("keydown", (event) => {
            const keys = ["ArrowDown", "ArrowRight", "ArrowUp", "ArrowLeft", "Home", "End"];
            if (!keys.includes(event.key)) return;
            event.preventDefault();
            let next = index;
            if (["ArrowDown", "ArrowRight"].includes(event.key)) next = (index + 1) % tabs.length;
            if (["ArrowUp", "ArrowLeft"].includes(event.key)) next = (index - 1 + tabs.length) % tabs.length;
            if (event.key === "Home") next = 0;
            if (event.key === "End") next = tabs.length - 1;
            setActiveSettingsCategory(tabs[next].dataset.settingsCategory, { focus: true });
        });
    });
}

export async function loadSettings() {
    contentEl.innerHTML = `
        <div class="skeleton-card" style="padding:20px;margin-bottom:12px">
            <div class="skeleton skeleton-line skeleton-short" style="margin-bottom:14px"></div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
                ${Array.from({length: 6}, () => `
                    <div class="skeleton-card" style="padding:12px">
                        <div class="skeleton skeleton-line skeleton-short" style="margin-bottom:6px"></div>
                        <div class="skeleton skeleton-line skeleton-medium"></div>
                    </div>`).join('')}
            </div>
        </div>
        <div class="skeleton-card" style="padding:20px;margin-bottom:12px">
            <div class="skeleton skeleton-line skeleton-medium" style="margin-bottom:14px"></div>
            <div class="skeleton skeleton-block" style="height:120px;border-radius:6px"></div>
        </div>
        <div class="skeleton-card" style="padding:20px">
            <div class="skeleton skeleton-line skeleton-short" style="margin-bottom:14px"></div>
            <div class="skeleton skeleton-block" style="height:200px;border-radius:6px"></div>
        </div>`;
    try {
        const [settingsData, diagData, secretsData, workspacesData] = await Promise.all([
            fetch("/api/settings").then((r) => r.json()),
            fetch("/api/diagnostics").then((r) => r.json()),
            fetch("/api/secrets").then((r) => r.json()),
            fetch("/api/workspaces").then((r) => r.json()),
        ]);
        renderSettings(settingsData, diagData, secretsData, workspacesData);
    } catch (e) {
        contentEl.innerHTML = `<p style="color:var(--red)">Failed to load settings: ${e.message}</p>`;
    }
}

function renderSettings(cfg, diag, secretsData = { secrets: [] }, workspacesData = { workspaces: [] }) {
    const statusColor = diag.status === "ok" ? "var(--green)" : "var(--yellow)";

    contentEl.innerHTML = `
        <!-- System Status -->
        <div class="settings-section">
            <h3 class="settings-section-title">System Status</h3>
            <div class="diag-grid">
                <div class="diag-item">
                    <span class="diag-label">Status</span>
                    <span class="diag-value" style="color:${statusColor}">${diag.status === "ok" ? "Running" : diag.status.toUpperCase()}</span>
                </div>
                <div class="diag-item">
                    <span class="diag-label">Uptime</span>
                    <span class="diag-value">${diag.uptime}</span>
                </div>
                <div class="diag-item">
                    <span class="diag-label">Model</span>
                    <span class="diag-value">${diag.model}</span>
                </div>
            </div>
        </div>

        <!-- Local Secrets -->
        <div class="settings-section">
            <h3 class="settings-section-title">Local Secrets</h3>
            <p class="settings-hint" style="margin-bottom:12px">
                API keys and tokens stay on this computer. Remy only uses a secret when a model or workflow block needs it.
            </p>
            <div id="local-secrets-list" class="settings-secrets-list">
                ${renderLocalSecrets(secretsData.secrets || [])}
            </div>
        </div>

        <!-- Model Registry -->
        <div class="settings-section">
            <h3 class="settings-section-title">Models</h3>
            <p class="settings-hint" style="margin-bottom:12px">Add models with their API keys. Use "+ Add model" to register any provider.</p>

            <!-- Per-model registry -->
            <div class="settings-subsection-title">Registered Models</div>
            <div id="model-registry-list">
                <span class="settings-hint">Loading...</span>
            </div>
            <details class="settings-add-model" style="margin-top:12px">
                <summary style="cursor:pointer;color:var(--accent);font-size:13px;font-weight:600">+ Add model</summary>
                <div style="margin-top:10px;display:flex;flex-direction:column;gap:8px">
                    <div style="display:flex;gap:8px;flex-wrap:wrap">
                        <select id="add-model-provider" class="input" style="width:140px;padding:6px 8px">
                            <option value="google">Google</option>
                            <option value="openai">OpenAI</option>
                            <option value="anthropic">Anthropic</option>
                            <option value="openrouter">OpenRouter</option>
                            <option value="nvidia">NVIDIA NIM</option>
                            <option value="deepseek">DeepSeek</option>
                            <option value="xai">xAI</option>
                        </select>
                        <input type="text" id="add-model-name" class="input" placeholder="Model name" style="flex:1;min-width:180px;padding:6px 8px">
                    </div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
                        <div style="position:relative;flex:1;min-width:140px">
                            <input type="password" id="add-model-key" class="input" placeholder="API key" style="width:100%;padding:6px 8px;box-sizing:border-box">
                            <span id="add-model-key-hint" style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:11px;color:var(--accent);cursor:pointer;white-space:nowrap" title="Click to reuse this key">reuse ↓</span>
                        </div>
                        <input type="number" id="add-model-input-price" class="input" placeholder="$/M in" step="0.01" min="0" style="width:90px;padding:6px 8px">
                        <input type="number" id="add-model-output-price" class="input" placeholder="$/M out" step="0.01" min="0" style="width:90px;padding:6px 8px">
                        <button class="btn btn-primary" id="btn-add-model">Add</button>
                    </div>
                    <div id="add-model-reuse-hint" style="display:none;font-size:12px;color:var(--accent);margin-top:-2px"></div>
                    <div id="add-model-provider-note" class="settings-hint"></div>
                </div>
            </details>

            <!-- Model assignments -->
            <div class="settings-subsection-title" style="margin-top:20px">Model assignments</div>
            <div class="settings-field">
                <label class="settings-label">Chat model</label>
                <div class="settings-input-row">
                    <select id="set-model" class="input settings-input" data-current="${esc(cfg.summary_model)}">
                        <option value="${esc(cfg.summary_model)}">${esc(cfg.summary_model)}</option>
                    </select>
                    <button class="btn btn-primary" id="btn-save-model">Save</button>
                </div>
            </div>
            <div class="settings-field">
                <label class="settings-label">Voice model</label>
                <div class="settings-input-row">
                    <select id="set-voice-model" class="input settings-input" data-current="${esc(cfg.gemini_model)}">
                        ${[
                            "gemini-3.1-flash-live-preview",
                            "gemini-2.5-flash-native-audio-preview-12-2025",
                            "gemini-2.5-flash-preview-native-audio-dialog",
                        ].map(v => `<option value="${v}" ${v === cfg.gemini_model ? "selected" : ""}>${v}</option>`).join("")}
                    </select>
                    <button class="btn btn-primary" id="btn-save-voice-model">Save</button>
                </div>
            </div>
        </div>

        <!-- Local Models -->
        <div class="settings-section" id="local-models-section">
            <h3 class="settings-section-title">Local Models (llama.cpp)</h3>
            <p class="settings-hint" style="margin-bottom:14px">
                Download GGUF models to your computer and run them fully locally. No cloud account or API key is required.
            </p>
            <div id="llamacpp-status-bar" style="margin-bottom:14px"></div>
            <div id="llamacpp-installed-list" style="margin-bottom:16px"></div>

            <div class="settings-subsection-title" style="margin-bottom:8px">Models folder</div>
            <p class="settings-hint" style="margin-bottom:10px">
                Choose where Remy should keep downloaded GGUF models. Existing GGUF files in that folder are discovered automatically.
            </p>
            <div id="llamacpp-models-dir-current" class="settings-current" style="margin-bottom:10px">Loading models folder...</div>
            <div class="llamacpp-model-install">
                <button class="btn btn-primary btn-sm" id="btn-llamacpp-local-folder">Choose models folder</button>
            </div>
            <div id="llamacpp-local-file-row" class="llamacpp-model-install hidden" style="margin-top:10px">
                <select id="llamacpp-local-file" class="input" aria-label="Model in selected folder"></select>
                <button class="btn btn-primary btn-sm" id="btn-llamacpp-add-model">Add to chat models</button>
            </div>
            <div id="llamacpp-local-status" class="settings-hint" style="margin:8px 0 18px"></div>

            <div class="settings-subsection-title" style="margin-bottom:8px">Download from Hugging Face</div>
            <p class="settings-hint" style="margin-bottom:10px">
                Paste a public Hugging Face repository ID, inspect its available GGUF files, then choose a quantization.
            </p>
            <div class="llamacpp-model-install">
                <input type="text" id="llamacpp-repo" class="input"
                    placeholder="owner/model-GGUF" autocomplete="off">
                <button class="btn btn-outline btn-sm" id="btn-llamacpp-find">Show repository files</button>
                <a class="btn btn-outline btn-sm" href="https://huggingface.co/models?library=gguf&amp;sort=trending"
                    target="_blank" rel="noopener noreferrer">Browse GGUF models</a>
            </div>
            <div id="llamacpp-file-row" class="llamacpp-model-install hidden" style="margin-top:8px">
                <select id="llamacpp-file" class="input" aria-label="GGUF quantization"></select>
                <button class="btn btn-primary btn-sm" id="btn-llamacpp-download">Download model</button>
            </div>
            <div id="llamacpp-repo-status" class="settings-hint" style="margin-top:8px"></div>
        </div>

        <!-- llama.cpp install and download progress modal -->
        <div id="llamacpp-progress-modal" class="llamacpp-modal hidden">
            <div class="llamacpp-modal-box">
                <div class="llamacpp-modal-title" id="llamacpp-progress-title">Preparing local runtime…</div>
                <div class="bulk-progress-bar-track" style="margin:12px 0">
                    <div id="llamacpp-progress-bar" class="bulk-progress-bar" style="width:0%"></div>
                </div>
                <div id="llamacpp-progress-log" class="bulk-log" style="max-height:160px"></div>
                <button id="llamacpp-progress-close" class="btn btn-outline btn-sm hidden" style="margin-top:10px">Close</button>
            </div>
        </div>

        <!-- Custom Prompt -->
        <div class="settings-section">
            <h3 class="settings-section-title">Custom Instructions</h3>
            <p class="settings-hint" style="margin-bottom:8px">Your preferences for Remy — how to communicate, what to focus on.</p>
            <textarea id="set-custom-prompt" class="input" rows="6"
                style="width:100%;font-size:13px;resize:vertical;padding:10px"
                placeholder="Example: Always reply in English. Be concise.">${esc(cfg.custom_system_prompt || "")}</textarea>
            <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
                <button class="btn btn-primary" id="btn-save-prompt">Save</button>
                <button class="btn btn-outline" id="btn-clear-prompt">Clear</button>
                <span id="prompt-status" class="settings-status" style="margin:0"></span>
            </div>
        </div>

        <!-- Appearance & Voice -->
        <div class="settings-section">
            <h3 class="settings-section-title">Appearance &amp; Voice</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
                <div class="settings-field">
                    <label class="settings-label">Theme</label>
                    <select id="set-theme" class="input settings-input">
                        <option value="dark">Dark</option>
                        <option value="light">Light</option>
                    </select>
                </div>
                <div class="settings-field">
                    <label class="settings-label">Voice</label>
                    <div class="settings-input-row">
                        <select id="set-voice" class="input settings-input">
                            ${["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Perseus"].map(
                                (v) => `<option value="${v}" ${v === cfg.gemini_voice ? "selected" : ""}>${v}</option>`
                            ).join("")}
                        </select>
                        <button class="btn btn-primary" id="btn-save-voice">Save</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Integrations -->
        <div class="settings-section">
            <h3 class="settings-section-title">Integrations</h3>

            <!-- Telegram -->
            <div class="settings-field">
                <label class="settings-label" style="font-size:14px;font-weight:600">&#9992; Telegram</label>
                <p class="settings-hint" style="margin:4px 0 10px">
                    Create a bot via <a href="https://t.me/BotFather" target="_blank" style="color:var(--accent)">@BotFather</a>,
                    copy the token, then send any message to your bot and paste your Chat ID.
                    After saving, Automations can deliver results directly to your Telegram.
                </p>
                <div class="settings-current">
                    Bot: <code>${cfg.has_telegram ? cfg.telegram_bot_masked : "Not configured"}</code>
                    &nbsp;·&nbsp; Chat ID: <code>${cfg.proactive_chat_id || "Not configured"}</code>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
                    <div class="settings-input-row">
                        <input type="password" id="set-telegram-token" class="input settings-input" placeholder="Bot Token (from @BotFather)">
                        <button class="btn btn-primary" id="btn-save-telegram-token">Save</button>
                    </div>
                    <div class="settings-input-row">
                        <input type="text" id="set-telegram-chat-id" class="input settings-input" placeholder="Your Chat ID">
                        <button class="btn btn-primary" id="btn-save-telegram-chat-id">Save</button>
                    </div>
                </div>
            </div>

            <!-- Email (Gmail App Password) -->
            <div class="settings-field" style="margin-top:20px">
                <label class="settings-label" style="font-size:14px;font-weight:600">&#9993; Email (Gmail)</label>
                <p class="settings-hint" style="margin:4px 0 10px">
                    Enable 2-Step Verification in your Google account, then generate an
                    <a href="https://myaccount.google.com/apppasswords" target="_blank" style="color:var(--accent)">App Password</a>
                    (select "Mail" + "Windows Computer"). Use that 16-character password below — not your regular Gmail password.
                </p>
                <div class="settings-current">
                    Status: <code style="color:${cfg.has_smtp ? "var(--green)" : "var(--text-muted)"}">${cfg.has_smtp ? "Configured ✓" : "Not configured"}</code>
                    ${cfg.has_smtp ? `&nbsp;·&nbsp; Account: <code>${cfg.smtp_user || ""}</code>` : ""}
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
                    <div class="settings-field" style="margin:0">
                        <label class="settings-label" style="font-size:11px">Gmail address</label>
                        <input type="email" id="set-smtp-user" class="input settings-input" placeholder="you@gmail.com" value="${esc(cfg.smtp_user || "")}">
                    </div>
                    <div class="settings-field" style="margin:0">
                        <label class="settings-label" style="font-size:11px">App Password (16 chars)</label>
                        <input type="password" id="set-smtp-password" class="input settings-input" placeholder="xxxx xxxx xxxx xxxx">
                    </div>
                </div>
                <div style="display:flex;gap:8px;margin-top:8px;align-items:center">
                    <button class="btn btn-primary" id="btn-save-smtp">Save Email Settings</button>
                    ${cfg.has_smtp ? `<button class="btn btn-outline btn-mini" id="btn-clear-smtp">Clear</button>` : ""}
                    <span id="smtp-status" class="settings-status" style="margin:0"></span>
                </div>
            </div>

            <div class="settings-field" id="push-section" style="margin-top:16px">
                <span class="settings-hint">Checking push support...</span>
            </div>
        </div>

        <!-- Local Workspaces -->
        <div class="settings-section">
            <h3 class="settings-section-title">Local Workspaces</h3>
            <p class="settings-hint" style="margin-bottom:12px">
                Give Remy access only to folders you choose. Read, Write, and Execute are separate capabilities and can be revoked at any time.
            </p>
            <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px">
                <label><input id="workspace-read" type="checkbox" checked disabled> Read</label>
                <label><input id="workspace-write" type="checkbox"> Write</label>
                <label title="Commands run with your Windows user privileges"><input id="workspace-execute" type="checkbox"> Execute</label>
                <button class="btn btn-primary" id="btn-workspace-choose">Choose folder</button>
            </div>
            <details style="margin-bottom:12px">
                <summary style="cursor:pointer;color:var(--accent);font-size:13px">Enter a path manually</summary>
                <div style="display:flex;gap:8px;margin-top:9px">
                    <input id="workspace-manual-path" class="input" style="flex:1" placeholder="D:\\Projects\\my-project">
                    <button class="btn btn-outline" id="btn-workspace-add-path">Add</button>
                </div>
            </details>
            <div id="workspace-warning" class="settings-hint" style="margin:8px 0;color:var(--yellow)"></div>
            <div id="workspace-list">${renderWorkspaces(workspacesData.workspaces || [])}</div>
        </div>

        <!-- Data -->
        <div class="settings-section">
            <h3 class="settings-section-title">Data</h3>
            <div class="settings-field" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <button class="btn btn-outline" id="btn-export">Export memory</button>
                <button class="btn btn-outline" id="btn-import">Import memory</button>
                <input type="file" id="import-file" style="display:none" accept=".json">
                <span class="settings-hint">${diag.brain.records} records</span>
            </div>
        </div>

        <!-- Aura Memory -->
        <div class="settings-section">
            <h3 class="settings-section-title">Aura Memory</h3>
            <p class="settings-hint" style="margin-bottom:12px">Remy's cognitive memory library. Updated independently of the main app.</p>
            <div id="aura-status-block">
                <span class="settings-hint">Checking version...</span>
            </div>
        </div>

        <div id="settings-status" class="settings-status"></div>
    `;

    buildSettingsNavigation();

    // Theme
    const themeSelect = document.getElementById("set-theme");
    const currentTheme = localStorage.getItem("theme") || "dark";
    if (themeSelect) {
        themeSelect.value = currentTheme;
        themeSelect.addEventListener("change", (e) => {
            const val = e.target.value;
            localStorage.setItem("theme", val);
            document.documentElement.setAttribute("data-theme", val);
        });
    }

    loadModelRegistry();
    loadAvailableModelOptions();
    bindLocalSecrets();
    document.getElementById("btn-add-model")?.addEventListener("click", addModel);
    document.getElementById("add-model-provider")?.addEventListener("change", _onProviderChange);
    document.querySelector(".settings-add-model")?.addEventListener("toggle", () => _onProviderChange());
    initPushSection();
    loadAuraStatus();
    loadLocalModels();
    bindWorkspaces();

    document.getElementById("btn-save-model").addEventListener("click", async () => {
        const val = document.getElementById("set-model").value.trim();
        if (val) await saveSetting({ summary_model: val });
    });
    document.getElementById("btn-save-voice-model").addEventListener("click", async () => {
        const val = document.getElementById("set-voice-model").value;
        if (val) await saveSetting({ gemini_model: val });
    });

    document.getElementById("btn-save-voice").addEventListener("click", async () => {
        await saveSetting({ gemini_voice: document.getElementById("set-voice").value });
    });

    document.getElementById("btn-save-telegram-token").addEventListener("click", async () => {
        const val = document.getElementById("set-telegram-token").value.trim();
        if (!val) return;
        await saveSetting({ telegram_bot_token: val });
        document.getElementById("set-telegram-token").value = "";
    });
    document.getElementById("btn-save-telegram-chat-id").addEventListener("click", async () => {
        const val = document.getElementById("set-telegram-chat-id").value.trim();
        if (!val) return;
        await saveSetting({ proactive_chat_id: parseInt(val) || val });
        document.getElementById("set-telegram-chat-id").value = "";
    });

    document.getElementById("btn-save-smtp")?.addEventListener("click", async () => {
        const user = document.getElementById("set-smtp-user").value.trim();
        const pass = document.getElementById("set-smtp-password").value.trim();
        const statusEl = document.getElementById("smtp-status");
        if (!user || !pass) {
            statusEl.textContent = "Enter both email and app password.";
            statusEl.style.color = "var(--red)";
            return;
        }
        try {
            await saveSetting({
                smtp_host: "smtp.gmail.com",
                smtp_port: 587,
                smtp_user: user,
                smtp_password: pass,
                smtp_from: user,
            }, false);
            document.getElementById("set-smtp-password").value = "";
            statusEl.textContent = "Email configured!";
            statusEl.style.color = "var(--green)";
            setTimeout(() => loadSettings(), 1200);
        } catch (e) {
            statusEl.textContent = `Error: ${e.message}`;
            statusEl.style.color = "var(--red)";
        }
    });
    document.getElementById("btn-clear-smtp")?.addEventListener("click", async () => {
        const statusEl = document.getElementById("smtp-status");
        try {
            await saveSetting({ smtp_user: "", smtp_password: "", smtp_from: "" }, false);
            statusEl.textContent = "Cleared.";
            statusEl.style.color = "var(--text-muted)";
            setTimeout(() => loadSettings(), 800);
        } catch (e) {
            statusEl.textContent = `Error: ${e.message}`;
            statusEl.style.color = "var(--red)";
        }
    });

    document.getElementById("btn-save-prompt").addEventListener("click", async () => {
        const text = document.getElementById("set-custom-prompt").value;
        const status = document.getElementById("prompt-status");
        try {
            await saveSetting({ custom_system_prompt: text }, false);
            status.textContent = "Saved!";
            status.style.color = "var(--green)";
        } catch (e) {
            status.textContent = `Error: ${e.message}`;
            status.style.color = "var(--red)";
        }
    });
    document.getElementById("btn-clear-prompt").addEventListener("click", async () => {
        document.getElementById("set-custom-prompt").value = "";
        const status = document.getElementById("prompt-status");
        try {
            await saveSetting({ custom_system_prompt: "" }, false);
            status.textContent = "Cleared.";
            status.style.color = "var(--text-muted)";
        } catch (e) {
            status.textContent = `Error: ${e.message}`;
            status.style.color = "var(--red)";
        }
    });

    document.getElementById("btn-export").addEventListener("click", exportBrain);
    document.getElementById("btn-import").addEventListener("click", () => {
        document.getElementById("import-file").click();
    });
    document.getElementById("import-file").addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) importBrain(file);
    });
}

function renderLocalSecrets(secrets) {
    if (!secrets.length) {
        return `<span class="settings-hint">No local secrets are available.</span>`;
    }
    return secrets.map((secret) => {
        const canTest = ["gemini_api_key", "openrouter_api_key", "telegram_bot_token"].includes(secret.key);
        return `
        <div class="settings-secret-row" data-secret-key="${esc(secret.key)}">
            <div class="settings-secret-main">
                <div class="settings-secret-title">
                    <strong>${esc(secret.label)}</strong>
                    <span>${esc(secret.kind)}</span>
                </div>
                <p>${esc(secret.description)}</p>
                <div class="settings-secret-status">
                    <span class="${secret.configured ? "settings-secret-ready" : "settings-secret-empty"}">
                        ${secret.configured ? "Ready" : "Not set"}
                    </span>
                    <code>${secret.configured ? esc(secret.masked) : "local only"}</code>
                    <span class="settings-secret-test-status" data-secret-test-status></span>
                </div>
            </div>
            <div class="settings-secret-actions">
                <input type="password" class="input settings-secret-input" placeholder="Paste new value">
                <button class="btn btn-primary btn-sm settings-secret-save">Save</button>
                ${canTest && secret.configured ? `<button class="btn btn-outline btn-sm settings-secret-test">Test</button>` : ""}
                ${secret.configured ? `<button class="btn btn-outline btn-sm settings-secret-clear">Clear</button>` : ""}
            </div>
        </div>
    `;
    }).join("");
}

function bindLocalSecrets() {
    document.querySelectorAll(".settings-secret-row").forEach((row) => {
        const key = row.dataset.secretKey;
        const input = row.querySelector(".settings-secret-input");
        row.querySelector(".settings-secret-save")?.addEventListener("click", async () => {
            const value = input?.value.trim() || "";
            if (!value) {
                showStatus("Paste a value before saving.", true);
                return;
            }
            await saveSecret(key, value);
            if (input) input.value = "";
        });
        row.querySelector(".settings-secret-clear")?.addEventListener("click", async () => {
            const confirmed = await showConfirm("Clear Secret", "Remove this local secret from Remy?");
            if (!confirmed) return;
            await saveSecret(key, "");
        });
        row.querySelector(".settings-secret-test")?.addEventListener("click", async () => {
            await testSecret(row, key);
        });
    });
}

async function testSecret(row, key) {
    const btn = row.querySelector(".settings-secret-test");
    const status = row.querySelector("[data-secret-test-status]");
    if (btn) btn.disabled = true;
    if (status) {
        status.textContent = "Testing...";
        status.className = "settings-secret-test-status";
    }
    try {
        const res = await fetch(`/api/secrets/${encodeURIComponent(key)}/test`, { method: "POST" });
        const data = await res.json().catch(() => ({}));
        const ok = !!data.ok;
        if (status) {
            status.textContent = data.message || (ok ? "Ready" : "Failed");
            status.className = `settings-secret-test-status ${ok ? "ok" : "error"}`;
        }
        showStatus(data.message || (ok ? "Secret works." : "Secret test failed."), !ok);
    } catch (e) {
        if (status) {
            status.textContent = "Test failed";
            status.className = "settings-secret-test-status error";
        }
        showStatus(`Secret test failed: ${e.message}`, true);
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function saveSecret(key, value) {
    const res = await fetch(`/api/secrets/${encodeURIComponent(key)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
    });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        showStatus(data.detail || "Secret save failed.", true);
        return;
    }
    showStatus(value ? "Secret saved locally." : "Secret cleared.");
    setTimeout(() => loadSettings(), 700);
}

// ============== Model Registry ==============

async function loadModelRegistry() {
    const container = document.getElementById("model-registry-list");
    if (!container) return;

    try {
        const res = await fetch("/api/model-registry");
        const data = await res.json();
        const models = data.models || [];
        _cachedRegisteredModels = models;

        if (models.length === 0) {
            container.innerHTML = `<span class="settings-hint">No custom models added yet. Use "+ Add model" below.</span>`;
            loadAvailableModelOptions();
            return;
        }

        container.innerHTML = `
            <div class="model-registry-grid">
                ${models.map(m => {
                    const hasPrice = m.input_price != null || m.output_price != null;
                    const priceText = hasPrice
                        ? `$${(m.input_price || 0).toFixed(2)}/$${(m.output_price || 0).toFixed(2)}`
                        : "";
                    return `
                    <div class="model-registry-row${m.auto_migrated ? " model-row-auto" : ""}" data-model="${esc(m.name)}">
                        <span class="model-provider-badge model-provider-${m.provider}">${m.provider}</span>
                        <span class="model-name">${esc(m.name)}${m.auto_migrated ? ` <span class="model-auto-tag">auto</span>` : ""}</span>
                        ${hasPrice ? `<span class="model-price" title="Input/Output per 1M tokens">${priceText}</span>` : `<span></span>`}
                        <code class="model-key">${m.has_key ? m.api_key_masked : "No key"}</code>
                        <div style="display:flex;gap:4px;align-items:center">
                            <button class="btn btn-outline btn-mini model-edit-key-btn" data-model="${esc(m.name)}" data-provider="${esc(m.provider)}">Edit key</button>
                            <button class="btn-icon model-delete-btn" data-model="${esc(m.name)}" title="Remove">&#10005;</button>
                        </div>
                        <div class="model-edit-key-row hidden" id="edit-key-${esc(m.name).replace(/[^a-z0-9]/gi,'_')}">
                            <input type="password" class="input model-new-key-input" placeholder="New API key" style="flex:1;padding:5px 8px;font-size:12px">
                            <button class="btn btn-primary btn-mini model-save-key-btn" data-model="${esc(m.name)}" data-provider="${esc(m.provider)}">Save</button>
                            <button class="btn btn-outline btn-mini model-cancel-key-btn">Cancel</button>
                        </div>
                    </div>`;
                }).join("")}
            </div>
        `;

        container.querySelectorAll(".model-edit-key-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const rowId = "edit-key-" + btn.dataset.model.replace(/[^a-z0-9]/gi, "_");
                const editRow = document.getElementById(rowId);
                editRow?.classList.toggle("hidden");
            });
        });

        container.querySelectorAll(".model-cancel-key-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                btn.closest(".model-edit-key-row")?.classList.add("hidden");
            });
        });

        container.querySelectorAll(".model-save-key-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                const name = btn.dataset.model;
                const provider = btn.dataset.provider;
                const input = btn.closest(".model-edit-key-row")?.querySelector(".model-new-key-input");
                const newKey = input?.value.trim();
                if (!newKey) return;
                try {
                    const response = await fetch("/api/model-registry", {
                        method: "PUT",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ model_name: name, api_key: newKey, provider }),
                    });
                    if (!response.ok) {
                        const data = await response.json().catch(() => ({}));
                        throw new Error(data.detail || `Could not update model (HTTP ${response.status}).`);
                    }
                    input.value = "";
                    await loadModelRegistry();
                    document.dispatchEvent(new CustomEvent("models-changed", {
                        detail: { reason: "registry-updated", model: name },
                    }));
                    showStatus(`Key updated: ${name}`);
                } catch (e) {
                    showStatus(`Error: ${e.message}`, true);
                }
            });
        });

        container.querySelectorAll(".model-delete-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                const name = btn.dataset.model;
                const confirmed = await showConfirm("Remove Model", `Remove "${name}" from registry?`);
                if (!confirmed) return;
                try {
                    const response = await fetch(`/api/model-registry/${encodeURIComponent(name)}`, { method: "DELETE" });
                    if (!response.ok) {
                        const data = await response.json().catch(() => ({}));
                        throw new Error(data.detail || `Could not remove model (HTTP ${response.status}).`);
                    }
                    await loadModelRegistry();
                    document.dispatchEvent(new CustomEvent("models-changed", {
                        detail: { reason: "registry-removed", model: name },
                    }));
                    showStatus(`Removed: ${name}`);
                } catch (e) {
                    showStatus(`Error: ${e.message}`, true);
                }
            });
        });

        loadAvailableModelOptions();
    } catch (e) {
        container.innerHTML = `<span class="settings-hint" style="color:var(--red)">Failed to load models.</span>`;
    }
}

function _populateModelSelects(models) {
    const selectIds = ["set-model"];
    for (const id of selectIds) {
        const sel = document.getElementById(id);
        if (!sel) continue;
        const current = sel.dataset.current || sel.value;
        sel.innerHTML = "";
        // Add all registry models as options
        for (const m of models) {
            const opt = document.createElement("option");
            opt.value = m.name;
            opt.textContent = `${m.label || m.name}  [${m.provider}]`;
            if (m.name === current) opt.selected = true;
            sel.appendChild(opt);
        }
        // If current model isn't in registry — keep it as an option
        if (!models.find(m => m.name === current) && current) {
            const opt = document.createElement("option");
            opt.value = current;
            opt.textContent = `${current}  [current]`;
            opt.selected = true;
            sel.insertBefore(opt, sel.firstChild);
        }
    }
}


function _onProviderChange() {
    const provider = document.getElementById("add-model-provider")?.value;
    const keyInput = document.getElementById("add-model-key");
    const hintEl = document.getElementById("add-model-reuse-hint");
    if (!provider || !keyInput || !hintEl) return;
    const nameInput = document.getElementById("add-model-name");
    const providerNote = document.getElementById("add-model-provider-note");
    const providerUi = {
        nvidia: {
            key: "NVIDIA API key (nvapi-...)",
            model: "e.g. deepseek-ai/deepseek-v4-flash",
            note: "NVIDIA hosted NIM trial endpoint. Development use may be rate limited.",
        },
        openrouter: {
            key: "OpenRouter API key (sk-or-...)",
            model: "e.g. moonshotai/kimi-k3",
            note: "OpenRouter model identifier in publisher/model format.",
        },
    };
    const ui = providerUi[provider] || {};
    if (nameInput) nameInput.placeholder = ui.model || "Model name";
    if (providerNote) providerNote.textContent = ui.note || "";
    keyInput.dataset.reuseFrom = "";
    keyInput.placeholder = ui.key || "API key";

    // Find an existing model with a key for this provider
    const existing = _cachedRegisteredModels.find(m => m.provider === provider && m.has_key);
    if (existing) {
        hintEl.style.display = "block";
        hintEl.innerHTML = `Key already saved for <b>${provider}</b> (${existing.api_key_masked}) — <a href="#" id="add-model-reuse-link" style="color:var(--accent)">reuse it</a>`;
        document.getElementById("add-model-reuse-link")?.addEventListener("click", (e) => {
            e.preventDefault();
            // Signal to addModel to copy the key from the existing model
            keyInput.dataset.reuseFrom = existing.name;
            keyInput.placeholder = `Using key from ${existing.name}`;
            keyInput.value = "";
            hintEl.innerHTML = `✓ Will reuse key from <b>${existing.name}</b>. Leave the field empty to confirm.`;
        });
    } else {
        hintEl.style.display = "none";
    }
}

async function loadAvailableModelOptions() {
    try {
        const response = await fetch("/api/models");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not load models");
        _populateModelSelects(data.models || []);
    } catch (_) {
        // Keep the current selection when the catalog is temporarily unavailable.
    }
}

async function addModel() {
    const name = document.getElementById("add-model-name").value.trim();
    const key = document.getElementById("add-model-key").value.trim();
    const provider = document.getElementById("add-model-provider").value;
    const inputPrice = parseFloat(document.getElementById("add-model-input-price").value) || null;
    const outputPrice = parseFloat(document.getElementById("add-model-output-price").value) || null;

    const keyInput = document.getElementById("add-model-key");
    const reuseFrom = keyInput?.dataset.reuseFrom || "";

    if (!name) { alert("Model name is required."); return; }
    if (!key && !reuseFrom) { alert("API key is required."); return; }

    try {
        // If reusing a key from another model of the same provider, copy it server-side
        const payload = { model_name: name, api_key: key || "", provider };
        if (!key && reuseFrom) payload.copy_key_from = reuseFrom;
        if (inputPrice != null) payload.input_price = inputPrice;
        if (outputPrice != null) payload.output_price = outputPrice;

        const response = await fetch("/api/model-registry", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || `Could not add model (HTTP ${response.status}).`);
        }
        document.getElementById("add-model-name").value = "";
        const ki = document.getElementById("add-model-key");
        if (ki) { ki.value = ""; ki.dataset.reuseFrom = ""; ki.placeholder = "API key"; }
        document.getElementById("add-model-input-price").value = "";
        document.getElementById("add-model-output-price").value = "";
        const hint = document.getElementById("add-model-reuse-hint");
        if (hint) hint.style.display = "none";
        await loadModelRegistry();
        document.dispatchEvent(new CustomEvent("models-changed", {
            detail: { reason: "registry-added", model: name },
        }));
        showStatus(`Added: ${name}`);
    } catch (e) {
        showStatus(`Error: ${e.message}`, true);
    }
}

// ============== Settings Save ==============

async function saveSetting(payload, reload = true) {
    const statusEl = document.getElementById("settings-status");
    try {
        const res = await fetch("/api/settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (statusEl) {
            statusEl.textContent = `Saved: ${data.updated.join(", ")}`;
            statusEl.style.color = "var(--green)";
        }
        if (reload) setTimeout(() => loadSettings(), 1500);
    } catch (e) {
        if (statusEl) {
            statusEl.textContent = `Error: ${e.message}`;
            statusEl.style.color = "var(--red)";
        }
        throw e;
    }
}

function showStatus(msg, isError = false) {
    const el = document.getElementById("settings-status");
    if (el) {
        el.textContent = msg;
        el.style.color = isError ? "var(--red)" : "var(--green)";
    }
}

// ============== Export/Import ==============

async function exportBrain() {
    const statusEl = document.getElementById("settings-status");
    try {
        statusEl.textContent = "Exporting...";
        statusEl.style.color = "var(--text-muted)";
        const res = await fetch("/api/export");
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "remy-brain-export.json";
        a.click();
        URL.revokeObjectURL(url);
        statusEl.textContent = "Export downloaded.";
        statusEl.style.color = "var(--green)";
    } catch (e) {
        statusEl.textContent = `Export failed: ${e.message}`;
        statusEl.style.color = "var(--red)";
    }
}

async function importBrain(file) {
    const confirmed = await showConfirm("Import Brain", "Importing will add/merge records from the file. Continue?");
    if (!confirmed) return;

    const statusEl = document.getElementById("settings-status");
    statusEl.textContent = "Importing...";
    statusEl.style.color = "var(--text-muted)";

    try {
        const text = await file.text();
        const json = JSON.parse(text);
        const res = await fetch("/api/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(json),
        });
        const result = await res.json();
        if (res.ok) {
            statusEl.textContent = `Imported: ${result.imported} records, ${result.connections_restored} connections.`;
            statusEl.style.color = "var(--green)";
            document.getElementById("import-file").value = "";
            setTimeout(() => loadSettings(), 2000);
        } else {
            throw new Error(result.detail || "Import failed");
        }
    } catch (e) {
        statusEl.textContent = `Import error: ${e.message}`;
        statusEl.style.color = "var(--red)";
        document.getElementById("import-file").value = "";
    }
}

// ============== Push Notifications ==============

async function initPushSection() {
    const container = document.getElementById("push-section");
    if (!container) return;

    if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
        container.innerHTML = `<span class="settings-hint">Push notifications not supported in this browser.</span>`;
        return;
    }

    const permission = Notification.permission;
    let serverStatus;
    try {
        const res = await fetch("/api/push/status");
        serverStatus = await res.json();
    } catch {
        container.innerHTML = `<span class="settings-hint" style="color:var(--red)">Failed to check push status.</span>`;
        return;
    }

    // Auto-resubscribe
    if (permission === "granted" && !serverStatus.subscribed) {
        const reg = await navigator.serviceWorker.ready;
        const existing = await reg.pushManager.getSubscription();
        if (existing) {
            try {
                await fetch("/api/push/subscribe", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(existing.toJSON()),
                });
                serverStatus.subscribed = true;
            } catch { /* ignore */ }
        }
    }

    renderPushUI(container, serverStatus.subscribed, permission);
}

function renderPushUI(container, subscribed, permission) {
    let statusText, statusColor, buttonText, buttonAction;

    if (permission === "denied") {
        statusText = "Blocked";
        statusColor = "var(--red)";
        buttonText = null;
    } else if (subscribed) {
        statusText = "Enabled";
        statusColor = "var(--green)";
        buttonText = "Disable";
        buttonAction = "disable";
    } else {
        statusText = "Disabled";
        statusColor = "var(--text-muted)";
        buttonText = "Enable";
        buttonAction = "enable";
    }

    container.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px">
            <span>Status: <strong style="color:${statusColor}">${statusText}</strong></span>
            ${buttonText ? `<button class="btn btn-primary" id="btn-push-toggle" style="font-size:13px;padding:6px 16px">${buttonText}</button>` : ""}
        </div>
        <div class="settings-hint" style="margin-top:4px">Background notifications when the tab is not focused.</div>
    `;

    const btn = document.getElementById("btn-push-toggle");
    if (btn) btn.addEventListener("click", () => togglePush(buttonAction));
}

async function togglePush(action) {
    const container = document.getElementById("push-section");

    if (action === "enable") {
        try {
            const permission = await Notification.requestPermission();
            if (permission !== "granted") { renderPushUI(container, false, permission); return; }

            const vapidRes = await fetch("/api/push/vapid-key");
            const { public_key } = await vapidRes.json();
            const reg = await navigator.serviceWorker.ready;
            const subscription = await reg.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: urlBase64ToUint8Array(public_key),
            });
            await fetch("/api/push/subscribe", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(subscription.toJSON()),
            });
            renderPushUI(container, true, "granted");
            showStatus("Push notifications enabled.");
        } catch (e) {
            showStatus(`Push setup failed: ${e.message}`, true);
        }
    } else {
        try {
            const reg = await navigator.serviceWorker.ready;
            const subscription = await reg.pushManager.getSubscription();
            if (subscription) await subscription.unsubscribe();
            await fetch("/api/push/unsubscribe", { method: "POST" });
            renderPushUI(container, false, Notification.permission);
            showStatus("Push notifications disabled.");
        } catch (e) {
            showStatus(`Push disable failed: ${e.message}`, true);
        }
    }
}

function urlBase64ToUint8Array(base64String) {
    const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
    const rawData = atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; i++) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

// ============== Aura Memory ==============

async function loadAuraStatus() {
    const block = document.getElementById("aura-status-block");
    if (!block) return;
    try {
        const res = await fetch("/api/aura/status");
        const data = await res.json();
        const installed = data.installed || "not installed";
        const latest = data.latest || "—";
        const upToDate = data.up_to_date;
        const badgeColor = upToDate === true ? "var(--green)" : upToDate === false ? "var(--yellow)" : "var(--text-muted)";
        const badgeText = upToDate === true ? "up to date" : upToDate === false ? "update available" : "unknown";

        block.innerHTML = `
            <div class="aura-status-row">
                <div class="aura-status-versions">
                    <div class="aura-version-item">
                        <span class="settings-hint">Installed</span>
                        <strong>${esc(installed)}</strong>
                    </div>
                    <div class="aura-version-item">
                        <span class="settings-hint">On PyPI</span>
                        <strong>${esc(latest)}</strong>
                    </div>
                    <span class="aura-badge" style="background:${badgeColor}20;color:${badgeColor};border:1px solid ${badgeColor}40">${badgeText}</span>
                </div>
                <div style="display:flex;gap:8px;align-items:center;margin-top:10px">
                    ${upToDate === false ? `<button class="btn btn-primary" id="btn-aura-update">Update to ${esc(latest)}</button>` : ""}
                    <button class="btn btn-outline" id="btn-aura-reinstall" style="font-size:12px">Reinstall</button>
                    <a href="${esc(data.pypi_url)}" target="_blank" class="settings-hint" style="font-size:12px">PyPI ↗</a>
                </div>
                <div id="aura-update-log" style="display:none;margin-top:10px"></div>
            </div>
        `;

        document.getElementById("btn-aura-update")?.addEventListener("click", () => runAuraUpdate());
        document.getElementById("btn-aura-reinstall")?.addEventListener("click", () => runAuraUpdate());
    } catch (e) {
        block.innerHTML = `<span class="settings-hint" style="color:var(--red)">Could not check version: ${esc(e.message)}</span>`;
    }
}

async function runAuraUpdate() {
    const block = document.getElementById("aura-status-block");
    if (!block) return;

    // Show install progress
    block.innerHTML = `
        <div class="aura-status-row">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                <div class="aura-spinner"></div>
                <span style="color:var(--text-muted)">Downloading update from PyPI…</span>
            </div>
            <div id="aura-update-log" style="font-family:monospace;font-size:11px;color:var(--text-muted);white-space:pre-wrap;max-height:200px;overflow:auto"></div>
        </div>
    `;

    try {
        const res = await fetch("/api/aura/update", { method: "POST" });
        const data = await res.json();
        const logEl = document.getElementById("aura-update-log");

        if (!data.success) {
            // Installation failed — show error
            if (logEl) {
                logEl.textContent = data.message + (data.stderr ? "\n\n" + data.stderr : "");
                logEl.style.color = "var(--red)";
            }
            block.querySelector(".aura-spinner")?.remove();
            block.querySelector("span").textContent = "Installation failed.";
            block.querySelector("span").style.color = "var(--red)";
            return;
        }

        // Success + restart in progress
        if (logEl) {
            logEl.textContent = data.stdout || "";
            logEl.style.color = "var(--text-muted)";
        }
        block.querySelector("span").textContent = "Installed. Remy is restarting…";
        block.querySelector("span").style.color = "var(--green)";

        // Poll until server is back up
        await _waitForRestart();

    } catch (e) {
        // Server went down mid-response (restart happened) — this is expected
        block.innerHTML = `
            <div class="aura-status-row">
                <div style="display:flex;align-items:center;gap:10px">
                    <div class="aura-spinner"></div>
                    <span style="color:var(--text-muted)">Remy is restarting…</span>
                </div>
            </div>
        `;
        await _waitForRestart();
    }
}

async function _waitForRestart() {
    const block = document.getElementById("aura-status-block");
    const MAX_WAIT_MS = 30_000;
    const POLL_MS = 1000;
    const start = Date.now();

    while (Date.now() - start < MAX_WAIT_MS) {
        await new Promise(r => setTimeout(r, POLL_MS));
        try {
            const r = await fetch("/api/aura/status", { cache: "no-store" });
            if (r.ok) {
                // Server is back — reload status and show success toast
                await loadAuraStatus();
                if (block) {
                    const toast = document.createElement("div");
                    toast.style.cssText = "color:var(--green);font-size:13px;margin-top:8px;font-weight:600";
                    toast.textContent = "✓ Aura Memory updated successfully!";
                    block.appendChild(toast);
                    setTimeout(() => toast.remove(), 4000);
                }
                return;
            }
        } catch {
            // Still restarting — keep polling
        }
    }

    // Timed out
    if (block) block.innerHTML = `<span class="settings-hint" style="color:var(--red)">Remy did not respond after restart. Please reopen the app.</span>`;
}

// ============== Local Models (llama.cpp) ==============

async function loadLocalModels() {
    await _renderLlamaCppStatus();
    _bindLlamaCppControls();
}

async function _renderLlamaCppStatus() {
    const bar = document.getElementById("llamacpp-status-bar");
    const list = document.getElementById("llamacpp-installed-list");
    if (!bar || !list) return;
    try {
        const status = await fetch("/api/llamacpp/status").then(async response => {
            if (!response.ok) throw new Error(await response.text());
            return response.json();
        });
        const modelsDirectory = document.getElementById("llamacpp-models-dir-current");
        if (modelsDirectory) {
            const prefix = status.models_dir_configured ? "Selected folder" : "Default folder";
            modelsDirectory.textContent = `${prefix}: ${status.models_dir}`;
        }
        _renderLocalFolderFiles(status.local_files || []);
        if (status.runtime_installed) {
            bar.innerHTML = `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <span style="color:var(--green);font-weight:600">llama.cpp runtime ready</span>
                <span class="settings-hint">${status.running ? "Model server running" : "No model loaded"}</span>
                ${status.running ? '<button class="btn btn-outline btn-mini" id="btn-llamacpp-stop">Stop model</button>' : ''}
            </div>`;
        } else {
            bar.innerHTML = `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <span class="settings-hint">llama.cpp runtime is not installed</span>
                <button class="btn btn-primary btn-sm" id="btn-llamacpp-runtime">Install local runtime</button>
            </div>`;
        }
        if (status.models?.length) {
            list.innerHTML = `<div class="settings-subsection-title" style="margin-bottom:8px">Installed GGUF models</div>
                <div class="llamacpp-installed-models">${status.models.map(model => `
                    <div class="llamacpp-installed-row">
                        <span class="llamacpp-model-name" title="${esc(model.repo_id || '')}">${esc(model.filename)}</span>
                        <span class="llamacpp-model-size">${Number(model.size_gb || 0).toFixed(2)} GB</span>
                        ${model.managed === false ? '<span class="settings-hint">linked</span>' : ''}
                        <button class="btn btn-outline btn-mini llamacpp-use-btn" data-model="${esc(model.name)}">Use in chat</button>
                        <button class="btn-icon llamacpp-delete-btn" data-model="${esc(model.id)}"
                            data-managed="${model.managed !== false}" title="${model.managed === false ? 'Remove from Remy' : 'Delete downloaded model'}">x</button>
                    </div>`).join('')}</div>`;
        } else {
            list.innerHTML = `<p class="settings-hint">No GGUF models downloaded yet.</p>`;
        }
    } catch (error) {
        bar.innerHTML = `<span class="settings-hint" style="color:var(--red)">Could not read llama.cpp status.</span>`;
        list.innerHTML = "";
        const modelsDirectory = document.getElementById("llamacpp-models-dir-current");
        if (modelsDirectory) {
            modelsDirectory.textContent = "Models folder unavailable. Restart Remy, then refresh this page.";
            modelsDirectory.style.color = "var(--red)";
        }
    }
}

function _localProgressElements(title) {
    const modal = document.getElementById("llamacpp-progress-modal");
    const titleEl = document.getElementById("llamacpp-progress-title");
    const log = document.getElementById("llamacpp-progress-log");
    const bar = document.getElementById("llamacpp-progress-bar");
    const close = document.getElementById("llamacpp-progress-close");
    if (titleEl) titleEl.textContent = title;
    if (log) log.innerHTML = "";
    if (bar) bar.style.width = "0%";
    close?.classList.add("hidden");
    modal?.classList.remove("hidden");
    return { modal, log, bar, close };
}

function _renderLocalFolderFiles(files) {
    const row = document.getElementById("llamacpp-local-file-row");
    const select = document.getElementById("llamacpp-local-file");
    const status = document.getElementById("llamacpp-local-status");
    if (!row || !select) return;
    const available = files.filter(file => !file.added);
    if (!files.length) {
        row.classList.add("hidden");
        if (status) status.textContent = "No GGUF files found in this folder.";
        return;
    }
    if (!available.length) {
        row.classList.add("hidden");
        if (status) status.textContent = "All GGUF models in this folder are already added.";
        return;
    }
    select.innerHTML = available.map(file => {
        const size = file.size ? ` (${(file.size / 1024 ** 3).toFixed(2)} GB)` : "";
        const shards = file.shards > 1 ? ` · ${file.shards} files` : "";
        return `<option value="${esc(file.relative_path)}">${esc(file.filename)}${size}${shards}</option>`;
    }).join("");
    row.classList.remove("hidden");
    if (status) status.textContent = `Choose one of ${available.length} available GGUF model(s).`;
}

async function _consumeLlamaCppSSE(response, ui) {
    if (!response.ok || !response.body) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const event = JSON.parse(line.slice(6));
            if (event.pct != null && ui.bar) ui.bar.style.width = `${event.pct}%`;
            if (event.message && ui.log) {
                const row = document.createElement("div");
                row.className = `bulk-log-line${event.phase === "error" ? " error" : ""}`;
                row.textContent = event.message;
                ui.log.appendChild(row);
                ui.log.scrollTop = ui.log.scrollHeight;
            }
        }
    }
    ui.close?.classList.remove("hidden");
    if (ui.close) ui.close.onclick = async () => {
        ui.modal?.classList.add("hidden");
        await loadLocalModels();
    };
}

function _bindLlamaCppControls() {
    const localStatus = document.getElementById("llamacpp-local-status");
    const chooseFolder = async () => {
        if (localStatus) {
            localStatus.style.color = "var(--text-muted)";
            localStatus.textContent = "Waiting for the Windows folder picker...";
        }
        const response = await fetch("/api/llamacpp/local/select-folder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            if (response.status === 404 || response.status === 405) {
                throw new Error("The running Remy server is outdated. Restart Remy and refresh the page.");
            }
            throw new Error(data.detail || "Could not choose the models folder");
        }
        if (data.cancelled) {
            if (localStatus) localStatus.textContent = "Folder selection cancelled.";
            return;
        }
        if (localStatus) {
            localStatus.style.color = "var(--green)";
            localStatus.textContent = `Folder selected: ${data.path}. Now choose a model below.`;
        }
        const modelsDirectory = document.getElementById("llamacpp-models-dir-current");
        if (modelsDirectory) modelsDirectory.textContent = `Selected folder: ${data.path}`;
        _renderLocalFolderFiles(data.files || []);
    };
    document.getElementById("btn-llamacpp-local-folder")?.addEventListener("click", async event => {
        const button = event.currentTarget;
        button.disabled = true;
        try {
            await chooseFolder();
        } catch (error) {
            if (localStatus) {
                localStatus.style.color = "var(--red)";
                localStatus.textContent = error.message || String(error);
            }
        } finally {
            button.disabled = false;
        }
    });
    document.getElementById("btn-llamacpp-add-model")?.addEventListener("click", async event => {
        const relativePath = document.getElementById("llamacpp-local-file")?.value;
        if (!relativePath) return;
        const button = event.currentTarget;
        button.disabled = true;
        try {
            const response = await fetch("/api/llamacpp/local/model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ relative_path: relativePath }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || "Could not add the selected model");
            await loadLocalModels();
            await loadAvailableModelOptions();
            document.dispatchEvent(new CustomEvent("models-changed"));
            if (localStatus) {
                localStatus.style.color = "var(--green)";
                localStatus.textContent = `${data.model?.filename || "Model"} added to chat models. Click “Use in chat” to activate it.`;
            }
        } catch (error) {
            if (localStatus) {
                localStatus.style.color = "var(--red)";
                localStatus.textContent = error.message || String(error);
            }
        } finally {
            button.disabled = false;
        }
    });
    document.getElementById("btn-llamacpp-runtime")?.addEventListener("click", async () => {
        const ui = _localProgressElements("Installing llama.cpp runtime...");
        try {
            await _consumeLlamaCppSSE(await fetch("/api/llamacpp/runtime/install", { method: "POST" }), ui);
        } catch (error) {
            if (ui.log) ui.log.textContent = error.message || String(error);
            ui.close?.classList.remove("hidden");
        }
    });
    document.getElementById("btn-llamacpp-stop")?.addEventListener("click", async () => {
        await fetch("/api/llamacpp/stop", { method: "POST" });
        await loadLocalModels();
    });
    document.getElementById("btn-llamacpp-find")?.addEventListener("click", async () => {
        const repo = document.getElementById("llamacpp-repo")?.value.trim();
        const row = document.getElementById("llamacpp-file-row");
        const select = document.getElementById("llamacpp-file");
        const status = document.getElementById("llamacpp-repo-status");
        if (!repo || !select) {
            if (status) status.textContent = "Paste a Hugging Face repository ID first.";
            return;
        }
        if (status) status.textContent = "Reading repository...";
        select.innerHTML = '<option>Loading repository...</option>';
        row?.classList.remove("hidden");
        try {
            const response = await fetch(`/api/llamacpp/repository?repo_id=${encodeURIComponent(repo)}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Repository lookup failed");
            if (!data.files?.length) throw new Error("No GGUF files found in this repository");
            select.innerHTML = data.files.filter(file => file.selectable !== false).map(file => {
                const size = file.size ? ` (${(file.size / 1024 ** 3).toFixed(2)} GB)` : "";
                const shards = file.shards > 1 ? ` · ${file.shards} files` : "";
                return `<option value="${esc(file.filename)}">${esc(file.filename)}${size}${shards}</option>`;
            }).join("");
            if (status) {
                status.style.color = "var(--green)";
                status.textContent = `${select.options.length} GGUF option(s) found.`;
            }
        } catch (error) {
            select.innerHTML = `<option value="">${esc(error.message || String(error))}</option>`;
            if (status) {
                status.style.color = "var(--red)";
                status.textContent = error.message || String(error);
            }
        }
    });
    document.getElementById("btn-llamacpp-download")?.addEventListener("click", async () => {
        const repo = document.getElementById("llamacpp-repo")?.value.trim();
        const filename = document.getElementById("llamacpp-file")?.value;
        if (!repo || !filename) return;
        const ui = _localProgressElements(`Downloading ${filename}...`);
        try {
            const response = await fetch("/api/llamacpp/models/download", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ repo_id: repo, filename }),
            });
            await _consumeLlamaCppSSE(response, ui);
        } catch (error) {
            if (ui.log) ui.log.textContent = error.message || String(error);
            ui.close?.classList.remove("hidden");
        }
    });
    document.querySelectorAll(".llamacpp-use-btn").forEach(button => {
        button.addEventListener("click", async () => {
            button.disabled = true;
            const model = button.dataset.model;
            const response = await fetch("/api/llamacpp/models/start", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_id: model }),
            });
            const data = await response.json();
            if (!response.ok) { alert(data.detail || "Could not load model"); button.disabled = false; return; }
            await fetch("/api/settings", {
                method: "PUT", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ summary_model: model }),
            });
            await loadLocalModels();
            await loadAvailableModelOptions();
            document.dispatchEvent(new CustomEvent("models-changed"));
            if (localStatus) {
                localStatus.style.color = "var(--green)";
                localStatus.textContent = "Local model loaded and selected for chat.";
            }
        });
    });
    document.querySelectorAll(".llamacpp-delete-btn").forEach(button => {
        button.addEventListener("click", async () => {
            const managed = button.dataset.managed === "true";
            const question = managed
                ? "Delete this downloaded GGUF model from disk?"
                : "Remove this model from Remy? The original GGUF file will remain untouched.";
            if (!confirm(question)) return;
            await fetch(`/api/llamacpp/models/${encodeURIComponent(button.dataset.model)}`, { method: "DELETE" });
            await loadLocalModels();
            await loadAvailableModelOptions();
            document.dispatchEvent(new CustomEvent("models-changed"));
        });
    });
}

// ============== Local Workspaces ==============

function renderWorkspaces(workspaces) {
    if (!workspaces.length) return `<span class="settings-hint">No folders connected.</span>`;
    return workspaces.map((workspace) => {
        const perms = new Set(workspace.permissions || []);
        const builtin = workspace.source === "builtin";
        const badge = (name) => builtin
            ? `<span style="padding:2px 7px;border:1px solid var(--border);border-radius:999px;font-size:11px;color:${perms.has(name) ? "var(--green)" : "var(--text-muted)"}">${name}</span>`
            : `<label style="font-size:12px"><input class="workspace-cap" data-cap="${name}" type="checkbox" ${perms.has(name) ? "checked" : ""}> ${name}</label>`;
        return `<div class="diag-item workspace-card" data-id="${esc(workspace.workspace_id)}" style="display:block;margin-bottom:8px;padding:11px 12px">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
                <div style="min-width:0">
                    <div style="font-weight:600">${esc(workspace.name)}</div>
                    <div class="settings-hint" style="word-break:break-all;margin:3px 0 7px">${esc(workspace.root_path)}</div>
                    <div style="display:flex;gap:5px">${badge("read")}${badge("write")}${badge("execute")}</div>
                </div>
                ${builtin ? `<span class="settings-hint">built-in</span>` : `<button class="btn btn-outline btn-mini workspace-revoke" data-id="${esc(workspace.workspace_id)}">Revoke</button>`}
            </div>
        </div>`;
    }).join("");
}

function workspacePayload(path = null) {
    return {
        path,
        read: true,
        write: Boolean(document.getElementById("workspace-write")?.checked),
        execute: Boolean(document.getElementById("workspace-execute")?.checked),
    };
}

async function workspaceRequest(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "Workspace request failed.");
    return data;
}

function bindWorkspaces() {
    const execute = document.getElementById("workspace-execute");
    const warning = document.getElementById("workspace-warning");
    execute?.addEventListener("change", () => {
        warning.textContent = execute.checked
            ? "Execute is powerful: approved commands run with your Windows user privileges and each run still requires confirmation."
            : "";
    });
    document.getElementById("btn-workspace-choose")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        button.disabled = true;
        warning.textContent = "Waiting for the Windows folder picker…";
        try {
            const data = await workspaceRequest("/api/workspaces/select-folder", {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(workspacePayload()),
            });
            warning.textContent = data.cancelled ? "Folder selection cancelled." : "Workspace connected.";
            if (!data.cancelled) setTimeout(() => loadSettings(), 350);
        } catch (error) {
            warning.textContent = error.message;
            warning.style.color = "var(--red)";
        } finally {
            button.disabled = false;
        }
    });
    document.getElementById("btn-workspace-add-path")?.addEventListener("click", async () => {
        const input = document.getElementById("workspace-manual-path");
        const path = input?.value.trim();
        if (!path) return;
        try {
            await workspaceRequest("/api/workspaces", {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(workspacePayload(path)),
            });
            setTimeout(() => loadSettings(), 350);
        } catch (error) {
            warning.textContent = error.message;
            warning.style.color = "var(--red)";
        }
    });
    document.querySelectorAll(".workspace-revoke").forEach((button) => button.addEventListener("click", async () => {
        const confirmed = await showConfirm("Revoke workspace", "Remove Remy's access to this folder?");
        if (!confirmed) return;
        try {
            await workspaceRequest(`/api/workspaces/${encodeURIComponent(button.dataset.id)}`, { method: "DELETE" });
            setTimeout(() => loadSettings(), 250);
        } catch (error) {
            warning.textContent = error.message;
            warning.style.color = "var(--red)";
        }
    }));
    document.querySelectorAll(".workspace-cap").forEach((checkbox) => checkbox.addEventListener("change", async () => {
        const card = checkbox.closest(".workspace-card");
        const id = card?.dataset.id;
        if (!id) return;
        if (checkbox.dataset.cap === "execute" && checkbox.checked) {
            const confirmed = await showConfirm(
                "Enable Execute",
                "Commands in this folder will run with your Windows user privileges. Each command will still require approval. Continue?",
            );
            if (!confirmed) {
                checkbox.checked = false;
                return;
            }
        }
        const enabled = [...card.querySelectorAll(".workspace-cap:checked")].map((item) => item.dataset.cap);
        if (!enabled.length) {
            checkbox.checked = true;
            warning.textContent = "Keep at least one capability or revoke the workspace.";
            return;
        }
        try {
            await workspaceRequest(`/api/workspaces/${encodeURIComponent(id)}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    read: enabled.includes("read"), write: enabled.includes("write"), execute: enabled.includes("execute"),
                }),
            });
            warning.textContent = "Workspace permissions updated.";
        } catch (error) {
            checkbox.checked = !checkbox.checked;
            warning.textContent = error.message;
            warning.style.color = "var(--red)";
        }
    }));
}

// ============== Helpers ==============

function esc(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

