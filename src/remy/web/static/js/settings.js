/**
 * Settings View — configuration, model registry, system prompt, diagnostics.
 */

import { showConfirm } from "./ui.js";

const contentEl = document.getElementById("settings-content");

// Cached registered models — used to auto-fill API key when adding a new model
let _cachedRegisteredModels = [];

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
        const [settingsData, diagData, secretsData] = await Promise.all([
            fetch("/api/settings").then((r) => r.json()),
            fetch("/api/diagnostics").then((r) => r.json()),
            fetch("/api/secrets").then((r) => r.json()),
        ]);
        renderSettings(settingsData, diagData, secretsData);
    } catch (e) {
        contentEl.innerHTML = `<p style="color:var(--red)">Failed to load settings: ${e.message}</p>`;
    }
}

function renderSettings(cfg, diag, secretsData = { secrets: [] }) {
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
                            <option value="deepseek">DeepSeek</option>
                            <option value="xai">xAI</option>
                            <option value="ollama">Ollama</option>
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
            <h3 class="settings-section-title">Local Models (Ollama)</h3>
            <p class="settings-hint" style="margin-bottom:14px">
                Run models directly on your computer — no internet, no API keys, full privacy.
                Remy will download everything needed automatically.
            </p>
            <div id="ollama-status-bar" style="margin-bottom:14px"></div>
            <div id="ollama-installed-list" style="margin-bottom:16px"></div>
            <div class="settings-subsection-title" style="margin-bottom:10px">Recommended models</div>
            <div id="ollama-popular-grid" class="ollama-popular-grid">
                <span class="settings-hint">Loading…</span>
            </div>
            <details style="margin-top:14px">
                <summary style="cursor:pointer;color:var(--accent);font-size:13px;font-weight:600">Install any model manually</summary>
                <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
                    <input type="text" id="ollama-custom-name" class="input" placeholder="gemma3:4b" style="flex:1;padding:6px 10px;max-width:260px">
                    <button class="btn btn-primary btn-sm" id="btn-ollama-custom-pull">Install</button>
                    <span style="color:var(--text-muted);font-size:12px">
                        Model names: <a href="https://ollama.com/library" target="_blank" style="color:var(--accent)">ollama.com/library</a>
                    </span>
                </div>
            </details>
        </div>

        <!-- Ollama pull progress modal -->
        <div id="ollama-pull-modal" class="ollama-modal hidden">
            <div class="ollama-modal-box">
                <div class="ollama-modal-title" id="ollama-pull-title">Downloading model…</div>
                <div class="bulk-progress-bar-track" style="margin:12px 0">
                    <div id="ollama-pull-bar" class="bulk-progress-bar" style="width:0%"></div>
                </div>
                <div id="ollama-pull-log" class="bulk-log" style="max-height:160px"></div>
                <button id="ollama-pull-close" class="btn btn-outline btn-sm hidden" style="margin-top:10px">Close</button>
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
    bindLocalSecrets();
    document.getElementById("btn-add-model")?.addEventListener("click", addModel);
    document.getElementById("add-model-provider")?.addEventListener("change", _onProviderChange);
    document.querySelector(".settings-add-model")?.addEventListener("toggle", () => _onProviderChange());
    initPushSection();
    loadAuraStatus();
    loadLocalModels();

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
            _populateModelSelects(models);
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
                    await fetch("/api/model-registry", {
                        method: "PUT",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ model_name: name, api_key: newKey, provider }),
                    });
                    input.value = "";
                    await loadModelRegistry();
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
                    await fetch(`/api/model-registry/${encodeURIComponent(name)}`, { method: "DELETE" });
                    loadModelRegistry();
                    showStatus(`Removed: ${name}`);
                } catch (e) {
                    showStatus(`Error: ${e.message}`, true);
                }
            });
        });

        _populateModelSelects(models);
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
            opt.textContent = `${m.name}  [${m.provider}]`;
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
        keyInput.dataset.reuseFrom = "";
        keyInput.placeholder = "API key";
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
    if (!key && provider !== "ollama" && !reuseFrom) { alert("API key is required."); return; }

    try {
        // If reusing a key from another model of the same provider, copy it server-side
        const payload = { model_name: name, api_key: key || "", provider };
        if (!key && reuseFrom) payload.copy_key_from = reuseFrom;
        if (inputPrice != null) payload.input_price = inputPrice;
        if (outputPrice != null) payload.output_price = outputPrice;

        await fetch("/api/model-registry", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        document.getElementById("add-model-name").value = "";
        const ki = document.getElementById("add-model-key");
        if (ki) { ki.value = ""; ki.dataset.reuseFrom = ""; ki.placeholder = "API key"; }
        document.getElementById("add-model-input-price").value = "";
        document.getElementById("add-model-output-price").value = "";
        const hint = document.getElementById("add-model-reuse-hint");
        if (hint) hint.style.display = "none";
        loadModelRegistry();
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

// ============== Local Models (Ollama) ==============

async function loadLocalModels() {
    await Promise.all([_renderOllamaStatus(), _renderPopularModels()]);
    _bindOllamaControls();
}

async function _renderOllamaStatus() {
    const bar = document.getElementById("ollama-status-bar");
    const list = document.getElementById("ollama-installed-list");
    if (!bar) return;

    let status;
    try {
        status = await fetch("/api/ollama/status").then(r => r.json());
    } catch {
        bar.innerHTML = `<span class="settings-hint" style="color:var(--red)">Could not check Ollama status.</span>`;
        return;
    }

    if (status.running) {
        bar.innerHTML = `
            <div style="display:flex;align-items:center;gap:8px;font-size:13px">
                <span style="color:var(--green);font-size:16px">●</span>
                <span style="color:var(--green);font-weight:600">Ollama is running</span>
                <span style="color:var(--text-muted)">${status.binary_path ? `(${status.binary_path})` : ''}</span>
            </div>`;
    } else {
        bar.innerHTML = `
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
                <span style="color:var(--text-muted);font-size:13px">● Ollama is not running</span>
                <button class="btn btn-outline btn-sm" id="btn-ollama-start">
                    ${status.binary_found ? 'Start' : 'Install &amp; Start'}
                </button>
            </div>`;
    }

    // Installed models list
    if (list) {
        if (status.models && status.models.length > 0) {
            list.innerHTML = `
                <div class="settings-subsection-title" style="margin-bottom:8px">Installed models</div>
                <div class="ollama-installed-models">
                    ${status.models.map(m => `
                        <div class="ollama-installed-row">
                            <span class="ollama-model-name">${esc(m.name)}</span>
                            <span class="ollama-model-size">${m.size_gb} GB</span>
                            <button class="btn btn-outline btn-mini ollama-use-btn" data-model="ollama:${esc(m.name)}">Use</button>
                            <button class="btn-icon ollama-delete-btn" data-model="${esc(m.name)}" title="Delete">🗑</button>
                        </div>
                    `).join('')}
                </div>`;
        } else if (status.running) {
            list.innerHTML = `<p class="settings-hint">No models yet. Choose one below.</p>`;
        } else {
            list.innerHTML = "";
        }
    }
}

async function _renderPopularModels() {
    const grid = document.getElementById("ollama-popular-grid");
    if (!grid) return;
    let data;
    try {
        data = await fetch("/api/ollama/models/popular").then(r => r.json());
    } catch {
        grid.innerHTML = `<span class="settings-hint" style="color:var(--red)">Could not load model list.</span>`;
        return;
    }

    const TAG_LABELS = { fast: "⚡ Fast", balanced: "⚖ Balanced", powerful: "💪 Powerful", reasoning: "🧠 Reasoning", multilingual: "🌍 Multilingual", small: "🪶 Small", popular: "⭐ Popular" };

    grid.innerHTML = (data.models || []).map(m => `
        <div class="ollama-model-card ${m.installed ? 'installed' : ''}">
            <div class="ollama-card-header">
                <span class="ollama-card-name">${esc(m.label)}</span>
                <span class="ollama-card-size">${esc(m.size)}</span>
            </div>
            <p class="ollama-card-desc">${esc(m.description)}</p>
            <div class="ollama-card-footer">
                <div class="ollama-card-tags">
                    ${(m.tags || []).map(t => `<span class="ollama-tag">${TAG_LABELS[t] || t}</span>`).join('')}
                </div>
                ${m.installed
                    ? `<span class="ollama-installed-badge">✓ Installed</span>`
                    : `<button class="btn btn-primary btn-sm ollama-pull-btn" data-model="${esc(m.name)}" data-label="${esc(m.label)}">Install</button>`
                }
            </div>
        </div>
    `).join('');
}

function _bindOllamaControls() {
    // Start button
    document.getElementById("btn-ollama-start")?.addEventListener("click", async () => {
        const modal = document.getElementById("ollama-pull-modal");
        const title = document.getElementById("ollama-pull-title");
        const log = document.getElementById("ollama-pull-log");
        const bar = document.getElementById("ollama-pull-bar");
        const closeBtn = document.getElementById("ollama-pull-close");
        if (title) title.textContent = "Starting Ollama…";
        if (log) log.innerHTML = "";
        if (bar) bar.style.width = "0%";
        if (closeBtn) closeBtn.classList.add("hidden");
        modal?.classList.remove("hidden");

        const resp = await fetch("/api/ollama/start", { method: "POST" });
        const reader = resp.body.getReader();
        await _consumeSSE(reader, log, bar, closeBtn, async () => {
            modal?.classList.add("hidden");
            await loadLocalModels();
        });
    });

    // Install buttons on popular cards
    document.querySelectorAll(".ollama-pull-btn").forEach(btn => {
        btn.addEventListener("click", () => _startPull(btn.dataset.model, btn.dataset.label));
    });

    // Custom model pull
    document.getElementById("btn-ollama-custom-pull")?.addEventListener("click", () => {
        const name = document.getElementById("ollama-custom-name")?.value.trim();
        if (name) _startPull(name, name);
    });

    // Delete buttons
    document.querySelectorAll(".ollama-delete-btn").forEach(btn => {
        btn.addEventListener("click", async () => {
            const name = btn.dataset.model;
            if (!confirm(`Delete model "${name}"?\nThis will free up disk space.`)) return;
            btn.disabled = true;
            try {
                await fetch("/api/ollama/model", { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
            } finally {
                await loadLocalModels();
            }
        });
    });

    // "Use" buttons — set as active model
    document.querySelectorAll(".ollama-use-btn").forEach(btn => {
        btn.addEventListener("click", async () => {
            const model = btn.dataset.model;
            await fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ summary_model: model }) });
            btn.textContent = "✓ Active";
            btn.disabled = true;
        });
    });
}

async function _startPull(modelName, modelLabel) {
    const modal = document.getElementById("ollama-pull-modal");
    const title = document.getElementById("ollama-pull-title");
    const log = document.getElementById("ollama-pull-log");
    const bar = document.getElementById("ollama-pull-bar");
    const closeBtn = document.getElementById("ollama-pull-close");

    if (title) title.textContent = `Installing ${modelLabel}…`;
    if (log) log.innerHTML = "";
    if (bar) bar.style.width = "0%";
    if (closeBtn) closeBtn.classList.add("hidden");
    modal?.classList.remove("hidden");

    const resp = await fetch("/api/ollama/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: modelName }),
    });
    const reader = resp.body.getReader();
    await _consumeSSE(reader, log, bar, closeBtn, async () => {
        await loadLocalModels();
    });
}

async function _consumeSSE(reader, log, bar, closeBtn, onDone) {
    const decoder = new TextDecoder();
    let buf = "";

    const _logLine = (text, cls = "") => {
        if (!log) return;
        const line = document.createElement("div");
        line.className = "bulk-log-line" + (cls ? " " + cls : "");
        line.textContent = text;
        log.appendChild(line);
        log.scrollTop = log.scrollHeight;
    };

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            let evt;
            try { evt = JSON.parse(line.slice(6)); } catch { continue; }
            const phase = evt.phase || "";
            const msg = evt.message || "";
            if (phase === "error") {
                _logLine(msg || "Error", "error");
            } else if (phase === "done") {
                if (bar) bar.style.width = "100%";
                _logLine(msg || "Done!", "done");
            } else if (msg) {
                if (bar && evt.pct != null) bar.style.width = evt.pct + "%";
                _logLine(msg);
            }
        }
    }

    if (closeBtn) {
        closeBtn.classList.remove("hidden");
        closeBtn.onclick = async () => {
            document.getElementById("ollama-pull-modal")?.classList.add("hidden");
            if (onDone) await onDone();
        };
    }
}

// ============== Helpers ==============

function esc(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

