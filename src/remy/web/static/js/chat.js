/**
 * Chat View — WebSocket messaging, voice recording, file upload, TTS.
 */

const messagesEl = document.getElementById("chat-messages");

// ============== USER-ACTIVATED TEAM MODE ==============

const _teamModeSelect = document.getElementById("chat-team-mode");
const _storedTeamMode = localStorage.getItem("remy.teamMode");
if (_teamModeSelect && ["off", "adaptive"].includes(_storedTeamMode)) {
    _teamModeSelect.value = _storedTeamMode;
}
_teamModeSelect?.addEventListener("change", () => {
    // Force is deliberately one-shot and is never persisted across tasks.
    localStorage.setItem(
        "remy.teamMode",
        _teamModeSelect.value === "adaptive" ? "adaptive" : "off",
    );
});

function consumeTeamMode() {
    const mode = ["adaptive", "force"].includes(_teamModeSelect?.value)
        ? _teamModeSelect.value
        : "off";
    if (mode === "force" && _teamModeSelect) {
        _teamModeSelect.value = "off";
        localStorage.setItem("remy.teamMode", "off");
    }
    return mode;
}

// ============== MODEL SWITCHER ==============

let _modelSwitcherBound = false;
let _modelRegistryPromise = null;

function _loadModelRegistry({ refresh = false } = {}) {
    if (refresh || !_modelRegistryPromise) {
        _modelRegistryPromise = fetch("/api/models")
            .then((response) => response.json())
            .catch((error) => {
                _modelRegistryPromise = null;
                throw error;
            });
    }
    return _modelRegistryPromise;
}

async function initModelSwitcher() {
    const sel = document.getElementById("chat-model-select");
    if (!sel) return;

    try {
        const [settingsRes, registryRes] = await Promise.all([
            fetch("/api/settings").then(r => r.json()),
            _loadModelRegistry(),
        ]);
        const current = settingsRes.summary_model || "";
        const models = registryRes.models || [];

        sel.innerHTML = "";
        for (const m of models) {
            const opt = document.createElement("option");
            opt.value = m.name;
            const providerLabel = m.provider === "openrouter" ? "OR" : m.provider.slice(0, 2).toUpperCase();
            opt.textContent = `${m.label || m.name}  [${providerLabel}]`;
            if (m.name === current) opt.selected = true;
            sel.appendChild(opt);
        }
        // current not in registry
        if (!models.find(m => m.name === current) && current) {
            const opt = document.createElement("option");
            opt.value = current;
            opt.textContent = current;
            opt.selected = true;
            sel.insertBefore(opt, sel.firstChild);
        }
    } catch (_) {
        sel.innerHTML = `<option value="">—</option>`;
    }

    if (!_modelSwitcherBound) {
        sel.addEventListener("change", async () => {
            const val = sel.value;
            if (!val) return;
            try {
                await fetch("/api/settings", {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ summary_model: val }),
                });
            } catch (_) {}
        });
        _modelSwitcherBound = true;
    }
}

initModelSwitcher();

// ============== COMPARE MODE ==============

let _compareActive = false;
let _compareSelectedModels = new Set();
let _compareAvailableModels = [];

const _compareBtn = document.getElementById("btn-compare");
const _comparePanel = document.getElementById("compare-panel");
const _compareModelList = document.getElementById("compare-model-list");
const _compareCancelBtn = document.getElementById("btn-compare-cancel");

async function _initComparePanel() {
    try {
        const data = await _loadModelRegistry();
        _compareAvailableModels = data.models || [];
    } catch (_) {}
}

_initComparePanel();

document.addEventListener("models-changed", () => {
    _modelRegistryPromise = null;
    initModelSwitcher();
    _initComparePanel();
});

_compareBtn?.addEventListener("click", () => {
    if (_compareActive) {
        _exitCompareMode();
        return;
    }
    _renderComparePanel();
    _comparePanel.classList.remove("hidden");
    _compareBtn.classList.add("active");
});

_compareCancelBtn?.addEventListener("click", _exitCompareMode);

function _exitCompareMode() {
    _compareActive = false;
    _compareSelectedModels.clear();
    _comparePanel.classList.add("hidden");
    _compareBtn.classList.remove("active");
}

function _renderComparePanel() {
    _compareModelList.innerHTML = "";
    const currentModel = document.getElementById("chat-model-select")?.value || "";
    for (const m of _compareAvailableModels) {
        const label = document.createElement("label");
        label.className = "compare-model-item";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = m.name;
        if (m.name === currentModel) { cb.checked = true; _compareSelectedModels.add(m.name); }
        cb.addEventListener("change", () => {
            if (cb.checked) _compareSelectedModels.add(m.name);
            else _compareSelectedModels.delete(m.name);
            _updateCompareSendBtn();
        });
        const providerLabel = m.provider === "openrouter" ? "OR" : m.provider.slice(0, 2).toUpperCase();
        const modelName = document.createElement("span");
        modelName.className = "compare-model-name";
        modelName.textContent = m.label || m.name;
        const modelProvider = document.createElement("span");
        modelProvider.className = "compare-model-provider";
        modelProvider.textContent = `[${providerLabel}]`;
        label.append(cb, " ", modelName, " ", modelProvider);
        _compareModelList.appendChild(label);
    }
    _updateCompareSendBtn();
}

function _updateCompareSendBtn() {
    _compareActive = _compareSelectedModels.size >= 2;
    _compareBtn.textContent = _compareActive
        ? `⚡ Compare (${_compareSelectedModels.size})`
        : "⚡ Compare";
}

async function _sendCompareMessage(text) {
    const models = [..._compareSelectedModels];
    if (models.length < 2) return;

    addMessage("user", text);
    _comparePanel.classList.add("hidden");

    // Build compare bubble with tabs
    const bubble = document.createElement("div");
    bubble.className = "chat-msg assistant compare-bubble";

    const tabs = document.createElement("div");
    tabs.className = "compare-tabs";

    const panels = document.createElement("div");
    panels.className = "compare-panels";

    const buffers = {};
    const panelEls = {};
    const terminalModels = new Set();

    for (const model of models) {
        buffers[model] = "";

        const tab = document.createElement("button");
        tab.className = "compare-tab";
        tab.textContent = model;
        tab.dataset.model = model;
        tab.addEventListener("click", () => {
            tabs.querySelectorAll(".compare-tab").forEach(t => t.classList.remove("active"));
            panels.querySelectorAll(".compare-panel-content").forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            panels.querySelector(`[data-panel="${model}"]`).classList.add("active");
        });
        tabs.appendChild(tab);

        const panel = document.createElement("div");
        panel.className = "compare-panel-content";
        panel.dataset.panel = model;
        panel.innerHTML = `<div class="compare-loading">Waiting for ${model}…</div>`;
        panels.appendChild(panel);
        panelEls[model] = panel;
    }

    // Activate first tab
    tabs.firstChild?.classList.add("active");
    panels.firstChild?.classList.add("active");

    bubble.appendChild(tabs);
    bubble.appendChild(panels);
    messagesEl.appendChild(bubble);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Open WebSocket
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}/api/ws/compare`);

    ws.onopen = () => {
        ws.send(JSON.stringify({ text, models }));
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "token" && data.model) {
            const panel = panelEls[data.model];
            if (panel) {
                buffers[data.model] += data.content;
                panel.innerHTML = formatMarkdown(buffers[data.model]);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            }
        } else if (data.type === "done" && data.model) {
            terminalModels.add(data.model);
            const tab = tabs.querySelector(`[data-model="${data.model}"]`);
            if (tab) tab.classList.add("compare-tab--done");
        } else if (data.type === "error" && data.model) {
            terminalModels.add(data.model);
            const panel = panelEls[data.model];
            if (panel) {
                panel.innerHTML = "";
                const error = document.createElement("div");
                error.className = "compare-error";
                error.textContent = `Error: ${data.content || "Model failed"}`;
                panel.appendChild(error);
            }
        } else if (data.type === "error") {
            const error = document.createElement("div");
            error.className = "compare-error";
            error.textContent = `Compare error: ${data.content || "Unknown error"}`;
            bubble.appendChild(error);
        } else if (data.type === "all_done") {
            ws.close(1000, "Comparison complete");
        }
    };

    ws.onerror = () => {
        bubble.insertAdjacentHTML("beforeend", '<div class="compare-error">Connection error</div>');
    };

    ws.onclose = () => {
        for (const model of models) {
            if (terminalModels.has(model)) continue;
            const panel = panelEls[model];
            if (!panel) continue;
            panel.innerHTML = "";
            const error = document.createElement("div");
            error.className = "compare-error";
            error.textContent = "Connection closed before this model returned a response.";
            panel.appendChild(error);
        }
    };
}

const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("btn-send");
const voiceBtn = document.getElementById("btn-voice");
const attachBtn = document.getElementById("btn-attach");
const codeWorkspaceBtn = document.getElementById("project-context-selector");
const fileInput = document.getElementById("file-input");
const filePreview = document.getElementById("file-preview");
const filePreviewName = document.getElementById("file-preview-name");
const removeFileBtn = document.getElementById("btn-remove-file");
const codeWorkspaceBar = document.getElementById("project-context-bar");
const codeWorkspaceName = document.getElementById("project-context-name");
const codeWorkspaceMeta = document.getElementById("project-context-meta");
const closeWorkspaceBtn = document.getElementById("project-context-close");
const chatInputArea = document.querySelector(".chat-input-area");
const dropOverlay = document.getElementById("file-drop-overlay");
const ttsCheckbox = document.getElementById("tts-enabled");
const contextReducerCompareCheckbox = document.getElementById("context-reducer-compare");
const contextReducerApplyCheckbox = document.getElementById("context-reducer-apply");
const contextReducerLabBtn = document.getElementById("btn-context-reducer-lab");
const contextReducerLab = document.getElementById("context-reducer-lab");
const contextReducerCloseBtn = document.getElementById("btn-context-reducer-close");
const contextReducerClearBtn = document.getElementById("btn-context-reducer-clear");
const contextReducerExportBtn = document.getElementById("btn-context-reducer-export");
const contextReducerSummaryEl = document.getElementById("context-reducer-summary");
const contextReducerRunsEl = document.getElementById("context-reducer-runs");
const contextReducerModelSelect = document.getElementById("context-reducer-model");
const contextReducerModelPriceEl = document.getElementById("context-reducer-model-price");
let contextReducerModels = [];

function getSelectedLabModel() {
    return contextReducerModelSelect?.value || "";
}

function updateLabModelPriceHint() {
    if (!contextReducerModelPriceEl) return;
    const sel = getSelectedLabModel();
    const m = contextReducerModels.find((x) => x.model === sel);
    if (!m) { contextReducerModelPriceEl.textContent = ""; return; }
    const inP = m.input_price_per_1m_usd;
    contextReducerModelPriceEl.textContent = (inP === null || inP === undefined)
        ? "no price in registry — cost will show $0"
        : `input $${inP}/1M tokens`;
}

async function populateLabModels() {
    if (!contextReducerModelSelect) return;
    try {
        const payload = await window.apiClient.getLlmOptimizationModels();
        contextReducerModels = Array.isArray(payload.models) ? payload.models : [];
        const prev = getSelectedLabModel();
        contextReducerModelSelect.innerHTML = contextReducerModels.map((m) => {
            const inP = m.input_price_per_1m_usd;
            const priceLabel = (inP === null || inP === undefined) ? "no price" : `$${inP}/1M`;
            return `<option value="${escapeHtml(m.model)}">${escapeHtml(m.model)} (${priceLabel})</option>`;
        }).join("");
        // Prefer a Gemini model with a real price so cost shows immediately.
        const priced = contextReducerModels.find((m) => String(m.model).toLowerCase().startsWith("gemini") && m.input_price_per_1m_usd);
        if (prev) contextReducerModelSelect.value = prev;
        else if (priced) contextReducerModelSelect.value = priced.model;
        updateLabModelPriceHint();
    } catch (err) {
        console.warn("Failed to load lab models", err);
    }
}

contextReducerModelSelect?.addEventListener("change", updateLabModelPriceHint);

let typingEl = null;

// ============== MARKDOWN ==============

function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatMarkdown(text) {
    if (!text) return "";

    // Extract code blocks first to protect them from other formatting
    const codeBlocks = [];
    let html = text.replace(/```([\s\S]*?)```/g, (_, code) => {
        codeBlocks.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
        return `\x00CB${codeBlocks.length - 1}\x00`;
    });

    // Extract inline code
    const inlineCode = [];
    html = html.replace(/`([^`]+)`/g, (_, code) => {
        inlineCode.push(`<code>${escapeHtml(code)}</code>`);
        return `\x00IC${inlineCode.length - 1}\x00`;
    });

    // Escape remaining HTML
    html = escapeHtml(html);

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Images (before links!) — ![alt](src)
    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,
        '<img src="$2" alt="$1" class="chat-msg-image" style="max-width:400px;border-radius:8px;margin:8px 0">');

    // JSON image objects — { "image": "/api/generated_images/file.jpg" }
    // LLM sometimes returns raw JSON instead of markdown
    html = html.replace(/\{\s*"image"\s*:\s*"(\/api\/(?:generated_images|browser_screenshots)\/[\w._-]+\.(?:png|jpe?g|webp))"\s*\}/gi,
        '<img src="$1" alt="Generated image" class="chat-msg-image" style="max-width:400px;border-radius:8px;margin:8px 0">');

    // Bare image URLs (generated_images, browser_screenshots) — auto-embed
    // Matches /api/generated_images/file.jpg or http://host/api/generated_images/file.jpg
    // but NOT when already inside an <img src="..."> tag (checks for no preceding src=")
    html = html.replace(/(?<![="'])(?:https?:\/\/[^/\s]+)?(\/api\/(?:generated_images|browser_screenshots)\/[\w._-]+\.(?:png|jpe?g|webp))/gi,
        '<img src="$1" alt="Generated image" class="chat-msg-image" style="max-width:400px;border-radius:8px;margin:8px 0">');

    // Markdown links to images — [text](/api/generated_images/file.jpg) → <img>
    // Must be BEFORE generic links regex so these don't become <a> tags
    html = html.replace(/\[([^\]]*)\]\(((?:https?:\/\/[^/\s]+)?\/api\/(?:generated_images|browser_screenshots)\/[\w._-]+\.(?:png|jpe?g|webp))\)/gi,
        '<img src="$2" alt="$1" class="chat-msg-image" style="max-width:400px;border-radius:8px;margin:8px 0">');

    // PDF report links — [text](/api/reports/file.pdf)
    html = html.replace(/\[([^\]]+)\]\((\/api\/reports\/[^\)]+\.pdf)\)/g,
        '<a href="$2" target="_blank" rel="noopener" class="chat-report-link" style="display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:#f0f4ff;border:1px solid #c5d5ff;border-radius:8px;text-decoration:none;color:#1a1a2e;margin:8px 0">' +
        '<span style="font-size:18px">&#128196;</span> $1</a>');

    // PPTX presentation links — [text](/api/presentations/file.pptx)
    html = html.replace(/\[([^\]]+)\]\((\/api\/presentations\/[^\)]+\.pptx)\)/g,
        '<a href="$2" rel="noopener" class="chat-report-link" style="display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:#fff4f0;border:1px solid #ffc5b5;border-radius:8px;text-decoration:none;color:#1a1a2e;margin:8px 0">' +
        '<span style="font-size:18px">&#128202;</span> $1</a>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Unordered lists
    html = html.replace(/(?:^|\n)((?:- .+\n?)+)/g, (_, block) => {
        const items = block.trim().split('\n').map(line =>
            `<li>${line.replace(/^- /, '')}</li>`
        ).join('');
        return `<ul>${items}</ul>`;
    });

    // Ordered lists
    html = html.replace(/(?:^|\n)((?:\d+\. .+\n?)+)/g, (_, block) => {
        const items = block.trim().split('\n').map(line =>
            `<li>${line.replace(/^\d+\. /, '')}</li>`
        ).join('');
        return `<ol>${items}</ol>`;
    });

    // Horizontal rule
    html = html.replace(/^---$/gm, '<hr>');

    // Newlines to <br> (but not inside block elements)
    html = html.replace(/\n/g, '<br>');

    // Restore code blocks and inline code
    html = html.replace(/\x00CB(\d+)\x00/g, (_, i) => codeBlocks[i]);
    html = html.replace(/\x00IC(\d+)\x00/g, (_, i) => inlineCode[i]);

    return html;
}

// ============== MESSAGES ==============

function estimateOutputTokens(text) {
    const value = String(text || "");
    if (!value) return 0;
    const nonAscii = (value.match(/[^\x00-\x7F]/g) || []).length;
    const ascii = value.length - nonAscii;
    return Math.max(1, Math.ceil((ascii / 4) + (nonAscii / 2.5)));
}

async function copyResponseText(button, text) {
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
        } else {
            const textarea = document.createElement("textarea");
            textarea.value = text;
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand("copy");
            textarea.remove();
        }
        const label = button.querySelector("span");
        if (label) label.textContent = "Copied";
        button.classList.add("copied");
        setTimeout(() => {
            if (label) label.textContent = "Copy";
            button.classList.remove("copied");
        }, 1400);
    } catch (error) {
        console.warn("Could not copy response", error);
        const label = button.querySelector("span");
        if (label) label.textContent = "Copy failed";
    }
}

function attachResponseControls(message, usage = null) {
    if (!message?.classList.contains("assistant")) return;
    let controls = message.querySelector(":scope > .chat-response-controls");
    if (!controls) {
        controls = document.createElement("div");
        controls.className = "chat-response-controls";

        const copyButton = document.createElement("button");
        copyButton.type = "button";
        copyButton.className = "chat-copy-response";
        copyButton.title = "Copy response as Markdown";
        copyButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="2"></rect><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"></path></svg><span>Copy</span>';
        copyButton.addEventListener("click", () => copyResponseText(
            copyButton,
            message.dataset.rawContent || ""
        ));
        controls.appendChild(copyButton);

        const tokenUsage = document.createElement("span");
        tokenUsage.className = "chat-token-usage";
        controls.appendChild(tokenUsage);
        message.appendChild(controls);
    }

    const tokenEl = controls.querySelector(".chat-token-usage");
    if (!tokenEl) return;
    tokenEl.classList.remove("partial", "estimated");
    const reportedCalls = Number(usage?.reported_calls || 0);
    const totalCalls = Number(usage?.total_calls || 0);
    const total = Number(usage?.total_tokens || 0);
    if (usage?.exact && total > 0) {
        const input = Number(usage.input_tokens || 0);
        const output = Number(usage.output_tokens || 0);
        tokenEl.textContent = `${total.toLocaleString()} tokens · ${input.toLocaleString()} in / ${output.toLocaleString()} out`;
        tokenEl.title = `Exact usage reported by all ${totalCalls} model call${totalCalls === 1 ? "" : "s"}.`;
    } else if (reportedCalls > 0 && total > 0) {
        tokenEl.textContent = `≥ ${total.toLocaleString()} tokens`;
        tokenEl.classList.add("partial");
        tokenEl.title = `Provider-reported minimum: ${reportedCalls} of ${totalCalls} model calls returned usage.`;
    } else {
        const estimate = estimateOutputTokens(message.dataset.rawContent || "");
        tokenEl.textContent = `≈ ${estimate.toLocaleString()} output tokens`;
        tokenEl.classList.add("estimated");
        tokenEl.title = "Estimated from response text because this provider did not report token usage. Input/context tokens are not included.";
    }
}

let _activeConversationId = "";
const _conversationTranscriptCache = new Map();
const _conversationTranscriptRequests = new Map();
const CONVERSATION_INITIAL_LIMIT = 120;
const CONVERSATION_CACHE_TTL_MS = 15_000;
const CONVERSATION_CACHE_SIZE = 6;

function addMessage(role, content, extraClass = "", options = {}) {
    const {
        target = messagesEl,
        scroll = true,
        removeTypingIndicator = true,
        trackConversationChange = true,
    } = options;
    if (removeTypingIndicator) removeTyping();
    if (trackConversationChange && _activeConversationId) {
        _conversationTranscriptCache.delete(_activeConversationId);
    }
    const div = document.createElement("div");
    div.className = `chat-msg ${role} ${extraClass}`.trim();
    if (role === "assistant") {
        div.dataset.rawContent = content;
        div.innerHTML = formatMarkdown(content);
        attachResponseControls(div);
    } else {
        div.textContent = content;
    }
    target.appendChild(div);
    if (scroll && target === messagesEl) {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    return div;
}

let _conversationLoadSequence = 0;

function _cacheConversationTranscript(conversationId, payload) {
    _conversationTranscriptCache.delete(conversationId);
    _conversationTranscriptCache.set(conversationId, {
        ...payload,
        fetchedAt: Date.now(),
    });
    while (_conversationTranscriptCache.size > CONVERSATION_CACHE_SIZE) {
        const oldestId = _conversationTranscriptCache.keys().next().value;
        _conversationTranscriptCache.delete(oldestId);
    }
}

function _renderConversationTranscript(conversation, payload) {
    const fragment = document.createDocumentFragment();
    if (payload.has_more) {
        const loadEarlier = document.createElement("button");
        loadEarlier.type = "button";
        loadEarlier.className = "chat-load-earlier";
        loadEarlier.textContent = "Load earlier messages";
        loadEarlier.addEventListener("click", () => {
            const nextLimit = Math.min(
                Math.max(Number(payload.limit) || CONVERSATION_INITIAL_LIMIT, CONVERSATION_INITIAL_LIMIT) * 2,
                1999,
            );
            loadConversationTranscript(conversation, { force: true, limit: nextLimit });
        });
        fragment.appendChild(loadEarlier);
    }
    for (const message of payload.messages || []) {
        if (message.role !== "user" && message.role !== "assistant") continue;
        addMessage(message.role, message.content || "", "", {
            target: fragment,
            scroll: false,
            removeTypingIndicator: false,
            trackConversationChange: false,
        });
    }
    removeTyping();
    messagesEl.replaceChildren(fragment);
    window.requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

function _conversationTranscriptRequest(conversationId, limit) {
    const requestKey = `${conversationId}:${limit}`;
    if (!_conversationTranscriptRequests.has(requestKey)) {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), 8000);
        const request = window.apiClient.getConversationMessages(
            conversationId,
            limit,
            { signal: controller.signal },
        ).finally(() => {
            window.clearTimeout(timeout);
            _conversationTranscriptRequests.delete(requestKey);
        });
        _conversationTranscriptRequests.set(requestKey, request);
    }
    return _conversationTranscriptRequests.get(requestKey);
}

async function loadConversationTranscript(
    conversation,
    { force = false, limit = CONVERSATION_INITIAL_LIMIT, preferCache = false } = {},
) {
    if (!conversation?.conversation_id || !window.apiClient) return;
    const sequence = ++_conversationLoadSequence;
    const conversationId = conversation.conversation_id;
    const cached = _conversationTranscriptCache.get(conversationId);
    const cacheIsFresh = cached
        && Date.now() - cached.fetchedAt < CONVERSATION_CACHE_TTL_MS
        && Number(cached.limit || 0) >= limit;
    if (cached && (preferCache || cacheIsFresh) && !force) {
        _renderConversationTranscript(conversation, cached);
        if (cacheIsFresh || preferCache) return;
    } else if (!cached) {
        const status = document.createElement("div");
        status.className = "conversation-load-status";
        status.textContent = "Loading conversation…";
        messagesEl.replaceChildren(status);
    }
    messagesEl.classList.add("loading-conversation");
    try {
        const payload = await _conversationTranscriptRequest(conversationId, limit);
        if (sequence !== _conversationLoadSequence) return;
        _cacheConversationTranscript(conversationId, payload);
        _renderConversationTranscript(conversation, payload);
    } catch (error) {
        console.error("Failed to load conversation transcript", error);
        if (sequence === _conversationLoadSequence) {
            messagesEl.innerHTML = "";
            addMessage(
                "assistant",
                error?.name === "AbortError"
                    ? "This chat history took too long to load. Your stored conversation is safe; try opening it again."
                    : "I could not load this chat history. The stored conversation was not deleted.",
            );
        }
    } finally {
        if (sequence === _conversationLoadSequence) {
            messagesEl.classList.remove("loading-conversation");
        }
    }
}

document.addEventListener("conversation-changed", (event) => {
    const conversation = event.detail?.conversation;
    if (conversation?.conversation_id) {
        _activeConversationId = conversation.conversation_id;
    }
    if (event.detail?.skipTranscriptReload) return;
    loadConversationTranscript(conversation, {
        preferCache: Boolean(event.detail?.preferCachedTranscript),
    });
});

document.addEventListener("fork-replay-ready", async (event) => {
    const fork = event.detail || {};
    const prompt = String(fork.next_prompt || "");
    inputEl.value = prompt;
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
    inputEl.placeholder = prompt
        ? "Review the replay prompt, then send when readyвЂ¦"
        : "Continue from the selected trajectory boundaryвЂ¦";

    const preferredModel = String(fork.preferred_model || "");
    if (preferredModel) {
        let select = document.getElementById("chat-model-select");
        if (!Array.from(select?.options || []).some((option) => option.value === preferredModel)) {
            await initModelSwitcher();
            select = document.getElementById("chat-model-select");
        }
        if (Array.from(select?.options || []).some((option) => option.value === preferredModel)) {
            select.value = preferredModel;
            select.dispatchEvent(new Event("change"));
        }
    }
    inputEl.focus();
});

function addImagePreview(base64, mimeType) {
    const container = document.createElement("div");
    container.className = "chat-msg user";

    const img = document.createElement("img");
    img.src = `data:${mimeType};base64,${base64}`;
    img.className = "chat-msg-image";
    container.appendChild(img);

    messagesEl.appendChild(container);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addToolCall(text) {
    const div = document.createElement("div");
    div.className = "chat-msg tool-call";
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function getLastAssistantMessage() {
    const items = messagesEl.querySelectorAll(".chat-msg.assistant");
    return items.length ? items[items.length - 1] : null;
}

// ============== BRAIN META PLATE ==============
// A small dim line under each assistant reply showing live brain state:
// belief count, thermal posture, routing cycle, plasticity health.
// Data comes from /api/chat/brain-meta — a cheap snapshot of already
// persisted artifacts (thermal map + maintenance routing + plasticity).

function _postureFromTemp(mean, hotZones) {
    if (mean >= 0.30 || hotZones >= 8) return { label: "hot", cls: "hot" };
    if (mean >= 0.18 || hotZones >= 3) return { label: "warm", cls: "warm" };
    if (mean <= 0.08)                  return { label: "cold", cls: "cold" };
    return { label: "cool", cls: "cool" };
}

function _routingHint(mode) {
    // Translate internal routing mode into a short human hint.
    switch ((mode || "").toLowerCase()) {
        case "cold_skip":    return "cold-skip → lighter maintenance";
        case "cold":         return "cold cycle";
        case "warm":         return "warm cycle";
        case "hot":          return "hot cycle → deep attention";
        case "hot_first":    return "hot-first routing";
        case "full_scan":    return "full scan";
        default:             return mode || "idle";
    }
}

function _renderBrainMetaPlate(meta) {
    if (!meta || meta.status === "no_brain") return "";
    const posture = _postureFromTemp(meta.mean_temp || 0, meta.hot_zones || 0);
    const pl = meta.plasticity || {};
    const parts = [];
    parts.push(`<span class="bm-chip bm-beliefs">🧠 ${meta.beliefs} beliefs · ${meta.edges} edges</span>`);
    parts.push(`<span class="bm-chip bm-posture bm-${posture.cls}">${posture.label} · ${(meta.mean_temp * 100).toFixed(1)}%</span>`);
    if ((meta.hot_zones || 0) > 0) {
        parts.push(`<span class="bm-chip bm-hot">${meta.hot_zones} hot zone${meta.hot_zones === 1 ? "" : "s"}</span>`);
    }
    parts.push(`<span class="bm-chip bm-routing">${_routingHint(meta.routing_mode)}</span>`);
    if (pl.weakened || pl.pruned) {
        parts.push(`<span class="bm-chip bm-plasticity">synapses: ${pl.healthy || 0}✓ · ${pl.weakened || 0}⚠ · ${pl.pruned || 0}✗</span>`);
    }
    return `<div class="brain-meta-plate">${parts.join("")}</div>`;
}

async function attachBrainMetaPlate() {
    const target = getLastAssistantMessage();
    if (!target) return;
    if (target.querySelector(".brain-meta-plate")) return; // don't duplicate
    try {
        const res = await fetch("/api/chat/brain-meta");
        if (!res.ok) return;
        const meta = await res.json();
        const html = _renderBrainMetaPlate(meta);
        if (html) target.insertAdjacentHTML("beforeend", html);
    } catch (e) {
        // Silent — meta plate is best-effort.
        console.debug("brain-meta plate skipped:", e.message);
    }
}

// ── Brain Voice — proactive messages from the brain ────────────────────────

const _BRAIN_VOICE_SHOWN = new Set();
let _brainVoicePollTimer = null;

function _brainVoiceLocale() {
    const lang = (navigator.language || "en").toLowerCase();
    if (lang.startsWith("uk") || lang.startsWith("ua")) return "ua";
    return "en";
}

function _renderBrainVoiceBubble(ev) {
    const severity = ev.severity || "notable";
    const kind = ev.kind || "unknown";
    const text = (ev.text || "").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const eid = ev.event_id || "";
    return `
        <div class="chat-msg brain-voice bv-${severity} bv-kind-${kind}" data-event-id="${eid}">
            <div class="bv-avatar">🧠</div>
            <div class="bv-body">
                <div class="bv-text">${text}</div>
                <div class="bv-actions">
                    <button class="bv-ack" data-event-id="${eid}" aria-label="acknowledge">✕</button>
                </div>
            </div>
        </div>`;
}

async function pollBrainVoice() {
    try {
        const locale = _brainVoiceLocale();
        const res = await fetch(`/api/chat/brain-voice?locale=${encodeURIComponent(locale)}&limit=10`);
        if (!res.ok) return;
        const data = await res.json();
        const events = Array.isArray(data.events) ? data.events : [];
        if (!events.length) return;

        const container = document.getElementById("chat-messages");
        if (!container) return;

        // Render oldest first so chronological order reads naturally
        events.reverse();
        for (const ev of events) {
            if (!ev.event_id || _BRAIN_VOICE_SHOWN.has(ev.event_id)) continue;
            _BRAIN_VOICE_SHOWN.add(ev.event_id);
            container.insertAdjacentHTML("beforeend", _renderBrainVoiceBubble(ev));
        }

        // Attach one-shot ack handlers
        container.querySelectorAll(".bv-ack:not([data-bound])").forEach((btn) => {
            btn.setAttribute("data-bound", "1");
            btn.addEventListener("click", async () => {
                const id = btn.getAttribute("data-event-id");
                if (!id) return;
                const bubble = btn.closest(".brain-voice");
                if (bubble) bubble.remove();
                try {
                    await fetch(`/api/chat/brain-voice/${encodeURIComponent(id)}/ack`, { method: "POST" });
                } catch (_) { /* best-effort */ }
            });
        });

        container.scrollTop = container.scrollHeight;
    } catch (e) {
        console.debug("brain-voice poll skipped:", e.message);
    }
}

function startBrainVoicePolling() {
    if (_brainVoicePollTimer) return;
    // Initial kick shortly after load, then every 60s
    setTimeout(() => { pollBrainVoice(); }, 2500);
    _brainVoicePollTimer = setInterval(pollBrainVoice, 60_000);
}

if (typeof window !== "undefined") {
    window.addEventListener("DOMContentLoaded", startBrainVoicePolling);
    window.addEventListener("focus", () => { pollBrainVoice(); });
}

function renderFactualitySummary(factuality) {
    if (!factuality) return "";
    const supported = factuality.supported_claims_total ?? 0;
    const unsupported = factuality.unsupported_claims_total ?? 0;
    const claims = Array.isArray(factuality.claims) ? factuality.claims : [];
    const evidenceIds = Array.isArray(factuality.evidence_record_ids) ? factuality.evidence_record_ids : [];

    const summaryBits = [
        `${supported} supported`,
        `${unsupported} unsupported`,
    ];
    if (factuality.unverified_current_claims) {
        summaryBits.push(`${factuality.unverified_current_claims} need verification`);
    }

    const claimLines = claims.slice(0, 4).map((claim) => {
        const label = claim.supported ? "supported" : "unsupported";
        const evidence = Array.isArray(claim.supporting_record_ids) && claim.supporting_record_ids.length
            ? ` [${claim.supporting_record_ids.join(", ")}]`
            : "";
        return `<li><span class="factuality-claim-${label}">${label}</span>: ${escapeHtml(claim.text)}${escapeHtml(evidence)}</li>`;
    }).join("");

    const evidenceLine = evidenceIds.length
        ? (
            `<div class="factuality-evidence">` +
            `<div>Evidence: ${escapeHtml(evidenceIds.slice(0, 5).join(", "))}</div>` +
            `<div class="factuality-actions">` +
            evidenceIds.slice(0, 5).map((id) =>
                `<button type="button" class="btn-small factuality-feedback-btn" data-record-id="${escapeHtml(id)}" data-useful="false">Mark wrong: ${escapeHtml(id)}</button>`
            ).join("") +
            `</div>` +
            `</div>`
        )
        : "";

    return (
        `<details class="factuality-summary">` +
        `<summary>Evidence check: ${escapeHtml(summaryBits.join(" • "))}</summary>` +
        `${evidenceLine}` +
        (claimLines ? `<ul>${claimLines}</ul>` : "") +
        `</details>`
    );
}

async function handleFactualityFeedback(button) {
    const recordId = button?.dataset?.recordId;
    if (!recordId) return;
    const useful = button.dataset.useful === "true";
    const defaultReason = useful
        ? "Operator confirmed this evidence record helped."
        : "Operator marked this evidence record as misleading from the factuality panel.";
    const reason = window.prompt("Optional feedback reason:", defaultReason);
    if (reason === null) return;

    const original = button.textContent;
    button.disabled = true;
    button.textContent = useful ? "Confirming..." : "Marking...";
    try {
        const result = await window.apiClient.submitRecordFeedback(recordId, useful, reason.trim());
        button.textContent = useful
            ? `Confirmed (${result.net_score ?? 0})`
            : `Marked wrong (${result.net_score ?? 0})`;
        button.classList.add("is-complete");
    } catch (err) {
        console.error("Record feedback failed:", err);
        button.disabled = false;
        button.textContent = original;
    }
}

const CONTEXT_REDUCER_RUNS_KEY = "remy.contextReducerRuns.v1";
let contextReducerRunsCache = null;
let contextReducerRunsSource = "local";

function getContextReducerRuns() {
    try {
        const parsed = JSON.parse(localStorage.getItem(CONTEXT_REDUCER_RUNS_KEY) || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch (_err) {
        return [];
    }
}

function saveContextReducerRuns(runs) {
    localStorage.setItem(CONTEXT_REDUCER_RUNS_KEY, JSON.stringify(runs.slice(0, 100)));
}

function summarizeContextReducerRuns(runs) {
    if (!runs.length) {
        return { count: 0, avgSaved: 0, avgTokenRatio: 0, avgLatencySaved: 0, wins: 0, llmCallsSaved: 0, memoryOnlyHits: 0, applyRuns: 0, compareRuns: 0, answerMemoryWrites: 0, routerDecisions: {}, costSavedUsd: 0, minRatio: 0, maxRatio: 0 };
    }
    let minRatio = Infinity;
    let maxRatio = 0;
    const totals = runs.reduce((acc, run) => {
        const delta = run.delta || {};
        const saved = Number(delta.prompt_tokens_saved_estimate || 0);
        const tokenRatio = Number(delta.prompt_token_reduction_ratio || 0);
        const latencySaved = Number(delta.latency_saved_seconds || 0);
        acc.saved += saved;
        acc.tokenRatio += tokenRatio;
        acc.latencySaved += latencySaved;
        acc.costSavedUsd += Number(delta.cost_saved_usd || 0);
        if (tokenRatio > 0) {
            minRatio = Math.min(minRatio, tokenRatio);
            maxRatio = Math.max(maxRatio, tokenRatio);
        }
        const claims = run.claims || {};
        acc.wins += saved > 0 ? 1 : 0;
        acc.llmCallsSaved += Number(delta.llm_calls_saved || (claims.memory_only_cache_hit ? 1 : 0));
        acc.memoryOnlyHits += claims.memory_only_cache_hit ? 1 : 0;
        acc.applyRuns += claims.apply_context_reducer ? 1 : 0;
        acc.compareRuns += claims.measured_ab_comparison ? 1 : 0;
        acc.answerMemoryWrites += run.blocks?.memory_writer?.answer_memory_written ? 1 : 0;
        const decision = run.blocks?.model_router?.decision || "unknown";
        acc.routerDecisions[decision] = (acc.routerDecisions[decision] || 0) + 1;
        return acc;
    }, { saved: 0, tokenRatio: 0, latencySaved: 0, wins: 0, llmCallsSaved: 0, memoryOnlyHits: 0, applyRuns: 0, compareRuns: 0, answerMemoryWrites: 0, routerDecisions: {}, costSavedUsd: 0 });
    return {
        count: runs.length,
        avgSaved: Math.round(totals.saved / runs.length),
        avgTokenRatio: (totals.tokenRatio / runs.length).toFixed(2),
        avgLatencySaved: (totals.latencySaved / runs.length).toFixed(2),
        wins: totals.wins,
        llmCallsSaved: totals.llmCallsSaved,
        memoryOnlyHits: totals.memoryOnlyHits,
        applyRuns: totals.applyRuns,
        compareRuns: totals.compareRuns,
        answerMemoryWrites: totals.answerMemoryWrites,
        routerDecisions: totals.routerDecisions,
        costSavedUsd: totals.costSavedUsd,
        minRatio: Number.isFinite(minRatio) ? minRatio : 0,
        maxRatio: maxRatio,
    };
}

function formatUsd(value) {
    const n = Number(value || 0);
    if (n === 0) return "$0";
    if (n < 0.01) return "$" + n.toFixed(6);
    return "$" + n.toFixed(4);
}

function formatContextReducerRunTime(value) {
    if (!value) return "-";
    const numeric = Number(value);
    const date = Number.isFinite(numeric)
        ? new Date(numeric < 1000000000000 ? numeric * 1000 : numeric)
        : new Date(value);
    return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString();
}

function renderContextReducerLab(runsOverride = null) {
    if (!contextReducerSummaryEl || !contextReducerRunsEl) return;
    const runs = runsOverride || contextReducerRunsCache || getContextReducerRuns();
    const summary = summarizeContextReducerRuns(runs);
    const fiveBlockRuns = runs.filter((run) => run.claims?.five_block_pipeline_report).length;
    const wrongAvoided = runs.reduce((acc, run) => acc + Number(run.delta?.wrong_answers_avoided_estimate || 0), 0);
    const routerSummary = Object.entries(summary.routerDecisions || {})
        .map(([key, value]) => `${escapeHtml(key)}:${value}`)
        .join(" / ") || "-";
    const avgCostSaved = summary.count ? summary.costSavedUsd / summary.count : 0;
    const projected1k = avgCostSaved * 1000;
    const ratioRange = summary.minRatio && summary.maxRatio
        ? `${Number(summary.minRatio).toFixed(1)}x - ${Number(summary.maxRatio).toFixed(1)}x`
        : "-";
    contextReducerSummaryEl.innerHTML = `
        <span class="crl-cost"><strong>${formatUsd(summary.costSavedUsd)}</strong> total cost saved</span>
        <span class="crl-cost"><strong>${formatUsd(avgCostSaved)}</strong> avg / call</span>
        <span class="crl-cost"><strong>${formatUsd(projected1k)}</strong> projected / 1k calls</span>
        <span><strong>${summary.count}</strong> runs (${escapeHtml(contextReducerRunsSource)})</span>
        <span><strong>${summary.avgTokenRatio}x</strong> avg token ratio (${ratioRange})</span>
        <span><strong>${fiveBlockRuns}</strong> five-block reports</span>
        <span><strong>${summary.wins}</strong> token-saving wins</span>
        <span><strong>${summary.llmCallsSaved}</strong> LLM calls saved</span>
        <span><strong>${summary.memoryOnlyHits}</strong> memory-only hits</span>
        <span><strong>${summary.answerMemoryWrites}</strong> answer-memory writes</span>
        <span><strong>${summary.applyRuns}</strong> apply runs / <strong>${summary.compareRuns}</strong> A/B</span>
        <span><strong>${routerSummary}</strong> router decisions</span>
        <span><strong>${summary.avgSaved}</strong> avg tokens saved</span>
        <span><strong>${summary.avgLatencySaved}s</strong> avg latency saved</span>
        <span><strong>${wrongAvoided}</strong> wrong answers avoided</span>
    `;
    contextReducerRunsEl.innerHTML = runs.length
        ? runs.map((run) => {
            const raw = run.raw || {};
            const reduced = run.reduced || {};
            const delta = run.delta || {};
            const created = formatContextReducerRunTime(run.created_at);
            const model = delta.model || run.blocks?.cost_latency_tracker?.model || "-";
            const price = delta.input_price_per_1m_usd;
            const costSaved = delta.cost_saved_usd;
            const tokSource = delta.billable_tokens_source || "estimate";
            const billableRatio = delta.prompt_token_reduction_ratio_billable;
            const costCell = (costSaved !== undefined && costSaved !== null)
                ? `<strong class="crl-cost">${formatUsd(costSaved)}</strong> saved<br><span class="crl-subcell">${escapeHtml(String(model))} @ $${price ?? "?"}/1M<br>${escapeHtml(tokSource)} tokens${billableRatio ? " · " + billableRatio + "x" : ""}</span>`
                : `<span class="crl-subcell">no price set</span>`;
            return `
                <tr>
                    <td>${escapeHtml(created)}</td>
                    <td title="${escapeHtml(run.user_text_preview || "")}">${escapeHtml(run.user_text_preview || "-")}</td>
                    <td>${raw.prompt_tokens_estimate ?? "-"} tok / ${raw.elapsed_seconds ?? "-"}s</td>
                    <td>${reduced.prompt_tokens_estimate ?? "-"} tok / ${reduced.elapsed_seconds ?? "-"}s</td>
                    <td>${delta.prompt_tokens_saved_estimate ?? "-"} tok (${delta.prompt_token_reduction_ratio ?? "-"}x)</td>
                    <td>${costCell}</td>
                    <td>${delta.latency_saved_seconds ?? "-"}s (${delta.latency_reduction_ratio ?? "-"}x)<br><span class="crl-subcell">${escapeHtml(run.blocks?.model_router?.decision || "router n/a")}${run.claims?.memory_only_cache_hit ? " · LLM saved" : ""}</span></td>
                </tr>`;
        }).join("")
        : `<tr><td colspan="7" class="crl-empty">No optimization runs yet.</td></tr>`;
}

async function loadContextReducerMeasurements() {
    try {
        const payload = await window.apiClient.getLlmOptimizationMeasurements(100);
        const runs = Array.isArray(payload.items) ? payload.items : [];
        contextReducerRunsCache = runs;
        contextReducerRunsSource = "backend";
        saveContextReducerRuns(runs);
        renderContextReducerLab(runs);
    } catch (err) {
        console.warn("Failed to load backend optimization report, using local mirror", err);
        contextReducerRunsCache = getContextReducerRuns();
        contextReducerRunsSource = "local fallback";
        renderContextReducerLab(contextReducerRunsCache);
    }
}

function rememberContextReducerRun(report) {
    if (!report) return;
    const runs = contextReducerRunsCache || getContextReducerRuns();
    runs.unshift({ ...report, created_at: new Date().toISOString() });
    contextReducerRunsCache = runs.slice(0, 100);
    contextReducerRunsSource = report.blocks?.memory_writer?.persistent_measurement_written ? "backend" : "local fallback";
    saveContextReducerRuns(contextReducerRunsCache);
    renderContextReducerLab(contextReducerRunsCache);
}

function exportContextReducerRuns() {
    const runs = contextReducerRunsCache || getContextReducerRuns();
    const blob = new Blob([JSON.stringify(runs, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "optimization-report.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function renderOptimizationBlocks(report) {
    const blocks = report?.blocks || {};
    const verifier = blocks.verifier || {};
    const router = blocks.model_router || {};
    const writer = blocks.memory_writer || {};
    const tracker = blocks.cost_latency_tracker || {};
    const routedModel = router.actual_model || router.preferred_model || "";
    return `
        <details class="crc-advanced">
            <summary>Technical details</summary>
            <div class="crc-blocks">
                <span class="crc-block is-on">1 Short Context</span>
                <span class="crc-block ${writer.event_written ? "is-on" : "is-muted"}">2 Answer Memory</span>
                <span class="crc-block ${verifier.enabled ? "is-on" : "is-muted"}">3 Safety Check${verifier.blocked ? " blocked" : ""}</span>
                <span class="crc-block ${router.enabled ? "is-on" : "is-muted"}">4 Model Choice: ${escapeHtml(router.decision || "default")}${routedModel ? " - " + escapeHtml(routedModel) : ""}</span>
                <span class="crc-block ${tracker.enabled ? "is-on" : "is-muted"}">5 Tracking</span>
            </div>
        </details>`;
}

function renderSavingsSummary(report, mode) {
    const delta = report?.delta || {};
    const saved = delta.prompt_tokens_saved_estimate ?? "-";
    const tokenRatio = delta.prompt_token_reduction_ratio ?? "-";
    const latencyRatio = delta.latency_reduction_ratio ?? "-";
    const memoryOnly = Boolean(report?.claims?.memory_only_cache_hit);
    if (memoryOnly) {
        return "Answered from a saved safe answer. No model call was needed.";
    }
    if (mode === "compare") {
        return `Test result: optimized mode used ${saved} fewer estimated prompt tokens (${tokenRatio}x smaller prompt) and ${latencyRatio}x latency in this run.`;
    }
    return `Optimized mode was used for this answer and reduced the prompt by ${saved} estimated tokens.`;
}

function renderContextReducerCompare(report) {
    if (!report) return "";
    const raw = report.raw || {};
    const reduced = report.reduced || {};
    const delta = report.delta || {};
    const ratio = delta.prompt_token_reduction_ratio ?? "-";
    const latencyRatio = delta.latency_reduction_ratio ?? "-";
    const wrongAvoided = delta.wrong_answers_avoided_estimate ?? 0;
    return `
        <div class="context-reducer-card">
            <div class="crc-title">Optimization Test</div>
            <div class="crc-user-summary">${escapeHtml(renderSavingsSummary(report, "compare"))}</div>
            <div class="crc-metrics">
                <span>Normal: ${raw.prompt_tokens_estimate ?? "-"} tok / ${raw.elapsed_seconds ?? "-"}s</span>
                <span>Optimized: ${reduced.prompt_tokens_estimate ?? "-"} tok / ${reduced.elapsed_seconds ?? "-"}s</span>
                <span>Tokens reduced: ${delta.prompt_tokens_saved_estimate ?? "-"} tok</span>
                <span>Prompt ratio: ${ratio}x tokens / ${latencyRatio}x latency</span>
                <span>Safety issues avoided: ${wrongAvoided}</span>
            </div>
            ${renderOptimizationBlocks(report)}
            <details>
                <summary>Normal answer</summary>
                <div class="crc-answer">${formatMarkdown(raw.answer || raw.answer_preview || "")}</div>
            </details>
            <details open>
                <summary>Optimized answer</summary>
                <div class="crc-answer">${formatMarkdown(reduced.answer || reduced.answer_preview || "")}</div>
            </details>
        </div>`;
}

function addContextReducerCompare(report) {
    removeTyping();
    rememberContextReducerRun(report);
    const div = document.createElement("div");
    div.className = "chat-msg assistant context-reducer-result";
    div.innerHTML = renderContextReducerCompare(report);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderContextReducerApply(report) {
    if (!report) return "";
    const raw = report.raw || {};
    const reduced = report.reduced || {};
    const delta = report.delta || {};
    const memoryOnly = Boolean(report.claims?.memory_only_cache_hit);
    return `
        <div class="context-reducer-card context-reducer-card--apply">
            <div class="crc-title">${memoryOnly ? "Saved Answer Used" : "Optimization Used"}</div>
            <div class="crc-user-summary">${escapeHtml(renderSavingsSummary(report, "apply"))}</div>
            <div class="crc-metrics">
                <span>Normal prompt avoided: ${raw.prompt_tokens_estimate ?? "-"} tok estimated</span>
                <span>${memoryOnly ? "Saved answer" : "Optimized prompt"}: ${reduced.prompt_tokens_estimate ?? "-"} tok / ${reduced.elapsed_seconds ?? "-"}s${memoryOnly && reduced.memory_source_store ? " - " + escapeHtml(reduced.memory_source_store) : ""}</span>
                <span>Tokens reduced: ${delta.prompt_tokens_saved_estimate ?? "-"} tok</span>
                <span>Ratio: ${delta.prompt_token_reduction_ratio ?? "-"}x tokens</span>
                <span>Model call: ${memoryOnly ? "avoided" : (report.claims?.raw_llm_call_skipped ? "optimized only" : "used")}</span>
                <span>Safety check: ${report.claims?.verifier_modified_answer ? "rewritten" : "no rewrite"}</span>
            </div>
            ${renderOptimizationBlocks(report)}
        </div>`;
}

function addLlmOptimizationApply(report) {
    rememberContextReducerRun(report);
    const div = document.createElement("div");
    div.className = "chat-msg assistant context-reducer-result";
    div.innerHTML = renderContextReducerApply(report);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function attachFactualitySummary(factuality) {
    if (!factuality) return;
    const assistantMsg = currentAssistantMsg || getLastAssistantMessage();
    if (!assistantMsg) return;

    const existing = assistantMsg.querySelector(".factuality-summary");
    if (existing) existing.remove();

    assistantMsg.insertAdjacentHTML("beforeend", renderFactualitySummary(factuality));
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping() {
    if (typingEl) return;
    typingEl = document.createElement("div");
    typingEl.className = "chat-msg typing";
    typingEl.innerHTML = '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>';
    messagesEl.appendChild(typingEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTyping() {
    if (typingEl) {
        typingEl.remove();
        typingEl = null;
    }
}

contextReducerCompareCheckbox?.addEventListener("change", () => {
    if (contextReducerCompareCheckbox.checked && contextReducerApplyCheckbox) {
        contextReducerApplyCheckbox.checked = false;
    }
});

contextReducerApplyCheckbox?.addEventListener("change", () => {
    if (contextReducerApplyCheckbox.checked && contextReducerCompareCheckbox) {
        contextReducerCompareCheckbox.checked = false;
    }
});

contextReducerLabBtn?.addEventListener("click", () => {
    contextReducerLab?.classList.toggle("hidden");
    if (!contextReducerLab?.classList.contains("hidden")) {
        populateLabModels();
        loadContextReducerMeasurements();
    }
});

contextReducerCloseBtn?.addEventListener("click", () => {
    contextReducerLab?.classList.add("hidden");
});

contextReducerClearBtn?.addEventListener("click", async () => {
    if (!window.confirm("Clear saved ContextReducer measurements?")) return;
    try {
        await window.apiClient.clearLlmOptimizationMeasurements();
    } catch (err) {
        console.warn("Backend measurement clear failed", err);
    }
    contextReducerRunsCache = [];
    contextReducerRunsSource = "backend";
    saveContextReducerRuns([]);
    renderContextReducerLab([]);
});

contextReducerExportBtn?.addEventListener("click", exportContextReducerRuns);

renderContextReducerLab();

// ============== TTS ==============

let availableVoices = [];
let preferredVoice = null;

function loadVoices() {
    if (!window.speechSynthesis) return;
    
    availableVoices = window.speechSynthesis.getVoices();
    
    // Voices might not be loaded yet (especially Firefox/Chrome async)
    if (availableVoices.length === 0) {
        window.speechSynthesis.onvoiceschanged = () => {
            availableVoices = window.speechSynthesis.getVoices();
            selectBestVoice();
        };
    } else {
        selectBestVoice();
    }
}

function selectBestVoice() {
    // Prioritize high-quality English voices
    const priority = [
        "Google US English",
        "Microsoft Zira", 
        "Microsoft David",
        "Samantha", 
    ];

    // Python code uses 'en-US' or similar logic
    // Try to find unmatched English voice if specific ones fail
    
    for (const name of priority) {
        const found = availableVoices.find(v => v.name.includes(name));
        if (found) {
            preferredVoice = found;
            console.log("TTS Voice selected:", found.name);
            return;
        }
    }
    
    // Fallback to any English voice
    preferredVoice = availableVoices.find(v => v.lang.startsWith("en"));
    
    if (!preferredVoice && availableVoices.length > 0) {
        preferredVoice = availableVoices[0];
    }
    
    if (preferredVoice) {
        console.log("TTS Voice fallback:", preferredVoice.name);
    }
}

// Initial load
if (window.speechSynthesis) {
    loadVoices();
}

function speakText(text) {
    if (!ttsCheckbox || !ttsCheckbox.checked) return;
    if (!window.speechSynthesis) return;

    // Reload voices if empty (Firefox quirk)
    if (!preferredVoice && availableVoices.length === 0) {
        loadVoices();
    }

    try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        utterance.onerror = (e) => {
            console.error("TTS Error:", e);
        };

        window.speechSynthesis.speak(utterance);
    } catch (e) {
        console.error("TTS Exception:", e);
    }
}

// ============== FILE HANDLING ==============

let stagedFile = null; // { name, mime_type, base64 }
let activeCodeWorkspace = null;

function setActiveCodeWorkspace(workspace) {
    activeCodeWorkspace = workspace || null;
    if (!activeCodeWorkspace) {
        localStorage.removeItem("remy-active-code-workspace");
        codeWorkspaceBar?.classList.remove("active");
        codeWorkspaceBtn?.setAttribute("title", "Choose a code project");
        if (codeWorkspaceName) codeWorkspaceName.textContent = "Choose project";
        if (codeWorkspaceMeta) codeWorkspaceMeta.textContent = "";
        closeWorkspaceBtn?.classList.add("hidden");
        if (inputEl) inputEl.placeholder = "Type a message...";
        return;
    }
    localStorage.setItem("remy-active-code-workspace", activeCodeWorkspace.workspace_id);
    if (codeWorkspaceName) codeWorkspaceName.textContent = activeCodeWorkspace.name || "Code workspace";
    if (codeWorkspaceMeta) {
        const rootPath = activeCodeWorkspace.root_path || "";
        const fileCount = activeCodeWorkspace.code_profile?.file_count;
        const countLabel = fileCount
            ? `${fileCount}${activeCodeWorkspace.code_profile?.truncated ? "+" : ""} files`
            : "read-only";
        codeWorkspaceMeta.textContent = rootPath ? `${rootPath} · ${countLabel}` : countLabel;
    }
    codeWorkspaceBtn?.setAttribute("title", `Current project: ${activeCodeWorkspace.root_path || activeCodeWorkspace.name}. Click to change.`);
    codeWorkspaceBar?.classList.add("active");
    closeWorkspaceBtn?.classList.remove("hidden");
    if (inputEl) inputEl.placeholder = `Ask about ${activeCodeWorkspace.name || "this codebase"}...`;
}

async function restoreActiveCodeWorkspace() {
    const savedId = localStorage.getItem("remy-active-code-workspace");
    if (!savedId) return;
    try {
        const response = await fetch("/api/workspaces");
        if (!response.ok) return;
        const data = await response.json();
        const workspace = (data.workspaces || []).find((item) => item.workspace_id === savedId);
        if (workspace && workspace.source !== "builtin") setActiveCodeWorkspace(workspace);
        else setActiveCodeWorkspace(null);
    } catch (_) {
        // The status will recover on the next page load; chat remains usable.
    }
}

codeWorkspaceBtn?.addEventListener("click", async () => {
    codeWorkspaceBtn.disabled = true;
    codeWorkspaceBtn.classList.add("workspace-picking");
    try {
        const response = await fetch("/api/workspaces/select-folder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ read: true, write: false, execute: false }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not open the folder picker.");
        if (data.cancelled) return;
        setActiveCodeWorkspace(data.workspace);
    } catch (error) {
        addErrorMessage(error.message || "Could not connect the code folder.");
    } finally {
        codeWorkspaceBtn.disabled = false;
        codeWorkspaceBtn.classList.remove("workspace-picking");
    }
});

closeWorkspaceBtn?.addEventListener("click", () => setActiveCodeWorkspace(null));
restoreActiveCodeWorkspace();

attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) stageFile(file);
    fileInput.value = "";
});

removeFileBtn.addEventListener("click", () => {
    stagedFile = null;
    filePreview.style.display = "none";
});

// Drag and drop
chatInputArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropOverlay.classList.add("visible");
});

chatInputArea.addEventListener("dragleave", () => {
    dropOverlay.classList.remove("visible");
});

chatInputArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropOverlay.classList.remove("visible");
    const file = e.dataTransfer.files[0];
    if (file) stageFile(file);
});

// Paste images from clipboard
inputEl.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith("image/")) {
            e.preventDefault();
            const file = item.getAsFile();
            if (file) stageFile(file);
            return;
        }
    }
});

async function stageFile(file) {
    const maxSize = 20 * 1024 * 1024;
    if (file.size > maxSize) {
        addMessage("assistant", "File too large (max 20 MB).");
        return;
    }

    const base64 = await fileToBase64(file);
    stagedFile = {
        name: file.name,
        mime_type: file.type || "application/octet-stream",
        base64,
    };
    filePreviewName.textContent = `${file.name} (${formatSize(file.size)})`;
    filePreview.style.display = "flex";
}

function fileToBase64(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(",")[1]);
        reader.readAsDataURL(file);
    });
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

// ============== VOICE RECORDING (TOGGLE MODE) ==============

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let activeStream = null;

// Toggle: click to start, click again to stop
voiceBtn.addEventListener("click", toggleRecording);

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    if (isRecording) return;

    try {
        activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Pick a supported MIME type
        const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus"
            : MediaRecorder.isTypeSupported("audio/mp4")
                ? "audio/mp4"
                : "";

        const options = mimeType ? { mimeType } : {};
        mediaRecorder = new MediaRecorder(activeStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            // Stop all tracks to release microphone
            if (activeStream) {
                activeStream.getTracks().forEach((t) => t.stop());
                activeStream = null;
            }
            if (audioChunks.length === 0) return;

            const audioBlob = new Blob(audioChunks, {
                type: mediaRecorder.mimeType || "audio/webm",
            });
            const base64 = await blobToBase64(audioBlob);
            const sendMime = (mediaRecorder.mimeType || "audio/webm").split(";")[0];

            addMessage("user", "[Voice message]", "voice-input");
            lastUserMessage = { type: "voice", audio: base64, mime_type: sendMime };
            showTyping();
            window.apiClient.sendVoice(base64, sendMime);
        };

        mediaRecorder.start(250); // collect data every 250ms for reliability
        isRecording = true;
        voiceBtn.classList.add("recording");
        voiceBtn.title = "Click to stop recording";
    } catch (err) {
        console.error("Microphone access denied:", err);
        addMessage("assistant", "Microphone access was denied. Please allow it in browser settings.");
    }
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    mediaRecorder.stop();
    isRecording = false;
    voiceBtn.classList.remove("recording");
    voiceBtn.title = "Click to speak";
}

function blobToBase64(blob) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(",")[1]);
        reader.readAsDataURL(blob);
    });
}

// ============== SEND MESSAGE ==============

function sendMessage() {
    const text = inputEl.value.trim();
    const emitBrainBurst = () => document.dispatchEvent(new CustomEvent("brain-activation-burst"));

    if (stagedFile) {
        const displayText = text
            ? `[${stagedFile.name}] ${text}`
            : `[${stagedFile.name}]`;

        if (limitedMode) {
            messageQueue.push({
                type: "file", data: stagedFile.base64,
                name: stagedFile.name, mime_type: stagedFile.mime_type,
                text, displayText,
            });
            addQueuedMessage(displayText);
        } else {
            addMessage("user", displayText);
            emitBrainBurst();
            if (stagedFile.mime_type.startsWith("image/")) {
                addImagePreview(stagedFile.base64, stagedFile.mime_type);
            }
            lastUserMessage = { type: "file", data: stagedFile.base64, name: stagedFile.name, mime_type: stagedFile.mime_type, text };
            window.apiClient.sendFile(stagedFile.base64, stagedFile.name, stagedFile.mime_type, text);
        }

        stagedFile = null;
        filePreview.style.display = "none";
        inputEl.value = "";
        inputEl.style.height = "auto";
        return;
    }

    if (!text) return;

    // Compare mode — send to multiple models in parallel
    if (_compareActive && _compareSelectedModels.size >= 2) {
        inputEl.value = "";
        inputEl.style.height = "auto";
        _sendCompareMessage(text);
        return;
    }

    const contextReducerCompare = Boolean(contextReducerCompareCheckbox?.checked);
    const contextReducerApply = !contextReducerCompare && Boolean(contextReducerApplyCheckbox?.checked);
    // When A/B testing, price the run against the model selected in the lab.
    const labModel = contextReducerCompare ? getSelectedLabModel() : "";
    const teamMode = (contextReducerCompare || contextReducerApply) ? "off" : consumeTeamMode();

    if (limitedMode) {
        messageQueue.push({
            type: "message", text, displayText: text, contextReducerCompare, contextReducerApply,
            workspaceId: activeCodeWorkspace?.workspace_id,
            teamMode,
        });
        addQueuedMessage(text);
        inputEl.value = "";
        inputEl.style.height = "auto";
        if (messageQueue.length === 1) {
            showTyping();
            emitBrainBurst();
            window.apiClient.sendMessage(text, {
                contextReducerCompare, contextReducerApply, model: labModel,
                workspaceId: activeCodeWorkspace?.workspace_id,
                teamMode,
            });
        }
        return;
    }

    addMessage("user", text);
    emitBrainBurst();
    lastUserMessage = {
        type: "message", text, contextReducerCompare, contextReducerApply,
        workspaceId: activeCodeWorkspace?.workspace_id,
        teamMode,
    };
    window.apiClient.sendMessage(text, {
        contextReducerCompare, contextReducerApply, model: labModel,
        workspaceId: activeCodeWorkspace?.workspace_id,
        teamMode,
    });
    inputEl.value = "";
    inputEl.style.height = "auto";
}

// Send button
sendBtn.addEventListener("click", sendMessage);

// Enter to send, Shift+Enter for newline
inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
});

// ============== PIPELINE LAUNCHER IN CHAT ==============

(function initChatPipelineLauncher() {
    const triggerBtn  = document.getElementById("btn-pipeline");
    const menu        = document.getElementById("chat-pipeline-menu");
    const listEl      = document.getElementById("cpm-list");
    if (!triggerBtn || !menu || !listEl) return;

    // ── state ──────────────────────────────────────────────────────────────────
    let _pipelines  = [];
    let _menuOpen   = false;
    let _active     = null;   // { id, name } of selected pipeline, or null
    let _runAbort   = null;
    let _conversationId = "";

    document.addEventListener("conversation-changed", (event) => {
        _conversationId = String(event.detail?.conversation?.conversation_id || "");
    });

    // ── helpers ────────────────────────────────────────────────────────────────
    function _setActive(pipeline) {
        _active = pipeline;
        if (pipeline) {
            triggerBtn.classList.add("active");
            triggerBtn.title = `Pipeline: ${pipeline.name} (click to cancel)`;
            inputEl.placeholder = `Query for "${pipeline.name}"…`;
        } else {
            triggerBtn.classList.remove("active");
            triggerBtn.title = "Run pipeline";
            inputEl.placeholder = "Type a message...";
        }
    }

    async function _loadPipelines() {
        try {
            const res = await fetch("/api/pipelines").then(r => r.json());
            _pipelines = res.pipelines || [];
        } catch (_) { _pipelines = []; }
    }

    function _renderList() {
        // First item: always "regular chat" to switch back
        const normalSelected = !_active;
        let html = `
            <div class="cpm-item cpm-item-normal${normalSelected ? " cpm-item-selected" : ""}" data-id="" data-name="">
                <span class="cpm-item-icon">💬</span>
                <div class="cpm-item-info">
                    <div class="cpm-item-name">Regular chat</div>
                    <div class="cpm-item-meta">AI response without a pipeline</div>
                </div>
                ${normalSelected ? '<span class="cpm-item-check">✓</span>' : ""}
            </div>`;

        if (_pipelines.length) {
            html += `<div class="cpm-divider"></div>`;
            html += _pipelines.map(p => `
                <div class="cpm-item${_active?.id === p.id ? " cpm-item-selected" : ""}" data-id="${p.id}" data-name="${escapeHtml(p.name)}">
                    <span class="cpm-item-icon">⚡</span>
                    <div class="cpm-item-info">
                        <div class="cpm-item-name">${escapeHtml(p.name)}</div>
                        <div class="cpm-item-meta">${p.step_count || 0} steps</div>
                    </div>
                    ${_active?.id === p.id ? '<span class="cpm-item-check">✓</span>' : ""}
                </div>`).join("");
        } else {
            html += `<div class="cpm-empty" style="padding:10px 14px">No saved pipelines.<br>Create them in the <b>Pipelines</b> section.</div>`;
        }

        listEl.innerHTML = html;

        listEl.querySelectorAll(".cpm-item").forEach(item => {
            item.addEventListener("click", () => {
                const id   = item.dataset.id;
                const name = item.dataset.name;
                _setActive(id ? { id, name } : null);
                _closeMenu();
                inputEl.focus();
            });
        });
    }

    async function _openMenu() {
        _menuOpen = true;
        menu.classList.remove("hidden");
        listEl.innerHTML = `<div class="cpm-empty">Loading…</div>`;
        await _loadPipelines();
        _renderList();
    }

    function _closeMenu() {
        _menuOpen = false;
        menu.classList.add("hidden");
    }

    // ── trigger button ─────────────────────────────────────────────────────────
    triggerBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        _menuOpen ? _closeMenu() : _openMenu();
    });

    document.addEventListener("click", (e) => {
        if (_menuOpen && !menu.contains(e.target) && e.target !== triggerBtn) _closeMenu();
    });

    // ── intercept Send when pipeline is active ─────────────────────────────────
    // We hook into the existing sendMessage by wrapping the send button and Enter key.
    // When _active is set, we run the pipeline instead of the normal WS send.
    const originalSend = window._pipelineSendOverride = async function pipelineSend() {
        if (!_active) return false;   // not our turn — let normal send proceed

        const inputText = inputEl.value.trim();
        inputEl.value = "";
        inputEl.style.height = "auto";

        const { id: pipelineId, name: pipelineName } = _active;

        // Show user bubble
        addMessage("user", inputText || `[${pipelineName}]`);

        // Show a "running" assistant bubble
        const runBubble = document.createElement("div");
        runBubble.className = "message assistant";
        runBubble.innerHTML = `<div class="cpm-running-inline">
            <div class="cpm-rl-title">⚡ ${escapeHtml(pipelineName)}</div>
            <div class="cpm-rl-steps" id="cpm-rl-steps-${pipelineId}"></div>
        </div>`;
        document.getElementById("chat-messages").appendChild(runBubble);
        runBubble.scrollIntoView({ behavior: "smooth", block: "end" });

        const stepsEl = runBubble.querySelector(`#cpm-rl-steps-${pipelineId}`);

        _runAbort = new AbortController();
        let lastOutput = "";
        let trajectoryConversationId = _conversationId;
        let trajectoryEventId = "";

        try {
            const resp = await fetch("/api/pipelines/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    pipeline_id: pipelineId,
                    input_text: inputText,
                    conversation_id: _conversationId,
                }),
                signal: _runAbort.signal,
            });
            if (!resp.ok) {
                const failure = await resp.json().catch(() => ({}));
                throw new Error(failure.detail || `Pipeline request failed (${resp.status})`);
            }

            const reader  = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split("\n");
                buf = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith("data:")) continue;
                    try {
                        const evt = JSON.parse(line.slice(5).trim());
                        if (evt.type === "run_started") {
                            trajectoryConversationId = evt.conversation_id || trajectoryConversationId;
                            trajectoryEventId = evt.trajectory_event_id || trajectoryEventId;
                        }
                        if (evt.type === "step_start") {
                            const d = document.createElement("div");
                            d.className = "cpm-step"; d.id = `cpmi-${evt.index}`;
                            d.innerHTML = `<span>⏳</span> ${escapeHtml(evt.label)}`;
                            stepsEl.appendChild(d);
                            runBubble.scrollIntoView({ behavior: "smooth", block: "end" });
                        }
                        if (evt.type === "step_done") {
                            const d = document.getElementById(`cpmi-${evt.index}`);
                            if (d) d.innerHTML = `<span>✅</span> ${escapeHtml(evt.label)}`;
                            if (d) d.className = "cpm-step done";
                        }
                        if (evt.type === "step_error") {
                            const d = document.getElementById(`cpmi-${evt.index}`);
                            if (d) d.innerHTML = `<span>❌</span> ${escapeHtml(evt.label)}: ${escapeHtml(evt.error||"")}`;
                            if (d) d.className = "cpm-step error";
                        }
                        if (evt.type === "done") lastOutput = evt.output || "";
                        if (evt.type === "trajectory_link") {
                            trajectoryConversationId = evt.conversation_id || trajectoryConversationId;
                            trajectoryEventId = evt.event_id || evt.run_event_id || trajectoryEventId;
                        }
                        if (evt.type === "error") {
                            lastOutput = `[Pipeline error: ${evt.error || "Unknown error"}]`;
                        }
                    } catch (_) {}
                }
            }
        } catch (err) {
            if (err.name === "AbortError") return true;
            lastOutput = `[Error: ${err.message}]`;
        }

        // Replace running bubble with final answer
        runBubble.innerHTML = "";
        if (lastOutput) {
            // Re-use addMessage formatting — remove temp bubble, add proper one
            runBubble.remove();
            const resultBubble = addMessage("assistant", lastOutput, "pipeline-result");
            if (trajectoryConversationId && trajectoryEventId) {
                const controls = resultBubble.querySelector(".chat-response-controls") || resultBubble;
                const trajectoryLink = document.createElement("button");
                trajectoryLink.type = "button";
                trajectoryLink.className = "pipeline-trajectory-link";
                trajectoryLink.textContent = "Trajectory ↗";
                trajectoryLink.title = "Inspect this pipeline run in Trajectory";
                trajectoryLink.addEventListener("click", () => {
                    document.dispatchEvent(new CustomEvent("trajectory-open-alert", {
                        detail: {
                            conversationId: trajectoryConversationId,
                            eventId: trajectoryEventId,
                        },
                    }));
                });
                controls.appendChild(trajectoryLink);
            }
        } else {
            runBubble.remove();
        }

        return true; // handled
    };

    // Patch sendMessage to check pipeline first
    const _origSendBtn = sendBtn.onclick;
    sendBtn.addEventListener("click", async (e) => {
        if (_active) { e.stopImmediatePropagation(); await originalSend(); }
    }, true);  // capture phase — runs before existing listener

    inputEl.addEventListener("keydown", async (e) => {
        if (e.key === "Enter" && !e.shiftKey && _active) {
            e.stopImmediatePropagation();
            e.preventDefault();
            await originalSend();
        }
    }, true);
})();

// ============== WEBSOCKET MESSAGES ==============

// Friendly tool name labels for progress indicator
const TOOL_LABELS = {
    recall:              "Searching memory",
    store:               "Storing memory",
    search:              "Searching records",
    store_person:        "Saving person",
    store_story:         "Saving story",
    family_tree:         "Loading family tree",
    insights:            "Analyzing memory health",
    connect_records:     "Connecting records",
    get_connections:     "Loading connections",
    store_user_profile:  "Updating profile",
    web_search:          "Searching the web",
    get_current_datetime:"Checking the time",
    schedule_task:       "Scheduling task",
    update_record:       "Updating record",
    delete_record:       "Deleting record",
    sandbox_create_tool: "Creating tool",
    sandbox_test_tool:   "Testing tool",
    sandbox_list_tools:  "Listing sandbox tools",
    store_research:      "Saving research report",
    create_subgoal:      "Creating sub-goal",
    complete_goal:       "Completing goal",
    read_file:           "Reading file",
    write_file:          "Writing file",
    list_directory:      "Listing directory",
    http_get:            "Fetching data",
    consolidate:         "Consolidating memory",
    start_research:      "Starting research",
    add_research_finding:"Recording finding",
    complete_research:   "Synthesizing research",
    track_metric:        "Logging metric",
    metric_summary:      "Summarizing metrics",
    event_correlate:     "Analyzing event correlations",
    extract_facts:       "Extracting facts",
    generate_image:      "Generating image",
    generate_report:     "Generating report",
    add_todo:            "Adding task",
    list_todos:          "Loading tasks",
    update_todo:         "Updating task",
    delete_todo:         "Removing task",
    browse_page:         "Browsing page",
    browser_act:         "Interacting with page",
    browser_close:       "Closing browser",
    list_local_workspaces:"Checking code workspaces",
    list_directory:      "Mapping codebase",
    fs_read:             "Reading code",
    fs_write:            "Writing file",
    fs_search:           "Searching code",
    shell_exec:          "Running command",
    delegate_task:       "Delegating to worker",
    enable_tools:        "Enabling tools",
    scratchpad:          "Using scratchpad",
    store_knowledge:     "Storing knowledge",
    recall_knowledge:    "Recalling knowledge",
    knowledge_stats:     "Knowledge stats",
    read_persona:        "Reading persona",
    update_persona:      "Updating persona",
    verify_record:       "Verifying record",
    aura_cognitive_ops:  "Cognitive operation",
    generate_presentation:"Generating presentation",
    // V11
    list_loaded_bases:   "Checking knowledge bases",
    check_base_version:  "Checking base version",
    list_cognitive_snapshots:"Listing brain snapshots",
    list_org_records:    "Listing org records",
    // V12
    introspect_drives:   "Checking active drives",
    introspect_goals:    "Reviewing goals",
    introspect_tensions: "Sensing tensions",
    claim_drive:         "Claiming drive",
    resolve_drive:       "Resolving drive",
    create_goal:         "Creating goal",
    revise_goal:         "Revising goal priority",
    // V13
    introspect_predictions:"Checking predictions",
    introspect_surprises:"Reviewing surprises",
    prediction_report:   "Prediction report",
    // V14
    introspect_curiosity:"Sensing knowledge gaps",
    curiosity_report:    "Curiosity report",
    // V15
    introspect_mood:     "Checking mood state",
    mood_history:        "Reviewing mood history",
    mood_modulation:     "Checking mood effects",
    // V17
    incubation_report:       "Checking incubation status",
    introspect_hypotheses:   "Reviewing hypotheses",
    review_hypothesis:       "Reviewing a hypothesis",
    set_incubation_enabled:      "Configuring incubation",
    clear_expired_hypotheses:    "Cleaning expired hypotheses",
    // Thermal
    get_thermal_map:             "Reading cognitive heat map",
    get_plasticity_audit:        "Auditing synaptic plasticity",
};

function friendlyToolName(rawName) {
    return TOOL_LABELS[rawName] || `Using ${rawName}`;
}

let currentAssistantMsg = null;
let currentToolIndicator = null;
let streamBuffer = "";
let lastUserMessage = null;
let currentAssistantProvisional = false;
let providerStatusEl = null;
let providerStatusStartedAt = null;
let providerStatusTimer = null;
let currentRun = null;
let trajectoryTokenNotified = false;

function notifyTrajectoryChange(phase) {
    document.dispatchEvent(new CustomEvent("trajectory-refresh", {
        detail: { phase: phase || "update" },
    }));
}

// Progress tracking state
let toolStepCount = 0;
let operationStartTime = null;
let elapsedTimerInterval = null;

// ============== LLM LIMITED MODE ==============

const LLM_ERROR_THRESHOLD = 2;
let consecutiveLlmErrors = 0;
let limitedMode = false;
let messageQueue = [];
let lastErrorClass = null;

const RECOVERY_ESTIMATES = {
    rate_limit: 60,
    server_error: 120,
    timeout: 15,
    network: 30,
    auth: 0,
    unknown: 60,
};

function enterLimitedMode(errorClass) {
    lastErrorClass = errorClass || "unknown";
    if (limitedMode) return;
    limitedMode = true;
    const estimateSeconds = RECOVERY_ESTIMATES[lastErrorClass] || 60;
    document.dispatchEvent(new CustomEvent("llm-status-change", {
        detail: { available: false, errorClass: lastErrorClass, estimateSeconds },
    }));
    inputEl.placeholder = "LLM unavailable \u2014 messages will be queued...";
    inputEl.classList.add("llm-unavailable");
    sendBtn.title = "Queue message";
    sendBtn.setAttribute("aria-label", "Queue message");
    sendBtn.classList.add("btn-queue");
}

function exitLimitedMode() {
    if (!limitedMode) return;
    limitedMode = false;
    consecutiveLlmErrors = 0;
    lastErrorClass = null;
    document.dispatchEvent(new CustomEvent("llm-status-change", { detail: { available: true } }));
    inputEl.placeholder = "Describe what you want Remy to do…";
    inputEl.classList.remove("llm-unavailable");
    sendBtn.title = "Send message";
    sendBtn.setAttribute("aria-label", "Send message");
    sendBtn.classList.remove("btn-queue");
    drainQueue();
}

function drainQueue() {
    if (messageQueue.length === 0) return;
    document.querySelectorAll(".chat-msg.queued").forEach((el) => el.remove());
    const queue = [...messageQueue];
    messageQueue = [];
    for (const msg of queue) {
        addMessage("user", msg.displayText || msg.text || "[Queued message]");
        showTyping();
        if (msg.type === "message") {
            window.apiClient.sendMessage(msg.text, {
                contextReducerCompare: Boolean(msg.contextReducerCompare),
                contextReducerApply: Boolean(msg.contextReducerApply),
                workspaceId: msg.workspaceId || activeCodeWorkspace?.workspace_id,
                teamMode: msg.teamMode || "off",
            });
        } else if (msg.type === "voice") {
            window.apiClient.sendVoice(msg.audio, msg.mime_type);
        } else if (msg.type === "file") {
            window.apiClient.sendFile(msg.data, msg.name, msg.mime_type, msg.text);
        }
    }
}

function addQueuedMessage(text) {
    const div = document.createElement("div");
    div.className = "chat-msg user queued";
    div.textContent = text;
    const badge = document.createElement("span");
    badge.className = "queued-badge";
    badge.textContent = "queued";
    div.appendChild(badge);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendToAssistantMessage(text) {
    if (!currentAssistantMsg) {
        removeTyping();
        const div = document.createElement("div");
        div.className = "chat-msg assistant";
        messagesEl.appendChild(div);
        currentAssistantMsg = div;
        streamBuffer = "";
    }
    streamBuffer += text;
    currentAssistantMsg.dataset.rawContent = streamBuffer;
    currentAssistantMsg.innerHTML = formatMarkdown(streamBuffer);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendProvisionalToken(text) {
    appendToAssistantMessage(text);
    currentAssistantProvisional = true;
    if (currentAssistantMsg) currentAssistantMsg.classList.add("provisional-stream");
}

function resetProvisionalMessage() {
    if (currentAssistantProvisional && currentAssistantMsg) {
        currentAssistantMsg.remove();
        currentAssistantMsg = null;
        streamBuffer = "";
    }
    currentAssistantProvisional = false;
}

function commitProvisionalMessage() {
    if (currentAssistantMsg) currentAssistantMsg.classList.remove("provisional-stream");
    currentAssistantProvisional = false;
}

const PROVIDER_PHASE_LABELS = {
    preparing: "Preparing request",
    waiting_first_token: "Waiting for first token",
    generating: "Generating response",
    working: "Model is working",
};

function showProviderStatus(data = {}) {
    removeTyping();
    if (!providerStatusEl) {
        providerStatusEl = document.createElement("div");
        providerStatusEl.className = "chat-msg provider-status";
        providerStatusEl.innerHTML = `<span class="provider-status-pulse"></span>`
            + `<span class="provider-status-text"></span>`
            + `<span class="provider-status-elapsed"></span>`
            + `<button type="button" class="provider-status-stop">Stop</button>`;
        providerStatusEl.querySelector(".provider-status-stop")?.addEventListener("click", () => {
            window.apiClient.cancelGeneration();
            const button = providerStatusEl?.querySelector(".provider-status-stop");
            if (button) {
                button.disabled = true;
                button.textContent = "Stopping…";
            }
        });
        messagesEl.appendChild(providerStatusEl);
        providerStatusStartedAt = Date.now();
        providerStatusTimer = setInterval(() => {
            if (!providerStatusEl || !providerStatusStartedAt) return;
            const elapsed = Math.floor((Date.now() - providerStatusStartedAt) / 1000);
            const elapsedEl = providerStatusEl.querySelector(".provider-status-elapsed");
            if (elapsedEl) elapsedEl.textContent = _formatElapsed(elapsed);
        }, 1000);
    }
    providerStatusEl.dataset.phase = data.phase || "working";
    const label = PROVIDER_PHASE_LABELS[data.phase] || data.message || "Model is working";
    const identity = [data.provider, data.model].filter(Boolean).join(" · ");
    const textEl = providerStatusEl.querySelector(".provider-status-text");
    if (textEl) textEl.textContent = identity ? `${label} — ${identity}` : label;
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

const TERMINAL_RUN_STATES = new Set([
    "completed", "completed_with_limits", "failed", "cancelled", "interrupted", "blocked",
]);

function showRunState(run = {}) {
    if (!run?.run_id) return;
    currentRun = run;
    if (!TERMINAL_RUN_STATES.has(run.status)) {
        const usage = run.usage || {};
        const limits = run.limits || {};
        const step = run.current_step || run.phase || "Working";
        const progress = `${usage.turns || 0}/${limits.max_turns || "∞"} steps`;
        showProviderStatus({
            phase: run.phase || "working",
            message: `${step} · ${progress}`,
        });
        return;
    }
    removeProviderStatus();
    const target = currentAssistantMsg || getLastAssistantMessage();
    if (!target || target.querySelector(`[data-run-summary="${run.run_id}"]`)) return;
    const usage = run.usage || {};
    const summary = document.createElement("div");
    summary.className = `run-envelope-summary run-envelope-${run.status}`;
    summary.dataset.runSummary = run.run_id;
    const reason = run.stop_reason ? ` · ${run.stop_reason.replaceAll("_", " ")}` : "";
    summary.textContent = `${run.status.replaceAll("_", " ")} · ${usage.turns || 0} steps · ${usage.total_tokens || 0} tokens${reason}`;
    target.appendChild(summary);
}

function removeProviderStatus() {
    if (providerStatusTimer) clearInterval(providerStatusTimer);
    providerStatusTimer = null;
    providerStatusStartedAt = null;
    if (providerStatusEl) providerStatusEl.remove();
    providerStatusEl = null;
}

function addToolIndicator(toolName, argsText) {
    removeTyping();
    toolStepCount++;

    if (!operationStartTime) {
        operationStartTime = Date.now();
    }

    const label = friendlyToolName(toolName);

    if (!currentToolIndicator) {
        // Create the tool activity log container
        const div = document.createElement("div");
        div.className = "chat-msg tool-indicator tool-activity-log";
        div.innerHTML = "";
        messagesEl.appendChild(div);
        currentToolIndicator = div;
    }

    // Mark previous active step as completed (without result — will be filled by tool_end)
    _finishActiveStep();

    // Add new active step
    const step = document.createElement("div");
    step.className = "tool-step-row tool-step-active";
    step.dataset.toolName = toolName;

    // Build label with args summary
    let displayLabel = label;
    if (argsText) {
        // Truncate long args for display
        const shortArgs = argsText.length > 100 ? argsText.slice(0, 97) + "..." : argsText;
        displayLabel += ` <span class="tool-step-args">${escapeHtml(shortArgs)}</span>`;
    }

    step.innerHTML = `<span class="tool-spinner"></span>`
        + `<span class="tool-step-label">${displayLabel}</span>`;
    step.dataset.startTime = Date.now();
    currentToolIndicator.appendChild(step);

    _startElapsedTimer();
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function markToolEnd(toolName, resultText) {
    if (!currentToolIndicator) return;
    const activeStep = currentToolIndicator.querySelector(".tool-step-active");
    if (!activeStep) return;

    // Add result snippet if available
    if (resultText) {
        const short = resultText.length > 120 ? resultText.slice(0, 117) + "..." : resultText;
        // Only show meaningful results (skip huge JSON)
        if (short && !short.startsWith("{") && short.length > 2) {
            let resultEl = activeStep.querySelector(".tool-step-result");
            if (!resultEl) {
                resultEl = document.createElement("span");
                resultEl.className = "tool-step-result";
                activeStep.appendChild(resultEl);
            }
            resultEl.textContent = `→ ${short}`;
        }
    }

    _finishActiveStep();
}

function addThinkingStep() {
    if (!currentToolIndicator) {
        // Create the tool activity log container if not exists
        const div = document.createElement("div");
        div.className = "chat-msg tool-indicator tool-activity-log";
        div.innerHTML = "";
        messagesEl.appendChild(div);
        currentToolIndicator = div;
    }

    _finishActiveStep();

    const step = document.createElement("div");
    step.className = "tool-step-row tool-step-active tool-step-thinking";
    step.innerHTML = `<span class="tool-spinner"></span>`
        + `<span class="tool-step-label">Analyzing results...</span>`;
    step.dataset.startTime = Date.now();
    currentToolIndicator.appendChild(step);

    _startElapsedTimer();
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addTeamStatusStep(data) {
    if (!currentToolIndicator) {
        const div = document.createElement("div");
        div.className = "chat-msg tool-indicator tool-activity-log";
        messagesEl.appendChild(div);
        currentToolIndicator = div;
    }
    _finishActiveStep();
    const step = document.createElement("div");
    const active = data.phase === "planning";
    step.className = `tool-step-row ${active ? "tool-step-active" : "tool-step-done"}`;
    const members = Array.isArray(data.members) ? data.members : [];
    const detail = members.length
        ? ` · ${members.map((item) => item.role).join(", ")}`
        : "";
    step.innerHTML = active
        ? `<span class="tool-spinner"></span><span class="tool-step-label"></span>`
        : `<span class="tool-check">✓</span><span class="tool-step-label"></span>`;
    step.querySelector(".tool-step-label").textContent = `${data.message || "Agent team"}${detail}`;
    currentToolIndicator.appendChild(step);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function _finishActiveStep() {
    if (!currentToolIndicator) return;
    const prevActive = currentToolIndicator.querySelector(".tool-step-active");
    if (prevActive) {
        prevActive.classList.remove("tool-step-active");
        prevActive.classList.add("tool-step-done");
        const spinner = prevActive.querySelector(".tool-spinner");
        if (spinner) spinner.outerHTML = '<span class="tool-check">✓</span>';
        // Remove elapsed from completed step
        const elapsed = prevActive.querySelector(".tool-elapsed");
        if (elapsed) elapsed.remove();
    }
}

function _elapsedSeconds() {
    if (!operationStartTime) return 0;
    return Math.floor((Date.now() - operationStartTime) / 1000);
}

function _formatElapsed(sec) {
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}m ${s}s`;
}

function _startElapsedTimer() {
    if (elapsedTimerInterval) clearInterval(elapsedTimerInterval);
    elapsedTimerInterval = setInterval(() => {
        if (!currentToolIndicator) {
            _stopElapsedTimer();
            return;
        }
        // Update elapsed time on the currently active step
        const activeStep = currentToolIndicator.querySelector(".tool-step-active");
        if (activeStep && activeStep.dataset.startTime) {
            const stepSec = Math.floor((Date.now() - parseInt(activeStep.dataset.startTime)) / 1000);
            if (stepSec >= 2) {
                let el = activeStep.querySelector(".tool-elapsed");
                if (el) {
                    el.textContent = _formatElapsed(stepSec);
                } else {
                    const span = document.createElement("span");
                    span.className = "tool-elapsed";
                    span.textContent = _formatElapsed(stepSec);
                    activeStep.appendChild(span);
                }
            }
        }
    }, 1000);
}

function _stopElapsedTimer() {
    if (elapsedTimerInterval) {
        clearInterval(elapsedTimerInterval);
        elapsedTimerInterval = null;
    }
}

function removeToolIndicator() {
    if (currentToolIndicator) {
        currentToolIndicator.remove();
        currentToolIndicator = null;
    }
    _stopElapsedTimer();
}

function _resetToolProgress() {
    toolStepCount = 0;
    operationStartTime = null;
}

function addErrorMessage(content, retryable = false) {
    removeTyping();
    removeToolIndicator();
    const div = document.createElement("div");
    div.className = "chat-msg assistant error";

    const textEl = document.createElement("span");
    textEl.textContent = content;
    div.appendChild(textEl);

    if (retryable && lastUserMessage) {
        const btn = document.createElement("button");
        btn.className = "btn-retry";
        btn.textContent = "Retry";
        btn.addEventListener("click", () => {
            div.remove();
            resendLastMessage();
        });
        div.appendChild(btn);
    }

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function resendLastMessage() {
    if (!lastUserMessage) return;
    showTyping();
    if (lastUserMessage.type === "message") {
        window.apiClient.sendMessage(lastUserMessage.text, {
            contextReducerCompare: Boolean(lastUserMessage.contextReducerCompare),
            contextReducerApply: Boolean(lastUserMessage.contextReducerApply),
            workspaceId: lastUserMessage.workspaceId || activeCodeWorkspace?.workspace_id,
            teamMode: lastUserMessage.teamMode || "off",
        });
    } else if (lastUserMessage.type === "voice") {
        window.apiClient.sendVoice(lastUserMessage.audio, lastUserMessage.mime_type);
    } else if (lastUserMessage.type === "file") {
        window.apiClient.sendFile(
            lastUserMessage.data, lastUserMessage.name,
            lastUserMessage.mime_type, lastUserMessage.text
        );
    }
}

// Auto-probe: app.js requests a probe when countdown expires
document.addEventListener("llm-probe-request", () => {
    if (!limitedMode) return;
    if (messageQueue.length > 0) {
        // Re-send first queued message as probe
        const msg = messageQueue[0];
        showTyping();
        if (msg.type === "message") {
            window.apiClient.sendMessage(msg.text, {
                contextReducerCompare: Boolean(msg.contextReducerCompare),
                contextReducerApply: Boolean(msg.contextReducerApply),
                workspaceId: msg.workspaceId || activeCodeWorkspace?.workspace_id,
                teamMode: msg.teamMode || "off",
            });
        } else if (msg.type === "voice") {
            window.apiClient.sendVoice(msg.audio, msg.mime_type);
        } else if (msg.type === "file") {
            window.apiClient.sendFile(msg.data, msg.name, msg.mime_type, msg.text);
        }
    } else {
        // No queued messages — send a lightweight probe
        showTyping();
        window.apiClient.sendMessage("ping", { workspaceId: activeCodeWorkspace?.workspace_id });
    }
});

window.apiClient.onMessage((data) => {
    switch (data.type) {
        case "typing":
            showTyping();
            break;
        case "token":
            removeProviderStatus();
            removeToolIndicator();
            consecutiveLlmErrors = 0;
            if (limitedMode) exitLimitedMode();
            appendToAssistantMessage(data.content);
            if (!trajectoryTokenNotified) {
                trajectoryTokenNotified = true;
                notifyTrajectoryChange("first-output");
            }
            break;
        case "provisional_token":
            consecutiveLlmErrors = 0;
            if (limitedMode) exitLimitedMode();
            appendProvisionalToken(data.content || "");
            if (!trajectoryTokenNotified) {
                trajectoryTokenNotified = true;
                notifyTrajectoryChange("first-output");
            }
            break;
        case "provisional_reset":
            resetProvisionalMessage();
            break;
        case "provisional_commit":
            commitProvisionalMessage();
            removeProviderStatus();
            break;
        case "provider_status":
            showProviderStatus(data);
            notifyTrajectoryChange("request");
            break;
        case "run_state":
            showRunState(data.run || {});
            notifyTrajectoryChange("run-state");
            break;
        case "stopped":
            removeProviderStatus();
            if (data.content && !streamBuffer) appendToAssistantMessage(data.content);
            notifyTrajectoryChange("stopped");
            break;
        case "tool_start":
            removeProviderStatus();
            addToolIndicator(data.content, data.args || "");
            notifyTrajectoryChange("tool-start");
            break;
        case "tool_end":
            markToolEnd(data.content, data.result || "");
            notifyTrajectoryChange("tool-end");
            break;
        case "thinking":
            addThinkingStep();
            break;
        case "team_status":
            addTeamStatusStep(data);
            notifyTrajectoryChange("agent-team");
            break;
        case "text":
            // Fallback for non-streaming or final blocks
            removeToolIndicator();
            consecutiveLlmErrors = 0;
            if (limitedMode) exitLimitedMode();
            // If we were streaming, this might be duplicative if we appended tokens.
            // But api.py only sends "text" on error or specific cases now.
            // For safety, start a new block if it's a full block
            currentAssistantMsg = null;
            addMessage("assistant", data.content);
            if (data.speak) {
                speakText(data.content);
            }
            break;
        case "tool_call":
            addToolCall(`Tool: ${data.name}(${JSON.stringify(data.args)})`);
            break;
        case "factuality":
            attachFactualitySummary(data.factuality);
            break;
        case "pipeline_candidate":
            currentAssistantMsg = null;
            addMessage(
                "assistant",
                `I noticed this task may be repeatable and created a safe pipeline draft: ${data.candidate?.title || "Untitled"}. Open Knowledge → Runtime to dry-run and approve it.`
            );
            break;
        case "context_reducer_compare":
            addContextReducerCompare(data.report);
            break;
        case "llm_optimization_apply":
            addLlmOptimizationApply(data.report);
            break;
        case "error":
            removeProviderStatus();
            resetProvisionalMessage();
            if (data.retryable) {
                consecutiveLlmErrors++;
                if (consecutiveLlmErrors >= LLM_ERROR_THRESHOLD && !limitedMode) {
                    enterLimitedMode(data.error_class);
                }
            }
            if (limitedMode && messageQueue.length > 0) {
                // Probe failed — suppress error display, just clean up
                removeTyping();
                removeToolIndicator();
            } else {
                addErrorMessage(data.content, data.retryable);
            }
            _resetToolProgress();
            notifyTrajectoryChange("error");
            break;
        case "done":
            if (data.run?.run_id) showRunState(data.run);
            commitProvisionalMessage();
            removeProviderStatus();
            removeTyping();
            removeToolIndicator();
            _resetToolProgress();
            attachResponseControls(currentAssistantMsg || getLastAssistantMessage(), data.token_usage);
            streamBuffer = "";
            currentAssistantMsg = null;
            currentRun = null;
            trajectoryTokenNotified = false;
            // Best-effort brain-state plate under the reply. Fire-and-forget.
            attachBrainMetaPlate();
            document.dispatchEvent(new CustomEvent("conversation-list-refresh"));
            notifyTrajectoryChange("done");
            break;
        case "session_reset":
            messagesEl.innerHTML = "";
            removeToolIndicator();
            _resetToolProgress();
            consecutiveLlmErrors = 0;
            messageQueue = [];
            if (limitedMode) exitLimitedMode();
            addMessage("assistant", "New session started.");
            break;
    }
});

// Research results arrive independently of the chat turn. Persist seen IDs so
// reconnecting clients receive missed results exactly once.
const _seenResearchNotifications = new Set(
    JSON.parse(localStorage.getItem("remy-seen-research-notifications") || "[]")
);

function _showResearchNotification(event) {
    const id = event.id || `${event.type}:${event.project_id || event.topic || event.timestamp}`;
    if (_seenResearchNotifications.has(id)) return;
    _seenResearchNotifications.add(id);
    localStorage.setItem(
        "remy-seen-research-notifications",
        JSON.stringify(Array.from(_seenResearchNotifications).slice(-100))
    );
    const fallback = event.type === "research.failed"
        ? `Research “${event.topic || event.project_id}” needs attention: ${event.error || "no grounded result"}`
        : `Research “${event.topic || event.project_id}” is complete.`;
    addMessage("assistant", event.message || event.markdown || fallback, "research-notification");
}

window.apiClient.onRuntimeEvent((event) => {
    if (event.type === "research.complete" || event.type === "research.failed") {
        _showResearchNotification(event);
    }
});
window.apiClient.connectRuntimeStream();
window.apiClient.getResearchNotifications(20)
    .then(data => (data.items || []).slice().reverse().forEach(_showResearchNotification))
    .catch(() => {});

messagesEl?.addEventListener("click", (event) => {
    const btn = event.target.closest(".factuality-feedback-btn");
    if (!btn) return;
    handleFactualityFeedback(btn);
});

export function clearChat() {
    messagesEl.innerHTML = "";
    removeToolIndicator();
    _resetToolProgress();
    consecutiveLlmErrors = 0;
    messageQueue = [];
    if (limitedMode) exitLimitedMode();
}
