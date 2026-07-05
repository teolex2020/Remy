/**
 * Automations — visual canvas builder (Drawflow) with Trigger → steps → Output.
 *
 * Differs from Pipelines:
 *  - "Trigger" node (instead of Start): configures when automation fires
 *  - "Output" node (instead of Result): where the result is delivered
 *  - Saved to /api/automations (not /api/pipelines)
 *  - No chat input/output — runs on schedule, delivers to Telegram/Email/Memory/Chat
 */

// ── Block catalogue (same as pipelines + trigger + output) ───────────────────

const BLOCKS = [
  { type: "llm_call",      icon: "🤖", label: "AI Response",    color: "#6366f1",
    inputs: 1, outputs: 1,
    defaults: { system_prompt: "", input_source: "{{prev}}", model: "" } },
  { type: "web_search",    icon: "🔍", label: "Web Search",      color: "#0ea5e9",
    inputs: 1, outputs: 1,
    defaults: { num_results: 5, input_source: "{{prev}}" } },
  { type: "page_scrape",   icon: "Pg", label: "Page Scraper",    color: "#0f766e",
    inputs: 1, outputs: 1,
    defaults: { url: "", mode: "text", max_chars: 12000 } },
  { type: "memory_search", icon: "🧠", label: "Memory Search",   color: "#8b5cf6",
    inputs: 1, outputs: 1,
    defaults: { limit: 5, input_source: "{{prev}}" } },
  { type: "memory_save",   icon: "💾", label: "Save to Memory",  color: "#10b981",
    inputs: 1, outputs: 1,
    defaults: { tags: "automation", input_source: "{{prev}}" } },
  { type: "http_request",  icon: "🌐", label: "HTTP Request",    color: "#f59e0b",
    inputs: 1, outputs: 1,
    defaults: { url: "", method: "GET", body: "", auth_secret_key: "", auth_scheme: "Bearer" } },
  { type: "router",        icon: "?", label: "Router",             color: "#f97316",
    inputs: 1, outputs: 2,
    defaults: {
      input_source: "{{prev}}",
      mode: "all_matching",
      routes: [
        { label: "Route 1", operator: "contains", value: "" },
        { label: "Fallback", operator: "fallback", value: "" },
      ],
    } },
  { type: "merge",         icon: "M", label: "Merge / Wait",       color: "#14b8a6",
    inputs: 2, outputs: 1,
    defaults: { input_count: 2, mode: "combine_text", separator: "\n\n---\n\n" } },
  { type: "delay",         icon: "T", label: "Delay / Wait",       color: "#a78bfa",
    inputs: 1, outputs: 1,
    defaults: { seconds: 1, input_source: "{{prev}}" } },
  { type: "filter",        icon: "F", label: "Filter",             color: "#22c55e",
    inputs: 1, outputs: 1,
    defaults: { input_source: "{{prev}}", operator: "contains", value: "" } },
  { type: "set_variable",  icon: "=", label: "Set Variable",       color: "#38bdf8",
    inputs: 1, outputs: 1,
    defaults: { name: "value", value: "{{prev}}" } },
  { type: "parse_json",    icon: "{}", label: "Parse JSON",        color: "#06b6d4",
    inputs: 1, outputs: 1,
    defaults: { text: "{{prev}}", path: "$" } },
  { type: "transform",     icon: "Tx", label: "Format / Transform", color: "#84cc16",
    inputs: 1, outputs: 1,
    defaults: { text: "{{prev}}", mode: "trim", find: "", replace: "", pattern: "", limit: 1000, separator: " " } },
  { type: "notification",  icon: "!", label: "Notification",       color: "#f43f5e",
    inputs: 1, outputs: 1,
    defaults: { title: "Workflow notification", message: "{{prev}}" } },
  { type: "file_read",     icon: "R", label: "File Read",          color: "#94a3b8",
    inputs: 1, outputs: 1,
    defaults: { filename: "workflow.txt", max_chars: 20000 } },
  { type: "file_write",    icon: "W", label: "File Write",         color: "#94a3b8",
    inputs: 1, outputs: 1,
    defaults: { filename: "workflow.txt", text: "{{prev}}", mode: "overwrite" } },
  { type: "code",          icon: "</>", label: "Code / Script",     color: "#ef4444",
    inputs: 1, outputs: 1,
    defaults: { mode: "safe_expression", language: "python", code: "input.strip()", allow_local_execution: false, timeout_seconds: 3, max_output_chars: 12000 } },
  { type: "error_handler", icon: "E", label: "Error Handler",      color: "#fb7185",
    inputs: 1, outputs: 1,
    defaults: { input_source: "{{prev}}", fallback_text: "Handled error: {{error}}" } },
  { type: "template",      icon: "📝", label: "Text / Template", color: "#64748b",
    inputs: 1, outputs: 1,
    defaults: { text: "" } },
];

function _routerRoutes(data) {
  if (Array.isArray(data?.routes) && data.routes.length) {
    return data.routes.map((r, i) => _normaliseRouterRoute(r, i, data.routes.length));
  }
  const routes = [];
  for (let i = 1; i <= 24; i++) {
    const label = data?.[`route_${i}_label`] || "";
    const condition = data?.[`route_${i}_condition`] || "";
    if (label || condition) routes.push(_normaliseRouterRoute({ label: label || `Route ${i}`, condition }, i - 1, 24));
  }
  return routes.length ? routes : [{ label: "Route 1", operator: "contains", value: "" }, { label: "Fallback", operator: "fallback", value: "" }];
}

function _normaliseRouterRoute(route, index, total) {
  const parsed = _parseRouterCondition(route?.condition || "");
  return {
    label: route?.label || (index === total - 1 ? "Fallback" : `Route ${index + 1}`),
    operator: route?.operator || parsed.operator,
    value: route?.value ?? parsed.value,
    condition: route?.condition || "",
  };
}

function _parseRouterCondition(condition) {
  const text = (condition || "").trim();
  const low = text.toLowerCase();
  if (!text) return { operator: "fallback", value: "" };
  if (["always", "true", "*"].includes(low)) return { operator: "always", value: "" };
  if (["never", "false"].includes(low)) return { operator: "never", value: "" };
  const rules = [
    ["not_contains", ["not contains:", "not contains "]],
    ["contains", ["contains:", "contains "]],
    ["equals", ["equals:", "equals "]],
  ];
  for (const [operator, prefixes] of rules) {
    for (const prefix of prefixes) {
      if (low.startsWith(prefix)) return { operator, value: text.slice(prefix.length).trim() };
    }
  }
  return text.length <= 120 && !text.includes("\n")
    ? { operator: "contains", value: text }
    : { operator: "ai", value: text };
}

function _routerOperatorOptions(current) {
  const options = [
    ["contains", "Contains"],
    ["not_contains", "Does not contain"],
    ["equals", "Equals"],
    ["not_equals", "Does not equal"],
    ["starts_with", "Starts with"],
    ["ends_with", "Ends with"],
    ["regex", "Matches regex"],
    ["is_empty", "Is empty"],
    ["is_not_empty", "Is not empty"],
    ["always", "Always pass"],
    ["never", "Never pass"],
    ["fallback", "Fallback if nothing matched"],
    ["ai", "AI condition"],
  ];
  return options.map(([value, label]) => `<option value="${value}" ${current === value ? "selected" : ""}>${label}</option>`).join("");
}

function _routerModeLabel(mode) {
  const labels = { all: "pass all", first_match: "first match", all_matching: "all matches" };
  return labels[mode || "all_matching"] || "all matches";
}

function _syncRouterOutputs(id, count) {
  const node = _editor?.getNodeFromId(id);
  if (!node) return;
  const target = Math.max(1, count || 1);
  let current = Object.keys(node.outputs || {}).length;
  while (current < target) {
    _editor.addNodeOutput(id);
    current++;
  }
  while (current > target) {
    _editor.removeNodeOutput(id, `output_${current}`);
    current--;
  }
}

function _mergeInputCount(data) {
  const n = parseInt(data?.input_count || 2, 10);
  return Math.max(2, Math.min(Number.isFinite(n) ? n : 2, 10));
}

function _syncMergeInputs(id, count) {
  const node = _editor?.getNodeFromId(id);
  if (!node) return;
  const target = Math.max(2, Math.min(parseInt(count || 2, 10) || 2, 10));
  let current = Object.keys(node.inputs || {}).length;
  while (current < target) {
    _editor.addNodeInput(id);
    current++;
  }
  while (current > target) {
    _editor.removeNodeInput(id, `input_${current}`);
    current--;
  }
}

function _block(type) {
  return BLOCKS.find(b => b.type === type)
    || { icon: "❓", label: type, color: "#64748b", inputs: 1, outputs: 1, defaults: {} };
}

// ── State ─────────────────────────────────────────────────────────────────────

function _blockHelpText(type) {
  const help = {
    trigger: {
      title: "What this block does",
      body: "Defines when the automation runs: manually, on app start, hourly, daily, weekly, or by cron.",
      tip: "Tip: keep Catch up enabled if you want missed scheduled runs to execute after the app starts.",
    },
    output: {
      title: "What this block does",
      body: "Chooses where the final automation result is delivered: chat, memory, Telegram, email, or webhook.",
      tip: "Tip: Chat is safest for local desktop use; Memory is useful when the result should become searchable later.",
    },
    llm_call: {
      title: "What this block does",
      body: "Uses the AI model to analyze, summarize, classify, rewrite, or generate text from previous block output.",
      tip: "Tip: write the system prompt as an instruction for the result you want, not as a conversation.",
    },
    web_search: {
      title: "What this block does",
      body: "Searches the web and passes found source text forward. Use it for scheduled monitoring, daily checks, or research routines.",
      tip: "Tip: leave query empty to use the previous block output as the search query.",
    },
    page_scrape: {
      title: "What this block does",
      body: "Fetches one web page, extracts readable content, and passes clean text, title, or links to the next block.",
      tip: "Tip: use it before AI Response for page analytics, changelog monitoring, and report extraction.",
    },
    memory_search: {
      title: "What this block does",
      body: "Searches local memory for relevant facts and passes them to the next block.",
      tip: "Tip: use it before AI Response when the automation should answer from stored user context.",
    },
    memory_save: {
      title: "What this block does",
      body: "Saves the current result to memory with tags. Use it to keep reports, summaries, decisions, or recurring observations.",
      tip: "Tip: tags make future memory search much easier; use stable names.",
    },
    http_request: {
      title: "What this block does",
      body: "Calls an external HTTP endpoint. Use it to read an API, send a webhook, or connect to another local service.",
      tip: "Tip: for scheduled automations, avoid endpoints that require manual login or captcha.",
    },
    router: {
      title: "What this block does",
      body: "Splits execution into one or more routes using conditions. Useful when different results need different actions.",
      tip: "Tip: add a Fallback route so the automation still has a path when no condition matches.",
    },
    merge: {
      title: "What this block does",
      body: "Waits for multiple branches and combines their outputs. Use it after Router when several paths must finish before delivery.",
      tip: "Tip: connect at least two inputs, then send Merge output to AI Response or Output.",
    },
    delay: {
      title: "What this block does",
      body: "Pauses the automation before continuing. Use it when another system needs time before the next step.",
      tip: "Tip: keep delays short so scheduled runs do not stack up.",
    },
    filter: {
      title: "What this block does",
      body: "Allows the automation to continue only if the condition matches. If not, this branch stops.",
      tip: "Tip: use Router for alternate paths; use Filter for pass or stop.",
    },
    set_variable: {
      title: "What this block does",
      body: "Stores a value under a name so later blocks can use it with {{name}}.",
      tip: "Tip: use stable names like report_title or alert_level.",
    },
    parse_json: {
      title: "What this block does",
      body: "Extracts a value from JSON using a path like $.items[0].title.",
      tip: "Tip: place it after HTTP Request when an API returns structured JSON.",
    },
    transform: {
      title: "What this block does",
      body: "Cleans or reshapes text: trim, upper/lowercase, replace, regex extract, truncate, or join lines.",
      tip: "Tip: use it before AI Response to reduce noisy input.",
    },
    notification: {
      title: "What this block does",
      body: "Creates a workflow notification entry and passes the message forward.",
      tip: "Tip: use it when a scheduled run finds something important.",
    },
    file_read: {
      title: "What this block does",
      body: "Reads a text file from the app workflow_files directory.",
      tip: "Tip: file access is limited to the app data folder for safety.",
    },
    file_write: {
      title: "What this block does",
      body: "Writes text to a file in the app workflow_files directory.",
      tip: "Tip: use append for logs and overwrite for the latest snapshot.",
    },
    code: {
      title: "What this block does",
      body: "Runs a safe expression by default. Local scripts require explicit opt-in inside the block.",
      tip: "Tip: use Safe Expression for data transforms; enable Local Script only when you intentionally want local code execution.",
    },
    error_handler: {
      title: "What this block does",
      body: "Handles errors from a connected second output. It receives the error text and returns the configured fallback text.",
      tip: "Tip: connect output_2 from a risky block into Error Handler, then connect Error Handler to Output or a recovery step.",
    },
    template: {
      title: "What this block does",
      body: "Creates static text or a prompt template. Use {{prev}} to insert the previous block output.",
      tip: "Tip: use it to format a clean instruction before AI Response or to prepare text for Memory Save.",
    },
  };
  return help[type] || {
    title: "What this block does",
    body: "Processes previous output and sends a result to the next connected block.",
    tip: "Tip: after running, click the block status badge to inspect what it produced.",
  };
}

function _blockHelpHtml(type) {
  const help = _blockHelpText(type);
  return `<div class="pf-block-help">
    <div class="pf-block-help-title">${_esc(help.title)}</div>
    <div class="pf-block-help-body">${_esc(help.body)}</div>
    <div class="pf-block-help-tip">${_esc(help.tip)}</div>
  </div>`;
}

let _editor         = null;
let _autoMeta       = null;   // { id, name }
let _selectedNodeId = null;
let _availableModels = [];
let _availableSecrets = [];
let _settingsSnapshot = null;
let _lastRunTraceByStepId = new Map();

// ── Public entry point ────────────────────────────────────────────────────────

export async function loadAutomations() {
  const pane = document.getElementById("automations-content");
  if (!pane) return;

  const alreadyMounted = !!pane.querySelector("#at-sidebar");
  if (!alreadyMounted) {
    _renderShell(pane);
    document.getElementById("btn-new-automation")
      ?.addEventListener("click", () => _openEditor(null));
    await _fetchModels();
  }

  await _renderAutomationList();
  await _renderAutomationTemplates();
  await _openPendingAutomationFromHome();
}

export async function openAutomationMemoryHistory(automationId) {
  const id = String(automationId || "");
  if (!id) return false;
  await loadAutomations();
  const data = await fetch(`/api/automations/${encodeURIComponent(id)}`).then(r => r.json()).catch(() => null);
  if (!data) return false;
  _openEditor(data);
  await _openAutomationHistory();
  return true;
}

// ── Shell ─────────────────────────────────────────────────────────────────────

function _renderShell(pane) {
  pane.innerHTML = `
  <div class="pf-shell pf-mode-list">
    <!-- Sidebar -->
    <div class="pf-sidebar" id="at-sidebar">
      <div class="pf-sidebar-header">
        <span class="pf-sidebar-title">Automations</span>
        <button class="btn btn-primary btn-sm" id="at-new-btn">+ New</button>
      </div>

      <div id="at-list-panel">
        <div class="pf-section-label">Saved</div>
        <div id="at-automation-list" class="at-automation-list-wrap">
          <div class="pf-loading">Loading…</div>
        </div>
        <div class="pf-section-label" style="margin-top:1rem">Templates</div>
        <div id="at-template-list" class="at-automation-list-wrap">
          <div class="pf-loading">Loading...</div>
        </div>
      </div>

      <!-- Palette (visible while editing) -->
      <div id="at-palette-panel" class="hidden">
        <div class="pf-section-label">← Drag a block to the canvas</div>
        ${BLOCKS.map(b => `
          <div class="pf-palette-item" draggable="true" data-type="${b.type}" style="--bc:${b.color}">
            <span class="pf-pi-icon">${b.icon}</span>
            <span class="pf-pi-label">${b.label}</span>
          </div>`).join("")}
        <div class="pf-section-label" style="margin-top:1rem">Hint</div>
        <div class="pf-hint-box">Drag blocks onto the canvas. Connect dots between blocks. Use the gear on a block to configure it.</div>
      </div>
    </div>

    <!-- Canvas -->
    <div class="pf-canvas-wrap hidden" id="at-canvas-wrap">
      <div class="pf-toolbar">
        <button class="btn btn-outline btn-sm" id="at-back-btn">← Back</button>
        <div class="pf-toolbar-meta">
          <input class="pf-title-input" id="at-title-input" placeholder="Automation name…">
        </div>
        <div class="pf-toolbar-actions">
          <button class="btn btn-outline btn-sm" id="at-zoom-in">+</button>
          <button class="btn btn-outline btn-sm" id="at-zoom-out">−</button>
          <button class="btn btn-outline btn-sm" id="at-zoom-reset">↺</button>
          <button class="btn btn-outline btn-sm" id="at-history-btn">History</button>
          <button class="btn btn-outline btn-sm" id="at-save-template-btn">Save as Template</button>
          <button class="btn btn-outline btn-sm" id="at-save-btn">💾 Save</button>
          <button class="btn btn-primary btn-sm"  id="at-run-btn">▶ Run now</button>
        </div>
      </div>
      <div id="at-drawflow" class="pf-drawflow-container"></div>
    </div>

    <!-- Config panel -->
    <div class="pf-config-panel hidden" id="at-config-panel">
      <div class="pf-config-resize" id="at-config-resize" title="Resize settings panel"></div>
      <div class="pf-config-header">
        <span id="at-config-title">Settings</span>
        <button class="pf-config-close" id="at-config-close">✕</button>
      </div>
      <div id="at-config-body" class="pf-config-body"></div>
    </div>
  </div>

  <!-- Run result modal -->
  <div class="pf-run-modal hidden" id="at-run-modal">
    <div class="pf-run-box">
      <div class="pf-run-header">
        <span id="at-run-title">Running automation</span>
        <button class="pf-config-close" id="at-run-close">✕</button>
      </div>
      <div id="at-run-steps" class="pf-run-steps"></div>
      <div class="pf-run-final hidden" id="at-run-final">
        <div class="pf-run-final-label">Result</div>
        <div class="pf-run-final-text" id="at-run-final-text"></div>
        <div class="pf-run-final-actions">
          <button class="btn btn-outline btn-sm" id="at-copy-btn">📋 Copy</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Run history modal -->
  <div class="pf-run-modal hidden" id="at-history-modal">
    <div class="pf-run-box">
      <div class="pf-run-header">
        <span>Automation history</span>
        <button class="pf-config-close" id="at-history-close">✕</button>
      </div>
      <div id="at-history-body" class="pf-run-steps"></div>
    </div>
  </div>

  <!-- Per-block result modal -->
  <div class="pf-step-modal hidden" id="at-step-modal">
    <div class="pf-step-box">
      <div class="pf-run-header">
        <span id="at-step-title">Block result</span>
        <button class="pf-config-close" id="at-step-close">&#10005;</button>
      </div>
      <div class="pf-step-meta" id="at-step-meta"></div>
      <pre class="pf-run-step-output pf-step-output" id="at-step-output"></pre>
      <div class="pf-step-actions">
        <button class=”btn btn-outline btn-sm” id=”at-step-copy”>&#128203; Copy</button>
        <button class="btn btn-outline btn-sm" id="at-step-pin">Pin result</button>
      </div>
    </div>
  </div>`;

  document.getElementById("at-new-btn").addEventListener("click", () => _openEditor(null));
  document.getElementById("at-back-btn")?.addEventListener("click", _closeEditor);
  document.getElementById("at-save-btn")?.addEventListener("click", _saveAutomation);
  document.getElementById("at-save-template-btn")?.addEventListener("click", _saveAutomationAsTemplate);
  document.getElementById("at-run-btn")?.addEventListener("click", _runNow);
  document.getElementById("at-history-btn")?.addEventListener("click", _openAutomationHistory);
  document.getElementById("at-zoom-in")?.addEventListener("click", () => _editor?.zoom_in());
  document.getElementById("at-zoom-out")?.addEventListener("click", () => _editor?.zoom_out());
  document.getElementById("at-zoom-reset")?.addEventListener("click", () => _editor?.zoom_reset());
  document.getElementById("at-config-close")?.addEventListener("click", _closeConfig);
  _bindConfigResize();
  document.getElementById("at-run-close")?.addEventListener("click", () => {
    document.getElementById("at-run-modal")?.classList.add("hidden");
  });
  document.getElementById("at-history-close")?.addEventListener("click", () => {
    document.getElementById("at-history-modal")?.classList.add("hidden");
  });
  document.getElementById("at-copy-btn")?.addEventListener("click", () => {
    const txt = document.getElementById("at-run-final-text")?.textContent || "";
    navigator.clipboard.writeText(txt);
  });
  document.getElementById("at-step-close")?.addEventListener("click", _hideStepResultModal);
  document.getElementById("at-step-copy")?.addEventListener("click", () => {
    const txt = document.getElementById("at-step-output")?.textContent || "";
    navigator.clipboard.writeText(txt);
  });
  document.getElementById("at-step-pin")?.addEventListener("click", _pinCurrentStepResult);
  document.getElementById("at-drawflow")?.addEventListener("click", (ev) => {
    const cfg = ev.target?.closest?.(".pf-node-config-btn");
    if (cfg) {
      ev.preventDefault();
      ev.stopPropagation();
      const nodeEl = cfg.closest(".drawflow-node");
      const nodeId = nodeEl?.id?.replace(/^node-/, "");
      if (nodeId) _openNodeConfig(nodeId);
      return;
    }

    const btn = ev.target?.closest?.(".pf-node-run-badge");
    if (!btn) return;
    ev.preventDefault();
    ev.stopPropagation();
    _openStepResult(btn.dataset.stepId);
  });
  document.getElementById("at-drawflow")?.addEventListener("pointerdown", (ev) => {
    if (!ev.target?.closest?.(".pf-node-config-btn")) return;
    ev.preventDefault();
    ev.stopPropagation();
  });
}

// ── Automation list ───────────────────────────────────────────────────────────

async function _renderAutomationList() {
  const res  = await fetch("/api/automations").then(r => r.json()).catch(() => ({ automations: [] }));
  const list = res.automations || [];
  const el   = document.getElementById("at-automation-list");
  if (!el) return;

  if (!list.length) {
    el.innerHTML = `<div class="pf-empty">No automations yet</div>`;
    return;
  }

  el.innerHTML = list.map(a => {
    const enabled = a.enabled !== false;
    const trig    = _triggerShortLabel(a.trigger || {});
    const status  = a.last_run_status
      ? ` · ${a.last_run_status}${a.consecutive_failures ? ` (${a.consecutive_failures} failed)` : ""}`
      : "";
    return `
      <div class="auto-card ${enabled ? "" : "auto-card-disabled"}" data-id="${_esc(a.id)}">
        <div class="auto-card-top">
          <span class="auto-dot ${enabled ? "auto-dot-on" : "auto-dot-off"}"></span>
          <div class="auto-card-info">
            <div class="auto-card-title" title="${_esc(a.name)}">${_esc(a.name)}</div>
            <div class="auto-card-meta">
              <span class="auto-chip">${_esc(trig)}</span>
              ${a.source_template_name ? `<span class="auto-chip-muted" title="Created from template ${_esc(a.source_template_name)}">From ${_esc(a.source_template_name)}</span>` : ""}
              <span class="workflow-memory-badge workflow-memory-loading" data-memory-report="${_esc(a.id)}"></span>
              <span class="workflow-memory-badge workflow-safety-loading" data-safety-report="${_esc(a.id)}"></span>
              ${status ? `<span class="auto-chip-muted">${_esc(status.replace(" · ", ""))}</span>` : ""}
            </div>
          </div>
        </div>
        <div class="pf-list-item-actions">
          <button class="pf-card-btn at-edit-btn" data-id="${_esc(a.id)}" title="Edit automation">
            <span class="pf-card-btn-icon">✏</span><span class="pf-card-btn-label">Edit</span>
          </button>
          <button class="pf-card-btn pf-card-btn-run at-run-list-btn" data-id="${_esc(a.id)}" title="Run now">
            <span class="pf-card-btn-icon">▶</span><span class="pf-card-btn-label">Run</span>
          </button>
          <button class="pf-card-btn at-toggle-btn" data-id="${_esc(a.id)}" data-enabled="${enabled}" title="${enabled ? "Pause automation" : "Enable automation"}">
            <span class="pf-card-btn-icon">${enabled ? "⏸" : "▶"}</span>
          </button>
          <button class="pf-card-btn pf-card-btn-danger at-del-btn" data-id="${_esc(a.id)}" title="Delete automation">
            <span class="pf-card-btn-icon">🗑</span>
          </button>
        </div>
      </div>`;
  }).join("");
  _hydrateAutomationMemoryBadges(list);
  _hydrateAutomationSafetyBadges(list);

  el.querySelectorAll(".at-edit-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const data = await fetch(`/api/automations/${btn.dataset.id}`).then(r => r.json()).catch(() => null);
      if (data) _openEditor(data);
    });
  });

  el.querySelectorAll(".at-run-list-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.id;
      if (!await _confirmAutomationPreflight(id)) return;
      _showRunModal("Running automation…");
      await _runAutomationById(id);
    });
  });

  el.querySelectorAll(".at-toggle-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const enabled = btn.dataset.enabled === "true";
      await fetch(`/api/automations/${btn.dataset.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !enabled }),
      });
      _renderAutomationList();
    });
  });

  el.querySelectorAll(".at-del-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!confirm("Delete this automation?")) return;
      await fetch(`/api/automations/${btn.dataset.id}`, { method: "DELETE" });
      _renderAutomationList();
    });
  });
}

// ── Editor ────────────────────────────────────────────────────────────────────

async function _hydrateAutomationMemoryBadges(items) {
  await Promise.all((items || []).map(async item => {
    const id = String(item.id || "");
    if (!id) return;
    const badge = document.querySelector(`.workflow-memory-badge[data-memory-report="${CSS.escape(id)}"]`);
    if (!badge) return;
    const report = await fetch(`/api/automations/${id}/memory-report`).then(r => r.json()).catch(() => null);
    _renderWorkflowMemoryBadge(badge, report);
  }));
}

async function _hydrateAutomationSafetyBadges(items) {
  await Promise.all((items || []).map(async item => {
    const id = String(item.id || "");
    if (!id) return;
    const badge = document.querySelector(`.workflow-memory-badge[data-safety-report="${CSS.escape(id)}"]`);
    if (!badge) return;
    const report = await fetch(`/api/automations/${encodeURIComponent(id)}/preflight`).then(r => r.json()).catch(() => null);
    _renderWorkflowSafetyBadge(badge, report, "Automation safety preflight");
  }));
}

function _renderWorkflowSafetyBadge(el, report, title) {
  if (!el || !report) {
    if (el) el.remove();
    return;
  }
  const blockers = report.blocker_count || (report.blockers || []).length || 0;
  const warnings = report.warning_count || (report.warnings || []).length || 0;
  const status = blockers ? "blocked" : warnings ? "warn" : "ok";
  el.className = `workflow-memory-badge workflow-memory-clickable workflow-safety-${status}`;
  el.textContent = blockers ? `Blocked ${blockers}` : warnings ? `Review ${warnings}` : "Ready";
  const first = (report.blockers || [])[0] || (report.warnings || [])[0];
  el.title = first ? first.message : "No safety issues detected.";
  el.setAttribute("role", "button");
  el.setAttribute("tabindex", "0");
  const open = () => _showSafetyPreflight(report, title);
  el.onclick = (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    open();
  };
  el.onkeydown = (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      open();
    }
  };
}

function _renderWorkflowMemoryBadge(el, report) {
  if (!el || !report || report.status === "no_data" || report.average_score === null || report.average_score === undefined) {
    if (el) el.remove();
    return;
  }
  const status = report.status || "unknown";
  el.className = `workflow-memory-badge workflow-memory-clickable workflow-memory-${status}`;
  el.textContent = `Memory ${report.average_score}/100`;
  el.title = `Open workflow memory history: ${report.average_score}/100 (${status})`;
  el.setAttribute("role", "button");
  el.setAttribute("tabindex", "0");
  const open = async () => {
    const id = el.dataset.memoryReport || "";
    const data = await fetch(`/api/automations/${id}`).then(r => r.json()).catch(() => null);
    if (data) {
      _openEditor(data);
    } else {
      const card = el.closest(".auto-card");
      const name = card?.querySelector(".auto-card-title")?.textContent || "";
      _autoMeta = { id, name };
    }
    _openAutomationHistory();
  };
  el.onclick = async (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    await open();
  };
  el.onkeydown = async (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      await open();
    }
  };
}

async function _openPendingAutomationFromHome() {
  const id = window.sessionStorage.getItem("remy_pending_automation_open") || "";
  if (!id) return;
  window.sessionStorage.removeItem("remy_pending_automation_open");
  const data = await fetch(`/api/automations/${encodeURIComponent(id)}`).then(r => r.json()).catch(() => null);
  if (data) {
    _openEditor(data);
  }
}

async function _renderAutomationTemplates() {
  const res = await fetch("/api/automations/templates/list").then(r => r.json()).catch(() => ({ templates: [] }));
  const templates = res.templates || [];
  const el = document.getElementById("at-template-list");
  if (!el) return;

  if (!templates.length) {
    el.innerHTML = `<div class="pf-empty">No templates available</div>`;
    return;
  }

  el.innerHTML = templates.map(t => {
    const steps = t.steps?.length || 0;
    const isCustomTemplate = t.source === "custom";
    return `
      <div class="auto-card auto-card-template" data-template-id="${_esc(t.id)}">
        <div class="auto-card-top">
          <span class="auto-dot auto-dot-on"></span>
          <div class="auto-card-info">
            <div class="auto-card-title" title="${_esc(t.name)}">${_esc(t.name)}</div>
            <div class="auto-card-meta">
              <span class="auto-chip">${_esc(t.pack || "Template")}</span>
              ${isCustomTemplate ? `<span class="auto-chip-muted">Custom</span>` : ""}
              <span class="auto-chip-muted">${steps} ${steps === 1 ? "block" : "blocks"}</span>
            </div>
          </div>
        </div>
        <div class="pf-list-item-actions">
          <button class="pf-card-btn at-template-use-btn" data-id="${_esc(t.id)}" title="Use automation template">
            <span class="pf-card-btn-icon">+</span><span class="pf-card-btn-label">Use</span>
          </button>
          ${isCustomTemplate ? `<button class="pf-card-btn pf-card-btn-danger at-template-del-btn" data-id="${_esc(t.id)}" title="Delete template">
            <span class="pf-card-btn-label">Delete</span>
          </button>` : ""}
        </div>
      </div>`;
  }).join("");

  el.querySelectorAll(".at-template-use-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const template = templates.find(t => t.id === btn.dataset.id);
      if (!template) return;
      _openEditor({
        id: null,
        name: template.name,
        description: template.description || "",
        enabled: false,
        trigger: template.trigger || { type: "manual" },
        steps: template.steps || [],
        output_destination: template.output_destination || { type: "chat" },
        drawflow_data: template.drawflow_data || null,
      });
    });
  });

  el.querySelectorAll(".at-template-del-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const template = templates.find(t => t.id === btn.dataset.id);
      if (!confirm(`Delete template "${template?.name}"?`)) return;
      const res = await fetch(`/api/automations/templates/${encodeURIComponent(btn.dataset.id)}`, { method: "DELETE" });
      if (!res.ok) {
        alert("Template delete failed.");
        return;
      }
      await _renderAutomationTemplates();
    });
  });
}

function _openEditor(automation) {
  _autoMeta = {
    id: automation?.id || null,
    name: automation?.name || "",
    source_template_id: automation?.source_template_id || "",
    source_template_name: automation?.source_template_name || "",
  };

  document.querySelector(".pf-shell")?.classList.remove("pf-mode-list");
  document.getElementById("at-list-panel").classList.add("hidden");
  document.getElementById("at-palette-panel").classList.remove("hidden");
  document.getElementById("at-canvas-wrap").classList.remove("hidden");
  document.getElementById("at-title-input").value = _autoMeta.name;
  _renderAutomationTemplateChip();

  const container = document.getElementById("at-drawflow");
  container.innerHTML = "";

  _editor = new Drawflow(container);
  _editor.reroute = true;
  _editor.start();

  _editor.on("nodeSelected",   id => { _selectedNodeId = String(id); });
  _editor.on("nodeUnselected", () => { _selectedNodeId = null; });
  _editor.on("nodeRemoved",    () => _closeConfig());

  _bindPaletteDrag(container);

  if (automation?.drawflow_data) {
    _editor.import(automation.drawflow_data);
  } else if (automation?.steps?.length) {
    _importSteps(automation);
  } else {
    _addTriggerNode();
    _addOutputNode();
  }
  _ensureErrorOutputs();
  setTimeout(_ensureNodeConfigButtons, 0);
}

function _renderAutomationTemplateChip() {
  const meta = document.querySelector("#at-canvas-wrap .pf-toolbar-meta");
  if (!meta) return;
  meta.querySelector(".pf-source-template-chip")?.remove();
  if (!_autoMeta?.source_template_name) return;
  const chip = document.createElement("span");
  chip.className = "auto-chip-muted pf-source-template-chip";
  chip.title = `Created from template ${_autoMeta.source_template_name}`;
  chip.textContent = `From ${_autoMeta.source_template_name}`;
  meta.appendChild(chip);
}

function _closeEditor() {
  _editor = null;
  _selectedNodeId = null;
  _closeConfig();
  document.querySelector(".pf-shell")?.classList.add("pf-mode-list");
  document.getElementById("at-list-panel").classList.remove("hidden");
  document.getElementById("at-palette-panel").classList.add("hidden");
  document.getElementById("at-canvas-wrap").classList.add("hidden");
  _renderAutomationList();
}

// ── Special nodes ─────────────────────────────────────────────────────────────

function _triggerNodeHtml(data) {
  const t = data.trigger_type || "daily";
  const labels = { daily: "Daily", weekly: "Weekly", hourly: "Every hour", custom: "Custom cron", on_start: "On app start", manual: "Manual" };
  const preview = t === "daily" || t === "weekly"
    ? `${labels[t]} at ${data.time_of_day || "09:00"}`
    : labels[t] || t;
  return `<div class="pf-node-inner pf-node-start">
    <div class="pf-node-header" style="background:#10b98122;border-left:3px solid #10b981">
      <span class="pf-node-icon">⏰</span>
      <span class="pf-node-label">Trigger</span>
      <button type="button" class="pf-node-config-btn" title="Configure block">⚙</button>
    </div>
    <div class="pf-node-preview">${_esc(preview)}</div>
  </div>`;
}

function _outputNodeHtml(data) {
  const types = { chat: "💬 Chat", memory: "🧠 Memory", telegram: "✈️ Telegram", email: "📧 Email", webhook: "🌐 Webhook" };
  const preview = types[data.output_type || "chat"] || "💬 Chat";
  return `<div class="pf-node-inner pf-node-result">
    <div class="pf-node-header" style="background:#6366f122;border-left:3px solid #6366f1">
      <span class="pf-node-icon">📤</span>
      <span class="pf-node-label">Output</span>
      <button type="button" class="pf-node-config-btn" title="Configure block">⚙</button>
    </div>
    <div class="pf-node-preview">${_esc(preview)}</div>
  </div>`;
}

function _addTriggerNode(data = {}) {
  const d = { trigger_type: "daily", time_of_day: "09:00", day_of_week: 0, cron: "", label: "Trigger", ...data };
  return _editor.addNode("trigger", 0, 1, 80, 200, "pf-start", d, _triggerNodeHtml(d), false);
}

function _addOutputNode(data = {}) {
  const d = { output_type: "chat", telegram_chat_id: "", email_to: "", email_subject: "", webhook_url: "", memory_tags: "automation", label: "Output", ...data };
  return _editor.addNode("output", 1, 0, 720, 200, "pf-result", d, _outputNodeHtml(d), false);
}

function _blockOutputCount(type, data = {}) {
  if (type === "router") return _routerRoutes(data).length;
  if (["trigger", "output", "error_handler"].includes(type)) return _block(type).outputs || 1;
  return Math.max(2, _block(type).outputs || 1);
}

function _ensureErrorOutputs() {
  if (!_editor) return;
  const nodes = _editor.export()?.drawflow?.Home?.data || {};
  Object.values(nodes).forEach(node => {
    if (!node || ["trigger", "output", "router", "error_handler"].includes(node.name)) return;
    if (Object.keys(node.outputs || {}).length < 2) _editor.addNodeOutput(node.id);
  });
}

function _addBlock(type, x, y, extraData = {}) {
  const b    = _block(type);
  const data = { ...b.defaults, label: b.label, ...extraData };
  const outputs = _blockOutputCount(type, data);
  const inputs = type === "merge" ? _mergeInputCount(data) : b.inputs;
  return _editor.addNode(type, inputs, outputs, x, y, `pf-block pf-block-${type}`, data, _nodeHtml(type, data), false);
}

function _nodeHtml(type, data) {
  const b = _block(type);
  return `<div class="pf-node-inner" data-type="${type}">
    <div class="pf-node-header" style="background:${b.color}22;border-left:3px solid ${b.color}">
      <span class="pf-node-icon">${b.icon}</span>
      <span class="pf-node-label">${_esc(data.label || b.label)}</span>
      <button type="button" class="pf-node-config-btn" title="Configure block">⚙</button>
    </div>
    <div class="pf-node-preview">${_nodePreview(type, data)}</div>
  </div>`;
}

function _nodePreview(type, data) {
  if (data?._pinned_enabled) return _esc(`Pinned · ${(data._pinned_output || "").slice(0, 42)}`);
  if (type === "llm_call")      return _esc((data.model || "AI") + (data.system_prompt ? " · " + data.system_prompt.slice(0, 30) : ""));
  if (type === "web_search")    return `Results: ${data.num_results || 5}`;
  if (type === "page_scrape")   return _esc(`${data.mode || "text"} ${(data.url || "").slice(0, 28)}`.trim());
  if (type === "memory_search") return `Results: ${data.limit || 5}${data.skip_empty_result ? " · skip empty" : ""}`;
  if (type === "memory_save")   return `Tags: ${data.tags || "automation"}${(data.dedup_guard || data.deduplicate) ? " · dedup" : ""}`;
  if (type === "merge")         return _esc(`Inputs: ${_mergeInputCount(data)} · ${data.mode || "combine_text"}`);
  if (type === "router")        return _esc(`Routes: ${_routerRoutes(data).length} · ${_routerModeLabel(data.mode)}`);
  if (type === "delay")         return `Wait ${data.seconds || 1}s`;
  if (type === "filter")        return _esc(`${data.operator || "contains"} ${data.value || ""}`.trim());
  if (type === "set_variable")  return _esc(`{{${data.name || "value"}}}`);
  if (type === "parse_json")    return _esc(data.path || "$");
  if (type === "transform")     return _esc(data.mode || "trim");
  if (type === "notification")  return _esc(data.title || "Notification");
  if (type === "file_read")     return _esc(data.filename || "workflow.txt");
  if (type === "file_write")    return _esc(data.filename || "workflow.txt");
  if (type === "code")          return (data.mode || "safe_expression") === "local_script" ? `${data.language || "python"} local script` : "Safe expression";
  if (type === "error_handler") return data.fallback_text ? "Fallback configured" : "Error recovery";
  if (type === "http_request")  return `${data.method || "GET"} ${_esc((data.url || "").slice(0, 28))}`;
  if (type === "template")      return _esc((data.text || "").slice(0, 50));
  return "";
}

// ── Import from saved automation ──────────────────────────────────────────────

function _ensureNodeConfigButtons() {
  document.querySelectorAll("#at-drawflow .drawflow-node").forEach(nodeEl => {
    const header = nodeEl.querySelector(".pf-node-header");
    if (!header || header.querySelector(".pf-node-config-btn")) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pf-node-config-btn";
    btn.title = "Configure block";
    btn.textContent = "...";
    header.appendChild(btn);
  });
}

function _importSteps(automation) {
  const trigData = automation.trigger || {};
  const trigNode = {
    trigger_type: trigData.schedule_type || trigData.type || "daily",
    time_of_day:  trigData.time_of_day  || "09:00",
    day_of_week:  trigData.day_of_week  || 0,
    cron:         trigData.cron         || "",
    catch_up:     trigData.catch_up !== false,
  };
  const triggerId = _addTriggerNode(trigNode);

  const dest      = automation.output_destination || {};
  const outData   = {
    output_type:       dest.type         || "chat",
    telegram_chat_id:  dest.chat_id      || "",
    email_to:          dest.to           || "",
    email_subject:     dest.subject      || "",
    webhook_url:       dest.url          || "",
    memory_tags:       dest.tags         || "automation",
  };

  const steps    = automation.steps || [];
  const colSpacing = 260;
  let prevId   = triggerId;
  let prevOut  = "output_1";

  steps.forEach((step, i) => {
    const x      = 80 + (i + 1) * colSpacing;
    const nodeId = _addBlock(step.type, x, 200, step.config || {});
    _editor.addConnection(prevId, nodeId, prevOut, "input_1");
    prevId  = nodeId;
    prevOut = "output_1";
  });

  const outputId = _addOutputNode(outData);
  _editor.addConnection(prevId, outputId, prevOut, "input_1");
}

// ── Palette drag ──────────────────────────────────────────────────────────────

function _bindPaletteDrag(container) {
  let _dragType = null;

  document.querySelectorAll("#at-palette-panel .pf-palette-item").forEach(item => {
    item.addEventListener("dragstart", e => {
      _dragType = item.dataset.type;
      e.dataTransfer.effectAllowed = "copy";
    });
  });

  container.addEventListener("dragover", e => e.preventDefault());
  container.addEventListener("drop", e => {
    e.preventDefault();
    if (!_dragType || !_editor) return;
    const rect = container.getBoundingClientRect();
    const zoom = _editor.zoom;
    const x    = (e.clientX - rect.left - _editor.canvas_x) / zoom;
    const y    = (e.clientY - rect.top  - _editor.canvas_y) / zoom;
    const id   = _addBlock(_dragType, x - 100, y - 40);
    _dragType  = null;
    setTimeout(() => _openNodeConfig(String(id)), 50);
  });
}

// ── Config panel ──────────────────────────────────────────────────────────────

function _openNodeConfig(id) {
  _selectedNodeId = id;
  const node = _editor?.getNodeFromId(id);
  if (!node) return;

  const panel = document.getElementById("at-config-panel");
  const title = document.getElementById("at-config-title");
  const body  = document.getElementById("at-config-body");

  title.innerHTML  = node.name === "trigger" ? "⏰ Trigger" : node.name === "output" ? "📤 Output" : `${_block(node.name).icon} ${_block(node.name).label}`;
  body.innerHTML   = _buildConfigForm(node.name, node.data, id);
  const deleteButton = body.querySelector(".pf-del-node-btn");
  if (deleteButton) {
    deleteButton.insertAdjacentHTML("afterend", _blockHelpHtml(node.name));
  } else {
    body.insertAdjacentHTML("beforeend", _blockHelpHtml(node.name));
  }
  panel.classList.remove("hidden");

  const deleteNodeButton = body.querySelector(".pf-del-node-btn");
  deleteNodeButton?.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    ev.stopPropagation();
  });
  deleteNodeButton?.addEventListener("click", ev => {
    ev.preventDefault();
    ev.stopPropagation();
    _deleteNodeFromCanvas(id);
  });

  body.querySelectorAll("input,textarea,select").forEach(el => {
    el.addEventListener("input",  () => _saveNodeConfig(id));
    el.addEventListener("change", () => _saveNodeConfig(id));
  });
  _bindOutputReadinessActions(body);
  body.querySelector('[data-key="auth_secret_key"]')?.addEventListener("change", () => _refreshHttpAuthReadiness(body));
  body.querySelector("[data-http-test]")?.addEventListener("click", () => _testHttpConnection(body));
  body.querySelector("[data-scrape-test]")?.addEventListener("click", () => _testPageScrape(body));
  body.querySelector(".pf-router-add-route")?.addEventListener("click", () => {
    const node = _editor?.getNodeFromId(id);
    if (!node) return;
    const routes = _routerRoutes(node.data);
    routes.push({ label: `Route ${routes.length + 1}`, operator: "contains", value: "" });
    _editor.updateNodeDataFromId(id, { ...node.data, routes });
    _syncRouterOutputs(id, routes.length);
    _openNodeConfig(id);
  });
  body.querySelectorAll(".pf-router-remove-route").forEach(btn => {
    btn.addEventListener("click", () => {
      const node = _editor?.getNodeFromId(id);
      if (!node) return;
      const routes = _routerRoutes(node.data);
      if (routes.length <= 1) return;
      routes.splice(Number(btn.dataset.routeIndex), 1);
      _editor.updateNodeDataFromId(id, { ...node.data, routes });
      _syncRouterOutputs(id, routes.length);
      _openNodeConfig(id);
    });
  });
}

function _canvasNodeExists(id) {
  const nodeId = String(id || "").replace(/^node-/, "");
  return !!_editor?.drawflow?.drawflow?.[_editor.module || "Home"]?.data?.[nodeId];
}

function _deleteNodeFromCanvas(id) {
  if (!_editor) return false;
  const nodeId = String(id || "").replace(/^node-/, "");
  if (!nodeId || !_canvasNodeExists(nodeId)) {
    _closeConfig();
    return false;
  }

  try {
    _editor.removeNodeId(`node-${nodeId}`);
  } catch (_err) {
    // Some Drawflow builds are picky about the id shape; fall back below.
  }

  if (_canvasNodeExists(nodeId)) {
    try {
      _editor.removeNodeId(nodeId);
    } catch (_err) {
      const moduleName = _editor.module || "Home";
      try { _editor.removeConnectionNodeId?.(`node-${nodeId}`); } catch (_connErr) {}
      document.getElementById(`node-${nodeId}`)?.remove();
      delete _editor.drawflow.drawflow[moduleName].data[nodeId];
      _editor.dispatch?.("nodeRemoved", nodeId);
    }
  }

  _closeConfig();
  return !_canvasNodeExists(nodeId);
}

function _secretConfigured(key) {
  return _availableSecrets.some(secret => secret.key === key && secret.configured);
}

function _outputReadinessHtml(type) {
  let ready = true;
  let label = "";
  let detail = "";
  if (type === "telegram") {
    ready = _secretConfigured("telegram_bot_token");
    label = ready ? "Telegram Ready" : "Telegram Not set";
    detail = ready
      ? "Telegram bot token is saved locally."
      : "Save a Telegram bot token in Local Secrets before running.";
  } else if (type === "email") {
    const hasAccount = !!(_settingsSnapshot?.smtp_user || "").trim();
    const hasPassword = _secretConfigured("smtp_password");
    ready = hasAccount && hasPassword;
    label = ready ? "Email Ready" : "Email Not set";
    if (!hasAccount && !hasPassword) detail = "Save Gmail address and Email app password before running.";
    else if (!hasAccount) detail = "Save Gmail address before running.";
    else detail = "Save Email app password in Local Secrets before running.";
  }
  const cls = ready ? "ready" : "missing";
  return `
    <div class="pf-output-readiness pf-output-readiness-${cls}">
      <div>
        <strong>${_esc(label)}</strong>
        <span>${_esc(detail)}</span>
      </div>
      ${ready ? "" : `<button type="button" class="btn btn-outline btn-sm at-open-settings">Open Settings</button>`}
    </div>`;
}

function _bindOutputReadinessActions(root) {
  root.querySelectorAll(".at-open-settings").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelector('.nav-item[data-view="settings"]')?.click();
    });
  });
}

function _httpAuthReadinessHtml(secretKey) {
  const key = String(secretKey || "").trim();
  if (!key) {
    return `
      <div class="pf-output-readiness" data-http-auth-readiness>
        <div>
          <strong>No Authorization header</strong>
          <span>This request will run without a vault secret.</span>
        </div>
      </div>`;
  }
  const secret = _availableSecrets.find(item => item.key === key);
  const ready = !!secret;
  const cls = ready ? "ready" : "missing";
  return `
    <div class="pf-output-readiness pf-output-readiness-${cls}" data-http-auth-readiness>
      <div>
        <strong>${ready ? "Authorization Ready" : "Authorization Not set"}</strong>
        <span>${ready ? `${_esc(secret.label)} is saved locally.` : "Save the selected secret in Local Secrets before running."}</span>
      </div>
      ${ready ? "" : `<button type="button" class="btn btn-outline btn-sm at-open-settings">Open Settings</button>`}
    </div>`;
}

function _refreshHttpAuthReadiness(root) {
  const slot = root.querySelector("[data-http-auth-readiness]");
  const select = root.querySelector('[data-key="auth_secret_key"]');
  if (!slot || !select) return;
  slot.outerHTML = _httpAuthReadinessHtml(select.value);
  _bindOutputReadinessActions(root);
}

function _collectHttpTestPayload(root) {
  const payload = {};
  root.querySelectorAll("[data-key]").forEach(el => {
    payload[el.dataset.key] = el.type === "checkbox" ? el.checked : el.value;
  });
  return {
    url: payload.url || "",
    method: payload.method || "GET",
    body: payload.body || "",
    auth_secret_key: payload.auth_secret_key || "",
    auth_scheme: payload.auth_scheme || "Bearer",
  };
}

function _setHttpTestStatus(root, cls, text) {
  const status = root.querySelector("[data-http-test-status]");
  if (!status) return;
  status.className = `pf-http-test-status ${cls || ""}`.trim();
  status.textContent = text;
}

async function _testHttpConnection(root) {
  const btn = root.querySelector("[data-http-test]");
  btn?.setAttribute("disabled", "disabled");
  _setHttpTestStatus(root, "", "Testing...");
  try {
    const res = await fetch("/api/workflows/http-test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_collectHttpTestPayload(root)),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      _setHttpTestStatus(root, "error", data.detail || "Connection test failed.");
    } else if (data.ok) {
      _setHttpTestStatus(root, "ok", `Connected: HTTP ${data.status_code} in ${data.duration_ms} ms`);
    } else {
      _setHttpTestStatus(root, "error", data.error || `HTTP ${data.status_code || "error"}`);
    }
  } catch (err) {
    _setHttpTestStatus(root, "error", err?.message || "Connection test failed.");
  } finally {
    btn?.removeAttribute("disabled");
  }
}

function _collectScrapeTestPayload(root) {
  const payload = {};
  root.querySelectorAll("[data-key]").forEach(el => {
    payload[el.dataset.key] = el.type === "checkbox" ? el.checked : el.value;
  });
  return {
    url: payload.url || "",
    mode: payload.mode || "text",
    max_chars: Number(payload.max_chars || 12000),
  };
}

async function _testPageScrape(root) {
  const btn = root.querySelector("[data-scrape-test]");
  btn?.setAttribute("disabled", "disabled");
  _setHttpTestStatus(root, "", "Testing...");
  try {
    const res = await fetch("/api/workflows/scrape-test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_collectScrapeTestPayload(root)),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      _setHttpTestStatus(root, "error", data.detail || "Scrape test failed.");
    } else if (data.ok) {
      const preview = String(data.preview || "").replace(/\s+/g, " ").trim().slice(0, 140);
      _setHttpTestStatus(root, "ok", preview ? `Preview: ${preview}` : `Scraped in ${data.duration_ms} ms`);
    } else {
      _setHttpTestStatus(root, "error", data.error || "Scrape test failed.");
    }
  } catch (err) {
    _setHttpTestStatus(root, "error", err?.message || "Scrape test failed.");
  } finally {
    btn?.removeAttribute("disabled");
  }
}

function _buildConfigForm(type, data, nodeId) {
  // ── Trigger node ──
  if (type === "trigger") {
    const t = data.trigger_type || "daily";
    return `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Trigger type</label>
        <select class="pf-cfg-select" data-key="trigger_type" id="at-trig-type">
          <option value="daily"    ${t==="daily"    ?"selected":""}>Daily</option>
          <option value="weekly"   ${t==="weekly"   ?"selected":""}>Weekly</option>
          <option value="hourly"   ${t==="hourly"   ?"selected":""}>Every hour</option>
          <option value="custom"   ${t==="custom"   ?"selected":""}>Custom (cron)</option>
          <option value="on_start" ${t==="on_start" ?"selected":""}>On app start</option>
          <option value="manual"   ${t==="manual"   ?"selected":""}>Manual only</option>
        </select>
      </div>
      <div id="at-trig-time-row" class="pf-cfg-row" style="${t==="custom"||t==="on_start"||t==="manual"?"display:none":""}">
        <label class="pf-cfg-label">Time</label>
        <input class="pf-cfg-input" data-key="time_of_day" type="time" value="${_esc(data.time_of_day||"09:00")}" style="width:120px">
      </div>
      <div id="at-trig-dow-row" class="pf-cfg-row" style="${t!=="weekly"?"display:none":""}">
        <label class="pf-cfg-label">Day of week</label>
        <select class="pf-cfg-select" data-key="day_of_week">
          ${["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            .map((d,i)=>`<option value="${i}" ${(data.day_of_week||0)==i?"selected":""}>${d}</option>`).join("")}
        </select>
      </div>
      <div id="at-trig-cron-row" class="pf-cfg-row" style="${t!=="custom"?"display:none":""}">
        <label class="pf-cfg-label">Cron expression</label>
        <input class="pf-cfg-input" data-key="cron" value="${_esc(data.cron||"")}" placeholder="0 9 * * 1-5">
        <div style="font-size:.72rem;color:var(--text-muted);margin-top:3px">minute hour day month weekday</div>
      </div>
      <div id="at-trig-catchup-row" class="pf-cfg-row" style="${t==="custom"||t==="on_start"||t==="manual"?"display:none":""}">
        <label class="pf-cfg-label">Catch up missed run</label>
        <label style="display:flex;align-items:center;gap:.45rem;font-size:.85rem;color:var(--text-secondary)">
          <input type="checkbox" data-key="catch_up" ${data.catch_up===false ? "" : "checked"}>
          Run once after app start if scheduled time was missed
        </label>
      </div>`;
  }

  // ── Output node ──
  if (type === "output") {
    const ot = data.output_type || "chat";
    return `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Send result to</label>
        <select class="pf-cfg-select" data-key="output_type" id="at-out-type">
          <option value="chat"     ${ot==="chat"    ?"selected":""}>💬 Chat (read when you open app)</option>
          <option value="memory"   ${ot==="memory"  ?"selected":""}>🧠 Memory</option>
          <option value="telegram" ${ot==="telegram"?"selected":""}>✈️ Telegram</option>
          <option value="email"    ${ot==="email"   ?"selected":""}>📧 Email</option>
          <option value="webhook"  ${ot==="webhook" ?"selected":""}>🌐 Webhook</option>
        </select>
      </div>
      <div id="at-out-tg" class="pf-cfg-row" style="${ot!=="telegram"?"display:none":""}">
        ${_outputReadinessHtml("telegram")}
        <label class="pf-cfg-label">Telegram Chat ID</label>
        <input class="pf-cfg-input" data-key="telegram_chat_id" value="${_esc(data.telegram_chat_id||"")}" placeholder="-100123456789">
        <p style="font-size:11px;color:var(--text-muted);margin:4px 0 0">
          Bot token is configured in Settings → Integrations.
          Get your Chat ID by messaging <code>@userinfobot</code> on Telegram.
        </p>
      </div>
      <div id="at-out-email-to" class="pf-cfg-row" style="${ot!=="email"?"display:none":""}">
        ${_outputReadinessHtml("email")}
        <label class="pf-cfg-label">Send to email</label>
        <input class="pf-cfg-input" data-key="email_to" value="${_esc(data.email_to||"")}" placeholder="recipient@example.com">
      </div>
      <div id="at-out-email-subj" class="pf-cfg-row" style="${ot!=="email"?"display:none":""}">
        <label class="pf-cfg-label">Subject</label>
        <input class="pf-cfg-input" data-key="email_subject" value="${_esc(data.email_subject||"Automation result")}">
        <p style="font-size:11px;color:var(--text-muted);margin:4px 0 0">
          Gmail credentials (App Password) are configured in Settings → Integrations.
        </p>
      </div>
      <div id="at-out-webhook" class="pf-cfg-row" style="${ot!=="webhook"?"display:none":""}">
        <label class="pf-cfg-label">Webhook URL</label>
        <input class="pf-cfg-input" data-key="webhook_url" value="${_esc(data.webhook_url||"")}" placeholder="https://…">
      </div>
      <div id="at-out-mem-tags" class="pf-cfg-row" style="${ot!=="memory"?"display:none":""}">
        <label class="pf-cfg-label">Memory tags</label>
        <input class="pf-cfg-input" data-key="memory_tags" value="${_esc(data.memory_tags||"automation")}">
      </div>`;
  }

  // ── Regular blocks ──
  let html = `
    <div class="pf-cfg-row">
      <label class="pf-cfg-label">Block name</label>
      <input class="pf-cfg-input" data-key="label" value="${_esc(data.label||"")}">
    </div>
    <div class="pf-cfg-row">
      <label class="pf-cfg-label">Retry on temporary failure</label>
      <label style="display:flex;align-items:center;gap:.45rem;font-size:.85rem;color:var(--text-secondary)">
        <input type="checkbox" data-key="_retry_enabled" ${data._retry_enabled ? "checked" : ""}>
        Retry this block before failing the automation
      </label>
    </div>
    <div class="pf-cfg-row">
      <label class="pf-cfg-label">Retry attempts</label>
      <input class="pf-cfg-input" type="number" min="0" max="5" data-key="_retry_count" value="${data._retry_count || 0}" style="width:90px">
    </div>
    <div class="pf-cfg-row">
      <label class="pf-cfg-label">Retry delay <span style="opacity:.6">(ms)</span></label>
      <input class="pf-cfg-input" type="number" min="0" max="30000" step="500" data-key="_retry_delay_ms" value="${data._retry_delay_ms || 0}" style="width:110px">
    </div>`;

  if (type === "llm_call") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Model</label>
        <select class="pf-cfg-select" data-key="model">${_modelOptions(data.model||"")}</select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">System prompt <span style="opacity:.6">(optional)</span></label>
        <textarea class="pf-cfg-textarea" data-key="system_prompt" rows="4"
          placeholder="Role or instruction for AI.">${_esc(data.system_prompt||"")}</textarea>
      </div>`;
  }
  if (type === "web_search") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Search query <span style="opacity:.6">(empty = use previous step)</span></label>
        <input class="pf-cfg-input" data-key="query" value="${_esc(data.query||"")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Number of results</label>
        <input class="pf-cfg-input" type="number" min="1" max="10" data-key="num_results" value="${data.num_results||5}" style="width:80px">
      </div>`;
  }
  if (type === "page_scrape") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Page URL</label>
        <input class="pf-cfg-input" data-key="url" value="${_esc(data.url||"")}" placeholder="https://example.com/report">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Extract</label>
        <select class="pf-cfg-select" data-key="mode">
          <option value="text" ${(data.mode||"text")==="text"?"selected":""}>Readable text</option>
          <option value="title" ${data.mode==="title"?"selected":""}>Page title</option>
          <option value="links" ${data.mode==="links"?"selected":""}>Links</option>
        </select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Max characters</label>
        <input class="pf-cfg-input" type="number" min="500" max="50000" step="500" data-key="max_chars" value="${data.max_chars||12000}" style="width:110px">
      </div>
      <div class="pf-cfg-row">
        <button type="button" class="btn btn-outline btn-sm" data-scrape-test>Test Scrape</button>
        <div class="pf-http-test-status" data-http-test-status></div>
      </div>`;
  }

  if (type === "memory_search") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Query <span style="opacity:.6">(empty = use previous step)</span></label>
        <input class="pf-cfg-input" data-key="query" value="${_esc(data.query||"")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Number of results</label>
        <input class="pf-cfg-input" type="number" min="1" max="20" data-key="limit" value="${data.limit||5}" style="width:80px">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-check">
          <input type="checkbox" data-key="skip_empty_result" ${data.skip_empty_result ? "checked" : ""}>
          Hide empty results
        </label>
        <p style="font-size:11px;color:var(--text-muted);margin:4px 0 0">Prevents empty memory results from becoming context for the next block.</p>
      </div>`;
  }
  if (type === "memory_save") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Tags <span style="opacity:.6">(comma-separated)</span></label>
        <input class="pf-cfg-input" data-key="tags" value="${_esc(data.tags||"automation")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-check">
          <input type="checkbox" data-key="dedup_guard" ${(data.dedup_guard || data.deduplicate) ? "checked" : ""}>
          Skip duplicate saves
        </label>
        <p style="font-size:11px;color:var(--text-muted);margin:4px 0 0">Searches local memory before writing and skips exact duplicate content.</p>
      </div>`;
  }
  if (type === "http_request") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">URL</label>
        <input class="pf-cfg-input" data-key="url" value="${_esc(data.url||"")}" placeholder="https://…">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Method</label>
        <select class="pf-cfg-select" data-key="method">
          <option ${(data.method||"GET")==="GET"?"selected":""}>GET</option>
          <option ${data.method==="POST"?"selected":""}>POST</option>
        </select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Body <span style="opacity:.6">(for POST)</span></label>
        <textarea class="pf-cfg-textarea" data-key="body" rows="3">${_esc(data.body||"")}</textarea>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Authorization secret <span style="opacity:.6">(optional)</span></label>
        <select class="pf-cfg-select" data-key="auth_secret_key">${_secretOptions(data.auth_secret_key || "")}</select>
        ${_httpAuthReadinessHtml(data.auth_secret_key || "")}
        <p style="font-size:11px;color:var(--text-muted);margin:4px 0 0">Select a local secret. The real value is inserted only during execution.</p>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Authorization scheme</label>
        <input class="pf-cfg-input" data-key="auth_scheme" value="${_esc(data.auth_scheme || "Bearer")}" placeholder="Bearer">
      </div>
      <div class="pf-cfg-row">
        <button type="button" class="btn btn-outline btn-sm" data-http-test>Test Connection</button>
        <div class="pf-http-test-status" data-http-test-status></div>
      </div>`;
  }
  if (type === "router") {
    const routes = _routerRoutes(data);
    const mode = data.mode || "all_matching";
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Routing mode</label>
        <select class="pf-cfg-select" data-key="mode">
          <option value="all_matching" ${mode === "all_matching" ? "selected" : ""}>All matching routes</option>
          <option value="first_match" ${mode === "first_match" ? "selected" : ""}>First matching route</option>
          <option value="all" ${mode === "all" ? "selected" : ""}>Pass through all routes</option>
        </select>
      </div>
      <div class="pf-router-routes">
        ${routes.map((route, i) => `
        <div class="pf-router-route" data-route-row="${i}">
          <div class="pf-cfg-row">
            <label class="pf-cfg-label">Route ${i + 1} name</label>
            <input class="pf-cfg-input" data-route-index="${i}" data-route-key="label" value="${_esc(route.label)}">
          </div>
          <div class="pf-cfg-row">
            <label class="pf-cfg-label">Route ${i + 1} condition</label>
            <select class="pf-cfg-select" data-route-index="${i}" data-route-key="operator">
              ${_routerOperatorOptions(route.operator)}
            </select>
          </div>
          <div class="pf-cfg-row">
            <label class="pf-cfg-label">Value <span style="opacity:.6">(not used by always/never/fallback/empty checks)</span></label>
            <textarea class="pf-cfg-textarea" data-route-index="${i}" data-route-key="value" rows="2"
              placeholder="invoice, urgent, spam, regex, or natural-language AI rule">${_esc(route.value || "")}</textarea>
          </div>
          <button type="button" class="btn btn-outline btn-sm pf-router-remove-route" data-route-index="${i}" ${routes.length <= 1 ? "disabled" : ""}>Remove route</button>
        </div>
        `).join("")}
      </div>
      <button type="button" class="btn btn-outline btn-sm pf-router-add-route" style="width:100%">+ Add route</button>`;
  }
  if (type === "merge") {
    const mode = data.mode || "combine_text";
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Inputs to wait for</label>
        <input class="pf-cfg-input" type="number" min="2" max="10" data-key="input_count" value="${_mergeInputCount(data)}" style="width:90px">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Merge mode</label>
        <select class="pf-cfg-select" data-key="mode">
          <option value="combine_text" ${mode === "combine_text" ? "selected" : ""}>Combine text</option>
          <option value="first_non_empty" ${mode === "first_non_empty" ? "selected" : ""}>First non-empty</option>
          <option value="json_array" ${mode === "json_array" ? "selected" : ""}>JSON array</option>
        </select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Separator <span style="opacity:.6">(combine text only)</span></label>
        <textarea class="pf-cfg-textarea" data-key="separator" rows="3">${_esc(data.separator || "\n\n---\n\n")}</textarea>
      </div>`;
  }
  if (type === "delay") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Seconds to wait</label>
        <input class="pf-cfg-input" type="number" min="0" max="300" step="0.5" data-key="seconds" value="${data.seconds || 1}" style="width:100px">
      </div>`;
  }
  if (type === "filter") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Condition</label>
        <select class="pf-cfg-select" data-key="operator">${_routerOperatorOptions(data.operator || "contains")}</select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Value</label>
        <textarea class="pf-cfg-textarea" data-key="value" rows="2">${_esc(data.value || "")}</textarea>
      </div>`;
  }
  if (type === "set_variable") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Variable name</label>
        <input class="pf-cfg-input" data-key="name" value="${_esc(data.name || "value")}" placeholder="customer_name">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Value</label>
        <textarea class="pf-cfg-textarea" data-key="value" rows="3">${_esc(data.value || "{{prev}}")}</textarea>
      </div>`;
  }
  if (type === "parse_json") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">JSON text</label>
        <textarea class="pf-cfg-textarea" data-key="text" rows="3">${_esc(data.text || "{{prev}}")}</textarea>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Path</label>
        <input class="pf-cfg-input" data-key="path" value="${_esc(data.path || "$")}" placeholder="$.items[0].title">
      </div>`;
  }
  if (type === "transform") {
    const mode = data.mode || "trim";
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Text</label>
        <textarea class="pf-cfg-textarea" data-key="text" rows="3">${_esc(data.text || "{{prev}}")}</textarea>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Mode</label>
        <select class="pf-cfg-select" data-key="mode">
          ${["trim","lower","upper","replace","regex_replace","extract_regex","truncate","join_lines"].map(v => `<option value="${v}" ${mode===v?"selected":""}>${v}</option>`).join("")}
        </select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Pattern</label>
        <input class="pf-cfg-input" data-key="pattern" value="${_esc(data.pattern || "")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Find text</label>
        <input class="pf-cfg-input" data-key="find" value="${_esc(data.find || "")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Replace</label>
        <input class="pf-cfg-input" data-key="replace" value="${_esc(data.replace || "")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Limit / separator</label>
        <input class="pf-cfg-input" data-key="limit" value="${_esc(data.limit || 1000)}" style="width:120px">
        <input class="pf-cfg-input" data-key="separator" value="${_esc(data.separator || " ")}" style="margin-top:.35rem">
      </div>`;
  }
  if (type === "notification") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Title</label>
        <input class="pf-cfg-input" data-key="title" value="${_esc(data.title || "Workflow notification")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Message</label>
        <textarea class="pf-cfg-textarea" data-key="message" rows="3">${_esc(data.message || "{{prev}}")}</textarea>
      </div>`;
  }
  if (type === "file_read") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">File name <span style="opacity:.6">(inside app workflow_files)</span></label>
        <input class="pf-cfg-input" data-key="filename" value="${_esc(data.filename || "workflow.txt")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Max characters</label>
        <input class="pf-cfg-input" type="number" min="1" max="100000" data-key="max_chars" value="${data.max_chars || 20000}" style="width:120px">
      </div>`;
  }
  if (type === "file_write") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">File name <span style="opacity:.6">(inside app workflow_files)</span></label>
        <input class="pf-cfg-input" data-key="filename" value="${_esc(data.filename || "workflow.txt")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Text</label>
        <textarea class="pf-cfg-textarea" data-key="text" rows="3">${_esc(data.text || "{{prev}}")}</textarea>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Mode</label>
        <select class="pf-cfg-select" data-key="mode">
          <option value="overwrite" ${(data.mode || "overwrite")==="overwrite"?"selected":""}>Overwrite</option>
          <option value="append" ${data.mode==="append"?"selected":""}>Append</option>
        </select>
      </div>`;
  }
  if (type === "code") {
    const mode = data.mode || "safe_expression";
    const language = data.language || "python";
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Execution mode</label>
        <select class="pf-cfg-select" data-key="mode">
          <option value="safe_expression" ${mode === "safe_expression" ? "selected" : ""}>Safe Python expression</option>
          <option value="local_script" ${mode === "local_script" ? "selected" : ""}>Local script on this computer</option>
        </select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Language</label>
        <select class="pf-cfg-select" data-key="language">
          <option value="python" ${language === "python" ? "selected" : ""}>Python</option>
          <option value="javascript" ${language === "javascript" ? "selected" : ""}>JavaScript / Node.js</option>
        </select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Code</label>
        <textarea class="pf-cfg-textarea" data-key="code" rows="7"
          placeholder="Safe expression: input.strip()
Local Python: result = input.upper()
Local JS: result = input.toUpperCase();">${_esc(data.code || "input.strip()")}</textarea>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">
          <input type="checkbox" data-key="allow_local_execution" ${data.allow_local_execution ? "checked" : ""}>
          Allow local script execution for this block
        </label>
        <p style="font-size:11px;color:var(--text-muted);margin:4px 0 0">Required only for Local Script. Safe Expression runs without system access.</p>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Timeout / max output</label>
        <input class="pf-cfg-input" type="number" min="0.1" max="30" step="0.5" data-key="timeout_seconds" value="${data.timeout_seconds || 3}" style="width:110px">
        <input class="pf-cfg-input" type="number" min="1" max="50000" data-key="max_output_chars" value="${data.max_output_chars || 12000}" style="width:140px;margin-top:.35rem">
      </div>`;
  }
  if (type === "error_handler") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Fallback text</label>
        <textarea class="pf-cfg-textarea" data-key="fallback_text" rows="4"
          placeholder="Handled error: {{error}}">${_esc(data.fallback_text || "Handled error: {{error}}")}</textarea>
      </div>`;
  }
  if (type === "template") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Text</label>
        <textarea class="pf-cfg-textarea" data-key="text" rows="5"
          placeholder="Static text. Use {{prev}} for previous step output.">${_esc(data.text||"")}</textarea>
      </div>`;
  }

  html += `<button class="btn btn-danger btn-sm pf-del-node-btn" style="margin-top:1rem;width:100%">🗑 Delete block</button>`;
  return html;
}

function _saveNodeConfig(id) {
  if (!_editor) return;
  const body = document.getElementById("at-config-body");
  if (!body) return;
  const node = _editor.getNodeFromId(id);
  if (!node) return;

  const newData = { ...node.data };
  body.querySelectorAll("[data-key]").forEach(el => {
    newData[el.dataset.key] = el.type === "checkbox" ? el.checked : el.value;
  });
  if (node.name === "router") {
    const routes = [];
    body.querySelectorAll("[data-route-key='label']").forEach(labelEl => {
      const idx = Number(labelEl.dataset.routeIndex);
      const operatorEl = body.querySelector(`[data-route-index="${idx}"][data-route-key="operator"]`);
      const valueEl = body.querySelector(`[data-route-index="${idx}"][data-route-key="value"]`);
      routes[idx] = { label: labelEl.value || `Route ${idx + 1}`, operator: operatorEl?.value || "contains", value: valueEl?.value || "" };
    });
    newData.routes = routes.filter(Boolean);
    for (let i = 1; i <= 24; i++) {
      delete newData[`route_${i}_label`];
      delete newData[`route_${i}_condition`];
    }
    _syncRouterOutputs(id, newData.routes.length);
  }
  if (node.name === "merge") {
    newData.input_count = _mergeInputCount(newData);
    _syncMergeInputs(id, newData.input_count);
  }
  _editor.updateNodeDataFromId(id, newData);

  // Live-update node preview text and dependent field visibility
  if (node.name === "trigger") {
    const t       = newData.trigger_type || "daily";
    const timeRow = body.querySelector("#at-trig-time-row");
    const dowRow  = body.querySelector("#at-trig-dow-row");
    const cronRow = body.querySelector("#at-trig-cron-row");
    const catchupRow = body.querySelector("#at-trig-catchup-row");
    if (timeRow) timeRow.style.display = (t==="custom"||t==="on_start"||t==="manual") ? "none" : "";
    if (dowRow)  dowRow.style.display  = t==="weekly"  ? "" : "none";
    if (cronRow) cronRow.style.display = t==="custom"  ? "" : "none";
    if (catchupRow) catchupRow.style.display = (t==="custom"||t==="on_start"||t==="manual") ? "none" : "";
    const previewEl = document.querySelector(`#node-${id} .pf-node-preview`);
    if (previewEl) previewEl.innerHTML = _esc(_triggerPreview(newData));
    // Re-render node HTML for output node
    const nodeEl = document.querySelector(`#node-${id} .pf-node-inner`);
    if (nodeEl) nodeEl.outerHTML = _triggerNodeHtml(newData);
  } else if (node.name === "output") {
    const ot = newData.output_type || "chat";
    body.querySelector("#at-out-tg")?.style.setProperty("display", ot==="telegram"?"":"none");
    body.querySelector("#at-out-email-to")?.style.setProperty("display", ot==="email"?"":"none");
    body.querySelector("#at-out-email-subj")?.style.setProperty("display", ot==="email"?"":"none");
    body.querySelector("#at-out-webhook")?.style.setProperty("display", ot==="webhook"?"":"none");
    body.querySelector("#at-out-mem-tags")?.style.setProperty("display", ot==="memory"?"":"none");
    const previewEl = document.querySelector(`#node-${id} .pf-node-preview`);
    if (previewEl) {
      const labels = { chat:"💬 Chat", memory:"🧠 Memory", telegram:"✈️ Telegram", email:"📧 Email", webhook:"🌐 Webhook" };
      previewEl.textContent = labels[ot] || ot;
    }
  } else {
    const previewEl = document.querySelector(`#node-${id} .pf-node-preview`);
    if (previewEl) previewEl.textContent = _nodePreview(node.name, newData);
    const labelEl = document.querySelector(`#node-${id} .pf-node-label`);
    if (labelEl) labelEl.textContent = newData.label || _block(node.name).label;
  }
}

function _closeConfig() {
  _selectedNodeId = null;
  document.getElementById("at-config-panel")?.classList.add("hidden");
}

// ── Save ──────────────────────────────────────────────────────────────────────

function _bindConfigResize() {
  const panel = document.getElementById("at-config-panel");
  const handle = document.getElementById("at-config-resize");
  if (!panel || !handle) return;

  let startX = 0;
  let startWidth = 0;

  const stop = () => {
    document.body.classList.remove("pf-resizing-config");
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
    window.removeEventListener("pointercancel", stop);
  };

  const move = (ev) => {
    const delta = startX - ev.clientX;
    const next = Math.max(320, Math.min(720, startWidth + delta));
    panel.style.width = `${next}px`;
  };

  handle.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    startX = ev.clientX;
    startWidth = panel.getBoundingClientRect().width;
    document.body.classList.add("pf-resizing-config");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  });
}

function _automationPayloadFromCanvas(nameOverride = null) {
  if (!_editor || !_autoMeta) return null;
  const name = (nameOverride || document.getElementById("at-title-input")?.value || "").trim() || "Untitled";
  const flow = _editor.export();
  const nodes = Object.values(flow.drawflow.Home.data);
  const trigNode = nodes.find(n => n.name === "trigger");
  const outNode = nodes.find(n => n.name === "output");
  const trigger = _buildTrigger(trigNode?.data || {});
  const output_destination = _buildOutput(outNode?.data || {});
  const steps = _orderedActionNodes(nodes)
    .map(n => ({
      id: `s${n.id}`,
      type: n.name,
      label: n.data.label || _block(n.name).label,
      config: _resolveStepConfig(n.name, n.data),
      _inputs: n.inputs,
      _connections: n.outputs,
    }));
  return {
    name,
    description: "",
    source_template_id: _autoMeta.source_template_id || "",
    source_template_name: _autoMeta.source_template_name || "",
    trigger,
    steps,
    output_destination,
    drawflow_data: flow,
  };
}

async function _saveAutomationAsTemplate() {
  if (!_editor || !_autoMeta) return false;
  const current = document.getElementById("at-title-input")?.value?.trim() || _autoMeta.name || "Automation template";
  const name = prompt("Template name:", current);
  if (name === null) return false;
  const payload = _automationPayloadFromCanvas(name);
  if (!payload) return false;
  const res = await fetch("/api/automations/templates", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(r => r.json().then(data => ({ ok: r.ok, data }))).catch(() => null);
  if (res?.ok && res.data?.template) {
    _showToast(`Template "${res.data.template.name}" saved`);
    await _renderAutomationTemplates();
    return true;
  }
  alert(_apiError(res?.data, "Template save failed"));
  return false;
}

async function _saveAutomation() {
  if (!_editor || !_autoMeta) return false;
  const name = document.getElementById("at-title-input")?.value?.trim() || "Untitled";
  _autoMeta.name = name;

  const flow  = _editor.export();
  const nodes = Object.values(flow.drawflow.Home.data);

  // Extract trigger config
  const trigNode = nodes.find(n => n.name === "trigger");
  const trigData = trigNode?.data || {};
  const trigger  = _buildTrigger(trigData);

  // Extract output config
  const outNode = nodes.find(n => n.name === "output");
  const outData  = outNode?.data || {};
  const output_destination = _buildOutput(outData);

  // Extract steps by the visible canvas path, not by node creation order.
  const steps = _orderedActionNodes(nodes)
    .map(n => ({
      id:     `s${n.id}`,
      type:   n.name,
      label:  n.data.label || _block(n.name).label,
      config: _resolveStepConfig(n.name, n.data),
      _inputs: n.inputs,
      _connections: n.outputs,
    }));

  const payload = {
    name,
    enabled: true,
    source_template_id: _autoMeta.source_template_id || "",
    source_template_name: _autoMeta.source_template_name || "",
    trigger,
    steps,
    output_destination,
    drawflow_data: flow,
  };

  const url    = _autoMeta.id ? `/api/automations/${_autoMeta.id}` : "/api/automations";
  const method = _autoMeta.id ? "PUT" : "POST";

  try {
    const res  = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (!res.ok) throw new Error(_apiError(data, "Save failed"));
    if (!_autoMeta.id && data.automation_id) _autoMeta.id = data.automation_id;
    // Flash save button
    const btn = document.getElementById("at-save-btn");
    const orig = btn.textContent;
    btn.textContent = "✓ Saved";
    setTimeout(() => { btn.textContent = orig; }, 1500);
    return true;
  } catch (err) {
    alert(`Save failed: ${err.message}`);
    return false;
  }
}

function _buildTrigger(d) {
  const type = d.trigger_type || "daily";
  const base = { type: type === "on_start" || type === "manual" ? type : "schedule" };
  if (base.type === "schedule") {
    base.schedule_type = type;
    base.time_of_day   = d.time_of_day  || "09:00";
    base.day_of_week   = parseInt(d.day_of_week || 0, 10);
    base.cron          = d.cron         || "";
    base.catch_up      = d.catch_up !== false;
  }
  return base;
}

function _buildOutput(d) {
  const type = d.output_type || "chat";
  const base = { type };
  if (type === "telegram") base.chat_id = d.telegram_chat_id || "";
  if (type === "email")    { base.to = d.email_to || ""; base.subject = d.email_subject || "Automation result"; }
  if (type === "webhook")  base.url  = d.webhook_url || "";
  if (type === "memory")   base.tags = d.memory_tags  || "automation";
  return base;
}

function _resolveStepConfig(type, data) {
  const cfg = { ...data };
  // For blocks that use previous output as implicit input — pass through as-is
  // The runner uses {{prev}} / {{input}} substitution
  if (type === "llm_call") {
    const sys = (cfg.system_prompt || "").trim();
    cfg.prompt = sys
      ? `Instruction:\n${sys}\n\nInput from previous block:\n{{prev}}`
      : "{{prev}}";
  }
  if (type === "web_search"    && !cfg.query)  cfg.query = "{{prev}}";
  if (type === "memory_search" && !cfg.query)  cfg.query = "{{prev}}";
  if (type === "memory_save"   && !cfg.text)   cfg.text  = "{{prev}}";
  if (type === "router") cfg._data_ref = "{{prev}}";
  if (type === "filter") cfg._data_ref = "{{prev}}";
  return cfg;
}

// ── Run now ───────────────────────────────────────────────────────────────────

function _orderedActionNodes(nodes) {
  const byId = new Map(nodes.map(n => [String(n.id), n]));
  const trigger = nodes.find(n => n.name === "trigger");
  const ordered = [];
  const seen = new Set();

  const visit = (node) => {
    if (!node || seen.has(String(node.id))) return;
    seen.add(String(node.id));
    const outputs = node.outputs || {};
    for (const outputName of Object.keys(outputs).sort()) {
      const connections = outputs[outputName]?.connections || [];
      for (const conn of connections) {
        const next = byId.get(String(conn.node));
        if (!next || next.name === "output") continue;
        if (seen.has(String(next.id))) continue;
        if (next.name !== "trigger") ordered.push(next);
        visit(next);
      }
    }
  };

  visit(trigger);

  const fallback = nodes
    .filter(n => n.name !== "trigger" && n.name !== "output")
    .sort((a, b) => a.id - b.id);
  return ordered.length ? ordered : fallback;
}

async function _runNow() {
  const saved = await _saveAutomation();
  if (!saved || !_autoMeta?.id) {
    alert("Please save the automation first.");
    return;
  }
  if (!await _confirmAutomationPreflight(_autoMeta.id)) return;
  _showRunModal(_autoMeta.name || "Automation");
  await _runAutomationById(_autoMeta.id);
}

async function _confirmAutomationPreflight(automationId) {
  const report = await fetch(`/api/automations/${encodeURIComponent(automationId)}/preflight`)
    .then(r => r.json())
    .catch(() => null);
  return _confirmSafetyPreflight(report, "Automation safety preflight");
}

function _confirmSafetyPreflight(report, title) {
  if (!report) return confirm(`${title}\n\nCould not run safety preflight. Continue anyway?`);
  const blockers = report.blockers || [];
  const warnings = report.warnings || [];
  if (!blockers.length && !warnings.length) return true;
  const lines = [`${title}`];
  if (blockers.length) {
    lines.push("", "Blocked:");
    blockers.forEach(item => lines.push(`- ${item.message}${item.detail ? ` ${item.detail}` : ""}`));
  }
  if (warnings.length) {
    lines.push("", "Warnings:");
    warnings.forEach(item => lines.push(`- ${item.message}${item.detail ? ` ${item.detail}` : ""}`));
  }
  if (blockers.length) {
    alert(lines.join("\n"));
    return false;
  }
  lines.push("", "Continue with this run?");
  return confirm(lines.join("\n"));
}

function _showSafetyPreflight(report, title) {
  if (!report) {
    alert(`${title}\n\nCould not run safety preflight.`);
    return;
  }
  const blockers = report.blockers || [];
  const warnings = report.warnings || [];
  const lines = [`${title}`];
  if (!blockers.length && !warnings.length) {
    lines.push("", "No safety issues detected.");
  }
  if (blockers.length) {
    lines.push("", "Blocked:");
    blockers.forEach(item => lines.push(`- ${item.message}${item.detail ? ` ${item.detail}` : ""}`));
  }
  if (warnings.length) {
    lines.push("", "Warnings:");
    warnings.forEach(item => lines.push(`- ${item.message}${item.detail ? ` ${item.detail}` : ""}`));
  }
  alert(lines.join("\n"));
}

async function _runAutomationById(id) {
  const stepsEl = document.getElementById("at-run-steps");
  const finalEl = document.getElementById("at-run-final");
  _clearNodeRunBadges();
  if (stepsEl) stepsEl.classList.remove("hidden");
  if (stepsEl) stepsEl.innerHTML = `<div class="pf-run-step pf-run-step-running">⏳ Running…</div>`;
  if (finalEl) finalEl.classList.add("hidden");

  try {
    const res  = await fetch(`/api/automations/${id}/run`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      const detail = data.detail || {};
      _applyRunTraceToCanvas(detail.trace || [], detail);
      if (stepsEl) stepsEl.innerHTML = "";
      throw new Error(_apiError(data, "Run failed"));
    }

    _applyRunTraceToCanvas(data.trace || [], data);
    if (stepsEl) stepsEl.classList.add("hidden");
    const textEl = document.getElementById("at-run-final-text");
    if (textEl) textEl.textContent = `${data.output || "(no output)"}${data.output_truncated ? "\n\n[Output truncated in UI response]" : ""}`;
    if (finalEl) finalEl.classList.remove("hidden");
    _renderAutomationList();
  } catch (err) {
    if (stepsEl) stepsEl.innerHTML += `<div class="pf-run-step pf-run-step-error">${_esc(err.message)}</div>`;
  }
}

function _showRunModal(title) {
  document.getElementById("at-run-title").textContent = title;
  document.getElementById("at-run-modal").classList.remove("hidden");
  document.getElementById("at-run-final").classList.add("hidden");
  const stepsEl = document.getElementById("at-run-steps");
  if (stepsEl) {
    stepsEl.classList.remove("hidden");
    stepsEl.innerHTML = "";
  }
}

// ── Models ────────────────────────────────────────────────────────────────────

function _clearNodeRunBadges() {
  _lastRunTraceByStepId = new Map();
  document.querySelectorAll(".pf-node-run-badge").forEach(el => el.remove());
}

function _applyRunTraceToCanvas(trace, runData = {}) {
  _clearNodeRunBadges();
  const steps = Array.isArray(trace) ? [...trace] : [];
  const outputStep = _outputTraceFromRun(runData, steps);
  if (outputStep) steps.push(outputStep);

  for (const step of steps) {
    if (!step?.id) continue;
    _lastRunTraceByStepId.set(String(step.id), step);
    const nodeEl = _nodeElementForStep(step);
    const inner = nodeEl?.querySelector?.(".pf-node-inner");
    if (!inner || inner.querySelector(".pf-node-run-badge")) continue;

    const ok = step.status === "ok";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `pf-node-run-badge ${ok ? "pf-node-run-ok" : "pf-node-run-error"}`;
    btn.dataset.stepId = String(step.id);
    btn.title = ok ? "View block result" : "View block error";
    btn.textContent = ok ? "✓" : "!";
    inner.appendChild(btn);
  }
}

function _outputTraceFromRun(runData, steps) {
  const outputNode = _findOutputNode();
  if (!outputNode) return null;
  if (steps.some(step => String(step?.id) === "output")) return null;
  const failed = runData?.status === "failed" || runData?.ok === false;
  return {
    id: "output",
    index: (steps.length || 0) + 1,
    type: "output",
    label: "Output",
    status: failed ? "error" : "ok",
    output: failed ? "" : (runData?.output || "(no output)"),
    error: failed ? (runData?.error || runData?.message || "Delivery failed") : "",
    output_length: runData?.output_length || 0,
    output_truncated: !!runData?.output_truncated,
  };
}

function _nodeElementForStep(step) {
  const id = String(step.id || "");
  if (id === "output") {
    const outputNode = _findOutputNode();
    return outputNode ? document.getElementById(`node-${outputNode.id}`) : null;
  }
  const drawflowId = id.startsWith("s") ? id.slice(1) : id;
  return drawflowId ? document.getElementById(`node-${drawflowId}`) : null;
}

function _findOutputNode() {
  try {
    const nodes = Object.values(_editor?.export?.()?.drawflow?.Home?.data || {});
    return nodes.find(n => n.name === "output") || null;
  } catch (_) {
    return null;
  }
}

function _openStepResult(stepId) {
  const step = _lastRunTraceByStepId.get(String(stepId || ""));
  if (!step) return;
  _showStepResultModal(step);
}

function _showStepResultModal(step) {
  const ok = step.status === "ok";
  const body = ok ? (step.output || "(no output)") : (step.error || "Step failed");
  const suffix = step.output_truncated
    ? `\n\n[Step output truncated: ${step.output_length} characters total]`
    : "";

  const title = document.getElementById("at-step-title");
  const meta = document.getElementById("at-step-meta");
  const output = document.getElementById("at-step-output");
  const pinBtn = document.getElementById("at-step-pin");
  if (title) title.textContent = `${ok ? "OK" : "ERROR"} ${step.index || ""}. ${step.label || step.type || "Block"}`.trim();
  if (meta) {
    meta.innerHTML = `
      <span class="pf-step-status ${ok ? "pf-step-status-ok" : "pf-step-status-error"}">${ok ? "Success" : "Error"}</span>
      <span>${_esc(step.type || "block")}</span>`;
  }
  if (output) output.textContent = body + suffix;
  if (pinBtn) {
    pinBtn.dataset.stepId = step.id || "";
    pinBtn.disabled = !ok || !String(step.id || "").startsWith("s");
    pinBtn.textContent = "Pin result";
  }
  document.getElementById("at-step-modal")?.classList.remove("hidden");
}

function _pinCurrentStepResult() {
  const btn = document.getElementById("at-step-pin");
  const stepId = btn?.dataset.stepId || "";
  const step = _lastRunTraceByStepId.get(String(stepId));
  if (!step || step.status !== "ok" || !String(step.id || "").startsWith("s")) return;
  const nodeId = String(step.id).slice(1);
  const node = _editor?.getNodeFromId(nodeId);
  if (!node) return;
  const data = {
    ...node.data,
    _pinned_enabled: true,
    _pinned_output: step.output || "",
  };
  _editor.updateNodeDataFromId(nodeId, data);
  const preview = document.querySelector(`#node-${nodeId} .pf-node-preview`);
  if (preview) preview.textContent = _nodePreview(node.name, data);
  _showToast("Pinned. Save the automation to keep it.");
}

function _hideStepResultModal() {
  document.getElementById("at-step-modal")?.classList.add("hidden");
}

function _renderRunTrace(trace) {
  if (!trace?.length) return "";
  return trace.map(step => {
    const ok = step.status === "ok";
    const body = ok ? (step.output || "(no output)") : (step.error || "Step failed");
    const suffix = step.output_truncated
      ? `\n\n[Step output truncated: ${step.output_length} characters total]`
      : "";
    return `
      <details class="pf-run-step ${ok ? "pf-run-step-done" : "pf-run-step-error"}" open>
        <summary>${ok ? "OK" : "ERROR"} ${_esc(step.index)}. ${_esc(step.label || step.type)}</summary>
        <pre class="pf-run-step-output">${_esc(body + suffix)}</pre>
      </details>`;
  }).join("");
}

function _renderMemoryEvaluation(report) {
  if (!report || typeof report !== "object" || report.score === undefined) return "";
  const recs = Array.isArray(report.recommendations) ? report.recommendations : [];
  return `
    <div class="pf-run-step">
      <strong>Memory discipline: ${_esc(String(report.score))}/100</strong>
      <div style="opacity:.72;margin-top:.35rem">
        Search ${_esc(String(report.memory_search_count || 0))} · Save ${_esc(String(report.memory_save_count || 0))} · Empty ${_esc(String(report.empty_search_count || 0))} · Duplicate candidates ${_esc(String(report.duplicate_save_candidate_count || 0))}
      </div>
      ${recs.length ? `<ul style="margin:.55rem 0 0 1.1rem">${recs.map(item => `<li>${_esc(item)}</li>`).join("")}</ul>` : ""}
    </div>`;
}

function _renderMemoryReportSummary(report) {
  if (!report || typeof report !== "object" || report.status === "no_data") return "";
  const totals = report.totals || {};
  const recs = Array.isArray(report.top_recommendations) ? report.top_recommendations : [];
  const actions = _memoryQuickFixButtons(totals);
  const trend = _renderMemoryTrend(report.trend);
  return `
    <div class="pf-run-step">
      <strong>Workflow memory: ${_esc(String(report.average_score ?? "?"))}/100</strong>
      <span style="opacity:.72;margin-left:.5rem">${_esc(report.status || "unknown")}</span>
      ${trend}
      <div style="opacity:.72;margin-top:.35rem">
        Runs ${_esc(String(report.evaluated_run_count || 0))}/${_esc(String(report.run_count || 0))} |
        Search ${_esc(String(totals.memory_search_count || 0))} |
        Save ${_esc(String(totals.memory_save_count || 0))} |
        Empty ${_esc(String(totals.empty_search_count || 0))} |
        Duplicate candidates ${_esc(String(totals.duplicate_save_candidate_count || 0))}
      </div>
      ${recs.length ? `<ul style="margin:.55rem 0 0 1.1rem">${recs.map(item => `<li>${_esc(item.text || item)}${item.count ? ` (${_esc(String(item.count))})` : ""}</li>`).join("")}</ul>` : ""}
      ${actions}
    </div>`;
}

function _renderMemoryTrend(trend) {
  if (!trend || trend.delta === null || trend.delta === undefined) return "";
  const delta = Number(trend.delta || 0);
  const sign = delta > 0 ? "+" : "";
  const label = `${trend.direction || "trend"} ${sign}${delta}`;
  const color = delta > 0 ? "#4ade80" : delta < 0 ? "#f87171" : "var(--text-muted)";
  return `<span style="opacity:.86;margin-left:.5rem;color:${color}">${_esc(label)}</span>`;
}

function _memoryQuickFixButtons(totals = {}) {
  const buttons = [
    `<button type="button" class="btn btn-primary btn-sm at-memory-verify">Run to verify</button>`,
  ];
  const needsRepair = (
    (totals.missed_search_before_save_count || 0) > 0
    || (totals.duplicate_save_candidate_count || 0) > 0
    || (totals.empty_search_count || 0) > 0
  );
  if (needsRepair) {
    buttons.push(`<button type="button" class="btn btn-outline btn-sm at-memory-fix" data-fix="auto"
      data-add-search="${(totals.missed_search_before_save_count || 0) > 0 ? "1" : "0"}"
      data-dedup-save="${(totals.duplicate_save_candidate_count || 0) > 0 ? "1" : "0"}"
      data-skip-empty="${(totals.empty_search_count || 0) > 0 ? "1" : "0"}">Auto repair</button>`);
  }
  return buttons.length
    ? `<div style="display:flex;gap:.45rem;flex-wrap:wrap;margin-top:.75rem">${buttons.join("")}</div>`
    : "";
}

function _bindMemoryQuickFixes(body) {
  body.querySelector(".at-memory-verify")?.addEventListener("click", async () => {
    if (!_autoMeta?.id) return;
    document.getElementById("at-history-modal")?.classList.add("hidden");
    _showRunModal(_autoMeta?.name || "Automation");
    await _runAutomationById(_autoMeta.id);
    document.getElementById("at-run-modal")?.classList.add("hidden");
    await _renderAutomationList();
    await _openAutomationHistory();
  });
  body.querySelectorAll(".at-memory-fix").forEach(btn => {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      const original = btn.textContent;
      btn.textContent = "Applying...";
      try {
        await _applyMemoryQuickFix(btn.dataset.fix, btn.dataset);
      } finally {
        btn.disabled = false;
        btn.textContent = original;
      }
    });
  });
}

function _refreshNodePreview(nodeId) {
  const node = _editor?.getNodeFromId(nodeId);
  const preview = document.querySelector(`#node-${nodeId} .pf-node-preview`);
  if (node && preview) preview.textContent = _nodePreview(node.name, node.data || {});
}

function _findFirstNodeByName(name) {
  const nodes = Object.values(_editor?.export?.()?.drawflow?.Home?.data || {});
  return nodes.filter(node => node.name === name).sort((a, b) => Number(a.id) - Number(b.id))[0] || null;
}

function _findIncomingConnection(targetId, inputName = "input_1") {
  const nodes = Object.values(_editor?.export?.()?.drawflow?.Home?.data || {});
  for (const node of nodes) {
    for (const [outputName, output] of Object.entries(node.outputs || {})) {
      for (const conn of output.connections || []) {
        if (String(conn.node) === String(targetId) && String(conn.output || inputName) === inputName) {
          return { sourceId: node.id, outputName };
        }
      }
    }
  }
  return null;
}

function _connectMemorySearchBeforeSave() {
  if (!_editor) return false;
  const saveNode = _findFirstNodeByName("memory_save");
  if (!saveNode) return false;
  const incoming = _findIncomingConnection(saveNode.id);
  if (incoming) {
    const incomingNode = _editor.getNodeFromId(incoming.sourceId);
    if (incomingNode?.name === "memory_search") {
      const data = { ...(incomingNode.data || {}), skip_empty_result: true };
      _editor.updateNodeDataFromId(incomingNode.id, data);
      _refreshNodePreview(incomingNode.id);
      return true;
    }
  }
  const x = Math.max(160, Number(saveNode.pos_x || 420) - 240);
  const y = Number(saveNode.pos_y || 220);
  const searchId = _addBlock("memory_search", x, y, {
    label: "Memory Search Guard",
    input_source: "{{prev}}",
    query: "{{prev}}",
    limit: 5,
    skip_empty_result: true,
  });
  if (incoming) {
    try { _editor.removeSingleConnection?.(incoming.sourceId, saveNode.id, incoming.outputName, "input_1"); } catch (_err) {}
    try { _editor.addConnection(incoming.sourceId, searchId, incoming.outputName, "input_1"); } catch (_err) {}
  }
  try { _editor.addConnection(searchId, saveNode.id, "output_1", "input_1"); } catch (_err) {}
  setTimeout(_ensureNodeConfigButtons, 0);
  return true;
}

function _updateNodesByName(name, updater) {
  const nodes = Object.values(_editor?.export?.()?.drawflow?.Home?.data || {}).filter(node => node.name === name);
  nodes.forEach(node => {
    const data = updater({ ...(node.data || {}) });
    _editor.updateNodeDataFromId(node.id, data);
    _refreshNodePreview(node.id);
  });
  return nodes.length;
}

async function _saveMemoryQuickFix(successMessage) {
  const saved = await _saveAutomation();
  _showToast(saved ? successMessage : "Quick fix applied locally, but save failed.");
}

async function _applyMemoryQuickFix(fix, options = {}) {
  if (!_editor) return;
  if (fix === "auto") {
    let changed = false;
    if (options.addSearch === "1") {
      changed = _connectMemorySearchBeforeSave() || changed;
    }
    if (options.dedupSave === "1") {
      changed = _updateNodesByName("memory_save", data => ({ ...data, dedup_guard: true, deduplicate: true })) > 0 || changed;
    }
    if (options.skipEmpty === "1") {
      changed = _updateNodesByName("memory_search", data => ({ ...data, skip_empty_result: true })) > 0 || changed;
    }
    if (changed) await _saveMemoryQuickFix("Workflow repaired and saved.");
    else _showToast("No repairable memory issues found.");
    return;
  }
  if (fix === "add-search") {
    if (_connectMemorySearchBeforeSave()) await _saveMemoryQuickFix("Memory Search guard applied and saved.");
    else _showToast("Add a Memory Save block first.");
    return;
  }
  if (fix === "dedup-save") {
    const count = _updateNodesByName("memory_save", data => ({ ...data, dedup_guard: true, deduplicate: true }));
    if (count) await _saveMemoryQuickFix("Dedup guard enabled and saved.");
    else _showToast("No Memory Save blocks found.");
    return;
  }
  if (fix === "skip-empty") {
    const count = _updateNodesByName("memory_search", data => ({ ...data, skip_empty_result: true }));
    if (count) await _saveMemoryQuickFix("Empty memory searches will be skipped and saved.");
    else _showToast("No Memory Search blocks found.");
  }
}

async function _openAutomationHistory() {
  if (!_autoMeta?.id) {
    alert("Save the automation before viewing history.");
    return;
  }
  const modal = document.getElementById("at-history-modal");
  const body = document.getElementById("at-history-body");
  if (!modal || !body) return;
  modal.classList.remove("hidden");
  body.innerHTML = `<div class="pf-run-step pf-run-step-running">Loading history...</div>`;

  const [res, memoryReport] = await Promise.all([
    fetch(`/api/automations/${_autoMeta.id}/runs`).then(r => r.json()).catch(() => ({ runs: [] })),
    fetch(`/api/automations/${_autoMeta.id}/memory-report`).then(r => r.json()).catch(() => null),
  ]);
  const runs = res.runs || [];
  if (!runs.length) {
    body.innerHTML = `${_renderMemoryReportSummary(memoryReport)}<div class="pf-run-step">No runs yet.</div>`;
    _bindMemoryQuickFixes(body);
    return;
  }
  body.innerHTML = `${_renderMemoryReportSummary(memoryReport)}${runs.map(run => `
    <button type="button" class="pf-run-step at-history-run" data-run-id="${_esc(run.run_id)}" style="width:100%;text-align:left">
      <strong>${_esc(run.status || "unknown")}</strong>
      <span style="opacity:.72;margin-left:.5rem">${_esc(run.started_at || "")}</span>
      ${run.memory_evaluation?.score !== undefined ? `<span style="opacity:.72;margin-left:.5rem">Memory ${_esc(String(run.memory_evaluation.score))}/100</span>` : ""}
      <div class="pf-rs-output">${_esc(run.error || run.output_preview || "(no output)")}</div>
    </button>`).join("")}`;
  _bindMemoryQuickFixes(body);

  body.querySelectorAll(".at-history-run").forEach(btn => {
    btn.addEventListener("click", async () => {
      const runId = btn.dataset.runId;
      const run = await fetch(`/api/automations/${_autoMeta.id}/runs/${runId}`).then(r => r.json()).catch(() => null);
      if (!run) return;
      body.innerHTML = `
        <button type="button" class="btn btn-outline btn-sm" id="at-history-back">← Runs</button>
        <button type="button" class="btn btn-primary btn-sm" id="at-history-rerun" style="margin-left:.5rem">Run again</button>
        <button type="button" class="btn btn-outline btn-sm" id="at-history-copy-output" style="margin-left:.5rem">Copy output</button>
        <div class="pf-run-step">
          <strong>${_esc(run.status || "unknown")}</strong>
          <span style="opacity:.72;margin-left:.5rem">${_esc(run.run_id || "")}</span>
          <div style="opacity:.72;margin-top:.35rem">${_esc(run.started_at || "")} · ${run.duration_ms ?? "?"} ms</div>
        </div>
        ${_renderMemoryEvaluation(run.memory_evaluation)}
        ${_renderRunTrace(run.trace || []) || `<div class="pf-run-step">No step trace recorded.</div>`}
        <details class="pf-run-step" open>
          <summary>Final output</summary>
          <pre class="pf-run-step-output">${_esc(run.output || run.error || "(no output)")}</pre>
        </details>`;
      body.querySelector("#at-history-back")?.addEventListener("click", _openAutomationHistory);
      body.querySelector("#at-history-rerun")?.addEventListener("click", async () => {
        document.getElementById("at-history-modal")?.classList.add("hidden");
        _showRunModal(_autoMeta?.name || "Automation");
        await _runAutomationById(_autoMeta.id);
      });
      body.querySelector("#at-history-copy-output")?.addEventListener("click", () => {
        navigator.clipboard?.writeText(run.output || run.error || "").then(() => _showToast("Copied"));
      });
    });
  });
}

async function _fetchModels() {
  _availableModels = [];
  _availableSecrets = [];
  _settingsSnapshot = null;
  const [s, res, secrets] = await Promise.all([
    fetch("/api/settings").then(r => r.json()).catch(() => null),
    fetch("/api/models").then(r => r.json()).catch(() => null),
    fetch("/api/secrets").then(r => r.json()).catch(() => null),
  ]);
  _settingsSnapshot = s || null;
  _availableSecrets = (secrets?.secrets || []).filter(secret => secret.configured);
  if (s?.summary_model) _availableModels.push({ id: s.summary_model, label: `${s.summary_model} (default)` });
  if (res?.models?.length) {
    for (const m of res.models) {
      if (!_availableModels.find(x => x.id === m.name))
        _availableModels.push({ id: m.name, label: m.label || m.name });
    }
  }
}

function _secretOptions(current) {
  let html = `<option value="" ${!current ? "selected" : ""}>No Authorization header</option>`;
  for (const secret of _availableSecrets) {
    html += `<option value="${_esc(secret.key)}" ${secret.key === current ? "selected" : ""}>${_esc(secret.label)} (${_esc(secret.kind)})</option>`;
  }
  if (current && !_availableSecrets.find(secret => secret.key === current)) {
    html += `<option value="${_esc(current)}" selected>${_esc(current)} (not configured)</option>`;
  }
  return html;
}

function _modelOptions(current) {
  let html = `<option value="" ${!current?"selected":""}>— Default model —</option>`;
  for (const m of _availableModels) {
    html += `<option value="${_esc(m.id)}" ${m.id===current?"selected":""}>${_esc(m.label)}</option>`;
  }
  return html;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _triggerPreview(d) {
  const t = d.trigger_type || "daily";
  if (t === "daily")    return `Daily at ${d.time_of_day || "09:00"}`;
  if (t === "weekly") {
    const days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
    return `${days[d.day_of_week||0]} at ${d.time_of_day||"09:00"}`;
  }
  if (t === "hourly")   return "Every hour";
  if (t === "custom")   return `cron: ${d.cron || "?"}`;
  if (t === "on_start") return "On app start";
  if (t === "manual")   return "Manual only";
  return t;
}

function _triggerShortLabel(trigger) {
  const st = trigger.schedule_type || trigger.type || "manual";
  if (st === "daily")    return `Daily ${trigger.time_of_day || "09:00"}`;
  if (st === "weekly")   return `Weekly ${trigger.time_of_day || "09:00"}`;
  if (st === "hourly")   return "Every hour";
  if (st === "custom")   return `cron: ${trigger.cron || "?"}`;
  if (st === "on_start") return "On start";
  return "Manual";
}

function _esc(text) {
  const d = document.createElement("div");
  d.textContent = text ?? "";
  return d.innerHTML;
}

function _apiError(data, fallback) {
  const detail = data?.detail ?? data?.error ?? data?.message;
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail?.errors)) return detail.errors.join("\n");
  if (typeof detail?.error === "string") return detail.error;
  try {
    return JSON.stringify(detail);
  } catch (_) {
    return fallback;
  }
}

function _showToast(msg) {
  const t = Object.assign(document.createElement("div"), { className: "pl-toast", textContent: msg });
  document.body.appendChild(t);
  setTimeout(() => t.classList.add("pl-toast-show"), 10);
  setTimeout(() => { t.classList.remove("pl-toast-show"); setTimeout(() => t.remove(), 300); }, 2500);
}
