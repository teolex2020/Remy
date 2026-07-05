/**
 * Pipelines — visual flow builder powered by Drawflow.
 *
 * UX principles (like Flowise):
 *  - Canvas fills the screen, drag nodes freely
 *  - "Start" and "Result" nodes are always present (fixed entry/exit)
 *  - Sidebar palette to drag-add blocks
 *  - Click a node → config panel slides in from right
 *  - Simple text for prompts, dropdown for model selection
 *  - Condition node has 2 outputs (true/false), Loop node loops back
 */

// ── Block catalogue ───────────────────────────────────────────────────────────

const BLOCKS = [
  { type: "llm_call",      icon: "🤖", label: "AI Response",      color: "#6366f1",
    inputs: 1, outputs: 1,
    defaults: { system_prompt: "", input_source: "{{input}}", model: "" } },
  { type: "web_search",    icon: "🔍", label: "Web Search",        color: "#0ea5e9",
    inputs: 1, outputs: 1,
    defaults: { num_results: 5, input_source: "{{input}}" } },
  { type: "page_scrape",   icon: "Pg", label: "Page Scraper",      color: "#0f766e",
    inputs: 1, outputs: 1,
    defaults: { url: "", mode: "text", max_chars: 12000 } },
  { type: "memory_search", icon: "🧠", label: "Memory Search",     color: "#8b5cf6",
    inputs: 1, outputs: 1,
    defaults: { limit: 5, input_source: "{{input}}" } },
  { type: "memory_save",   icon: "💾", label: "Save to Memory",    color: "#10b981",
    inputs: 1, outputs: 1,
    defaults: { tags: "pipeline", input_source: "{{input}}" } },
  { type: "http_request",  icon: "🌐", label: "HTTP Request",      color: "#f59e0b",
    inputs: 1, outputs: 1,
    defaults: { url: "", method: "GET", body: "", auth_secret_key: "", auth_scheme: "Bearer" } },
  { type: "router",        icon: "?", label: "Router",             color: "#f97316",
    inputs: 1, outputs: 2,
    defaults: {
      input_source: "{{input}}",
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
  { type: "condition",     icon: "❓", label: "Condition",         color: "#f97316",
    inputs: 1, outputs: 2,
    defaults: { condition: "", description_true: "True", description_false: "False", input_source: "{{input}}" } },
  { type: "loop",          icon: "🔁", label: "Loop",              color: "#ec4899",
    inputs: 1, outputs: 1,
    defaults: { max_iterations: 5, stop_condition: "" } },
  { type: "template",      icon: "📝", label: "Text / Template",   color: "#64748b",
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
  return BLOCKS.find(b => b.type === type) || { icon: "❓", label: type, color: "#64748b", inputs:1, outputs:1, defaults:{} };
}

// ── State ─────────────────────────────────────────────────────────────────────

function _blockHelpText(type) {
  const help = {
    llm_call: {
      title: "What this block does",
      body: "Sends selected input text to the AI model and returns a generated answer. Use it when text must be analyzed, rewritten, classified, or summarized.",
      tip: "Tip: keep the prompt specific: what to produce, what to ignore, and what format to return.",
    },
    web_search: {
      title: "What this block does",
      body: "Searches the web and passes source text to the next block. Use it when the answer depends on fresh or external information.",
      tip: "Tip: connect it to AI Response when you want the model to summarize or evaluate found sources.",
    },
    page_scrape: {
      title: "What this block does",
      body: "Fetches one web page, extracts readable content, and passes clean text, title, or links to the next block.",
      tip: "Tip: use it before AI Response for page analytics, changelog monitoring, and report extraction.",
    },
    memory_search: {
      title: "What this block does",
      body: "Finds relevant records in local memory. Use it when the answer should rely on saved user facts, prior decisions, or project knowledge.",
      tip: "Tip: fewer results are better for precise answers; more results are better for broad context.",
    },
    memory_save: {
      title: "What this block does",
      body: "Stores selected text into memory with tags. Use it to keep summaries, decisions, research findings, or extracted facts.",
      tip: "Tip: use clear tags like research, client, finance, project, or a project name.",
    },
    http_request: {
      title: "What this block does",
      body: "Calls an external HTTP endpoint and passes the response forward. Use it for APIs, webhooks, and simple integrations.",
      tip: "Tip: use GET for reading data and POST when sending a request body.",
    },
    router: {
      title: "What this block does",
      body: "Splits the pipeline into multiple routes. Each route can use contains, equals, regex, fallback, always, or AI condition.",
      tip: "Tip: use Merge / Wait after Router when several branches must be joined before continuing.",
    },
    merge: {
      title: "What this block does",
      body: "Waits for multiple connected branches and combines their outputs into one result. Use it after Router fan-out.",
      tip: "Tip: connect at least two inputs. Combine text is best before AI Response; JSON array is best for structured processing.",
    },
    delay: {
      title: "What this block does",
      body: "Pauses the workflow before continuing. Use it when an API, page, or external process needs time before the next step.",
      tip: "Tip: keep delays short; long waits make scheduled workflows feel broken.",
    },
    filter: {
      title: "What this block does",
      body: "Allows the flow to continue only when the condition matches. If it does not match, this branch stops.",
      tip: "Tip: use Router when you need alternate paths; use Filter when you only need pass or stop.",
    },
    set_variable: {
      title: "What this block does",
      body: "Stores a value under a name so later blocks can use it with {{name}}.",
      tip: "Tip: variable names should be short and stable, for example customer_email or topic.",
    },
    parse_json: {
      title: "What this block does",
      body: "Extracts a value from JSON using a simple path like $.items[0].title.",
      tip: "Tip: place it after HTTP Request when the API returns JSON.",
    },
    transform: {
      title: "What this block does",
      body: "Cleans or reshapes text: trim, upper/lowercase, replace, regex extract, truncate, or join lines.",
      tip: "Tip: use it before AI Response to reduce noisy input.",
    },
    notification: {
      title: "What this block does",
      body: "Creates a local workflow notification entry and passes the message forward.",
      tip: "Tip: use it near the end of an automation when something important happened.",
    },
    file_read: {
      title: "What this block does",
      body: "Reads a text file from the app workflow_files directory and passes its content forward.",
      tip: "Tip: file access is intentionally limited to the app data folder for safety.",
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
      tip: "Tip: connect output_2 from a risky block into Error Handler, then connect Error Handler back to the normal flow.",
    },
    condition: {
      title: "What this block does",
      body: "Legacy two-way True/False branch. Router is preferred for new workflows because it supports many routes and structured operators.",
      tip: "Tip: use it only for very simple yes/no branches.",
    },
    loop: {
      title: "What this block does",
      body: "Repeats work until a stop condition is met or max iterations is reached. Use it for iterative refinement.",
      tip: "Tip: keep max iterations low to avoid long runs and excessive model calls.",
    },
    template: {
      title: "What this block does",
      body: "Creates fixed text or a prompt template. Use variables like {{input}}, {{prev}}, or previous step outputs.",
      tip: "Tip: place it before AI Response to shape the exact instruction sent to the model.",
    },
  };
  return help[type] || {
    title: "What this block does",
    body: "Processes input and sends its result to the next connected block.",
    tip: "Tip: run the pipeline and click the block status badge to inspect exact output.",
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

let _editor = null;           // Drawflow instance
let _pipelineMeta = null;     // { id, name, description }
let _selectedNodeId = null;   // currently selected node id (string)
let _availableModels = [];    // fetched from /api/settings or /api/ollama/status
let _availableSecrets = [];
let _runAbort = null;
let _lastRunTraceByStepId = new Map();
let _currentRunTrace = [];

// ── Public API ────────────────────────────────────────────────────────────────

export async function loadPipelines() {
  const pane = document.getElementById("pipelines-content");
  if (!pane) return;

  const alreadyMounted = !!pane.querySelector("#pf-sidebar");
  if (!alreadyMounted) {
    _renderShell(pane);
    await _fetchModels();
  }

  await _renderPipelineList();
  await _openPendingPipelineFromHome();
}

export async function openPipelineMemoryHistory(pipelineId) {
  const id = String(pipelineId || "");
  if (!id) return false;
  await loadPipelines();
  const data = await fetch(`/api/pipelines/${encodeURIComponent(id)}`).then(r => r.json()).catch(() => null);
  if (!data) return false;
  _openEditor(data);
  await _openPipelineHistory();
  return true;
}

// ── Shell layout ──────────────────────────────────────────────────────────────

function _renderShell(pane) {
  pane.innerHTML = `
  <div class="pf-shell pf-mode-list">
    <!-- Left sidebar: list + palette -->
    <div class="pf-sidebar" id="pf-sidebar">
      <div class="pf-sidebar-header">
        <span class="pf-sidebar-title">Pipelines</span>
        <button class="btn btn-primary btn-sm" id="pf-new-btn">+ New</button>
      </div>

      <!-- Pipeline list (shown when not editing) -->
      <div id="pf-list-panel">
        <div class="pf-section-label">Saved</div>
        <div id="pf-pipeline-list" class="pf-pipeline-list">
          <div class="pf-loading">Loading…</div>
        </div>
        <div class="pf-section-label" style="margin-top:1rem">Templates</div>
        <div id="pf-template-list" class="pf-pipeline-list">
          <div class="pf-loading">Loading…</div>
        </div>
      </div>

      <!-- Block palette (shown when editing) -->
      <div id="pf-palette-panel" class="hidden">
        <div class="pf-section-label">← Drag a block to the canvas</div>
        ${BLOCKS.filter(b => b.type !== "condition").map(b => `
          <div class="pf-palette-item" draggable="true"
               data-type="${b.type}"
               style="--bc:${b.color}">
            <span class="pf-pi-icon">${b.icon}</span>
            <span class="pf-pi-label">${b.label}</span>
          </div>`).join("")}
        <div class="pf-section-label" style="margin-top:1rem">Hint</div>
        <div class="pf-hint-box">Drag blocks onto the canvas. Connect dots between blocks. Use the block menu to configure processing steps.</div>
      </div>
    </div>

    <!-- Canvas area -->
    <div class="pf-canvas-wrap hidden" id="pf-canvas-wrap">
      <!-- Toolbar -->
      <div class="pf-toolbar">
        <button class="btn btn-outline btn-sm" id="pf-back-btn">← Back</button>
        <div class="pf-toolbar-meta">
          <input class="pf-title-input" id="pf-title-input" placeholder="Pipeline name…">
        </div>
        <div class="pf-toolbar-actions">
          <button class="btn btn-outline btn-sm" id="pf-zoom-in">+</button>
          <button class="btn btn-outline btn-sm" id="pf-zoom-out">−</button>
          <button class="btn btn-outline btn-sm" id="pf-zoom-reset">↺</button>
          <button class="btn btn-outline btn-sm" id="pf-history-btn">History</button>
          <button class="btn btn-outline btn-sm" id="pf-save-template-btn">Save as Template</button>
          <button class="btn btn-outline btn-sm" id="pf-save-btn">💾 Save</button>
          <button class="btn btn-primary btn-sm" id="pf-run-btn">▶ Run</button>
        </div>
      </div>
      <!-- Drawflow container -->
      <div id="pf-drawflow" class="pf-drawflow-container"></div>
    </div>

    <!-- Right config panel -->
    <div class="pf-config-panel hidden" id="pf-config-panel">
      <div class="pf-config-resize" id="pf-config-resize" title="Resize settings panel"></div>
      <div class="pf-config-header">
        <span id="pf-config-title">Settings</span>
        <button class="pf-config-close" id="pf-config-close">✕</button>
      </div>
      <div id="pf-config-body" class="pf-config-body"></div>
    </div>
  </div>

  <!-- Run modal -->
  <div class="pf-run-modal hidden" id="pf-run-modal">
    <div class="pf-run-box">
      <div class="pf-run-header">
        <span id="pf-run-title">Running pipeline</span>
        <button class="pf-config-close" id="pf-run-close">✕</button>
      </div>
      <div id="pf-run-steps" class="pf-run-steps"></div>
      <div class="pf-run-final hidden" id="pf-run-final">
        <div class="pf-run-final-label">Result</div>
        <div class="pf-run-final-text" id="pf-run-final-text"></div>
        <div class="pf-run-final-actions">
          <button class="btn btn-outline btn-sm" id="pf-copy-btn">📋 Copy</button>
          <button class="btn btn-primary btn-sm" id="pf-to-chat-btn">💬 Send to Chat</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Per-block result modal -->
  <div class="pf-step-modal hidden" id="pf-step-modal">
    <div class="pf-step-box">
      <div class="pf-run-header">
        <span id="pf-step-title">Block result</span>
        <button class="pf-config-close" id="pf-step-close">&#10005;</button>
      </div>
      <div class="pf-step-meta" id="pf-step-meta"></div>
      <pre class="pf-run-step-output pf-step-output" id="pf-step-output"></pre>
      <div class="pf-step-actions">
        <button class=”btn btn-outline btn-sm” id=”pf-step-copy”>&#128203; Copy</button>
        <button class="btn btn-outline btn-sm" id="pf-step-pin">Pin result</button>
      </div>
    </div>
  </div>
  <div class="pf-run-modal hidden" id="pf-history-modal">
    <div class="pf-run-box">
      <div class="pf-run-header">
        <span>Pipeline history</span>
        <button class="pf-config-close" id="pf-history-close">✕</button>
      </div>
      <div id="pf-history-body" class="pf-run-steps"></div>
    </div>
  </div>`;

  document.getElementById("pf-new-btn").addEventListener("click", () => _openEditor(null));
  document.getElementById("pf-back-btn")?.addEventListener("click", _closeEditor);
  document.getElementById("pf-save-btn")?.addEventListener("click", _savePipeline);
  document.getElementById("pf-save-template-btn")?.addEventListener("click", _savePipelineAsTemplate);
  document.getElementById("pf-run-btn")?.addEventListener("click", _promptRun);
  document.getElementById("pf-history-btn")?.addEventListener("click", _openPipelineHistory);
  document.getElementById("pf-zoom-in")?.addEventListener("click", () => _editor?.zoom_in());
  document.getElementById("pf-zoom-out")?.addEventListener("click", () => _editor?.zoom_out());
  document.getElementById("pf-zoom-reset")?.addEventListener("click", () => _editor?.zoom_reset());
  document.getElementById("pf-config-close")?.addEventListener("click", _closeConfig);
  _bindConfigResize();
  document.getElementById("pf-run-close")?.addEventListener("click", _closeRunModal);
  document.getElementById("pf-history-close")?.addEventListener("click", () => {
    document.getElementById("pf-history-modal")?.classList.add("hidden");
  });
  document.getElementById("pf-step-close")?.addEventListener("click", _hideStepResultModal);
  document.getElementById("pf-step-copy")?.addEventListener("click", () => {
    const txt = document.getElementById("pf-step-output")?.textContent || "";
    navigator.clipboard.writeText(txt);
  });
  document.getElementById("pf-step-pin")?.addEventListener("click", _pinCurrentStepResult);
  document.getElementById("pf-drawflow")?.addEventListener("click", (ev) => {
    const cfg = ev.target?.closest?.(".pf-node-config-btn");
    if (cfg) {
      ev.preventDefault();
      ev.stopPropagation();
      const nodeId = cfg.closest(".drawflow-node")?.id?.replace(/^node-/, "");
      if (nodeId) _openNodeConfig(nodeId);
      return;
    }

    const badge = ev.target?.closest?.(".pf-node-run-badge");
    if (!badge) return;
    ev.preventDefault();
    ev.stopPropagation();
    _openStepResult(badge.dataset.stepId);
  });
  document.getElementById("pf-drawflow")?.addEventListener("pointerdown", (ev) => {
    if (!ev.target?.closest?.(".pf-node-config-btn,.pf-node-run-badge")) return;
    ev.preventDefault();
    ev.stopPropagation();
  });
}

// ── Pipeline list ─────────────────────────────────────────────────────────────

async function _renderPipelineList() {
  const [savedRes, tplRes] = await Promise.all([
    fetch("/api/pipelines").then(r => r.json()).catch(() => ({ pipelines: [] })),
    fetch("/api/pipelines/templates/list").then(r => r.json()).catch(() => ({ templates: [] })),
  ]);

  _fillList("pf-pipeline-list", savedRes.pipelines || [], false);
  _fillList("pf-template-list", tplRes.templates || [], true);
}

async function _openPendingPipelineFromHome() {
  const id = window.sessionStorage.getItem("remy_pending_pipeline_open") || "";
  if (!id) return;
  window.sessionStorage.removeItem("remy_pending_pipeline_open");
  const data = await fetch(`/api/pipelines/${encodeURIComponent(id)}`).then(r => r.json()).catch(() => null);
  if (data) {
    _openEditor(data);
  }
}

function _fillList(containerId, items, isTemplate) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!items.length) {
    el.innerHTML = `<div class="pf-empty">${isTemplate ? "No templates available" : "No saved pipelines"}</div>`;
    return;
  }
  el.innerHTML = items.map(p => {
    const steps = p.step_count ?? (p.steps?.length ?? 0);
    const isCustomTemplate = isTemplate && p.source === "custom";
    return `
    <div class="pf-list-item ${isTemplate ? "pf-list-tpl" : ""}" data-id="${p.id}">
      <div class="pf-list-item-top">
        <span class="pf-list-item-badge">${isTemplate ? "TPL" : "⛓"}</span>
        <div class="pf-list-item-info">
          <div class="pf-list-item-name">${_esc(p.name)}</div>
          <div class="pf-list-item-meta">
            <span>${steps} ${steps === 1 ? "step" : "steps"}</span>
            ${isCustomTemplate ? `<span class="auto-chip-muted">Custom</span>` : ""}
            ${!isTemplate && p.source_template_name ? `<span class="auto-chip-muted" title="Created from template ${_esc(p.source_template_name)}">From ${_esc(p.source_template_name)}</span>` : ""}
            ${!isTemplate ? `<span class="workflow-memory-badge workflow-memory-loading" data-memory-report="${_esc(p.id)}"></span>` : ""}
            ${!isTemplate ? `<span class="workflow-memory-badge workflow-safety-loading" data-safety-report="${_esc(p.id)}"></span>` : ""}
          </div>
        </div>
      </div>
      <div class="pf-list-item-actions">
        <button class="pf-card-btn pf-edit-btn" data-id="${p.id}" title="Edit pipeline">
          <span class="pf-card-btn-icon">✏</span><span class="pf-card-btn-label">Edit</span>
        </button>
        ${!isTemplate ? `<button class="pf-card-btn pf-card-btn-run pf-run-list-btn" data-id="${p.id}" title="Run pipeline">
          <span class="pf-card-btn-icon">▶</span><span class="pf-card-btn-label">Run</span>
        </button>` : ""}
        ${!isTemplate ? `<button class="pf-card-btn pf-card-btn-danger pf-del-btn" data-id="${p.id}" title="Delete pipeline">
          <span class="pf-card-btn-icon">🗑</span>
        </button>` : ""}
      </div>
    </div>`;
  }).join("");
  if (!isTemplate) {
    _hydratePipelineMemoryBadges(items);
    _hydratePipelineSafetyBadges(items);
  }
  if (isTemplate) {
    items.filter(item => item.source === "custom").forEach(item => {
      const row = el.querySelector(`.pf-list-item[data-id="${CSS.escape(String(item.id))}"] .pf-list-item-actions`);
      if (!row || row.querySelector(".pf-template-del-btn")) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pf-card-btn pf-card-btn-danger pf-template-del-btn";
      btn.dataset.id = item.id;
      btn.title = "Delete template";
      btn.innerHTML = `<span class="pf-card-btn-label">Delete</span>`;
      row.appendChild(btn);
    });
  }

  el.querySelectorAll(".pf-edit-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      let data;
      if (isTemplate) {
        const tplRes = await fetch("/api/pipelines/templates/list").then(r => r.json()).catch(() => null);
        data = (tplRes?.templates || []).find(t => t.id === btn.dataset.id);
        if (data) data = { ...data, id: null };
      } else {
        data = await fetch(`/api/pipelines/${btn.dataset.id}`).then(r => r.json()).catch(() => null);
      }
      if (data) _openEditor(data);
    });
  });

  el.querySelectorAll(".pf-run-list-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const data = await fetch(`/api/pipelines/${btn.dataset.id}`).then(r => r.json()).catch(() => null);
        if (!data) return;
        const input = prompt(`Enter input for "${data.name}":`, "");
        if (input === null) return;
        if (!await _confirmPipelinePreflight(btn.dataset.id)) return;
        _openRunModal(data.name);
        _startRun(btn.dataset.id, input);
    });
  });

  el.querySelectorAll(".pf-del-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const item = items.find(p => p.id === btn.dataset.id);
      if (!confirm(`Delete "${item?.name}"?`)) return;
      await fetch(`/api/pipelines/${btn.dataset.id}`, { method: "DELETE" });
      await _renderPipelineList();
    });
  });

  el.querySelectorAll(".pf-template-del-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const item = items.find(p => p.id === btn.dataset.id);
      if (!confirm(`Delete template "${item?.name}"?`)) return;
      const res = await fetch(`/api/pipelines/templates/${encodeURIComponent(btn.dataset.id)}`, { method: "DELETE" });
      if (!res.ok) {
        alert("Template delete failed.");
        return;
      }
      await _renderPipelineList();
    });
  });
}

// ── Editor ─────────────────────────────────────────────────────────────────────

async function _hydratePipelineMemoryBadges(items) {
  await Promise.all((items || []).map(async item => {
    const id = String(item.id || "");
    if (!id) return;
    const badge = document.querySelector(`.workflow-memory-badge[data-memory-report="${CSS.escape(id)}"]`);
    if (!badge) return;
    const report = await fetch(`/api/pipelines/${id}/memory-report`).then(r => r.json()).catch(() => null);
    _renderWorkflowMemoryBadge(badge, report);
  }));
}

async function _hydratePipelineSafetyBadges(items) {
  await Promise.all((items || []).map(async item => {
    const id = String(item.id || "");
    if (!id) return;
    const badge = document.querySelector(`.workflow-memory-badge[data-safety-report="${CSS.escape(id)}"]`);
    if (!badge) return;
    const report = await fetch(`/api/pipelines/${encodeURIComponent(id)}/preflight`).then(r => r.json()).catch(() => null);
    _renderWorkflowSafetyBadge(badge, report, "Pipeline safety preflight");
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
    const data = await fetch(`/api/pipelines/${id}`).then(r => r.json()).catch(() => null);
    if (data) {
      _openEditor(data);
    } else {
      const card = el.closest(".pf-list-item");
      const name = card?.querySelector(".pf-list-item-name")?.textContent || "";
      _pipelineMeta = { id, name, description: "" };
    }
    _openPipelineHistory();
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

function _openEditor(pipeline) {
  _pipelineMeta = {
    id: pipeline?.id || null,
    name: pipeline?.name || "",
    description: pipeline?.description || "",
    source_template_id: pipeline?.source_template_id || "",
    source_template_name: pipeline?.source_template_name || "",
  };

  // Show canvas, palette; hide list
  document.querySelector(".pf-shell")?.classList.remove("pf-mode-list");
  document.getElementById("pf-list-panel").classList.add("hidden");
  document.getElementById("pf-palette-panel").classList.remove("hidden");
  document.getElementById("pf-canvas-wrap").classList.remove("hidden");
  document.getElementById("pf-title-input").value = _pipelineMeta.name;
  _renderPipelineTemplateChip();

  // Init Drawflow
  const container = document.getElementById("pf-drawflow");
  container.innerHTML = "";

  _editor = new Drawflow(container);
  _editor.reroute = true;
  _editor.start();

  // Style overrides for dark theme applied via CSS
  _editor.on("nodeSelected", id => { _selectedNodeId = String(id); });
  _editor.on("nodeUnselected", () => { _selectedNodeId = null; });
  _editor.on("nodeRemoved", () => _closeConfig());

  // Drag-from-palette
  _bindPaletteDrag(container);

  // Load existing pipeline or create fresh with Start+Result
  if (pipeline?.drawflow_data) {
    _editor.import(pipeline.drawflow_data);
  } else if (pipeline?.steps?.length) {
    _importPipelineSteps(pipeline.steps);
  } else {
    _addStartNode();
    _addResultNode();
  }
  _ensureErrorOutputs();
  setTimeout(_ensureNodeActionButtons, 0);
}

function _renderPipelineTemplateChip() {
  const meta = document.querySelector("#pf-canvas-wrap .pf-toolbar-meta");
  if (!meta) return;
  meta.querySelector(".pf-source-template-chip")?.remove();
  if (!_pipelineMeta?.source_template_name) return;
  const chip = document.createElement("span");
  chip.className = "auto-chip-muted pf-source-template-chip";
  chip.title = `Created from template ${_pipelineMeta.source_template_name}`;
  chip.textContent = `From ${_pipelineMeta.source_template_name}`;
  meta.appendChild(chip);
}

function _closeEditor() {
  _editor = null;
  _selectedNodeId = null;
  _closeConfig();
  document.querySelector(".pf-shell")?.classList.add("pf-mode-list");
  document.getElementById("pf-list-panel").classList.remove("hidden");
  document.getElementById("pf-palette-panel").classList.add("hidden");
  document.getElementById("pf-canvas-wrap").classList.add("hidden");
  _renderPipelineList();
}

// ── Node builders ─────────────────────────────────────────────────────────────

function _nodeHtml(type, data) {
  const b = _block(type);
  return `<div class="pf-node-inner" data-type="${type}">
    <div class="pf-node-header" style="background:${b.color}22;border-left:3px solid ${b.color}">
      <span class="pf-node-icon">${b.icon}</span>
      <span class="pf-node-label">${_esc(data.label || b.label)}</span>
      <button type="button" class="pf-node-config-btn" title="Configure block">...</button>
    </div>
    <div class="pf-node-preview">${_nodePreview(type, data)}</div>
  </div>`;
}

function _nodePreview(type, data) {
  if (data?._pinned_enabled) return _esc(`Pinned · ${(data._pinned_output || "").slice(0, 42)}`);
  const modelName = data.model ? data.model.split("/").pop() : "Current model";
  if (type === "llm_call") {
    const sysTxt = (data.system_prompt || "").slice(0, 35);
    return _esc(modelName + (sysTxt ? " · " + sysTxt + (data.system_prompt?.length > 35 ? "…" : "") : ""));
  }
  if (type === "web_search") return `Results: ${data.num_results || 5}`;
  if (type === "page_scrape") return _esc(`${data.mode || "text"} ${(data.url || "").slice(0, 30)}`.trim());
  if (type === "memory_search") return `Results: ${data.limit || 5}${data.skip_empty_result ? " · skip empty" : ""}`;
  if (type === "memory_save") return `Tags: ${data.tags || "pipeline"}${(data.dedup_guard || data.deduplicate) ? " · dedup" : ""}`;
  if (type === "merge") return _esc(`Inputs: ${_mergeInputCount(data)} · ${data.mode || "combine_text"}`);
  if (type === "router") return _esc(`Routes: ${_routerRoutes(data).length} · ${_routerModeLabel(data.mode)}`);
  if (type === "delay") return `Wait ${data.seconds || 1}s`;
  if (type === "filter") return _esc(`${data.operator || "contains"} ${data.value || ""}`.trim());
  if (type === "set_variable") return _esc(`{{${data.name || "value"}}}`);
  if (type === "parse_json") return _esc(data.path || "$");
  if (type === "transform") return _esc(data.mode || "trim");
  if (type === "notification") return _esc(data.title || "Notification");
  if (type === "file_read") return _esc(data.filename || "workflow.txt");
  if (type === "file_write") return _esc(data.filename || "workflow.txt");
  if (type === "code") return (data.mode || "safe_expression") === "local_script" ? `${data.language || "python"} local script` : "Safe expression";
  if (type === "error_handler") return data.fallback_text ? "Fallback configured" : "Error recovery";
  if (type === "http_request") return `${data.method || "GET"} ${_esc((data.url || "").slice(0, 30))}`;
  if (type === "condition") return _esc((data.condition || "Condition…").slice(0, 50));
  if (type === "loop") return `Max ${data.max_iterations || 5} iterations`;
  if (type === "template") return _esc((data.text || "").slice(0, 50));
  return "";
}

function _ensureNodeActionButtons() {
  document.querySelectorAll("#pf-drawflow .drawflow-node").forEach(nodeEl => {
    const nodeId = nodeEl.id?.replace(/^node-/, "");
    const node = nodeId ? _editor?.getNodeFromId(nodeId) : null;
    if (!node || node.name === "start" || node.name === "result") return;

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

function _startNodeHtml() {
  return `<div class="pf-node-inner pf-node-start">
    <div class="pf-node-header" style="background:#10b98122;border-left:3px solid #10b981">
      <span class="pf-node-icon">▶</span>
      <span class="pf-node-label">Start (chat input)</span>
    </div>
    <div class="pf-node-preview">User message</div>
  </div>`;
}

function _resultNodeHtml() {
  return `<div class="pf-node-inner pf-node-result">
    <div class="pf-node-header" style="background:#6366f122;border-left:3px solid #6366f1">
      <span class="pf-node-icon">✅</span>
      <span class="pf-node-label">Result</span>
    </div>
    <div class="pf-node-preview">Final answer</div>
  </div>`;
}

function _addStartNode() {
  return _editor.addNode("start", 0, 1, 80, 200, "pf-start", { label: "Start" }, _startNodeHtml(), false);
}

function _addResultNode() {
  return _editor.addNode("result", 1, 0, 700, 200, "pf-result", { label: "Result" }, _resultNodeHtml(), false);
}

function _blockOutputCount(type, data = {}) {
  if (type === "router") return _routerRoutes(data).length;
  if (type === "condition") return 2;
  if (["start", "result", "error_handler"].includes(type)) return _block(type).outputs || 1;
  return Math.max(2, _block(type).outputs || 1);
}

function _ensureErrorOutputs() {
  if (!_editor) return;
  const nodes = _editor.export()?.drawflow?.Home?.data || {};
  Object.values(nodes).forEach(node => {
    if (!node || ["start", "result", "router", "condition", "error_handler"].includes(node.name)) return;
    if (Object.keys(node.outputs || {}).length < 2) _editor.addNodeOutput(node.id);
  });
}

function _addBlock(type, x, y, extraData = {}) {
  const b = _block(type);
  const data = { ...b.defaults, label: b.label, ...extraData };
  const outputs = _blockOutputCount(type, data);
  const inputs = type === "merge" ? _mergeInputCount(data) : b.inputs;
  return _editor.addNode(type, inputs, outputs, x, y, `pf-block pf-block-${type}`, data, _nodeHtml(type, data), false);
}

// ── Drag from palette ─────────────────────────────────────────────────────────

function _bindPaletteDrag(container) {
  let _dragType = null;

  document.querySelectorAll(".pf-palette-item").forEach(item => {
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
    const x = (e.clientX - rect.left - _editor.canvas_x) / zoom;
    const y = (e.clientY - rect.top - _editor.canvas_y) / zoom;

    const nodeId = _addBlock(_dragType, x - 100, y - 40);
    _dragType = null;

    setTimeout(_ensureNodeActionButtons, 0);
  });
}

// ── Restore config from saved format (reverse of _resolveNodeConfig) ──────────

function _restoreNodeConfig(type, config) {
  const cfg = { ...config };

  if (type === "llm_call" && cfg.prompt) {
    // Detect input_source: last line if it looks like {{...}}
    const lines = cfg.prompt.split("\n");
    const lastLine = lines[lines.length - 1].trim();
    if (/^\{\{.+\}\}$/.test(lastLine)) {
      cfg.input_source = lastLine;
      cfg.system_prompt = lines.slice(0, -1).join("\n").trim();
      // Remove trailing blank line
      if (cfg.system_prompt.endsWith("\n")) cfg.system_prompt = cfg.system_prompt.trimEnd();
    } else {
      cfg.input_source = "{{input}}";
      cfg.system_prompt = cfg.prompt;
    }
  }

  if (type === "web_search" && cfg.query) {
    cfg.input_source = cfg.query;
  }

  if (type === "memory_search" && cfg.query) {
    cfg.input_source = cfg.query;
  }

  if (type === "memory_save" && cfg.text) {
    cfg.input_source = cfg.text;
  }

  if (type === "condition" && cfg._data_ref) {
    cfg.input_source = cfg._data_ref;
  }
  if (type === "router" && cfg._data_ref) {
    cfg.input_source = cfg._data_ref;
  }

  return cfg;
}

// ── Import existing pipeline steps → Drawflow nodes ──────────────────────────

function _importPipelineSteps(steps) {
  const startId = _addStartNode();
  let prevId = startId;
  let prevOut = "output_1";
  const colSpacing = 260;
  const startX = 80;
  const startY = 200;

  steps.forEach((step, i) => {
    if (step.type === "start" || step.type === "result") return;
    const restoredConfig = _restoreNodeConfig(step.type, step.config || {});
    const x = startX + (i + 1) * colSpacing;
    const nodeId = _addBlock(step.type, x, startY, restoredConfig);
    _editor.addConnection(prevId, nodeId, prevOut, "input_1");
    prevId = nodeId;
    prevOut = "output_1";
  });

  const resultId = _addResultNode();
  _editor.addConnection(prevId, resultId, prevOut, "input_1");
}

// ── Source picker helpers ─────────────────────────────────────────────────────

// Returns ordered list of upstream nodes that can feed data to this node.
// Each entry: { value: "{{input}}" | "{{sN.output}}", label: "Start (chat input)" | "Block N: Web search" }
function _getSourceOptions(currentNodeId) {
  if (!_editor) return [];
  const flow = _editor.export();
  const nodes = Object.values(flow.drawflow.Home.data);

  const opts = [{ value: "{{input}}", label: "Start (chat input)" }];

  // Sort by drawflow id so numbering is stable
  nodes
    .filter(n => n.name !== "start" && n.name !== "result" && String(n.id) !== String(currentNodeId))
    .sort((a, b) => a.id - b.id)
    .forEach((n, idx) => {
      const b = _block(n.name);
      opts.push({
        value: `{{s${n.id}.output}}`,
        label: `${b.icon} ${n.data.label || b.label}`,
      });
    });

  return opts;
}

// Render a <select> for picking the data source (replaces {{s1.output}} in UI)
function _sourceSelect(currentNodeId, currentValue, dataKey) {
  const opts = _getSourceOptions(currentNodeId);
  // Try to detect what the current value maps to
  let selected = currentValue || "{{input}}";
  const optsHtml = opts.map(o =>
    `<option value="${_esc(o.value)}" ${o.value === selected ? "selected" : ""}>${_esc(o.label)}</option>`
  ).join("");
  return `<select class="pf-cfg-select pf-source-select" data-key="${dataKey}">${optsHtml}</select>`;
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
      ${ready ? "" : `<button type="button" class="btn btn-outline btn-sm pf-open-settings">Open Settings</button>`}
    </div>`;
}

function _bindHttpAuthReadinessActions(root) {
  root.querySelectorAll(".pf-open-settings").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelector('.nav-item[data-view="settings"]')?.click();
    });
  });
}

function _refreshHttpAuthReadiness(root) {
  const slot = root.querySelector("[data-http-auth-readiness]");
  const select = root.querySelector('[data-key="auth_secret_key"]');
  if (!slot || !select) return;
  slot.outerHTML = _httpAuthReadinessHtml(select.value);
  _bindHttpAuthReadinessActions(root);
}

// ── Node config panel ─────────────────────────────────────────────────────────

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

function _openNodeConfig(id) {
  _selectedNodeId = id;
  const node = _editor?.getNodeFromId(id);
  if (!node) return;

  const panel = document.getElementById("pf-config-panel");
  const title = document.getElementById("pf-config-title");
  const body = document.getElementById("pf-config-body");

  if (node.name === "start" || node.name === "result") {
    panel.classList.add("hidden");
    return;
  }

  const b = _block(node.name);
  title.innerHTML = `${b.icon} ${b.label}`;
  body.innerHTML = _buildConfigForm(node.name, node.data, id);
  body.querySelector(".pf-del-node-btn")?.insertAdjacentHTML("afterend", _blockHelpHtml(node.name));
  panel.classList.remove("hidden");

  // Bind delete button immediately after render
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

  // Bind save-on-change
  body.querySelectorAll("input,textarea,select").forEach(el => {
    el.addEventListener("input", () => _saveNodeConfig(id));
    el.addEventListener("change", () => _saveNodeConfig(id));
  });
  _bindHttpAuthReadinessActions(body);
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

function _buildConfigForm(type, data, nodeId) {
  let html = `
    <div class="pf-cfg-row">
      <label class="pf-cfg-label">Block name</label>
      <input class="pf-cfg-input" data-key="label" value="${_esc(data.label || "")}">
    </div>`;

  if (type === "llm_call") {
    const modelOpts = _modelOptions(data.model || "");
    const srcSelect = _sourceSelect(nodeId, data.input_source || "{{input}}", "input_source");
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Model</label>
        <select class="pf-cfg-select" data-key="model">${modelOpts}</select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Input data</label>
        ${srcSelect}
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">System prompt <span style="opacity:.6">(optional)</span></label>
        <textarea class="pf-cfg-textarea" data-key="system_prompt" rows="4"
          placeholder="Role or instruction for AI. Leave empty to just pass the text through.">${_esc(data.system_prompt || "")}</textarea>
      </div>`;
  }

  if (type === "web_search") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Number of results</label>
        <input class="pf-cfg-input" type="number" min="1" max="10" data-key="num_results" value="${data.num_results || 5}" style="width:80px">
      </div>`;
  }

  if (type === "page_scrape") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Page URL</label>
        <input class="pf-cfg-input" data-key="url" value="${_esc(data.url || "")}" placeholder="https://example.com/report">
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
        <label class="pf-cfg-label">Number of results</label>
        <input class="pf-cfg-input" type="number" min="1" max="20" data-key="limit" value="${data.limit || 5}" style="width:80px">
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
    const srcSelect = _sourceSelect(nodeId, data.input_source || "{{input}}", "input_source");
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">What to save</label>
        ${srcSelect}
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Tags <span style="opacity:.6">(comma-separated)</span></label>
        <input class="pf-cfg-input" data-key="tags" value="${_esc(data.tags || "pipeline")}">
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
        <input class="pf-cfg-input" data-key="url" value="${_esc(data.url || "")}" placeholder="https://…">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Method</label>
        <select class="pf-cfg-select" data-key="method">
          <option ${(data.method||"GET")==="GET"?"selected":""}>GET</option>
          <option ${data.method==="POST"?"selected":""}>POST</option>
        </select>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Request body <span style="opacity:.6">(for POST)</span></label>
        <textarea class="pf-cfg-textarea" data-key="body" rows="3">${_esc(data.body || "")}</textarea>
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
    const srcSelect = _sourceSelect(nodeId, data.input_source || "{{input}}", "input_source");
    const routes = _routerRoutes(data);
    const mode = data.mode || "all_matching";
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Route using data from</label>
        ${srcSelect}
      </div>
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
    const srcSelect = _sourceSelect(nodeId, data.input_source || "{{prev}}", "input_source");
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Filter data from</label>
        ${srcSelect}
      </div>
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
        <label class="pf-cfg-label">Find / pattern</label>
        <input class="pf-cfg-input" data-key="pattern" value="${_esc(data.pattern || "")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Find text</label>
        <input class="pf-cfg-input" data-key="find" value="${_esc(data.find || "")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Replace / separator</label>
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

  if (type === "condition") {
    const srcSelect = _sourceSelect(nodeId, data.input_source || "{{input}}", "input_source");
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Check data from</label>
        ${srcSelect}
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Condition <span style="opacity:.6">(when is it "True"?)</span></label>
        <textarea class="pf-cfg-textarea" data-key="condition" rows="3"
          placeholder="e.g. search results contain relevant information">${_esc(data.condition || "")}</textarea>
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Top output (True ✓)</label>
        <input class="pf-cfg-input" data-key="description_true" value="${_esc(data.description_true || "True")}">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Bottom output (False ✗)</label>
        <input class="pf-cfg-input" data-key="description_false" value="${_esc(data.description_false || "False")}">
      </div>`;
  }

  if (type === "loop") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Max iterations</label>
        <input class="pf-cfg-input" type="number" min="1" max="20" data-key="max_iterations" value="${data.max_iterations || 5}" style="width:80px">
      </div>
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Stop condition <span style="opacity:.6">(when to stop the loop)</span></label>
        <textarea class="pf-cfg-textarea" data-key="stop_condition" rows="3"
          placeholder="e.g. enough information has been gathered to answer">${_esc(data.stop_condition || "")}</textarea>
      </div>`;
  }

  if (type === "template") {
    html += `
      <div class="pf-cfg-row">
        <label class="pf-cfg-label">Text</label>
        <textarea class="pf-cfg-textarea" data-key="text" rows="6"
          placeholder="Enter template text…">${_esc(data.text || "")}</textarea>
      </div>`;
  }

  html += `<button class="btn btn-danger btn-sm pf-del-node-btn" style="margin-top:1rem;width:100%">🗑 Delete block</button>`;
  return html;
}

function _saveNodeConfig(id) {
  if (!_editor) return;
  const body = document.getElementById("pf-config-body");
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

  // Refresh preview inside the node
  const nodeEl = document.querySelector(`#node-${id} .pf-node-preview`);
  if (nodeEl) nodeEl.textContent = _nodePreview(node.name, newData);
  const labelEl = document.querySelector(`#node-${id} .pf-node-label`);
  if (labelEl) labelEl.textContent = newData.label || _block(node.name).label;
}

function _closeConfig() {
  _selectedNodeId = null;
  document.getElementById("pf-config-panel")?.classList.add("hidden");
}

// ── Model list ────────────────────────────────────────────────────────────────

function _bindConfigResize() {
  const panel = document.getElementById("pf-config-panel");
  const handle = document.getElementById("pf-config-resize");
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

async function _fetchModels() {
  _availableModels = [];
  _availableSecrets = [];

  const [s, res, reg, secrets] = await Promise.all([
    fetch("/api/settings").then(r => r.json()).catch(() => null),
    fetch("/api/models").then(r => r.json()).catch(() => null),
    fetch("/api/model-registry").then(r => r.json()).catch(() => null),
    fetch("/api/secrets").then(r => r.json()).catch(() => null),
  ]);

  _availableSecrets = (secrets?.secrets || []).filter(secret => secret.configured);

  // 1. Current model from settings (always first)
  if (s?.summary_model) {
    _availableModels.push({ id: s.summary_model, label: `${s.summary_model} · current` });
  }

  // 2. All available models from all providers
  if (res?.models?.length) {
    for (const m of res.models) {
      if (!_availableModels.find(x => x.id === m.name)) {
        _availableModels.push({ id: m.name, label: m.label || m.name });
      }
    }
  }

  // 3. Custom-registered models from model registry
  if (reg?.models?.length) {
    for (const m of reg.models) {
      const id = m.model_name || m.name;
      if (id && !_availableModels.find(x => x.id === id)) {
        _availableModels.push({ id, label: `${id} · custom` });
      }
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
  let html = `<option value="" ${!current ? "selected" : ""}>— Current model (from settings) —</option>`;

  // Group by provider label hint
  const groups = {};
  for (const m of _availableModels) {
    if (!m.id) continue;
    // Detect group from label suffix "· Gemini", "· Ollama", etc.
    let group = "Other";
    const lbl = m.label || "";
    if (lbl.includes("Gemini") || lbl.includes("gemini")) group = "Google Gemini";
    else if (lbl.includes("GPT") || lbl.includes("gpt") || lbl.includes("OpenAI")) group = "OpenAI";
    else if (lbl.includes("Claude") || lbl.includes("claude") || lbl.includes("Anthropic")) group = "Anthropic";
    else if (lbl.includes("Ollama") || m.id.startsWith("ollama:")) group = "Ollama (local)";
    else if (lbl.includes("OpenRouter") || lbl.includes("FREE") || lbl.includes("openrouter")) group = "OpenRouter";
    else if (lbl.includes("DeepSeek") || lbl.includes("Grok") || lbl.includes("Mistral")) group = "Other AI";
    else if (lbl.includes("custom")) group = "Custom";
    else if (lbl.includes("current")) { group = "Current"; }

    if (!groups[group]) groups[group] = [];
    groups[group].push(m);
  }

  // Current model first (if not already in list)
  if (current && !_availableModels.find(m => m.id === current)) {
    html += `<option value="${_esc(current)}" selected>${_esc(current)} · current</option>`;
  }

  // Render grouped options
  const groupOrder = ["Current", "Google Gemini", "OpenAI", "Anthropic", "Ollama (local)", "OpenRouter", "Other AI", "Custom", "Other"];
  for (const gname of groupOrder) {
    const items = groups[gname];
    if (!items?.length) continue;
    html += `<optgroup label="${gname}">`;
    for (const m of items) {
      html += `<option value="${_esc(m.id)}" ${m.id === current ? "selected" : ""}>${_esc(m.label)}</option>`;
    }
    html += `</optgroup>`;
  }

  return html;
}

// ── Resolve input_source → actual config fields before saving ─────────────────

function _resolveNodeConfig(type, data) {
  const cfg = { ...data };
  const src = cfg.input_source || "{{input}}";

  if (type === "llm_call") {
    // Build prompt from system_prompt + input_source reference
    const sysPart = (cfg.system_prompt || "").trim();
    cfg.prompt = sysPart
      ? `${sysPart}\n\n${src}`
      : src;
    // Keep raw fields for round-trip restore
  }

  if (type === "web_search") {
    cfg.query = src;
  }

  if (type === "memory_search") {
    cfg.query = src;
  }

  if (type === "memory_save") {
    cfg.text = src;
  }

  if (type === "condition") {
    // Prepend data reference to condition text for the evaluator
    const condText = (cfg.condition || "").trim();
    cfg._data_ref = src;  // runner will substitute before evaluation
  }
  if (type === "router") {
    cfg._data_ref = src;
  }
  if (type === "filter") {
    cfg._data_ref = src;
  }

  return cfg;
}

// ── Save pipeline ─────────────────────────────────────────────────────────────

function _pipelinePayloadFromCanvas(nameOverride = null) {
  if (!_editor || !_pipelineMeta) return null;
  const name = (nameOverride || document.getElementById("pf-title-input")?.value || "").trim() || "Untitled";
  const flow = _editor.export();
  const nodes = Object.values(flow.drawflow.Home.data);
  const steps = nodes
    .filter(n => n.name !== "start" && n.name !== "result")
    .map(n => ({
      id: `s${n.id}`,
      type: n.name,
      label: n.data.label || _block(n.name).label,
      config: _resolveNodeConfig(n.name, n.data),
      _df_id: n.id,
      _inputs: n.inputs,
      _connections: n.outputs,
    }));
  return {
    name,
    description: _pipelineMeta.description || "",
    source_template_id: _pipelineMeta.source_template_id || "",
    source_template_name: _pipelineMeta.source_template_name || "",
    steps,
    drawflow_data: flow,
  };
}

async function _savePipelineAsTemplate() {
  if (!_editor || !_pipelineMeta) return false;
  const current = document.getElementById("pf-title-input")?.value?.trim() || _pipelineMeta.name || "Pipeline template";
  const name = prompt("Template name:", current);
  if (name === null) return false;
  const payload = _pipelinePayloadFromCanvas(name);
  if (!payload) return false;
  const res = await fetch("/api/pipelines/templates", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(r => r.json().then(data => ({ ok: r.ok, data }))).catch(() => null);
  if (res?.ok && res.data?.template) {
    _showToast(`Template "${res.data.template.name}" saved`);
    await _renderPipelineList();
    return true;
  }
  alert(_apiError(res?.data, "Template save failed"));
  return false;
}

async function _savePipeline() {
  if (!_editor || !_pipelineMeta) return false;
  const name = document.getElementById("pf-title-input")?.value?.trim() || "Untitled";
  _pipelineMeta.name = name;

  const flow = _editor.export();
  const nodes = Object.values(flow.drawflow.Home.data);

  // Convert Drawflow nodes to our step format
  const steps = nodes
    .filter(n => n.name !== "start" && n.name !== "result")
    .map(n => ({
      id: `s${n.id}`,
      type: n.name,
      label: n.data.label || _block(n.name).label,
      config: _resolveNodeConfig(n.name, n.data),
      _df_id: n.id,
      _inputs: n.inputs,
      _connections: n.outputs,
    }));

  const body = {
    id: _pipelineMeta.id || undefined,
    name: _pipelineMeta.name,
    description: _pipelineMeta.description,
    source_template_id: _pipelineMeta.source_template_id || "",
    source_template_name: _pipelineMeta.source_template_name || "",
    steps,
    drawflow_data: flow,
  };

  const saved = await fetch("/api/pipelines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(r => r.json()).catch(() => null);

  if (saved?.id) {
    _pipelineMeta.id = saved.id;
    _showToast(`"${saved.name}" saved`);
    return true;
  } else {
    alert("Save failed. Please try again.");
    return false;
  }
}

// ── Run ───────────────────────────────────────────────────────────────────────

async function _promptRun() {
  if (!_pipelineMeta?.id) {
    alert("Save the pipeline first.");
    return;
  }
  const input = prompt("Enter your message:", "");
  if (input === null) return;
  if (!await _confirmPipelinePreflight(_pipelineMeta.id)) return;
  _openRunModal(_pipelineMeta.name);
  _startRun(_pipelineMeta.id, input);
}

async function _confirmPipelinePreflight(pipelineId) {
  const report = await fetch(`/api/pipelines/${encodeURIComponent(pipelineId)}/preflight`)
    .then(r => r.json())
    .catch(() => null);
  return _confirmSafetyPreflight(report, "Pipeline safety preflight");
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

function _openRunModal(name) {
  document.getElementById("pf-run-title").textContent = name || "Running";
  const stepsEl = document.getElementById("pf-run-steps");
  if (stepsEl) {
    stepsEl.classList.remove("hidden");
    stepsEl.innerHTML = "";
  }
  document.getElementById("pf-run-final").classList.add("hidden");
  document.getElementById("pf-run-modal").classList.remove("hidden");
  _clearNodeRunBadges();
}

function _closeRunModal() {
  _runAbort?.abort();
  document.getElementById("pf-run-modal").classList.add("hidden");
}

async function _startRun(pipelineId, inputText) {
  _runAbort = new AbortController();
  _currentRunTrace = [];
  try {
    const resp = await fetch("/api/pipelines/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pipeline_id: pipelineId, input_text: inputText }),
      signal: _runAbort.signal,
    });

    const reader = resp.body.getReader();
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
        try { _handleRunEvent(JSON.parse(line.slice(5).trim())); } catch (_) {}
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") console.error("Run error:", err);
  }
}

function _handleRunEvent(evt) {
  const stepsEl = document.getElementById("pf-run-steps");

  if (evt.type === "run_started") {
    const title = document.getElementById("pf-run-title");
    if (title && evt.run_id) title.textContent = `${_pipelineMeta?.name || "Pipeline"} · ${evt.run_id}`;
    return;
  }

  if (evt.type === "start") {
    if (stepsEl) {
      stepsEl.classList.remove("hidden");
      stepsEl.innerHTML = `<div class="pf-run-step pf-run-step-running">Running pipeline...</div>`;
    }
    return;
  }
  if (evt.type === "step_start") {
    _recordRunStep({ ...evt, status: "running" });
    _applyPipelineStepEventToCanvas(evt, "running");
    return;
  }
  if (evt.type === "step_done") {
    _recordRunStep({ ...evt, status: "ok" });
    _applyPipelineStepEventToCanvas(evt, "ok");
    return;
  }
  if (evt.type === "step_error") {
    _recordRunStep({ ...evt, status: "error" });
    _applyPipelineStepEventToCanvas(evt, "error");
    return;
  }
  if (evt.type === "done") {
    if (stepsEl) stepsEl.classList.add("hidden");
    const finalEl = document.getElementById("pf-run-final");
    finalEl.classList.remove("hidden");
    document.getElementById("pf-run-final-text").textContent = evt.output || "";
    _recordResultStep(evt.output || "");
    _applyResultBadge(evt.output || "");

    const copyBtn = document.getElementById("pf-copy-btn");
    const chatBtn = document.getElementById("pf-to-chat-btn");
    copyBtn.onclick = () => navigator.clipboard.writeText(evt.output || "").then(() => _showToast("Copied"));
    chatBtn.onclick = () => {
      const ci = document.getElementById("chat-input");
      if (ci) { ci.value = evt.output || ""; ci.dispatchEvent(new Event("input")); }
      document.querySelector('.nav-item[data-view="chat"]')?.click();
      _closeRunModal();
    };
    return;
  }

  if (evt.type === "start") {
    stepsEl.innerHTML = "";
  }
  if (evt.type === "step_start") {
    const div = document.createElement("div");
    div.className = "pf-run-step pf-run-running";
    div.id = `pf-rs-${evt.index}`;
    div.innerHTML = `<span class="pf-rs-icon">⏳</span><span class="pf-rs-label">${_esc(evt.label)}</span>`;
    stepsEl.appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  if (evt.type === "step_done") {
    const div = document.getElementById(`pf-rs-${evt.index}`);
    if (div) {
      div.className = "pf-run-step pf-run-done";
      div.innerHTML = `<span class="pf-rs-icon">✅</span><span class="pf-rs-label">${_esc(evt.label)}</span>
        <div class="pf-rs-output">${_esc((evt.output || "").slice(0, 200))}${(evt.output||"").length > 200 ? "…" : ""}</div>`;
    }
  }
  if (evt.type === "step_error") {
    const div = document.getElementById(`pf-rs-${evt.index}`);
    if (div) {
      div.className = "pf-run-step pf-run-error";
      div.innerHTML = `<span class="pf-rs-icon">❌</span><span class="pf-rs-label">${_esc(evt.label)}</span>
        <div class="pf-rs-output" style="color:#f87171">${_esc(evt.error || "")}</div>`;
    }
  }
  if (evt.type === "done") {
    const finalEl = document.getElementById("pf-run-final");
    finalEl.classList.remove("hidden");
    document.getElementById("pf-run-final-text").textContent = evt.output || "";

    const copyBtn = document.getElementById("pf-copy-btn");
    const chatBtn = document.getElementById("pf-to-chat-btn");
    copyBtn.onclick = () => navigator.clipboard.writeText(evt.output || "").then(() => _showToast("Copied"));
    chatBtn.onclick = () => {
      const ci = document.getElementById("chat-input");
      if (ci) { ci.value = evt.output || ""; ci.dispatchEvent(new Event("input")); }
      document.querySelector('.nav-item[data-view="chat"]')?.click();
      _closeRunModal();
    };
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

async function _openPipelineHistory() {
  if (!_pipelineMeta?.id) {
    alert("Save the pipeline before viewing history.");
    return;
  }
  const modal = document.getElementById("pf-history-modal");
  const body = document.getElementById("pf-history-body");
  if (!modal || !body) return;
  modal.classList.remove("hidden");
  body.innerHTML = `<div class="pf-run-step pf-run-step-running">Loading history...</div>`;

  const [res, memoryReport] = await Promise.all([
    fetch(`/api/pipelines/${_pipelineMeta.id}/runs`).then(r => r.json()).catch(() => ({ runs: [] })),
    fetch(`/api/pipelines/${_pipelineMeta.id}/memory-report`).then(r => r.json()).catch(() => null),
  ]);
  const runs = res.runs || [];
  if (!runs.length) {
    body.innerHTML = `${_renderMemoryReportSummary(memoryReport)}<div class="pf-run-step">No runs yet.</div>`;
    _bindMemoryQuickFixes(body);
    return;
  }
  body.innerHTML = `${_renderMemoryReportSummary(memoryReport)}${runs.map(run => `
    <button type="button" class="pf-run-step pf-history-run" data-run-id="${_esc(run.run_id)}" style="width:100%;text-align:left">
      <strong>${_esc(run.status || "unknown")}</strong>
      <span style="opacity:.72;margin-left:.5rem">${_esc(run.started_at || "")}</span>
      ${run.memory_evaluation?.score !== undefined ? `<span style="opacity:.72;margin-left:.5rem">Memory ${_esc(String(run.memory_evaluation.score))}/100</span>` : ""}
      <div class="pf-rs-output">${_esc(run.error || run.output_preview || "(no output)")}</div>
    </button>`).join("")}`;
  _bindMemoryQuickFixes(body);

  body.querySelectorAll(".pf-history-run").forEach(btn => {
    btn.addEventListener("click", async () => {
      const runId = btn.dataset.runId;
      const run = await fetch(`/api/pipelines/${_pipelineMeta.id}/runs/${runId}`).then(r => r.json()).catch(() => null);
      if (!run) return;
      body.innerHTML = `
        <button type="button" class="btn btn-outline btn-sm" id="pf-history-back">← Runs</button>
        <button type="button" class="btn btn-primary btn-sm" id="pf-history-rerun" style="margin-left:.5rem">Run again</button>
        <button type="button" class="btn btn-outline btn-sm" id="pf-history-copy-output" style="margin-left:.5rem">Copy output</button>
        <div class="pf-run-step">
          <strong>${_esc(run.status || "unknown")}</strong>
          <span style="opacity:.72;margin-left:.5rem">${_esc(run.run_id || "")}</span>
          <div style="opacity:.72;margin-top:.35rem">${_esc(run.started_at || "")} · ${run.duration_ms ?? "?"} ms</div>
        </div>
        ${_renderMemoryEvaluation(run.memory_evaluation)}
        ${_renderWorkflowRunTrace(run.trace || [])}
        <details class="pf-run-step" open>
          <summary>Final output</summary>
          <pre class="pf-run-step-output">${_esc(run.output || run.error || "(no output)")}</pre>
        </details>`;
      body.querySelector("#pf-history-back")?.addEventListener("click", _openPipelineHistory);
      body.querySelector("#pf-history-rerun")?.addEventListener("click", () => {
        document.getElementById("pf-history-modal")?.classList.add("hidden");
        _openRunModal(_pipelineMeta?.name || "Pipeline");
        _startRun(_pipelineMeta.id, run.input || "");
      });
      body.querySelector("#pf-history-copy-output")?.addEventListener("click", () => {
        navigator.clipboard?.writeText(run.output || run.error || "").then(() => _showToast("Copied"));
      });
    });
  });
}

function _renderWorkflowRunTrace(trace) {
  if (!trace?.length) return `<div class="pf-run-step">No step trace recorded.</div>`;
  return trace.map(step => {
    const ok = step.status === "ok";
    const body = ok ? (step.output || "(no output)") : (step.error || "Step failed");
    const suffix = step.output_truncated ? `\n\n[Step output truncated: ${step.output_length} characters total]` : "";
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
    `<button type="button" class="btn btn-primary btn-sm pf-memory-verify">Run to verify</button>`,
  ];
  const needsRepair = (
    (totals.missed_search_before_save_count || 0) > 0
    || (totals.duplicate_save_candidate_count || 0) > 0
    || (totals.empty_search_count || 0) > 0
  );
  if (needsRepair) {
    buttons.push(`<button type="button" class="btn btn-outline btn-sm pf-memory-fix" data-fix="auto"
      data-add-search="${(totals.missed_search_before_save_count || 0) > 0 ? "1" : "0"}"
      data-dedup-save="${(totals.duplicate_save_candidate_count || 0) > 0 ? "1" : "0"}"
      data-skip-empty="${(totals.empty_search_count || 0) > 0 ? "1" : "0"}">Auto repair</button>`);
  }
  return buttons.length
    ? `<div style="display:flex;gap:.45rem;flex-wrap:wrap;margin-top:.75rem">${buttons.join("")}</div>`
    : "";
}

function _bindMemoryQuickFixes(body) {
  body.querySelector(".pf-memory-verify")?.addEventListener("click", async () => {
    if (!_pipelineMeta?.id) return;
    const input = prompt(`Enter input for "${_pipelineMeta.name || "Pipeline"}":`, "");
    if (input === null) return;
    document.getElementById("pf-history-modal")?.classList.add("hidden");
    _openRunModal(_pipelineMeta?.name || "Pipeline");
    await _startRun(_pipelineMeta.id, input);
    document.getElementById("pf-run-modal")?.classList.add("hidden");
    await _renderPipelineList();
    await _openPipelineHistory();
  });
  body.querySelectorAll(".pf-memory-fix").forEach(btn => {
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
  setTimeout(_ensureNodeActionButtons, 0);
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
  const saved = await _savePipeline();
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

function _clearNodeRunBadges() {
  _lastRunTraceByStepId = new Map();
  _currentRunTrace = [];
  document.querySelectorAll("#pf-drawflow .pf-node-run-badge").forEach(el => el.remove());
}

function _recordRunStep(evt) {
  const id = String(evt.id || `idx-${evt.index}`);
  const existing = _lastRunTraceByStepId.get(id) || {};
  const item = {
    ...existing,
    id,
    index: evt.index,
    type: evt.step_type || evt.type || "step",
    label: evt.label || existing.label || "Step",
    status: evt.status,
    output: evt.output ?? existing.output ?? "",
    error: evt.error ?? existing.error ?? "",
  };
  _lastRunTraceByStepId.set(id, item);
  _currentRunTrace = [..._lastRunTraceByStepId.values()];
}

function _recordResultStep(output) {
  const item = {
    id: "result",
    index: _currentRunTrace.length + 1,
    type: "result",
    label: "Result",
    status: "ok",
    output,
    error: "",
  };
  _lastRunTraceByStepId.set("result", item);
}

function _applyPipelineStepEventToCanvas(evt, status) {
  const id = String(evt.id || `idx-${evt.index}`);
  const nodeEl = _nodeElementForPipelineStep(evt);
  _setNodeRunBadge(nodeEl, id, status);
}

function _applyResultBadge(output) {
  const node = _findNodeByName("result");
  const nodeEl = node ? document.getElementById(`node-${node.id}`) : null;
  _setNodeRunBadge(nodeEl, "result", "ok");
}

function _setNodeRunBadge(nodeEl, stepId, status) {
  const inner = nodeEl?.querySelector?.(".pf-node-inner");
  if (!inner) return;
  inner.querySelector(".pf-node-run-badge")?.remove();

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = `pf-node-run-badge ${status === "ok" ? "pf-node-run-ok" : status === "error" ? "pf-node-run-error" : "pf-node-run-running"}`;
  btn.dataset.stepId = stepId;
  btn.title = status === "ok" ? "View block result" : status === "error" ? "View block error" : "Block is running";
  btn.textContent = status === "ok" ? "✓" : status === "error" ? "!" : "...";
  inner.appendChild(btn);
}

function _nodeElementForPipelineStep(evt) {
  const id = String(evt.id || "");
  if (id.startsWith("s")) {
    const el = document.getElementById(`node-${id.slice(1)}`);
    if (el) return el;
  }

  const nodes = _orderedProcessingNodes();
  const fallback = nodes[Number(evt.index)];
  return fallback ? document.getElementById(`node-${fallback.id}`) : null;
}

function _orderedProcessingNodes() {
  try {
    const nodes = Object.values(_editor?.export?.()?.drawflow?.Home?.data || {});
    return nodes
      .filter(n => n.name !== "start" && n.name !== "result")
      .sort((a, b) => Number(a.id) - Number(b.id));
  } catch (_) {
    return [];
  }
}

function _findNodeByName(name) {
  try {
    const nodes = Object.values(_editor?.export?.()?.drawflow?.Home?.data || {});
    return nodes.find(n => n.name === name) || null;
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
  const running = step.status === "running";
  const body = running ? "Block is still running." : ok ? (step.output || "(no output)") : (step.error || "Step failed");

  const title = document.getElementById("pf-step-title");
  const meta = document.getElementById("pf-step-meta");
  const output = document.getElementById("pf-step-output");
  const pinBtn = document.getElementById("pf-step-pin");
  if (title) title.textContent = `${running ? "RUNNING" : ok ? "OK" : "ERROR"} ${step.index ?? ""}. ${step.label || step.type || "Block"}`.trim();
  if (meta) {
    meta.innerHTML = `
      <span class="pf-step-status ${ok ? "pf-step-status-ok" : running ? "pf-step-status-running" : "pf-step-status-error"}">${running ? "Running" : ok ? "Success" : "Error"}</span>
      <span>${_esc(step.type || "block")}</span>`;
  }
  if (output) output.textContent = body;
  if (pinBtn) {
    pinBtn.dataset.stepId = step.id || "";
    pinBtn.disabled = !ok || !String(step.id || "").startsWith("s");
    pinBtn.textContent = pinBtn.disabled ? "Pin result" : "Pin result";
  }
  document.getElementById("pf-step-modal")?.classList.remove("hidden");
}

function _pinCurrentStepResult() {
  const btn = document.getElementById("pf-step-pin");
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
  if (preview) preview.textContent = `Pinned · ${_nodePreview(node.name, data)}`;
  _showToast("Pinned. Save the pipeline to keep it.");
}

function _hideStepResultModal() {
  document.getElementById("pf-step-modal")?.classList.add("hidden");
}

function _esc(s) {
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function _apiError(data, fallback) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail?.errors)) return detail.errors.join("\n");
  if (Array.isArray(detail)) return detail.map(item => item.msg || JSON.stringify(item)).join("\n");
  return data?.error || fallback;
}

function _showToast(msg) {
  const t = Object.assign(document.createElement("div"), { className: "pl-toast", textContent: msg });
  document.body.appendChild(t);
  setTimeout(() => t.classList.add("pl-toast-show"), 10);
  setTimeout(() => { t.classList.remove("pl-toast-show"); setTimeout(() => t.remove(), 300); }, 2500);
}
