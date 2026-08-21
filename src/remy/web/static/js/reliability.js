const STATUS_ORDER = {
  needs_attention: 0,
  watch: 1,
  ok: 2,
  no_data: 3,
};

export async function loadReliability() {
  const root = document.getElementById("reliability-content");
  if (!root) return;
  root.innerHTML = `<div class="pf-loading">Loading reliability overview...</div>`;
  const refresh = document.getElementById("btn-reliability-refresh");
  if (refresh) refresh.onclick = loadReliability;

  const [pipelineData, automationData] = await Promise.all([
    _loadPipelineItems(),
    _loadAutomationItems(),
  ]);
  const items = [...pipelineData.items, ...automationData.items];
  const hydrated = await _withReports(items);
  _renderReliability(root, hydrated, pipelineData.scope || automationData.scope || {});
  _bindReliabilityActions(root);
}

async function _loadPipelineItems() {
  const data = await fetch("/api/pipelines").then(r => r.json()).catch(() => ({ pipelines: [] }));
  return { scope: data.scope || {}, items: (data.pipelines || []).map(item => ({
    kind: "pipeline",
    id: item.id,
    name: item.name || "Untitled pipeline",
    description: item.description || "",
    stepCount: item.step_count ?? 0,
  })) };
}

async function _loadAutomationItems() {
  const data = await fetch("/api/automations").then(r => r.json()).catch(() => ({ automations: [] }));
  return { scope: data.scope || {}, items: (data.automations || []).map(item => ({
    kind: "automation",
    id: item.id,
    name: item.name || "Untitled automation",
    description: item.description || "",
    stepCount: (item.steps || []).length,
    enabled: item.enabled !== false,
    lastRunStatus: item.last_run_status || "",
    failureCount: item.consecutive_failures || 0,
  })) };
}

async function _withReports(items) {
  const rows = await Promise.all(items.map(async item => {
    const report = await fetch(`/api/${item.kind === "pipeline" ? "pipelines" : "automations"}/${encodeURIComponent(item.id)}/memory-report`)
      .then(r => r.json())
      .catch(() => null);
    return { ...item, report: report || { status: "no_data" } };
  }));
  return rows.sort((a, b) => {
    const aStatus = STATUS_ORDER[a.report?.status || "no_data"] ?? 9;
    const bStatus = STATUS_ORDER[b.report?.status || "no_data"] ?? 9;
    if (aStatus !== bStatus) return aStatus - bStatus;
    return _score(a) - _score(b);
  });
}

function _renderReliability(root, items, scope) {
  const summary = _summary(items);
  root.innerHTML = `
    <div class="stats-scope-bar">
      <div>
        <span class="stats-scope-kicker">Project reliability</span>
        <strong>${_esc(scope.project_name || "Active project")}</strong>
      </div>
      <span>Only this project's pipelines, automations, and run memory are included.</span>
    </div>
    <section class="reliability-summary">
      ${_summaryTile("Workflows", summary.total)}
      ${_summaryTile("Need Attention", summary.needsAttention)}
      ${_summaryTile("Watch", summary.watch)}
      ${_summaryTile("Average Memory", summary.averageScore === null ? "No data" : `${summary.averageScore}/100`)}
    </section>
    <section class="reliability-panel">
      <div class="reliability-panel-header">
        <div>
          <h3>Workflow Reliability</h3>
          <p>Memory discipline across saved pipelines and automations. Open History to apply the repair actions already available in the builder.</p>
        </div>
      </div>
      <div class="reliability-list">
        ${items.length ? items.map(_workflowRow).join("") : `<div class="reliability-empty">No saved workflows yet.</div>`}
      </div>
    </section>
  `;
}

function _summary(items) {
  const scored = items.filter(item => Number.isFinite(item.report?.average_score));
  const totalScore = scored.reduce((sum, item) => sum + Number(item.report.average_score), 0);
  return {
    total: items.length,
    needsAttention: items.filter(item => item.report?.status === "needs_attention").length,
    watch: items.filter(item => item.report?.status === "watch").length,
    averageScore: scored.length ? Math.round(totalScore / scored.length) : null,
  };
}

function _summaryTile(label, value) {
  return `
    <div class="reliability-tile">
      <span>${_esc(label)}</span>
      <strong>${_esc(value)}</strong>
    </div>`;
}

function _workflowRow(item) {
  const report = item.report || {};
  const status = report.status || "no_data";
  const score = Number.isFinite(report.average_score) ? `${report.average_score}/100` : "No runs";
  const totals = report.totals || {};
  const rec = (report.top_recommendations || [])[0];
  const recommendation = rec ? (rec.text || rec) : _fallbackRecommendation(status);
  const trend = _trendBadge(report.trend);
  return `
    <article class="reliability-row reliability-row-${_esc(status)}">
      <div class="reliability-row-main">
        <div class="reliability-row-title">
          <span class="reliability-kind">${_esc(item.kind)}</span>
          <strong>${_esc(item.name)}</strong>
        </div>
        <div class="reliability-row-meta">
          <span>${_esc(item.stepCount)} ${item.stepCount === 1 ? "step" : "steps"}</span>
          ${item.kind === "automation" ? `<span>${item.enabled ? "enabled" : "paused"}</span>` : ""}
          ${item.lastRunStatus ? `<span>last: ${_esc(item.lastRunStatus)}</span>` : ""}
        </div>
        <p>${_esc(recommendation)}</p>
      </div>
      <div class="reliability-row-metrics">
        <span class="workflow-memory-badge workflow-memory-${_esc(status)}">${_esc(score)}</span>
        ${trend}
        <span>Runs ${_esc(report.evaluated_run_count || 0)}/${_esc(report.run_count || 0)}</span>
        <span>Empty ${_esc(totals.empty_search_count || 0)}</span>
        <span>Duplicates ${_esc(totals.duplicate_save_candidate_count || 0)}</span>
      </div>
      <div class="reliability-row-actions">
        <button class="btn btn-primary btn-sm reliability-open-history" type="button" data-kind="${_esc(item.kind)}" data-id="${_esc(item.id)}">Open History</button>
      </div>
    </article>`;
}

function _trendBadge(trend) {
  if (!trend || trend.delta === null || trend.delta === undefined) return "";
  const delta = Number(trend.delta || 0);
  const status = delta > 0 ? "ok" : delta < 0 ? "needs_attention" : "no_data";
  const sign = delta > 0 ? "+" : "";
  return `<span class="workflow-memory-badge workflow-memory-${status}" title="Latest run vs previous evaluated run">${_esc(trend.direction || "trend")} ${sign}${_esc(delta)}</span>`;
}

function _fallbackRecommendation(status) {
  if (status === "no_data") return "Run this workflow once to collect a reliability baseline.";
  if (status === "ok") return "No memory discipline issues detected in recent runs.";
  return "Review memory report and apply the suggested repair actions.";
}

function _bindReliabilityActions(root) {
  root.querySelectorAll(".reliability-open-history").forEach(button => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "Opening...";
      try {
        await _openWorkflowHistory(button.dataset.kind, button.dataset.id);
      } finally {
        button.disabled = false;
        button.textContent = "Open History";
      }
    });
  });
}

async function _openWorkflowHistory(kind, id) {
  if (kind === "pipeline") {
    document.querySelector('.nav-item[data-view="pipelines"]')?.click();
    const mod = await import("./pipelines.js?v=3.0");
    await mod.openPipelineMemoryHistory(id);
    return;
  }
  document.querySelector('.nav-item[data-view="automations"]')?.click();
  const mod = await import("./automations.js?v=2.9");
  await mod.openAutomationMemoryHistory(id);
}

function _score(item) {
  return Number.isFinite(item.report?.average_score) ? Number(item.report.average_score) : 101;
}

function _esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
