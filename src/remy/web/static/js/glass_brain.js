/**
 * Glass Brain — 3D cognitive heat map using ForceGraph3D (same engine as graph.js).
 * Loads ALL memory records from /api/graph, overlays thermal temperatures from
 * /api/glass-brain/belief-graph, then renders with the same colour gradient as
 * graph.js thermal mode: cold (blue) → warm (green) → hot (red).
 */

let _graph3d = null;
let _refreshTimer = null;
let _container = null;
let _activeTab = "graph";
let _fg3dLoaded = false;
let _lastGraphHadNodes = false;
const _REFRESH_INTERVAL_MS = 10 * 60 * 1000;

// ── Thermal colour gradient (identical to graph.js) ───────────────────────────

function _thermalColorHex(t) {
    const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
    const stops = [
        { p: 0.00, r: 0x38, g: 0xbd, b: 0xf8 },
        { p: 0.50, r: 0x34, g: 0xd3, b: 0x99 },
        { p: 1.00, r: 0xef, g: 0x44, b: 0x44 },
    ];
    let a = stops[0], b = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
        if (x >= stops[i].p && x <= stops[i + 1].p) { a = stops[i]; b = stops[i + 1]; break; }
    }
    const span = (b.p - a.p) || 1;
    const k = (x - a.p) / span;
    return (Math.round(a.r + (b.r - a.r) * k) << 16)
         | (Math.round(a.g + (b.g - a.g) * k) << 8)
         |  Math.round(a.b + (b.b - a.b) * k);
}

const _HEALTH_COLOR = {
    healthy:  "rgba(148,163,184,0.14)",
    weakened: "rgba(250,204,21,0.80)",
    pruned:   "rgba(239,68,68,0.70)",
};
const _HEALTH_WIDTH = { healthy: 0.4, weakened: 2.0, pruned: 2.4 };

// ── Entry points ─────────────────────────────────────────────────────────────

export async function loadGlassBrain() {
    _container = document.getElementById("view-glass-brain");
    if (!_container) return;
    _renderShell();
    await _loadAll();
    _startRefresh();
}

export function stopGlassBrainRefresh() {
    clearInterval(_refreshTimer);
    _refreshTimer = null;
    _destroyGraph();
}

// ── Shell ────────────────────────────────────────────────────────────────────

function _renderShell() {
    _container.innerHTML = `
        <div class="gb-shell">
            <div class="gb-header">
                <div class="gb-title-row">
                    <h2>Glass Brain</h2>
                    <span class="gb-subtitle">Cognitive heat map — memory graph coloured by thermal activity</span>
                </div>
                <div class="gb-tabs">
                    <button class="gb-tab active" data-tab="graph">Graph</button>
                    <button class="gb-tab" data-tab="thermal">Thermal</button>
                    <button class="gb-tab" data-tab="plasticity">Plasticity</button>
                </div>
            </div>
            <div class="gb-body">
                <div id="gb-panel-graph" class="gb-panel active">
                    <div id="gb-graph-container" class="gb-graph-container"></div>
                </div>
                <div id="gb-panel-thermal" class="gb-panel hidden">
                    <div id="gb-thermal-content" class="gb-info-content">
                        <div class="gb-loading">Loading thermal data…</div>
                    </div>
                </div>
                <div id="gb-panel-plasticity" class="gb-panel hidden">
                    <div id="gb-plasticity-content" class="gb-info-content">
                        <div class="gb-loading">Loading plasticity data…</div>
                    </div>
                </div>
            </div>
        </div>`;

    _container.querySelectorAll(".gb-tab").forEach(btn => {
        btn.addEventListener("click", () => {
            _activeTab = btn.dataset.tab;
            _container.querySelectorAll(".gb-tab").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            _container.querySelectorAll(".gb-panel").forEach(p => { p.classList.add("hidden"); p.classList.remove("active"); });
            const panel = document.getElementById(`gb-panel-${_activeTab}`);
            panel?.classList.remove("hidden");
            panel?.classList.add("active");
            if (_activeTab !== "graph") _destroyGraph();
            else _loadGraph();
        });
    });
}

// ── Load all tabs in parallel ─────────────────────────────────────────────────

async function _loadAll() {
    await Promise.all([_loadGraph(), _loadThermal(), _loadPlasticity()]);
}

// ── Graph tab — full memory graph + thermal overlay ──────────────────────────

async function _loadGraph(options = {}) {
    if (_activeTab !== "graph") return;
    const silent = Boolean(options.silent);
    const gc = document.getElementById("gb-graph-container");
    if (!gc) return;
    if (!silent && !_graph3d) {
        gc.innerHTML = `<div class="gb-loading" style="padding:40px;text-align:center">Loading belief graph…</div>`;
    }

    try {
        // Fetch both in parallel: full memory graph + thermal temperatures
        const [graphRes, thermalRes] = await Promise.all([
            fetch("/api/graph?mode=full"),
            fetch("/api/glass-brain/belief-graph"),
        ]);
        const graphData   = await graphRes.json();
        const thermalData = await thermalRes.json();

        const memNodes = graphData.nodes || [];
        const memEdges = graphData.edges || [];

        if (!memNodes.length) {
            if (_lastGraphHadNodes || _graph3d) {
                _showGraphNotice("Memory graph temporarily returned no records. Keeping the last visible graph.");
                return;
            }
            _showGraphEmpty("No memory records yet");
            return;
        }
        _lastGraphHadNodes = true;

        // Build temperature map: belief key → rescaled temp
        // beliefs.cog uses "key" field like "default:tag1,tag2:type"
        // We match by belief id → find nodes whose label contains the key
        const beliefNodes = thermalData.nodes || [];
        const rawTemps = beliefNodes.map(n => Number(n.temp) || 0);
        const tMax = rawTemps.length ? Math.max(...rawTemps) : 1;
        const tMin = rawTemps.length ? Math.min(...rawTemps) : 0;
        const span = Math.max(1e-4, tMax - tMin);
        const rescale = t => Math.pow(Math.max(0, Math.min(1, (t - tMin) / span)), 0.7);

        // Fetch plasticity summary for HUD stats
        let plasticityStats = { weakened: 0, pruned: 0 };
        try {
            const pRes = await fetch("/api/glass-brain/plasticity");
            const pData = await pRes.json();
            if (pData.summary) {
                plasticityStats = { weakened: pData.summary.weakened || 0, pruned: pData.summary.pruned || 0 };
            }
        } catch (_) {}

        // Build tag → max_temperature map from beliefs
        // Each belief key: "default:tag1,tag2,tag3:state" — extract tag segment
        const tagTempMap = new Map();
        for (const bn of beliefNodes) {
            const parts = (bn.key || "").split(":");
            const tagSegment = parts[1] || "";
            const tags = tagSegment.split(",").map(t => t.trim()).filter(Boolean);
            const temp = rescale(bn.temp);
            for (const tag of tags) {
                const prev = tagTempMap.get(tag) || 0;
                tagTempMap.set(tag, Math.max(prev, temp));
            }
        }

        // Assign temperature to each memory node via its tags
        const nodes = memNodes.map(n => {
            const nodeTags = (n.tags || []).map(t => String(t).trim().toLowerCase());
            let temp = 0;
            for (const tag of nodeTags) {
                const t = tagTempMap.get(tag) || 0;
                if (t > temp) temp = t;
            }
            const displayTemp = temp || 0.04; // cold nodes show blue
            return {
                id:        n.id,
                _label:    (n.label || "").slice(0, 60),
                _temp:     displayTemp,
                _level:    n.level || "",
                _tags:     n.tags || [],
                _strength: n.strength || 0.1,
                _color:    _thermalColorHex(displayTemp),
            };
        });

        // Cap edges at 2000 strongest
        const MAX_EDGES = 2000;
        const links = memEdges
            .sort((a, b) => (b.weight || 0) - (a.weight || 0))
            .slice(0, MAX_EDGES)
            .map(e => ({ source: e.source, target: e.target, _weight: e.weight || 0.1 }));

        const hotCount = nodes.filter(n => n._temp > 0.6).length;

        await _render3d({ nodes, links }, {
            hotCount,
            weakenedCount: plasticityStats.weakened,
            prunedCount:   plasticityStats.pruned,
        });
    } catch (e) {
        if (_lastGraphHadNodes || _graph3d) {
            _showGraphNotice("Refresh failed: " + e.message);
            return;
        }
        _showGraphEmpty("Failed to load: " + e.message);
    }
}

function _showGraphEmpty(msg) {
    _lastGraphHadNodes = false;
    _destroyGraph();
    const c = document.getElementById("gb-graph-container");
    if (c) c.innerHTML = `<div class="gb-empty">${_esc(msg)}</div>`;
}

function _showGraphNotice(msg) {
    const c = document.getElementById("gb-graph-container");
    if (!c) return;
    c.querySelector(".gb-refresh-notice")?.remove();
    c.insertAdjacentHTML("beforeend", `<div class="gb-refresh-notice">${_esc(msg)}</div>`);
    window.setTimeout(() => c.querySelector(".gb-refresh-notice")?.remove(), 5000);
}

async function _load3dLib() {
    if (_fg3dLoaded || typeof ForceGraph3D !== "undefined") { _fg3dLoaded = true; return; }
    await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "/js/vendor/3d-force-graph.min.js";
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
    });
    _fg3dLoaded = true;
}

async function _render3d({ nodes, links }, stats) {
    const gc = document.getElementById("gb-graph-container");
    if (!gc) return;

    await _load3dLib();
    _destroyGraph();
    gc.innerHTML = "";

    const W = gc.clientWidth  || 860;
    const H = gc.clientHeight || 620;

    _graph3d = ForceGraph3D({ antialias: true, alpha: true })(gc)
        .width(W)
        .height(H)
        .backgroundColor("#030b16")
        .nodeRelSize(3)
        .cooldownTicks(180)
        .d3AlphaDecay(0.025)
        .d3VelocityDecay(0.4)
        .warmupTicks(40)
        .onEngineStop((() => {
            let done = false;
            return () => { if (!done && _graph3d) { done = true; _graph3d.zoomToFit(400, 30); } };
        })())
        .nodeColor(n => "#" + n._color.toString(16).padStart(6, "0"))
        .nodeVal(n => {
            const base = { IDENTITY: 5, DOMAIN: 3, DECISIONS: 2 }[n._level] ?? 1.5;
            const heat = n._temp > 0.6 ? 1.5 : 1.0;
            return base * Math.max(0.3, n._strength) * heat;
        })
        .nodeOpacity(0.92)
        .nodeResolution(12)
        .nodeLabel(n => `${n._label} | Temp: ${(n._temp * 100).toFixed(0)}%`)
        .linkColor(l => {
            const w = l._weight || 0.1;
            const a = Math.max(0.06, Math.min(0.28, w * 0.35));
            return `rgba(148,163,184,${a})`;
        })
        .linkWidth(l => Math.max(0.15, (l._weight || 0.1) * 0.5))
        .linkOpacity(1)
        .linkCurvature(0.08)
        .linkDirectionalParticles(l => (l._weight || 0) > 0.6 ? 1 : 0)
        .linkDirectionalParticleWidth(1.2)
        .linkDirectionalParticleSpeed(0.004)
        .linkDirectionalParticleColor(() => "#93c5fd")
        .onNodeHover((node, prev, event) => {
            gc.style.cursor = node ? "pointer" : "default";
            if (node && event) _showTooltip(node, event);
            else _hideTooltip();
        })
        .graphData({ nodes, links });

    gc.insertAdjacentHTML("beforeend", `
        <div class="gb-hud">
            <div class="gb-hud-chip"><span>Nodes</span><strong>${nodes.length}</strong></div>
            <div class="gb-hud-chip"><span>Edges</span><strong>${links.length}</strong></div>
            <div class="gb-hud-chip"><span>Hot (&gt;60%)</span><strong>${stats.hotCount}</strong></div>
            <div class="gb-hud-chip"><span>Weakened</span><strong>${stats.weakenedCount}</strong></div>
            <div class="gb-hud-chip"><span>Pruned</span><strong>${stats.prunedCount}</strong></div>
        </div>
        <div class="gb-legend-bar">
            <span class="gb-lgd cold"></span><span>Cold</span>
            <span class="gb-lgd warm"></span><span>Warm</span>
            <span class="gb-lgd hot"></span><span>Hot</span>
            <span style="margin-left:14px;color:rgba(250,204,21,0.9)">— Weakened</span>
            <span style="margin-left:8px;color:rgba(239,68,68,0.9)">— Pruned</span>
        </div>`);

    gc.addEventListener("mousemove", evt => {
        const tip = document.getElementById("gb-tooltip");
        if (tip?.style.display === "block") _positionTooltip(tip, evt);
    }, { passive: true });

    window.addEventListener("resize", () => {
        if (_graph3d && gc) { _graph3d.width(gc.clientWidth); _graph3d.height(gc.clientHeight); }
    });
}

function _destroyGraph() {
    if (_graph3d) { try { _graph3d._destructor?.(); } catch (_) {} _graph3d = null; }
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

function _showTooltip(node, event) {
    let tip = document.getElementById("gb-tooltip");
    if (!tip) {
        tip = document.createElement("div");
        tip.id = "gb-tooltip";
        tip.className = "gb-tooltip";
        document.body.appendChild(tip);
    }
    const hex = "#" + node._color.toString(16).padStart(6, "0");
    const tags = (node._tags || []).slice(0, 5).join(", ");
    tip.innerHTML = `
        <div class="gb-tip-label"><span style="background:${hex}" class="gb-tip-dot"></span>${_esc(node._label)}</div>
        <div class="gb-tip-row">Temp <strong>${(node._temp * 100).toFixed(0)}%</strong></div>
        <div class="gb-tip-row">Level <strong>${_esc(node._level || "—")}</strong></div>
        <div class="gb-tip-row">Strength <strong>${(node._strength * 100).toFixed(0)}%</strong></div>
        ${tags ? `<div class="gb-tip-row" style="color:var(--text-muted)">${_esc(tags)}</div>` : ""}`;
    tip.style.display = "block";
    _positionTooltip(tip, event);
}

function _positionTooltip(tip, evt) {
    const w = 240;
    let left = evt.clientX + 14;
    let top  = evt.clientY - 14;
    if (left + w > window.innerWidth) left = evt.clientX - w - 14;
    if (top < 0) top = 8;
    tip.style.left = `${left}px`;
    tip.style.top  = `${top}px`;
}

function _hideTooltip() {
    const tip = document.getElementById("gb-tooltip");
    if (tip) tip.style.display = "none";
}

// ── Thermal tab ───────────────────────────────────────────────────────────────

async function _loadThermal() {
    const el = document.getElementById("gb-thermal-content");
    if (!el) return;
    try {
        const res = await fetch("/api/glass-brain/thermal-map");
        const data = await res.json();
        _renderThermal(el, data);
    } catch (e) {
        if (el) el.innerHTML = `<div class="gb-error">Failed: ${_esc(e.message)}</div>`;
    }
}

function _renderThermal(el, data) {
    if (!data || data.status === "no_data") {
        el.innerHTML = `<div class="gb-empty">${_esc(data?.message || "No thermal data yet")}</div>`;
        return;
    }
    if (data.status === "error") {
        el.innerHTML = `<div class="gb-error">${_esc(data.message)}</div>`;
        return;
    }

    const clusters = (data.clusters || []).slice(0, 8).map(c => {
        const tags = (c.dominant_tags || []).map(t => `<span class="tag">${_esc(t.tag)}</span>`).join(" ");
        const flags = [
            c.has_conflict   ? '<span class="gb-badge conflict">conflict</span>'   : "",
            c.has_unresolved ? '<span class="gb-badge unresolved">unresolved</span>' : "",
        ].filter(Boolean).join(" ");
        return `
            <div class="gb-cluster-card">
                <div class="gb-cluster-temp">${(c.max_temperature * 100).toFixed(0)}°</div>
                <div class="gb-cluster-body">
                    <div class="gb-cluster-size">${c.size} beliefs ${flags}</div>
                    <div class="gb-cluster-tags">${tags}</div>
                </div>
            </div>`;
    }).join("") || "<em style='color:var(--text-muted)'>No hot clusters</em>";

    const advice = (data.routing_advice || [])
        .map(a => `<div class="gb-advice-item">${_esc(a)}</div>`).join("")
        || "<em style='color:var(--text-muted)'>No advice</em>";

    el.innerHTML = `
        <div class="gb-stats-row">
            ${_stat("Total energy",  data.total_energy?.toFixed(3))}
            ${_stat("Mean temp",     ((data.mean_temperature || 0) * 100).toFixed(1) + "%")}
            ${_stat("Hot zones",     data.hot_zone_count)}
            ${_stat("Cold mass",     data.cold_mass_count)}
            ${_stat("Nodes",         data.node_count)}
            ${_stat("Edges",         data.edge_count)}
        </div>
        <div class="gb-section-label">Hot Clusters</div>
        <div class="gb-clusters">${clusters}</div>
        <div class="gb-section-label">Routing Advice</div>
        <div class="gb-advice">${advice}</div>`;
}

// ── Plasticity tab ────────────────────────────────────────────────────────────

async function _loadPlasticity() {
    const el = document.getElementById("gb-plasticity-content");
    if (!el) return;
    try {
        const res = await fetch("/api/glass-brain/plasticity");
        const data = await res.json();
        _renderPlasticity(el, data);
    } catch (e) {
        if (el) el.innerHTML = `<div class="gb-error">Failed: ${_esc(e.message)}</div>`;
    }
}

function _renderPlasticity(el, data) {
    if (!data || data.status === "no_data") {
        el.innerHTML = `<div class="gb-empty">${_esc(data?.message || "No plasticity data yet")}</div>`;
        return;
    }
    if (data.status === "error") {
        el.innerHTML = `<div class="gb-error">${_esc(data.message)}</div>`;
        return;
    }

    const summary = data.summary || {};

    // pruned_edges have fields: a, b, shared_tags, leaks, productive, penalty
    const prunedList = (data.pruned_edges || []).slice(0, 15).map(e => `
        <div class="gb-pruned-item">
            <span class="gb-pruned-label">${_esc(_shortKey(e.a))} ↔ ${_esc(_shortKey(e.b))}</span>
            <span class="gb-badge pruned" title="leaks:${e.leaks} penalty:${e.penalty?.toFixed(2)}">pruned</span>
        </div>`).join("") || "<em style='color:var(--text-muted)'>None</em>";

    // at_risk_edges
    const atRiskList = (data.at_risk_edges || []).slice(0, 15).map(e => `
        <div class="gb-pruned-item">
            <span class="gb-pruned-label">${_esc(_shortKey(e.a))} ↔ ${_esc(_shortKey(e.b))}</span>
            <span class="gb-badge weakened" title="penalty:${e.penalty?.toFixed(2)}">at risk</span>
        </div>`).join("") || "<em style='color:var(--text-muted)'>None</em>";

    el.innerHTML = `
        <div class="gb-stats-row">
            ${_stat("Total edges",  summary.total_edges ?? "—")}
            ${_stat("Healthy",      summary.healthy ?? "—")}
            ${_stat("Weakened",     summary.weakened ?? "—")}
            ${_stat("Pruned",       summary.pruned ?? "—")}
            ${_stat("Leaks",        summary.total_leaks ?? "—")}
            ${_stat("Productive",   summary.total_productive ?? "—")}
        </div>
        <div class="gb-section-label">Pruned Synapses</div>
        <div class="gb-pruned-list">${prunedList}</div>
        <div class="gb-section-label">At Risk</div>
        <div class="gb-pruned-list">${atRiskList}</div>`;
}

// Extract just the meaningful part of a belief key "default:tags:type"
function _shortKey(key) {
    if (!key) return "?";
    const parts = String(key).split(":");
    // take middle (tags) + last (type), skip "default"
    return parts.slice(1).join(":").slice(0, 50) || key.slice(0, 50);
}

// ── Auto-refresh ──────────────────────────────────────────────────────────────

function _startRefresh() {
    clearInterval(_refreshTimer);
    _refreshTimer = setInterval(() => {
        if (_activeTab === "graph")           _loadGraph({ silent: true });
        else if (_activeTab === "thermal")    _loadThermal();
        else if (_activeTab === "plasticity") _loadPlasticity();
    }, _REFRESH_INTERVAL_MS);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _stat(label, value) {
    return `
        <div class="gb-stat-card">
            <div class="gb-stat-value">${_esc(String(value ?? "—"))}</div>
            <div class="gb-stat-label">${_esc(label)}</div>
        </div>`;
}

function _esc(str) {
    const d = document.createElement("div");
    d.textContent = str ?? "";
    return d.innerHTML;
}
