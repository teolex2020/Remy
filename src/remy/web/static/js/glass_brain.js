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
let _allGraphData = { nodes: [], links: [] };
let _graphAudit = {};
let _selectedNodeId = "";
let _resizeHandler = null;
let _filterTimer = null;
let _highlightNodeIds = new Set();
let _highlightLinkKeys = new Set();
const _localFocus = { id: "", depth: 0 };
const _graphFilters = { query: "", level: "all", hideIsolated: false, edgeLimit: 2000 };
const _LAYOUT_STORAGE_KEY = "remy_glass_brain_layout_v1";
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
            const tags = tagSegment.split(",").map(t => t.trim().toLowerCase()).filter(Boolean);
            const temp = rescale(bn.temp);
            for (const tag of tags) {
                const prev = tagTempMap.get(tag) || 0;
                tagTempMap.set(tag, Math.max(prev, temp));
            }
        }

        // Assign temperature to each memory node via its tags
        const savedLayout = _readSavedLayout();
        let thermallyMatched = 0;
        const nodes = memNodes.map(n => {
            const nodeTags = (n.tags || []).map(t => String(t).trim().toLowerCase());
            let temp = 0;
            let thermalMatched = false;
            for (const tag of nodeTags) {
                if (!tagTempMap.has(tag)) continue;
                thermalMatched = true;
                const t = tagTempMap.get(tag) || 0;
                if (t > temp) temp = t;
            }
            if (thermalMatched) thermallyMatched += 1;
            const displayTemp = temp || 0.04; // cold nodes show blue
            const saved = savedLayout[n.id];
            return {
                id:        n.id,
                _label:    (n.label || "").slice(0, 60),
                _temp:     displayTemp,
                _level:    n.level || "",
                _tags:     n.tags || [],
                _strength: n.strength || 0.1,
                _importance: n.importance || 0,
                _excerpt: n.excerpt || n.label || "",
                _type: n.record_type || "memory",
                _timestamp: n.timestamp || "",
                _thermalMatched: thermalMatched,
                _color:    _thermalColorHex(displayTemp),
                ...(saved && Number.isFinite(saved.x) && Number.isFinite(saved.y) && Number.isFinite(saved.z)
                    ? { x: saved.x, y: saved.y, z: saved.z }
                    : {}),
            };
        });

        const links = memEdges
            .sort((a, b) => (b.weight || 0) - (a.weight || 0))
            .map(e => ({ source: e.source, target: e.target, _weight: e.weight || 0.1 }));

        const hotCount = nodes.filter(n => n._temp > 0.6).length;
        _allGraphData = { nodes, links };
        _graphAudit = {
            ...(graphData.coverage || {}),
            hotCount,
            thermallyMatched,
            thermalUnmatched: Math.max(0, nodes.length - thermallyMatched),
            beliefNodes: beliefNodes.length,
            weakenedCount: plasticityStats.weakened,
            prunedCount: plasticityStats.pruned,
        };
        await _render3d(_filteredGraphData(), {
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

function _endpointId(value) {
    return typeof value === "object" && value ? value.id : value;
}

function _linkKey(source, target) {
    return [String(_endpointId(source)), String(_endpointId(target))].sort().join("::");
}

function _readSavedLayout() {
    try {
        const value = JSON.parse(localStorage.getItem(_LAYOUT_STORAGE_KEY) || "{}");
        return value && typeof value === "object" ? value : {};
    } catch (_) {
        return {};
    }
}

function _saveCurrentLayout() {
    if (!_graph3d) return;
    try {
        const positions = {};
        for (const node of _graph3d.graphData().nodes.slice(0, 10000)) {
            if (![node.x, node.y, node.z].every(Number.isFinite)) continue;
            positions[node.id] = {
                x: Math.round(node.x * 100) / 100,
                y: Math.round(node.y * 100) / 100,
                z: Math.round(node.z * 100) / 100,
            };
        }
        localStorage.setItem(_LAYOUT_STORAGE_KEY, JSON.stringify(positions));
    } catch (_) {}
}

function _localNodeIds(focusId, depth) {
    if (!focusId || depth < 1) return null;
    const visited = new Set([focusId]);
    let frontier = new Set([focusId]);
    for (let hop = 0; hop < depth; hop++) {
        const next = new Set();
        for (const link of _allGraphData.links) {
            const source = _endpointId(link.source);
            const target = _endpointId(link.target);
            if (frontier.has(source) && !visited.has(target)) next.add(target);
            if (frontier.has(target) && !visited.has(source)) next.add(source);
        }
        for (const id of next) visited.add(id);
        frontier = next;
        if (!frontier.size) break;
    }
    return visited;
}

function _filteredGraphData() {
    const query = _graphFilters.query.trim().toLowerCase();
    const localIds = _localNodeIds(_localFocus.id, _localFocus.depth);
    const connected = new Set();
    for (const link of _allGraphData.links) {
        connected.add(_endpointId(link.source));
        connected.add(_endpointId(link.target));
    }
    const nodes = _allGraphData.nodes.filter(node => {
        if (localIds && !localIds.has(node.id)) return false;
        if (_graphFilters.level !== "all" && node._level !== _graphFilters.level) return false;
        if (_graphFilters.hideIsolated && !connected.has(node.id)) return false;
        if (!query) return true;
        const haystack = [
            node._label, node._excerpt, node._level, node._type, ...(node._tags || []),
        ].join(" ").toLowerCase();
        return haystack.includes(query);
    });
    const ids = new Set(nodes.map(node => node.id));
    let links = _allGraphData.links.filter(link => (
        ids.has(_endpointId(link.source)) && ids.has(_endpointId(link.target))
    ));
    if (Number.isFinite(_graphFilters.edgeLimit)) {
        links = links.slice(0, _graphFilters.edgeLimit);
    }
    return {
        nodes,
        links: links.map(link => ({
            source: _endpointId(link.source),
            target: _endpointId(link.target),
            _weight: link._weight,
        })),
    };
}

function _applyGraphFilters() {
    if (!_graph3d) return;
    const data = _filteredGraphData();
    if (_selectedNodeId && !data.nodes.some(node => node.id === _selectedNodeId)) {
        _selectedNodeId = "";
    }
    _graph3d.graphData(data);
    _setGraphHighlight(
        data.nodes.find(node => node.id === _selectedNodeId) || null
    );
    _updateGraphHud(data);
    _updateLocalIndicator();
    if (!_selectedNodeId) _hideNodeDetail();
    window.setTimeout(() => _graph3d?.zoomToFit(350, 42), 80);
}

function _baseNodeColor(node) {
    return "#" + node._color.toString(16).padStart(6, "0");
}

function _nodeDisplayColor(node) {
    if (!_highlightNodeIds.size) return _baseNodeColor(node);
    if (node.id === _selectedNodeId) return "#f8fafc";
    return _highlightNodeIds.has(node.id) ? _baseNodeColor(node) : "#111c2c";
}

function _linkDisplayColor(link) {
    if (!_highlightLinkKeys.size) {
        const alpha = Math.max(0.06, Math.min(0.28, (link._weight || 0.1) * 0.35));
        return `rgba(148,163,184,${alpha})`;
    }
    return _highlightLinkKeys.has(_linkKey(link.source, link.target))
        ? "rgba(147,197,253,0.9)"
        : "rgba(51,65,85,0.025)";
}

function _linkDisplayWidth(link) {
    const base = Math.max(0.15, (link._weight || 0.1) * 0.5);
    if (!_highlightLinkKeys.size) return base;
    return _highlightLinkKeys.has(_linkKey(link.source, link.target)) ? Math.max(1.4, base * 2.5) : 0.08;
}

function _setGraphHighlight(node) {
    _highlightNodeIds = new Set();
    _highlightLinkKeys = new Set();
    if (node && _graph3d) {
        _highlightNodeIds.add(node.id);
        for (const link of _graph3d.graphData().links) {
            const source = _endpointId(link.source);
            const target = _endpointId(link.target);
            if (source !== node.id && target !== node.id) continue;
            _highlightNodeIds.add(source);
            _highlightNodeIds.add(target);
            _highlightLinkKeys.add(_linkKey(source, target));
        }
    }
    _graph3d?.nodeColor(_nodeDisplayColor);
    _graph3d?.linkColor(_linkDisplayColor);
    _graph3d?.linkWidth(_linkDisplayWidth);
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
            return () => {
                _saveCurrentLayout();
                if (!done && _graph3d) { done = true; _graph3d.zoomToFit(400, 30); }
            };
        })())
        .nodeColor(_nodeDisplayColor)
        .nodeVal(n => {
            const base = { IDENTITY: 5, DOMAIN: 3, DECISIONS: 2 }[n._level] ?? 1.5;
            const heat = n._temp > 0.6 ? 1.5 : 1.0;
            return base * Math.max(0.3, n._strength) * heat;
        })
        .nodeOpacity(0.92)
        .nodeResolution(12)
        .nodeLabel(n => `${n._label} | Temp: ${(n._temp * 100).toFixed(0)}%`)
        .linkColor(_linkDisplayColor)
        .linkWidth(_linkDisplayWidth)
        .linkOpacity(1)
        .linkCurvature(0.08)
        .linkDirectionalParticles(l => (l._weight || 0) > 0.6 ? 1 : 0)
        .linkDirectionalParticleWidth(1.2)
        .linkDirectionalParticleSpeed(0.004)
        .linkDirectionalParticleColor(() => "#93c5fd")
        .onNodeHover((node, prev, event) => {
            gc.style.cursor = node ? "pointer" : "default";
            if (node) {
                _setGraphHighlight(node);
                if (event) _showTooltip(node, event);
            } else {
                _hideTooltip();
                _setGraphHighlight(
                    _graph3d?.graphData().nodes.find(item => item.id === _selectedNodeId) || null
                );
            }
        })
        .onNodeClick(node => {
            _selectedNodeId = node.id;
            _setGraphHighlight(node);
            _showNodeDetail(node);
            const distance = 110;
            const length = Math.hypot(node.x || 0, node.y || 0, node.z || 0) || 1;
            const ratio = 1 + distance / length;
            _graph3d?.cameraPosition(
                { x: (node.x || 0) * ratio, y: (node.y || 0) * ratio, z: (node.z || 0) * ratio },
                node,
                700,
            );
        })
        .onBackgroundClick(() => {
            _selectedNodeId = "";
            _setGraphHighlight(null);
            _hideNodeDetail();
        })
        .graphData({ nodes, links });

    const levels = [...new Set(_allGraphData.nodes.map(node => node._level).filter(Boolean))].sort();
    gc.insertAdjacentHTML("beforeend", `
        <div class="gb-graph-tools">
            <input id="gb-search" class="gb-search" type="search" placeholder="Search memories or tags" value="${_esc(_graphFilters.query)}">
            <select id="gb-level-filter" class="gb-filter-select" aria-label="Memory level">
                <option value="all">All levels</option>
                ${levels.map(level => `<option value="${_esc(level)}" ${_graphFilters.level === level ? "selected" : ""}>${_esc(level)}</option>`).join("")}
            </select>
            <select id="gb-edge-limit" class="gb-filter-select" aria-label="Connection detail">
                <option value="2000" ${_graphFilters.edgeLimit === 2000 ? "selected" : ""}>2k connections</option>
                <option value="10000" ${_graphFilters.edgeLimit === 10000 ? "selected" : ""}>10k connections</option>
                <option value="all" ${!Number.isFinite(_graphFilters.edgeLimit) ? "selected" : ""}>All connections</option>
            </select>
            <label class="gb-check"><input id="gb-hide-isolated" type="checkbox" ${_graphFilters.hideIsolated ? "checked" : ""}> Hide isolated</label>
            <button id="gb-fit" class="gb-tool-button">Fit</button>
            <button id="gb-reset-layout" class="gb-tool-button">Reset layout</button>
            <div id="gb-local-mode" class="gb-local-mode"></div>
        </div>
        <details class="gb-coverage">
            <summary>Coverage audit</summary>
            <div class="gb-coverage-body">
                <div><span>Record store</span><strong>${_graphAudit.records_returned ?? _allGraphData.nodes.length}/${_graphAudit.records_scanned ?? _allGraphData.nodes.length}</strong></div>
                <div><span>Connected records</span><strong>${_graphAudit.connected_records ?? "—"}</strong></div>
                <div><span>Isolated records</span><strong>${_graphAudit.isolated_records ?? "—"}</strong></div>
                <div><span>Thermal mapping</span><strong>${_graphAudit.thermallyMatched ?? 0}/${_allGraphData.nodes.length}</strong></div>
                <div><span>ACL belief layer</span><strong>${_graphAudit.beliefNodes ?? 0}</strong></div>
                <div><span>Dangling references</span><strong>${_graphAudit.dangling_connections ?? 0}</strong></div>
                <p>The graph shows Aura memory records. Thermal beliefs are a separate ACL layer matched to records by normalized tags. Activity and event logs remain in their dedicated views.</p>
            </div>
        </details>
        <aside id="gb-node-detail" class="gb-node-detail gb-node-detail-overlay hidden"></aside>`);

    gc.insertAdjacentHTML("beforeend", `
        <div class="gb-hud">
            <div class="gb-hud-chip"><span>Nodes</span><strong>${nodes.length}</strong></div>
            <div class="gb-hud-chip"><span>Edges</span><strong>${links.length}</strong></div>
            <div class="gb-hud-chip"><span>Hot (&gt;60%)</span><strong>${stats.hotCount}</strong></div>
            <div class="gb-hud-chip"><span>ACL weak</span><strong>${stats.weakenedCount}</strong></div>
            <div class="gb-hud-chip"><span>ACL pruned</span><strong>${stats.prunedCount}</strong></div>
        </div>
        <div class="gb-legend-bar">
            <span class="gb-lgd cold"></span><span>Cold</span>
            <span class="gb-lgd warm"></span><span>Warm</span>
            <span class="gb-lgd hot"></span><span>Hot</span>
            <span class="gb-legend-note">Colour = thermal match · size = level/strength</span>
        </div>`);
    _updateGraphHud({ nodes, links });
    _bindGraphControls();
    _updateLocalIndicator();

    gc.addEventListener("mousemove", evt => {
        const tip = document.getElementById("gb-tooltip");
        if (tip?.style.display === "block") _positionTooltip(tip, evt);
    }, { passive: true });

    _resizeHandler = () => {
        if (_graph3d && gc) { _graph3d.width(gc.clientWidth); _graph3d.height(gc.clientHeight); }
    };
    window.addEventListener("resize", _resizeHandler);
}

function _updateGraphHud(data) {
    const hud = document.querySelector("#gb-graph-container .gb-hud");
    if (!hud) return;
    const visibleHot = data.nodes.filter(node => node._temp > 0.6).length;
    const visibleMapped = data.nodes.filter(node => node._thermalMatched).length;
    hud.innerHTML = `
        <div class="gb-hud-chip"><span>Records</span><strong>${data.nodes.length}/${_allGraphData.nodes.length}</strong></div>
        <div class="gb-hud-chip"><span>Connections</span><strong>${data.links.length}/${_allGraphData.links.length}</strong></div>
        <div class="gb-hud-chip"><span>Thermal match</span><strong>${visibleMapped}</strong></div>
        <div class="gb-hud-chip"><span>Hot (&gt;60%)</span><strong>${visibleHot}</strong></div>
        <div class="gb-hud-chip"><span>ACL weak</span><strong>${_graphAudit.weakenedCount || 0}</strong></div>
        <div class="gb-hud-chip"><span>ACL pruned</span><strong>${_graphAudit.prunedCount || 0}</strong></div>`;
}

function _bindGraphControls() {
    const search = document.getElementById("gb-search");
    search?.addEventListener("input", () => {
        clearTimeout(_filterTimer);
        _filterTimer = window.setTimeout(() => {
            _graphFilters.query = search.value;
            _applyGraphFilters();
        }, 160);
    });
    document.getElementById("gb-level-filter")?.addEventListener("change", event => {
        _graphFilters.level = event.target.value;
        _applyGraphFilters();
    });
    document.getElementById("gb-edge-limit")?.addEventListener("change", event => {
        _graphFilters.edgeLimit = event.target.value === "all" ? Infinity : Number(event.target.value);
        _applyGraphFilters();
    });
    document.getElementById("gb-hide-isolated")?.addEventListener("change", event => {
        _graphFilters.hideIsolated = event.target.checked;
        _applyGraphFilters();
    });
    document.getElementById("gb-fit")?.addEventListener("click", () => _graph3d?.zoomToFit(350, 42));
    document.getElementById("gb-reset-layout")?.addEventListener("click", () => {
        localStorage.removeItem(_LAYOUT_STORAGE_KEY);
        for (const node of _allGraphData.nodes) {
            delete node.fx; delete node.fy; delete node.fz;
            node.x = (Math.random() - 0.5) * 20;
            node.y = (Math.random() - 0.5) * 20;
            node.z = (Math.random() - 0.5) * 20;
        }
        _graph3d?.graphData(_filteredGraphData());
        _graph3d?.d3ReheatSimulation?.();
    });
}

function _setLocalFocus(nodeId, depth) {
    _localFocus.id = nodeId;
    _localFocus.depth = Math.max(1, Math.min(3, Number(depth) || 1));
    _graphFilters.query = "";
    const search = document.getElementById("gb-search");
    if (search) search.value = "";
    _applyGraphFilters();
}

function _clearLocalFocus() {
    _localFocus.id = "";
    _localFocus.depth = 0;
    _applyGraphFilters();
}

function _updateLocalIndicator() {
    const indicator = document.getElementById("gb-local-mode");
    if (!indicator) return;
    if (!_localFocus.id) {
        indicator.innerHTML = "";
        indicator.classList.remove("active");
        return;
    }
    const node = _allGraphData.nodes.find(item => item.id === _localFocus.id);
    indicator.classList.add("active");
    indicator.innerHTML = `<span>Local ${_localFocus.depth}-hop: ${_esc(node?._label || _localFocus.id)}</span><button id="gb-local-clear" aria-label="Show full graph">×</button>`;
    document.getElementById("gb-local-clear")?.addEventListener("click", _clearLocalFocus);
}

function _showNodeDetail(node) {
    const panel = document.getElementById("gb-node-detail");
    if (!panel) return;
    const related = new Set();
    for (const link of _allGraphData.links) {
        const source = _endpointId(link.source);
        const target = _endpointId(link.target);
        if (source === node.id) related.add(target);
        if (target === node.id) related.add(source);
    }
    const tags = (node._tags || []).map(tag => `<span class="gb-node-tag">${_esc(tag)}</span>`).join("");
    panel.innerHTML = `
        <button id="gb-node-close" class="gb-node-close" aria-label="Close">×</button>
        <div class="gb-node-key">${_esc(node._label)}</div>
        <div class="gb-node-stats">
            <div>Level <strong>${_esc(node._level || "—")}</strong></div>
            <div>Type <strong>${_esc(node._type || "memory")}</strong></div>
            <div>Strength <strong>${(node._strength * 100).toFixed(0)}%</strong></div>
            <div>Importance <strong>${Number(node._importance || 0).toFixed(2)}</strong></div>
            <div>Thermal match <strong>${(node._temp * 100).toFixed(0)}%</strong></div>
            <div>Connections <strong>${related.size}</strong></div>
            ${node._timestamp ? `<div>Updated <strong>${_esc(String(node._timestamp))}</strong></div>` : ""}
        </div>
        ${tags ? `<div class="gb-node-tags">${tags}</div>` : ""}
        <div class="gb-node-excerpt">${_esc(node._excerpt || node._label)}</div>
        <div class="gb-local-actions">
            <span>Local Brain</span>
            <button class="gb-tool-button gb-local-depth" data-depth="1">1 hop</button>
            <button class="gb-tool-button gb-local-depth" data-depth="2">2 hops</button>
            <button class="gb-tool-button gb-local-depth" data-depth="3">3 hops</button>
        </div>
        <button id="gb-copy-node-id" class="gb-tool-button gb-copy-id">Copy memory ID</button>`;
    panel.classList.remove("hidden");
    document.getElementById("gb-node-close")?.addEventListener("click", _hideNodeDetail);
    panel.querySelectorAll(".gb-local-depth").forEach(button => {
        button.addEventListener("click", () => _setLocalFocus(node.id, Number(button.dataset.depth)));
    });
    document.getElementById("gb-copy-node-id")?.addEventListener("click", async event => {
        await navigator.clipboard.writeText(String(node.id));
        event.currentTarget.textContent = "Copied";
    });
}

function _hideNodeDetail() {
    document.getElementById("gb-node-detail")?.classList.add("hidden");
}

function _destroyGraph() {
    if (_graph3d) { try { _graph3d._destructor?.(); } catch (_) {} _graph3d = null; }
    if (_resizeHandler) window.removeEventListener("resize", _resizeHandler);
    _resizeHandler = null;
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
