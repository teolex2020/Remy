/**
 * Graph View — 3D force-directed neural visualization via 3d-force-graph + Three.js
 */

// ── Debounce ─────────────────────────────────────────────────────────────────
function _debounce(fn, delay) {
    let timer = null;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

// ── DOM refs ─────────────────────────────────────────────────────────────────
const graphContainer = document.getElementById("graph-container");
const graphModeButtons = Array.from(document.querySelectorAll("[data-graph-mode]"));
const graphTimeButtons = Array.from(document.querySelectorAll("[data-graph-time]"));
const graphTimelineControls = document.getElementById("graph-timeline-controls");
const graphTimelinePlayBtn = document.getElementById("graph-timeline-play");
const graphTimelineRange = document.getElementById("graph-timeline-range");
const graphTimelineLabel = document.getElementById("graph-timeline-label");
const scopeFilter    = document.getElementById("graph-scope-filter");
const levelFilter    = document.getElementById("graph-level-filter");
const tagFilter      = document.getElementById("graph-tag-filter");
const resetBtn       = document.getElementById("btn-graph-reset");
const edgeSlider     = document.getElementById("graph-edge-threshold");
const edgeValueEl    = document.getElementById("graph-edge-value");
const nodeCountEl    = document.getElementById("graph-node-count");
const edgeCountEl    = document.getElementById("graph-edge-count");
const avgStrengthEl  = document.getElementById("graph-avg-strength");
const levelBreakdownEl = document.getElementById("graph-level-breakdown");
const topTagsEl      = document.getElementById("graph-top-tags");

// ── State ─────────────────────────────────────────────────────────────────────
let graph3d   = null;   // ForceGraph3D instance
let rawGraph  = { nodes: [], edges: [] };
let thermalGraph = { nodes: [], edges: [], status: "idle" };  // belief-space graph
let currentGraphMode = "network";
let currentBrainTimeWindow = "all";
let currentBrainPlaybackSec = null;
let brainTimelineBounds = null;
let brainPulseTimer = null;
let brainBurstUntil = 0;
let brainTimelineTimer = null;

// ── Neural palette (hex integers for Three.js) ────────────────────────────────
const LEVEL_COLORS_HEX = {
    working:   0x38bdf8,
    decisions: 0xfb923c,
    domain:    0x34d399,
    identity:  0xe879f9,
    unknown:   0x64748b,
};
const LEVEL_COLORS_CSS = {
    working:   "#38bdf8",
    decisions: "#fb923c",
    domain:    "#34d399",
    identity:  "#e879f9",
    unknown:   "#64748b",
};
const LEVEL_LABELS = {
    WORKING: "Working", DECISIONS: "Decisions", DOMAIN: "Domain", IDENTITY: "Identity",
};

function normalizeLevel(level) {
    const t = String(level || "").toUpperCase();
    if (t.includes("WORK"))     return "WORKING";
    if (t.includes("DECISION")) return "DECISIONS";
    if (t.includes("DOMAIN"))   return "DOMAIN";
    if (t.includes("IDENTITY")) return "IDENTITY";
    return t || "UNKNOWN";
}

function levelColor(level) {
    return LEVEL_COLORS_HEX[level.toLowerCase()] ?? LEVEL_COLORS_HEX.unknown;
}
function levelColorCss(level) {
    return LEVEL_COLORS_CSS[level.toLowerCase()] ?? LEVEL_COLORS_CSS.unknown;
}

// ── Thermal palette — maps temperature ∈ [0,1] to a hex color ─────────────────
//  Cold (blue) → warm (green) → hot (orange/red)
function thermalColorHex(t) {
    const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
    // 3-stop gradient: 0.0 cold #38bdf8, 0.5 warm #34d399, 1.0 hot #ef4444
    const stops = [
        { p: 0.00, r: 0x38, g: 0xbd, b: 0xf8 },
        { p: 0.50, r: 0x34, g: 0xd3, b: 0x99 },
        { p: 1.00, r: 0xef, g: 0x44, b: 0x44 },
    ];
    let a = stops[0], b = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
        if (x >= stops[i].p && x <= stops[i + 1].p) {
            a = stops[i]; b = stops[i + 1]; break;
        }
    }
    const span = (b.p - a.p) || 1;
    const k = (x - a.p) / span;
    const r = Math.round(a.r + (b.r - a.r) * k);
    const g = Math.round(a.g + (b.g - a.g) * k);
    const bl = Math.round(a.b + (b.b - a.b) * k);
    return (r << 16) | (g << 8) | bl;
}

function thermalColorCss(t) {
    const hex = thermalColorHex(t).toString(16).padStart(6, "0");
    return `#${hex}`;
}

const EDGE_HEALTH_COLOR_CSS = {
    healthy:  "rgba(148,163,184,0.22)",
    weakened: "rgba(250,204,21,0.85)",
    pruned:   "rgba(239,68,68,0.75)",
};
// Visual weight multiplier per health — thicker = more visible
const EDGE_HEALTH_WIDTH_MULT = {
    healthy:  0.6,
    weakened: 2.2,
    pruned:   2.6,
};

function setGraphMode(mode) {
    const allowed = new Set(["network", "brain", "thermal"]);
    const nextMode = allowed.has(mode) ? mode : "network";
    if (currentGraphMode === nextMode) return;
    currentGraphMode = nextMode;
    if (currentGraphMode !== "brain") {
        _stopBrainTimelinePlayback();
    }
    graphModeButtons.forEach((btn) => {
        const active = btn.dataset.graphMode === currentGraphMode;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    if (currentGraphMode === "thermal") {
        loadThermalGraph().then(renderFilteredGraph);
    } else {
        renderFilteredGraph();
    }
}

function setBrainTimeWindow(windowKey) {
    const allowed = new Set(["all", "30d", "7d", "1d"]);
    currentBrainTimeWindow = allowed.has(windowKey) ? windowKey : "all";
    if (currentBrainTimeWindow === "all") {
        currentBrainPlaybackSec = null;
        _stopBrainTimelinePlayback();
    }
    graphTimeButtons.forEach((btn) => {
        const active = btn.dataset.graphTime === currentBrainTimeWindow;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    renderFilteredGraph();
}

function _formatTimelineDate(sec) {
    if (!Number.isFinite(sec) || sec <= 0) return "Latest";
    const dt = new Date(sec * 1000);
    const day = dt.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    return day;
}

function _stopBrainTimelinePlayback() {
    if (brainTimelineTimer) {
        clearInterval(brainTimelineTimer);
        brainTimelineTimer = null;
    }
    if (graphTimelinePlayBtn) {
        graphTimelinePlayBtn.classList.remove("active");
        graphTimelinePlayBtn.textContent = "Play";
    }
}

function _syncBrainTimelineUI(data) {
    if (!graphTimelineControls || !graphTimelineRange || !graphTimelineLabel || !graphTimelinePlayBtn) return;
    const nodes = data?.nodes || [];
    const times = nodes
        .map((node) => _parseGraphTimestamp(node.timestamp))
        .filter((value) => value > 0)
        .sort((a, b) => a - b);

    if (currentGraphMode !== "brain" || currentBrainTimeWindow === "all" || !times.length) {
        graphTimelineControls.classList.add("hidden");
        brainTimelineBounds = null;
        if (currentGraphMode !== "brain" || currentBrainTimeWindow === "all") {
            currentBrainPlaybackSec = null;
        }
        _stopBrainTimelinePlayback();
        return;
    }

    const minSec = times[0];
    const maxSec = times[times.length - 1];
    const spanDays = Math.max(1, Math.ceil((maxSec - minSec) / 86400));
    brainTimelineBounds = { minSec, maxSec, spanDays };
    if (!Number.isFinite(currentBrainPlaybackSec) || currentBrainPlaybackSec < minSec || currentBrainPlaybackSec > maxSec) {
        currentBrainPlaybackSec = maxSec;
    }

    const offsetDays = Math.max(0, Math.min(spanDays, Math.round((currentBrainPlaybackSec - minSec) / 86400)));
    graphTimelineRange.min = "0";
    graphTimelineRange.max = String(spanDays);
    graphTimelineRange.step = "1";
    graphTimelineRange.value = String(offsetDays);
    graphTimelineLabel.textContent = _formatTimelineDate(currentBrainPlaybackSec);
    graphTimelineControls.classList.remove("hidden");
}

// ── Filters ───────────────────────────────────────────────────────────────────
function getActiveFilters() {
    return {
        level:         levelFilter?.value || "all",
        tag:           (tagFilter?.value || "").trim().toLowerCase(),
        edgeThreshold: edgeSlider ? parseFloat(edgeSlider.value) : 0,
    };
}

function applyFilters(data) {
    const { level, tag, edgeThreshold } = getActiveFilters();

    const filteredNodes = (data.nodes || []).filter(node => {
        const nl = normalizeLevel(node.level);
        if (level !== "all" && nl !== level) return false;
        if (tag) {
            const tags = (node.tags || []).map(t => String(t).toLowerCase());
            if (!tags.some(t => t.includes(tag))) return false;
        }
        return true;
    });

    const validIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = (data.edges || []).filter(
        e => validIds.has(e.source) && validIds.has(e.target) && (e.weight || 0) >= edgeThreshold
    );

    return { nodes: filteredNodes, edges: filteredEdges };
}

function _parseGraphTimestamp(value) {
    if (!value) return 0;
    if (typeof value === "number" && Number.isFinite(value)) return value > 1e12 ? value / 1000 : value;
    const ts = Date.parse(String(value));
    return Number.isFinite(ts) ? ts / 1000 : 0;
}

function _brainWindowSeconds(windowKey) {
    return {
        "30d": 30 * 86400,
        "7d": 7 * 86400,
        "1d": 86400,
    }[windowKey] || 0;
}

function _annotateBrainRecency(data) {
    const nodes = (data.nodes || []).map((node) => ({ ...node }));
    const edges = data.edges || [];
    if (currentBrainTimeWindow === "all") {
        nodes.forEach((node) => {
            node._brainRecent = 0;
            node._brainAgeSec = _parseGraphTimestamp(node.timestamp) > 0
                ? Math.max(0, Date.now() / 1000 - _parseGraphTimestamp(node.timestamp))
                : null;
        });
        return { nodes, edges };
    }

    const nowSec = Number.isFinite(currentBrainPlaybackSec) ? currentBrainPlaybackSec : Date.now() / 1000;
    const windowSec = _brainWindowSeconds(currentBrainTimeWindow);
    if (!windowSec) return { nodes, edges };

    let datedCount = 0;
    const visibleIds = new Set();
    nodes.forEach((node) => {
        const ts = _parseGraphTimestamp(node.timestamp);
        if (ts > 0) {
            if (ts > nowSec) {
                node._brainAgeSec = null;
                node._brainRecent = 0;
                node._brainFuture = true;
                return;
            }
            datedCount += 1;
            const ageSec = Math.max(0, nowSec - ts);
            node._brainAgeSec = ageSec;
            node._brainRecent = ageSec <= windowSec ? Math.max(0.12, 1 - (ageSec / windowSec)) : 0;
            node._brainFuture = false;
            visibleIds.add(node.id);
        } else {
            node._brainAgeSec = null;
            node._brainRecent = 0;
            node._brainFuture = false;
            visibleIds.add(node.id);
        }
    });
    if (datedCount === 0) {
        return { nodes, edges };
    }
    const visibleNodes = nodes.filter((node) => visibleIds.has(node.id));
    const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
    const visibleEdges = edges.filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));
    return { nodes: visibleNodes, edges: visibleEdges };
}

// ── Stats panel ───────────────────────────────────────────────────────────────
function renderStats(data) {
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    const levelCounts = new Map();
    const tagCounts   = new Map();
    let totalStrength = 0;

    for (const n of nodes) {
        const lv = normalizeLevel(n.level);
        levelCounts.set(lv, (levelCounts.get(lv) || 0) + 1);
        totalStrength += (n.strength || 0);
        for (const tag of n.tags || []) {
            const t = String(tag || "").trim();
            if (t) tagCounts.set(t, (tagCounts.get(t) || 0) + 1);
        }
    }

    if (nodeCountEl) nodeCountEl.textContent = String(nodes.length);
    // Show capped edge count with indicator if truncated
    if (edgeCountEl) {
        const MAX_LINKS = 1500;
        edgeCountEl.textContent = edges.length > MAX_LINKS
            ? `${MAX_LINKS}+`
            : String(edges.length);
    }
    if (avgStrengthEl) {
        const avg = nodes.length > 0 ? totalStrength / nodes.length : 0;
        avgStrengthEl.textContent = `${(avg * 100).toFixed(0)}%`;
    }
    if (levelBreakdownEl) {
        levelBreakdownEl.innerHTML = Array.from(levelCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([lv, cnt]) => {
                const label = LEVEL_LABELS[lv] || lv;
                const color = levelColorCss(lv);
                return `<span class="graph-pill"><span class="graph-legend-dot" style="background:${color}"></span><strong>${label}</strong><em>${cnt}</em></span>`;
            }).join("") || `<span class="graph-muted">No nodes match current filters.</span>`;
    }
    if (topTagsEl) {
        topTagsEl.innerHTML = Array.from(tagCounts.entries())
            .sort((a, b) => b[1] - a[1]).slice(0, 8)
            .map(([tag, cnt]) => `<span class="graph-pill graph-pill-clickable" data-tag="${tag}"><strong>${tag}</strong><em>${cnt}</em></span>`)
            .join("") || `<span class="graph-muted">No tags available.</span>`;

        topTagsEl.querySelectorAll(".graph-pill-clickable").forEach(pill => {
            pill.addEventListener("click", () => {
                if (tagFilter) { tagFilter.value = pill.dataset.tag; renderFilteredGraph(); }
            });
        });
    }
}

function _normalizeTag(tag) {
    return String(tag || "").trim().toLowerCase();
}

function _inferRegion(node) {
    const tags = (node.tags || []).map(_normalizeTag);
    const text = String(node.label || node.content || "").toLowerCase();

    if (tags.includes("profile") || normalizeLevel(node.level) === "IDENTITY") return "Identity Core";
    if (tags.includes("goal") || tags.includes("task")) return "Goals";
    if (tags.includes("strategy")) return "Strategy";
    if (tags.includes("financial")) return "Finance";
    if (tags.includes("contact")) return "Contacts";
    if (tags.includes("grants")) return "Grants";
    if (tags.includes("research") || tags.includes("web-search")) return "Research";
    if (tags.includes("aurasdk") || tags.includes("v10") || tags.includes("v9") || text.includes("aurasdk")) return "AuraSDK";
    if (normalizeLevel(node.level) === "DECISIONS") return "Decisions";
    if (normalizeLevel(node.level) === "WORKING") return "Working";
    return "Knowledge";
}

function _buildCognitiveModel(data) {
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const regionMap = new Map();
    let datedNodes = 0;
    let recentNodes = 0;

    for (const node of nodes) {
        const regionName = _inferRegion(node);
        const level = normalizeLevel(node.level);
        if (!regionMap.has(regionName)) {
            regionMap.set(regionName, {
                id: regionName,
                label: regionName,
                count: 0,
                recentCount: 0,
                recentStrength: 0,
                strengthSum: 0,
                edgeWeight: 0,
                nodeIds: new Set(),
                levels: new Map(),
                topTags: new Map(),
            });
        }
        const region = regionMap.get(regionName);
        region.count += 1;
        region.strengthSum += Number(node.strength || 0);
        if (Number.isFinite(node._brainAgeSec) && node._brainAgeSec !== null) {
            datedNodes += 1;
        }
        const recent = Number(node._brainRecent || 0);
        if (recent > 0) {
            recentNodes += 1;
            region.recentCount += 1;
            region.recentStrength += recent;
        }
        region.nodeIds.add(node.id);
        region.levels.set(level, (region.levels.get(level) || 0) + 1);
        for (const tag of node.tags || []) {
            const normalized = String(tag || "").trim();
            if (normalized) {
                region.topTags.set(normalized, (region.topTags.get(normalized) || 0) + 1);
            }
        }
    }

    const regionLinks = new Map();
    for (const edge of edges) {
        const sourceNode = nodeById.get(edge.source);
        const targetNode = nodeById.get(edge.target);
        if (!sourceNode || !targetNode) continue;
        const sourceRegion = _inferRegion(sourceNode);
        const targetRegion = _inferRegion(targetNode);
        if (sourceRegion === targetRegion) {
            const region = regionMap.get(sourceRegion);
            if (region) region.edgeWeight += Number(edge.weight || 0);
            continue;
        }
        const linkKey = [sourceRegion, targetRegion].sort().join("::");
        regionLinks.set(linkKey, (regionLinks.get(linkKey) || 0) + Number(edge.weight || 0));
    }

    const regions = Array.from(regionMap.values())
        .map((region) => {
            const dominantLevel = Array.from(region.levels.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] || "DOMAIN";
            const topTag = Array.from(region.topTags.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
            return {
                ...region,
                avgStrength: region.count ? region.strengthSum / region.count : 0,
                recentShare: region.count ? region.recentCount / region.count : 0,
                recentGlow: region.count ? region.recentStrength / region.count : 0,
                dominantLevel,
                topTag,
            };
        })
        .sort((a, b) => b.count - a.count);

    const links = Array.from(regionLinks.entries())
        .map(([key, weight]) => {
            const [source, target] = key.split("::");
            return { source, target, weight };
        })
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 18);

    const strongestRegion = regions[0] || null;
    const bridgeWeightByRegion = new Map();
    for (const link of links) {
        bridgeWeightByRegion.set(link.source, (bridgeWeightByRegion.get(link.source) || 0) + link.weight);
        bridgeWeightByRegion.set(link.target, (bridgeWeightByRegion.get(link.target) || 0) + link.weight);
    }
    const bridgeRegion = regions
        .map((region) => ({ ...region, bridgeWeight: bridgeWeightByRegion.get(region.id) || 0 }))
        .sort((a, b) => b.bridgeWeight - a.bridgeWeight)[0] || null;

    return {
        regions,
        links,
        strongestRegion,
        bridgeRegion,
        datedNodes,
        recentNodes,
        recentRegions: regions.filter((region) => region.recentCount > 0).length,
        totalNodes: nodes.length,
        totalEdges: edges.length,
    };
}

function _renderCognitiveMap(data) {
    if (!graphContainer) return;
    if (graph3d) { graph3d._destructor?.(); graph3d = null; }
    graphContainer.classList.add("graph-container--brain");
    graphContainer.innerHTML = "";

    const model = _buildCognitiveModel(data);
    if (!model.regions.length) {
        graphContainer.innerHTML = `<div class="brain-map-empty">No regions match current filters.</div>`;
        return;
    }

    const stageW = 980;
    const stageH = 500;
    const centerX = stageW / 2;
    const centerY = stageH / 2 - 12;
    const maxCount = Math.max(...model.regions.map((region) => region.count), 1);

    const placed = model.regions.slice(0, 8).map((region, index, list) => {
        const angle = (-Math.PI / 2) + ((Math.PI * 2) / Math.max(list.length, 1)) * index;
        const radius = region.id === "Identity Core" ? 120 : 180 + (index % 2) * 42;
        return {
            ...region,
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
            r: 24 + (region.count / maxCount) * 34,
        };
    });

    const placedMap = new Map(placed.map((region) => [region.id, region]));
    const regionCards = placed
        .map((region) => {
            const fill = levelColorCss(region.dominantLevel);
            const bridgePct = model.bridgeRegion?.bridgeWeight
                ? Math.round(((region.bridgeWeight || 0) / model.bridgeRegion.bridgeWeight) * 100)
                : 0;
            return `
                <button class="brain-map-region-card" type="button" data-region-tag="${escapeHtml(region.topTag || "")}">
                    <div class="brain-map-region-card-header">
                        <div class="brain-map-region-card-title">
                            <span class="graph-legend-dot" style="background:${fill}"></span>
                            <span>${escapeHtml(region.label)}</span>
                        </div>
                        <div class="brain-map-region-card-meta">${region.count} nodes</div>
                    </div>
                    <div class="brain-map-region-card-meta">Dominant layer: ${escapeHtml(LEVEL_LABELS[region.dominantLevel] || region.dominantLevel)}</div>
                    <div class="brain-map-bar"><div class="brain-map-bar-fill" style="width:${Math.max(8, bridgePct)}%;background:${fill}"></div></div>
                    <div class="brain-map-region-card-meta">Bridge load ${bridgePct}%${region.topTag ? ` • top tag ${escapeHtml(region.topTag)}` : ""}</div>
                </button>
            `;
        })
        .join("");

    const linkMarkup = model.links
        .map((link) => {
            const src = placedMap.get(link.source);
            const dst = placedMap.get(link.target);
            if (!src || !dst) return "";
            const strong = link.weight >= (model.links[0]?.weight || 0) * 0.55;
            return `<line class="brain-map-link ${strong ? "brain-map-link--strong" : ""}" x1="${src.x}" y1="${src.y}" x2="${dst.x}" y2="${dst.y}" stroke-width="${Math.max(1, Math.min(5, link.weight * 3.5))}"></line>`;
        })
        .join("");

    const nodeMarkup = placed
        .map((region) => {
            const fill = levelColorCss(region.dominantLevel);
            return `
                <g class="brain-map-region" data-region-tag="${escapeHtml(region.topTag || "")}" transform="translate(${region.x}, ${region.y})">
                    <circle r="${region.r}" fill="${fill}" opacity="0.78"></circle>
                    <circle r="${Math.max(8, region.r * 0.56)}" fill="rgba(255,255,255,0.08)"></circle>
                    <text class="brain-map-label" y="-2">${escapeHtml(region.label)}</text>
                    <text class="brain-map-sublabel" y="16">${region.count} nodes • ${(region.avgStrength * 100).toFixed(0)}%</text>
                </g>
            `;
        })
        .join("");

    const strongestText = model.strongestRegion
        ? `${model.strongestRegion.label} holds ${model.strongestRegion.count} nodes`
        : "No dominant region";
    const bridgeText = model.bridgeRegion
        ? `${model.bridgeRegion.label} acts as the main bridge`
        : "No bridge region";

    graphContainer.innerHTML = `
        <div class="brain-map">
            <div class="brain-map-summary">
                <div class="brain-map-card">
                    <div class="brain-map-card-label">Cognitive Regions</div>
                    <div class="brain-map-card-value">${model.regions.length}</div>
                    <div class="brain-map-card-note">Compressed semantic view instead of raw node noise.</div>
                </div>
                <div class="brain-map-card">
                    <div class="brain-map-card-label">Primary Memory Mass</div>
                    <div class="brain-map-card-value">${escapeHtml(model.strongestRegion?.label || "None")}</div>
                    <div class="brain-map-card-note">${escapeHtml(strongestText)}</div>
                </div>
                <div class="brain-map-card">
                    <div class="brain-map-card-label">Bridge Region</div>
                    <div class="brain-map-card-value">${escapeHtml(model.bridgeRegion?.label || "None")}</div>
                    <div class="brain-map-card-note">${escapeHtml(bridgeText)}</div>
                </div>
            </div>
            <div class="brain-map-stage">
                <svg class="brain-map-svg" viewBox="0 0 ${stageW} ${stageH}" preserveAspectRatio="xMidYMid meet">
                    <circle class="brain-map-core" cx="${centerX}" cy="${centerY}" r="84"></circle>
                    <circle class="brain-map-core-secondary" cx="${centerX}" cy="${centerY}" r="145"></circle>
                    <text class="brain-map-label" x="${centerX}" y="${centerY - 8}">Cognitive Core</text>
                    <text class="brain-map-sublabel" x="${centerX}" y="${centerY + 14}">${model.totalNodes} memories • ${model.totalEdges} links</text>
                    ${linkMarkup}
                    ${nodeMarkup}
                </svg>
                <div class="brain-map-caption">
                    <span>Brain mode compresses the network into semantic regions, bridge load, and dominant memory masses.</span>
                    <span>Scroll for region cards. Click a region node or card to filter the raw graph.</span>
                </div>
            </div>
            <div class="brain-map-regions">${regionCards}</div>
        </div>
    `;

    graphContainer.querySelectorAll("[data-region-tag]").forEach((el) => {
        el.addEventListener("click", () => {
            const tag = String(el.getAttribute("data-region-tag") || "").trim();
            if (!tag || !tagFilter) return;
            tagFilter.value = tag;
            renderFilteredGraph();
        });
    });
}

// ── Lazy-load 3d-force-graph (Three.js bundle) ────────────────────────────────
function _makeTextSprite(text, color = "#dbeafe") {
    if (!text || typeof THREE === "undefined") return null;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 512;
    canvas.height = 160;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = "600 52px Segoe UI";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = color;
    ctx.shadowColor = "rgba(7, 16, 28, 0.95)";
    ctx.shadowBlur = 12;
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(60, 18, 1);
    return sprite;
}

function _seedFromString(value) {
    const text = String(value || "");
    let hash = 2166136261;
    for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function _seededUnit(seed) {
    let t = seed + 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function _seededRange(seed, min, max) {
    return min + _seededUnit(seed) * (max - min);
}

function _stopBrainFx() {
    if (brainPulseTimer) {
        clearInterval(brainPulseTimer);
        brainPulseTimer = null;
    }
}

function _triggerBrainBurst(durationMs = 2600) {
    brainBurstUntil = Math.max(brainBurstUntil, Date.now() + durationMs);
}

function _startBrainFx(sceneGraph, model) {
    _stopBrainFx();
    const strongestId = model.strongestRegion ? `region:${model.strongestRegion.id}` : "";
    const bridgeId = model.bridgeRegion ? `region:${model.bridgeRegion.id}` : "";
    const nodeSeedMap = new Map(sceneGraph.nodes.map((node) => [node.id, _seedFromString(node.id)]));
    const linkSeedMap = new Map(sceneGraph.links.map((link, idx) => [`${link.source}->${link.target}:${idx}`, _seedFromString(`${link.source}:${link.target}:${idx}`)]));

    brainPulseTimer = setInterval(() => {
        if (!graph3d || currentGraphMode !== "brain") return;
        const now = Date.now();
        const burst = Math.max(0, (brainBurstUntil - now) / 2600);
        const t = now / 1000;

        sceneGraph.nodes.forEach((node) => {
            const seed = nodeSeedMap.get(node.id) || 1;
            const phase = _seededRange(seed, 0, Math.PI * 2);
            const drift = Math.sin(t * _seededRange(seed + 1, 0.35, 0.85) + phase);
            const shimmer = Math.cos(t * _seededRange(seed + 2, 0.25, 0.6) + phase * 0.7);
            let pulse = 0;
            if (node._kind === "core") pulse = 0.16 + drift * 0.06 + burst * 0.35;
            else if (node._kind === "core-shell") pulse = 0.08 + drift * 0.04 + burst * 0.2;
            else if (node._kind === "core-aura") pulse = 0.05 + shimmer * 0.03 + burst * 0.12;
            else if (node._kind === "region") pulse = 0.08 + drift * 0.05 + (node.id === strongestId ? 0.08 : 0) + (node.id === bridgeId ? 0.05 : 0) + burst * 0.16;
            else if (node._kind === "halo") pulse = 0.03 + shimmer * 0.02 + burst * 0.08;
            else pulse = 0.015 + drift * 0.015 + burst * 0.05;
            node._pulse = Math.max(0, pulse);
        });

        sceneGraph.links.forEach((link, idx) => {
            const seed = linkSeedMap.get(`${link.source}->${link.target}:${idx}`) || 1;
            const phase = _seededRange(seed, 0, Math.PI * 2);
            const wave = Math.sin(t * _seededRange(seed + 3, 0.45, 0.95) + phase);
            let pulse = 0;
            if (link._type === "bridge") pulse = 0.12 + wave * 0.07 + burst * 0.28;
            else if (link._type === "spine") pulse = 0.04 + wave * 0.03 + burst * 0.1;
            else pulse = 0.01 + wave * 0.01;
            link._pulse = Math.max(0, pulse);
        });

        graph3d.refresh();
    }, 140);
}

function _buildBrainSceneGraph(model) {
    const regions = model.regions.slice(0, 8);
    const nodes = [];
    const links = [];
    const regionIds = new Set(regions.map((region) => region.id));
    const maxCount = Math.max(...regions.map((region) => region.count), 1);
    const maxBridge = Math.max(...model.links.map((link) => link.weight), 0.1);
    const maxRecentGlow = Math.max(...regions.map((region) => region.recentGlow || 0), 0.001);

    nodes.push({
        id: "brain-core",
        _kind: "core",
        _label: "Cognitive Core",
        _fullText: `${model.totalNodes} memories • ${model.totalEdges} active links`,
        _strength: 1,
        _level: "IDENTITY",
        _color: 0x60a5fa,
        fx: 0,
        fy: 0,
        fz: 0,
    });

    nodes.push({
        id: "brain-core-shell",
        _kind: "core-shell",
        _label: "Cognitive Field",
        _fullText: "Outer shell around the active core",
        _strength: 1,
        _level: "IDENTITY",
        _color: 0x93c5fd,
        fx: 0,
        fy: 0,
        fz: 0,
    });

    nodes.push({
        id: "brain-core-aura",
        _kind: "core-aura",
        _label: "Memory Aura",
        _fullText: "Diffuse field around the cognitive core",
        _strength: 1,
        _level: "IDENTITY",
        _color: 0x7dd3fc,
        fx: 0,
        fy: 0,
        fz: 0,
    });

    regions.forEach((region, index, list) => {
        const baseSeed = _seedFromString(region.id);
        const angle = (-Math.PI / 2)
            + ((Math.PI * 2) / Math.max(list.length, 1)) * index
            + _seededRange(baseSeed + 1, -0.22, 0.22);
        const orbit = _seededRange(baseSeed + 2, 150, 245) + (region.count / maxCount) * 18;
        const fx = Math.cos(angle) * orbit + _seededRange(baseSeed + 3, -32, 32);
        const fy = Math.sin(angle) * orbit * _seededRange(baseSeed + 4, 0.62, 0.98) + _seededRange(baseSeed + 5, -26, 26);
        const fz = _seededRange(baseSeed + 6, -120, 120);
        const regionColor = levelColor(region.dominantLevel);
        const particleCount = Math.min(22, Math.max(8, Math.round(region.count / 8)));
        const haloCount = Math.min(10, Math.max(4, Math.round(region.count / 18)));
        const growth = Math.max(0, (region.recentGlow || 0) / maxRecentGlow);

        nodes.push({
            id: `region:${region.id}`,
            _kind: "region",
            _label: region.label,
            _fullText: `${region.label}: ${region.count} nodes, ${Math.round(region.avgStrength * 100)}% average strength`,
            _strength: Math.max(0.45, region.count / maxCount),
            _growth: growth,
            _recentShare: region.recentShare || 0,
            _level: region.dominantLevel,
            _color: regionColor,
            _regionTag: region.topTag || "",
            fx,
            fy,
            fz,
        });

        links.push({
            source: "brain-core",
            target: `region:${region.id}`,
            _type: "spine",
            _weight: 0.28 + region.count / maxCount,
        });

        for (let i = 0; i < particleCount; i += 1) {
            const particleSeed = baseSeed + 40 + i * 17;
            const cloudRadius = _seededRange(particleSeed + 1, 20, 62) * (0.85 + region.count / maxCount);
            const theta = _seededRange(particleSeed + 2, 0, Math.PI * 2);
            const phi = _seededRange(particleSeed + 3, -Math.PI / 3, Math.PI / 3);
            nodes.push({
                id: `particle:${region.id}:${i}`,
                _kind: "particle",
                _label: region.label,
                _fullText: `${region.label} memory particle`,
                _strength: Math.max(0.18, region.avgStrength * (0.65 + ((i % 5) * 0.08))),
                _growth: growth * _seededRange(particleSeed + 6, 0.6, 1),
                _level: region.dominantLevel,
                _color: regionColor,
                _regionTag: region.topTag || "",
                x: fx + Math.cos(theta) * Math.cos(phi) * cloudRadius,
                y: fy + Math.sin(theta) * Math.cos(phi) * cloudRadius * _seededRange(particleSeed + 4, 0.65, 1.05),
                z: fz + Math.sin(phi) * cloudRadius * _seededRange(particleSeed + 5, 0.8, 1.4),
            });
            links.push({
                source: `region:${region.id}`,
                target: `particle:${region.id}:${i}`,
                _type: "filament",
                _weight: 0.09 + region.avgStrength * 0.16,
            });
        }

        for (let i = 0; i < haloCount; i += 1) {
            const haloSeed = baseSeed + 400 + i * 29;
            const haloRadius = _seededRange(haloSeed + 1, 56, 96);
            const theta = _seededRange(haloSeed + 2, 0, Math.PI * 2);
            const phi = _seededRange(haloSeed + 3, -Math.PI / 2.6, Math.PI / 2.6);
            nodes.push({
                id: `halo:${region.id}:${i}`,
                _kind: "halo",
                _label: region.label,
                _fullText: `${region.label} ambient field`,
                _strength: Math.max(0.12, region.avgStrength * 0.45),
                _growth: growth * _seededRange(haloSeed + 6, 0.65, 1),
                _level: region.dominantLevel,
                _color: regionColor,
                _regionTag: region.topTag || "",
                x: fx + Math.cos(theta) * Math.cos(phi) * haloRadius,
                y: fy + Math.sin(theta) * Math.cos(phi) * haloRadius * _seededRange(haloSeed + 4, 0.7, 1.15),
                z: fz + Math.sin(phi) * haloRadius * _seededRange(haloSeed + 5, 0.8, 1.25),
            });
        }
    });

    model.links
        .filter((link) => regionIds.has(link.source) && regionIds.has(link.target))
        .slice(0, 10)
        .forEach((link) => {
            links.push({
                source: `region:${link.source}`,
                target: `region:${link.target}`,
                _type: "bridge",
                _weight: 0.32 + (link.weight / maxBridge) * 1.1,
                _growth: Math.max(
                    0,
                    ((regions.find((region) => region.id === link.source)?.recentGlow || 0)
                    + (regions.find((region) => region.id === link.target)?.recentGlow || 0))
                    / (2 * maxRecentGlow)
                ),
            });
        });

    return { nodes, links };
}

function _formatBrainWindowLabel(windowKey) {
    return {
        all: "All time",
        "30d": "Last 30 days",
        "7d": "Last 7 days",
        "1d": "Today",
    }[windowKey] || "All time";
}

async function _renderBrain3d(data) {
    if (!graphContainer) return;

    await _load3dLib();

    _stopBrainFx();
    if (graph3d) { graph3d._destructor?.(); graph3d = null; }
    graphContainer.classList.remove("graph-container--brain");
    graphContainer.innerHTML = "";

    const model = _buildCognitiveModel(data);
    if (!model.regions.length) {
        graphContainer.innerHTML = `<div class="brain-map-empty">No regions match current filters.</div>`;
        return;
    }

    const sceneGraph = _buildBrainSceneGraph(model);
    const W = graphContainer.clientWidth || 800;
    const H = graphContainer.clientHeight || 620;

    graph3d = ForceGraph3D({ antialias: true, alpha: true })(graphContainer)
        .width(W)
        .height(H)
        .backgroundColor("#030b16")
        .nodeRelSize(4)
        .cooldownTicks(120)
        .d3AlphaDecay(0.03)
        .d3VelocityDecay(0.4)
        .nodeColor((node) => {
            const pulse = node._pulse || 0;
            const growth = node._growth || 0;
            const recencyMode = currentBrainTimeWindow !== "all";
            if (node._kind === "core") return `rgba(96, 165, 250, ${Math.min(0.98, 0.9 + pulse * 0.45)})`;
            if (node._kind === "core-shell") return `rgba(147, 197, 253, ${0.12 + pulse * 0.18 + (recencyMode ? 0.04 : 0)})`;
            if (node._kind === "core-aura") return `rgba(125, 211, 252, ${0.05 + pulse * 0.1 + (recencyMode ? 0.03 : 0)})`;
            if (node._kind === "halo") {
                const hex = "#" + node._color.toString(16).padStart(6, "0");
                const r = parseInt(hex.slice(1, 3), 16);
                const g = parseInt(hex.slice(3, 5), 16);
                const b = parseInt(hex.slice(5, 7), 16);
                const base = recencyMode ? 0.012 : 0.04;
                return `rgba(${r}, ${g}, ${b}, ${base + pulse * 0.08 + growth * 0.2})`;
            }
            if (node._kind === "region") {
                const hex = "#" + node._color.toString(16).padStart(6, "0");
                const r = parseInt(hex.slice(1, 3), 16);
                const g = parseInt(hex.slice(3, 5), 16);
                const b = parseInt(hex.slice(5, 7), 16);
                const base = recencyMode ? 0.22 : 0.55;
                return `rgba(${r}, ${g}, ${b}, ${base + pulse * 0.18 + growth * 0.42})`;
            }
            if (node._kind === "particle") {
                const hex = "#" + node._color.toString(16).padStart(6, "0");
                const r = parseInt(hex.slice(1, 3), 16);
                const g = parseInt(hex.slice(3, 5), 16);
                const b = parseInt(hex.slice(5, 7), 16);
                const base = recencyMode ? 0.045 : 0.18;
                return `rgba(${r}, ${g}, ${b}, ${base + pulse * 0.08 + growth * 0.28})`;
            }
            return "#" + node._color.toString(16).padStart(6, "0");
        })
        .nodeVal((node) => {
            const pulse = node._pulse || 0;
            const growth = node._growth || 0;
            if (node._kind === "core") return 28 + pulse * 16;
            if (node._kind === "core-shell") return 60 + pulse * 28;
            if (node._kind === "core-aura") return 110 + pulse * 36;
            if (node._kind === "region") return 12 + node._strength * 20 + pulse * 9 + growth * 8;
            if (node._kind === "halo") return 5 + node._strength * 10 + pulse * 7 + growth * 7;
            return 1.2 + node._strength * 3.2 + pulse * 1.2 + growth * 1.8;
        })
        .nodeOpacity(0.88)
        .nodeResolution(16)
        .nodeLabel((node) => {
            if (node._kind === "core") return `Cognitive Core | ${model.totalNodes} memories | ${model.totalEdges} active links`;
            if (node._kind === "region") return `${node._label} | ${node._fullText}`;
            return "";
        })
        .linkColor((link) => {
            const pulse = link._pulse || 0;
            const growth = link._growth || 0;
            if (link._type === "bridge") return `rgba(125, 211, 252, ${0.2 + pulse * 0.35 + growth * 0.3})`;
            if (link._type === "spine") return `rgba(96, 165, 250, ${0.1 + pulse * 0.18})`;
            return "rgba(52, 211, 153, 0.04)";
        })
        .linkWidth((link) => {
            const pulse = link._pulse || 0;
            const growth = link._growth || 0;
            if (link._type === "bridge") return Math.max(1.2, link._weight * 3.2 + pulse * 2.4 + growth * 2.6);
            if (link._type === "spine") return Math.max(0.3, link._weight * 0.9 + pulse * 0.8);
            return Math.max(0.04, link._weight * 0.28);
        })
        .linkOpacity(1)
        .linkCurvature((link) => link._type === "bridge" ? 0.34 : (link._type === "spine" ? 0.08 : 0.03))
        .linkDirectionalParticles((link) => {
            const pulse = link._pulse || 0;
            if (link._type === "bridge") return 4 + Math.round(pulse * 6);
            if (link._type === "spine") return pulse > 0.06 ? 1 : 0;
            return 0;
        })
        .linkDirectionalParticleWidth((link) => {
            const pulse = link._pulse || 0;
            return link._type === "bridge" ? 2.2 + pulse * 2.1 : 0.8 + pulse * 0.7;
        })
        .linkDirectionalParticleSpeed((link) => {
            const pulse = link._pulse || 0;
            return link._type === "bridge" ? 0.004 + pulse * 0.008 : 0.0014 + pulse * 0.002;
        })
        .linkDirectionalParticleColor((link) => link._type === "bridge" ? "#a5f3fc" : "#93c5fd")
        .onEngineStop((() => {
            let zoomed = false;
            return () => {
                if (!zoomed && graph3d) {
                    zoomed = true;
                    graph3d.cameraPosition({ x: 0, y: 40, z: 520 }, { x: 0, y: 0, z: 0 }, 900);
                }
            };
        })())
        .onNodeHover((node, prevNode, event) => {
            graphContainer.style.cursor = node ? "pointer" : "default";
            if (node && event && (node._kind === "region" || node._kind === "core")) {
                showTooltip3d(getTooltip(), node, event);
            } else {
                hideTooltip();
            }
        })
        .onNodeClick((node) => {
            if (!node) return;
            if (node._regionTag && tagFilter) {
                tagFilter.value = node._regionTag;
                renderFilteredGraph();
            }
        })
        .graphData(sceneGraph);

    const strongestText = model.strongestRegion
        ? `${model.strongestRegion.label} ${model.strongestRegion.count} nodes`
        : "No dominant region";
    const bridgeText = model.bridgeRegion
        ? `${model.bridgeRegion.label} bridge`
        : "No bridge region";
    const windowLabel = _formatBrainWindowLabel(currentBrainTimeWindow);
    const recentText = currentBrainTimeWindow === "all"
        ? `${model.datedNodes}/${model.totalNodes} dated`
        : `${model.recentNodes} recent memories`;
    const activeText = currentBrainTimeWindow === "all"
        ? `${model.regions.length} mapped`
        : `${model.recentRegions} active regions`;
    const captionText = currentBrainTimeWindow === "all"
        ? "Brain mode abstracts the graph into glowing memory regions, bridge streams, and active particles."
        : `Brain mode keeps long-term memory visible while highlighting ${model.recentNodes} recent memories across ${model.recentRegions} active regions in ${windowLabel.toLowerCase()}.`;

    graphContainer.insertAdjacentHTML("beforeend", `
        <div class="brain-graph-hud">
            <div class="brain-graph-chip"><span>Window</span><strong>${escapeHtml(windowLabel)}</strong></div>
            <div class="brain-graph-chip"><span>Regions</span><strong>${model.regions.length}</strong></div>
            <div class="brain-graph-chip"><span>Recent</span><strong>${escapeHtml(recentText)}</strong></div>
            <div class="brain-graph-chip"><span>Active</span><strong>${escapeHtml(activeText)}</strong></div>
            <div class="brain-graph-chip"><span>Primary</span><strong>${escapeHtml(strongestText)}</strong></div>
            <div class="brain-graph-chip"><span>Bridge</span><strong>${escapeHtml(bridgeText)}</strong></div>
        </div>
        <div class="brain-graph-caption">${escapeHtml(captionText)}</div>
    `);

    _startBrainFx(sceneGraph, model);
}

let _fg3dLoaded = false;
async function _load3dLib() {
    if (_fg3dLoaded || typeof ForceGraph3D !== "undefined") { _fg3dLoaded = true; return; }
    await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "/js/vendor/3d-force-graph.min.js";
        s.onload = resolve; s.onerror = reject;
        document.head.appendChild(s);
    });
    _fg3dLoaded = true;
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function getTooltip() {
    let el = document.getElementById("graph-tooltip");
    if (!el) {
        el = document.createElement("div");
        el.id = "graph-tooltip";
        el.className = "graph-tooltip";
        document.body.appendChild(el);
    }
    return el;
}

function showTooltip3d(el, node, event) {
    const level  = node._level || "UNKNOWN";
    const color  = levelColorCss(level);
    const str    = ((node._strength || 0) * 100).toFixed(0);
    const tags   = (node._tags || []).slice(0, 5).join(", ");
    const text   = node._fullText || node._label || "";
    const trunc  = text.length > 140 ? text.slice(0, 140) + "\u2026" : text;

    el.innerHTML = `
        <div class="graph-tooltip-header">
            <span class="graph-legend-dot" style="background:${color}"></span>
            <strong>${LEVEL_LABELS[level] || level}</strong>
            <span style="color:var(--text-muted);margin-left:auto">str ${str}%</span>
        </div>
        <div class="graph-tooltip-text">${escapeHtml(trunc)}</div>
        ${tags ? `<div class="graph-tooltip-tags">${tags}</div>` : ""}`;
    el.style.display = "block";

    const tipW = 280;
    let left = event.clientX + 14;
    let top  = event.clientY - 14;
    if (left + tipW > window.innerWidth) left = event.clientX - tipW - 14;
    if (top < 0) top = 8;
    el.style.left = `${left}px`;
    el.style.top  = `${top}px`;
}

function hideTooltip() {
    const el = document.getElementById("graph-tooltip");
    if (el) el.style.display = "none";
}

function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}

// ── Main render ───────────────────────────────────────────────────────────────
async function renderGraph(data) {
    if (!graphContainer) return;

    await _load3dLib();

    // Destroy previous instance
    _stopBrainFx();
    if (graph3d) { graph3d._destructor?.(); graph3d = null; }
    graphContainer.classList.remove("graph-container--brain");
    graphContainer.innerHTML = "";

    const isThermal = currentGraphMode === "thermal";
    const nodes = (data.nodes || []).map(n => {
        let label = n.label || n.content || "";
        if (label.length > 40) label = label.slice(0, 40) + "\u2026";
        const level = normalizeLevel(n.level);
        const color = isThermal && n._thermal
            ? thermalColorHex(n._thermal.temp_display ?? n._thermal.temp)
            : levelColor(level);
        return {
            id:        n.id,
            _label:    label,
            _fullText: n.label || n.content || "",
            _strength: n.strength || 0.1,
            _tags:     n.tags || [],
            _level:    level,
            _color:    color,
            _thermal:  n._thermal || null,
        };
    });

    // Cap edges — keep only the strongest MAX_LINKS by weight.
    const MAX_LINKS = 1500;
    const links = (data.edges || [])
        .map(e => ({
            source: e.source,
            target: e.target,
            _weight: e.weight || 0.1,
            _thermal: e._thermal || null,
        }))
        .sort((a, b) => b._weight - a._weight)
        .slice(0, MAX_LINKS);

    console.log(`[graph] rendering ${nodes.length} nodes, ${links.length} links`);
    if (nodes.length === 0) {
        graphContainer.innerHTML = `<p style="color:var(--text-muted);padding:40px;text-align:center">No nodes match current filters.</p>`;
        return;
    }

    const W = graphContainer.clientWidth  || 800;
    const H = graphContainer.clientHeight || 620;

    graph3d = ForceGraph3D({ antialias: true, alpha: true })(graphContainer)
        .width(W)
        .height(H)
        .backgroundColor("#060d18")
        .d3AlphaDecay(0.03)
        .d3VelocityDecay(0.4)
        .warmupTicks(60)
        .cooldownTicks(200)
        // Auto-zoom once after simulation stops (tighter padding in thermal mode)
        .onEngineStop((() => {
            let done = false;
            const padding = isThermal ? 20 : 60;
            return () => { if (!done && graph3d) { done = true; graph3d.zoomToFit(400, padding); } };
        })())
        // ── Node appearance ─────────────────────────────────────────────
        .nodeColor(n => "#" + n._color.toString(16).padStart(6, "0"))
        // nodeVal = volume (radius ≈ √val), keep sizes modest
        .nodeVal(n => {
            if (isThermal && n._thermal) {
                // Size by degree (1 + log(1+deg)); slight boost for hot nodes
                const deg = Math.max(0, n._thermal.degree || 0);
                const hot = (n._thermal.temp || 0) > 0.2 ? 1.4 : 1.0;
                return (1 + Math.log(1 + deg) * 0.8) * hot;
            }
            const base = { IDENTITY: 6, DOMAIN: 3, DECISIONS: 2, WORKING: 1 }[n._level] ?? 2;
            return base * Math.max(0.3, Math.min(1.5, n._strength || 0.3));
        })
        .nodeOpacity(0.92)
        .nodeResolution(12)
        // ── Edge appearance — colored by source node level (or edge health in thermal mode) ───────────────
        .linkColor(l => {
            if (isThermal && l._thermal) {
                return EDGE_HEALTH_COLOR_CSS[l._thermal.health] || EDGE_HEALTH_COLOR_CSS.healthy;
            }
            const src = nodes.find(n => n.id === (l.source?.id ?? l.source));
            const col = src ? levelColorCss(src._level) : "#94a3b8";
            const opacity = Math.max(0.08, Math.min(0.35, (l._weight || 0.1) * 0.4));
            if (col.startsWith("#") && col.length === 7) {
                return `rgba(${parseInt(col.slice(1,3),16)},${parseInt(col.slice(3,5),16)},${parseInt(col.slice(5,7),16)},${opacity})`;
            }
            return `rgba(148,163,184,${opacity})`;
        })
        .linkWidth(l => {
            if (isThermal && l._thermal) {
                const mult = EDGE_HEALTH_WIDTH_MULT[l._thermal.health] ?? 0.6;
                return Math.max(0.3, mult);
            }
            return Math.max(0.1, (l._weight || 0.1) * 0.6);
        })
        .linkOpacity(1)              // opacity handled via rgba in linkColor
        .linkCurvature(0.1)
        // ── Particles on edges — "signal firing" / pulsing pruned edges ───
        .linkDirectionalParticles(l => {
            if (isThermal && l._thermal) {
                if (l._thermal.health === "pruned")   return 4;
                if (l._thermal.health === "weakened") return 2;
                return 0;
            }
            return (l._weight || 0) > 0.5 ? 2 : 0;
        })
        .linkDirectionalParticleWidth(l => (isThermal && l._thermal && l._thermal.health !== "healthy") ? 2.5 : 1.5)
        .linkDirectionalParticleSpeed(l => (isThermal && l._thermal?.health === "pruned") ? 0.008 : 0.004)
        .linkDirectionalParticleColor(l => {
            if (isThermal && l._thermal) {
                if (l._thermal.health === "pruned")   return "#fca5a5";
                if (l._thermal.health === "weakened") return "#fde047";
                return "rgba(148,163,184,0.6)";
            }
            const src = nodes.find(n => n.id === (l.source?.id ?? l.source));
            return src ? "#" + src._color.toString(16).padStart(6, "0") : "#ffffff";
        })
        // ── Tooltip ───────────────────────────────────────────────────────
        .onNodeHover((node, prevNode, event) => {
            graphContainer.style.cursor = node ? "pointer" : "default";
            if (node && event) {
                showTooltip3d(getTooltip(), node, event);
            } else {
                hideTooltip();
            }
        })
        .onNodeClick(node => {
            document.dispatchEvent(new CustomEvent("graph-node-selected", { detail: { id: node.id } }));
            const dist = 80;
            const distRatio = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
            graph3d.cameraPosition(
                { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
                node, 800
            );
        })
        // graphData last — starts the simulation
        .graphData({ nodes, links });

    // Keep tooltip synced with mouse position during hover
    graphContainer.addEventListener("mousemove", evt => {
        const tooltip = document.getElementById("graph-tooltip");
        if (tooltip && tooltip.style.display === "block") {
            const tipW = 280;
            let left = evt.clientX + 14;
            let top  = evt.clientY - 14;
            if (left + tipW > window.innerWidth) left = evt.clientX - tipW - 14;
            if (top < 0) top = 8;
            tooltip.style.left = `${left}px`;
            tooltip.style.top  = `${top}px`;
        }
    }, { passive: true });
}

// ── Public entry point ────────────────────────────────────────────────────────
async function renderFilteredGraph() {
    if (currentGraphMode === "thermal") {
        const data = _annotateThermal(thermalGraph);
        renderThermalStats(thermalGraph);
        await renderGraph(data);
        return;
    }
    const filtered = applyFilters(rawGraph);
    _syncBrainTimelineUI(filtered);
    const activeData = currentGraphMode === "brain" ? _annotateBrainRecency(filtered) : filtered;
    renderStats(activeData);
    if (currentGraphMode === "brain") {
        await _renderBrain3d(activeData);
        return;
    }
    await renderGraph(activeData);
}

// ── Thermal-mode helpers ──────────────────────────────────────────────────────
async function loadThermalGraph() {
    try {
        const res = await fetch("/api/glass-brain/belief-graph");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        thermalGraph = await res.json();
    } catch (e) {
        console.error("Thermal graph load error:", e);
        thermalGraph = { status: "error", message: e.message, nodes: [], edges: [] };
    }
}

// Translate belief-graph payload into the shape renderGraph expects, marking
// nodes/edges with thermal fields so the renderer can switch palette.
//
// Temperatures in beliefs.cog are typically bounded well below 1.0 (a healthy
// cold brain rarely exceeds ~0.35). A naive [0,1] color map would show almost
// everything blue. Instead, we rescale each snapshot so the hottest node in
// the dataset always maps to the top of the gradient — preserving the
// *relative* thermal contrast while staying honest about the underlying value.
function _annotateThermal(g) {
    const rawNodes = g.nodes || [];
    const temps = rawNodes.map(n => Number(n.temp) || 0).filter(t => Number.isFinite(t));
    const tMax = temps.length ? Math.max(...temps) : 1.0;
    const tMin = temps.length ? Math.min(...temps) : 0.0;
    const span = Math.max(1e-4, tMax - tMin);
    const rescale = (t) => {
        const raw = Number(t) || 0;
        // Gentle γ so mid values lift toward warm instead of staying near blue
        const norm = Math.max(0, Math.min(1, (raw - tMin) / span));
        return Math.pow(norm, 0.7);
    };

    const nodes = rawNodes.map(n => ({
        id: n.id,
        label: n.label || n.key || n.id,
        level: "thermal",              // keeps normalizeLevel happy
        tags: [],
        strength: Math.max(0.15, n.temp || 0.1),
        _thermal: {
            temp: n.temp,
            temp_display: rescale(n.temp),
            state: n.state,
            confidence: n.confidence,
            conflict_mass: n.conflict_mass,
            stability: n.stability,
            degree: n.degree,
            key: n.key,
        },
    }));
    const edges = (g.edges || []).map(e => ({
        source: e.source,
        target: e.target,
        weight: Math.max(0.05, e.conductivity || 0.05),
        _thermal: {
            health: e.health,
            conductivity: e.conductivity,
            base_conductivity: e.base_conductivity,
        },
    }));
    return { nodes, edges };
}

function renderThermalStats(g) {
    if (nodeCountEl) nodeCountEl.textContent = String(g.node_count ?? (g.nodes || []).length);
    if (edgeCountEl) edgeCountEl.textContent = String(g.edge_count ?? (g.edges || []).length);
    if (avgStrengthEl) {
        const temps = (g.nodes || []).map(n => n.temp).filter(v => Number.isFinite(v));
        if (temps.length) {
            const mean = temps.reduce((a, b) => a + b, 0) / temps.length;
            const max  = Math.max(...temps);
            avgStrengthEl.textContent = `${(mean * 100).toFixed(1)}% avg · ${(max * 100).toFixed(1)}% max`;
        } else {
            avgStrengthEl.textContent = "—";
        }
    }
    if (levelBreakdownEl) {
        const pruned   = (g.edges || []).filter(e => e.health === "pruned").length;
        const weakened = (g.edges || []).filter(e => e.health === "weakened").length;
        const healthy  = (g.edges || []).filter(e => e.health === "healthy").length;
        levelBreakdownEl.innerHTML = `
            <div class="level-breakdown-item"><span class="color-dot" style="background:${EDGE_HEALTH_COLOR_CSS.healthy.replace('0.35','1')}"></span>Healthy <b>${healthy}</b></div>
            <div class="level-breakdown-item"><span class="color-dot" style="background:${EDGE_HEALTH_COLOR_CSS.weakened.replace('0.35','1')}"></span>Weakened <b>${weakened}</b></div>
            <div class="level-breakdown-item"><span class="color-dot" style="background:${EDGE_HEALTH_COLOR_CSS.pruned.replace('0.20','1')}"></span>Pruned <b>${pruned}</b></div>
        `;
    }
    if (topTagsEl) topTagsEl.innerHTML = "";
}

const renderFilteredGraphDebounced = _debounce(renderFilteredGraph, 300);

export async function loadGraph() {
    try {
        rawGraph = await window.apiClient.getGraph(scopeFilter?.value || "user");
        await renderFilteredGraph();
    } catch (e) {
        console.error("Graph load error:", e);
        if (graphContainer) graphContainer.innerHTML =
            `<p style="color:var(--red);padding:20px">Failed to load graph: ${e.message}</p>`;
    }
}

// ── Event listeners ───────────────────────────────────────────────────────────
resetBtn?.addEventListener("click", () => {
    if (scopeFilter) scopeFilter.value = "user";
    if (levelFilter) levelFilter.value = "all";
    if (tagFilter)   tagFilter.value = "";
    if (edgeSlider)  { edgeSlider.value = "0"; if (edgeValueEl) edgeValueEl.textContent = "0%"; }
    loadGraph();
});

scopeFilter?.addEventListener("change", loadGraph);
levelFilter?.addEventListener("change", renderFilteredGraphDebounced);
tagFilter?.addEventListener("keydown", e => { if (e.key === "Enter") renderFilteredGraph(); });
tagFilter?.addEventListener("input", renderFilteredGraphDebounced);
graphModeButtons.forEach((btn) => {
    btn.addEventListener("click", () => setGraphMode(btn.dataset.graphMode || "network"));
});
graphTimeButtons.forEach((btn) => {
    btn.addEventListener("click", () => setBrainTimeWindow(btn.dataset.graphTime || "all"));
});
graphTimelineRange?.addEventListener("input", () => {
    if (!brainTimelineBounds) return;
    const offsetDays = Number.parseInt(graphTimelineRange.value || "0", 10) || 0;
    currentBrainPlaybackSec = brainTimelineBounds.minSec + (offsetDays * 86400);
    if (graphTimelineLabel) {
        graphTimelineLabel.textContent = _formatTimelineDate(currentBrainPlaybackSec);
    }
    renderFilteredGraph();
});
graphTimelinePlayBtn?.addEventListener("click", () => {
    if (!brainTimelineBounds) return;
    if (brainTimelineTimer) {
        _stopBrainTimelinePlayback();
        return;
    }
    graphTimelinePlayBtn.classList.add("active");
    graphTimelinePlayBtn.textContent = "Pause";
    brainTimelineTimer = setInterval(() => {
        if (!graphTimelineRange || !brainTimelineBounds) {
            _stopBrainTimelinePlayback();
            return;
        }
        const max = Number.parseInt(graphTimelineRange.max || "0", 10) || 0;
        let next = (Number.parseInt(graphTimelineRange.value || "0", 10) || 0) + 1;
        if (next > max) {
            next = 0;
        }
        graphTimelineRange.value = String(next);
        currentBrainPlaybackSec = brainTimelineBounds.minSec + (next * 86400);
        if (graphTimelineLabel) {
            graphTimelineLabel.textContent = _formatTimelineDate(currentBrainPlaybackSec);
        }
        renderFilteredGraph();
    }, 900);
});

document.addEventListener("brain-activation-burst", () => {
    _triggerBrainBurst();
});

if (edgeSlider) {
    // Sync label with initial value
    if (edgeValueEl) edgeValueEl.textContent = `${Math.round(parseFloat(edgeSlider.value) * 100)}%`;
    edgeSlider.addEventListener("input", () => {
        if (edgeValueEl) edgeValueEl.textContent = `${Math.round(edgeSlider.value * 100)}%`;
        renderFilteredGraphDebounced();
    });
}

// Resize handler — keep 3D canvas filling container
window.addEventListener("resize", _debounce(() => {
    if (graph3d && graphContainer) {
        graph3d.width(graphContainer.clientWidth);
        graph3d.height(graphContainer.clientHeight);
    }
}, 200));
