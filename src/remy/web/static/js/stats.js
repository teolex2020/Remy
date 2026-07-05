/**
 * Stats view.
 */

import { skeletonGrid, EMPTY, errorState } from "./ui.js";

function getStatsElements() {
    return {
        cardsEl: document.getElementById("stats-cards"),
        metricsEl: document.getElementById("metrics-cards"),
    };
}

let _statsRefreshTimer = null;
const _STATS_REFRESH_MS = 15000;

export async function loadStats() {
    const { cardsEl, metricsEl } = getStatsElements();
    if (!cardsEl) return;
    stopHealthRefresh();
    cardsEl.innerHTML = skeletonGrid(6);
    if (metricsEl) metricsEl.innerHTML = "";
    try {
        const [statsData, metricsData, execData, evalData] = await Promise.all([
            window.apiClient.getStats(),
            window.apiClient.getTaskMetrics(),
            window.apiClient.getExecutionLogSummary(),
            window.apiClient.getEvalMetrics().catch(() => ({})),
        ]);
        renderStats(cardsEl, metricsEl, statsData, metricsData, execData, evalData);
        _statsRefreshTimer = setInterval(() => {
            if (!document.getElementById("view-stats")?.classList.contains("active")) {
                stopHealthRefresh();
                return;
            }
            loadStats();
        }, _STATS_REFRESH_MS);
    } catch (err) {
        console.error("Failed to load stats:", err);
        cardsEl.innerHTML = errorState(err.message);
    }
}

export function stopHealthRefresh() {
    if (_statsRefreshTimer) {
        clearInterval(_statsRefreshTimer);
        _statsRefreshTimer = null;
    }
}

function renderStats(cardsEl, metricsEl, statsData, metricsData, execData, evalData) {
    const usage = statsData.usage || {};
    const lifetime = usage.lifetime || {};
    const session = usage.session || {};
    const lifetimeTotalTokens = lifetime.total_tokens ?? usage.total_tokens ?? 0;
    const sessionTotalTokens = session.total_tokens || 0;
    const metrics = metricsData.totals || metricsData || {};
    const families = Object.entries(metricsData.families || {}).sort((a, b) => familyOrder(a[0]) - familyOrder(b[0]));

    if (metricsEl) metricsEl.innerHTML = "";
    cardsEl.className = "stats-shell";

    const activeGoals = metrics.active_goals ?? 0;
    const totalGoals = metrics.total_goals ?? 0;
    const completionRate = Math.round((metrics.completion_rate || 0) * 100);
    const blockedGoals = metrics.blocked_goals ?? 0;
    const totalCycles = metrics.total_cycles ?? 0;
    const brainRecords = statsData.total_records || 0;
    const successPct = completionRate;
    const blockedPct = Math.round((metrics.blocked_rate || 0) * 100);

    cardsEl.innerHTML = `
        <div class="dash-grid">

            <!-- Hero: success rate -->
            <div class="dash-card dash-card-hero">
                <div class="dash-card-icon">&#9889;</div>
                <div class="dash-card-label">Task success rate</div>
                <div class="dash-card-big">${successPct}%</div>
                <div class="dash-progress-wrap">
                    <div class="dash-progress-fill dash-progress-green" style="width:${successPct}%"></div>
                </div>
                <div class="dash-card-sub">${totalGoals} tasks total · ${activeGoals} active</div>
            </div>

            <!-- Cycles -->
            <div class="dash-card">
                <div class="dash-card-icon">&#128257;</div>
                <div class="dash-card-label">Cycles completed</div>
                <div class="dash-card-big">${totalCycles.toLocaleString()}</div>
                <div class="dash-card-sub">autonomous mode iterations</div>
            </div>

            <!-- Blocked -->
            <div class="dash-card ${blockedGoals > 0 ? "dash-card-warn" : ""}">
                <div class="dash-card-icon">${blockedGoals > 0 ? "&#9888;" : "&#9989;"}</div>
                <div class="dash-card-label">Blocked</div>
                <div class="dash-card-big">${blockedGoals}</div>
                ${blockedPct > 0 ? `<div class="dash-progress-wrap"><div class="dash-progress-fill dash-progress-red" style="width:${blockedPct}%"></div></div>` : ""}
                <div class="dash-card-sub">${blockedGoals === 0 ? "All running without blocks" : `${blockedPct}% of all tasks`}</div>
            </div>

            <!-- Memory -->
            <div class="dash-card">
                <div class="dash-card-icon">&#129504;</div>
                <div class="dash-card-label">Memory records</div>
                <div class="dash-card-big">${brainRecords.toLocaleString()}</div>
                <div class="dash-card-sub">knowledge accumulated by Remy</div>
            </div>

            <!-- Tokens -->
            <div class="dash-card">
                <div class="dash-card-icon">&#128201;</div>
                <div class="dash-card-label">Tokens used</div>
                <div class="dash-card-big">${formatTokensShort(lifetimeTotalTokens)}</div>
                <div class="dash-card-sub">lifetime · ${formatTokensShort(sessionTotalTokens)} this session</div>
            </div>

        </div>

        ${families.length ? `
        <div class="dash-section-title">Task types</div>
        <div class="dash-families">
            ${families.map(([family, info]) => renderFamilyCard(family, info)).join("")}
        </div>` : ""}

        ${renderMemoryQuality(evalData)}
    `;
}

function renderMemoryQuality(evalData) {
    const e = evalData || {};
    if (!e.total_responses) return "";

    // empty_recall_rate is only populated once the AuraSDK exposes
    // recall_hit_stats() (rebuilt ≥1.5.5). Until then recall_calls_total is 0,
    // so we hide the card rather than show a misleading 0%.
    const hasEmptyRecall = (e.recall_calls_total || 0) > 0;
    const emptyRate = hasEmptyRecall ? e.empty_recall_rate : null;

    const cells = [];
    if (hasEmptyRecall) {
        // Lower is better: a low empty-recall rate means searches usually find something.
        cells.push(memoryStatCell(
            "Empty recall rate", `${emptyRate}%`,
            `${e.empty_recall_count_total}/${e.recall_calls_total} recalls found nothing`,
            emptyRate <= 25 ? "green" : emptyRate <= 50 ? "amber" : "red",
        ));
    }
    cells.push(memoryStatCell(
        "Recall hit rate", `${e.avg_recall_hit_rate ?? 0}%`,
        "recalled items the reply used",
        (e.avg_recall_hit_rate ?? 0) >= 50 ? "green" : "amber",
    ));
    cells.push(memoryStatCell(
        "Duplicate writes", `${e.avg_duplicate_store_rate ?? 0}%`,
        "stores that hit an existing record",
        (e.avg_duplicate_store_rate ?? 0) <= 15 ? "green" : "red",
    ));
    cells.push(memoryStatCell(
        "Recall usage", `${e.recall_usage_rate ?? 0}%`,
        "responses that consulted memory",
        "neutral",
    ));

    return `
        <div class="dash-section-title">Memory quality</div>
        <div class="dash-families">
            ${cells.join("")}
        </div>`;
}

function memoryStatCell(label, value, sub, tone) {
    return `
        <div class="stat-card stat-card-mem-${tone}">
            <div class="stat-card-title">${escapeHtml(label)}</div>
            <div class="stat-card-value">${escapeHtml(value)}</div>
            <div class="goal-card-meta"><span>${escapeHtml(sub)}</span></div>
        </div>`;
}

function formatTokensShort(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
    if (n >= 1_000) return (n / 1_000).toFixed(0) + "K";
    return String(n);
}

function renderFamilyCard(family, info) {
    const label = info.pack_label || family.replace(/_/g, " ");
    return `
        <div class="stat-card stat-card-${familyAccentClass(family)}">
            <div class="stat-card-title">${escapeHtml(label)}</div>
            <div class="stat-card-value">${formatPercent(info.completion_rate)}</div>
            <div class="stat-card-rates">
                ${ratePill("success", info.completion_rate)}
                ${ratePill("verified", info.verified_rate)}
                ${ratePill("blocked", info.blocked_rate)}
                ${ratePill("repeated", info.repeated_failure_rate)}
            </div>
            <div class="goal-card-meta">
                <span>${info.total_cycles || 0} cycles</span>
                <span>${formatDuration(info.avg_duration_ms || 0)} avg</span>
                ${info.memory_assisted ? `<span>${info.memory_assisted} memory</span>` : ""}
            </div>
        </div>
    `;
}

function ratePill(label, rate) {
    const tone = label === "blocked" || label === "repeated" ? "danger" : "success";
    return `<span class="stat-rate stat-rate-${tone}">${escapeHtml(label)} ${formatPercent(rate)}</span>`;
}

function familyOrder(family) {
    const order = { signup_operator: 0, publisher: 1, market_research: 2, monitoring: 3, general: 4 };
    return order[family] ?? 99;
}

function familyAccentClass(family) {
    return {
        signup_operator: "signup",
        publisher: "publisher",
        market_research: "research",
        monitoring: "monitoring",
        general: "general",
    }[family] || "general";
}

function formatPercent(rate) {
    return `${Math.round((rate || 0) * 100)}%`;
}

function formatDuration(ms) {
    if (!ms) return "0s";
    if (ms >= 60000) return `${Math.round(ms / 60000)}m`;
    return `${Math.round(ms / 1000)}s`;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
}
