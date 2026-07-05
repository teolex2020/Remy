/**
 * History View - browse, filter, and read past session transcripts.
 * Conversation-first rendering: user/assistant messages stay primary,
 * tool activity is grouped and collapsed underneath each turn.
 */

import { showConfirm } from "./ui.js";

const listEl = document.getElementById("history-list");
const detailEl = document.getElementById("history-detail");
const periodSelect = document.getElementById("history-period");
const countEl = document.getElementById("history-count");
const clearBtn = document.getElementById("btn-history-clear");

let allSessions = [];
let currentFile = null;

export async function loadHistory() {
    listEl.innerHTML = Array.from({length: 6}, () => `
        <div class="skeleton-row" style="padding:10px 14px;border-bottom:1px solid var(--border)">
            <div class="skeleton skeleton-line skeleton-short" style="width:48px"></div>
            <div class="skeleton skeleton-line skeleton-medium" style="flex:1;margin:0 12px"></div>
            <div class="skeleton skeleton-line" style="width:32px"></div>
        </div>`).join('');
    try {
        const res = await fetch("/api/history");
        const data = await res.json();
        allSessions = data.sessions || [];
        renderFiltered();
    } catch (e) {
        listEl.innerHTML = `<div class="empty-state" style="color:var(--red)">Failed to load history: ${e.message}</div>`;
    }
}

function getFilteredSessions() {
    const period = periodSelect?.value || "all";
    if (period === "all") return allSessions;

    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;

    let cutoff = null;
    if (period === "today") {
        cutoff = todayStart;
    } else if (period === "week") {
        cutoff = todayStart - 6 * 86400;
    } else if (period === "month") {
        cutoff = todayStart - 29 * 86400;
    }

    return allSessions.filter((s) => s.timestamp >= cutoff);
}

function renderFiltered() {
    const filtered = getFilteredSessions();
    if (countEl) countEl.textContent = `${filtered.length} session${filtered.length !== 1 ? "s" : ""}`;
    // Reset expanded state on each filter change so first group auto-opens
    _expandedDates.clear();
    _firstRender = true;
    renderList(filtered);
}

// Track which date groups are expanded; default: expand first group
const _expandedDates = new Set();
let _firstRender = true;

function renderList(sessions) {
    if (!sessions.length) {
        listEl.innerHTML = `<div class="empty-state">No sessions found for this period.</div>`;
        return;
    }

    const groups = new Map();
    for (const s of sessions) {
        const dateKey = s.date_str.split(" ")[0];
        if (!groups.has(dateKey)) groups.set(dateKey, []);
        groups.get(dateKey).push(s);
    }

    // On first render auto-expand the latest date group
    if (_firstRender) {
        _firstRender = false;
        const firstDate = groups.keys().next().value;
        if (firstDate) _expandedDates.add(firstDate);
    }

    let html = "";
    for (const [date, items] of groups) {
        const isExpanded = _expandedDates.has(date);
        const activeInGroup = items.some((s) => s.filename === currentFile);
        const label = formatDateLabel(date);
        const totalSize = items.reduce((sum, s) => sum + (s.size || 0), 0);

        html += `
            <div class="history-date-header${isExpanded ? " expanded" : ""}" data-date="${date}">
                <span class="history-date-chevron">${isExpanded ? "▾" : "▸"}</span>
                <span class="history-date-label">${label}</span>
                <span class="history-date-meta">${items.length} sessions · ${formatSize(totalSize)}</span>
            </div>
            <div class="history-date-sessions${isExpanded ? "" : " hidden"}" data-date="${date}">
        `;

        for (const s of items) {
            const time = s.date_str.split(" ")[1] || "";
            const active = currentFile === s.filename ? " active" : "";
            html += `
                <div class="history-item${active}" data-file="${s.filename}">
                    <div class="history-item-time">${time}</div>
                    <div class="history-item-info">
                        <span class="history-item-size">${formatSize(s.size)}</span>
                    </div>
                    <button class="btn-icon history-delete-btn" data-file="${s.filename}" title="Delete">&#10005;</button>
                </div>
            `;
        }

        html += `</div>`;
    }

    listEl.innerHTML = html;

    // Date header toggle
    listEl.querySelectorAll(".history-date-header").forEach((header) => {
        header.addEventListener("click", () => {
            const date = header.dataset.date;
            const sessions = listEl.querySelector(`.history-date-sessions[data-date="${date}"]`);
            const isNowExpanded = _expandedDates.has(date);
            if (isNowExpanded) {
                _expandedDates.delete(date);
                header.classList.remove("expanded");
                header.querySelector(".history-date-chevron").textContent = "▸";
                sessions?.classList.add("hidden");
            } else {
                _expandedDates.add(date);
                header.classList.add("expanded");
                header.querySelector(".history-date-chevron").textContent = "▾";
                sessions?.classList.remove("hidden");
            }
        });
    });

    listEl.querySelectorAll(".history-item").forEach((el) => {
        el.addEventListener("click", (e) => {
            if (e.target.closest(".history-delete-btn")) return;
            listEl.querySelectorAll(".history-item").forEach((item) => item.classList.remove("active"));
            el.classList.add("active");
            loadSessionDetail(el.dataset.file);
        });
    });

    listEl.querySelectorAll(".history-delete-btn").forEach((btn) => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const file = btn.dataset.file;
            const confirmed = await showConfirm("Delete Session", "Delete this session log?");
            if (!confirmed) return;
            try {
                await fetch(`/api/history/${file}`, { method: "DELETE" });
                allSessions = allSessions.filter((s) => s.filename !== file);
                if (currentFile === file) {
                    currentFile = null;
                    detailEl.innerHTML = `<div class="empty-state">Select a session to view the conversation</div>`;
                }
                renderFiltered();
            } catch (err) {
                console.error("Delete failed:", err);
            }
        });
    });
}

function formatDateLabel(dateStr) {
    const now = new Date();
    const today = now.toISOString().slice(0, 10);
    const yesterday = new Date(now - 86400000).toISOString().slice(0, 10);

    if (dateStr === today) return "Today";
    if (dateStr === yesterday) return "Yesterday";

    try {
        const d = new Date(`${dateStr}T00:00:00`);
        return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
    } catch {
        return dateStr;
    }
}

async function loadSessionDetail(filename) {
    currentFile = filename;
    detailEl.innerHTML = `
        <div style="padding:20px">
            ${Array.from({length: 5}, (_, i) => `
                <div class="skeleton-row" style="margin-bottom:14px;justify-content:${i % 2 ? 'flex-end' : 'flex-start'}">
                    <div class="skeleton skeleton-line ${i % 2 ? 'skeleton-medium' : 'skeleton-long'}" style="max-width:70%"></div>
                </div>`).join('')}
        </div>`;

    try {
        const res = await fetch(`/api/history/${filename}`);
        if (!res.ok) throw new Error("Failed to load");
        const data = await res.json();
        renderTranscript(data);
    } catch (e) {
        detailEl.innerHTML = `<div style="padding:20px;color:var(--red)">Error: ${e.message}</div>`;
    }
}

function renderTranscript(data) {
    const log = Array.isArray(data.log) ? data.log : [];
    if (!log.length) {
        detailEl.innerHTML = `<div class="empty-state">Empty session.</div>`;
        return;
    }

    const userMsgs = log.filter((e) => e.type === "user_text" || e.type === "user_voice").length;
    const agentMsgs = log.filter((e) => e.type === "model_response" || e.type === "text" || e.type === "final").length;
    const toolCalls = log.filter((e) => e.type === "tool_call").length;
    const workerEvents = log.filter((e) => e.type === "worker_event").length;

    const turns = buildTurns(log);
    const transcriptHtml = turns.map((turn, index) => renderTurn(turn, index)).join("");
    const noteHtml = agentMsgs === 0
        ? `<div class="history-note-banner">This session includes tool activity, but no final model reply was saved in the transcript.</div>`
        : "";
    const statsBits = [`${userMsgs} user`, `${agentMsgs} agent`, `${toolCalls} tools`];
    if (workerEvents) statsBits.push(`${workerEvents} worker events`);

    detailEl.innerHTML = `
        <div class="history-transcript-header">
            <span>${data.timestamp || ""}</span>
            <span style="color:var(--text-muted)">${statsBits.join(" · ")}</span>
        </div>
        <div class="history-transcript-messages">
            ${noteHtml}
            ${transcriptHtml}
        </div>
    `;
}

function buildTurns(log) {
    const turns = [];
    let current = null;

    for (const entry of log) {
        if (entry.type === "user_text" || entry.type === "user_voice") {
            if (current) turns.push(current);
            current = { user: entry, tools: [], agent: [] };
            continue;
        }

        if (entry.type === "tool_call" || entry.type === "worker_event") {
            if (!current) current = { user: null, tools: [], agent: [] };
            current.tools.push(entry);
            continue;
        }

        if (entry.type === "model_response" || entry.type === "text" || entry.type === "final") {
            if (!current) current = { user: null, tools: [], agent: [] };
            current.agent.push(entry);
        }
    }

    if (current) turns.push(current);
    return turns;
}

function renderTurn(turn, index) {
    const parts = [];

    if (turn.user) {
        const prefix = turn.user.type === "user_voice" ? "[Voice] " : "";
        const text = decodeDisplayText(turn.user.text || "");
        parts.push(`<div class="chat-msg user">${escapeHtml(prefix + text)}</div>`);
    }

    for (const entry of turn.agent) {
        const text = decodeDisplayText(entry.full_text || entry.text || entry.content || "");
        parts.push(`<div class="chat-msg assistant">${formatMarkdown(text)}</div>`);
    }

    if (turn.tools.length) {
        const openAttr = turn.agent.length ? "" : "open";
        const toolItems = turn.tools.map((entry) => {
            if (entry.type === "worker_event") {
                const eventName = entry.event || "worker_event";
                let label = eventName;
                if (eventName === "worker_role_resolution") {
                    const requested = entry.requested_role || entry.role || "";
                    const resolved = entry.resolved_role || "";
                    const status = entry.status || "";
                    label = status === "unknown_role"
                        ? `role resolution failed: ${requested}`
                        : `role ${requested}${resolved && resolved !== requested ? ` -> ${resolved}` : ""}`;
                } else if (eventName === "worker_started") {
                    label = `started ${entry.worker_channel || entry.role || "worker"}`;
                } else if (eventName === "worker_completed") {
                    label = `${entry.role || "worker"} ${entry.status || "completed"}`;
                }
                const metaBits = [];
                if (entry.worker_channel) metaBits.push(entry.worker_channel);
                if (entry.timeout_sec) metaBits.push(`${entry.timeout_sec}s`);
                if (entry.step_budget) metaBits.push(`${entry.step_budget} steps`);
                if (entry.elapsed_sec) metaBits.push(`${entry.elapsed_sec}s`);
                if (entry.tool_calls !== undefined && entry.tool_calls !== null) metaBits.push(`${entry.tool_calls} calls`);
                if (entry.error) metaBits.push(String(entry.error));
                const meta = metaBits.join(" | ");
                return `
                    <div class="history-tool-item history-worker-event">
                        <div class="history-tool-item-head">
                            <span class="history-tool-name">worker</span>
                            <span class="history-tool-args">${escapeHtml(label)}</span>
                        </div>
                        ${meta ? `<div class="history-tool-result">${escapeHtml(truncate(meta, 220))}</div>` : ""}
                    </div>
                `;
            }

            const toolName = escapeHtml(entry.tool || "unknown");
            const argsPreview = entry.args ? truncate(JSON.stringify(entry.args), 140) : "";
            const resultPreview = entry.result ? truncate(previewToolResult(entry.result), 220) : "";
            return `
                <div class="history-tool-item">
                    <div class="history-tool-item-head">
                        <span class="history-tool-name">${toolName}</span>
                        ${argsPreview ? `<span class="history-tool-args">${escapeHtml(argsPreview)}</span>` : ""}
                    </div>
                    ${resultPreview ? `<div class="history-tool-result">${escapeHtml(resultPreview)}</div>` : ""}
                </div>
            `;
        }).join("");

        parts.push(`
            <details class="history-tool-group" ${openAttr}>
                <summary>
                    <span>Agent activity</span>
                    <span class="history-tool-count">${turn.tools.length} item${turn.tools.length !== 1 ? "s" : ""}</span>
                </summary>
                <div class="history-tool-list">${toolItems}</div>
            </details>
        `);
    }

    return `<div class="history-turn" data-turn="${index}">${parts.join("")}</div>`;
}

function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(decodeDisplayText(text));
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
    html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    html = html.replace(/\n/g, "<br>");
    return html;
}

function truncate(str, max) {
    if (!str) return "";
    return str.length > max ? `${str.slice(0, max)}...` : str;
}

function previewToolResult(result) {
    if (typeof result !== "string") {
        try {
            return JSON.stringify(result);
        } catch {
            return String(result);
        }
    }
    return decodeDisplayText(result).replace(/\s+/g, " ").trim();
}

function decodeDisplayText(text) {
    if (!text || typeof text !== "string") return text || "";
    if (!/[Р РЎРѓГђГ‘]/.test(text)) return text;
    try {
        return decodeURIComponent(escape(text));
    } catch {
        return text;
    }
}

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1024).toFixed(1)} KB`;
}

periodSelect?.addEventListener("change", renderFiltered);

clearBtn?.addEventListener("click", async () => {
    const confirmed = await showConfirm("Clear All History", `Delete all ${allSessions.length} session logs? This cannot be undone.`);
    if (!confirmed) return;
    try {
        await fetch("/api/history", { method: "DELETE" });
        allSessions = [];
        currentFile = null;
        detailEl.innerHTML = `<div class="empty-state">Select a session to view the conversation</div>`;
        renderFiltered();
    } catch (e) {
        console.error("Clear history failed:", e);
    }
});
