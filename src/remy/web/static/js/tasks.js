/**
 * Tasks View — reminders and todos for the user + agent activity summary.
 */

import { skeletonCards, EMPTY, errorState } from "./ui.js";
import { showConfirm } from "./ui.js";

let _statusFilter = "all";
let _daysFilter = 7;

// ── Entry point ───────────────────────────────────────────────────────────────

export async function loadTasks() {
    _renderControls();
    const container = document.getElementById("tasks-content");
    if (!container) return;
    container.innerHTML = skeletonCards(4);

    try {
        const data = await window.apiClient.getTodos(_statusFilter, null, 250, _daysFilter);
        renderTasks(container, data.todos || []);
    } catch (err) {
        console.error("Failed to load tasks:", err);
        container.innerHTML = errorState(err.message);
    }
}

function _renderControls() {
    const bar = document.getElementById("tasks-tab-controls");
    if (!bar) return;
    bar.innerHTML = `
        <select id="tasks-filter" class="input" style="width:auto;padding:6px 10px;">
            <option value="all"    ${_statusFilter==="all"   ?"selected":""}>All</option>
            <option value="active" ${_statusFilter==="active"?"selected":""}>Active</option>
            <option value="done"   ${_statusFilter==="done"  ?"selected":""}>Done</option>
        </select>
        <select id="tasks-period" class="input" style="width:auto;padding:6px 10px;">
            <option value="1"    ${_daysFilter===1   ?"selected":""}>Today</option>
            <option value="7"    ${_daysFilter===7   ?"selected":""}>Week</option>
            <option value="30"   ${_daysFilter===30  ?"selected":""}>Month</option>
            <option value="3650" ${_daysFilter===3650?"selected":""}>All time</option>
        </select>
        <button id="btn-add-task" class="btn btn-primary">+ Add Task</button>`;
    document.getElementById("tasks-filter")?.addEventListener("change", e => {
        _statusFilter = e.target.value; loadTasks();
    });
    document.getElementById("tasks-period")?.addEventListener("change", e => {
        _daysFilter = Number(e.target.value || 7); loadTasks();
    });
    document.getElementById("btn-add-task")?.addEventListener("click", openAddModal);
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderTasks(container, items) {
    const userItems  = items.filter(i => i.source === "todo" || i.source === "reminder");
    const agentItems = items.filter(i => i.source === "goal");
    const activeAgent = agentItems.filter(i => i.status === "in_progress");

    let html = renderUserSection(userItems);
    if (activeAgent.length > 0) html += renderAgentSection(activeAgent);

    container.innerHTML = html || EMPTY.activity();
    bindActions(container);
}

function renderUserSection(items) {
    const pending = items.filter(i => i.status !== "done");
    const done    = items.filter(i => i.status === "done");

    if (!items.length) {
        return `<section class="task-section">${EMPTY.activity()}</section>`;
    }
    return `<section class="task-section">${[...pending, ...done].map(renderReminderCard).join("")}</section>`;
}

function renderAgentSection(items) {
    const cards = items.map(goal => {
        const statusText = goal.status === "in_progress" ? "Working..." : goal.status || "pending";
        return `
        <div class="task-agent-card">
            <div class="task-agent-indicator"></div>
            <div class="task-agent-body">
                <div class="task-agent-title">${escapeHtml(goal.title)}</div>
                ${goal.blocked_reason
                    ? `<div class="task-agent-note task-agent-note-blocked">Blocked: ${escapeHtml(goal.blocked_reason)}</div>`
                    : `<div class="task-agent-note">${escapeHtml(statusText)}</div>`}
            </div>
        </div>`;
    }).join("");
    return `
        <section class="task-section task-section-agent">
            <div class="task-section-header"><h3>Remy is working on</h3></div>
            ${cards}
        </section>`;
}

function renderReminderCard(todo) {
    const isDone   = todo.status === "done";
    const metaBits = [
        todo.due_date ? `Due ${formatDue(todo.due_date)}` : "",
        todo.repeat   ? `Repeats ${todo.repeat}` : "",
        !isDone && todo.created_at ? formatRelativeTime(todo.created_at) : "",
    ].filter(Boolean);

    return `
        <div class="task-reminder-card ${isDone ? "task-reminder-done" : ""}" data-task-id="${escapeHtml(todo.id)}">
            <label class="task-reminder-check">
                <input type="checkbox" ${isDone ? "checked" : ""} data-task-id="${escapeHtml(todo.id)}">
                <span class="task-checkmark"></span>
            </label>
            <div class="task-reminder-main">
                <div class="task-reminder-title">${escapeHtml(todo.title)}</div>
                ${metaBits.length ? `<div class="task-reminder-meta">${metaBits.map(escapeHtml).join(" · ")}</div>` : ""}
            </div>
            <div class="task-reminder-actions">
                ${todo.priority && todo.priority !== "medium" ? priorityDot(todo.priority) : ""}
                <button class="btn-icon task-delete-btn" data-task-id="${escapeHtml(todo.id)}" title="Delete">&times;</button>
            </div>
        </div>`;
}

function bindActions(container) {
    container.querySelectorAll('input[type="checkbox"][data-task-id]').forEach(cb => {
        cb.addEventListener("change", async () => {
            try {
                await window.apiClient.toggleTodo(cb.dataset.taskId);
                loadTasks();
            } catch (err) { console.error("Toggle failed:", err); }
        });
    });
    container.querySelectorAll(".task-delete-btn").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const confirmed = await showConfirm("Delete task", "Delete this task?");
            if (!confirmed) return;
            try {
                await window.apiClient.deleteTodo(btn.dataset.taskId);
                loadTasks();
            } catch (err) { console.error("Delete failed:", err); }
        });
    });
}

// ── Add task modal ────────────────────────────────────────────────────────────

function openAddModal() {
    let modal = document.getElementById("add-task-modal");
    if (!modal) {
        modal = document.createElement("div");
        modal.id = "add-task-modal";
        modal.className = "modal-overlay";
        modal.innerHTML = `
            <div class="modal-box" style="max-width:420px">
                <div class="modal-header">
                    <h3>New Task</h3>
                    <button class="btn-icon" id="add-task-close">&times;</button>
                </div>
                <div class="modal-body">
                    <input id="add-task-title" class="input" placeholder="What needs to be done?" autofocus style="width:100%;margin-bottom:10px">
                    <div style="display:flex;gap:10px">
                        <select id="add-task-priority" class="input" style="flex:1">
                            <option value="low">Low priority</option>
                            <option value="medium" selected>Medium priority</option>
                            <option value="high">High priority</option>
                        </select>
                        <input id="add-task-due" class="input" type="date" style="flex:1" placeholder="Due date">
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline" id="add-task-cancel">Cancel</button>
                    <button class="btn btn-primary" id="add-task-save">Add Task</button>
                </div>
            </div>`;
        document.body.appendChild(modal);
        modal.addEventListener("click", e => { if (e.target === modal) closeAddModal(); });
        document.getElementById("add-task-close").addEventListener("click", closeAddModal);
        document.getElementById("add-task-cancel").addEventListener("click", closeAddModal);
        document.getElementById("add-task-save").addEventListener("click", saveNewTask);
        document.getElementById("add-task-title").addEventListener("keydown", e => {
            if (e.key === "Enter") saveNewTask();
            if (e.key === "Escape") closeAddModal();
        });
    }
    modal.classList.add("active");
    document.getElementById("add-task-title").value = "";
    document.getElementById("add-task-due").value = "";
    document.getElementById("add-task-priority").value = "medium";
    setTimeout(() => document.getElementById("add-task-title")?.focus(), 50);
}

function closeAddModal() {
    document.getElementById("add-task-modal")?.classList.remove("active");
}

async function saveNewTask() {
    const title = document.getElementById("add-task-title")?.value.trim();
    if (!title) { document.getElementById("add-task-title")?.focus(); return; }
    const priority = document.getElementById("add-task-priority")?.value || "medium";
    const due_date = document.getElementById("add-task-due")?.value || null;
    const saveBtn  = document.getElementById("add-task-save");
    saveBtn.disabled = true;
    saveBtn.textContent = "Adding...";
    try {
        await window.apiClient.createTodo({ title, priority, due_date, category: "personal" });
        closeAddModal();
        loadTasks();
    } catch (err) {
        alert(`Failed to create task: ${err.message}`);
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = "Add Task";
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function priorityDot(priority) {
    const colors = { high: "var(--red)", critical: "var(--red)", low: "var(--text-muted)" };
    return `<span style="width:7px;height:7px;border-radius:50%;background:${colors[priority]||"var(--yellow)"};display:inline-block;flex-shrink:0" title="${escapeHtml(priority)}"></span>`;
}

function formatRelativeTime(value) {
    if (!value) return "";
    const dt = new Date(value);
    if (Number.isNaN(dt.getTime())) return "";
    const diffHours = Math.round((Date.now() - dt.getTime()) / 3600000);
    if (diffHours < 1) return "just now";
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.round(diffHours / 24)}d ago`;
}

function formatDue(value) {
    if (!value) return "";
    const dt = new Date(value);
    if (Number.isNaN(dt.getTime())) return String(value);
    const isOverdue = dt < new Date();
    const str = dt.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
    return isOverdue ? `⚠ ${str}` : str;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
}
