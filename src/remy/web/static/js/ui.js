/**
 * UI Utilities — Modal confirmation, skeleton loaders, empty states.
 */

// ── SKELETON LOADERS ──

export function skeletonCards(count = 3) {
    return Array.from({ length: count }, () => `
        <div class="skeleton-card">
            <div class="skeleton-row">
                <div class="skeleton skeleton-avatar"></div>
                <div style="flex:1">
                    <div class="skeleton skeleton-line medium"></div>
                    <div class="skeleton skeleton-line short"></div>
                </div>
            </div>
            <div class="skeleton skeleton-line long"></div>
            <div class="skeleton skeleton-line medium"></div>
        </div>
    `).join("");
}

export function skeletonLines(count = 4) {
    const widths = ["long", "medium", "full", "short", "long", "medium"];
    return Array.from({ length: count }, (_, i) =>
        `<div class="skeleton skeleton-line ${widths[i % widths.length]}"></div>`
    ).join("");
}

export function skeletonGrid(count = 6) {
    return `<div class="system-grid">` +
        Array.from({ length: count }, () =>
            `<div class="skeleton-card"><div class="skeleton skeleton-block"></div></div>`
        ).join("") +
    `</div>`;
}

// ── EMPTY STATES ──

export function emptyState({ icon = "📭", title, hint = "" } = {}) {
    return `
        <div class="empty-state">
            <div class="empty-state-icon">${icon}</div>
            <div class="empty-state-title">${title}</div>
            ${hint ? `<div class="empty-state-hint">${hint}</div>` : ""}
        </div>
    `;
}

export const EMPTY = {
    memory:    () => emptyState({ icon: "🧠", title: "No memories yet",       hint: "Start a conversation — Remy will remember what matters." }),
    tasks:     () => emptyState({ icon: "✅", title: "No tasks here",          hint: "Add a task or let Remy create one autonomously." }),
    goals:     () => emptyState({ icon: "🎯", title: "No active goals",        hint: "Goals are created automatically when Remy works on missions." }),
    history:   () => emptyState({ icon: "🗂️", title: "No sessions yet",       hint: "Conversations appear here after you close a session." }),
    activity:  () => emptyState({ icon: "⚡", title: "No activity yet",        hint: "Remy's autonomous actions will appear here in real time." }),
    search:    () => emptyState({ icon: "🔍", title: "No results found",       hint: "Try different keywords or remove filters." }),
    graph:     () => emptyState({ icon: "🕸️", title: "No connections yet",    hint: "Add more memories to see the knowledge graph." }),
    stats:     () => emptyState({ icon: "📊", title: "No data yet",            hint: "Stats accumulate as Remy handles requests and runs missions." }),
    approvals: () => emptyState({ icon: "✔️", title: "No pending approvals",  hint: "Remy will ask for approval before sensitive actions." }),
    alerts:    () => emptyState({ icon: "🔔", title: "No operator alerts",     hint: "System alerts appear here when something needs attention." }),
};

// ── ERROR STATES ──

export function errorState(message) {
    return `
        <div class="empty-state" style="border-color:var(--red)">
            <div class="empty-state-icon">⚠️</div>
            <div class="empty-state-title" style="color:var(--red)">Failed to load</div>
            <div class="empty-state-hint">${escHtml(message)}</div>
        </div>
    `;
}

// ── ESCAPE HTML ──

export function escHtml(str) {
    return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ── CORRELATION ID ──

let _reqCounter = 0;
export function newRequestId() {
    _reqCounter++;
    return `r${Date.now().toString(36)}-${_reqCounter}`;
}

const modal = document.getElementById("confirm-modal");
const modalTitle = document.getElementById("modal-title");
const modalMessage = document.getElementById("modal-message");
const btnCancel = document.getElementById("btn-modal-cancel");
const btnConfirm = document.getElementById("btn-modal-confirm");

let resolvePromise = null;

function close() {
    modal.classList.remove("active");
    resolvePromise = null;
}

if (btnCancel) {
    btnCancel.addEventListener("click", () => {
        if (resolvePromise) resolvePromise(false);
        close();
    });
}

if (btnConfirm) {
    btnConfirm.addEventListener("click", () => {
        if (resolvePromise) resolvePromise(true);
        close();
    });
}

/**
 * Show a confirmation modal.
 * @param {string} title 
 * @param {string} message 
 * @returns {Promise<boolean>}
 */
export function showConfirm(title, message) {
    return new Promise((resolve) => {
        if (!modal) {
            // Fallback if modal not in DOM
            resolve(confirm(message));
            return;
        }
        
        modalTitle.textContent = title || "Confirm";
        modalMessage.textContent = message || "Are you sure?";
        resolvePromise = resolve;
        modal.classList.add("active");
    });
}
