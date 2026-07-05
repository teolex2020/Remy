/**
 * Memory View — list, search, view details, edit, delete, stats dashboard.
 */

import { skeletonCards, EMPTY, errorState } from "./ui.js";

const listEl = document.getElementById("memory-list");
const statsBar = document.getElementById("memory-stats-bar");
const searchInput = document.getElementById("memory-search");
const periodSelect = document.getElementById("memory-period");
const searchBtn = document.getElementById("btn-memory-search");
const addBtn = document.getElementById("btn-add-record");
const exportBtn = document.getElementById("btn-export");
const importBtn = document.getElementById("btn-import-kb");

import { showConfirm } from "./ui.js";

let currentOffset = 0;
const LIMIT = 50;

// ============== STATS DASHBOARD ==============

export async function loadMemoryStats() {
    try {
        const statsData = await window.apiClient._fetch("/api/stats").then(r => r.json());
        const total = statsData.total_records || 0;
        statsBar.innerHTML = `<div class="mem-stat"><span class="mem-stat-value">${total}</span> records</div>`;
    } catch (e) {
        console.error("Failed to load memory stats:", e);
        statsBar.innerHTML = "";
    }
}

// ============== CONSOLIDATE & EXPORT ==============

async function doExport() {
    try {
        const res = await window.apiClient._fetch("/api/export");
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "remy-brain-export.json";
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) {
        alert("Export failed: " + e.message);
    }
}

// ============== RECORDS LIST ==============

export async function loadRecords(append = false) {
    if (!append) {
        currentOffset = 0;
        listEl.innerHTML = skeletonCards(5);
    }

    const period = periodSelect.value;
    const data = await window.apiClient.getRecords(null, "all", period, currentOffset, LIMIT);

    renderList(data.records, data.total, append, { mode: "browse" });

    currentOffset += data.records.length;

    // Manage Load More button
    const existingBtn = document.getElementById("btn-load-more");
    if (existingBtn) existingBtn.remove();

    if (currentOffset < data.total) {
        const btn = document.createElement("button");
        btn.id = "btn-load-more";
        btn.className = "btn btn-outline";
        btn.style.width = "100%";
        btn.style.marginTop = "20px";
        btn.textContent = `Load More (${data.total - currentOffset} remaining)`;
        btn.onclick = () => loadRecords(true);
        listEl.appendChild(btn);
    }
}

async function doSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        await loadRecords();
        return;
    }
    currentOffset = 0;
    const period = periodSelect.value;
    const data = await window.apiClient.searchRecords(query, null, "all", period, "hybrid");
    renderList(data.results || [], (data.results || []).length, false, { mode: "hybrid" });
}


document.addEventListener("open-memory-search", (event) => {
    const detail = event?.detail || {};
    const query = String(detail.query || "").trim();
    if (searchInput) searchInput.value = query;
    if (periodSelect) periodSelect.value = "all";
    if (query) {
        doSearch().catch((e) => console.error("Failed to open memory search", e));
    } else {
        loadRecords(false).catch((e) => console.error("Failed to load memory records", e));
    }
});

document.addEventListener("open-memory-restored-set", async (event) => {
    const detail = event?.detail || {};
    const labels = Array.isArray(detail.labels)
        ? detail.labels.map((item) => String(item || "").trim()).filter(Boolean)
        : [];
    if (searchInput) searchInput.value = labels.join(" | ");
    if (periodSelect) periodSelect.value = "all";
    if (!labels.length) {
        loadRecords(false).catch((e) => console.error("Failed to load memory records", e));
        return;
    }
    try {
        const resultSets = await Promise.all(
            labels.map((label) => window.apiClient.searchRecords(label, null, "all", "all", "hybrid")),
        );
        const merged = [];
        const seen = new Set();
        for (const data of resultSets) {
            for (const record of (data?.results || [])) {
                if (!record?.id || seen.has(record.id)) continue;
                seen.add(record.id);
                merged.push(record);
            }
        }
        renderList(merged, merged.length, false, { mode: "browse" });
    } catch (e) {
        console.error("Failed to open restored memory set", e);
    }
});

const INTERNAL_TAGS = new Set([
    "decision_dossier", "pinned_snapshot", "reconstruction_review", "incident_snapshot",
    "contact", "financial",
]);

function recordDateKey(r) {
    const ts = r.created_at;
    if (ts) {
        const d = new Date(ts * 1000);
        return d.toISOString().slice(0, 10); // "2026-04-13"
    }
    // fallback: parse from content like "[2026-04-13 ..."
    const m = (r.content || "").match(/^\[(\d{4}-\d{2}-\d{2})/);
    return m ? m[1] : "Unknown";
}

function formatDateGroupLabel(dateKey) {
    if (dateKey === "Unknown") return "No date";
    try {
        const d = new Date(`${dateKey}T00:00:00`);
        const now = new Date();
        const today = now.toISOString().slice(0, 10);
        const yesterday = new Date(now - 86400000).toISOString().slice(0, 10);
        if (dateKey === today) return "Today";
        if (dateKey === yesterday) return "Yesterday";
        return d.toLocaleDateString("uk-UA", { day: "numeric", month: "long", year: "numeric" });
    } catch {
        return dateKey;
    }
}

function renderCard(r) {
    const content = escapeHtml(r.content || "");
    const isTruncated = (r.content || "").length > 180;
    const tags = (Array.isArray(r.tags) ? r.tags : []).filter(t => !INTERNAL_TAGS.has(t));
    return `
    <div class="memory-card" data-id="${r.id}">
        <div class="memory-card-content${isTruncated ? " truncated" : ""}">${content}</div>
        ${tags.length ? `<div class="memory-card-meta">${tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>` : ""}
        <button class="btn-card-delete" data-id="${r.id}" title="Delete">&#x2715;</button>
    </div>`;
}

function renderList(records, total, append, options = {}) {
    if ((!records || records.length === 0) && !append) {
        listEl.innerHTML = EMPTY.memory();
        return;
    }

    // Group by date descending
    const groups = new Map();
    for (const r of records) {
        const key = recordDateKey(r);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(r);
    }
    // Sort groups newest first
    const sortedGroups = [...groups.entries()].sort((a, b) => {
        if (a[0] === "Unknown") return 1;
        if (b[0] === "Unknown") return -1;
        return b[0].localeCompare(a[0]);
    });

    const html = sortedGroups.map(([dateKey, items], groupIdx) => {
        const label = formatDateGroupLabel(dateKey);
        const isFirst = groupIdx === 0;
        const cardsHtml = items.map(renderCard).join("");
        return `
        <div class="mem-date-group" data-date="${dateKey}">
            <button class="mem-date-header${isFirst ? " open" : ""}" aria-expanded="${isFirst}">
                <span class="mem-date-label">${label}</span>
                <span class="mem-date-count">${items.length}</span>
                <span class="mem-date-chevron">›</span>
            </button>
            <div class="mem-date-body${isFirst ? "" : " hidden"}">
                ${cardsHtml}
            </div>
        </div>`;
    }).join("");

    if (append) {
        const temp = document.createElement("div");
        temp.innerHTML = html;
        while (temp.firstChild) listEl.appendChild(temp.firstChild);
    } else {
        listEl.innerHTML = html;
    }

    _bindGroupListeners(append ? null : listEl);
}

function _bindGroupListeners(root) {
    const container = root || listEl;

    // Accordion toggle
    container.querySelectorAll(".mem-date-header").forEach(btn => {
        btn.addEventListener("click", () => {
            const body = btn.nextElementSibling;
            const open = btn.classList.toggle("open");
            btn.setAttribute("aria-expanded", open);
            body.classList.toggle("hidden", !open);
        });
    });

    // Card click → detail panel
    container.querySelectorAll(".memory-card").forEach(card => {
        card.addEventListener("click", (e) => {
            if (e.target.closest(".btn-card-delete")) return;
            const contentEl = card.querySelector(".memory-card-content");
            if (contentEl && (contentEl.classList.contains("truncated") || contentEl.classList.contains("expanded"))) {
                contentEl.classList.toggle("expanded");
                contentEl.classList.toggle("truncated");
                return;
            }
            openRecordDetail(card.dataset.id);
        });
    });

    // Delete button
    container.querySelectorAll(".btn-card-delete").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const confirmed = await showConfirm("Delete Record", "Permanently delete this memory?");
            if (!confirmed) return;
            try {
                await window.apiClient.deleteRecord(btn.dataset.id);
                const card = btn.closest(".memory-card");
                const body = card.closest(".mem-date-body");
                card.remove();
                // remove group if empty
                if (body && !body.querySelector(".memory-card")) {
                    body.closest(".mem-date-group")?.remove();
                }
                await loadMemoryStats();
            } catch (err) {
                alert("Delete failed: " + err.message);
            }
        });
    });
}

export async function openRecordDetail(id) {
    try {
        const record = await window.apiClient.getRecord(id);
        showDetailPanel(record);
    } catch (e) {
        console.error("Failed to load record:", e);
    }
}

function showDetailPanel(record) {
    const app = document.querySelector(".app");
    const panelContent = document.getElementById("panel-content");
    const panelTitle = document.getElementById("panel-title");

    panelTitle.textContent = "Record Details";
    app.classList.add("panel-open");

    let connectionsHtml = "";
    if (record.connections && record.connections.length > 0) {
        connectionsHtml = `
            <div class="panel-connections">
                <div class="panel-field-label">Connections</div>
                ${record.connections.map((c) => `
                    <div class="panel-connection-item" data-id="${c.id}">
                        <strong>${escapeHtml(c.content)}</strong>
                        <br><span style="color:var(--text-muted)">weight: ${c.weight} | tags: ${(c.tags || []).join(", ")}</span>
                    </div>
                `).join("")}
            </div>
        `;
    }

    // Metadata section
    const meta = record.metadata || {};
    let metaHtml = "";
    const metaKeys = Object.keys(meta).filter(k => meta[k] != null && meta[k] !== "");
    if (metaKeys.length > 0) {
        metaHtml = `
            <div class="panel-field">
                <div class="panel-field-label">Metadata</div>
                <div class="panel-field-value" style="font-size:12px;font-family:monospace">
                    ${metaKeys.map(k => `<div>${escapeHtml(k)}: ${escapeHtml(String(meta[k]))}</div>`).join("")}
                </div>
            </div>
        `;
    }

    const lvlCls = levelClass(record.level);
    const strength = record.strength || 0;

    panelContent.innerHTML = `
        <div class="panel-field">
            <div class="panel-field-label">ID</div>
            <div class="panel-field-value" style="font-family:monospace;font-size:11px;word-break:break-all">${record.id}</div>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Content</div>
            <div class="panel-field-value">${escapeHtml(record.content)}</div>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Level</div>
            <div class="panel-field-value"><span class="memory-card-level ${lvlCls}">${record.level}</span></div>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Strength</div>
            <div class="panel-field-value" style="display:flex;align-items:center;gap:8px">
                <div class="strength-bar" style="width:100px">
                    <div class="strength-fill" style="width:${strength * 100}%;background:${strengthColor(strength)}"></div>
                </div>
                ${(strength * 100).toFixed(1)}%
            </div>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Activations</div>
            <div class="panel-field-value">${record.activation_count}</div>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Tags</div>
            <div class="panel-field-value">${(record.tags || []).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join(" ")}</div>
        </div>
        ${metaHtml}
        ${connectionsHtml}
        <div class="panel-actions">
            <button class="btn btn-outline" id="btn-edit-record">Edit</button>
            <button class="btn btn-danger" id="btn-delete-record">Delete</button>
        </div>
    `;

    // Connection click -> load that record
    panelContent.querySelectorAll(".panel-connection-item").forEach((item) => {
        item.addEventListener("click", () => openRecordDetail(item.dataset.id));
    });

    // Edit button
    const editBtn = document.getElementById("btn-edit-record");
    if (editBtn) {
        editBtn.addEventListener("click", () => enterEditMode(record));
    }

    // Delete button
    const deleteBtn = document.getElementById("btn-delete-record");
    if (deleteBtn) {
        deleteBtn.addEventListener("click", async () => {
            const confirmed = await showConfirm(
                "Delete Record",
                "Are you sure you want to permanently delete this memory?"
            );
            if (!confirmed) return;

            try {
                await window.apiClient.deleteRecord(record.id);
                app.classList.remove("panel-open");
                await loadRecords();
                await loadMemoryStats();
            } catch (e) {
                alert("Delete failed: " + e.message);
            }
        });
    }
}

function enterEditMode(record) {
    const panelContent = document.getElementById("panel-content");
    const panelTitle = document.getElementById("panel-title");
    panelTitle.textContent = "Edit Record";

    const tagsStr = (record.tags || []).join(", ");
    const currentLevel = (record.level || "").replace("Level.", "").toUpperCase();

    const levels = ["WORKING", "DECISIONS", "DOMAIN", "IDENTITY"];
    const levelOptions = levels.map((l) =>
        `<option value="${l.toLowerCase()}" ${l === currentLevel ? "selected" : ""}>${l}</option>`
    ).join("");

    panelContent.innerHTML = `
        <div class="panel-field">
            <div class="panel-field-label">ID</div>
            <div class="panel-field-value" style="font-family:monospace;font-size:11px;word-break:break-all">${record.id}</div>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Content</div>
            <textarea id="edit-content" class="panel-edit-textarea">${escapeHtml(record.content || "")}</textarea>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Tags (comma-separated)</div>
            <input id="edit-tags" class="panel-edit-input" type="text" value="${escapeHtml(tagsStr)}">
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Level</div>
            <select id="edit-level" class="panel-edit-select">${levelOptions}</select>
        </div>
        <div class="panel-actions">
            <button class="btn btn-save" id="btn-save-record">Save</button>
            <button class="btn btn-outline" id="btn-cancel-edit">Cancel</button>
        </div>
    `;

    document.getElementById("btn-save-record").addEventListener("click", () => saveRecord(record));
    document.getElementById("btn-cancel-edit").addEventListener("click", () => showDetailPanel(record));
}

async function saveRecord(originalRecord) {
    const content = document.getElementById("edit-content").value.trim();
    const tags = document.getElementById("edit-tags").value.trim();
    const level = document.getElementById("edit-level").value;

    const payload = {};
    if (content && content !== originalRecord.content) payload.content = content;
    if (tags !== (originalRecord.tags || []).join(", ")) payload.tags = tags;

    const originalLevel = (originalRecord.level || "").replace("Level.", "").toLowerCase();
    if (level !== originalLevel) payload.level = level;

    if (Object.keys(payload).length === 0) {
        showDetailPanel(originalRecord);
        return;
    }

    const saveBtn = document.getElementById("btn-save-record");
    saveBtn.textContent = "Saving...";
    saveBtn.disabled = true;

    try {
        await window.apiClient.updateRecord(originalRecord.id, payload);
        const updated = await window.apiClient.getRecord(originalRecord.id);
        showDetailPanel(updated);
        await loadRecords();
        await loadMemoryStats();
    } catch (e) {
        alert("Save failed: " + e.message);
        saveBtn.textContent = "Save";
        saveBtn.disabled = false;
    }
}

function openCreateForm() {
    const app = document.querySelector(".app");
    const panelContent = document.getElementById("panel-content");
    const panelTitle = document.getElementById("panel-title");

    panelTitle.textContent = "New Record";
    app.classList.add("panel-open");

    const levels = ["WORKING", "DECISIONS", "DOMAIN", "IDENTITY"];
    const levelOptions = levels.map((l) =>
        `<option value="${l.toLowerCase()}" ${l === "DOMAIN" ? "selected" : ""}>${l}</option>`
    ).join("");

    panelContent.innerHTML = `
        <div class="panel-field">
            <div class="panel-field-label">Content</div>
            <textarea id="create-content" class="panel-edit-textarea" placeholder="What do you want to remember?"></textarea>
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Tags (comma-separated)</div>
            <input id="create-tags" class="panel-edit-input" type="text" placeholder="e.g. person, family, project">
        </div>
        <div class="panel-field">
            <div class="panel-field-label">Level</div>
            <select id="create-level" class="panel-edit-select">${levelOptions}</select>
        </div>
        <div class="panel-actions">
            <button class="btn btn-save" id="btn-create-save">Create</button>
            <button class="btn btn-outline" id="btn-create-cancel">Cancel</button>
        </div>
    `;

    document.getElementById("btn-create-save").addEventListener("click", createRecord);
    document.getElementById("btn-create-cancel").addEventListener("click", () => {
        app.classList.remove("panel-open");
    });

    document.getElementById("create-content").focus();
}

async function createRecord() {
    const content = document.getElementById("create-content").value.trim();
    const tags = document.getElementById("create-tags").value.trim();
    const level = document.getElementById("create-level").value;

    if (!content) {
        alert("Content is required.");
        return;
    }

    const saveBtn = document.getElementById("btn-create-save");
    saveBtn.textContent = "Creating...";
    saveBtn.disabled = true;

    const payload = { content };
    if (tags) payload.tags = tags;
    if (level) payload.level = level;

    try {
        await window.apiClient.createRecord(payload);
        const app = document.querySelector(".app");
        app.classList.remove("panel-open");
        await loadRecords();
        await loadMemoryStats();
    } catch (e) {
        alert("Create failed: " + e.message);
        saveBtn.textContent = "Create";
        saveBtn.disabled = false;
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ============== IMPORT KB ==============

function openImportPanel() {
    const app = document.querySelector(".app");
    const panelContent = document.getElementById("panel-content");
    const panelTitle = document.getElementById("panel-title");

    panelTitle.textContent = "Import Knowledge";
    app.classList.add("panel-open");

    panelContent.innerHTML = `
        <div class="panel-field">
            <div class="panel-field-label">Paste Text</div>
            <textarea id="import-text" class="panel-edit-textarea" rows="8"
                      placeholder="Paste text to import into memory. Long text will be split into paragraphs."></textarea>
        </div>
        <div class="panel-field">
            <label style="display:flex;align-items:center;gap:6px;font-size:13px">
                <input type="checkbox" id="import-pin"> Pin to IDENTITY level
            </label>
        </div>
        <div class="panel-actions">
            <button class="btn btn-primary" id="btn-import-text">Import Text</button>
        </div>
        <div style="border-top:1px solid var(--border);margin:16px 0;padding-top:16px">
            <div class="panel-field-label">Or Upload File</div>
            <p style="font-size:12px;color:var(--text-muted);margin:4px 0 8px">Supported: .txt, .md, .csv (max 10 MB)</p>
            <input type="file" id="import-file" accept=".txt,.md,.csv" style="font-size:13px">
            <div class="panel-actions" style="margin-top:8px">
                <button class="btn btn-primary" id="btn-import-file" disabled>Upload File</button>
            </div>
        </div>
        <div id="import-status" class="kb-status" style="margin-top:8px"></div>
    `;

    const fileInput = document.getElementById("import-file");
    const uploadBtn = document.getElementById("btn-import-file");
    fileInput.addEventListener("change", () => {
        uploadBtn.disabled = !fileInput.files.length;
    });

    document.getElementById("btn-import-text").addEventListener("click", doImportText);
    uploadBtn.addEventListener("click", doImportFile);
}

async function doImportText() {
    const text = document.getElementById("import-text").value.trim();
    if (!text) { alert("Enter some text to import."); return; }
    const pin = document.getElementById("import-pin").checked;
    const status = document.getElementById("import-status");
    const btn = document.getElementById("btn-import-text");

    btn.disabled = true;
    btn.textContent = "Importing...";
    status.textContent = "";

    try {
        const res = await window.apiClient.ingestKnowledge(text, pin);
        status.textContent = `Imported ${res.ingested} record(s). Total: ${res.total}`;
        status.className = "kb-status kb-status-ok";
        document.getElementById("import-text").value = "";
        await loadRecords();
        await loadMemoryStats();
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
        status.className = "kb-status kb-status-err";
    } finally {
        btn.disabled = false;
        btn.textContent = "Import Text";
    }
}

async function doImportFile() {
    const fileInput = document.getElementById("import-file");
    const file = fileInput.files[0];
    if (!file) return;
    const pin = document.getElementById("import-pin").checked;
    const status = document.getElementById("import-status");
    const btn = document.getElementById("btn-import-file");

    btn.disabled = true;
    btn.textContent = "Uploading...";
    status.textContent = "";

    try {
        const res = await window.apiClient.uploadKnowledgeFile(file, pin);
        status.textContent = `Uploaded "${res.filename}": ${res.chunks} chunk(s). Total: ${res.total}`;
        status.className = "kb-status kb-status-ok";
        fileInput.value = "";
        btn.disabled = true;
        await loadRecords();
        await loadMemoryStats();
    } catch (e) {
        status.textContent = `Error: ${e.message}`;
        status.className = "kb-status kb-status-err";
        btn.disabled = false;
    } finally {
        btn.textContent = "Upload File";
    }
}

// Event listeners
searchBtn.addEventListener("click", doSearch);
addBtn.addEventListener("click", openCreateForm);
exportBtn.addEventListener("click", doExport);
if (importBtn) importBtn.addEventListener("click", openImportPanel);
searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
});
periodSelect.addEventListener("change", () => loadRecords(false));
