/**
 * Documents & Reports UI
 * Manages agent-created .md files and PDF reports.
 */

// ── State ──────────────────────────────────────────────────────────────────

let _currentDoc = null;   // { name, content, location }
let _editMode = false;
let _docsCache = [];
let _reportsCache = [];
let _activeTab = 'documents'; // 'documents' | 'reports' | 'import'
let _docsFilter = 'all'; // 'all' | '1' | '3' | '7' (days)
let _bulkFiles = [];   // File[] selected for import

// ── Init ───────────────────────────────────────────────────────────────────

export function initDocuments() {
    _bindTabButtons();
    _bindNewDocButton();
    _bindFilterButtons();
    _initBulkImport();
}

function _bindFilterButtons() {
    document.querySelectorAll('.docs-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            _docsFilter = btn.dataset.docsFilter || 'all';
            document.querySelectorAll('.docs-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (_activeTab === 'documents') _renderDocumentsList();
            else _renderReportsList();
        });
    });
}

export async function loadDocuments() {
    _activeTab = 'documents';
    _showTab('documents');
    await _fetchAndRenderDocuments();
}

export async function loadReports() {
    _activeTab = 'reports';
    _showTab('reports');
    await _fetchAndRenderReports();
}

// ── Tab switching ──────────────────────────────────────────────────────────

function _showTab(tab) {
    const isImport = tab === 'import';
    document.getElementById('docs-tab-documents')?.classList.toggle('active', tab === 'documents');
    document.getElementById('docs-tab-reports')?.classList.toggle('active', tab === 'reports');
    document.getElementById('docs-tab-import')?.classList.toggle('active', isImport);
    document.getElementById('docs-pane-documents')?.classList.toggle('hidden', tab !== 'documents');
    document.getElementById('docs-pane-reports')?.classList.toggle('hidden', tab !== 'reports');
    document.getElementById('docs-pane-import')?.classList.toggle('hidden', !isImport);
    // Hide date filters on import tab — they don't apply
    const filtersBar = document.getElementById('docs-filters-bar');
    if (filtersBar) filtersBar.style.display = isImport ? 'none' : '';
    // Hide "New Document" button on import tab
    const newDocBtn = document.getElementById('btn-new-document');
    if (newDocBtn) newDocBtn.style.display = isImport ? 'none' : '';
}

function _bindTabButtons() {
    document.getElementById('docs-tab-documents')?.addEventListener('click', async () => {
        _activeTab = 'documents';
        _showTab('documents');
        _closeEditor();
        await _fetchAndRenderDocuments();
    });
    document.getElementById('docs-tab-reports')?.addEventListener('click', async () => {
        _activeTab = 'reports';
        _showTab('reports');
        _closeEditor();
        await _fetchAndRenderReports();
    });
    document.getElementById('docs-tab-import')?.addEventListener('click', () => {
        _activeTab = 'import';
        _showTab('import');
        _closeEditor();
    });
}

function _bindNewDocButton() {
    document.getElementById('btn-new-document')?.addEventListener('click', () => {
        _openEditor(null);
    });
    // Close modal on overlay click or Escape
    document.getElementById('docs-editor-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'docs-editor-modal') _closeEditor();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') _closeEditor();
    });
}

// ── Documents list ─────────────────────────────────────────────────────────

async function _fetchAndRenderDocuments() {
    const list = document.getElementById('docs-list');
    if (!list) return;
    list.innerHTML = Array.from({length: 5}, () => `
        <li style="padding:12px 14px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
            <div style="flex:1">
                <div class="skeleton skeleton-line skeleton-medium" style="margin-bottom:6px"></div>
                <div class="skeleton skeleton-line skeleton-short"></div>
            </div>
            <div class="skeleton skeleton-line" style="width:48px;margin-left:12px"></div>
        </li>`).join('');

    try {
        const data = await window.apiClient._fetch('/api/documents').then(r => r.json());
        _docsCache = data.documents || [];
        _renderDocumentsList();
    } catch (e) {
        list.innerHTML = `<li class="docs-error">Error: ${e.message}</li>`;
    }
}

function _filterByDate(items) {
    if (_docsFilter === 'all') return items;
    const days = Number(_docsFilter);
    const cutoff = (Date.now() / 1000) - days * 86400;
    return items.filter(item => (item.modified || 0) >= cutoff);
}

function _renderDocumentsList() {
    const list = document.getElementById('docs-list');
    if (!list) return;

    const filtered = _filterByDate(_docsCache);

    if (filtered.length === 0) {
        list.innerHTML = _docsCache.length === 0
            ? '<li class="docs-empty">No documents yet. Agent will create them automatically.</li>'
            : '<li class="docs-empty">No documents in this time range.</li>';
        return;
    }

    list.innerHTML = filtered.map(doc => {
        const date = new Date(doc.modified * 1000).toLocaleDateString('uk-UA', {
            day: '2-digit', month: '2-digit', year: 'numeric'
        });
        const size = _formatSize(doc.size);
        return `
            <li class="docs-item" data-name="${_esc(doc.name)}">
                <div class="docs-item-info">
                    <span class="docs-item-name">${_esc(doc.name)}</span>
                    <span class="docs-item-meta">${date} · ${size}</span>
                </div>
                <div class="docs-item-actions">
                    <button class="btn-icon" title="View/Edit" data-action="edit" data-name="${_esc(doc.name)}">&#9998;</button>
                    <button class="btn-icon btn-danger" title="Delete" data-action="delete" data-name="${_esc(doc.name)}">&#128465;</button>
                </div>
            </li>`;
    }).join('');

    list.querySelectorAll('[data-action="edit"]').forEach(btn => {
        btn.addEventListener('click', () => _openDocForEdit(btn.dataset.name));
    });
    list.querySelectorAll('[data-action="delete"]').forEach(btn => {
        btn.addEventListener('click', () => _deleteDocument(btn.dataset.name));
    });
}

async function _openDocForEdit(name) {
    try {
        const data = await window.apiClient._fetch(`/api/documents/${encodeURIComponent(name)}`).then(r => r.json());
        _openEditor({ name: data.name, content: data.content });
    } catch (e) {
        alert(`Cannot load document: ${e.message}`);
    }
}

async function _deleteDocument(name) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
        await window.apiClient._fetch(`/api/documents/${encodeURIComponent(name)}`, { method: 'DELETE' });
        await _fetchAndRenderDocuments();
        if (_currentDoc?.name === name) _closeEditor();
    } catch (e) {
        alert(`Delete failed: ${e.message}`);
    }
}

// ── Editor ─────────────────────────────────────────────────────────────────

function _openEditor(doc) {
    _currentDoc = doc;
    _editMode = !doc; // new doc starts in edit mode

    const panel = document.getElementById('docs-editor-modal');
    if (!panel) return;
    panel.classList.remove('hidden');

    const nameInput = document.getElementById('docs-editor-name');
    const contentArea = document.getElementById('docs-editor-content');
    const previewDiv = document.getElementById('docs-editor-preview');
    const btnEdit = document.getElementById('btn-docs-edit');
    const btnSave = document.getElementById('btn-docs-save');
    const btnCancel = document.getElementById('btn-docs-editor-cancel');

    if (nameInput) nameInput.value = doc ? doc.name : '';
    if (nameInput) nameInput.disabled = !!doc; // can't rename existing
    if (contentArea) contentArea.value = doc ? doc.content : '';

    _setEditorMode(_editMode);

    btnEdit?.addEventListener('click', () => {
        _editMode = true;
        _setEditorMode(true);
    }, { once: false });

    if (btnSave) btnSave.onclick = _saveDocument;
    if (btnCancel) btnCancel.onclick = _closeEditor;
}

function _setEditorMode(editing) {
    const contentArea = document.getElementById('docs-editor-content');
    const previewDiv = document.getElementById('docs-editor-preview');
    const btnEdit = document.getElementById('btn-docs-edit');
    const btnSave = document.getElementById('btn-docs-save');

    if (contentArea) contentArea.classList.toggle('hidden', !editing);
    if (previewDiv) {
        previewDiv.classList.toggle('hidden', editing);
        if (!editing && contentArea) {
            previewDiv.innerHTML = _renderMarkdown(contentArea.value);
        }
    }
    if (btnEdit) btnEdit.classList.toggle('hidden', editing);
    if (btnSave) btnSave.classList.toggle('hidden', !editing);
}

async function _saveDocument() {
    const nameInput = document.getElementById('docs-editor-name');
    const contentArea = document.getElementById('docs-editor-content');
    if (!nameInput || !contentArea) return;

    let name = nameInput.value.trim();
    if (!name) { alert('Enter a file name'); return; }
    if (!name.endsWith('.md')) name += '.md';

    const content = contentArea.value;
    try {
        await window.apiClient._fetch(`/api/documents/${encodeURIComponent(name)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
        });
        _currentDoc = { name, content };
        _editMode = false;
        _setEditorMode(false);
        nameInput.disabled = true;
        await _fetchAndRenderDocuments();
    } catch (e) {
        alert(`Save failed: ${e.message}`);
    }
}

function _closeEditor() {
    _currentDoc = null;
    _editMode = false;
    document.getElementById('docs-editor-modal')?.classList.add('hidden');
}

// ── Reports list ───────────────────────────────────────────────────────────

async function _fetchAndRenderReports() {
    const list = document.getElementById('reports-list');
    if (!list) return;
    list.innerHTML = Array.from({length: 4}, () => `
        <li style="padding:12px 14px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
            <div style="flex:1">
                <div class="skeleton skeleton-line skeleton-medium" style="margin-bottom:6px"></div>
                <div class="skeleton skeleton-line skeleton-short"></div>
            </div>
            <div class="skeleton skeleton-line" style="width:48px;margin-left:12px"></div>
        </li>`).join('');

    try {
        const data = await window.apiClient._fetch('/api/reports').then(r => r.json());
        _reportsCache = data.reports || [];
        _renderReportsList();
    } catch (e) {
        list.innerHTML = `<li class="docs-error">Error: ${e.message}</li>`;
    }
}

function _renderReportsList() {
    const list = document.getElementById('reports-list');
    if (!list) return;

    const filtered = _filterByDate(_reportsCache);

    if (filtered.length === 0) {
        list.innerHTML = _reportsCache.length === 0
            ? '<li class="docs-empty">No reports yet.</li>'
            : '<li class="docs-empty">No reports in this time range.</li>';
        return;
    }

    list.innerHTML = filtered.map(r => {
        const date = new Date(r.modified * 1000).toLocaleDateString('uk-UA', {
            day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        const size = _formatSize(r.size);
        const label = r.name.replace(/_/g, ' ').replace(/\.pdf$/, '');
        return `
            <li class="docs-item" data-name="${_esc(r.name)}">
                <div class="docs-item-info">
                    <span class="docs-item-name" title="${_esc(r.name)}">${_esc(label)}</span>
                    <span class="docs-item-meta">${date} · ${size}</span>
                </div>
                <div class="docs-item-actions">
                    <a class="btn-icon" title="Download" href="/api/reports/${encodeURIComponent(r.name)}" target="_blank" download>&#8681;</a>
                    <button class="btn-icon btn-danger" title="Delete" data-action="delete-report" data-name="${_esc(r.name)}">&#128465;</button>
                </div>
            </li>`;
    }).join('');

    list.querySelectorAll('[data-action="delete-report"]').forEach(btn => {
        btn.addEventListener('click', () => _deleteReport(btn.dataset.name));
    });
}

async function _deleteReport(name) {
    if (!confirm(`Delete report "${name}"?`)) return;
    try {
        await window.apiClient._fetch(`/api/reports/${encodeURIComponent(name)}`, { method: 'DELETE' });
        await _fetchAndRenderReports();
    } catch (e) {
        alert(`Delete failed: ${e.message}`);
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function _esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function _formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Very simple markdown → HTML renderer (headings, bold, italic, code, lists, links). */
function _renderMarkdown(md) {
    if (!md) return '';
    let html = md
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        // headings
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // bold / italic
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        // unordered lists
        .replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>')
        // paragraphs
        .replace(/\n\n+/g, '</p><p>')
        .replace(/\n/g, '<br>');
    return `<p>${html}</p>`.replace(/<p><\/p>/g, '');
}

// ── Bulk Import ────────────────────────────────────────────────────────────

function _fmtSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

function _renderBulkFileList() {
    const listEl = document.getElementById('bulk-file-list');
    const optRow = document.getElementById('bulk-options-row');
    if (!listEl) return;

    if (_bulkFiles.length === 0) {
        listEl.classList.add('hidden');
        if (optRow) optRow.style.display = 'none';
        return;
    }

    listEl.classList.remove('hidden');
    if (optRow) optRow.style.display = 'flex';

    listEl.innerHTML = _bulkFiles.map((f, i) => `
        <div class="bulk-file-item">
            <span class="bulk-file-name" title="${f.name}">${f.name}</span>
            <span class="bulk-file-size">${_fmtSize(f.size)}</span>
            <button class="bulk-file-remove" data-idx="${i}" title="Remove">&times;</button>
        </div>
    `).join('');

    listEl.querySelectorAll('.bulk-file-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            _bulkFiles.splice(Number(btn.dataset.idx), 1);
            _renderBulkFileList();
        });
    });
}

function _addFiles(fileList) {
    const allowed = new Set(['.pdf','.docx','.xlsx','.csv','.html','.htm','.xml',
                              '.json','.jsonl','.txt','.md','.markdown','.rst','.zip']);
    for (const f of fileList) {
        const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
        if (!allowed.has(ext)) continue;
        if (!_bulkFiles.some(x => x.name === f.name && x.size === f.size)) {
            _bulkFiles.push(f);
        }
    }
    _renderBulkFileList();
}

function _logLine(text, cls = '') {
    const log = document.getElementById('bulk-log');
    if (!log) return;
    const line = document.createElement('div');
    line.className = 'bulk-log-line' + (cls ? ' ' + cls : '');
    line.textContent = text;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

async function _startBulkImport() {
    if (_bulkFiles.length === 0) return;

    const pin = document.getElementById('bulk-pin-chk')?.checked || false;
    const tags = document.getElementById('bulk-tags-input')?.value || '';
    const progressWrap = document.getElementById('bulk-progress-wrap');
    const progressBar = document.getElementById('bulk-progress-bar');
    const progressLabel = document.getElementById('bulk-progress-label');
    const progressCount = document.getElementById('bulk-progress-count');
    const doneBtn = document.getElementById('bulk-done-btn');
    const startBtn = document.getElementById('bulk-start-btn');
    const log = document.getElementById('bulk-log');

    if (progressWrap) progressWrap.classList.remove('hidden');
    if (log) log.innerHTML = '';
    if (progressBar) progressBar.style.width = '0%';
    if (progressLabel) progressLabel.textContent = 'Importing…';
    if (progressCount) progressCount.textContent = '';
    if (startBtn) startBtn.disabled = true;
    if (doneBtn) doneBtn.classList.add('hidden');

    const formData = new FormData();
    for (const f of _bulkFiles) formData.append('files', f);
    formData.append('pin', pin ? 'true' : 'false');
    formData.append('tags', tags);
    formData.append('source_label', 'bulk-upload');

    try {
        const resp = await fetch('/api/knowledge/bulk-ingest', {
            method: 'POST',
            body: formData,
            headers: window.apiClient?._authHeaders?.() || {},
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            _logLine('Error: ' + (err.detail || resp.statusText), 'error');
            if (startBtn) startBtn.disabled = false;
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let storedTotal = 0;
        let totalFiles = _bulkFiles.length;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                let evt;
                try { evt = JSON.parse(line.slice(6)); } catch { continue; }

                const phase = evt.phase || '';
                const msg = evt.message || '';

                if (phase === 'start') {
                    totalFiles = evt.total_files || totalFiles;
                    _logLine(msg);
                } else if (phase === 'extracting') {
                    if (progressLabel) progressLabel.textContent = 'Extracting…';
                    const pct = totalFiles > 0 ? Math.round((evt.processed || 0) / totalFiles * 50) : 0;
                    if (progressBar) progressBar.style.width = pct + '%';
                    if (msg) _logLine(msg);
                } else if (phase === 'ingesting') {
                    storedTotal = evt.stored || storedTotal;
                    if (progressLabel) progressLabel.textContent = 'Storing in memory…';
                    const pct = totalFiles > 0 ? 50 + Math.round((evt.processed || 0) / totalFiles * 50) : 50;
                    if (progressBar) progressBar.style.width = Math.min(pct, 99) + '%';
                    if (progressCount) progressCount.textContent = storedTotal + ' chunks';
                    if (msg) _logLine(msg);
                } else if (phase === 'done') {
                    if (progressBar) progressBar.style.width = '100%';
                    if (progressLabel) progressLabel.textContent = 'Complete';
                    if (progressCount) progressCount.textContent = (evt.stored || storedTotal) + ' chunks stored';
                    _logLine(msg || 'Done.', 'done');
                    if (doneBtn) doneBtn.classList.remove('hidden');
                } else if (phase === 'error') {
                    _logLine(msg || 'Error', 'error');
                }
            }
        }
    } catch (e) {
        _logLine('Network error: ' + e.message, 'error');
    }

    if (startBtn) startBtn.disabled = false;
}

function _initBulkImport() {
    const dropZone = document.getElementById('bulk-drop-zone');
    const fileInput = document.getElementById('bulk-file-input');
    const pickBtn = document.getElementById('bulk-pick-btn');
    const startBtn = document.getElementById('bulk-start-btn');
    const doneBtn = document.getElementById('bulk-done-btn');

    if (!dropZone) return;

    // Click to pick
    pickBtn?.addEventListener('click', (e) => { e.stopPropagation(); fileInput?.click(); });
    dropZone.addEventListener('click', () => fileInput?.click());
    fileInput?.addEventListener('change', () => { _addFiles(fileInput.files); fileInput.value = ''; });

    // Drag & drop
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        _addFiles(e.dataTransfer.files);
    });

    startBtn?.addEventListener('click', _startBulkImport);

    doneBtn?.addEventListener('click', () => {
        _bulkFiles = [];
        _renderBulkFileList();
        document.getElementById('bulk-progress-wrap')?.classList.add('hidden');
        document.getElementById('bulk-log') && (document.getElementById('bulk-log').innerHTML = '');
        doneBtn.classList.add('hidden');
    });
}
