/**
 * Knowledge Dashboard Controller.
 */

export async function loadKnowledge() {
    const container = document.getElementById("knowledge-content");
    container.innerHTML = `
        <div style="display:flex;gap:8px;margin-bottom:16px">
            ${Array.from({length: 6}, () => '<div class="skeleton skeleton-line" style="width:80px;height:32px;border-radius:6px"></div>').join('')}
        </div>
        <div class="skeleton-card" style="padding:20px">
            <div class="skeleton skeleton-line skeleton-short" style="margin-bottom:14px"></div>
            ${Array.from({length: 4}, () => `
                <div class="skeleton-row" style="margin-bottom:10px">
                    <div class="skeleton skeleton-line skeleton-full"></div>
                </div>`).join('')}
        </div>
        <div class="skeleton-card" style="padding:20px;margin-top:12px">
            <div class="skeleton skeleton-line skeleton-medium" style="margin-bottom:14px"></div>
            ${Array.from({length: 3}, () => `
                <div class="skeleton-row" style="margin-bottom:10px">
                    <div class="skeleton skeleton-line skeleton-long"></div>
                </div>`).join('')}
        </div>`;

    try {
        const [research, metrics, facts, kb, identity, calendar] = await Promise.all([
            window.apiClient.getKnowledgeResearch(),
            window.apiClient.getKnowledgeMetrics(),
            window.apiClient.getKnowledgeFacts(),
            window.apiClient.getKnowledgeBase().catch(() => ({ items: [], total: 0 })),
            window.apiClient.getIdentity().catch(() => ({ profile: {}, people: [] })),
            window.apiClient.getCalendar().catch(() => ({ tasks: [] })),
        ]);

        renderKnowledgeView(container, research, metrics, facts, {}, kb, identity, calendar);

    } catch (err) {
        console.error("Failed to load knowledge:", err);
        container.innerHTML = `<div class="error">Failed to load knowledge data: ${err.message}</div>`;
    }
}

function renderKnowledgeView(container, research, metrics, facts, _stats, kb, identity, calendar) {
    container.innerHTML = `
        <div class="knowledge-tabs">
            <button class="tab-btn active" onclick="switchKnowledgeTab('identity')">Identity</button>
            <button class="tab-btn" onclick="switchKnowledgeTab('calendar')">Calendar</button>
            <button class="tab-btn" onclick="switchKnowledgeTab('research')">Research</button>
            <button class="tab-btn" onclick="switchKnowledgeTab('metrics')">Metrics</button>
            <button class="tab-btn" onclick="switchKnowledgeTab('facts')">Facts</button>
            <button class="tab-btn" onclick="switchKnowledgeTab('kb')">KB</button>
        </div>

        <div id="tab-identity" class="tab-content active">
            ${renderIdentitySection(identity || { profile: {}, people: [] })}
        </div>

        <div id="tab-calendar" class="tab-content" style="display:none">
            ${renderCalendarSection(calendar)}
        </div>

        <div id="tab-research" class="tab-content" style="display:none">
            ${renderResearchSection(research)}
        </div>

        <div id="tab-metrics" class="tab-content" style="display:none">
            ${renderMetricsSection(metrics)}
        </div>

        <div id="tab-facts" class="tab-content" style="display:none">
            ${renderFactsSection(facts)}
        </div>

        <div id="tab-kb" class="tab-content" style="display:none">
            ${renderKBManagerSection(kb)}
        </div>
    `;

    // Simple window global binding for tab switch
    window.switchKnowledgeTab = (tabName) => {
        document.querySelectorAll(".knowledge-tabs .tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelector(`.knowledge-tabs .tab-btn[onclick="switchKnowledgeTab('${tabName}')"]`).classList.add("active");

        document.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");
        document.getElementById(`tab-${tabName}`).style.display = "block";
    };

    // Bind interactive events
    bindKBEvents();
    bindIdentityEvents();
}

function renderResearchSection(data) {
    if ((!data.active || !data.active.length) && (!data.completed || !data.completed.length)) {
        return '<div class="empty-state">No research projects found.</div>';
    }

    let html = '';

    // Active
    if (data.active && data.active.length > 0) {
        html += '<h3>Active Projects</h3><div class="card-grid">';
        data.active.forEach(p => {
            const progress = p.queries_total ? Math.round((p.queries_done / p.queries_total) * 100) : 0;
            html += `
                <div class="card research-card active">
                    <div class="card-header">
                        <h4>${p.topic}</h4>
                        <span class="badge badge-primary">Active</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:${progress}%"></div>
                    </div>
                    <div class="card-meta">
                        <span>Queries: ${p.queries_done}/${p.queries_total}</span>
                        <span>Findings: ${p.findings_count}</span>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }

    // Completed
    if (data.completed && data.completed.length > 0) {
        html += '<h3>Completed Reports</h3><div class="list-group">';
        data.completed.forEach(p => {
            const date = p.completed_at ? new Date(p.completed_at).toLocaleDateString() : 'Unknown date';
            const preview = p.report_preview || 'No preview available.';
            html += `
                <div class="list-item research-item">
                    <div class="item-main">
                        <div class="item-title">${p.topic}</div>
                        <div class="item-desc">${preview.substring(0, 100)}...</div>
                    </div>
                    <div class="item-meta">
                        <span class="badge badge-outline">Completed</span>
                        <span>${date}</span>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }

    return html;
}

function renderMetricsSection(data) {
    if (!data.data || !data.data.length) {
        return '<div class="empty-state">No metrics recorded. Try "Track project score is 8 points".</div>';
    }

    let html = `
        <div class="table-container">
            <table class="data-table" style="width:100%">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Date</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
    `;

    data.data.forEach(row => {
        const date = row.timestamp ? new Date(row.timestamp).toLocaleString() : '';
        html += `
            <tr>
                <td><strong>${row.metric}</strong></td>
                <td>${row.value} ${row.unit}</td>
                <td>${date}</td>
                <td class="text-muted">${row.notes || ''}</td>
            </tr>
        `;
    });

    html += '</tbody></table></div>';
    return html;
}

function renderFactsSection(data) {
    if (!data.data || !data.data.length) {
        return '<div class="empty-state">No extracted facts found.</div>';
    }

    let html = '<div class="facts-list" style="display:grid;gap:10px">';
    data.data.forEach(f => {
        const s = f.structure || {};
        html += `
            <div class="fact-card" style="padding:10px;border:1px solid var(--border-color);border-radius:6px">
                <div class="fact-structure">
                    <span class="fact-subj" style="font-weight:bold">${s.subject || '?'}</span>
                    <span class="fact-pred" style="color:var(--primary-color)"> ${s.predicate || '->'} </span>
                    <span class="fact-obj" style="font-weight:bold">${s.object || '?'}</span>
                </div>
                <div class="fact-source" style="font-size:0.8em;color:var(--text-muted)">
                    Source: ${f.source || 'Unknown'}
                </div>
            </div>
        `;
    });
    html += '</div>';
    return html;
}



function renderIdentitySection(data) {
    const p = data.profile || {};
    const people = data.people || [];

    let html = `
        <div class="identity-section">
            <h3>Your Profile ${p.verified ? '<span class="badge badge-primary">Verified</span>' : ''}</h3>
            <div class="identity-form" id="profile-form">
                ${profileField("name", "Name", p.name)}
                ${profileField("age", "Date of Birth", p.age)}
                ${profileField("location", "Location", p.location)}
                ${profileField("occupation", "Occupation", p.occupation)}
                ${profileField("languages", "Languages", p.languages)}
                ${profileField("family", "Family", p.family)}
                ${profileField("personal_focus", "Personal Focus", p.personal_focus)}
                ${profileField("interests", "Interests", p.interests)}
                ${profileField("notes", "Notes", p.notes)}
                <div style="display:flex;align-items:center;gap:12px;margin-top:4px">
                    <button class="btn btn-primary" id="btn-save-profile">Save Profile</button>
                    <span id="profile-status" class="kb-status"></span>
                </div>
            </div>
        </div>
    `;

    html += `
        <div class="identity-section" style="margin-top:24px">
            <h3>People (${people.length})</h3>
            <div id="people-list">
                ${people.length > 0
                    ? people.map(person => renderPersonCard(person)).join('')
                    : '<div class="empty-state">No people stored yet. Tell the agent about someone.</div>'}
            </div>
        </div>
    `;

    return html;
}

function profileField(key, label, value) {
    return `
        <div class="settings-field">
            <label class="settings-label">${label}</label>
            <input type="text" class="input settings-input"
                   id="profile-${key}" value="${escapeHtml(value || '')}"
                   placeholder="${label}">
        </div>
    `;
}

function renderPersonCard(person) {
    const trust = person.trust_score || 0;
    const verifiedBadge = person.verified
        ? '<span class="badge badge-primary">Verified</span>'
        : '<span class="badge badge-outline">Unverified</span>';
    return `
        <div class="kb-record person-card" data-person-id="${person.id}">
            <div class="kb-record-header">
                ${verifiedBadge}
                <strong>${escapeHtml(person.full_name)}</strong>
                ${person.role ? `<span class="text-muted">(${escapeHtml(person.role)})</span>` : ''}
                <button class="btn-icon person-edit-btn" title="Edit" style="margin-left:auto;cursor:pointer">&#9998;</button>
            </div>
            <div class="person-details">
                ${person.birth_date ? `Born: ${escapeHtml(person.birth_date)}` : ''}
                ${person.birth_place ? ` in ${escapeHtml(person.birth_place)}` : ''}
            </div>
            <div class="person-edit-form" style="display:none" data-id="${person.id}">
                <input type="text" class="input" data-field="full_name" value="${escapeHtml(person.full_name || '')}" placeholder="Full name">
                <input type="text" class="input" data-field="role" value="${escapeHtml(person.role || '')}" placeholder="Role (brother, friend...)">
                <input type="text" class="input" data-field="birth_date" value="${escapeHtml(person.birth_date || '')}" placeholder="Birth date">
                <input type="text" class="input" data-field="birth_place" value="${escapeHtml(person.birth_place || '')}" placeholder="Birth place">
                <div style="display:flex;gap:8px;margin-top:4px">
                    <button class="btn btn-primary person-save-btn" data-id="${person.id}">Save</button>
                    <button class="btn person-cancel-btn">Cancel</button>
                </div>
            </div>
        </div>
    `;
}

function bindIdentityEvents() {
    // Save profile
    const saveBtn = document.getElementById('btn-save-profile');
    if (saveBtn) {
        saveBtn.addEventListener('click', async () => {
            const fields = {};
            for (const key of ['name','age','location','occupation','languages','family','personal_focus','interests','notes']) {
                const val = document.getElementById(`profile-${key}`)?.value.trim();
                if (val) fields[key] = val;
            }
            const status = document.getElementById('profile-status');
            saveBtn.disabled = true;
            status.textContent = 'Saving...';
            status.className = 'kb-status';
            try {
                await window.apiClient.updateProfile(fields);
                status.textContent = 'Saved!';
                status.className = 'kb-status kb-status-ok';
            } catch (e) {
                status.textContent = `Error: ${e.message}`;
                status.className = 'kb-status kb-status-err';
            } finally {
                saveBtn.disabled = false;
            }
        });
    }

    // Edit/Save/Cancel person (event delegation)
    const peopleList = document.getElementById('people-list');
    if (peopleList) {
        peopleList.addEventListener('click', async (e) => {
            const editBtn = e.target.closest('.person-edit-btn');
            if (editBtn) {
                const card = editBtn.closest('.person-card');
                card.querySelector('.person-edit-form').style.display = 'grid';
                return;
            }
            const cancelBtn = e.target.closest('.person-cancel-btn');
            if (cancelBtn) {
                cancelBtn.closest('.person-edit-form').style.display = 'none';
                return;
            }
            const savePersonBtn = e.target.closest('.person-save-btn');
            if (savePersonBtn) {
                const id = savePersonBtn.dataset.id;
                const form = savePersonBtn.closest('.person-edit-form');
                const fields = {};
                form.querySelectorAll('input[data-field]').forEach(inp => {
                    fields[inp.dataset.field] = inp.value.trim();
                });
                savePersonBtn.disabled = true;
                try {
                    await window.apiClient.updatePerson(id, fields);
                    loadKnowledge(); // Reload to show updated data
                } catch (err) {
                    console.error('Update person failed:', err);
                } finally {
                    savePersonBtn.disabled = false;
                }
            }
        });
    }
}

function renderKBManagerSection(data) {
    const items = data.items || [];
    const total = data.total || 0;

    let recordsHtml = '';
    if (items.length > 0) {
        items.forEach(r => {
            const preview = (r.text || '').substring(0, 120);
            const dna = r.dna || 'general';
            const badgeClass = dna === 'user_core' ? 'kb-badge-anchor' : 'kb-badge-synapse';
            const badgeLabel = dna === 'user_core' ? 'Anchor' : 'General';
            const ts = r.timestamp ? new Date(r.timestamp * 1000).toLocaleDateString() : '';
            recordsHtml += `
                <div class="kb-record">
                    <div class="kb-record-header">
                        <span class="badge ${badgeClass}">${badgeLabel}</span>
                        <span class="kb-record-ts">${ts}</span>
                        <button class="btn-icon kb-delete-btn" data-id="${r.id}" title="Delete">&times;</button>
                    </div>
                    <div class="kb-record-text">${escapeHtml(preview)}${r.text && r.text.length > 120 ? '...' : ''}</div>
                </div>
            `;
        });
    } else {
        recordsHtml = '<div class="empty-state">No records in knowledge base yet.</div>';
    }

    return `
        <div class="kb-input-section">
            <h3>Add Knowledge</h3>
            <textarea id="kb-textarea" class="kb-textarea" placeholder="Paste text here to add to the knowledge base..." rows="4"></textarea>
            <div class="kb-controls">
                <label class="kb-pin-label">
                    <input type="checkbox" id="kb-pin-toggle"> Pin as permanent (anchor)
                </label>
                <button class="btn btn-primary" id="kb-add-btn">Add</button>
            </div>

            <div class="kb-drop-zone" id="kb-drop-zone">
                <div class="kb-drop-zone-inner">
                    <span class="kb-drop-icon">&#128196;</span>
                    <span>Drop files here or click to browse</span>
                    <span class="kb-drop-hint">.txt, .md, .csv — max 10MB</span>
                </div>
                <input type="file" id="kb-file-input" accept=".txt,.md,.csv" style="display:none">
            </div>
            <div id="kb-file-info" class="kb-file-info" style="display:none"></div>
            <div id="kb-status" class="kb-status"></div>
        </div>

        <div class="kb-records-section">
            <div class="kb-records-header">
                <h3>Records <span class="text-muted">(${total})</span></h3>
                <input type="text" id="kb-search" class="input kb-search" placeholder="Search knowledge base...">
            </div>
            <div id="kb-records-list">
                ${recordsHtml}
            </div>
        </div>
    `;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function bindKBEvents() {
    // Add text button
    const addBtn = document.getElementById('kb-add-btn');
    if (addBtn) {
        addBtn.addEventListener('click', async () => {
            const textarea = document.getElementById('kb-textarea');
            const text = textarea.value.trim();
            if (!text) return;
            const pin = document.getElementById('kb-pin-toggle').checked;
            const status = document.getElementById('kb-status');
            addBtn.disabled = true;
            status.textContent = 'Adding...';
            status.className = 'kb-status';
            try {
                const result = await window.apiClient.ingestKnowledge(text, pin);
                status.textContent = `Added ${result.chunks_stored || 1} chunk(s).`;
                status.className = 'kb-status kb-status-ok';
                textarea.value = '';
                reloadKBRecords();
            } catch (e) {
                status.textContent = `Error: ${e.message}`;
                status.className = 'kb-status kb-status-err';
            } finally {
                addBtn.disabled = false;
            }
        });
    }

    // Drop zone
    const dropZone = document.getElementById('kb-drop-zone');
    const fileInput = document.getElementById('kb-file-input');
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('active'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('active');
            if (e.dataTransfer.files.length > 0) handleKBFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) handleKBFile(fileInput.files[0]);
        });
    }

    // Search
    const searchInput = document.getElementById('kb-search');
    if (searchInput) {
        let debounce = null;
        searchInput.addEventListener('input', () => {
            clearTimeout(debounce);
            debounce = setTimeout(() => reloadKBRecords(searchInput.value.trim()), 400);
        });
    }

    // Delete buttons (event delegation)
    const recordsList = document.getElementById('kb-records-list');
    if (recordsList) {
        recordsList.addEventListener('click', async (e) => {
            const btn = e.target.closest('.kb-delete-btn');
            if (!btn) return;
            const id = btn.dataset.id;
            if (!id) return;
            btn.disabled = true;
            try {
                await window.apiClient.deleteKnowledgeItem(id);
                reloadKBRecords(document.getElementById('kb-search')?.value.trim() || '');
            } catch (err) {
                console.error('Delete KB item failed:', err);
            }
        });
    }
}

async function handleKBFile(file) {
    const maxSize = 10 * 1024 * 1024;
    const status = document.getElementById('kb-status');
    const fileInfo = document.getElementById('kb-file-info');

    if (file.size > maxSize) {
        status.textContent = 'File too large (max 10MB).';
        status.className = 'kb-status kb-status-err';
        return;
    }

    const allowed = ['.txt', '.md', '.csv'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
        status.textContent = `Unsupported file type: ${ext}. Use .txt, .md, or .csv.`;
        status.className = 'kb-status kb-status-err';
        return;
    }

    fileInfo.style.display = 'flex';
    fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    status.textContent = 'Uploading...';
    status.className = 'kb-status';

    const pin = document.getElementById('kb-pin-toggle').checked;
    try {
        const result = await window.apiClient.uploadKnowledgeFile(file, pin);
        status.textContent = `Uploaded: ${result.chunks_stored || 0} chunk(s) from ${file.name}.`;
        status.className = 'kb-status kb-status-ok';
        reloadKBRecords();
    } catch (e) {
        status.textContent = `Upload failed: ${e.message}`;
        status.className = 'kb-status kb-status-err';
    } finally {
        fileInfo.style.display = 'none';
        const fileInput = document.getElementById('kb-file-input');
        if (fileInput) fileInput.value = '';
    }
}

async function reloadKBRecords(query = '') {
    const list = document.getElementById('kb-records-list');
    if (!list) return;
    try {
        const data = await window.apiClient.getKnowledgeBase(100, 0, query);
        const items = data.items || [];
        if (items.length === 0) {
            list.innerHTML = '<div class="empty-state">No records found.</div>';
            return;
        }
        let html = '';
        items.forEach(r => {
            const preview = (r.text || '').substring(0, 120);
            const dna = r.dna || 'general';
            const badgeClass = dna === 'user_core' ? 'kb-badge-anchor' : 'kb-badge-synapse';
            const badgeLabel = dna === 'user_core' ? 'Anchor' : 'General';
            const ts = r.timestamp ? new Date(r.timestamp * 1000).toLocaleDateString() : '';
            html += `
                <div class="kb-record">
                    <div class="kb-record-header">
                        <span class="badge ${badgeClass}">${badgeLabel}</span>
                        <span class="kb-record-ts">${ts}</span>
                        <button class="btn-icon kb-delete-btn" data-id="${r.id}" title="Delete">&times;</button>
                    </div>
                    <div class="kb-record-text">${escapeHtml(preview)}${r.text && r.text.length > 120 ? '...' : ''}</div>
                </div>
            `;
        });
        list.innerHTML = html;
    } catch (e) {
        console.error('Reload KB records failed:', e);
    }
}

// ============== Calendar Section ==============

function renderCalendarSection(data) {
    const tasks = (data && data.tasks) || [];
    if (!tasks.length) {
        return '<div class="empty-state">No scheduled tasks, todos, or goals found.</div>';
    }

    // Group by date
    const byDate = {};
    for (const t of tasks) {
        const date = t.date || "No date";
        if (!byDate[date]) byDate[date] = [];
        byDate[date].push(t);
    }

    const sortedDates = Object.keys(byDate).sort();
    let html = '<div class="calendar-list">';
    for (const date of sortedDates) {
        const items = byDate[date];
        html += `<div class="calendar-day">
            <div class="calendar-day-header">${date} <span class="text-muted">(${items.length})</span></div>
            <div class="calendar-day-items">`;
        for (const item of items) {
            const sourceBadge = _sourceBadge(item.source);
            html += `<div class="calendar-item">
                ${sourceBadge}
                <span class="calendar-item-text">${escapeHtml(item.title || item.content || "")}</span>
            </div>`;
        }
        html += '</div></div>';
    }
    html += '</div>';
    return html;
}

function _sourceBadge(source) {
    if (source === "goal") return '<span class="badge badge-goal">Goal</span>';
    if (source === "todo") return '<span class="badge badge-todo">Todo</span>';
    return '<span class="badge badge-scheduled">Scheduled</span>';
}


