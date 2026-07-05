/**
 * Profile View - personal context editor + people list.
 */

const container = document.getElementById("profile-content");

export async function loadProfile() {
    container.innerHTML = `
        <div class="profile-shell">
            <div class="skeleton-card" style="padding:20px;display:flex;gap:14px;align-items:center;margin-bottom:12px">
                <div class="skeleton skeleton-avatar" style="width:48px;height:48px;border-radius:50%;flex-shrink:0"></div>
                <div style="flex:1">
                    <div class="skeleton skeleton-line skeleton-medium" style="margin-bottom:8px"></div>
                    <div class="skeleton skeleton-line skeleton-long"></div>
                </div>
            </div>
            <div class="skeleton-card" style="padding:20px">
                <div class="skeleton skeleton-block" style="height:280px;border-radius:6px"></div>
            </div>
        </div>`;

    try {
        const identity = await window.apiClient.getIdentity().catch(() => ({ profile: {}, people: [] }));
        renderProfileView(identity);
    } catch (err) {
        console.error("Failed to load profile:", err);
        container.innerHTML = `<div class="empty-state">Failed to load profile: ${escapeHtml(String(err.message || err))}</div>`;
    }
}

function renderProfileView(identity) {
    const profile = (identity || {}).profile || {};
    const people = Array.isArray(identity?.people) ? identity.people : [];
    const notesText = buildContextText(identity);
    const name = profile.name || "";
    const initial = (name || "?").charAt(0).toUpperCase();

    container.innerHTML = `
        <div class="profile-shell">
            <div class="profile-header-card">
                <div class="profile-avatar">${escapeHtml(initial)}</div>
                <div class="profile-header-info">
                    <div class="profile-name">${escapeHtml(name || "Profile")}</div>
                    ${profile.location ? `<div class="profile-location">${escapeHtml(profile.location)}</div>` : ""}
                </div>
            </div>

            <div class="profile-section">
                <div class="profile-section-label">Personal context</div>
                <p class="profile-section-hint">This is what the agent currently knows about you. Edit directly — changes apply immediately to future responses.</p>
                <textarea
                    id="profile-notes-editor"
                    class="input profile-textarea"
                    placeholder="Write about yourself: name, age, city, language, habits, important facts — anything the agent should consider."
                >${escapeHtml(notesText)}</textarea>
                <div class="profile-actions">
                    <button class="btn btn-primary" id="btn-save-profile-notes">Save</button>
                    <button class="btn btn-outline" id="btn-reset-profile-notes">Reset</button>
                    <span id="profile-save-status" class="profile-save-status"></span>
                </div>
            </div>

            ${people.length ? renderPeopleSection(people) : ""}
        </div>`;

    autoResizeTextarea(document.getElementById("profile-notes-editor"));
    bindEvents(notesText);
}

function renderPeopleSection(people) {
    const cards = people.map(p => {
        const role = p.role ? `<span class="profile-person-role">${escapeHtml(p.role)}</span>` : "";
        const birth = p.birth_date ? `<span class="profile-person-meta">${escapeHtml(p.birth_date)}</span>` : "";
        return `
            <div class="profile-person-card">
                <div class="profile-person-avatar">${escapeHtml((p.full_name || "?").charAt(0).toUpperCase())}</div>
                <div class="profile-person-info">
                    <div class="profile-person-name">${escapeHtml(p.full_name || "—")}</div>
                    <div class="profile-person-details">${[role, birth].filter(Boolean).join(" · ")}</div>
                </div>
            </div>`;
    }).join("");

    return `
        <div class="profile-section">
            <div class="profile-section-label">Important people</div>
            <div class="profile-people-grid">${cards}</div>
        </div>`;
}

function bindEvents(initialText) {
    const textarea = document.getElementById("profile-notes-editor");
    const saveBtn = document.getElementById("btn-save-profile-notes");
    const resetBtn = document.getElementById("btn-reset-profile-notes");
    const status = document.getElementById("profile-save-status");

    resetBtn?.addEventListener("click", () => {
        if (textarea) textarea.value = initialText;
        if (status) status.textContent = "";
        autoResizeTextarea(textarea);
    });

    saveBtn?.addEventListener("click", async () => {
        if (!textarea) return;
        saveBtn.disabled = true;
        if (status) { status.textContent = "Saving..."; status.className = "profile-save-status"; }
        try {
            await window.apiClient.updateProfile({ notes: textarea.value.trim() });
            if (status) { status.textContent = "Saved ✓"; status.className = "profile-save-status profile-save-ok"; }
            setTimeout(() => { if (status) status.textContent = ""; }, 3000);
        } catch (err) {
            console.error("Failed to save profile:", err);
            if (status) { status.textContent = `Error: ${err.message || err}`; status.className = "profile-save-status profile-save-err"; }
        } finally {
            saveBtn.disabled = false;
        }
    });

    // Ctrl+S / Cmd+S to save
    textarea?.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "s") {
            e.preventDefault();
            saveBtn?.click();
        }
    });
}

function autoResizeTextarea(el) {
    if (!el) return;
    const resize = () => {
        el.style.height = "auto";
        el.style.height = Math.max(200, el.scrollHeight) + "px";
    };
    resize();
    el.addEventListener("input", resize);
}

// Builds a clean plain-text summary to put in the textarea
function buildContextText(identity) {
    const profile = identity?.profile || {};
    const people = Array.isArray(identity?.people) ? identity.people : [];
    const lines = [];

    // Core fields
    const basic = [
        profile.name ? `Мене звати ${profile.name}.` : "",
        profile.age ? `Мені ${profile.age} роки.` : "",
        profile.location ? `Живу в ${profile.location}.` : "",
        profile.languages ? `Мова: ${profile.languages}.` : "",
        profile.occupation ? `Заняття: ${profile.occupation}.` : "",
        profile.email ? `Email: ${profile.email}.` : "",
        profile.phone ? `Телефон: ${profile.phone}.` : "",
    ].filter(Boolean);
    if (basic.length) lines.push(basic.join(" "));

    if (profile.family) lines.push(`Родина: ${profile.family}.`);

    // Notes — deduplicated, trimmed
    const notes = simplifyNotes(profile.notes);
    if (notes.length) lines.push(`Додатково:\n${notes.map(n => `- ${n}`).join("\n")}`);

    // People block
    if (people.length) {
        const peopleLines = people.map(p => {
            const bits = [];
            if (p.birth_date) bits.push(`нар. ${p.birth_date}`);
            if (p.birth_place) bits.push(p.birth_place);
            if (p.role) bits.push(p.role);
            return `- ${p.full_name || "—"}${bits.length ? ` (${bits.join(", ")})` : ""}`;
        });
        lines.push(`Важливі люди:\n${peopleLines.join("\n")}`);
    }

    if (!lines.length) {
        return "";
    }
    return lines.join("\n\n");
}

function simplifyNotes(rawNotes) {
    const text = String(rawNotes || "").trim();
    if (!text) return [];
    return text
        .split(";")
        .map(s => s.trim())
        .filter(Boolean)
        .filter(s => !/root project origins/i.test(s))
        .filter(s => !/^email\s*:/i.test(s))
        .filter(s => !/^(телефон|номер телефону)\s*:/i.test(s))
        .reduce((acc, item) => {
            const key = item.toLowerCase().replace(/[\W_]+/g, "");
            if (key && !acc.seen.has(key)) { acc.seen.add(key); acc.list.push(item.endsWith(".") ? item : `${item}.`); }
            return acc;
        }, { seen: new Set(), list: [] }).list.slice(0, 6);
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
}
