// Calendar view — monthly grid with tasks/todos/scheduled items

let _currentYear, _currentMonth; // 0-based month
let _allEvents = [];
let _calendarInitialized = false;

const STATUS_COLORS = {
    done: "cal-ev-done",
    completed: "cal-ev-done",
    active: "cal-ev-active",
    pending: "cal-ev-pending",
    in_progress: "cal-ev-inprogress",
};

const SOURCE_ICONS = {
    todo: "✅",
    scheduled: "⏰",
    goal: "🎯",
};

function _pad(n) { return String(n).padStart(2, "0"); }
function _isoDate(y, m, d) { return `${y}-${_pad(m+1)}-${_pad(d)}`; }

async function _loadEvents() {
    try {
        const data = await window.apiClient._fetch("/api/knowledge/calendar").then(r => r.json());
        _allEvents = data.tasks || [];
    } catch (e) {
        _allEvents = [];
    }
}

function _eventsForDate(isoDate) {
    return _allEvents.filter(ev => ev.due_date === isoDate);
}

function _renderGrid(year, month) {
    const grid = document.getElementById("cal-grid");
    if (!grid) return;

    const label = document.getElementById("cal-month-label");
    if (label) {
        const dt = new Date(year, month, 1);
        label.textContent = dt.toLocaleString("default", { month: "long", year: "numeric" });
    }

    const firstDay = new Date(year, month, 1).getDay(); // 0=Sun
    const startOffset = (firstDay + 6) % 7; // Mon-first
    const daysInMonth = new Date(year, month + 1, 0).getDate();

    const today = new Date();
    const todayIso = _isoDate(today.getFullYear(), today.getMonth(), today.getDate());

    let html = '<div class="cal-weekdays">';
    for (const d of ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]) {
        html += `<div class="cal-weekday">${d}</div>`;
    }
    html += "</div><div class='cal-days'>";

    // empty cells before first day
    for (let i = 0; i < startOffset; i++) html += '<div class="cal-cell cal-empty"></div>';

    for (let day = 1; day <= daysInMonth; day++) {
        const iso = _isoDate(year, month, day);
        const evs = _eventsForDate(iso);
        const isToday = iso === todayIso;

        let dots = "";
        const shown = evs.slice(0, 3);
        for (const ev of shown) {
            const cls = STATUS_COLORS[ev.status] || "cal-ev-pending";
            dots += `<span class="cal-dot ${cls}" title="${ev.description || ""}"></span>`;
        }
        if (evs.length > 3) dots += `<span class="cal-dot-more">+${evs.length - 3}</span>`;

        html += `<div class="cal-cell${isToday ? " cal-today" : ""}${evs.length ? " cal-has-events" : ""}" data-date="${iso}">
            <span class="cal-day-num">${day}</span>
            <div class="cal-dots">${dots}</div>
        </div>`;
    }

    html += "</div>";
    grid.innerHTML = html;

    // click to show event list
    grid.querySelectorAll(".cal-cell.cal-has-events").forEach(cell => {
        cell.addEventListener("click", () => _showEventList(cell.dataset.date));
    });
}

function _showEventList(isoDate) {
    const panel = document.getElementById("cal-event-list");
    const dateLabel = document.getElementById("cal-event-list-date");
    const ul = document.getElementById("cal-events-ul");
    if (!panel || !ul) return;

    const evs = _eventsForDate(isoDate);
    const dt = new Date(isoDate + "T00:00:00");
    if (dateLabel) dateLabel.textContent = dt.toLocaleDateString("default", { weekday: "long", day: "numeric", month: "long" });

    ul.innerHTML = evs.map(ev => {
        const icon = SOURCE_ICONS[ev.source] || "•";
        const cls = STATUS_COLORS[ev.status] || "cal-ev-pending";
        const statusBadge = ev.status ? `<span class="cal-ev-badge ${cls}">${ev.status}</span>` : "";
        const desc = ev.description || ev.content || "";
        return `<li class="cal-event-item">
            <span class="cal-ev-icon">${icon}</span>
            <span class="cal-ev-desc">${escapeHtml(desc.substring(0, 120))}${desc.length > 120 ? "…" : ""}</span>
            ${statusBadge}
        </li>`;
    }).join("");

    panel.classList.remove("hidden");
}

function escapeHtml(s) {
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

async function _refresh() {
    _ensureCalendarInitialized();
    await _loadEvents();
    _renderGrid(_currentYear, _currentMonth);
}

function _initCalendar() {
    if (_calendarInitialized) return;
    _calendarInitialized = true;
    const now = new Date();
    _currentYear = now.getFullYear();
    _currentMonth = now.getMonth();

    document.getElementById("cal-prev")?.addEventListener("click", () => {
        _currentMonth--;
        if (_currentMonth < 0) { _currentMonth = 11; _currentYear--; }
        _renderGrid(_currentYear, _currentMonth);
        document.getElementById("cal-event-list")?.classList.add("hidden");
    });

    document.getElementById("cal-next")?.addEventListener("click", () => {
        _currentMonth++;
        if (_currentMonth > 11) { _currentMonth = 0; _currentYear++; }
        _renderGrid(_currentYear, _currentMonth);
        document.getElementById("cal-event-list")?.classList.add("hidden");
    });

    document.getElementById("cal-today")?.addEventListener("click", () => {
        const now = new Date();
        _currentYear = now.getFullYear();
        _currentMonth = now.getMonth();
        _renderGrid(_currentYear, _currentMonth);
        document.getElementById("cal-event-list")?.classList.add("hidden");
    });

    document.getElementById("cal-event-list-close")?.addEventListener("click", () => {
        document.getElementById("cal-event-list")?.classList.add("hidden");
    });

    _refresh();
}

function _ensureCalendarInitialized() {
    if (!_calendarInitialized) {
        _initCalendar();
    }
}

window.loadCalendar = () => _refresh();

if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => {
        _ensureCalendarInitialized();
    });
} else {
    _ensureCalendarInitialized();
}
