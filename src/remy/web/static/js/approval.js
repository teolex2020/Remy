/**
 * Approval Panel — real-time human-in-the-loop approval notifications.
 *
 * Connects to the shared /ws/human-loop WebSocket.
 * When the autonomous agent triggers a financial/registration action,
 * a notification card slides in from the bottom-right corner.
 * The user clicks "Approve" or "Reject" — the agent resumes or stops.
 */

import { eventField, eventName } from "./runtime-events.js";

const area = document.getElementById("approval-notification-area");

/** action_id → { card: HTMLElement, interval: number } */
const _cards = {};

// ============== Card rendering ==============

function _renderCard(event) {
    const actionId = eventField(event, "action_id", "");
    const description = eventField(event, "description", "");
    const timeoutSec = Number(eventField(event, "timeout_sec", 0));
    const createdAt = Number(eventField(event, "created_at", Date.now() / 1000));

    // Guard: don't show duplicates
    if (!actionId || _cards[actionId]) return;

    const card = document.createElement("div");
    card.className = "approval-card";
    card.dataset.actionId = actionId;

    const expiresAt = (createdAt + timeoutSec) * 1000;
    const shortId = actionId.slice(0, 8);

    card.innerHTML = `
        <div class="approval-card-header">
            <span class="approval-icon">&#9888;</span>
            <span class="approval-title">Approval required</span>
            <span class="approval-timer" id="approval-timer-${shortId}"></span>
        </div>
        <div class="approval-card-body">${_escapeHtml(description)}</div>
        <div class="approval-card-actions">
            <button class="btn btn-primary btn-approve" data-id="${actionId}">&#10003; Approve</button>
            <button class="btn btn-outline btn-reject" data-id="${actionId}">&#10005; Reject</button>
        </div>
    `;

    // Countdown timer
    const timerEl = card.querySelector(`#approval-timer-${shortId}`);
    const interval = setInterval(() => {
        const remaining = Math.max(0, Math.ceil((expiresAt - Date.now()) / 1000));
        if (timerEl) {
            timerEl.textContent = remaining > 0 ? `${remaining}s` : "expired";
        }
        if (remaining === 0) {
            clearInterval(interval);
            // Auto-remove timed-out card after a short grace period
            setTimeout(() => _removeCard(actionId), 2000);
        }
    }, 1000);

    card.querySelector(".btn-approve").addEventListener("click", async () => {
        try {
            await window.apiClient.approveAction(actionId);
        } catch (e) {
            console.warn("Approve request failed:", e);
        }
        _removeCard(actionId);
    });

    card.querySelector(".btn-reject").addEventListener("click", async () => {
        try {
            await window.apiClient.rejectAction(actionId);
        } catch (e) {
            console.warn("Reject request failed:", e);
        }
        _removeCard(actionId);
    });

    _cards[actionId] = { card, interval };
    area.appendChild(card);

    // Small delay so the animation plays from the initial off-screen position
    requestAnimationFrame(() => card.classList.add("approval-card-visible"));
}

function _resolveCard(event) {
    const actionId = eventField(event, "action_id", "");
    const entry = _cards[actionId];
    if (!entry) return;

    clearInterval(entry.interval);
    const decision = String(eventField(event, "decision", eventField(event, "approved", false) ? "approved" : "denied"));
    const description = eventField(event, "description", "");
    const routingPressure = Boolean(eventField(event, "routing_pressure", false));
    const titleEl = entry.card.querySelector(".approval-title");
    const bodyEl = entry.card.querySelector(".approval-card-body");
    const actionsEl = entry.card.querySelector(".approval-card-actions");
    const timerEl = entry.card.querySelector(".approval-timer");
    if (titleEl) {
        titleEl.textContent = decision === "approved" ? "Approval approved" : "Approval denied";
    }
    if (bodyEl && description) {
        bodyEl.textContent = routingPressure ? `${description} • routing pressure` : description;
    }
    if (actionsEl) {
        actionsEl.innerHTML = "";
    }
    if (timerEl) {
        timerEl.textContent = decision;
    }
    entry.card.classList.add("approval-card-resolved");
    setTimeout(() => _removeCard(actionId), 1200);
}

function _removeCard(action_id) {
    const entry = _cards[action_id];
    if (!entry) return;

    clearInterval(entry.interval);
    entry.card.classList.add("approval-card-resolved");

    // Remove from DOM after slide-out animation (400ms)
    setTimeout(() => {
        if (entry.card.parentNode) entry.card.remove();
        delete _cards[action_id];
    }, 400);
}

function _escapeHtml(text) {
    return (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ============== Public init ==============

export function initApprovals() {
    if (!area) return; // Safety: element not present

    window.apiClient.connectApprovals();

    window.apiClient.onApprovalEvent((event) => {
        const currentEventName = eventName(event);
        if (currentEventName === "approval.pending") {
            _renderCard(event);
        } else if (currentEventName === "approval.resolved") {
            _resolveCard(event);
        }
    });
}
