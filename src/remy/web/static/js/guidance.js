/**
 * Guidance Panel — real-time guidance request notifications.
 *
 * Connects to the shared /ws/human-loop WebSocket.
 * When the autonomous agent needs user help, a notification card
 * slides in with the question + text input for the user to answer.
 */

import { eventField, eventName } from "./runtime-events.js";

const guidanceArea = document.getElementById("guidance-notification-area");

/** request_id → { card: HTMLElement, interval: number } */
const _guidanceCards = {};

// ============== Card rendering ==============

function _renderGuidanceCard(event) {
    const requestId = eventField(event, "request_id", "");
    const question = eventField(event, "question", "");
    const context = eventField(event, "context", "");
    const timeoutSec = Number(eventField(event, "timeout_sec", 0));
    const createdAt = Number(eventField(event, "created_at", Date.now() / 1000));

    if (!requestId || _guidanceCards[requestId]) return;

    const card = document.createElement("div");
    card.className = "guidance-card";
    card.dataset.requestId = requestId;

    const expiresAt = (createdAt + timeoutSec) * 1000;
    const shortId = requestId.slice(0, 8);
    const contextHtml = context
        ? `<div class="guidance-card-context">${_escapeGuidanceHtml(context)}</div>`
        : "";

    card.innerHTML = `
        <div class="guidance-card-header">
            <span class="guidance-icon">&#10067;</span>
            <span class="guidance-title">Agent needs help</span>
            <span class="guidance-timer" id="guidance-timer-${shortId}"></span>
        </div>
        <div class="guidance-card-body">${_escapeGuidanceHtml(question)}</div>
        ${contextHtml}
        <div class="guidance-card-input">
            <input type="text" class="guidance-input" placeholder="Type your answer..." />
            <button class="btn btn-primary btn-guidance-send" data-id="${requestId}">Send</button>
        </div>
    `;

    // Countdown timer
    const timerEl = card.querySelector(`#guidance-timer-${shortId}`);
    const interval = setInterval(() => {
        const remaining = Math.max(0, Math.ceil((expiresAt - Date.now()) / 1000));
        if (timerEl) {
            timerEl.textContent = remaining > 0 ? `${remaining}s` : "expired";
        }
        if (remaining === 0) {
            clearInterval(interval);
            setTimeout(() => _removeGuidanceCard(requestId), 2000);
        }
    }, 1000);

    const input = card.querySelector(".guidance-input");
    const sendBtn = card.querySelector(".btn-guidance-send");

    async function submitAnswer() {
        const answer = input.value.trim();
        if (!answer) return;
        try {
            await window.apiClient.submitGuidanceAnswer(requestId, answer);
        } catch (e) {
            console.warn("Guidance answer failed:", e);
        }
        _removeGuidanceCard(requestId);
    }

    sendBtn.addEventListener("click", submitAnswer);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") submitAnswer();
    });

    _guidanceCards[requestId] = { card, interval };
    guidanceArea.appendChild(card);

    requestAnimationFrame(() => card.classList.add("guidance-card-visible"));
}

function _removeGuidanceCard(request_id) {
    const entry = _guidanceCards[request_id];
    if (!entry) return;

    clearInterval(entry.interval);
    entry.card.classList.add("guidance-card-resolved");

    setTimeout(() => {
        if (entry.card.parentNode) entry.card.remove();
        delete _guidanceCards[request_id];
    }, 400);
}

function _escapeGuidanceHtml(text) {
    return (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ============== Public init ==============

export function initGuidance() {
    if (!guidanceArea) return;

    window.apiClient.connectGuidance();

    window.apiClient.onGuidanceEvent((event) => {
        const currentEventName = eventName(event);
        if (currentEventName === "guidance.pending") {
            _renderGuidanceCard(event);
        } else if (currentEventName === "guidance.resolved") {
            _removeGuidanceCard(eventField(event, "request_id", ""));
        }
    });
}
