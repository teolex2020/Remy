/**
 * API Client — WebSocket for chat, REST for data.
 */

import { normalizeRuntimeEvent } from "./runtime-events.js";
import { newRequestId } from "./ui.js";

const MAX_RECONNECT_ATTEMPTS = 20;

class ApiClient {
    constructor() {
        this.ws = null;
        this.runtimeWs = null;
        this._messageHandlers = [];
        this._statusHandlers = [];
        this._activityHandlers = [];
        this._activityStatusHandlers = [];
        this._approvalHandlers = [];
        this._guidanceHandlers = [];
        this._runtimeHandlers = [];
        this._runtimeReconnect = 0;
        this._activityActive = false;
        this.reconnectAttempt = 0;
    }

    // ============== WebSocket ==============

    connectChat() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${protocol}//${location.host}/api/ws/chat`;

        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            this._emitStatus("connected");
            this.reconnectAttempt = 0; // Reset on success
        };

        this.ws.onclose = (event) => {
            this._emitStatus("disconnected");


            // Stop after max attempts
            if (this.reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
                console.warn(`WebSocket: gave up after ${MAX_RECONNECT_ATTEMPTS} attempts.`);
                this._emitStatus("failed");
                return;
            }

            // Exponential backoff
            const delay = Math.min(1000 * (2 ** this.reconnectAttempt), 30000); // Max 30s
            console.log(`WebSocket closed. Reconnecting in ${delay}ms (Attempt ${this.reconnectAttempt + 1}/${MAX_RECONNECT_ATTEMPTS})...`);

            setTimeout(() => {
                this._emitStatus("reconnecting");
                this.reconnectAttempt++;
                this.connectChat();
            }, delay);
        };

        this.ws.onerror = () => this._emitStatus("disconnected");

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this._messageHandlers.forEach((fn) => fn(data));
        };
    }

    sendMessage(text, options = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: "message",
                text,
                context_reducer_compare: Boolean(options.contextReducerCompare),
                context_reducer_apply: Boolean(options.contextReducerApply),
                model: options.model || undefined,
            }));
        }
    }

    sendVoice(audioBase64, mimeType = "audio/webm") {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: "voice",
                audio: audioBase64,
                mime_type: mimeType,
            }));
        }
    }

    sendFile(fileBase64, fileName, mimeType, text = "") {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: "file",
                data: fileBase64,
                name: fileName,
                mime_type: mimeType,
                text: text,
            }));
        }
    }

    sendNewSession() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "new_session" }));
        }
    }

    onMessage(fn) {
        this._messageHandlers.push(fn);
    }

    onStatus(fn) {
        this._statusHandlers.push(fn);
    }

    manualReconnect() {
        this.reconnectAttempt = 0;
        this.connectChat();
    }

    _emitStatus(status) {
        this._statusHandlers.forEach((fn) => fn(status));
    }

    async getLlmOptimizationMeasurements(limit = 100) {
        const resp = await fetch(`/api/llm-optimization/measurements?limit=${encodeURIComponent(limit)}`);
        if (!resp.ok) throw new Error(`Failed to load measurements: ${resp.status}`);
        return await resp.json();
    }

    async clearLlmOptimizationMeasurements() {
        const resp = await fetch("/api/llm-optimization/measurements", { method: "DELETE" });
        if (!resp.ok) throw new Error(`Failed to clear measurements: ${resp.status}`);
        return await resp.json();
    }

    async getLlmOptimizationModels() {
        const resp = await fetch("/api/llm-optimization/models");
        if (!resp.ok) throw new Error(`Failed to load models: ${resp.status}`);
        return await resp.json();
    }

    // ============== Activity WebSocket ==============

    connectActivity() {
        this._activityActive = true;
        this._connectRuntime();
        if (this.runtimeWs && this.runtimeWs.readyState === WebSocket.OPEN) {
            this._emitActivityStatus("connected");
        }
    }

    disconnectActivity() {
        this._activityActive = false;
        this._emitActivityStatus("disconnected");
    }

    onActivityEvent(fn) {
        this._activityHandlers.push(fn);
    }

    onActivityStatus(fn) {
        this._activityStatusHandlers.push(fn);
    }

    _emitActivityStatus(status) {
        this._activityStatusHandlers.forEach((fn) => fn(status));
    }

    // ============== Runtime WebSocket ==============

    _connectRuntime() {
        if (this.runtimeWs && (
            this.runtimeWs.readyState === WebSocket.OPEN ||
            this.runtimeWs.readyState === WebSocket.CONNECTING
        )) return;

        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        this.runtimeWs = new WebSocket(`${protocol}//${location.host}/api/ws/runtime`);

        this.runtimeWs.onopen = () => {
            this._runtimeReconnect = 0;
            if (this._activityActive) {
                this._emitActivityStatus("connected");
            }
        };

        this.runtimeWs.onmessage = (event) => {
            const data = normalizeRuntimeEvent(JSON.parse(event.data));
            this._runtimeHandlers.forEach((fn) => fn(data));
            if (this._activityActive) {
                this._activityHandlers.forEach((fn) => fn(data));
            }
            if (data.event_domain === "approval" || data.type.startsWith("approval.")) {
                this._approvalHandlers.forEach((fn) => fn(data));
            }
            if (data.event_domain === "guidance" || data.type.startsWith("guidance.")) {
                this._guidanceHandlers.forEach((fn) => fn(data));
            }
        };

        this.runtimeWs.onclose = (event) => {
            if (this._activityActive) {
                this._emitActivityStatus("disconnected");
            }
            if (event.code === 4001) return;

            if (this._runtimeReconnect >= MAX_RECONNECT_ATTEMPTS) {
                if (this._activityActive) {
                    console.warn(`Runtime WebSocket: gave up after ${MAX_RECONNECT_ATTEMPTS} attempts.`);
                    this._emitActivityStatus("failed");
                }
                return;
            }

            const delay = Math.min(1000 * (2 ** this._runtimeReconnect), 30000);
            this._runtimeReconnect++;
            setTimeout(() => {
                if (this._activityActive) {
                    this._emitActivityStatus("reconnecting");
                }
                this._connectRuntime();
            }, delay);
        };

        this.runtimeWs.onerror = () => {
            if (this._activityActive) {
                this._emitActivityStatus("disconnected");
            }
        };
    }

    connectApprovals() {
        this._connectRuntime();
    }

    connectRuntimeStream() {
        this._connectRuntime();
    }

    onApprovalEvent(fn) {
        this._approvalHandlers.push(fn);
    }

    async approveAction(actionId) {
        return this._fetch(`/api/approvals/${actionId}/approve`, { method: "POST" });
    }

    async rejectAction(actionId) {
        return this._fetch(`/api/approvals/${actionId}/reject`, { method: "POST" });
    }

    // ============== REST ==============

    async _fetch(url, options = {}) {
        const reqId = newRequestId();
        const headers = { ...options.headers, "X-Request-Id": reqId };
        const res = await fetch(url, { ...options, headers });

        if (!res.ok) {
            console.warn(`[${reqId}] API error ${res.status} for ${url}`);
        }
        return res;
    }

    async getStats() {
        const res = await this._fetch("/api/stats");
        return res.json();
    }

    async getDiagnostics() {
        const res = await this._fetch("/api/diagnostics");
        return res.json();
    }

    async getEvalMetrics(limit = 50) {
        const res = await this._fetch(`/api/eval-metrics?limit=${limit}`);
        return res.json();
    }

    async getRecords(tags = null, tier = "all", period = "all", offset = 0, limit = 50) {
        let url = `/api/records?limit=${limit}&offset=${offset}`;
        if (tags) url += `&tags=${encodeURIComponent(tags)}`;
        if (tier && tier !== "all") url += `&tier=${encodeURIComponent(tier)}`;
        if (period && period !== "all") url += `&period=${encodeURIComponent(period)}`;
        const res = await this._fetch(url);
        return res.json();
    }

    async getRecord(id) {
        const res = await this._fetch(`/api/records/${encodeURIComponent(id)}`);
        if (!res.ok) throw new Error(`Record not found`);
        return res.json();
    }

    async createRecord(payload) {
        const res = await this._fetch("/api/records", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error("Failed to create record");
        return res.json();
    }

    async updateRecord(id, payload) {
        const res = await this._fetch(`/api/records/${encodeURIComponent(id)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        return res.json();
    }

    async deleteRecord(id) {
        const res = await this._fetch(`/api/records/${encodeURIComponent(id)}`, {
            method: "DELETE",
        });
        return res.json();
    }

    async submitRecordFeedback(id, useful, reason = "") {
        const res = await this._fetch(`/api/records/${encodeURIComponent(id)}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ useful, reason }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Failed to submit record feedback");
        }
        return res.json();
    }

    async searchRecords(query, tags = null, tier = "all", period = "all", mode = "hybrid") {
        const res = await this._fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, tags, tier, period, mode, limit: 50 }),
        });
        return res.json();
    }

    async getGraph(mode = "user") {
        const search = new URLSearchParams();
        if (mode) search.set("mode", mode);
        const suffix = search.toString() ? `?${search.toString()}` : "";
        const res = await this._fetch(`/api/graph${suffix}`);
        return res.json();
    }

    // ============== Knowledge API (RM-8) ==============

    async getKnowledgeResearch() {
        const res = await this._fetch("/api/knowledge/research");
        return res.json();
    }

    async getKnowledgeMetrics(limit = 50) {
        const res = await this._fetch(`/api/knowledge/metrics?limit=${limit}`);
        return res.json();
    }

    async getKnowledgeFacts(limit = 50) {
        const res = await this._fetch(`/api/knowledge/facts?limit=${limit}`);
        return res.json();
    }

    async getKnowledgeStats() {
        const res = await this._fetch("/api/knowledge/stats");
        return res.json();
    }

    // ============== Knowledge Base (Aura Memory) ==============

    async getKnowledgeBase(limit = 100, offset = 0, query = "") {
        let url = `/api/knowledge/base?limit=${limit}&offset=${offset}`;
        if (query) url += `&query=${encodeURIComponent(query)}`;
        const res = await this._fetch(url);
        return res.json();
    }

    async ingestKnowledge(text, pin = false) {
        const res = await this._fetch("/api/knowledge/ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, pin }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Ingest failed");
        }
        return res.json();
    }

    async uploadKnowledgeFile(file, pin = false) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("pin", pin.toString());
        const res = await this._fetch("/api/knowledge/upload", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Upload failed");
        }
        return res.json();
    }

    async deleteKnowledgeItem(id) {
        const res = await this._fetch(`/api/knowledge/base/${encodeURIComponent(id)}`, {
            method: "DELETE",
        });
        return res.json();
    }

    // ── Identity (Profile + People) ──

    async getIdentity() {
        const res = await this._fetch("/api/knowledge/identity");
        return res.json();
    }

    async updateProfile(fields) {
        const res = await this._fetch("/api/knowledge/identity/profile", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(fields),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Update failed");
        }
        return res.json();
    }

    async updatePerson(id, fields) {
        const res = await this._fetch(`/api/knowledge/identity/person/${encodeURIComponent(id)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(fields),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Update failed");
        }
        return res.json();
    }

    // ============== Guidance WebSocket ==============

    connectGuidance() {
        this._connectRuntime();
    }

    onGuidanceEvent(fn) {
        this._guidanceHandlers.push(fn);
    }

    onRuntimeEvent(fn) {
        this._runtimeHandlers.push(fn);
    }

    async submitGuidanceAnswer(requestId, answer) {
        return this._fetch(`/api/guidance/${requestId}/answer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ answer }),
        });
    }

    // ============== Calendar ==============

    async getCalendar() {
        const res = await this._fetch("/api/knowledge/calendar");
        return res.json();
    }

    async getTodos(status = "active", category = null, limit = 50, days = null) {
        let url = `/api/todos?status=${status}&limit=${limit}`;
        if (category) url += `&category=${encodeURIComponent(category)}`;
        if (days) url += `&days=${encodeURIComponent(days)}`;
        const res = await this._fetch(url);
        return res.json();
    }

    async getTaskMetrics() {
        const res = await this._fetch("/api/task-metrics");
        return res.json();
    }

    async getAutonomyStatus() {
        const res = await this._fetch("/api/autonomy/status");
        return res.json();
    }

    async getSystemStatus() {
        const res = await this._fetch("/api/system/status");
        return res.json();
    }

    async toggleAutonomy() {
        const res = await this._fetch("/api/autonomy/toggle", {
            method: "POST",
        });
        return res.json();
    }

    async shutdownServer() {
        const res = await this._fetch("/api/server/shutdown", {
            method: "POST",
        });
        return res.json();
    }

    async getExecutionLogSummary() {
        const res = await this._fetch("/api/execution-log/summary");
        return res.json();
    }

    async getHarnessEvalHistory(limit = 20) {
        const res = await this._fetch(`/api/harness-eval-history?limit=${encodeURIComponent(limit)}`);
        return res.json();
    }

    async getGoalHistory(goalId) {
        const res = await this._fetch(`/api/goal-history/${encodeURIComponent(goalId)}`);
        return res.json();
    }

    async archiveAutonomyGoal(goalId, reason = "archived_by_user") {
        const res = await this._fetch(`/api/autonomy/goals/${encodeURIComponent(goalId)}/archive`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason }),
        });
        return res.json();
    }

    async unblockAutonomyGoal(goalId) {
        const res = await this._fetch(`/api/autonomy/goals/${encodeURIComponent(goalId)}/unblock`, {
            method: "POST",
        });
        return res.json();
    }

    async resumeAutonomyGoal(goalId, note = "") {
        const res = await this._fetch(`/api/autonomy/goals/${encodeURIComponent(goalId)}/resume`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ note }),
        });
        return res.json();
    }

    async getLiveValidation() {
        const res = await this._fetch("/api/autonomy/live-validation");
        return res.json();
    }

    async runLiveValidation() {
        const res = await this._fetch("/api/autonomy/live-validation/run", { method: "POST" });
        return res.json();
    }

    async getLiveValidationScenarios() {
        const res = await this._fetch("/api/autonomy/live-validation/scenarios");
        return res.json();
    }

    async saveLiveValidationScenarios(payload) {
        const res = await this._fetch("/api/autonomy/live-validation/scenarios", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        return res.json();
    }

    async createTodo(payload) {
        const res = await this._fetch("/api/todos", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error("Failed to create todo");
        return res.json();
    }

    async toggleTodo(id) {
        const res = await this._fetch(`/api/todos/${encodeURIComponent(id)}/toggle`, {
            method: "POST",
        });
        return res.json();
    }

    async deleteTodo(id) {
        const res = await this._fetch(`/api/todos/${encodeURIComponent(id)}`, {
            method: "DELETE",
        });
        return res.json();
    }
}

// Singleton
window.apiClient = new ApiClient();

// Backup: close session on page unload (tab close, navigate away, refresh)
// Primary close happens via WebSocket disconnect handler on the server.
// sendBeacon is fire-and-forget — works even if page is already closing.
window.addEventListener("beforeunload", () => {
    navigator.sendBeacon("/api/end-session", "");
});
