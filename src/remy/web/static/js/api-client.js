/**
 * API Client — WebSocket for chat, REST for data.
 */

import { normalizeRuntimeEvent } from "./runtime-events.js";
import { newRequestId } from "./ui.js";

const MAX_RECONNECT_ATTEMPTS = 20;

const API_FIELD_LABELS = {
    name: "Project name",
    domain: "Project area",
    description: "Project purpose",
    workspace_id: "Project folder",
};

function formatApiErrorDetail(detail, fallbackMessage = "Request failed") {
    if (typeof detail === "string" && detail.trim()) return detail.trim();

    if (Array.isArray(detail)) {
        const messages = detail.map((item) => {
            if (typeof item === "string") return item.trim();
            if (!item || typeof item !== "object") return "";

            const location = Array.isArray(item.loc)
                ? item.loc.filter((part) => !["body", "query", "path"].includes(String(part)))
                : [];
            const field = location.length
                ? API_FIELD_LABELS[String(location.at(-1))] || location.join(".")
                : "";
            const message = item.msg || item.message || item.detail || "";
            if (field && message) return `${field}: ${message}`;
            if (message) return String(message);
            try {
                return JSON.stringify(item);
            } catch (_) {
                return "";
            }
        }).filter(Boolean);
        if (messages.length) return messages.join(". ");
    }

    if (detail && typeof detail === "object") {
        const nested = detail.message || detail.error || detail.detail;
        if (nested && nested !== detail) {
            return formatApiErrorDetail(nested, fallbackMessage);
        }
        try {
            const serialized = JSON.stringify(detail);
            if (serialized && serialized !== "{}") return serialized;
        } catch (_) {
            // Fall through to the caller-provided message.
        }
    }

    return fallbackMessage;
}

async function responsePayload(res) {
    try {
        return await res.json();
    } catch (_) {
        return {};
    }
}

async function readApiJson(res, fallbackMessage = "Request failed") {
    const payload = await responsePayload(res);
    if (!res.ok) {
        throw new Error(formatApiErrorDetail(
            payload.detail ?? payload.error ?? payload.message,
            fallbackMessage,
        ));
    }
    return payload;
}

function hasLegacyProjectProfileValidation(detail) {
    if (!Array.isArray(detail) || !detail.length) return false;
    const unsupported = new Set(["domain", "description"]);
    return detail.every((item) => {
        if (!item || typeof item !== "object" || item.type !== "extra_forbidden") return false;
        const location = Array.isArray(item.loc) ? item.loc : [];
        return unsupported.has(String(location.at(-1)));
    });
}

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
        this._runtimeStatusHandlers = [];
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
                workspace_id: options.workspaceId || undefined,
                team_mode: options.teamMode || "off",
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

    cancelGeneration() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "cancel" }));
        }
    }

    async getProjects(includeArchived = false) {
        const query = includeArchived ? "?include_archived=true" : "";
        const res = await this._fetch(`/api/projects${query}`);
        return readApiJson(res, "Could not load projects");
    }

    async createProject(name, activate = true, profile = {}) {
        const body = { name, activate };
        const domain = String(profile.domain || "").trim();
        const description = String(profile.description || "").trim();
        if (domain) body.domain = domain;
        if (description) body.description = description;

        const request = (payload) => this._fetch("/api/projects", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        let res = await request(body);
        if (!res.ok && res.status === 422 && (domain || description)) {
            const validation = await responsePayload(res.clone());
            if (hasLegacyProjectProfileValidation(validation.detail)) {
                // A browser can receive newer static files from disk while the
                // already-running Python process still has the previous request
                // schema in memory. Preserve the essential operation: create the
                // project by title, then let a restart enable the richer profile.
                res = await request({ name, activate });
                const created = await readApiJson(res, "Could not create project");
                created.compatibility_warning =
                    "Project created from its title. Restart Remy before adding its area or purpose.";
                return created;
            }
        }
        return readApiJson(res, "Could not create project");
    }

    async activateProject(projectId) {
        const res = await this._fetch(`/api/projects/${encodeURIComponent(projectId)}/activate`, {
            method: "POST",
        });
        return readApiJson(res, "Could not open project");
    }

    async updateProject(projectId, changes) {
        const res = await this._fetch(`/api/projects/${encodeURIComponent(projectId)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(changes || {}),
        });
        return readApiJson(res, "Could not update project");
    }

    async getProjectAgent(projectId) {
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/agent`,
        );
        return readApiJson(res, "Could not load Project Agent");
    }

    async updateProjectAgent(projectId, changes) {
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/agent`,
            {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(changes || {}),
            },
        );
        return readApiJson(res, "Could not update Project Agent");
    }

    async createKnowledgePack(projectId, name, description = "") {
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/knowledge-packs`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, description }),
            },
        );
        return readApiJson(res, "Could not create Knowledge Pack");
    }

    async updateKnowledgePack(projectId, packId, changes) {
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/knowledge-packs/${encodeURIComponent(packId)}`,
            {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(changes || {}),
            },
        );
        return readApiJson(res, "Could not update Knowledge Pack");
    }

    async deleteKnowledgePack(projectId, packId) {
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/knowledge-packs/${encodeURIComponent(packId)}`,
            { method: "DELETE" },
        );
        return readApiJson(res, "Could not delete Knowledge Pack");
    }

    async uploadKnowledgePackSource(projectId, packId, file) {
        const body = new FormData();
        body.append("file", file);
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/knowledge-packs/${encodeURIComponent(packId)}/sources`,
            { method: "POST", body },
        );
        return readApiJson(res, "Could not upload knowledge source");
    }

    async deleteKnowledgePackSource(projectId, packId, sourceId) {
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/knowledge-packs/${encodeURIComponent(packId)}/sources/${encodeURIComponent(sourceId)}`,
            { method: "DELETE" },
        );
        return readApiJson(res, "Could not delete knowledge source");
    }

    async archiveProject(projectId) {
        const res = await this._fetch(`/api/projects/${encodeURIComponent(projectId)}`, {
            method: "DELETE",
        });
        return readApiJson(res, "Could not archive project");
    }

    async restoreProject(projectId) {
        const res = await this._fetch(
            `/api/projects/${encodeURIComponent(projectId)}/restore`,
            { method: "POST" },
        );
        return readApiJson(res, "Could not restore project");
    }

    async getConversations() {
        const res = await this._fetch("/api/conversations");
        return res.json();
    }

    async createConversation(title = "New conversation") {
        const res = await this._fetch("/api/conversations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title }),
        });
        return res.json();
    }

    async activateConversation(conversationId) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}/activate`,
            { method: "POST" },
        );
        return res.json();
    }

    async renameConversation(conversationId, title) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}`,
            {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title }),
            },
        );
        return res.json();
    }

    async archiveConversation(conversationId) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}`,
            { method: "DELETE" },
        );
        return res.json();
    }

    async getConversationMessages(conversationId, limit = 120, options = {}) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}/messages?limit=${encodeURIComponent(limit)}`,
            { signal: options.signal },
        );
        return res.json();
    }

    async getConversationTrajectory(conversationId, limit = 1500) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}/trajectory?limit=${encodeURIComponent(limit)}`,
        );
        return readApiJson(res, "Could not load trajectory");
    }

    async getTrajectoryAnalytics(days = 30, limit = 20000) {
        const query = new URLSearchParams({
            days: String(days),
            limit: String(limit),
        });
        const res = await this._fetch(`/api/trajectory/analytics?${query.toString()}`);
        return readApiJson(res, "Could not load project trajectory analytics");
    }

    async createTrajectoryBaseline(name, days = 30, activate = true) {
        const res = await this._fetch("/api/trajectory/analytics/baselines", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, days, activate }),
        });
        return readApiJson(res, "Could not create trajectory baseline");
    }

    async activateTrajectoryBaseline(baselineId) {
        const res = await this._fetch(
            `/api/trajectory/analytics/baselines/${encodeURIComponent(baselineId)}/activate`,
            { method: "PUT" },
        );
        return readApiJson(res, "Could not activate trajectory baseline");
    }

    async useTrajectoryWindowMedian() {
        const res = await this._fetch("/api/trajectory/analytics/baselines/window-median", {
            method: "PUT",
        });
        return readApiJson(res, "Could not switch to the live window median");
    }

    async deleteTrajectoryBaseline(baselineId) {
        const res = await this._fetch(
            `/api/trajectory/analytics/baselines/${encodeURIComponent(baselineId)}`,
            { method: "DELETE" },
        );
        return readApiJson(res, "Could not delete trajectory baseline");
    }

    async updateTrajectoryAlert(alertId, status) {
        const res = await this._fetch(
            `/api/trajectory/analytics/alerts/${encodeURIComponent(alertId)}`,
            {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ status }),
            },
        );
        return readApiJson(res, "Could not update regression alert");
    }

    async getTrajectoryAlertHistory(alertId = "", limit = 200) {
        const query = new URLSearchParams({ limit: String(limit) });
        if (alertId) query.set("alert_id", alertId);
        const res = await this._fetch(
            `/api/trajectory/analytics/alert-history?${query.toString()}`,
        );
        return readApiJson(res, "Could not load trajectory alert history");
    }

    async updateTrajectorySlo(config) {
        const res = await this._fetch("/api/trajectory/analytics/slo", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config || {}),
        });
        return readApiJson(res, "Could not update trajectory SLO");
    }

    async getTrajectorySloIncidents(status = "", limit = 100) {
        const query = new URLSearchParams({ limit: String(limit) });
        if (status) query.set("status", status);
        const res = await this._fetch(
            `/api/trajectory/analytics/slo/incidents?${query.toString()}`,
        );
        return readApiJson(res, "Could not load trajectory SLO incidents");
    }

    async updateTrajectorySloIncident(incidentId, status) {
        const res = await this._fetch(
            `/api/trajectory/analytics/slo/incidents/${encodeURIComponent(incidentId)}`,
            {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ status }),
            },
        );
        return readApiJson(res, "Could not update trajectory SLO incident");
    }

    async getTrajectoryIncidentDossier(incidentId) {
        const res = await this._fetch(
            `/api/trajectory/analytics/incidents/${encodeURIComponent(incidentId)}/dossier`,
        );
        return readApiJson(res, "Could not build trajectory incident dossier");
    }

    async createTrajectoryEvalCase(incidentId, name = "") {
        const res = await this._fetch(
            `/api/trajectory/analytics/incidents/${encodeURIComponent(incidentId)}/eval-cases`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            },
        );
        return readApiJson(res, "Could not create trajectory regression eval");
    }

    async getTrajectoryEvalCases(limit = 100) {
        const res = await this._fetch(
            `/api/trajectory/analytics/eval-cases?limit=${encodeURIComponent(limit)}`,
        );
        return readApiJson(res, "Could not load trajectory regression evals");
    }

    async runTrajectoryEvalCase(caseId, conversationId = "") {
        const res = await this._fetch(
            `/api/trajectory/analytics/eval-cases/${encodeURIComponent(caseId)}/runs`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ conversation_id: conversationId }),
            },
        );
        return readApiJson(res, "Could not run trajectory regression eval");
    }

    async sandboxReplayTrajectoryEvalCase(caseId) {
        const res = await this._fetch(
            `/api/trajectory/analytics/eval-cases/${encodeURIComponent(caseId)}/sandbox-replay`,
            { method: "POST" },
        );
        return readApiJson(res, "Could not run trajectory sandbox replay");
    }

    async runTrajectoryEvalMatrix({ name = "", preferredModel = "", caseIds = [] } = {}) {
        const res = await this._fetch("/api/trajectory/analytics/eval-matrices", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name,
                preferred_model: preferredModel,
                case_ids: caseIds,
            }),
        });
        return readApiJson(res, "Could not run trajectory replay matrix");
    }

    async getTrajectoryEvalMatrices(limit = 20) {
        const res = await this._fetch(
            `/api/trajectory/analytics/eval-matrices?limit=${encodeURIComponent(limit)}`,
        );
        return readApiJson(res, "Could not load trajectory replay matrices");
    }

    async runTrajectoryEvalComparison({ name = "", models = [], caseIds = [] } = {}) {
        const res = await this._fetch("/api/trajectory/analytics/eval-comparisons", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, models, case_ids: caseIds }),
        });
        return readApiJson(res, "Could not run trajectory model comparison");
    }

    async getTrajectoryEvalComparisons(limit = 20) {
        const res = await this._fetch(
            `/api/trajectory/analytics/eval-comparisons?limit=${encodeURIComponent(limit)}`,
        );
        return readApiJson(res, "Could not load trajectory model comparisons");
    }

    async startTrajectoryModelPromotion({ candidateModel, confirmModel, canaryPercent = 10 }) {
        const res = await this._fetch("/api/trajectory/analytics/model-promotions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                candidate_model: candidateModel,
                confirm_model: confirmModel,
                canary_percent: canaryPercent,
            }),
        });
        return readApiJson(res, "Could not start model promotion canary");
    }

    async finalizeTrajectoryModelPromotion(promotionId, confirmModel) {
        const res = await this._fetch(
            `/api/trajectory/analytics/model-promotions/${encodeURIComponent(promotionId)}/promote`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ confirm_model: confirmModel }),
            },
        );
        return readApiJson(res, "Could not finalize model promotion");
    }

    async rollbackTrajectoryModelPromotion(promotionId, confirmModel) {
        const res = await this._fetch(
            `/api/trajectory/analytics/model-promotions/${encodeURIComponent(promotionId)}/rollback`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ confirm_model: confirmModel }),
            },
        );
        return readApiJson(res, "Could not roll back model promotion");
    }

    async getTrajectoryEvalRuns(caseId, limit = 50) {
        const res = await this._fetch(
            `/api/trajectory/analytics/eval-cases/${encodeURIComponent(caseId)}/runs?limit=${encodeURIComponent(limit)}`,
        );
        return readApiJson(res, "Could not load trajectory eval runs");
    }

    async deleteTrajectoryEvalCase(caseId) {
        const res = await this._fetch(
            `/api/trajectory/analytics/eval-cases/${encodeURIComponent(caseId)}`,
            { method: "DELETE" },
        );
        return readApiJson(res, "Could not delete trajectory regression eval");
    }

    async createTrajectoryAlertPolicy(policy) {
        const res = await this._fetch("/api/trajectory/analytics/policies", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(policy || {}),
        });
        return readApiJson(res, "Could not create trajectory alert policy");
    }

    async updateTrajectoryAlertPolicy(policyId, policy) {
        const res = await this._fetch(
            `/api/trajectory/analytics/policies/${encodeURIComponent(policyId)}`,
            {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(policy || {}),
            },
        );
        return readApiJson(res, "Could not update trajectory alert policy");
    }

    async deleteTrajectoryAlertPolicy(policyId) {
        const res = await this._fetch(
            `/api/trajectory/analytics/policies/${encodeURIComponent(policyId)}`,
            { method: "DELETE" },
        );
        return readApiJson(res, "Could not delete trajectory alert policy");
    }

    async forkConversationTrajectory(conversationId, eventId, options = {}) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}/trajectory/${encodeURIComponent(eventId)}/fork`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(options),
            },
        );
        return readApiJson(res, "Could not fork trajectory");
    }

    async updateTrajectoryAnnotation(conversationId, eventId, annotation) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}/trajectory/${encodeURIComponent(eventId)}/annotation`,
            {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(annotation || {}),
            },
        );
        return readApiJson(res, "Could not save trajectory annotation");
    }

    async deleteTrajectoryAnnotation(conversationId, eventId) {
        const res = await this._fetch(
            `/api/conversations/${encodeURIComponent(conversationId)}/trajectory/${encodeURIComponent(eventId)}/annotation`,
            { method: "DELETE" },
        );
        return readApiJson(res, "Could not delete trajectory annotation");
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
            this._emitRuntimeStatus("connected");
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
            this._emitRuntimeStatus("disconnected");
            if (this._activityActive) {
                this._emitActivityStatus("disconnected");
            }
            if (event.code === 4001) return;

            if (this._runtimeReconnect >= MAX_RECONNECT_ATTEMPTS) {
                this._emitRuntimeStatus("failed");
                if (this._activityActive) {
                    console.warn(`Runtime WebSocket: gave up after ${MAX_RECONNECT_ATTEMPTS} attempts.`);
                    this._emitActivityStatus("failed");
                }
                return;
            }

            const delay = Math.min(1000 * (2 ** this._runtimeReconnect), 30000);
            this._runtimeReconnect++;
            this._emitRuntimeStatus("reconnecting");
            setTimeout(() => {
                if (this._activityActive) {
                    this._emitActivityStatus("reconnecting");
                }
                this._connectRuntime();
            }, delay);
        };

        this.runtimeWs.onerror = () => {
            this._emitRuntimeStatus("disconnected");
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

    async getWorkflowFiles() {
        const res = await this._fetch("/api/workflow-files");
        return res.json();
    }

    async uploadWorkflowFile(file) {
        const form = new FormData();
        form.append("file", file);
        const res = await this._fetch("/api/workflow-files", { method: "POST", body: form });
        return res.json();
    }

    async deleteWorkflowFile(name) {
        const res = await this._fetch(`/api/workflow-files/${encodeURIComponent(name)}`, { method: "DELETE" });
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

    async getResearchNotifications(limit = 20) {
        const res = await this._fetch(`/api/knowledge/research/notifications?limit=${limit}`);
        return res.json();
    }

    async pauseResearch(projectId) {
        const res = await this._fetch(`/api/knowledge/research/${encodeURIComponent(projectId)}/pause`, {
            method: "POST",
        });
        return res.json();
    }

    async resumeResearch(projectId) {
        const res = await this._fetch(`/api/knowledge/research/${encodeURIComponent(projectId)}/resume`, {
            method: "POST",
        });
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

    async getLearningReviews(status = "pending", limit = 100) {
        const res = await this._fetch(`/api/knowledge/learning-reviews?status=${encodeURIComponent(status)}&limit=${limit}`);
        return res.json();
    }

    async decideLearningReview(reviewId, decision) {
        const res = await this._fetch(`/api/knowledge/learning-reviews/${encodeURIComponent(reviewId)}/decision`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ decision }),
        });
        return res.json();
    }

    async getExecutionAttempts(limit = 100) {
        const res = await this._fetch(`/api/knowledge/execution-attempts?limit=${limit}`);
        return res.json();
    }

    // ============== Background task handles ==============

    async launchWorkerTasks(tasks, conversationId = "") {
        const res = await this._fetch("/api/tasks/workers", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tasks, conversation_id: conversationId, channel: "web" }),
        });
        return readApiJson(res, "Could not launch background task");
    }

    async getWorkerTasks(status = "", limit = 100) {
        const query = new URLSearchParams({ status, limit: String(limit) });
        const res = await this._fetch(`/api/tasks?${query.toString()}`);
        return readApiJson(res, "Could not load background tasks");
    }

    async getWorkerTask(taskId) {
        const res = await this._fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
        return readApiJson(res, "Could not load background task");
    }

    async cancelWorkerTask(taskId, reason = "Stopped by user") {
        const res = await this._fetch(`/api/tasks/${encodeURIComponent(taskId)}/cancel`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason }),
        });
        return readApiJson(res, "Could not stop background task");
    }

    async resumeWorkerTask(taskId, confirmSideEffects = false) {
        const res = await this._fetch(`/api/tasks/${encodeURIComponent(taskId)}/resume`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ confirm_side_effects: Boolean(confirmSideEffects) }),
        });
        return readApiJson(res, "Could not resume background task");
    }

    async getChildSessions(parentSessionId = "", limit = 100) {
        const query = new URLSearchParams({
            parent_session_id: parentSessionId,
            limit: String(limit),
        });
        const res = await this._fetch(`/api/children?${query.toString()}`);
        return readApiJson(res, "Could not load child sessions");
    }

    async getChildReport(childId) {
        const res = await this._fetch(`/api/children/${encodeURIComponent(childId)}/report`);
        return readApiJson(res, "Could not load child report");
    }

    async followUpChild(childId, message, confirmSideEffects = false) {
        const res = await this._fetch(`/api/children/${encodeURIComponent(childId)}/follow-up`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message,
                confirm_side_effects: Boolean(confirmSideEffects),
            }),
        });
        return readApiJson(res, "Could not send child follow-up");
    }

    async interruptChild(childId, reason = "Interrupted by parent") {
        const res = await this._fetch(`/api/children/${encodeURIComponent(childId)}/interrupt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason }),
        });
        return readApiJson(res, "Could not interrupt child session");
    }

    async resumeChild(childId, confirmSideEffects = false) {
        const res = await this._fetch(`/api/children/${encodeURIComponent(childId)}/resume`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ confirm_side_effects: Boolean(confirmSideEffects) }),
        });
        return readApiJson(res, "Could not resume child session");
    }

    async getPtcTools() {
        const res = await this._fetch("/api/ptc/tools");
        return readApiJson(res, "Could not load PTC tool allowlist");
    }

    async validatePtcProgram(program, limits = {}) {
        const res = await this._fetch("/api/ptc/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ program, limits }),
        });
        return readApiJson(res, "Could not validate PTC program");
    }

    async runPtcProgram(program, limits = {}, sessionId = "") {
        const res = await this._fetch("/api/ptc/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ program, limits, session_id: sessionId, channel: "web" }),
        });
        return readApiJson(res, "Could not run PTC program");
    }

    async searchTranscripts(query, limit = 20) {
        const res = await this._fetch(`/api/knowledge/transcripts/search?q=${encodeURIComponent(query)}&limit=${limit}`);
        return res.json();
    }

    async getPipelineCandidates(status = "all", limit = 100) {
        const res = await this._fetch(`/api/pipeline-candidates?status=${encodeURIComponent(status)}&limit=${limit}`);
        return res.json();
    }

    async dryRunPipelineCandidate(candidateId, inputText = "") {
        const res = await this._fetch(`/api/pipeline-candidates/${encodeURIComponent(candidateId)}/dry-run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input_text: inputText }),
        });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Dry-run failed");
        return res.json();
    }

    async decidePipelineCandidate(candidateId, decision) {
        const res = await this._fetch(`/api/pipeline-candidates/${encodeURIComponent(candidateId)}/decision`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ decision }),
        });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Decision failed");
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

    onRuntimeStatus(fn) {
        this._runtimeStatusHandlers.push(fn);
        if (!this.runtimeWs) return;
        if (this.runtimeWs.readyState === WebSocket.OPEN) fn("connected");
        else if (this.runtimeWs.readyState === WebSocket.CONNECTING) fn("connecting");
        else fn("disconnected");
    }

    _emitRuntimeStatus(status) {
        this._runtimeStatusHandlers.forEach((fn) => fn(status));
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

    async getOperatorAlerts(limit = 8, options = {}) {
        const query = new URLSearchParams({ limit: String(limit) });
        const res = await this._fetch(`/api/system/operator-alerts?${query.toString()}`, {
            signal: options.signal,
        });
        return readApiJson(res, "Could not load operator incidents");
    }

    async acknowledgeOperatorAlert(alertId) {
        const res = await this._fetch(
            `/api/system/operator-alerts/${encodeURIComponent(alertId)}/ack`,
            { method: "POST" },
        );
        return readApiJson(res, "Could not acknowledge operator incident");
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
