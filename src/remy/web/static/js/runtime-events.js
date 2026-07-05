export function eventName(event) {
    return event?.event_name || event?.type || "";
}

export function eventCssName(event) {
    return eventName(event).replace(/[^\w-]/g, "_");
}

export function eventDomain(event) {
    return event?.event_domain || "";
}

export function eventPayload(event) {
    if (event?.payload && typeof event.payload === "object" && !Array.isArray(event.payload)) {
        return event.payload;
    }
    return {};
}

export function eventField(event, key, fallback = "") {
    const payload = eventPayload(event);
    if (payload[key] !== undefined) {
        return payload[key];
    }
    if (event?.[key] !== undefined) {
        return event[key];
    }
    return fallback;
}

export function normalizeRuntimeEvent(event) {
    if (!event || typeof event !== "object") {
        return event;
    }
    const normalized = { ...event };
    normalized.type = eventName(normalized);
    normalized.event_name = eventName(normalized);
    normalized.event_domain = eventDomain(normalized);
    normalized.payload = eventPayload(normalized);
    return normalized;
}
