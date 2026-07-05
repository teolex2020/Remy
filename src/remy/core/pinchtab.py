"""
PinchTab backend adapter.

Provides an optional text-first browser backend that can complement the
existing Playwright + vision path. The goal is to use PinchTab for cheap
page extraction and stable element references, while keeping Playwright as a
fallback for legacy selectors and visual verification.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from urllib.parse import urlparse

import httpx

from remy.config.settings import settings

logger = logging.getLogger("PinchTab")


class PinchTabError(RuntimeError):
    """Raised when the PinchTab backend cannot complete a request."""


def _normalize_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.scheme:
        return f"https://{text}"
    return text


class PinchTabManager:
    """Singleton manager for an external PinchTab service."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._instance_id: str | None = None
        self._tab_id: str | None = None
        self._last_activity: float = 0.0

    @classmethod
    def get(cls) -> "PinchTabManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    @staticmethod
    def enabled() -> bool:
        return bool(settings.PINCHTAB_ENABLED and settings.PINCHTAB_BASE_URL.strip())

    @staticmethod
    def _timeout() -> float:
        return max(5, int(settings.PINCHTAB_TIMEOUT_SEC))

    @staticmethod
    def _base_url() -> str:
        return settings.PINCHTAB_BASE_URL.rstrip("/")

    async def _request(self, method: str, path: str, *, json_data: dict | None = None) -> object:
        url = f"{self._base_url()}{path}"
        async with httpx.AsyncClient(timeout=self._timeout()) as client:
            resp = await client.request(method, url, json=json_data)
            resp.raise_for_status()
            if not resp.content:
                return {}
            return resp.json()

    async def ensure_instance(self) -> str:
        if self._instance_id:
            return self._instance_id
        payload = {
            "mode": "headless" if settings.BROWSER_HEADLESS else "headed",
        }
        if settings.PINCHTAB_PROFILE_ID:
            payload["profileId"] = settings.PINCHTAB_PROFILE_ID
        data = await self._request("POST", "/instances/launch", json_data=payload)
        instance_id = str(data.get("id") or data.get("instanceId") or "")
        if not instance_id:
            raise PinchTabError("PinchTab did not return an instance id.")
        self._instance_id = instance_id
        return instance_id

    async def open_tab(self, url: str) -> tuple[str, str]:
        normalized_url = _normalize_url(url)
        if not normalized_url:
            raise PinchTabError("url is required")
        instance_id = await self.ensure_instance()
        data = await self._request(
            "POST",
            f"/instances/{instance_id}/tabs/open",
            json_data={"url": normalized_url},
        )
        tab = data.get("tab") if isinstance(data, dict) else None
        tab_id = str(
            data.get("id")
            or data.get("tabId")
            or (tab or {}).get("id")
            or ""
        )
        page_url = str(data.get("url") or (tab or {}).get("url") or normalized_url)
        if not tab_id:
            raise PinchTabError("PinchTab did not return a tab id.")
        self._tab_id = tab_id
        self._last_activity = time.time()
        return tab_id, page_url

    async def get_text(self, tab_id: str) -> str:
        data = await self._request("GET", f"/tabs/{tab_id}/text")
        if isinstance(data, dict):
            return str(data.get("text") or data.get("content") or "")
        return ""

    async def get_snapshot(self, tab_id: str) -> list[dict]:
        data = await self._request("GET", f"/tabs/{tab_id}/snapshot?filter=interactive")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            nodes = data.get("items") or data.get("nodes") or data.get("elements") or []
            if isinstance(nodes, list):
                return [item for item in nodes if isinstance(item, dict)]
        return []

    async def browse_page(self, url: str, question: str = "") -> dict:
        tab_id, page_url = await self.open_tab(url)
        page_text = await self.get_text(tab_id)
        snapshot = await self.get_snapshot(tab_id)
        dom_elements = _snapshot_to_dom_elements(snapshot)
        description = _describe_page(page_text, dom_elements)
        answer = _answer_question(question, page_text)
        return {
            "backend": "pinchtab",
            "tab_id": tab_id,
            "url": page_url,
            "requested_url": _normalize_url(url),
            "question": question,
            "page_text": page_text[:5000],
            "dom_elements": dom_elements,
            "dom_form_fields": _dom_form_fields(dom_elements),
            "description": description,
            "answer": answer,
            "page_state": _page_state(page_text, dom_elements),
            "snapshot_count": len(snapshot),
        }

    async def act(self, *, action: str, selector: str | None = None, text: str | None = None, url: str | None = None) -> dict:
        import asyncio

        normalized_url = _normalize_url(url or "")
        if action in {"goto", "open"}:
            tab_id, page_url = await self.open_tab(normalized_url)
            page_text = await self.get_text(tab_id)
            return {
                "backend": "pinchtab",
                "tab_id": tab_id,
                "url": page_url,
                "requested_url": normalized_url,
                "page_text": page_text[:5000],
                "action": action,
                "description": _describe_page(page_text, []),
                "page_state": _page_state(page_text, []),
            }

        if action == "wait":
            timeout_ms = int(text) if text and str(text).isdigit() else 2000
            await asyncio.sleep(min(timeout_ms, 10000) / 1000)
            return {
                "backend": "pinchtab",
                "tab_id": self._tab_id or "",
                "url": "",
                "requested_url": normalized_url,
                "page_text": "",
                "action": action,
                "description": "Wait completed.",
                "page_state": "unknown",
            }

        if not self._tab_id:
            raise PinchTabError("No active PinchTab tab. Call browse_page first.")

        payload = {"kind": action}
        ref = _selector_to_ref(selector)
        if ref:
            payload["ref"] = ref
        elif selector:
            payload["selector"] = selector
        if text is not None:
            payload["text"] = text
            payload["value"] = text

        await self._request("POST", f"/tabs/{self._tab_id}/actions", json_data=payload)
        page_text = await self.get_text(self._tab_id)
        page_url = ""
        try:
            info = await self._request("GET", f"/tabs/{self._tab_id}")
            if isinstance(info, dict):
                page_url = str(info.get("url") or "")
        except Exception:
            page_url = ""
        return {
            "backend": "pinchtab",
            "tab_id": self._tab_id,
            "url": page_url,
            "requested_url": normalized_url,
            "page_text": page_text[:5000],
            "action": action,
            "description": _describe_page(page_text, []),
            "page_state": _page_state(page_text, []),
        }

    async def close(self) -> None:
        instance_id = self._instance_id
        self._tab_id = None
        self._instance_id = None
        if not instance_id:
            return
        try:
            await self._request("POST", f"/instances/{instance_id}/stop")
        except Exception as exc:
            logger.warning("PinchTab close failed: %s", exc)


def _selector_to_ref(selector: str | None) -> str | None:
    if not selector:
        return None
    text = selector.strip()
    if text.startswith("ref:"):
        text = text[4:]
    if text.startswith("pinch:"):
        text = text[6:]
    if text and text[0] == "e" and text[1:].isdigit():
        return text
    return None


def _snapshot_to_dom_elements(nodes: list[dict]) -> list[dict]:
    dom_elements: list[dict] = []
    for node in nodes:
        ref = str(node.get("ref") or "")
        label = str(node.get("label") or node.get("name") or node.get("text") or "")
        role = str(node.get("role") or node.get("type") or "")
        selector = f"ref:{ref}" if ref else ""
        dom_elements.append(
            {
                "tag": role or "element",
                "text": label,
                "selector": selector,
                "visible_label": label,
                "aria_label": label,
                "role": role,
                "ref": ref,
                "backend": "pinchtab",
            }
        )
    return dom_elements


def _dom_form_fields(dom_elements: list[dict]) -> list[dict]:
    fields: list[dict] = []
    for el in dom_elements:
        role = str(el.get("role") or el.get("tag") or "").lower()
        if role not in {"textbox", "input", "textarea", "select", "combobox", "checkbox", "radio"}:
            continue
        fields.append(
            {
                "label": el.get("visible_label") or el.get("text") or role,
                "selector": el.get("selector", ""),
                "type": role,
                "backend": "pinchtab",
            }
        )
    return fields


def _describe_page(page_text: str, dom_elements: list[dict]) -> str:
    lines = [line.strip() for line in (page_text or "").splitlines() if line.strip()]
    headline = lines[0][:160] if lines else "Text-first page extraction completed."
    interactive = len(dom_elements)
    if interactive:
        return f"{headline} ({interactive} interactive elements extracted)"
    return headline


def _answer_question(question: str, page_text: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return ""
    text = (page_text or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokens = [tok for tok in re.findall(r"\w+", q) if len(tok) >= 4]
    if not tokens:
        return lines[0][:300] if lines else ""
    scored: list[tuple[int, str]] = []
    for line in lines[:80]:
        hay = line.lower()
        score = sum(1 for tok in tokens if tok in hay)
        if score:
            scored.append((score, line))
    if not scored:
        return lines[0][:300] if lines else ""
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return scored[0][1][:300]


def _page_state(page_text: str, dom_elements: list[dict]) -> str:
    hay = (page_text or "").lower()
    if "captcha" in hay or "robot" in hay:
        return "captcha"
    if "error" in hay or "invalid" in hay or "required" in hay:
        return "error"
    if dom_elements:
        return "interactive"
    return "normal" if hay else "unknown"
