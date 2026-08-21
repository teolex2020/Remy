"""HTTP adapter for the server API shipped by Aura Memory.

The adapter intentionally implements only operations that the upstream HTTP
contract can represent without losing information.  In particular, the
current ``/process`` endpoint does not accept tags, metadata, namespaces, or a
specific Aura level.  Callers that request those features fail explicitly
instead of silently storing a weaker record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

_STORED_RECORD_RE = re.compile(r"^Stored record (?P<id>.+?) \(level=.*\)$")


class AuraServerError(RuntimeError):
    """Base error for a remote Aura MicroBrain."""


class AuraServerUnavailable(AuraServerError):
    """The configured Aura server could not be reached."""


class AuraServerAuthenticationError(AuraServerError):
    """The Aura server rejected its API key."""


class AuraServerContractError(AuraServerError):
    """The server response does not match the supported Aura HTTP contract."""


class AuraServerCapabilityError(AuraServerError):
    """The requested operation cannot be represented by Aura's server API."""


@dataclass(slots=True)
class AuraServerRecord:
    """Small record facade compatible with Remy's common memory consumers."""

    id: str
    content: str
    timestamp: float = 0.0
    intensity: float = 0.0
    dna: str = ""
    score: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    level: str = ""
    strength: float = 0.0
    activation_count: int = 0
    connections: dict[str, float] = field(default_factory=dict)
    importance: float | None = None
    namespace: str | None = None
    source_type: str | None = None

    @property
    def text(self) -> str:
        return self.content

    @property
    def created_at(self) -> float:
        return self.timestamp

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "text": self.content,
            "timestamp": self.timestamp,
            "created_at": self.timestamp,
            "intensity": self.intensity,
            "dna": self.dna,
            "score": self.score,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "level": self.level,
            "strength": self.strength,
            "activation_count": self.activation_count,
            "connections": dict(self.connections),
            "importance": self.importance,
            "namespace": self.namespace,
            "source_type": self.source_type,
        }


class AuraServerBrain:
    """Synchronous Remy facade over Aura Memory's official HTTP API v2."""

    provider_id = "aura-server"
    api_profile = "aura-http-v2-core"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        verify_on_open: bool = True,
    ):
        self.base_url = self._normalize_base_url(base_url)
        self._owns_client = client is None
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers
        self._client = client or httpx.Client(
            timeout=max(0.1, float(timeout)),
        )
        self._closed = False
        if verify_on_open:
            self.health()

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        raw = str(value or "").strip().rstrip("/")
        parts = urlsplit(raw)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError(
                "Aura server location must be a complete http:// or https:// URL"
            )
        if parts.username or parts.password:
            raise ValueError("Aura server credentials must not be embedded in its URL")
        if parts.query or parts.fragment:
            raise ValueError("Aura server URL must not contain a query or fragment")
        path = parts.path.rstrip("/")
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "core_memory": True,
            "health": True,
            "record_metadata": False,
            "record_tags": False,
            "record_levels": False,
            "namespaces": False,
            "cognitive_runtime": False,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise AuraServerError("Aura server MicroBrain is already closed")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_open()
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers,
                json=json_body,
                params=params,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise AuraServerUnavailable(
                f"Aura server is unavailable at {self.base_url}"
            ) from exc
        if allow_not_found and response.status_code == 404:
            return None
        if response.status_code == 401:
            raise AuraServerAuthenticationError(
                "Aura server rejected the configured API key"
            )
        if response.status_code == 429:
            raise AuraServerUnavailable(
                "Aura server rate limit was reached; retry this operation"
            )
        if response.status_code >= 400:
            raise AuraServerError(
                f"Aura server request failed with HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise AuraServerContractError(
                f"Aura server returned non-JSON data for {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise AuraServerContractError(
                f"Aura server returned an invalid object for {path}"
            )
        return payload

    @staticmethod
    def _record(payload: dict[str, Any]) -> AuraServerRecord:
        record_id = str(payload.get("id") or "")
        if not record_id:
            raise AuraServerContractError("Aura record has no id")
        content = str(payload.get("text") or payload.get("content") or "")
        intensity = float(payload.get("intensity") or 0.0)
        dna = str(payload.get("dna") or "")
        return AuraServerRecord(
            id=record_id,
            content=content,
            timestamp=float(payload.get("timestamp") or 0.0),
            intensity=intensity,
            dna=dna,
            score=(
                None
                if payload.get("score") is None
                else float(payload.get("score"))
            ),
            level=dna,
            strength=intensity,
            metadata={"aura_dna": dna, "memory_provider": "aura-server"},
        )

    @staticmethod
    def _reject_lossy_store(
        *,
        level: Any,
        tags: Any,
        metadata: Any,
        kwargs: dict[str, Any],
    ) -> None:
        unsupported = []
        if tags:
            unsupported.append("tags")
        if metadata:
            unsupported.append("metadata")
        if kwargs.get("namespace"):
            unsupported.append("namespace")
        if kwargs.get("source_type") or kwargs.get("semantic_type"):
            unsupported.append("semantic/source type")
        # Aura's HTTP API can express only normal or pinned storage.  A caller
        # may still pass Remy's default WORKING level, so reject only an
        # explicitly meaningful non-default value.
        level_name = str(getattr(level, "name", level) or "").lower()
        if level_name and level_name not in {"0", "working", "level.working"}:
            unsupported.append("specific memory level")
        if unsupported:
            raise AuraServerCapabilityError(
                "Aura server API v2 cannot safely store "
                + ", ".join(unsupported)
            )

    def health(self) -> dict[str, Any]:
        payload = self._request("GET", "/health")
        if payload is None or payload.get("status") != "ok":
            raise AuraServerContractError("Aura server health response is invalid")
        return payload

    def store(
        self,
        content: str,
        level: Any = None,
        tags: list[str] | None = None,
        deduplicate: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuraServerRecord:
        del deduplicate
        self._reject_lossy_store(
            level=level,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )
        pin = bool(kwargs.get("pin", False))
        payload = self._request(
            "POST",
            "/process",
            json_body={"text": str(content), "pin": pin},
        )
        status = str((payload or {}).get("status") or "")
        match = _STORED_RECORD_RE.match(status)
        if match is None:
            raise AuraServerContractError(
                "Aura server stored data but did not return a record id"
            )
        record_id = match.group("id")
        return AuraServerRecord(
            id=record_id,
            content=str(content),
            metadata={"memory_provider": "aura-server"},
        )

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[AuraServerRecord]:
        if tags or any(
            kwargs.get(name) is not None
            for name in ("level", "content_type", "source_type", "namespace")
        ):
            raise AuraServerCapabilityError(
                "Aura server API v2 cannot filter by tags, level, type, or namespace"
            )
        bounded_limit = max(1, min(int(limit), 5000))
        if not str(query or "").strip():
            payload = self._request(
                "GET",
                "/memories",
                params={"offset": 0, "limit": bounded_limit, "dna": "all"},
            )
            values = (payload or {}).get("memories")
        else:
            payload = self._request(
                "POST",
                "/retrieve",
                json_body={"query": str(query), "top_k": bounded_limit},
            )
            values = (payload or {}).get("results")
        if not isinstance(values, list):
            raise AuraServerContractError("Aura server returned an invalid record list")
        return [self._record(item) for item in values if isinstance(item, dict)]

    def list_records(
        self,
        tags: list[str] | None = None,
        min_strength: float = 0.0,
        limit: int = 5000,
    ) -> list[AuraServerRecord]:
        records = self.search(query="", tags=tags, limit=limit)
        if min_strength > 0:
            records = [
                record for record in records if record.strength >= min_strength
            ]
        return records

    @property
    def records(self) -> dict[str, AuraServerRecord]:
        return {record.id: record for record in self.list_records(limit=5000)}

    def get(self, record_id: str) -> AuraServerRecord | None:
        # Aura server v2 has no GET-by-id route.  Paginate deterministically
        # instead of using semantic retrieval, which can return a wrong record.
        offset = 0
        page_size = 250
        while True:
            payload = self._request(
                "GET",
                "/memories",
                params={"offset": offset, "limit": page_size, "dna": "all"},
            )
            values = (payload or {}).get("memories")
            total = int((payload or {}).get("total") or 0)
            if not isinstance(values, list):
                raise AuraServerContractError(
                    "Aura server returned an invalid record page"
                )
            for item in values:
                if isinstance(item, dict) and str(item.get("id") or "") == record_id:
                    return self._record(item)
            offset += len(values)
            if not values or offset >= total:
                return None

    def update(
        self,
        record_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        if content is None:
            raise AuraServerCapabilityError(
                "Aura server API v2 cannot update metadata without replacing content"
            )
        self._reject_lossy_store(
            level=kwargs.get("level"),
            tags=kwargs.get("tags"),
            metadata=metadata,
            kwargs=kwargs,
        )
        payload = self._request(
            "POST",
            "/update",
            json_body={"id": str(record_id), "text": str(content)},
        )
        return str((payload or {}).get("status") or "").lower() == "updated"

    def delete(self, record_id: str) -> bool:
        payload = self._request(
            "POST",
            "/delete",
            json_body={"id": str(record_id)},
            allow_not_found=True,
        )
        return bool(payload and payload.get("success"))

    def count(self, *_args: Any, **_kwargs: Any) -> int:
        payload = self._request("GET", "/stats")
        try:
            return int((payload or {})["total_memories"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuraServerContractError(
                "Aura server stats response has no total_memories"
            ) from exc

    def stats(self) -> dict[str, Any]:
        return dict(self._request("GET", "/stats") or {})

    def recall_structured(
        self,
        query: str,
        top_k: int = 15,
        min_strength: float | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        del session_id
        records = self.search(query=query, limit=top_k)
        if min_strength is not None:
            records = [
                item for item in records if item.strength >= float(min_strength)
            ]
        return [item.to_dict() for item in records]

    def recall_full(
        self,
        query: str,
        top_k: int = 20,
        include_failures: bool = True,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        del include_failures
        return self.recall_structured(query, top_k=top_k)

    def explain_recall(
        self,
        query: str,
        top_k: int = 10,
        min_strength: float | None = None,
        expand_connections: bool | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query, "top_k": top_k}
        if min_strength is not None:
            params["min_strength"] = min_strength
        if expand_connections is not None:
            params["expand_connections"] = str(bool(expand_connections)).lower()
        if namespace:
            params["namespaces"] = namespace
        payload = self._request("GET", "/explain-recall", params=params)
        explanation = (payload or {}).get("explanation")
        if not isinstance(explanation, dict):
            raise AuraServerContractError(
                "Aura server returned an invalid recall explanation"
            )
        return explanation

    def memory_health_digest(self, limit: int = 10) -> dict[str, Any]:
        payload = self._request(
            "GET",
            "/memory-health",
            params={"limit": max(1, int(limit))},
        )
        digest = (payload or {}).get("digest")
        if not isinstance(digest, dict):
            raise AuraServerContractError(
                "Aura server returned an invalid memory health digest"
            )
        return digest

    def run_maintenance(self) -> None:
        raise AuraServerCapabilityError(
            "Aura server API v2 does not expose a maintenance command"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self._client.close()

    def __repr__(self) -> str:
        return (
            f"AuraServerBrain(base_url={self.base_url!r}, "
            f"profile={self.api_profile!r})"
        )
