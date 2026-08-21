from __future__ import annotations

import json

import httpx
import pytest

from remy.core.aura_server import (
    AuraServerAuthenticationError,
    AuraServerBrain,
    AuraServerCapabilityError,
    AuraServerContractError,
)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_aura_server_core_memory_contract_and_bearer_auth():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer secret"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/process":
            body = json.loads(request.content)
            assert body == {"text": "remember this", "pin": False}
            return httpx.Response(
                200,
                json={"status": "Stored record rec-1 (level=Working)"},
            )
        if request.url.path == "/retrieve":
            body = json.loads(request.content)
            assert body == {"query": "remember", "top_k": 4}
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "rec-1",
                            "text": "remember this",
                            "timestamp": 12.5,
                            "intensity": 0.8,
                            "dna": "general",
                            "score": 0.91,
                        }
                    ]
                },
            )
        if request.url.path == "/stats":
            return httpx.Response(
                200,
                json={
                    "total_memories": 1,
                    "license": "Unlocked",
                    "version": "v2.0",
                    "phantom_count": 0,
                },
            )
        raise AssertionError(request.url)

    brain = AuraServerBrain(
        "https://memory.example.test/",
        api_key="secret",
        client=_client(handler),
    )

    stored = brain.store("remember this")
    recalled = brain.search("remember", limit=4)

    assert stored.id == "rec-1"
    assert recalled[0].content == "remember this"
    assert recalled[0].score == pytest.approx(0.91)
    assert recalled[0].metadata["memory_provider"] == "aura-server"
    assert brain.count() == 1
    assert brain.capabilities["core_memory"] is True
    assert brain.capabilities["record_metadata"] is False
    assert len(requests) == 4


def test_aura_server_lists_gets_updates_and_deletes_records():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/memories":
            return httpx.Response(
                200,
                json={
                    "memories": [
                        {
                            "id": "rec-2",
                            "text": "two",
                            "timestamp": 2,
                            "intensity": 0.5,
                            "dna": "general",
                        }
                    ],
                    "total": 1,
                },
            )
        if request.url.path == "/update":
            assert json.loads(request.content) == {"id": "rec-2", "text": "updated"}
            return httpx.Response(200, json={"status": "Updated"})
        if request.url.path == "/delete":
            assert json.loads(request.content) == {"id": "rec-2"}
            return httpx.Response(200, json={"success": True})
        raise AssertionError(request.url)

    brain = AuraServerBrain(
        "http://127.0.0.1:9898",
        client=_client(handler),
        verify_on_open=False,
    )

    assert [record.id for record in brain.list_records()] == ["rec-2"]
    assert brain.get("rec-2").content == "two"
    assert brain.update("rec-2", content="updated") is True
    assert brain.delete("rec-2") is True


@pytest.mark.parametrize(
    ("kwargs", "feature"),
    [
        ({"tags": ["private"]}, "tags"),
        ({"metadata": {"source": "user"}}, "metadata"),
        ({"namespace": "project"}, "namespace"),
        ({"semantic_type": "fact"}, "semantic/source type"),
        ({"level": "Identity"}, "specific memory level"),
    ],
)
def test_aura_server_rejects_lossy_store_features(kwargs, feature):
    brain = AuraServerBrain(
        "https://memory.example.test",
        client=_client(lambda _request: httpx.Response(500)),
        verify_on_open=False,
    )

    with pytest.raises(AuraServerCapabilityError, match=feature):
        brain.store("content", **kwargs)


def test_aura_server_distinguishes_auth_and_invalid_contract():
    unauthorized = AuraServerBrain(
        "https://memory.example.test",
        client=_client(lambda _request: httpx.Response(401)),
        verify_on_open=False,
    )
    with pytest.raises(AuraServerAuthenticationError, match="API key"):
        unauthorized.health()

    invalid = AuraServerBrain(
        "https://memory.example.test",
        client=_client(lambda _request: httpx.Response(200, text="not-json")),
        verify_on_open=False,
    )
    with pytest.raises(AuraServerContractError, match="non-JSON"):
        invalid.health()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "memory.example.test",
        "ftp://memory.example.test",
        "https://user:pass@memory.example.test",
        "https://memory.example.test?token=secret",
    ],
)
def test_aura_server_rejects_unsafe_or_incomplete_urls(url):
    with pytest.raises(ValueError):
        AuraServerBrain(url, verify_on_open=False)
