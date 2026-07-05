"""Phase 1 invariants for web_search (analysis-only path).

Roadmap:
  D:\\AuraSDK-verify\\private\\roadmaps\\BRAIN_NATIVE_RETRIEVAL_ROADMAP_2026-04-13.md

These tests enforce:
  - web_search never writes durable knowledge (only short-lived search cache)
  - same-intent retry cap fires on the 4th identical call
  - honest_refusal is emitted with a stable schema
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from remy.core import brain_tools


@pytest.fixture(autouse=True)
def _reset_counter():
    brain_tools._reset_search_intent_counter()
    yield
    brain_tools._reset_search_intent_counter()


def _fake_ddgs_results():
    return [
        {"title": "t1", "href": "https://example.com/1", "body": "b1"},
        {"title": "t2", "href": "https://example.com/2", "body": "b2"},
    ]


def _call_web_search(query: str, session_id: str = "s1") -> dict:
    # Patch DDGS to avoid real network calls.
    with patch("ddgs.DDGS") as MockDDGS:
        MockDDGS.return_value.text.return_value = _fake_ddgs_results()
        # Also skip cache reads/writes so tests don't depend on brain state.
        with patch.object(brain_tools, "_get_cached_search", return_value=None), \
             patch.object(brain_tools, "_cache_search_result"):
            raw = brain_tools._execute_tool_locked(
                "web_search", {"query": query}, session_id=session_id,
            )
    return json.loads(raw)


def test_web_search_returns_candidate_discovery_on_first_call():
    r = _call_web_search("rust memory model 2025")
    assert r["mode"] == "candidate_discovery"
    assert r["candidate_count"] == 2
    assert r["sources"][0]["uri"] == "https://example.com/1"


def test_same_intent_retry_cap_fires_on_fourth_call():
    q = "some identical query"
    for i in range(brain_tools._SAME_INTENT_RETRY_CAP):
        r = _call_web_search(q)
        assert r["mode"] == "candidate_discovery", f"call {i+1} should still succeed"

    # 4th call: cap trips.
    r = _call_web_search(q)
    assert r["mode"] == "honest_refusal"
    assert r["reason"] == "same_intent_retry_cap"
    assert r["candidate_count"] == 0


def test_intent_normalization_treats_variants_as_same():
    # Different punctuation / whitespace / case, but same tokens.
    variants = [
        'foo bar baz',
        'FOO  Bar  baz',
        '"foo", bar; baz!',
    ]
    for v in variants:
        _call_web_search(v)
    # 4th variant should still trip the cap because all three above normalize
    # to the same intent.
    r = _call_web_search("baz foo bar")  # reordered tokens
    assert r["mode"] == "honest_refusal"


def test_different_session_ids_have_independent_counters():
    q = "same query across sessions"
    for _ in range(5):
        r = _call_web_search(q, session_id="session-A")
    assert r["mode"] == "honest_refusal"

    # Session B is fresh; must succeed.
    r2 = _call_web_search(q, session_id="session-B")
    assert r2["mode"] == "candidate_discovery"


def test_web_search_does_not_call_brain_store():
    # Patch brain.store to detect any durable write attempt.
    # The only permitted write path (search cache) is already patched out
    # above via _cache_search_result; if web_search tries ANY other store
    # call, this test will fail.
    with patch.object(brain_tools.brain, "store") as mock_store:
        _call_web_search("no-store check query")
        assert mock_store.call_count == 0, (
            f"web_search called brain.store {mock_store.call_count} times; "
            "analysis path must be write-free"
        )


def test_answer_string_lists_top_candidates_explicitly():
    r = _call_web_search("surface top candidates")
    assert "https://example.com/1" in r["answer"]
    assert "NEXT STEP" in r["answer"]
    assert "extract_content" in r["answer"]


def test_forward_progress_cap_fires_without_fetch():
    # Different queries each time so same-intent cap doesn't fire first.
    cap = brain_tools._FORWARD_PROGRESS_CAP
    for i in range(cap):
        r = _call_web_search(f"distinct query number {i}")
        assert r["mode"] == "candidate_discovery"

    # Next call (cap+1): forward-progress cap trips.
    r = _call_web_search("yet another distinct query")
    assert r["mode"] == "honest_refusal"
    assert r["reason"] == "forward_progress_cap"
    # The refusal must surface last candidates so the agent can still fetch.
    assert r["sources"], "forward-progress refusal must include last candidates"
    assert r["sources"][0]["uri"] == "https://example.com/1"


def test_forward_progress_cap_resets_after_fetch():
    cap = brain_tools._FORWARD_PROGRESS_CAP
    for i in range(cap):
        _call_web_search(f"pre-fetch query {i}")

    # Simulate a successful fetch — this is what extract_content / http_get would do.
    brain_tools._reset_web_search_no_fetch("s1")

    # Next call should succeed again, counter has been cleared.
    r = _call_web_search("post-fetch query")
    assert r["mode"] == "candidate_discovery"


def test_last_candidates_stashed_per_session():
    _call_web_search("stash test query", session_id="sess-X")
    stashed = brain_tools._get_last_candidates("sess-X")
    assert stashed and stashed[0]["uri"] == "https://example.com/1"
    # Different session = no stash.
    assert brain_tools._get_last_candidates("sess-Y") == []
