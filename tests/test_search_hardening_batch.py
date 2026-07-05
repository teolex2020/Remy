
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from remy.core.agent import AgentState, call_tools, _choose_best_candidate_source
from remy.core.claim_provenance import clear_turn_fetch_evidence, get_turn_fetch_evidence, record_turn_fetch_evidence
from remy.core.external_claim_verifier import verify_external_claims
from remy.core import brain_tools as brain_tools_mod
from remy.core.hybrid_search import search_exact_structured
from remy.core.langgraph_tools import build_langchain_tools, invalidate_tool_cache
from remy.core.tool_handlers import research as research_mod


class FakeTool:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def invoke(self, args):
        return self._fn(args)


class DummyBrain:
    def __init__(self, project_rec=None, finding_map=None, store_result=None):
        self.project_rec = project_rec
        self.finding_map = finding_map or {}
        self.store_result = store_result or SimpleNamespace(id="stored-rec", metadata={})
        self.connected = []
        self.updated = []
        self.stored = []

    def store(self, content, level, tags, metadata):
        self.stored.append({"content": content, "level": level, "tags": tags, "metadata": metadata})
        return self.store_result

    def connect(self, left, right, weight=0.0):
        self.connected.append((left, right, weight))

    def update(self, record_id, metadata=None):
        self.updated.append((record_id, metadata or {}))

    def get(self, record_id):
        if self.project_rec and getattr(self.project_rec, "id", None) == record_id:
            return self.project_rec
        if self.store_result and getattr(self.store_result, "id", None) == record_id:
            return self.store_result
        return self.finding_map.get(record_id)


class DummyVerification:
    verified = True
    repair_required = False
    reason = ""

    def to_dict(self):
        return {"verified": True, "repair_required": False, "reason": ""}


def test_choose_best_candidate_source_prefers_trusted_domain():
    selected = _choose_best_candidate_source([
        {"title": "Mirror copy", "uri": "https://mirror.example.com/paper"},
        {"title": "Paper", "uri": "https://arxiv.org/abs/2411.02534"},
    ])
    assert selected is not None
    assert selected["uri"] == "https://arxiv.org/abs/2411.02534"
    assert selected["trust_score"] > 0


def test_call_tools_auto_follow_uses_trusted_source_and_logs_full_payload():
    ai_msg = AIMessage(content="", tool_calls=[{"id": "call_search", "name": "web_search", "args": {"query": "epistemic integrity"}}])
    state = AgentState(messages=[ai_msg], session_id="sess-auto-follow", channel="desktop", session_log=[])
    web_search_tool = FakeTool("web_search", lambda args: json.dumps({
        "mode": "candidate_discovery",
        "answer": "Found candidates.",
        "sources": [
            {"title": "Mirror page", "uri": "https://mirror.example.com/paper"},
            {"title": "Real paper", "uri": "https://arxiv.org/abs/2411.02534"},
        ],
    }))
    extract_tool = FakeTool("extract_content", lambda args: json.dumps({
        "url": args["url"],
        "title": "Real paper title",
        "site": "arXiv",
        "content": "Fetched evidence body",
    }))
    clear_turn_fetch_evidence("sess-auto-follow")
    with patch("remy.core.agent.get_all_tools", return_value=[web_search_tool, extract_tool]):
        result = call_tools(state)
    payload = json.loads(result["messages"][0].content)
    assert payload["selected_source"]["uri"] == "https://arxiv.org/abs/2411.02534"
    assert payload["auto_extract"]["url"] == "https://arxiv.org/abs/2411.02534"
    assert any(entry["tool"] == "extract_content" for entry in result["session_log"])
    assert result["session_log"][0]["args_full"] == {"query": "epistemic integrity"}
    fetched = get_turn_fetch_evidence("sess-auto-follow")
    assert any(item["url"] == "https://arxiv.org/abs/2411.02534" for item in fetched)


def test_call_tools_auto_follow_failure_does_not_break_turn():
    ai_msg = AIMessage(content="", tool_calls=[{"id": "call_search_fail", "name": "web_search", "args": {"query": "epistemic integrity"}}])
    state = AgentState(messages=[ai_msg], session_id="sess-auto-fail", channel="desktop", session_log=[])
    web_search_tool = FakeTool("web_search", lambda args: json.dumps({
        "mode": "candidate_discovery",
        "answer": "Found candidates.",
        "sources": [{"title": "Real paper", "uri": "https://arxiv.org/abs/2411.02534"}],
    }))
    extract_tool = FakeTool("extract_content", lambda args: (_ for _ in ()).throw(RuntimeError("fetch failed")))
    with patch("remy.core.agent.get_all_tools", return_value=[web_search_tool, extract_tool]):
        result = call_tools(state)
    payload = json.loads(result["messages"][0].content)
    assert payload["mode"] == "candidate_discovery"
    assert payload["auto_extract"]["error"] == "fetch failed"


def test_verify_external_claims_flags_reference_identity_mismatch():
    response_text = '"Epistemic Integrity in Large Language Models" (2024) arXiv:2411.02534'
    session_log = [{
        "type": "tool_call",
        "tool": "extract_content",
        "result_full": json.dumps({
            "url": "https://arxiv.org/abs/2411.02534",
            "title": "Real Different Paper Title",
            "author": "Actual Author",
            "content": "abstract text",
        }),
    }]
    report = verify_external_claims(response_text, session_log)
    assert report.reference_identity_mismatch_count >= 1
    assert report.grounded_count == 0


def test_add_research_finding_rejects_unanchored_source_url():
    # Arrange a project so we pass the project-lookup guard and actually hit
    # the fetch-anchor check in ingest_grounded_evidence.
    project_rec = SimpleNamespace(id="project-rec", metadata={"topic": "research", "finding_ids": []})
    brain = DummyBrain(project_rec=project_rec)
    clear_turn_fetch_evidence("sess-research-miss")
    with patch.object(research_mod, "_get_brain", return_value=brain), \
         patch.object(research_mod, "_get_brain_lock", return_value=nullcontext()), \
         patch.object(research_mod, "_get_research_project", return_value=project_rec), \
         patch("remy.core.tool_utils._check_duplicates", return_value=[]):
        result = json.loads(research_mod._add_research_finding({
            "project_id": "proj-1",
            "content": "Finding text",
            "source_url": "https://arxiv.org/abs/2411.02534",
        }, session_id="sess-research-miss"))
    # Current canonical wording from ingest_grounded_evidence.
    assert "not anchored by a fetch this turn" in result["error"]


def test_add_research_finding_accepts_anchored_source_url():
    clear_turn_fetch_evidence("sess-research-ok")
    record_turn_fetch_evidence("sess-research-ok", tool="extract_content", url="https://arxiv.org/abs/2411.02534", title="Real Paper", site="arXiv")
    project_rec = SimpleNamespace(id="project-rec", metadata={"topic": "research", "finding_ids": []})
    stored_rec = SimpleNamespace(id="finding-rec", metadata={})
    brain = DummyBrain(project_rec=project_rec, store_result=stored_rec)
    with patch.object(research_mod, "_get_brain", return_value=brain), patch.object(research_mod, "_get_brain_lock", return_value=nullcontext()), patch.object(research_mod, "_get_research_project", return_value=project_rec), patch("remy.core.tool_utils._check_duplicates", return_value=[]):
        result = json.loads(research_mod._add_research_finding({
            "project_id": "proj-1",
            "content": "Finding text",
            "source_url": "https://arxiv.org/abs/2411.02534",
        }, session_id="sess-research-ok"))
    assert result["stored"] is True
    assert brain.stored[0]["metadata"]["source_anchored"] is True


def test_complete_research_rejects_unanchored_findings():
    project_rec = SimpleNamespace(id="project-rec", metadata={"topic": "research", "finding_ids": ["f1"]})
    finding_rec = SimpleNamespace(id="f1", content="Finding one", metadata={"source_url": "https://example.com/source", "source_anchored": False, "confidence": 0.8})
    brain = DummyBrain(project_rec=project_rec, finding_map={"f1": finding_rec})
    with patch.object(research_mod, "_get_brain", return_value=brain), patch.object(research_mod, "_get_brain_lock", return_value=nullcontext()), patch.object(research_mod, "_get_research_project", return_value=project_rec):
        result = json.loads(research_mod._complete_research({"project_id": "proj-1"}, session_id="sess-complete-bad"))
    assert "anchored source_url" in result["error"]


def test_complete_research_succeeds_with_anchored_findings():
    project_rec = SimpleNamespace(id="project-rec", metadata={"topic": "research", "finding_ids": ["f1"]})
    finding_rec = SimpleNamespace(id="f1", content="Finding one", metadata={"source_url": "https://arxiv.org/abs/2411.02534", "source_anchored": True, "confidence": 0.9})
    report_rec = SimpleNamespace(id="report-rec", metadata={})
    brain = DummyBrain(project_rec=project_rec, finding_map={"f1": finding_rec}, store_result=report_rec)
    with patch.object(research_mod, "_get_brain", return_value=brain), patch.object(research_mod, "_get_brain_lock", return_value=nullcontext()), patch.object(research_mod, "_get_research_project", return_value=project_rec), patch("remy.core.llm.call_llm", return_value=SimpleNamespace(content="Anchored report")), patch("remy.core.brain_tools._generate_report", return_value=json.dumps({"generated": False})), patch("remy.core.verification_gate.run_research_completion_verification_gate", return_value=DummyVerification()), patch("remy.core.verification_gate.emit_verification_incident"), patch("remy.core.verification_gate.resolve_verification_incident"):
        result = json.loads(research_mod._complete_research({"project_id": "proj-1"}, session_id="sess-complete-ok"))
    assert result["completed"] is True
    assert result["source_count"] == 1


def test_extract_content_is_present_in_langgraph_tools():
    invalidate_tool_cache()
    tool_names = {tool.name for tool in build_langchain_tools()}
    assert "extract_content" in tool_names
    assert "extract_content" in brain_tools_mod.CORE_TOOL_NAMES


def test_get_cached_search_normalizes_legacy_summary_payload():
    # The cache is backend-pinned (currently "ddgs-v3-pinned"). Records without
    # that pin are rejected wholesale by design so stale-backend answers don't
    # leak in. This test verifies legacy *payload shape* normalization on a
    # properly-pinned record: even with a summary-style "answer", the cache
    # returns the canonical candidate-discovery wrapper, not the summary text.
    rec = SimpleNamespace(metadata={
        "cached_at": "9999-12-31T23:59:59",
        "query": "epistemic governance",
        "backend": brain_tools_mod._SEARCH_CACHE_BACKEND,
        "answer": "Research on epistemic governance reveals several architectures.",
        "sources": [{"title": "Paper", "uri": "https://arxiv.org/abs/2411.02534"}],
    }, content="legacy cache")

    with patch.object(brain_tools_mod, "brain_lock", nullcontext()), patch.object(brain_tools_mod, "brain", SimpleNamespace(search=lambda **kwargs: [rec])):
        cached = brain_tools_mod._get_cached_search("epistemic governance")

    assert cached is not None
    assert cached["mode"] == "candidate_discovery"
    assert cached["candidate_count"] == 1
    assert "These are discovery candidates" in cached["answer"]
    assert "Research on epistemic governance" not in cached["answer"]


def test_search_exact_structured_excludes_web_search_cache_records():
    cache_rec = SimpleNamespace(
        id="cache-1",
        content="Web search: epistemic governance",
        tags=["web-search-cache"],
        level="DOMAIN",
        strength=0.1,
        activation_count=1,
        metadata={},
        importance=None,
    )
    good_rec = SimpleNamespace(
        id="real-1",
        content="Verified user memory",
        tags=["preference"],
        level="DOMAIN",
        strength=0.8,
        activation_count=5,
        metadata={},
        importance=None,
    )
    brain = SimpleNamespace(search=lambda **kwargs: [cache_rec, good_rec])

    results = search_exact_structured(brain, "epistemic governance", top_k=5, lexical_limit=5)

    assert [item["id"] for item in results] == ["real-1"]


def test_choose_best_candidate_source_respects_site_constraint():
    selected = _choose_best_candidate_source([
        {"title": "ResearchGate mirror", "uri": "https://www.researchgate.net/publication/123"},
        {"title": "Actual arXiv paper", "uri": "https://arxiv.org/abs/2411.02534"},
    ], query='site:arxiv.org "epistemic governance"')
    assert selected is not None
    assert selected["uri"] == "https://arxiv.org/abs/2411.02534"


def test_call_tools_filters_candidates_that_violate_site_constraint():
    ai_msg = AIMessage(content="", tool_calls=[{"id": "call_search_constraint", "name": "web_search", "args": {"query": 'site:arxiv.org "epistemic governance"'}}])
    state = AgentState(messages=[ai_msg], session_id="sess-site-constraint", channel="desktop", session_log=[])
    web_search_tool = FakeTool("web_search", lambda args: json.dumps({
        "mode": "candidate_discovery",
        "answer": "Found candidates.",
        "sources": [
            {"title": "ResearchGate mirror", "uri": "https://www.researchgate.net/publication/123"},
            {"title": "Blog post", "uri": "https://medium.com/example/post"},
        ],
    }))
    extract_tool = FakeTool("extract_content", lambda args: json.dumps({"url": args["url"], "title": "Should not run"}))
    with patch("remy.core.agent.get_all_tools", return_value=[web_search_tool, extract_tool]):
        result = call_tools(state)
    payload = json.loads(result["messages"][0].content)
    assert payload["candidate_count"] == 0
    assert payload["filtered_candidate_count"] == 0
    assert payload["raw_candidate_count"] == 2
    assert payload.get("auto_extract") is None
    assert "matched the query constraints" in payload["answer"]
