"""Tests for the research worker wrapper."""

from unittest.mock import AsyncMock, patch

import pytest

from remy.core.workers.contracts import WorkerExecutionResult
from remy.core.workers.reporter import format_worker_report
from remy.core.workers.research_worker import (
    RESEARCH_TOOL_WHITELIST,
    _apply_marginal_evidence_controls,
    _claim_matrix_allows_completion,
    _contradiction_blocks_completion,
    _resolve_lifecycle_project_id,
    _build_research_query_plan,
    _derive_research_status,
    _extract_completion_threshold,
    _extract_research_evidence,
    _plan_research_queries,
    _rerank_sources,
    _store_research_source_consequences,
    build_research_worker_prompt,
)

# ============== Prompt building ==============


class TestBuildResearchWorkerPrompt:
    def test_lifecycle_scope_prefers_workspace_project_over_research_project(self):
        with patch("remy.core.microbrain.current_project_id", return_value="workspace-1"):
            assert _resolve_lifecycle_project_id(
                {"project_id": "transient-research-42"}
            ) == "workspace-1"

    def test_lifecycle_scope_accepts_explicit_owner_boundary(self):
        assert _resolve_lifecycle_project_id(
            {"owner_project_id": "workspace-2", "project_id": "research-7"}
        ) == "workspace-2"

    def test_includes_task_description(self):
        prompt = build_research_worker_prompt({"description": "Analyze competitor pricing"})
        assert "Analyze competitor pricing" in prompt

    def test_includes_worker_identity(self):
        prompt = build_research_worker_prompt({"description": "test"})
        assert "RESEARCH_WORKER" in prompt

    def test_includes_rules(self):
        prompt = build_research_worker_prompt({"description": "test"})
        assert "recall first" in prompt
        assert "Do NOT use browse_page" in prompt

    def test_includes_task_action(self):
        prompt = build_research_worker_prompt(
            {
                "description": "Research X",
                "task_action": "Search for competitor pricing data",
                "task_done_when": "3+ competitors found with pricing",
            }
        )
        assert "Search for competitor pricing data" in prompt
        assert "3+ competitors found" in prompt

    def test_includes_resume_context(self):
        prompt = build_research_worker_prompt(
            {
                "description": "Research X",
                "resume_context": "Found 2 competitors so far",
                "blocked_reason": "Rate limited by search API",
            }
        )
        assert "Found 2 competitors" in prompt
        assert "Rate limited" in prompt

    def test_includes_existing_knowledge(self):
        prompt = build_research_worker_prompt(
            {"description": "test"},
            existing_knowledge="Previous research found 5 competitors in the AI memory space.",
        )
        assert "5 competitors" in prompt

    def test_default_template(self):
        prompt = build_research_worker_prompt({"description": "test"})
        assert "market_research" in prompt

    def test_includes_cycle_budget_rules(self):
        prompt = build_research_worker_prompt(
            {"description": "Research AI memory tools", "goal_template": "market_research"}
        )
        assert "Max queries this cycle" in prompt
        assert "partial findings" in prompt


class TestResearchQueryPlanning:
    def test_balanced_plan_has_primary_corroboration_and_counterevidence(self):
        plan = _build_research_query_plan(
            {"description": "Research AI agent memory architecture"},
            {"research_mode": "balanced", "source_scope": "web"},
        )
        assert [query["intent"] for query in plan["queries"]] == [
            "primary",
            "corroboration",
            "counterevidence",
        ]

    def test_prefers_seed_queries_for_influencer_goals(self):
        queries = _plan_research_queries(
            {
                "description": (
                    "Market research and competitive analysis. "
                    "Search queries: 'twitter AI agent memory influencer 2025 2026', "
                    "'site:x.com agent memory'. Minimum 5 real influencers."
                )
            },
            {"research_mode": "balanced", "source_scope": "web"},
        )
        assert queries[0] == "twitter AI agent memory influencer 2025 2026"
        assert "site:x.com agent memory" in queries
        assert all("Minimum 5 real influencers" not in q for q in queries)
        assert len(queries) <= 3

    def test_compact_query_generation_for_non_seed_influencer_goal(self):
        queries = _plan_research_queries(
            {
                "description": "Find Twitter influencers who discuss AI agent memory and cognitive architecture"
            },
            {"research_mode": "balanced", "source_scope": "web"},
        )
        assert any('site:x.com "AI agent memory" influencer' in q for q in queries)
        assert all(len(q) < 120 for q in queries)


class TestResearchProvenanceIntegration:
    def test_shared_origin_reopens_superficially_complete_schedule(self):
        origin = "https://origin.example/database-study"
        distinct_commentary = [
            "quorum majority node failure replicated state machine safety proof",
            "serializable transactions write skew snapshot isolation anomaly audit",
            "leader election network partition replicated log production metrics",
        ]
        source_records = [
            {
                "url": f"https://publisher{index}.example/report",
                "original_url": origin,
                "content": (
                    f"Database consistency {distinct_commentary[index - 1]}. " * 18
                ),
            }
            for index in range(1, 4)
        ]
        schedule, _, marginal, graph = _apply_marginal_evidence_controls(
            "database consistency",
            {"source_records": source_records},
            {"sufficient": True, "repair_queries": [], "next_actions": []},
        )

        assert graph["independent_root_count"] == 1
        assert marginal["decision"] == "continue_provenance_diversify"
        assert schedule["sufficient"] is False
        assert (
            "marginal:not_enough_independent_evidence_roots" in schedule["reasons"]
        )


class TestCompletionThreshold:
    def test_extracts_top_count_threshold(self):
        goal = {"description": "Find top 10 Twitter/X influencers who talk about AI agent memory"}
        assert _extract_completion_threshold(goal) == 10

    def test_extracts_minimum_threshold(self):
        goal = {"task_done_when": "Store at least 5 validated competitors with URLs"}
        assert _extract_completion_threshold(goal) == 5

    def test_claim_matrix_blocks_citation_bound_completion(self):
        matrix = {"publication_ready": False, "unsupported_claims": 1}
        assert not _claim_matrix_allows_completion(matrix, citation_required=True)

    def test_claim_matrix_allows_verified_completion(self):
        matrix = {"publication_ready": True, "unsupported_claims": 0}
        assert _claim_matrix_allows_completion(matrix, citation_required=True)

    def test_non_citation_research_does_not_require_matrix(self):
        assert _claim_matrix_allows_completion({}, citation_required=False)

    def test_resolved_temporal_conflict_does_not_block_completion(self):
        assert not _contradiction_blocks_completion(
            {
                "status": "resolved",
                "resolution_status": "resolved_by_temporal_supersession",
            }
        )
        assert _contradiction_blocks_completion({"status": "unresolved"})


# ============== Evidence extraction ==============


class TestExtractResearchEvidence:
    def test_extracts_queries(self):
        log = [
            {"type": "tool_call", "tool": "web_search", "args": {"query": "AI memory competitors"}},
            {"type": "tool_call", "tool": "web_search", "args": {"query": "agent memory pricing"}},
        ]
        evidence = _extract_research_evidence(log, "")
        assert evidence["queries"] == ["AI memory competitors", "agent memory pricing"]

    def test_extracts_sources(self):
        log = [
            {
                "type": "tool_call",
                "tool": "add_research_finding",
                "args": {"source_url": "https://example.com/a"},
            },
            {
                "type": "tool_call",
                "tool": "extract_content",
                "args": {"url": "https://example.com/b"},
            },
        ]
        evidence = _extract_research_evidence(log, "")
        assert "https://example.com/a" in evidence["sources"]
        assert "https://example.com/b" in evidence["sources"]

    def test_deduplicates_sources(self):
        log = [
            {
                "type": "tool_call",
                "tool": "extract_content",
                "args": {"url": "https://example.com/same"},
            },
            {
                "type": "tool_call",
                "tool": "add_research_finding",
                "args": {"source_url": "https://example.com/same"},
            },
        ]
        evidence = _extract_research_evidence(log, "")
        assert evidence["sources"].count("https://example.com/same") == 1

    def test_extract_content_preserves_temporal_and_origin_metadata(self):
        log = [
            {
                "type": "tool_call",
                "tool": "extract_content",
                "args": {"url": "https://example.com/article"},
                "result": {
                    "url": "https://example.com/article",
                    "content": "Current price evidence with enough readable material. " * 4,
                    "published_at": "2026-08-21T09:00:00Z",
                    "original_url": "https://origin.example/report",
                    "metadata": {"dateModified": "2026-08-22"},
                },
            }
        ]

        evidence = _extract_research_evidence(log, "")

        record = evidence["source_records"][0]
        assert record["published_at"] == "2026-08-21T09:00:00Z"
        assert record["original_url"] == "https://origin.example/report"
        assert record["metadata"]["dateModified"] == "2026-08-22"

    def test_counts_findings(self):
        log = [
            {"type": "tool_call", "tool": "add_research_finding", "args": {"source_url": "a"}},
            {"type": "tool_call", "tool": "add_research_finding", "args": {"source_url": "b"}},
            {"type": "tool_call", "tool": "add_research_finding", "args": {"source_url": "c"}},
        ]
        evidence = _extract_research_evidence(log, "")
        assert evidence["findings_count"] == 3

    def test_store_counts_as_partial_finding(self):
        log = [
            {
                "type": "tool_call",
                "tool": "store",
                "args": {
                    "tags": "influencer-research,twitter-ai-memory",
                    "content": "Jerry Liu (@jerryjliu0) recent post https://x.com/jerryjliu0/status/123",
                },
            }
        ]
        evidence = _extract_research_evidence(log, "")
        assert evidence["findings_count"] == 1
        assert "https://x.com/jerryjliu0/status/123" in evidence["sources"]
        assert evidence["findings"][0]["summary"].startswith("Jerry Liu")

    def test_store_infers_x_profile_url_from_handle(self):
        log = [
            {
                "type": "tool_call",
                "tool": "store",
                "args": {
                    "tags": "influencer-research,twitter-ai-memory",
                    "content": "Harrison Chase (@hwchase17) discusses agent memory and LangGraph.",
                },
            }
        ]
        evidence = _extract_research_evidence(log, "")
        assert "https://x.com/hwchase17" in evidence["sources"]
        assert evidence["findings"][0]["source_url"] == "https://x.com/hwchase17"

    def test_empty_log(self):
        evidence = _extract_research_evidence([], "")
        assert evidence["queries"] == []
        assert evidence["sources"] == []
        assert evidence["findings_count"] == 0


class TestResearchSourceMemory:
    def test_rerank_sources_uses_consequence_memory(self, monkeypatch):
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def policy_hint(self, situation, action, namespace=None):
                if action == "source_host:bad.example":
                    return {
                        "hint": "avoid",
                        "verdict": "refutes",
                        "supports": 0,
                        "refutes": 2,
                    }
                return {"hint": "verify_first", "supports": 0, "refutes": 0}

        monkeypatch.setattr(memory_api, "get_memory", lambda: StubMemory())
        accepted, rejected = _rerank_sources(
            {"description": "neutral source"},
            [
                {"url": "https://bad.example/a", "title": "Neutral source", "snippet": ""},
                {"url": "https://good.example/a", "title": "Neutral source", "snippet": ""},
            ],
            {"research_mode": "speed", "source_scope": "web"},
        )

        assert accepted[0]["url"] == "https://good.example/a"
        assert any(item["url"] == "https://bad.example/a" for item in accepted + rejected)

    def test_store_research_source_consequences(self, monkeypatch):
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.units = []

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return f"unit-{len(self.units)}"

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)
        _store_research_source_consequences(
            evidence={
                "accepted_sources": [
                    {"url": "https://arxiv.org/abs/2411.02534", "title": "Paper"}
                ],
                "accepted_sources_count": 1,
                "citation_coverage_rate": 1.0,
            },
            status="completed",
            session_id="sess-research",
        )

        actions = {unit["action"]: unit for unit in memory.units}
        assert actions["source_class:research"]["namespace"] == "remy-sources"
        assert actions["source_class:research"]["consequence"] == "SUPPORTS"
        assert actions["source_host:arxiv.org"]["consequence"] == "SUPPORTS"
        assert actions["source_tool:research_worker"]["consequence"] == "SUPPORTS"
        assert "research-worker" in actions["source_class:research"]["scope"]


# ============== Status derivation ==============


class TestDeriveResearchStatus:
    def test_no_action(self):
        assert _derive_research_status([], "") == "no_action"

    def test_completed(self):
        log = [{"type": "tool_call", "tool": "complete_research"}]
        assert _derive_research_status(log, "") == "completed"

    def test_findings_collected(self):
        log = [
            {"type": "tool_call", "tool": "web_search"},
            {"type": "tool_call", "tool": "add_research_finding"},
        ]
        assert _derive_research_status(log, "") == "findings_collected"

    def test_searching(self):
        log = [{"type": "tool_call", "tool": "web_search"}]
        assert _derive_research_status(log, "") == "searching"

    def test_store_counts_as_findings(self):
        log = [{"type": "tool_call", "tool": "store"}]
        assert _derive_research_status(log, "") == "findings_collected"

    def test_attempted_for_other_tools(self):
        log = [{"type": "tool_call", "tool": "recall"}]
        assert _derive_research_status(log, "") == "attempted"


# ============== Reporter: research format ==============


class TestReporterResearchFormat:
    def test_formats_research_result(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="findings_collected",
            response_text="Found 3 competitors: Mem0, Zep, LangMem.",
            evidence={
                "queries": ["AI agent memory competitors"],
                "sources": ["https://mem0.ai", "https://zep.ai"],
                "findings_count": 3,
                "project_id": "rp-abc123",
            },
        )
        text = format_worker_report(result)
        assert "Status: findings_collected" in text
        assert "Findings: 3" in text
        assert "mem0.ai" in text
        assert "rp-abc123" in text

    def test_completed_next_step(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="completed",
            response_text="Research complete.",
            evidence={"findings_count": 5},
        )
        text = format_worker_report(result)
        assert "review the research report" in text

    def test_formats_execution_scheduler_coverage(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="partial_progress",
            response_text="",
            evidence={
                "execution_schedule": {
                    "required_lanes": 3,
                    "executed_lanes": 2,
                    "fetched_lanes": 1,
                    "distinct_domains": 1,
                    "domain_target": 3,
                },
                "same_run_recovery": {
                    "should_retry": True,
                    "resolved": False,
                    "tool_calls": 2,
                },
            },
        )
        text = format_worker_report(result)
        assert "2/3 lanes searched" in text
        assert "1/3 independent domains" in text
        assert "Same-run recovery: still incomplete" in text

    def test_formats_claim_level_provenance_warning(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="partial_progress",
            response_text="",
            evidence={
                "claim_source_matrix": {
                    "claim_count": 2,
                    "supported_claims": 2,
                    "partial_claims": 0,
                    "unsupported_claims": 0,
                    "conflicting_claims": 0,
                    "corroborated_claims": 1,
                    "false_corroborated_claims": 1,
                    "provenance_coverage_rate": 1.0,
                    "time_sensitive_claims": 2,
                    "temporally_ready_claims": 1,
                    "stale_claims": 1,
                    "undated_temporal_claims": 0,
                }
            },
        )

        text = format_worker_report(result)

        assert "Claim provenance" in text
        assert "1 false corroboration" in text
        assert "coverage=100%" in text
        assert "Temporal evidence: 1/2 current, 1 stale, 0 undated" in text

    def test_formats_temporal_supersession_audit_summary(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="completed",
            response_text="",
            evidence={
                "claim_source_matrix": {
                    "claim_count": 3,
                    "supported_claims": 2,
                    "partial_claims": 0,
                    "unsupported_claims": 0,
                    "conflicting_claims": 0,
                    "corroborated_claims": 1,
                    "false_corroborated_claims": 0,
                    "provenance_coverage_rate": 1.0,
                    "resolved_temporal_conflicts": 1,
                    "unresolved_contradictions": 0,
                    "superseded_claims": 1,
                }
            },
        )

        text = format_worker_report(result)

        assert "Temporal supersession: 1 resolved" in text
        assert "0 active conflicts" in text
        assert "1 historical claims retained" in text

    def test_formats_cross_run_claim_lifecycle(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="completed",
            response_text="",
            evidence={
                "claim_lifecycle": {
                    "scope_key": "scope-1",
                    "summary": {
                        "tracked_subjects": 4,
                        "confirmed_transitions": 2,
                        "pending_changes": 1,
                    },
                }
            },
        )

        text = format_worker_report(result)

        assert "Claim lifecycle: 4 tracked subjects" in text
        assert "2 confirmed transitions" in text
        assert "1 pending changes" in text

    def test_searching_next_step(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="searching",
            response_text="Searching...",
            evidence={},
        )
        text = format_worker_report(result)
        assert "continue research" in text

    def test_partial_progress_next_step(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="partial_progress",
            response_text="Stored 3 findings.",
            evidence={"findings_count": 3},
        )
        text = format_worker_report(result)
        assert "Status: partial_progress" in text
        assert "continue research" in text

    def test_browser_result_not_affected(self):
        result = WorkerExecutionResult(
            worker="browser_worker",
            status="verified",
            response_text="Done.",
            evidence={"current_url": "https://example.com"},
        )
        text = format_worker_report(result)
        assert "URL: https://example.com" in text
        assert "Findings:" not in text


# ============== Tool whitelist ==============


class TestResearchToolWhitelist:
    def test_no_browser_tools(self):
        assert "browse_page" not in RESEARCH_TOOL_WHITELIST
        assert "browser_act" not in RESEARCH_TOOL_WHITELIST
        assert "browser_close" not in RESEARCH_TOOL_WHITELIST

    def test_has_search_tools(self):
        assert "web_search" in RESEARCH_TOOL_WHITELIST
        assert "extract_content" in RESEARCH_TOOL_WHITELIST

    def test_has_research_tools(self):
        assert "start_research" in RESEARCH_TOOL_WHITELIST


class TestResearchWorkerSessionLog:
    @pytest.mark.asyncio
    async def test_same_run_recovery_executes_only_missing_evidence_actions(self):
        from remy.core.worker import WorkerResult
        from remy.core.workers.research_worker import run_research_worker

        goal = {
            "goal_id": "goal-same-run-recovery",
            "description": "Research AI agent memory architecture",
            "goal_template": "market_research",
        }
        config = {"research_mode": "balanced", "source_scope": "web"}
        plan = _build_research_query_plan(goal, config)

        def search_event(query, url):
            return {
                "type": "tool_call",
                "tool": "web_search",
                "args": {"query": query},
                "result": {"results": [{"url": url, "title": "Evidence"}]},
            }

        def fetch_event(url):
            return {
                "type": "tool_call",
                "tool": "extract_content",
                "args": {"url": url},
                "result": {
                    "url": url,
                    "content": (
                        f"AI agent memory architecture research evidence from {url}. " * 20
                    ),
                },
            }

        first_url = "https://primary.example/evidence"
        first_log = [
            search_event(plan["queries"][0]["text"], first_url),
            fetch_event(first_url),
        ]
        recovery_log = []
        for index, query in enumerate(plan["queries"][1:], start=2):
            url = f"https://source{index}.example/evidence"
            recovery_log.extend((search_event(query["text"], url), fetch_event(url)))

        with (
            patch("remy.core.worker.execute_single_worker", new_callable=AsyncMock) as mock_exec,
            patch("remy.core.research_sessions.load_research_session", return_value=None),
            patch("remy.core.research_sessions.save_research_session", side_effect=lambda s: s),
            patch("remy.core.research_sessions.append_queries"),
            patch("remy.core.research_sessions.record_source_fetch"),
            patch("remy.core.research_sessions.record_source_decision"),
            patch("remy.core.research_sessions.record_finding"),
            patch("remy.core.research_sessions.record_contradictions"),
            patch("remy.core.research_sessions.mark_session_completed"),
        ):
            mock_exec.side_effect = [
                WorkerResult(
                    role="osint",
                    status="success",
                    output="Initial evidence.",
                    tool_calls=2,
                    session_log=first_log,
                ),
                WorkerResult(
                    role="osint",
                    status="success",
                    output="Recovered evidence.",
                    tool_calls=4,
                    session_log=recovery_log,
                ),
            ]
            result = await run_research_worker(
                goal=goal,
                session_id="sess-same-run-recovery",
                session_log=[],
                history=[],
            )

        assert mock_exec.await_count == 2
        recovery_call = mock_exec.await_args_list[1]
        assert "SAME-RUN EVIDENCE RECOVERY" in recovery_call.kwargs["task"].instruction
        assert recovery_call.kwargs["step_budget"] <= 4
        assert recovery_call.kwargs["timeout_override"] == 30
        assert result.evidence["execution_schedule"]["sufficient"] is True
        assert result.evidence["same_run_recovery"]["resolved"] is True
        assert result.evidence["source_provenance_graph"]["independent_root_count"] == 3
        assert any(
            event.get("type") == "source_provenance_graph"
            for event in result.session_log
        )
        assert any(
            event.get("type") == "claim_source_matrix"
            and "resolved_temporal_conflicts" in event
            for event in result.session_log
        )
        assert any(
            event.get("type") == "research_same_run_recovery"
            for event in result.session_log
        )

    @pytest.mark.asyncio
    async def test_uses_worker_session_log_for_evidence(self):
        from remy.core.worker import WorkerResult
        from remy.core.workers.research_worker import run_research_worker

        fake_log = [
            {
                "type": "tool_call",
                "tool": "web_search",
                "args": {"query": "AI agent memory competitors"},
                "result": "",
            },
            {
                "type": "tool_call",
                "tool": "add_research_finding",
                "args": {
                    "summary": "Mem0 is a competitor",
                    "source_url": "https://mem0.ai",
                },
                "result": '{"auto_contradictions":[]}',
            },
        ]

        with (
            patch("remy.core.worker.execute_single_worker", new_callable=AsyncMock) as mock_exec,
            patch("remy.core.research_sessions.load_research_session", return_value=None),
            patch("remy.core.research_sessions.save_research_session", side_effect=lambda s: s),
            patch("remy.core.research_sessions.append_queries"),
            patch("remy.core.research_sessions.record_source_fetch"),
            patch("remy.core.research_sessions.record_source_decision"),
            patch("remy.core.research_sessions.record_finding"),
            patch("remy.core.research_sessions.record_contradictions"),
            patch("remy.core.research_sessions.mark_session_completed"),
        ):
            mock_exec.return_value = WorkerResult(
                role="osint",
                status="success",
                output="Found competitors.",
                tool_calls=2,
                session_log=fake_log,
            )
            result = await run_research_worker(
                goal={
                    "goal_id": "goal-1",
                    "description": "Research competitors",
                    "goal_template": "market_research",
                },
                session_id="sess-1",
                session_log=[],
                history=[],
            )

            assert result.session_log[: len(fake_log)] == fake_log
            assert result.session_log[-1]["type"] == "research_execution_schedule"
            assert result.evidence["findings_count"] == 1
            assert result.evidence["sources"] == ["https://mem0.ai"]
        assert "add_research_finding" in RESEARCH_TOOL_WHITELIST
        assert "complete_research" in RESEARCH_TOOL_WHITELIST

    @pytest.mark.asyncio
    async def test_run_research_worker_delegates_to_osint_role(self):
        from remy.core.worker import WorkerResult
        from remy.core.workers.research_worker import run_research_worker

        with (
            patch("remy.core.worker.execute_single_worker", new_callable=AsyncMock) as mock_exec,
            patch("remy.core.research_sessions.load_research_session", return_value=None),
            patch("remy.core.research_sessions.save_research_session", side_effect=lambda s: s),
            patch("remy.core.research_sessions.append_queries"),
            patch("remy.core.research_sessions.record_source_fetch"),
            patch("remy.core.research_sessions.record_source_decision"),
            patch("remy.core.research_sessions.record_finding"),
            patch("remy.core.research_sessions.record_contradictions"),
            patch("remy.core.research_sessions.mark_session_completed"),
        ):
            mock_exec.return_value = WorkerResult(
                role="osint",
                status="success",
                output="Found competitors.",
                tool_calls=1,
                session_log=[],
            )
            result = await run_research_worker(
                goal={
                    "goal_id": "goal-2",
                    "description": "Research competitors",
                    "goal_template": "market_research",
                },
                session_id="sess-2",
                session_log=[],
                history=[],
            )

        delegated_task = mock_exec.await_args_list[0].kwargs["task"]
        assert delegated_task.role == "osint"
        assert "EXECUTION QUERY PLAN" in delegated_task.instruction
        assert "[primary]" in delegated_task.instruction
        assert result.worker == "research_worker"
        assert result.status != "error"
        assert result.evidence["query_plan"]["strategy"] == "evidence_lanes"
        assert "Unknown role: osint" not in result.response_text

    def test_has_memory_tools(self):
        assert "recall" in RESEARCH_TOOL_WHITELIST
        assert "store" in RESEARCH_TOOL_WHITELIST


# ============== Dispatch routing ==============


class TestResearchDispatchRouting:
    def test_market_research_template(self):
        from remy.core.autonomy import _should_use_research_worker

        assert _should_use_research_worker({"goal_template": "market_research"}) is True

    def test_research_keyword_in_description(self):
        from remy.core.autonomy import _should_use_research_worker

        assert (
            _should_use_research_worker({"description": "Research competitor pricing strategies"})
            is True
        )

    def test_osint_keyword(self):
        from remy.core.autonomy import _should_use_research_worker

        assert _should_use_research_worker({"description": "Run OSINT on target companies"}) is True

    def test_non_research_goal(self):
        from remy.core.autonomy import _should_use_research_worker

        assert (
            _should_use_research_worker(
                {"description": "Sign up for platform X", "goal_template": "signup_operator"}
            )
            is False
        )

    def test_none_goal(self):
        from remy.core.autonomy import _should_use_research_worker

        assert _should_use_research_worker(None) is False

    def test_empty_goal(self):
        from remy.core.autonomy import _should_use_research_worker

        assert _should_use_research_worker({}) is False
