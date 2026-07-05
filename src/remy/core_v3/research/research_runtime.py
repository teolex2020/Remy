"""
Research Runtime for Remy v3.

Orchestrates the full research pipeline:
  plan → collect → rank → extract → synthesize

Bridges v2 tools (web_search, extract_content) with v3 structured research.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .research_models import (
    ResearchProject, ResearchStatus, ResearchQuestion,
    Source, SourceCredibility, Finding, FindingConfidence,
)
from .research_context import ResearchContextRuntime
from .source_ranking import SourceRanker
from .synthesis import SynthesisEngine

log = logging.getLogger(__name__)


class ResearchRuntime:
    """Orchestrates a structured research project through its lifecycle."""

    def __init__(
        self,
        max_sources: int = 10,
        max_queries: int = 5,
        min_corroboration: int = 2,
        persistence_runtime=None,
    ):
        self.ranker = SourceRanker()
        self.synthesizer = SynthesisEngine(min_corroboration=min_corroboration)
        self.context_runtime = ResearchContextRuntime(persistence_runtime=persistence_runtime)
        self.max_sources = max_sources
        self.max_queries = max_queries

    # -------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------

    async def run(self, project: ResearchProject) -> ResearchProject:
        """Run the full research pipeline on a project.

        Phases: planning → collecting → analyzing → synthesizing → completed.
        """
        try:
            self.context_runtime.hydrate_project(project)
            if project.status == ResearchStatus.PLANNING:
                await self._phase_plan(project)

            if project.status == ResearchStatus.COLLECTING:
                await self._phase_collect(project)

            if project.status == ResearchStatus.ANALYZING:
                await self._phase_analyze(project)

            if project.status == ResearchStatus.SYNTHESIZING:
                await self._phase_synthesize(project)
                self.context_runtime.store_project_summary(project)

        except Exception as e:
            log.exception("Research pipeline error: %s", e)
            project.status = ResearchStatus.FAILED

        project.updated_at = time.time()
        return project

    # -------------------------------------------------------------------
    # Phase 1: Planning
    # -------------------------------------------------------------------

    async def _phase_plan(self, project: ResearchProject):
        """Generate research questions and search queries."""
        project.status = ResearchStatus.PLANNING
        log.info("Research planning: %s", project.objective[:60])

        if not project.questions:
            project.questions = self._generate_questions(project.objective)

        if project.prior_context:
            project.questions.insert(0, ResearchQuestion(
                question=f"What changed recently for {project.objective}?",
                priority=1,
                search_queries=[
                    f"{project.objective} latest updates",
                    f"{project.objective} recent changes",
                ],
            ))

        # Generate search queries from questions
        for q in project.questions:
            if not q.search_queries:
                q.search_queries = self._question_to_queries(q.question)
            for hint in project.strategy_hints[:2]:
                if hint not in q.search_queries:
                    q.search_queries.append(hint)
            q.search_queries = q.search_queries[:3]

        project.status = ResearchStatus.COLLECTING

    def _generate_questions(self, objective: str) -> list[ResearchQuestion]:
        """Break objective into research questions."""
        # Default decomposition — can be enhanced with LLM later
        questions = [
            ResearchQuestion(
                question=f"What is {objective}?",
                priority=1,
            ),
            ResearchQuestion(
                question=f"What are the key facts about {objective}?",
                priority=2,
            ),
            ResearchQuestion(
                question=f"What are recent developments in {objective}?",
                priority=3,
            ),
        ]
        return questions

    def _question_to_queries(self, question: str) -> list[str]:
        """Convert a research question to search queries."""
        # Simple: use the question itself + a shortened version
        queries = [question]
        # Remove question words for a keyword query
        for prefix in ("What is ", "What are ", "How does ", "Why is ",
                        "When did ", "Who is ", "Where is "):
            if question.startswith(prefix):
                queries.append(question[len(prefix):].rstrip("?"))
                break
        return queries[:2]

    # -------------------------------------------------------------------
    # Phase 2: Collection
    # -------------------------------------------------------------------

    async def _phase_collect(self, project: ResearchProject):
        """Collect sources using search and content extraction."""
        project.status = ResearchStatus.COLLECTING
        log.info("Collecting sources for: %s", project.objective[:60])

        queries_to_run = []
        for q in project.questions:
            for query in q.search_queries:
                if query not in project.queries_executed:
                    queries_to_run.append(query)

        # Limit queries
        queries_to_run = queries_to_run[:self.max_queries - len(project.queries_executed)]

        for query in queries_to_run:
            if len(project.sources) >= self.max_sources:
                break
            sources = await self._execute_search(query)
            for src in sources:
                if len(project.sources) >= self.max_sources:
                    break
                # Deduplicate by URL
                if not any(s.url == src.url for s in project.sources):
                    project.add_source(src)
            project.queries_executed.append(query)

        # Fetch content for top-ranked sources
        ranked = self.ranker.rank(project.sources)
        for src in ranked:
            if not src.fetched:
                await self._fetch_source(src)

        project.status = ResearchStatus.ANALYZING

    async def _execute_search(self, query: str) -> list[Source]:
        """Execute a web search via v2 tools."""
        try:
            from remy.core.langgraph_tools import get_all_tools
            tools = get_all_tools()

            # Find web_search tool
            search_fn = None
            for tool in tools:
                if getattr(tool, "name", "") == "web_search":
                    search_fn = tool
                    break

            if not search_fn:
                log.warning("web_search tool not available")
                return []

            result = await self._invoke_tool(search_fn, {"query": query})
            return self._parse_search_results(result, query)

        except Exception as e:
            log.warning("Search failed for '%s': %s", query, e)
            return []

    async def _fetch_source(self, source: Source):
        """Fetch full content for a source via v2 extract_content."""
        if not source.url:
            return
        try:
            from remy.core.langgraph_tools import get_all_tools
            tools = get_all_tools()

            extract_fn = None
            for tool in tools:
                if getattr(tool, "name", "") == "extract_content":
                    extract_fn = tool
                    break

            if not extract_fn:
                source.fetch_error = "extract_content tool not available"
                return

            result = await self._invoke_tool(extract_fn, {"url": source.url})
            if isinstance(result, str) and result:
                source.full_text = result[:5000]  # Cap content size
                source.snippet = result[:300]
                source.fetched = True
                source.fetched_at = time.time()
            else:
                source.fetch_error = "Empty content"

        except Exception as e:
            source.fetch_error = str(e)
            log.warning("Fetch failed for %s: %s", source.url, e)

    async def _invoke_tool(self, tool: Any, args: dict) -> Any:
        """Invoke a LangGraph tool (sync or async)."""
        import asyncio
        if asyncio.iscoroutinefunction(getattr(tool, "ainvoke", None)):
            return await tool.ainvoke(args)
        elif hasattr(tool, "invoke"):
            return await asyncio.to_thread(tool.invoke, args)
        elif callable(tool):
            return await asyncio.to_thread(tool, **args)
        return None

    def _parse_search_results(self, result: Any, query: str) -> list[Source]:
        """Parse search results into Source objects.

        Handles multiple formats:
        - JSON string from web_search tool: {"answer": "...", "sources": [...]}
        - Plain dict with "sources" / "answer" keys
        - List of {"url", "title", "snippet"} dicts
        - Plain text with URLs embedded per line
        """
        import json as _json

        sources = []

        # Unwrap JSON string → dict
        if isinstance(result, str):
            try:
                parsed = _json.loads(result)
                if isinstance(parsed, dict):
                    result = parsed
            except (ValueError, TypeError):
                pass

        # --- Format 1: our web_search JSON {"answer": "...", "sources": [{title, uri}]} ---
        if isinstance(result, dict):
            raw_sources = result.get("sources", [])
            answer_text = result.get("answer", "")

            for i, item in enumerate(raw_sources):
                url = item.get("uri", item.get("url", item.get("link", "")))
                title = item.get("title", "")
                if url:
                    sources.append(Source(
                        url=url,
                        title=title[:200],
                        snippet=answer_text[:300] if not sources else "",
                        query=query,
                        search_rank=i + 1,
                    ))

            # If no structured sources but answer has URLs — fall through to text parsing
            if not sources and answer_text:
                result = answer_text  # re-parse as plain text below

        # --- Format 2: list of dicts ---
        if isinstance(result, list):
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    url = item.get("url", item.get("uri", item.get("link", "")))
                    if url:
                        sources.append(Source(
                            url=url,
                            title=item.get("title", "")[:200],
                            snippet=item.get("snippet", item.get("description", ""))[:300],
                            query=query,
                            search_rank=i + 1,
                        ))

        # --- Format 3: plain text with http URLs embedded ---
        if not sources and isinstance(result, str):
            for i, line in enumerate(result.strip().split("\n")):
                line = line.strip()
                if not line or line.startswith("---"):
                    continue
                url = ""
                title = line
                for word in line.split():
                    if word.startswith("http"):
                        url = word.rstrip(",;)")
                        title = line.replace(url, "").strip(" •-–—|")
                        break
                if url:
                    sources.append(Source(
                        url=url,
                        title=title[:200],
                        query=query,
                        search_rank=i + 1,
                    ))

        if not sources:
            log.debug("_parse_search_results: no sources extracted from result type=%s", type(result).__name__)

        return sources[:10]

    # -------------------------------------------------------------------
    # Phase 3: Analysis
    # -------------------------------------------------------------------

    async def _phase_analyze(self, project: ResearchProject):
        """Extract findings from collected sources."""
        project.status = ResearchStatus.ANALYZING
        log.info("Analyzing %d sources", len(project.sources))

        usable = self.ranker.filter_usable(project.sources)
        if not usable:
            log.warning("No usable sources to analyze")
            project.status = ResearchStatus.SYNTHESIZING
            return

        for source in usable:
            findings = self._extract_findings_from_source(source)
            for finding in findings:
                project.add_finding(finding)

        # Score findings by corroboration
        self.synthesizer.score_findings(project.findings)

        # Try to answer questions
        self.synthesizer.try_answer_questions(project)

        project.status = ResearchStatus.SYNTHESIZING

    def _extract_findings_from_source(self, source: Source) -> list[Finding]:
        """Extract findings from a single source's content."""
        content = source.full_text or source.snippet
        if not content:
            return []

        findings = []
        # Split into paragraphs, treat substantial ones as findings
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]

        for para in paragraphs[:5]:  # Max 5 findings per source
            finding = Finding(
                content=para[:300],
                source_ids=[source.id],
                category=self._categorize_content(para),
                confidence=self._initial_confidence(source),
            )
            findings.append(finding)

        return findings

    def _categorize_content(self, text: str) -> str:
        """Simple keyword-based content categorization."""
        text_lower = text.lower()
        categories = {
            "pricing": {"price", "cost", "$", "€", "fee", "subscription", "plan"},
            "feature": {"feature", "capability", "supports", "includes", "offers"},
            "comparison": {"compared", "versus", "vs", "better", "worse", "alternative"},
            "technical": {"api", "sdk", "library", "framework", "integration", "protocol"},
            "market": {"market", "industry", "growth", "trend", "competitor"},
        }
        best_cat = "general"
        best_count = 0
        for cat, keywords in categories.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_cat = cat
        return best_cat

    def _initial_confidence(self, source: Source) -> FindingConfidence:
        """Set initial finding confidence from source credibility."""
        return {
            SourceCredibility.HIGH: FindingConfidence.MEDIUM,
            SourceCredibility.MEDIUM: FindingConfidence.MEDIUM,
            SourceCredibility.LOW: FindingConfidence.LOW,
            SourceCredibility.UNKNOWN: FindingConfidence.LOW,
        }.get(source.credibility, FindingConfidence.LOW)

    # -------------------------------------------------------------------
    # Phase 4: Synthesis
    # -------------------------------------------------------------------

    async def _phase_synthesize(self, project: ResearchProject):
        """Run synthesis engine to produce final output."""
        project.status = ResearchStatus.SYNTHESIZING
        log.info("Synthesizing %d findings", len(project.findings))

        self.synthesizer.synthesize(project)

        project.status = ResearchStatus.COMPLETED
        project.completed_at = time.time()
        log.info(
            "Research complete: %d sources, %d findings, confidence %.2f",
            len(project.sources), len(project.findings),
            project.synthesis.confidence if project.synthesis else 0.0,
        )

    # -------------------------------------------------------------------
    # Convenience: create + run
    # -------------------------------------------------------------------

    async def research(
        self,
        objective: str,
        mission_id: str = "",
        goal_id: str = "",
    ) -> ResearchProject:
        """Create a research project and run it end-to-end."""
        project = ResearchProject(
            objective=objective,
            mission_id=mission_id,
            goal_id=goal_id,
            max_sources=self.max_sources,
            max_queries=self.max_queries,
        )
        return await self.run(project)
