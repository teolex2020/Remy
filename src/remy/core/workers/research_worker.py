"""Research worker wrapper with a focused prompt and scoped tool access.

Handles market research, competitive analysis, and source gathering.
Uses the existing worker.py LangGraph execution engine but with a
research-specific system instruction and tool whitelist.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

# Saturation: if this many consecutive cycles add zero new findings → stop early
_SATURATION_WINDOW = 3

from remy.core.workers.contracts import WorkerExecutionResult

logger = logging.getLogger("ResearchWorker")

RESEARCH_WORKER_CHANNEL = "worker-osint"

# Tools the research worker is allowed to use
RESEARCH_TOOL_WHITELIST = frozenset(
    {
        "recall",
        "web_search",
        "extract_content",
        "http_get",
        "store",
        "search",
        "start_research",
        "add_research_finding",
        "complete_research",
        "store_research",
        "scratchpad",
        "get_current_datetime",
    }
)

_MODE_QUERY_LIMITS = {
    "speed": 2,
    "balanced": 3,
    "deep": 7,
}

_MODE_SOURCE_LIMITS = {
    "speed": 5,
    "balanced": 8,
    "deep": 12,
}

_MODE_STEP_BUDGETS = {
    "speed": 5,
    "balanced": 8,
    "deep": 12,
}

_MODE_TIMEOUT_LIMITS = {
    "speed": 45,
    "balanced": 60,
    "deep": 75,
}

_SCOPE_HINTS = {
    "web": (),
    "discussions": ("reddit", "hacker news", "forum", "discussion", "community"),
    "papers": ("paper", "arxiv", "research", "preprint", "documentation", "technical"),
}


def _resolve_goal_id(goal: dict) -> str:
    return str(
        (goal or {}).get("todo_id") or (goal or {}).get("goal_id") or (goal or {}).get("id") or ""
    )


def _goal_topic(goal: dict) -> str:
    return str((goal or {}).get("task_action") or (goal or {}).get("description") or "").strip()


def _extract_completion_threshold(goal: dict) -> int:
    text = " ".join(
        str((goal or {}).get(key, "") or "")
        for key in ("task_done_when", "description", "task_action")
    )
    lowered = text.lower()
    patterns = (
        r"\b(?:at least|minimum|min\.?)\s+(\d+)\b",
        r"\bfind\s+top\s+(\d+)\b",
        r"\bfind\s+(\d+)\+\b",
        r"\b(\d+)\+\s+(?:influencers|competitors|partners|sources|leads)\b",
        r"\b(\d+)\s+(?:influencers|competitors|partners|sources|leads)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return max(1, int(match.group(1)))
            except Exception:
                continue
    return 5


def _normalize_research_topic(topic: str) -> str:
    text = (topic or "").strip()
    if not text:
        return ""
    for marker in ("Search queries:", "For each", "Minimum ", "Store each"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return re.sub(r"\s+", " ", text).strip(" .")


def _extract_seed_queries(topic: str) -> list[str]:
    if not topic:
        return []
    matches = re.findall(r"'([^']{4,180})'|\"([^\"]{4,180})\"", topic)
    queries: list[str] = []
    for left, right in matches:
        query = (left or right or "").strip()
        if query:
            queries.append(query)
    return queries


def _extract_handles(topic: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"@([A-Za-z0-9_]{2,32})", topic or "")))


def _infer_source_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)]+", text or "")
    handles = re.findall(r"@([A-Za-z0-9_]{2,32})", text or "")
    inferred = [f"https://x.com/{handle}" for handle in handles]
    return list(dict.fromkeys([*(u.strip() for u in urls if u.strip()), *inferred]))


def _is_influencer_goal(topic: str) -> bool:
    lowered = (topic or "").lower()
    return any(
        token in lowered for token in ("influencer", "twitter", "x.com", "followers", "handle")
    )


def _plan_influencer_queries(
    base_topic: str, seed_queries: list[str], handles: list[str]
) -> list[str]:
    queries: list[str] = []
    queries.extend(seed_queries)
    concise_defaults = [
        'site:x.com "AI agent memory" influencer',
        'site:x.com "LLM memory" influencer',
        'site:x.com "cognitive architecture" influencer',
    ]
    if not seed_queries:
        queries.extend(concise_defaults)
    if handles:
        batch = " ".join(f"@{handle}" for handle in handles[:5])
        queries.append(f"twitter follower count and recent tweet URL for {batch}")
    elif base_topic:
        queries.append(f"{base_topic} twitter handles recent content")
    return queries


def _plan_research_queries(goal: dict, config: dict[str, Any]) -> list[str]:
    topic = _goal_topic(goal)
    if not topic:
        return []
    base_topic = _normalize_research_topic(topic)
    seed_queries = _extract_seed_queries(topic)
    handles = _extract_handles(topic)
    queries = []
    scope = str(config.get("source_scope", "web"))
    domains = list(config.get("source_domains", []) or [])
    if _is_influencer_goal(topic):
        queries.extend(_plan_influencer_queries(base_topic, seed_queries, handles))
    elif seed_queries:
        queries.extend(seed_queries)
    elif base_topic:
        queries.append(base_topic)
    if scope in _SCOPE_HINTS and not _is_influencer_goal(topic):
        for hint in _SCOPE_HINTS[scope]:
            seed = base_topic or topic
            queries.append(f"{seed} {hint}")
    if "competitor" in topic.lower() or "compare" in topic.lower():
        seed = base_topic or topic
        queries.append(f"{seed} pricing features")
        queries.append(f"{seed} reviews comparison")
    else:
        seed = base_topic or topic
        queries.append(f"{seed} overview")
        queries.append(f"{seed} latest updates")
    if scope == "domain":
        for domain in domains:
            seed = base_topic or topic
            queries.append(f"site:{domain} {seed}")
    seen: list[str] = []
    for query in queries:
        q = query.strip()
        if q and q not in seen:
            seen.append(q)
    return seen[: _MODE_QUERY_LIMITS.get(str(config.get("research_mode", "balanced")), 4)]


def _source_scope_score(url: str, scope: str, domains: list[str]) -> int:
    host = (urlparse(url).hostname or "").lower() if url else ""
    if not host:
        return 0
    if scope == "domain":
        return 4 if any(host == d or host.endswith("." + d) for d in domains) else -3
    if scope == "discussions":
        return (
            3
            if any(
                token in host
                for token in ("reddit", "news.ycombinator", "x.com", "twitter", "forum", "discuss")
            )
            else 0
        )
    if scope == "papers":
        return (
            3
            if any(
                token in host
                for token in (
                    "arxiv",
                    "doi.org",
                    "acm.org",
                    "ieee.org",
                    "docs.",
                    "research",
                    "paperswithcode",
                )
            )
            else 0
        )
    return 0


def _source_host(url: str) -> str:
    try:
        return urlparse(url or "").netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _source_class(source: dict) -> str:
    try:
        from remy.core.retrieval.source_filter import classify

        return str(classify({"uri": source.get("url") or source.get("uri") or "", "title": source.get("title", "")}).source_class or "unknown")
    except Exception:
        return "unknown"


def _source_memory_bias(source: dict) -> tuple[int, str]:
    """Apply lived source outcomes to research-worker candidate ranking."""
    try:
        from remy.core.consequence_gate import consult_policy_hint
        from remy.core_v3.memory.memory_api import get_memory

        memory = get_memory()
        url = str(source.get("url") or source.get("uri") or "")
        source_class = _source_class(source)
        host = _source_host(url)
        bias = 0
        reasons: list[str] = []
        for action in (
            f"source_class:{source_class}",
            f"source_host:{host}" if host else "",
        ):
            if not action:
                continue
            hint = consult_policy_hint(
                memory,
                situation="source-selection:global",
                action=action,
                namespace="remy-sources",
            )
            context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
            supports = int(context.get("supports", 0) or 0)
            refutes = int(context.get("refutes", 0) or 0)
            policy = str(context.get("hint") or "")
            if policy == "avoid" and refutes > 0:
                bias -= min(8, 3 + refutes)
                reasons.append(f"memory_avoid:{action}")
            elif policy == "prefer" and supports > 0:
                bias += min(5, 1 + supports)
                reasons.append(f"memory_prefer:{action}")
        return bias, "+".join(reasons)
    except Exception as exc:
        logger.debug("Research source memory bias skipped: %s", exc)
        return 0, ""


def _store_research_source_consequences(
    *,
    evidence: dict[str, Any],
    status: str,
    session_id: str,
) -> None:
    """Persist research-worker accepted sources as source-selection supports."""
    if status not in {"completed", "findings_collected", "partial_progress"}:
        return
    accepted = list((evidence or {}).get("accepted_sources") or [])
    if not accepted:
        return
    try:
        from remy.core_v3.memory.memory_api import get_memory

        memory = get_memory()
        seen_actions: set[str] = set()
        for source in accepted:
            url = str(source.get("url") or source.get("uri") or "").strip()
            if not url:
                continue
            source_class = _source_class(source)
            host = _source_host(url)
            for action in (
                f"source_class:{source_class}",
                f"source_host:{host}" if host else "",
                "source_tool:research_worker",
            ):
                if not action or action in seen_actions:
                    continue
                seen_actions.add(action)
                memory.capture_consequence(
                    situation="source-selection:global",
                    action=action,
                    consequence="SUPPORTS",
                    trust=1,
                    scope=[
                        "source-grounding",
                        "research-worker",
                        f"status:{status}",
                        f"source_class:{source_class}",
                        f"host:{host}" if host else "host:",
                        f"session:{session_id}" if session_id else "session:",
                        f"accepted_sources_count:{int((evidence or {}).get('accepted_sources_count', 0) or 0)}",
                        f"citation_coverage_rate:{(evidence or {}).get('citation_coverage_rate', 0)}",
                    ],
                    provenance=[
                        "remy:research_worker_source_grounding",
                        f"session:{session_id}" if session_id else "session:",
                    ],
                    links={"session": session_id or "", "url": url, "host": host},
                    namespace="remy-sources",
                )
    except Exception as exc:
        logger.debug("Failed to store research source consequences: %s", exc)


def _rerank_sources(
    goal: dict, sources: list[dict], config: dict[str, Any]
) -> tuple[list[dict], list[dict]]:
    topic = _goal_topic(goal).lower()
    scope = str(config.get("source_scope", "web"))
    domains = list(config.get("source_domains", []) or [])
    accepted: list[dict] = []
    rejected: list[dict] = []
    ranked: list[tuple[int, dict]] = []
    seen_urls: set[str] = set()

    for source in sources:
        url = str(source.get("url", "") or "")
        if not url or url in seen_urls:
            if url:
                rejected.append({**source, "reason": "duplicate"})
            continue
        seen_urls.add(url)
        text = " ".join(
            str(source.get(key, "") or "") for key in ("title", "snippet", "url")
        ).lower()
        score = 0
        if topic:
            overlap = sum(1 for token in set(topic.split()) if len(token) > 3 and token in text)
            score += overlap
        if source.get("fresh"):
            score += 1
        score += _source_scope_score(url, scope, domains)
        if any(
            token in url
            for token in ("blog", "docs", "github", "dev.to", "arxiv", "reddit", "news.ycombinator")
        ):
            score += 1
        memory_bias, memory_reason = _source_memory_bias(source)
        score += memory_bias
        if memory_reason:
            source = {**source, "memory_reason": memory_reason}
        ranked.append((score, source))

    ranked.sort(key=lambda item: item[0], reverse=True)
    limit = _MODE_SOURCE_LIMITS.get(str(config.get("research_mode", "balanced")), 8)
    for idx, (score, source) in enumerate(ranked):
        payload = {**source, "score": score, "source_class": _source_class(source)}
        if scope == "domain" and score < 0:
            rejected.append({**payload, "reason": "out_of_scope"})
            continue
        if idx < limit and score >= 0:
            accepted.append(payload)
        else:
            rejected.append({**payload, "reason": "low_rank"})
    return accepted, rejected


def _store_research_summary_artifact(
    goal: dict, session_summary: dict, findings: list[dict]
) -> str:
    """Store a research session summary artifact.

    D-02 / D-04 boundary rule:
    - This is an auto-generated synthesis artifact from a worker session.
    - It is NOT durable domain knowledge — it is a research artifact.
    - Stored at Level.WORKING with explicit admission_class=research_artifact.
    - Findings with source URLs are listed for reference but not promoted here.
    """
    from remy.core.agent_tools import brain

    has_cited_sources = bool(session_summary.get("accepted_sources_count", 0))
    citation_complete = has_cited_sources

    lines = [
        f"Research summary for: {_goal_topic(goal)}",
        f"Findings count: {session_summary.get('findings_count', 0)}",
        f"Accepted sources: {session_summary.get('accepted_sources_count', 0)}",
        f"Research mode: {session_summary.get('research_mode', '')}",
        f"Source scope: {session_summary.get('source_scope', '')}",
        "",
        "Top findings:",
    ]
    for idx, finding in enumerate(findings[:10], start=1):
        summary = str(finding.get("summary", "") or finding.get("text", "") or "").strip()
        source_url = str(finding.get("source_url", "") or "").strip()
        if source_url:
            lines.append(f"{idx}. {summary} [{source_url}]")
        else:
            lines.append(f"{idx}. {summary}")

    content = "\n".join(lines).strip()
    try:
        return str(
            brain.store(
                content,
                level="L1_WORKING",
                tags="research-summary,auto-generated,research-artifact",
                metadata={
                    "goal_id": str((goal or {}).get("goal_id") or ""),
                    "goal_template": str((goal or {}).get("goal_template") or ""),
                    "source_scope": str(session_summary.get("source_scope", "") or ""),
                    "research_mode": str(session_summary.get("research_mode", "") or ""),
                    "citation_complete": citation_complete,
                    "learning_channel": "internet_evidence",
                    "admission_class": "research_artifact",
                    "requires_promotion": True,
                },
            )
        )
    except Exception:
        return ""


def build_research_worker_prompt(
    goal: dict,
    current_plan: object | None = None,
    existing_knowledge: str = "",
) -> str:
    """Build a compact operational prompt for research/osint tasks."""
    desc = (goal or {}).get("description", "") or ""
    template = (goal or {}).get("goal_template", "") or ""
    resume_context = (goal or {}).get("resume_context", "") or ""
    blocked_reason = (goal or {}).get("blocked_reason", "") or ""
    task_action = (goal or {}).get("task_action", "") or ""
    task_done_when = (goal or {}).get("task_done_when", "") or ""

    plan_line = ""
    if task_action:
        plan_line = f"\nCURRENT TASK ACTION: {task_action}"
        if task_done_when:
            plan_line += f"\nDONE WHEN: {task_done_when}"
    elif current_plan:
        try:
            plan_line = f"\nPLAN CONTEXT: {str(current_plan)[:400]}"
        except Exception:
            plan_line = ""

    resume_text = ""
    if resume_context or blocked_reason:
        resume_text = "\nRESUME CONTEXT:\n"
        if blocked_reason:
            resume_text += f"- Previous blocker: {blocked_reason}\n"
        if resume_context:
            resume_text += f"- Continue from: {resume_context}\n"

    knowledge_text = ""
    if existing_knowledge:
        knowledge_text = f"\nEXISTING KNOWLEDGE:\n{existing_knowledge[:600]}\n"

    # Inject prior research findings and contradictions from research memory
    research_memory_text = ""
    research_session_text = ""
    config = {}
    try:
        from remy.core.capability_packs import resolve_research_config
        from remy.core.research_memory import format_existing_research_for_prompt
        from remy.core.research_sessions import get_research_session_summary, load_research_session

        config = resolve_research_config(goal)
        research_memory_text = format_existing_research_for_prompt(desc)
        session_summary = get_research_session_summary(_resolve_goal_id(goal))
        if session_summary:
            open_qs: list[str] = []
            try:
                _sess_obj = load_research_session(_resolve_goal_id(goal))
                if _sess_obj:
                    open_qs = [
                        w[len("[open] "):].strip()
                        for w in (_sess_obj.warnings or [])
                        if w.startswith("[open] ")
                    ]
            except Exception:
                pass
            open_qs_text = ""
            if open_qs:
                open_qs_text = "\nUNANSWERED QUESTIONS FROM PRIOR CYCLE:\n" + "\n".join(f"- {q}" for q in open_qs[:3]) + "\n"
            # Surface unresolved contradictions so the worker can seek tie-breakers
            contradictions_prompt_text = ""
            try:
                _sess_obj2 = load_research_session(_resolve_goal_id(goal))
                if _sess_obj2 and _sess_obj2.contradictions:
                    c_lines = []
                    for c in _sess_obj2.contradictions[:3]:
                        s = str(c.get("summary") or "")
                        a = str(c.get("claim_a") or c.get("source_a") or "")
                        b = str(c.get("claim_b") or c.get("source_b") or "")
                        if a and b:
                            c_lines.append(f"  - {s}: A says «{a[:80]}», B says «{b[:80]}»")
                        elif s:
                            c_lines.append(f"  - {s[:120]}")
                    if c_lines:
                        contradictions_prompt_text = (
                            "\nUNRESOLVED CONTRADICTIONS (seek a third source to resolve):\n"
                            + "\n".join(c_lines) + "\n"
                        )
            except Exception:
                pass
            research_session_text = (
                "\nRESEARCH SESSION:\n"
                f"- Session: {session_summary['session_id']} ({session_summary['status']})\n"
                f"- Accepted sources: {session_summary['accepted_sources_count']}\n"
                f"- Rejected sources: {session_summary['rejected_sources_count']}\n"
                f"- Contradictions: {session_summary['contradictions_count']}\n"
            ) + open_qs_text + contradictions_prompt_text
    except Exception:
        pass
    planned_queries = _plan_research_queries(goal, config)
    planned_query_text = ""
    if planned_queries:
        planned_query_text = (
            "\nPLANNED QUERIES:\n" + "\n".join(f"- {q}" for q in planned_queries[:7]) + "\n"
        )
    mode = str(config.get("research_mode", "balanced") or "balanced")
    scope = str(config.get("source_scope", "web") or "web")
    domains = list(config.get("source_domains", []) or [])
    citation_required = bool(config.get("citation_required", True))
    step_budget = int(config.get("step_budget", 0) or _MODE_STEP_BUDGETS.get(mode, 8))
    max_queries = _MODE_QUERY_LIMITS.get(mode, 4)
    max_sources = _MODE_SOURCE_LIMITS.get(mode, 8)
    config_text = (
        "\nRESEARCH CONFIG:\n"
        f"- Mode: {mode}\n"
        f"- Source scope: {scope}\n"
        f"- Citation required: {'yes' if citation_required else 'no'}\n"
        f"- Max queries this cycle: {max_queries}\n"
        f"- Max sources this cycle: {max_sources}\n"
        f"- Step budget this cycle: {step_budget}\n"
        f"{'- Source domains: ' + ', '.join(domains) if domains else ''}\n"
    )

    return (
        "You are RESEARCH_WORKER.\n"
        "Execute a focused research step for this task.\n"
        "Do not operate websites interactively. Do not ask follow-up questions.\n"
        "Return structured findings only.\n"
        "\nTASK:\n"
        f"{desc}\n"
        f"JOB TEMPLATE: {template or 'market_research'}\n"
        f"{plan_line}"
        f"{resume_text}"
        f"{knowledge_text}"
        f"{research_memory_text}"
        f"{research_session_text}"
        f"{config_text}"
        f"{planned_query_text}"
        "\nRULES:\n"
        "- Use recall first (free) before web_search (costs tokens).\n"
        "- Use extract_content for article pages, http_get only for APIs/JSON.\n"
        "- Use start_research for multi-query investigations.\n"
        "- Store each finding with add_research_finding including source URL.\n"
        "- Cross-reference multiple sources before concluding.\n"
        "- Prefer accepted/relevant sources and ignore low-quality or out-of-scope links.\n"
        "- Every final claim must cite at least one source URL; comparisons should cite 2 when possible.\n"
        "- If evidence is weak or conflicting, say so explicitly.\n"
        "- Stay within the research config budget for THIS cycle.\n"
        "- If you have partial findings, store them and stop; do not try to finish the whole report in one cycle.\n"
        "- Prefer 2-3 strong leads now over a long search loop that times out.\n"
        "- Do NOT use browse_page or browser_act.\n"
        "- Do NOT create goals, files, or delegate tasks.\n"
        "- Keep the final response under 10 lines: key findings + sources.\n"
        f"{_pack_guardrails(goal)}"
    )


def _pack_guardrails(goal: dict) -> str:
    """Inject capability pack guardrails into the worker prompt."""
    try:
        from remy.core.capability_packs import format_guardrails_for_prompt, resolve_pack

        pack = resolve_pack(goal)
        return format_guardrails_for_prompt(pack)
    except Exception:
        return ""


def _extract_research_evidence(session_log: list[dict], response_text: str) -> dict[str, Any]:
    """Extract structured evidence from a research worker's session log."""
    sources: list[str] = []
    findings_count = 0
    queries: list[str] = []
    project_id = ""
    findings: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    contradiction_count = 0
    final_artifact_id = ""
    for entry in session_log or []:
        if entry.get("type") != "tool_call":
            continue
        tool = entry.get("tool", "")
        args = entry.get("args") or {}

        if tool == "web_search":
            q = args.get("query") or args.get("q") or ""
            if q:
                queries.append(q)
            result = entry.get("result") or {}
            if isinstance(result, str):
                extracted_urls = _infer_source_urls_from_text(result)
                for url in extracted_urls:
                    source_records.append({"url": url, "title": "", "snippet": result[:180]})
            if isinstance(result, dict):
                for item in result.get("results", []) or []:
                    if isinstance(item, dict):
                        source_records.append(
                            {
                                "url": item.get("url", ""),
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                            }
                        )
        elif tool == "add_research_finding":
            findings_count += 1
            src = args.get("source_url", "")
            if src:
                sources.append(src)
            result = entry.get("result") or {}
            if isinstance(result, dict) and result.get("auto_contradictions"):
                for ac in (result.get("auto_contradictions") or []):
                    contradiction_count += 1
                    if isinstance(ac, dict):
                        contradictions.append(ac)
                    elif isinstance(ac, str) and ac:
                        contradictions.append({"summary": ac})
            findings.append(
                {
                    "source_url": src,
                    "summary": args.get("summary") or args.get("finding") or "",
                }
            )
        elif tool == "start_research":
            pid = args.get("project_id", "")
            if pid:
                project_id = pid
        elif tool == "extract_content":
            url = args.get("url", "")
            if url:
                sources.append(url)
                source_records.append(
                    {
                        "url": url,
                        "title": args.get("title", "") or "",
                        "snippet": args.get("question", "") or "",
                    }
                )
        elif tool == "store":
            content = str(args.get("content", "") or "")
            tags = str(args.get("tags", "") or "").lower()
            if content and (
                "research" in tags or "influencer" in tags or "twitter-ai-memory" in tags
            ):
                findings_count += 1
                urls = _infer_source_urls_from_text(content)
                if urls:
                    sources.extend(urls)
                    for url in urls:
                        source_records.append({"url": url, "title": "", "snippet": content[:180]})
                findings.append(
                    {
                        "source_url": urls[0] if urls else "",
                        "summary": content[:240],
                    }
                )
        elif tool == "complete_research":
            result = entry.get("result") or {}
            if isinstance(result, dict):
                final_artifact_id = str(
                    result.get("record_id", "") or result.get("report_id", "") or ""
                )

    return {
        "queries": queries,
        "sources": list(dict.fromkeys(sources)),  # deduplicate, preserve order
        "source_records": source_records,
        "findings": findings,
        "findings_count": findings_count,
        "contradictions": contradictions,
        "contradictions_count": contradiction_count,
        "final_artifact_id": final_artifact_id,
        "project_id": project_id,
    }


def _derive_research_status(session_log: list[dict], response_text: str) -> str:
    """Derive a machine-readable status from the research worker's execution."""
    tool_calls = [e for e in (session_log or []) if e.get("type") == "tool_call"]
    if not tool_calls:
        return "no_action"

    tool_names = [e.get("tool") for e in tool_calls]

    if "complete_research" in tool_names:
        return "completed"

    if any(t in ("add_research_finding", "store_research", "store") for t in tool_names):
        return "findings_collected"

    if any(t in ("web_search", "extract_content", "http_get") for t in tool_names):
        return "searching"

    return "attempted"


async def _evaluate_research_outcome(
    goal: dict,
    findings: list[dict],
    contradictions: list[dict],
) -> dict:
    """Ask the LLM whether the research goal is answered and what still needs work.

    Contradictions are treated as signals: each unresolved contradiction generates
    a tie-breaker search query for the next research cycle.

    Returns:
      answered (bool)
      confidence (str)          — "high" | "medium" | "low"
      open_questions (list[str])
      refinement_queries (list[str]) — gap-fill + contradiction tie-breaker queries (max 5)
      summary (str)
    """
    question = _goal_topic(goal)
    done_when = (goal or {}).get("task_done_when", "") or ""

    if not question or not findings:
        return {
            "answered": False,
            "confidence": "low",
            "open_questions": ["No findings collected yet."],
            "refinement_queries": [],
            "summary": "No findings to evaluate.",
        }

    findings_text = "\n".join(
        f"- {f.get('summary', '')[:180]} [{f.get('source_url', '')}]"
        for f in findings[:15]
    )

    contradictions_text = ""
    if contradictions:
        lines = []
        for c in contradictions[:5]:
            s = str(c.get("summary") or c.get("text") or "")
            a = str(c.get("source_a") or c.get("claim_a") or "")
            b = str(c.get("source_b") or c.get("claim_b") or "")
            if a and b:
                lines.append(f"  - Conflict: {s} | A: {a[:100]} | B: {b[:100]}")
            elif s:
                lines.append(f"  - {s[:160]}")
        if lines:
            contradictions_text = "\nUNRESOLVED CONTRADICTIONS:\n" + "\n".join(lines) + "\n"

    prompt = (
        "You are a research quality evaluator. Answer in JSON only.\n\n"
        f"RESEARCH QUESTION: {question}\n"
        + (f"DONE WHEN: {done_when}\n" if done_when else "")
        + f"\nFINDINGS SO FAR:\n{findings_text}\n"
        + contradictions_text
        + "\nEvaluate whether the research question is sufficiently answered.\n"
        "For any unresolved contradictions, generate a tie-breaker search query that would find a third authoritative source.\n"
        "Return JSON with exactly these fields:\n"
        '{\n'
        '  "answered": true/false,\n'
        '  "confidence": "high"/"medium"/"low",\n'
        '  "open_questions": ["max 3 specific unanswered sub-questions"],\n'
        '  "refinement_queries": ["max 5: gap-fill queries + one per contradiction tie-breaker"],\n'
        '  "summary": "one sentence"\n'
        '}'
    )

    try:
        from remy.core.llm import call_llm_async
        import json as _json

        response = await call_llm_async(prompt, purpose="research-outcome-eval")
        text = (response.content or "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = _json.loads(text[start:end])
            return {
                "answered": bool(data.get("answered", False)),
                "confidence": str(data.get("confidence", "low")),
                "open_questions": list(data.get("open_questions", []))[:3],
                "refinement_queries": list(data.get("refinement_queries", []))[:5],
                "summary": str(data.get("summary", ""))[:200],
            }
    except Exception as exc:
        logger.debug("Outcome evaluation failed: %s", exc)

    return {
        "answered": False,
        "confidence": "low",
        "open_questions": [],
        "refinement_queries": [],
        "summary": "Evaluation unavailable.",
    }


def _check_saturation(goal_id: str) -> bool:
    """Return True if the last _SATURATION_WINDOW cycles added no new findings.

    Reads the per-goal research session history and counts how many findings
    were added in each recent resumed run. If every run in the window is empty
    the research has saturated and we should stop rather than waste more tokens.
    """
    try:
        from remy.core.research_sessions import load_research_session

        session = load_research_session(goal_id)
        if not session:
            return False
        runs = list(getattr(session, "findings_per_run", []) or [])
        if len(runs) < _SATURATION_WINDOW:
            return False
        recent = runs[-_SATURATION_WINDOW:]
        added = [int(n) for n in recent]
        saturated = all(n == 0 for n in added)
        if saturated:
            logger.info(
                "Research saturation detected for goal %s — last %d cycles added 0 findings",
                goal_id,
                _SATURATION_WINDOW,
            )
        return saturated
    except Exception as exc:
        logger.debug("Saturation check skipped: %s", exc)
        return False


def _deduplicate_findings(findings: list[dict]) -> list[dict]:
    """Remove semantically redundant findings using aura_memory search.

    For each finding we query the brain for near-identical content already
    stored in this session.  If similarity exceeds the threshold we skip
    storing it again.  Falls back to exact-text dedup if brain is unavailable.
    """
    if not findings:
        return findings

    # Fast path: exact-text dedup (always safe, zero cost)
    seen_summaries: set[str] = set()
    unique: list[dict] = []
    for f in findings:
        key = (f.get("summary") or f.get("text") or "").strip().lower()[:120]
        if key and key in seen_summaries:
            logger.debug("Exact-dedup: skipping duplicate finding: %s…", key[:60])
            continue
        if key:
            seen_summaries.add(key)
        unique.append(f)

    # Semantic dedup via aura brain search (best-effort)
    try:
        from remy.core.agent_tools import brain, brain_lock

        SIMILARITY_THRESHOLD = 0.82  # cosine similarity — tune if needed
        semantic_unique: list[dict] = []
        with brain_lock:
            for f in unique:
                text = (f.get("summary") or f.get("text") or "").strip()
                if not text or len(text) < 20:
                    semantic_unique.append(f)
                    continue
                results = brain.search(text, top_k=1, tags="research-finding")
                if results:
                    top = results[0]
                    score = float(getattr(top, "score", 0) or top.get("score", 0) if isinstance(top, dict) else getattr(top, "score", 0))
                    if score >= SIMILARITY_THRESHOLD:
                        logger.debug(
                            "Semantic-dedup: skipping finding (score=%.2f): %s…", score, text[:60]
                        )
                        continue
                semantic_unique.append(f)
        removed = len(unique) - len(semantic_unique)
        if removed:
            logger.info("Semantic dedup removed %d redundant findings", removed)
        return semantic_unique
    except Exception as exc:
        logger.debug("Semantic dedup skipped (brain unavailable): %s", exc)
        return unique


async def run_research_worker(
    goal: dict,
    session_id: str,
    session_log: list,
    history: list | None = None,
    current_plan: object | None = None,
) -> WorkerExecutionResult:
    """Run the specialized research worker via the existing worker engine."""
    from remy.core.worker import WorkerTask, execute_single_worker

    desc = (goal or {}).get("description", "") or ""
    task_action = (goal or {}).get("task_action", "") or ""
    instruction = task_action if task_action else desc
    goal_id = _resolve_goal_id(goal)

    from remy.core.capability_packs import resolve_research_config
    from remy.core.research_sessions import (
        append_queries,
        get_or_create_research_session,
        get_research_session_summary,
        load_research_session,
        mark_session_completed,
        reconcile_session_sources,
        record_contradictions,
        record_finding,
        record_run_findings_count,
        record_source_decision,
        record_source_fetch,
    )

    config = resolve_research_config(goal)
    mode = str(config.get("research_mode", "balanced") or "balanced")
    planned_queries = _plan_research_queries(goal, config)

    # Saturation guard: skip execution if recent cycles produced nothing new
    if _check_saturation(goal_id or session_id):
        logger.info("Research worker skipping execution — goal %s is saturated", goal_id)
        return WorkerExecutionResult(
            worker="research_worker",
            status="saturated",
            response_text="Research saturated: last cycles added no new findings. Stopping early.",
            history=history or [],
            session_log=[],
            evidence={"saturation": True, "findings_count": 0},
            tool_calls=[],
        )

    session, resumed = get_or_create_research_session(
        goal_id=goal_id or session_id,
        pack_id=str(config.get("pack_id", "market_research")),
        topic=_goal_topic(goal),
        research_mode=str(config.get("research_mode", "balanced")),
        source_scope=str(config.get("source_scope", "web")),
        source_domains=list(config.get("source_domains", []) or []),
        citation_required=bool(config.get("citation_required", True)),
        warnings=list(config.get("warnings", []) or []),
    )

    # Crок 4: Adaptive query refinement — inject queries from prior outcome evaluation
    if resumed:
        refinement_from_session = [
            w[len("[refinement] "):].strip()
            for w in (getattr(session, "warnings", []) or [])
            if w.startswith("[refinement] ")
        ]
        if refinement_from_session:
            logger.info(
                "Adaptive refinement: injecting %d queries from prior outcome evaluation",
                len(refinement_from_session),
            )
            # Prepend refinement queries so they take priority over auto-planned ones
            mode_limit = _MODE_QUERY_LIMITS.get(mode, 4)
            combined = (refinement_from_session + planned_queries)[:mode_limit]
            planned_queries = combined

    if planned_queries:
        append_queries(session.goal_id, planned_queries)

    # Clear consumed refinement queries from session warnings so they don't repeat
    if resumed and any(w.startswith("[refinement] ") for w in (getattr(session, "warnings", []) or [])):
        try:
            from remy.core.research_sessions import save_research_session as _srs, load_research_session as _lrs2

            _s2 = _lrs2(session.goal_id)
            if _s2:
                _s2.warnings = [w for w in (_s2.warnings or []) if not w.startswith("[refinement] ")]
                _srs(_s2)
        except Exception as _ce:
            logger.debug("Could not clear refinement tags: %s", _ce)

    # Build context from goal metadata
    resume_context = (goal or {}).get("resume_context", "") or ""
    blocked_reason = (goal or {}).get("blocked_reason", "") or ""
    context_parts = []
    if resume_context:
        context_parts.append(f"Resume from: {resume_context}")
    if blocked_reason:
        context_parts.append(f"Previous blocker: {blocked_reason}")

    task = WorkerTask(
        role="osint",
        instruction=instruction,
        context="\n".join(context_parts),
    )

    effective_step_budget = int(goal.get("_pack_step_budget", 0) or 0)
    mode_step_budget = _MODE_STEP_BUDGETS.get(mode, 8)
    if effective_step_budget <= 0:
        effective_step_budget = mode_step_budget
    else:
        # Pack budget wins — it's the operator-level config; mode is just a hint
        effective_step_budget = max(effective_step_budget, mode_step_budget)

    effective_timeout = int(goal.get("_pack_timeout_sec", 0) or 0)
    mode_timeout = _MODE_TIMEOUT_LIMITS.get(mode, 60)
    if effective_timeout <= 0:
        effective_timeout = mode_timeout
    else:
        # Pack timeout wins — mode timeout is a lower bound, not a cap
        effective_timeout = max(effective_timeout, mode_timeout)

    result = await execute_single_worker(
        task=task,
        session_id=session_id,
        channel=RESEARCH_WORKER_CHANNEL,
        step_budget=effective_step_budget,
        timeout_override=effective_timeout,
    )

    # Convert worker.WorkerResult → WorkerExecutionResult with research evidence
    worker_session_log = list(result.session_log or [])
    evidence = _extract_research_evidence(worker_session_log, result.output)
    status = _derive_research_status(worker_session_log, result.output)
    accepted_sources, rejected_sources = _rerank_sources(
        goal, evidence.get("source_records", []), config
    )
    if not accepted_sources and evidence.get("sources"):
        accepted_sources = [
            {"url": url, "title": "", "snippet": "", "score": 1}
            for url in list(dict.fromkeys(evidence.get("sources", [])))[
                : _MODE_SOURCE_LIMITS.get(mode, 8)
            ]
        ]
    for source in evidence.get("source_records", []):
        record_source_fetch(session.goal_id, source)
    for url in evidence.get("sources", []):
        record_source_fetch(session.goal_id, {"url": url, "title": "", "snippet": ""})
    for source in accepted_sources:
        record_source_decision(session.goal_id, source, accepted=True)
    for source in rejected_sources:
        record_source_decision(
            session.goal_id, source, accepted=False, reason=source.get("reason", "rejected")
        )
    raw_findings = evidence.get("findings", [])
    deduped_findings = _deduplicate_findings(raw_findings)
    evidence["findings"] = deduped_findings
    evidence["findings_count"] = len(deduped_findings)
    evidence["findings_deduped"] = len(raw_findings) - len(deduped_findings)
    for finding in deduped_findings:
        record_finding(session.goal_id, finding)
    # Track per-run finding count for saturation detection
    record_run_findings_count(session.goal_id, len(deduped_findings))
    raw_contradictions = list(evidence.get("contradictions") or [])
    if raw_contradictions:
        enriched = []
        for idx, c in enumerate(raw_contradictions):
            entry = dict(c) if isinstance(c, dict) else {"summary": str(c)}
            if not entry.get("id"):
                entry["id"] = f"auto-{idx}"
            enriched.append(entry)
        record_contradictions(session.goal_id, enriched)

    # Outcome evaluation: ask LLM whether the research goal is sufficiently answered
    # Pass real contradiction objects so the evaluator can generate tie-breaker queries
    all_contradictions = list(evidence.get("contradictions") or [])
    try:
        from remy.core.research_sessions import load_research_session as _lrs_c
        _sess_c = _lrs_c(session.goal_id)
        if _sess_c and _sess_c.contradictions:
            # Merge session-level contradictions (may have richer structure)
            seen_ids = {c.get("id") for c in all_contradictions if c.get("id")}
            for sc in _sess_c.contradictions:
                if sc.get("id") not in seen_ids:
                    all_contradictions.append(sc)
    except Exception:
        pass

    outcome = await _evaluate_research_outcome(
        goal=goal,
        findings=deduped_findings,
        contradictions=all_contradictions,
    )
    evidence["outcome_evaluation"] = outcome
    # Store open_questions and refinement_queries in the session for Крок 4
    if outcome.get("open_questions") or outcome.get("refinement_queries"):
        try:
            from remy.core.research_sessions import save_research_session, load_research_session as _lrs

            _sess = _lrs(session.goal_id)
            if _sess:
                _sess.warnings = list(dict.fromkeys(
                    list(_sess.warnings or []) + [
                        f"[open] {q}" for q in (outcome.get("open_questions") or [])[:3]
                    ]
                ))
                # Store refinement queries so next cycle can pick them up
                refinement = list(outcome.get("refinement_queries") or [])
                if refinement:
                    existing_rq = [w for w in _sess.warnings if w.startswith("[refinement] ")]
                    for rq in refinement:
                        tag = f"[refinement] {rq}"
                        if tag not in existing_rq:
                            _sess.warnings.append(tag)
                save_research_session(_sess)
        except Exception as _e:
            logger.debug("Could not persist outcome evaluation to session: %s", _e)

    # If outcome says research IS answered with high confidence → upgrade status to completed
    if (
        outcome.get("answered")
        and outcome.get("confidence") == "high"
        and status in ("findings_collected", "partial_progress")
    ):
        logger.info(
            "Outcome evaluation: research goal answered with high confidence — upgrading status to completed"
        )
        status = "completed"

    # Reconcile worker engine status with research-level status.
    # Key insight: a timeout with stored findings is partial_progress, not a full failure.
    if result.status == "timeout":
        if status in ("findings_collected", "completed"):
            status = "partial_progress"
            logger.info(
                "Research worker timed out but had %d findings — marking partial_progress",
                evidence.get("findings_count", 0),
            )
        else:
            status = "timeout"
    elif result.status == "error":
        status = "error"
    refreshed_session = (
        reconcile_session_sources(session.goal_id)
        or load_research_session(session.goal_id)
        or session
    )
    threshold = _extract_completion_threshold(goal)
    total_findings = len(getattr(refreshed_session, "findings", []) or [])
    artifact_id = str(
        getattr(refreshed_session, "final_artifact_id", "")
        or evidence.get("final_artifact_id", "")
        or ""
    )

    if (
        status in ("completed", "findings_collected", "partial_progress")
        and total_findings >= threshold
    ):
        if not artifact_id:
            artifact_id = _store_research_summary_artifact(
                goal, refreshed_session.summary(), list(refreshed_session.findings or [])
            )
        mark_session_completed(session.goal_id, final_artifact_id=artifact_id)
        refreshed_session = load_research_session(session.goal_id) or refreshed_session
        status = "completed"
    elif status == "completed":
        mark_session_completed(
            session.goal_id, final_artifact_id=str(evidence.get("final_artifact_id", ""))
        )
        refreshed_session = load_research_session(session.goal_id) or refreshed_session

    session_summary = get_research_session_summary(session.goal_id) or refreshed_session.summary()
    accepted_count = int(session_summary.get("accepted_sources_count", 0) or 0)
    rejected_count = int(session_summary.get("rejected_sources_count", 0) or 0)
    findings_count = int(session_summary.get("findings_count", 0) or 0)
    citation_coverage_rate = accepted_count / findings_count if findings_count else 0.0
    refreshed_accepted = list(getattr(refreshed_session, "accepted_sources", []) or [])
    refreshed_rejected = list(getattr(refreshed_session, "rejected_sources", []) or [])
    evidence.update(
        {
            "research_mode": mode,
            "source_scope": config.get("source_scope", "web"),
            "source_domains": list(config.get("source_domains", []) or []),
            "citation_required": bool(config.get("citation_required", True)),
            "planned_queries_count": len(planned_queries),
            "step_budget": effective_step_budget,
            "timeout_sec": effective_timeout,
            "accepted_sources_count": accepted_count,
            "rejected_sources_count": rejected_count,
            "citation_coverage_rate": round(citation_coverage_rate, 3),
            "accepted_sources": refreshed_accepted[:5]
            if refreshed_accepted
            else accepted_sources[:5],
            "rejected_sources": refreshed_rejected[:5]
            if refreshed_rejected
            else rejected_sources[:5],
            "final_artifact_id": str(
                getattr(refreshed_session, "final_artifact_id", "") or artifact_id
            ),
            "research_session": {
                **session_summary,
                "resumed": resumed,
            },
        }
    )
    _store_research_source_consequences(
        evidence=evidence,
        status=status,
        session_id=session_id,
    )

    return WorkerExecutionResult(
        worker="research_worker",
        status=status,
        response_text=result.output,
        history=history or [],
        session_log=worker_session_log,
        evidence=evidence,
        tool_calls=result.tool_calls,
    )
