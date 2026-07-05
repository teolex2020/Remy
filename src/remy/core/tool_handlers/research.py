"""
Research Orchestrator Handlers — structured multi-query research projects.

Handles start_research, add_research_finding, complete_research, and
active project queries. Uses LLM for plan generation and synthesis.
"""

import json
import logging
import re
import uuid
from datetime import datetime

from remy.core.source_credibility import credibility_scorer

logger = logging.getLogger("BrainTools")


def _get_brain():
    """Lazy accessor — reads brain from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain


def _get_brain_lock():
    """Lazy accessor — reads brain_lock from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain_lock


_RESEARCH_PROJECT_TAG = "research-project"
_RESEARCH_FINDING_TAG = "research-finding"
_DEPTH_QUERY_COUNT = {"quick": 2, "standard": 4, "deep": 7}


def _build_cited_markdown_report(
    *,
    topic: str,
    summary: str,
    findings: list[dict],
    unique_sources: list[str],
    evidence_note: str = "",
) -> tuple[str, list[dict]]:
    citation_index: dict[str, int] = {
        url: idx for idx, url in enumerate(unique_sources, start=1) if url
    }
    citations = [
        {"id": f"S{idx}", "url": url}
        for url, idx in citation_index.items()
    ]

    lines = [
        f"# {topic}",
        "",
        "## Executive Summary",
        summary.strip(),
    ]
    if evidence_note:
        lines.extend(["", f"> {evidence_note.strip()}"])

    lines.extend(["", "## Key Findings"])
    for finding in findings:
        content = str(finding.get("content", "") or "").strip()
        if not content:
            continue
        source_url = str(finding.get("source_url", "") or "").strip()
        confidence = finding.get("confidence")
        line = f"- {content}"
        citation_marker = ""
        if source_url and source_url in citation_index:
            citation_marker = f" [S{citation_index[source_url]}]"
        if confidence is not None:
            try:
                confidence_text = f"{float(confidence):.2f}"
                line = f"{line}{citation_marker} (confidence {confidence_text})"
            except Exception:
                line = f"{line}{citation_marker}"
        else:
            line = f"{line}{citation_marker}"
        lines.append(line)

    lines.extend(["", "## Sources"])
    if unique_sources:
        for source_url in unique_sources:
            lines.append(f"- [S{citation_index[source_url]}] {source_url}")
    else:
        lines.append("- No accepted source URLs were attached.")

    return "\n".join(lines).strip(), citations


def _get_research_project(project_id: str):
    """Load a research project record by its project_id metadata field."""
    brain = _get_brain()
    brain_lock = _get_brain_lock()

    # Primary: tag-based search
    with brain_lock:
        records = brain.search(query="", tags=[_RESEARCH_PROJECT_TAG], limit=100)
        for rec in records:
            meta = rec.metadata or {}
            if meta.get("project_id") == project_id:
                return rec

        # Fallback: content-based search (in case tag search missed it)
        records2 = brain.search(query=project_id, limit=10)
        for rec in records2:
            meta = rec.metadata or {}
            if meta.get("project_id") == project_id:
                return rec

    logger.warning(
        "Research project '%s' not found. Tag search returned %d records.", project_id, len(records)
    )
    return None




def _canonicalize_source_url(url: str) -> str:
    url = (url or "").strip().rstrip("/ ")
    return url.lower()


def _source_url_is_anchored(session_id: str | None, source_url: str) -> bool:
    if not session_id or not source_url:
        return False
    try:
        from remy.core.claim_provenance import get_turn_fetch_evidence
        target = _canonicalize_source_url(source_url)
        for item in get_turn_fetch_evidence(session_id):
            if _canonicalize_source_url(str(item.get("url") or "")) == target:
                return True
    except Exception as e:
        logger.debug("Could not inspect turn fetch evidence for %s: %s", source_url, e)
    return False


def get_active_research_projects() -> list[dict]:
    """Return active (non-complete) research projects as dicts for decision prompt."""
    brain = _get_brain()
    records = brain.search(query="", tags=[_RESEARCH_PROJECT_TAG], limit=20)
    projects = []
    for rec in records:
        meta = rec.metadata or {}
        if meta.get("status") in ("complete", "abandoned"):
            continue
        projects.append(
            {
                "project_id": meta.get("project_id", ""),
                "topic": meta.get("topic", ""),
                "status": meta.get("status", "planning"),
                "depth": meta.get("depth", "standard"),
                "queries_total": len(meta.get("query_plan", [])),
                "queries_done": meta.get("queries_done", 0),
                "findings_count": meta.get("findings_count", 0),
            }
        )
    return projects


def _start_research(args: dict, session_id: str | None = None, channel: str | None = None) -> str:
    """Create a research project with an LLM-generated query plan."""
    from remy.core.agent_tools import Level
    from remy.core.provenance import _stamp_provenance

    brain = _get_brain()
    brain_lock = _get_brain_lock()

    topic = str(
        args.get("topic")
        or args.get("question")
        or args.get("query")
        or args.get("prompt")
        or args.get("description")
        or ""
    ).strip()
    if not topic:
        return json.dumps({"error": "start_research requires topic, question, query, prompt, or description."})
    depth = args.get("depth", "standard").strip().lower()
    context = args.get("context", "").strip()
    research_mode = args.get("research_mode", "").strip().lower()
    source_scope = args.get("source_scope", "").strip().lower() or "web"
    source_domains = args.get("source_domains", []) or []
    citation_required = bool(args.get("citation_required", False))

    if depth not in _DEPTH_QUERY_COUNT:
        depth = "standard"
    query_count = _DEPTH_QUERY_COUNT[depth]

    project_id = f"rp-{uuid.uuid4().hex[:12]}"
    topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]

    # Check existing knowledge
    existing_knowledge = ""
    try:
        with brain_lock:
            recall_result = brain.recall(topic, token_budget=512)
        if recall_result and "No relevant" not in recall_result:
            existing_knowledge = recall_result[:300]
    except Exception:
        pass

    # Generate research plan via LLM
    plan_prompt = (
        "You are a research planner. Generate search queries for a research project.\n"
        f"Respond ONLY with a JSON array of {query_count} search query strings.\n\n"
        f"TOPIC: {topic}\n"
    )
    if context:
        plan_prompt += f"CONTEXT: {context}\n"
    if existing_knowledge:
        plan_prompt += f"EXISTING KNOWLEDGE: {existing_knowledge}\n"
    plan_prompt += (
        f"\nGenerate exactly {query_count} specific, diverse search queries "
        "that will cover the topic comprehensively. Respond with Valid JSON Array of strings only.\n"
        'Example: ["query 1", "query 2"]\n'
        "Do NOT use single quotes. Do NOT include markdown formatting."
    )

    query_plan = []
    try:
        from remy.core.llm import call_llm

        result = call_llm(plan_prompt, purpose="research_plan")
        raw = result.content
        if isinstance(raw, list):
            raw = " ".join(str(c) for c in raw)
        raw = str(raw).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        # Cleanup: remove JSON prefix if present
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

        # Cleanup: replace single quotes if it looks like a python list
        if raw.startswith("['") and "']" in raw:
            try:
                import ast

                # safe usage for literals
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    query_plan = [str(q).strip() for q in parsed[:query_count] if str(q).strip()]
            except Exception:
                pass

        if not query_plan:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    query_plan = [str(q).strip() for q in parsed[:query_count] if str(q).strip()]
            except json.JSONDecodeError:
                pass  # Will fall back to topic below
    except Exception as e:
        logger.warning("Research plan generation failed, using topic as query: %s", e)

    # Fallback: use topic itself as queries
    if not query_plan:
        query_plan = [topic]

    # Pad if LLM returned fewer queries than requested
    if len(query_plan) < query_count:
        logger.info(
            "Research plan: LLM returned %d/%d queries, padding with variations",
            len(query_plan),
            query_count,
        )
        suffixes = [
            "latest developments",
            "best practices",
            "comparisons and alternatives",
            "implementation guides",
            "case studies",
            "technical challenges",
        ]
        for suffix in suffixes:
            if len(query_plan) >= query_count:
                break
            variant = f"{topic} {suffix}"
            if variant not in query_plan:
                query_plan.append(variant)

    # Store project in brain
    content = f"Research Project: {topic}\nDepth: {depth}\nQueries: {len(query_plan)}"
    with brain_lock:
        rec = brain.store(
            content=content,
            level=Level.DOMAIN,
            tags=[_RESEARCH_PROJECT_TAG, topic_slug],
            metadata=_stamp_provenance(
                {
                    "type": "research_project",
                    "project_id": project_id,
                    "topic": topic,
                    "depth": depth,
                    "research_mode": research_mode,
                    "source_scope": source_scope,
                    "source_domains": source_domains,
                    "citation_required": citation_required,
                    "status": "researching",
                    "query_plan": query_plan,
                    "queries_done": 0,
                    "findings_count": 0,
                    "finding_ids": [],
                    "started_at": datetime.now().isoformat(),
                    # D-04: structural project metadata record, not a knowledge claim.
                    "admission_class": "research_project",
                },
                channel,
                tags=[_RESEARCH_PROJECT_TAG, topic_slug],
            ),
        )

    return json.dumps(
        {
            "created": True,
            "project_id": project_id,
            "record_id": rec.id,
            "topic": topic,
            "depth": depth,
            "research_mode": research_mode,
            "source_scope": source_scope,
            "source_domains": source_domains,
            "citation_required": citation_required,
            "query_plan": query_plan,
            "queries_total": len(query_plan),
        },
        ensure_ascii=False,
    )


def _add_research_finding(
    args: dict, session_id: str | None = None, channel: str | None = None
) -> str:
    """Record a finding and attach it to a research project."""
    from remy.core.ingestion import ingest_grounded_evidence
    from remy.core.provenance import _stamp_provenance
    from remy.core.tool_utils import _check_duplicates

    brain = _get_brain()
    brain_lock = _get_brain_lock()

    project_id = args["project_id"].strip()
    # Fallback: LLM sometimes sends "summary" instead of "content"
    content = (args.get("content") or args.get("summary") or "").strip()
    if not content:
        return json.dumps({"error": "Missing required field 'content' (the finding text)"})
    source_url = args.get("source_url", "").strip()
    if not source_url:
        return json.dumps({
            "error": "add_research_finding requires source_url. Candidate discovery alone is not enough - fetch the chosen source first."
        }, ensure_ascii=False)

    # RM-2: Apply credibility default if not provided
    if "confidence" not in args and source_url:
        args["confidence"] = credibility_scorer.get_score(source_url)

    confidence = float(args.get("confidence", 0.7))
    contradicts_id = args.get("contradicts_finding_id", "").strip()

    # Find the project
    with brain_lock:
        project_rec = _get_research_project(project_id)
        if not project_rec:
            return json.dumps({"error": f"Research project '{project_id}' not found"})

        project_meta = dict(project_rec.metadata or {})
        topic = project_meta.get("topic", "research")

        topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]

        # Dedup check
        existing = _check_duplicates(content[:100], tags=[_RESEARCH_FINDING_TAG])

        # Build finding content
        finding_content = f"Research finding ({topic}): {content}"
        if source_url:
            finding_content += f"\nSource: {source_url}"

        # Phase 2: canonical grounded ingestion.
        ingestion = ingest_grounded_evidence(
            content=finding_content,
            source_url=source_url,
            session_id=session_id or "",
            channel=channel,
            extract_class="grounded_external_fact",
            extra_tags=[_RESEARCH_FINDING_TAG, topic_slug],
            extra_meta={
                "type": "research_finding",
                "project_id": project_id,
                "timestamp": datetime.now().isoformat(),
            },
            confidence=confidence,
        )
        if not ingestion.admitted:
            return json.dumps({
                "error": ingestion.reason,
                "source_url": source_url,
            }, ensure_ascii=False)

        rec = brain.store(
            content=ingestion.content,
            level=ingestion.level,
            tags=ingestion.tags,
            metadata=_stamp_provenance(
                ingestion.metadata, channel, tags=ingestion.tags,
            ),
        )

        # Connect to project (promotion-gated)
        from remy.core.agent_tools import gated_connect
        gated_connect(brain, rec.id, project_rec.id, weight=0.8)

        # Handle explicit contradiction
        if contradicts_id and contradicts_id != rec.id:
            contradicting_rec = brain.get(contradicts_id)
            if contradicting_rec:
                gated_connect(brain, rec.id, contradicts_id, weight=0.3)

    # Auto-detect contradictions against prior findings
    try:
        from remy.core.research_memory import check_finding_contradictions

        auto_contradictions = check_finding_contradictions(
            new_content=content,
            new_source=source_url,
            new_finding_id=rec.id,
            topic=topic,
            project_id=project_id,
        )
    except Exception as e:
        logger.debug("Contradiction check failed: %s", e)
        auto_contradictions = []

    with brain_lock:
        # Update project metadata
        finding_ids = project_meta.get("finding_ids", [])
        finding_ids.append(rec.id)
        project_meta["finding_ids"] = finding_ids
        project_meta["findings_count"] = len(finding_ids)
        brain.update(project_rec.id, metadata=project_meta)

    result = {
        "stored": True,
        "finding_id": rec.id,
        "project_id": project_id,
        "findings_count": len(finding_ids),
    }
    if existing:
        result["duplicate_warning"] = existing
    if contradicts_id:
        result["contradicts"] = contradicts_id
    if auto_contradictions:
        result["auto_contradictions"] = [
            {"finding_id": c["finding_id"], "score": c["score"]} for c in auto_contradictions[:3]
        ]

    return json.dumps(result, ensure_ascii=False)


def _complete_research(
    args: dict, session_id: str | None = None, channel: str | None = None
) -> str:
    """Synthesize all findings into a final report and mark project complete."""
    from remy.core.agent_tools import Level
    from remy.core.provenance import _stamp_provenance
    from remy.core.verification_gate import (
        emit_verification_incident,
        resolve_verification_incident,
        run_research_completion_verification_gate,
    )

    brain = _get_brain()
    brain_lock = _get_brain_lock()

    project_id = args["project_id"].strip()

    project_rec = _get_research_project(project_id)
    if not project_rec:
        return json.dumps({"error": f"Research project '{project_id}' not found"})

    project_meta = dict(project_rec.metadata or {})
    topic = project_meta.get("topic", "research")
    finding_ids = project_meta.get("finding_ids", [])

    # Gather findings
    findings = []
    sources = []
    total_confidence = 0.0
    with brain_lock:
        for fid in finding_ids:
            frec = brain.get(fid)
            if not frec:
                continue
            fmeta = frec.metadata or {}
            findings.append(
                {
                    "content": frec.content,
                    "source_url": fmeta.get("source_url", ""),
                    "source_anchored": bool(fmeta.get("source_anchored")),
                    "confidence": fmeta.get("confidence", 0.7),
                }
            )
            total_confidence += fmeta.get("confidence", 0.7)
            if fmeta.get("source_url"):
                sources.append(fmeta["source_url"])

    if not findings:
        return json.dumps({"error": "No findings to synthesize"})

    invalid_findings = [
        {
            "source_url": str(f.get("source_url") or ""),
            "source_anchored": bool(f.get("source_anchored")),
        }
        for f in findings
        if not f.get("source_url") or not f.get("source_anchored")
    ]
    if invalid_findings:
        return json.dumps({
            "error": "complete_research requires every finding to have an anchored source_url from fetched evidence.",
            "invalid_findings": invalid_findings[:10],
        }, ensure_ascii=False)

    avg_confidence = round(total_confidence / len(findings), 2)
    unique_sources = list(dict.fromkeys(sources))
    citation_complete = bool(unique_sources)
    if not citation_complete:
        return json.dumps({
            "error": "complete_research requires accepted source URLs on findings. Fetch evidence first, then attach source_url on each finding."
        }, ensure_ascii=False)
    evidence_note = ""

    # Synthesize via LLM
    findings_text = "\n".join(
        f"- {f['content'][:300]}"
        + (f" [confidence: {f['confidence']}]" if f["confidence"] != 0.7 else "")
        for f in findings
    )

    synth_prompt = (
        "You are synthesizing research findings into a clear, structured report.\n\n"
        f"TOPIC: {topic}\n"
        f"FINDINGS ({len(findings)} total):\n{findings_text}\n\n"
        "Write a concise research report (3-6 sentences) that:\n"
        "1. Summarizes the key findings\n"
        "2. Notes any contradictions or uncertainties\n"
        "3. Draws conclusions\n"
        "4. Uses the same language as the findings\n\n"
        "Every concrete claim should cite its supporting source URLs when available.\n"
        "If evidence is weak, say so explicitly.\n\n"
        "Report:"
    )

    report = None
    try:
        from remy.core.llm import call_llm

        result = call_llm(synth_prompt, purpose="research_synthesis")
        raw = result.content
        if isinstance(raw, list):
            raw = " ".join(str(c) for c in raw)
        report = str(raw).strip()
    except Exception as e:
        logger.warning("Research synthesis failed: %s", e)

    if not report or len(report) < 10:
        # Fallback: concatenate findings
        report = f"Research on '{topic}':\n" + "\n".join(
            f"- {f['content'][:200]}" for f in findings
        )
    if evidence_note and evidence_note not in report:
        report = f"{report}\n\n{evidence_note}"

    cited_markdown, citations = _build_cited_markdown_report(
        topic=topic,
        summary=report,
        findings=findings,
        unique_sources=unique_sources,
        evidence_note=evidence_note,
    )

    # Store final report via store_research logic
    topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]

    with brain_lock:
        report_store_meta = _stamp_provenance(
            {
                "type": "research_report",
                "artifact_format": "markdown",
                "markdown_body": cited_markdown,
                "citations": citations,
                "topic": topic,
                "project_id": project_id,
                "sources": unique_sources,
                "findings_count": len(findings),
                "confidence_avg": avg_confidence,
                "citation_complete": citation_complete,
                "citation_count": len(unique_sources),
                "evidence_note": evidence_note,
                "research_mode": project_meta.get("research_mode", ""),
                "source_scope": project_meta.get("source_scope", "web"),
                "source_domains": project_meta.get("source_domains", []),
                "learning_channel": "internet_evidence",
                # D-04: research_report is LLM-synthesized text over grounded findings.
                # The synthesis itself is not factual knowledge — it requires explicit
                # downstream promotion before it can influence durable brain state.
                "admission_class": "research_report",
                "requires_promotion": True,
                "timestamp": datetime.now().isoformat(),
            },
            channel,
            tags=["research", topic_slug],
        )
        report_rec = brain.store(
            content=cited_markdown,
            # D-04: DECISIONS level — synthesis output, not raw domain knowledge.
            # Promotion to DOMAIN requires explicit operator/user review.
            level=Level.DECISIONS,
            tags=["research", topic_slug],
            metadata=report_store_meta,
        )

    report_record_id = getattr(report_rec, "id", None) or str(report_rec or "")
    hydrated_report_rec = brain.get(report_record_id) if report_record_id else None

    pdf_artifact = {}
    pdf_result = {}
    try:
        import remy.core.brain_tools as _bt

        pdf_result = json.loads(
            _bt._generate_report(
                {
                    "title": topic,
                    "subtitle": "Research Report",
                    "content": cited_markdown,
                    "report_type": "standard",
                    "include_toc": True,
                    "metadata": {
                        "topic": topic,
                        "source_count": len(unique_sources),
                        "citation_complete": citation_complete,
                    },
                },
                session_id,
                channel,
            )
        )
        if pdf_result.get("generated"):
            pdf_artifact = {
                "pdf_url": pdf_result.get("url"),
                "pdf_filename": pdf_result.get("filename"),
                "pdf_record_id": pdf_result.get("record_id"),
            }
    except Exception as e:
        logger.warning("Research PDF render failed for %s: %s", project_id, e)

    verification = run_research_completion_verification_gate(
        project_id=project_id,
        report_record_id=report_record_id,
        stored_report_record=hydrated_report_rec,
        markdown_body=cited_markdown,
        findings_count=len(findings),
        pdf_result=pdf_result if isinstance(pdf_result, dict) else None,
    )

    if not verification.verified and verification.repair_required:
        emit_verification_incident(
            source="complete_research",
            verification=verification,
            artifact_label=topic,
            extra={"project_id": project_id},
        )
        report_meta = dict((getattr(hydrated_report_rec, "metadata", None) or report_store_meta))
        report_meta["verification"] = verification.to_dict()
        if report_record_id:
            brain.update(report_record_id, metadata=report_meta)
        return json.dumps(
            {
                "completed": False,
                "project_id": project_id,
                "topic": topic,
                "error": verification.reason,
                "verification": verification.to_dict(),
            },
            ensure_ascii=False,
        )

    # Connect report to project (promotion-gated)
    from remy.core.agent_tools import gated_connect
    gated_connect(brain, report_record_id, project_rec.id, weight=0.9)

    # Mark project complete
    project_meta["status"] = "complete"
    project_meta["completed_at"] = datetime.now().isoformat()
    project_meta["report_id"] = report_record_id
    project_meta["verification"] = verification.to_dict()
    if pdf_artifact:
        project_meta.update(pdf_artifact)
    brain.update(project_rec.id, metadata=project_meta)

    report_meta = dict((getattr(hydrated_report_rec, "metadata", None) or report_store_meta))
    report_meta["verification"] = verification.to_dict()
    if pdf_artifact:
        report_meta.update(pdf_artifact)
    brain.update(report_record_id, metadata=report_meta)
    resolve_verification_incident(
        source="complete_research",
        artifact_label=topic,
        extra={"project_id": project_id},
    )

    return json.dumps(
        {
            "completed": True,
            "project_id": project_id,
            "report_id": report_record_id,
            "topic": topic,
            "report": report[:500],
            "markdown": cited_markdown,
            "artifact_format": "markdown",
            "citations": citations,
            **pdf_artifact,
            "verification": verification.to_dict(),
            "source_count": len(unique_sources),
            "findings_count": len(findings),
            "confidence_avg": avg_confidence,
            "citation_complete": citation_complete,
            "evidence_note": evidence_note,
        },
        ensure_ascii=False,
    )
