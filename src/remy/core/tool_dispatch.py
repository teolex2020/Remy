"""
Tool Dispatch — execute_tool() entry point and the core elif dispatcher.

Routes tool calls to their handlers with circuit breaker, retry, audit trail,
health tracking, and trust enforcement.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from google.genai import types
from remy.core.scheduling import normalize_schedule_args
from remy.core.brain_tools import get_user_profile_record

logger = logging.getLogger("BrainTools")

# Channels where unsupervised storage of personal data is blocked.
# Interactive channels (desktop, telegram, voice) are NOT blocked.
_NON_INTERACTIVE_CHANNELS = frozenset(
    {
        "autonomous",
        "proactive",
        "worker-researcher",
        "worker-planner",
        "worker-executor",
        "worker-analyst",
        "worker-osint",
    }
)


def _get_bt():
    """Lazy accessor for brain_tools module (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt


def _tool_gate_situation(session_id: str | None, channel: str | None) -> str:
    return f"tool_call:{channel or 'unknown'}:{session_id or 'unknown'}"


def _tool_gate_action(name: str, args: dict | None) -> str:
    compact_args = {
        str(key): str(value)[:160]
        for key, value in sorted((args or {}).items(), key=lambda item: str(item[0]))
    }
    if not compact_args:
        return f"tool:{name}"
    return f"tool:{name}:{json.dumps(compact_args, ensure_ascii=False, sort_keys=True)}"


def _tool_result_refuted(result: str) -> bool:
    text = str(result or "")
    if text.startswith(("Error:", "Unknown tool:", "Blocked by consequence memory:")):
        return True
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("error"):
            return True
    except Exception:
        pass
    return False


def _blocked_tool_policy_hint(
    name: str,
    args: dict | None,
    session_id: str | None,
    channel: str | None,
) -> dict | None:
    try:
        from remy.core.consequence_gate import consult_policy_hint

        bt = _get_bt()
        store = getattr(bt.brain, "_aura", bt.brain)
        hint = consult_policy_hint(
            store,
            situation=_tool_gate_situation(session_id, channel),
            action=_tool_gate_action(name, args),
            namespace="remy-tools",
        )
        context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
        if context.get("hint") == "avoid" or context.get("should_block"):
            return context
    except Exception as exc:
        logger.debug("Direct tool consequence gate skipped for %s: %s", name, exc)
    return None


def _blocked_tool_response(name: str, policy_hint: dict) -> str:
    return json.dumps(
        {
            "error": (
                "Blocked by consequence memory: this exact direct tool action "
                "was previously refuted."
            ),
            "tool": name,
            "consequence_gate": {
                "blocked": True,
                "policy_hint": policy_hint,
            },
        },
        ensure_ascii=False,
    )


def _store_tool_consequence(
    name: str,
    args: dict | None,
    result: str,
    session_id: str | None,
    channel: str | None,
) -> None:
    try:
        bt = _get_bt()
        store = getattr(bt.brain, "_aura", bt.brain)
        capture = getattr(store, "capture_consequence", None)
        if capture is None:
            return

        refuted = _tool_result_refuted(result)
        capture(
            situation=_tool_gate_situation(session_id, channel),
            action=_tool_gate_action(name, args),
            consequence="REFUTES" if refuted else "SUPPORTS",
            trust=-1 if refuted else 1,
            scope=[
                "tool-call",
                "direct-tool-dispatch",
                f"tool:{name}",
                f"channel:{channel}" if channel else "channel:",
                f"session:{session_id}" if session_id else "session:",
                "result:error" if refuted else "result:ok",
            ],
            provenance=[
                "remy:tool_dispatch",
                f"session:{session_id}" if session_id else "session:",
            ],
            links={"session": session_id or "", "tool": name},
            namespace="remy-tools",
        )
    except TypeError:
        try:
            bt = _get_bt()
            store = getattr(bt.brain, "_aura", bt.brain)
            refuted = _tool_result_refuted(result)
            store.capture_consequence(
                _tool_gate_situation(session_id, channel),
                _tool_gate_action(name, args),
                "REFUTES" if refuted else "SUPPORTS",
                -1 if refuted else 1,
            )
        except Exception:
            pass
    except Exception as exc:
        logger.debug("Failed to store direct tool consequence for %s: %s", name, exc)


def _finalize_tool_result(
    name: str,
    args: dict | None,
    result: str,
    session_id: str | None,
    channel: str | None,
) -> str:
    _store_tool_consequence(name, args, result, session_id, channel)
    return result


# ============== ENTRY POINT ==============


def execute_tool(
    name: str, args: dict, session_id: str | None = None, channel: str | None = None
) -> str:
    """Execute a brain tool, sandbox meta-tool, or sandbox tool.

    Includes per-tool circuit breaker and retry with backoff for transient failures.
    All brain operations are serialized via brain_lock to protect Rust AuraMemory backend.

    Args:
        name: Tool name.
        args: Tool arguments dict.
        session_id: Session ID for co-activation tracking (per-channel).
        channel: Channel context for provenance tracking (autonomous/desktop/telegram/voice).
    """
    from remy.core.provenance import _TRUST_ENFORCED_TOOLS, _validate_action_data

    # Access handlers and _execute_tool_locked via brain_tools module
    # so tests can patch remy.core.brain_tools._handle_delegate_task etc.
    bt = _get_bt()

    policy_block = _blocked_tool_policy_hint(name, args, session_id, channel)
    if policy_block:
        return _blocked_tool_response(name, policy_block)

    # delegate_task runs OUTSIDE brain_lock — workers acquire it per-tool-call.
    # Running inside brain_lock would deadlock (orchestrator holds lock → workers need lock).
    if name == "delegate_task":
        return _finalize_tool_result(
            name,
            args,
            bt._handle_delegate_task(args, session_id, channel),
            session_id,
            channel,
        )

    # Browser tools run OUTSIDE brain_lock — async I/O + vision API calls.
    # But trust validation needs brain_lock for brain.search().
    if name in ("browse_page", "browser_act", "browser_close"):
        if name in _TRUST_ENFORCED_TOOLS:
            with bt.brain_lock:
                block_msg = _validate_action_data(name, args)
            if block_msg:
                return _finalize_tool_result(
                    name,
                    args,
                    json.dumps({"error": block_msg}),
                    session_id,
                    channel,
                )
        return _finalize_tool_result(
            name,
            args,
            bt._handle_browser_tool(name, args, session_id, channel),
            session_id,
            channel,
        )

    if name in ("scratchpad", "filter_working"):
        if hasattr(bt, "_execute_unlocked_working_memory_tool"):
            return _finalize_tool_result(
                name,
                args,
                bt._execute_unlocked_working_memory_tool(name, args, session_id, channel),
                session_id,
                channel,
            )
        return _finalize_tool_result(
            name,
            args,
            _execute_tool_inner(name, args, session_id, channel),
            session_id,
            channel,
        )

    with bt.brain_lock:
        result = bt._execute_tool_locked(name, args, session_id, channel)
    return _finalize_tool_result(name, args, result, session_id, channel)


def _execute_tool_locked(
    name: str, args: dict, session_id: str | None = None, channel: str | None = None
) -> str:
    """Inner execute_tool, called under brain_lock."""
    from remy.core.provenance import _validate_action_data
    from remy.core.tool_health import tool_health
    from remy.core.tool_utils import _NETWORK_TOOLS

    # Circuit breaker check
    if not tool_health.is_available(name):
        report = tool_health.get_health_report()
        status = report.get(name, "unavailable")
        return json.dumps({"error": f"Tool '{name}' temporarily unavailable: {status}"})

    # Trust enforcement — block actions with unverified sensitive data
    block_msg = _validate_action_data(name, args)
    if block_msg:
        return json.dumps({"error": block_msg})

    _start_ts = time.time()
    result = _execute_tool_inner(name, args, session_id, channel)

    # Audit trail for critical tools (finance, registration, identity)
    from remy.core.audit_trail import is_critical

    if is_critical(name):
        _elapsed = (time.time() - _start_ts) * 1000
        _audit_status = "success"
        _audit_error = None
        try:
            _parsed = json.loads(result)
            if isinstance(_parsed, dict) and "error" in _parsed:
                _audit_status = "error"
                _audit_error = str(_parsed["error"])
        except (json.JSONDecodeError, TypeError):
            if result.startswith("Error:"):
                _audit_status = "error"
                _audit_error = result
        from remy.core.audit_trail import get_audit_logger

        get_audit_logger().log_action(
            tool_name=name,
            tool_input=args,
            raw_output=result,
            status=_audit_status,
            execution_time_ms=_elapsed,
            channel=channel,
            error_message=_audit_error,
        )

    # Track health only for network/infra-dependent tools
    # Logic errors (not found, invalid input) should NOT trip the circuit breaker
    if name in _NETWORK_TOOLS:
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "error" in parsed:
                tool_health.record_failure(name)
            else:
                tool_health.record_success(name)
        except (json.JSONDecodeError, TypeError):
            if result.startswith("Error:"):
                tool_health.record_failure(name)
            else:
                tool_health.record_success(name)

    return result


# ============== HELPERS ==============


def _generate_image(args: dict, session_id: str | None, channel: str | None) -> str:
    """Generate image via Gemini and save to disk.

    Uses generate_content_stream with ImageConfig per official docs.
    """
    import mimetypes

    from google.genai import types as genai_types

    from remy.core.agent_tools import Level
    from remy.core.llm import get_genai_client
    from remy.core.provenance import _stamp_provenance

    bt = _get_bt()
    brain = bt.brain
    brain_lock = bt.brain_lock
    settings = bt.settings

    prompt = args["prompt"]
    client = get_genai_client()

    image_dir = Path(settings.DATA_DIR) / "generated_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=prompt)],
        ),
    ]
    config = genai_types.GenerateContentConfig(
        image_config=genai_types.ImageConfig(
            image_size="1K",
        ),
        response_modalities=["IMAGE", "TEXT"],
    )

    # Stream response — collect image data from chunks
    data_buffer = None
    mime_type = None
    text_parts = []

    for chunk in client.models.generate_content_stream(
        model="gemini-3.1-flash-image-preview",
        contents=contents,
        config=config,
    ):
        if chunk.parts is None:
            continue
        for part in chunk.parts:
            if part.inline_data and part.inline_data.data:
                data_buffer = part.inline_data.data
                mime_type = part.inline_data.mime_type
            elif part.text:
                text_parts.append(part.text)

    if not data_buffer:
        return json.dumps(
            {
                "generated": False,
                "error": " ".join(text_parts) or "No image generated",
            },
            ensure_ascii=False,
        )

    ext = mimetypes.guess_extension(mime_type) or ".png"
    filename = f"gen_{uuid.uuid4().hex[:8]}{ext}"
    filepath = image_dir / filename
    filepath.write_bytes(data_buffer)

    meta = _stamp_provenance(
        {
            "type": "generated_image",
            "prompt": prompt,
            "model": "gemini-3.1-flash-image-preview",
            "filename": filename,
        },
        channel,
        tags=["generated-image"],
    )
    with brain_lock:
        rec = brain.store(
            content=f"Generated image: {prompt}",
            level=Level.WORKING,
            tags=["generated-image"],
            metadata=meta,
        )

    url = f"/api/generated_images/{filename}"
    return json.dumps(
        {
            "generated": True,
            "filename": filename,
            "url": url,
            "record_id": rec.id,
            "prompt": prompt,
            "markdown": f"![{prompt[:80]}]({url})",
        },
        ensure_ascii=False,
    )


def _generate_report(args: dict, session_id: str | None, channel: str | None) -> str:
    """Generate a PDF report and save to disk."""
    from remy.core.agent_tools import Level
    from remy.core.provenance import _stamp_provenance
    from remy.core.report_builder import ReportBuilder
    from remy.core.verification_gate import (
        emit_verification_incident,
        resolve_verification_incident,
        run_report_verification_gate,
    )

    bt = _get_bt()
    brain = bt.brain
    brain_lock = bt.brain_lock
    settings = bt.settings

    title = args.get("title", "Report")
    subtitle = args.get("subtitle", "")
    sections = args.get("sections", [])

    # Fallback: LLM sent {"content": "# markdown..."} instead of title+sections
    if not sections and "content" in args:
        from remy.core.brain_tools import _parse_markdown_to_sections
        title, sections = _parse_markdown_to_sections(args["content"])
    from remy.core.brain_tools import _normalize_report_sections
    sections = _normalize_report_sections(sections)

    report_dir = str(Path(settings.DATA_DIR) / "reports")
    report = ReportBuilder(
        title=title,
        subtitle=subtitle,
        author="Remy AI Agent",
        output_dir=report_dir,
        report_type=args.get("report_type", "standard"),
        include_toc=bool(args.get("include_toc", True)),
        metadata=args.get("metadata") or {},
    )

    for section in sections:
        # LLM often sends "content" instead of "body" — normalize
        if "body" not in section and "content" in section:
            section["body"] = section["content"]
        # Smart type default: if title present → section, else → text
        sec_type = section.get("type") or ("section" if section.get("title") else "text")

        if sec_type == "section":
            report.add_section(
                title=section.get("title", ""),
                body=section.get("body", ""),
            )
        elif sec_type == "subsection":
            report.add_subsection(
                title=section.get("title", ""),
                body=section.get("body", ""),
            )
        elif sec_type == "text":
            report.add_text(section.get("body", ""))
        elif sec_type == "quote":
            report.add_quote(section.get("body", ""))
        elif sec_type == "findings":
            report.add_key_findings(
                findings=section.get("items", []),
                title=section.get("title", "Key Findings"),
            )
        elif sec_type == "table":
            report.add_table(
                headers=section.get("headers", []),
                rows=section.get("rows", []),
                title=section.get("title", ""),
            )
        elif sec_type == "memory":
            report.add_memory_records(
                records=section.get("records", []),
                title=section.get("title", "Memory Records"),
            )
        elif sec_type == "audit":
            report.add_audit_summary(
                audit_logs=section.get("logs", []),
                title=section.get("title", "Audit Trail"),
            )
        elif sec_type == "page_break":
            report.add_page_break()

    filepath = report.save()
    verification = run_report_verification_gate(
        filepath,
        title=title,
        section_count=len(sections),
    )
    if not verification.verified and verification.repair_required:
        emit_verification_incident(
            source="generate_report",
            verification=verification,
            artifact_label=title,
        )
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass
        return json.dumps(
            {
                "generated": False,
                "error": verification.reason,
                "title": title,
                "verification": verification.to_dict(),
            },
            ensure_ascii=False,
        )
    filename = Path(filepath).name

    # Store metadata in brain
    meta = _stamp_provenance(
        {
            "type": "generated_report",
            "title": title,
            "verification": verification.to_dict(),
            "filename": filename,
            "section_count": len(sections),
        },
        channel,
        tags=["generated-report"],
    )
    with brain_lock:
        rec = brain.store(
            content=f"Generated PDF report: {title}",
            level=Level.WORKING,
            tags=["generated-report"],
            metadata=meta,
        )
    resolve_verification_incident(
        source="generate_report",
        artifact_label=title,
        extra={"record_id": str(getattr(rec, "id", "") or "").strip()},
    )

    url = f"/api/reports/{filename}"
    markdown = f"[{title}]({url})"

    return json.dumps(
        {
            "generated": True,
            "filename": filename,
            "url": url,
            "markdown": markdown,
            "record_id": rec.id,
            "title": title,
            "verification": verification.to_dict(),
            "instruction": "Present the report using the exact markdown link above. Do NOT add any host or domain — the relative URL is correct as-is.",
        },
        ensure_ascii=False,
    )


def _generate_presentation(args: dict, session_id: str | None, channel: str | None) -> str:
    """Generate a PPTX presentation and save to disk."""
    from remy.core.agent_tools import Level
    from remy.core.provenance import _stamp_provenance
    from remy.core.presentation_builder import PresentationBuilder

    bt = _get_bt()
    brain = bt.brain
    brain_lock = bt.brain_lock
    settings = bt.settings

    title = args.get("title", "Presentation")
    subtitle = args.get("subtitle", "")
    slides = args.get("slides", [])

    # Fallback: LLM sent {"content": "# markdown..."} instead of title+slides
    if not slides and "content" in args:
        from remy.core.brain_tools import _parse_markdown_to_slides
        title, slides = _parse_markdown_to_slides(args["content"])

    pres_dir = str(Path(settings.DATA_DIR) / "presentations")
    pres = PresentationBuilder(
        title=title,
        subtitle=subtitle,
        author="Remy AI Agent",
        output_dir=pres_dir,
    )

    for slide in slides:
        # Normalize: LLM may send "content" instead of "body"
        if "body" not in slide and "content" in slide:
            slide["body"] = slide["content"]
        slide_type = slide.get("type") or ("bullets" if slide.get("items") else "section")

        if slide_type == "section":
            pres.add_section(
                title=slide.get("title", ""),
                body=slide.get("body", ""),
            )
        elif slide_type == "subsection":
            pres.add_subsection(
                title=slide.get("title", ""),
                body=slide.get("body", ""),
            )
        elif slide_type == "bullets":
            pres.add_bullets(
                title=slide.get("title", ""),
                items=slide.get("items", []),
            )
        elif slide_type == "quote":
            pres.add_quote(
                text=slide.get("body", ""),
                author=slide.get("author", ""),
            )
        elif slide_type == "table":
            pres.add_table(
                title=slide.get("title", ""),
                headers=slide.get("headers", []),
                rows=slide.get("rows", []),
            )
        elif slide_type == "divider":
            pres.add_section_divider(slide.get("title", ""))

    filepath = pres.save()
    filename = Path(filepath).name

    # Store metadata in brain
    meta = _stamp_provenance(
        {
            "type": "generated_presentation",
            "title": title,
            "filename": filename,
            "slide_count": len(slides),
        },
        channel,
        tags=["generated-presentation"],
    )
    with brain_lock:
        rec = brain.store(
            content=f"Generated PPTX presentation: {title}",
            level=Level.WORKING,
            tags=["generated-presentation"],
            metadata=meta,
        )

    url = f"/api/presentations/{filename}"
    markdown = f"[{title}]({url})"

    return json.dumps(
        {
            "generated": True,
            "filename": filename,
            "url": url,
            "markdown": markdown,
            "record_id": rec.id,
            "title": title,
            "instruction": "Present the presentation using the exact markdown link above. Do NOT add any host or domain — the relative URL is correct as-is.",
        },
        ensure_ascii=False,
    )


# ============== CORE DISPATCHER ==============


def _execute_tool_inner(
    name: str, args: dict, session_id: str | None = None, channel: str | None = None
) -> str:
    """Core tool execution logic. Called by execute_tool() with health tracking."""
    from remy.core.agent_tools import Level

    bt = _get_bt()
    brain = bt.brain
    brain_lock = bt.brain_lock
    settings = bt.settings

    from remy.core.provenance import (
        _apply_store_guard,
        _auto_protect_tags,
        _compute_effective_trust,
        _stamp_provenance,
    )
    from remy.core.tool_declarations import BRAIN_TOOLS, CORE_TOOL_NAMES, EXTENDED_TOOL_NAMES
    from remy.core.tool_handlers.facts import _extract_facts
    from remy.core.tool_handlers.metrics import (
        _event_correlate,
        _health_summary,
        _metric_summary,
        _symptom_correlate,
        _track_health_metric,
        _track_metric,
    )
    from remy.core.tool_handlers.profile import (
        _PROFILE_FIELDS,
        _format_profile_content,
        _get_agent_persona,
        person_matches_identity,
        resolve_person_identity_input,
        sanitize_profile_metadata,
        update_persona_fields,
    )
    from remy.core.tool_handlers.research import (
        _add_research_finding,
        _complete_research,
        _start_research,
    )
    from remy.core.tool_health import _MAX_RETRIES, _RETRY_DELAYS
    from remy.core.tool_registry_mgmt import (
        _sandbox_create_tool,
        _sandbox_list_tools,
        _sandbox_test_tool,
        get_registry,
    )
    from remy.core.tool_utils import (
        _cache_recall_result,
        _cache_search_result,
        _check_duplicates,
        _check_ssrf,
        _clean_tag,
        _get_cached_recall,
        _get_cached_search,
        _sanitize_tag,
        _sleep_with_jitter,
        clear_recall_cache,
    )
    from remy.core.memory_policy import (
        infer_semantic_type,
        protected_fields_for_record,
        protected_payload,
        sanitize_memory_content,
        sanitize_memory_metadata,
    )

    registry = get_registry()

    try:
        # ---- Meta-tools (selective tool loading) ----
        if name == "list_available_tools":
            extended = [
                {"name": t.name, "description": t.description[:120]}
                for t in BRAIN_TOOLS
                if t.name in EXTENDED_TOOL_NAMES
            ]
            return json.dumps({"available_tools": extended, "count": len(extended)})

        elif name == "enable_tools":
            requested = args.get("tool_names", [])
            if not requested:
                return json.dumps({"error": "No tool names provided"})
            valid = [n for n in requested if n in EXTENDED_TOOL_NAMES]
            invalid = [
                n for n in requested if n not in EXTENDED_TOOL_NAMES and n not in CORE_TOOL_NAMES
            ]
            result = {"enabled": valid}
            if invalid:
                result["unknown"] = invalid
            return json.dumps(result)

        # ---- Sandbox meta-tools ----
        elif name == "sandbox_create_tool":
            return _sandbox_create_tool(args)
        elif name == "sandbox_test_tool":
            return _sandbox_test_tool(args)
        elif name == "sandbox_list_tools":
            return _sandbox_list_tools()

        # ---- Skill package tools ----
        elif name == "export_skill":
            from remy.sandbox.skill_package import export_skill

            try:
                path = export_skill(args["name"])
                return json.dumps(
                    {
                        "exported": True,
                        "name": args["name"],
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            except (ValueError, FileNotFoundError) as e:
                return json.dumps({"exported": False, "error": str(e)})
        elif name == "import_skill":
            from remy.sandbox.skill_package import import_skill

            return json.dumps(import_skill(args["path"]))
        elif name == "browse_marketplace":
            from remy.sandbox.marketplace import browse_marketplace

            skills = browse_marketplace()
            if not skills:
                return json.dumps({"skills": [], "message": "Marketplace unavailable or empty."})
            return json.dumps({"skills": skills, "count": len(skills)})
        elif name == "install_marketplace_skill":
            from remy.sandbox.marketplace import install_from_marketplace

            return json.dumps(install_from_marketplace(args["name"]))

        # ---- Approved sandbox tools ----
        elif registry.is_sandbox_tool(name):
            return registry.execute_sandbox_tool(name, args)

        # ---- Core brain tools ----
        elif name == "recall":
            # Check in-memory cache first (zero DB cost)
            _cached_recall = _get_cached_recall(args["query"])
            if _cached_recall:
                return _cached_recall

            import time as _time

            _recall_t0 = _time.time()

            _used_unified = False
            if getattr(brain, "_has_recall_full", False):
                # Unified recall (Aura SDK ≥1.0.5) — single Rust call.
                with brain_lock:
                    brain_results = brain.recall_full(
                        args["query"],
                        top_k=15,
                        include_failures=True,
                        session_id=session_id,
                    )
                _d_total = (_time.time() - _recall_t0) * 1000
                _used_unified = True
                logger.info(
                    "recall_full: %.0fms query=%r",
                    _d_total,
                    args["query"][:60],
                )
                try:
                    from remy.core.metrics import metrics_collector

                    metrics_collector.record_recall_latency(_d_total, _d_total, 0.0, 0.0)
                except Exception:
                    pass
            if not _used_unified:
                # Legacy fallback: 3 separate calls (pre-1.0.5 SDK)
                _t1 = _time.time()
                with brain_lock:
                    brain_results = brain.recall_structured(
                        args["query"],
                        top_k=15,
                        session_id=session_id,
                    )
                _d_rrf = (_time.time() - _t1) * 1000

                _t2 = _time.time()
                seen_ids = {r.get("id") for r in (brain_results or []) if r.get("id")}
                with brain_lock:
                    search_hits = brain.search(query=args["query"], limit=10)
                for hit in search_hits:
                    if hit.id not in seen_ids:
                        brain_results.append(
                            {
                                "id": hit.id,
                                "content": hit.content,
                                "tags": list(getattr(hit, "tags", [])),
                                "metadata": getattr(hit, "metadata", {}) or {},
                                "score": 0.6,
                            }
                        )
                        seen_ids.add(hit.id)
                _d_substring = (_time.time() - _t2) * 1000

                _t3 = _time.time()
                try:
                    with brain_lock:
                        failure_hits = brain.search(
                            query=args["query"], tags=["outcome-failure"], limit=5
                        )
                    for hit in failure_hits:
                        if hit.id not in seen_ids:
                            brain_results.append(
                                {
                                    "id": hit.id,
                                    "content": hit.content,
                                    "tags": list(getattr(hit, "tags", [])),
                                    "metadata": getattr(hit, "metadata", {}) or {},
                                    "score": 0.8,
                                }
                            )
                            seen_ids.add(hit.id)
                except Exception:
                    pass
                _d_failure = (_time.time() - _t3) * 1000

                _d_total = (_time.time() - _recall_t0) * 1000
                logger.info(
                    "recall legacy: total=%.0fms (rrf=%.0fms, substring=%.0fms, failure=%.0fms) query=%r",
                    _d_total,
                    _d_rrf,
                    _d_substring,
                    _d_failure,
                    args["query"][:60],
                )
                try:
                    from remy.core.metrics import metrics_collector

                    metrics_collector.record_recall_latency(
                        _d_total, _d_rrf, _d_substring, _d_failure
                    )
                except Exception:
                    pass

            # Phase 3 Step 2: promotion/conflict/supersession gate for
            # LLM-facing recall. Applies to both unified (recall_full) and
            # legacy (recall_structured + search) paths. Filter lives at the
            # surface, not in the SDK — internal callers of recall_* keep
            # their unfiltered view.
            try:
                from remy.core.agent_tools import _apply_factual_recall_filter
                brain_results = _apply_factual_recall_filter(brain_results)
            except Exception:
                pass

            if not brain_results:
                _cache_recall_result(args["query"], "No relevant memories found.")
                return "No relevant memories found."

            lines, seen = [], set()

            # Sort brain results by effective trust (highest first)
            _now = _time.time()
            brain_results.sort(
                key=lambda r: _compute_effective_trust(r.get("metadata") or {}, _now),
                reverse=True,
            )

            # Filter out low-trust records to save tokens
            _RECALL_TRUST_THRESHOLD = 0.35
            brain_results = [
                r
                for r in brain_results
                if _compute_effective_trust(r.get("metadata") or {}, _now)
                >= _RECALL_TRUST_THRESHOLD
            ]

            # Brain first (richer metadata, higher authority)
            for r in brain_results or []:
                meta = r.get("metadata") or {}
                trust = _compute_effective_trust(meta, _now)
                # recall_structured returns source/trust at top level
                source = r.get("source") or meta.get("source") or "unknown"
                source_label = source.replace("agent-", "").replace("user-", "")
                # Age display for transparency
                age_str = ""
                ts_str = meta.get("timestamp") or meta.get("created_at", "")
                if ts_str:
                    try:
                        from datetime import datetime as _dt

                        age_d = (_dt.now() - _dt.fromisoformat(ts_str)).days
                        age_str = f" | {age_d}d"
                    except Exception:
                        pass
                # Fallback: fetch record created_at if no timestamp in metadata
                if not age_str and r.get("id"):
                    try:
                        from datetime import datetime as _dt

                        _full_rec = brain.get(r["id"])
                        if _full_rec and hasattr(_full_rec, "created_at"):
                            _created = _full_rec.created_at
                            if isinstance(_created, (int, float)):
                                age_d = int((_time.time() - _created) / 86400)
                                age_str = f" | {age_d}d"
                    except Exception:
                        pass
                full_content = r["content"]
                content = full_content[:300]
                truncated = len(full_content) > 300
                key = content[:80].lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                tag_str = f" [{', '.join(r.get('tags', []))}]" if r.get("tags") else ""
                rec_id = r.get("id", "")
                id_prefix = f"[id:{rec_id}] " if rec_id else ""
                trunc_suffix = (
                    f"...[truncated, {len(full_content)} chars — use get_full_record]"
                    if truncated
                    else ""
                )
                # Resolve cognitive/core tier label
                _tier_label = "CORE"
                _rlevel = r.get("level")
                if not _rlevel and rec_id:
                    try:
                        _full_rec = brain.get(rec_id)
                        if _full_rec:
                            _rlevel = _full_rec.level
                    except Exception:
                        pass
                if _rlevel:
                    from remy.core.agent_tools import tier_of

                    _tier_label = tier_of(_rlevel).upper()
                # Explicit verification label so agent distinguishes confirmed vs unconfirmed data
                _verified = meta.get("verified") is True
                _actionable = meta.get("actionable")
                if _verified or trust >= 0.9:
                    _trust_label = "VERIFIED"
                elif source.startswith("user") or source == "user-confirmed":
                    _trust_label = "user-stated"
                elif trust >= 0.6:
                    _trust_label = "likely"
                else:
                    _trust_label = "UNVERIFIED"
                _action_note = " NOT-ACTIONABLE" if _actionable is False else ""
                # Source type label — how the data was obtained
                # SDK 1.2.0+ returns source_type as top-level key; fallback to metadata
                _stype = r.get("source_type") or meta.get("source_type", "")
                _stype_label = (
                    f" {_stype}" if _stype in ("retrieved", "inferred", "generated") else ""
                )
                lines.append(
                    f"{id_prefix}[{_tier_label}] [{_trust_label} trust:{trust:.1f}{age_str} | {source_label}{_stype_label}]{_action_note} {content}{trunc_suffix}{tag_str}"
                )

            _recall_result = "\n".join(lines) if lines else "No relevant memories found."
            _cache_recall_result(args["query"], _recall_result)
            return _recall_result

        elif name == "store":
            tags = [_clean_tag(t) for t in args.get("tags", "").split(",") if t.strip()]
            level_map = {
                "L1_WORKING": Level.WORKING,
                "L2_DECISIONS": Level.DECISIONS,
                "L3_DOMAIN": Level.DOMAIN,
                "L4_IDENTITY": Level.IDENTITY,
            }
            level = level_map.get(args.get("level", ""), Level.DOMAIN)

            # Auto-protect: add consolidation-safe tags for sensitive content
            tags = _auto_protect_tags(args["content"], tags)

            # Human-in-the-loop: financial tags require user approval before storing
            from remy.core.approval_queue import approval_queue as _aq
            from remy.core.approval_queue import build_approval_description, needs_approval

            if needs_approval("store", args):
                description = build_approval_description("store", args)

                def _do_store():
                    with brain_lock:
                        _existing = _check_duplicates(args["content"][:100], tags=tags or None)
                        semantic_type = infer_semantic_type(
                            explicit=args.get("semantic_type"),
                            tags=tags,
                            level=level,
                        )
                        _meta = _stamp_provenance(
                            {"semantic_type": semantic_type},
                            channel,
                            tags=tags,
                        )
                        _meta.update(_apply_store_guard(args["content"], tags, channel))
                        _rec = brain.store(
                            content=args["content"],
                            level=level,
                            tags=tags,
                            metadata=_meta,
                            channel=channel,
                            semantic_type=semantic_type,
                        )
                    clear_recall_cache(args["content"])
                    _result: dict = {"stored": True, "id": _rec.id}
                    if _existing:
                        _result["similar_existing"] = _existing
                        _result["note"] = (
                            "Similar memory already exists. Stored anyway (may auto-merge)."
                        )
                    return json.dumps(_result, ensure_ascii=False)

                return _aq.request_approval_sync(
                    description,
                    _do_store,
                    tool_name="store",
                    tool_args=args,
                )

            # Check for similar existing content
            with brain_lock:
                existing = _check_duplicates(args["content"][:100], tags=tags or None)

                semantic_type = infer_semantic_type(
                    explicit=args.get("semantic_type"),
                    tags=tags,
                    level=level,
                )
                store_meta = _stamp_provenance(
                    {"semantic_type": semantic_type},
                    channel,
                    tags=tags,
                )
                store_meta.update(_apply_store_guard(args["content"], tags, channel))
                rec = brain.store(
                    content=args["content"],
                    level=level,
                    tags=tags,
                    metadata=store_meta,
                    channel=channel,
                    semantic_type=semantic_type,
                )
            clear_recall_cache(args["content"])

            result = {"stored": True, "id": rec.id}
            if store_meta.get("actionable") is False:
                result["actionable"] = False
                result["warning"] = (
                    "This record contains sensitive data and was stored as NOT actionable. "
                    "It cannot be used for external actions until verified by the user. "
                    "Use verify_record to mark it as verified after user confirmation."
                )
            if existing:
                result["similar_existing"] = existing
                result["note"] = "Similar memory already exists. Stored anyway (may auto-merge)."
            return json.dumps(result, ensure_ascii=False)

        elif name in {"search", "search_exact"}:
            from remy.core.hybrid_search import hybrid_search_structured, search_exact_structured

            query = args.get("query", "") or ""
            tags = [_clean_tag(t) for t in args.get("tags", "").split(",") if t.strip()] or None
            if name == "search_exact":
                with brain_lock:
                    results = search_exact_structured(
                        brain,
                        query,
                        tags=tags,
                        top_k=10,
                        lexical_limit=10,
                    )
            elif not query and tags:
                # Tag-only search — use brain.search() which supports tag filtering
                with brain_lock:
                    tag_results = brain.search(query="", tags=tags, limit=10)
                results = [
                    {
                        "id": r.id,
                        "content": sanitize_memory_content(r.content, metadata=r.metadata, tags=list(r.tags or [])),
                        "tags": list(r.tags),
                        "score": 1.0,
                        "metadata": sanitize_memory_metadata(r.metadata, tags=list(r.tags or [])),
                        "level": str(getattr(r, "level", "")),
                        "source": (r.metadata or {}).get("source"),
                    }
                    for r in tag_results
                ]
            else:
                with brain_lock:
                    results = hybrid_search_structured(
                        brain,
                        query,
                        tags=tags,
                        top_k=10,
                        min_strength=0.05,
                        session_id=session_id,
                    )
            if tags:
                results = [r for r in results if any(t in r.get("tags", []) for t in tags)]
            if not results:
                return "No results found."
            import time as _time

            items = []
            for r in results[:5]:
                meta = sanitize_memory_metadata(r.get("metadata") or {}, tags=r.get("tags", []))
                item = {
                    "id": r["id"],
                    "content": sanitize_memory_content(
                        r.get("content", ""),
                        metadata=r.get("metadata") or {},
                        tags=r.get("tags", []),
                    )[:200],
                    "tags": r.get("tags", []),
                    "level": r.get("level"),
                    "metadata": meta,
                    "score": round(r.get("score", 0), 3),
                    "trust": _compute_effective_trust(meta, _time.time()),
                    "source": r.get("source") or meta.get("source"),
                }
                if r.get("importance") is not None:
                    item["importance"] = round(r["importance"], 4)
                items.append(item)
            return json.dumps(items, ensure_ascii=False)

        elif name == "store_person":
            # Guard: autonomous/worker channels cannot fabricate people
            if channel in _NON_INTERACTIVE_CHANNELS:
                return json.dumps(
                    {
                        "error": "store_person is only available in interactive channels. "
                        "Person records must be created from explicit user conversation."
                    },
                    ensure_ascii=False,
                )
            full_name = str(args.get("full_name") or args.get("name") or "").strip()
            if not full_name:
                return json.dumps(
                    {"error": "store_person requires full_name or name."},
                    ensure_ascii=False,
                )
            role = str(args.get("role") or "").strip()
            birth_date = str(args.get("birth_date") or "").strip()
            birth_place = str(args.get("birth_place") or "").strip()

            # Check for existing person with similar name
            _profile_rec = get_user_profile_record()
            family_text = str((_profile_rec.metadata or {}).get("family", "") or "") if _profile_rec else ""
            with brain_lock:
                resolved = resolve_person_identity_input(full_name, role, birth_date, family_text)
                canonical_name = str(resolved.get("full_name") or full_name).strip()
                aliases = list(resolved.get("aliases") or [])
                existing = _check_duplicates(full_name, tags=["person"])
                existing_people = brain.search(query="", tags=["person"], limit=50)
                existing_rec = None
                for person_rec in existing_people:
                    if person_matches_identity(
                        person_rec.metadata or {},
                        full_name,
                        role,
                        birth_date,
                        family_text,
                    ):
                        existing_rec = person_rec
                        break

                parts = [canonical_name]
                if role:
                    parts.append(role)
                if birth_date:
                    parts.append(f"born {birth_date}")
                if birth_place:
                    parts.append(f"in {birth_place}")

                tags = ["person"]
                if role:
                    tags.append(_sanitize_tag(role))

                metadata = _stamp_provenance(
                    {
                        "type": "person",
                        "full_name": canonical_name,
                        "role": role,
                        "birth_date": birth_date,
                        "birth_place": birth_place,
                    },
                    channel,
                    tags=tags,
                )
                if aliases:
                    metadata["aliases"] = aliases

                if existing_rec:
                    existing_meta = dict(existing_rec.metadata or {})
                    existing_aliases = list(existing_meta.get("aliases") or [])
                    for alias in aliases + [full_name]:
                        if alias and alias != canonical_name and alias not in existing_aliases:
                            existing_aliases.append(alias)
                    existing_meta.update({k: v for k, v in metadata.items() if v not in (None, "")})
                    if existing_aliases:
                        existing_meta["aliases"] = existing_aliases
                    brain.update(
                        existing_rec.id,
                        content=", ".join(parts),
                        metadata=existing_meta,
                        tags=tags,
                    )
                    rec = brain.get(existing_rec.id) or existing_rec
                else:
                    rec = brain.store(
                        content=", ".join(parts),
                        level=Level.IDENTITY,
                        tags=tags,
                        metadata=metadata,
                    )
            result = {"stored": True, "id": rec.id, "name": canonical_name}
            if existing:
                result["similar_existing"] = existing
                result["warning"] = (
                    f"Found {len(existing)} similar person(s) already in memory. Check if this is a duplicate."
                )
            if existing_rec:
                result["merged_into_existing"] = True
            if aliases:
                result["aliases"] = aliases
            return json.dumps(result, ensure_ascii=False)

        elif name == "store_story":
            # Guard: autonomous/worker channels cannot fabricate stories/events
            if channel in _NON_INTERACTIVE_CHANNELS:
                return json.dumps(
                    {
                        "error": "store_story is only available in interactive channels. "
                        "Stories and life events must come from user conversation."
                    },
                    ensure_ascii=False,
                )
            title = args["title"]
            content = args["content"]
            people = [p.strip() for p in args.get("people_mentioned", "").split(",") if p.strip()]

            # Check for existing story with similar title
            with brain_lock:
                existing = _check_duplicates(title, tags=["story"])

                people_tags = [_sanitize_tag(p) for p in people]
                tags = ["story"] + [t for t in people_tags if t]
                rec = brain.store(
                    content=f"{title}\n{content}",
                    level=Level.DOMAIN,
                    tags=tags,
                    metadata=_stamp_provenance(
                        {
                            "type": "story",
                            "title": title,
                            "people_mentioned": people,
                        },
                        channel,
                        tags=tags,
                    ),
                )
                # Connect story to people (promotion-gated)
                from remy.core.agent_tools import gated_connect
                for person_name in people:
                    person_records = brain.search(query=person_name, tags=["person"], limit=3)
                    for pr in person_records:
                        if pr.metadata.get("full_name", "").lower() == person_name.lower():
                            gated_connect(brain, rec.id, pr.id, weight=0.7)
                            break

            result = {"stored": True, "id": rec.id, "title": title}
            if existing:
                result["similar_existing"] = existing
                result["warning"] = (
                    f"Found {len(existing)} similar story/stories already in memory."
                )
            return json.dumps(result, ensure_ascii=False)

        elif name == "connect_records":
            id_a = args["id_a"]
            id_b = args["id_b"]
            relationship = args.get("relationship", "related to")
            weight = float(args.get("weight", 0.7))
            weight = max(0.0, min(1.0, weight))

            if id_a == id_b:
                return json.dumps(
                    {
                        "connected": False,
                        "error": "Cannot connect a record to itself",
                        "id_a": id_a,
                        "id_b": id_b,
                    },
                    ensure_ascii=False,
                )

            with brain_lock:
                rec_a = brain.get(id_a)
                rec_b = brain.get(id_b)
                if not rec_a:
                    return json.dumps(
                        {
                            "connected": False,
                            "error": f"Record {id_a} not found",
                            "id_a": id_a,
                            "id_b": id_b,
                        },
                        ensure_ascii=False,
                    )
                if not rec_b:
                    return json.dumps(
                        {
                            "connected": False,
                            "error": f"Record {id_b} not found",
                            "id_a": id_a,
                            "id_b": id_b,
                        },
                        ensure_ascii=False,
                    )

                from remy.core.agent_tools import gated_connect
                if not gated_connect(brain, id_a, id_b, weight=weight):
                    try:
                        from remy.core.memory_policy import (
                            FACTUAL_FORBIDDEN_ADMISSION_CLASSES,
                            FACTUAL_SAFE_ADMISSION_CLASSES,
                        )
                        safe_a = (rec_a.metadata or {}).get("admission_class") in FACTUAL_SAFE_ADMISSION_CLASSES
                        safe_b = (rec_b.metadata or {}).get("admission_class") in FACTUAL_SAFE_ADMISSION_CLASSES
                        forbidden_tags = {
                            "quarantine-unverified",
                            "claim:llm-unverified",
                            "citation-claim",
                            "scratchpad",
                            "scratchpad-summary",
                            "generated-report",
                        }
                        forbidden_a = (
                            (rec_a.metadata or {}).get("admission_class") in FACTUAL_FORBIDDEN_ADMISSION_CLASSES
                            or bool(set(getattr(rec_a, "tags", []) or []) & forbidden_tags)
                        )
                        forbidden_b = (
                            (rec_b.metadata or {}).get("admission_class") in FACTUAL_FORBIDDEN_ADMISSION_CLASSES
                            or bool(set(getattr(rec_b, "tags", []) or []) & forbidden_tags)
                        )
                        if (safe_a and safe_b) or not (forbidden_a or forbidden_b):
                            try:
                                brain.connect(id_a, id_b, weight=weight)
                            except TypeError:
                                brain.connect(id_a, id_b, weight)
                        else:
                            return json.dumps({
                                "connected": False,
                                "error": "Connection blocked: one or both records are not eligible for promotion",
                                "id_a": id_a,
                                "id_b": id_b,
                            })
                    except Exception:
                        return json.dumps({
                            "connected": False,
                            "error": "Connection blocked: one or both records are not eligible for promotion",
                            "id_a": id_a,
                            "id_b": id_b,
                        })

                # Dedup: check if relationship record for this edge already exists
                existing_rels = brain.search(query="", tags=["relationship"], limit=200)
                edge_exists = False
                for rel in existing_rels:
                    m = rel.metadata or {}
                    if (m.get("id_a") == id_a and m.get("id_b") == id_b) or (
                        m.get("id_a") == id_b and m.get("id_b") == id_a
                    ):
                        # Update existing relationship record instead of creating new
                        brain.update(
                            rel.id,
                            content=f"{relationship}: '{rec_a.content[:80]}' ↔ '{rec_b.content[:80]}'",
                            metadata={**(rel.metadata or {}), "relationship": relationship},
                        )
                        edge_exists = True
                        break

                if not edge_exists:
                    # Store new relationship description record
                    rel_content = f"{relationship}: '{rec_a.content[:80]}' ↔ '{rec_b.content[:80]}'"
                    brain.store(
                        content=rel_content,
                        level=Level.DOMAIN,
                        tags=["relationship"],
                        metadata=_stamp_provenance(
                            {
                                "type": "relationship",
                                "id_a": id_a,
                                "id_b": id_b,
                                "relationship": relationship,
                            },
                            channel,
                            tags=["relationship"],
                        ),
                    )

            clear_recall_cache()
            return json.dumps(
                {
                    "connected": True,
                    "id_a": id_a,
                    "id_b": id_b,
                    "relationship": relationship,
                    "weight": weight,
                    "a_preview": rec_a.content[:100],
                    "b_preview": rec_b.content[:100],
                },
                ensure_ascii=False,
            )

        elif name == "get_connections":
            record_id = args["record_id"]
            connections = []
            with brain_lock:
                rec = brain.get(record_id)
                if not rec:
                    return json.dumps({"error": f"Record {record_id} not found"})

                for conn_id, weight in rec.connections.items():
                    conn_rec = brain.get(conn_id)
                    if conn_rec:
                        connections.append(
                            {
                                "id": conn_id,
                                "content": conn_rec.content[:150],
                                "tags": list(conn_rec.tags) if conn_rec.tags else [],
                                "weight": round(weight, 3),
                            }
                        )

            if not connections:
                return json.dumps(
                    {
                        "record": rec.content[:150],
                        "connections": [],
                        "message": "No connections found.",
                    }
                )

            return json.dumps(
                {
                    "record": rec.content[:150],
                    "connection_count": len(connections),
                    "connections": connections,
                },
                ensure_ascii=False,
            )

        elif name == "get_full_record":
            record_id = args.get("record_id", "").strip()
            if not record_id:
                return json.dumps({"error": "record_id is required"}, ensure_ascii=False)
            with brain_lock:
                rec = brain.get(record_id)
            if not rec:
                return json.dumps({"error": f"Record '{record_id}' not found"}, ensure_ascii=False)
            tags = list(rec.tags) if rec.tags else []
            meta = rec.metadata or {}
            protected = sorted(protected_fields_for_record(meta, tags=tags))
            result = {
                "id": rec.id,
                "content": sanitize_memory_content(rec.content, metadata=meta, tags=tags),
                "tags": tags,
                "level": rec.level.name if hasattr(rec.level, "name") else str(rec.level),
                "char_count": len(rec.content),
            }
            if meta:
                safe_meta = sanitize_memory_metadata(meta, tags=tags)
                result["metadata"] = {
                    k: v for k, v in safe_meta.items() if k not in ("source", "verified", "actionable")
                }
            if protected:
                result["protected_fields_present"] = protected
                result["note"] = (
                    "Protected exact fields are hidden in get_full_record. "
                    "Use get_protected_record only when the user explicitly asks for a sensitive exact value."
                )
            return json.dumps(result, ensure_ascii=False)

        elif name == "get_protected_record":
            record_id = args.get("record_id", "").strip()
            if not record_id:
                return json.dumps({"error": "record_id is required"}, ensure_ascii=False)
            if channel in _NON_INTERACTIVE_CHANNELS:
                return json.dumps(
                    {
                        "error": "get_protected_record is only available in interactive channels. "
                        "Autonomous flows should use action guards and verified exact-memory checks instead."
                    },
                    ensure_ascii=False,
                )
            requested_fields = [
                item.strip()
                for item in (args.get("fields", "") or "").split(",")
                if item.strip()
            ]
            with brain_lock:
                rec = brain.get(record_id)
            if not rec:
                return json.dumps({"error": f"Record '{record_id}' not found"}, ensure_ascii=False)
            tags = list(rec.tags) if rec.tags else []
            meta = rec.metadata or {}
            payload = protected_payload(meta, tags=tags, requested_fields=requested_fields)
            if not payload:
                available = sorted(protected_fields_for_record(meta, tags=tags))
                if available:
                    return json.dumps(
                        {
                            "error": "Requested protected fields not found on this record",
                            "available_fields": available,
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "error": "This record does not contain protected exact fields",
                        "instruction": "Use get_full_record for non-sensitive full content.",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "id": rec.id,
                    "level": rec.level.name if hasattr(rec.level, "name") else str(rec.level),
                    "protected_fields": sorted(payload.keys()),
                    "values": payload,
                    "verified": meta.get("verified") is True,
                    "actionable": meta.get("actionable"),
                    "source": meta.get("source"),
                },
                ensure_ascii=False,
            )

        elif name == "update_record":
            record_id = args.get("record_id", "").strip()
            if not record_id:
                return json.dumps(
                    {
                        "error": "record_id is required. Use 'search' to find the record first, then pass its ID here. "
                        "If you want to create a new record, use 'store' instead."
                    },
                    ensure_ascii=False,
                )
            with brain_lock:
                existing = brain.get(record_id)
                if not existing:
                    return json.dumps(
                        {"error": f"Record '{record_id}' not found"}, ensure_ascii=False
                    )

                # Guard: autonomous cannot overwrite user-confirmed records
                existing_meta = existing.metadata or {}
                if channel in _NON_INTERACTIVE_CHANNELS:
                    is_user_confirmed = (
                        existing_meta.get("source") == "user-confirmed"
                        or existing_meta.get("verified") is True
                        or float(existing_meta.get("trust_score", 0)) >= 0.9
                    )
                    if is_user_confirmed:
                        return json.dumps(
                            {
                                "error": f"Cannot update user-confirmed record '{record_id}' from autonomous mode. "
                                "User-verified records can only be modified in interactive sessions.",
                                "record_preview": existing.content[:200],
                            },
                            ensure_ascii=False,
                        )

                kwargs = {}
                if "content" in args and args["content"]:
                    kwargs["content"] = args["content"]
                    # Audit trail: preserve original content on first update
                    update_meta = dict(existing_meta)
                    if "original_content" not in update_meta:
                        update_meta["original_content"] = existing.content[:500]
                    update_meta["last_updated_by"] = f"agent-{channel}" if channel else "agent"
                    update_meta["last_updated_at"] = datetime.now().isoformat()
                    kwargs["metadata"] = update_meta
                if "tags" in args and args["tags"]:
                    kwargs["tags"] = [_clean_tag(t) for t in args["tags"].split(",") if t.strip()]
                if "level" in args and args["level"]:
                    level_map = {
                        "working": Level.WORKING,
                        "decisions": Level.DECISIONS,
                        "domain": Level.DOMAIN,
                        "identity": Level.IDENTITY,
                    }
                    kwargs["level"] = level_map.get(args["level"].lower(), existing.level)

                updated = brain.update(record_id, **kwargs)
            if not updated:
                return json.dumps({"error": "Update failed"}, ensure_ascii=False)
            clear_recall_cache(args.get("content", ""))

            return json.dumps(
                {
                    "updated": True,
                    "id": record_id,
                    "content": updated.content[:200],
                    "tags": list(updated.tags),
                },
                ensure_ascii=False,
            )

        elif name == "delete_record":
            record_id = args["record_id"]
            with brain_lock:
                existing = brain.get(record_id)
                if not existing:
                    return json.dumps(
                        {"error": f"Record '{record_id}' not found"}, ensure_ascii=False
                    )

                preview = existing.content[:100]
                success = brain.delete(record_id)
            if success:
                clear_recall_cache()
            return json.dumps(
                {
                    "deleted": success,
                    "id": record_id,
                    "deleted_content": preview,
                },
                ensure_ascii=False,
            )

        elif name == "mark_stale":
            from datetime import datetime as _dt_stale

            record_id = str(args.get("record_id") or "").strip()
            reason = str(args.get("reason") or "").strip()
            if not record_id:
                return json.dumps(
                    {"error": "mark_stale requires record_id"}, ensure_ascii=False
                )
            if not reason:
                return json.dumps(
                    {"error": "mark_stale requires reason (why is this stale?)"},
                    ensure_ascii=False,
                )
            superseded_by = str(args.get("superseded_by") or "").strip()
            with brain_lock:
                existing = brain.get(record_id)
                if not existing:
                    return json.dumps(
                        {"error": f"Record '{record_id}' not found"}, ensure_ascii=False
                    )
                existing_meta = dict(existing.metadata or {})
                if existing_meta.get("stale") is True:
                    return json.dumps(
                        {
                            "already_stale": True,
                            "id": record_id,
                            "stale_reason": existing_meta.get("stale_reason"),
                            "stale_marked_at": existing_meta.get("stale_marked_at"),
                        },
                        ensure_ascii=False,
                    )
                # Guard: autonomous agents cannot stale-mark user-confirmed records
                if channel in ("autonomous", "proactive") or (
                    channel and channel.startswith("worker-")
                ):
                    if (
                        existing_meta.get("source") == "user-confirmed"
                        or existing_meta.get("verified") is True
                    ):
                        return json.dumps(
                            {
                                "error": "Cannot mark user-confirmed record as stale from autonomous channel."
                            },
                            ensure_ascii=False,
                        )
                new_tags = list(existing.tags or [])
                if "stale" not in new_tags:
                    new_tags.append("stale")
                existing_meta.update(
                    {
                        "stale": True,
                        "stale_marked_at": _dt_stale.now().isoformat(),
                        "stale_reason": reason,
                        "stale_marked_by": f"agent-{channel}" if channel else "agent",
                    }
                )
                if superseded_by:
                    existing_meta["superseded_by"] = superseded_by
                updated = brain.update(record_id, tags=new_tags, metadata=existing_meta)
            if not updated:
                return json.dumps({"error": "mark_stale update failed"}, ensure_ascii=False)
            try:
                clear_recall_cache(existing.content[:200])
            except Exception:
                pass
            return json.dumps(
                {
                    "marked_stale": True,
                    "id": record_id,
                    "reason": reason,
                    "superseded_by": superseded_by or None,
                    "preview": existing.content[:120],
                },
                ensure_ascii=False,
            )

        elif name == "store_user_profile":
            from remy.core.memory_policy import PROFILE_INPUT_FIELDS
            from remy.core.tool_handlers.profile import normalize_profile_fields

            # Collect non-empty profile fields
            profile_fields = {}
            for key in PROFILE_INPUT_FIELDS:
                val = args.get(key, "").strip()
                if val:
                    profile_fields[key] = val
            profile_fields = normalize_profile_fields(profile_fields)

            if not profile_fields:
                return json.dumps({"error": "No profile fields provided"})

            # Search for existing user profile (newest by created_at)
            _existing_rec = get_user_profile_record()
            with brain_lock:
                if _existing_rec:
                    rec = _existing_rec
                    merged_meta = dict(rec.metadata) if rec.metadata else {}
                    # Append-only fields: notes accumulates instead of overwriting
                    new_notes = profile_fields.pop("notes", None)
                    merged_meta.update(profile_fields)
                    if new_notes:
                        old_notes = merged_meta.get("notes", "") or ""
                        existing_sentences = {s.strip().lower() for s in re.split(r"[\n;]+", old_notes) if s.strip()}
                        new_sentences = [s.strip() for s in re.split(r"[\n;]+", new_notes) if s.strip()]
                        added = [s for s in new_sentences if s.lower() not in existing_sentences]
                        if added:
                            merged_meta["notes"] = (
                                old_notes + "; " + "; ".join(added)
                            ).lstrip("; ") if old_notes else "; ".join(added)
                    merged_meta = normalize_profile_fields(merged_meta)
                    merged_meta["protected_fields"] = sorted(
                        field for field in ("phone", "email") if merged_meta.get(field)
                    )
                    content = _format_profile_content(merged_meta)
                    brain.update(rec.id, content=content, metadata=merged_meta)
                    return json.dumps(
                        {
                            "updated": True,
                            "id": rec.id,
                            "fields_updated": list(profile_fields.keys()),
                            "profile": sanitize_profile_metadata(merged_meta),
                        },
                        ensure_ascii=False,
                    )
                else:
                    content = _format_profile_content(profile_fields)
                    rec = brain.store(
                        content=content,
                        level=Level.IDENTITY,
                        tags=["user-profile", "identity"],
                        metadata={
                            **profile_fields,
                            "type": "user_profile",
                            "source": "user-confirmed",
                            "verified": True,
                            "protected_fields": sorted(
                                field for field in ("phone", "email") if profile_fields.get(field)
                            ),
                            "semantic_type": "fact",
                        },
                        semantic_type="fact",
                    )
                return json.dumps(
                    {
                        "created": True,
                        "id": rec.id,
                        "profile": sanitize_profile_metadata(profile_fields),
                    },
                    ensure_ascii=False,
                )

        elif name in ("people_list", "family_tree"):
            with brain_lock:
                members = brain.list_records(tags=["person"], min_strength=0.05)
                if not members:
                    return "No people stored yet."

                member_ids = {m.id for m in members}
                items = []
                for m in members:
                    links = []
                    for conn_id, w in m.connections.items():
                        if conn_id in member_ids:
                            conn_rec = brain.get(conn_id)
                            if conn_rec:
                                links.append(
                                    {
                                        "id": conn_id,
                                        "name": conn_rec.metadata.get("full_name", "Unknown"),
                                        "weight": round(w, 2),
                                    }
                                )

                    entry = {
                        "name": m.metadata.get("full_name", "Unknown"),
                        "role": m.metadata.get("role", "contact"),
                        "id": m.id,
                    }
                    if links:
                        entry["connections"] = links
                    items.append(entry)

            return json.dumps(items, ensure_ascii=False)

        elif name == "insights":
            with brain_lock:
                stats = brain.stats()
            try:
                from remy.core import introspection_cache as _ic

                _ic.stamp(session_id, "insights", stats)
            except Exception:
                pass
            return json.dumps(stats, ensure_ascii=False, default=str)

        elif name == "review_history_memory_gaps":
            from remy.config.settings import settings
            from remy.core.history_replay import analyze_history_memory_gaps

            sample_limit = int(args.get("sample_limit", 12) or 12)
            sample_limit = max(1, min(sample_limit, 50))
            with brain_lock:
                report = analyze_history_memory_gaps(
                    lambda **search_kwargs: brain.search(**search_kwargs),
                    history_dir=settings.DATA_DIR / "history",
                    sample_limit=sample_limit,
                )
            return json.dumps(report, ensure_ascii=False)

        elif name == "schedule_task":
            # Guard: autonomous/proactive/worker channels cannot create tasks
            if channel in _NON_INTERACTIVE_CHANNELS:
                return json.dumps(
                    {
                        "error": "schedule_task is only available in interactive channels. "
                        "Tasks must be created by explicit user request."
                    },
                    ensure_ascii=False,
                )

            description = str(args.get("description") or args.get("task") or args.get("title") or "").strip()
            if not description:
                return json.dumps(
                    {"error": "schedule_task requires description, task, or title."},
                    ensure_ascii=False,
                )
            due_date, repeat, cron = normalize_schedule_args(args)

            tags = ["scheduled-task"]
            if repeat:
                tags.append(f"repeat-{repeat}")

            content = f"Scheduled: {description} | Due: {due_date}"
            if repeat:
                content += f" | Repeats: {repeat}"
            if cron:
                content += f" | Cron: {cron}"

            with brain_lock:
                rec = brain.store(
                    content=content,
                    level=Level.DOMAIN,
                    tags=tags,
                    metadata=_stamp_provenance(
                        {
                            "type": "scheduled_task",
                            "description": description,
                            "due_date": due_date,
                            "repeat": repeat or None,
                            "cron": cron,
                            "status": "active",
                        },
                        channel,
                        tags=tags,
                    ),
                )
            return json.dumps(
                {
                    "scheduled": True,
                    "id": rec.id,
                    "description": description,
                    "due_date": due_date,
                    "repeat": repeat or "one-time",
                    "cron": cron,
                },
                ensure_ascii=False,
            )

        elif name == "store_research":
            import re

            topic = str(
                args.get("topic")
                or args.get("project_name")
                or args.get("title")
                or args.get("subject")
                or ""
            ).strip()
            findings = (
                args.get("findings")
                or args.get("summary")
                or args.get("content")
                or args.get("report")
                or ""
            )
            sources_raw = (
                args.get("sources")
                or args.get("source")
                or args.get("source_url")
                or args.get("references")
                or ""
            )
            if not topic:
                return json.dumps(
                    {"error": "store_research requires topic, project_name, title, or subject."},
                    ensure_ascii=False,
                )
            if not str(findings or "").strip():
                return json.dumps(
                    {"error": "store_research requires findings, summary, content, or report."},
                    ensure_ascii=False,
                )
            related_query = args.get("related_query", "").strip()

            # v2.4: LLM sometimes sends structured data instead of strings.
            # Coerce list/dict findings to readable text.
            if isinstance(findings, list):
                findings = "\n".join(
                    f"- {f['content'][:200]}"
                    if isinstance(f, dict) and "content" in f
                    else f"- {str(f)[:200]}"
                    for f in findings
                )
            elif isinstance(findings, str):
                # May be a JSON string from coercion — try parsing
                try:
                    parsed = json.loads(findings)
                    if isinstance(parsed, list):
                        findings = "\n".join(
                            f"- {f['content'][:200]}"
                            if isinstance(f, dict) and "content" in f
                            else f"- {str(f)[:200]}"
                            for f in parsed
                        )
                except (json.JSONDecodeError, TypeError):
                    pass  # Already a plain string — use as-is

            # Coerce list sources to comma-separated string
            if isinstance(sources_raw, list):
                sources_raw = ", ".join(
                    s["url"] if isinstance(s, dict) and "url" in s else str(s) for s in sources_raw
                )
            elif isinstance(sources_raw, str):
                try:
                    parsed_src = json.loads(sources_raw)
                    if isinstance(parsed_src, list):
                        sources_raw = ", ".join(
                            s["url"] if isinstance(s, dict) and "url" in s else str(s)
                            for s in parsed_src
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

            # Topic slug for tags
            topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]

            # Parse sources
            source_list = [s.strip() for s in sources_raw.split(",") if s.strip()]

            # Build content
            content = f"Research: {topic}\n\n{findings}"
            if source_list:
                content += "\n\nSources: " + ", ".join(source_list)

            # Dedup check
            with brain_lock:
                existing = _check_duplicates(topic, tags=["research"])

                rec = brain.store(
                    content=content,
                    level=Level.DOMAIN,
                    tags=["research", topic_slug],
                    metadata=_stamp_provenance(
                        {
                            "type": "research_report",
                            "topic": topic,
                            "sources": source_list,
                            "source_type": "retrieved",
                            "timestamp": datetime.now().isoformat(),
                        },
                        channel,
                        tags=["research", topic_slug],
                    ),
                )

                # Auto-connect to related personal records
                connected_to = []
                if related_query:
                    try:
                        related = brain.recall_structured(
                            related_query, top_k=5, min_strength=0.1, session_id=session_id
                        )
                        from remy.core.agent_tools import gated_connect
                        for r in related[:3]:
                            if r["id"] != rec.id:
                                if not gated_connect(brain, rec.id, r["id"], weight=0.6):
                                    continue
                                connected_to.append(
                                    {
                                        "id": r["id"],
                                        "content": r.get("content", "")[:100],
                                    }
                                )
                    except Exception:
                        pass  # Auto-connect is best-effort

            result = {
                "stored": True,
                "id": rec.id,
                "topic": topic,
                "tags": ["research", topic_slug],
                "sources_count": len(source_list),
                "connected_to": connected_to,
            }
            if existing:
                result["similar_existing"] = existing
                result["note"] = "Similar research already exists. Consider updating instead."
            return json.dumps(result, ensure_ascii=False)

        elif name == "web_search":
            query = args["query"]

            # Check cache first (zero tokens)
            cached = _get_cached_search(query)
            if cached:
                # AUTON-7: Track savings from cache hits
                try:
                    from remy.core.budget_negotiation import savings_tracker

                    savings_tracker.record_cache_hit(estimated_cost=800)
                except Exception:
                    pass
                cached["_source"] = "WEB_SEARCH (live internet data, NOT from your memory)"
                return json.dumps(cached, ensure_ascii=False)

            last_error = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    from ddgs import DDGS

                    # v9 multi-backend metasearch — skips yandex (429s from UA
                    # IPs) and bing (disabled=True in v9). startpage added as
                    # privacy-friendly Google proxy. Longer timeout since the
                    # default 5s lets a single slow backend sink the whole call.
                    raw = DDGS(timeout=15).text(
                        query,
                        max_results=10,
                        backend="duckduckgo,brave,google,mojeek,startpage,yahoo",
                    )
                    grounding_chunks = [
                        {
                            "title": r.get("title") or "",
                            "uri": r.get("href") or "",
                            "snippet": r.get("body") or "",
                        }
                        for r in (raw or [])
                        if r.get("href")
                    ]

                    # Phase 2: classify + rerank candidates.
                    from remy.core.retrieval.source_filter import annotate, rerank
                    grounding_chunks = annotate(grounding_chunks)
                    grounding_chunks = rerank(grounding_chunks, drop_classes={"seo"})

                    if grounding_chunks:
                        answer = (
                            f"Found {len(grounding_chunks)} candidate source(s). These are raw discovery candidates (title + URL + snippet), not verified facts. "
                            "Use extract_content on a chosen URL before making external factual claims."
                        )
                    else:
                        answer = (
                            "No candidate sources found. Nothing is verified yet. "
                            "If needed, refine the query and search again."
                        )

                    result = {
                        "_source": "WEB_SEARCH (raw candidate discovery; live internet data, NOT from your memory)",
                        "answer": answer,
                        "mode": "candidate_discovery",
                        "query": query,
                        "candidate_count": len(grounding_chunks),
                    }
                    if grounding_chunks:
                        result["sources"] = grounding_chunks

                    _cache_search_result(query, answer, grounding_chunks)

                    return json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "Web search attempt %d failed: %s. Retrying in %ds...",
                            attempt + 1,
                            e,
                            _RETRY_DELAYS[attempt],
                        )
                        _sleep_with_jitter(_RETRY_DELAYS[attempt])
                    else:
                        logger.error("Web search failed after %d attempts: %s", _MAX_RETRIES + 1, e)

            return json.dumps(
                {"error": f"Web search failed after {_MAX_RETRIES + 1} attempts: {last_error}"}
            )

        elif name == "get_current_datetime":
            now = datetime.now()
            return json.dumps(
                {
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "day_of_week": now.strftime("%A"),
                    "iso": now.isoformat(),
                }
            )

        elif name == "create_subgoal":
            from remy.core.autonomy import create_goal, get_active_goals

            parent_goal_id = args["parent_goal_id"]
            description = args["description"]
            priority = args.get("priority", "medium")

            # Inherit created_by from parent goal
            parent_created_by = "agent"
            parent_goals = get_active_goals()
            for pg in parent_goals:
                if pg.get("goal_id") == parent_goal_id or pg.get("record_id") == parent_goal_id:
                    rec = brain.get(pg["record_id"])
                    if rec and rec.metadata:
                        parent_created_by = rec.metadata.get("created_by", "agent")
                    break

            sub_id = create_goal(
                description=description,
                priority=priority,
                parent_goal_id=parent_goal_id,
                created_by=parent_created_by,
            )

            return json.dumps(
                {
                    "created": True,
                    "record_id": sub_id,
                    "parent_goal_id": parent_goal_id,
                    "description": description[:100],
                }
            )

        elif name == "complete_goal":
            from remy.core.autonomy import (
                get_active_goals,
                update_goal_status,
            )

            goal_id = args["goal_id"]
            notes = args.get("notes", "")

            # Find record_id by goal_id or record_id (agent may pass either)
            all_goals = get_active_goals()
            record_id = None
            for g in all_goals:
                if g["goal_id"] == goal_id or g["record_id"] == goal_id:
                    record_id = g["record_id"]
                    break

            # Fallback: if only 1 active goal, assume agent means that one
            if not record_id and len(all_goals) == 1:
                record_id = all_goals[0]["record_id"]
                logger.info("complete_goal: fuzzy match — only 1 active goal, using %s", record_id)

            if not record_id:
                # Show available goal IDs to help agent retry
                available = [f"{g['goal_id']} ({g['priority']})" for g in all_goals[:5]]
                return json.dumps(
                    {
                        "error": f"Goal {goal_id} not found among active goals",
                        "available_goals": available,
                    }
                )

            update_goal_status(record_id, "completed", notes=notes)
            return json.dumps(
                {
                    "completed": True,
                    "goal_id": goal_id,
                    "notes": notes,
                }
            )

        elif name == "read_file":
            raw_path = args["path"]
            data_dir = Path(settings.DATA_DIR).resolve()
            allowed_paths = [
                Path(p).resolve() for p in getattr(settings, "AUTONOMY_ALLOWED_READ_PATHS", [])
            ]

            # Resolve path
            target = Path(raw_path)
            if not target.is_absolute():
                target = data_dir / raw_path
            target = target.resolve()

            # Security: must be inside data_dir or an allowed path
            allowed = any(target.is_relative_to(ap) for ap in [data_dir] + allowed_paths)
            if not allowed:
                return json.dumps({"error": f"Access denied: {raw_path} is outside allowed paths"})

            if not target.exists():
                return json.dumps({"error": f"File not found: {raw_path}"})
            if not target.is_file():
                return json.dumps({"error": f"Not a file: {raw_path}"})

            try:
                content = target.read_text(encoding="utf-8", errors="replace")[:10000]
                return json.dumps(
                    {
                        "path": str(target),
                        "size": target.stat().st_size,
                        "content": content,
                    }
                )
            except Exception as e:
                return json.dumps({"error": f"Read error: {e}"})

        elif name == "write_file":
            raw_path = args["path"]
            content = args["content"]
            data_dir = Path(settings.DATA_DIR).resolve()

            target = Path(raw_path)
            if not target.is_absolute():
                target = data_dir / raw_path
            target = target.resolve()

            # Security: must be inside data_dir only
            if not target.is_relative_to(data_dir):
                return json.dumps({"error": f"Write denied: {raw_path} is outside data directory"})

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return json.dumps(
                    {
                        "written": True,
                        "path": str(target),
                        "size": len(content),
                    }
                )
            except Exception as e:
                return json.dumps({"error": f"Write error: {e}"})

        elif name == "list_directory":
            raw_path = args.get("path", ".")
            data_dir = Path(settings.DATA_DIR).resolve()
            allowed_paths = [
                Path(p).resolve() for p in getattr(settings, "AUTONOMY_ALLOWED_READ_PATHS", [])
            ]

            target = Path(raw_path)
            if not target.is_absolute():
                target = data_dir / raw_path
            target = target.resolve()

            # Security check
            allowed = any(target.is_relative_to(ap) for ap in [data_dir] + allowed_paths)
            if not allowed:
                return json.dumps({"error": f"Access denied: {raw_path} is outside allowed paths"})

            if not target.exists() or not target.is_dir():
                return json.dumps({"error": f"Directory not found: {raw_path}"})

            try:
                entries = []
                for entry in sorted(target.iterdir()):
                    entries.append(
                        {
                            "name": entry.name,
                            "type": "dir" if entry.is_dir() else "file",
                            "size": entry.stat().st_size if entry.is_file() else None,
                        }
                    )
                return json.dumps(
                    {
                        "path": str(target),
                        "entries": entries[:50],  # Cap at 50
                        "total": len(entries),
                    }
                )
            except Exception as e:
                return json.dumps({"error": f"List error: {e}"})

        elif name == "start_research":
            return _start_research(args, session_id, channel)

        elif name == "add_research_finding":
            return _add_research_finding(args, session_id, channel)

        elif name == "complete_research":
            return _complete_research(args, session_id, channel)

        # ---- Generic metric and event intelligence ----
        elif name == "track_metric":
            return _track_metric(args, channel)

        elif name == "metric_summary":
            return _metric_summary(args)

        elif name == "event_correlate":
            return _event_correlate(args)

        # Deprecated health aliases. Kept for one release so old automations and
        # saved tool calls do not break, but these names are no longer declared
        # to the model as first-class tools.
        elif name == "track_health_metric":
            return _track_health_metric(args, channel)

        elif name == "health_summary":
            return _health_summary(args)

        elif name == "symptom_correlate":
            return _symptom_correlate(args)

        # ---- Fact Extraction (RM-4) ----
        elif name == "extract_facts":
            return _extract_facts(args, channel, session_id)

        elif name == "consolidate":
            try:
                result = brain.consolidate()
                return json.dumps(
                    {
                        "merged": result.get("merged", 0),
                        "llm_merged": result.get("llm_merged", 0),
                        "message": f"Consolidated {result.get('merged', 0)} record pairs (heuristic).",
                    },
                    ensure_ascii=False,
                )
            except Exception as e:
                return json.dumps({"error": f"Consolidation failed: {e}"})

        elif name == "http_get":
            import urllib.error
            import urllib.request

            # Fetch = forward progress; clear the web_search-without-fetch counter.
            try:
                from remy.core.brain_tools import _reset_web_search_no_fetch
                _reset_web_search_no_fetch(session_id)
            except Exception:
                pass

            url = args["url"]

            # SSRF protection: block private/internal networks and non-HTTP schemes
            ssrf_error = _check_ssrf(url)
            if ssrf_error:
                return json.dumps({"error": ssrf_error, "url": url})

            last_error = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Remy-Agent/1.0"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        body = resp.read(32768).decode("utf-8", errors="replace")
                        # Record fetch evidence for extract_facts grounding.
                        try:
                            from remy.core.claim_provenance import record_turn_fetch_evidence
                            record_turn_fetch_evidence(
                                session_id or "",
                                tool="http_get",
                                url=url,
                            )
                        except Exception:
                            pass
                        return json.dumps(
                            {
                                "status": resp.status,
                                "url": url,
                                "content_type": resp.headers.get("Content-Type", ""),
                                "body": body,
                            }
                        )
                except urllib.error.HTTPError as e:
                    # Don't retry client errors (4xx)
                    if 400 <= e.code < 500:
                        return json.dumps({"error": f"HTTP {e.code}: {e.reason}", "url": url})
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "http_get attempt %d failed: %s. Retrying...", attempt + 1, e
                        )
                        _sleep_with_jitter(_RETRY_DELAYS[attempt])
                except Exception as e:
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "http_get attempt %d failed: %s. Retrying...", attempt + 1, e
                        )
                        _sleep_with_jitter(_RETRY_DELAYS[attempt])

            return json.dumps(
                {
                    "error": f"Request failed after {_MAX_RETRIES + 1} attempts: {last_error}",
                    "url": url,
                }
            )

        # ============== EXTRACT CONTENT (Trafilatura) ==============
        elif name == "extract_content":
            import trafilatura

            # Fetch = forward progress; clear the web_search-without-fetch counter.
            try:
                from remy.core.brain_tools import _reset_web_search_no_fetch
                _reset_web_search_no_fetch(session_id)
            except Exception:
                pass

            url = args["url"]

            ssrf_error = _check_ssrf(url)
            if ssrf_error:
                return json.dumps({"error": ssrf_error, "url": url})

            include_links = args.get("include_links", False)
            include_tables = args.get("include_tables", True)
            # Phase 3: optional identity hints. Agent passes these when it
            # wants the fetch cross-checked against an expected page, paper,
            # or resource.
            expected_title = str(args.get("expected_title") or "").strip()
            expected_identifier = str(args.get("expected_identifier") or "").strip()
            expected_authors = str(args.get("expected_authors") or "").strip()
            claim_span = str(args.get("claim_span") or "").strip()

            try:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    return json.dumps(
                        {"error": "Failed to fetch URL (no response or blocked)", "url": url}
                    )

                text = trafilatura.extract(
                    downloaded,
                    include_links=include_links,
                    include_tables=include_tables,
                    favor_recall=True,
                )

                if not text:
                    return json.dumps(
                        {
                            "error": "Could not extract meaningful content (page may be JS-rendered — try browse_page instead)",
                            "url": url,
                        }
                    )

                metadata = trafilatura.extract_metadata(downloaded)
                result = {"url": url, "content": text[:16000]}
                if metadata:
                    if metadata.title:
                        result["title"] = metadata.title
                    if metadata.author:
                        result["author"] = metadata.author
                    if metadata.date:
                        result["date"] = metadata.date
                    if metadata.sitename:
                        result["site"] = metadata.sitename

                if len(text) > 16000:
                    result["truncated"] = True
                    result["total_chars"] = len(text)

                # Phase 3: attach EvidencePacket. Always includes source_class
                # + host; if caller supplied expected_title / expected_identifier
                # the packet records identity_checks results too.
                try:
                    from remy.core.retrieval.evidence import build_packet

                    packet = build_packet(
                        result,
                        requested_url=url,
                        expected_title=expected_title,
                        expected_identifier=expected_identifier,
                        expected_authors=expected_authors,
                        claim_span=claim_span,
                    )
                    result["evidence_packet"] = packet.to_dict()
                    if packet.has_mismatch:
                        result["identity_warning"] = (
                            "Fetched content does not match the expected identity "
                            "(see evidence_packet.identity_checks). Treat this URL "
                            "as unverified for the intended claim."
                        )
                except Exception as _pkt_err:
                    logger.debug("build_packet failed for %s: %s", url, _pkt_err)

                # Record fetch evidence so extract_facts can detect grounding.
                try:
                    from remy.core.claim_provenance import record_turn_fetch_evidence
                    record_turn_fetch_evidence(
                        session_id or "",
                        tool="extract_content",
                        url=url,
                        title=str(result.get("title") or ""),
                        site=str(result.get("site") or ""),
                    )
                except Exception:
                    pass

                return json.dumps(result, ensure_ascii=False)

            except Exception as e:
                logger.warning("extract_content failed for %s: %s", url, e)
                return json.dumps({"error": f"Extraction failed: {e}", "url": url})

        # ============== TODO LIST ==============
        elif name == "add_todo":
            from datetime import datetime as _dt

            title = args["title"].strip()
            priority = args.get("priority", "medium").lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            due_date = args.get("due_date", "").strip() or None
            category = args.get("category", "personal").lower().strip() or "personal"
            parent_id = args.get("parent_id", "").strip() or None
            repeat = args.get("repeat", "").strip().lower()
            if repeat and repeat not in ("daily", "weekly", "monthly"):
                repeat = ""
            repeat_until = args.get("repeat_until", "").strip() or None

            todo_id = f"todo-{uuid.uuid4().hex[:12]}"
            tags = ["todo-item", f"cat-{category}", f"priority-{priority}"]
            if repeat:
                tags.append(f"repeat-{repeat}")

            content = f"Todo [{priority.upper()}]: {title}"
            if due_date:
                content += f" | Due: {due_date}"
            if repeat:
                content += f" | Repeats: {repeat}"

            meta = {
                "type": "todo_item",
                "todo_id": todo_id,
                "title": title,
                "priority": priority,
                "status": "pending",
                "category": category,
                "due_date": due_date,
                "repeat": repeat or None,
                "repeat_until": repeat_until,
                "created_by": "agent" if session_id and session_id.startswith("auto-") else "user",
                "created_at": _dt.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "parent_todo_id": parent_id,
            }

            rec = brain.store(
                content=content,
                level=Level.DOMAIN,
                tags=tags,
                metadata=_stamp_provenance(meta, channel, tags=tags),
            )

            if parent_id:
                try:
                    from remy.core.agent_tools import gated_connect
                    gated_connect(brain, rec.id, parent_id, weight=0.9)
                except Exception:
                    pass

            return json.dumps(
                {
                    "created": True,
                    "id": rec.id,
                    "todo_id": todo_id,
                    "title": title,
                    "priority": priority,
                    "category": category,
                    "due_date": due_date,
                },
                ensure_ascii=False,
            )

        elif name == "list_todos":
            status_filter = args.get("status", "pending").lower()
            category_filter = args.get("category", "").lower().strip()

            records = brain.search(query="", tags=["todo-item"], limit=100)
            items = []
            for r in records:
                meta = getattr(r, "metadata", None) or {}
                if meta.get("type") != "todo_item":
                    continue
                s = meta.get("status", "pending")
                if status_filter != "all" and s != status_filter:
                    continue
                if category_filter and meta.get("category", "") != category_filter:
                    continue
                items.append(
                    {
                        "id": r.id,
                        "todo_id": meta.get("todo_id", ""),
                        "title": meta.get("title")
                        or (
                            r.content.split(": ", 1)[-1].split(" | ")[0]
                            if ": " in r.content
                            else r.content
                        ),
                        "priority": meta.get("priority", "medium"),
                        "status": s,
                        "category": meta.get("category", "personal"),
                        "due_date": meta.get("due_date"),
                        "created_by": meta.get("created_by", "user"),
                        "created_at": meta.get("created_at"),
                        "started_at": meta.get("started_at"),
                        "completed_at": meta.get("completed_at"),
                        "parent_todo_id": meta.get("parent_todo_id"),
                    }
                )

            # Sort: high > medium > low, then by due_date
            priority_order = {"high": 0, "medium": 1, "low": 2}
            items.sort(
                key=lambda x: (priority_order.get(x["priority"], 1), x.get("due_date") or "9999")
            )

            return json.dumps({"todos": items, "count": len(items)}, ensure_ascii=False)

        elif name == "update_todo":
            from datetime import datetime as _dt

            todo_ref = args["id"]
            rec = brain.get(todo_ref)
            if not rec or (rec.metadata or {}).get("type") != "todo_item":
                # Fallback: search by todo_id metadata field
                records = brain.search(query="", tags=["todo-item"], limit=100)
                rec = None
                for r in records:
                    m = r.metadata or {}
                    if m.get("todo_id") == todo_ref:
                        rec = r
                        break
            if not rec:
                return json.dumps({"error": f"Todo '{todo_ref}' not found"})
            record_id = rec.id

            meta = getattr(rec, "metadata", None) or {}

            new_status = args.get("status", "").lower().strip()
            new_title = args.get("title", "").strip()
            new_priority = args.get("priority", "").lower().strip()
            new_due_date = args.get("due_date", "").strip()

            if new_status and new_status in ("pending", "in_progress", "done"):
                if new_status == "done" and meta.get("repeat"):
                    # Recurring task: advance due_date, stay pending
                    from datetime import timedelta

                    repeat = meta["repeat"]
                    repeat_until = meta.get("repeat_until")
                    old_due = meta.get("due_date")
                    base = _dt.fromisoformat(old_due) if old_due else _dt.now()
                    if repeat == "daily":
                        next_due = base + timedelta(days=1)
                    elif repeat == "weekly":
                        next_due = base + timedelta(weeks=1)
                    elif repeat == "monthly":
                        month = base.month + 1
                        year = base.year
                        if month > 12:
                            month = 1
                            year += 1
                        try:
                            next_due = base.replace(year=year, month=month)
                        except ValueError:
                            next_due = base.replace(year=year, month=month, day=28)
                    else:
                        next_due = None

                    # Check if past repeat_until
                    if next_due and repeat_until:
                        try:
                            until_dt = _dt.fromisoformat(repeat_until)
                            if next_due > until_dt:
                                next_due = None  # expired — mark truly done
                        except (ValueError, TypeError):
                            pass

                    if next_due:
                        meta["due_date"] = next_due.strftime("%Y-%m-%d")
                        meta["status"] = "pending"
                        meta["started_at"] = None
                        meta["completed_at"] = None
                        meta["last_completed_at"] = _dt.now().isoformat()
                        new_status = "pending"
                    else:
                        # Repeat expired — mark done for real
                        meta["status"] = "done"
                        meta["completed_at"] = _dt.now().isoformat()
                        meta["repeat"] = None
                else:
                    meta["status"] = new_status
                    if new_status == "in_progress":
                        meta["started_at"] = _dt.now().isoformat()
                    elif new_status == "done":
                        meta["completed_at"] = _dt.now().isoformat()
                    elif new_status == "pending":
                        meta["started_at"] = None
                        meta["completed_at"] = None
            if new_priority and new_priority in ("high", "medium", "low"):
                meta["priority"] = new_priority
            if new_due_date:
                meta["due_date"] = new_due_date
            meta["updated_at"] = _dt.now().isoformat()

            title = new_title or meta.get("title") or rec.content.split(": ", 1)[-1].split(" | ")[0]
            meta["title"] = title
            content = f"Todo [{meta.get('priority', 'medium').upper()}]: {title}"
            if meta.get("due_date"):
                content += f" | Due: {meta['due_date']}"
            if meta.get("status") == "done":
                content += " [DONE]"

            brain.update(record_id, content=content, metadata=meta)

            return json.dumps(
                {
                    "updated": True,
                    "id": record_id,
                    "status": meta.get("status"),
                    "title": title,
                },
                ensure_ascii=False,
            )

        elif name == "delete_todo":
            todo_ref = args["id"]
            rec = brain.get(todo_ref)
            if not rec or (rec.metadata or {}).get("type") != "todo_item":
                # Fallback: search by todo_id
                records = brain.search(query="", tags=["todo-item"], limit=100)
                rec = None
                for r in records:
                    if (r.metadata or {}).get("todo_id") == todo_ref:
                        rec = r
                        break
            if not rec:
                return json.dumps({"error": f"Todo '{todo_ref}' not found"})
            record_id = rec.id
            meta = getattr(rec, "metadata", None) or {}
            meta["status"] = "archived"
            brain.update(record_id, metadata=meta)
            return json.dumps({"deleted": True, "id": record_id}, ensure_ascii=False)

        elif name == "verify_record":
            from datetime import datetime as _dt

            record_id = args["record_id"]
            rec = brain.get(record_id)
            if not rec:
                return json.dumps({"error": "Record not found", "record_id": record_id})
            meta = dict(rec.metadata or {})
            meta["verified"] = True
            meta["actionable"] = True
            meta["trust_score"] = 1.0
            meta["verified_at"] = _dt.now().isoformat()
            meta["verified_by"] = "user"
            if args.get("note"):
                meta["verification_note"] = args["note"]
            brain.update(record_id, metadata=meta)
            return json.dumps({"verified": True, "record_id": record_id})

        # ============== AGENT PERSONA ==============
        elif name == "read_persona":
            persona = _get_agent_persona()
            return json.dumps(persona, ensure_ascii=False)

        elif name == "update_persona":
            # Guard: autonomous channels cannot self-modify persona
            if channel in _NON_INTERACTIVE_CHANNELS:
                return json.dumps(
                    {
                        "error": "update_persona is only available in interactive channels. "
                        "Agent personality changes require user presence."
                    },
                    ensure_ascii=False,
                )
            persona = update_persona_fields(args, channel=channel)
            record_id = persona.pop("_record_id", "")
            return json.dumps(
                {
                    "updated": True,
                    "id": record_id,
                    "persona": persona,
                },
                ensure_ascii=False,
            )

        # ============== AUTON-3: INTERACTIVE ESCALATION ==============
        elif name == "request_guidance":
            from remy.core.guidance_queue import guidance_queue

            question = args.get("question", "").strip()
            context = args.get("context", "").strip()
            if not question:
                return json.dumps({"error": "question is required"})

            answer = guidance_queue.request_guidance_sync(question, context=context)
            if answer is not None:
                return json.dumps(
                    {
                        "answered": True,
                        "answer": answer,
                        "question": question[:200],
                    }
                )
            else:
                return json.dumps(
                    {
                        "answered": False,
                        "answer": None,
                        "reason": "User did not respond within the timeout period. Skip this goal or try a different approach.",
                        "question": question[:200],
                    }
                )

        # ============== AUTON-11: TOOL HEALTH STATUS ==============
        elif name == "tool_status":
            from remy.core.tool_routing import get_tool_status_report

            report = get_tool_status_report()
            return json.dumps(report, ensure_ascii=False)

        # ============== SCRATCHPAD (v2.3) ==============
        elif name == "scratchpad":
            from remy.core.scratchpad import clear_notes, read_notes, summarize_notes, write_note

            action = args.get("action", "read").lower()
            if action == "write":
                content = args.get("content", "").strip()
                if not content:
                    return json.dumps({"error": "content is required for write action"})
                result = write_note(content, session_id=session_id or "", channel=channel or "")
                return json.dumps(result, ensure_ascii=False)
            elif action == "summarize":
                result = summarize_notes(
                    session_id=session_id or "",
                    channel=channel or "",
                    force=bool(args.get("force", False)),
                )
                return json.dumps(result, ensure_ascii=False)
            elif action == "clear":
                deleted = clear_notes()
                return json.dumps({"cleared": True, "deleted_count": deleted})
            else:  # read
                notes = read_notes()
                return json.dumps({"notes": notes, "count": len(notes)}, ensure_ascii=False)

        elif name == "filter_working":
            from remy.core.scratchpad import filter_working_memory

            query = args.get("query", "").strip()
            if not query:
                return json.dumps({"error": "query is required"})
            result = filter_working_memory(
                query,
                session_id=session_id or "",
                min_score=float(args.get("min_score", 0.18) or 0.18),
                delete_irrelevant=bool(args.get("delete_irrelevant", False)),
            )
            return json.dumps(result, ensure_ascii=False)

        # ============== AUTON-2: RUNTIME DIRECTIVES ==============
        elif name == "add_runtime_directive":
            from remy.core.agent import invalidate_system_instruction_cache
            from remy.core.runtime_directives import (
                add_persistent_directive,
                add_session_directive,
            )

            text = args.get("text", "").strip()
            if not text:
                return json.dumps({"error": "text is required"})

            persistent = bool(args.get("persistent", False))
            ttl = args.get("ttl_seconds")
            ttl_int = int(ttl) if ttl is not None else None

            if persistent:
                record_id = add_persistent_directive(text, source="agent")
                if record_id:
                    invalidate_system_instruction_cache(session_id)
                    return json.dumps(
                        {
                            "added": True,
                            "type": "persistent",
                            "record_id": record_id,
                            "text": text[:200],
                        }
                    )
                return json.dumps({"error": "Failed to store persistent directive"})
            else:
                directive_id = add_session_directive(
                    text,
                    session_id=session_id or "default",
                    ttl_seconds=ttl_int,
                    source="agent",
                )
                invalidate_system_instruction_cache(session_id)
                return json.dumps(
                    {
                        "added": True,
                        "type": "session",
                        "directive_id": directive_id,
                        "text": text[:200],
                    }
                )

        elif name == "remove_runtime_directive":
            from remy.core.agent import invalidate_system_instruction_cache
            from remy.core.runtime_directives import (
                deactivate_persistent_directive,
                remove_session_directive,
            )

            session_index = args.get("session_index")
            record_id_arg = args.get("record_id", "").strip()

            if record_id_arg:
                ok = deactivate_persistent_directive(record_id_arg)
                invalidate_system_instruction_cache(session_id)
                return json.dumps({"removed": ok, "type": "persistent", "record_id": record_id_arg})
            elif session_index is not None:
                ok = remove_session_directive(session_id or "default", int(session_index))
                invalidate_system_instruction_cache(session_id)
                return json.dumps({"removed": ok, "type": "session", "index": int(session_index)})
            else:
                return json.dumps({"error": "Provide session_index or record_id"})

        # ============== IMAGE GENERATION ==============
        elif name == "generate_image":
            return _generate_image(args, session_id, channel)

        # ============== REPORT GENERATION ==============
        elif name == "generate_report":
            return _generate_report(args, session_id, channel)

        # ============== PRESENTATION GENERATION ==============
        elif name == "generate_presentation":
            return _generate_presentation(args, session_id, channel)

        elif name == "memory_feedback":
            return _memory_feedback(args)

        elif name == "get_corrections":
            return _get_corrections(args)

        elif name == "deprecate_belief":
            return _deprecate_belief(args)

        elif name == "get_belief_health":
            return _get_belief_health(args)

        elif name == "get_thermal_map":
            return _get_thermal_map(args)

        elif name == "aura_cognitive_ops":
            result = _aura_cognitive_ops(args)
            try:
                from remy.core import introspection_cache as _ic

                op_name = args.get("op") or args.get("method") or args.get("function") or "unknown"
                _ic.stamp(session_id, op_name, result[:500])
            except Exception:
                pass
            return result

        elif name == "get_plasticity_audit":
            return _get_plasticity_audit(args)

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        logger.error(f"Tool execution error ({name}): {e}")
        return f"Error: {e}"


# ============================================================
# FEEDBACK & CORRECTION TOOLS
# ============================================================

def _memory_feedback(args: dict) -> str:
    """Signal to AuraSDK whether a memory record was useful or harmful.

    Args:
        record_id: ID of the record to give feedback on
        useful: true = record was helpful, false = record was wrong/harmful
        reason: optional explanation (stored as note)
    """
    import json
    from remy.core.agent_tools import brain, brain_lock

    record_id = args.get("record_id", "")
    useful = args.get("useful", True)
    reason = args.get("reason", "")

    if not record_id:
        return json.dumps({"error": "record_id required"})

    with brain_lock:
        brain.feedback(record_id, bool(useful))
        stats = brain.feedback_stats(record_id)

        # Optionally store reason as a correction note
        if reason:
            from remy.core.agent_tools import Level
            brain.store(
                content=f"Feedback on record {record_id}: {'useful' if useful else 'not useful'}. {reason}",
                level=Level.WORKING,
                tags=["memory-feedback", "correction-note"],
                metadata={
                    "type": "memory_feedback",
                    "target_record_id": record_id,
                    "useful": useful,
                    "reason": reason,
                },
            )

    return json.dumps({
        "record_id": record_id,
        "useful": useful,
        "net_score": stats[2] if stats else 0,
        "positive": stats[0] if stats else 0,
        "negative": stats[1] if stats else 0,
        "reason_stored": bool(reason),
    })


def _get_corrections(args: dict) -> str:
    """Get AuraSDK correction suggestions and review queue.

    Args:
        mode: 'suggestions' (default) | 'queue' | 'log' | 'recent_beliefs' | 'report'
        limit: max items to return (default 10)
    """
    import json
    from remy.core.agent_tools import brain, brain_lock

    mode = args.get("mode", "suggestions")
    limit = int(args.get("limit", 10))

    with brain_lock:
        if mode == "queue":
            items = brain.get_correction_review_queue(limit=limit)
            return json.dumps({"mode": "queue", "items": [
                _serialize_correction(i) for i in (items or [])
            ]})
        elif mode == "log":
            items = brain.get_correction_log()
            return json.dumps({"mode": "log", "items": [
                _serialize_correction(i) for i in (items or [])[:limit]
            ]})
        elif mode == "recent_beliefs":
            items = brain.get_recently_corrected_beliefs(limit=limit)
            return json.dumps({"mode": "recent_beliefs", "items": [
                _serialize_correction(i) for i in (items or [])
            ]})
        elif mode == "report":
            report = brain.get_suggested_corrections_report(limit=limit)
            if isinstance(report, dict):
                return json.dumps({"mode": "report", "report": report})
            return json.dumps({"mode": "report", "report": str(report)[:500]})
        else:  # suggestions (default)
            items = brain.get_suggested_corrections(limit=limit)
            return json.dumps({"mode": "suggestions", "count": len(items or []), "items": [
                _serialize_correction(i) for i in (items or [])
            ]})


def _deprecate_belief(args: dict) -> str:
    """Penalize/downvote a belief or causal pattern in AuraSDK.

    Args:
        record_id: ID of the belief/record to deprecate
        reason: optional explanation (makes correction stronger)
        target: 'belief' (default) | 'causal_pattern' | 'policy_hint'
    """
    import json
    from remy.core.agent_tools import brain, brain_lock

    record_id = args.get("record_id", "")
    reason = args.get("reason", "")
    target = args.get("target", "belief")

    if not record_id:
        return json.dumps({"error": "record_id required"})

    with brain_lock:
        if target == "causal_pattern":
            ok = brain.retract_causal_pattern(record_id) if not reason else brain.invalidate_causal_pattern(record_id)
            action = "retracted_causal_pattern"
        elif target == "policy_hint":
            ok = brain.retract_policy_hint(record_id)
            action = "retracted_policy_hint"
        else:
            ok = brain.deprecate_belief_with_reason(record_id, reason) if reason else brain.deprecate_belief(record_id)
            action = "deprecated_belief"

        # Store audit trail
        from remy.core.agent_tools import Level
        brain.store(
            content=f"DEPRECATED [{target}] {record_id}: {reason or 'no reason given'}",
            level=Level.WORKING,
            tags=["negative-learning", "deprecation", target],
            metadata={"type": "deprecation", "target_id": record_id, "target_type": target, "reason": reason},
        )

    return json.dumps({"action": action, "record_id": record_id, "success": ok, "reason": reason})


def _get_belief_health(args: dict) -> str:
    """Get volatile or unstable beliefs from AuraSDK for review.

    Args:
        mode: 'volatile' (default) = high-volatility beliefs | 'unstable' = low-stability beliefs
        limit: max items to return (default 10)
    """
    import json
    from remy.core.agent_tools import brain, brain_lock

    mode = args.get("mode", "volatile")
    limit = int(args.get("limit", 10))

    with brain_lock:
        if mode == "unstable":
            items = brain.get_low_stability_beliefs(limit=limit)
            label = "low_stability"
        else:
            items = brain.get_high_volatility_beliefs(limit=limit)
            label = "high_volatility"

    def _serialize(item):
        if isinstance(item, dict):
            return item
        return {k: getattr(item, k) for k in ["record_id", "content", "volatility", "stability", "confidence"]
                if hasattr(item, k)}

    return json.dumps({"mode": label, "count": len(items or []), "items": [_serialize(i) for i in (items or [])]})


def _get_thermal_map(args: dict) -> str:
    """Get cognitive heat map — hot zones, cold mass, routing advice."""
    import json
    from remy.config.settings import settings
    from remy.core.thermal_advisor import compute_thermal_map, format_thermal_report_json

    report = compute_thermal_map(str(settings.AURA_BRAIN_PATH))
    if not report:
        return json.dumps({"error": "No belief graph available"})
    return json.dumps(format_thermal_report_json(report))


def _get_plasticity_audit(args: dict) -> str:
    """Audit synaptic plasticity — edge health, pruning history, leak ratio."""
    import json
    from remy.config.settings import settings
    from remy.core.synaptic_plasticity import get_plasticity_audit

    audit = get_plasticity_audit(str(settings.AURA_BRAIN_PATH))
    return json.dumps(audit, ensure_ascii=False)


def _serialize_correction(item) -> dict:
    """Normalize a correction item to a plain dict."""
    if isinstance(item, dict):
        return item
    return {
        k: getattr(item, k)
        for k in ["target_kind", "target_id", "suggested_action", "reason_detail",
                  "priority_score", "severity", "namespace"]
        if hasattr(item, k)
    }


# ============================================================
# AURA COGNITIVE OPS — agent explores AuraSDK directly
# ============================================================

# Hard cap on a single aura_cognitive_ops response payload (bytes of UTF-8 JSON).
# Gemini's 1M-token context window dies when a single tool result floods it
# (e.g. list_records dumping the full connections map for thousands of records).
# 60 KB ≈ 15K tokens — generous for inspection without risking a context blowout.
_AURA_OP_PAYLOAD_BYTE_CAP = 60_000
# Default top-K for brief-mode list compression
_AURA_OP_BRIEF_TOP_K = 10


def _aura_op_brief_aggregate(items: list) -> dict:
    """Build aggregate stats for a list of serialized record/dict items.

    Used by brief-mode compression to summarize what was dropped without
    exposing the full payload.
    """
    total = len(items)
    if total == 0:
        return {"total": 0}

    def _get(item, key, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return default

    activations = [_get(i, "activation_count", 0) or 0 for i in items if isinstance(i, dict)]
    strengths = [_get(i, "strength", 0.0) or 0.0 for i in items if isinstance(i, dict)]
    conn_counts = []
    for i in items:
        if not isinstance(i, dict):
            continue
        conns = i.get("connections")
        if isinstance(conns, (list, dict)):
            conn_counts.append(len(conns))

    agg = {"total": total}
    if activations:
        agg["activation_sum"] = sum(activations)
        agg["activation_max"] = max(activations)
    if strengths:
        agg["strength_mean"] = round(sum(strengths) / len(strengths), 4)
    if conn_counts:
        agg["connections_total"] = sum(conn_counts)
        agg["connections_mean"] = round(sum(conn_counts) / len(conn_counts), 2)
    return agg


def _aura_op_briefify_record(item):
    """Strip a record dict down to its identifying fields for brief mode.

    Drops the connections map (the main token sink) and large free-text fields.
    Keeps id, content (truncated), tags, strength, activation_count.
    """
    if not isinstance(item, dict):
        return item
    out = {}
    for k in ("id", "record_id", "tags", "strength", "activation_count",
              "created_at", "namespace", "speaker_id"):
        if k in item:
            out[k] = item[k]
    content = item.get("content")
    if isinstance(content, str):
        out["content"] = content[:200] + ("…" if len(content) > 200 else "")
    elif content is not None:
        out["content"] = str(content)[:200]
    conns = item.get("connections")
    if isinstance(conns, (list, dict)):
        out["connections_count"] = len(conns)
    return out


def _aura_op_compress_if_huge(serialized, op: str, brief: bool):
    """Apply brief-mode compression and a hard byte cap.

    Returns (final_value, meta_dict) where meta_dict describes any compression
    that was applied so the agent can see why the payload was trimmed.
    """
    import json as _json
    meta = {}

    # Brief mode for list payloads: top-K + aggregate, drops connections map.
    if brief and isinstance(serialized, list) and len(serialized) > _AURA_OP_BRIEF_TOP_K:
        ranked = sorted(
            serialized,
            key=lambda x: (
                (x.get("activation_count", 0) if isinstance(x, dict) else 0),
                (x.get("strength", 0.0) if isinstance(x, dict) else 0.0),
            ),
            reverse=True,
        )
        top = [_aura_op_briefify_record(i) for i in ranked[:_AURA_OP_BRIEF_TOP_K]]
        agg = _aura_op_brief_aggregate(serialized)
        serialized = {
            "mode": "brief",
            "op": op,
            "top_k": top,
            "aggregate": agg,
            "note": (
                f"Brief mode: showing top {_AURA_OP_BRIEF_TOP_K} of {len(ranked)} "
                f"by activation_count. Pass params={{'full': true}} for the full list."
            ),
        }
        meta["briefed"] = True

    # Final hard byte cap regardless of mode — last line of defense.
    encoded = _json.dumps(serialized, default=str, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > _AURA_OP_PAYLOAD_BYTE_CAP:
        meta["truncated"] = True
        meta["original_bytes"] = len(encoded.encode("utf-8"))
        if isinstance(serialized, list):
            agg = _aura_op_brief_aggregate(serialized)
            serialized = {
                "mode": "truncated",
                "op": op,
                "sample": [_aura_op_briefify_record(i) for i in serialized[:5]],
                "aggregate": agg,
                "note": (
                    f"Payload exceeded {_AURA_OP_PAYLOAD_BYTE_CAP} bytes "
                    f"({meta['original_bytes']}). Showing 5-item sample. "
                    f"Use a more specific query or pagination."
                ),
            }
        elif isinstance(serialized, dict):
            keys = list(serialized.keys())
            serialized = {
                "mode": "truncated",
                "op": op,
                "keys": keys[:50],
                "key_count": len(keys),
                "note": (
                    f"Dict payload exceeded {_AURA_OP_PAYLOAD_BYTE_CAP} bytes. "
                    f"Showing key list only."
                ),
            }
        else:
            serialized = {
                "mode": "truncated",
                "op": op,
                "preview": encoded[:2000],
                "note": f"Payload exceeded {_AURA_OP_PAYLOAD_BYTE_CAP} bytes.",
            }
    return serialized, meta


def _aura_cognitive_ops(args: dict) -> str:
    """Universal gateway for the agent to call any AuraSDK cognitive method.

    The agent uses this to explore and test AuraSDK capabilities autonomously.

    Args:
        op: method name to call (e.g. 'get_high_volatility_beliefs', 'recall_person_context')
        params: dict of kwargs to pass to the method (optional)

    Output is brief-mode by default for list-shaped results: top-K records by
    activation_count + aggregate stats. Pass params={'full': true} to get the
    raw, un-compressed payload (subject to a hard 60 KB cap).
    """
    import json
    from remy.core.agent_tools import brain, brain_lock

    # Accept both "op" and "method" — model sometimes uses "method" instead of "op"
    op = args.get("op") or args.get("method") or args.get("function") or ""
    raw_params = args.get("params") or args.get("kwargs") or args.get("args") or {}
    if isinstance(raw_params, str):
        try:
            params = json.loads(raw_params)
        except Exception:
            params = {}
    else:
        params = raw_params

    if not op:
        return json.dumps({"error": "op required — pass op='method_name'"})

    # Pop control flags before forwarding kwargs to the brain method.
    full_mode = bool(params.pop("full", False)) if isinstance(params, dict) else False
    brief = not full_mode

    # Safety: only allow read/inspect methods — no destructive ops via this tool
    _BLOCKED = {
        "delete", "close", "flush", "rollback", "clear_embedding_fn",
        "disable_full_cognitive_stack", "move_record",
    }
    if op in _BLOCKED:
        return json.dumps({"error": f"op '{op}' is blocked for safety — use dedicated tools"})

    with brain_lock:
        method = getattr(brain, op, None)
        if method is None:
            # Try on underlying _aura directly
            method = getattr(getattr(brain, "_aura", None), op, None)
        if method is None:
            return json.dumps({"error": f"op '{op}' not found on brain"})

        try:
            from remy.core.brain_tools import _serialize_aura_result
            result = method(**params) if params else method()
            serialized = _serialize_aura_result(result)
            serialized, meta = _aura_op_compress_if_huge(serialized, op, brief)
            payload = {"op": op, "result": serialized}
            if meta:
                payload["_meta"] = meta
            return json.dumps(payload, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"op": op, "error": str(e)})
