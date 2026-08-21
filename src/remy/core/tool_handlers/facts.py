"""
Fact extraction handlers.

This module owns the neutral RM-4 fact extraction path. The deprecated
tool_handlers.health module re-exports this handler for import stability.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger("BrainTools")


def _get_brain():
    """Lazy accessor - reads brain from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain


def _extract_facts(
    args: dict, channel: str | None = None, session_id: str | None = None
) -> str:
    """Extract structured facts from text with explicit learning boundaries.

    Learning boundary rule (D-01, Phase 2):
    - Grounded path: if a URL has been fetched this turn, SPO triples are
      admitted through ingest_grounded_evidence.
    - Unverified path: no fetch this turn -> quarantine at Level.WORKING with
      admission_class=unverified_claim. Unverified claims are not factual
      substrate.
    """
    from remy.core.provenance import _stamp_provenance
    from remy.core.ingestion import ingest_grounded_evidence

    brain = _get_brain()
    text = args["text"]
    source = args.get("source", "unknown")

    fetch_evidence: list[dict] = []
    try:
        from remy.core.claim_provenance import get_turn_fetch_evidence

        fetch_evidence = get_turn_fetch_evidence(session_id or "")
    except Exception:
        pass

    has_fetch_grounding = bool(fetch_evidence)
    grounding_url = fetch_evidence[0].get("url", "") if fetch_evidence else ""

    prompt = (
        f"Extract key facts from the following text into distinct Subject-Predicate-Object statements.\n"
        f"Text: {text}\n\n"
        "Format: JSON list of objects with keys 'subject', 'predicate', 'object', 'context'.\n"
        "Example: [{'subject': 'Project Aurora', 'predicate': 'uses', 'object': 'daily reports', 'context': 'workflow tracking'}]"
    )

    try:
        from remy.core.agent_tools import Level
        from remy.core.llm import call_llm

        result = call_llm(prompt, purpose="extract_facts").content

        clean = str(result).replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)

        if isinstance(data, list):
            stored_count = 0
            for item in data:
                subject = item.get("subject")
                predicate = item.get("predicate")
                obj = item.get("object")

                if subject and predicate and obj:
                    content = f"{subject} {predicate} {obj}."

                    if has_fetch_grounding:
                        ingestion = ingest_grounded_evidence(
                            content=content,
                            source_url=grounding_url or source,
                            session_id=session_id or "",
                            channel=channel,
                            extract_class="grounded_source_extract",
                            extra_tags=["fact", "extracted-fact"],
                            extra_meta={
                                "type": "fact",
                                "verified": True,
                                "extraction_method": "llm",
                                "structure": item,
                                "extracted_at": datetime.now().isoformat(),
                            },
                        )
                        if not ingestion.admitted:
                            continue
                        store_level = ingestion.level
                        store_tags = ingestion.tags
                        store_meta = _stamp_provenance(
                            ingestion.metadata,
                            channel,
                            tags=store_tags,
                        )
                    else:
                        store_level = Level.WORKING
                        store_tags = [
                            "extracted-fact",
                            "unverified-extraction",
                            "quarantine-unverified",
                        ]
                        store_meta = _stamp_provenance(
                            {
                                "type": "fact",
                                "verified": False,
                                "extraction_method": "llm",
                                "structure": item,
                                "extracted_at": datetime.now().isoformat(),
                                "source": source,
                                "learning_channel": "unverified",
                                "admission_class": "unverified_claim",
                                "requires_grounding": True,
                            },
                            channel,
                            tags=store_tags,
                        )

                    brain.store(
                        content=content,
                        level=store_level,
                        tags=store_tags,
                        metadata=store_meta,
                    )
                    stored_count += 1

            if not has_fetch_grounding and stored_count > 0:
                return (
                    f"Extracted {stored_count} items from text. "
                    "No fetch evidence found this turn - stored as unverified working notes "
                    "(Level.WORKING, quarantine-unverified). "
                    "Use extract_content on a source URL first if you want durable domain facts."
                )
            return f"Extracted and stored {stored_count} facts."

        return "Failed to parse facts: format unexpected."

    except Exception as e:
        return f"Fact extraction failed: {e}"
