"""
Spatial Seed — loads user location context into brain on first startup.

Reads data/spatial_seed.json and stores records tagged 'spatial-context'
if they don't already exist. Idempotent — safe to call on every startup.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_spatial_context() -> None:
    """Store spatial seed records if not yet in brain."""
    from remy.config.settings import settings
    from remy.core.agent_tools import brain

    seed_path = settings.DATA_DIR / "spatial_seed.json"
    if not seed_path.exists():
        return

    # Check if already seeded
    try:
        existing = brain.search(query="", tags=["spatial-context"], limit=1)
        if existing:
            return  # already seeded
    except Exception:
        pass

    try:
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        records = data.get("records", [])
    except Exception as e:
        logger.warning("spatial_seed: failed to read %s: %s", seed_path, e)
        return

    stored = 0
    for rec in records:
        try:
            content = rec.get("content", "")
            tags = rec.get("tags", ["spatial-context"])
            metadata = rec.get("metadata", {})
            level = rec.get("level", "L3_DOMAIN")
            trust = rec.get("trust_score", 1.0)

            brain.store(
                content=content,
                tags=tags,
                metadata=metadata,
                level=level,
                trust_score=trust,
            )
            stored += 1
        except Exception as e:
            logger.debug("spatial_seed: failed to store record: %s", e)

    if stored:
        logger.info("spatial_seed: stored %d spatial context records", stored)
