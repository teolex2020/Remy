#!/usr/bin/env python3
"""
Brain Cleanup Script — fix level promotions, remove duplicates, stamp provenance.

Run: python scripts/brain_cleanup.py [--dry-run]

Fixes:
  1. Downgrade transient records wrongly promoted to IDENTITY
  2. Remove duplicate records (keep newest, delete older copies)
  3. Stamp missing provenance on records without source field
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aura import Aura, Level

DRY_RUN = "--dry-run" in sys.argv

# Tags that should NEVER be at IDENTITY level
TRANSIENT_TAGS = {
    "autonomous-outcome",
    "outcome-failure",
    "outcome-success",
    "action-plan",
    "session-summary",
    "session-reflection",
    "web-search-cache",
    "feedback-signal",
    "autonomous-session",
}

# What level should each tag type be at (max)
TAG_LEVEL_MAP = {
    "autonomous-outcome": Level.Domain,
    "outcome-failure": Level.Domain,
    "outcome-success": Level.Domain,
    "action-plan": Level.Decisions,
    "session-summary": Level.Domain,
    "session-reflection": Level.Domain,
    "web-search-cache": Level.Working,
    "feedback-signal": Level.Working,
    "autonomous-session": Level.Domain,
}


def get_target_level(tags: list[str]) -> Level | None:
    """Determine the correct level for a record based on its tags."""
    for tag in tags:
        if tag in TAG_LEVEL_MAP:
            return TAG_LEVEL_MAP[tag]
    return None


def main():
    brain_path = os.path.join(os.path.dirname(__file__), "..", "data", "brain")
    brain = Aura(brain_path)

    all_recs = brain.search(limit=5000)
    print(f"Total records: {len(all_recs)}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print()

    # ── Phase 1: Fix wrong IDENTITY levels ──────────────────────────
    print("=" * 60)
    print("PHASE 1: Fix wrongly promoted IDENTITY records")
    print("=" * 60)

    downgraded = 0
    for rec in all_recs:
        if str(rec.level) != "IDENTITY":
            continue
        tags = rec.tags or []
        tag_set = set(tags)
        if not (tag_set & TRANSIENT_TAGS):
            continue

        target = get_target_level(tags)
        if target is None:
            continue

        print(f"  DOWNGRADE [{rec.id[:8]}] IDENTITY -> {target}")
        print(f"    tags: {tags}")
        print(f"    content: {rec.content[:80]}")

        if not DRY_RUN:
            brain.update(rec.id, level=target)
        downgraded += 1

    print(f"\nDowngraded: {downgraded} records")
    print()

    # ── Phase 2: Remove duplicates ──────────────────────────────────
    print("=" * 60)
    print("PHASE 2: Remove duplicate records")
    print("=" * 60)

    # Re-fetch after level changes
    if not DRY_RUN:
        all_recs = brain.search(limit=5000)

    # Group by content prefix (first 80 chars)
    prefix_groups: dict[str, list] = {}
    for rec in all_recs:
        prefix = rec.content[:80].strip()
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(rec)

    deleted = 0
    for prefix, group in prefix_groups.items():
        if len(group) < 2:
            continue

        # Sort by created_at descending (keep newest)
        group.sort(
            key=lambda r: getattr(r, "created_at", 0) or 0,
            reverse=True,
        )

        keeper = group[0]
        dupes = group[1:]

        print(f"  DUPLICATES ({len(group)}x): {prefix[:60]}...")
        print(f"    Keep: [{keeper.id[:8]}] str={keeper.strength:.2f}")

        for dupe in dupes:
            print(f"    Delete: [{dupe.id[:8]}] str={dupe.strength:.2f}")
            if not DRY_RUN:
                brain.delete(dupe.id)
            deleted += 1

    print(f"\nDeleted: {deleted} duplicate records")
    print()

    # ── Phase 3: Stamp missing provenance ───────────────────────────
    print("=" * 60)
    print("PHASE 3: Stamp missing provenance")
    print("=" * 60)

    # Re-fetch after deletions
    if not DRY_RUN:
        all_recs = brain.search(limit=5000)

    stamped = 0
    for rec in all_recs:
        meta = dict(rec.metadata or {})
        if meta.get("source"):
            continue

        # Infer source from tags/content
        tags = set(rec.tags or [])
        if tags & {"autonomous-outcome", "autonomous-goal", "autonomous-session"}:
            source = "agent-autonomous"
            trust = "0.4"
        elif tags & {"feedback-signal"}:
            source = "agent-interactive"
            trust = "0.7"
        elif tags & {"action-plan"}:
            source = "agent-autonomous"
            trust = "0.4"
        elif tags & {"consolidated-meta", "background-insights-latest"}:
            source = "system"
            trust = "0.6"
        else:
            source = "agent"
            trust = "0.5"

        meta["source"] = source
        meta.setdefault("verified", "false")
        meta.setdefault("trust_score", trust)

        print(f"  STAMP [{rec.id[:8]}] source={source} tags={list(tags)[:3]}")

        if not DRY_RUN:
            brain.update(rec.id, metadata=meta)
        stamped += 1

    print(f"\nStamped: {stamped} records")
    print()

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Downgraded levels: {downgraded}")
    print(f"  Deleted duplicates: {deleted}")
    print(f"  Stamped provenance: {stamped}")

    if DRY_RUN:
        print("\n  *** DRY RUN — no changes made ***")
        print("  Run without --dry-run to apply changes.")
    else:
        # Final stats
        final_recs = brain.search(limit=5000)
        print(f"\n  Records before: {len(all_recs)}")
        print(f"  Records after: {len(final_recs)}")

        # Level distribution
        level_counts: dict[str, int] = {}
        for r in final_recs:
            lvl = str(r.level)
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
        print("\n  Level distribution:")
        for lvl, count in sorted(level_counts.items(), key=lambda x: -x[1]):
            print(f"    {lvl}: {count}")

    brain.close()


if __name__ == "__main__":
    main()
