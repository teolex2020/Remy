"""
Brain Cleanup — remove bloated background-insight and cross-connection records.

These records accumulated due to Aura Cognitive v1.1.0 auto-promotion
(WORKING → IDENTITY) when background_brain.py repeatedly accessed them.

Usage:
    python scripts/cleanup_brain.py          # dry run (show what would be deleted)
    python scripts/cleanup_brain.py --apply  # actually delete

Safe to run multiple times. Only deletes records with specific tags.
"""

import sys
from collections import Counter
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from remy.core.agent_tools import brain


def main():
    apply = "--apply" in sys.argv

    all_records = brain.list_records()
    total = len(all_records)

    print(f"=== Brain Cleanup {'(DRY RUN)' if not apply else '(APPLYING)'} ===")
    print(f"Total records: {total}")
    print()

    # Categorize records to delete
    to_delete = []
    keep_tags = {"user-profile", "identity", "autonomous-goal", "scheduled-task",
                 "research-project", "health-metric", "session-summary",
                 "action-plan", "session-reflection", "extracted-fact"}

    for r in all_records:
        tags = set(r.tags or [])

        # Delete background-insight records (they're transient by design)
        if "background-insight" in tags:
            to_delete.append((r, "background-insight"))
            continue

        # Delete cross-connection records
        if "cross-connection" in tags:
            to_delete.append((r, "cross-connection"))
            continue

        # Delete web-search-cache (expired caches)
        if "web-search-cache" in tags:
            to_delete.append((r, "web-search-cache"))
            continue

    # Show summary
    by_reason = Counter(reason for _, reason in to_delete)
    print("Records to delete:")
    for reason, count in by_reason.most_common():
        print(f"  {reason}: {count}")
    print(f"  TOTAL: {len(to_delete)}")
    print(f"  Keeping: {total - len(to_delete)}")
    print()

    if not to_delete:
        print("Nothing to clean up!")
        return

    if not apply:
        print("Run with --apply to actually delete these records.")
        print(f"  python scripts/cleanup_brain.py --apply")
        return

    # Delete
    deleted = 0
    failed = 0
    for r, reason in to_delete:
        try:
            brain.delete(r.id)
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete {r.id[:8]}: {e}")
            failed += 1

    print(f"Deleted: {deleted}")
    if failed:
        print(f"Failed: {failed}")

    # Show final stats
    remaining = brain.count()
    print(f"\nFinal record count: {remaining}")


if __name__ == "__main__":
    main()
