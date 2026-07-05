#!/usr/bin/env python3
"""
inspect_brain.py — snapshot of Aura cognitive memory state.

Usage:
    python scripts/inspect_brain.py
    python scripts/inspect_brain.py --path path/to/brain
    python scripts/inspect_brain.py --full       # show all records, not just top-10
"""

import argparse
import sys
from pathlib import Path

# ── resolve brain path ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inspect Aura cognitive memory.")
parser.add_argument("--path", default=None, help="Path to brain directory")
parser.add_argument("--full", action="store_true", help="Show all records")
args = parser.parse_args()

if args.path:
    BRAIN_PATH = Path(args.path)
else:
    # default: repo-root data directory, independent of where this script lives
    BRAIN_PATH = Path(__file__).resolve().parents[1] / "data" / "brain"

if not BRAIN_PATH.exists():
    print(f"[error] Brain path not found: {BRAIN_PATH}")
    sys.exit(1)

# ── import aura ───────────────────────────────────────────────────────────────
try:
    from aura import Aura, Level
except ImportError:
    print("[error] aura_memory not installed. Run: pip install vendor/aura_memory-*.whl")
    sys.exit(1)

print(f"\n{'═'*60}")
print(f"  AURA BRAIN INSPECTOR")
print(f"  {BRAIN_PATH}")
print(f"{'═'*60}\n")

brain = Aura(str(BRAIN_PATH))

# ── 1. record counts by level ─────────────────────────────────────────────────
print("── RECORDS BY LEVEL ──────────────────────────────────────")
def recall_all(brain, top_k=2000):
    """Fetch all records via search() — works across all wheel versions."""
    try:
        return brain.search(query="", limit=top_k)
    except Exception:
        pass
    # fallback: recall_structured with real query to bypass empty-string RRF skip
    try:
        return brain.recall_structured("the", top_k=top_k, min_strength=0.0)
    except Exception:
        pass
    return []

all_records = recall_all(brain)
total = 0
for level in [Level.Identity, Level.Domain, Level.Decisions, Level.Working]:
    count = sum(1 for r in all_records if r.level == level)
    total += count
    name = str(level).replace("Level.", "")
    bar = "█" * min(count, 40)
    print(f"  {name:<12} {count:>4}  {bar}")

total = len(all_records)
print(f"  {'TOTAL':<12} {total:>4}")

# ── 2. strength distribution ──────────────────────────────────────────────────
print("\n── STRENGTH DISTRIBUTION ─────────────────────────────────")
buckets = {"0.8-1.0": 0, "0.6-0.8": 0, "0.4-0.6": 0, "0.2-0.4": 0, "0.0-0.2": 0}
for r in all_records:
    s = r.strength
    if s >= 0.8:
        buckets["0.8-1.0"] += 1
    elif s >= 0.6:
        buckets["0.6-0.8"] += 1
    elif s >= 0.4:
        buckets["0.4-0.6"] += 1
    elif s >= 0.2:
        buckets["0.2-0.4"] += 1
    else:
        buckets["0.0-0.2"] += 1

for label, count in buckets.items():
    bar = "█" * min(count, 40)
    print(f"  {label}  {count:>4}  {bar}")

# ── 3. semantic type breakdown ────────────────────────────────────────────────
print("\n── SEMANTIC TYPES ────────────────────────────────────────")
types: dict[str, int] = {}
for r in all_records:
    st = getattr(r, "semantic_type", None) or "unknown"
    types[st] = types.get(st, 0) + 1

for st, count in sorted(types.items(), key=lambda x: -x[1]):
    bar = "█" * min(count, 40)
    print(f"  {st:<20} {count:>4}  {bar}")

# ── 4. top 10 strongest records ───────────────────────────────────────────────
limit = total if args.full else 10
sorted_records = sorted(all_records, key=lambda r: r.strength, reverse=True)
print(f"\n── TOP {limit} STRONGEST RECORDS {'(all)' if args.full else ''} ──────────────────────────")
for i, r in enumerate(sorted_records[:limit], 1):
    content = r.content[:80].replace("\n", " ")
    level_name = str(r.level).replace("Level.", "")
    st = getattr(r, "semantic_type", "?") or "?"
    print(f"  {i:>3}. [{r.strength:.3f}] [{level_name:<9}] [{st:<15}] {content}")

# ── 5. surfaced concepts ──────────────────────────────────────────────────────
print("\n── SURFACED CONCEPTS ─────────────────────────────────────")
try:
    concepts = brain.get_surfaced_concepts(10)
    if not concepts:
        print("  (none — need more maintenance cycles or more records per topic)")
    else:
        for c in concepts:
            print(f"  [{c.score:.3f}] [{c.state}] {c.label}")
            print(f"         ns={c.namespace}  beliefs={len(c.belief_ids)}  records={len(c.record_ids)}")
            if c.tags:
                print(f"         tags={c.tags[:5]}")
except AttributeError:
    print("  (get_surfaced_concepts not available — upgrade aura_memory wheel)")

# ── 6. surfaced causal patterns ───────────────────────────────────────────────
print("\n── SURFACED CAUSAL PATTERNS ──────────────────────────────")
try:
    patterns = brain.get_surfaced_causal_patterns(10)
    if not patterns:
        print("  (none — need caused_by links + repeated co-occurrence)")
    else:
        for p in patterns:
            print(f"  [{p.score:.3f}] [{p.state}] {p.cause_label} → {p.effect_label}")
            print(f"         ns={p.namespace}  support={p.support}  consistency={p.temporal_consistency:.2f}")
except AttributeError:
    print("  (get_surfaced_causal_patterns not available — upgrade aura_memory wheel)")

# ── 7. surfaced policy hints ──────────────────────────────────────────────────
print("\n── SURFACED POLICY HINTS ─────────────────────────────────")
try:
    hints = brain.get_surfaced_policy_hints(10)
    if not hints:
        print("  (none — need stable causal patterns first)")
    else:
        for h in hints:
            action = getattr(h, "action_kind", "?")
            domain = getattr(h, "domain", "?")
            print(f"  [{h.strength:.3f}] [{action:<12}] [{domain}]")
            print(f"         {h.description[:80]}")
except AttributeError:
    print("  (get_surfaced_policy_hints not available — upgrade aura_memory wheel)")

# ── 8. maintenance report (quick run) ────────────────────────────────────────
print("\n── MAINTENANCE SNAPSHOT ──────────────────────────────────")
try:
    report = brain.run_maintenance()
    print(f"  phases completed: {getattr(report, 'phases_completed', '?')}")
    print(f"  records retained: {getattr(report, 'records_retained', '?')}")
    print(f"  records archived: {getattr(report, 'records_archived', '?')}")
    print(f"  records promoted: {getattr(report, 'records_promoted', '?')}")
    timings = getattr(report, "phase_timings", None)
    if timings:
        total_ms = getattr(timings, "total_ms", None)
        if total_ms:
            print(f"  total cycle time: {total_ms:.2f} ms")
except Exception as e:
    print(f"  (maintenance error: {e})")

print(f"\n{'═'*60}\n")
brain.close()
