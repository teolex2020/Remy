"""
AuraSDK Detailed Benchmark — isolate where latency comes from.

Tests recall vs recall_structured vs search separately,
cold vs warm, and shows distribution histograms.

Usage:
    python scripts/bench_aura_detailed.py
"""

import os
import random
import shutil
import statistics
import tempfile
import time

from aura import Aura, Level

# Reuse generators from main bench
from bench_aura_live import RECALL_QUERIES, generate_records


def _measure(fn, n_runs=50):
    """Run fn() n times, return sorted list of times in ms."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        elapsed = (time.perf_counter_ns() - t0) / 1_000_000
        times.append(elapsed)
    times.sort()
    return times


def _stats(times):
    n = len(times)
    return {
        "avg": statistics.mean(times),
        "p50": times[n // 2],
        "p95": times[int(n * 0.95)],
        "p99": times[int(n * 0.99)],
        "min": times[0],
        "max": times[-1],
    }


def _print_stats(label, times):
    s = _stats(times)
    print(
        f"  {label:<25} avg={s['avg']:>8.3f}ms  p50={s['p50']:>8.3f}ms  "
        f"p95={s['p95']:>8.3f}ms  min={s['min']:>8.3f}ms  max={s['max']:>8.3f}ms"
    )


def _print_histogram(label, times, buckets=None):
    """Show distribution of times."""
    if buckets is None:
        buckets = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0]
    counts = [0] * (len(buckets) + 1)
    for t in times:
        placed = False
        for i, b in enumerate(buckets):
            if t <= b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    print(f"\n  {label} distribution ({len(times)} samples):")
    prev = 0
    for i, b in enumerate(buckets):
        bar = "#" * counts[i]
        if counts[i] > 0:
            print(f"    {prev:>6.2f} - {b:>6.1f} ms: {counts[i]:>4}  {bar}")
        prev = b
    if counts[-1] > 0:
        print(f"    {buckets[-1]:>6.1f}+     ms: {counts[-1]:>4}  {'#' * counts[-1]}")


def run():
    tmpdir = tempfile.mkdtemp(prefix="aura_detail_")
    brain_path = os.path.join(tmpdir, "detail_brain")

    print("=" * 70)
    print("  AuraSDK Detailed Benchmark")
    print("=" * 70)

    for record_count in [1000, 5000, 10000]:
        print(f"\n{'=' * 70}")
        print(f"  {record_count:,} RECORDS")
        print(f"{'=' * 70}")

        # Fresh brain for each scale
        if os.path.exists(brain_path):
            shutil.rmtree(brain_path, ignore_errors=True)
        brain = Aura(brain_path)

        # Fill
        print(f"\n  Filling {record_count:,} records...")
        records = generate_records(record_count)
        t0 = time.perf_counter_ns()
        for rec in records:
            brain.store(
                content=rec["content"],
                level=rec["level"],
                tags=rec["tags"],
                metadata=rec["metadata"],
            )
        fill_ms = (time.perf_counter_ns() - t0) / 1_000_000
        print(f"  Fill time: {fill_ms:.0f}ms ({fill_ms / record_count:.3f}ms/record)")

        queries = RECALL_QUERIES * 5  # 50 queries

        # --- 1. recall (unified) — COLD ---
        print("\n  --- recall (unified, token_budget=2000) ---")
        cold_times = _measure(
            lambda: brain.recall(random.choice(queries), token_budget=2000), n_runs=10
        )
        _print_stats("COLD (first 10)", cold_times)

        warm_times = _measure(
            lambda: brain.recall(random.choice(queries), token_budget=2000), n_runs=50
        )
        _print_stats("WARM (next 50)", warm_times)
        _print_histogram("recall", warm_times)

        # --- 2. recall_structured (RRF only) ---
        print("\n  --- recall_structured (RRF, top_k=10) ---")
        rrf_times = _measure(
            lambda: brain.recall_structured(random.choice(queries), top_k=10), n_runs=50
        )
        _print_stats("recall_structured", rrf_times)
        _print_histogram("recall_structured", rrf_times)

        # --- 3. search (tag-based) ---
        print("\n  --- search (tag-based, limit=20) ---")
        search_times = _measure(
            lambda: brain.search(query="", tags=["autonomous-goal"], limit=20), n_runs=50
        )
        _print_stats("search by tag", search_times)

        search_q_times = _measure(lambda: brain.search(query="agent memory", limit=20), n_runs=50)
        _print_stats("search by query", search_q_times)

        # --- 4. store (single record) ---
        print(f"\n  --- store (single record into {record_count:,}) ---")
        store_times = _measure(
            lambda: brain.store(
                content=f"Test record {random.randint(0, 999999)}",
                level=Level.Working,
                tags=["bench-test"],
                metadata={"type": "bench"},
            ),
            n_runs=50,
        )
        _print_stats("store", store_times)

        # --- 5. recall_full if available ---
        if hasattr(brain, "recall_full"):
            print("\n  --- recall_full (unified v2) ---")
            rf_times = _measure(
                lambda: brain.recall_full(random.choice(queries), top_k=10), n_runs=50
            )
            _print_stats("recall_full", rf_times)

        brain.close()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\n  Cleaned up. Done!")


if __name__ == "__main__":
    run()
