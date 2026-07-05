"""
AuraSDK Live Benchmark — realistic latency at scale.

Creates a temporary brain, fills it with realistic agent data,
measures recall/search/store latency at 1K, 5K, 10K records.
Does NOT touch the production brain.

Usage:
    python scripts/bench_aura_live.py
"""

import os
import random
import shutil
import statistics
import string
import tempfile
import time

from aura import Aura, Level

# ============== Test data generators ==============

GOAL_TEMPLATES = [
    "Research {} integration patterns for AI agents",
    "Analyze {} performance benchmarks vs competitors",
    "Write technical blog post about {} architecture",
    "Investigate {} pricing model and cost optimization",
    "Build proof of concept for {} memory layer",
    "Deploy {} monitoring dashboard for production",
    "Create documentation for {} API endpoints",
    "Optimize {} query latency under heavy load",
    "Set up CI/CD pipeline for {} module",
    "Implement {} caching strategy for embeddings",
]

OUTCOME_TEMPLATES = [
    "Successfully completed {} — latency improved by {}%",
    "Failed to {} due to API rate limiting, will retry",
    "Partially completed {} — need user approval for next step",
    "Research on {} found {} relevant papers and {} tools",
    "Deployed {} to staging — all tests passing",
    "Benchmark results: {} ops/sec with {}ms p99 latency",
    "User feedback on {}: positive, requesting {} improvements",
    "Cost analysis: {} saves ${}/ month vs current solution",
]

FACT_TEMPLATES = [
    "{} uses {} architecture with {} backend",
    "Market size for {} estimated at ${} billion by 2027",
    "{} raised ${} million in Series {} funding",
    "Key competitor {} has {} GitHub stars and {} monthly downloads",
    "{} performance: {}ms latency, {} requests/sec throughput",
    "User survey: {}% prefer {} over {} for agent memory",
]

TOPICS = [
    "vector database",
    "LLM memory",
    "agent orchestration",
    "RAG pipeline",
    "knowledge graph",
    "semantic search",
    "embedding model",
    "context window",
    "tool calling",
    "function dispatch",
    "prompt engineering",
    "fine-tuning",
    "reinforcement learning",
    "RLHF",
    "constitutional AI",
    "chain of thought",
    "multi-agent system",
    "autonomous agent",
    "memory consolidation",
    "decay",
    "Rust FFI",
    "PyO3 binding",
    "WebSocket streaming",
    "event bus",
    "TronGrid API",
    "crypto wallet",
    "survival economics",
    "token budget",
    "AuraSDK",
    "Mem0",
    "Zep",
    "Letta",
    "LangGraph",
    "CrewAI",
    "AutoGPT",
    "Redis cache",
    "SQLite",
    "PostgreSQL",
    "MongoDB",
    "Elasticsearch",
]

TAGS_POOL = [
    "autonomous-goal",
    "autonomous-outcome",
    "research-finding",
    "fact",
    "todo-item",
    "session-summary",
    "scheduled-task",
    "proactive-session",
    "identity-core",
    "user-preference",
    "tool-status",
    "error-log",
    "competitive-analysis",
]


def _rand_text(template_list: list[str]) -> str:
    tmpl = random.choice(template_list)
    # Count {} placeholders
    count = tmpl.count("{}")
    fillers = []
    for _ in range(count):
        fillers.append(
            random.choice(
                TOPICS
                + [
                    str(random.randint(1, 999)),
                    f"{random.uniform(0.1, 99.9):.1f}",
                    random.choice(string.ascii_uppercase),
                ]
            )
        )
    return tmpl.format(*fillers)


def generate_records(n: int) -> list[dict]:
    """Generate n realistic records."""
    records = []
    for i in range(n):
        kind = random.choice(["goal", "outcome", "fact"])
        if kind == "goal":
            content = _rand_text(GOAL_TEMPLATES)
            tags = ["autonomous-goal", f"priority-{random.choice(['high', 'medium', 'low'])}"]
            level = Level.Domain
        elif kind == "outcome":
            content = _rand_text(OUTCOME_TEMPLATES)
            tags = ["autonomous-outcome", random.choice(TAGS_POOL)]
            level = Level.Working
        else:
            content = _rand_text(FACT_TEMPLATES)
            tags = ["fact", random.choice(TAGS_POOL)]
            level = Level.Domain

        records.append(
            {
                "content": content,
                "level": level,
                "tags": tags,
                "metadata": {
                    "type": kind,
                    "index": str(i),
                    "created_at": f"2026-03-0{random.randint(1, 5)}T{random.randint(10, 23)}:00:00",
                },
            }
        )
    return records


# ============== Benchmark runners ==============


def bench_store(brain: Aura, records: list[dict]) -> float:
    """Store records, return avg latency in ms."""
    times = []
    for rec in records:
        t0 = time.perf_counter_ns()
        brain.store(
            content=rec["content"],
            level=rec["level"],
            tags=rec["tags"],
            metadata=rec["metadata"],
        )
        elapsed = (time.perf_counter_ns() - t0) / 1_000_000  # ns → ms
        times.append(elapsed)
    return statistics.mean(times)


def bench_recall(brain: Aura, queries: list[str], n_runs: int = 3) -> dict:
    """Run recall benchmark. Returns {avg, p50, p95, p99, min, max} in ms."""
    times = []
    for _ in range(n_runs):
        for q in queries:
            t0 = time.perf_counter_ns()
            brain.recall(q, token_budget=2000)
            elapsed = (time.perf_counter_ns() - t0) / 1_000_000
            times.append(elapsed)

    times.sort()
    n = len(times)
    return {
        "avg": statistics.mean(times),
        "p50": times[n // 2],
        "p95": times[int(n * 0.95)],
        "p99": times[int(n * 0.99)],
        "min": times[0],
        "max": times[-1],
        "samples": n,
    }


def bench_search(brain: Aura, n_runs: int = 30) -> dict:
    """Benchmark tag-based search."""
    tags_to_test = ["autonomous-goal", "autonomous-outcome", "fact"]
    times = []
    for _ in range(n_runs):
        for tag in tags_to_test:
            t0 = time.perf_counter_ns()
            brain.search(query="", tags=[tag], limit=20)
            elapsed = (time.perf_counter_ns() - t0) / 1_000_000
            times.append(elapsed)

    times.sort()
    n = len(times)
    return {
        "avg": statistics.mean(times),
        "p50": times[n // 2],
        "p95": times[int(n * 0.95)],
        "p99": times[int(n * 0.99)],
        "min": times[0],
        "max": times[-1],
        "samples": n,
    }


# ============== Main ==============


RECALL_QUERIES = [
    "AuraSDK performance benchmarks",
    "how to optimize agent memory latency",
    "competitor analysis Mem0 vs Zep",
    "survival economics wallet balance",
    "Reddit marketing strategy for SDK",
    "LLM token cost optimization",
    "autonomous goal completion rate",
    "Rust FFI Python binding overhead",
    "vector search vs keyword recall",
    "knowledge graph connection patterns",
]

SCALE_POINTS = [1000, 5000, 10000]


def run_benchmark():
    tmpdir = tempfile.mkdtemp(prefix="aura_bench_")
    brain_path = os.path.join(tmpdir, "bench_brain")

    print("=" * 65)
    print("  AuraSDK Live Benchmark (aura-memory 1.3.0)")
    print("=" * 65)
    print(f"  Temp brain: {brain_path}")
    print(f"  Scale points: {SCALE_POINTS}")
    print(f"  Recall queries: {len(RECALL_QUERIES)} x 3 runs each")
    print()

    brain = Aura(brain_path)
    current_count = 0

    results = []

    for target in SCALE_POINTS:
        to_add = target - current_count
        print(f"--- Filling to {target:,} records (+{to_add:,}) ---")

        records = generate_records(to_add)
        store_avg = bench_store(brain, records)
        current_count = target

        print(f"  Store avg: {store_avg:.3f} ms/record")

        # Warmup recall
        for q in RECALL_QUERIES[:3]:
            brain.recall(q, token_budget=2000)

        recall = bench_recall(brain, RECALL_QUERIES)
        search = bench_search(brain)

        results.append(
            {
                "records": target,
                "store_avg": store_avg,
                "recall": recall,
                "search": search,
            }
        )

        print(
            f"  Recall:  avg={recall['avg']:.3f}ms  p50={recall['p50']:.3f}ms  "
            f"p95={recall['p95']:.3f}ms  p99={recall['p99']:.3f}ms  "
            f"min={recall['min']:.3f}ms  max={recall['max']:.3f}ms"
        )
        print(
            f"  Search:  avg={search['avg']:.3f}ms  p50={search['p50']:.3f}ms  "
            f"p95={search['p95']:.3f}ms  p99={search['p99']:.3f}ms"
        )
        print()

    # Summary table
    print("=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    print(
        f"  {'Records':>8}  {'Store':>8}  {'Recall avg':>10}  {'Recall p50':>10}  "
        f"{'Recall p95':>10}  {'Recall p99':>10}  {'Search avg':>10}"
    )
    print(
        f"  {'':>8}  {'(ms)':>8}  {'(ms)':>10}  {'(ms)':>10}  "
        f"{'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}"
    )
    print("  " + "-" * 76)

    for r in results:
        print(
            f"  {r['records']:>8,}  {r['store_avg']:>8.3f}  "
            f"{r['recall']['avg']:>10.3f}  {r['recall']['p50']:>10.3f}  "
            f"{r['recall']['p95']:>10.3f}  {r['recall']['p99']:>10.3f}  "
            f"{r['search']['avg']:>10.3f}"
        )

    print()

    # Scaling analysis
    if len(results) >= 2:
        r1, r_last = results[0], results[-1]
        scale_factor = r_last["records"] / r1["records"]
        recall_factor = (
            r_last["recall"]["avg"] / r1["recall"]["avg"] if r1["recall"]["avg"] > 0 else 0
        )
        print(
            f"  Scaling: {r1['records']:,} -> {r_last['records']:,} records "
            f"({scale_factor:.0f}x data)"
        )
        print(
            f"  Recall degradation: {recall_factor:.2f}x "
            f"({'sublinear - GOOD' if recall_factor < scale_factor else 'linear or worse'})"
        )

    print()

    # Cleanup
    brain.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("  Cleaned up temp brain.")
    print("  Done!")


if __name__ == "__main__":
    run_benchmark()
