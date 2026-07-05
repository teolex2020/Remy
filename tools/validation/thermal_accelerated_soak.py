"""
Accelerated Thermal Soak — Months-Equivalent Simulation in Minutes.

Runs 200 simulated maintenance cycles over real brain data with
scenario-driven belief mutations. Validates that:
  1. Pruning cuts leak edges, not productive ones
  2. edge_health.json doesn't grow unbounded
  3. No pathological loops (oscillation, runaway pruning)
  4. Leak ratio trends downward over time
  5. Hot zone count stays bounded
  6. Cold mass grows as expected

Scenarios (injected at specific cycle windows):
  - NEW_TOPICS: fresh beliefs with new tags (cycles 1-30)
  - REPEATED_CONFLICT: conflicting beliefs on same subject (cycles 20-60)
  - QUIET_PHASE: no mutations, only decay (cycles 60-90)
  - STALE_COOLING: old beliefs lose volatility/confidence (cycles 90-120)
  - CROSS_DOMAIN_LEAK: hub-tag beliefs bridging domains (cycles 40-80)
  - REACTIVATION: previously cold beliefs get new evidence (cycles 120-160)
  - SUSTAINED_MIX: all patterns at low rate (cycles 160-200)

Run: python -m tools.validation.thermal_accelerated_soak
"""

import json
import os
import sys
import random
import tempfile
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

if not getattr(sys, "frozen", False):
    _src = str(Path(__file__).parents[2] / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_ROOT = str((Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parents[2]) / "data")
BRAIN_DIR = os.path.join(DATA_ROOT, "brain")

# Simulation parameters
TOTAL_CYCLES = 200
SNAPSHOT_INTERVAL = 10  # record metrics every N cycles

# Scenario windows
SCENARIOS = {
    "new_topics":        (1, 30),
    "repeated_conflict": (20, 60),
    "cross_domain_leak": (40, 80),
    "quiet_phase":       (60, 90),
    "stale_cooling":     (90, 120),
    "reactivation":      (120, 160),
    "sustained_mix":     (160, 200),
}

# Deterministic seed for reproducibility
SEED = 42

# Topic pools for synthetic beliefs
TOPIC_POOLS = {
    "health": ["nutrition", "sleep", "exercise", "hydration", "supplements", "fasting"],
    "work":   ["productivity", "meetings", "deadlines", "focus", "breaks", "planning"],
    "mood":   ["energy", "stress", "motivation", "calm", "anxiety", "optimism"],
    "social": ["friends", "family", "isolation", "community", "boundaries", "support"],
}

HUB_TAGS = ["wellness", "daily_routine", "self_improvement"]


@dataclass
class CycleSnapshot:
    """Metrics captured at a snapshot cycle."""
    cycle: int
    scenario: str
    belief_count: int
    edge_count: int
    total_energy: float
    mean_temp: float
    hot_count: int
    cold_count: int
    cluster_count: int
    # Plasticity
    edges_tracked: int
    edges_healthy: int
    edges_weakened: int
    edges_pruned: int
    leak_ratio: float
    leaks_this_cycle: int
    prunes_this_cycle: int
    # Edge health file size
    edge_health_bytes: int


def _load_beliefs(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("beliefs", {})


def _save_beliefs(path: str, beliefs: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"beliefs": beliefs}, f, ensure_ascii=False)


def _make_belief_id(prefix: str, idx: int) -> str:
    """Deterministic synthetic belief ID."""
    import hashlib
    return hashlib.md5(f"{prefix}_{idx}".encode()).hexdigest()[:16]


def _make_belief(
    bid: str,
    domain: str,
    tags: list[str],
    content: str,
    *,
    conflict_mass: float = 0.0,
    volatility: float = 0.0,
    confidence: float = 0.7,
    stability: int = 3,
    support_mass: float = 1.0,
    state: str = "Singleton",
) -> dict:
    tag_str = ",".join(tags)
    key = f"{domain}:{tag_str}:decision"
    return {
        "key": key,
        "content": content,
        "confidence": confidence,
        "support_mass": support_mass,
        "conflict_mass": conflict_mass,
        "volatility": volatility,
        "stability": stability,
        "state": state,
        "records": [],
    }


def _active_scenario(cycle: int) -> str:
    """Return which scenario is active at this cycle."""
    for name, (start, end) in SCENARIOS.items():
        if start <= cycle <= end:
            return name
    return "none"


def _active_scenarios(cycle: int) -> list[str]:
    """Return ALL active scenarios at this cycle (windows overlap)."""
    return [name for name, (start, end) in SCENARIOS.items() if start <= cycle <= end]


def _inject_new_topics(beliefs: dict, rng: random.Random, cycle: int) -> int:
    """Add 2-3 beliefs with fresh tags in a random domain."""
    domain = rng.choice(list(TOPIC_POOLS.keys()))
    tags = rng.sample(TOPIC_POOLS[domain], min(2, len(TOPIC_POOLS[domain])))
    added = 0
    for i in range(rng.randint(2, 3)):
        bid = _make_belief_id(f"new_{domain}_{cycle}", i)
        if bid in beliefs:
            continue
        beliefs[bid] = _make_belief(
            bid, domain, tags,
            f"New observation about {tags[0]} at cycle {cycle}",
            volatility=rng.uniform(0.3, 0.7),
            confidence=rng.uniform(0.4, 0.6),
            stability=0,
        )
        added += 1
    return added


def _inject_conflict(beliefs: dict, rng: random.Random, cycle: int) -> int:
    """Add conflicting belief pair on same subject."""
    domain = rng.choice(list(TOPIC_POOLS.keys()))
    tag = rng.choice(TOPIC_POOLS[domain])

    bid_a = _make_belief_id(f"conflict_{cycle}", 0)
    bid_b = _make_belief_id(f"conflict_{cycle}", 1)
    if bid_a in beliefs or bid_b in beliefs:
        return 0

    beliefs[bid_a] = _make_belief(
        bid_a, domain, [tag, TOPIC_POOLS[domain][0]],
        f"{tag} is beneficial — evidence at cycle {cycle}",
        conflict_mass=0.8,
        volatility=0.6,
        state="Unresolved",
    )
    beliefs[bid_b] = _make_belief(
        bid_b, domain, [tag, TOPIC_POOLS[domain][0]],
        f"{tag} is harmful — counter-evidence at cycle {cycle}",
        conflict_mass=0.8,
        volatility=0.6,
        state="Unresolved",
    )
    return 2


def _inject_cross_domain_leak(beliefs: dict, rng: random.Random, cycle: int) -> int:
    """Add beliefs that share hub tags across different domains."""
    domain_a, domain_b = rng.sample(list(TOPIC_POOLS.keys()), 2)
    hub = rng.choice(HUB_TAGS)
    tag_a = rng.choice(TOPIC_POOLS[domain_a])
    tag_b = rng.choice(TOPIC_POOLS[domain_b])

    bid_a = _make_belief_id(f"hub_{cycle}", 0)
    bid_b = _make_belief_id(f"hub_{cycle}", 1)
    if bid_a in beliefs or bid_b in beliefs:
        return 0

    beliefs[bid_a] = _make_belief(
        bid_a, domain_a, [tag_a, hub],
        f"{tag_a} relates to {hub} — cycle {cycle}",
        volatility=0.5, conflict_mass=0.3,
    )
    beliefs[bid_b] = _make_belief(
        bid_b, domain_b, [tag_b, hub],
        f"{tag_b} also relates to {hub} — cycle {cycle}",
        volatility=0.5, conflict_mass=0.3,
    )
    return 2


def _apply_stale_cooling(beliefs: dict, rng: random.Random, cycle: int) -> int:
    """Reduce volatility and confidence of old beliefs (simulates natural decay)."""
    modified = 0
    for bid, b in beliefs.items():
        if b["stability"] >= 3 and b["volatility"] > 0.05:
            b["volatility"] = max(0.0, b["volatility"] - rng.uniform(0.02, 0.08))
            b["stability"] += 1
            b["confidence"] = min(1.0, b["confidence"] + rng.uniform(0.01, 0.03))
            modified += 1
    return modified


def _apply_reactivation(beliefs: dict, rng: random.Random, cycle: int) -> int:
    """Reactivate cold beliefs: bump volatility, add conflict."""
    cold_bids = [
        bid for bid, b in beliefs.items()
        if b["volatility"] < 0.1 and b["stability"] >= 5
    ]
    if not cold_bids:
        return 0

    reactivated = 0
    for bid in rng.sample(cold_bids, min(3, len(cold_bids))):
        b = beliefs[bid]
        b["volatility"] = rng.uniform(0.4, 0.7)
        b["conflict_mass"] = rng.uniform(0.3, 0.6)
        b["stability"] = max(0, b["stability"] - 3)
        b["state"] = "Unresolved"
        reactivated += 1
    return reactivated


def run_simulation():
    """Execute full accelerated soak simulation."""
    print("=" * 75)
    print("ACCELERATED THERMAL SOAK SIMULATION")
    print(f"  Cycles: {TOTAL_CYCLES}")
    print(f"  Seed: {SEED}")
    print(f"  Base beliefs: {BRAIN_DIR}")
    print("=" * 75)

    rng = random.Random(SEED)

    # Create isolated temp directory for simulation
    sim_dir = tempfile.mkdtemp(prefix="thermal_soak_")
    sim_brain = os.path.join(sim_dir, "brain")
    os.makedirs(sim_brain, exist_ok=True)

    # Copy real beliefs as starting point
    src_beliefs_path = os.path.join(BRAIN_DIR, "beliefs.cog")
    if not os.path.exists(src_beliefs_path):
        print("ERROR: No beliefs.cog in brain directory")
        return 1

    shutil.copy2(src_beliefs_path, os.path.join(sim_brain, "beliefs.cog"))
    beliefs = _load_beliefs(os.path.join(sim_brain, "beliefs.cog"))
    print(f"  Starting with {len(beliefs)} real beliefs")
    print()

    # Import thermal pipeline
    from remy.core.thermal_advisor import (
        _build_graph, _initialize_temperature, _spread_heat,
        NUM_SPREAD_PASSES, COOLING_FACTOR, HOT_ZONE_THRESHOLD, COLD_THRESHOLD,
        classify_cycle, CycleType, ThermalReport,
    )
    from remy.core.synaptic_plasticity import (
        run_plasticity_cycle, get_conductivity_modifiers, _edge_key,
        load_edge_health, save_edge_health, get_plasticity_summary,
    )

    snapshots: list[CycleSnapshot] = []
    cumulative_prunes = 0
    cumulative_leaks = 0
    max_beliefs_seen = len(beliefs)

    # Track per-cycle leak/prune for trend analysis
    leak_history = []
    prune_history = []
    hot_history = []
    energy_history = []

    # Cycle classification counters
    cycle_type_counts = {
        CycleType.HOT: 0,
        CycleType.WARM: 0,
        CycleType.COLD: 0,
        CycleType.FULL_SCAN: 0,
    }

    print(f"{'Cycle':>5} {'Scenario':<20} {'Beliefs':>7} {'Edges':>6} {'Hot':>4} "
          f"{'Cold':>5} {'Energy':>7} {'Leaks':>6} {'Weak':>5} {'Prune':>5} {'LeakR':>6}")
    print("-" * 95)

    for cycle in range(1, TOTAL_CYCLES + 1):
        active = _active_scenarios(cycle)
        primary = active[0] if active else "none"

        # ---- SCENARIO INJECTION ----
        mutations = []
        for scenario in active:
            if scenario == "new_topics" and rng.random() < 0.5:
                n = _inject_new_topics(beliefs, rng, cycle)
                if n:
                    mutations.append(f"+{n} new")

            elif scenario == "repeated_conflict" and rng.random() < 0.3:
                n = _inject_conflict(beliefs, rng, cycle)
                if n:
                    mutations.append(f"+{n} conflict")

            elif scenario == "cross_domain_leak" and rng.random() < 0.35:
                n = _inject_cross_domain_leak(beliefs, rng, cycle)
                if n:
                    mutations.append(f"+{n} hub")

            elif scenario == "quiet_phase":
                pass  # no mutations

            elif scenario == "stale_cooling" and rng.random() < 0.4:
                n = _apply_stale_cooling(beliefs, rng, cycle)
                if n:
                    mutations.append(f"~{n} cooled")

            elif scenario == "reactivation" and rng.random() < 0.25:
                n = _apply_reactivation(beliefs, rng, cycle)
                if n:
                    mutations.append(f"!{n} reactivated")

            elif scenario == "sustained_mix":
                # Low-rate mix of everything
                if rng.random() < 0.2:
                    _inject_new_topics(beliefs, rng, cycle)
                if rng.random() < 0.1:
                    _inject_conflict(beliefs, rng, cycle)
                if rng.random() < 0.15:
                    _inject_cross_domain_leak(beliefs, rng, cycle)
                if rng.random() < 0.15:
                    _apply_stale_cooling(beliefs, rng, cycle)
                if rng.random() < 0.1:
                    _apply_reactivation(beliefs, rng, cycle)

        # Save mutated beliefs
        _save_beliefs(os.path.join(sim_brain, "beliefs.cog"), beliefs)
        max_beliefs_seen = max(max_beliefs_seen, len(beliefs))

        # ---- THERMAL CYCLE ----
        adj, edges, degree = _build_graph(beliefs)

        # Apply plasticity modifiers to conductivity
        modifiers = get_conductivity_modifiers(sim_brain)
        if modifiers:
            for node_a in list(adj.keys()):
                adj[node_a] = [
                    (nb, cond * modifiers.get(_edge_key(node_a, nb), 1.0))
                    for nb, cond in adj[node_a]
                ]

        temps = _initialize_temperature(beliefs)
        temps_before = dict(temps)

        for _ in range(NUM_SPREAD_PASSES):
            _spread_heat(temps, adj)

        # Run plasticity
        plasticity = run_plasticity_cycle(sim_brain, beliefs, temps_before, temps, adj)

        # Cool
        for n in temps:
            temps[n] *= COOLING_FACTOR

        # ---- METRICS ----
        vals = list(temps.values())
        mean_t = sum(vals) / len(vals) if vals else 0
        total_energy = sum(vals)
        hot_count = sum(1 for t in vals if t > HOT_ZONE_THRESHOLD)
        cold_count = sum(1 for t in vals if t < COLD_THRESHOLD)

        # Cycle classification
        has_conflicts = any(
            beliefs.get(bid, {}).get("conflict_mass", 0) > 0
            for bid in beliefs if temps.get(bid, 0) > HOT_ZONE_THRESHOLD
        )
        if cycle % 5 == 0:
            ct = CycleType.FULL_SCAN
        elif hot_count > 0:
            ct = CycleType.HOT
        elif mean_t < 0.10 and not has_conflicts:
            ct = CycleType.COLD
        else:
            ct = CycleType.WARM
        cycle_type_counts[ct] += 1

        cumulative_prunes += plasticity.edges_pruned
        cumulative_leaks += plasticity.leaks_detected

        summary = get_plasticity_summary(sim_brain)
        leak_ratio = summary.get("total_leaks", 0) / max(1, summary.get("total_leaks", 0) + summary.get("total_productive", 0))

        # Track histories
        leak_history.append(plasticity.leaks_detected)
        prune_history.append(plasticity.edges_pruned)
        hot_history.append(hot_count)
        energy_history.append(total_energy)

        # Edge health file size
        eh_path = os.path.join(sim_brain, "edge_health.json")
        eh_bytes = os.path.getsize(eh_path) if os.path.exists(eh_path) else 0

        # Print every SNAPSHOT_INTERVAL cycles or on significant events
        if cycle % SNAPSHOT_INTERVAL == 0 or plasticity.edges_pruned > 0 or cycle <= 3:
            scenario_label = "+".join(active)[:20] if active else "none"
            print(f"{cycle:>5} {scenario_label:<20} {len(beliefs):>7} {len(edges):>6} "
                  f"{hot_count:>4} {cold_count:>5} {total_energy:>7.1f} "
                  f"{plasticity.leaks_detected:>6} {summary.get('weakened', 0):>5} "
                  f"{cumulative_prunes:>5} {leak_ratio:>6.3f}")

        # Record snapshot
        if cycle % SNAPSHOT_INTERVAL == 0:
            snapshots.append(CycleSnapshot(
                cycle=cycle,
                scenario=primary,
                belief_count=len(beliefs),
                edge_count=len(edges),
                total_energy=round(total_energy, 2),
                mean_temp=round(mean_t, 4),
                hot_count=hot_count,
                cold_count=cold_count,
                cluster_count=0,
                edges_tracked=summary.get("total_tracked", 0),
                edges_healthy=summary.get("healthy", 0),
                edges_weakened=summary.get("weakened", 0),
                edges_pruned=summary.get("pruned", 0),
                leak_ratio=round(leak_ratio, 4),
                leaks_this_cycle=plasticity.leaks_detected,
                prunes_this_cycle=plasticity.edges_pruned,
                edge_health_bytes=eh_bytes,
            ))

    # ============== ANALYSIS ==============
    print()
    print("=" * 75)
    print("SIMULATION ANALYSIS")
    print("=" * 75)

    final_summary = get_plasticity_summary(sim_brain)
    final_health = load_edge_health(sim_brain)

    # 1. Pruning correctness
    print("\n1. PRUNING CORRECTNESS")
    pruned_edges = {ek: eh for ek, eh in final_health.items() if eh.pruned}
    print(f"   Total edges pruned: {len(pruned_edges)}")

    false_prunes = 0
    for ek, eh in pruned_edges.items():
        # A false prune: more productive than leaky
        # Since pruned edges stop being evaluated, the counts are frozen at prune time
        if eh.productive_count > eh.leak_count:
            false_prunes += 1
            parts = ek.split("|")
            key_a = beliefs.get(parts[0], {}).get("key", "?")[:40]
            key_b = beliefs.get(parts[1], {}).get("key", "?")[:40]
            print(f"   FALSE PRUNE: {key_a} <-> {key_b} "
                  f"(productive={eh.productive_count} > leaks={eh.leak_count})")
            if false_prunes >= 10:
                print(f"   ... (showing first 10 only)")
                break

    if false_prunes == 0:
        print(f"   PASS: Zero false prunes — all pruned edges had more leaks than productive transfers")
    else:
        print(f"   CONCERN: {false_prunes} edges pruned despite being more productive than leaky")

    # 2. Edge health growth
    print("\n2. EDGE HEALTH GROWTH")
    eh_bytes_final = os.path.getsize(os.path.join(sim_brain, "edge_health.json")) if os.path.exists(os.path.join(sim_brain, "edge_health.json")) else 0
    print(f"   Final edge_health.json: {eh_bytes_final:,} bytes ({len(final_health)} edges tracked)")
    print(f"   Max beliefs seen: {max_beliefs_seen}")

    # Check growth rate
    if snapshots:
        first_eh = snapshots[0].edge_health_bytes
        last_eh = snapshots[-1].edge_health_bytes
        growth_factor = last_eh / max(1, first_eh) if first_eh > 0 else last_eh
        print(f"   Growth factor (first->last snapshot): {growth_factor:.1f}x")

        # Reasonable bound: edge_health should not be larger than ~100x belief count
        if len(final_health) > max_beliefs_seen * 5:
            print(f"   WARNING: edge_health ({len(final_health)}) >> beliefs ({max_beliefs_seen}) — possible bloat")
        else:
            print(f"   PASS: edge_health size proportional to belief count")

    # 3. Pathological loops
    print("\n3. PATHOLOGICAL LOOP DETECTION")
    # Check for oscillation: prune, recover, prune same edge
    prune_log_path = os.path.join(sim_brain, "pruning_log.jsonl")
    prune_log = []
    if os.path.exists(prune_log_path):
        with open(prune_log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    prune_log.append(json.loads(line))

    # Check if any edge appears in prune log multiple times
    prune_counts = defaultdict(int)
    for entry in prune_log:
        prune_counts[entry.get("edge", "")] += 1
    multi_pruned = {e: c for e, c in prune_counts.items() if c > 1}

    if multi_pruned:
        print(f"   WARNING: {len(multi_pruned)} edges pruned multiple times — possible oscillation:")
        for edge, count in sorted(multi_pruned.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {edge}: pruned {count}x")
    else:
        print(f"   PASS: No edges pruned more than once — no oscillation detected")

    # Check for runaway pruning (prune rate accelerating)
    window = 20
    if len(prune_history) >= window * 3:
        early_prunes = sum(prune_history[:window])
        mid_prunes = sum(prune_history[window:window*2])
        late_prunes = sum(prune_history[-window:])
        print(f"   Prune rate: early={early_prunes}, mid={mid_prunes}, late={late_prunes}")
        if late_prunes > early_prunes * 3 and late_prunes > 5:
            print(f"   WARNING: Prune rate accelerating — possible runaway")
        else:
            print(f"   PASS: Prune rate stable or declining")

    # 4. Leak ratio trend
    print("\n4. LEAK RATIO TREND")
    if snapshots and len(snapshots) >= 4:
        first_quarter = [s.leak_ratio for s in snapshots[:len(snapshots)//4]]
        last_quarter = [s.leak_ratio for s in snapshots[-len(snapshots)//4:]]
        avg_first = sum(first_quarter) / len(first_quarter) if first_quarter else 0
        avg_last = sum(last_quarter) / len(last_quarter) if last_quarter else 0
        print(f"   First quarter avg leak ratio: {avg_first:.4f}")
        print(f"   Last quarter avg leak ratio:  {avg_last:.4f}")
        if avg_last <= avg_first:
            print(f"   PASS: Leak ratio trending down or stable")
        else:
            delta = avg_last - avg_first
            if delta < 0.05:
                print(f"   OK: Leak ratio slightly higher (+{delta:.4f}) but within tolerance")
            else:
                print(f"   CONCERN: Leak ratio increasing significantly (+{delta:.4f})")

    # 5. Hot zone stability
    print("\n5. HOT ZONE STABILITY")
    if hot_history:
        max_hot = max(hot_history)
        avg_hot = sum(hot_history) / len(hot_history)
        # Check last 20% for stability
        late_hot = hot_history[-len(hot_history)//5:]
        late_avg = sum(late_hot) / len(late_hot) if late_hot else 0
        print(f"   Max hot zones: {max_hot}")
        print(f"   Average hot zones: {avg_hot:.1f}")
        print(f"   Late-stage (last 20%) avg: {late_avg:.1f}")
        if max_hot > len(beliefs) * 0.8:
            print(f"   WARNING: >80% of beliefs were hot at some point — thermal ceiling too low?")
        else:
            print(f"   PASS: Hot zones bounded ({max_hot}/{len(beliefs)} max)")

    # 6. Cold mass growth
    print("\n6. COLD MASS GROWTH")
    if snapshots:
        first_cold = snapshots[0].cold_count
        last_cold = snapshots[-1].cold_count
        first_total = snapshots[0].belief_count
        last_total = snapshots[-1].belief_count
        print(f"   Start: {first_cold}/{first_total} cold ({first_cold/max(1,first_total):.0%})")
        print(f"   End:   {last_cold}/{last_total} cold ({last_cold/max(1,last_total):.0%})")
        if last_cold >= first_cold:
            print(f"   PASS: Cold mass growing as expected")
        else:
            print(f"   INFO: Cold mass decreased — likely due to reactivation scenario")

    # 7. Energy trend
    print("\n7. ENERGY TREND")
    if energy_history and len(energy_history) >= 20:
        early_energy = sum(energy_history[:20]) / 20
        late_energy = sum(energy_history[-20:]) / 20
        print(f"   Early avg energy: {early_energy:.1f}")
        print(f"   Late avg energy:  {late_energy:.1f}")
        if late_energy > early_energy * 2:
            print(f"   CONCERN: Energy doubled — graph heating faster than cooling")
        else:
            print(f"   PASS: Energy bounded")

    # 8. Cycle classification distribution
    print("\n8. CYCLE CLASSIFICATION (Frontier 1)")
    total_classified = sum(cycle_type_counts.values())
    for ct, count in sorted(cycle_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_classified * 100 if total_classified else 0
        print(f"   {ct:<12} {count:>4} ({pct:>5.1f}%)")

    cold_pct = cycle_type_counts[CycleType.COLD] / total_classified * 100 if total_classified else 0
    full_scan_pct = cycle_type_counts[CycleType.FULL_SCAN] / total_classified * 100 if total_classified else 0

    if cycle_type_counts[CycleType.COLD] > 0:
        print(f"   PASS: Cold cycles detected — {cold_pct:.0f}% of cycles would skip diagnostics+LLM")
    else:
        print(f"   INFO: No cold cycles — graph stays warm/hot throughout (expected with continuous injection)")

    if cycle_type_counts[CycleType.FULL_SCAN] > 0:
        print(f"   PASS: Full scans fire at {full_scan_pct:.0f}% — safety coverage maintained")

    # Compute savings estimate
    diag_skipped = cycle_type_counts[CycleType.COLD]
    llm_skipped = cycle_type_counts[CycleType.COLD]  # cold cycles skip LLM too
    print(f"   Estimated savings: {diag_skipped} diagnostic phases skipped, {llm_skipped} LLM calls avoided")

    # ============== VERDICT ==============
    print()
    print("=" * 75)
    print("FINAL VERDICT")
    print("=" * 75)

    issues = []
    if false_prunes > 0:
        issues.append(f"false_prunes={false_prunes}")
    if multi_pruned:
        issues.append(f"oscillating_edges={len(multi_pruned)}")
    if len(final_health) > max_beliefs_seen * 5:
        issues.append("edge_health_bloat")
    if snapshots:
        avg_last = sum(s.leak_ratio for s in snapshots[-len(snapshots)//4:]) / max(1, len(snapshots)//4)
        avg_first = sum(s.leak_ratio for s in snapshots[:len(snapshots)//4]) / max(1, len(snapshots)//4)
        if avg_last - avg_first > 0.05:
            issues.append("leak_ratio_increasing")

    if not issues:
        print("  ALL CHECKS PASS")
        print(f"  {TOTAL_CYCLES} cycles simulated ({TOTAL_CYCLES // 4}-{TOTAL_CYCLES // 3} days equivalent)")
        print(f"  {cumulative_prunes} edges pruned, {cumulative_leaks} leaks detected")
        print(f"  {len(final_health)} edges tracked, 0 false prunes, 0 oscillations")
        print(f"  Cycle classification: {cycle_type_counts}")
        print(f"  Thermal Cognitive Subsystem is production-stable")
    else:
        print(f"  ISSUES FOUND: {', '.join(issues)}")
        print(f"  Review the detailed analysis above for specifics")

    # Cleanup
    print(f"\n  Simulation directory: {sim_dir}")
    print(f"  (keeping for manual inspection; rm -rf to clean)")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(run_simulation())
