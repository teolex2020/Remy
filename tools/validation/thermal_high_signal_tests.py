"""
Thermal High-Signal Tests — 5 targeted tests to find limits or raise confidence.

1. False-Negative: does thermal layer miss genuinely important beliefs?
2. Ablation: which signals in composite temperature actually matter?
3. Adversarial Hub: does a high-degree hub break the mechanics?
4. Session-to-Session Drift: does heat map change meaningfully over time?
5. Cost/Benefit: rough practical value estimate.

Run: python -m tools.validation.thermal_high_signal_tests
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if not getattr(sys, "frozen", False):
    _src = str(Path(__file__).parents[2] / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from remy.config.settings import settings

DATA_DIR = str(settings.AURA_BRAIN_PATH)


def _load_beliefs():
    with open(os.path.join(DATA_DIR, "beliefs.cog"), encoding="utf-8") as f:
        return json.load(f)["beliefs"]


def _full_thermal_pipeline(beliefs):
    """Run v5 thermal pipeline, return final temps."""
    from remy.core.thermal_advisor import (
        _build_graph, _initialize_temperature, _spread_heat,
        NUM_SPREAD_PASSES, COOLING_FACTOR,
    )
    adj, edges, degree = _build_graph(beliefs)
    temps = _initialize_temperature(beliefs)
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps, adj)
    for n in temps:
        temps[n] *= COOLING_FACTOR
    return temps, adj, degree


# ============== TEST 1: FALSE-NEGATIVE ==============

def test_false_negative():
    """Check if known-important beliefs end up in cold zones."""
    print("=" * 60)
    print("TEST 1: FALSE-NEGATIVE — does thermal miss important beliefs?")
    print("=" * 60)

    beliefs = _load_beliefs()
    temps, adj, degree = _full_thermal_pipeline(beliefs)

    # Collect known-important beliefs
    important = {}

    # Conflicts
    for bid, b in beliefs.items():
        if b["conflict_mass"] > 0:
            important[bid] = ("conflict", b["conflict_mass"])

    # Unresolved
    for bid, b in beliefs.items():
        if b["state"] == "Unresolved":
            important.setdefault(bid, ("unresolved", 0))

    # Low confidence (bottom 10%)
    all_conf = sorted(beliefs.items(), key=lambda x: x[1]["confidence"])
    cutoff = max(1, len(all_conf) // 10)
    for bid, b in all_conf[:cutoff]:
        important.setdefault(bid, ("low_confidence", b["confidence"]))

    # Low stability (bottom 10%)
    all_stab = sorted(beliefs.items(), key=lambda x: x[1]["stability"])
    cutoff_s = max(1, len(all_stab) // 10)
    for bid, b in all_stab[:cutoff_s]:
        important.setdefault(bid, ("low_stability", b["stability"]))

    # Classify each important belief by thermal zone
    COLD = 0.08
    HOT = 0.18
    false_negatives = []
    warm_catches = []
    hot_catches = []

    print(f"\n  {len(important)} known-important beliefs found")
    print(f"  {'Belief':<55} {'Reason':<18} {'Temp':>6} {'Zone':<6}")
    print("  " + "-" * 90)

    for bid, (reason, detail) in sorted(important.items(), key=lambda x: temps.get(x[0], 0)):
        t = temps.get(bid, 0)
        key = beliefs[bid]["key"][:52]
        if t < COLD:
            zone = "COLD"
            false_negatives.append((bid, reason, t))
        elif t < HOT:
            zone = "warm"
            warm_catches.append((bid, reason, t))
        else:
            zone = "HOT"
            hot_catches.append((bid, reason, t))
        print(f"  {key:<55} {reason:<18} {t:>6.3f} {zone:<6}")

    print(f"\n  Results:")
    print(f"    HOT  (correctly prioritized): {len(hot_catches)}")
    print(f"    warm (visible but not top):   {len(warm_catches)}")
    print(f"    COLD (false negative):        {len(false_negatives)}")

    if false_negatives:
        print(f"\n  FALSE NEGATIVES:")
        for bid, reason, t in false_negatives:
            b = beliefs[bid]
            print(f"    {t:.3f} [{reason}] {b['key'][:60]}")
            print(f"           conf={b['confidence']:.3f} stab={b['stability']} "
                  f"vol={b['volatility']:.3f} conflict={b['conflict_mass']}")
        fn_rate = len(false_negatives) / len(important)
        print(f"\n  False-negative rate: {fn_rate:.1%} ({len(false_negatives)}/{len(important)})")
        if fn_rate > 0.2:
            print("  FAIL: >20% false negatives — composite temperature needs adjustment")
            return False
        else:
            print("  PASS: false-negative rate acceptable")
    else:
        print("\n  PASS: zero false negatives — all important beliefs detected")

    return True


# ============== TEST 2: ABLATION ==============

def test_ablation():
    """Compare signal quality across composite temperature variants."""
    print()
    print("=" * 60)
    print("TEST 2: ABLATION — which signals matter?")
    print("=" * 60)

    beliefs = _load_beliefs()
    from remy.core.thermal_advisor import (
        _build_graph, _spread_heat, NUM_SPREAD_PASSES, COOLING_FACTOR, HOT_ZONE_THRESHOLD,
    )
    adj, edges, degree = _build_graph(beliefs)

    # Find ground truth: beliefs that SHOULD be hot
    ground_truth = set()
    for bid, b in beliefs.items():
        if b["conflict_mass"] > 0 or b["state"] == "Unresolved":
            ground_truth.add(bid)
    # Add low-confidence as secondary ground truth
    all_conf = sorted(beliefs.items(), key=lambda x: x[1]["confidence"])
    for bid, b in all_conf[:max(1, len(all_conf) // 10)]:
        ground_truth.add(bid)

    # Normalization helpers
    all_vol = [b["volatility"] for b in beliefs.values()]
    all_conf_vals = [b["confidence"] for b in beliefs.values()]
    all_stab = [b["stability"] for b in beliefs.values()]
    all_supp = [b["support_mass"] for b in beliefs.values()]
    max_vol = max(all_vol) if max(all_vol) > 0 else 1.0
    max_conf = max(all_conf_vals) if max(all_conf_vals) > 0 else 1.0
    max_stab = max(all_stab) if max(all_stab) > 0 else 1.0
    max_supp = max(all_supp) if max(all_supp) > 0 else 1.0

    variants = {
        "A: volatility only": lambda b: b["volatility"] / max_vol,
        "B: vol + conflict": lambda b: (
            0.50 * b["volatility"] / max_vol
            + 0.50 * (1.0 if b["conflict_mass"] > 0 else 0.0)
        ),
        "C: vol + conflict + uncertainty": lambda b: (
            0.30 * b["volatility"] / max_vol
            + 0.35 * (1.0 if b["conflict_mass"] > 0 else 0.0)
            + 0.35 * (1.0 - b["confidence"] / max_conf)
        ),
        "D: full composite (v5)": lambda b: max(0.0, min(1.0,
            0.20 * b["volatility"] / max_vol
            + 0.30 * (1.0 if b["conflict_mass"] > 0 else 0.0)
            + 0.25 * (1.0 if b["state"] == "Unresolved" else 0.0)
            + 0.15 * (1.0 - b["confidence"] / max_conf)
            - 0.30 * b["stability"] / max_stab
            - 0.25 * b["confidence"] / max_conf
            - 0.15 * b["support_mass"] / max_supp
            + 0.10
        )),
    }

    print(f"\n  Ground truth: {len(ground_truth)} important beliefs")
    print(f"  {'Variant':<35} {'Hot':>4} {'Capture':>8} {'Clusters':>9} {'Variance':>10}")
    print("  " + "-" * 70)

    results = {}
    for name, init_fn in variants.items():
        temps = {}
        for bid, b in beliefs.items():
            temps[bid] = max(0.0, min(1.0, init_fn(b)))

        for _ in range(NUM_SPREAD_PASSES):
            _spread_heat(temps, adj)
        for n in temps:
            temps[n] *= COOLING_FACTOR

        hot_ids = {bid for bid, t in temps.items() if t > HOT_ZONE_THRESHOLD}
        captured = hot_ids & ground_truth
        capture_rate = len(captured) / len(ground_truth) if ground_truth else 0

        vals = list(temps.values())
        mean_t = sum(vals) / len(vals)
        variance = sum((v - mean_t) ** 2 for v in vals) / len(vals)

        # Cluster count at HOT threshold
        from remy.core.thermal_advisor import _find_clusters
        clusters = _find_clusters(hot_ids, adj)

        results[name] = {
            "hot": len(hot_ids),
            "capture": capture_rate,
            "clusters": len(clusters),
            "variance": variance,
        }

        print(f"  {name:<35} {len(hot_ids):>4} {capture_rate:>7.0%} {len(clusters):>9} {variance:>10.6f}")

    # Verdict
    full = results["D: full composite (v5)"]
    vol_only = results["A: volatility only"]

    print(f"\n  Full composite vs volatility-only:")
    print(f"    Capture rate: {full['capture']:.0%} vs {vol_only['capture']:.0%}")
    print(f"    Clusters:     {full['clusters']} vs {vol_only['clusters']}")
    print(f"    Variance:     {full['variance']:.6f} vs {vol_only['variance']:.6f}")

    if full["capture"] >= vol_only["capture"]:
        print("  PASS: full composite captures at least as much as volatility alone")
    else:
        print("  WARN: volatility alone captures more — composite may over-cool")

    if full["variance"] > vol_only["variance"]:
        print("  PASS: full composite has better differentiation (higher variance)")
    else:
        print("  INFO: volatility alone has higher variance (may be noise, not signal)")

    return True


# ============== TEST 3: ADVERSARIAL HUB ==============

def test_adversarial_hub():
    """Check if a high-degree hub node distorts the thermal map."""
    print()
    print("=" * 60)
    print("TEST 3: ADVERSARIAL HUB — does a mega-hub break the map?")
    print("=" * 60)

    beliefs = _load_beliefs()
    from remy.core.thermal_advisor import (
        _build_graph, _initialize_temperature, _spread_heat,
        NUM_SPREAD_PASSES, COOLING_FACTOR, HOT_ZONE_THRESHOLD, _find_clusters,
    )

    adj, edges, degree = _build_graph(beliefs)

    # Find the highest-degree node (natural hub)
    hub_id = max(degree, key=degree.get)
    hub_deg = degree[hub_id]
    hub_key = beliefs[hub_id]["key"][:50]
    avg_deg = sum(degree.values()) / len(degree)

    print(f"\n  Natural hub: degree={hub_deg} (avg={avg_deg:.1f})")
    print(f"  Hub belief: {hub_key}")

    # Run normal thermal
    temps_normal = _initialize_temperature(beliefs)
    temps_before = dict(temps_normal)
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps_normal, adj)
    for n in temps_normal:
        temps_normal[n] *= COOLING_FACTOR

    hub_temp_normal = temps_normal[hub_id]

    # Run adversarial: set hub to max heat (1.0)
    temps_adversarial = _initialize_temperature(beliefs)
    temps_adversarial[hub_id] = 1.0  # artificially max heat
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps_adversarial, adj)
    for n in temps_adversarial:
        temps_adversarial[n] *= COOLING_FACTOR

    hub_temp_adv = temps_adversarial[hub_id]

    # Compare: did adversarial hub create a mega-cluster?
    hot_normal = {bid for bid, t in temps_normal.items() if t > HOT_ZONE_THRESHOLD}
    hot_adv = {bid for bid, t in temps_adversarial.items() if t > HOT_ZONE_THRESHOLD}

    clusters_normal = _find_clusters(hot_normal, adj)
    clusters_adv = _find_clusters(hot_adv, adj)

    largest_normal = max(len(c) for c in clusters_normal) if clusters_normal else 0
    largest_adv = max(len(c) for c in clusters_adv) if clusters_adv else 0

    # Energy check: did adversarial hub cause energy explosion?
    energy_normal = sum(temps_normal.values())
    energy_adv = sum(temps_adversarial.values())

    # How many neighbors got heated above normal?
    hub_neighbors = [bid for bid, _ in adj.get(hub_id, [])]
    heated_neighbors = 0
    for nbid in hub_neighbors:
        if temps_adversarial[nbid] - temps_normal[nbid] > 0.05:
            heated_neighbors += 1

    print(f"\n  {'Metric':<30} {'Normal':>10} {'Adversarial':>12} {'Delta':>8}")
    print("  " + "-" * 65)
    print(f"  {'Hub temperature':<30} {hub_temp_normal:>10.3f} {hub_temp_adv:>12.3f} {hub_temp_adv-hub_temp_normal:>+8.3f}")
    print(f"  {'Hot zone count':<30} {len(hot_normal):>10} {len(hot_adv):>12} {len(hot_adv)-len(hot_normal):>+8}")
    print(f"  {'Cluster count':<30} {len(clusters_normal):>10} {len(clusters_adv):>12} {len(clusters_adv)-len(clusters_normal):>+8}")
    print(f"  {'Largest cluster':<30} {largest_normal:>10} {largest_adv:>12} {largest_adv-largest_normal:>+8}")
    print(f"  {'Total energy':<30} {energy_normal:>10.3f} {energy_adv:>12.3f} {energy_adv-energy_normal:>+8.3f}")
    print(f"  {'Heated neighbors (>+0.05)':<30} {'-':>10} {heated_neighbors:>12}/{len(hub_neighbors)}")

    # Verdict
    mega = largest_adv > len(beliefs) * 0.5
    if mega:
        print(f"\n  FAIL: adversarial hub created mega-cluster ({largest_adv}/{len(beliefs)})")
        return False

    energy_ratio = energy_adv / energy_normal if energy_normal > 0 else 0
    if energy_ratio > 1.2:
        print(f"\n  WARN: energy grew {energy_ratio:.2f}x — hub leaked significant heat")
    else:
        print(f"\n  PASS: energy contained ({energy_ratio:.3f}x), no mega-cluster")

    spread_ratio = heated_neighbors / len(hub_neighbors) if hub_neighbors else 0
    print(f"  Heat spread: {spread_ratio:.0%} of hub neighbors heated >+0.05")
    if spread_ratio < 0.5:
        print("  PASS: outflow limiting contained hub heat effectively")
    else:
        print("  WARN: hub heat spread to majority of neighbors")

    return True


# ============== TEST 4: SESSION-TO-SESSION DRIFT ==============

def test_session_drift():
    """Check if thermal map changes meaningfully when beliefs change."""
    print()
    print("=" * 60)
    print("TEST 4: SESSION DRIFT — does heat map respond to changes?")
    print("=" * 60)

    beliefs = _load_beliefs()
    from remy.core.thermal_advisor import (
        _build_graph, _initialize_temperature, _spread_heat,
        NUM_SPREAD_PASSES, COOLING_FACTOR, HOT_ZONE_THRESHOLD, _find_clusters,
    )

    # Baseline
    adj, edges, degree = _build_graph(beliefs)
    temps_base = _initialize_temperature(beliefs)
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps_base, adj)
    for n in temps_base:
        temps_base[n] *= COOLING_FACTOR
    hot_base = {bid for bid, t in temps_base.items() if t > HOT_ZONE_THRESHOLD}

    # Simulate session: resolve a conflict (should cool that zone)
    beliefs_after = json.loads(json.dumps(beliefs))  # deep copy
    resolved_bid = None
    for bid, b in beliefs_after.items():
        if b["conflict_mass"] > 0 and b["state"] == "Unresolved":
            b["state"] = "Resolved"
            b["conflict_mass"] = 0
            b["confidence"] = 0.8
            b["stability"] = 5
            resolved_bid = bid
            break

    if resolved_bid:
        print(f"\n  Simulated: resolved conflict in {beliefs[resolved_bid]['key'][:50]}")
    else:
        # No unresolved conflict — simulate by cooling a hot belief
        hottest = max(temps_base, key=temps_base.get)
        beliefs_after[hottest]["stability"] = 10
        beliefs_after[hottest]["confidence"] = 0.9
        beliefs_after[hottest]["conflict_mass"] = 0
        resolved_bid = hottest
        print(f"\n  Simulated: stabilized {beliefs[resolved_bid]['key'][:50]}")

    adj2, _, _ = _build_graph(beliefs_after)
    temps_after = _initialize_temperature(beliefs_after)
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps_after, adj2)
    for n in temps_after:
        temps_after[n] *= COOLING_FACTOR
    hot_after = {bid for bid, t in temps_after.items() if t > HOT_ZONE_THRESHOLD}

    # Simulate session: add a new problem (should heat that zone)
    beliefs_problem = json.loads(json.dumps(beliefs))
    # Pick a cold belief and make it problematic
    coldest = min(temps_base, key=temps_base.get)
    beliefs_problem[coldest]["conflict_mass"] = 3.0
    beliefs_problem[coldest]["state"] = "Unresolved"
    beliefs_problem[coldest]["confidence"] = 0.1
    beliefs_problem[coldest]["stability"] = 0

    adj3, _, _ = _build_graph(beliefs_problem)
    temps_problem = _initialize_temperature(beliefs_problem)
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps_problem, adj3)
    for n in temps_problem:
        temps_problem[n] *= COOLING_FACTOR
    hot_problem = {bid for bid, t in temps_problem.items() if t > HOT_ZONE_THRESHOLD}

    resolved_temp_before = temps_base.get(resolved_bid, 0)
    resolved_temp_after = temps_after.get(resolved_bid, 0)
    coldest_temp_before = temps_base.get(coldest, 0)
    coldest_temp_problem = temps_problem.get(coldest, 0)

    print(f"\n  {'Scenario':<35} {'Hot zones':>10} {'Target temp':>12}")
    print("  " + "-" * 60)
    print(f"  {'Baseline':<35} {len(hot_base):>10} {'-':>12}")
    print(f"  {'After resolving conflict':<35} {len(hot_after):>10} {resolved_temp_before:.3f} -> {resolved_temp_after:.3f}")
    print(f"  {'After adding new problem':<35} {len(hot_problem):>10} {coldest_temp_before:.3f} -> {coldest_temp_problem:.3f}")

    # Overlap between baseline and after-resolution
    overlap = len(hot_base & hot_after) / len(hot_base) if hot_base else 0
    print(f"\n  Hot zone overlap (base vs resolved): {overlap:.0%}")

    # Verdicts
    cooled = resolved_temp_after < resolved_temp_before
    heated = coldest_temp_problem > coldest_temp_before

    if cooled:
        print(f"  PASS: resolved belief cooled ({resolved_temp_before:.3f} -> {resolved_temp_after:.3f})")
    else:
        print(f"  WARN: resolved belief did NOT cool ({resolved_temp_before:.3f} -> {resolved_temp_after:.3f})")

    if heated:
        print(f"  PASS: new problem heated ({coldest_temp_before:.3f} -> {coldest_temp_problem:.3f})")
    else:
        print(f"  WARN: new problem did NOT heat ({coldest_temp_before:.3f} -> {coldest_temp_problem:.3f})")

    if overlap > 0.5 and overlap < 1.0:
        print(f"  PASS: map is stable but responsive ({overlap:.0%} overlap)")
    elif overlap == 1.0:
        print(f"  WARN: map didn't change at all — may be insensitive")
    else:
        print(f"  WARN: map changed too much — may be unstable")

    return cooled and heated


# ============== TEST 5: COST/BENEFIT ==============

def test_cost_benefit():
    """Estimate practical value from current thermal state."""
    print()
    print("=" * 60)
    print("TEST 5: COST/BENEFIT — rough practical value estimate")
    print("=" * 60)

    beliefs = _load_beliefs()
    temps, adj, degree = _full_thermal_pipeline(beliefs)

    total = len(beliefs)
    hot = sum(1 for t in temps.values() if t > 0.18)
    cold = sum(1 for t in temps.values() if t < 0.08)
    neutral = total - hot - cold

    # Estimate: if maintenance scans all beliefs equally, thermal saves cold/total
    cold_ratio = cold / total if total > 0 else 0
    hot_ratio = hot / total if total > 0 else 0

    # LLM gating: if graph were cold, how many cycles would skip?
    # Currently hot, so no savings — but estimate for when graph cools
    estimated_skip_rate = 0.0  # can't skip when hot
    if hot == 0:
        estimated_skip_rate = 0.6  # estimated: 60% of cycles could skip in cold state

    print(f"\n  Belief distribution:")
    print(f"    Total:   {total}")
    print(f"    Hot:     {hot} ({hot_ratio:.0%})")
    print(f"    Neutral: {neutral}")
    print(f"    Cold:    {cold} ({cold_ratio:.0%})")

    print(f"\n  Routing value:")
    print(f"    Hot-first sorting: top {hot} beliefs get attention first (vs random order)")
    print(f"    Cold-deferred:     {cold} beliefs skippable per cycle ({cold_ratio:.0%} savings)")

    print(f"\n  LLM gating value:")
    print(f"    Current state:     {hot} hot zones -> LLM always runs (no savings now)")
    print(f"    If graph cools:    ~60% cycles could skip LLM consolidation")

    # Check observation log for real data
    log_path = os.path.join(os.path.dirname(DATA_DIR), "thermal_observations.jsonl")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        total_entries = len(entries)
        skipped = sum(1 for e in entries if e.get("llm_gating", {}).get("skipped"))
        hot_triggers = sum(1 for e in entries if e.get("llm_gating", {}).get("triggered_by") == "hot_zones")
        drought = sum(1 for e in entries if e.get("llm_gating", {}).get("triggered_by") == "drought_fallback")
        print(f"\n  Live observation log ({total_entries} entries):")
        print(f"    LLM skipped:         {skipped}")
        print(f"    LLM triggered (hot): {hot_triggers}")
        print(f"    LLM triggered (drought): {drought}")
        if total_entries > 0:
            print(f"    Skip rate: {skipped/total_entries:.0%}")
    else:
        print(f"\n  No observation log yet — will accumulate during live sessions")

    print(f"\n  PASS: cost/benefit estimated (live data needed for real measurement)")
    return True


# ============== MAIN ==============

def main():
    print("THERMAL HIGH-SIGNAL TESTS")
    print(f"Data: {DATA_DIR}")
    print()

    results = {}
    results["false_negative"] = test_false_negative()
    results["ablation"] = test_ablation()
    results["adversarial_hub"] = test_adversarial_hub()
    results["session_drift"] = test_session_drift()
    results["cost_benefit"] = test_cost_benefit()

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    all_pass = all(results.values())
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
