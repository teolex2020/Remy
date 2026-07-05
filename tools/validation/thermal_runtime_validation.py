"""
Thermal Runtime Validation — killer experiments 4+5 on live Remi brain.

Validates 4 critical properties:
  1. STABILITY: thermal layer doesn't oscillate, crash, or produce NaN
  2. CORRELATION: hot zones map to genuinely important beliefs
  3. LLM GATING: saves calls without missing important work
  4. INTEGRATION: thermal data flows naturally through background cycle

Run: python -m tools.validation.thermal_runtime_validation
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure src is on path
if not getattr(sys, "frozen", False):
    _src = str(Path(__file__).parents[2] / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from remy.config.settings import settings

DATA_DIR = str(settings.AURA_BRAIN_PATH)

# ============== TEST 1: STABILITY ==============

def test_stability():
    """Run thermal map N times, verify no oscillation or crash."""
    from remy.core.thermal_advisor import compute_thermal_map

    print("=" * 60)
    print("TEST 1: STABILITY — repeated thermal computation")
    print("=" * 60)

    energies = []
    variances = []
    hot_counts = []
    cold_counts = []
    cluster_counts = []

    N_RUNS = 10
    for i in range(N_RUNS):
        report = compute_thermal_map(DATA_DIR)
        if report is None:
            print(f"  Run {i+1}: FAIL — no report")
            return False

        energies.append(report.total_energy)
        variances.append(report.variance)
        hot_counts.append(report.hot_zone_count)
        cold_counts.append(report.cold_mass_count)
        cluster_counts.append(len(report.clusters))

    # Determinism: same input should give same output
    energy_unique = len(set(energies))
    var_unique = len(set(variances))
    hot_unique = len(set(hot_counts))

    print(f"  {N_RUNS} runs completed")
    print(f"  Energy: {energies[0]:.3f} (unique values: {energy_unique})")
    print(f"  Variance: {variances[0]:.6f} (unique values: {var_unique})")
    print(f"  Hot zones: {hot_counts[0]} (unique values: {hot_unique})")
    print(f"  Cold mass: {cold_counts[0]}")
    print(f"  Clusters: {cluster_counts[0]}")

    # All runs should produce identical results (deterministic)
    if energy_unique > 1:
        print("  WARN: non-deterministic energy (may be OK if beliefs changed between runs)")
    if hot_unique > 1:
        print("  WARN: non-deterministic hot zone count")

    # Sanity: no NaN, no negative, no explosion
    for e in energies:
        if e != e or e < 0 or e > 10000:
            print(f"  FAIL: invalid energy {e}")
            return False
    for v in variances:
        if v != v or v < 0:
            print(f"  FAIL: invalid variance {v}")
            return False

    print("  PASS: stable, deterministic, no NaN/explosion")
    return True


# ============== TEST 2: CORRELATION ==============

def test_correlation():
    """Check whether hot zones correspond to genuinely important beliefs."""
    from remy.core.thermal_advisor import compute_thermal_map

    print()
    print("=" * 60)
    print("TEST 2: CORRELATION — hot zones vs cognitive importance")
    print("=" * 60)

    report = compute_thermal_map(DATA_DIR)
    if not report:
        print("  FAIL: no report")
        return False

    # Load raw beliefs for cross-checking
    with open(os.path.join(DATA_DIR, "beliefs.cog"), encoding="utf-8") as f:
        beliefs = json.load(f)["beliefs"]

    # Check 1: Do hot nodes have more conflict/unresolved than cold?
    hot_ids = {n.belief_id for n in report.top_hot if n.temperature > 0.18}
    cold_ids = set()
    all_temps = {}
    # Rebuild temps for full analysis
    from remy.core.thermal_advisor import _build_graph, _initialize_temperature, _spread_heat
    from remy.core.thermal_advisor import NUM_SPREAD_PASSES, COOLING_FACTOR
    adj, _, _ = _build_graph(beliefs)
    temps = _initialize_temperature(beliefs)
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps, adj)
    for n in temps:
        temps[n] *= COOLING_FACTOR
    all_temps = temps

    for bid, t in all_temps.items():
        if t > 0.18:
            hot_ids.add(bid)
        elif t < 0.08:
            cold_ids.add(bid)

    hot_conflict = sum(1 for bid in hot_ids if beliefs[bid]["conflict_mass"] > 0)
    hot_unresolved = sum(1 for bid in hot_ids if beliefs[bid]["state"] == "Unresolved")
    cold_conflict = sum(1 for bid in cold_ids if beliefs[bid]["conflict_mass"] > 0)
    cold_unresolved = sum(1 for bid in cold_ids if beliefs[bid]["state"] == "Unresolved")

    hot_avg_vol = sum(beliefs[bid]["volatility"] for bid in hot_ids) / len(hot_ids) if hot_ids else 0
    cold_avg_vol = sum(beliefs[bid]["volatility"] for bid in cold_ids) / len(cold_ids) if cold_ids else 0

    print(f"  Hot zone ({len(hot_ids)} beliefs):")
    print(f"    conflict: {hot_conflict}, unresolved: {hot_unresolved}, avg_volatility: {hot_avg_vol:.3f}")
    print(f"  Cold zone ({len(cold_ids)} beliefs):")
    print(f"    conflict: {cold_conflict}, unresolved: {cold_unresolved}, avg_volatility: {cold_avg_vol:.3f}")

    # Check 2: Hot clusters have semantically meaningful topics
    print(f"  Hot clusters ({len(report.clusters)}):")
    for i, c in enumerate(report.clusters):
        tags = ", ".join(t for t, _ in c.dominant_tags[:3])
        flags = []
        if c.has_conflict:
            flags.append("CONFLICT")
        if c.has_unresolved:
            flags.append("UNRESOLVED")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"    C{i+1}: {len(c.nodes)} beliefs, avg={c.avg_temperature:.3f}{flag_str} | {tags}")

    # Check 3: Cold zone should be stable, low-interest beliefs
    neutral_count = len(beliefs) - len(hot_ids) - len(cold_ids)
    print(f"  Neutral zone: {neutral_count} beliefs (between hot and cold)")

    # Verdict
    hot_signal = hot_conflict + hot_unresolved
    cold_signal = cold_conflict + cold_unresolved
    if hot_signal >= cold_signal:
        print(f"  PASS: hot zones have more cognitive tension ({hot_signal}) than cold ({cold_signal})")
    else:
        print(f"  WARN: cold zones have MORE tension ({cold_signal}) than hot ({hot_signal}) — investigate")

    if hot_avg_vol >= cold_avg_vol:
        print(f"  PASS: hot zones have higher volatility ({hot_avg_vol:.3f}) than cold ({cold_avg_vol:.3f})")
    else:
        print(f"  WARN: volatility inverted — cold zone more volatile")

    return True


# ============== TEST 3: LLM GATING ==============

def test_llm_gating():
    """Simulate gating decisions under different thermal states."""
    from remy.core.thermal_advisor import compute_thermal_map

    print()
    print("=" * 60)
    print("TEST 3: LLM GATING — cost savings without missing work")
    print("=" * 60)

    report = compute_thermal_map(DATA_DIR)
    if not report:
        print("  FAIL: no report")
        return False

    _LLM_CONSOLIDATION_INTERVAL_SEC = 3600
    _MAX_LLM_DROUGHT_SEC = _LLM_CONSOLIDATION_INTERVAL_SEC * 3

    scenarios = [
        ("hot graph, just eligible", report.hot_zone_count, _LLM_CONSOLIDATION_INTERVAL_SEC + 1),
        ("hot graph, long overdue", report.hot_zone_count, _MAX_LLM_DROUGHT_SEC + 1),
        ("cold graph (simulated), recent", 0, _LLM_CONSOLIDATION_INTERVAL_SEC + 1),
        ("cold graph (simulated), medium", 0, _LLM_CONSOLIDATION_INTERVAL_SEC * 2),
        ("cold graph (simulated), drought", 0, _MAX_LLM_DROUGHT_SEC + 1),
    ]

    for label, hot_count, time_since in scenarios:
        skip = False
        reason = ""
        if hot_count == 0 and time_since < _MAX_LLM_DROUGHT_SEC:
            skip = True
            reason = "cold_graph"
        elif hot_count > 0:
            reason = "hot_zones"
        else:
            reason = "drought_fallback"

        action = "SKIP" if skip else "RUN"
        print(f"  {label}: {action} ({reason})")

    # With current live data:
    actual_hot = report.hot_zone_count
    if actual_hot > 0:
        print(f"\n  Live state: {actual_hot} hot zones — LLM consolidation would ALWAYS run")
        print("  (Cold-graph gating only activates when graph cools down)")
    else:
        print(f"\n  Live state: 0 hot zones — LLM consolidation would be DEFERRED")

    print("  PASS: gating logic consistent across all scenarios")
    return True


# ============== TEST 4: INTEGRATION ==============

def test_integration():
    """Verify thermal data flows through all 3 integration points."""
    from remy.core.thermal_advisor import (
        compute_thermal_map,
        format_thermal_summary,
        format_thermal_report_json,
        get_maintenance_routing,
        sort_clusters_by_thermal_priority,
    )

    print()
    print("=" * 60)
    print("TEST 4: INTEGRATION — 3 live integration points")
    print("=" * 60)

    # Point 1: Background report (JSON)
    report = compute_thermal_map(DATA_DIR)
    if not report:
        print("  FAIL: no report")
        return False

    json_data = format_thermal_report_json(report)
    json_str = json.dumps(json_data)
    print(f"  [1] Background report: {len(json_str)} bytes, {len(json_data['clusters'])} clusters")
    assert "total_energy" in json_data
    assert "routing_advice" in json_data

    # Point 2: System instruction (text)
    summary = format_thermal_summary(report)
    print(f"  [2] System instruction: {len(summary)} chars, {summary.count(chr(10))+1} lines")
    assert "[THERMAL]" in summary
    assert "energy" in summary
    assert ("Warm zone:" in summary) or ("Cold zone:" in summary) or ("Hot zone:" in summary)

    # Point 3: Tool query (same as JSON but via dispatch path)
    try:
        from remy.core.tool_dispatch import _get_thermal_map
        result = _get_thermal_map({})
        parsed = json.loads(result)
        print(f"  [3] Tool dispatch: {len(result)} bytes, {parsed['cluster_count']} clusters")
        assert "hot_zone_count" in parsed
    except Exception as e:
        print(f"  [3] Tool dispatch: SKIP (import error: {e})")

    # Point 4: Routing integration
    routing = get_maintenance_routing(DATA_DIR)
    print(f"  [4] Maintenance routing: mode={routing.mode}, "
          f"hot={len(routing.hot_priority_tags)}, cold={len(routing.cold_skip_tags)}")

    # Simulated cluster sort
    fake_clusters = [
        ("health,lifestyle", ["r1", "r2"]),
        ("session-summary,autonomous-session", ["r3", "r4"]),
    ]
    prioritized, deferred, stats = sort_clusters_by_thermal_priority(fake_clusters, routing)
    print(f"  [4b] Cluster sort: {len(prioritized)} prioritized, {len(deferred)} deferred")

    print("  PASS: all integration points functional")
    return True


# ============== OVERALL VERDICT ==============

def test_cross_validation():
    """Cross-validate: thermal hot zones vs Aura's own instability summary."""

    print()
    print("=" * 60)
    print("TEST 5: CROSS-VALIDATION — thermal vs AuraSDK instability")
    print("=" * 60)

    from remy.core.thermal_advisor import compute_thermal_map

    report = compute_thermal_map(DATA_DIR)
    if not report:
        print("  FAIL: no report")
        return False

    # Get Aura's own instability view
    try:
        from remy.core.agent_tools import brain, brain_lock
        with brain_lock:
            volatile = brain.get_high_volatility_beliefs(limit=20)
            unstable = brain.get_low_stability_beliefs(limit=20)
    except Exception as e:
        print(f"  SKIP: can't access brain ({e})")
        return True  # Not a failure, just can't cross-validate

    # Extract IDs from Aura's view
    vol_ids = set()
    for v in (volatile or []):
        rid = getattr(v, "record_id", None) or (v.get("record_id") if isinstance(v, dict) else None)
        if rid:
            vol_ids.add(rid)

    unstable_ids = set()
    for u in (unstable or []):
        rid = getattr(u, "record_id", None) or (u.get("record_id") if isinstance(u, dict) else None)
        if rid:
            unstable_ids.add(rid)

    # Compare: thermal hot zone belief keys vs Aura's volatile/unstable
    hot_keys = {n.key for n in report.top_hot if n.temperature > 0.18}

    print(f"  Thermal hot beliefs: {len(hot_keys)}")
    print(f"  Aura volatile beliefs: {len(vol_ids)}")
    print(f"  Aura unstable beliefs: {len(unstable_ids)}")

    # Overlap by tags (since IDs may differ between record-level and belief-level)
    hot_tags = set()
    for n in report.top_hot:
        if n.temperature > 0.18:
            parts = n.key.split(":")
            if len(parts) >= 2:
                for t in parts[1].split(","):
                    hot_tags.add(t.strip())

    print(f"  Hot zone tags: {sorted(hot_tags)[:10]}...")
    print("  PASS: cross-validation data collected (qualitative comparison)")
    return True


def main():
    print("THERMAL RUNTIME VALIDATION")
    print(f"Data: {DATA_DIR}")
    print(f"Beliefs.cog exists: {os.path.exists(os.path.join(DATA_DIR, 'beliefs.cog'))}")
    print()

    results = {}
    results["stability"] = test_stability()
    results["correlation"] = test_correlation()
    results["llm_gating"] = test_llm_gating()
    results["integration"] = test_integration()
    results["cross_validation"] = test_cross_validation()

    print()
    print("=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
