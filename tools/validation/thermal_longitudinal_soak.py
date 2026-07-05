"""
Thermal Longitudinal Soak Tests — 3 deep studies on real historical brain snapshots.

Uses real Remi brain backups spanning 2026-03-22 to 2026-04-05 (14 days).

1. Chronic Heat (Tinnitus): beliefs hot across all snapshots without resolution
2. Freezing Trajectory: old beliefs cooling naturally over time
3. Semantic Ripple (Butterfly Effect): heat propagation paths across the graph

Run: python -m tools.validation.thermal_longitudinal_soak
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

DATA_ROOT = str((Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parents[2]) / "data")

# Snapshots in chronological order
SNAPSHOTS = [
    ("2026-03-22", os.path.join(DATA_ROOT, "brain_incompatible_20260322_142029")),
    ("2026-03-23", os.path.join(DATA_ROOT, "brain_incompatible_20260323_144640")),
    ("2026-03-31a", os.path.join(DATA_ROOT, "brain_incompatible_20260331_090818")),
    ("2026-03-31b", os.path.join(DATA_ROOT, "brain_incompatible_20260331_223805")),
    ("2026-04-05", os.path.join(DATA_ROOT, "brain")),
]


def _load_beliefs(data_dir: str) -> dict:
    path = os.path.join(data_dir, "beliefs.cog")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("beliefs", {})


def _normalize_beliefs(beliefs: dict) -> dict:
    """Ensure all beliefs have required fields with defaults for older formats."""
    for b in beliefs.values():
        b.setdefault("volatility", 0.0)
        b.setdefault("support_mass", 0.0)
        b.setdefault("confidence", 0.5)
        b.setdefault("stability", 0)
        b.setdefault("conflict_mass", 0.0)
        b.setdefault("state", "Singleton")
    return beliefs


def _run_thermal(beliefs: dict) -> tuple[dict, dict, dict]:
    """Run full v5 thermal pipeline. Returns (temps, adj, degree)."""
    if not beliefs or len(beliefs) < 2:
        return {}, {}, {}
    beliefs = _normalize_beliefs(beliefs)
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


def _extract_tags(key: str) -> set:
    parts = key.split(":")
    if len(parts) >= 2:
        return {t.strip() for t in parts[1].split(",")}
    return set()


# ============== STUDY 1: CHRONIC HEAT ==============

def study_chronic_heat():
    """Find beliefs that stay hot across multiple snapshots."""
    print("=" * 70)
    print("STUDY 1: CHRONIC HEAT (Tinnitus)")
    print("  Do any beliefs stay hot across all snapshots without resolution?")
    print("=" * 70)

    # Run thermal on each snapshot
    snapshot_temps = []
    snapshot_beliefs = []
    valid_snapshots = []

    for label, path in SNAPSHOTS:
        beliefs = _load_beliefs(path)
        if len(beliefs) < 3:
            print(f"\n  [{label}] {len(beliefs)} beliefs — too small, skipping")
            continue
        temps, _, _ = _run_thermal(beliefs)
        snapshot_temps.append(temps)
        snapshot_beliefs.append(beliefs)
        valid_snapshots.append(label)
        hot = sum(1 for t in temps.values() if t > 0.18)
        print(f"\n  [{label}] {len(beliefs)} beliefs, {hot} hot (>0.18)")

    if len(valid_snapshots) < 2:
        print("\n  Not enough snapshots for longitudinal analysis")
        return True

    # Track beliefs by key across snapshots (IDs may change between snapshots)
    key_history: dict[str, list[tuple[str, float, str]]] = defaultdict(list)

    for i, (label, beliefs, temps) in enumerate(
        zip(valid_snapshots, snapshot_beliefs, snapshot_temps)
    ):
        for bid, b in beliefs.items():
            key = b["key"]
            t = temps.get(bid, 0)
            key_history[key].append((label, t, b["state"]))

    # Find chronic heat: keys that appear in 3+ snapshots and are hot in all
    print(f"\n  Tracking {len(key_history)} unique belief keys across {len(valid_snapshots)} snapshots")

    chronic = []
    persistent_warm = []

    for key, history in key_history.items():
        if len(history) < 2:
            continue
        temps_list = [t for _, t, _ in history]
        all_hot = all(t > 0.18 for t in temps_list)
        all_warm = all(t > 0.10 for t in temps_list)

        if all_hot and len(history) >= 2:
            chronic.append((key, history))
        elif all_warm and len(history) >= 3:
            persistent_warm.append((key, history))

    print(f"\n  Chronic hot (>0.18 in all appearances): {len(chronic)}")
    for key, history in chronic[:10]:
        trajectory = " -> ".join(f"{t:.3f}({lbl[:6]})" for lbl, t, _ in history)
        states = set(s for _, _, s in history)
        print(f"    {key[:55]}")
        print(f"      {trajectory}  states={states}")

    print(f"\n  Persistent warm (>0.10 in 3+ appearances): {len(persistent_warm)}")
    for key, history in persistent_warm[:5]:
        trajectory = " -> ".join(f"{t:.3f}" for _, t, _ in history)
        print(f"    {key[:55]}  [{trajectory}]")

    # Verdict
    print(f"\n  VERDICT:")
    if chronic:
        boiling = [k for k, h in chronic if any(t > 0.6 for _, t, _ in h)]
        if boiling:
            print(f"    WARNING: {len(boiling)} beliefs above 0.6 for sustained period")
            print(f"    These may need a burnout/scar mechanism")
        else:
            print(f"    {len(chronic)} chronically warm beliefs, but none boiling (>0.6)")
            print(f"    Current cooling is sufficient — no burnout mechanism needed yet")
    else:
        print(f"    No chronic heat found — graph cools and heats naturally")

    return True


# ============== STUDY 2: FREEZING TRAJECTORY ==============

def study_freezing_trajectory():
    """Track how old beliefs cool over time."""
    print()
    print("=" * 70)
    print("STUDY 2: FREEZING TRAJECTORY")
    print("  Do old unused beliefs cool toward zero? What's the cooling rate?")
    print("=" * 70)

    # Compare earliest meaningful snapshot to current
    early_beliefs = None
    early_label = None
    for label, path in SNAPSHOTS:
        beliefs = _load_beliefs(path)
        if len(beliefs) >= 10:
            early_beliefs = beliefs
            early_label = label
            break

    if early_beliefs is None:
        print("\n  No early snapshot with enough beliefs")
        return True

    current_beliefs = _load_beliefs(SNAPSHOTS[-1][1])
    current_label = SNAPSHOTS[-1][0]

    early_temps, _, _ = _run_thermal(early_beliefs)
    current_temps, _, _ = _run_thermal(current_beliefs)

    # Find beliefs that existed in early snapshot
    early_keys = {b["key"]: (bid, b) for bid, b in early_beliefs.items()}
    current_keys = {b["key"]: (bid, b) for bid, b in current_beliefs.items()}

    survived = []
    disappeared = []

    for key, (ebid, eb) in early_keys.items():
        early_t = early_temps.get(ebid, 0)
        if key in current_keys:
            cbid, cb = current_keys[key]
            current_t = current_temps.get(cbid, 0)
            survived.append((key, early_t, current_t, eb, cb))
        else:
            disappeared.append((key, early_t, eb))

    print(f"\n  [{early_label}] -> [{current_label}]")
    print(f"  Early beliefs: {len(early_beliefs)}")
    print(f"  Current beliefs: {len(current_beliefs)}")
    print(f"  Survived: {len(survived)}")
    print(f"  Disappeared: {len(disappeared)}")

    if survived:
        print(f"\n  Surviving beliefs — temperature trajectory:")
        print(f"  {'Key':<50} {'Early':>6} {'Now':>6} {'Delta':>7} {'Cooled?':>8}")
        print("  " + "-" * 80)

        cooled = 0
        heated = 0
        stable = 0
        for key, et, ct, eb, cb in sorted(survived, key=lambda x: x[2] - x[1]):
            delta = ct - et
            status = "COOLED" if delta < -0.02 else ("HEATED" if delta > 0.02 else "stable")
            if delta < -0.02:
                cooled += 1
            elif delta > 0.02:
                heated += 1
            else:
                stable += 1
            print(f"  {key[:50]:<50} {et:>6.3f} {ct:>6.3f} {delta:>+7.3f} {status:>8}")

        print(f"\n  Summary: {cooled} cooled, {heated} heated, {stable} stable")

    if disappeared:
        print(f"\n  Disappeared beliefs (existed in {early_label}, gone by {current_label}):")
        for key, et, eb in disappeared[:10]:
            print(f"    {key[:55]} (was temp={et:.3f}, state={eb['state']})")

    # Freezing analysis: what temperature do unused beliefs reach?
    if survived:
        now_temps = [ct for _, _, ct, _, _ in survived]
        print(f"\n  Surviving beliefs temperature distribution:")
        frozen = sum(1 for t in now_temps if t < 0.03)
        cold = sum(1 for t in now_temps if 0.03 <= t < 0.08)
        warm = sum(1 for t in now_temps if 0.08 <= t < 0.18)
        hot = sum(1 for t in now_temps if t >= 0.18)
        print(f"    Frozen (<0.03): {frozen}")
        print(f"    Cold (0.03-0.08): {cold}")
        print(f"    Warm (0.08-0.18): {warm}")
        print(f"    Hot (>0.18): {hot}")

    # Verdict
    print(f"\n  VERDICT:")
    if survived:
        avg_delta = sum(ct - et for _, et, ct, _, _ in survived) / len(survived)
        print(f"    Average temperature change: {avg_delta:+.3f}")
        if avg_delta < -0.02:
            print(f"    Graph is naturally cooling old beliefs — freezing trajectory works")
        elif avg_delta > 0.02:
            print(f"    Old beliefs got hotter — possible structural reheating from new connections")
        else:
            print(f"    Temperature roughly stable — slow cooling or reactivation balances out")
    else:
        print(f"    No surviving beliefs to compare")

    return True


# ============== STUDY 3: SEMANTIC RIPPLE ==============

def study_semantic_ripple():
    """Trace heat propagation paths through the graph."""
    print()
    print("=" * 70)
    print("STUDY 3: SEMANTIC RIPPLE (Butterfly Effect)")
    print("  How far does heat travel? Does it cross semantic boundaries?")
    print("=" * 70)

    beliefs = _load_beliefs(SNAPSHOTS[-1][1])
    from remy.core.thermal_advisor import (
        _build_graph, _initialize_temperature, _spread_heat,
        NUM_SPREAD_PASSES, COOLING_FACTOR,
    )

    adj, edges, degree = _build_graph(beliefs)

    # Find a hot source node (conflict or high initial temp)
    temps_init = _initialize_temperature(beliefs)
    hottest_bid = max(temps_init, key=temps_init.get)
    hottest_key = beliefs[hottest_bid]["key"]
    hottest_init = temps_init[hottest_bid]
    hottest_tags = _extract_tags(hottest_key)

    print(f"\n  Source node: {hottest_key[:60]}")
    print(f"  Initial temp: {hottest_init:.3f}")
    print(f"  Tags: {hottest_tags}")
    print(f"  Degree: {degree.get(hottest_bid, 0)}")

    # Track heat spread pass by pass
    temps = dict(temps_init)
    print(f"\n  Heat propagation from source node:")
    print(f"  {'Pass':<6} {'Source':>7} {'Neighbors avg':>14} {'2-hop avg':>10} {'Cross-domain':>13}")
    print("  " + "-" * 55)

    neighbors = {bid for bid, _ in adj.get(hottest_bid, [])}
    # 2-hop neighbors
    two_hop = set()
    for nbid in neighbors:
        for nb2, _ in adj.get(nbid, []):
            if nb2 != hottest_bid and nb2 not in neighbors:
                two_hop.add(nb2)

    # Cross-domain: beliefs with completely different tags
    cross_domain = set()
    for bid in beliefs:
        if bid == hottest_bid:
            continue
        bid_tags = _extract_tags(beliefs[bid]["key"])
        if bid_tags and hottest_tags and not (bid_tags & hottest_tags):
            cross_domain.add(bid)

    # Reset temps and track spreading
    temps = dict(temps_init)
    for p in range(NUM_SPREAD_PASSES):
        _spread_heat(temps, adj)

        src_t = temps[hottest_bid]
        nb_avg = sum(temps[n] for n in neighbors) / len(neighbors) if neighbors else 0
        th_avg = sum(temps[n] for n in two_hop) / len(two_hop) if two_hop else 0
        cd_avg = sum(temps[n] for n in cross_domain) / len(cross_domain) if cross_domain else 0

        print(f"  {p+1:<6} {src_t:>7.3f} {nb_avg:>14.3f} {th_avg:>10.3f} {cd_avg:>13.3f}")

    # Apply cooling
    for n in temps:
        temps[n] *= COOLING_FACTOR

    # Analyze: which tags received heat from source?
    print(f"\n  Heat received by tag group (after spreading):")

    tag_heat: dict[str, list[float]] = defaultdict(list)
    tag_init: dict[str, list[float]] = defaultdict(list)
    for bid, b in beliefs.items():
        for tag in _extract_tags(b["key"]):
            tag_heat[tag].append(temps[bid])
            tag_init[tag].append(temps_init[bid])

    print(f"  {'Tag':<25} {'Init avg':>9} {'Final avg':>10} {'Delta':>7} {'Same domain?':>13}")
    print("  " + "-" * 68)

    tag_results = []
    for tag in sorted(tag_heat.keys()):
        init_avg = sum(tag_init[tag]) / len(tag_init[tag])
        final_avg = sum(tag_heat[tag]) / len(tag_heat[tag])
        delta = final_avg - init_avg
        same = "YES" if tag in hottest_tags else "no"
        tag_results.append((tag, init_avg, final_avg, delta, same))

    # Sort by delta to show most heated tags
    tag_results.sort(key=lambda x: x[3], reverse=True)
    for tag, ia, fa, delta, same in tag_results[:15]:
        print(f"  {tag:<25} {ia:>9.3f} {fa:>10.3f} {delta:>+7.3f} {same:>13}")

    # Verdict: did heat cross semantic boundaries?
    cross_heated = [(t, d) for t, _, _, d, s in tag_results if s == "no" and d > 0.02]
    same_heated = [(t, d) for t, _, _, d, s in tag_results if s == "YES" and d > 0.02]

    print(f"\n  VERDICT:")
    print(f"    Tags in source domain heated >0.02: {len(same_heated)}")
    print(f"    Tags OUTSIDE source domain heated >0.02: {len(cross_heated)}")

    if cross_heated:
        print(f"    Cross-domain heat leak:")
        for tag, delta in cross_heated[:5]:
            print(f"      {tag}: {delta:+.3f}")
        if any(d > 0.05 for _, d in cross_heated):
            print(f"    WARNING: significant cross-domain heat leak (>0.05)")
            print(f"    Spreading may be too strong or conductivity too high")
        else:
            print(f"    Minor cross-domain warming — acceptable (shared tags create bridges)")
    else:
        print(f"    No cross-domain heat leak — spreading respects semantic boundaries")

    return True


# ============== MAIN ==============

def main():
    print("THERMAL LONGITUDINAL SOAK TESTS")
    print(f"Data root: {DATA_ROOT}")
    print(f"Snapshots: {len(SNAPSHOTS)}")
    for label, path in SNAPSHOTS:
        exists = os.path.exists(os.path.join(path, "beliefs.cog"))
        print(f"  [{label}] {path.split('data')[-1]} {'OK' if exists else 'MISSING'}")
    print()

    results = {}
    results["chronic_heat"] = study_chronic_heat()
    results["freezing_trajectory"] = study_freezing_trajectory()
    results["semantic_ripple"] = study_semantic_ripple()

    print()
    print("=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "CONCERN"
        print(f"  {name}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
