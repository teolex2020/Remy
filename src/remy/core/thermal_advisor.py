"""
Thermal Advisor — cognitive heat map for routing and observability.

Computes a thermodynamic map over the belief graph:
  - Tension/conflict = heat that spreads through shared-tag edges
  - Stable/confident = cold (heat sink)
  - Hot zones = clusters needing attention
  - Cold mass = stable knowledge that can sleep

This is purely advisory — it never mutates brain state.
Uses the v5 algorithm: composite initial temperature + outflow-limited spreading.

Integration: called from background_brain after run_maintenance(),
results surfaced in the background report and system instruction context.
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("ThermalAdvisor")

# --- Observation log ---
_THERMAL_LOG_FILENAME = "thermal_observations.jsonl"

# --- Algorithm parameters (validated in robustness sweep: 22/22 configs stable) ---
CONDUCTIVITY_PER_TAG = 0.12
MAX_OUTFLOW_FRACTION = 0.25
COOLING_FACTOR = 0.85
NUM_SPREAD_PASSES = 8

# --- Reporting thresholds ---
HOT_ZONE_THRESHOLD = 0.18
COLD_THRESHOLD = 0.08
CLUSTER_MIN_SIZE = 2


@dataclass
class ThermalNode:
    """Single belief's thermal state."""
    belief_id: str
    key: str
    temperature: float
    initial_temperature: float
    state: str
    confidence: float
    conflict_mass: float
    volatility: float
    stability: int
    degree: int = 0


@dataclass
class ThermalCluster:
    """A connected component of hot beliefs."""
    nodes: list[str]
    avg_temperature: float
    max_temperature: float
    dominant_tags: list[tuple[str, int]]
    has_conflict: bool
    has_unresolved: bool


@dataclass
class ThermalReport:
    """Full thermal map result."""
    total_energy: float
    mean_temperature: float
    variance: float
    hot_zone_count: int
    cold_mass_count: int
    clusters: list[ThermalCluster]
    top_hot: list[ThermalNode]
    routing_advice: list[str]
    node_count: int
    edge_count: int


def _load_beliefs_from_file(data_dir: str) -> Optional[dict]:
    """Load beliefs from beliefs.cog file."""
    path = os.path.join(data_dir, "beliefs.cog")
    if not os.path.exists(path):
        logger.debug("No beliefs.cog at %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("beliefs", {})
    except Exception as e:
        logger.warning("Failed to load beliefs.cog: %s", e)
        return None


def _build_graph(beliefs: dict):
    """Build adjacency graph from shared tags between beliefs."""
    tag_to_beliefs = defaultdict(set)
    for bid, b in beliefs.items():
        parts = b["key"].split(":")
        if len(parts) >= 2:
            for tag in parts[1].split(","):
                tag_to_beliefs[tag.strip()].add(bid)

    raw_edges = defaultdict(float)
    for tag, bids in tag_to_beliefs.items():
        bids_list = list(bids)
        for i in range(len(bids_list)):
            for j in range(i + 1, len(bids_list)):
                pair = tuple(sorted([bids_list[i], bids_list[j]]))
                raw_edges[pair] += CONDUCTIVITY_PER_TAG

    degree = defaultdict(int)
    for (a, b) in raw_edges:
        degree[a] += 1
        degree[b] += 1

    adj = defaultdict(list)
    for (a, b), conductivity in raw_edges.items():
        adj[a].append((b, conductivity))
        adj[b].append((a, conductivity))

    return adj, raw_edges, degree


def _initialize_temperature(beliefs: dict) -> dict[str, float]:
    """Composite initial temperature from cognitive signals (v5 algorithm)."""
    all_conf = [b["confidence"] for b in beliefs.values()]
    all_stab = [b["stability"] for b in beliefs.values()]
    all_supp = [b["support_mass"] for b in beliefs.values()]
    all_vol = [b["volatility"] for b in beliefs.values()]

    max_stab = max(all_stab) if max(all_stab) > 0 else 1.0
    max_supp = max(all_supp) if max(all_supp) > 0 else 1.0
    max_vol = max(all_vol) if max(all_vol) > 0 else 1.0
    max_conf = max(all_conf) if max(all_conf) > 0 else 1.0

    temps = {}
    for bid, b in beliefs.items():
        vol_norm = b["volatility"] / max_vol
        conflict = 1.0 if b["conflict_mass"] > 0 else 0.0
        unresolved = 1.0 if b["state"] == "Unresolved" else 0.0
        uncertainty = 1.0 - (b["confidence"] / max_conf)

        stab_norm = b["stability"] / max_stab
        conf_norm = b["confidence"] / max_conf
        supp_norm = b["support_mass"] / max_supp

        heat = (
            0.20 * vol_norm
            + 0.30 * conflict
            + 0.25 * unresolved
            + 0.15 * uncertainty
        )
        cold = (
            0.30 * stab_norm
            + 0.25 * conf_norm
            + 0.15 * supp_norm
        )
        temps[bid] = max(0.0, min(1.0, heat - cold + 0.10))

    return temps


def _spread_heat(temps: dict[str, float], adj: dict) -> int:
    """One pass of outflow-limited heat spreading. Returns transfer count."""
    transfers_list = []
    for node_a, neighbors in adj.items():
        for node_b, conductivity in neighbors:
            if node_a >= node_b:
                continue
            diff = temps[node_a] - temps[node_b]
            if abs(diff) < 1e-8:
                continue
            delta = conductivity * diff
            if delta > 0:
                transfers_list.append((node_a, node_b, delta))
            else:
                transfers_list.append((node_b, node_a, -delta))

    outflow_budget = {n: temps[n] * MAX_OUTFLOW_FRACTION for n in temps}
    total_desired = defaultdict(float)
    for hot, cold, delta in transfers_list:
        total_desired[hot] += delta

    scale = {}
    for node, total in total_desired.items():
        if total > 0:
            scale[node] = min(1.0, outflow_budget.get(node, 0) / total)

    deltas = defaultdict(float)
    actual = 0
    for hot, cold, delta in transfers_list:
        d = delta * scale.get(hot, 1.0)
        max_recv = 1.0 - (temps[cold] + deltas[cold])
        d = min(d, max_recv)
        if d > 1e-8:
            deltas[hot] -= d
            deltas[cold] += d
            actual += 1

    for node, d in deltas.items():
        temps[node] = max(0.0, min(1.0, temps[node] + d))
    return actual


def _find_clusters(hot_ids: set, adj: dict) -> list[set]:
    """Find connected components among hot nodes."""
    h_adj = defaultdict(set)
    for a in hot_ids:
        for bn, _ in adj.get(a, []):
            if bn in hot_ids:
                h_adj[a].add(bn)

    visited = set()
    clusters = []
    for n in hot_ids:
        if n in visited:
            continue
        cluster = set()
        queue = [n]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            cluster.add(cur)
            queue.extend(nb for nb in h_adj.get(cur, set()) if nb not in visited)
        if len(cluster) >= CLUSTER_MIN_SIZE:
            clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


def compute_thermal_map(data_dir: str) -> Optional[ThermalReport]:
    """Compute the full thermal map over the belief graph.

    Args:
        data_dir: path to brain data directory containing beliefs.cog

    Returns:
        ThermalReport with hot zones, cold mass, clusters, and routing advice.
        None if beliefs.cog doesn't exist or is empty.
    """
    beliefs = _load_beliefs_from_file(data_dir)
    if not beliefs:
        return None

    adj, edges, degree = _build_graph(beliefs)

    # Apply synaptic plasticity: modify conductivity based on edge health
    try:
        from remy.core.synaptic_plasticity import get_conductivity_modifiers, _edge_key
        modifiers = get_conductivity_modifiers(data_dir)
        if modifiers:
            for node_a in list(adj.keys()):
                adj[node_a] = [
                    (nb, cond * modifiers.get(_edge_key(node_a, nb), 1.0))
                    for nb, cond in adj[node_a]
                ]
    except Exception:
        pass

    temps = _initialize_temperature(beliefs)
    temps_before = dict(temps)

    # Spread heat
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps, adj)

    # Run plasticity cycle: track edge utility, weaken/prune leaky edges
    try:
        from remy.core.synaptic_plasticity import run_plasticity_cycle
        plasticity = run_plasticity_cycle(data_dir, beliefs, temps_before, temps, adj)
        if plasticity.edges_pruned > 0 or plasticity.edges_weakened > 0:
            logger.info(
                "Plasticity: %d evaluated, %d leaks, %d weakened, %d pruned, %d recovered",
                plasticity.edges_evaluated, plasticity.leaks_detected,
                plasticity.edges_weakened, plasticity.edges_pruned, plasticity.edges_recovered,
            )
    except Exception as e:
        logger.debug("Plasticity cycle skipped: %s", e)

    # Cool
    for n in temps:
        temps[n] *= COOLING_FACTOR

    # Analyze
    vals = list(temps.values())
    mean_t = sum(vals) / len(vals) if vals else 0
    variance = sum((v - mean_t) ** 2 for v in vals) / len(vals) if vals else 0

    hot_ids = {bid for bid, t in temps.items() if t > HOT_ZONE_THRESHOLD}
    cold_ids = {bid for bid, t in temps.items() if t < COLD_THRESHOLD}

    # Build clusters
    raw_clusters = _find_clusters(hot_ids, adj)
    clusters = []
    for cl in raw_clusters[:10]:
        tags = defaultdict(int)
        has_conflict = False
        has_unresolved = False
        for bid in cl:
            b = beliefs[bid]
            if b["conflict_mass"] > 0:
                has_conflict = True
            if b["state"] == "Unresolved":
                has_unresolved = True
            parts = b["key"].split(":")
            if len(parts) >= 2:
                for tag in parts[1].split(","):
                    tags[tag.strip()] += 1

        dominant = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:4]
        avg_t = sum(temps[n] for n in cl) / len(cl)
        max_t = max(temps[n] for n in cl)
        clusters.append(ThermalCluster(
            nodes=list(cl),
            avg_temperature=round(avg_t, 4),
            max_temperature=round(max_t, 4),
            dominant_tags=dominant,
            has_conflict=has_conflict,
            has_unresolved=has_unresolved,
        ))

    # Top hot nodes
    sorted_by_temp = sorted(temps.items(), key=lambda x: x[1], reverse=True)
    top_hot = []
    for bid, t in sorted_by_temp[:15]:
        b = beliefs[bid]
        top_hot.append(ThermalNode(
            belief_id=bid,
            key=b["key"],
            temperature=round(t, 4),
            initial_temperature=round(temps_before[bid], 4),
            state=b["state"],
            confidence=b["confidence"],
            conflict_mass=b["conflict_mass"],
            volatility=b["volatility"],
            stability=b["stability"],
            degree=degree.get(bid, 0),
        ))

    # Generate routing advice
    advice = _generate_routing_advice(clusters, top_hot, len(cold_ids), len(beliefs))

    return ThermalReport(
        total_energy=round(sum(vals), 3),
        mean_temperature=round(mean_t, 4),
        variance=round(variance, 6),
        hot_zone_count=len(hot_ids),
        cold_mass_count=len(cold_ids),
        clusters=clusters,
        top_hot=top_hot,
        routing_advice=advice,
        node_count=len(beliefs),
        edge_count=len(edges),
    )


def _generate_routing_advice(
    clusters: list[ThermalCluster],
    top_hot: list[ThermalNode],
    cold_count: int,
    total_count: int,
) -> list[str]:
    """Generate human-readable routing recommendations."""
    advice = []

    # Conflict clusters get priority
    conflict_clusters = [c for c in clusters if c.has_conflict]
    if conflict_clusters:
        for c in conflict_clusters[:3]:
            tags_str = ", ".join(t for t, _ in c.dominant_tags[:3])
            advice.append(
                f"CONFLICT HOT ZONE ({len(c.nodes)} beliefs, avg {c.avg_temperature:.3f}): "
                f"topics [{tags_str}] — resolve contradictions first"
            )

    # Unresolved clusters next
    unresolved_clusters = [c for c in clusters if c.has_unresolved and not c.has_conflict]
    for c in unresolved_clusters[:2]:
        tags_str = ", ".join(t for t, _ in c.dominant_tags[:3])
        advice.append(
            f"UNRESOLVED HOT ZONE ({len(c.nodes)} beliefs): "
            f"topics [{tags_str}] — needs investigation"
        )

    # High-heat individual nodes not in clusters
    clustered_ids = set()
    for c in clusters:
        clustered_ids.update(c.nodes)
    isolated_hot = [n for n in top_hot if n.belief_id not in clustered_ids and n.temperature > HOT_ZONE_THRESHOLD]
    if isolated_hot:
        advice.append(
            f"{len(isolated_hot)} isolated hot beliefs (not in clusters) — may be emerging concerns"
        )

    # Cold mass ratio
    if total_count > 0:
        cold_ratio = cold_count / total_count
        if cold_ratio > 0.7:
            advice.append(
                f"COLD MASS: {cold_count}/{total_count} ({cold_ratio:.0%}) beliefs stable — "
                f"safe to skip in maintenance"
            )
        elif cold_ratio > 0.4:
            advice.append(
                f"WARM GRAPH: {cold_count}/{total_count} ({cold_ratio:.0%}) cold — "
                f"moderate activity across graph"
            )

    return advice


def format_thermal_summary(report: ThermalReport, locale: str = "en") -> str:
    """Format thermal report as compact text for system instruction context.

    Uses ACL renderer for deterministic, locale-aware output.
    The brain speaks without LLM.
    """
    if not report:
        return ""

    from remy.core.acl_renderer import (
        Locale,
        thermal_summary_from_report,
        render_thermal_summary as _render,
    )

    loc = Locale.from_str(locale)
    expr = thermal_summary_from_report(report)
    return _render(expr, loc)


def format_thermal_report_json(report: ThermalReport) -> dict:
    """Serialize thermal report to JSON-safe dict for background report."""
    if not report:
        return {}
    return {
        "total_energy": report.total_energy,
        "mean_temperature": report.mean_temperature,
        "variance": report.variance,
        "hot_zone_count": report.hot_zone_count,
        "cold_mass_count": report.cold_mass_count,
        "node_count": report.node_count,
        "edge_count": report.edge_count,
        "cluster_count": len(report.clusters),
        "clusters": [
            {
                "size": len(c.nodes),
                "avg_temp": c.avg_temperature,
                "max_temp": c.max_temperature,
                "tags": [t for t, _ in c.dominant_tags[:4]],
                "has_conflict": c.has_conflict,
                "has_unresolved": c.has_unresolved,
            }
            for c in report.clusters[:10]
        ],
        "top_hot": [
            {
                "key": n.key[:60],
                "temp": n.temperature,
                "state": n.state,
                "conflict": n.conflict_mass,
                "degree": n.degree,
            }
            for n in report.top_hot[:10]
        ],
        "routing_advice": report.routing_advice,
    }


# ============== OBSERVATION LOG ==============

def append_thermal_observation(
    data_dir: str,
    report: ThermalReport,
    routing: Optional[dict] = None,
    llm_gating: Optional[dict] = None,
) -> None:
    """Append one observation to the JSONL thermal log.

    Each line is a self-contained JSON object with timestamp, thermal metrics,
    cluster summary, routing decision, and LLM gating outcome.
    Written to data_dir/../thermal_observations.jsonl (next to brain/).
    """
    log_path = os.path.join(os.path.dirname(data_dir), _THERMAL_LOG_FILENAME)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "energy": report.total_energy,
        "mean_temp": report.mean_temperature,
        "variance": report.variance,
        "hot": report.hot_zone_count,
        "cold": report.cold_mass_count,
        "nodes": report.node_count,
        "edges": report.edge_count,
        "clusters": [
            {
                "size": len(c.nodes),
                "avg": c.avg_temperature,
                "tags": [t for t, _ in c.dominant_tags[:3]],
                "conflict": c.has_conflict,
                "unresolved": c.has_unresolved,
            }
            for c in report.clusters[:6]
        ],
        "top5": [
            {"key": n.key[:50], "temp": n.temperature, "state": n.state}
            for n in report.top_hot[:5]
        ],
    }

    if routing:
        entry["routing"] = routing

    if llm_gating:
        entry["llm_gating"] = llm_gating

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to write thermal log: %s", e)


# ============== MAINTENANCE ROUTING (Experiment 4) ==============

# Cycle counter for periodic full scans
_cycle_count: int = 0
FULL_SCAN_INTERVAL: int = 5  # every Nth cycle, ignore thermal routing and scan everything

# Cold-zone skip threshold: clusters whose tags are ALL cold get deferred
COLD_SKIP_THRESHOLD = 0.06


@dataclass
class RoutingDecision:
    """Result of thermal routing for one maintenance cycle."""
    mode: str  # "hot_first" | "full_scan" | "no_thermal"
    cycle_number: int
    hot_priority_tags: list[str]     # tags in hot zones, sorted by temperature
    cold_skip_tags: list[str]        # tags safe to defer this cycle
    neutral_tags: list[str]          # everything else — process normally
    thermal_available: bool


def get_maintenance_routing(data_dir: str) -> RoutingDecision:
    """Compute thermal routing for current maintenance cycle.

    Three modes:
      - hot_first: thermal map available, route by temperature
      - full_scan: periodic fallback, process everything regardless
      - no_thermal: thermal map unavailable, process everything

    Returns RoutingDecision with tag priority lists.
    """
    global _cycle_count
    _cycle_count += 1

    # Periodic full scan — don't skip anything
    if _cycle_count % FULL_SCAN_INTERVAL == 0:
        return RoutingDecision(
            mode="full_scan",
            cycle_number=_cycle_count,
            hot_priority_tags=[],
            cold_skip_tags=[],
            neutral_tags=[],
            thermal_available=True,
        )

    report = compute_thermal_map(data_dir)
    if not report:
        return RoutingDecision(
            mode="no_thermal",
            cycle_number=_cycle_count,
            hot_priority_tags=[],
            cold_skip_tags=[],
            neutral_tags=[],
            thermal_available=False,
        )

    # Build tag → max temperature mapping from belief graph
    tag_temps: dict[str, float] = {}
    beliefs = _load_beliefs_from_file(data_dir)
    if not beliefs:
        return RoutingDecision(
            mode="no_thermal",
            cycle_number=_cycle_count,
            hot_priority_tags=[],
            cold_skip_tags=[],
            neutral_tags=[],
            thermal_available=False,
        )

    # Get final temperatures from report's top_hot (which has the thermal map values)
    # Rebuild temps quickly from the cached belief data
    adj, _, _ = _build_graph(beliefs)
    temps = _initialize_temperature(beliefs)
    for _ in range(NUM_SPREAD_PASSES):
        _spread_heat(temps, adj)
    for n in temps:
        temps[n] *= COOLING_FACTOR

    # Map tags to their max belief temperature
    for bid, b in beliefs.items():
        parts = b["key"].split(":")
        if len(parts) >= 2:
            for tag in parts[1].split(","):
                tag = tag.strip()
                t = temps.get(bid, 0)
                if tag not in tag_temps or t > tag_temps[tag]:
                    tag_temps[tag] = t

    hot_tags = []
    cold_tags = []
    neutral_tags = []

    for tag, max_t in tag_temps.items():
        if max_t > HOT_ZONE_THRESHOLD:
            hot_tags.append((tag, max_t))
        elif max_t < COLD_SKIP_THRESHOLD:
            cold_tags.append(tag)
        else:
            neutral_tags.append(tag)

    # Sort hot tags by temperature descending
    hot_tags.sort(key=lambda x: x[1], reverse=True)

    return RoutingDecision(
        mode="hot_first",
        cycle_number=_cycle_count,
        hot_priority_tags=[t for t, _ in hot_tags],
        cold_skip_tags=cold_tags,
        neutral_tags=neutral_tags,
        thermal_available=True,
    )


def sort_clusters_by_thermal_priority(
    clusters: list[tuple[str, list[str]]],
    routing: RoutingDecision,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]], dict]:
    """Sort consolidation clusters by thermal priority.

    Args:
        clusters: list of (tag_key, record_ids) from _find_consolidation_clusters
        routing: current RoutingDecision

    Returns:
        (prioritized_clusters, deferred_clusters, stats)
        - prioritized: hot-first, then neutral, ordered by max tag temperature
        - deferred: cold-zone clusters skipped this cycle
        - stats: routing diagnostics
    """
    if routing.mode in ("full_scan", "no_thermal"):
        return clusters, [], {"mode": routing.mode, "deferred": 0}

    hot_set = set(routing.hot_priority_tags)
    cold_set = set(routing.cold_skip_tags)

    prioritized = []
    deferred = []

    for tag_key, record_ids in clusters:
        cluster_tags = set(t.strip() for t in tag_key.split(",") if t.strip())

        # Cluster is "cold" only if ALL its tags are in cold set
        if cluster_tags and cluster_tags <= cold_set:
            deferred.append((tag_key, record_ids))
            continue

        # Score by hottest tag
        has_hot = bool(cluster_tags & hot_set)
        prioritized.append((tag_key, record_ids, has_hot))

    # Hot clusters first, then neutral
    prioritized.sort(key=lambda x: (not x[2], -len(x[1])))
    sorted_clusters = [(tk, rids) for tk, rids, _ in prioritized]

    stats = {
        "mode": routing.mode,
        "cycle": routing.cycle_number,
        "prioritized": len(sorted_clusters),
        "deferred": len(deferred),
        "hot_first": sum(1 for _, _, h in prioritized if h),
    }

    return sorted_clusters, deferred, stats


# ============== CYCLE CLASSIFICATION (Frontier 1) ==============


class CycleType:
    """Thermal cycle classification for maintenance orchestration."""
    HOT = "hot_cycle"        # hot zones present → full processing on hot, shallow on cold
    WARM = "warm_cycle"      # no hot zones, moderate activity → standard processing
    COLD = "cold_cycle"      # graph mostly cold → minimal processing, skip expensive phases
    FULL_SCAN = "full_scan"  # periodic override → process everything


@dataclass
class CycleClassification:
    """Result of thermal cycle classification."""
    cycle_type: str
    cycle_number: int
    hot_zone_count: int
    mean_temperature: float
    has_conflicts: bool
    skip_diagnostics: bool   # skip trajectory, drift, V16 diagnostics
    skip_insights: bool      # skip insight formatting
    skip_llm: bool           # skip LLM consolidation entirely
    reason: str


# Thresholds for cycle classification
_WARM_MEAN_THRESHOLD = 0.10  # below this = cold cycle


def classify_cycle(report: Optional[ThermalReport], cycle_number: int) -> CycleClassification:
    """Classify the current maintenance cycle based on thermal state.

    Called BEFORE any processing. Returns classification that guides
    which phases run, how deep, and what gets skipped.
    """
    # Periodic full scan overrides everything
    if cycle_number % FULL_SCAN_INTERVAL == 0:
        return CycleClassification(
            cycle_type=CycleType.FULL_SCAN,
            cycle_number=cycle_number,
            hot_zone_count=0,
            mean_temperature=0.0,
            has_conflicts=False,
            skip_diagnostics=False,
            skip_insights=False,
            skip_llm=False,
            reason="periodic full scan",
        )

    # No thermal data — run everything (safe fallback)
    if not report:
        return CycleClassification(
            cycle_type=CycleType.WARM,
            cycle_number=cycle_number,
            hot_zone_count=0,
            mean_temperature=0.0,
            has_conflicts=False,
            skip_diagnostics=False,
            skip_insights=False,
            skip_llm=False,
            reason="no thermal data, safe fallback",
        )

    has_conflicts = any(c.has_conflict for c in report.clusters)

    # Hot cycle: hot zones or active conflicts
    if report.hot_zone_count > 0:
        return CycleClassification(
            cycle_type=CycleType.HOT,
            cycle_number=cycle_number,
            hot_zone_count=report.hot_zone_count,
            mean_temperature=report.mean_temperature,
            has_conflicts=has_conflicts,
            skip_diagnostics=False,
            skip_insights=False,
            skip_llm=False,
            reason=f"{report.hot_zone_count} hot zones"
                   + (", conflicts present" if has_conflicts else ""),
        )

    # Cold cycle: low mean temp, no conflicts
    if report.mean_temperature < _WARM_MEAN_THRESHOLD and not has_conflicts:
        return CycleClassification(
            cycle_type=CycleType.COLD,
            cycle_number=cycle_number,
            hot_zone_count=0,
            mean_temperature=report.mean_temperature,
            has_conflicts=False,
            skip_diagnostics=True,
            skip_insights=True,
            skip_llm=True,
            reason=f"mean_temp={report.mean_temperature:.3f} < {_WARM_MEAN_THRESHOLD}",
        )

    # Warm cycle: moderate activity
    return CycleClassification(
        cycle_type=CycleType.WARM,
        cycle_number=cycle_number,
        hot_zone_count=0,
        mean_temperature=report.mean_temperature,
        has_conflicts=has_conflicts,
        skip_diagnostics=False,
        skip_insights=False,
        skip_llm=False,
        reason=f"mean_temp={report.mean_temperature:.3f}, warm",
    )
