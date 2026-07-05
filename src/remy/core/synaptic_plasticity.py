"""
Synaptic Plasticity — Autonomous Structural Rewiring, Slice 1.

Edge Weakening and Pruning based on thermal leak detection.

Problem: shared hub tags (e.g. "aurasdk") create high-conductivity bridges
between semantically unrelated belief clusters, causing cross-domain heat leak.

Solution: track edge utility. Edges that repeatedly conduct heat between
domains with no productive outcome (no conflict resolved, no cooling achieved)
accumulate leak penalty. Their conductivity weakens over cycles. After
sustained penalty, the edge is pruned (conductivity set to zero).

Three constraints:
  1. No instant deletion — weaken first, prune only after repeated evidence
  2. Protected edge classes — same-domain edges are exempt from pruning
  3. Operator-visible pruning log — every action is inspectable

Persistence: edge_health.jsonl in brain data directory.
"""

import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("SynapticPlasticity")

# --- Parameters ---
# How much penalty an edge gets per cycle of unproductive heat transfer
LEAK_PENALTY_PER_CYCLE = 0.05
# How much penalty decays per cycle (forgiveness if edge stops leaking)
PENALTY_DECAY_PER_CYCLE = 0.02
# Penalty threshold to start weakening conductivity
WEAKENING_THRESHOLD = 0.15  # ~3 cycles of leak
# Penalty threshold to prune (set conductivity to 0)
PRUNING_THRESHOLD = 0.40  # ~8 cycles of sustained leak
# Minimum conductivity — never weaken below this (keeps edge alive for recovery)
MIN_CONDUCTIVITY_FRACTION = 0.1
# Maximum edges to prune per cycle
MAX_PRUNES_PER_CYCLE = 3

_EDGE_HEALTH_FILE = "edge_health.json"
_PRUNING_LOG_FILE = "pruning_log.jsonl"


@dataclass
class EdgeHealth:
    """Health state of a single edge."""
    edge_key: str  # "bid_a|bid_b" sorted
    leak_penalty: float  # accumulated penalty [0, 1+]
    productive_count: int  # times this edge conducted useful heat
    leak_count: int  # times this edge conducted cross-domain heat
    conductivity_modifier: float  # multiplier on base conductivity [0, 1]
    pruned: bool
    last_updated: str


def _edge_key(a: str, b: str) -> str:
    """Deterministic edge key from two belief IDs."""
    return "|".join(sorted([a, b]))


def _extract_tags(key: str) -> set:
    parts = key.split(":")
    if len(parts) >= 2:
        return {t.strip() for t in parts[1].split(",")}
    return set()


def _extract_namespace(key: str) -> str:
    parts = key.split(":")
    return parts[0] if parts else "default"


def _is_same_domain(tags_a: set, tags_b: set) -> bool:
    """Two beliefs are same-domain if they share at least 2 non-hub tags."""
    shared = tags_a & tags_b
    # Exclude very common tags (hubs) — they don't indicate true domain overlap
    return len(shared) >= 2


def _is_protected_edge(beliefs: dict, bid_a: str, bid_b: str) -> bool:
    """Protected edges: same namespace + same semantic type, or explicit causal links."""
    a = beliefs.get(bid_a, {})
    b = beliefs.get(bid_b, {})
    key_a = a.get("key", "")
    key_b = b.get("key", "")

    # Same namespace
    ns_a = _extract_namespace(key_a)
    ns_b = _extract_namespace(key_b)
    if ns_a != ns_b:
        return False  # cross-namespace = not protected

    # Same semantic type (last part of key before #)
    parts_a = key_a.split(":")
    parts_b = key_b.split(":")
    if len(parts_a) >= 3 and len(parts_b) >= 3:
        st_a = parts_a[2].split("#")[0]
        st_b = parts_b[2].split("#")[0]
        if st_a == st_b:
            return True  # same type = protected

    # Strong tag overlap = protected
    tags_a = _extract_tags(key_a)
    tags_b = _extract_tags(key_b)
    if _is_same_domain(tags_a, tags_b):
        return True

    return False


def load_edge_health(data_dir: str) -> dict[str, EdgeHealth]:
    """Load persisted edge health state."""
    path = os.path.join(data_dir, _EDGE_HEALTH_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as decode_error:
            decoder = json.JSONDecoder()
            data, end = decoder.raw_decode(raw)
            if raw[end:].strip():
                logger.info(
                    "Recovered edge health from %s by ignoring trailing corrupted data after char %d: %s",
                    path,
                    end,
                    decode_error,
                )
        health = {k: EdgeHealth(**v) for k, v in data.items()}
        if len(raw) > 0 and "end" in locals() and raw[end:].strip():
            save_edge_health(data_dir, health)
        return health
    except Exception as e:
        logger.warning("Failed to load edge health: %s", e)
        return {}


def save_edge_health(data_dir: str, health: dict[str, EdgeHealth]) -> None:
    """Persist edge health state."""
    path = os.path.join(data_dir, _EDGE_HEALTH_FILE)
    os.makedirs(data_dir, exist_ok=True)
    data = {}
    for k, eh in health.items():
        data[k] = {
            "edge_key": eh.edge_key,
            "leak_penalty": round(eh.leak_penalty, 4),
            "productive_count": eh.productive_count,
            "leak_count": eh.leak_count,
            "conductivity_modifier": round(eh.conductivity_modifier, 4),
            "pruned": eh.pruned,
            "last_updated": eh.last_updated,
        }
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=data_dir,
            prefix=".edge_health.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception as e:
        logger.warning("Failed to save edge health: %s", e)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _append_pruning_log(data_dir: str, entry: dict) -> None:
    """Append to operator-visible pruning log."""
    path = os.path.join(data_dir, _PRUNING_LOG_FILE)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to write pruning log: %s", e)


@dataclass
class PlasticityResult:
    """Result of one plasticity cycle."""
    edges_evaluated: int
    leaks_detected: int
    edges_weakened: int
    edges_pruned: int
    edges_recovered: int  # penalty decayed below threshold
    pruned_edges: list[dict]  # details of pruned edges


def run_plasticity_cycle(
    data_dir: str,
    beliefs: dict,
    temps_before: dict[str, float],
    temps_after: dict[str, float],
    adj: dict,
) -> PlasticityResult:
    """Run one cycle of edge utility tracking and pruning.

    Called after thermal spreading. Compares temps before and after spreading
    to identify which edges conducted heat unproductively (cross-domain leak
    without cooling the source or resolving conflict).

    Args:
        data_dir: brain data directory
        beliefs: current belief dict
        temps_before: temperatures before spreading
        temps_after: temperatures after spreading
        adj: adjacency list from _build_graph
    """
    health = load_edge_health(data_dir)
    now = datetime.now(timezone.utc).isoformat()

    result = PlasticityResult(
        edges_evaluated=0,
        leaks_detected=0,
        edges_weakened=0,
        edges_pruned=0,
        edges_recovered=0,
        pruned_edges=[],
    )

    # Phase 1: Evaluate each edge for productive vs leaky heat transfer
    evaluated_edges = set()

    for node_a, neighbors in adj.items():
        for node_b, conductivity in neighbors:
            ek = _edge_key(node_a, node_b)
            if ek in evaluated_edges:
                continue
            evaluated_edges.add(ek)
            result.edges_evaluated += 1

            # Skip already-pruned edges — no point tracking utility on dead connections
            if ek in health and health[ek].pruned:
                continue

            # Skip protected edges
            if _is_protected_edge(beliefs, node_a, node_b):
                continue

            # Did heat flow across this edge?
            diff_before = abs(temps_before.get(node_a, 0) - temps_before.get(node_b, 0))
            diff_after = abs(temps_after.get(node_a, 0) - temps_after.get(node_b, 0))

            if diff_before < 0.01:
                continue  # no temperature gradient, no flow

            # Was the transfer productive?
            # Productive = source cooled meaningfully, or gradient reduced
            source = node_a if temps_before.get(node_a, 0) > temps_before.get(node_b, 0) else node_b
            sink = node_b if source == node_a else node_a

            source_cooled = temps_after.get(source, 0) < temps_before.get(source, 0) - 0.01
            gradient_reduced = diff_after < diff_before * 0.9

            # Cross-domain check
            tags_a = _extract_tags(beliefs.get(node_a, {}).get("key", ""))
            tags_b = _extract_tags(beliefs.get(node_b, {}).get("key", ""))
            cross_domain = not _is_same_domain(tags_a, tags_b)

            # Leak = cross-domain heat flow that didn't cool source or reduce gradient
            is_leak = cross_domain and not source_cooled and not gradient_reduced

            # Update edge health
            if ek not in health:
                health[ek] = EdgeHealth(
                    edge_key=ek,
                    leak_penalty=0.0,
                    productive_count=0,
                    leak_count=0,
                    conductivity_modifier=1.0,
                    pruned=False,
                    last_updated=now,
                )

            eh = health[ek]
            eh.last_updated = now

            if is_leak:
                eh.leak_penalty = min(1.0, eh.leak_penalty + LEAK_PENALTY_PER_CYCLE)
                eh.leak_count += 1
                result.leaks_detected += 1
            else:
                eh.productive_count += 1
                # Decay penalty for productive use (stronger than leak accumulation)
                eh.leak_penalty = max(0.0, eh.leak_penalty - PENALTY_DECAY_PER_CYCLE * 3)

    # Phase 2: Apply penalty decay to all edges (forgiveness over time)
    for ek, eh in health.items():
        if ek not in evaluated_edges:
            eh.leak_penalty = max(0.0, eh.leak_penalty - PENALTY_DECAY_PER_CYCLE)
            eh.last_updated = now

    # Phase 3: Compute conductivity modifiers and prune
    prune_candidates = []

    for ek, eh in health.items():
        old_modifier = eh.conductivity_modifier

        if eh.pruned:
            continue

        if eh.leak_penalty >= PRUNING_THRESHOLD:
            # Safety gate: never prune edges that are more productive than leaky
            if eh.productive_count > eh.leak_count:
                continue
            prune_candidates.append((ek, eh))
        elif eh.leak_penalty >= WEAKENING_THRESHOLD:
            # Linear weakening: penalty 0.15-0.40 maps to modifier 1.0-0.1
            t = (eh.leak_penalty - WEAKENING_THRESHOLD) / (PRUNING_THRESHOLD - WEAKENING_THRESHOLD)
            eh.conductivity_modifier = max(MIN_CONDUCTIVITY_FRACTION, 1.0 - t * (1.0 - MIN_CONDUCTIVITY_FRACTION))
            if old_modifier > eh.conductivity_modifier:
                result.edges_weakened += 1
        else:
            # Recovery: if penalty dropped below threshold, restore
            if eh.conductivity_modifier < 1.0:
                eh.conductivity_modifier = min(1.0, eh.conductivity_modifier + 0.1)
                result.edges_recovered += 1

    # Phase 4: Prune (limited per cycle)
    prune_candidates.sort(key=lambda x: x[1].leak_penalty, reverse=True)
    for ek, eh in prune_candidates[:MAX_PRUNES_PER_CYCLE]:
        eh.pruned = True
        eh.conductivity_modifier = 0.0
        result.edges_pruned += 1

        # Build log entry
        parts = ek.split("|")
        key_a = beliefs.get(parts[0], {}).get("key", "?")[:40] if len(parts) > 0 else "?"
        key_b = beliefs.get(parts[1], {}).get("key", "?")[:40] if len(parts) > 1 else "?"
        pruned_info = {
            "edge": ek,
            "belief_a": key_a,
            "belief_b": key_b,
            "leak_penalty": round(eh.leak_penalty, 3),
            "leak_count": eh.leak_count,
            "productive_count": eh.productive_count,
        }
        result.pruned_edges.append(pruned_info)

        _append_pruning_log(data_dir, {
            "ts": now,
            "action": "pruned",
            "edge": ek,
            "belief_a": key_a,
            "belief_b": key_b,
            "penalty": round(eh.leak_penalty, 3),
            "leaks": eh.leak_count,
            "productive": eh.productive_count,
        })
        logger.info("PRUNED edge %s <-> %s (penalty=%.3f, leaks=%d)",
                     key_a, key_b, eh.leak_penalty, eh.leak_count)

    save_edge_health(data_dir, health)
    return result


def get_conductivity_modifiers(data_dir: str) -> dict[str, float]:
    """Get current conductivity modifiers for all tracked edges.

    Returns dict of edge_key -> modifier (0.0 = pruned, 1.0 = healthy).
    Used by thermal_advisor to apply plasticity to spreading.
    """
    health = load_edge_health(data_dir)
    return {ek: eh.conductivity_modifier for ek, eh in health.items() if eh.conductivity_modifier < 1.0}


def get_plasticity_summary(data_dir: str) -> dict:
    """Get summary of edge health state for diagnostics."""
    health = load_edge_health(data_dir)
    if not health:
        return {"total_tracked": 0}

    pruned = sum(1 for eh in health.values() if eh.pruned)
    weakened = sum(1 for eh in health.values() if not eh.pruned and eh.conductivity_modifier < 1.0)
    healthy = sum(1 for eh in health.values() if eh.conductivity_modifier == 1.0 and not eh.pruned)
    total_leaks = sum(eh.leak_count for eh in health.values())
    total_productive = sum(eh.productive_count for eh in health.values())

    return {
        "total_tracked": len(health),
        "healthy": healthy,
        "weakened": weakened,
        "pruned": pruned,
        "total_leaks": total_leaks,
        "total_productive": total_productive,
    }


def get_plasticity_audit(data_dir: str, beliefs: Optional[dict] = None) -> dict:
    """Full audit of plasticity state for operator review.

    Returns structured report with:
      - summary stats
      - pruned edges with belief context
      - weakened edges approaching prune threshold
      - pruning rate (prunes per tracked edge)
      - health distribution
      - pruning log tail (last 10 entries)
    """
    health = load_edge_health(data_dir)
    if not health:
        return {"status": "no_data", "message": "No edge state data yet — plasticity has not run"}

    # Load beliefs for context if not provided
    if beliefs is None:
        beliefs_path = os.path.join(data_dir, "beliefs.cog")
        if os.path.exists(beliefs_path):
            try:
                with open(beliefs_path, encoding="utf-8") as f:
                    beliefs = json.load(f).get("beliefs", {})
            except Exception:
                beliefs = {}
        else:
            beliefs = {}

    def _edge_context(ek: str) -> dict:
        parts = ek.split("|")
        a_key = beliefs.get(parts[0], {}).get("key", "?")[:50] if len(parts) > 0 else "?"
        b_key = beliefs.get(parts[1], {}).get("key", "?")[:50] if len(parts) > 1 else "?"
        tags_a = _extract_tags(a_key) if a_key != "?" else set()
        tags_b = _extract_tags(b_key) if b_key != "?" else set()
        shared = tags_a & tags_b
        return {"a": a_key, "b": b_key, "shared_tags": list(shared)}

    # Pruned edges with context
    pruned_list = []
    for ek, eh in health.items():
        if eh.pruned:
            ctx = _edge_context(ek)
            pruned_list.append({
                **ctx,
                "leaks": eh.leak_count,
                "productive": eh.productive_count,
                "penalty": round(eh.leak_penalty, 3),
            })

    # Weakened edges approaching prune
    at_risk = []
    for ek, eh in health.items():
        if not eh.pruned and eh.leak_penalty >= WEAKENING_THRESHOLD:
            ctx = _edge_context(ek)
            at_risk.append({
                **ctx,
                "penalty": round(eh.leak_penalty, 3),
                "modifier": round(eh.conductivity_modifier, 2),
                "leaks": eh.leak_count,
                "productive": eh.productive_count,
                "cycles_to_prune": max(0, int((PRUNING_THRESHOLD - eh.leak_penalty) / LEAK_PENALTY_PER_CYCLE)),
            })
    at_risk.sort(key=lambda x: x["penalty"], reverse=True)

    # Pruning rate
    total = len(health)
    pruned_count = sum(1 for eh in health.values() if eh.pruned)
    weakened_count = sum(1 for eh in health.values() if not eh.pruned and eh.conductivity_modifier < 1.0)

    # Pruning log tail
    log_tail = []
    log_path = os.path.join(data_dir, _PRUNING_LOG_FILE)
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-10:]:
                if line.strip():
                    log_tail.append(json.loads(line))
        except Exception:
            pass

    # Leak-to-productive ratio
    total_leaks = sum(eh.leak_count for eh in health.values())
    total_productive = sum(eh.productive_count for eh in health.values())
    leak_ratio = total_leaks / (total_leaks + total_productive) if (total_leaks + total_productive) > 0 else 0

    return {
        "status": "active",
        "summary": {
            "total_edges": total,
            "healthy": total - pruned_count - weakened_count,
            "weakened": weakened_count,
            "pruned": pruned_count,
            "prune_rate": round(pruned_count / total, 3) if total > 0 else 0,
            "leak_ratio": round(leak_ratio, 3),
            "total_leaks": total_leaks,
            "total_productive": total_productive,
        },
        "pruned_edges": pruned_list[:20],
        "at_risk_edges": at_risk[:10],
        "pruning_log_tail": log_tail,
    }
