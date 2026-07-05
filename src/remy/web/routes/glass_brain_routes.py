"""
Glass Brain routes — cognitive heat map, plasticity, and belief graph.
Exposes thermal advisor and synaptic plasticity data for the Glass Brain view.
"""

import json
import logging
import os
import time

from fastapi import APIRouter

from remy.web.routes._helpers import _get_api, run_in_thread

logger = logging.getLogger("GlassBrain")

router = APIRouter()

_TIMEOUT = 12.0


def _data_dir() -> str:
    from remy.config.settings import settings
    return str(settings.AURA_BRAIN_PATH)


# ── Thermal map ───────────────────────────────────────────────────────────────

@router.get("/glass-brain/thermal-map")
async def get_thermal_map():
    """Full thermal map: hot zones, clusters, top hot beliefs, routing advice."""
    def _compute():
        from remy.core.thermal_advisor import compute_thermal_map
        data_dir = _data_dir()
        report = compute_thermal_map(data_dir)
        if not report:
            return {"status": "no_data", "message": "No beliefs.cog yet — brain hasn't been trained"}

        clusters = []
        for cl in report.clusters:
            clusters.append({
                "size": len(cl.nodes),
                "avg_temperature": cl.avg_temperature,
                "max_temperature": cl.max_temperature,
                "dominant_tags": [{"tag": t, "count": c} for t, c in cl.dominant_tags],
                "has_conflict": cl.has_conflict,
                "has_unresolved": cl.has_unresolved,
            })

        top_hot = []
        for node in report.top_hot:
            top_hot.append({
                "belief_id": node.belief_id,
                "key": node.key,
                "temperature": node.temperature,
                "initial_temperature": node.initial_temperature,
                "state": node.state,
                "confidence": node.confidence,
                "conflict_mass": node.conflict_mass,
                "volatility": node.volatility,
                "stability": node.stability,
                "degree": node.degree,
            })

        return {
            "status": "ok",
            "total_energy": report.total_energy,
            "mean_temperature": report.mean_temperature,
            "variance": report.variance,
            "hot_zone_count": report.hot_zone_count,
            "cold_mass_count": report.cold_mass_count,
            "node_count": report.node_count,
            "edge_count": report.edge_count,
            "clusters": clusters,
            "top_hot": top_hot,
            "routing_advice": report.routing_advice,
        }

    try:
        return await run_in_thread(_compute, timeout=_TIMEOUT, error_msg="Thermal map timed out")
    except Exception as exc:
        logger.warning("Thermal map failed: %s", exc)
        return {"status": "error", "message": str(exc)}


# ── Plasticity ────────────────────────────────────────────────────────────────

@router.get("/glass-brain/plasticity")
async def get_plasticity():
    """Edge health audit: pruned, weakened, at-risk edges, leak ratio."""
    def _compute():
        from remy.core.synaptic_plasticity import get_plasticity_audit
        data_dir = _data_dir()
        return get_plasticity_audit(data_dir)

    try:
        return await run_in_thread(_compute, timeout=_TIMEOUT, error_msg="Plasticity audit timed out")
    except Exception as exc:
        logger.warning("Plasticity failed: %s", exc)
        return {"status": "error", "message": str(exc)}


# ── Observations ──────────────────────────────────────────────────────────────

@router.get("/glass-brain/observations")
async def get_observations(limit: int = 30):
    """Recent thermal observations from the observation log."""
    def _read():
        data_dir = _data_dir()
        log_path = os.path.join(data_dir, "thermal_observations.jsonl")
        if not os.path.exists(log_path):
            return {"status": "no_data", "observations": []}

        lines = []
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as exc:
            return {"status": "error", "message": str(exc), "observations": []}

        observations = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                observations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(observations) >= limit:
                break

        return {"status": "ok", "observations": observations, "total": len(lines)}

    try:
        return await run_in_thread(_read, timeout=_TIMEOUT, error_msg="Observations timed out")
    except Exception as exc:
        logger.warning("Observations failed: %s", exc)
        return {"status": "error", "message": str(exc), "observations": []}


# ── Pruning log ───────────────────────────────────────────────────────────────

@router.get("/glass-brain/pruning-log")
async def get_pruning_log(limit: int = 50):
    """Recent synaptic pruning events."""
    def _read():
        data_dir = _data_dir()
        log_path = os.path.join(data_dir, "pruning_log.jsonl")
        if not os.path.exists(log_path):
            return {"status": "no_data", "entries": []}

        lines = []
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as exc:
            return {"status": "error", "message": str(exc), "entries": []}

        entries = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= limit:
                break

        return {"status": "ok", "entries": entries, "total": len(lines)}

    try:
        return await run_in_thread(_read, timeout=_TIMEOUT, error_msg="Pruning log timed out")
    except Exception as exc:
        logger.warning("Pruning log failed: %s", exc)
        return {"status": "error", "message": str(exc), "entries": []}


# ── Routing ───────────────────────────────────────────────────────────────────

@router.get("/glass-brain/routing")
async def get_routing():
    """Current maintenance routing decision: mode, hot/cold/neutral tags."""
    def _compute():
        from remy.core.thermal_advisor import get_maintenance_routing
        data_dir = _data_dir()
        routing = get_maintenance_routing(data_dir)
        return {
            "status": "ok",
            "mode": routing.mode,
            "cycle_number": routing.cycle_number,
            "thermal_available": routing.thermal_available,
            "hot_priority_tags": routing.hot_priority_tags[:20],
            "cold_skip_tags": routing.cold_skip_tags[:20],
            "neutral_tags": routing.neutral_tags[:20],
        }

    try:
        return await run_in_thread(_compute, timeout=_TIMEOUT, error_msg="Routing timed out")
    except Exception as exc:
        logger.warning("Routing failed: %s", exc)
        return {"status": "error", "message": str(exc)}


# ── Belief graph ──────────────────────────────────────────────────────────────

@router.get("/glass-brain/belief-graph")
async def get_belief_graph():
    """
    Full belief graph for visualization.
    nodes: [{id, key, temp, state, confidence, degree}]
    edges: [{source, target, conductivity, health}]
    health in {"healthy", "weakened", "pruned"}
    """
    def _build():
        from remy.core.thermal_advisor import (
            _load_beliefs_from_file, _build_graph, _initialize_temperature,
            _spread_heat, NUM_SPREAD_PASSES, COOLING_FACTOR,
        )
        from remy.core.synaptic_plasticity import load_edge_health, _edge_key

        data_dir = _data_dir()
        beliefs = _load_beliefs_from_file(data_dir)
        if not beliefs:
            return {"status": "no_data", "nodes": [], "edges": []}

        adj, raw_edges, degree = _build_graph(beliefs)
        temps = _initialize_temperature(beliefs)
        for _ in range(NUM_SPREAD_PASSES):
            _spread_heat(temps, adj)
        for n in temps:
            temps[n] *= COOLING_FACTOR

        health = load_edge_health(data_dir)

        nodes = []
        for bid, b in beliefs.items():
            nodes.append({
                "id": bid,
                "key": b.get("key", "")[:80],
                "temp": round(temps.get(bid, 0), 4),
                "state": b.get("state", ""),
                "confidence": round(b.get("confidence", 0), 3),
                "conflict_mass": round(b.get("conflict_mass", 0), 3),
                "degree": degree.get(bid, 0),
            })

        edges = []
        seen = set()
        for (a, b_id), cond in raw_edges.items():
            ek = _edge_key(a, b_id)
            if ek in seen:
                continue
            seen.add(ek)
            eh = health.get(ek)
            if eh and eh.pruned:
                state = "pruned"
            elif eh and eh.conductivity_modifier < 1.0:
                state = "weakened"
            else:
                state = "healthy"
            edges.append({
                "source": a,
                "target": b_id,
                "conductivity": round(cond, 3),
                "health": state,
            })

        return {
            "status": "ok",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    try:
        return await run_in_thread(_build, timeout=_TIMEOUT, error_msg="Belief graph timed out")
    except Exception as exc:
        logger.warning("Belief graph failed: %s", exc)
        return {"status": "error", "message": str(exc), "nodes": [], "edges": []}
