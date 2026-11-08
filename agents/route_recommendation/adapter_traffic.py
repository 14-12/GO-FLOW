"""The other agents' messages, turned into this agent's alerts: {"agent", "route" (a SUMO edge id), "type", "penalty"}.

A penalty multiplies the road's free-flow time in the route search (router.py). The coefficients are not calibrated
yet: they rank routes, they do not predict a time.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NODES = REPO / "models" / "goflow" / "graph" / "nodes.json"
A_COEF = 2.0        # congestion: penalty = 1 + A_COEF * fill (to calibrate)
FILL_MIN = 0.3      # congestion: below this forecast fill, no alert (to calibrate)


def load_approach_map(path=NODES):
    """DSO4 approach (e.g. '1427-1') -> SUMO edge id, from the shipped graph (models/goflow/graph/nodes.json)
    in this working copy (8 Nov: it was read from `git show origin/main:...`, the remote's copy)."""
    return {n["node"]: n["edge"] for n in json.loads(Path(path).read_text(encoding="utf-8"))}


def forecast_to_alerts(msg, approach_map, horizon_index=1):
    """DSO4 'forecast' message -> congestion alerts (horizon_index 1 = +120 s)."""
    alerts = []
    for node, v in (msg.get("nodes") or {}).items():
        edge = approach_map.get(node)
        fills = (v or {}).get("fill") or []
        fill = fills[horizon_index] if len(fills) > horizon_index else None
        if edge and fill is not None and fill > FILL_MIN:
            alerts.append({"agent": 2, "route": edge, "type": "congestion", "node": node,
                           "penalty": 1 + A_COEF * min(fill, 1.0), "fill": fill})
    return alerts


def accident_to_alert(msg):
    """Accident message {edge, closed_lanes, total_lanes, severity} -> alert with its penalty."""
    closed, total = msg.get("closed_lanes", 0), max(msg.get("total_lanes", 1), 1)
    if closed >= total:
        penalty = 50.0                       # road as good as blocked
    else:
        penalty = 1 + 4 * closed / total     # slowed in proportion
    return {"agent": 1, "type": "accident", "route": msg["edge"], "penalty": penalty,
            "severity": msg.get("severity", "unknown"), "closed_lanes": closed, "total_lanes": total}
