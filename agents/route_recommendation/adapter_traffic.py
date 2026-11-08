import json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
A_COEF = 2.0   # pénalité = 1 + A_COEF * fill (à calibrer plus tard)


def load_approach_map():
    """approche DSO4 (ex. '1427-1') -> edge_id SUMO, lu dans main (sans rien copier)."""
    raw = subprocess.run(
        ["git", "show", "origin/main:models/goflow/graph/nodes.json"],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO).stdout
    return {n["node"]: n["edge"] for n in json.loads(raw)}


def forecast_to_alerts(msg, approach_map, horizon_index=1):
    """Message 'forecast' de DSO4 -> alertes de congestion (horizon_index 1 = +120 s)."""
    alerts = []
    for node, v in msg["nodes"].items():
        edge = approach_map.get(node)
        fill = v["fill"][horizon_index]
        if edge and fill is not None and fill > 0.3:      # seuil à calibrer
            alerts.append({"agent": 2, "route": edge, "type": "congestion",
                           "penalty": 1 + A_COEF * min(fill, 1.0), "fill": fill})
    return alerts