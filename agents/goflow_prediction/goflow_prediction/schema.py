"""The JSON-lines messages between the town and this agent (plan, section 4.2; GO-FLOW additions, section 9.3).

Town -> agent:  hello, obs, reset          Agent -> town:  ready, forecast, alert, error
Version 1. Times are SUMO seconds, fill and occupancy fractions, speed m/s. A message that does not pass `check_in`
is answered with an `error` message; it never stops the agent.
"""
from __future__ import annotations

import math

VERSION = 1
IN_TYPES = {"hello", "obs", "reset"}
NODE_FIELDS = {"veh", "halt", "speed", "occ", "green"}
GOFLOW_FIELDS = {"pred_occ", "traffic_state", "risk_level"}


def _num(x, nullable=True) -> bool:
    if x is None:
        return nullable
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_in(msg) -> list[str]:
    """Problems with a message from the town ([] = fine)."""
    if not isinstance(msg, dict):
        return ["not a JSON object"]
    p = []
    if msg.get("v") != VERSION:
        p.append(f"v must be {VERSION}")
    typ = msg.get("type")
    if typ not in IN_TYPES:
        return p + [f"unknown type {typ!r}"]
    if typ == "hello":
        if not isinstance(msg.get("nodes"), list):
            p.append("hello.nodes must be a list")
        if not _num(msg.get("dt_s"), nullable=False):
            p.append("hello.dt_s must be a number")
    if typ in ("obs", "reset") and not _num(msg.get("t"), nullable=False):
        p.append(f"{typ}.t must be a number (SUMO seconds)")
    if typ == "obs":
        nodes = msg.get("nodes")
        if not isinstance(nodes, dict):
            p.append("obs.nodes must be an object")
        else:
            for n, v in nodes.items():
                if not isinstance(v, dict) or set(v) - NODE_FIELDS:
                    p.append(f"obs.nodes[{n}] has unknown fields {sorted(set(v) - NODE_FIELDS) if isinstance(v, dict) else v}")
                    break
                bad = [k for k, x in v.items() if not _num(x)]
                if bad:
                    p.append(f"obs.nodes[{n}].{bad[0]} is not a finite number or null")
                    break
        if not _num(msg.get("demand")):
            p.append("obs.demand must be a number or null")
        da = msg.get("demand_at", {})
        if not isinstance(da, dict) or not all(_num(x) for x in da.values()):
            p.append("obs.demand_at must map horizons to numbers")
        gf = msg.get("goflow")
        if gf is not None:
            if not isinstance(gf, dict):
                p.append("obs.goflow must be an object or null")
            else:
                for n, v in gf.items():
                    if v is not None and (not isinstance(v, dict) or set(v) - GOFLOW_FIELDS
                                          or not _num(v.get("pred_occ"))):
                        p.append(f"obs.goflow[{n}] is malformed")
                        break
    return p


def error(msg: str, seq=None) -> dict:
    return {"type": "error", "v": VERSION, "seq": seq, "msg": msg}


def ready(model: str, horizons_s, warmup_steps: int, nodes: int) -> dict:
    return {"type": "ready", "v": VERSION, "model": model, "horizons_s": list(horizons_s),
            "warmup_steps": warmup_steps, "nodes": nodes}
