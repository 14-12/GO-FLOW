"""The DSO4 agent on Town05: perceive, self-evaluate, decide, act, once per observation (every dt_s SUMO seconds).

As in DSO4's agent/core.py, but online and on seconds:
1. perceive      : the observation joins a ring buffer (12 steps of input, plus what the baselines need);
2. self_evaluate : every candidate's absolute error on forecasts whose target has now been observed, over the last
                   `eval_window` matured forecasts per horizon; the best candidate per horizon is used;
3. decide        : forecast fill per approach and horizon;
4. act           : an alert for an approach still under `low` now and forecast at `full` or more:
                   `propagation` when a downstream approach is already full (the queue spills back), else
                   `spillback_soon`; cooldown per approach. With GO-FLOW on, an alert also needs her state to be
                   congested or severe (the PEMS08 confirmation rule); without her answer it goes out unconfirmed.

`replay()` runs a dataset through exactly this code, so offline evaluation and live use cannot differ.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np

from .config import AlertRules, TimeBase
from .features import Window, build, neighbor_mats

CONGESTED_STATES = {"congested", "severe"}


class ModelBundle:
    """A trained model folder (train_lgbm.py output)."""

    def __init__(self, folder):
        import lightgbm as lgb

        with open(os.path.join(folder, "model_card.json"), encoding="utf-8") as f:
            self.card = json.load(f)
        tb = self.card["timebase"]
        self.tb = TimeBase(dt_s=float(tb["dt_s"]), n_in=int(tb["n_in"]), horizons_s=tuple(tb["horizons_s"]))
        with open(os.path.join(folder, "nodes.json"), encoding="utf-8") as f:
            self.nodes = json.load(f)
        self.adj = np.load(os.path.join(folder, "adjacency.npy"))
        self.storage = np.load(os.path.join(folder, "storage.npy"))
        self.use_node_id = bool(self.card.get("use_node_id", True))
        self.use_goflow = bool(self.card.get("use_goflow", False))
        self.models = {h: lgb.Booster(model_file=os.path.join(folder, f"lgbm_{h}s.txt")) for h in self.tb.steps}
        self.version = self.card["version"]
        self.name = "lgbm_gf" if self.use_goflow else "lgbm"


class ForecastAgent:
    def __init__(self, models: list[ModelBundle], rules: AlertRules | None = None):
        if not models:
            raise ValueError("at least one model")
        m0 = models[0]
        for m in models[1:]:
            if m.nodes != m0.nodes or m.tb != m0.tb:
                raise ValueError("all models must share the nodes and the time base")
        self.bundles = {m.name: m for m in models}
        self.tb, self.nodes, self.storage = m0.tb, list(m0.nodes), m0.storage
        self.N = len(self.nodes)
        self.col = {n: j for j, n in enumerate(self.nodes)}
        self.A_up, self.A_down = neighbor_mats(m0.adj)
        self.down = [np.flatnonzero(m0.adj[k] > 0) for k in range(self.N)]   # k -> j: traffic from k enters j
        self.down = [d[d != k] for k, d in enumerate(self.down)]
        self.rules = rules or AlertRules()
        self.keep = max(self.tb.n_in, self.tb.cycle_steps + 1)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.buf = deque(maxlen=self.keep)          # dicts: t, fill [N], veh, speed, occ, green, demand, demand_at
        self.step_i = 0                              # observations since the last reset
        self.pending: deque = deque()                # (due step index, horizon, {candidate: pred [N]})
        self.errors = {h: {} for h in self.tb.steps}  # horizon -> candidate -> deque of MAE over nodes
        self.last_alert = np.full(self.N, -math.inf)
        self.last_t = None

    @property
    def candidates(self) -> list[str]:
        return list(self.bundles) + ["last_value", "cycle_persistence"]

    # ------------------------------------------------------------------ 1. perceive
    def perceive(self, obs: dict) -> list[str]:
        """Adds one observation; returns problems found in it (missing approaches), not fatal."""
        t = float(obs["t"])
        if self.last_t is not None and abs(t - self.last_t - self.tb.dt_s) > 1e-6:
            self.reset()                              # a gap or a jump: the window would mix two moments
        problems = []
        arr = {c: np.full(self.N, np.nan) for c in ("halt", "veh", "speed", "occ", "green")}
        nodes = obs.get("nodes") or {}
        for n, v in nodes.items():
            j = self.col.get(n)
            if j is None:
                problems.append(f"unknown node {n}")
                continue
            for c in arr:
                x = v.get(c)
                arr[c][j] = np.nan if x is None else float(x)
        missing = [n for n in self.nodes if n not in nodes]
        if missing:
            problems.append(f"{len(missing)} nodes missing from obs, e.g. {missing[0]}")
        st = np.where(self.storage >= 1.0, self.storage, np.nan)
        da = obs.get("demand_at") or {}
        gf = obs.get("goflow")
        g, states = None, None
        if gf:
            g = np.full(self.N, np.nan)
            states = [None] * self.N
            for n, v in gf.items():
                j = self.col.get(n)
                if j is not None and v is not None:
                    if v.get("pred_occ") is not None:
                        g[j] = float(v["pred_occ"])
                    states[j] = v.get("traffic_state")
        self.buf.append(dict(t=t, fill=arr["halt"] / st, veh=arr["veh"], speed=arr["speed"], occ=arr["occ"],
                             green=arr["green"], demand=float(obs.get("demand", np.nan)),
                             demand_at=np.array([float(da.get(h, np.nan)) for h in self.tb.steps]),
                             goflow=g, goflow_state=states))
        self.last_t = t
        self.step_i += 1
        return problems

    # ------------------------------------------------------------------ 2. self-evaluate
    def self_evaluate(self) -> dict[str, str]:
        now = self.buf[-1]["fill"]
        while self.pending and self.pending[0][0] <= self.step_i:
            due, h, preds = self.pending.popleft()
            if due != self.step_i:
                continue
            for name, p in preds.items():
                m = np.isfinite(p) & np.isfinite(now)
                if m.any():
                    q = self.errors[h].setdefault(name, deque(maxlen=self.rules.eval_window))
                    q.append(float(np.abs(p[m] - now[m]).mean()))
        chosen = {}
        avail = self._available()
        models = [m for m in ("lgbm_gf", "lgbm") if m in avail]
        default = models[0] if models else "last_value"
        for h in self.tb.steps:
            scores = {c: np.mean(q) for c, q in self.errors[h].items() if len(q)}
            usable = {c: s for c, s in scores.items() if c in avail}
            chosen[h] = min(usable, key=usable.get) if usable else default
        return chosen

    def _available(self) -> set[str]:
        out = {"last_value", "cycle_persistence"}
        has_gf = self.buf and self.buf[-1]["goflow"] is not None
        for name, b in self.bundles.items():
            if not b.use_goflow or has_gf:
                out.add(name)
        return out

    # ------------------------------------------------------------------ 3. decide
    def window(self) -> Window:
        b = list(self.buf)[-self.tb.n_in:]
        stack = lambda k: np.stack([x[k] for x in b])[None]  # noqa: E731
        last = b[-1]
        return Window(fill=stack("fill"), veh=stack("veh"), speed=stack("speed"), occ=stack("occ"),
                      green=stack("green"), demand=np.array([last["demand"]]), demand_at=last["demand_at"][None],
                      goflow=None if last["goflow"] is None else last["goflow"][None])

    def forecast_all(self) -> dict[str, dict[str, np.ndarray]]:
        """horizon -> candidate -> fill [N]."""
        w = self.window()
        fills = np.stack([x["fill"] for x in self.buf])          # [L, N], oldest first
        out = {}
        for hi, (h, k) in enumerate(self.tb.steps.items()):
            res = {"last_value": fills[-1]}
            c = self.tb.cycle_steps
            back = c - k
            while back < 0:
                back += c
            res["cycle_persistence"] = fills[-1 - back] if back < len(fills) else fills[-1]
            for name, bnd in self.bundles.items():
                if bnd.use_goflow and w.goflow is None:
                    continue
                F, _ = build(w, hi, self.storage, self.A_up, self.A_down, bnd.use_node_id, bnd.use_goflow)
                res[name] = np.clip(bnd.models[h].predict(F), 0.0, None)
            out[h] = res
        return out

    # ------------------------------------------------------------------ 4. act
    def act(self, t: float, pred: dict[str, np.ndarray], chosen: dict[str, str]) -> list[dict]:
        r = self.rules
        now = self.buf[-1]["fill"]
        states = self.buf[-1]["goflow_state"]
        full_now = np.nan_to_num(now, nan=-1) >= r.full
        alerts = []
        for h in self.tb.steps:
            p = pred[h]
            cand = (np.nan_to_num(p, nan=-1) >= r.full) & (np.nan_to_num(now, nan=9) < r.low)
            for j in np.flatnonzero(cand):
                if t - self.last_alert[j] < r.cooldown_s:
                    continue
                confirmed = None
                if states is not None and any(b.use_goflow for b in self.bundles.values()):
                    confirmed = states[j] in CONGESTED_STATES
                    if not confirmed:
                        continue
                src = [d for d in self.down[j] if full_now[d]]
                kind = "propagation" if src else "spillback_soon"
                node = self.nodes[j]
                a = {"type": "alert", "v": 1, "t": t, "kind": kind, "node": node, "tls": node.split("-")[0],
                     "horizon_s": int(h), "fill_now": round(float(now[j]), 4), "pred_fill": round(float(p[j]), 4),
                     "source": self.nodes[src[0]] if src else None, "model": chosen[h],
                     "action": _action(kind, node, int(h), self.nodes[src[0]] if src else None)}
                if states is not None:
                    g = self.buf[-1]["goflow"]
                    a["goflow_pred_occ"] = None if g is None or not np.isfinite(g[j]) else round(float(g[j]), 5)
                    a["confirmed_by_goflow"] = confirmed
                alerts.append(a)
                self.last_alert[j] = t
        return alerts

    # ------------------------------------------------------------------ one step
    def step(self, obs: dict) -> dict:
        """-> {"problems", "forecast" (message or None while warming up), "alerts"}."""
        problems = self.perceive(obs)
        chosen = self.self_evaluate()
        if len(self.buf) < self.tb.n_in:
            return {"problems": problems, "forecast": None, "alerts": [], "warming_up": True}
        allp = self.forecast_all()
        for h, k in self.tb.steps.items():
            self.pending.append((self.step_i + k, h, allp[h]))
        self.pending = deque(sorted(self.pending, key=lambda x: x[0]))
        pred = {h: allp[h].get(chosen[h], allp[h]["last_value"]) for h in self.tb.steps}
        t = float(obs["t"])
        alerts = self.act(t, pred, chosen)
        hs = list(self.tb.steps)
        msg = {"type": "forecast", "v": 1, "seq": obs.get("seq"), "t": t,
               "model": "+".join(b.version for b in self.bundles.values()), "chosen": chosen,
               "nodes": {n: {"fill": [_r(pred[h][j]) for h in hs],
                             "halt": [_r(pred[h][j] * self.storage[j]) for h in hs]}
                         for j, n in enumerate(self.nodes)}}
        return {"problems": problems, "forecast": msg, "alerts": alerts, "warming_up": False}


def _r(x) -> float | None:
    return None if not np.isfinite(x) else round(float(x), 4)


def _action(kind: str, node: str, horizon_s: int, source: str | None) -> str:
    if kind == "propagation":
        return (f"Queue on {source} is spilling back: hold the green longer on {node} before its queue reaches the "
                f"upstream junction (within {horizon_s} s)")
    return f"Hold the green longer on {node} before its queue reaches the upstream junction (within {horizon_s} s)"


def obs_from_dataset(ds, i: int, seq: int, goflow_states=None) -> dict:
    """One dataset row as the town would send it (the same message the live adapter writes)."""
    tb = ds.timebase()
    nodes = {}
    for j, n in enumerate(ds.nodes):
        nodes[n] = {c: (None if not np.isfinite(ds.x[c][i, j]) else float(ds.x[c][i, j]))
                    for c in ("veh", "halt", "speed", "occ", "green")}
    msg = {"type": "obs", "v": 1, "seq": seq, "t": float(ds.t[i]), "tod_s": float(ds.tod_s[i]),
           "demand": float(ds.demand[i]),
           "demand_at": {h: float(ds.demand_at[i, k]) for k, h in enumerate(tb.steps)}, "nodes": nodes}
    if ds.goflow is not None:
        msg["goflow"] = {n: {"pred_occ": None if not np.isfinite(ds.goflow[i, j]) else float(ds.goflow[i, j]),
                             "traffic_state": _state(goflow_states, ds, i, j)}
                         for j, n in enumerate(ds.nodes)}
    return msg


def _state(override, ds, i, j):
    src = override if override is not None else ds.goflow_state
    if src is None:
        return None
    v = src[i][j]
    return str(v) if v else None


def replay(agent: ForecastAgent, ds, seeds, goflow_states=None):
    """Runs the agent over the runs of `seeds`, one observation at a time. -> (alerts, forecasts by row index)."""
    alerts, forecasts = [], {}
    rows = np.flatnonzero(np.isin(ds.seed, list(seeds)))
    seq, prev_run = 0, None
    for i in rows:
        if ds.run[i] != prev_run:
            agent.reset()
            prev_run = ds.run[i]
        out = agent.step(obs_from_dataset(ds, i, seq, goflow_states))
        seq += 1
        alerts += out["alerts"]
        if out["forecast"] is not None:
            forecasts[int(i)] = out["forecast"]
    return alerts, forecasts
