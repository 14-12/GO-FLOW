"""The junction agents with and without the GO-FLOW forecasters' warnings, SUMO alone, paired seeds.

    python tools/eval_agents_forecast.py --seeds 11-20 --scales 2,3 --out out/agents_forecast/dev
    python tools/eval_agents_forecast.py --seeds 51-60 --scales 2,3 --out out/agents_forecast/final \\
        --variants agents_observed,agents_observed+forecast@forecast_w=0.5

Each run is the forecasters' own training recipe (tools/goflow_prediction_dataset.one_run: the viewer's town with
crashes, buses and the ring road, the clock at 30, 5 min warm-up from 06:00, then --minutes measured), the junction
agents wired as the viewer wires them (cameras, emergency requests). Both forecasters run in this process, step by
step, with the live agents' own code (goflow_traffic.agent.GoFlowTown05, then goflow_prediction.agent.ForecastAgent
with GO-FLOW's answer in the message), in every variant, so the two are measured alike; only "+forecast" variants read
their warnings (cosim/agents_actuated.py, `forecast`). In process rather than as the viewer's child processes: the
answer to a step is then always there before the next, and two runs of a seed differ only by what the agents did.

Measured over the trips that depart after the warm-up (SUMO tripinfo, unfinished ones included):
  waiting      mean waiting time per trip (s), every vehicle but the emergency vehicles
  time_loss    mean time loss per trip (s), the same vehicles
  throughput   those trips completed per hour
  ems          emergency vehicles on a run: mean time loss (s)
  long_queue   minutes x approaches in a long queue: the approach's fill averaged over a signal cycle at or over its
               level (the --models classifier's levels, dso4/congestion.json, the same for every variant), from the
               same 30 s samples the forecasters get (kept in each run's file, counted at the summary)
Writes OUT/runs/<variant>_x<scale>_seed<seed>.json, OUT/summary.json and OUT/summary.md (paired differences against
the first variant, with the seeds where the other was better). Runs already written are kept (--skip-done).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "agents" / "goflow_prediction"),
                str(ROOT / "agents" / "goflow_traffic")]

MODELS = ROOT / "models" / "goflow"


def seeds_arg(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


class LocalForecast:
    """Both forecasters in this process, fed as cosim/goflow_prediction.GoflowPrediction feeds its children.
    run=False: only the 30 s samples (the variants that do not read the forecasts need no model)."""

    def __init__(self, conn, models: Path, clock_speed: float, scale: float, run: bool = True):
        from cosim.goflow_prediction import Aggregator

        self.nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
        self.agg = Aggregator(conn, self.nodes, 30.0)
        self.gf = self.dso = None
        if run:
            from goflow_prediction.agent import ForecastAgent, ModelBundle
            from goflow_traffic.agent import GoFlowTown05

            self.gf = GoFlowTown05(str(models / "goflow_xgb"))
            self.dso = ForecastAgent([ModelBundle(str(models / "dso4")), ModelBundle(str(models / "dso4_goflow"))])
        self.clock_speed, self.scale = clock_speed, scale
        self.seq = 0
        self.latest = None
        self.rows: list[dict] = []      # (t, halt per approach): the long-queue measure
        self.forecasts = 0

    def on_step(self, cur, conn, sod: float) -> None:
        from cosim.goflow_prediction import obs_message

        row = self.agg.sample(conn, cur.time, cur.tls)
        if row is None:
            return
        msg = obs_message(row, self.nodes, self.seq, sod, self.clock_speed, self.scale)
        self.seq += 1
        self.rows.append({"t": row["t"], "halt": [round(x, 3) for x in row["halt"]]})
        if self.dso is None:
            return
        msg = json.loads(json.dumps(msg))                 # as it would travel to a child
        res, _ = self.gf.step(msg)
        msg = dict(msg, goflow=json.loads(json.dumps(res))["nodes"] if res is not None else None)
        out = self.dso.step(msg)
        if out["forecast"] is not None:
            self.latest = json.loads(json.dumps(out["forecast"]))
            self.forecasts += 1

    def risk(self, now: float) -> dict[str, float]:
        from cosim.goflow_prediction import long_queue_risk

        return long_queue_risk(self.latest, now)


def long_queue_minutes(rows: list[dict], nodes: list[dict], card: dict, t_from: float) -> float:
    """Minutes x approaches with the cycle-mean fill at or over the approach's level, from t_from on."""
    import numpy as np

    k = int(card["cycle_steps"])
    level = np.array([card["level"][n["node"]] for n in nodes])
    st_ = np.array([n.get("storage") or 0.0 for n in nodes])
    st_ = np.where(st_ >= 1.0, st_, np.nan)
    halt = np.array([r["halt"] for r in rows], dtype=float)
    t = np.array([r["t"] for r in rows])
    fill = np.nan_to_num(halt / st_)
    total = 0
    for i in range(k - 1, len(rows)):
        if t[i] < t_from:
            continue
        total += int((fill[i - k + 1:i + 1].mean(0) >= level).sum())
    return round(total * float(card["dt_s"]) / 60.0, 1)


def run_one(variant: str, scale: float, seed: int, args: dict, port: int) -> dict:
    import goflow_prediction_dataset as gen

    from cosim.daily import GAME_SPEED

    out = Path(args["out"])
    tag = f"{variant}_x{scale:g}_seed{seed}"
    dst = out / "runs" / f"{tag}.json"
    if args["skip_done"] and dst.is_file():
        return json.loads(dst.read_text(encoding="utf-8"))
    work = out / "work" / tag
    (work / "runs").mkdir(parents=True, exist_ok=True)
    trip = work / "tripinfo.xml"
    models = Path(args["models"])
    a = {"minutes": args["minutes"], "warmup": args["warmup"], "start_hour": args["start_hour"],
         "clock_speed": GAME_SPEED, "dt": 30.0, "controller": variant, "scale": scale, "accidents": True,
         "careless_share": None, "buses": True, "ring": True, "junction_blocker": 60.0, "trips": "purpose",
         "sumo_extra": ["--tripinfo-output", str(trip), "--tripinfo-output.write-unfinished", "true"]}
    nodes = json.loads((models / "graph" / "nodes.json").read_text(encoding="utf-8"))
    state: dict = {"ems": set(), "k": 0}

    def hook(sumo, im) -> None:
        pol = sumo.policy
        reads = pol is not None and bool({"forecast", "gate"} & set(getattr(pol, "on", ())))
        fc = LocalForecast(sumo.conn, models, GAME_SPEED, scale, run=reads)
        state.update(sumo=sumo, im=im, fc=fc, t0=sumo.time)
        if reads:
            pol.forecast = lambda: fc.risk(sumo.time)
        if pol is not None and "accgate" in getattr(pol, "on", ()):   # the roads the drivers' navigation counts as slow
            pol.incident_roads = lambda: {e for inc in im.incidents for e in getattr(inc, "diverted", ())}

    def probe(cur, conn, sod) -> None:
        state["fc"].on_step(cur, conn, sod)
        state["k"] += 1
        if state["k"] % 10 == 0:
            sumo, im = state["sumo"], state["im"]
            ex = sumo.extras
            state["ems"].update(v for v in (ex.emergency if ex is not None else ()) if sumo.on_run(v))
            state["ems"].update(v for v in list(im.role) if im.on_run(v))

    t_wall = time.time()
    res = gen.one_run(seed, a, nodes, str(work), port, probe=probe, hook=hook)
    t_from = state["t0"] + 60.0 * args["warmup"]
    root = ET.parse(trip).getroot()
    trips = [x for x in root.findall("tripinfo") if float(x.get("depart", -1)) >= t_from]
    ems = state["ems"]
    cars = [x for x in trips if x.get("id") not in ems and not x.get("id", "").startswith("pt_")]
    on_run = [x for x in trips if x.get("id") in ems]

    def mean(xs, key):
        return round(st.mean(float(x.get(key)) for x in xs), 2) if xs else None

    pol = state["sumo"].policy
    r = {"variant": variant, "scale": scale, "seed": seed, "minutes": args["minutes"], "models": str(models),
         "trips": len(cars), "completed": sum(float(x.get("arrival", -1)) >= 0 for x in cars),
         "waiting": mean(cars, "waitingTime"), "time_loss": mean(cars, "timeLoss"),
         "throughput": round(sum(float(x.get("arrival", -1)) >= 0 for x in cars) * 60.0 / args["minutes"], 1),
         "ems_n": len(on_run), "ems": mean(on_run, "timeLoss"),
         "t_from": t_from, "rows": state["fc"].rows,      # the 30 s samples: long_queue at the summary
         "forecasts": state["fc"].forecasts, "stuck_over_300s": res["stuck_over_300s"], "gridlock": res["gridlock"],
         "incidents": res["incidents"], "removed": res.get("removed"), "agent_stats": dict(pol.stats) if pol is not None and hasattr(pol, "stats") else None,
         "wall_s": round(time.time() - t_wall)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(r, indent=1), encoding="utf-8")
    trip.unlink(missing_ok=True)
    return r


METRICS = [("waiting", "waiting /trip (s)", -1), ("time_loss", "time loss /trip (s)", -1),
           ("throughput", "throughput /h", 1), ("ems", "ambulance time loss (s)", -1),
           ("long_queue", "long-queue approach-minutes", -1)]


def summarize(results: list[dict], variants: list[str], out: Path, models: Path) -> dict:
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    card = json.loads((models / "dso4" / "congestion.json").read_text(encoding="utf-8"))
    for r in results:   # the same levels for every variant: those of the models evaluated
        r["long_queue"] = long_queue_minutes(r["rows"], nodes, card, r["t_from"])
    by = {(r["variant"], r["scale"], r["seed"]): r for r in results}
    base = variants[0]
    rows = []
    for scale in sorted({r["scale"] for r in results}):
        seeds = sorted({r["seed"] for r in results if r["scale"] == scale})
        for v in variants:
            row = {"variant": v, "scale": scale, "seeds": seeds}
            for key, _, sign in METRICS:
                xs = [by[(v, scale, s)][key] for s in seeds if by.get((v, scale, s)) and by[(v, scale, s)][key] is not None]
                row[key] = round(st.mean(xs), 2) if xs else None
                if v != base:
                    d = [by[(v, scale, s)][key] - by[(base, scale, s)][key] for s in seeds
                         if by.get((v, scale, s)) and by.get((base, scale, s))
                         and by[(v, scale, s)][key] is not None and by[(base, scale, s)][key] is not None]
                    row[f"d_{key}"] = round(st.mean(d), 2) if d else None
                    row[f"better_{key}"] = sum(1 for x in d if x * sign > 0)
                    row[f"n_{key}"] = len(d)
            rows.append(row)
    md = [f"Paired against `{base}`: mean difference, then the seeds where the variant was better.", "",
          "| traffic | variant | " + " | ".join(m[1] for m in METRICS) + " |",
          "|---|---|" + "---|" * len(METRICS)]
    for row in rows:
        cells = []
        for key, _, _ in METRICS:
            if row["variant"] == base or row.get(f"d_{key}") is None:
                cells.append(f"{row[key]}")
            else:
                cells.append(f"{row[key]} (Δ {row[f'd_{key}']:+}, {row[f'better_{key}']}/{row[f'n_{key}']})")
        md.append(f"| x{row['scale']:g} | {row['variant']} | " + " | ".join(cells) + " |")
    summary = {"variants": variants, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--scales", default="2,3")
    ap.add_argument("--variants", default="agents_observed,agents_observed+forecast",
                    help="controller names (cosim/agents_actuated.make); the first is the reference")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--start-hour", type=float, default=6.0)
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--port", type=int, default=9700)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="only the tables, from the runs written")
    a = ap.parse_args(argv)
    variants = a.variants.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = {"out": str(out), "models": a.models, "minutes": a.minutes, "warmup": a.warmup,
            "start_hour": a.start_hour, "skip_done": a.skip_done}
    jobs = [(v, float(sc), sd) for sc in a.scales.split(",") for sd in seeds_arg(a.seeds) for v in variants]
    if a.summary_only:
        runs = [out / "runs" / f"{v}_x{sc:g}_seed{sd}.json" for v, sc, sd in jobs]
        summarize([json.loads(p.read_text(encoding="utf-8")) for p in runs if p.is_file()], variants, out,
                  Path(a.models))
        return 0
    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, v, sc, sd, args, a.port + 3 * i) for i, (v, sc, sd) in enumerate(jobs)]
        for f in futs:
            r = f.result()
            results.append(r)
            print(json.dumps({k: r[k] for k in ("variant", "scale", "seed", "waiting", "time_loss", "throughput",
                                                 "ems", "forecasts", "wall_s")}), flush=True)
    summarize(results, variants, out, Path(a.models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
