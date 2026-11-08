"""The route recommendation agent in SUMO: a share of the town's cars routed by it, against the same share left on
their own routes, paired seeds.

    python tools/eval_route_guidance.py --seeds 11-20 --scales 2,3 --out out/route_guidance/dev
    python tools/eval_route_guidance.py --seeds 61-70 --scales 2,3 --out out/route_guidance/final   (test: once)

Each run is tools/eval_agents_forecast.py's (the forecasters' recipe: the viewer's town with crashes, buses and the
ring road, the clock at 30, 5 min warm-up from 06:00, then --minutes measured; the junction agents wired, under
--controller), both forecasters in this process, step by step, in every variant. The agent reads DSO4's forecasts
and the crashes once reported (cosim/route_guidance.py); nothing else of SUMO's.

Variants (cosim/route_guidance.FleetGuide), the same cars marked in each (a hash of the car's id and the seed):
  none       the marked cars keep the routes they were given (the town's own routes, duarouter's)
  shortest   the marked cars get the free-flow shortest route as they set off, nothing after
  agent      the marked cars get the agent's route as they set off (alerts in), and a new one whenever the alerts
             change, if cheaper by 5 s and 10 % (no flip-flopping); costs: free-flow times x penalties (as built)
  agent_typical  the same with the typical times from past trips (signals included, train seeds,
             tools/route_delay_table.py) as the costs instead of the free-flow times

Measured (SUMO tripinfo), over the trips that set off after the warm-up and at least --tail minutes before the end
(so they can finish):
  marked_delay   the marked cars' mean trip time minus the free-flow time of the shortest route between their start
                 and their end roads (s): what a trip lost against the best it could have done on an empty town
  marked_trip    their mean trip time (s), marked_done their share finished
  waiting, time_loss, throughput   every car but the emergency vehicles, as eval_agents_forecast.py
  ems            emergency vehicles on a run: mean time loss (s)
  long_queue     long-queue approach-minutes (eval_agents_forecast.long_queue_minutes)
Writes OUT/runs/<variant>_x<scale>_seed<seed>.json, OUT/summary.json and OUT/summary.md (paired differences against
the first variant, with the seeds where the other was better).
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
TEST_FROM = 61   # test seeds: 21-60 are used (docs/HANDOVER_8nov.md)


def run_one(variant: str, scale: float, seed: int, args: dict, port: int) -> dict:
    import eval_agents_forecast as eaf
    import goflow_prediction_dataset as gen

    from cosim.daily import GAME_SPEED
    from cosim.route_guidance import FleetGuide

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
         "clock_speed": GAME_SPEED, "dt": 30.0, "controller": args["controller"], "scale": scale, "accidents": True,
         "careless_share": None, "buses": True, "ring": True, "junction_blocker": 60.0,
         "sumo_extra": ["--tripinfo-output", str(trip), "--tripinfo-output.write-unfinished", "true"]}
    nodes = json.loads((models / "graph" / "nodes.json").read_text(encoding="utf-8"))
    guide = FleetGuide(variant, args["share"], seed)
    state: dict = {"ems": set(), "k": 0}

    def hook(sumo, im) -> None:
        pol = sumo.policy
        reads = pol is not None and "forecast" in getattr(pol, "on", ())
        fc = eaf.LocalForecast(sumo.conn, models, GAME_SPEED, scale, run=True)   # the agent reads DSO4 in every variant
        state.update(sumo=sumo, im=im, fc=fc, t0=sumo.time)
        if reads:
            pol.forecast = lambda: fc.risk(sumo.time)

    def probe(cur, conn, sod) -> None:
        state["fc"].on_step(cur, conn, sod)
        guide.step(cur, conn, state["fc"].latest, state["im"])
        state["k"] += 1
        if state["k"] % 10 == 0:
            sumo, im = state["sumo"], state["im"]
            ex = sumo.extras
            state["ems"].update(v for v in (ex.emergency if ex is not None else ()) if sumo.on_run(v))
            state["ems"].update(v for v in list(im.role) if im.on_run(v))

    t_wall = time.time()
    res = gen.one_run(seed, a, nodes, str(work), port, probe=probe, hook=hook)
    t_from = state["t0"] + 60.0 * args["warmup"]
    t_to = state["t0"] + 60.0 * (args["warmup"] + args["minutes"] - args["tail"])
    root = ET.parse(trip).getroot()
    trips = [x for x in root.findall("tripinfo") if t_from <= float(x.get("depart", -1)) <= t_to]
    ems = state["ems"]
    cars = [x for x in trips if x.get("id") not in ems and not x.get("id", "").startswith("pt_")]
    on_run = [x for x in trips if x.get("id") in ems]
    marked = [x for x in cars if x.get("id") in guide.flag]
    done = [x for x in marked if float(x.get("arrival", -1)) >= 0]
    from cosim.route_guidance import load_router

    r_, _ = load_router(typical_routing=False)   # the free-flow shortest route, whatever the variant's costs
    delays = []
    for x in done:
        o, d = x.get("departLane", "").rsplit("_", 1)[0], x.get("arrivalLane", "").rsplit("_", 1)[0]
        best = r_.best(o, d) if o in r_.roads and d in r_.roads else None
        if best is not None:
            delays.append(float(x.get("duration")) - r_.free_s(best, skip_first=False))

    def mean(xs, key):
        return round(st.mean(float(x.get(key)) for x in xs), 2) if xs else None

    # the agent's estimate of a trip as it set off, against the trip: free-flow, and with the typical waiting at
    # signals (models/route_guidance/movement_times.json, from the train seeds)
    err_f, err_t = [], []
    for x in done:
        e = guide.est.get(x.get("id"))
        if e is not None:
            real = float(x.get("duration"))
            err_f.append(real - e[0])
            err_t.append(real - e[1])
    eta = {"n": len(err_f),
           "free_mae": round(st.mean(abs(e) for e in err_f), 2) if err_f else None,
           "free_bias": round(st.mean(err_f), 2) if err_f else None,
           "typical_mae": round(st.mean(abs(e) for e in err_t), 2) if err_t else None,
           "typical_bias": round(st.mean(err_t), 2) if err_t else None,
           "typical_table": guide.router.typical and len(guide.router.typical)}

    pol = state["sumo"].policy
    r = {"variant": variant, "scale": scale, "seed": seed, "minutes": args["minutes"], "share": args["share"],
         "controller": args["controller"], "models": str(models),
         "trips": len(cars), "completed": sum(float(x.get("arrival", -1)) >= 0 for x in cars),
         "waiting": mean(cars, "waitingTime"), "time_loss": mean(cars, "timeLoss"),
         "throughput": round(sum(float(x.get("arrival", -1)) >= 0 for x in cars) * 60.0 / args["minutes"], 1),
         "marked": len(marked), "marked_done": round(len(done) / len(marked), 3) if marked else None,
         "marked_trip": mean(done, "duration"), "marked_delay": round(st.mean(delays), 2) if delays else None,
         "marked_waiting": mean(done, "waitingTime"), "eta": eta,
         "ems_n": len(on_run), "ems": mean(on_run, "timeLoss"),
         "t_from": t_from, "rows": state["fc"].rows, "forecasts": state["fc"].forecasts,
         "guide": guide.stats, "stuck_over_300s": res["stuck_over_300s"], "gridlock": res["gridlock"],
         "incidents": res["incidents"],
         "agent_stats": dict(pol.stats) if pol is not None and hasattr(pol, "stats") else None,
         "wall_s": round(time.time() - t_wall)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(r, indent=1), encoding="utf-8")
    trip.unlink(missing_ok=True)
    return r


METRICS = [("marked_delay", "marked cars: delay /trip (s)", -1), ("marked_trip", "marked cars: trip (s)", -1),
           ("marked_done", "marked cars: finished", 1), ("waiting", "all: waiting /trip (s)", -1),
           ("time_loss", "all: time loss /trip (s)", -1), ("throughput", "all: throughput /h", 1),
           ("ems", "ambulance time loss (s)", -1), ("long_queue", "long-queue approach-min", -1)]


def summarize(results: list[dict], variants: list[str], out: Path, models: Path) -> dict:
    import eval_agents_forecast as eaf

    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    card = json.loads((models / "dso4" / "congestion.json").read_text(encoding="utf-8"))
    for r in results:
        r["long_queue"] = eaf.long_queue_minutes(r["rows"], nodes, card, r["t_from"])
    by = {(r["variant"], r["scale"], r["seed"]): r for r in results}
    base = variants[0]
    rows = []
    for scale in sorted({r["scale"] for r in results}):
        seeds = sorted({r["seed"] for r in results if r["scale"] == scale})
        for v in variants:
            row = {"variant": v, "scale": scale, "seeds": seeds,
                   "reroutes": sum((by.get((v, scale, s)) or {}).get("guide", {}).get("reroutes", 0) for s in seeds),
                   "changed_at_start": sum((by.get((v, scale, s)) or {}).get("guide", {}).get("changed_at_start", 0)
                                           for s in seeds)}
            for key, _, sign in METRICS:
                xs = [by[(v, scale, s)][key] for s in seeds if by.get((v, scale, s)) and by[(v, scale, s)][key] is not None]
                row[key] = round(st.mean(xs), 3) if xs else None
                if v != base:
                    d = [by[(v, scale, s)][key] - by[(base, scale, s)][key] for s in seeds
                         if by.get((v, scale, s)) and by.get((base, scale, s))
                         and by[(v, scale, s)][key] is not None and by[(base, scale, s)][key] is not None]
                    row[f"d_{key}"] = round(st.mean(d), 3) if d else None
                    row[f"better_{key}"] = sum(1 for x in d if x * sign > 0)
                    row[f"n_{key}"] = len(d)
            rows.append(row)
    md = [f"Paired against `{base}`: mean, then (mean difference, seeds where the variant was better).", "",
          "| traffic | variant | routes changed at start / later | " + " | ".join(m[1] for m in METRICS) + " |",
          "|---|---|---|" + "---|" * len(METRICS)]
    for row in rows:
        cells = []
        for key, _, _ in METRICS:
            if row["variant"] == base or row.get(f"d_{key}") is None:
                cells.append(f"{row[key]}")
            else:
                cells.append(f"{row[key]} (Δ {row[f'd_{key}']:+}, {row[f'better_{key}']}/{row[f'n_{key}']})")
        md.append(f"| x{row['scale']:g} | {row['variant']} | {row['changed_at_start']} / {row['reroutes']} | "
                  + " | ".join(cells) + " |")
    md += ["", "The agent's estimate of a marked trip as it set off, against the real trip (mean absolute error / mean "
           "real minus estimate, s; trips pooled over the seeds' means):", "",
           "| traffic | variant | trips | free-flow: error / bias | with typical waiting: error / bias |",
           "|---|---|---|---|---|"]
    for scale in sorted({r["scale"] for r in results}):
        for v in variants:
            rs = [r for r in results if r["scale"] == scale and r["variant"] == v and (r.get("eta") or {}).get("n")]
            if not rs:
                continue

            def m(key, rs=rs):
                xs = [r["eta"][key] for r in rs if r["eta"].get(key) is not None]
                return round(st.mean(xs), 1) if xs else None
            md.append(f"| x{scale:g} | {v} | {sum(r['eta']['n'] for r in rs)} | {m('free_mae')} / {m('free_bias')} | "
                      f"{m('typical_mae')} / {m('typical_bias')} |")
    summary = {"variants": variants, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return summary


def main(argv=None) -> int:
    from eval_agents_forecast import seeds_arg

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--scales", default="2,3")
    ap.add_argument("--variants", default="none,shortest,agent")
    ap.add_argument("--share", type=float, default=0.3, help="share of the town's cars the agent routes")
    ap.add_argument("--controller", default="agents_observed")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--tail", type=float, default=8.0, help="trips set off this many minutes before the end at the latest")
    ap.add_argument("--start-hour", type=float, default=6.0)
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--port", type=int, default=9800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reason", default="", help="required for test seeds (61 and above): why they are opened now")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="only the tables, from the runs written")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds_arg(a.seeds)
    scales = [float(x) for x in a.scales.split(",")]
    variants = a.variants.split(",")
    args = {"out": str(out), "models": a.models, "minutes": a.minutes, "warmup": a.warmup, "tail": a.tail,
            "start_hour": a.start_hour, "share": a.share, "controller": a.controller, "skip_done": a.skip_done}
    if not a.summary_only:
        test = [s for s in seeds if s >= TEST_FROM]
        if test:
            lock = out / "TEST_OPENED.lock"
            if lock.is_file():
                print(f"test seeds already opened here: {lock.read_text(encoding='utf-8')}")
                return 2
            if not a.reason:
                print("test seeds need --reason")
                return 2
            lock.write_text(json.dumps({"seeds": test, "reason": a.reason, "when": time.strftime("%Y-%m-%d %H:%M"),
                                        "args": vars(a)}), encoding="utf-8")
        jobs = [(v, sc, s) for sc in scales for s in seeds for v in variants]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(run_one, v, sc, s, args, a.port + 3 * i) for i, (v, sc, s) in enumerate(jobs)]
            for f, (v, sc, s) in zip(futs, jobs, strict=True):
                r = f.result()
                print(f"{v} x{sc:g} seed {s}: marked delay {r['marked_delay']} s, waiting {r['waiting']} s, "
                      f"guide {r['guide']}, {r['wall_s']} s", flush=True)
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*.json"))]
    results = [r for r in results if r["variant"] in variants]
    summarize(results, variants, out, Path(a.models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
