"""How close every car's trip estimate (cosim/trip_estimates.py, made when it sets off) comes to its real time, on the
viewer's town in SUMO alone (tools/goflow_prediction_dataset.one_run: trips between places, crashes, buses, the ring,
the junction agents on the cameras). Dev seeds by default; train seeds are where the typical times come from.

    python tools/eval_trip_estimates.py --seeds 11-20 --start-hours 7,12,17 --out out/trip_estimates/dev

Per trip: estimate, real time, how it ended, whether the town gave it another way. Trips started in the warm-up, still
on their way at the end, given another destination, or taken off are left out of the accuracy (reported apart).
Accuracy: median absolute error, share within 20 % and within 60 s of the real time, overall, by kind of trip
(to a place / through traffic), by length and by town hour.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def run_one(seed: int, start: float, args: dict, port: int) -> dict:
    import goflow_prediction_dataset as gen
    import sumolib

    from cosim.daily import GAME_SPEED
    from cosim.highway import HIGHWAY_SPEED, find_loops
    from cosim.trip_estimates import TripEstimates, area_names, plan_names

    work = Path(args["out"]) / f"h{start:g}_seed{seed}"
    done = work / "legs.json"
    if done.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    (work / "runs").mkdir(parents=True, exist_ok=True)
    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    ring = {e: HIGHWAY_SPEED for lp in find_loops(net) for e in lp}
    a = {"minutes": args["minutes"], "warmup": args["warmup"], "start_hour": start, "clock_speed": GAME_SPEED,
         "dt": 30.0, "controller": args["controller"], "scale": args["scale"], "accidents": True,
         "careless_share": None, "buses": True, "ring": True, "ring_trips": args["ring_trips"], "junction_blocker": 0.0,
         "trips": "purpose"}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    st_ = {}

    def hook(sumo, im) -> None:
        pop = getattr(sumo, "population", None)   # (one_run hands it over)
        names = area_names(pop.demand.places) if pop is not None and getattr(pop, "demand", None) else {}
        te = st_["te"] = TripEstimates(net, typical_file=Path(args["table"]), speeds=ring, area_names=names,
                                       stat=args["stat"],
                                       place_name=plan_names(pop.demand) if pop is not None and pop.demand else None)
        te.setup(sumo.conn, sumo)
        st_["t0"] = sumo.time

    hours = {}

    def probe(cur, c, sod) -> None:
        st_["te"].step(c, cur.time, cur.arrived)
        hours[round(cur.time, 1)] = sod / 3600.0

    t_wall = time.time()
    gen.one_run(seed, a, nodes, str(work), port, probe=probe, hook=hook)
    te = st_["te"]
    t_from = st_["t0"] + 60.0 * args["warmup"]
    legs = [{"t0": lg.t0, "hour": round(hours.get(round(lg.t0, 1), 0.0), 2), "est": lg.est_s, "free": lg.free_s,
             "share": lg.table_share, "real": lg.real_s, "how": lg.how, "new_way": bool(lg.changes),
             "kind": "place" if lg.dest_name else "through", "roads": len(lg.route),
             "route": lg.route, "pos0": round(lg.start_pos, 1), "end_pos": round(lg.dest_pos, 1)}   # (to score other tables)
            for lg in te.done if lg.t0 >= t_from]
    out = {"seed": seed, "start": start, "legs": legs, "stats": te.stats, "wall_s": round(time.time() - t_wall)}
    done.write_text(json.dumps(out), encoding="utf-8")
    return out


def accuracy(legs: list[dict]) -> dict:
    ok = [lg for lg in legs if lg["how"] in ("parked", "arrived", "stopped") and lg["real"] and lg["real"] > 0]
    if not ok:
        return {"n": 0}
    err = [lg["real"] - lg["est"] for lg in ok]
    rel = [abs(e) / lg["real"] for e, lg in zip(err, ok, strict=True)]
    return {"n": len(ok), "median_abs_err_s": round(st.median(abs(e) for e in err), 1),
            "median_err_s": round(st.median(err), 1),
            "within_20pct": round(sum(r <= 0.2 for r in rel) / len(ok), 3),
            "within_60s": round(sum(abs(e) <= 60 for e in err) / len(ok), 3),
            "median_real_s": round(st.median(lg["real"] for lg in ok), 1),
            "median_est_s": round(st.median(lg["est"] for lg in ok), 1)}


def rescore(out: str, tables: list[str]) -> dict:
    """The trips saved in out/*/legs.json (with their routes) scored again with other tables, without simulating:
    TABLE (each road's median) or TABLE:mean_s. The first entry should give the run's own
    estimates back (checked)."""
    import glob

    import sumolib

    from cosim.highway import HIGHWAY_SPEED, find_loops
    from cosim.trip_estimates import TripEstimates

    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    ring = {e: HIGHWAY_SPEED for lp in find_loops(net) for e in lp}
    legs = [lg for f in sorted(glob.glob(str(Path(out) / "*" / "legs.json")))
            for lg in json.loads(Path(f).read_text(encoding="utf-8"))["legs"] if "route" in lg]
    rep = {"trips": len(legs)}
    for spec in tables:
        path, _, stat = spec.partition(":")
        te = TripEstimates(net, typical_file=Path(path), speeds=ring, stat=stat or "median_s")
        new = []
        for lg in legs:
            est, free, share = te.estimate(lg["route"], lg["pos0"], lg["end_pos"])
            new.append({**lg, "est": round(est, 1), "free": round(free, 1), "share": round(share, 2)})
        same = sum(abs(a["est"] - b["est"]) <= 0.15 for a, b in zip(legs, new, strict=True))
        rep[spec] = {"same_as_run": round(same / max(len(legs), 1), 3), "all": accuracy(new),
                     "by_kind": {k: accuracy([lg for lg in new if lg["kind"] == k]) for k in ("place", "through")},
                     "one_road": accuracy([lg for lg in new if lg["roads"] == 1]),
                     "two_roads_or_more": accuracy([lg for lg in new if lg["roads"] > 1])}
    return rep


def main(argv=None) -> int:
    from eval_agents_forecast import seeds_arg

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", default="11-20")
    ap.add_argument("--start-hours", default="7,12,17")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--controller", default="agents_observed")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--ring-trips", action="store_true")
    ap.add_argument("--table", default=str(ROOT / "models" / "route_guidance" / "movement_times.json"),
                    help="the viewer's (cosim/trip_estimates.TYPICAL); models/trip_estimates/movement_times.json: "
                         "the viewer's own traffic (10 Nov), a little worse on dev")
    ap.add_argument("--stat", default="median_s", help="median_s | mean_s: each road's time from the table")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--port", type=int, default=9700)
    ap.add_argument("--out", default=str(ROOT / "out" / "trip_estimates" / "dev"))
    ap.add_argument("--rescore", default="", help="TABLE[:mean_s],...: score the trips saved in --out again")
    a = ap.parse_args(argv)
    if a.rescore:
        rep = rescore(a.out, a.rescore.split(","))
        (Path(a.out) / "rescore.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1))
        return 0
    seeds = seeds_arg(a.seeds)
    hours = [float(h) for h in a.start_hours.split(",")]
    args = {k: getattr(a, k) for k in ("out", "minutes", "warmup", "controller", "scale", "ring_trips", "table", "stat")}
    jobs = [(s, h) for s in seeds for h in hours]
    res = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, s, h, args, a.port + 3 * i) for i, (s, h) in enumerate(jobs)]
        for f, (s, h) in zip(futs, jobs, strict=True):
            r = f.result()
            res.append(r)
            print(f"seed {s} from {h:g}h: {len(r['legs'])} trips, {r['wall_s']} s", flush=True)
    legs = [lg for r in res for lg in r["legs"]]
    how = {}
    for lg in legs:
        how[lg["how"]] = how.get(lg["how"], 0) + 1
    rep = {"seeds": seeds, "start_hours": hours, "table": a.table, "stat": a.stat, "trips": len(legs), "ended": how,
           "all": accuracy(legs),
           "by_kind": {k: accuracy([lg for lg in legs if lg["kind"] == k]) for k in ("place", "through")},
           "given_another_way": accuracy([lg for lg in legs if lg["new_way"]]),
           "same_way": accuracy([lg for lg in legs if not lg["new_way"]]),
           "by_length": {f"{lo}-{hi} s": accuracy([lg for lg in legs if lg["real"] and lo <= lg["real"] < hi])
                         for lo, hi in ((0, 60), (60, 180), (180, 600), (600, 10 ** 6))},
           "by_town_hour": {f"{h:02d}-{h + 3:02d}": accuracy([lg for lg in legs if h <= lg["hour"] % 24 < h + 3])
                            for h in range(0, 24, 3)},
           "from_table_mostly": accuracy([lg for lg in legs if lg["share"] >= 0.5])}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("trips", "ended", "all", "by_kind")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
