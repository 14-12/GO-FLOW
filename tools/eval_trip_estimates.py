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
    from cosim.live_eta import ProbeTraffic, time_left
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
         "trips": "purpose", "ring_planned": not args.get("ring_laps", False)}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    st_ = {}

    def hook(sumo, im) -> None:
        pop = getattr(sumo, "population", None)   # (one_run hands it over)
        names = area_names(pop.demand.places) if pop is not None and getattr(pop, "demand", None) else {}
        te = st_["te"] = TripEstimates(net, typical_file=Path(args["table"]), speeds=ring, area_names=names,
                                       stat=args["stat"],
                                       place_name=plan_names(pop.demand) if pop is not None and pop.demand else None,
                                       end_of=lambda v: sumo.ring.end_of(v) if getattr(sumo, "ring", None) else None)
        te.setup(sumo.conn, sumo)
        st_["t0"] = sumo.time
        st_["pop"] = pop
        st_["probes"] = {sh: ProbeTraffic(sh) for sh in SHARES}

    hours = {}
    town = {"ring_cars": [], "standing_share": [], "town_cars": []}

    def probe(cur, c, sod) -> None:
        te = st_["te"]
        te.step(c, cur.time, cur.arrived)
        hours[round(cur.time, 1)] = sod / 3600.0
        subs = c.vehicle.getAllSubscriptionResults()
        for pr in st_["probes"].values():
            pr.step(c, cur.time, cur.arrived, subs)
        checkpoints(te, c, cur.time, subs)
        st_["k"] = k = st_.get("k", 0) + 1
        if k % 120 or cur.time < st_["t0"] + 60.0 * args["warmup"]:
            return
        # (10 Nov) the town, every minute: cars on the ring road, the share of the town's cars standing (under 1 m/s,
        # parked ones, buses and the ring's own left out, as the mini-map counts them)
        subs = c.vehicle.getAllSubscriptionResults()
        pop = st_.get("pop")
        parked = pop.demand.parked if pop is not None and getattr(pop, "demand", None) else {}
        on_ring = sum(1 for v in cur.vehicles if (subs.get(v) or {}).get(0x50) in ring)
        tv = [x for v, x in cur.vehicles.items() if not v.startswith(("hw", "pt_")) and v not in parked]
        town["ring_cars"].append(on_ring)
        town["town_cars"].append(len(tv))
        town["standing_share"].append(sum(x.speed < 1.0 for x in tv) / max(len(tv), 1))

    def checkpoints(te, c, now, subs) -> None:
        """(10 Nov) Every CHECK_S of each trip: its time left by each method, graded at its end (real time left)."""
        for vid, lg in te.active.items():
            nxt = getattr(lg, "next_cp", None)
            if nxt is None:
                nxt = lg.next_cp = lg.t0 + CHECK_S
            if now < nxt:
                continue
            r = subs.get(vid) or {}
            road, pos = r.get(0x50, ""), r.get(0x56)
            if not road or road.startswith(":") or pos is None:
                continue   # on a junction: at the next step
            try:
                route, idx = list(c.vehicle.getRoute(vid)), c.vehicle.getRouteIndex(vid)
                j = route.index(lg.dest_edge, max(idx, 0))
            except Exception:  # noqa: BLE001 - its end no longer on its way: no checkpoint
                lg.next_cp = nxt + CHECK_S
                continue
            roads = route[idx:j + 1]
            if not roads or roads[0] != road:
                continue
            lg.next_cp = nxt + CHECK_S
            el = now - lg.t0
            cp = {"t": round(el, 1), "start": round(max(0.0, lg.est_s - el), 1),
                  "usual": round(time_left(te, None, roads, pos, lg.dest_pos, now)["left_s"], 1)}
            for sh, pr in st_["probes"].items():
                for b in BLENDS:
                    for k_ in SHRINKS:
                        cp[f"p{int(sh * 100)}_b{b}_k{k_}"] = round(
                            time_left(te, pr, roads, pos, lg.dest_pos, now, b, k_)["left_s"], 1)
            if not hasattr(lg, "cps"):
                lg.cps = []
            lg.cps.append(cp)

    t_wall = time.time()
    res = gen.one_run(seed, a, nodes, str(work), port, probe=probe, hook=hook)
    te = st_["te"]
    t_from = st_["t0"] + 60.0 * args["warmup"]
    legs = [{"t0": lg.t0, "hour": round(hours.get(round(lg.t0, 1), 0.0), 2), "est": lg.est_s, "free": lg.free_s,
             "share": lg.table_share, "real": lg.real_s, "how": lg.how, "new_way": bool(lg.changes),
             "kind": "ring" if lg.vid.startswith("hw") else "place" if lg.dest_name else "through",
             "roads": len(lg.route), "vid": lg.vid, "cps": getattr(lg, "cps", []),
             "route": lg.route, "pos0": round(lg.start_pos, 1), "end_pos": round(lg.dest_pos, 1)}   # (to score other tables)
            for lg in te.done if lg.t0 >= t_from]
    rem = (res or {}).get("removed") or {}
    out = {"seed": seed, "start": start, "legs": legs, "stats": te.stats, "wall_s": round(time.time() - t_wall),
           "town": {"ring_cars": round(st.mean(town["ring_cars"]), 1) if town["ring_cars"] else None,
                    "town_cars": round(st.mean(town["town_cars"]), 1) if town["town_cars"] else None,
                    "standing_share": round(st.mean(town["standing_share"]), 3) if town["standing_share"] else None,
                    "trips_parked": ((res or {}).get("purposes") or {}).get("arrived_parked"),
                    "taken_off": sum(v for k_, v in rem.items() if k_ not in ("population_excess", "trips_not_started",
                                                                             "ring_excess") and isinstance(v, int)),
                    "ring_excess": rem.get("ring_excess"), "removed": rem}}
    done.write_text(json.dumps(out), encoding="utf-8")
    return out


SHARES = (0.10, 0.25, 0.50)     # cars running the app (cosim/live_eta.py)
BLENDS = (30, 120, 100000)      # live weight exp(-tau / b); 100000: live only where known
SHRINKS = (0, 3, 10)            # a road's live time shrunk toward its usual one by how few app cars it rests on
CHECK_S = 30.0                  # a trip's time left checked every 30 s


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


def left_accuracy(legs: list[dict]) -> dict:
    """(10 Nov) The time left at each checkpoint (every CHECK_S of a trip) by each method against the real time left:
    start (the estimate at the start minus the time driven), usual (the table from where it is), pXX_bYY_kZZ (with the
    app's live times: XX % of cars, blend YY s, shrink ZZ); overall and by the real time left."""
    rows = [(lg["real"] - cp["t"], cp) for lg in legs
            if lg["how"] in ("parked", "arrived", "stopped") and lg["real"] for cp in lg.get("cps") or []]
    rows = [(r, cp) for r, cp in rows if r > 0]
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0][1] if k != "t"]

    def acc(sel):
        out = {"n": len(sel)}
        if not sel:
            return out
        for k in keys:
            e = [r - cp[k] for r, cp in sel]
            out[k] = {"mae": round(st.mean(abs(x) for x in e), 1), "median_abs": round(st.median(abs(x) for x in e), 1),
                      "bias": round(st.median(e), 1),
                      "within_20pct": round(sum(abs(x) <= 0.2 * r for x, (r, _) in zip(e, sel)) / len(sel), 3),
                      "within_30s": round(sum(abs(x) <= 30 for x in e) / len(sel), 3)}
        return out
    return {"all": acc(rows), "by_left": {f"{lo}-{hi} s": acc([x for x in rows if lo <= x[0] < hi])
                                           for lo, hi in ((0, 60), (60, 180), (180, 600), (600, 10 ** 6))}}


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
    ap.add_argument("--ring-laps", action="store_true", help="the ring cars' old 200 laps (no planned trips)")
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
    args = {k: getattr(a, k) for k in ("out", "minutes", "warmup", "controller", "scale", "ring_trips", "table", "stat", "ring_laps")}
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
           "by_kind": {k: accuracy([lg for lg in legs if lg["kind"] == k]) for k in ("place", "through", "ring")},
           "given_another_way": accuracy([lg for lg in legs if lg["new_way"]]),
           "same_way": accuracy([lg for lg in legs if not lg["new_way"]]),
           "by_length": {f"{lo}-{hi} s": accuracy([lg for lg in legs if lg["real"] and lo <= lg["real"] < hi])
                         for lo, hi in ((0, 60), (60, 180), (180, 600), (600, 10 ** 6))},
           "by_town_hour": {f"{h:02d}-{h + 3:02d}": accuracy([lg for lg in legs if h <= lg["hour"] % 24 < h + 3])
                            for h in range(0, 24, 3)},
           "from_table_mostly": accuracy([lg for lg in legs if lg["share"] >= 0.5]),
           "time_left": left_accuracy(legs),
           "town": {k: round(st.mean(r["town"][k] for r in res if r.get("town", {}).get(k) is not None), 3)
                    for k in ("ring_cars", "town_cars", "standing_share", "trips_parked", "taken_off", "ring_excess")
                    if any(r.get("town", {}).get(k) is not None for r in res)}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("trips", "ended", "all", "by_kind")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
