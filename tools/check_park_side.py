"""(12 Nov) How far the town's cars drive round the block because of the side of the street they park on, SUMO alone.

A place's kerb parking has an area on each side of its street (scenarios/town05/town05.places.add.xml); a trip's
destination area, and its home area for the way back, are drawn among the place's areas with room
(PurposeDemand.area_for). For every trip launched this records the route length to the area drawn and to the
place's other areas with room at that moment (the router's best route from the same start), out and back:

    python tools/check_park_side.py --seeds 11-13 --park-side random --out out/park_side/random
    python tools/check_park_side.py --seeds 11-13 --park-side route --out out/park_side/route

The run is the evaluation's (tools/goflow_prediction_dataset.one_run: the viewer's town, buses, the ring road,
emergency runs, purpose trips, the clock at 30, x1, junction agents with bus priority), without accidents.
Writes OUT/seed<k>.json and OUT/summary.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def route_len(r, edges) -> float:
    return sum(r.roads[e].length for e in edges if e in r.roads) if edges else float("nan")


def instrument(log: list) -> None:
    """Wraps PurposeDemand.launch: after a trip is set, its two choices against the alternatives."""
    from cosim import purposes as P

    orig = P.PurposeDemand.launch

    def launch(self, vid, *aa, **kw):
        ok = orig(self, vid, *aa, **kw)
        plan = self.plans.get(vid) if ok else None
        if plan is None:
            return ok
        r, pl = self.places.router, self.places
        try:
            route = list(self.c.vehicle.getRoute(vid))
        except Exception:  # noqa: BLE001
            return ok
        a_dest = plan.areas[0]
        a_home = plan.areas[1] if len(plan.areas) > 1 else None
        dest_edge = pl.areas[a_dest]["edge"]
        start = route[0]
        own = sorted({a for a in pl.park_of.get(plan.dest, []) if self.free(a) > 0} | {a_dest})
        out = {a: route_len(r, r.best(start, pl.areas[a]["edge"])) for a in own}
        rec = {"vid": vid, "purpose": plan.purpose, "dest": plan.dest, "a_dest": a_dest, "start": start,
               "route_m": route_len(r, route), "out_m": out[a_dest], "out_best_m": min(out.values()),
               "out_options": len(own)}
        if a_home is not None and plan.home:
            homes = sorted({a for a in pl.park_of.get(plan.home, []) if self.free(a) > 0} | {a_home})
            back = {a: route_len(r, r.best(dest_edge, pl.areas[a]["edge"])) for a in homes}
            rec.update(a_home=a_home, back_m=back[a_home], back_best_m=min(back.values()), back_options=len(homes))
        log.append(rec)
        return ok

    P.PurposeDemand.launch = launch


def one(seed: int, out: Path, port: int, minutes: float, controller: str) -> dict:
    import goflow_prediction_dataset as gen

    from cosim.daily import GAME_SPEED

    log: list = []
    instrument(log)
    a = {"minutes": minutes, "warmup": 5.0, "start_hour": 6.0, "clock_speed": GAME_SPEED, "dt": 30.0,
         "controller": controller, "scale": 1.0, "accidents": False, "careless_share": None, "buses": True,
         "ring": True, "junction_blocker": 60.0, "trips": "purpose"}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    t = time.time()
    (out / "work" / f"s{seed}" / "runs").mkdir(parents=True, exist_ok=True)   # (one_run writes its samples there)
    gen.one_run(seed, a, nodes, str(out / "work" / f"s{seed}"), port)
    res = {"seed": seed, "trips": log, "wall_s": round(time.time() - t)}
    (out / f"seed{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def summarize(runs: list[dict]) -> dict:
    trips = [t for r in runs for t in r["trips"]]
    out_x = [t["out_m"] - t["out_best_m"] for t in trips]
    back_x = [t["back_m"] - t["back_best_m"] for t in trips if "back_m" in t]
    km = sum(t["route_m"] for t in trips) / 1000.0
    s = {"trips": len(trips), "route_km": round(km, 1),
         "out_extra_km": round(sum(out_x) / 1000.0, 2), "back_extra_km": round(sum(back_x) / 1000.0, 2),
         "share_of_km": round((sum(out_x) + sum(back_x)) / 1000.0 / km, 4) if km else None,
         "out_trips_over_50m": sum(1 for x in out_x if x > 50.0), "back_trips_over_50m": sum(1 for x in back_x if x > 50.0),
         "out_extra_mean_m": round(st.mean(out_x), 1) if out_x else None,
         "back_extra_mean_m": round(st.mean(back_x), 1) if back_x else None,
         "out_extra_when_over_50m_mean": round(st.mean([x for x in out_x if x > 50.0]), 1) if any(x > 50 for x in out_x) else None,
         "by_dest": {}}
    for t in trips:
        d = s["by_dest"].setdefault(t["dest"], {"n": 0, "extra_m": 0.0})
        d["n"] += 1
        d["extra_m"] = round(d["extra_m"] + t["out_m"] - t["out_best_m"], 1)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", default="11-13")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--controller", default="agents_observed+bus@bus_cut=1")
    ap.add_argument("--port", type=int, default=9880)
    ap.add_argument("--park-side", default="random", choices=("random", "route"),
                    help="cosim/purposes.PARK_SIDE for the run (random: as before 12 Nov)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_agents_forecast import seeds_arg

    from cosim import purposes as P

    P.PARK_SIDE = a.park_side
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for i, seed in enumerate(seeds_arg(a.seeds)):
        f = out / f"seed{seed}.json"
        runs.append(json.loads(f.read_text(encoding="utf-8")) if f.is_file() else one(seed, out, a.port + i, a.minutes, a.controller))
        print(seed, len(runs[-1]["trips"]), "trips", runs[-1].get("wall_s"), "s", flush=True)
    s = summarize(runs)
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "by_dest"}, indent=1))
    print(json.dumps(s["by_dest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
