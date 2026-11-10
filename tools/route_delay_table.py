"""The route agent's typical times: how long cars take on each road of Town05 by the way they leave it (the wait at
its signal included), from SUMO runs on the TRAIN seeds only. What a navigation service has from its users' past
trips. Written to models/route_guidance/movement_times.json, read by agents/route_recommendation/router.py
(Router.set_typical) for the agent's estimate and, in the "agent_typical" variant, its route costs.

    python tools/route_delay_table.py --seeds 101-110 --scales 1,2,3 --jobs 2

Each run is the forecasters' recipe (tools/goflow_prediction_dataset.one_run: the viewer's town with crashes, buses
and the ring road, the clock at 30, 5 min warm-up from 06:00, then --minutes), under --controller, no guidance. SUMO
writes every finished car's roads and the moment it left each (--vehroute-output.exit-times); the time on road i
(entered when it left road i-1) is filed under (road i, the road it turned into). The first road of a trip is left
out (it starts part way along it). Cars of the town only ("live" ids), not buses, emergency or ring-road vehicles;
cars whose route was changed on the way (a crash ahead) are left out.

--road-times live (10 Nov): the same times read from the run's own step (the road of every car, which SUMO sends with
each step) instead of SUMO's route output: every car counts, also those that park and never "arrive" (the route
output only has cars that finished, so it leant on through traffic), and SUMO's route output is not used (with
exit times it quit with "vector too long" on two train runs). A road where the car parked or stopped is left out,
the first road of every trip too (it starts part way along it), and a road left while teleporting.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
OUT = ROOT / "models" / "route_guidance" / "movement_times.json"
TRAIN = range(101, 131)


class LiveRoadTimes:
    """(10 Nov) Each car's time on a road, from leaving the road before it to leaving this one (the junction between
    and the wait at this road's signal included, as SUMO's exit times), filed under (road, road taken next). Fed after
    every step with the vehicles' subscription results (their road). Only roads entered from t_from on."""

    def __init__(self, t_from: float, prefix: str = "live"):
        self.t_from, self.prefix = t_from, prefix
        self.state: dict[str, list] = {}   # vid -> [road, entered at (None: unknown), ok, left at (None: still on it)]
        self.samples: dict[str, list] = defaultdict(list)
        self.left_out = defaultdict(int)

    def step(self, c, now: float, arrived=()) -> None:
        import traci.constants as tc

        try:
            halted = set(c.simulation.getParkingStartingVehiclesIDList()) | set(c.simulation.getStopStartingVehiclesIDList())
        except Exception:  # noqa: BLE001
            halted = set()
        for vid, r in c.vehicle.getAllSubscriptionResults().items():
            if not vid.startswith(self.prefix):
                continue
            road = r.get(tc.VAR_ROAD_ID, "")
            s = self.state.get(vid)
            if s is None:   # set off part way along its first road
                self.state[vid] = [road if road and not road.startswith(":") else "", None, False, None]
                continue
            if vid in halted:   # parks or stops on this road: no time on the road
                s[2] = False
            if not road:   # teleporting
                if s[1] is not None:
                    self.left_out["teleport"] += 1
                s[:] = ["", None, False, None]
            elif road.startswith(":"):
                if s[3] is None:
                    s[3] = now
            elif road != s[0]:
                t_out = s[3] if s[3] is not None else now
                if s[0] and s[1] is not None:
                    if not s[2]:
                        self.left_out["halted"] += 1
                    elif s[1] >= self.t_from:
                        self.samples[f"{s[0]}>{road}"].append(round(t_out - s[1], 2))
                s[:] = [road, t_out if s[0] else None, True, None]   # (after a teleport: part way along it)
            else:
                s[3] = None
        for vid in arrived:   # its last road: not filed (where it ends on that road is not its end)
            self.state.pop(vid, None)


def run_one(scale: float, seed: int, args: dict, port: int, start: float = 6.0) -> dict:
    import goflow_prediction_dataset as gen

    from cosim.daily import GAME_SPEED

    work = Path(args["work"]) / (f"x{scale:g}_seed{seed}" if start == 6.0 else f"x{scale:g}_h{start:g}_seed{seed}")
    done = work / "times.json"
    if done.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    (work / "runs").mkdir(parents=True, exist_ok=True)
    live = args.get("road_times", "vehroutes") == "live"
    vr = work / "vehroutes.xml"
    a = {"minutes": args["minutes"], "warmup": args["warmup"], "start_hour": start, "clock_speed": GAME_SPEED,
         "dt": 30.0, "controller": args["controller"], "scale": scale, "accidents": True, "careless_share": None,
         "buses": True, "ring": True, "ring_trips": args.get("ring_trips", False),
         "junction_blocker": args.get("junction_blocker", 60.0),
         "trips": args.get("trips", "random"),
         "sumo_extra": [] if live else ["--vehroute-output", str(vr), "--vehroute-output.exit-times", "true"]}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    state = {}

    def hook(sumo, im) -> None:
        state["t0"] = sumo.time
        if live:
            state["rec"] = LiveRoadTimes(sumo.time + 60.0 * args["warmup"], prefix=args.get("prefix", "live"))

    def probe(cur, c, sod) -> None:
        state["rec"].step(c, cur.time, cur.arrived)

    t_wall = time.time()
    gen.one_run(seed, a, nodes, str(work), port, hook=hook, probe=probe if live else None)
    t_from = state["t0"] + 60.0 * args["warmup"]
    samples: dict[str, list] = defaultdict(list)
    if live:
        out = {"scale": scale, "seed": seed, "samples": state["rec"].samples, "left_out": state["rec"].left_out,
               "road_times": "live", "wall_s": round(time.time() - t_wall)}
        done.write_text(json.dumps(out), encoding="utf-8")
        return out
    for v in ET.parse(vr).getroot().iter("vehicle"):
        vid = v.get("id", "")
        if not vid.startswith("live") or float(v.get("depart", -1)) < t_from:
            continue
        r = v.find("route")   # a car whose route was changed on the way has a routeDistribution instead: left out
        if r is None or not r.get("exitTimes"):
            continue
        edges, exits = r.get("edges").split(), [float(x) for x in r.get("exitTimes").split()]
        if len(edges) != len(exits) or any(b < a for a, b in zip(exits, exits[1:])):
            continue
        # (10 Nov) a road on which the car stopped (parked at a place, trips="purpose") is not a time on the road:
        # SUMO's exit time there includes the stop. Those samples are left out.
        stops = [(float(x.get("started")), float(x.get("ended"))) for x in v.findall("stop")
                 if x.get("started") is not None and x.get("ended") is not None]
        for i in range(1, len(edges)):
            if any(exits[i - 1] - 1e-6 <= a_ <= exits[i] + 1e-6 for a_, _ in stops):
                continue
            nxt = edges[i + 1] if i + 1 < len(edges) else ""
            samples[f"{edges[i]}>{nxt}"].append(round(exits[i] - exits[i - 1], 2))
    out = {"scale": scale, "seed": seed, "samples": samples, "wall_s": round(time.time() - t_wall)}
    done.write_text(json.dumps(out), encoding="utf-8")
    vr.unlink(missing_ok=True)
    return out


def main(argv=None) -> int:
    from eval_agents_forecast import seeds_arg

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", default="101-110")
    ap.add_argument("--scales", default="1,2,3")
    ap.add_argument("--controller", default="agents_observed")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--port", type=int, default=9900)
    ap.add_argument("--start-hours", default="6", help="town hours the runs start at, e.g. 7,12,17 (10 Nov)")
    ap.add_argument("--trips", default="random", help="random | purpose (the viewer's: trips between places)")
    ap.add_argument("--ring-trips", action="store_true", help="the ring carries the town's trips (cosim/highway.py)")
    ap.add_argument("--junction-blocker", type=float, default=60.0, help="0: as the viewer (SUMO's own rule)")
    ap.add_argument("--road-times", default="vehroutes", choices=["vehroutes", "live"],
                    help="live: read from each step, every car (10 Nov); vehroutes: SUMO's route output, finished cars")
    ap.add_argument("--prefix", default="live", help="--road-times live: whose times (live: the town's cars; hw: the "
                                                     "ring road's, cosim/trip_estimates.RING_TIMES)")
    ap.add_argument("--work", default=str(ROOT / "out" / "route_guidance" / "train"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    seeds = seeds_arg(a.seeds)
    bad = [s for s in seeds if s not in TRAIN]
    if bad:
        print(f"train seeds only (101-130): {bad}")
        return 2
    args = {"work": a.work, "minutes": a.minutes, "warmup": a.warmup, "controller": a.controller, "trips": a.trips,
            "ring_trips": a.ring_trips, "junction_blocker": a.junction_blocker,
            "road_times": a.road_times, "prefix": a.prefix}
    hours = [float(h) for h in a.start_hours.split(",")]
    jobs = [(float(sc), s, h) for sc in a.scales.split(",") for s in seeds for h in hours]
    pooled: dict[str, list] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, sc, s, args, a.port + 3 * i, h) for i, (sc, s, h) in enumerate(jobs)]
        for f, (sc, s, h) in zip(futs, jobs, strict=True):
            r = f.result()
            n = sum(len(x) for x in r["samples"].values())
            print(f"x{sc:g} seed {s} from {h:g}h: {n} road times, {r['wall_s']} s", flush=True)
            for k, xs in r["samples"].items():
                pooled[k] += xs
    table = {k: {"n": len(xs), "mean_s": round(st.mean(xs), 2), "median_s": round(st.median(xs), 2)}
             for k, xs in sorted(pooled.items())}
    card = {"what": "time on a road by the road taken next (s), the wait at its signal included; first road of a trip "
                    "left out", "seeds": seeds, "scales": a.scales, "controller": a.controller,
            "recipe": {"minutes": a.minutes, "warmup": a.warmup, "start_hours": hours, "clock_speed": 30,
                       "trips": a.trips, "ring_trips": a.ring_trips, "junction_blocker": a.junction_blocker,
                       "road_times": a.road_times, "prefix": a.prefix},
            "made": time.strftime("%Y-%m-%d %H:%M"), "movements": table}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"{len(table)} movements, {sum(v['n'] for v in table.values())} samples -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
