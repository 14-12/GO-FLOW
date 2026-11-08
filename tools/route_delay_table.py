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


def run_one(scale: float, seed: int, args: dict, port: int) -> dict:
    import goflow_prediction_dataset as gen

    from cosim.daily import GAME_SPEED

    work = Path(args["work"]) / f"x{scale:g}_seed{seed}"
    done = work / "times.json"
    if done.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    (work / "runs").mkdir(parents=True, exist_ok=True)
    vr = work / "vehroutes.xml"
    a = {"minutes": args["minutes"], "warmup": args["warmup"], "start_hour": 6.0, "clock_speed": GAME_SPEED, "dt": 30.0,
         "controller": args["controller"], "scale": scale, "accidents": True, "careless_share": None, "buses": True,
         "ring": True, "junction_blocker": 60.0,
         "sumo_extra": ["--vehroute-output", str(vr), "--vehroute-output.exit-times", "true"]}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    state = {}

    def hook(sumo, im) -> None:
        state["t0"] = sumo.time

    t_wall = time.time()
    gen.one_run(seed, a, nodes, str(work), port, hook=hook)
    t_from = state["t0"] + 60.0 * args["warmup"]
    samples: dict[str, list] = defaultdict(list)
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
        for i in range(1, len(edges)):
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
    ap.add_argument("--work", default=str(ROOT / "out" / "route_guidance" / "train"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    seeds = seeds_arg(a.seeds)
    bad = [s for s in seeds if s not in TRAIN]
    if bad:
        print(f"train seeds only (101-130): {bad}")
        return 2
    args = {"work": a.work, "minutes": a.minutes, "warmup": a.warmup, "controller": a.controller}
    jobs = [(float(sc), s) for sc in a.scales.split(",") for s in seeds]
    pooled: dict[str, list] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, sc, s, args, a.port + 3 * i) for i, (sc, s) in enumerate(jobs)]
        for f, (sc, s) in zip(futs, jobs, strict=True):
            r = f.result()
            n = sum(len(x) for x in r["samples"].values())
            print(f"x{sc:g} seed {s}: {n} road times, {r['wall_s']} s", flush=True)
            for k, xs in r["samples"].items():
                pooled[k] += xs
    table = {k: {"n": len(xs), "mean_s": round(st.mean(xs), 2), "median_s": round(st.median(xs), 2)}
             for k, xs in sorted(pooled.items())}
    card = {"what": "time on a road by the road taken next (s), the wait at its signal included; first road of a trip "
                    "left out", "seeds": seeds, "scales": a.scales, "controller": a.controller,
            "recipe": {"minutes": a.minutes, "warmup": a.warmup, "start_hour": 6.0, "clock_speed": 30},
            "made": time.strftime("%Y-%m-%d %H:%M"), "movements": table}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"{len(table)} movements, {sum(v['n'] for v in table.values())} samples -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
