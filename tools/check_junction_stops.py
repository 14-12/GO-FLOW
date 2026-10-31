"""Do cars stop inside the signalised junctions? (27 Oct: with permissive turns they did, waiting for oncoming cars or
walkers in the middle of the junction; with the protected signals a car that has its green goes.)

    python tools/check_junction_stops.py --controller actuated --scale 0.8 --seeds 21,22 --seconds 1800

Runs the town as tools/eval_town05_baseline.py does and counts, every step, the vehicles standing (under 0.1 m/s)
on a signalised junction's internal lanes, i.e. inside the junction past the stop line: vehicle-seconds and distinct
vehicles, split by why: the lane it leaves by is full (a queue backing into the junction, not the signal's doing)
or not (it stopped for someone crossing its path). Also SUMO's collisions inside junctions, and the trips' waiting.
Writes out/signal_checks/junction_stops_<tag>.json and prints a line per run.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(controller: str, scale: float, seed: int, seconds: float, port: int, tag: str) -> dict:
    import traci.constants as tc

    from cosim.agents_actuated import make as make_agents
    from cosim.sumo_side import SumoSide

    out_dir = ROOT / "out" / "signal_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    coll = out_dir / f"stops_{tag}_{controller}_{scale:g}_{seed}.collisions.xml"
    trip = out_dir / f"stops_{tag}_{controller}_{scale:g}_{seed}.tripinfo.xml"
    policy = make_agents(controller) if controller.startswith("agents") else None
    s = SumoSide(ROOT / "scenarios" / "town05", controller if policy is None else "agents", seed=seed, port=port,
                 label=f"stops{port}", policy=policy, extras=True, emergency_every=120.0,
                 vehicle_vars=[tc.VAR_LANE_ID],
                 extra_args=["--scale", f"{scale:g}", "--end", f"{seconds:g}", "--collision.check-junctions", "true",
                             "--collision.action", "warn", "--intermodal-collision.action", "warn",
                             "--collision-output", str(coll), "--tripinfo-output", str(trip),
                             "--tripinfo-output.write-unfinished", "true"])
    s.extras.config.pedestrian_trips = True
    if policy is not None:
        ex = s.extras
        policy.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    tls = set(s.tls_ids)
    stood: dict[str, float] = {}
    by_why = {"exit_full": 0.0, "signal_conflict": 0.0}
    vehicles_inside = {"exit_full": set(), "signal_conflict": set()}
    c = s.conn
    try:
        while s.time < seconds:
            s.step()
            dt = s.step_length
            for vid, sub in s.vehicle_subs.items():
                lane = sub.get(tc.VAR_LANE_ID, "")
                if not lane.startswith(":") or sub.get(tc.VAR_SPEED, 1.0) >= 0.1:
                    continue
                j = lane[1:].rsplit("_", 2)[0]
                if j not in tls or "_w" in lane or "_c" in lane:
                    continue
                # held by a queue: a standing vehicle less than 8 m ahead of it (the exit is full)
                leader = c.vehicle.getLeader(vid, 30.0)
                full = leader is not None and leader[1] < 8.0 and c.vehicle.getSpeed(leader[0]) < 0.5
                why = "exit_full" if full else "signal_conflict"
                by_why[why] += dt
                vehicles_inside[why].add(vid)
                stood[vid] = stood.get(vid, 0.0) + dt
    finally:
        s.close()
    collisions = 0
    if coll.is_file():
        collisions = sum(1 for e in ET.parse(coll).getroot().iter("collision") if e.get("type") == "junction"
                         and "ped" not in (e.get("victimType", "") + e.get("colliderType", "")))
        ped_coll = sum(1 for e in ET.parse(coll).getroot().iter("collision")
                       if "ped" in (e.get("victimType", "") + e.get("colliderType", "")))
        coll.unlink()
    else:
        ped_coll = 0
    trips = ET.parse(trip).getroot()
    tr = trips.findall("tripinfo")
    walks = [w for p in trips.findall("personinfo") for w in p.findall("walk")]
    trip.unlink(missing_ok=True)
    res = {"tag": tag, "controller": controller, "scale": scale, "seed": seed, "seconds": seconds,
           "stood_inside_s": {k: round(v, 1) for k, v in by_why.items()},
           "vehicles_stopping_inside": {k: len(v) for k, v in vehicles_inside.items()},
           "vehicles": len(tr), "completed": sum(float(t.get("arrival", -1)) >= 0 for t in tr),
           "mean_waiting_s": round(st.mean(float(t.get("waitingTime")) for t in tr), 2),
           "mean_time_loss_s": round(st.mean(float(t.get("timeLoss")) for t in tr), 2),
           "ped_time_loss_s": round(st.mean(float(w.get("timeLoss", 0)) for w in walks), 2) if walks else None,
           "junction_collisions_vehicles": collisions, "collisions_pedestrians": ped_coll}
    (ROOT / "out" / "signal_checks" / f"junction_stops_{tag}_{controller}_{scale:g}_{seed}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controller", default="actuated")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--seeds", default="21")
    ap.add_argument("--seconds", type=float, default=1800.0)
    ap.add_argument("--port", type=int, default=9600)
    ap.add_argument("--tag", default="protected")
    a = ap.parse_args(argv)
    for k, sd in enumerate(int(x) for x in a.seeds.split(",")):
        r = run(a.controller, a.scale, sd, a.seconds, a.port + k, a.tag)
        print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
