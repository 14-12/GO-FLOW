"""Several cars driving into the same car park and out of it at once (cosim/parking_motion.py), SUMO alone: how close
the cars on drives come to each other and to the cars SUMO drives on the road beside the car park (the bodies as
three circles each, 1.9 m wide), and how long the drives were held up giving way.

    python tools/check_parking_crowd.py --place marche --cars 8 --every 4

A car is sent every --every seconds to --place (shopping: a short stay at a fast clock, so the drives in and out
cross). Writes out/parking_crowd_<place>.json. An overlap step: two bodies overlapping by more than 0.2 m at a step.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WIDTH_M = 1.9


def circles(x, y, face, length):
    """A car's body as three circles (its middle at x, y): middle, and half a length less half a width either way."""
    a = math.radians(face)
    hx, hy = math.sin(a), math.cos(a)
    off = max(length / 2 - WIDTH_M / 2, 0.0)
    return [(x + hx * k * off, y + hy * k * off) for k in (-1, 0, 1)]


def clearance(c1, c2):
    """The gap between two bodies (negative: they overlap), from their circles."""
    return min(math.dist(p, q) for p in c1 for q in c2) - WIDTH_M


def _state(motion, d, vs, vid, t):
    """What a car is doing, for the overlaps' list: its drive (kind, progress / duration, pace, P: SUMO has it in
    its space), or SUMO's (the space it is in, its speed)."""
    dv = motion.drives.get(vid)
    if dv is None:
        return f"sumo {d.parked.get(vid)} {vs[vid].speed:.1f}"
    return f"{dv.kind} {'done' if dv.done(t) else round(dv.progress(t), 1)}/{dv.duration:.0f} r{dv.rate:.2f} " \
           f"{'P' if vid in d.parked else ''}"


def run(place="marche", cars=8, every=4.0, seed=5, clock=240.0, steps=2400, port_label="parking_crowd"):
    import sumolib
    import traci

    from cosim import parking_motion as pm
    from cosim import purposes as P
    from cosim.sumo_side import Vehicle

    scen = ROOT / "scenarios" / "town05"
    traci.start([sumolib.checkBinary("sumo"), "-n", str(scen / "town05.net.xml"), "-a", str(P.ADD), "--step-length",
                 "0.5", "--no-step-log", "--no-warnings", "--time-to-teleport", "-1", "--seed", str(seed)],
                label=port_label)
    c = traci.getConnection(port_label)
    res = {"place": place, "cars": cars, "every_s": every, "seed": seed}
    try:
        d = P.PurposeDemand(c, clock_speed=clock, seed=seed)
        d.rng = random.Random(seed)
        d.purpose_at = lambda hour, vclass: "shopping"
        d.places.choose = lambda rng, what, hour, exclude=None: place
        motion = pm.ParkingMotion(d, c)
        launched, t_next = 0, 1.0
        worst, overlaps, pairs_close, drive_steps = [], 0, 0, 0
        seen_in, seen_out = set(), set()
        for _ in range(steps):
            c.simulationStep()
            t = c.simulation.getTime()
            if launched < cars and t >= t_next:
                if d.launch(f"crowd{launched}", "DEFAULT_VEHTYPE", "passenger", 10.0, (), ()):
                    launched += 1
                t_next = t + every
            vs = {}
            for vid in c.vehicle.getIDList():
                x, y = c.vehicle.getPosition(vid)
                vs[vid] = Vehicle(vid, "passenger", "t", c.vehicle.getLength(vid), x, y, c.vehicle.getAngle(vid),
                                  c.vehicle.getSpeed(vid), 0)
            d.step([])
            for vid, aid in d.parked.items():
                v = vs.get(vid)
                if v is not None and "spaces" in d.places.areas[aid]:
                    a = math.radians(v.angle)
                    v.x += math.sin(a) * v.length / 2
                    v.y += math.cos(a) * v.length / 2
            motion.step(t, vs)
            for vid, dv in motion.drives.items():
                (seen_in if dv.kind == "in" else seen_out).add(vid)
            on = set(motion.drives)
            if not on:
                if launched >= cars and len(seen_out) >= launched and t > t_next + 30:
                    break
                continue
            drive_steps += 1
            bodies = {}
            for vid, v in vs.items():
                if abs(v.x) > 1e5 or (vid in d.parked and vid not in on):
                    continue          # (standing in a space: the paths keep off them)
                a = math.radians(v.angle)
                bodies[vid] = circles(v.x - math.sin(a) * v.length / 2, v.y - math.cos(a) * v.length / 2, v.angle,
                                      v.length)
            for a_, b_ in itertools.combinations(sorted(bodies), 2):
                if a_ not in on and b_ not in on:
                    continue
                g = clearance(bodies[a_], bodies[b_])
                if g < 1.0:
                    pairs_close += 1
                if g < -0.2:
                    overlaps += 1

                    worst.append((round(g, 2), a_, b_, round(t, 1), _state(motion, d, vs, a_, t),
                                  _state(motion, d, vs, b_, t)))
        res.update(launched=launched, drives_in=len(seen_in), drives_out=len(seen_out), drive_steps=drive_steps,
                   overlap_steps=overlaps, close_steps=pairs_close, worst=worst[::max(len(worst) // 25, 1)], stats=motion.stats)
    finally:
        c.close()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--place", default="marche")
    ap.add_argument("--cars", type=int, default=8)
    ap.add_argument("--every", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(argv)
    res = run(a.place, a.cars, a.every, a.seed)
    out = ROOT / "out" / f"parking_crowd_{a.place}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
