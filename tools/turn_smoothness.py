"""How smoothly the free-fly viewer turns its vehicles: SUMO alone on Town05 (the viewer's evening demand),
then every vehicle posed 60 times a second by cosim.run.Poser with each heading mode, as the viewer would.

    python tools/turn_smoothness.py [--seconds 300]

Per mode: forward motion (speed jumps between two poses, acceleration, jerk: a car whose speed jumps at
every SUMO step looks as if it lagged), how often a vehicle's turning reverses (left-right-left: a shake), yaw acceleration, the largest
turning rate, and slip: the angle between where the body points and where its centre moves (a car's body
follows its path; a body swinging across it reads as shaking).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import daily  # noqa: E402
from cosim.interp import Trails  # noqa: E402
from cosim.live import Population  # noqa: E402
from cosim.run import Poser  # noqa: E402
from cosim import scenario as scenario_mod  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402

FPS = 60


def wrap(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--port", type=int, default=8975)
    a = ap.parse_args(argv)
    scen = scenario_mod.load("town05", ROOT)
    pool = daily.load_pool(scen.folder)
    pf = ROOT / "out" / "freefly" / "turn_smoothness.rou.xml"
    rids = daily.write_pool_routes(pool, pf)
    curve = daily.population_curve(None, pool, mean_trip_min=140 / 60.0)
    sod = 17.5 * 3600
    sumo = SumoSide(scen.folder, "actuated", seed=7, port=a.port, label="turns", extras=True, emergency_every=45,
                    extra_args=["--route-files", str(pf), "--end", "100000000", "--lanechange.duration", "2"])
    pop = Population(sumo.conn, pool, rids, curve, seed=7)
    pop.add(int(pop.target(sod)), anywhere=True)
    steps = []
    while sumo.time < 200 + a.seconds:
        cur = sumo.step()
        pop.step(cur.vehicles, sod, lambda vid, v: True)
        if sumo.time >= 200:
            steps.append(cur)
    dt = sumo.step_length
    sumo.close()
    sub = int(round(dt * FPS))
    for mode in ("turn", "trail"):
        trails = Trails() if mode == "trail" else None
        seq: dict[str, list] = {}
        for k in range(1, len(steps) - 1):
            pz = Poser(scen.frame, None, heading=mode)
            pz.begin_step(steps[k - 1], steps[k])
            pz.pre = steps[k - 2].vehicles if k >= 2 else None
            pz.nxt = steps[k + 1].vehicles
            for i in range(sub):
                for vid, tf in pz.poses(i / sub, dt, trails).items():
                    seq.setdefault(vid, []).append((tf.location.x, tf.location.y, tf.rotation.yaw))
        rev, acc, slip, rates = 0, [], [], []
        # forward motion: speed from one pose to the next, its jumps, acceleration and jerk
        dv, lacc, jerk = [], [], []
        for s in seq.values():
            sp = [math.hypot(s[j][0] - s[j - 1][0], s[j][1] - s[j - 1][1]) * FPS for j in range(1, len(s))]
            for j in range(2, len(sp)):
                a1, a0 = (sp[j] - sp[j - 1]) * FPS, (sp[j - 1] - sp[j - 2]) * FPS
                dv.append(abs(sp[j] - sp[j - 1]))
                lacc.append(abs(a1))
                jerk.append(abs(a1 - a0) * FPS)
        for s in seq.values():
            for j in range(2, len(s)):
                r1 = wrap(s[j][2] - s[j - 1][2]) * FPS
                r0 = wrap(s[j - 1][2] - s[j - 2][2]) * FPS
                acc.append(abs(r1 - r0) * FPS)
                rates.append(abs(r1))
                if abs(r1) > 3 and abs(r0) > 3 and r1 * r0 < 0:
                    rev += 1
                vx, vy = s[j][0] - s[j - 1][0], s[j][1] - s[j - 1][1]
                if math.hypot(vx, vy) * FPS > 2.0:
                    slip.append(abs(wrap(math.degrees(math.atan2(vy, vx)) - s[j][2])))
        acc.sort(), slip.sort(), rates.sort(), dv.sort(), lacc.sort(), jerk.sort()
        q = lambda v, p: v[min(len(v) - 1, int(len(v) * p))]  # noqa: E731
        print(f"{mode:6s} turning reversals {rev}, yaw accel p99 {q(acc, .99):.0f} p99.9 {q(acc, .999):.0f} deg/s2, "
              f"max turn rate {rates[-1]:.0f} deg/s, slip p99 {q(slip, .99):.1f} p99.9 {q(slip, .999):.1f} deg", flush=True)
        print(f"{'':6s} forward: speed jumps over 0.5 m/s between two poses {sum(x > 0.5 for x in dv)}, "
              f"acceleration p99 {q(lacc, .99):.1f} p99.9 {q(lacc, .999):.1f} m/s2, "
              f"jerk p99 {q(jerk, .99):.0f} p99.9 {q(jerk, .999):.0f} m/s3", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
