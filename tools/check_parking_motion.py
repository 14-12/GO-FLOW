"""The drives into and out of the spaces (cosim/parking_motion.py) on a town hour, SUMO alone, as the viewer makes
them: how many, how far a car moves between two steps on a drive (no jump), and how far from SUMO's own car it is
when a drive out hands it back to SUMO (the seam).

    python tools/check_parking_motion.py --seed 101 --start 7.5 --hours 1.5

The forecasters' recipe (tools/goflow_prediction_dataset.one_run, trips with a purpose), the drives made from the
2nd real minute on. Writes out/parking_motion_check.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def _count_by(overlaps):
    """Overlap steps by kind: in a car park with painted bays or at a kerb, with another drive or a car of the
    road."""
    out: dict = {}
    for o in overlaps:
        k = f"{'kerb' if 'kerb' in (o[6] or '') else 'lot'} {o[4]} x {'drive' if o[5] else 'road car'}"
        out[k] = out.get(k, 0) + 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--start", type=float, default=7.5)
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--port", type=int, default=9980)
    ap.add_argument("--out", default=str(ROOT / "out" / "parking_motion_check.json"))
    ap.add_argument("--trace", default=None, help="a car to follow step by step (its SUMO and shown places)")
    a = ap.parse_args(argv)
    import goflow_prediction_dataset as gen

    from cosim import live
    from cosim.daily import GAME_SPEED
    from cosim.parking_motion import ParkingMotion

    st: dict = {"pop": None, "motion": None, "last": {}, "steps": [], "drive_steps": [], "t0": None}
    orig_init = live.Population.__init__

    def init(self, *aa, **kw):
        orig_init(self, *aa, **kw)
        st["pop"] = self
    live.Population.__init__ = init

    def probe(cur, conn, sod) -> None:
        d = getattr(st["pop"], "demand", None)
        if d is None:
            return
        if st["t0"] is None:
            st["t0"] = cur.time
        if st["motion"] is None:
            if cur.time - st["t0"] < 60.0:
                return
            st["motion"] = ParkingMotion(d, conn)
        m = st["motion"]
        for vid, aid in d.parked.items():                   # as cosim/freefly.produce
            v = cur.vehicles.get(vid)
            if v is not None and "spaces" in d.places.areas[aid]:
                r = math.radians(v.angle)
                v.x += math.sin(r) * v.length / 2
                v.y += math.cos(r) * v.length / 2
        on_drive = set(m.drives)
        if a.trace and a.trace in cur.vehicles:
            v_ = cur.vehicles[a.trace]
            dv_ = m.drives.get(a.trace)
            st.setdefault("trace", []).append([round(cur.time, 1), "sumo", round(v_.x, 1), round(v_.y, 1),
                                               d.parked.get(a.trace), dv_ and dv_.kind, dv_ and dv_.why])
        m.step(cur.time, cur.vehicles)
        if a.trace and a.trace in cur.vehicles:
            v_ = cur.vehicles[a.trace]
            st["trace"][-1] += ["shown", round(v_.x, 1), round(v_.y, 1)]
        on_drive |= set(m.drives)
        # (9 Nov) how close the cars on drives come to each other and to the road's cars (bodies as circles)
        if m.drives:
            from check_parking_crowd import circles, clearance
            bodies = {}
            for vid, v in cur.vehicles.items():
                if abs(v.x) > 1e5 or (vid in d.parked and vid not in m.drives):
                    continue
                a_ = math.radians(v.angle)
                bodies[vid] = circles(v.x - math.sin(a_) * v.length / 2, v.y - math.cos(a_) * v.length / 2,
                                      v.angle, v.length)
            for vid in m.drives:
                b = bodies.get(vid)
                if b is None:
                    continue
                for oid, ob in bodies.items():
                    if oid == vid or (oid in m.drives and oid < vid) or math.dist(b[1], ob[1]) > 12.0:
                        continue
                    g = clearance(b, ob)
                    if g < -0.2:
                        st.setdefault("overlaps", []).append((round(g, 2), vid, oid, round(cur.time, 1),
                                                              m.drives[vid].kind, oid in m.drives,
                                                              m.drives[vid].area))
        last, nxt = st["last"], {}
        for vid in on_drive | {v for v in last if v in cur.vehicles}:
            v = cur.vehicles.get(vid)
            if v is None:
                continue
            h = math.radians(v.angle)
            mid = (v.x - math.sin(h) * v.length / 2, v.y - math.cos(h) * v.length / 2)
            if abs(v.x) > 1e5 or abs(v.y) > 1e5:             # (SUMO has no place for it at this step)
                st.setdefault("no_place", []).append((vid, round(cur.time, 1), vid in m.drives, d.parked.get(vid),
                                                      vid in d.plans, vid in d.resume_at))
                continue
            if vid in last:
                step = math.dist(mid, last[vid][0])
                st["drive_steps" if (vid in m.drives or last[vid][1]) else "steps"].append(
                    (round(step, 2), vid, round(cur.time, 1)))
            if vid in on_drive or vid in last:
                nxt[vid] = (mid, vid in m.drives)
        # keep the cars of a drive (and 4 steps after it) in view
        st["last"] = {k: v for k, v in nxt.items() if v[1] or k in on_drive}

    args = {"minutes": a.hours * 3600.0 / GAME_SPEED / 60.0, "warmup": 1.0, "start_hour": a.start,
            "clock_speed": GAME_SPEED, "dt": 30.0, "controller": "agents_observed", "scale": 1.0, "accidents": True,
            "careless_share": None, "buses": True, "ring": True, "junction_blocker": 60.0, "trips": "purpose"}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    work = ROOT / "out" / "parking_motion_check"
    (work / "runs").mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        gen.one_run(a.seed, args, nodes, str(work), a.port, probe=probe)
    finally:
        live.Population.__init__ = orig_init
    m = st["motion"]
    ds = sorted(st["drive_steps"], reverse=True)
    res = {"seed": a.seed, "start": a.start, "hours": a.hours, "stats": m.stats if m else None,
           "drive_step_max_m": ds[0][0] if ds else None, "drive_steps_over_2m": [x for x in ds if x[0] > 2.0][:20],
           "overlap_steps": len(st.get("overlaps", [])), "overlaps": st.get("overlaps", [])[:40],
           "overlaps_by": _count_by(st.get("overlaps", [])),
           "seams": m.seams if m else None, "no_place": st.get("no_place", [])[:20],
           "trace": st.get("trace"), "wall_s": round(time.time() - t)}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "seams"}, indent=1))
    print("seams over 0.5 m:", [s for s in (m.seams if m else []) if s["seam_m"] > 0.5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
