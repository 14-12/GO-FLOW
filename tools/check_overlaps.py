"""Do the free-fly viewer's vehicles drive into each other, or slide sideways? SUMO alone on Town05 (the viewer's
rush-hour demand and options), every vehicle posed 30 times a SUMO step as the pose process does
(cosim/pose_proc.FramePoser: Catmull-Rom along SUMO's positions, heading from the path, Trails), then:

* overlaps: two vehicles' footprints (SUMO length and width, or the CARLA model's with --carla-dims) cutting
  into each other by more than --depth metres, counted as episodes (a pair, until they part), with where it
  happened (a junction, a lane change, a queue) and how deep;
* sideways while standing: a vehicle under 0.5 m/s whose front moved over 5 cm across its heading in a step
  (a car can't move sideways on the spot);
* SUMO's own collisions (--collision.check-junctions, reported, not acted on).

    python tools/check_overlaps.py [--seconds 300] [--variant base] [--json out.json]

Variants change only SUMO's options, to compare (see VARIANTS).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import traci.constants as tc  # noqa: E402

from cosim import daily  # noqa: E402
from cosim import scenario as scenario_mod  # noqa: E402
from cosim.interp import Timed, Trails  # noqa: E402
from cosim.lanegeo import LaneGeometry  # noqa: E402
from cosim.live import Population  # noqa: E402
from cosim.mapping import front_to_centre  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402

SUB = 30  # poses per SUMO step (60 a second at 0.5 s)
VARIANTS = {
    "base": ["--lanechange.duration", "2"],
    "instant": [],
    "lc3": ["--lanechange.duration", "3"],
    "sublane": ["--lateral-resolution", "0.8"],
    # the viewer since 23 Oct (cosim/freefly.py): sublane, cosim/vmodels.LANE_CHANGE_ATTRS on every type, each
    # vehicle typed and sized as its CARLA model
    "viewer": ["--lateral-resolution", "0.8"],
}


def corners(cx, cy, ang_deg, length, width):
    a = math.radians(ang_deg)
    fx, fy = math.sin(a), math.cos(a)   # SUMO heading: forward
    lx, ly = -fy, fx
    hl, hw = length / 2, width / 2
    return [(cx + fx * hl + lx * hw, cy + fy * hl + ly * hw), (cx + fx * hl - lx * hw, cy + fy * hl - ly * hw),
            (cx - fx * hl - lx * hw, cy - fy * hl - ly * hw), (cx - fx * hl + lx * hw, cy - fy * hl + ly * hw)]


def penetration(p, q) -> float:
    """Separating-axis test of two rectangles: 0 if apart, else the smallest overlap along any axis (m)."""
    best = 1e9
    for poly in (p, q):
        for i in range(4):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % 4]
            ax, ay = y0 - y1, x1 - x0
            n = math.hypot(ax, ay) or 1.0
            ax, ay = ax / n, ay / n
            pa = [x * ax + y * ay for x, y in p]
            qa = [x * ax + y * ay for x, y in q]
            o = min(max(pa), max(qa)) - max(min(pa), min(qa))
            if o <= 0:
                return 0.0
            best = min(best, o)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--warmup", type=float, default=200.0)
    ap.add_argument("--hour", type=float, default=17.5)
    ap.add_argument("--port", type=int, default=8977)
    ap.add_argument("--variant", default="base", choices=sorted(VARIANTS))
    ap.add_argument("--extra", default="", help="more SUMO options, space separated")
    ap.add_argument("--depth", type=float, default=0.15)
    ap.add_argument("--json", default=None)
    ap.add_argument("--vtype", default="", help='attributes added to every vehicle type, e.g. "lcMaxSpeedLatStanding=0"')
    ap.add_argument("--trace", default="", help="vehicle ids (comma separated): their poses, step by step, to stderr")
    ap.add_argument("--carla-dims", default=None, help="JSON {sumo type id or vclass: [length, width]}")
    a = ap.parse_args(argv)
    scen = scenario_mod.load("town05", ROOT)
    pool = daily.load_pool(scen.folder)
    if a.vtype:
        add = " ".join(f'{kv.split("=")[0]}="{kv.split("=")[1]}"' for kv in a.vtype.split())
        pool.vtypes = {t: vt.replace("<vType ", f"<vType {add} ", 1) for t, vt in pool.vtypes.items()}
    from cosim import vmodels

    models = vmodels.load()
    extras_types = None
    if a.variant == "viewer":
        pool.vtypes = {t: vmodels.with_attrs(vt, {k: v for k, v in vmodels.LANE_CHANGE_ATTRS.items() if f'{k}=' not in vt})
                       for t, vt in pool.vtypes.items()}
        extras_types = vmodels.extras_types_file(models, ROOT / "out" / "freefly" / f"extras_types_{a.port}.add.xml",
                                                 vmodels.LANE_CHANGE_ATTRS)
    pf = ROOT / "out" / "freefly" / f"check_overlaps_{a.port}.rou.xml"
    rids = daily.write_pool_routes(pool, pf)
    curve = daily.population_curve(None, pool, mean_trip_min=140 / 60.0)
    sod = a.hour * 3600
    extra = VARIANTS[a.variant] + (a.extra.split() if a.extra else [])
    sumo = SumoSide(scen.folder, "actuated", seed=7, port=a.port, label="overlaps", extras=True, emergency_every=45,
                    extra_args=["--route-files", str(pf), "--end", "100000000", "--collision.action", "warn",
                                "--collision.check-junctions", "true"] + extra,
                    vehicle_vars=[tc.VAR_LANE_ID, tc.VAR_LANEPOSITION_LAT], extras_types=extras_types,
                    lane_geometry=LaneGeometry(scen.net_file) if a.variant == "viewer" else None)
    c = sumo.conn
    pop = Population(c, pool, rids, curve, seed=7)
    if a.variant == "viewer":
        import re

        from cosim.blueprints import Candidate

        cands = [Candidate(m, i["base_type"].lower(), i["special_type"].lower(), int(i["generation"] or 1), ("0,0,0",),
                           i["has_lights"]) for m, i in models.items()]
        base = {t: (re.search(r'vClass="([^"]+)"', vt) or [None, "passenger"])[1] for t, vt in pool.vtypes.items()}
        pop.type_for = vmodels.Typer(c, cands, models, base).type_for
    pop.add(int(pop.target(sod)), anywhere=True)
    steps, widths, lanes = [], {}, []
    sumo_collisions = []
    while sumo.time < a.warmup + a.seconds:
        cur = sumo.step()
        pop.step(cur.vehicles, sod, lambda vid, v: True)
        for vid in cur.vehicles:
            if vid not in widths:
                widths[vid] = c.vehicle.getWidth(vid)
        if sumo.time >= a.warmup:
            steps.append(cur)
            lanes.append({vid: (r.get(tc.VAR_LANE_ID, ""), r.get(tc.VAR_LANEPOSITION_LAT, 0.0))
                          for vid, r in sumo.vehicle_subs.items()})
            for col in c.simulation.getCollisions():
                sumo_collisions.append({"t": sumo.time, "a": col.collider, "b": col.victim, "type": col.type,
                                        "lane": col.lane, "speeds": [round(col.colliderSpeed, 1),
                                                                     round(col.victimSpeed, 1)]})
    sumo.close()
    dims = json.loads(Path(a.carla_dims).read_text()) if a.carla_dims else {}

    import sumolib

    net = sumolib.net.readNet(str(scen.net_file), withInternal=True)

    def nodes_of(lane: str) -> set:
        if not lane:
            return set()
        if lane.startswith(":"):
            return {lane[1:].rsplit("_", 2)[0]}
        try:
            e = net.getEdge(lane.rsplit("_", 1)[0])
        except KeyError:
            return set()
        return {e.getFromNode().getID(), e.getToNode().getID()}

    trails = Trails()
    open_pairs: dict[tuple, dict] = {}
    episodes = []
    sideways = []
    for k in range(1, len(steps) - 1):
        prev, cur = steps[k - 1].vehicles, steps[k].vehicles
        pre = steps[k - 2].vehicles if k >= 2 else {}
        nxt = steps[k + 1].vehicles
        timed = {}
        for vid, v in cur.items():
            p = prev.get(vid)
            if p is None:
                continue
            p1, p2 = (p.x, p.y), (v.x, v.y)
            q0, q3 = pre.get(vid), nxt.get(vid)
            p0 = (q0.x, q0.y) if q0 is not None else (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])
            p3 = (q3.x, q3.y) if q3 is not None else (2 * p2[0] - p1[0], 2 * p2[1] - p1[1])
            timed[vid] = Timed(p0, p1, p2, p3)
            # sideways while standing
            if max(p.speed, v.speed) < 0.5:
                h = math.radians(p.angle)
                side = (v.x - p.x) * math.cos(h) - (v.y - p.y) * math.sin(h)
                if abs(side) > 0.05:
                    sideways.append({"t": steps[k].time, "veh": vid, "side_m": round(side, 3),
                                     "lane": lanes[k].get(vid, ("", 0))[0]})
        for i in range(SUB):
            s = i / SUB
            boxes = {}
            for vid, tm in timed.items():
                v = cur[vid]
                x, y = tm.at(s)
                la = v.lane_angle
                pv = prev.get(vid)
                if la is not None and pv is not None and pv.lane_angle is not None:  # as the pose process: blended
                    la = pv.lane_angle + ((la - pv.lane_angle + 180.0) % 360.0 - 180.0) * s
                ang = trails.update(vid, x, y, v.length, v.angle, la, (pv.speed if pv is not None else v.speed) * (1 - s) + v.speed * s)
                if a.trace and vid in a.trace.split(",") and i % 15 == 0:
                    print(f"TRACE {vid} t={steps[k].time + s * 0.5:.2f} x={x:.2f} y={y:.2f} v={v.speed:.2f} sumo={v.angle:.1f} "
                          f"trail={ang:.1f} lane={lanes[k].get(vid, ('', 0))}", file=sys.stderr)
                length, width = dims.get(v.type_id, dims.get(v.vclass, (v.length, widths.get(vid, 1.8))))
                cx, cy = front_to_centre(x, y, ang, v.length / 2)  # the pose's centre: SUMO's length
                boxes[vid] = (cx, cy, corners(cx, cy, ang, length, width))
            ids = sorted(boxes)
            grid: dict[tuple, list] = {}
            for vid in ids:
                cx, cy, _ = boxes[vid]
                grid.setdefault((int(cx // 15), int(cy // 15)), []).append(vid)
            now_pairs = set()
            for (gx, gy), members in grid.items():
                near = [w for dx in (-1, 0, 1) for dy in (-1, 0, 1) for w in grid.get((gx + dx, gy + dy), [])]
                for va in members:
                    for vb in near:
                        if vb <= va:
                            continue
                        ax, ay, pa = boxes[va]
                        bx, by, pb = boxes[vb]
                        if (ax - bx) ** 2 + (ay - by) ** 2 > 200:
                            continue
                        d = penetration(pa, pb)
                        if d > a.depth and not (nodes_of(lanes[k].get(va, ("", 0))[0]) and nodes_of(lanes[k].get(vb, ("", 0))[0])
                                                and not nodes_of(lanes[k].get(va, ("", 0))[0]) & nodes_of(lanes[k].get(vb, ("", 0))[0])):
                            key = (va, vb)
                            now_pairs.add(key)
                            ep = open_pairs.get(key)
                            if ep is None:
                                la, lb = lanes[k].get(va, ("", 0)), lanes[k].get(vb, ("", 0))
                                ep = open_pairs[key] = {"t": steps[k].time + s * 0.5, "a": va, "b": vb,
                                                        "lanes": [la[0], lb[0]], "lat": [round(la[1], 2), round(lb[1], 2)],
                                                        "speeds": [round(cur[va].speed, 1), round(cur[vb].speed, 1)],
                                                        "depth": 0.0, "frames": 0}
                            ep["depth"] = max(ep["depth"], round(d, 2))
                            ep["frames"] += 1
            for key in [p for p in open_pairs if p not in now_pairs]:
                episodes.append(open_pairs.pop(key))
    episodes += list(open_pairs.values())

    def where(ep):
        la, lb = ep["lanes"]
        if la.startswith(":") or lb.startswith(":"):
            return "junction"
        if abs(ep["lat"][0]) > 0.3 or abs(ep["lat"][1]) > 0.3:
            return "lane change"
        return "same road"

    by = {}
    for ep in episodes:
        by.setdefault(where(ep), []).append(ep)
    out = {"variant": a.variant, "extra": extra, "vtype": a.vtype, "seconds": a.seconds, "vehicles_seen": len(widths),
           "overlap_episodes": len(episodes),
           "overlaps_by_place": {w: len(v) for w, v in by.items()},
           "overlap_depth_max": max((e["depth"] for e in episodes), default=0),
           "overlaps_over_0.5m": sum(1 for e in episodes if e["depth"] > 0.5),
           "sideways_standing": len(sideways),
           "sumo_collisions": len(sumo_collisions),
           "sumo_collision_types": {t: sum(1 for x in sumo_collisions if x["type"] == t)
                                    for t in {x["type"] for x in sumo_collisions}},
           "worst": sorted(episodes, key=lambda e: -e["depth"])[:12],
           "sideways_examples": sideways[:10], "sumo_collision_examples": sumo_collisions[:10]}
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
