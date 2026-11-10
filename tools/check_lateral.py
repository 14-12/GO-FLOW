"""Do the viewer's vehicles slide sideways? The viewer's town in SUMO alone (tools/goflow_prediction_dataset.one_run:
the day's traffic, trips with a purpose, buses, the ring road, accidents, the junction agents), every vehicle posed 30
times a SUMO step as the viewer's pose process does (Catmull-Rom along SUMO's front positions, the body's heading
from its path, interp.Trails), then the body's rear axle followed from pose to pose.

A car's rear wheels roll, they do not slide: the rear axle moves along the body, never across it. Here the rear axle
is taken 0.75 x length behind the front; its speed across the body, averaged over 0.25 s, is the slide. An episode is
a vehicle sliding over --slide m/s, its rear axle moving more than --crab degrees off the body's axis, for at least
0.25 s; each is put down with what the vehicle was doing (a lane
change, a drift inside its lane, standing), what was around it (an accident's wreck or vehicle, a bus at its stop, a
junction) and where, so that the worst can be looked at in CARLA.

    python tools/check_lateral.py --seed 11 --start 7.5 --minutes 10 --out out/lateral/s11.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

SUB = 30            # poses a SUMO step (60 a second)
STEP_S = 0.5
WIN = 15            # poses averaged (0.25 s)
PARK_S = 20.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--start", type=float, default=7.5, help="town hour at the start (clock 30)")
    ap.add_argument("--warmup", type=float, default=3.0, help="SUMO minutes not looked at")
    ap.add_argument("--minutes", type=float, default=10.0, help="SUMO minutes looked at")
    ap.add_argument("--controller", default="agents_observed")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--slide", type=float, default=0.2, help="m/s across the body")
    ap.add_argument("--crab", type=float, default=8.0, help="deg between the rear axle's motion and the body")
    ap.add_argument("--port", type=int, default=8990)
    ap.add_argument("--rear", type=float, default=0.75, help="the rear axle, behind the front, x length")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None, help="keep the run's steps (pickle) to look at them again with --load")
    ap.add_argument("--load", default=None)
    a = ap.parse_args(argv)

    import goflow_prediction_dataset as gpd

    from cosim.interp import Timed, Trails

    nodes = json.loads((ROOT / "models/goflow/graph/nodes.json").read_text())
    args = {"minutes": a.minutes, "warmup": a.warmup, "start_hour": a.start, "clock_speed": 30.0, "dt": 30.0,
            "controller": a.controller, "scale": a.scale, "accidents": True, "careless_share": None, "buses": True,
            "ring": True, "junction_blocker": 0, "trips": "purpose"}
    steps = []      # per step: {vid: (x, y, angle, speed, length, lane_angle, lane, lat, ctx)}
    st = {"im": None, "t0": None}

    def hook(sumo, im):
        st["im"] = im

    def probe(cur, c, sod):
        if st["t0"] is None:
            st["t0"] = cur.time
        if cur.time - st["t0"] < 60.0 * a.warmup - 2 * STEP_S:
            return
        im = st["im"]
        role = im.role if im is not None else {}
        squeezed = getattr(im, "squeezed", {}) if im is not None else {}
        # what stands in the way: wrecks and incident vehicles, buses at a stop
        blockers = []
        for vid, v in cur.vehicles.items():
            if vid in role:
                blockers.append(("accident", v.x, v.y))
            elif vid.startswith("pt_") and v.speed < 0.3:
                blockers.append(("bus", v.x, v.y))
        snap = {}
        for vid, v in cur.vehicles.items():
            try:
                lane = c.vehicle.getLaneID(vid)
                lat = c.vehicle.getLateralLanePosition(vid)
            except Exception:  # noqa: BLE001
                lane, lat = "", 0.0
            ctx = []
            if vid in role:
                ctx.append("incident vehicle")
            if vid in squeezed:
                ctx.append("squeezed")
            for kind, bx, by in blockers:
                if (bx - v.x) ** 2 + (by - v.y) ** 2 < 25.0 ** 2:
                    ctx.append("near " + kind)
                    break
            snap[vid] = (v.x, v.y, v.angle, v.speed, v.length, v.lane_angle, lane, lat, tuple(ctx), v.vclass)
        steps.append((cur.time, snap))

    import pickle

    if a.load:
        steps, res = pickle.loads(Path(a.load).read_bytes())
    else:
        work = Path(a.out).parent / f"_work_{a.seed}_{a.port}"
        (work / "runs").mkdir(parents=True, exist_ok=True)
        res = gpd.one_run(a.seed, args, nodes, str(work), a.port, probe=probe, hook=hook)
        if a.save:
            Path(a.save).write_bytes(pickle.dumps((steps, res)))

    # a car driven into its parking space or out of it: the viewer poses it on its own (cosim/parking_motion.py,
    # a heading given); left out here within PARK_S of being off the road
    off_road: dict[str, list] = {}
    for t, snap in steps:
        for vid, v in snap.items():
            if not v[6]:
                off_road.setdefault(vid, []).append(t)

    def parking_move(vid, t):
        return any(abs(t - u) <= PARK_S for u in off_road.get(vid, ()))

    trails = Trails()
    hist: dict[str, list] = {}   # vid -> recent (t, rx, ry, heading, fwd speed)
    open_ep: dict[str, dict] = {}
    episodes = []
    poses = 0
    yaw, yaw_acc, yaw_flip, boxes = {}, [], [0], {}

    def close(vid):
        ep = open_ep.pop(vid, None)
        if ep is not None and ep["n"] >= WIN and not parking_move(vid, ep["t"]):
            ep["dur_s"] = round(ep["n"] / (SUB / STEP_S), 2)
            ep["slide_max"] = round(ep["slide_max"], 2)
            ep["side_m"] = round(ep["side_m"], 2)
            episodes.append(ep)

    for k in range(2, len(steps) - 1):
        t, cur = steps[k]
        prev, pre, nxt = steps[k - 1][1], steps[k - 2][1], steps[k + 1][1]
        for vid in [v for v in open_ep if v not in cur]:
            close(vid)
        for vid, v in cur.items():
            p = prev.get(vid)
            if p is None:
                continue
            p1, p2 = (p[0], p[1]), (v[0], v[1])
            q0, q3 = pre.get(vid), nxt.get(vid)
            p0 = (q0[0], q0[1]) if q0 is not None else (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])
            p3 = (q3[0], q3[1]) if q3 is not None else (2 * p2[0] - p1[0], 2 * p2[1] - p1[1])
            tm = Timed(p0, p1, p2, p3)
            h = hist.setdefault(vid, [])
            for i in range(SUB):
                s = i / SUB
                x, y = tm.at(s)
                la = v[5]
                if la is not None and p[5] is not None:
                    la = p[5] + ((la - p[5] + 180.0) % 360.0 - 180.0) * s
                sp = p[3] * (1 - s) + v[3] * s
                ang = trails.update(vid, x, y, v[4], v[2], la, sp)
                poses += 1
                yh = yaw.setdefault(vid, [])
                yh.append(ang)
                if len(yh) > 3:
                    del yh[0]
                if len(yh) == 3:
                    r1 = (yh[1] - yh[0] + 180.0) % 360.0 - 180.0
                    r2 = (yh[2] - yh[1] + 180.0) % 360.0 - 180.0
                    yaw_acc.append(abs(r2 - r1) * (SUB / STEP_S) ** 2)
                    if r1 * r2 < 0 and min(abs(r1), abs(r2)) * SUB / STEP_S > 5.0:
                        yaw_flip[0] += 1
                if i % 6 == 0:
                    cxx, cyy = x - math.sin(math.radians(ang)) * v[4] / 2, y - math.cos(math.radians(ang)) * v[4] / 2
                    boxes.setdefault((k, i), {})[vid] = (cxx, cyy, ang, v[4], 2.5 if v[9] == "bus" else 1.8, v[6])
                r = math.radians(ang)
                fx, fy = math.sin(r), math.cos(r)
                rx, ry = x - fx * a.rear * v[4], y - fy * a.rear * v[4]
                h.append((t - STEP_S + s * STEP_S, rx, ry, fx, fy))
                if len(h) > WIN + 1:
                    del h[0]
                if len(h) <= WIN:
                    continue
                t0, x0, y0, _, _ = h[0]
                dt = (t - STEP_S + s * STEP_S) - t0
                if dt <= 0:
                    continue
                dx, dy = (rx - x0) / dt, (ry - y0) / dt
                side = dx * fy - dy * fx          # across the body (m/s), + to the right
                fwd = dx * fx + dy * fy
                if math.hypot(dx, dy) > 25.0:     # put somewhere else (a car park, a reroute): not driving
                    h.clear()
                    close(vid)
                    continue
                crab = math.degrees(math.atan2(abs(side), abs(fwd)))
                if abs(side) > a.slide and crab > a.crab:
                    ep = open_ep.get(vid)
                    if ep is None:
                        lane, lat = v[6], v[7]
                        doing = ("standing" if max(p[3], v[3]) < 0.3 else
                                 "junction" if lane.startswith(":") else
                                 "lane change" if (p[6] and lane and p[6] != lane) or abs(lat) > 0.6 else
                                 "drift in lane")
                        ep = open_ep[vid] = {"t": round(t - STEP_S + s * STEP_S, 2), "veh": vid, "vclass": v[9],
                                             "x": round(x, 1), "y": round(y, 1), "lane": lane, "lat": round(lat, 2),
                                             "speed": round(v[3], 1), "doing": doing, "ctx": list(v[8]),
                                             "n": 0, "slide_max": 0.0, "side_m": 0.0, "fwd": round(fwd, 2)}
                    ep["n"] += 1
                    ep["slide_max"] = max(ep["slide_max"], abs(side))
                    ep["crab_max"] = round(max(ep.get("crab_max", 0.0), crab), 1)
                    ep["side_m"] += abs(side) * STEP_S / SUB
                    if p[6] and v[6] and p[6] != v[6] and ep["doing"] == "drift in lane":
                        ep["doing"] = "lane change"
                else:
                    close(vid)
    for vid in list(open_ep):
        close(vid)

    # overlaps: two bodies on roads (not inside a junction) cutting into each other over 0.15 m, counted per pair
    from check_overlaps import corners, penetration

    pairs = set()
    for _, bx in boxes.items():
        grid = {}
        for vid, b in bx.items():
            grid.setdefault((int(b[0] // 12), int(b[1] // 12)), []).append(vid)
        for (gx, gy), mem in grid.items():
            near = [w for ddx in (-1, 0, 1) for ddy in (-1, 0, 1) for w in grid.get((gx + ddx, gy + ddy), [])]
            for va in mem:
                A = bx[va]
                if not A[5] or A[5].startswith(":"):
                    continue
                for vb in near:
                    if vb <= va:
                        continue
                    B = bx[vb]
                    if not B[5] or B[5].startswith(":") or (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 > 150:
                        continue
                    if penetration(corners(*A[:5]), corners(*B[:5])) > 0.15:
                        pairs.add((va, vb))
    yaw_acc.sort()

    def group(key):
        out = {}
        for e in episodes:
            kk = key(e)
            d = out.setdefault(kk, {"episodes": 0, "seconds": 0.0, "side_m": 0.0})
            d["episodes"] += 1
            d["seconds"] = round(d["seconds"] + e["dur_s"], 1)
            d["side_m"] = round(d["side_m"] + e["side_m"], 1)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["seconds"]))

    vehicle_s = sum(len(s) for _, s in steps) * STEP_S
    out = {"seed": a.seed, "start": a.start, "minutes": a.minutes, "slide_threshold": a.slide, "crab_threshold": a.crab,
           "vehicle_seconds": round(vehicle_s), "poses": poses, "episodes": len(episodes),
           "slide_seconds": round(sum(e["dur_s"] for e in episodes), 1),
           "slide_seconds_per_vehicle_hour": round(3600 * sum(e["dur_s"] for e in episodes) / max(vehicle_s, 1), 1),
           "by_doing": group(lambda e: e["doing"]),
           "by_context": group(lambda e: ",".join(e["ctx"]) or "-"),
           "overlap_pairs": len(pairs),
           "yaw_acc_p99_p999": [round(yaw_acc[int(q * (len(yaw_acc) - 1))]) for q in (0.99, 0.999)] if yaw_acc else None,
           "yaw_reversals": yaw_flip[0],
           "by_class": group(lambda e: e["vclass"]),
           "by_speed": group(lambda e: "<0.3" if e["speed"] < 0.3 else "<3" if e["speed"] < 3 else "<8" if e["speed"] < 8 else ">=8"),
           "run": {k: res.get(k) for k in ("vehicles", "removed", "incidents", "gridlock")} if isinstance(res, dict) else None,
           "by_doing_speed": group(lambda e: e["doing"] + (" <0.3" if e["speed"] < 0.3 else " <3" if e["speed"] < 3 else " <8" if e["speed"] < 8 else " >=8")),
           "worst": sorted(episodes, key=lambda e: -e["side_m"])[:40], "all": episodes}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k not in ("worst", "all")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
