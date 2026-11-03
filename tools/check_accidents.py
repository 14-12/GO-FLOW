"""Accidents end to end, SUMO alone (no CARLA): the viewer's traffic and options at the evening rush, the careless
drivers and the response of cosim/incidents.py, for --minutes. Prints each incident's timeline (crash, police
sent, on scene, lane closed, recovery trucks, lifted, reopened) and the figures: incidents, collisions, near
misses, response times, and vehicles stopped over 2 minutes that are not part of an incident (a jam the
incident caused that does not clear).

    python tools/check_accidents.py [--minutes 20] [--hour 17.5] [--json out/check_accidents.json]

31 Oct: --ring adds the ring road's own cars (cosim/highway.py, where pile-ups happen) and --buses the three bus lines
(their long stops are look-alikes); the figures then count the breakdowns, pile-ups, speeders and look-alikes too
(cosim/incidents.py). --no-new leaves those out (the incidents as before 31 Oct).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import daily, vmodels  # noqa: E402
from cosim import scenario as scenario_mod  # noqa: E402
from cosim.blueprints import Candidate  # noqa: E402
from cosim.incidents import IncidentManager  # noqa: E402
from cosim.lanegeo import LaneGeometry  # noqa: E402
from cosim.live import Population  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402

VIEWER_SUMO_ARGS = ["--lateral-resolution", "0.8", "--collision.action", "warn", "--collision.check-junctions", "true",
                    "--collision.mingap-factor", "0"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--hour", type=float, default=17.5)
    ap.add_argument("--port", type=int, default=8995)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--share", type=float, default=None, help="careless share of new cars (default cosim/incidents)")
    ap.add_argument("--no-incidents", action="store_true", help="no incident manager (collisions left to SUMO)")
    ap.add_argument("--lc", choices=["sublane", "old"], default="sublane",
                    help="old: the viewer's lane changes before 23 Oct (--lanechange.duration 2, no sublane model)")
    ap.add_argument("--no-unjam", dest="unjam", action="store_false",
                    help="leave cars standing in a knot (the viewer takes them off out of sight after 150 s)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--ring", action="store_true", help="the ring road's own cars, as in the viewer")
    ap.add_argument("--buses", action="store_true", help="the bus lines, as in the viewer")
    ap.add_argument("--no-new", dest="new", action="store_false",
                    help="no breakdowns, speeders, pile-ups or look-alikes (the incidents before 31 Oct)")
    ap.add_argument("--no-breakdowns", dest="breakdowns", action="store_false")
    ap.add_argument("--no-lookalikes", dest="lookalikes", action="store_false")
    ap.add_argument("--police-delay", default=None, metavar="MEDIAN[:SPREAD]",
                    help="seconds from a crash to the police car being sent (cosim/freefly.py --police-delay; default "
                         "240:0.5, 3 Nov; 6:0 as before)")
    a = ap.parse_args(argv)
    scen = scenario_mod.load("town05", ROOT)
    pool = daily.load_pool(scen.folder)
    models = vmodels.load()
    pool.vtypes = {t: vmodels.with_attrs(vt, vmodels.LANE_CHANGE_ATTRS) for t, vt in pool.vtypes.items()}
    out_dir = ROOT / "out" / "freefly"
    pf = out_dir / f"check_accidents_{a.port}.rou.xml"
    rids = daily.write_pool_routes(pool, pf)
    et = vmodels.extras_types_file(models, out_dir / f"extras_types_{a.port}.add.xml", vmodels.LANE_CHANGE_ATTRS)
    curve = daily.population_curve(None, pool, mean_trip_min=140 / 60.0)
    from cosim.freefly import BusLines

    sumo = SumoSide(scen.folder, "actuated", seed=a.seed, port=a.port, label=f"acc{a.port}", extras=True,
                    additional=BusLines.additional(scen.folder, "town05") if a.buses else [],
                    emergency_every=45, extras_types=et, lane_geometry=LaneGeometry(scen.net_file),
                    extra_args=["--route-files", str(pf), "--end", "100000000"]
                    + (VIEWER_SUMO_ARGS if a.lc == "sublane" else ["--lanechange.duration", "2", "--collision.action", "warn",
                                                                  "--collision.check-junctions", "true"]))
    c = sumo.conn
    pop = Population(c, pool, rids, curve, seed=a.seed)
    cands = [Candidate(m, i["base_type"].lower(), i["special_type"].lower(), int(i["generation"] or 1), ("0",),
                       i["has_lights"]) for m, i in models.items()]
    base = {t: (re.search(r'vClass="([^"]+)"', vt) or [None, "passenger"])[1] for t, vt in pool.vtypes.items()}
    base.update({t: vc for t, vc in vmodels.EMERGENCY_TYPES.items()})
    typer = vmodels.Typer(c, cands, models, base)
    sinks = sorted({r.split()[-1] for routes in pool.routes.values() for r in routes})
    kw = {} if a.share is None else {"share": a.share}
    if not a.new:
        kw.update(breakdown_mean_s=0.0, speeder_share=0.0, pileups=False, lookalikes=False)
    if not a.breakdowns:
        kw["breakdown_mean_s"] = 0.0
    if not a.lookalikes:
        kw["lookalikes"] = False
    from cosim.incidents import police_delay_args

    kw.update(police_delay_args(a.police_delay))
    im = IncidentManager(c, scen.net_file, typer, seed=a.seed, walk=sumo.walk, sinks=sinks, **kw)
    pop.type_for = im.type_for
    sod = a.hour * 3600
    bus = ring = None
    if a.buses:
        bus_types = [t for t, vt in pool.vtypes.items() if 'vClass="bus"' in vt]
        bus = BusLines(c, scen.folder, "town05", bus_types[0], type_for=typer.type_for)
        for i, t in enumerate(pop.types):  # buses run the lines, not random trips
            if t in bus_types:
                pop.weights[i] = 0
    if a.ring:
        import sumolib

        from cosim.highway import HighwayTraffic, find_loops

        net = sumolib.net.readNet(str(scen.net_file))
        loops = find_loops(net)
        top = max(curve) or 1.0
        ring = HighwayTraffic(c, loops, lambda s_: daily.target_at(curve, s_) / top,
                              types=[(t, n) for t, n in pool.per_hour.items() if base.get(t) != "bus"],
                              type_for=pop.type_for, seed=a.seed + 3,
                              lengths={e: net.getEdge(e).getLength() for lp in loops for e in lp}, policy=sumo.policy)
        ring.protected = im.protected
        ring.setup()
    k = 0
    pop.add(int(pop.target(sod)), anywhere=True)
    stopped: dict[str, float] = {}
    long_stops: dict[str, float] = {}
    stop_where: dict[str, tuple] = {}
    # around each incident, while it is on (police on scene to reopened): its edge's open lanes 0-80 m before the
    # wrecks (are drivers moving past?), anyone not part of it on the closed lane near the wrecks, and sideways slides
    # of standing cars there (lateral moves at a standstill look like glitches)
    scene: dict[int, dict] = {}
    lat_prev: dict[str, float] = {}
    t_wall = time.time()
    end = sumo.time + 60.0 * a.minutes
    while sumo.time < end:
        cur = sumo.step()
        k += 1
        pop.step(cur.vehicles, sod, lambda vid, v: not im.protected(vid) and not vid.startswith("pt_"))
        if ring is not None and k % 4 == 0:
            ring.step(cur.vehicles, sod, set(c.simulation.getPendingVehicles()))
        if bus is not None and k % 2 == 0:
            bus.step(cur.time, a.hour)
            for vid in [v for v in cur.vehicles if v.startswith("pt_")]:
                try:
                    route, idx = c.vehicle.getRoute(vid), c.vehicle.getRouteIndex(vid)
                except Exception:  # noqa: BLE001
                    continue
                if idx >= len(route) - 3:
                    bus.next_lap(vid, idx, route)
        if not a.no_incidents:
            im.step(cur.time, cur.vehicles, cur.tls)
        if a.unjam and int(cur.time * 2) % 10 == 0:
            im.unjam(cur.time, cur.vehicles)
        else:
            for col in c.simulation.getCollisions():
                im._pairs.add((col.collider, col.victim))
        for inc in im.incidents:
            if inc.phase not in ("secured", "recovery") or not inc.open_lane:
                continue
            lane, rear, front = im._anchor(inc)
            if not lane:
                continue
            m = scene.setdefault(inc.id, {"samples": 0, "standing": 0, "passed": set(), "closed_intruders": set(),
                                          "standing_slides": 0, "seconds": 0.0, "stand_since": {}, "longest_stand_s": 0.0,
                                          "free_samples": 0, "free_standing": 0})
            m["seconds"] += 0.5
            holding = bool(inc.held) or any(h[0] <= cur.time < h[1] for h in inc.holds)  # the police stop the traffic
            edge = lane.rsplit("_", 1)[0]
            role = set(im.role) | {t.vid for t in inc.tows}
            for ln in [f"{edge}_{k}" for k in range(c.edge.getLaneNumber(edge))]:
                for o in c.lane.getLastStepVehicleIDs(ln):
                    if o in role or o not in cur.vehicles:
                        continue
                    p = c.vehicle.getLanePosition(o)
                    sp = cur.vehicles[o].speed
                    lat = c.vehicle.getLateralLanePosition(o)
                    if sp < 0.3 and o in lat_prev and abs(lat - lat_prev[o]) > 0.05 and rear - 80 < p < front + 20:
                        m["standing_slides"] += 1
                    lat_prev[o] = lat
                    if ln in [cl for cl, _ in inc.closed]:
                        if rear - 25.0 < p < front + 15.0:
                            m["closed_intruders"].add(o)
                        continue
                    if rear - 80.0 < p < rear:
                        nt = c.vehicle.getNextTLS(o)
                        if sp < 0.5 and nt and nt[0][3] in "rRyu" and nt[0][2] < 80.0:
                            continue  # queuing at a red light just ahead: not the accident's doing
                        m["samples"] += 1
                        if not holding:
                            m["free_samples"] += 1
                            m["free_standing"] += sp < 0.5
                        if sp < 0.5:
                            m["standing"] += 1
                            t0s = m["stand_since"].setdefault(o, cur.time)
                            m["longest_stand_s"] = max(m["longest_stand_s"], cur.time - t0s)
                        else:
                            m["stand_since"].pop(o, None)
                    elif p > front + 5.0:
                        m["passed"].add(o)
        for vid, v in cur.vehicles.items():
            if v.speed < 0.1:
                stopped.setdefault(vid, cur.time)
                if cur.time - stopped[vid] > 120.0 and not im.protected(vid) and not vid.startswith("pt_"):
                    if vid not in long_stops:
                        try:
                            stop_where[vid] = (c.vehicle.getLaneID(vid), round(cur.time))
                        except Exception:  # noqa: BLE001
                            pass
                    long_stops[vid] = cur.time - stopped[vid]
            else:
                stopped.pop(vid, None)
    where: dict = {}
    for vid in long_stops:
        try:
            ln = c.vehicle.getLaneID(vid)
            where[ln] = where.get(ln, 0) + 1
        except Exception:  # noqa: BLE001
            pass
    sumo.close()
    incs = im.done + im.incidents

    def at(i, text):
        return next((t for t, e in i.events if text in e), None)

    rows = []
    for i in incs:
        rows.append({"id": i.id, "t0": i.t0, "kind": i.kind, "label": i.label, "kmh": round(i.severity * 3.6),
                     "edge": i.edge, "lane": i.lane, "x": round(i.x, 1), "y": round(i.y, 1),
                     "wrecks": len(i.wrecks), "phase": i.phase, "cause": getattr(i, "cause", None), "police_s": at(i, "police on scene"),
                     "closed": bool(at(i, "lane closed")), "last_load_s": max((t for t, e in i.events if "loaded onto" in e),
                                                                             default=None),
                     "reopened_s": at(i, "lane reopened"), "t_contact": i.t_contact, "police": dict(i.police_record),
                     "events": i.events})
    done = [r for r in rows if r["reopened_s"] is not None]
    pol = [r["police"] for r in rows if r["police"]]
    res = {"variant": {"lc": a.lc, "incidents": not a.no_incidents, "share": a.share, "police_delay": a.police_delay},
           "collisions_seen": len(im._pairs),
           # (3 Nov) the police: sent and on scene, seconds after the crash; incidents whose police never got there
           "police_sent_after_s": sorted(round(p["sent"] - r["t0"]) for r, p in ((r, r["police"]) for r in rows)
                                         if p and p.get("sent") is not None),
           "police_on_scene_after_s": sorted(round(p["on_scene"] - r["t0"]) for r, p in ((r, r["police"]) for r in rows)
                                             if p and p.get("on_scene") is not None),
           "police_never_on_scene": sum(1 for p in pol if p.get("on_scene") is None),
           "police_how": {h: sum(1 for p in pol if str(p.get("how")) == h) for h in sorted({str(p.get("how")) for p in pol})},
           "minutes": a.minutes, "hour": a.hour, "wall_s": round(time.time() - t_wall), "stats": im.stats, "near_misses_by_reason": im.near_why,
           "incidents": len(incs), "completed": len(done),
           "police_on_scene_s": sorted(r["police_s"] for r in rows if r["police_s"] is not None),
           "cleared_s": sorted(r["reopened_s"] for r in done),
           "stopped_over_2min_not_incident": len(long_stops), "longest_stop_s": round(max(long_stops.values(), default=0)),
           "long_stops_where": sorted(stop_where.values(), key=lambda x: x[1]),
           "stuck_lanes_at_end": dict(sorted(where.items(), key=lambda kv: -kv[1])[:12]), "squeezed": len(im.squeezed),
           "scenes": {k: {"seconds": m["seconds"], "open_lane_standing_share": round(m["standing"] / max(m["samples"], 1), 3),
                          "standing_share_waved_on": round(m["free_standing"] / max(m["free_samples"], 1), 3),
                          "longest_stand_s": m["longest_stand_s"],
                          "passed_per_min": round(60.0 * len(m["passed"]) / max(m["seconds"], 1.0), 1),
                          "closed_lane_intruders": len(m["closed_intruders"]), "standing_slides": m["standing_slides"]}
                      for k, m in scene.items()},
           "rows": rows,
           "by_kind": {kd: sum(1 for r in rows if r["kind"] == kd) for kd in sorted({r["kind"] for r in rows})},
           "speeding_crashes": sum(1 for i in incs if any(w.speeding for w in i.wrecks)),
           "lookalikes": im.lookalike_snapshot()}
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
