"""An accident from crash to clearance through the viewer's own code, without CARLA's server (23-24 Oct): SUMO alone with
the viewer's options and cosim/incidents.py, every step's snapshot given to cosim/incident_scene.IncidentScene (as
FreeFly.apply does), and what it sends played by cosim/pose_proc.Driver (as the pose process does), the poses
worked out with CARLA's Transform maths (the client library, no server; or a stand-in with the same maths).

Every vehicle within 150 m of an accident is posed, so the people and the recovery truck can be checked against
the traffic, at 10 poses a second:
* walkers inside a vehicle (a driver who got out into a car, an officer in a passing car): their feet inside a
  vehicle's body (its box less 15 cm), not their own car's while getting out;
* the recovery truck while it backs in and loads, against any other vehicle's body;
* where the drivers end up (on the pavement), where the officer stands, the wreck on the deck when loaded and the
  truck where SUMO has it when the loading ends.
Prints the figures and the timeline.

    python tools/check_incident_scene.py [--minutes 12] [--hour 17.5] [--share 0.3]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import daily, vmodels  # noqa: E402
from cosim import scenario as scenario_mod  # noqa: E402
from cosim.blueprints import Candidate  # noqa: E402
from cosim.incident_scene import IncidentScene, SceneWalkers  # noqa: E402
from cosim.incidents import IncidentManager  # noqa: E402
from cosim.lanegeo import LaneGeometry  # noqa: E402
from cosim.live import Population  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402

NEAR_M = 150.0


def inside(tf, ext, px: float, py: float, shrink: float = 0.15) -> bool:
    """(px, py) inside the box of half sizes ext (x, y) posed tf, less `shrink`."""
    yaw = math.radians(tf.rotation.yaw)
    dx, dy = px - tf.location.x, py - tf.location.y
    lx = dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return abs(lx) < ext[0] - shrink and abs(ly) < ext[1] - shrink


def boxes_overlap(a, ea, b, eb, shrink: float = 0.2) -> bool:
    """Two posed boxes (separating axes, 2D)."""
    def corners(tf, e):
        y = math.radians(tf.rotation.yaw)
        c, s = math.cos(y), math.sin(y)
        hx, hy = e[0] - shrink, e[1] - shrink
        return [(tf.location.x + c * u - s * v, tf.location.y + s * u + c * v) for u, v in
                ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))]
    pa, pb = corners(a, ea), corners(b, eb)
    for poly in (pa, pb):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            ra = [nx * x + ny * y for x, y in pa]
            rb = [nx * x + ny * y for x, y in pb]
            if max(ra) < min(rb) or max(rb) < min(ra):
                return False
    return True


def _foes(c, vid):
    try:
        return [(f[0], f[1]) for f in c.vehicle.getJunctionFoes(vid, 30.0)][:4]
    except Exception as e:  # noqa: BLE001
        return str(e)[:40]


def dump_queues(c, inc, vehicles, im=None) -> None:
    """The head of each queue standing within 200 m of an accident: where it stands, why (leader, signal, its next
    road and that road's lanes), for the diagnosis of a scene that does not move."""
    heads = {}
    for vid, v in vehicles.items():
        if v.speed > 0.3 or math.hypot(v.x - inc.x, v.y - inc.y) > 200.0:
            continue
        try:
            ln, p = c.vehicle.getLaneID(vid), c.vehicle.getLanePosition(vid)
        except Exception:  # noqa: BLE001
            continue
        if ln not in heads or p > heads[ln][1]:
            heads[ln] = (vid, p)
    print(f"--- incident {inc.id} at {c.simulation.getTime() - inc.t0:.0f} s: phase {inc.phase}, closed "
          f"{[x for x, _ in inc.closed]}, open {inc.open_lane}, wrecks {[(w.vid, w.lane, round(w.pos)) for w in inc.wrecks]}, "
          f"police {inc.police}, blocker {getattr(inc, 'blocker', None)}", flush=True)
    for ln, (vid, p) in sorted(heads.items()):
        try:
            r, i = c.vehicle.getRoute(vid), c.vehicle.getRouteIndex(vid)
            nxt = r[i + 1] if i + 1 < len(r) else None
            tl = c.vehicle.getNextTLS(vid)[:1]
            lead = c.vehicle.getLeader(vid, 40.0)
            lcl = c.vehicle.getLaneChangeState(vid, 1)[0], c.vehicle.getLaneChangeState(vid, -1)[0]
            print(f"  {ln:12s} len {c.lane.getLength(ln):5.0f} head {vid} {c.vehicle.getTypeID(vid)[:14]} at {p:5.1f} "
                  f"wait {c.vehicle.getWaitingTime(vid):4.0f} next {nxt} tls {tl} leader {lead} lc {lcl} "
                  f"stop {c.vehicle.getStopState(vid)} n {sum(1 for o in c.lane.getLastStepVehicleIDs(ln))}"
                  + (f" | NO LEADER: v_noTraci {c.vehicle.getSpeedWithoutTraCI(vid):.1f} sm {c.vehicle.getSpeedMode(vid)} "
                     f"lcm {c.vehicle.getLaneChangeMode(vid)} held {vid in inc.held} halted {(im.halted.get(vid) if im else None)} squeezed {(vid in im.squeezed) if im else None} role {(im.role.get(vid) if im else None)} tls_dist {tl[0][2] if tl else None} "
                     f"lat {c.vehicle.getLateralLanePosition(vid):.2f} L {c.vehicle.getNeighbors(vid, 3)} R {c.vehicle.getNeighbors(vid, 2)} "
                     f"foes {_foes(c, vid)} maxv {c.vehicle.getAllowedSpeed(vid):.1f}"
                     if lead is None else ""), flush=True)
        except Exception as e:  # noqa: BLE001
            print("  ", ln, vid, e)
    if os.environ.get("DUMP_TLS"):
        for tl in os.environ["DUMP_TLS"].split(","):
            print("  TLS", tl, c.trafficlight.getPhase(tl), c.trafficlight.getRedYellowGreenState(tl),
                  round(c.trafficlight.getNextSwitch(tl) - c.simulation.getTime(), 1), c.trafficlight.getProgram(tl),
                  flush=True)
    if os.environ.get("DUMP_LINKS"):
        for ln in os.environ["DUMP_LINKS"].split(","):
            for lk in c.lane.getLinks(ln, True):
                to, via = lk[0], lk[4]
                occ = [(o, round(c.vehicle.getLanePosition(o), 1), round(c.vehicle.getSpeed(o), 1))
                       for o in c.lane.getLastStepVehicleIDs(to)][:4]
                vocc = [(o, round(c.vehicle.getLanePosition(o), 1)) for o in c.lane.getLastStepVehicleIDs(via)] if via else []
                print("  LINK", ln, "->", to, "prio", lk[1], "open", lk[2], "foe", lk[3], "via", via, "state", lk[5],
                      "dir", lk[6], "to-occ", occ, "via-occ", vocc, "to-len", round(c.lane.getLength(to), 1), flush=True)
    if os.environ.get("DUMP_VEH"):
        for vid in os.environ["DUMP_VEH"].split(","):
            try:
                print("  VEH", vid, c.vehicle.getLaneID(vid), round(c.vehicle.getLanePosition(vid), 1), "sm",
                      c.vehicle.getSpeedMode(vid), "v", c.vehicle.getSpeed(vid), "vNoTraci",
                      round(c.vehicle.getSpeedWithoutTraCI(vid), 2), "route", c.vehicle.getRoute(vid)[c.vehicle.getRouteIndex(vid):][:4],
                      "best", [(b[0], b[3], b[5][:3]) for b in c.vehicle.getBestLanes(vid)], "wait", c.vehicle.getWaitingTime(vid),
                      "leader", c.vehicle.getLeader(vid, 60.0), "lc", c.vehicle.getLaneChangeState(vid, 1)[0],
                      c.vehicle.getLaneChangeState(vid, -1)[0], "stops", [(st.lane, round(st.endPos, 1)) for st in
                                                                         c.vehicle.getStops(vid, 2)],
                      "lat", round(c.vehicle.getLateralLanePosition(vid), 2), "R", c.vehicle.getNeighbors(vid, 0),
                      "Rlead", c.vehicle.getNeighbors(vid, 2), "Llead", c.vehicle.getNeighbors(vid, 3),
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print("  VEH", vid, e)
    if os.environ.get("DUMP_GEOM"):
        for vid in os.environ["DUMP_GEOM"].split(","):
            try:
                print("  GEOM", vid, c.vehicle.getLaneID(vid), round(c.vehicle.getLanePosition(vid), 1), c.vehicle.getPosition(vid), c.vehicle.getWidth(vid),
                      c.vehicle.getLength(vid), c.vehicle.getMinGap(vid), c.vehicle.getMinGapLat(vid),
                      c.vehicle.getLateralLanePosition(vid), c.vehicle.getAngle(vid), flush=True)
                ln = c.vehicle.getLaneID(vid)
                print("     lane", ln, c.lane.getWidth(ln), c.lane.getShape(ln)[:2], flush=True)
            except Exception as e:  # noqa: BLE001
                print("  GEOM", vid, e)
    if os.environ.get("DUMP_MOVE_BLOCKER") and getattr(inc, "blocker", None) and not getattr(inc, "_moved", False):
        inc._moved = True
        ln = c.vehicle.getLaneID(inc.blocker)
        print("  MOVING", inc.blocker, "to", os.environ["DUMP_MOVE_BLOCKER"], flush=True)
        c.vehicle.moveTo(inc.blocker, ln, float(os.environ["DUMP_MOVE_BLOCKER"]))
    for t in inc.tows:  # each truck: where, its leaders, what stands on its lane
        try:
            ln, p = c.vehicle.getLaneID(t.vid), c.vehicle.getLanePosition(t.vid)
        except Exception:  # noqa: BLE001
            continue
        chain, cur = [], t.vid
        for _ in range(12):
            lead = c.vehicle.getLeader(cur, 100.0)
            if not lead or not lead[0]:
                break
            cur = lead[0]
            chain.append(f"{cur}@{c.vehicle.getLaneID(cur)}:{c.vehicle.getLanePosition(cur):.0f}"
                         f" v{c.vehicle.getSpeed(cur):.1f} w{c.vehicle.getWaitingTime(cur):.0f}"
                         f" stop{c.vehicle.getStopState(cur)} tls{c.vehicle.getNextTLS(cur)[:1]}"
                         f" lat{c.vehicle.getLateralLanePosition(cur):.1f} R{[n[0] for n in c.vehicle.getNeighbors(cur, 2)][:2]}"
                         f" L{[n[0] for n in c.vehicle.getNeighbors(cur, 3)][:2]}")
        on = [(o, round(c.vehicle.getLanePosition(o), 1)) for o in c.lane.getLastStepVehicleIDs(ln) if o != t.vid]
        print(f"  TOW {t.vid} {t.state} {ln}@{p:.1f} v{c.vehicle.getSpeed(t.vid):.1f} hold {getattr(t, 'hold', None)} "
              f"leaders {chain} on-lane {on}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--hour", type=float, default=17.5)
    ap.add_argument("--port", type=int, default=8961)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--share", type=float, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump-after", type=lambda x: [float(v) for v in x.split(",")], default=[], metavar="S",
                    help="S s after each accident: print the heads of the queues standing within 200 m of it")
    ap.add_argument("--no-depots", dest="depots", action="store_false",
                    help="police cars and recovery trucks start on the roads (before 25 Oct), not at their depots "
                         "(cosim/depots.py), and the emergency vehicles cross the town")
    ap.add_argument("--no-depot-dispatch", dest="depot_dispatch", action="store_false",
                    help="with the depots: the accidents' vehicles are put on the road near them, not sent from there")
    ap.add_argument("--watch", type=float, default=0.0, metavar="M",
                    help="someone watches every accident: whatever is within M m of one counts as in sight (as "
                         "freefly's in_view does for the camera), so police cars and recovery trucks must come from "
                         "out of sight")
    a = ap.parse_args(argv)
    from cosim import towing
    from cosim.pose_proc import Driver, compose

    scen = scenario_mod.load("town05", ROOT)
    pool = daily.load_pool(scen.folder)
    models = vmodels.load()
    pool.vtypes = {t: vmodels.with_attrs(vt, vmodels.LANE_CHANGE_ATTRS) for t, vt in pool.vtypes.items()}
    out = ROOT / "out" / "freefly"
    pf = out / f"check_scene_{a.port}.rou.xml"
    rids = daily.write_pool_routes(pool, pf)
    et = vmodels.extras_types_file(models, out / f"extras_types_{a.port}.add.xml", vmodels.LANE_CHANGE_ATTRS)
    curve = daily.population_curve(None, pool, mean_trip_min=140 / 60.0)
    sumo = SumoSide(scen.folder, "actuated", seed=a.seed, port=a.port, label=f"scene{a.port}", extras=True,
                    emergency_every=45, extras_types=et, lane_geometry=LaneGeometry(scen.net_file),
                    extra_args=["--route-files", str(pf), "--end", "100000000", "--lateral-resolution", "0.8",
                                "--collision.action", "warn", "--collision.check-junctions", "true",
                                "--collision.mingap-factor", "0"] + ([ "--log", os.environ["SUMO_LOG"]] if os.environ.get("SUMO_LOG") else []))
    c = sumo.conn
    pop = Population(c, pool, rids, curve, seed=a.seed)
    cands = [Candidate(m, i["base_type"].lower(), i["special_type"].lower(), int(i["generation"] or 1), ("0",),
                       i["has_lights"]) for m, i in models.items()]
    base = {t: (re.search(r'vClass="([^"]+)"', vt) or [None, "passenger"])[1] for t, vt in pool.vtypes.items()}
    base.update(vmodels.EMERGENCY_TYPES)
    typer = vmodels.Typer(c, cands, models, base)
    sinks = sorted({r.split()[-1] for routes in pool.routes.values() for r in routes})
    kw = {} if a.share is None else {"share": a.share}
    im = IncidentManager(c, scen.net_file, typer, seed=a.seed, walk=sumo.walk, sinks=sinks, **kw)
    pop.type_for = im.type_for
    if a.watch > 0:
        im.visible = lambda x, y: any(math.hypot(x - i.x, y - i.y) <= a.watch for i in im.incidents)
    sod = a.hour * 3600
    pop.add(int(pop.target(sod)), anywhere=True)

    actors: dict[str, int] = {}
    model_of: dict[int, str] = {}
    cs = NS(actors=actors, lights={}, vclass_of={}, idle={})
    ids = iter(range(1000, 10 ** 6))
    sent: list = []
    clock = {"t": 0.0}

    def send(msg):
        sent.append((round(clock["t"], 1), msg))

    def ground(x, y, yaw):
        return 0.0, 0.0

    walkers = SceneWalkers({"police": [1, 2], "worker": [3, 4], "civil": [5, 6, 7, 8, 9, 10]})
    scene = IncidentScene(scen.frame, ground, im.net, send, walkers, cs, models)
    scene.add_decks([901, 902, 903, 904, 905, 906, 907])
    depots = calls = None
    if a.depots:
        import random as _random

        from cosim.depots import TOWN05, Depots, EmergencyCalls

        depots = Depots(c, im.net, TOWN05)
        depots.setup()
        im.tow_free = lambda vid: not scene.carrying(vid)
        im.stock_depots(depots)
        im.depot_dispatch = a.depot_dispatch
        from cosim.depots import CALL_EVERY_S

        sumo.extras.config.emergency_every = max(sumo.extras.config.emergency_every, CALL_EVERY_S)
        calls = EmergencyCalls(c, im.net, depots, sumo.extras, _random.Random(a.seed + 5))
        calls.stock()
        calls.avoid = im.blocked_roads
        sumo.extras.spawn_hook = calls.spawn
        scene.extra_tows = lambda: depots.depots["tow"].members if "tow" in depots.depots else ()
        figs_depots = {"unloaded": 0}
    import cosim.incident_scene as isc

    isc.time.time = lambda: clock["t"]  # the scene's clock follows SUMO here
    drv = Driver(scen.frame.cx, scen.frame.cy)
    timeline = []
    seen_status = {}
    figs = {"walker_in_vehicle": [], "truck_overlaps": [], "driver_off_pavement_m": [], "officer_off_post_m": [],
            "loaded_off_deck_m": [], "truck_off_sumo_m": [], "poses": 0}
    done_ids = set()
    end = sumo.time + 60.0 * a.minutes
    prev = None
    t_wall0 = time.time()
    while sumo.time < end:
        cur = sumo.step()
        clock["t"] = cur.time
        pop.step(cur.vehicles, sod, lambda vid, v: not im.protected(vid))
        im.step(cur.time, cur.vehicles, cur.tls)
        if depots is not None and int(cur.time * 2) % 4 == 0:
            depots.step(cur.vehicles)
            calls.step(cur.time)
            for tv in depots.parked("tow"):  # a wreck brought to the yard is taken off its truck (out of sight)
                if scene.carrying(tv):
                    xy = cur.vehicles.get(tv)
                    if xy is None or im.visible is None or not im.visible(xy.x, xy.y):
                        figs_depots["unloaded"] += scene.unload(tv)
        if int(cur.time * 2) % 10 == 0:
            im.unjam(cur.time, cur.vehicles)
        if a.dump_after:
            for inc_ in im.incidents:
                if any(abs(cur.time - (inc_.t0 + d)) < 0.25 for d in a.dump_after):
                    dump_queues(c, inc_, cur.vehicles, im)
        snap = im.snapshot()
        near = {}
        for vid, v in cur.vehicles.items():
            if any(math.hypot(v.x - i["x"], v.y - i["y"]) < NEAR_M for i in snap):
                near[vid] = v
        for vid in near:
            if vid not in actors and vid not in scene.taken and not vid.startswith("incblock"):
                actors[vid] = next(ids)
                cs.vclass_of[vid] = "k"
                model_of[actors[vid]] = vmodels.model_of(near[vid].type_id) or "vehicle.audi.tt"
        for vid in [v for v in actors if v not in near]:
            actors.pop(vid)
        n0 = len(sent)
        scene.update(snap, cur.vehicles)
        for t, msg in sent[n0:]:
            drv.receive(msg)
            if msg[0] == "tow":
                plan = drv.tows[msg[1]]
                tid = next((v_ for v_, a_ in actors.items() if a_ == msg[2]), None)
                vt = cur.vehicles.get(tid)
                sumo_tf = scene._carla(vt.x, vt.y, vt.angle, vt.length) if vt else None
                figs.setdefault("tow_starts", []).append({"t": t, "truck": tid, "truck0": str(plan.truck0),
                                                          "sumo_now": [round(v_, 2) for v_ in sumo_tf] if sumo_tf else None,
                                                          "load": [round(v_, 2) for v_ in msg[3]["load"]]})
            if msg[0] not in ("walk",):
                timeline.append((t, msg[0], str(msg[1:])[:60]))
        for inc in snap:
            m = next((m for m in scene.markers if m["id"] == inc["id"]), None)
            if m and seen_status.get(inc["id"]) != m["status"]:
                seen_status[inc["id"]] = m["status"]
                timeline.append((cur.time, "STATUS", f"incident {inc['id']}: {m['status']}"))
        veh = []
        for vid, v in near.items():
            if vid not in actors:
                continue
            p = prev.vehicles.get(vid) if prev else None
            if p is None:
                veh.append((actors[vid], vid, None, None, (v.x, v.y), None, v.length, v.angle, 0, 0, 0, 0,
                            None, v.lane_angle, None, v.speed, scene.damage_for(vid)))
            else:
                p1, p2 = (p.x, p.y), (v.x, v.y)
                veh.append((actors[vid], vid, (2 * p1[0] - p2[0], 2 * p1[1] - p2[1]), p1, p2,
                            (2 * p2[0] - p1[0], 2 * p2[1] - p1[1]), v.length, v.angle, 0, 0, 0, 0,
                            p.lane_angle, v.lane_angle, p.speed, v.speed, scene.damage_for(vid)))
        drv.receive(("seg", {"start": cur.time, "dt": 0.5, "veh": veh, "walk": []}, True))
        name = {a_: v_ for v_, a_ in actors.items()}
        name.update({wa: f"wreck@{wa}" for wa in drv.tows})
        own_car = {}
        for inc in snap:
            for w in inc["wrecks"]:
                wid = scene.state.get(inc["id"], {}).get("drivers", {}).get(w["vid"])
                if wid:
                    own_car[wid] = actors.get(w["vid"]) or scene.state[inc["id"]]["towing"].get(w["vid"])
        tow_aids = {actors.get(t["vid"]) for i in snap for t in i["tows"] if t["state"] in ("backing", "loading", "loaded")}
        for k in range(5):
            t_show = cur.time + 0.1 * k
            vs, ws, *_ = drv.picture(t_show)
            figs["poses"] += 1
            allv = {aid: tf for aid, tf in drv.last_tf.items() if aid in model_of or aid in drv.tows}
            ext = {aid: models.get(model_of.get(aid, ""), {"extent": [2.3, 1.0]})["extent"][:2] for aid in allv}
            ext = {aid: [e[0], min(e[1] * 0.92, 1.275)] for aid, e in ext.items()}  # the body without mirrors (as SUMO)
            for wa, plan in drv.tows.items():  # a carried wreck
                ext[wa] = [plan.car_len / 2, 0.95]
            live_aids = {actors.get(v) for v in near} | set(drv.tows)
            for wid, keys in drv.walks.items():
                if t_show < keys[0][0]:
                    continue
                from cosim.pose_proc import walker_at

                x, y, *_ = walker_at(keys, t_show)
                for aid, tf in allv.items():
                    if aid not in live_aids or aid == own_car.get(wid):
                        continue
                    if inside(tf, ext[aid], x, y):
                        figs["walker_in_vehicle"].append((round(t_show, 1), wid, name.get(aid, aid)))
                        key = f"{wid}|{name.get(aid, aid)}"
                        if key not in figs.setdefault("walker_detail", {}):
                            vid_ = name.get(aid)
                            figs["walker_detail"][key] = {
                                "t": round(t_show, 1), "walker": (round(x, 2), round(y, 2)), "vehicle": str(tf),
                                "lane": c.vehicle.getLaneID(vid_) if vid_ in cur.vehicles else None,
                                "pos": round(c.vehicle.getLanePosition(vid_), 1) if vid_ in cur.vehicles else None,
                                "speed": round(cur.vehicles[vid_].speed, 1) if vid_ in cur.vehicles else None,
                                "keys": [[round(v, 1) for v in k_] for k_ in keys][:6],
                                "posts": {i_: str(st_.get("post")) for i_, st_ in scene.state.items()},
                                "officers": {i_: st_.get("officer") for i_, st_ in scene.state.items()},
                                "incidents": [(i["id"], i["phase"], i["closed"], i["open_lane"], i["police"],
                                               [(w["vid"], w["lane"], round(w["pos"], 1)) for w in i["wrecks"]]) for i in snap]}
            for ta in tow_aids:
                if ta is None or ta not in allv:
                    continue
                for aid, tf in allv.items():
                    if aid == ta or aid not in live_aids or aid in drv.tows:
                        continue
                    if boxes_overlap(allv[ta], ext[ta], tf, ext[aid]):
                        figs["truck_overlaps"].append((round(t_show, 1), name.get(ta, ta), name.get(aid, aid)))
                        key = (name.get(ta, ta), name.get(aid, aid))
                        if key not in figs.setdefault("overlap_detail", {}):
                            plan = next((p_ for p_ in drv.tows.values() if p_.ta == ta), None)
                            vt = cur.vehicles.get(key[0])
                            vo = cur.vehicles.get(key[1])
                            figs["overlap_detail"][key] = {
                                "t": round(t_show, 1), "truck": str(allv[ta]), "other": str(tf),
                                "phase": None if plan is None else __import__("cosim.towing", fromlist=["x"]).phase_at(plan.tl, t_show - plan.t0),
                                "sumo_truck": None if vt is None else (vt.lane_id if hasattr(vt, "lane_id") else None, round(vt.x, 1), round(vt.y, 1), round(vt.angle, 1), round(vt.speed, 1)),
                                "sumo_other": None if vo is None else (round(vo.x, 1), round(vo.y, 1), round(vo.angle, 1), round(vo.speed, 1)),
                                "truck_lane": c.vehicle.getLaneID(key[0]) if key[0] in cur.vehicles else None,
                                "other_lane": c.vehicle.getLaneID(key[1]) if key[1] in cur.vehicles else None}
        # the loaded wreck on its deck, the truck back where SUMO has it, at the end of each loading
        for wa, plan in drv.tows.items():
            tend = plan.t0 + plan.tl["total"]
            if tend <= cur.time < tend + 0.5 and plan.ta in drv.last_tf:
                truck = drv.last_tf[plan.ta]
                want = compose(compose(truck, towing.deck_level()), (towing.CAR_X, 0.0, towing.DECK_T, 0.0, 0.0))
                got = drv.last_tf.get(wa)
                if got is not None:
                    figs["loaded_off_deck_m"].append(round(got.location.distance(want.location), 3))
                tid = next((v for v, a_ in actors.items() if a_ == plan.ta), None)
                if tid in cur.vehicles:
                    v = cur.vehicles[tid]
                    x, y, z, yaw = scene._carla(v.x, v.y, v.angle, v.length)
                    figs["truck_off_sumo_m"].append(round(math.hypot(truck.location.x - x, truck.location.y - y), 3))
        for inc in snap:
            st = scene.state.get(inc["id"])
            if not st:
                continue
            if st.get("officer") and st.get("at_post") and cur.time > st["at_post"] + 1.0 and \
                    inc["phase"] in ("secured", "recovery"):
                keys = drv.walks.get(st["officer"])
                if keys:
                    from cosim.pose_proc import walker_at

                    x, y, *_ = walker_at(keys, cur.time)
                    figs["officer_off_post_m"].append(round(math.hypot(x - st["post"][0], y - st["post"][1]), 2))
            for vid, wid in st["drivers"].items():
                spot = st.get("driver_spot", {}).get(vid)
                keys = drv.walks.get(wid) if wid else None
                if spot and keys and cur.time > keys[-1][0] + 1.0 and inc["phase"] != "clearing":
                    from cosim.pose_proc import walker_at

                    x, y, *_ = walker_at(keys, cur.time)
                    figs["driver_off_pavement_m"].append(round(math.hypot(x - spot[2], y - spot[3]), 2))
        prev = cur
        done_ids |= {i.id for i in im.done}
        if im.done and not im.incidents and not scene.carried and len(done_ids) >= 1 and sumo.time > 120:
            break
    sumo.close()
    kinds: dict = {}
    for _, k, _ in timeline:
        kinds[k] = kinds.get(k, 0) + 1
    res = {"wall_s": round(time.time() - t_wall0), "incidents": im.stats, "done": len(im.done), "messages": kinds,
           "poses_checked": figs["poses"],
           "walker_in_vehicle": len(figs["walker_in_vehicle"]),
           "walker_in_vehicle_pairs": sorted({(w, v) for _, w, v in figs["walker_in_vehicle"]}, key=str),
           "truck_overlaps": len(figs["truck_overlaps"]),
           "truck_overlap_pairs": sorted({(a_, b_, round(t_)) for t_, a_, b_ in figs["truck_overlaps"]}, key=str)[:40],
           "driver_off_pavement_m_max": max(figs["driver_off_pavement_m"], default=None),
           "officer_off_post_m_max": max(figs["officer_off_post_m"], default=None),
           "loaded_off_deck_m": figs["loaded_off_deck_m"], "truck_off_sumo_m": figs["truck_off_sumo_m"],
           "walkers_left": walkers.free, "walker_detail": figs.get("walker_detail"), "tow_starts": figs.get("tow_starts"), "overlap_detail": {f"{k[0]}|{k[1]}": v for k, v in figs.get("overlap_detail", {}).items()},
           "timeline": timeline,
           "depots": None if depots is None else {
               "stats": depots.stats, "calls": calls.stats, "unloaded": figs_depots["unloaded"],
               "parked_at_end": {n: depots.parked(n) for n in depots.depots},
               "out_at_end": {n: sorted(d.members - set(d.parked.values())) for n, d in depots.depots.items()}},
           "events": {i.id: i.events for i in im.done + im.incidents}}
    txt = json.dumps(res, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
