"""Town05's ring road with and without its own traffic (cosim/highway.py), SUMO as the free-fly viewer runs it.

    python tools/check_highway.py                       # 20 min at 17:30, ring traffic on
    python tools/check_highway.py --no-highway          # the same without it (the town's trips only)
    python tools/check_highway.py --minutes 30 --hour 8 --json out/highway/check.json

Every minute: cars on each loop of the ring and their mean speed, how many stand there; the town's vehicles, how
many stand, how many have stood over 5 minutes (a lock-up); the halting vehicles at the ring's two junctions.
The town's population follows the day's curve as in the viewer (cosim/live.Population).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import daily, vmodels  # noqa: E402
from cosim import scenario as scenario_mod  # noqa: E402
from cosim.blueprints import Candidate  # noqa: E402
from cosim.highway import HighwayTraffic, find_loops  # noqa: E402
from cosim.lanegeo import LaneGeometry  # noqa: E402
from cosim.live import Population  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402


def _where(c, vid):
    if not vid:
        return ""
    try:
        return (c.vehicle.getLaneID(vid), round(c.vehicle.getLanePosition(vid), 1), round(c.vehicle.getSpeed(vid), 1),
                round(c.vehicle.getWaitingTime(vid)), c.vehicle.getNextTLS(vid)[:1], c.vehicle.getLeader(vid, 30.0),
                round(c.vehicle.getLateralLanePosition(vid), 2), c.vehicle.getLaneChangeState(vid, 1)[0],
                c.vehicle.getLaneChangeState(vid, -1)[0])
    except Exception as e:  # noqa: BLE001
        return str(e)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--hour", type=float, default=17.5)
    ap.add_argument("--port", type=int, default=8971)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--controller", default="actuated")
    ap.add_argument("--no-highway", dest="highway", action="store_false")
    ap.add_argument("--peak", type=int, default=None, help="cars per loop at the peak (cosim/highway.PEAK_PER_LOOP)")
    ap.add_argument("--green-max", type=float, default=None, help="cosim/highway.RING_GREEN_MAX_S (0: no cap)")
    ap.add_argument("--lanes-of", default=None, help="every minute: the halting vehicles on this junction's lanes")
    ap.add_argument("--unjam", action="store_true",
                    help="as the viewer: knots undone (cosim/incidents.IncidentManager.unjam, no accidents; everything "
                         "counts as out of sight, so a car standing 150 s is taken off)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    import sumolib

    scen = scenario_mod.load("town05", ROOT)
    net = sumolib.net.readNet(str(scen.net_file))
    pool = daily.load_pool(scen.folder)
    models = vmodels.load()
    pool.vtypes = {t: vmodels.with_attrs(vt, vmodels.LANE_CHANGE_ATTRS) for t, vt in pool.vtypes.items()}
    out = ROOT / "out" / "highway"
    out.mkdir(parents=True, exist_ok=True)
    pf = out / f"check_{a.port}.rou.xml"
    rids = daily.write_pool_routes(pool, pf)
    et = vmodels.extras_types_file(models, out / f"extras_types_{a.port}.add.xml", vmodels.LANE_CHANGE_ATTRS)
    curve = daily.population_curve(None, pool, mean_trip_min=140 / 60.0)
    sumo = SumoSide(scen.folder, a.controller, seed=a.seed, port=a.port, label=f"hw{a.port}", extras=True,
                    emergency_every=45, extras_types=et, lane_geometry=LaneGeometry(scen.net_file),
                    extra_args=["--route-files", str(pf), "--end", "100000000", "--lateral-resolution", "0.8",
                                "--collision.action", "warn", "--collision.check-junctions", "true",
                                "--collision.mingap-factor", "0"])
    c = sumo.conn
    pop = Population(c, pool, rids, curve, seed=a.seed)
    cands = [Candidate(m, i["base_type"].lower(), i["special_type"].lower(), int(i["generation"] or 1), ("0",),
                       i["has_lights"]) for m, i in models.items()]
    base = {t: (re.search(r'vClass="([^"]+)"', vt) or [None, "passenger"])[1] for t, vt in pool.vtypes.items()}
    base.update(vmodels.EMERGENCY_TYPES)
    typer = vmodels.Typer(c, cands, models, base)
    pop.type_for = typer.type_for
    loops = find_loops(net)
    top = max(curve)
    hw = None
    if a.highway:
        types = [(t, n) for t, n in pool.per_hour.items() if base.get(t) != "bus"]
        kw = {} if a.peak is None else {"peak": a.peak}
        if a.green_max is not None:
            kw["green_max"] = a.green_max or None
        hw = HighwayTraffic(c, loops, lambda s: daily.target_at(curve, s) / top, types=types,
                            type_for=typer.type_for, seed=a.seed, policy=sumo.policy, **kw)
        hw.setup()
        ag = getattr(sumo.policy, "junctions", {})
        print("ring greens capped:", hw.capped, {t: ag[t].max_dur for t in hw.capped if t in ag}, flush=True)
    sod0 = a.hour * 3600
    pop.add(int(pop.target(sod0)), anywhere=True)
    ring_edges = {e for lp in loops for e in lp}
    ring_tls = sorted({net.getEdge(e).getToNode().getID() for e in ring_edges})
    tls_in = {t: [ln for ln in c.trafficlight.getControlledLanes(t) if not ln.startswith(":")] for t in ring_tls}
    all_in = {t: sorted({ln for ln in c.trafficlight.getControlledLanes(t) if not ln.startswith(":")})
              for t in c.trafficlight.getIDList()}
    im = None
    if a.unjam:
        from cosim.incidents import IncidentManager

        im = IncidentManager(c, scen.net_file, share=0.0)
    rows, stood = [], {}
    collisions_n: set = set()  # vehicle-vehicle (SUMO warns, nobody is taken off)
    collision_kinds: list = []
    steps = int(a.minutes * 120)
    for k in range(steps):
        cur = sumo.step()
        sod = sod0 + k * 0.5
        town = {v: x for v, x in cur.vehicles.items() if not (hw and hw.is_ring_car(v))}
        pop.step(town, sod, lambda vid, v: False)
        if hw is not None and k % 4 == 0:
            hw.step(cur.vehicles, sod, set(c.simulation.getPendingVehicles()))
        try:
            for col in c.simulation.getCollisions():
                if col.collider in cur.vehicles and col.victim in cur.vehicles:
                    if (col.collider, col.victim) not in collisions_n:
                        collision_kinds.append((round(cur.time), col.type, col.lane, col.collider, col.victim))
                    collisions_n.add((col.collider, col.victim))
        except Exception:  # noqa: BLE001
            pass
        if im is not None and k % 10 == 0:
            im.now = cur.time
            im.unjam(cur.time, cur.vehicles)
        for vid, v in cur.vehicles.items():
            if v.speed < 0.1:
                stood.setdefault(vid, cur.time)
            else:
                stood.pop(vid, None)
        if k % 120 == 119:
            on_ring = [(vid, v) for vid, v in cur.vehicles.items() if c.vehicle.getRoadID(vid) in ring_edges]
            sp = [v.speed for _, v in on_ring]
            row = {"t_min": round((k + 1) / 120), "on_ring": len(on_ring),
                   "ring_cars": sum(len(s) for s in hw.cars) if hw else 0,
                   "ring_cars_off_ring": sorted({c.vehicle.getRoadID(v) for s in (hw.cars if hw else []) for v in s
                                                 if v in cur.vehicles and c.vehicle.getRoadID(v) not in ring_edges
                                                 and not c.vehicle.getRoadID(v).startswith(":")}),
                   "ring_speed_kmh": round(statistics.mean(sp) * 3.6, 1) if sp else None,
                   "ring_standing": sum(1 for s in sp if s < 0.1),
                   "town": len(town), "town_standing": sum(1 for v in town.values() if v.speed < 0.1),
                   "stood_5min": sum(1 for vid, t0 in stood.items() if cur.time - t0 > 300),
                   "ring_junction_halting": {t: sum(c.lane.getLastStepHaltingNumber(ln) for ln in lanes)
                                             for t, lanes in tls_in.items()},
                   "halting_by_junction": {t: sum(c.lane.getLastStepHaltingNumber(ln) for ln in lanes)
                                           for t, lanes in all_in.items()}}
            rows.append(row)
            print({k: v for k, v in row.items() if k != "halting_by_junction"}, flush=True)
            if a.lanes_of:
                for ln in all_in.get(a.lanes_of, []):
                    ids = c.lane.getLastStepVehicleIDs(ln)
                    halt = [v for v in ids if c.vehicle.getSpeed(v) < 0.1]
                    if halt:
                        h = halt[-1]
                        print(f"   {ln}: {len(halt)} halting; head {h} next "
                              f"{c.vehicle.getRoute(h)[c.vehicle.getRouteIndex(h):][:3]} tls {c.vehicle.getNextTLS(h)[:1]} "
                              f"leader {c.vehicle.getLeader(h, 50.0)} {_where(c, (c.vehicle.getLeader(h, 50.0) or [None])[0])}",
                              flush=True)
    sumo.close() if hasattr(sumo, "close") else None
    half = rows[len(rows) // 2:] or rows
    if not half:
        print("no full minute run")
        return 1
    summary = {"highway": a.highway, "minutes": a.minutes, "hour": a.hour, "seed": a.seed, "loops": loops,
               "on_ring_mean": round(statistics.mean(r["on_ring"] for r in half), 1),
               "ring_speed_kmh_mean": round(statistics.mean(r["ring_speed_kmh"] or 0 for r in half), 1),
               "town_mean": round(statistics.mean(r["town"] for r in half), 1),
               "town_standing_share": round(statistics.mean(r["town_standing"] / max(r["town"], 1) for r in half), 3),
               "stood_5min_end": rows[-1]["stood_5min"] if rows else None,
               "halting_by_junction": {t: round(statistics.mean(r["halting_by_junction"][t] for r in half), 1)
                                       for t in (rows[0]["halting_by_junction"] if rows else {})},
               "added": hw.added if hw else 0, "removed": hw.removed if hw else 0, "rows": rows,
               "unjam": dict(im.stats) if im is not None else None, "collisions": len(collisions_n),
               "collision_list": collision_kinds}
    print({k: v for k, v in summary.items() if k != "rows"})
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
