"""How many TraCI round trips the free-fly viewer's SUMO thread makes per step, and from where (SUMO alone).

    python tools/count_traci.py [--steps 120] [--start 08:00]

Each round trip to SUMO releases Python's lock and has to win it back from the viewer's other threads (the
main loop, the pose feeder, the camera): in the viewer every one cost ~1 ms more than here. Builds SUMO as
cosim/freefly.FreeFly.load does (the day's population, the extras, the bus lines), fills the town, then runs
the SUMO part of FreeFly.produce (sumo.step, pop.step, every second keep_driving / bus / preempt /
keep_walking) with the camera over a Town05 junction, and counts the commands sent, per calling function.
Writes out/freefly/count_traci.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--start", default="08:00")
    ap.add_argument("--fill-s", type=float, default=30.0)
    ap.add_argument("--port", type=int, default=8871)
    a = ap.parse_args(argv)

    import random

    import sumolib
    import traci.connection

    from cosim import daily
    from cosim import scenario as scenario_mod
    from cosim.freefly import BusLines, FreeFly, hhmm_to_s, population_class
    from cosim.sumo_side import SumoSide

    calls: collections.Counter = collections.Counter()
    counting = [False]
    orig = traci.connection.Connection._sendCmd

    def counted(self, cmdID, varID, objID, *rest, **kw):
        if counting[0]:
            f = sys._getframe(1)
            while f is not None and ("traci" + os.sep) in f.f_code.co_filename:
                f = f.f_back
            where = f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}" if f else "?"
            calls[(where, hex(cmdID), hex(varID) if isinstance(varID, int) else "sub")] += 1
        return orig(self, cmdID, varID, objID, *rest, **kw)

    traci.connection.Connection._sendCmd = counted

    scen = scenario_mod.load("town05", ROOT)
    pool = daily.load_pool(scen.folder)
    pool_file = ROOT / "out" / "freefly" / "town05.pool.rou.xml"
    pool_file.parent.mkdir(parents=True, exist_ok=True)
    route_ids = daily.write_pool_routes(pool, pool_file)
    curve = daily.population_curve(None, pool, mean_trip_min=140.0 / 60.0)
    sumo = SumoSide(scen.folder, "actuated", seed=42, port=a.port, label="count", vehicle_vars=[0x69], person_vars=[0xc2],
                    extras=True, emergency_every=45.0,
                    additional=BusLines.additional(scen.folder, "town05"),
                    extra_args=["--route-files", str(pool_file), "--end", "100000000", "--lanechange.duration", "2"])
    ff = FreeFly.__new__(FreeFly)
    ff.scen, ff.sumo = scen, sumo
    ff.args = SimpleNamespace(radius=500.0)
    net = sumolib.net.readNet(str(scen.net_file))
    # the camera over the busiest junction, looking along +x
    tls = net.getTrafficLights()
    nodes = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    node = max(nodes, key=lambda n: len(n.getIncoming()))
    cx, cy = scen.frame.to_carla_xy(*node.getCoord())
    ff._view = (cx, cy, 30.0, 1.0, 0.0)
    ex = sumo.extras
    ex.config.pedestrians_per_km = 20.0
    ex.config.on_run_share = 1.0
    ex.config.pedestrian_trips = True
    ex.config.near_share = 0.75
    ex.near_xy = node.getCoord()
    ex.spawn_ok = lambda edge, pos: not ff.in_view(*scen.frame.to_carla_xy(*sumo.conn.simulation.convert2D(edge, pos)))
    ff.pop = population_class()(sumo.conn, pool, route_ids, curve, seed=42)
    bus_types = [t for t, vt in pool.vtypes.items() if 'vClass="bus"' in vt]
    ff.bus = BusLines(sumo.conn, scen.folder, "town05", bus_types[0]) if bus_types else None
    for i, t in enumerate(ff.pop.types):
        if t in bus_types:
            ff.pop.weights[i] = 0
    ff.edge_start = {}
    for e in net.getEdges():
        x, y = e.getLanes()[0].getShape()[0]
        ff.edge_start[e.getID()] = scen.frame.to_carla_xy(x, y)
    ff.sinks = sorted({r.split()[-1] for routes in pool.routes.values() for r in routes})
    ff.rng = random.Random(43)
    ff.pop.start_ok = lambda edge: not ff.in_view(*ff.edge_start.get(edge, (1e9, 1e9)))
    ff.extended = ff.walks_extended = ff.preempted = 0
    sod = hhmm_to_s(a.start)
    ff.target = ff.pop.target(sod)
    ff.pop.add(int(ff.target), anywhere=True)
    for _ in range(int(a.fill_s / sumo.step_length)):
        cur = sumo.step()
        ff.pop.step(cur.vehicles, sod, lambda vid, v: True)
    print(f"filled: {len(cur.vehicles)} vehicles, {len(cur.persons)} persons", flush=True)

    counting[0] = True
    frame = scen.frame
    t_all = time.perf_counter()
    secs: collections.Counter = collections.Counter()
    n_veh = n_per = 0
    for k in range(a.steps):
        t = time.perf_counter()
        cur = sumo.step()
        secs["sumo.step"] += time.perf_counter() - t
        t = time.perf_counter()
        alive = set(sumo.conn.vehicle.getIDList())
        ff.target = ff.pop.step(cur.vehicles, sod + k * sumo.step_length,
                                lambda vid, v: vid in alive and not vid.startswith("pt_")
                                and not ff.in_view(*frame.to_carla_xy(v.x, v.y)))
        secs["pop.step"] += time.perf_counter() - t
        if k % 2 == 0:
            for name, fn in (("keep_driving", lambda: ff.keep_driving(cur.vehicles)),
                             ("bus", lambda: ff.bus.step(cur.time, (sod / 3600.0) % 24) if ff.bus else None),
                             ("preempt", lambda: ff.preempt(cur.vehicles)),
                             ("keep_walking", lambda: ff.keep_walking(cur.persons))):
                t = time.perf_counter()
                fn()
                secs[name] += time.perf_counter() - t
        n_veh += len(cur.vehicles)
        n_per += len(cur.persons)
    span = time.perf_counter() - t_all
    counting[0] = False
    sumo.close()

    total = sum(calls.values())
    by_fn: collections.Counter = collections.Counter()
    for (where, _, _), n in calls.items():
        by_fn[where] += n
    res = {"steps": a.steps, "start": a.start, "trips_extended": ff.extended, "walks_extended": ff.walks_extended, "vehicles_mean": round(n_veh / a.steps, 1),
           "persons_mean": round(n_per / a.steps, 1), "round_trips_per_step": round(total / a.steps, 1),
           "ms_per_step": round(1000 * span / a.steps, 2),
           "ms_per_step_by_part": {k: round(1000 * v / a.steps, 2) for k, v in secs.most_common()},
           "per_step_by_function": {w: round(n / a.steps, 1) for w, n in by_fn.most_common()},
           "per_step_by_command": {f"{w} {c} {v}": round(n / a.steps, 2) for (w, c, v), n in calls.most_common(30)}}
    out = ROOT / "out" / "freefly" / "count_traci.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
