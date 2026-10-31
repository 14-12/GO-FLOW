"""Pictures of pedestrians on the signalised crosswalks (tools/build_town05_peds.py), for checking by eye.

    python tools/probe_crosswalk.py      (CARLA running; tools/probe_crosswalk.bat starts it off screen)

SUMO runs Town05 with the free-fly viewer's pedestrian settings; CARLA draws every vehicle and the
pedestrians near the junction, synchronously, with the pedestrian lights (cosim/ped_lights.py). Whenever
someone is walking on a crossing, a camera 12 m back along the road and 5 m up looks at that crossing:
out/freefly/crosswalk_probe_<n>.png, 4 pictures at 4 different junctions.
"""

from __future__ import annotations

import json
import math
import queue
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import carla  # noqa: E402
import sumolib  # noqa: E402

from cosim import scenario as scenario_mod  # noqa: E402
from cosim.carla_side import CarlaSide  # noqa: E402
from cosim.convert import light_state  # noqa: E402
from cosim.ped_lights import PedLights  # noqa: E402
from cosim.run import Poser, load_world  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402


def main() -> int:
    scen = scenario_mod.load("town05", ROOT)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300.0)
    world = load_world(client, scen)
    cmap = world.get_map()
    net = sumolib.net.readNet(str(scen.net_file), withInternal=True, withPedestrianConnections=True)
    cross = {}
    for e in net.getEdges():
        if e.getFunction() == "crossing":
            sh = e.getLanes()[0].getShape()
            cross[e.getID()] = (sum(p[0] for p in sh) / len(sh), sum(p[1] for p in sh) / len(sh))
    sumo = SumoSide(scen.folder, "actuated", seed=3, label="probe-cw", port=8813, extras=True, emergency_every=60,
                    extra_args=["--end", "100000", "--lanechange.duration", "2"])
    sumo.extras.config.pedestrians_per_km = 25
    sumo.extras.config.pedestrian_trips = True
    c = sumo.conn
    while sumo.time < 600:
        cur = sumo.step()
    side = CarlaSide(client, world, scen.frame, fps=20.0, links=scen.links([lm.id for lm in cmap.get_all_landmarks_of_type("1000001")]))
    side.use_road_levels()
    poser = Poser(scen.frame, side.ground, heading="turn")
    lights = PedLights(world, scen.net_file, scen.frame, cmap)
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    for k, v in (("image_size_x", "1280"), ("image_size_y", "720"), ("fov", "80")):
        bp.set_attribute(k, v)
    cam = world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, 100)))
    q: queue.Queue = queue.Queue()
    cam.listen(q.put)
    done_tls, shots, n = set(), [], 0
    try:
        while sumo.time < 3000 and n < 4:
            cur = sumo.step()
            walking = []
            for pid in c.person.getIDList():
                r = c.person.getRoadID(pid)
                if r.startswith(":") and "_c" in r and c.person.getSpeed(pid) > 0.3:
                    walking.append((pid, r))
            pick = next(((p, r) for p, r in walking if r.split("_")[0][1:] not in done_tls), None)
            if pick is None:
                continue
            pid, road = pick
            tid = road.split("_")[0][1:]
            jx, jy = net.getNode(tid).getCoord()
            mx, my = cross[road]
            d = math.hypot(mx - jx, my - jy) or 1.0
            cx_, cy_ = scen.frame.to_carla_xy(mx, my)
            px_, py_ = scen.frame.to_carla_xy(mx + (mx - jx) / d * 12.0, my + (my - jy) / d * 12.0)
            from cosim.run import ground_z

            z = ground_z(cmap, cx_, cy_)
            yaw = math.degrees(math.atan2(cy_ - py_, cx_ - px_))
            tf = carla.Transform(carla.Location(px_, py_, z + 5.0),
                                 carla.Rotation(-math.degrees(math.atan2(5.0, 12.0)), yaw, 0.0))
            # draw this SUMO state in CARLA
            side.destroy([v for v in side.actors if v not in cur.vehicles])
            poser.begin_step(cur, cur)
            poses = poser.poses(1.0, sumo.step_length)
            side.spawn([(v, cur.vehicles[v].vclass, poses[v]) for v in cur.vehicles
                        if v not in side.actors and v not in side.failed and v in poses])
            side.set_traffic_lights(cur.tls)
            side.move(poses, {v: light_state(x.signals, False, emergency_run=sumo.on_run(v)) for v, x in cur.vehicles.items()})
            near = {p: pp for p, pp in cur.persons.items()
                    if math.hypot(*(a - b for a, b in zip(scen.frame.to_carla_xy(pp.x, pp.y), (cx_, cy_), strict=True))) < 60}
            side.destroy_walkers([p for p in side.walkers if p not in near])
            side.spawn_walkers([(p, side.walker_pose(pp.x, pp.y, pp.angle)) for p, pp in near.items()
                                if p not in side.walkers and p not in side.failed])
            for _ in range(3):
                side.move_walkers({p: (side.walker_pose(pp.x, pp.y, pp.angle), pp.speed) for p, pp in near.items()})
                lights._t = 0.0
                lights.draw(cx_, cy_, cur.tls)
                cam.set_transform(tf)
                fid = side.tick()
            while True:
                im = q.get(timeout=60)
                if im.frame >= fid:
                    break
            rgb = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(im.height, im.width, 4)[:, :, 2::-1]
            out = ROOT / "out" / "freefly" / f"crosswalk_probe_{n}.png"
            Image.fromarray(np.ascontiguousarray(rgb)).save(out)
            st = cur.tls.get(tid, "")
            shots.append({"file": out.name, "junction": tid, "crossing": road, "sim_time": cur.time,
                          "walking_on_crossings": sum(1 for _, r in walking if r.startswith(f":{tid}_c")),
                          "pedestrians_drawn": len(side.walkers), "signal": st})
            print(shots[-1], flush=True)
            done_tls.add(tid)
            n += 1
    finally:
        cam.stop()
        cam.destroy()
        side.close()
        w = sumo.walk
        res = {"shots": shots, "walk_phases_served": w.served if w else None, "skipped": w.skipped if w else None}
        sumo.close()
    (ROOT / "out" / "freefly" / "crosswalk_probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
