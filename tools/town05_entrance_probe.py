"""Where a car can really drive into the car parks inside buildings (Galerie Atlas, Parking Central, the towers, the
apartment blocks), from CARLA itself (9 Nov).

The viewer drives cars into their painted bays (cosim/parking_motion.py); the car parks inside buildings were left
out, the car would have gone through a wall. This looks for real openings:

* around each building, a grid of vertical rays (CARLA's ray casting, every surface the ray meets, with its label):
  where the ground is, whether anything stands between 0.3 m and 2.2 m over it (a wall, a fence, a post) and whether
  something covers it (a floor, a roof) -> out/town05_places/entrances_probe.json;
* pictures at a driver's eye from the road beside each car park and from a drone -> out/town05_places/door_*.jpg.

CARLA must be running (tools/town05_entrance_probe.bat starts it off screen).
"""
from __future__ import annotations

import json
import math
import queue
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import carla

from cosim import scenario as scenario_mod

OUT = ROOT / "out" / "town05_places"
STEP = 0.6
HALF = 0.35         # the sideways rays: this far either side of a cell's middle
PAD = 14.0
TOP, BOTTOM = 90.0, -16.0


def shoot(world, lib, tf, w=1600, h=900, fov=90.0):
    bp = lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    cam = world.spawn_actor(bp, tf)
    q: queue.Queue = queue.Queue()
    cam.listen(lambda im: q.put((im.frame, np.array(np.frombuffer(im.raw_data, dtype=np.uint8)
                                                    .reshape(im.height, im.width, 4)))))
    img = None
    try:
        frame = None
        for _ in range(10):
            frame = world.tick()
        t = time.time()
        while time.time() - t < 20.0:
            try:
                f, a = q.get(timeout=2.0)
            except queue.Empty:
                continue
            if f >= frame:
                img = a
                break
    finally:
        cam.stop()
        time.sleep(0.3)
        cam.destroy()
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scen = scenario_mod.load("town05", ROOT)
    fr = scen.frame
    import sumolib

    net = sumolib.net.readNet(str(scen.net_file))
    places = json.loads((ROOT / "scenarios" / "town05" / "places.json").read_text(encoding="utf-8"))
    areas = {a["id"]: a for a in places["parking_areas"]}
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300.0)
    world = client.get_world()
    if not world.get_map().name.endswith("Town05"):
        world = client.load_world("Town05")
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    lib = world.get_blueprint_library()
    for _ in range(20):
        world.tick()

    def ground_z(x, y):
        cx, cy = fr.to_carla_xy(x, y)
        hits = world.cast_ray(carla.Location(cx, cy, 5.0), carla.Location(cx, cy, -5.0))
        return hits[0].location.z if hits else 0.0

    def cam_at(x, y, z, look_x, look_y, look_z):
        cx, cy = fr.to_carla_xy(x, y)
        lx, ly = fr.to_carla_xy(look_x, look_y)
        dx, dy, dz = lx - cx, ly - cy, look_z - z
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
        return carla.Transform(carla.Location(cx, cy, z), carla.Rotation(pitch, yaw, 0.0))

    res = {"step": STEP, "what": __doc__.split("\n\n")[0], "areas": {}}
    for pl in places["places"]:
        aid = pl.get("parking")
        if not aid or not areas[aid].get("hidden"):
            continue
        a = areas[aid]
        xs = [p[0] for p in pl["outline"]]
        ys = [p[1] for p in pl["outline"]]
        x0, x1, y0, y1 = min(xs) - PAD, max(xs) + PAD, min(ys) - PAD, max(ys) + PAD
        t = time.time()
        nx, ny = int((x1 - x0) / STEP) + 1, int((y1 - y0) / STEP) + 1
        cols, side = [], []
        for j in range(ny):
            row, srow = [], []
            for i in range(nx):
                x, y = x0 + i * STEP, y0 + j * STEP
                cx, cy = fr.to_carla_xy(x, y)
                hits = world.cast_ray(carla.Location(cx, cy, TOP), carla.Location(cx, cy, BOTTOM))
                row.append([[round(h.location.z, 2), int(h.label)] for h in hits][:24])
                # (9 Nov, 2nd run) what stands at a car's height, seen sideways: a vertical ray meets only what faces
                # up, it misses a wall under a roof and a pillar under a floor. Short rays across the cell at 0.5 and
                # 1.2 m over its ground (the highest surface within 1 m of the street): '1' if any meets something
                g = max((h.location.z for h in hits if -1.0 <= h.location.z <= 1.0), default=0.0)
                hit = False
                for zz in (0.5, 1.2):
                    for dx, dy in ((HALF, 0.0), (0.0, HALF)):
                        a_ = carla.Location(cx - dx, cy - dy, g + zz)
                        b_ = carla.Location(cx + dx, cy + dy, g + zz)
                        if world.cast_ray(a_, b_) or world.cast_ray(b_, a_):
                            hit = True
                            break
                    if hit:
                        break
                srow.append("1" if hit else "0")
            cols.append(row)
            side.append("".join(srow))
        lane = net.getLane(a["lane"])
        mid = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), (a["start"] + a["end"]) / 2)
        res["areas"][aid] = {"place": pl["id"], "x0": x0, "y0": y0, "nx": nx, "ny": ny, "cols": cols, "side": side,
                             "lane": a["lane"], "lane_shape": [[round(v, 2) for v in p] for p in lane.getShape()],
                             "outline": pl["outline"], "seconds": round(time.time() - t, 1)}
        print(f"{aid}: {nx}x{ny} rays in {time.time() - t:.0f} s", flush=True)
        # pictures: from the road beside the car park at a driver's eye, 20 m either side, and a drone
        bx, by = sum(xs) / len(xs), sum(ys) / len(ys)
        sh = lane.getShape()
        L = lane.getLength()
        shots = []
        for k, off in enumerate((-22.0, 0.0, 22.0)):
            p = sumolib.geomhelper.positionAtShapeOffset(sh, min(max((a["start"] + a["end"]) / 2 + off, 1.0), L - 1.0))
            gz = ground_z(*p)
            shots.append((f"eye{k}", cam_at(p[0], p[1], gz + 1.6, bx, by, gz + 3.0)))
        d = math.hypot(mid[0] - bx, mid[1] - by) or 1.0
        ux, uy = (mid[0] - bx) / d, (mid[1] - by) / d
        shots.append(("drone", cam_at(mid[0] + ux * 25, mid[1] + uy * 25, 28.0, (mid[0] + bx) / 2, (mid[1] + by) / 2,
                                      0.0)))
        for name, tf in shots:
            img = shoot(world, lib, tf)
            if img is not None:
                Image.fromarray(img[:, :, :3][:, :, ::-1]).save(OUT / f"door_{pl['id']}_{name}.jpg", quality=88)
        (OUT / "entrances_probe.json").write_text(json.dumps(res), encoding="utf-8")

    # the doors and gates Town05 has (out/town05_places/env_objects.json): the Galerie Atlas's metal shutters, seen
    # shut and with the shutter hidden (world.enable_environment_objects: what is behind it); Parking Central's
    # gates and ground floor
    objs = {o.name: o for o in world.get_environment_objects(carla.CityObjectLabel.Any)}
    doors = {}
    for name, o in objs.items():
        if "MetallicDoor" not in name:
            continue
        bb = o.bounding_box
        sx, sy = fr.to_sumo_xy(bb.location.x, bb.location.y)
        doors[name] = {"id": int(o.id), "sumo": [round(sx, 2), round(sy, 2)], "ext": [bb.extent.x, bb.extent.y,
                                                                                      bb.extent.z],
                       "yaw": bb.rotation.yaw}
    res["doors"] = doors
    views = []
    for name, dd in doors.items():            # each shutter from 9 m out, both sides (the outside is not known)
        sx, sy = dd["sumo"]
        across_x = abs(math.cos(math.radians(dd["yaw"]))) > 0.7     # its thin side along CARLA x: faces +-x
        for sgn in (-1, 1):
            ex, ey = (sx + sgn * 9.0, sy) if across_x else (sx, sy + sgn * 9.0)
            views.append((f"{name}_{'a' if sgn < 0 else 'b'}", ex, ey, sx, sy, dd["id"]))
    for tag, ex, ey, lx, ly, did in views:
        gz = ground_z(ex, ey)
        tf = cam_at(ex, ey, gz + 1.6, lx, ly, gz + 1.4)
        img = shoot(world, lib, tf, w=1280, h=720)
        if img is not None:
            Image.fromarray(img[:, :, :3][:, :, ::-1]).save(OUT / f"door_atlas_{tag}_shut.jpg", quality=85)
        world.enable_environment_objects({did}, False)
        img = shoot(world, lib, tf, w=1280, h=720)
        # how deep a car could go behind it: horizontal rays at 0.6, 1.2 and 1.8 m from 9 m out, through its middle
        deep = []
        for zz in (0.6, 1.2, 1.8):
            a_ = fr.to_carla_xy(ex, ey)
            b_ = fr.to_carla_xy(ex + (lx - ex) * 5.0, ey + (ly - ey) * 5.0)
            hits = world.cast_ray(carla.Location(a_[0], a_[1], gz + zz), carla.Location(b_[0], b_[1], gz + zz))
            deep.append([[round(math.dist(a_, (h.location.x, h.location.y)), 2), int(h.label)] for h in hits][:6])
        world.enable_environment_objects({did}, True)
        res.setdefault("door_rays", {})[tag] = deep
        if img is not None:
            Image.fromarray(img[:, :, :3][:, :, ::-1]).save(OUT / f"door_atlas_{tag}_open.jpg", quality=85)
    silo_views = {"gate_east": ((216.0, 261.0, 1.6), (200.0, 259.0, 1.2)),
                  "gate_west": ((150.0, 261.0, 1.6), (168.0, 259.0, 1.2)),
                  "compound": ((203.0, 257.5, 1.6), (196.0, 238.0, 1.0)),
                  "compound_w": ((165.0, 257.5, 1.6), (172.0, 238.0, 1.0)),
                  "ground_floor": ((205.0, 247.0, 1.6), (175.0, 222.0, 1.0)),
                  "south_road": ((185.0, 204.0, 1.6), (185.0, 230.0, 1.5)),
                  "drone_n": ((185.0, 285.0, 30.0), (185.0, 245.0, 0.0))}
    for tag, ((ex, ey, ez), (lx, ly, lz)) in silo_views.items():
        gz = ground_z(ex, ey) if ez < 5 else 0.0
        img = shoot(world, lib, cam_at(ex, ey, gz + ez, lx, ly, gz + lz), w=1280, h=720)
        if img is not None:
            Image.fromarray(img[:, :, :3][:, :, ::-1]).save(OUT / f"door_silo_{tag}.jpg", quality=85)
    (OUT / "entrances_probe.json").write_text(json.dumps(res), encoding="utf-8")
    s.synchronous_mode = False
    world.apply_settings(s)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
