"""Drivers in the cars, tried in CARLA before they are used (25 Oct).

    python tools/probe_drivers.py   (CARLA running; tools/probe_drivers.bat)

CARLA's cars drive empty. To seat a person at the wheel, two things are needed: a sitting pose (a walker's leg
bones turned, as the officer's arms are in cosim/pose_proc.SceneOps) and where the driver's seat is in each model.

Run 1 (the first version of this file): which way each leg bone turns: a walker standing, each of crl_thigh__L/R and crl_leg__L/R turned +-90 about
   pitch, yaw and roll, photographed from its left side (legs_*.jpg).
Run 1: every car model of the library: its bounding box (report.json), photographed from its left side with the
   driver's door open and a ruler of debug points beside it: every 0.25 m along the car at seat height, red at the
   bounding box's centre, blue every metre (seat_<model>.jpg).
Found (run 1): the bones' overrides add up (set_bones keeps the bones set before); thigh roll -90 swings the leg
forward (yaw: sideways, pitch: about its own axis), leg (knee) roll +90 bends the knee; the same signs on both sides.
Car models open the driver's door only for a few (Charger 2020); the ruler shows the seat at about the box's centre.

Run 2 (default): a driver seated (cosim/drivers.py's pose and seat) in each model, photographed from the side and
from the front through the windscreen (driver_<model>_side/front.jpg); in the Charger 2020 (door open) the seat's
place tried around the guess (fit_*.jpg).
Pictures in out/_probe/drivers/.
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "_probe" / "drivers"
sys.path.insert(0, str(ROOT))
import carla  # noqa: E402

SKIP = ("bh.crossbike", "diamondback.century", "gazelle.omafiets", "harley", "kawasaki", "yamaha", "vespa")


def look(eye, target):
    d = target - eye
    return carla.Transform(eye, carla.Rotation(math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))),
                                               math.degrees(math.atan2(d.y, d.x)), 0.0))


class Cam:
    def __init__(self, world, lib, w=1000, h=640, fov=50):
        bp = lib.find("sensor.camera.rgb")
        for k, v in (("image_size_x", w), ("image_size_y", h), ("fov", fov)):
            bp.set_attribute(k, str(v))
        self.world = world
        self.cam = world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, 400)))
        self.q: queue.Queue = queue.Queue()
        self.cam.listen(self.q.put)

    def shot(self, tf, name, ticks=6):
        self.cam.set_transform(tf)
        frame = None
        for _ in range(ticks):
            frame = self.world.tick()
        img = None
        t = time.time()
        while time.time() - t < 5:
            try:
                im = self.q.get(timeout=1)
            except queue.Empty:
                continue
            if im.frame >= frame:
                img = im
                break
        while not self.q.empty():
            self.q.get_nowait()
        if img is not None:
            a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3][:, :, ::-1]
            Image.fromarray(a).save(OUT / f"{name}.jpg", quality=85)


def main() -> int:
    from cosim.drivers import POSE, seat_of

    OUT.mkdir(parents=True, exist_ok=True)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300)
    world = client.get_world()
    if not world.get_map().name.endswith("Town05"):
        world = client.load_world("Town05")
    client.set_timeout(60)
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    lib = world.get_blueprint_library()
    rep: dict = {"errors": [], "vehicles": {}, "seats": {}}
    cam = Cam(world, lib)
    actors = []
    try:
        wp = world.get_map().get_waypoint(world.get_map().get_spawn_points()[10].location)
        base = wp.transform
        yaw = base.rotation.yaw
        fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        rx, ry = -fy, fx

        def at(f, r, z=0.0):
            return carla.Location(base.location.x + fx * f + rx * r, base.location.y + fy * f + ry * r,
                                  base.location.z + z)

        w = world.spawn_actor(lib.find("walker.pedestrian.0002"), carla.Transform(at(0, 0, 50.0)))
        actors.append(w)
        w.set_simulate_physics(False)
        world.tick()
        rel = {b.name: b.relative for b in w.get_bones().bone_transforms}
        bones = []
        for name, (dp, dy, dr) in POSE.items():
            r0 = rel[name]
            bones.append((name, carla.Transform(r0.location, carla.Rotation(r0.rotation.pitch + dp, r0.rotation.yaw + dy,
                                                                              r0.rotation.roll + dr))))
        w.get_pose_from_animation()
        w.set_bones(carla.WalkerBoneControlIn(bones))
        w.show_pose()

        def seat(v, dx=0.0, dz=0.0):
            bb = v.bounding_box
            x, y, z = seat_of(bp_id_of(v), (bb.extent.x, bb.extent.y, bb.extent.z), (bb.location.x, bb.location.z))
            vtf = v.get_transform()
            loc = vtf.transform(carla.Location(x + dx, y, z + dz))
            w.set_transform(carla.Transform(loc, vtf.rotation))
            return x, y, z

        def bp_id_of(v):
            return v.type_id

        def shots(v, name, x, y, z):
            vtf = v.get_transform()
            bb = v.bounding_box
            hip = vtf.transform(carla.Location(x, y, z))
            far = 1.0 + 0.6 * max(bb.extent.z - 0.8, 0.0) / 0.8  # tall vehicles from further
            side = vtf.transform(carla.Location(x, -(bb.extent.y + 3.2 * far), z + 0.5 * far))
            front = vtf.transform(carla.Location(bb.location.x + bb.extent.x + 3.5 * far, y - 0.8, z + 1.0))
            cam.shot(look(side, hip + carla.Location(0, 0, 0.3)), f"{name}_side", ticks=4)
            cam.shot(look(front, hip + carla.Location(0, 0, 0.45)), f"{name}_front", ticks=4)

        # ---- the seat's place in the Charger 2020 (its door opens)
        v = world.spawn_actor(lib.find("vehicle.dodge.charger_2020"), carla.Transform(at(0, 0, 0.3), carla.Rotation(0, yaw, 0)))
        actors.append(v)
        v.set_simulate_physics(False)
        world.tick()
        v.open_door(carla.VehicleDoor.FL)
        for _ in range(25):
            world.tick()
        for dx in ():
            for dz in ():
                x, y, z = seat(v, dx, dz)
                vtf = v.get_transform()
                hip = vtf.transform(carla.Location(x + dx, y, z + dz))
                eye = vtf.transform(carla.Location(x + dx, y - 3.0, z + dz + 0.4))
                cam.shot(look(eye, hip + carla.Location(0, 0, 0.2)), f"fit_dx{dx:+.1f}_dz{dz:+.1f}", ticks=3)
        v.destroy()
        actors.remove(v)

        # ---- every model with its driver
        for bp in sorted(lib.filter("vehicle.*"), key=lambda b: b.id):
            if any(x in bp.id for x in SKIP):
                continue
            if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels").as_int()) < 4:
                continue
            if os.environ.get("PROBE_ONLY") and not any(m in bp.id for m in os.environ["PROBE_ONLY"].split(",")):
                continue
            t0 = time.time()
            try:
                w.set_transform(carla.Transform(at(0, 0, 50.0)))  # out of the way: a spawn fails on a collision
                world.tick()
                v = world.spawn_actor(bp, carla.Transform(at(0, 0, 0.3), carla.Rotation(0, yaw, 0)))
                actors.append(v)
                v.set_simulate_physics(False)
                world.tick()
                bb = v.bounding_box
                rep["vehicles"][bp.id] = {"extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                                          "center": [bb.location.x, bb.location.y, bb.location.z],
                                          "base_type": bp.get_attribute("base_type").as_str()
                                          if bp.has_attribute("base_type") else "",
                                          "special_type": bp.get_attribute("special_type").as_str()
                                          if bp.has_attribute("special_type") else ""}
                x, y, z = seat(v)
                rep["seats"][bp.id] = [round(x, 3), round(y, 3), round(z, 3)]
                shots(v, f"driver_{bp.id[8:]}", x, y, z)
                if os.environ.get("SEAT_GRID"):  # the seat moved around the guess; where the walker really went
                    for dx in (-0.6, 0.0, 0.6):
                        for dz in (-0.3, 0.0, 0.3):
                            seat(v, dx, dz)
                            world.tick()
                            want = v.get_transform().transform(carla.Location(x + dx, y, z + dz))
                            got = w.get_transform().location
                            rep.setdefault("grid", []).append([bp.id, dx, dz, round(got.distance(want), 3)])
                            vtf = v.get_transform()
                            hip = vtf.transform(carla.Location(x + dx, y, z + dz))
                            eye = vtf.transform(carla.Location(x + dx, y - 4.5, z + dz + 0.6))
                            cam.shot(look(eye, hip + carla.Location(0, 0, 0.3)),
                                     f"grid_{bp.id[8:]}_dx{dx:+.1f}_dz{dz:+.1f}", ticks=3)
                v.destroy()
                actors.remove(v)
            except Exception as e:  # noqa: BLE001
                rep["errors"].append(f"{bp.id}: {e!r}")
                print(bp.id, repr(e), flush=True)
            print(bp.id, f"{time.time() - t0:.1f} s", flush=True)
    except Exception:  # noqa: BLE001
        rep["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        for a in actors:
            try:
                a.destroy()
            except Exception:  # noqa: BLE001
                pass
        cam.cam.stop()
        cam.cam.destroy()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
        (OUT / "report2.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("done", len(rep["vehicles"]), "vehicles", len(rep["errors"]), "errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
