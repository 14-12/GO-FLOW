"""Accident scene parts, tried in CARLA before they are used (23 Oct).

    python tools/probe_incident_assets.py   (CARLA running; tools/probe_incident_assets.bat)

1. The police officer (walker 0030): his skeleton (bone names, transforms: out/_probe/incident/bones.json), and his
   arms posed for the traffic gestures (stop: right arm up, palm out; pass: left arm out, pointing), several
   rotations each, photographed from the front: which bone axis does what.
2. The police car (Charger 2020 police): driver's door open, beacons and hazard lights.
3. The recovery truck (European HGV) carrying a car on its back, from the side: the car's place on the deck.
4. A wreck scene: two cars after a rear-end crash (tilted, turned), debris, cones and the warning sign.
5. A car on its side (severe crash).
Pictures in out/_probe/incident/.
"""

from __future__ import annotations

import json
import math
import queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "_probe" / "incident"
import carla  # noqa: E402


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

    def shot(self, tf, name, ticks=8):
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
            Image.fromarray(a).save(OUT / f"{name}.jpg", quality=88)


def tf_of(loc, yaw=0.0, pitch=0.0, roll=0.0):
    return carla.Transform(loc, carla.Rotation(pitch, yaw, roll))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300)
    world = client.get_world()
    if not world.get_map().name.endswith("Town05"):
        world = client.load_world("Town05")
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    lib = world.get_blueprint_library()
    cmap = world.get_map()
    rep: dict = {"errors": []}
    cam = Cam(world, lib)
    actors = []

    def spawn(bp_id, tf, physics=False, attrs=None):
        bp = lib.find(bp_id)
        for k, v in (attrs or {}).items():
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        a = world.spawn_actor(bp, tf)
        try:
            a.set_simulate_physics(physics)
        except Exception:  # noqa: BLE001
            pass
        actors.append(a)
        return a

    try:
        # a straight multi-lane road: the waypoint of the first spawn point, its lane and the one beside
        wp = cmap.get_waypoint(world.get_map().get_spawn_points()[10].location)
        base = wp.transform
        yaw = base.rotation.yaw
        fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        rx, ry = -fy, fx

        def at(f, r, z=0.0):
            return carla.Location(base.location.x + fx * f + rx * r, base.location.y + fy * f + ry * r,
                                  base.location.z + z)

        # ---- 0. the plate mesh: can it be moved after spawning, attached to a car? and the bus's plates, lower
        pb = lib.find("static.prop.mesh")
        pb.set_attribute("mesh_path", "/Game/Carla/Static/Other/SM_TnPlate.SM_TnPlate")
        pb.set_attribute("mass", "1.0")  # with a mass CARLA simulates it, so it is made movable; physics off at once
        r0 = client.apply_batch_sync([carla.command.SpawnActor(pb, tf_of(at(0, 0, 5.0))).then(
            carla.command.SetSimulatePhysics(carla.command.FutureActor, False))], True)
        p0 = world.get_actor(r0[0].actor_id)
        actors.append(p0)
        world.tick()
        p0.set_transform(tf_of(at(3, 1, 6.0), 30))
        world.tick()
        world.tick()
        l0 = p0.get_transform()
        want = at(3, 1, 6.0)
        rep["prop_moved_m"] = round(l0.location.distance(want), 3)
        carx = spawn("vehicle.lincoln.mkz_2020", tf_of(at(0, 6, 0.3), yaw))
        world.tick()
        res = client.apply_batch_sync([carla.command.SpawnActor(pb, carla.Transform(carla.Location(-2.5, 0, 0.5)), carx.id).then(
            carla.command.SetSimulatePhysics(carla.command.FutureActor, False))], True)
        world.tick()
        pa = world.get_actor(res[0].actor_id) if not res[0].error else None
        rep["attach_batch"] = res[0].error or None
        if pa is not None:
            actors.append(pa)
            carx.set_transform(tf_of(at(10, 6, 0.3), yaw))
            world.tick()
            world.tick()
            rel_now = pa.get_transform().location - carx.get_transform().location
            rep["attached_follows"] = [round(rel_now.x, 2), round(rel_now.y, 2), round(rel_now.z, 2)]
            rep["attached_parent"] = pa.parent.id if pa.parent is not None else None
        for a in (p0, carx) + ((pa,) if pa is not None else ()):
            a.destroy()
            actors.remove(a)
        p1 = world.spawn_actor(pb, tf_of(at(0, 3, 5.0)))
        actors.append(p1)
        p1.set_simulate_physics(False)
        world.tick()
        client.apply_batch([carla.command.ApplyTransform(p1.id, tf_of(at(4, 3, 6.0), 45))])
        world.tick()
        world.tick()
        rep["prop_batch_moved_m"] = round(p1.get_transform().location.distance(at(4, 3, 6.0)), 3)
        p1.destroy()
        actors.remove(p1)
        bus = spawn("vehicle.mitsubishi.fusorosa", tf_of(at(0, 0, 0.3), yaw))
        world.tick()
        btf = bus.get_transform()
        bb = bus.bounding_box
        rep["bus_bbox"] = [bb.extent.x, bb.extent.y, bb.extent.z, bb.location.x]
        for end, sg in ():
            e = btf.transform(carla.Location(sg * (bb.extent.x + 3.0) + bb.location.x, 0.0, 0.7))
            t = btf.transform(carla.Location(sg * bb.extent.x + bb.location.x, 0.0, 0.6))
            cam.shot(look(e, t), f"bus_plate_{end}")
        bus.destroy()
        actors.remove(bus)

        rep["road"] = {"x": base.location.x, "y": base.location.y, "z": base.location.z, "yaw": yaw,
                       "lane_width": wp.lane_width}

        # ---- 1. the officer's skeleton and arm gestures
        off = spawn("walker.pedestrian.0030", tf_of(at(0, 0, 1.0), yaw + 180))
        world.tick()
        world.tick()
        bones = off.get_bones()
        rep_b = {}
        for bt in bones.bone_transforms:
            name, c, r = bt.name, bt.component, bt.relative
            rep_b[name] = {"component": [c.location.x, c.location.y, c.location.z, c.rotation.pitch, c.rotation.yaw,
                                         c.rotation.roll],
                           "relative": [r.location.x, r.location.y, r.location.z, r.rotation.pitch, r.rotation.yaw,
                                        r.rotation.roll]}
        (OUT / "bones.json").write_text(json.dumps(rep_b, indent=1), encoding="utf-8")
        rel = {n: carla.Transform(carla.Location(*v["relative"][:3]),
                                  carla.Rotation(v["relative"][3], v["relative"][4], v["relative"][5]))
               for n, v in rep_b.items()}
        eye = at(-4.2, 0, 1.4)
        front = look(eye, at(0, 0, 1.2))
        off.get_pose_from_animation()
        off.show_pose()
        cam.shot(front, "officer_00_idle")
        arm = next((n for n in rel if n.lower().endswith("arm__r") and "fore" not in n.lower()), None)
        arml = next((n for n in rel if n.lower().endswith("arm__l") and "fore" not in n.lower()), None)
        rep["arms"] = [arm, arml]
        k = 1
        for bone in (arm, arml):
            if bone is None:
                continue
            for axis in ("pitch", "yaw", "roll"):
                for ang in (-80, 80):
                    r0 = rel[bone]
                    rot = carla.Rotation(r0.rotation.pitch + (ang if axis == "pitch" else 0),
                                         r0.rotation.yaw + (ang if axis == "yaw" else 0),
                                         r0.rotation.roll + (ang if axis == "roll" else 0))
                    off.get_pose_from_animation()
                    off.set_bones(carla.WalkerBoneControlIn([(bone, carla.Transform(r0.location, rot))]))
                    off.show_pose()
                    cam.shot(front, f"officer_{k:02d}_{bone}_{axis}{ang:+d}")
                    k += 1
        off.hide_pose()
        off.destroy()
        actors.remove(off)

        # ---- 2. the police car
        pol = spawn("vehicle.dodge.charger_police_2020", tf_of(at(0, 0, 0.3), yaw))
        world.tick()
        pol.open_door(carla.VehicleDoor.FL)
        v = carla.VehicleLightState
        pol.set_light_state(v(int(v.Special1) | int(v.LeftBlinker) | int(v.RightBlinker) | int(v.Position)))
        for _ in range(30):
            world.tick()
        cam.shot(look(at(-6, 4, 2.2), at(0, 0, 0.8)), "police_door_beacons")
        rep["police_bbox"] = [pol.bounding_box.extent.x, pol.bounding_box.extent.y, pol.bounding_box.extent.z]
        pol.destroy()
        actors.remove(pol)

        # ---- 3. the recovery truck with a car on its back
        hgv = spawn("vehicle.carlamotors.european_hgv", tf_of(at(0, 0, 0.3), yaw))
        world.tick()
        bb = hgv.bounding_box
        rep["hgv_bbox"] = [bb.extent.x, bb.extent.y, bb.extent.z, bb.location.x, bb.location.y, bb.location.z]
        htf = hgv.get_transform()
        car = spawn("vehicle.lincoln.mkz_2020", carla.Transform(carla.Location(0, 0, 500)), attrs={"color": "40,60,140"})
        for i, (dx, dz) in enumerate(((-1.6, 1.15), (-2.2, 1.3))):
            loc = htf.transform(carla.Location(dx, 0.0, dz))
            car.set_transform(carla.Transform(loc, carla.Rotation(2.0, htf.rotation.yaw, 0.0)))
            cam.shot(look(at(-1.5, 9, 2.0), at(-1.5, 0, 1.0)), f"hgv_carried_{i}")
        cam.shot(look(at(-12, 3, 3.0), at(-1.5, 0, 1.0)), "hgv_carried_rear")
        hgv.destroy()
        car.destroy()
        actors.remove(hgv)
        actors.remove(car)

        # ---- 4. a wreck scene (rear-end), cones and the warning sign
        a1 = spawn("vehicle.tesla.model3", tf_of(at(0, 0, 0.05), yaw + 11, pitch=-1.5, roll=3.0), attrs={"color": "160,20,20"})
        a2 = spawn("vehicle.audi.tt", tf_of(at(-4.7, 0.3, 0.0), yaw - 6, pitch=1.5, roll=-1.5), attrs={"color": "200,200,200"})
        for a in (a1, a2):
            a.set_light_state(carla.VehicleLightState(int(v.LeftBlinker) | int(v.RightBlinker)))
        try:
            a1.open_door(carla.VehicleDoor.FL)
        except Exception:  # noqa: BLE001
            pass
        import random

        rng = random.Random(3)
        for i in range(8):
            kind = rng.choice(["static.prop.brokentile01", "static.prop.brokentile02", "static.prop.brokentile03",
                               "static.prop.brokentile04", "static.prop.dirtdebris01", "static.prop.garbage05"])
            spawn(kind, tf_of(at(-2.3 + rng.uniform(-1.5, 1.5), rng.uniform(-1.2, 1.2), 0.02), rng.uniform(0, 360)))
        for i in range(6):
            spawn("static.prop.constructioncone", tf_of(at(-12 - 3.0 * i, -1.6 + i * 0.32 * wp.lane_width / 1.0, 0.0), 0))
        spawn("static.prop.warningaccident", tf_of(at(-34, -2.2, 0.0), yaw + 180))
        drv = spawn("walker.pedestrian.0015", tf_of(at(-0.5, -1.7, 1.0), yaw + 90))
        world.tick()
        cam.shot(look(at(-22, 5, 4.0), at(-3, 0, 0.5)), "scene_rear_end")
        cam.shot(look(at(4, -7, 2.5), at(-2.5, 0, 0.5)), "scene_rear_end_side")
        for a in list(actors):
            a.destroy()
            actors.remove(a)

        # ---- 5. a car on its side
        rol = spawn("vehicle.mercedes.coupe_2020", tf_of(at(0, 0, 0.95), yaw + 30, roll=88), attrs={"color": "30,30,30"})
        world.tick()
        cam.shot(look(at(-9, 5, 3.0), at(0, 0, 0.5)), "rollover")
        rol.destroy()
        actors.remove(rol)
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
        (OUT / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
