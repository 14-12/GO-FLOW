"""Pictures and facts about CARLA's own assets, for the accident scenes and the plates (23 Oct).

    python tools/probe_carla_assets.py      (CARLA running; tools/probe_carla_assets.bat starts it off screen)

Every vehicle model is put in the sky (no physics, sun above) and photographed from behind, in front (plate
close-ups too) and from the side with its doors opened; every pedestrian model from the front; the props that
could dress an accident scene (cones, warnings, barriers, debris). Facts to out/_probe/carla/probe.json:
each model's bounding box (for SUMO's vehicle lengths), attributes, the props' ids, whether static.prop.mesh
exists. Pictures to out/_probe/carla/*.png.
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
OUT = ROOT / "out" / "_probe" / "carla"
import carla  # noqa: E402

X0, Y0, Z0 = -150.0, -250.0, 300.0
PROP_WORDS = ("cone", "warning", "barrier", "debris", "tile", "glass", "plank", "box", "garbage", "bag", "barrel",
              "trash", "kiosk", "clothcontainer", "streetsign", "trafficwarning", "accident", "construction")


def save(img, name: str) -> None:
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3][:, :, ::-1]
    Image.fromarray(arr).save(OUT / f"{name}.png")


def shoot(world, lib, tf, name: str, w=960, h=540, fov=35.0, ticks=14) -> bool:
    bp = lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    cam = world.spawn_actor(bp, tf)
    q: queue.Queue = queue.Queue()
    cam.listen(q.put)
    img = None
    try:
        for _ in range(ticks):
            world.tick()
            time.sleep(0.04)  # let the textures stream in
            try:
                while True:
                    img = q.get(timeout=2.0 if img is None else 0.01)
            except queue.Empty:
                pass
        if img is not None:
            save(img, name)
    finally:
        cam.stop()
        cam.destroy()
    return img is not None


def look_at(eye: carla.Location, target: carla.Location) -> carla.Transform:
    d = target - eye
    yaw = math.degrees(math.atan2(d.y, d.x))
    pitch = math.degrees(math.atan2(d.z, math.hypot(d.x, d.y)))
    return carla.Transform(eye, carla.Rotation(pitch, yaw, 0.0))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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
    facts: dict = {"vehicles": {}, "walkers": {}, "props": [], "errors": []}
    facts["static_prop_mesh"] = None
    try:
        bp = lib.find("static.prop.mesh")
        facts["static_prop_mesh"] = [a.id for a in bp]
    except Exception as e:  # noqa: BLE001
        facts["static_prop_mesh"] = f"missing: {e}"
    facts["props"] = sorted(b.id for b in lib.filter("static.prop.*"))
    facts["vehicle_doors"] = [d for d in dir(carla.VehicleDoor) if not d.startswith("_")]
    try:
        names = world.get_names_of_all_objects()
        facts["objects_total"] = len(names)
    except Exception as e:  # noqa: BLE001
        facts["objects_total"] = f"error: {e}"

    # ------------------------------------------------------------------ vehicles
    vbps = sorted(lib.filter("vehicle.*"), key=lambda b: b.id)
    actors = []
    for i, bp in enumerate(vbps):
        if bp.has_attribute("color"):
            bp.set_attribute("color", "200,200,200")
        tf = carla.Transform(carla.Location(X0 + 14.0 * i, Y0, Z0), carla.Rotation(0.0, 90.0, 0.0))
        a = world.try_spawn_actor(bp, tf)
        if a is None:
            facts["errors"].append(f"spawn {bp.id}")
            continue
        a.set_simulate_physics(False)
        actors.append((bp, a, tf))
    world.tick()
    try:
        names2 = world.get_names_of_all_objects()
        facts["objects_after_vehicles"] = len(names2)
        facts["object_names_sample"] = sorted(set(names2) - set(names))[:80] if isinstance(names, list) else []
    except Exception as e:  # noqa: BLE001
        facts["objects_after_vehicles"] = f"error: {e}"
    for bp, a, tf in actors:
        bb = a.bounding_box
        attrs = {}
        for at in bp:
            try:
                attrs[at.id] = at.as_str() if at.type == carla.ActorAttributeType.String else str(at)
            except Exception:  # noqa: BLE001
                attrs[at.id] = "?"
        facts["vehicles"][bp.id] = {"extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                                    "centre": [bb.location.x, bb.location.y, bb.location.z], "attributes": attrs}
    for bp, a, tf in actors:
        tag = bp.id.replace("vehicle.", "").replace(".", "_")
        try:
            loc = a.get_transform().location
            bb = a.bounding_box
            c = carla.Location(loc.x + bb.location.y * 0, loc.y + bb.location.x, loc.z + bb.location.z)
            back = carla.Location(c.x, c.y - bb.extent.x, loc.z + 0.55)
            front = carla.Location(c.x, c.y + bb.extent.x, loc.z + 0.55)
            shoot(world, lib, look_at(carla.Location(c.x, c.y - bb.extent.x - 5.5, loc.z + 1.2), back),
                  f"veh_{tag}_rear", fov=40)
            shoot(world, lib, look_at(carla.Location(c.x, c.y - bb.extent.x - 3.0, loc.z + 0.7), back),
                  f"veh_{tag}_rear_close", fov=22)
            shoot(world, lib, look_at(carla.Location(c.x, c.y + bb.extent.x + 3.0, loc.z + 0.7), front),
                  f"veh_{tag}_front_close", fov=22)
            try:
                a.open_door(carla.VehicleDoor.All)
            except Exception as e:  # noqa: BLE001
                facts["errors"].append(f"doors {bp.id}: {e}")
            for _ in range(30):
                world.tick()
            shoot(world, lib, look_at(carla.Location(c.x + 6.5, c.y + 1.0, loc.z + 1.6), c), f"veh_{tag}_doors",
                  fov=55)
        except Exception:  # noqa: BLE001
            facts["errors"].append(f"{bp.id}: {traceback.format_exc()}")
        print("vehicle", bp.id, flush=True)
    for _, a, _ in actors:
        a.destroy()

    # ------------------------------------------------------------------ walkers
    wbps = sorted(lib.filter("walker.pedestrian.*"), key=lambda b: b.id)
    for i, bp in enumerate(wbps):
        attrs = {}
        for at in bp:
            try:
                attrs[at.id] = str(at)
            except Exception:  # noqa: BLE001
                attrs[at.id] = "?"
        facts["walkers"][bp.id] = attrs
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        loc = carla.Location(X0 + 6.0 * i, Y0 + 60.0, Z0)
        a = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(0, -90.0, 0)))
        if a is None:
            facts["errors"].append(f"spawn {bp.id}")
            continue
        a.set_simulate_physics(False)
        world.tick()
        shoot(world, lib, look_at(carla.Location(loc.x, loc.y - 3.2, loc.z + 0.2), carla.Location(loc.x, loc.y, loc.z)),
              f"walker_{bp.id.split('.')[-1]}", w=360, h=540, fov=45, ticks=10)
        a.destroy()
        print("walker", bp.id, flush=True)

    # ------------------------------------------------------------------ props
    for i, pid in enumerate(p for p in facts["props"] if any(w in p for w in PROP_WORDS)):
        loc = carla.Location(X0 + 8.0 * i, Y0 + 120.0, Z0)
        a = world.try_spawn_actor(lib.find(pid), carla.Transform(loc))
        if a is None:
            facts["errors"].append(f"spawn {pid}")
            continue
        try:
            a.set_simulate_physics(False)
        except Exception:  # noqa: BLE001
            pass
        world.tick()
        bb = a.bounding_box
        facts.setdefault("prop_boxes", {})[pid] = [bb.extent.x, bb.extent.y, bb.extent.z, bb.location.z]
        r = max(bb.extent.x, bb.extent.y, bb.extent.z, 0.3)
        shoot(world, lib, look_at(carla.Location(loc.x - 3.2 * r, loc.y - 3.2 * r, loc.z + bb.location.z + 2.0 * r),
                                  carla.Location(loc.x, loc.y, loc.z + bb.location.z)),
              f"prop_{pid.split('.')[-1]}", w=480, h=360, fov=50, ticks=8)
        a.destroy()
        print("prop", pid, flush=True)

    s.synchronous_mode, s.fixed_delta_seconds = False, None
    world.apply_settings(s)
    (OUT / "probe.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print("done", len(facts["vehicles"]), "vehicles", len(facts["walkers"]), "walkers", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
