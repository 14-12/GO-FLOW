"""What Town05 really has, from CARLA itself (step 1 of giving the town's trips real destinations, 8 Nov):

* every environment object the map holds (buildings, static props, the cars parked for good, vegetation...): its
  CARLA label, mesh name and box, in CARLA and in SUMO coordinates -> out/town05_places/env_objects.json;
* the town seen from straight above at noon, a colour picture and CARLA's semantic labels from the same camera, with
  the camera's numbers so a pixel can be turned into a place -> out/town05_places/sat_rgb.png, sat_tags.png, sat.json
  (sat_tags.png: the red channel is the CityObjectLabel value of each pixel).

CARLA must be running (tools/town05_places_probe.bat starts it off screen).
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
SIZE = 2048
HEIGHT = 1000.0


def shoot(world, lib, kind: str, tf, fov: float):
    bp = lib.find(kind)
    bp.set_attribute("image_size_x", str(SIZE))
    bp.set_attribute("image_size_y", str(SIZE))
    bp.set_attribute("fov", str(fov))
    cam = world.spawn_actor(bp, tf)
    q: queue.Queue = queue.Queue()
    # the picture copied out at once, in CARLA's own thread: holding its buffer past the camera's end crashed the
    # client (8 Nov, access violation)
    cam.listen(lambda im: q.put((im.frame, np.array(np.frombuffer(im.raw_data, dtype=np.uint8)
                                                    .reshape(im.height, im.width, 4)))))
    img = None
    try:
        frame = None
        for _ in range(12):            # let the streaming settle
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
        time.sleep(0.5)
        cam.destroy()
    if img is None:
        raise RuntimeError(f"no picture from {kind}")
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scen = scenario_mod.load("town05", ROOT)
    frame = scen.frame
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

    objs = []
    for o in ([] if (OUT / "env_objects.json").is_file() else world.get_environment_objects(carla.CityObjectLabel.Any)):
        bb = o.bounding_box
        loc = bb.location
        sx, sy = frame.to_sumo_xy(loc.x, loc.y)
        objs.append({"id": int(o.id), "name": o.name, "label": str(o.type), "carla": [round(loc.x, 2), round(loc.y, 2),
                                                                                      round(loc.z, 2)],
                     "sumo": [round(sx, 2), round(sy, 2)], "ext": [round(bb.extent.x, 2), round(bb.extent.y, 2),
                                                                    round(bb.extent.z, 2)],
                     "yaw": round(bb.rotation.yaw, 1)})
    if objs:
        (OUT / "env_objects.json").write_text(json.dumps(objs), encoding="utf-8")
        print(f"{len(objs)} environment objects")

    # the town from above: SUMO's bounds, the camera over their middle, north (SUMO +y = CARLA -y) up
    import sumolib

    net = sumolib.net.readNet(str(scen.net_file))
    (x0, y0), (x1, y1) = net.getBBoxXY()
    pad = 60.0
    span = max(x1 - x0, y1 - y0) + 2 * pad
    cx, cy = frame.to_carla_xy((x0 + x1) / 2, (y0 + y1) / 2)
    fov = math.degrees(2 * math.atan(span / 2 / HEIGHT))
    tf = carla.Transform(carla.Location(cx, cy, HEIGHT), carla.Rotation(-90.0, -90.0, 0.0))
    rgb = shoot(world, lib, "sensor.camera.rgb", tf, fov)
    Image.fromarray(rgb[:, :, :3][:, :, ::-1]).save(OUT / "sat_rgb.png")
    tags = shoot(world, lib, "sensor.camera.semantic_segmentation", tf, fov)
    Image.fromarray(tags[:, :, 2]).save(OUT / "sat_tags.png")    # BGRA: the label is in R
    sat = {"size": SIZE, "fov": fov, "height": HEIGHT, "carla_centre": [cx, cy],
           "sumo_centre": [(x0 + x1) / 2, (y0 + y1) / 2], "span_m": span,
           "metres_per_pixel_at_ground": span / SIZE,
           "orientation": "image up = SUMO +y (north), image right = SUMO +x",
           "labels": {str(int(v)): k for k, v in carla.CityObjectLabel.names.items()}}
    (OUT / "sat.json").write_text(json.dumps(sat, indent=1), encoding="utf-8")
    print(f"pictures {SIZE}x{SIZE}, fov {fov:.1f}, {span / SIZE:.3f} m a pixel")
    s.synchronous_mode = False
    world.apply_settings(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
