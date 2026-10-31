"""Can each vehicle get a plate texture of its own through CARLA's runtime texture API? (23 Oct)

    python tools/probe_plates_api.py      (CARLA running; tools/probe_plates_api.bat)

CARLA's plate materials take their texture from a parameter the API does not set; tools/plates/install_plates.py
renames it to "BaseColor", which `World.apply_color_texture_to_object` sets on every static mesh component of an
actor. This spawns every car model in the sky, works out which scene object is which actor (names of objects
before and after the spawn), gives each a plate texture with its actor id and a fine grid (to see which part of
the texture the plate shows), and photographs the rear, the front and the side (did anything else change?).
Times the texture building and the calls. out/_probe/plates_api/*.png, report.json.
"""

from __future__ import annotations

import json
import math
import queue
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "_probe" / "plates_api"
import carla  # noqa: E402

X0, Y0, Z0 = -150.0, -250.0, 300.0
MODELS = ["vehicle.lincoln.mkz_2020", "vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.dodge.charger_2020",
          "vehicle.mini.cooper_s_2021", "vehicle.nissan.patrol_2021", "vehicle.mercedes.coupe_2020", "vehicle.ford.crown",
          "vehicle.toyota.prius", "vehicle.seat.leon", "vehicle.citroen.c3", "vehicle.mercedes.sprinter",
          "vehicle.carlamotors.carlacola", "vehicle.mitsubishi.fusorosa", "vehicle.ford.mustang", "vehicle.audi.a2",
          "vehicle.lincoln.mkz_2020", "vehicle.lincoln.mkz_2020"]


def pattern(n: int, w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h), (20, 20, 20))
    d = ImageDraw.Draw(im)
    for j in range(32):  # bands of v, 1/32 each, alternating
        y0, y1 = j * h / 32, (j + 1) * h / 32
        d.rectangle([0, y0, w, y1], fill=(230, 230, 230) if j % 2 else (60, 120, 230))
        d.text((2, y0), f"{j}", fill=(200, 0, 0))
    for i in range(1, 16):
        x = i * w / 16
        d.line([x, 0, x, h], fill=(255, 0, 0), width=1)
    font = ImageFont.load_default(size=max(10, h // 6)) if hasattr(ImageFont, "load_default") else None
    d.text((w * 0.25, h * 0.40), f"ID {n}", fill=(0, 0, 0), font=font, stroke_width=2, stroke_fill=(255, 255, 0))
    return im


def to_texture(img: Image.Image) -> carla.TextureColor:
    w, h = img.size
    tex = carla.TextureColor(w, h)
    a = np.asarray(img.convert("RGB"))
    for y in range(h):
        row = a[y]
        for x in range(w):
            r, g, b = row[x]
            tex.set(x, y, carla.Color(int(r), int(g), int(b), 255))
    return tex


def save(img, name: str) -> None:
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3][:, :, ::-1]
    Image.fromarray(arr).save(OUT / f"{name}.png")


def shoot(world, lib, tf, name: str, w=960, h=540, fov=35.0, ticks=14) -> None:
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
            time.sleep(0.04)
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


def look_at(eye, target):
    d = target - eye
    return carla.Transform(eye, carla.Rotation(math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))),
                                               math.degrees(math.atan2(d.y, d.x)), 0.0))


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
    rep: dict = {"errors": []}
    t = time.perf_counter()
    before = set(world.get_names_of_all_objects())
    rep["names_call_s"] = round(time.perf_counter() - t, 3)
    batch = []
    for i, mid in enumerate(MODELS):
        bp = lib.find(mid)
        if bp.has_attribute("color"):
            bp.set_attribute("color", "200,200,200")
        tf = carla.Transform(carla.Location(X0 + 14.0 * i, Y0, Z0), carla.Rotation(0.0, 90.0, 0.0))
        batch.append(carla.command.SpawnActor(bp, tf).then(carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
    res = client.apply_batch_sync(batch, True)
    ids = [r.actor_id for r in res]
    world.tick()
    after = world.get_names_of_all_objects()
    new = [n for n in after if n not in before]
    rep["new_names"] = new
    # actor -> name: by class prefix, ordered (ids ascending, name numbers descending: spawn order, measured)
    actors = {a.id: a for a in world.get_actors(ids)}
    by_cls: dict = {}
    for n in new:
        m = re.match(r"(.+)_C_(\d+)$", n)
        if m:
            by_cls.setdefault(m.group(1), []).append((int(m.group(2)), n))
    rep["by_class"] = {k: [x[1] for x in v] for k, v in by_cls.items()}
    for size in ((256, 128), (512, 256)):
        t = time.perf_counter()
        to_texture(pattern(0, *size))
        rep[f"texture_build_s_{size[0]}x{size[1]}"] = round(time.perf_counter() - t, 3)
    mapping = {}
    for (aid, mid) in zip(ids, MODELS, strict=True):
        cands = [n for n in new if n.lower().startswith("bp_")]
        mapping[aid] = mid
    # each actor: try every new name that could be its class, in order; record which one is its
    names_by_model: dict = {}
    for aid, mid in zip(ids, MODELS, strict=True):
        a = actors.get(aid)
        names_by_model[mid] = None
    calls = []
    for k, n in enumerate(new):
        tex = to_texture(pattern(k, 256, 128))
        t = time.perf_counter()
        try:
            world.apply_color_texture_to_object(n, carla.MaterialParameter.Diffuse, tex)
        except Exception as e:  # noqa: BLE001
            rep["errors"].append(f"apply {n}: {e}")
        calls.append(time.perf_counter() - t)
    rep["apply_call_s"] = [round(c, 3) for c in calls]
    rep["texture_index_of_name"] = {n: k for k, n in enumerate(new)}
    rep["actor_ids"] = list(zip(MODELS, ids, strict=True))
    for i, (aid, mid) in enumerate(zip(ids, MODELS, strict=True)):
        a = actors.get(aid)
        if a is None:
            continue
        tag = f'{i:02d}_{aid}_' + mid.replace("vehicle.", "").replace(".", "_")
        loc = a.get_transform().location
        bb = a.bounding_box
        c = carla.Location(loc.x, loc.y + bb.location.x, loc.z + bb.location.z)
        back = carla.Location(c.x, c.y - bb.extent.x, loc.z + 0.55)
        front = carla.Location(c.x, c.y + bb.extent.x, loc.z + 0.55)
        try:
            shoot(world, lib, look_at(carla.Location(c.x, c.y - bb.extent.x - 3.0, loc.z + 0.7), back), f"{tag}_rear", fov=22)
            shoot(world, lib, look_at(carla.Location(c.x, c.y + bb.extent.x + 3.0, loc.z + 0.7), front), f"{tag}_front", fov=22)
            shoot(world, lib, look_at(carla.Location(c.x + 6.5, c.y - 3.0, loc.z + 1.8), c), f"{tag}_side", fov=60)
            shoot(world, lib, look_at(carla.Location(c.x, c.y - bb.extent.x - 14.0, loc.z + 1.5), back), f"{tag}_far", fov=30)
        except Exception:  # noqa: BLE001
            rep["errors"].append(traceback.format_exc())
        print("shot", mid, flush=True)
    s.synchronous_mode, s.fixed_delta_seconds = False, None
    world.apply_settings(s)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
