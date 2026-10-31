"""Plates, second probe (23 Oct): where each car model's plates are, and whether a plate actor of our own works.

    python tools/probe_plates2.py      (CARLA running, T_LicensePlate_d magenta, SM_TnPlate installed;
                                         tools/probe_plates2.bat does all of it)

A. The plate mesh (tools/plates/make_plate_mesh.py): one static.prop.mesh SM_TnPlate in the sky, its scene name
   found by the names before and after, given a texture with its corners marked through the runtime texture API;
   photographed from both sides (which way the texture lies on the faces).
B. Every car model's plates: T_LicensePlate_d is installed plain magenta, so a car's plates are the only magenta in
   the picture. An RGB and a depth camera at the rear and at the front of each model, 3.2 m away at the body's
   middle height; each magenta pixel, put in 3D with its depth, then in the car's own frame, gives the plate's
   centre, width, height and tilt (out/_probe/plates2/plates.json, used by cosim/plate_proc.py).
C. Two plates of our own attached to a few models where B found theirs, with Tunisian plates; photographed.
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out" / "_probe" / "plates2"
import carla  # noqa: E402

MESH = "/Game/Carla/Static/Other/SM_TnPlate.SM_TnPlate"
X0, Y0, Z0 = -150.0, -260.0, 300.0
W, H, FOV = 800, 600, 50.0


class Cams:
    """An RGB and a depth camera moved together; shot() returns both pictures of one frame."""

    def __init__(self, world, lib):
        self.world = world
        self.cams, self.q = [], []
        for kind in ("sensor.camera.rgb", "sensor.camera.depth"):
            bp = lib.find(kind)
            bp.set_attribute("image_size_x", str(W))
            bp.set_attribute("image_size_y", str(H))
            bp.set_attribute("fov", str(FOV))
            c = world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, 500)))
            q: queue.Queue = queue.Queue()
            c.listen(q.put)
            self.cams.append(c)
            self.q.append(q)

    def shot(self, tf, ticks: int = 6):
        for c in self.cams:
            c.set_transform(tf)
        frame = None
        for _ in range(ticks):
            frame = self.world.tick()
        out = []
        for q in self.q:
            img = None
            t = time.time()
            while time.time() - t < 5.0:
                try:
                    im = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if im.frame >= frame:
                    img = im
                    break
            out.append(img)
            while not q.empty():
                q.get_nowait()
        return out

    def destroy(self):
        for c in self.cams:
            c.stop()
            c.destroy()


def rgb_of(img) -> np.ndarray:
    return np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3][:, :, ::-1]


def depth_of(img) -> np.ndarray:
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4).astype(np.float64)
    return (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / 16777215.0 * 1000.0


def look(eye: carla.Location, target: carla.Location) -> carla.Transform:
    d = target - eye
    return carla.Transform(eye, carla.Rotation(math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))),
                                               math.degrees(math.atan2(d.y, d.x)), 0.0))


def magenta_points(rgb: np.ndarray, depth: np.ndarray, cam_tf: carla.Transform, actor_tf: carla.Transform):
    """The magenta pixels as points in the actor's frame (x forward, y right, z up), and the mask."""
    r, g, b = (rgb[:, :, k].astype(np.int32) for k in range(3))
    # magenta, also when shaded or lit pale by the sky (230, 160, 225): red and blue both well above green
    mask = (r - g > 35) & (b - g > 25) & (np.abs(r - b) < 90) & (depth < 12.0)
    v, u = np.nonzero(mask)
    if len(u) < 30:
        return None, mask
    f = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
    d = depth[v, u]
    xc, yc, zc = d, (u - W / 2.0) / f * d, -(v - H / 2.0) / f * d
    m = np.array(cam_tf.get_matrix())
    pw = m @ np.vstack([xc, yc, zc, np.ones_like(xc)])
    inv = np.array(actor_tf.get_inverse_matrix())
    pl = inv @ pw
    return pl[:3].T, mask


def describe(pts: np.ndarray) -> dict:
    """Centre, width (y), height (z) and tilt of a plate from its points (robust: 3rd-97th percentiles)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    lo, hi = np.percentile(y, [3, 97])
    zl, zh = np.percentile(z, [3, 97])
    # tilt: x as a straight line of z (a plate leaning back has x changing with height)
    A = np.vstack([z, np.ones_like(z)]).T
    k, _ = np.linalg.lstsq(A, x, rcond=None)[0]
    return {"x": round(float(np.median(x)), 4), "y": round(float((lo + hi) / 2), 4), "z": round(float((zl + zh) / 2), 4),
            "w": round(float(hi - lo), 4), "h": round(float(zh - zl), 4),
            "tilt_deg": round(math.degrees(math.atan(k)), 2), "n": int(len(x))}


def names(world) -> set:
    return set(world.get_names_of_all_objects())


def corner_texture(w: int = 208, h: int = 44):
    im = Image.new("RGB", (w, h), (30, 30, 30))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w // 4, h // 2], fill=(230, 40, 40))       # top-left red
    d.rectangle([3 * w // 4, 0, w - 1, h // 2], fill=(40, 200, 40))  # top-right green
    d.rectangle([0, h // 2, w // 4, h - 1], fill=(40, 80, 230))    # bottom-left blue
    d.text((w * 0.35, h * 0.3), "TOP L>R", fill=(255, 255, 0))
    return im


def main() -> int:
    from cosim import plates

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
    cams = Cams(world, lib)
    try:
        # ---- A: the plate mesh
        bp = lib.find("static.prop.mesh")
        bp.set_attribute("mesh_path", MESH)
        bp.set_attribute("scale", "1.0")
        before = names(world)
        tfA = carla.Transform(carla.Location(X0, Y0 - 30.0, Z0), carla.Rotation(0, 0, 0))
        prop = world.spawn_actor(bp, tfA)
        prop.set_simulate_physics(False)
        world.tick()
        new = sorted(names(world) - before)
        rep["prop_names"] = new
        rep["prop_bbox"] = [prop.bounding_box.extent.x, prop.bounding_box.extent.y, prop.bounding_box.extent.z]
        t = time.perf_counter()
        tex = plates.to_texture(corner_texture())
        rep["texture_s"] = round(time.perf_counter() - t, 3)
        for n in new:
            t = time.perf_counter()
            world.apply_color_texture_to_object(n, carla.MaterialParameter.Diffuse, tex)
            rep.setdefault("apply_s", []).append(round(time.perf_counter() - t, 4))
        for side, dx in (("plusX", 1.2), ("minusX", -1.2)):
            eye = carla.Location(tfA.location.x + dx, tfA.location.y, tfA.location.z)
            rgb, _ = cams.shot(look(eye, tfA.location))
            if rgb is not None:
                Image.fromarray(rgb_of(rgb)).save(OUT / f"A_prop_{side}.png")
        prop.destroy()

        # ---- B: every model's plates
        models = sorted(b.id for b in lib.filter("vehicle.*")
                        if not (b.has_attribute("base_type") and b.get_attribute("base_type").as_str().lower() == "bicycle"))
        rep["models"] = models
        found: dict = {}
        for i, mid in enumerate(models):
            vb = lib.find(mid)
            if vb.has_attribute("color"):
                vb.set_attribute("color", "120,120,120")
            tf = carla.Transform(carla.Location(X0 + 20.0 * (i % 10), Y0 + 25.0 * (i // 10), Z0), carla.Rotation(0, 0, 0))
            try:
                a = world.spawn_actor(vb, tf)
            except Exception as e:  # noqa: BLE001
                rep["errors"].append(f"{mid}: {e}")
                continue
            a.set_simulate_physics(False)
            world.tick()
            atf = a.get_transform()
            bb = a.bounding_box
            res = {"extent": [bb.extent.x, bb.extent.y, bb.extent.z], "centre": [bb.location.x, bb.location.y, bb.location.z]}
            for end, sgn in (("rear", -1.0), ("front", 1.0)):
                cz = atf.location.z + bb.location.z
                eye = carla.Location(atf.location.x + bb.location.x + sgn * (bb.extent.x + 3.2), atf.location.y, cz)
                tgt = carla.Location(atf.location.x + bb.location.x + sgn * bb.extent.x, atf.location.y, cz)
                ctf = look(eye, tgt)
                rgb, dep = cams.shot(ctf)
                if rgb is None or dep is None:
                    res[end] = None
                    continue
                pts, mask = magenta_points(rgb_of(rgb), depth_of(dep), ctf, atf)
                res[end] = describe(pts) if pts is not None else None
                im = Image.fromarray(rgb_of(rgb)).convert("RGB")
                ov = Image.new("RGB", im.size, (0, 255, 0))
                im = Image.composite(ov, im, Image.fromarray((mask * 120).astype(np.uint8)))
                im.resize((400, 300)).save(OUT / f"B_{mid.replace('vehicle.', '').replace('.', '_')}_{end}.jpg", quality=85)
            found[mid] = res
            print(mid, {k: res.get(k) for k in ("rear", "front")}, flush=True)
            a.destroy()
            world.tick()
        (OUT / "plates.json").write_text(json.dumps(found, indent=1), encoding="utf-8")

        # ---- C: our plates on a few models
        import random

        rng = random.Random(5)
        pick = [m for m in ("vehicle.lincoln.mkz_2020", "vehicle.mini.cooper_s_2021", "vehicle.mercedes.sprinter",
                            "vehicle.carlamotors.european_hgv", "vehicle.tesla.model3", "vehicle.audi.a2") if m in found]
        for i, mid in enumerate(pick):
            res = found[mid]
            vb = lib.find(mid)
            if vb.has_attribute("color"):
                vb.set_attribute("color", "200,200,200")
            tf = carla.Transform(carla.Location(X0 + 20.0 * i, Y0 + 120.0, Z0), carla.Rotation(0, 0, 0))
            a = world.spawn_actor(vb, tf)
            a.set_simulate_physics(False)
            world.tick()
            plate = plates.random_plate(rng)
            tex = plates.to_texture(plates.render(plate))
            for end, yaw, size in (("rear", 180.0, plates.REAR_MM), ("front", 0.0, plates.FRONT_MM)):
                p = res.get(end)
                if not p:
                    continue
                pb = lib.find("static.prop.mesh")
                pb.set_attribute("mesh_path", MESH)
                pb.set_attribute("scale", f"{size[0] / 520.0:.3f}")
                off = 0.006 if end == "front" else -0.006
                rel = carla.Transform(carla.Location(p["x"] + off, p["y"], p["z"]),
                                      carla.Rotation(-p["tilt_deg"] if end == "rear" else p["tilt_deg"], yaw, 0.0))
                before = names(world)
                pa = world.spawn_actor(pb, rel, attach_to=a)
                pa.set_simulate_physics(False)  # a prop with physics falls off its parent
                world.tick()
                rep.setdefault("attached", []).append({"model": mid, "end": end, "want": [p["x"], p["y"], p["z"]],
                                                       "got": [round(v, 3) for v in (pa.get_transform().location.x - a.get_transform().location.x,
                                                                                     pa.get_transform().location.y - a.get_transform().location.y,
                                                                                     pa.get_transform().location.z - a.get_transform().location.z)]})
                for n in sorted(names(world) - before):
                    world.apply_color_texture_to_object(n, carla.MaterialParameter.Diffuse, tex)
            atf = a.get_transform()
            bb = a.bounding_box
            for end, sgn in (("rear", -1.0), ("front", 1.0)):
                p = res.get(end) or {"z": bb.location.z}
                cz = atf.location.z + p["z"]
                eye = carla.Location(atf.location.x + sgn * (bb.extent.x + 2.2), atf.location.y + 0.6, cz + 0.4)
                tgt = carla.Location(atf.location.x + sgn * bb.extent.x, atf.location.y, cz)
                rgb, _ = cams.shot(look(eye, tgt))
                if rgb is not None:
                    Image.fromarray(rgb_of(rgb)).save(OUT / f"C_{mid.replace('vehicle.', '').replace('.', '_')}_{end}.png")
            eye = carla.Location(atf.location.x - bb.extent.x - 9.0, atf.location.y + 3.0, atf.location.z + 2.0)
            rgb, _ = cams.shot(look(eye, atf.location))
            if rgb is not None:
                Image.fromarray(rgb_of(rgb)).save(OUT / f"C_{mid.replace('vehicle.', '').replace('.', '_')}_far.png")
            print("C", mid, plate.text, flush=True)
    except Exception:  # noqa: BLE001
        rep["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        cams.destroy()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
        (OUT / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
