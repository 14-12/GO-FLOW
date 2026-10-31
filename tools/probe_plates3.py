"""Plates, third probe (23 Oct): where to put a plate on the model ends that have no plate of their own.

    python tools/probe_plates3.py      (tools/probe_plates3.bat starts CARLA off screen and runs it)

tools/probe_plates2.py found each model's plates by making CARLA's plate texture magenta. Seven ends have no plate
material (or one far wider than a plate) and were left bare: the BMW Gran Tourer's rear, both police Chargers' and the
European HGV's fronts, the Vespa's rear. Here each such end is photographed with a depth camera; the body's surface
behind a plate-sized patch (plate width x 12 cm) is found at every height of the end's plate band, and the plate goes
where that patch is flattest (a bumper face or a tailgate, not a grille's corner), 8 mm proud of its outermost point,
tilted as the surface leans. Then the plate is put there (a prop, posed with the car's transform) and photographed.

Out: out/_probe/plates3/plates.json ({model: {end: [x, y, z, pitch, yaw, scale]}}, the format of
config/vehicle_models.json "plates") and D_<model>_<end>.png pictures.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import carla  # noqa: E402
from probe_plates2 import FOV, MESH, H, W, Cams, depth_of, look, rgb_of  # noqa: E402

OUT = ROOT / "out" / "_probe" / "plates3"
X0, Y0, Z0 = -150.0, -200.0, 300.0
# (model, end): (plate band low, high in m above the car's origin, plate width m)
WANTED = {
    ("vehicle.bmw.grandtourer", "rear"): (0.70, 0.95, 0.52),
    ("vehicle.dodge.charger_police", "front"): (0.35, 0.60, 0.45),
    ("vehicle.dodge.charger_police_2020", "front"): (0.35, 0.60, 0.45),
    ("vehicle.carlamotors.european_hgv", "front"): (0.45, 0.85, 0.52),
    ("vehicle.vespa.zx125", "rear"): (0.45, 0.85, 0.30),
}
PROUD = 0.008
PLATE_H = 0.11


def surface_points(rgb, depth, cam_tf, actor_tf, actor_bb, end_sign):
    """Points of the car itself (inside its box, slightly grown) seen at one end, in the actor's frame."""
    f = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
    v, u = np.nonzero(depth < 15.0)
    d = depth[v, u]
    xc, yc, zc = d, (u - W / 2.0) / f * d, -(v - H / 2.0) / f * d
    pw = np.array(cam_tf.get_matrix()) @ np.vstack([xc, yc, zc, np.ones_like(xc)])
    p = (np.array(actor_tf.get_inverse_matrix()) @ pw)[:3].T
    c, e = actor_bb.location, actor_bb.extent
    inside = (np.abs(p[:, 0] - c.x) < e.x + 0.15) & (np.abs(p[:, 1] - c.y) < e.y + 0.15) & \
             (p[:, 2] > c.z - e.z - 0.1) & (p[:, 2] < c.z + e.z + 0.15) & (end_sign * (p[:, 0] - c.x) > 0.3 * e.x)
    return p[inside]


def best_spot(pts, end_sign, lo, hi, width):
    """The flattest plate-sized patch of the surface between heights lo and hi: (x, z, tilt_deg, rms_m)."""
    half = width / 2.0
    best = None
    for z in np.arange(lo, hi + 1e-6, 0.02):
        sel = pts[(np.abs(pts[:, 1]) < half) & (np.abs(pts[:, 2] - z) < PLATE_H / 2.0)]
        if len(sel) < 40:
            continue
        # the outer surface: per (y, z) cell the outermost point (a camera sees the nearest surface anyway)
        x_out = end_sign * sel[:, 0]
        # covered: the patch must be seen over most of its width (no hole: a grille opening, a gap under a bumper)
        cover = len(np.unique(np.round(sel[:, 1] / 0.04))) / max(int(width / 0.04), 1)
        if cover < 0.8:
            continue
        A = np.vstack([sel[:, 2], np.ones(len(sel))]).T
        k, b = np.linalg.lstsq(A, x_out, rcond=None)[0]
        rms = float(np.sqrt(np.mean((x_out - (k * sel[:, 2] + b)) ** 2)))
        outer = float(np.percentile(x_out, 98))
        score = rms + 0.02 * abs(z - (lo + hi) / 2.0)  # flattest, then nearest the band's middle
        if best is None or score < best[4]:
            best = (end_sign * (outer + PROUD), float(z), math.degrees(math.atan(end_sign * k)), rms, score)  # tilt as probe_plates2's: x(z) in the car's frame
    return best


def main() -> int:
    from cosim import plates
    from cosim.pose_proc import compose

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
    cams = Cams(world, lib)
    found: dict = {}
    rep: dict = {"errors": [], "spots": {}}
    import random

    rng = random.Random(9)
    try:
        for i, ((mid, end), (lo, hi, width)) in enumerate(WANTED.items()):
            vb = lib.find(mid)
            if vb.has_attribute("color"):
                vb.set_attribute("color", "200,200,200")
            tf = carla.Transform(carla.Location(X0 + 25.0 * i, Y0, Z0), carla.Rotation(0, 0, 0))
            a = world.spawn_actor(vb, tf)
            a.set_simulate_physics(False)
            world.tick()
            atf, bb = a.get_transform(), a.bounding_box
            sgn = -1.0 if end == "rear" else 1.0
            pts_all = []
            # three views (straight, a little left and right) so the surface is seen without holes
            for side in (0.0, -0.8, 0.8):
                cz = atf.location.z + (lo + hi) / 2.0
                eye = carla.Location(atf.location.x + bb.location.x + sgn * (bb.extent.x + 2.6), atf.location.y + side, cz)
                tgt = carla.Location(atf.location.x + bb.location.x + sgn * bb.extent.x, atf.location.y, cz)
                ctf = look(eye, tgt)
                rgb, dep = cams.shot(ctf)
                if rgb is None or dep is None:
                    continue
                pts_all.append(surface_points(rgb_of(rgb), depth_of(dep), ctf, atf, bb, sgn))
                if side == 0.0:
                    Image.fromarray(rgb_of(rgb)).save(OUT / f"S_{mid.replace('vehicle.', '').replace('.', '_')}_{end}.jpg")
            pts = np.vstack(pts_all) if pts_all else np.zeros((0, 3))
            spot = best_spot(pts, sgn, lo, hi, width)
            rep["spots"][f"{mid} {end}"] = None if spot is None else [round(v, 4) for v in spot[:4]] + [len(pts)]
            print(mid, end, rep["spots"][f"{mid} {end}"], flush=True)
            if spot is not None:
                x, z, tilt, _, _ = spot
                # the probe_plates2 convention: rear yaw 0, pitch -tilt; front yaw 180, pitch +tilt
                rel = [round(x, 4), 0.0, round(z, 4), round(-tilt if end == "rear" else tilt, 2),
                       0.0 if end == "rear" else 180.0, round(width / 0.52, 4)]
                found.setdefault(mid, {})[end] = rel
                # the plate put there, photographed
                pb = lib.find("static.prop.mesh")
                pb.set_attribute("mesh_path", MESH)
                pb.set_attribute("scale", f"{rel[5]:.4f}")
                before = set(world.get_names_of_all_objects())
                pa = world.spawn_actor(pb, compose(atf, tuple(rel[:5])))
                world.tick()
                tex = plates.to_texture(plates.render(plates.random_plate(rng)))
                for n in set(world.get_names_of_all_objects()) - before:
                    world.apply_color_texture_to_object(n, carla.MaterialParameter.Diffuse, tex)
                cz = atf.location.z + z
                for tag, side, back, up in (("near", 0.7, 2.2, 0.35), ("far", 2.5, 7.0, 1.2)):
                    eye = carla.Location(atf.location.x + sgn * (bb.extent.x + back), atf.location.y + side, cz + up)
                    tgt = carla.Location(atf.location.x + sgn * bb.extent.x, atf.location.y, cz)
                    rgb, _ = cams.shot(look(eye, tgt))
                    if rgb is not None:
                        Image.fromarray(rgb_of(rgb)).save(
                            OUT / f"D_{mid.replace('vehicle.', '').replace('.', '_')}_{end}_{tag}.png")
                pa.destroy()
            a.destroy()
            world.tick()
    except Exception:  # noqa: BLE001
        rep["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        cams.destroy()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
        (OUT / "plates.json").write_text(json.dumps(found, indent=1), encoding="utf-8")
        (OUT / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
