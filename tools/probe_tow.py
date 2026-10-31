"""The recovery truck's deck on the HGV (24 Oct): where the chassis is, and how the deck and a loaded car look.

    python tools/probe_tow.py      (tools/probe_tow.bat starts CARLA off screen, SM_TnDeck installed)

A. The HGV's outline from the side (depth camera, 14 m off, at half its height): the top of the body in 10 cm steps
   along it, in its own frame -> the cab's back (where the top drops) and the chassis' top behind it
   (out/_probe/tow/profile.json). cosim/towing.py's DECK_FRONT_X, CHASSIS_Z and PIVOT_X come from it.
B. The deck (SM_TnDeck with cosim/towing.deck_texture) put on it as cosim/towing.py says, photographed level, slid
   back, tilted, and with a car at the start of the winch, half way up and loaded (out/_probe/tow/*.png).
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
from probe_plates2 import FOV, H, W, Cams, depth_of, look, rgb_of  # noqa: E402

from cosim import towing  # noqa: E402
from cosim.pose_proc import compose, deck_tf  # noqa: E402

OUT = ROOT / "out" / "_probe" / "tow"
X0, Y0, Z0 = -150.0, -160.0, 300.0
CAR = "vehicle.lincoln.mkz_2020"


def points(depth, cam_tf, actor_tf):
    f = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
    v, u = np.nonzero(depth < 40.0)
    d = depth[v, u]
    xc, yc, zc = d, (u - W / 2.0) / f * d, -(v - H / 2.0) / f * d
    pw = np.array(cam_tf.get_matrix()) @ np.vstack([xc, yc, zc, np.ones_like(xc)])
    return (np.array(actor_tf.get_inverse_matrix()) @ pw)[:3].T


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
    cams = Cams(world, lib)
    rep: dict = {"errors": []}
    actors = []
    try:
        bp = lib.find(towing.TRUCK_MODEL)
        if bp.has_attribute("color"):
            bp.set_attribute("color", "230,230,230")
        truck = world.spawn_actor(bp, carla.Transform(carla.Location(X0, Y0, Z0)))
        actors.append(truck)
        truck.set_simulate_physics(False)
        world.tick()
        ttf = truck.get_transform()
        bb = truck.bounding_box
        rep["truck_bbox"] = {"loc": [bb.location.x, bb.location.y, bb.location.z],
                             "ext": [bb.extent.x, bb.extent.y, bb.extent.z]}
        # ---- A: the outline from both sides
        pts = []
        for side in (-1.0, 1.0):
            eye = carla.Location(ttf.location.x, ttf.location.y + side * 14.0, ttf.location.z + 1.7)
            ctf = look(eye, carla.Location(ttf.location.x, ttf.location.y, ttf.location.z + 1.7))
            rgb, dep = cams.shot(ctf)
            if dep is not None:
                p = points(depth_of(dep), ctf, ttf)
                pts.append(p[(np.abs(p[:, 0]) < 4.3) & (np.abs(p[:, 1]) < 1.6) & (p[:, 2] > -0.2) & (p[:, 2] < 4.0)])
            if rgb is not None:
                Image.fromarray(rgb_of(rgb)).save(OUT / f"A_side_{'L' if side < 0 else 'R'}.png")
        # and from above: the chassis' top between the frame rails
        eye = carla.Location(ttf.location.x - 1.0, ttf.location.y, ttf.location.z + 9.0)
        ctf = carla.Transform(eye, carla.Rotation(-89.0, 0.0, 0.0))
        rgb, dep = cams.shot(ctf)
        if dep is not None:
            p = points(depth_of(dep), ctf, ttf)
            top = p[(np.abs(p[:, 0]) < 4.3) & (np.abs(p[:, 1]) < 1.3) & (p[:, 2] > -0.2) & (p[:, 2] < 4.0)]
            prof_top = {}
            for x in np.arange(-4.2, 4.2, 0.1):
                sel = top[(top[:, 0] >= x) & (top[:, 0] < x + 0.1) & (np.abs(top[:, 1]) < 0.9)]
                if len(sel) > 5:
                    prof_top[round(float(x), 1)] = [round(float(np.percentile(sel[:, 2], 90)), 3),
                                                    round(float(np.median(sel[:, 2])), 3)]
            rep["top_view_z_by_x"] = prof_top
        if rgb is not None:
            Image.fromarray(rgb_of(rgb)).save(OUT / "A_top.png")
        if pts:
            p = np.vstack(pts)
            prof = {}
            for x in np.arange(-4.2, 4.2, 0.1):
                sel = p[(p[:, 0] >= x) & (p[:, 0] < x + 0.1)]
                if len(sel) > 5:
                    prof[round(float(x), 1)] = round(float(np.percentile(sel[:, 2], 98)), 3)
            rep["side_top_z_by_x"] = prof
        print("outline", json.dumps({"side": rep.get("side_top_z_by_x"), "top": rep.get("top_view_z_by_x")}), flush=True)

        # ---- B: the deck and a car
        dbp = lib.find("static.prop.mesh")
        dbp.set_attribute("mesh_path", towing.DECK_MESH)
        dbp.set_attribute("mass", "1.0")
        before = set(world.get_names_of_all_objects())
        deck = world.spawn_actor(dbp, deck_tf(ttf, towing.deck_level()))
        actors.append(deck)
        deck.set_simulate_physics(False)
        world.tick()
        new = [n for n in set(world.get_names_of_all_objects()) - before]
        rep["deck_names"] = new
        from cosim import plates

        tex = plates.to_texture(towing.deck_texture())
        world.apply_color_texture_to_objects(new, carla.MaterialParameter.Diffuse, tex)
        cbp = lib.find(CAR)
        if cbp.has_attribute("color"):
            cbp.set_attribute("color", "150,20,24")
        car = world.spawn_actor(cbp, carla.Transform(carla.Location(X0, Y0 + 40.0, Z0)))
        actors.append(car)
        car.set_simulate_physics(False)
        world.tick()
        car_len = 2 * car.bounding_box.extent.x
        path = towing.Path()
        park = carla.Transform(carla.Location(X0, Y0 + 40.0, Z0))

        def car_at(s_arc):
            x, z, pitch = path.car(s_arc, car_len)
            return compose(ttf, (x, 0.0, z, pitch, 0.0))

        shots = [("level", towing.deck_pose(0, 0), None), ("slid", towing.deck_pose(1, 0), None),
                 ("tilt_half", towing.deck_pose(1, 0.5), None),
                 ("tilted_car_start", towing.deck_pose(1, 1), path.s_start(car_len)),
                 ("winch_half", towing.deck_pose(1, 1), 1.2), ("winch_up", towing.deck_pose(1, 1), 3.0),
                 ("winch_end", towing.deck_pose(1, 1), path.s_end(car_len)), ("loaded", towing.deck_pose(0, 0), "on")]
        for name, drel, s_arc in shots:
            deck.set_transform(deck_tf(ttf, drel))
            if s_arc is None:
                car.set_transform(park)
            elif s_arc == "on":
                car.set_transform(compose(ttf, towing.car_on_deck()))
            else:
                car.set_transform(car_at(s_arc))
            for tag, eye in (("side", carla.Location(ttf.location.x - 3.0, ttf.location.y + 15.0, ttf.location.z + 2.0)),
                             ("rear", carla.Location(ttf.location.x - 17.0, ttf.location.y + 6.0, ttf.location.z + 3.5))):
                rgb, _ = cams.shot(look(eye, carla.Location(ttf.location.x - 3.5, ttf.location.y, ttf.location.z + 1.0)))
                if rgb is not None:
                    Image.fromarray(rgb_of(rgb)).save(OUT / f"B_{name}_{tag}.png")
            print("shot", name, flush=True)
    except Exception:  # noqa: BLE001
        rep["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        for a in actors:
            try:
                a.destroy()
            except Exception:  # noqa: BLE001
                pass
        cams.destroy()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
        (OUT / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
