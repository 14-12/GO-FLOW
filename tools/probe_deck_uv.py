"""Why the deck shows one colour (24 Oct): SM_TnDeck and SM_TnPlate (scaled up) side by side, each given a test
texture of four coloured quadrants and a grid, photographed from above and from the side (out/_probe/deck_uv)."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import carla  # noqa: E402
from probe_plates2 import Cams, look, rgb_of  # noqa: E402

from cosim import plates, towing  # noqa: E402
from cosim.pose_proc import deck_tf  # noqa: E402

OUT = ROOT / "out" / "_probe" / "deck_uv"
X0, Y0, Z0 = -150.0, -120.0, 300.0


def quad(w=256, h=128):
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w // 2 - 1, h // 2 - 1], fill=(220, 30, 30))       # top-left red
    d.rectangle([w // 2, 0, w - 1, h // 2 - 1], fill=(30, 200, 30))       # top-right green
    d.rectangle([0, h // 2, w // 2 - 1, h - 1], fill=(30, 60, 220))       # bottom-left blue
    d.rectangle([w // 2, h // 2, w - 1, h - 1], fill=(240, 220, 30))      # bottom-right yellow
    for x in range(0, w, 16):
        d.line([x, 0, x, h], fill=(0, 0, 0))
    for y in range(0, h, 16):
        d.line([0, y, w, y], fill=(0, 0, 0))
    return im


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
    rep = {"errors": []}
    actors = []
    try:
        tex = plates.to_texture(quad())
        specs = [("deck", towing.DECK_MESH, 1.0, deck_tf(carla.Transform(carla.Location(X0, Y0, Z0)), (0, 0, 0, 0, 0))),
                 ("deck_flat_raw", towing.DECK_MESH, 1.0, carla.Transform(carla.Location(X0, Y0 + 10, Z0))),
                 ("plate_big", "/Game/Carla/Static/Other/SM_TnPlate.SM_TnPlate", 8.0,
                  carla.Transform(carla.Location(X0, Y0 + 20, Z0 + 1), carla.Rotation(0, 0, 0)))]
        for name, mesh, scale, tf in specs:
            bp = lib.find("static.prop.mesh")
            bp.set_attribute("mesh_path", mesh)
            bp.set_attribute("scale", str(scale))
            before = set(world.get_names_of_all_objects())
            a = world.spawn_actor(bp, tf)
            actors.append(a)
            world.tick()
            new = sorted(set(world.get_names_of_all_objects()) - before)
            rep[name] = new
            world.apply_color_texture_to_objects(new, carla.MaterialParameter.Diffuse, tex)
            world.tick()
            loc = tf.location
            for tag, eye in (("above", carla.Location(loc.x - 0.5, loc.y, loc.z + 9.0)),
                             ("side", carla.Location(loc.x - 2.0, loc.y - 8.0, loc.z + 3.0)),
                             ("plusx", carla.Location(loc.x + 8.0, loc.y + 0.5, loc.z + 1.0)),
                             ("minusx", carla.Location(loc.x - 8.0, loc.y + 0.5, loc.z + 1.0))):
                rgb, _ = cams.shot(look(eye, loc))
                if rgb is not None:
                    Image.fromarray(rgb_of(rgb)).save(OUT / f"{name}_{tag}.png")
            print(name, new, flush=True)
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
