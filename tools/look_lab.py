"""Look lab: the same views of Town05 under several weathers, for choosing the town's look (cosim/look.py).

    python tools/look_lab.py --tag base --times 12:30+17:45+20:45+21:45+23:30 --looks old+town
    python tools/look_lab.py --tag sweep --sweep --times 17:45+20:45

CARLA must be running (tools/look_lab.bat starts it with a render profile and full screen, as the viewer does).
No SUMO: a row of standing cars on the street views. For every time, view and look: a screenshot of CARLA's
window and the frame rate over 3 s (the server's own frame times). Contact sheets (looks side by side) and
fps.json go to --out/<tag>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import carla  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from cosim import look as look_mod  # noqa: E402
from cosim import scenario as scenario_mod  # noqa: E402
from cosim.scenario import INGOLSTADT  # noqa: E402
from cosim.solar import carla_sun_azimuth, scenario_sun  # noqa: E402

SWEEP = [("base", {}), ("cloud30", {"cloudiness": 30}), ("cloud55", {"cloudiness": 55}),
         ("fog5", {"fog_density": 5}), ("fog15", {"fog_density": 15}), ("fog5_d40", {"fog_density": 5, "fog_distance": 40}),
         ("fog15_d80", {"fog_density": 15, "fog_distance": 80}), ("fog10_fall1", {"fog_density": 10, "fog_falloff": 1.0}),
         ("fog10_fall3", {"fog_density": 10, "fog_falloff": 3.0}),
         ("scat0.5", {"scattering_intensity": 0.5}), ("scat2", {"scattering_intensity": 2.0}),
         ("mie0.01", {"mie_scattering_scale": 0.01}), ("mie0.1", {"mie_scattering_scale": 0.1}),
         ("ray0.015", {"rayleigh_scattering_scale": 0.015}), ("ray0.07", {"rayleigh_scattering_scale": 0.07}),
         ("wet40", {"wetness": 40}), ("dust20", {"dust_storm": 20}), ("wind50", {"wind_intensity": 50})]


def hhmm_s(t: str) -> float:
    h, m = t.split(":")
    return int(h) * 3600 + int(m) * 60


def views(world, centres):
    xs = [v[0] for v in centres.values()]
    ys = [v[1] for v in centres.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    r = min(max(0.3 * span, 80.0), 220.0)
    cmap = world.get_map()
    wp = cmap.get_waypoint(carla.Location(cx, cy, 0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
    route = [wp]
    for _ in range(600):
        nxt = wp.next(1.0)
        if not nxt:
            break
        wp = nxt[0]
        route.append(wp)
    out = {}
    for name, deg, z in (("high_a", 40.0, 60.0), ("high_b", 220.0, 60.0), ("mid", 130.0, 28.0)):
        a = math.radians(deg)
        rr = r if z > 40 else 0.55 * r
        px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
        out[name] = carla.Transform(carla.Location(px, py, z),
                                    carla.Rotation(math.degrees(math.atan2(-z, rr)), math.degrees(math.atan2(cy - py, cx - px)), 0.0))
    for name, i, z, pitch in (("street_a", 40, 3.0, -5.0), ("street_b", 330, 3.0, -5.0), ("eye", 150, 1.7, 0.0)):
        w = route[min(i, len(route) - 1)].transform
        out[name] = carla.Transform(carla.Location(w.location.x, w.location.y, w.location.z + z),
                                    carla.Rotation(pitch, w.rotation.yaw, 0.0))
    return out, route


def park_cars(world, route, n=26):
    lib = world.get_blueprint_library()
    bps = [b for b in lib.filter("vehicle.*") if b.has_attribute("number_of_wheels")
           and b.get_attribute("number_of_wheels").as_int() == 4]
    bps.sort(key=lambda b: b.id)
    actors = []
    k = 0
    for i in list(range(48, 140, 9)) + list(range(160, 320, 11)) + list(range(338, 420, 9)):
        if len(actors) >= n or i >= len(route):
            break
        for wp in [route[i]] + ([route[i].get_left_lane()] if k % 3 == 0 else []):
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                continue
            bp = bps[(k * 7) % len(bps)]
            k += 1
            if bp.has_attribute("color"):
                cols = bp.get_attribute("color").recommended_values
                bp.set_attribute("color", cols[k % len(cols)])
            tf = wp.transform
            tf.location.z += 0.3
            a = world.try_spawn_actor(bp, tf)
            if a is not None:
                actors.append(a)
    return actors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lab")
    ap.add_argument("--out", default="Claude outputs/vq/lab")
    ap.add_argument("--times", default="12:30,17:45,20:45,21:45,23:30")
    ap.add_argument("--looks", default="old,town")
    ap.add_argument("--views", default="high_a,high_b,mid,street_a,street_b,eye")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--measure-s", type=float, default=3.0)
    ap.add_argument("--no-fullscreen", action="store_true")
    a = ap.parse_args(argv)
    out = ROOT / a.out / a.tag
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("fps_test", ROOT / "tools" / "town05_fps_test.py")
    shooter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shooter)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(180.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != "Town05":
        world = client.load_world("Town05")
    s = world.get_settings()
    s.synchronous_mode, s.no_rendering_mode, s.fixed_delta_seconds = False, False, None
    world.apply_settings(s)
    if not a.no_fullscreen:
        from cosim.overlay import Overlay, dpi_aware

        dpi_aware()
        ov = Overlay(0)
        for _ in range(20):
            if ov.fullscreen_carla():
                break
            time.sleep(0.5)
    scen = scenario_mod.load("town05", ROOT)
    centres = scenario_mod.tls_centres(scen.net_file, scen.frame)
    vs, route = views(world, centres)
    cars = park_cars(world, route)
    for c in cars:
        c.set_simulate_physics(False)
    print("weather fields:", [n for n in dir(carla.WeatherParameters) if not n.startswith("_") and n[0].islower()], flush=True)
    print("default weather:", world.get_weather(), flush=True)
    print("lights by day:", look_mod.light_report(world), flush=True)
    spectator = world.get_spectator()
    lat, lon = INGOLSTADT
    frames: list[float] = []
    world.on_tick(lambda snap: frames.append(snap.timestamp.delta_seconds))
    try:
        font = ImageFont.truetype(str(ROOT / "assets" / "fonts" / "Barlow-SemiBold.ttf"), 22)
    except OSError:
        font = ImageFont.load_default()
    fps = {}
    variants = SWEEP if a.sweep else [(n, None) for n in re.split("[+,]", a.looks)]
    for t in re.split("[+,]", a.times):
        az, el = scenario_sun(hhmm_s(t), lat, lon)
        night = el < 3.0
        v = carla.VehicleLightState
        ls = v(int(v.Position) | int(v.LowBeam)) if night else v.NONE
        client.apply_batch([carla.command.SetVehicleLightState(c.id, ls) for c in cars])
        shots = {}
        for label, over in variants:
            if over is None:
                fields = look_mod.weather_fields(el, hhmm_s(t) / 3600.0, look=label)
            else:
                fields = dict(look_mod.weather_fields(el, hhmm_s(t) / 3600.0, look="old"), **over)
            w = carla.WeatherParameters(**fields)
            w.sun_altitude_angle, w.sun_azimuth_angle = el, carla_sun_azimuth(az)
            world.set_weather(w)
            if hasattr(look_mod, "apply_lights"):
                look_mod.apply_lights(world, el, look=label if over is None else "old")
            time.sleep(2.5)
            if night:
                print(t, label, "lights:", look_mod.light_report(world), flush=True)
            for vn in re.split("[+,]", a.views):
                spectator.set_transform(vs[vn])
                time.sleep(1.6)
                frames.clear()
                time.sleep(a.measure_s)
                d = sorted(frames)
                f = round(len(d) / sum(d), 1) if d else None
                fps[f"{t}/{vn}/{label}"] = f
                p = out / f"{t.replace(':', '')}_{vn}_{label}.jpg"
                tmp = out / "_grab.png"
                if shooter.grab_window(tmp):
                    Image.open(tmp).convert("RGB").save(p, quality=92)
                    shots[(vn, label)] = (p, f)
                print(t, vn, label, f, flush=True)
        # contact sheets: one per view, the variants side by side (or a grid for the sweep)
        for vn in re.split("[+,]", a.views):
            ims = [(lab, *shots[(vn, lab)]) for lab, _ in variants if (vn, lab) in shots]
            if not ims:
                continue
            cols = 2 if len(ims) <= 4 else 3
            tw = 960 if cols == 2 else 640
            tiles = []
            for lab, p, f in ims:
                im = Image.open(p)
                im = im.resize((tw, round(im.height * tw / im.width)), Image.LANCZOS)
                d = ImageDraw.Draw(im)
                txt = f"{t}  {vn}  {lab}  {f} fps  sun {el:.0f} deg"
                d.rectangle((0, 0, d.textlength(txt, font=font) + 16, 30), fill=(0, 0, 0))
                d.text((8, 2), txt, font=font, fill=(255, 255, 255))
                tiles.append(im)
            th = tiles[0].height
            rows = (len(tiles) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
            for i, im in enumerate(tiles):
                sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
            sheet.save(out / f"sheet_{t.replace(':', '')}_{vn}.jpg", quality=85)
    (out / "fps.json").write_text(json.dumps(fps, indent=1), encoding="utf-8")
    client.apply_batch([carla.command.DestroyActor(c.id) for c in cars])
    return 0


if __name__ == "__main__":
    sys.exit(main())
