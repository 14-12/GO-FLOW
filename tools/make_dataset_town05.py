"""Town05 training set for the free-fly traffic cameras, rendered by CARLA and labelled from it.

    python tools/make_dataset_town05.py [--out out/dataset_town05] [--frames 3]

Why: the detector (models/carla-yolo26n.pt) was trained on Town10, Ingolstadt7 and Cologne3 from
cameras 7 m up and 22-30 m out. The free-fly inspector (cosim/inspector.py) looks through other
cameras (on a mast 10 m into the junction, 9 m up, pitch -22, 960x540) at another town, and there it
drew emergency boxes on facades. This set is rendered from those very cameras, plus one moved copy of
each per capture (a few metres, a few degrees), at Town05, at the inspector's resolution.

Labels come from the instance-segmentation camera at the same pose (cosim/labels.py): a box is the
extent of the object's *visible* pixels, so a vehicle hidden behind a building or a bus is not
labelled, and one half hidden gets a box of the half that shows. Images with nothing in them are kept
(empty label file): they teach the facades, poles and signs as background.

* Traffic: Town05's own demand (scenarios/town05), actuated signals, the main project's extras:
  pedestrians (40 per km of pavement, twice the default, for examples), and an emergency vehicle about every --emergency-every s, 60 % of them on a run
  (beacons on: ambulance_lights / police_lights), the rest driving normally (ambulance / police).
* Conditions: sunrise to night, clear, cloudy and rain (Town05 has street lights; frames too dark
  to see anything are dropped).
* Split by junction: 3 of the 15 signals are test only, 2 are val, never seen in training.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import carla  # noqa: E402
import sumolib  # noqa: E402

from cosim import scenario as scenario_mod  # noqa: E402
from cosim.carla_side import CarlaSide  # noqa: E402
from cosim.convert import light_state  # noqa: E402
from cosim.inspector import FOV, RES  # noqa: E402
from cosim.labels import CLASSES, UNLIT_LUMA, VCLASS, instances, label_frame, mean_luma  # noqa: E402
from cosim.perception import plan_cameras  # noqa: E402
from cosim.run import WEATHER, Poser, load_world  # noqa: E402
from cosim.solar import carla_sun_azimuth, scenario_sun  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402

# (hour, weather): sunrise, morning, midday, afternoon, rain, evening, sunset, dusk, night
CONDITIONS = [(5.6, "clear"), (8.0, "clear"), (11.0, "cloudy"), (13.5, "clear"), (15.0, "rain"),
              (18.0, "clear"), (20.8, "clear"), (21.6, "cloudy"), (23.5, "clear")]
WARMUP_S = 600.0
WALKER_RADIUS_M = 90.0  # pedestrians further than this from every camera are not drawn (too small anyway)
FREE_LABELS = {"Roads", "RoadLines", "Sidewalks", "Ground", "Terrain", "Sky", "NONE", "Other"}


def split_tls(tls_ids: list[str], seed: int) -> dict[str, str]:
    ids = sorted(tls_ids)
    random.Random(seed).shuffle(ids)
    out = {t: "train" for t in ids}
    for t in ids[:3]:
        out[t] = "test"
    for t in ids[3:5]:
        out[t] = "val"
    return out


def base_tf(c) -> carla.Transform:
    return carla.Transform(carla.Location(c.x, c.y, c.z), carla.Rotation(c.pitch, c.yaw, 0.0))


def moved_tf(world, c, rng: random.Random) -> carla.Transform | None:
    """A copy of camera c moved a little: along and across its view, up or down, turned and tilted.
    None when the way from the real camera to it is not clear (inside a building, a tree)."""
    for _ in range(6):
        along, side, up = rng.uniform(-6.0, 8.0), rng.uniform(-3.0, 3.0), rng.uniform(-2.0, 2.0)
        yaw = math.radians(c.yaw)
        x = c.x + along * math.cos(yaw) - side * math.sin(yaw)
        y = c.y + along * math.sin(yaw) + side * math.cos(yaw)
        z = c.z + up
        hits = world.cast_ray(carla.Location(c.x, c.y, c.z), carla.Location(x, y, z))
        if any(getattr(h.label, "name", str(h.label)) not in FREE_LABELS for h in hits):
            continue
        return carla.Transform(carla.Location(x, y, z),
                               carla.Rotation(rng.uniform(-30.0, -15.0), c.yaw + rng.uniform(-12.0, 12.0), 0.0))
    return None


def render(side, sensors, qs, tf, ticks=2):
    for s in sensors:
        s.set_transform(tf)
    for _ in range(ticks - 1):
        side.tick()  # settle (temporal anti-aliasing after the jump)
    fid = side.tick()
    imgs = []
    for q in qs:
        while True:
            im = q.get(timeout=60)
            if im.frame >= fid:
                imgs.append(im)
                break
    return imgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/dataset_town05")
    ap.add_argument("--frames", type=int, default=4, help="captures per condition")
    ap.add_argument("--interval", type=float, default=20.0, help="simulated seconds between captures")
    ap.add_argument("--emergency-every", type=float, default=10.0)
    ap.add_argument("--pedestrians-per-km", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit-cams", type=int, default=0, help="only the first N cameras (quick check)")
    ap.add_argument("--conditions", type=int, default=0, help="only the first N conditions (quick check)")
    a = ap.parse_args(argv)
    W, H = RES
    out = ROOT / a.out
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    scen = scenario_mod.load("town05", ROOT)
    net = sumolib.net.readNet(str(scen.net_file))
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300.0)
    world = load_world(client, scen)
    cmap = world.get_map()
    # Town05's parked lorries and vans are static meshes tagged Truck: no actor, so no class (van or
    # truck?) can be given to them, and an unlabelled vehicle in a training picture teaches "background".
    # They are hidden while rendering this set (also parked motorbikes and bicycles, which have no class).
    hidden = [o.id for lab in (carla.CityObjectLabel.Truck, carla.CityObjectLabel.Motorcycle,
                               carla.CityObjectLabel.Bicycle, carla.CityObjectLabel.Rider)
              for o in world.get_environment_objects(lab)]
    if hidden:
        world.enable_environment_objects(set(hidden), False)
    print(f"{len(hidden)} static vehicles without a class hidden", flush=True)

    def ground_z(x, y):  # as the inspector places its cameras
        wp = cmap.get_waypoint(carla.Location(x, y, 0.0), project_to_road=True, lane_type=carla.LaneType.Any)
        return float(wp.transform.location.z) if wp is not None else 0.0

    cams = plan_cameras(net, scen.frame, ground_z, back_m=-10.0, pitch=-22.0)
    if a.limit_cams:
        cams = cams[:a.limit_cams]
    split = split_tls(sorted({c.tls for c in cams}), a.seed)
    print(f"{len(cams)} cameras; split {json.dumps(split)}", flush=True)
    links = scen.links([lm.id for lm in cmap.get_all_landmarks_of_type("1000001")])
    sumo = SumoSide(scen.folder, "actuated", seed=a.seed, label="ds-town05", port=8813, extras=True,
                    emergency_every=a.emergency_every,
                    extra_args=["--end", "100000", "--lanechange.duration", "2"])
    sumo.extras.config.pedestrians_per_km = a.pedestrians_per_km
    meta = (out / "meta.jsonl").open("w", encoding="utf-8")
    totals = {sp: {"images": 0, "empty": 0, "objects": {}} for sp in ("train", "val", "test")}
    dropped: dict[str, int] = {}
    n_img = 0
    t_start = time.time()
    sensors = []
    side = None
    try:
        t_end = sumo.time + WARMUP_S
        while sumo.time < t_end:
            cur = sumo.step()
        side = CarlaSide(client, world, scen.frame, fps=20.0, links=links)
        side.use_road_levels()
        poser = Poser(scen.frame, side.ground, heading="turn")
        lib = world.get_blueprint_library()
        qs = []
        for name in ("sensor.camera.rgb", "sensor.camera.instance_segmentation"):
            bp = lib.find(name)
            for k, v in (("image_size_x", str(W)), ("image_size_y", str(H)), ("fov", str(FOV)),
                         ("motion_blur_intensity", "0.0")):
                if bp.has_attribute(k):
                    bp.set_attribute(k, v)
            s = world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, 100)))
            q: queue.Queue = queue.Queue()
            s.listen(q.put)
            sensors.append(s)
            qs.append(q)
        # the empty town through every camera: a static vehicle that cannot be labelled would drop its frames
        bad = []
        for i, c in enumerate(cams):
            inst = np.frombuffer(render(side, sensors, qs, base_tf(c))[1].raw_data, dtype=np.uint8).reshape(H, W, 4)
            if label_frame(instances(inst), {}, W, H)[0] is None:
                bad.append(i)
        print(f"cameras seeing an unlabellable static vehicle: {bad}", flush=True)
        cam_xy = np.array([[c.x, c.y] for c in cams])
        weather = carla.WeatherParameters()
        conditions = CONDITIONS[:a.conditions] if a.conditions else CONDITIONS
        for hour, wname in conditions:
            for k, v in WEATHER[wname].items():
                setattr(weather, k, v)
            az, el = scenario_sun(hour * 3600, scen.lat, scen.lon)
            weather.sun_altitude_angle, weather.sun_azimuth_angle = el, carla_sun_azimuth(az)
            world.set_weather(weather)
            night = el < 3.0
            for _ in range(a.frames):
                t_next = sumo.time + a.interval
                while sumo.time < t_next:
                    cur = sumo.step()
                drawable = {vid: v for vid, v in cur.vehicles.items() if VCLASS.get(v.vclass, "car") is not None}
                side.destroy([vid for vid in side.actors if vid not in drawable])
                poser.begin_step(cur, cur)
                poses = {vid: tf for vid, tf in poser.poses(1.0, sumo.step_length).items() if vid in drawable}
                side.spawn([(vid, drawable[vid].vclass, poses[vid]) for vid in drawable
                            if vid not in side.actors and vid not in side.failed])
                side.set_traffic_lights(cur.tls)
                side.move(poses, {vid: light_state(v.signals, night, emergency_run=sumo.on_run(vid))
                                  for vid, v in drawable.items()})
                near = {}
                for pid, pp in cur.persons.items():
                    cx, cy = scen.frame.to_carla_xy(pp.x, pp.y)
                    if np.min(np.hypot(cam_xy[:, 0] - cx, cam_xy[:, 1] - cy)) < WALKER_RADIUS_M:
                        near[pid] = pp
                side.destroy_walkers([p for p in side.walkers if p not in near])
                side.spawn_walkers([(pid, side.walker_pose(pp.x, pp.y, pp.angle)) for pid, pp in near.items()
                                    if pid not in side.walkers and pid not in side.failed])
                side.move_walkers({pid: (side.walker_pose(pp.x, pp.y, pp.angle), pp.speed)
                                   for pid, pp in near.items()})
                actor_class = {}
                for vid, v in drawable.items():
                    if vid in side.actors:
                        cls = VCLASS.get(v.vclass, "car")
                        if cls in ("ambulance", "police") and sumo.on_run(vid):
                            cls += "_lights"
                        actor_class[side.actors[vid]] = cls
                actor_class.update({wid: "pedestrian" for wid in side.walkers.values()})
                for ci, c in enumerate(cams):
                    for variant, tf in (("cam", base_tf(c)), ("moved", moved_tf(world, c, rng))):
                        if tf is None:
                            continue
                        imgs = render(side, sensors, qs, tf)
                        rgb = np.frombuffer(imgs[0].raw_data, dtype=np.uint8).reshape(H, W, 4)
                        if mean_luma(rgb) < UNLIT_LUMA:
                            dropped["unlit"] = dropped.get("unlit", 0) + 1
                            continue
                        inst = np.frombuffer(imgs[1].raw_data, dtype=np.uint8).reshape(H, W, 4)
                        lines, info = label_frame(instances(inst), actor_class, W, H)
                        if lines is None:
                            key = info["rejected"][0].split(" id ")[0]
                            dropped[key] = dropped.get(key, 0) + 1
                            continue
                        sp = split[c.tls]
                        stem = f"town05_{c.tls}_c{ci:02d}{variant[0]}_{wname}-{hour:g}h_{int(cur.time)}_{n_img:05d}"
                        Image.fromarray(np.ascontiguousarray(rgb[:, :, 2::-1])).save(
                            out / "images" / sp / f"{stem}.jpg", quality=95)
                        (out / "labels" / sp / f"{stem}.txt").write_text(
                            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                        meta.write(json.dumps({"image": f"images/{sp}/{stem}.jpg", "split": sp, "junction": c.tls,
                                               "camera": ci, "variant": variant, "condition": f"{wname}-{hour:g}h",
                                               "sim_time": cur.time, "labels": info}) + "\n")
                        totals[sp]["images"] += 1
                        totals[sp]["empty"] += 0 if lines else 1
                        for k2, n in info.items():
                            totals[sp]["objects"][k2] = totals[sp]["objects"].get(k2, 0) + n
                        n_img += 1
                print(f"  {wname}-{hour:g}h t={cur.time:.0f}: {n_img} images, {len(drawable)} vehicles, "
                      f"{len(side.walkers)} walkers, {time.time() - t_start:.0f} s, dropped {dropped}", flush=True)
    finally:
        for s in sensors:
            s.stop()
            s.destroy()
        if side is not None:
            side.close()
        sumo.close()
        meta.close()
    (out / "dataset.yaml").write_text(
        "path: .  # run from this folder, or set an absolute path\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8")
    manifest = {"classes": CLASSES, "totals": totals, "dropped_frames": dropped, "conditions": CONDITIONS,
                "split": split, "cameras": len(cams), "resolution": [W, H], "fov": FOV,
                "image_format": "JPEG RGB quality 95", "frames_per_condition": a.frames,
                "emergency_every": a.emergency_every, "seconds": round(time.time() - t_start)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
