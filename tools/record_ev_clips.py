"""Short clips of emergency vehicles, lights on and off, through the inspector's cameras (siren step 8).

The detector's Town05 set is single pictures 20 s apart: nothing to learn a flashing light from. This records, for
each inspector camera (cosim/perception.plan_cameras, the H window's poses and lens) and a few times of day, one
ambulance and one police car at a time, each with its lights on and off, driving along a lane in front of the camera
(12-50 m away, 0-12 m/s). A clip is FRAMES pictures, one every 0.1 s (CARLA synchronous at 0.05 s), the full
960 x 540 picture as JPEG and the vehicle's visible box from the instance segmentation (None when under MIN_PIXELS).
No SUMO: the vehicle is moved along the lane's waypoints. The junctions are split as the detector's dataset
(make_dataset_town05.split_tls, seed 11), so the clips of its test junctions are never trained on.

    python tools/record_ev_clips.py --out out/ev_clips          (CARLA running, Town05)
    -> out/ev_clips/clips/<n>.npz (frames as JPEG bytes, boxes, meta) and out/ev_clips/index.jsonl
"""

from __future__ import annotations

import argparse
import io
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
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import carla  # noqa: E402
import sumolib  # noqa: E402

from cosim import scenario as scenario_mod  # noqa: E402
from cosim.inspector import FOV, RES  # noqa: E402
from cosim.labels import UNLIT_LUMA, instances, mean_luma  # noqa: E402
from cosim.perception import plan_cameras  # noqa: E402
from cosim.run import WEATHER, load_world  # noqa: E402
from cosim.solar import carla_sun_azimuth, scenario_sun  # noqa: E402
from cosim.vmodels import FIXED  # noqa: E402

DT = 0.05            # CARLA step
EVERY = 2            # a picture every 2 steps: 10 a second
FRAMES = 16          # 1.6 s
MIN_PIXELS = 150
KINDS = (("ambulance", FIXED["emergency"]), ("police", FIXED["authority"]))
CONDITIONS = [(8.0, "clear"), (15.0, "rain"), (23.5, "clear")]


def split_tls(tls_ids, seed):
    from make_dataset_town05 import split_tls as s

    return s(tls_ids, seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/ev_clips")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit-cams", type=int, default=0)
    ap.add_argument("--conditions", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="clips per camera, kind, lights and condition")
    a = ap.parse_args(argv)
    W, H = RES
    out = ROOT / a.out
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    scen = scenario_mod.load("town05", ROOT)
    net = sumolib.net.readNet(str(scen.net_file))
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300.0)
    world = load_world(client, scen)
    cmap = world.get_map()
    old = world.get_settings()
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, DT
    world.apply_settings(st)

    def ground_z(x, y):
        wp = cmap.get_waypoint(carla.Location(x, y, 0.0), project_to_road=True, lane_type=carla.LaneType.Any)
        return float(wp.transform.location.z) if wp is not None else 0.0

    cams = plan_cameras(net, scen.frame, ground_z, back_m=-10.0, pitch=-22.0)
    split = split_tls(sorted({c.tls for c in cams}), a.seed)
    if a.limit_cams:
        cams = cams[:a.limit_cams]
    lib = world.get_blueprint_library()
    sensors, qs = [], []
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

    def tick_get():
        fid = world.tick()
        got = []
        for q in qs:
            while True:
                im = q.get(timeout=60)
                if im.frame >= fid:
                    got.append(im)
                    break
        return got

    index = (out / "index.jsonl").open("a", encoding="utf-8")
    n = sum(1 for _ in (out / "clips").glob("*.npz"))
    t0 = time.time()
    conditions = CONDITIONS[:a.conditions] if a.conditions else CONDITIONS
    weather = carla.WeatherParameters()
    try:
        for hour, wname in conditions:
            for k, v in WEATHER[wname].items():
                setattr(weather, k, v)
            az, el = scenario_sun(hour * 3600, scen.lat, scen.lon)
            weather.sun_altitude_angle, weather.sun_azimuth_angle = el, carla_sun_azimuth(az)
            world.set_weather(weather)
            night = el < 3.0
            for ci, c in enumerate(cams):
                cam_tf = carla.Transform(carla.Location(c.x, c.y, c.z), carla.Rotation(c.pitch, c.yaw, 0.0))
                for s in sensors:
                    s.set_transform(cam_tf)
                for rep in range(a.repeats):
                    for kind, bp_id in KINDS:
                        for lights in (True, False):
                            d = rng.uniform(12.0, 50.0)
                            yaw = math.radians(c.yaw)
                            wp = cmap.get_waypoint(carla.Location(c.x + d * math.cos(yaw), c.y + d * math.sin(yaw), 0),
                                                   project_to_road=True, lane_type=carla.LaneType.Driving)
                            if wp is None:
                                continue
                            speed = rng.choice([0.0, rng.uniform(2.0, 12.0)])
                            forward = rng.random() < 0.5
                            bp = lib.find(bp_id)
                            tf = wp.transform
                            if not forward:
                                tf = carla.Transform(tf.location, carla.Rotation(0, tf.rotation.yaw + 180.0, 0))
                            tf.location.z += 0.4
                            v = world.try_spawn_actor(bp, tf)
                            if v is None:
                                continue
                            try:
                                v.set_simulate_physics(False)
                                ls = int(carla.VehicleLightState.NONE)
                                if night:
                                    ls |= int(carla.VehicleLightState.Position) | int(carla.VehicleLightState.LowBeam)
                                if lights:
                                    ls |= int(carla.VehicleLightState.Special1)
                                v.set_light_state(carla.VehicleLightState(ls))
                                for _ in range(rng.randint(3, 12)):  # settle; lights start at a random phase
                                    tick_get()
                                frames, boxes = [], []
                                cur = wp
                                for k in range(FRAMES * EVERY):
                                    if speed > 0:
                                        nxt = (cur.next if forward else cur.previous)(speed * DT)
                                        if nxt:
                                            cur = nxt[0]
                                            t2 = cur.transform
                                            yaw2 = t2.rotation.yaw + (0.0 if forward else 180.0)
                                            v.set_transform(carla.Transform(
                                                carla.Location(t2.location.x, t2.location.y, t2.location.z + 0.05),
                                                carla.Rotation(t2.rotation.pitch, yaw2, 0)))
                                    imgs = tick_get()
                                    if k % EVERY:
                                        continue
                                    rgb = np.frombuffer(imgs[0].raw_data, dtype=np.uint8).reshape(H, W, 4)
                                    inst = np.frombuffer(imgs[1].raw_data, dtype=np.uint8).reshape(H, W, 4)
                                    box = None
                                    for it in instances(inst):
                                        if it.id == v.id and it.pixels >= MIN_PIXELS:
                                            box = [int(x) for x in it.box] + [int(it.pixels)]
                                    buf = io.BytesIO()
                                    Image.fromarray(np.ascontiguousarray(rgb[:, :, 2::-1])).save(buf, "JPEG", quality=90)
                                    frames.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
                                    boxes.append(box)
                                    luma = mean_luma(rgb)
                                seen = sum(b is not None for b in boxes)
                                if seen < FRAMES // 2 or luma < UNLIT_LUMA:
                                    continue
                                meta = {"clip": n, "camera": ci, "junction": c.tls, "split": split[c.tls],
                                        "condition": f"{wname}-{hour:g}h", "night": night, "kind": kind,
                                        "lights": lights, "class": kind + ("_lights" if lights else ""),
                                        "distance_m": round(d, 1), "speed_ms": round(speed, 1),
                                        "towards_camera": not forward, "frames_seen": seen, "fps": 1.0 / (DT * EVERY)}
                                np.savez(out / "clips" / f"{n:05d}.npz",
                                         frames=np.array(frames, dtype=object), boxes=np.array(
                                             [b if b is not None else [-1, -1, -1, -1, 0] for b in boxes]),
                                         meta=json.dumps(meta))
                                index.write(json.dumps(meta) + "\n")
                                index.flush()
                                n += 1
                            finally:
                                v.destroy()
                print(f"{wname}-{hour:g}h camera {ci + 1}/{len(cams)}: {n} clips, {time.time() - t0:.0f} s", flush=True)
    finally:
        index.close()
        for s in sensors:
            s.stop()
            s.destroy()
        world.apply_settings(old)
    print(json.dumps({"clips": n, "seconds": round(time.time() - t0)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
