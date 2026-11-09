"""The pillars of Parking Central's ground floor, from CARLA (9 Nov): CARLA's ray casting does not meet them (no
collision), so a depth camera under the first floor looks straight down at the ground floor, in a grid of pictures:
a point of a picture more than 0.4 m over the floor (and under the ceiling) is something standing there (a pillar,
a wall, a parked car). -> out/town05_places/silo_pillars.json: the cells (0.3 m) where something stands, SUMO x, y.
Read by tools/town05_entrances_build.py (those cells are not a way).

CARLA must be running (tools/town05_pillars_probe.bat starts it off screen).
"""
from __future__ import annotations

import json
import math
import queue
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import carla

from cosim import scenario as scenario_mod

OUT = ROOT / "out" / "town05_places"
CELL = 0.3
BOX = (157.0, 210.0, 211.0, 258.0)       # SUMO x0, x1, y0, y1: the ground floor and its north side
FLOOR_Z = 0.58                           # its floor (vertical rays, entrances_probe.json)
CAM_UP = 6.3                             # the cameras this far over it (the ceiling is at 8.75 m)
FOV, SIZE = 100.0, 800


def main() -> int:
    scen = scenario_mod.load("town05", ROOT)
    fr = scen.frame
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300.0)
    world = client.get_world()
    if not world.get_map().name.endswith("Town05"):
        world = client.load_world("Town05")
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    lib = world.get_blueprint_library()
    bp = lib.find("sensor.camera.depth")
    bp.set_attribute("image_size_x", str(SIZE))
    bp.set_attribute("image_size_y", str(SIZE))
    bp.set_attribute("fov", str(FOV))
    f = SIZE / (2.0 * math.tan(math.radians(FOV) / 2.0))
    half = CAM_UP * math.tan(math.radians(FOV) / 2.0) * 0.8
    x0, x1, y0, y1 = BOX
    nx, ny = int((x1 - x0) / CELL) + 1, int((y1 - y0) / CELL) + 1
    stand = np.zeros((ny, nx), dtype=bool)
    seen = np.zeros((ny, nx), dtype=bool)
    xs = np.arange(x0 + half, x1 + half, 2 * half * 0.9)
    ys = np.arange(y0 + half, y1 + half, 2 * half * 0.9)
    u, v = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    shots = 0
    for sx in xs:
        for sy in ys:
            cx, cy = fr.to_carla_xy(float(sx), float(sy))
            tf = carla.Transform(carla.Location(cx, cy, FLOOR_Z + CAM_UP), carla.Rotation(-90.0, 0.0, 0.0))
            cam = world.spawn_actor(bp, tf)
            q: queue.Queue = queue.Queue()
            cam.listen(lambda im, q=q: q.put((im.frame, np.array(np.frombuffer(im.raw_data, dtype=np.uint8)
                                                                  .reshape(im.height, im.width, 4)))))
            img = None
            try:
                frame = None
                for _ in range(8):
                    frame = world.tick()
                t = time.time()
                while time.time() - t < 20.0:
                    try:
                        fno, a = q.get(timeout=2.0)
                    except queue.Empty:
                        continue
                    if fno >= frame:
                        img = a
                        break
            finally:
                cam.stop()
                time.sleep(0.2)
                cam.destroy()
            if img is None:
                continue
            shots += 1
            bgr = img[:, :, :3].astype(np.float64)       # CARLA's depth: (R + G*256 + B*65536) / (2^24 - 1) * 1000 m
            depth = (bgr[:, :, 2] + bgr[:, :, 1] * 256.0 + bgr[:, :, 0] * 65536.0) / (256.0 ** 3 - 1) * 1000.0
            # camera space (UE: x forward = down here, y right, z up) -> world, by the camera's own matrix
            pc = np.stack([depth, (u - SIZE / 2) / f * depth, -(v - SIZE / 2) / f * depth, np.ones_like(depth)])
            m = np.array(tf.get_matrix())
            w = np.tensordot(m, pc, axes=1)                # 4 x H x W
            wx, wy, wz = w[0], w[1], w[2]
            height = wz - FLOOR_Z
            # SUMO x, y from CARLA x, y: the frame's own conversion, point by point (vectorised: it is affine)
            ox, oy = fr.to_sumo_xy(0.0, 0.0)
            ax_, ay_ = fr.to_sumo_xy(1.0, 0.0)
            bx_, by_ = fr.to_sumo_xy(0.0, 1.0)
            sumo_x = ox + (ax_ - ox) * wx + (bx_ - ox) * wy
            sumo_y = oy + (ay_ - oy) * wx + (by_ - oy) * wy
            i = np.round((sumo_x - x0) / CELL).astype(int)
            j = np.round((sumo_y - y0) / CELL).astype(int)
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (depth < CAM_UP + 3.0)
            seen[j[ok], i[ok]] = True
            up = ok & (height > 0.4) & (height < CAM_UP - 0.3)
            stand[j[up], i[up]] = True
    res = {"what": __doc__.split("\n\n")[0], "cell": CELL, "x0": x0, "y0": y0, "nx": nx, "ny": ny, "shots": shots,
           "floor_z": FLOOR_Z, "stand": ["".join("1" if c else "0" for c in row) for row in stand],
           "seen": ["".join("1" if c else "0" for c in row) for row in seen]}
    (OUT / "silo_pillars.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"{shots} pictures, {int(stand.sum())} cells with something standing of {int(seen.sum())} seen", flush=True)
    s.synchronous_mode = False
    world.apply_settings(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
