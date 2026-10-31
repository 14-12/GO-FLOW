"""Camera moves for the CARLA view and the recording.

One RGB camera sensor is moved every frame; the CARLA window's spectator follows the same
pose, so what is recorded is what is shown. Moves:

* orbit  - circles a point (a signalised junction) at a given radius, height and speed
* chase  - behind and above a vehicle, re-targeting when it leaves the network
* fixed  - a fixed pose (x, y, z, yaw, pitch in CARLA coordinates)

All positions are CARLA coordinates. Smoothing is exponential on the camera itself so a
chase target that turns does not make the picture jerk.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass

import carla


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float

    def transform(self) -> carla.Transform:
        return carla.Transform(carla.Location(self.x, self.y, self.z), carla.Rotation(self.pitch, self.yaw, 0.0))


def look_at(px: float, py: float, pz: float, tx: float, ty: float, tz: float) -> Pose:
    dx, dy, dz = tx - px, ty - py, tz - pz
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return Pose(px, py, pz, yaw, pitch)


def orbit_pose(cx: float, cy: float, t: float, radius: float = 45.0, height: float = 22.0,
               deg_per_s: float = 6.0, start_deg: float = 0.0) -> Pose:
    a = math.radians(start_deg + deg_per_s * t)
    return look_at(cx + radius * math.cos(a), cy + radius * math.sin(a), height, cx, cy, 1.5)


# What a camera may see through: the ground, markings, thin street furniture and vehicles.
_SEE_THROUGH = {"Roads", "RoadLines", "Sidewalks", "Ground", "Terrain", "Water", "Poles", "TrafficSigns",
                "TrafficLight", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Rider", "Pedestrians", "NONE",
                "Sky", "Dynamic"}


NEAR_M = 15.0  # the camera's surroundings checked towards the edges of the picture
EDGE_M = 12.0  # how far from the target those edge directions aim


def _towards(a: tuple, b: tuple, d: float) -> tuple:
    """The point `d` metres from a towards b (b itself if closer)."""
    v = [bb - aa for aa, bb in zip(a, b, strict=True)]
    n = math.sqrt(sum(x * x for x in v))
    return b if n <= d else tuple(aa + x * d / n for aa, x in zip(a, v, strict=True))


def clear_arc(cast_ray, cx: float, cy: float, inward: tuple[float, float] | None, radius: float = 45.0,
              height: float = 22.0, step_deg: float = 5.0, inward_min: float = -0.2,
              edge_m: float | None = None, view_ok=None) -> tuple[float, float] | None:
    """The longest arc of orbit angles from which the camera sees the target unobstructed and
    looks into the map rather than out of it.

    cast_ray(a, b) -> labels hit along the segment (CARLA's World.cast_ray). A position is
    good when nothing but `_SEE_THROUGH` lies between it and the target, nor within NEAR_M of
    the camera towards points EDGE_M around and above the target (a sign or a building corner
    beside the camera fills the picture even when the centre ray passes it), and when the view
    direction points towards the map centre (`inward`, a vector from the target to the centre;
    cos >= inward_min), so junctions at the edge of a town are not filmed against the void
    beyond it. view_ok(angle_deg) -> bool, if given, is asked last about each position that
    passed (run.py renders the view: `NearView`). Returns (start_deg, span_deg), or None if no
    angle is good.
    """
    n = int(round(360 / step_deg))
    good = []
    for k in range(n):
        a = math.radians(k * step_deg)
        px, py = cx + radius * math.cos(a), cy + radius * math.sin(a)
        cam = (px, py, height)
        ok = not any(label not in _SEE_THROUGH for label in cast_ray(cam, (cx, cy, 1.5)))
        e_m = EDGE_M if edge_m is None else edge_m
        d = e_m / math.sqrt(2)
        edges = [(cx + e_m, cy, 1.5), (cx - e_m, cy, 1.5), (cx, cy + e_m, 1.5), (cx, cy - e_m, 1.5),
                 (cx + d, cy + d, 1.5), (cx + d, cy - d, 1.5), (cx - d, cy + d, 1.5), (cx - d, cy - d, 1.5),
                 (cx, cy, e_m)]
        ok = ok and not any(label not in _SEE_THROUGH
                            for e in edges for label in cast_ray(cam, _towards(cam, e, NEAR_M)))
        if ok and inward is not None:
            vx, vy = cx - px, cy - py  # view direction
            norm = math.hypot(*inward) * math.hypot(vx, vy)
            ok = norm == 0 or (vx * inward[0] + vy * inward[1]) / norm >= inward_min
        if ok and view_ok is not None:
            ok = view_ok(k * step_deg)
        good.append(ok)
    if not any(good):
        return None
    if all(good):
        return 0.0, 360.0
    best, run, best_start, start = 0, 0, 0, 0
    for k in range(2 * n):  # twice round, for arcs crossing 0 deg
        if good[k % n]:
            if run == 0:
                start = k
            run += 1
            if run > best and run <= n:
                best, best_start = run, start
        else:
            run = 0
    return (best_start * step_deg) % 360.0, best * step_deg


def arc_pose(cx: float, cy: float, t: float, arc: tuple[float, float] | None, radius: float = 45.0,
             height: float = 22.0, deg_per_s: float = 6.0) -> Pose:
    """Orbit within the clear arc: round and round if it is the full circle, else swinging
    back and forth across it (sinusoidally, so it slows down at the ends)."""
    if arc is None:  # nothing clear: look down from high above
        return look_at(cx - 15.0, cy, 45.0, cx, cy, 0.0)
    start, span = arc
    if span >= 360.0:
        return orbit_pose(cx, cy, t, radius, height, deg_per_s, start)
    if span <= 10.0:
        a = start
    else:
        period = 2 * span / deg_per_s
        a = start + span / 2 + (span / 2 - 5.0) * math.sin(2 * math.pi * t / period)
    return orbit_pose(cx, cy, 0.0, radius, height, 0.0, a)


def chase_pose(tf: carla.Transform, back: float = 9.0, up: float = 3.2, ahead: float = 6.0) -> Pose:
    yaw = math.radians(tf.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    p = tf.location
    return look_at(p.x - fx * back, p.y - fy * back, p.z + up, p.x + fx * ahead, p.y + fy * ahead, p.z + 1.0)


def smooth(prev: Pose | None, target: Pose, k: float) -> Pose:
    """Exponential smoothing; angles take the short way round."""
    if prev is None:
        return target
    def ang(a, b):
        return a + ((b - a + 180) % 360 - 180) * k
    return Pose(prev.x + (target.x - prev.x) * k, prev.y + (target.y - prev.y) * k,
                prev.z + (target.z - prev.z) * k, ang(prev.yaw, target.yaw), ang(prev.pitch, target.pitch))


def _near_spectator(world: carla.World) -> carla.Transform:
    """Where to spawn a sensor: at the spectator (the camera's current place), since on a
    large map that is where the tiles are streamed in."""
    tf = world.get_spectator().get_transform()
    return carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z + 2.0), tf.rotation)


class NearView:
    """Renders a camera pose (small depth and semantic images) and measures how much of the
    picture is taken by something solid close to the lens: a palm frond or a roof edge the
    ray test misses (foliage is thin, rays pass between the leaves)."""

    # CARLA semantic tags that block the view when close: Building, Wall, Fence, Pole,
    # Vegetation, Static, Other, Bridge, GuardRail
    SOLID = (3, 4, 5, 6, 9, 20, 22, 26, 28)

    def __init__(self, world: carla.World, tick, fov: float, near_m: float = 15.0, size=(320, 180)):
        self.world, self.tick, self.near_m, self.size = world, tick, near_m, size
        lib = world.get_blueprint_library()
        self.sensors, self.queues = [], []
        for name in ("sensor.camera.depth", "sensor.camera.semantic_segmentation"):
            bp = lib.find(name)
            for k, v in (("image_size_x", str(size[0])), ("image_size_y", str(size[1])), ("fov", str(fov))):
                if bp.has_attribute(k):
                    bp.set_attribute(k, v)
            # At the spectator, not at the map origin: on a large map (Town12/13) a sensor
            # left kilometres from the streamed tiles goes dormant, and the 0.9.16 server
            # then crashed in streaming::token_type (see tools/probe_large_map.py).
            s = world.spawn_actor(bp, _near_spectator(world))
            q: queue.Queue = queue.Queue()
            s.listen(q.put)
            self.sensors.append(s)
            self.queues.append(q)

    def near_fraction(self, pose: Pose) -> float:
        return self.assess(pose)[0]

    def assess(self, pose: Pose, target_m: float | None = None) -> tuple[float, float]:
        """(share of the picture taken by something solid within near_m, share of the picture's
        centre (the middle fifth each way) taken by something solid nearer than target_m - 8 m,
        i.e. standing between the camera and what it films; 0 if target_m is None)."""
        import numpy as np

        for s in self.sensors:
            s.set_transform(pose.transform())
        fid = self.tick()
        imgs = []
        for q in self.queues:
            while True:
                im = q.get(timeout=60)
                if im.frame >= fid:
                    imgs.append(im)
                    break
        d, sem = (np.frombuffer(im.raw_data, dtype=np.uint8).reshape(im.height, im.width, 4) for im in imgs)
        metres = (d[..., 2].astype(np.float64) + d[..., 1] * 256.0 + d[..., 0] * 65536.0) / (256 ** 3 - 1) * 1000.0
        solid = np.isin(sem[..., 2], self.SOLID)
        near_solid = (metres < self.near_m) & solid
        blocked = 0.0
        if target_m is not None:
            h, w = metres.shape
            c = (slice(2 * h // 5, 3 * h // 5), slice(2 * w // 5, 3 * w // 5))
            blocked = float(((metres[c] < target_m - 8.0) & solid[c]).mean())
        return float(near_solid.mean()), blocked

    def close(self) -> None:
        for s in self.sensors:
            s.stop()
            s.destroy()


class Recorder:
    """An RGB camera whose frames, in synchronous mode, belong to the tick that rendered them."""

    def __init__(self, world: carla.World, width: int, height: int, fov: float = 75.0,
                 path: str | None = None, fps: float = 20.0):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        for name, value in (("motion_blur_intensity", "0.35"), ("enable_postprocess_effects", "True")):
            if bp.has_attribute(name):
                bp.set_attribute(name, value)
        self.world = world
        self.sensor = world.spawn_actor(bp, _near_spectator(world))  # see NearView
        self.frames: queue.Queue = queue.Queue()
        self.sensor.listen(self.frames.put)
        self.size = (width, height)
        self.writer = None
        self.written = 0
        if path:
            import imageio_ffmpeg  # bundled ffmpeg binary, no system install needed

            self.writer = imageio_ffmpeg.write_frames(path, self.size, fps=fps, codec="libx264",
                                                      quality=8, macro_block_size=8, pix_fmt_in="rgb24")
            self.writer.send(None)

    def place(self, pose: Pose) -> None:
        tf = pose.transform()
        self.sensor.set_transform(tf)
        self.world.get_spectator().set_transform(tf)

    def collect(self, frame_id: int, timeout: float = 30.0, still: str | None = None) -> None:
        """After world.tick(): take this tick's image and append it to the video
        (and, if `still` is a path, also save it there as an image)."""
        while True:
            img = self.frames.get(timeout=timeout)
            if img.frame >= frame_id:
                break
        if still:
            img.save_to_disk(still)
        if self.writer is not None:
            import numpy as np

            bgra = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            self.writer.send(np.ascontiguousarray(bgra[:, :, 2::-1]).tobytes())
            self.written += 1

    def close(self) -> None:
        try:
            self.sensor.stop()
            self.sensor.destroy()
        finally:
            if self.writer is not None:
                self.writer.close()
