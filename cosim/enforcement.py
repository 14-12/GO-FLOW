"""A red-light enforcement camera at one approach of a signalised junction (31 Oct).

As a real red-light camera site: a camera of its own on a gantry over the approach lanes, some metres before the
stop line, looking down the road at the stop line and into the junction, sharp enough to read a number plate; and
the signal's state from the junction's controller (here SUMO's signal program, the state each picture shows), with
both on one clock. The agent that decides (cosim/redlight.py) gets only these two: the pictures and the controller's
signal log. Where the vehicles really are, and who ran the red, it never sees; that is the simulation's own record
(cosim/red_runners.RedLightTruth), written next to the recording to grade it.

This module: the site (`plan_site`: lanes, stop line, the cameras' poses), the controller's log (`SignalLog`), and the
cameras (`SiteCamera`: a CARLA RGB sensor, every picture timed on the simulation's clock and kept as video,
optionally handed to the agent).

Two cameras, as real red-light installations have: the overview camera (`cam`: the approach, the stop line and the
junction, for the decision and the photos) and, on the same pole, a narrow plate camera (`plate_cam`: the last metres
before the line, both lanes, about twice the overview's pixels on a plate: characters about 18 pixels high instead of
10, what number-plate readers need). The agent finds a vehicle's plate in it from where the overview camera has the
vehicle on the road (both cameras' poses known: calibrated).

The time of a picture. SUMO runs a step or two ahead of what is shown (cosim/freefly.SumoThread); the viewer shows
step k's state (its vehicles moving from where step k-1 had them to where step k has them, its signals) from the
moment the pose process starts it (FrameFeeder: `start`) for one step. So a picture taken at wall time w shows SUMO
time t_k - step + (w - start): FreeFly.depicted_at(w). The signals are set with that step, so a picture's signal is
the state SUMO had after step k: SignalLog.state_at(t) is the state of the first step at or after t.
"""

from __future__ import annotations

import bisect
import csv
import json
import logging
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("enforcement")

CAM_BACK_M = 24.0       # the pole this far before the stop line, 7 m up, looking 17 degrees down: the road from 9 m
CAM_HEIGHT_M = 7.0      # before the line (a car's back there: its front 4.5 m before the line, two or three pictures
CAM_PITCH_DEG = -17.0   # before it reaches it even at 50 km/h) to about 28 m past it, into the junction; a rear
CAM_FOV_DEG = 30.0      # plate about 110 pixels wide nearest the camera, 75 at the line (0.52 m wide)
CAM_SIZE = (1920, 1080)
CAM_FPS = 10.0
PLATE_FOV_DEG = 17.0    # the plate camera: 7.8 m across (both lanes) at 26 m
PLATE_AIM_S = -3.0      # aimed at a plate (0.5 m up) this far before the line: it sees the backs from about 8 m
PLATE_Z_M = 0.5         # before the line to 2 m past it


@dataclass
class Site:
    """One approach of a signalised junction, watched by one camera (CARLA coordinates unless said)."""
    name: str                     # e.g. "1292-6": the junction (SUMO's signal) and the road
    tls: str
    edge: str
    lanes: list                   # SUMO lane ids for cars, right to left
    links: dict                   # lane -> [the signal's link indices from it]
    line: list                    # the stop line: [[x, y], [x, y]], CARLA, across all its lanes (right, left)
    lane_lines: dict              # lane -> [[x, y], [x, y]]: its part of the stop line
    heading: float                # CARLA yaw of the road, towards the junction (deg)
    lane_width: float
    cam: dict = field(default_factory=dict)   # x, y, z, yaw, pitch, fov, width, height (CARLA)
    plate_cam: dict = field(default_factory=dict)   # the plate camera (same keys); empty: none

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Site":
        return Site(**d)


def plan_site(net, frame, tls: str, edge: str, back_m: float = CAM_BACK_M, height_m: float = CAM_HEIGHT_M,
              pitch: float = CAM_PITCH_DEG, fov: float = CAM_FOV_DEG, size=CAM_SIZE,
              plate_fov: float | None = PLATE_FOV_DEG) -> Site:
    """The site at road `edge` into signal `tls`. net: sumolib net; frame: cosim.mapping.Frame. The camera's height
    is above SUMO's (flat) road: the viewer adds the CARLA road's own height (FreeFly: ground)."""
    e = net.getEdge(edge)
    lanes = [ln for ln in e.getLanes() if ln.allows("passenger")]
    if not lanes:
        raise ValueError(f"road {edge} has no lane for cars")
    links = {}
    for ln in lanes:
        idx = sorted({c.getTLLinkIndex() for c in ln.getOutgoing() if c.getTLSID() == tls and c.getTLLinkIndex() >= 0})
        if idx:
            links[ln.getID()] = idx
    if not links:
        raise ValueError(f"road {edge} is not controlled by signal {tls}")
    # the road's direction at the stop line (SUMO), from the lanes' last stretch
    (x0, y0), (x1, y1) = lanes[0].getShape()[-2], lanes[0].getShape()[-1]
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / d, (y1 - y0) / d          # along the road (SUMO)
    rx, ry = uy, -ux                                # to the right of the road (SUMO: y north)
    width = lanes[0].getWidth()
    lane_lines, ends = {}, []
    for ln in lanes:
        ex, ey = ln.getShape()[-1]
        w = ln.getWidth() / 2.0
        a = frame.to_carla_xy(ex + rx * w, ey + ry * w)
        b = frame.to_carla_xy(ex - rx * w, ey - ry * w)
        lane_lines[ln.getID()] = [list(a), list(b)]
        ends.append((ex, ey))
    right = lane_lines[lanes[0].getID()][0]
    left = lane_lines[lanes[-1].getID()][1]
    mx = sum(p[0] for p in ends) / len(ends)
    my = sum(p[1] for p in ends) / len(ends)
    cx, cy = frame.to_carla_xy(mx - ux * back_m, my - uy * back_m)
    yaw = math.degrees(math.atan2(-uy, ux))         # CARLA: y south
    plate_cam = {}
    if plate_fov:
        p_pitch = -math.degrees(math.atan2(height_m - PLATE_Z_M, back_m + PLATE_AIM_S))
        plate_cam = {"x": cx, "y": cy, "z": height_m, "height_m": height_m, "yaw": yaw, "pitch": p_pitch,
                     "fov": plate_fov, "width": size[0], "height": size[1]}
    return Site(name=f"{tls}-{edge}", tls=tls, edge=edge, lanes=[ln.getID() for ln in lanes], links=links,
                line=[list(right), list(left)], lane_lines=lane_lines, heading=yaw, lane_width=width,
                cam={"x": cx, "y": cy, "z": height_m, "height_m": height_m, "yaw": yaw, "pitch": pitch, "fov": fov,
                     "width": size[0], "height": size[1]}, plate_cam=plate_cam)


# ------------------------------------------------------------------ the camera's geometry
def basis(yaw: float, pitch: float):
    """Unit vectors (forward, right, up) of a CARLA camera (left-handed: y south)."""
    cy_, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    f = np.array([cp * cy_, cp * sy, sp])
    r = np.array([-sy, cy_, 0.0])
    u = np.array([-sp * cy_, -sp * sy, cp])  # as cosim/perception.basis
    return f, r, u


def focal(cam: dict) -> float:
    return cam["width"] / (2.0 * math.tan(math.radians(cam["fov"]) / 2.0))


def project(points, cam: dict):
    """CARLA points (N, 3) -> pixels (N, 2), depth (N,)."""
    f, r, u = basis(cam["yaw"], cam["pitch"])
    d = np.asarray(points, dtype=float) - np.array([cam["x"], cam["y"], cam["z"]])
    depth, right, up = d @ f, d @ r, d @ u
    fl = focal(cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.stack([cam["width"] / 2 + fl * right / depth, cam["height"] / 2 - fl * up / depth], axis=1)
    return px, depth


def to_ground(px, cam: dict, ground_z: float):
    """Pixels (N, 2) -> CARLA points on the plane z = ground_z (N, 3); NaN above the horizon."""
    f, r, u = basis(cam["yaw"], cam["pitch"])
    fl = focal(cam)
    px = np.asarray(px, dtype=float).reshape(-1, 2)
    rays = (f[None, :] + ((px[:, 0:1] - cam["width"] / 2) / fl) * r[None, :]
            - ((px[:, 1:2] - cam["height"] / 2) / fl) * u[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (ground_z - cam["z"]) / rays[:, 2]
    s[s <= 0] = np.nan
    return np.array([cam["x"], cam["y"], cam["z"]])[None, :] + s[:, None] * rays


# ------------------------------------------------------------------ the controller's log
@dataclass
class SignalLog:
    """The junction controller's signal states, one per SUMO step, on the simulation's clock (module doc)."""
    times: list = field(default_factory=list)
    states: list = field(default_factory=list)

    def add(self, t: float, state: str) -> None:
        if self.times and t <= self.times[-1]:
            return
        self.times.append(t)
        self.states.append(state)

    def state_at(self, t: float) -> str | None:
        i = bisect.bisect_left(self.times, t - 1e-6)
        return self.states[i] if i < len(self.states) else None

    def red_since(self, t: float, links) -> float | None:
        """How long all of `links` had been red at time t (s), or None if one of them was not red then. A step's
        state holds from the step before (module doc)."""
        i = bisect.bisect_left(self.times, t - 1e-6)
        if i >= len(self.states):
            return None
        if not all(k < len(self.states[i]) and self.states[i][k] in "rR" for k in links):
            return None
        j = i
        while j > 0 and all(k < len(self.states[j - 1]) and self.states[j - 1][k] in "rR" for k in links):
            j -= 1
        start = self.times[j - 1] if j > 0 else self.times[0] - 0.5
        return t - start

    @staticmethod
    def from_changes(changes, step: float = 0.5, until: float | None = None) -> "SignalLog":
        """A log rebuilt from its changes [(t, state)] (signals.jsonl): every step from the first change on."""
        log_ = SignalLog()
        if not changes:
            return log_
        changes = sorted(changes)
        end = until if until is not None else changes[-1][0] + 120.0
        i, t = 0, changes[0][0]
        while t <= end + 1e-6:
            while i + 1 < len(changes) and changes[i + 1][0] <= t + 1e-6:
                i += 1
            log_.add(round(t, 3), changes[i][1])
            t += step
        return log_

    def changes(self, links) -> list:
        """[(t, state of `links`)] each time it changed: what a controller's log would hold."""
        out, last = [], None
        for t, st in zip(self.times, self.states, strict=False):
            s = "".join(st[k] if k < len(st) else "?" for k in links)
            if s != last:
                out.append((t, s))
                last = s
        return out


# ------------------------------------------------------------------ the camera
class SiteCamera:
    """The site's camera in CARLA: every picture timed on the simulation's clock (depicted(wall) -> SUMO seconds),
    written to video (record) and/or handed to `consumer(frame, sim_t, rgb array)` (the agent)."""

    def __init__(self, world, site: Site, out_dir: Path | None, depicted, fps: float = CAM_FPS,
                 consumer=None, crf: int = 18, which: str = "cam"):
        self.world, self.site, self.depicted = world, site, depicted
        self.which = which                       # "cam" (overview: camera.mp4, frames.csv) or "plate_cam"
        self.stem = "camera" if which == "cam" else "plate"   # (plate.mp4, plate_frames.csv)
        self.out = Path(out_dir) if out_dir is not None else None
        self.fps, self.crf, self.consumer = fps, crf, consumer
        self.q: queue.Queue = queue.Queue(maxsize=120)
        self.frames = self.dropped = 0
        self.sensor = None
        self.thread = None
        self.stopped = False
        self.last_sim = None
        self.offset = None   # wall time - CARLA's elapsed time of a picture (its lowest seen: the delivery's delay
        self._tick_id = None  # left out), from the world's ticks

    def start(self) -> None:
        import carla

        cam = getattr(self.site, self.which)
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", str(cam["width"])), ("image_size_y", str(cam["height"])),
                     ("fov", str(cam["fov"])), ("sensor_tick", f"{1.0 / self.fps:.4f}"),
                     ("motion_blur_intensity", "0.0"), ("shutter_speed", "500.0")):
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        tf = carla.Transform(carla.Location(cam["x"], cam["y"], cam["z"]), carla.Rotation(cam["pitch"], cam["yaw"], 0))
        self.sensor = self.world.spawn_actor(bp, tf)

        def on_tick(snap):  # a picture's time on the wall clock: its CARLA time + the offset (the lowest seen)
            off = time.time() - snap.timestamp.elapsed_seconds
            self.offset = off if self.offset is None else min(self.offset + 1e-4, off)

        self._tick_id = self.world.on_tick(on_tick)
        writer = index = cw = None
        if self.out is not None:
            import imageio_ffmpeg

            self.out.mkdir(parents=True, exist_ok=True)
            writer = imageio_ffmpeg.write_frames(str(self.out / f"{self.stem}.mp4"), (cam["width"], cam["height"]),
                                                 fps=self.fps, codec="libx264", pix_fmt_in="rgb24",
                                                 macro_block_size=2,
                                                 output_params=["-crf", str(self.crf), "-preset", "veryfast",
                                                                "-pix_fmt", "yuv420p"])
            writer.send(None)
            index = (self.out / ("frames.csv" if self.which == "cam" else "plate_frames.csv")).open(
                "w", newline="", encoding="utf-8")
            cw = csv.writer(index)
            cw.writerow(["frame", "sim_s", "wall_s", "carla_s", "frame_wall_s"])
        w_, h_ = cam["width"], cam["height"]

        def on_image(img):
            if self.stopped:
                return
            wall = time.time()
            try:
                self.q.put_nowait((bytes(img.raw_data), img.timestamp, wall))
            except queue.Full:
                self.dropped += 1

        def work():
            try:
                while True:
                    item = self.q.get()
                    if item is None:
                        break
                    raw, ts, wall = item
                    frame_wall = ts + self.offset if self.offset is not None else wall
                    sim = self.depicted(frame_wall)
                    rgb = np.ascontiguousarray(np.frombuffer(raw, dtype=np.uint8).reshape(h_, w_, 4)[:, :, 2::-1])
                    if writer is not None:
                        writer.send(rgb.tobytes())
                        cw.writerow([self.frames, f"{sim:.3f}" if sim is not None else "", f"{wall:.3f}",
                                     f"{ts:.3f}", f"{frame_wall:.3f}"])
                    if self.consumer is not None and sim is not None:
                        try:
                            self.consumer(self.frames, sim, rgb)
                        except Exception:  # noqa: BLE001 - the agent's trouble never stops the recording
                            if not getattr(self, "_warned", False):
                                log.exception("site camera: handing a picture to the agent failed")
                                self._warned = True
                    self.frames += 1
                    self.last_sim = sim
            except Exception:  # noqa: BLE001 - the viewer goes on
                log.exception("enforcement camera %s failed", self.site.name)
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as e:  # noqa: BLE001
                        log.warning("closing the site's video: %s", e)
                    index.close()

        self.thread = threading.Thread(target=work, name=f"site {self.site.name}", daemon=True)
        self.thread.start()
        self.sensor.listen(on_image)
        log.info("enforcement %s at %s: %s", "camera" if self.which == "cam" else "plate camera", self.site.name, cam)

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            if self._tick_id is not None:
                self.world.remove_on_tick(self._tick_id)
            self.sensor.stop()
            self.sensor.destroy()
        except Exception as e:  # noqa: BLE001
            log.debug("site camera: %s", e)
        self.q.put(None)
        if self.thread is not None:
            self.thread.join(timeout=60)


def write_jsonl(path: Path, rows) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------ one site in the viewer
class SiteRun:
    """A site in the running viewer: the controller's log and the simulation's own record (SUMO thread:
    sumo_step), the camera (main thread: start, tick), and what is written to out_dir:
    site.json, camera.mp4 + frames.csv and plate.mp4 + plate_frames.csv (record), signals.jsonl (the controller's log: each change of the
    approach's signal), truth.jsonl (every vehicle entering the junction from the site's lanes, with its plate, model
    and colour: for grading only)."""

    def __init__(self, site: Site, conn, out_dir: Path, record: bool = True, on_run=None):
        from cosim.red_runners import RedLightTruth

        self.site = site
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.record = record
        self.signals = SignalLog()
        self.truth = RedLightTruth(conn, {site.tls: set(site.lanes)}, on_run=on_run)
        self.pending: list = []          # truth entries not written yet (SUMO thread -> main thread)
        self.camera: SiteCamera | None = None
        self.plate_camera: SiteCamera | None = None
        self._sig_written = 0
        self._lock = threading.Lock()
        self.links = sorted({k for ks in site.links.values() for k in ks})
        self.agent_q = None          # the agent's control queue (cosim/redlight_proc.py), when it runs

    def sumo_step(self, cur) -> None:
        st = cur.tls.get(self.site.tls)
        if st is not None:
            with self._lock:
                self.signals.add(cur.time, st)
            if self.agent_q is not None:
                self.agent_q.put(("signal", cur.time, st))
        new = self.truth.step(cur.time, cur.vehicles, cur.tls)
        if new:
            with self._lock:
                self.pending.extend(new)

    def start(self, world, depicted, ground_z: float, consumer=None, plate_consumer=None) -> None:
        for cam in (self.site.cam, self.site.plate_cam):
            if cam:
                cam["z"] = ground_z + cam.get("height_m", CAM_HEIGHT_M)
                cam["ground_z"] = ground_z
        (self.out / "site.json").write_text(json.dumps(self.site.to_json(), indent=2), encoding="utf-8")
        self.camera = SiteCamera(world, self.site, self.out if self.record else None, depicted, consumer=consumer)
        self.camera.start()
        if self.site.plate_cam:
            self.plate_camera = SiteCamera(world, self.site, self.out if self.record else None, depicted,
                                           consumer=plate_consumer, which="plate_cam")
            self.plate_camera.start()

    def tick(self, plate_of=None, car_of=None) -> None:
        """Main thread, now and then: the new truth entries (with the plate the car carries: plate_of(vid), and its
        model and colour: car_of(vid)) and the controller's new log lines to their files."""
        with self._lock:
            new, self.pending = self.pending, []
            times, states = list(self.signals.times), list(self.signals.states)
        rows = []
        for e in new:
            r = dict(vars(e))
            r["plate"] = plate_of(e.vid) if plate_of else None
            r["car"] = car_of(e.vid) if car_of else None
            rows.append(r)
        if rows:
            write_jsonl(self.out / "truth.jsonl", rows)
        if len(times) > self._sig_written:
            log_ = SignalLog(times, states)
            ch = log_.changes(self.links)
            done = getattr(self, "_changes_written", 0)
            if len(ch) > done:
                write_jsonl(self.out / "signals.jsonl", [{"t": t, "links": self.links, "state": s} for t, s in ch[done:]])
                self._changes_written = len(ch)
            self._sig_written = len(times)

    def stop(self) -> None:
        for c in (self.camera, self.plate_camera):
            if c is not None:
                c.stop()
