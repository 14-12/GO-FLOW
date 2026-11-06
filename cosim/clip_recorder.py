"""Video clips from Town05's traffic cameras, with the accidents labelled (30 Oct).

For the GO-FLOW incident detection agent (an MP4 in, a verdict per 5 s window out): fixed traffic cameras record
the free-fly viewer's town as MP4 files while it runs, every accident it has is logged with the second it happened,
and two clips are cut from the recordings with their labels:

* `normal_traffic.mp4`: a camera with no accident in its picture for the whole clip;
* `crash.mp4`: a camera with a crash in its picture, from some time before the crash to minutes after it
  (the wrecks standing, the police coming, the recovery truck).

Cameras: the 55 of out/town05_cameras/layout.json (one per road at each signalised junction, 9 m up, pitch -22 deg,
960x540, 75 deg), the same poses as the viewer's traffic cameras. A CARLA RGB sensor per recorded camera takes a
picture every 1/fps CARLA seconds (`sensor_tick`); each picture is written to the camera's MP4 as it comes, with
its SUMO second in a CSV next to it (`<camera>.frames.csv`), so a label is placed by the simulation's own clock,
not by the frame count. Nothing is staged: the accidents are cosim/incidents.py's, careless drivers who crash or
not depending on the traffic around them. What is decided here is only which cameras film.

A camera "sees" an accident when a wreck's middle projects into its picture, 2-90 m ahead (cosim/perception.project,
the cameras' own pinhole model). Buildings or trees in between are not checked.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cosim.perception import CamSpec, project

log = logging.getLogger("clips")

SEE_MAX_M = 90.0
SEE_MIN_M = 2.0
EDGE_PX = 8          # a wreck this close to the picture's edge does not count as seen
CAM_HEIGHT_M = 9.0   # the layout's cameras are this far above the road (cosim/perception.CAM_HEIGHT_M)
WRECK_Z_M = 0.7      # the middle of a car above the road


def sees(spec: CamSpec, width: int, height: int, fov: float, x: float, y: float) -> tuple[bool, float]:
    """Whether CARLA point (x, y) on the road is in this camera's picture, and how far ahead (m)."""
    z = spec.z - CAM_HEIGHT_M + WRECK_Z_M
    px, depth = project(np.array([[x, y, z]]), spec, width, height, fov)
    d = float(depth[0])
    if not (SEE_MIN_M <= d <= SEE_MAX_M):
        return False, d
    u, v = float(px[0, 0]), float(px[0, 1])
    return (EDGE_PX <= u <= width - EDGE_PX and EDGE_PX <= v <= height - EDGE_PX), d


def near_view(spec: CamSpec, fov: float, x: float, y: float, reach_m: float = 140.0) -> bool:
    """Rough 2-D test for the viewer's `in_view` (what may be in this camera's picture: nobody is put on the road or
    taken off there): within reach and inside the picture's width plus 15 deg each side."""
    dx, dy = x - spec.x, y - spec.y
    d = math.hypot(dx, dy)
    if d < 12.0:
        return True
    if d > reach_m:
        return False
    ang = math.degrees(math.atan2(dy, dx)) - spec.yaw
    ang = (ang + 180.0) % 360.0 - 180.0
    return abs(ang) <= fov / 2 + 15.0


# (6 Nov) Option A of the GO-FLOW crash delivery: where a crash is "clearly visible" to a camera (the team's request,
# 2 Nov: the wreck within about 30 m and fully in the picture). A vehicle's middle and both its ends project into the
# picture at least FOCUS_EDGE_PX from its edge, FOCUS_MIN_M to FOCUS_MAX_M ahead.
FOCUS_MAX_M = 30.0
FOCUS_MIN_M = 6.0
FOCUS_EDGE_PX = 24
FOCUS_HALF_LEN_M = 2.4


class Focus:
    """The places clearly visible to some cameras (option A, cosim/incidents.IncidentManager.focus): called with a SUMO
    point and the heading there (degrees, SUMO: 0 north, clockwise), it tells whether a car standing there would be
    whole in a camera's picture, within FOCUS_MAX_M; `seen_by` names that camera. to_carla: SUMO (x, y) -> CARLA."""

    def __init__(self, cams: list[tuple[str, CamSpec]], width: int, height: int, fov: float, to_carla,
                 max_m: float = FOCUS_MAX_M, min_m: float = FOCUS_MIN_M):
        self.cams = list(cams)
        self.w, self.h, self.fov = width, height, fov
        self.to_carla = to_carla
        self.max_m, self.min_m = max_m, min_m

    def seen_by(self, x: float, y: float, angle: float | None = None) -> str | None:
        cx, cy = self.to_carla(x, y)
        pts = [(cx, cy)]
        if angle is not None:  # SUMO heading -> CARLA (x east, y south): the car's two ends
            a = math.radians(angle)
            hx, hy = self.to_carla(x + math.sin(a), y + math.cos(a))
            ux, uy = hx - cx, hy - cy
            n = math.hypot(ux, uy) or 1.0
            ux, uy = ux / n, uy / n
            pts += [(cx + ux * FOCUS_HALF_LEN_M, cy + uy * FOCUS_HALF_LEN_M),
                    (cx - ux * FOCUS_HALF_LEN_M, cy - uy * FOCUS_HALF_LEN_M)]
        for name, spec in self.cams:
            dx, dy = cx - spec.x, cy - spec.y
            if math.hypot(dx, dy) > self.max_m + 3.0:
                continue
            ang = (math.degrees(math.atan2(dy, dx)) - spec.yaw + 180.0) % 360.0 - 180.0
            if abs(ang) > self.fov / 2 + 5.0:
                continue
            z = spec.z - CAM_HEIGHT_M + WRECK_Z_M
            px, depth = project(np.array([[px_, py_, z] for px_, py_ in pts]), spec, self.w, self.h, self.fov)
            if not all(self.min_m <= float(d) <= self.max_m for d in depth):
                continue
            e = FOCUS_EDGE_PX
            if all(e <= float(u) <= self.w - e and e <= float(v) <= self.h - e for u, v in px):
                return name
        return None

    def __call__(self, x: float, y: float, angle: float | None = None) -> bool:
        return self.seen_by(x, y, angle) is not None

    def near(self, x: float, y: float, reach_m: float) -> bool:
        """SUMO (x, y) within reach_m of a camera (a cheap test before the others)."""
        cx, cy = self.to_carla(x, y)
        return any(math.hypot(cx - spec.x, cy - spec.y) <= reach_m for _, spec in self.cams)


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


@dataclass
class Recording:
    index: int                       # camera index in the layout
    name: str
    spec: CamSpec
    path: Path
    started_sim: float | None = None
    started_wall: float = field(default_factory=time.time)
    reason: str = "planned"          # planned | after crash <id>
    frames: int = 0
    dropped: int = 0
    last_sim: float | None = None
    sensor: object = None
    q: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=600))
    thread: threading.Thread | None = None
    stopped: bool = False


class ClipRecorder:
    """Records some of the layout's cameras to MP4 while the viewer runs (see the module doc)."""

    def __init__(self, world, layout, out_dir: Path, sim_now, fps: float = 10.0, crf: int = 20):
        self.world, self.lay, self.out, self.sim_now = world, layout, Path(out_dir), sim_now
        self.fps, self.crf = fps, crf
        self.out.mkdir(parents=True, exist_ok=True)
        self.recs: dict[int, Recording] = {}
        self.w, self.h, self.fov = layout.cams[0].width, layout.cams[0].height, layout.cams[0].fov

    # -- cameras --------------------------------------------------------------------------------

    def index_of(self, name: str) -> int:
        return self.lay.names.index(name)

    def add_camera(self, name: str, spec: CamSpec) -> int:
        """A camera of its own beside the layout's (an overview camera over a road, looking back at its junction),
        same picture size and field of view. Its index."""
        from cosim.perception import LaneCamera

        self.lay.cams.append(LaneCamera(spec, self.w, self.h, self.fov))
        self.lay.names.append(name)
        return len(self.lay.cams) - 1

    def start(self, i: int, reason: str = "planned") -> Recording:
        import carla
        import imageio_ffmpeg

        if i in self.recs:
            return self.recs[i]
        spec = self.lay.cams[i].spec
        name = self.lay.names[i]
        rec = Recording(i, name, spec, self.out / f"{slug(name)}.mp4", reason=reason)
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", str(self.w)), ("image_size_y", str(self.h)), ("fov", str(self.fov)),
                     ("sensor_tick", f"{1.0 / self.fps:.4f}"), ("motion_blur_intensity", "0.0")):
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        tf = carla.Transform(carla.Location(spec.x, spec.y, spec.z), carla.Rotation(spec.pitch, spec.yaw, 0.0))
        rec.sensor = self.world.spawn_actor(bp, tf)
        writer = imageio_ffmpeg.write_frames(str(rec.path), (self.w, self.h), fps=self.fps, codec="libx264",
                                             pix_fmt_in="rgb24", macro_block_size=2,  # 540 rows stay 540
                                             output_params=["-crf", str(self.crf), "-preset", "veryfast",
                                                            "-pix_fmt", "yuv420p"])
        writer.send(None)
        index = (self.out / f"{slug(name)}.frames.csv").open("w", newline="", encoding="utf-8")
        cw = csv.writer(index)
        cw.writerow(["frame", "video_s", "sim_s", "carla_s", "wall_s"])

        def on_image(img, rec=rec):
            if rec.stopped:
                return
            try:
                rec.q.put_nowait((bytes(img.raw_data), img.timestamp, self.sim_now(), time.time()))
            except queue.Full:
                rec.dropped += 1

        def write(rec=rec):
            try:
                while True:
                    item = rec.q.get()
                    if item is None:
                        break
                    raw, ts, sim_s, wall = item
                    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 4)
                    writer.send(np.ascontiguousarray(bgra[:, :, 2::-1]).tobytes())
                    if rec.started_sim is None:
                        rec.started_sim = sim_s
                    cw.writerow([rec.frames, f"{rec.frames / self.fps:.2f}", f"{sim_s:.2f}", f"{ts:.3f}",
                                 f"{wall:.3f}"])
                    rec.frames += 1
                    rec.last_sim = sim_s
            except Exception:  # a failed recording is logged, the viewer goes on
                log.exception("recording %s failed", rec.name)
            finally:
                try:
                    writer.close()
                except Exception as e:  # noqa: BLE001
                    log.warning("closing %s: %s", rec.path.name, e)
                index.close()

        rec.thread = threading.Thread(target=write, name=f"clip {name}", daemon=True)
        rec.thread.start()
        rec.sensor.listen(on_image)
        self.recs[i] = rec
        log.info("recording camera %s (%s) to %s", name, reason, rec.path)
        return rec

    def stop(self, i: int) -> None:
        rec = self.recs.get(i)
        if rec is None or rec.stopped:
            return
        rec.stopped = True
        try:
            rec.sensor.stop()
            rec.sensor.destroy()
        except Exception as e:  # noqa: BLE001
            log.debug("sensor of %s: %s", rec.name, e)
        rec.q.put(None)
        if rec.thread is not None:
            rec.thread.join(timeout=60)

    def close(self) -> None:
        for i in list(self.recs):
            self.stop(i)

    # -- what the cameras see -------------------------------------------------------------------

    def seen_by(self, x: float, y: float, cams=None) -> list[tuple[int, float]]:
        """Cameras (all of the layout's, or `cams`) whose picture holds CARLA point (x, y): (index, metres ahead),
        nearest first."""
        out = []
        for i in (range(len(self.lay.cams)) if cams is None else cams):
            ok, d = sees(self.lay.cams[i].spec, self.w, self.h, self.fov, x, y)
            if ok:
                out.append((i, d))
        return sorted(out, key=lambda t: t[1])

    def near_recorded(self, x: float, y: float) -> bool:
        return any(near_view(r.spec, self.fov, x, y) for r in self.recs.values() if not r.stopped)


def cut(src: Path, dst: Path, start_s: float, seconds: float, crf: int = 20) -> None:
    """Copy `seconds` of video from `start_s` (video seconds) of `src` to `dst`, re-encoded so it starts on a
    picture of its own."""
    import subprocess

    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{start_s:.2f}", "-i", str(src), "-t", f"{seconds:.2f}",
           "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(dst)]
    subprocess.run(cmd, check=True)


def frames_index(path: Path) -> list[tuple[int, float, float]]:
    """(frame, video s, sim s) of a recording's frames CSV."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame"]), float(r["video_s"]), float(r["sim_s"])))
    return rows


def video_s_at(rows, sim_s: float) -> float:
    """The video second of the first frame at or after simulation second `sim_s`."""
    for _, v, s in rows:
        if s >= sim_s:
            return v
    return rows[-1][1] if rows else 0.0


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
