"""Traffic cameras at every signalised junction, with the CARLA-trained YOLO26n detector, for the free-fly
viewer (cosim/freefly.py).

* Cameras: one per road at each signalised junction, the layout measured best on Ingolstadt7
  (cosim/perception.plan_cameras, "mast": 10 m into the junction, 9 m up, pitch -22, 960x540, 75 deg),
  each looking down its road at the queue coming in. In the CARLA window each shows as a small blue box
  on a grey pole (debug drawing) when you are within 120 m.
* Inspect: fly within INSPECT_M of a camera and press H. Native (default, 22 Oct): CARLA's own view is put
  at the camera's pose, so the feed is CARLA's window itself, at its full frame rate and resolution, and only
  the street-camera overlay and the detector's boxes are drawn over it (cosim/cctv.py). A hidden camera
  sensor at the same pose (and the view's field of view), the detector process's own, feeds the detector
  about 10 pictures a second; only the boxes come back to the viewer;
  the boxes are scaled to the screen and carried forward along each vehicle's motion to the moment drawn
  (`boxes_now`). Before, the picture shown was the sensor's own, copied to Python, and it only changed
  when a detection came back (5-12 a second): it lagged. `--cctv sensor` keeps that way. Sensor: a camera
  sensor is put at that camera's pose, its pictures go through the detector (models/carla-yolo26n.pt: car, van, bus, truck, ambulance,
  ambulance_lights, police, police_lights, pedestrian, trained on CARLA images, tools/train_detector.py),
  and the picture with the boxes and the counts is shown over the CARLA window. H again closes it.
  Only the inspected camera runs (one sensor and the detector would cost every camera's frames).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("inspector")
ROOT = Path(__file__).resolve().parents[1]
# Fine-tuned on Town05 from these very cameras (tools/make_dataset_town05.py, out/detector/eval_town05.md);
# the first CARLA model (Town10, Ingolstadt7, Cologne3) if it is missing
MODEL = next(p for p in (ROOT / "models" / "carla-yolo26n-t05.pt", ROOT / "models" / "carla-yolo26n.pt")
             if p.exists() or p.name == "carla-yolo26n.pt")
RES, FOV = (960, 540), 75.0
NATIVE_TICK_S = 0.2     # native: pictures a second for the detector (5; at 10 and 960 px YOLO took half of the
                        # GTX 1650 and CARLA fell from ~55 to 30 FPS while a camera was open)
NATIVE_IMGSZ = 640      # native: YOLO's input size (the sensor feed keeps 960)
NATIVE_WIDTH = 640      # native: the detector sensor's width. Measured at 08:00 with the camera open 20 s
                        # (tools/freefly_camera_compare.bat): 960 px, CARLA 23 FPS, 57 % of its pictures uneven,
                        # judder 28 %; 640 px, 35 FPS, 0 %, 0.06 %, the same boxes (YOLO works at 640 anyway)
VIEW_FOV = 90.0         # CARLA's spectator's horizontal field of view (UE4's default; --cctv-fov)
CARRY_MAX_S = 0.5       # native: boxes are carried forward at most this long after their picture
FLY_AWAY_M = 1.5        # native: you flew the view this far from the camera: the camera closes
TURN_AWAY_DEG = 3.0     # native: or turned it this much
ARRIVE_S = 2.0          # native: the view gets this long to reach the camera after H
INSPECT_M = 25.0
MARK_M = 120.0

# class -> (group shown in the counts, BGR colour of its boxes, label)
GROUPS = {
    "car": ("vehicles", (80, 200, 80), "car"),
    "van": ("vehicles", (80, 200, 80), "van"),
    "truck": ("vehicles", (80, 200, 80), "truck"),
    "bus": ("public transport", (0, 190, 255), "bus"),
    "pedestrian": ("pedestrians", (255, 220, 60), "pedestrian"),
    "ambulance": ("emergency, siren off", (255, 140, 60), "ambulance"),
    "police": ("emergency, siren off", (255, 140, 60), "police"),
    "ambulance_lights": ("emergency, SIREN ON", (60, 60, 255), "ambulance SIREN ON"),
    "police_lights": ("emergency, SIREN ON", (60, 60, 255), "police SIREN ON"),
}
GROUP_ORDER = ["vehicles", "public transport", "pedestrians", "emergency, SIREN ON", "emergency, siren off"]


def compass(yaw: float) -> str:
    """Direction a camera looks, CARLA yaw (0 = +x = east, 90 = +y = south)."""
    return ("east", "south", "west", "north")[int(((yaw % 360.0) + 45.0) // 90.0) % 4]


class TrafficCameras:
    def __init__(self, world, net, frame, cmap, host: str = "127.0.0.1", port: int = 2000):
        import carla

        from cosim.perception import plan_cameras

        def ground_z(x, y):
            wp = cmap.get_waypoint(carla.Location(x, y, 0.0), project_to_road=True, lane_type=carla.LaneType.Any)
            return float(wp.transform.location.z) if wp is not None else 0.0

        self.world = world
        self.host, self.port = host, port  # the detector process connects to CARLA itself (native feed)
        self.cams = plan_cameras(net, frame, ground_z, back_m=-10.0, pitch=-22.0)
        self.ground = [ground_z(c.x, c.y) for c in self.cams]
        self.names = []
        per_tls: dict[str, int] = {}
        for c in self.cams:
            per_tls[c.tls] = per_tls.get(c.tls, 0) + 1
            self.names.append(f"{c.tls}-{per_tls[c.tls]} looking {compass(c.yaw)}")
        log.info("%d traffic cameras at %d signals", len(self.cams), len(per_tls))
        self.active: int | None = None
        self.sensor = None
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self.view = None  # (RGB uint8 (H, W, 3) or None in native mode, counts dict, detector fps, camera name)
        self.native = False
        self.view_fov = VIEW_FOV
        self.native_tick = NATIVE_TICK_S   # native feed: the detector sensor's seconds between pictures
        self.native_width = NATIVE_WIDTH   # and its width (its height follows the window's shape)
        self.det: list[tuple[float, list, tuple[int, int]]] = []  # the last two detections: (picture time, boxes, size)
        self._back = None  # native: where the view was before the camera opened
        self._at_pose, self._opened_t = False, 0.0
        self.pose = None   # native: the camera's pose (x, y, z) the view was put at
        self.model = None
        self.model_state = "loading the detector"
        self._stop = False
        self._start_detector()
        self._det = threading.Thread(target=self._detect_loop, daemon=True)
        self._det.start()
        self._mark_t = 0.0
        self.hide_near: list[tuple[float, float, float]] = []   # (x, y, radius): no marker there (an enforcement
        # camera's view: a debug drawing is seen by every camera, and no real street has these markers)

    # ------------------------------------------------------------ the detector
    def _start_detector(self) -> None:
        """The detector runs in a process of its own: loading PyTorch and running YOLO in this process
        held Python's lock long enough to stall the vehicles (measured: 6 updates a second while the
        model loaded, against 60)."""
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._in = ctx.Queue()  # pictures (sensor feed: one at a time, see `busy`) and commands (native feed)
        self._out = ctx.Queue()
        self._proc = ctx.Process(target=detector_process, args=(self._in, self._out, str(MODEL), self.host, self.port),
                                 daemon=True)
        self._proc.start()

    def _on_image(self, img) -> None:
        with self._lock:
            self._frame = img
            self._frame_id += 1
            self._frame_t = time.perf_counter()

    def _detect_loop(self) -> None:
        import cv2

        sent: dict[int, np.ndarray] = {}
        sent_t: dict[int, float] = {}
        seen, busy, times = 0, False, []
        while not self._stop:
            # results and messages from the detector process. With no picture of ours to send (no camera open, or
            # the native feed, whose pictures the detector process takes itself) wait for them, instead of asking
            # 200 times a second: each ask let go of Python's lock and won it back from the viewer's busy threads
            idle = self.active is None or (self.native and self._frame is None)
            first = True
            try:
                while True:
                    if first and idle:
                        msg = self._out.get(timeout=0.05)
                    else:
                        msg = self._out.get_nowait()
                    first = False
                    if msg[0] == "ready":
                        self.model, self.model_state = True, msg[1]
                        log.info("detector ready: %s", msg[1])
                    elif msg[0] == "error":
                        self.model_state = f"detector not available: {msg[1]}"
                        log.warning("detector: %s", msg[1])
                    elif msg[0] == "ndet":  # native feed: the detector process's own sensor
                        _, boxes, dt, t_pic, size = msg
                        if self.active is None or not self.native:
                            continue
                        boxes = self._voted(t_pic, boxes)
                        times = (times + [dt])[-20:]
                        self.det = (self.det + [(t_pic, boxes, size)])[-2:]
                        self.view = (None, count(boxes), len(times) / sum(times), self.names[self.active])
                    elif msg[0] == "det":
                        _, fid, boxes, dt = msg
                        busy = False
                        bgr = sent.pop(fid, None)
                        t_pic = sent_t.pop(fid, time.perf_counter())
                        sent.clear()
                        sent_t.clear()
                        if bgr is None or self.active is None:
                            continue
                        boxes = self._voted(t_pic, boxes)
                        times = (times + [dt])[-20:]
                        rate = len(times) / sum(times)
                        if self.native:  # the picture is CARLA's window: only the boxes and the counts
                            self.det = (self.det + [(t_pic, boxes, (bgr.shape[1], bgr.shape[0]))])[-2:]
                            self.view = (None, count(boxes), rate, self.names[self.active])
                        else:
                            self.view = (draw_boxes(cv2, bgr, boxes), count(boxes), rate, self.names[self.active])
            except Exception:  # noqa: BLE001 - queue empty
                pass
            with self._lock:
                img, fid, t_img = self._frame, self._frame_id, getattr(self, "_frame_t", 0.0)
            if img is None or fid == seen or self.active is None:
                time.sleep(0.005)
                continue
            bgr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy()
            seen = fid
            if self.model is None:  # still loading: show the picture without boxes
                self.view = (None if self.native else bgr[:, :, ::-1].copy(), count([]), 0.0, self.names[self.active])
                continue
            if busy:
                continue
            try:
                self._in.put_nowait((fid, bgr.shape[0], bgr.shape[1], bgr.tobytes()))
                sent[fid] = bgr
                sent_t[fid] = t_img
                busy = True
            except Exception:  # noqa: BLE001 - the detector is still on the last one
                pass

    def _voted(self, t: float, boxes: list) -> list:
        vote = getattr(self, "vote", None)
        if vote is not None:
            try:
                return vote.relabel(t, boxes)
            except Exception as e:  # noqa: BLE001 - the window keeps the picture's own labels
                log.debug("track vote: %s", e)
        return [tuple(b[:6]) for b in boxes]

    # ------------------------------------------------------------ cameras in the world
    def nearest(self, x: float, y: float, z: float) -> tuple[int | None, float]:
        best, bd = None, 1e9
        for i, c in enumerate(self.cams):
            d = math.sqrt((c.x - x) ** 2 + (c.y - y) ** 2 + (c.z - z) ** 2)
            if d < bd:
                best, bd = i, d
        return best, bd

    def draw_markers(self, x: float, y: float) -> None:
        """Small blue boxes on dark poles at the cameras (blue: red ones were taken for traffic lights that
        never turned green), drawn once and for good (CARLA debug drawing with
        life_time 0). They were redrawn twice a second with a 0.6 s life: each redraw overlapped the last
        and the poles flashed. Each box sits 0.8 m behind its camera's lens, so an open camera never sees
        its own marker (it used to fill the live picture)."""
        if self._mark_t:
            return
        import carla

        self._mark_t = time.time()
        dbg = self.world.debug
        for i, c in enumerate(self.cams):
            if any(math.hypot(c.x - hx, c.y - hy) < hr for hx, hy, hr in self.hide_near):
                continue
            a = math.radians(c.yaw)
            bx, by = c.x - 0.8 * math.cos(a), c.y - 0.8 * math.sin(a)
            dbg.draw_line(carla.Location(bx, by, self.ground[i]), carla.Location(bx, by, c.z - 0.2),
                          thickness=0.05, color=carla.Color(25, 25, 25), life_time=0.0)
            dbg.draw_box(carla.BoundingBox(carla.Location(bx, by, c.z), carla.Vector3D(0.35, 0.2, 0.18)),
                         carla.Rotation(c.pitch, c.yaw, 0.0), thickness=0.06, color=carla.Color(40, 110, 255),
                         life_time=0.0)

    def open(self, i: int, native: bool = False, screen: tuple[int, int] | None = None) -> None:
        """Camera i. native: CARLA's view goes to the camera (screen: the window's size, for the sensor's
        shape, so its boxes map onto the window by scaling alone)."""
        import carla

        self.close()
        c = self.cams[i]
        tf = carla.Transform(carla.Location(c.x, c.y, c.z), carla.Rotation(c.pitch, c.yaw, 0.0))
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.native = native
        self.det = []
        self.vote = TrackVote()   # a new camera: new tracks
        self.active = i
        self.view = None
        if native:
            w, h = screen or RES
            size = (self.native_width, max(2, round(self.native_width * h / max(w, 1))))
            spectator = self.world.get_spectator()
            self._back = spectator.get_transform()
            spectator.set_transform(tf)
            self.pose = (c.x, c.y, c.z)
            self._at_pose, self._opened_t = False, time.time()
            # The detector process puts its own sensor there and sends back only the boxes. With the sensor
            # here, every picture (960 x 540, 10 a second) was copied and pickled in the viewer's process:
            # its SUMO thread fell behind and the vehicles juddered while a camera was open (measured on the
            # GTX 1650 laptop: judder 53-60 % of moving-vehicle pictures, against 0-1 % with it closed).
            self._in.put(("open", (c.x, c.y, c.z, c.pitch, c.yaw), size, self.view_fov, self.native_tick))
        else:
            for k, v in (("image_size_x", RES[0]), ("image_size_y", RES[1]), ("fov", FOV), ("sensor_tick", 0.066)):
                bp.set_attribute(k, str(v))
            self.sensor = self.world.spawn_actor(bp, tf)
            self.sensor.listen(self._on_image)
        log.info("inspecting camera %s (%s)", self.names[i], "native" if native else "sensor")

    def flown_away(self, tf) -> bool:
        """Native: the view (transform tf) was flown or turned away from the camera (Z Q S D, the arrows, the
        mouse): it is no longer the camera's."""
        if not self.native or self.pose is None or self.active is None:
            return False
        c = self.cams[self.active]
        loc, rot = tf.location, tf.rotation
        turned = max(abs((rot.yaw - c.yaw + 180.0) % 360.0 - 180.0), abs(rot.pitch - c.pitch))
        away = math.dist((loc.x, loc.y, loc.z), self.pose) > FLY_AWAY_M or turned > TURN_AWAY_DEG
        if not self._at_pose:
            # CARLA moves the view on its next picture: until the view has been seen at the camera, it is
            # still where it was (it was read there right after H, and the camera closed at once)
            if not away:
                self._at_pose = True
            return away and time.time() - self._opened_t > ARRIVE_S
        return away

    def boxes_now(self, t: float, screen: tuple[int, int]) -> list[tuple[float, float, float, float, str, float]]:
        """Native: the last detection's boxes in window pixels, each carried forward to time t along its
        motion since the detection before (matched by overlap and class), at most CARRY_MAX_S."""
        if not self.det:
            return []
        t1, boxes, (iw, ih) = self.det[-1]
        sx, sy = screen[0] / iw, screen[1] / ih
        prev = self.det[-2] if len(self.det) > 1 else None
        dt = min(max(t - t1, 0.0), CARRY_MAX_S)
        out = []
        for x0, y0, x1, y1, cls, p in boxes:
            vx = vy = 0.0
            if prev is not None and t1 > prev[0]:
                best, bi = 0.3, None
                for b in prev[1]:
                    if b[4] == cls:
                        o = _iou((x0, y0, x1, y1), b[:4])
                        if o > best:
                            best, bi = o, b
                if bi is not None:
                    span = t1 - prev[0]
                    vx = ((x0 + x1) - (bi[0] + bi[2])) / 2.0 / span
                    vy = ((y0 + y1) - (bi[1] + bi[3])) / 2.0 / span
            ox, oy = vx * dt, vy * dt
            out.append(((x0 + ox) * sx, (y0 + oy) * sy, (x1 + ox) * sx, (y1 + oy) * sy, cls, p))
        return out

    def close(self, restore: bool = True) -> None:
        """restore (native): the view goes back where it was before the camera opened."""
        if getattr(self, "native", False) and getattr(self, "active", None) is not None:
            try:
                self._in.put(("close",))
            except Exception:  # noqa: BLE001 - the detector is gone
                pass
        if getattr(self, "native", False) and restore and self._back is not None:
            try:
                self.world.get_spectator().set_transform(self._back)
            except Exception:  # noqa: BLE001 - CARLA gone
                pass
        self._back = None
        self.pose = None
        self.det = []
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception:  # noqa: BLE001 - gone already
                pass
        self.sensor = None
        self.active = None
        with self._lock:
            self._frame = None
        self.view = None

    def shutdown(self) -> None:
        self._stop = True
        self.close()
        try:
            self._in.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        if self._proc.is_alive():
            self._proc.terminate()
        self._in.cancel_join_thread()  # pictures not read yet must not hold the viewer up at exit


def detector_process(q_in, q_out, model_path: str, host: str | None = None, port: int | None = None) -> None:
    """In its own process: load YOLO (on the GPU if PyTorch sees one) and detect on every picture sent
    (sensor feed), or, for the native feed, on the pictures of a camera sensor of its own: ("open", (x, y, z,
    pitch, yaw), (w, h), fov, tick) puts it there (its own CARLA connection), ("close",) takes it away; only
    the boxes go back, with the moment the picture came."""
    import queue
    import threading

    try:
        import torch
        from ultralytics import YOLO

        dev = os.environ.get("FREEFLY_YOLO_DEVICE") or (0 if torch.cuda.is_available() else "cpu")
        model = YOLO(model_path)
        prec = 16 if dev != "cpu" else None  # FP16 on the GPU (ultralytics 8.4: `quantize`, formerly `half`)
        model.predict(np.zeros((540, 960, 3), np.uint8), imgsz=960, device=dev, verbose=False, quantize=prec)
        q_out.put(("ready", f"YOLO26n (CARLA-trained) on {'GPU' if dev != 'cpu' else 'CPU'}"))
    except Exception as e:  # noqa: BLE001
        q_out.put(("error", str(e)))
        return

    from cosim.siren_flash import light_numbers

    def detect(bgr, imgsz: int = 960):
        r = model.predict(bgr, imgsz=imgsz, conf=0.35, device=dev, verbose=False, quantize=prec)[0]
        out = []
        rgb = bgr[:, :, ::-1]
        for (x0, y0, x1, y1), k, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                          r.boxes.conf.cpu().numpy(), strict=True):
            name = r.names[int(k)]
            # (10 Nov) the light numbers of a vehicle's box, for the siren-light model (TrackVote, cosim/siren_flash)
            lights = light_numbers(rgb, (x0, y0, x1, y1)) if name != "pedestrian" else None
            out.append((float(x0), float(y0), float(x1), float(y1), name, float(p), lights))
        return out

    world = sensor = None
    latest: list = [None]  # (picture, time it came), the newest only
    lock = threading.Lock()

    def on_image(img):
        with lock:
            latest[0] = (img, time.time())

    def close():
        nonlocal sensor
        if sensor is not None:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:  # noqa: BLE001 - gone already
                pass
        sensor = None
        with lock:
            latest[0] = None

    try:
        while True:
            try:
                item = q_in.get(timeout=0.01)
            except queue.Empty:
                item = ()
            if item is None:
                return
            if item and item[0] == "open":
                import carla

                close()
                _, (x, y, z, pitch, yaw), (w, h), fov, tick = item
                if world is None:
                    client = carla.Client(host or "127.0.0.1", port or 2000)
                    client.set_timeout(20.0)
                    world = client.get_world()
                bp = world.get_blueprint_library().find("sensor.camera.rgb")
                for k, v in (("image_size_x", w), ("image_size_y", h), ("fov", fov), ("sensor_tick", tick)):
                    bp.set_attribute(k, str(v))
                sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch, yaw, 0.0)))
                sensor.listen(on_image)
            elif item and item[0] == "close":
                close()
            elif item:  # a picture from the viewer (sensor feed)
                fid, h, w, buf = item
                bgr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                t = time.perf_counter()
                boxes = detect(bgr)
                q_out.put(("det", fid, boxes, time.perf_counter() - t))
            with lock:
                got, latest[0] = latest[0], None
            if got is not None and sensor is not None:
                img, t_pic = got
                bgr = np.ascontiguousarray(np.frombuffer(img.raw_data, dtype=np.uint8)
                                           .reshape(img.height, img.width, 4)[:, :, :3])
                t = time.perf_counter()
                boxes = detect(bgr, NATIVE_IMGSZ)
                q_out.put(("ndet", boxes, time.perf_counter() - t, t_pic, (img.width, img.height)))
    finally:
        close()


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


EMERGENCY = {"ambulance": "ambulance", "ambulance_lights": "ambulance", "police": "police", "police_lights": "police"}


class TrackVote:
    """(10 Nov) The H window's boxes labelled by what each vehicle has been over its last pictures, not by one
    picture: the boxes are followed from picture to picture (camera_reports.Tracker, as the cameras' reports) and an
    emergency vehicle's label is voted over its track; ambulance or police by the majority of its emergency pictures.
    Siren on: once the track has `flash.pictures` pictures, the siren-light model says (cosim/siren_flash.py: does its
    bar flash?); before that, or without the model, the detector's vote, `vote_need` of its last `vote_frames`
    pictures (kept while the track lives, as in the camera reports). Before 10 Nov each picture was counted on its
    own: the detector finds a siren-on vehicle as siren-on in 34 % of pictures and calls a siren-off one siren-on in
    9 % (out/detector/eval_town05.md), so the label flickered. Other vehicles keep the picture's class. The boxes come
    back as (x0, y0, x1, y1, class, conf)."""

    def __init__(self, rules=None, flash=None):
        from collections import deque

        from cosim.camera_reports import ReportRules, Tracker

        self.rules = rules or ReportRules()
        self.tracker = Tracker(vote_frames=self.rules.vote_frames)
        if flash is None:
            try:
                from cosim.siren_flash import MODEL, SirenFlash

                flash = SirenFlash() if MODEL.is_file() else False
            except Exception as e:  # noqa: BLE001 - the detector's vote alone
                log.debug("siren-light model: %s", e)
                flash = False
        self.flash = flash or None
        self._deque = deque

    def relabel(self, t: float, boxes: list) -> list:
        from cosim.camera_reports import SIREN

        live = self.tracker.update(t, [b[:4] for b in boxes], [b[4] for b in boxes])
        by_box = {tr.box: tr for tr in live}
        out = []
        for b in boxes:
            tr = by_box.get(tuple(float(v) for v in b[:4]))
            if tr is None:
                out.append(tuple(b[:6]))
                continue
            names = list(tr.names)
            if self.flash is not None:
                lights = getattr(tr, "lights", None)
                if lights is None:
                    lights = tr.lights = self._deque(maxlen=self.flash.pictures)
                if len(b) > 6 and b[6] is not None:
                    lights.append(b[6])
            p = self.flash.prob(list(tr.lights)) if self.flash is not None else None
            if p is not None:
                tr.siren = p >= self.flash.threshold
            elif not tr.siren and sum(1 for n in names if n in SIREN) >= self.rules.vote_need:
                tr.siren = True
            kinds = [EMERGENCY[n] for n in names if n in EMERGENCY]
            if kinds and 2 * len(kinds) >= len(names) or (tr.siren and kinds):
                kind = max(("ambulance", "police"), key=kinds.count)
                out.append((*b[:4], kind + ("_lights" if tr.siren else ""), b[5]))
            else:
                out.append((*b[:4], b[4] if b[4] not in EMERGENCY else (names[-1] if names[-1] not in EMERGENCY
                                                                          else "car"), b[5]))
        return out


def count(boxes) -> dict:
    out = {g: 0 for g in GROUP_ORDER}
    for *_, cls, _ in boxes:
        out[GROUPS.get(cls, ("vehicles",))[0]] += 1
    return out


def draw_boxes(cv2, bgr: np.ndarray, boxes) -> np.ndarray:
    """The picture (RGB) with a coloured box and label per detection."""
    for x0, y0, x1, y1, cls, p in boxes:
        _, colour, label = GROUPS.get(cls, ("vehicles", (200, 200, 200), cls))
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(bgr, p0, p1, colour, 2)
        text = f"{label} {p:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(bgr, (p0[0], p0[1] - th - 6), (p0[0] + tw + 6, p0[1]), colour, -1)
        cv2.putText(bgr, text, (p0[0] + 3, p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return bgr[:, :, ::-1].copy()
