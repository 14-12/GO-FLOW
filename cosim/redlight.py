"""The red-light agent (31 Oct): from a site camera's pictures and the junction controller's signal log, the vehicles
that entered the junction on red, each with its evidence and its plate as read from the picture.

What it is given, as a real red-light camera system is: the pictures (each with its time on the simulation's clock:
cosim/enforcement.py), the camera's calibration (pose, field of view, the road's height: the site survey), where the
stop line and the lanes are (painted on the road; here from the site plan), and the controller's signal log. What it
is not given: where the vehicles are, who they are, or who ran the red. That is cosim/red_runners.RedLightTruth's,
used only to grade it (tools/eval_redlight.py).

For each picture:
1. Vehicles are found (the Town05 detector, cosim/inspector's model: car, van, bus, truck, ambulance and police with
   and without their lights) and followed from picture to picture (`Tracker`: overlap of boxes).
2. Each vehicle's place on the road: the middle of its box's bottom edge is where its back meets the road, seen
   from behind and above; the camera's calibration turns that pixel into a point on the road (`to_ground`), and the
   point into a distance along the road from the stop line (negative before it) and an offset across (its lane).
   Its front is about one vehicle length further on (LENGTH_M by class: the camera does not see the front).
3. The moment its front crossed the line is interpolated between the two pictures either side of it; the signal of
   its lane at that moment and how long it had been red come from the controller's log.
4. A violation (`RULES`): the signal red for at least GRACE_S when the front crossed, the vehicle then seen going on
   into the junction (its back CONFIRM_M past the line within CONFIRM_S: not one that stopped just over it), seen
   before the line too, and not an emergency vehicle with its lights on (the detector's own classes, by vote).
5. Its case: the pictures before and after (A: before the line on red; B: in the junction), the frames of the
   seconds around it for a clip, the best views of its back for the plate (`PlateReader`), the plate as read and how
   sure the reading is, the times, the signal, its speed. A case is a proposal for the police office
   (tools/police_office.py), never a fine by itself.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from cosim import enforcement as enf

log = logging.getLogger("redlight")

NAMES = {0: "car", 1: "van", 2: "bus", 3: "truck", 4: "ambulance", 5: "ambulance_lights", 6: "police",
         7: "police_lights", 8: "pedestrian"}
VEHICLE = {0, 1, 2, 3, 4, 5, 6, 7}
EMERGENCY_ON = {5, 7}
LENGTH_M = {"car": 4.5, "van": 5.2, "bus": 11.0, "truck": 7.5, "ambulance": 5.8, "ambulance_lights": 5.8,
            "police": 4.9, "police_lights": 4.9}
GRACE_S = 0.5          # red for at least this long when the front crossed (shorter: not reported)
CONFIRM_M = 3.0        # ... and its back then this far past the line,
CONFIRM_S = 5.0        # within this many seconds
EXTRAPOLATE_S = 1.2    # a crossing timed from the vehicle's first second in view at most this far back
MIN_CONF = 0.35
EMERGENCY_VOTE = 0.3   # of a track's pictures with the lights-on classes: an emergency vehicle on a run
RULES = (f"Entering on red: the signal of the vehicle's lane red for at least {GRACE_S} s when its front crossed the "
         f"stop line, the vehicle then going on into the junction (its back {CONFIRM_M} m past the line within "
         f"{CONFIRM_S} s); emergency vehicles with their lights on are exempt.")


# ------------------------------------------------------------------ detection
@dataclass
class Det:
    box: tuple       # x1, y1, x2, y2 (pixels)
    cls: int
    conf: float


class OnnxDetector:
    """The Town05 detector through onnxruntime (tools/export_detector_onnx.py), CPU or GPU: boxes in the picture's
    pixels."""

    def __init__(self, path: str, imgsz: int = 960, providers=None):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

    def __call__(self, rgb: np.ndarray) -> list[Det]:
        import cv2

        h, w = rgb.shape[:2]
        s = self.imgsz / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = img
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0]          # (4 + classes, anchors)
        boxes, scores = out[:4].T, out[4:].T
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= MIN_CONF
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if not len(boxes):
            return []
        xy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                       boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        xy[:, [0, 2]] = (xy[:, [0, 2]] - left) / s
        xy[:, [1, 3]] = (xy[:, [1, 3]] - top) / s
        idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xy],
                               conf.astype(float).tolist(), MIN_CONF, 0.5)
        return [Det(tuple(float(v) for v in xy[i]), int(cls[i]), float(conf[i])) for i in np.array(idx).reshape(-1)]


class UltralyticsDetector:
    """The same detector through Ultralytics and PyTorch (the viewer's own environment: cosim/inspector.py)."""

    def __init__(self, path: str, imgsz: int = 960):
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.imgsz = imgsz

    def __call__(self, rgb: np.ndarray) -> list[Det]:
        r = self.model.predict(rgb[:, :, ::-1], imgsz=self.imgsz, conf=MIN_CONF, verbose=False)[0]
        b = r.boxes
        return [Det(tuple(float(v) for v in xy), int(c), float(p))
                for xy, c, p in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy(), strict=False)]


# ------------------------------------------------------------------ tracking
@dataclass
class Obs:
    k: int           # picture
    t: float         # its time (SUMO s)
    box: tuple
    cls: int
    conf: float
    s: float         # back of the vehicle: metres along the road from the stop line (negative before it)
    d: float         # metres across from the road's right edge of the stop line


@dataclass
class Track:
    id: int
    obs: list = field(default_factory=list)
    done: bool = False
    case: object = None

    @property
    def last(self) -> Obs:
        return self.obs[-1]

    def kind(self) -> str:
        votes: dict[int, int] = {}
        for o in self.obs:
            votes[o.cls] = votes.get(o.cls, 0) + 1
        c = max(votes, key=votes.get)
        if c in EMERGENCY_ON:
            c = c - 1
        return NAMES[c]

    def emergency_on(self) -> bool:
        n = sum(1 for o in self.obs if o.cls in EMERGENCY_ON)
        return n >= max(2, EMERGENCY_VOTE * len(self.obs))


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    """Boxes followed from picture to picture by their overlap (and, failing that, by their place on the road)."""

    def __init__(self, max_gap_s: float = 1.0):
        self.tracks: list[Track] = []
        self.n = 0
        self.max_gap_s = max_gap_s

    def update(self, obs: list[Obs]) -> list[Track]:
        live = [tr for tr in self.tracks if not tr.done]
        pairs = []
        for i, o in enumerate(obs):
            for tr in live:
                if o.t - tr.last.t > self.max_gap_s:
                    continue
                v = iou(o.box, tr.last.box)
                near = math.hypot(o.s - tr.last.s, o.d - tr.last.d)
                if v > 0.2 or near < 3.0:
                    pairs.append((v - 0.05 * near, i, tr))
        pairs.sort(key=lambda p: -p[0])
        used_o, used_t = set(), set()
        for _, i, tr in pairs:
            if i in used_o or tr.id in used_t:
                continue
            tr.obs.append(obs[i])
            used_o.add(i)
            used_t.add(tr.id)
        for i, o in enumerate(obs):
            if i not in used_o:
                self.n += 1
                self.tracks.append(Track(self.n, [o]))
        if obs:
            now = obs[0].t
            for tr in live:
                if now - tr.last.t > self.max_gap_s:
                    tr.done = True
        return [tr for tr in self.tracks if not tr.done]


# ------------------------------------------------------------------ the site's road
class Road:
    """The stop line and the lanes on the road plane (CARLA): along (s, metres past the line), across (d, metres
    from its right end)."""

    def __init__(self, site: enf.Site):
        (rx, ry), (lx, ly) = site.line
        self.o = np.array([rx, ry])
        across = np.array([lx - rx, ly - ry])
        self.width = float(np.linalg.norm(across))
        self.v = across / self.width                       # right -> left along the line
        h = math.radians(site.heading)
        self.u = np.array([math.cos(h), math.sin(h)])     # towards the junction
        self.lane_w = site.lane_width
        self.lanes = site.lanes                            # right to left

    def sd(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y]) - self.o
        return float(p @ self.u), float(p @ self.v)

    def lane(self, d: float) -> str | None:
        i = int(d // self.lane_w)
        return self.lanes[i] if 0 <= i < len(self.lanes) else None

    def xy(self, s: float, d: float) -> tuple[float, float]:
        """The point s along, d across (the inverse of sd)."""
        m = np.array([self.u, self.v]).T
        p = self.o + m @ np.linalg.solve(np.array([[self.u @ self.u, self.u @ self.v], [self.v @ self.u, self.v @ self.v]]),
                                        np.array([s, d]))
        return float(p[0]), float(p[1])


@dataclass
class Case:
    """A proposed violation (module doc), with what the police office needs to judge it."""
    id: str
    site: str
    track: int
    kind: str                # the vehicle's class, by vote
    lane: str
    t_cross: float           # its front crossed the stop line (SUMO s)
    red_s: float             # the signal had been red this long then
    signal: str              # the lane's signal then ("r")
    speed_kmh: float
    frame_a: int             # before the line, on red
    frame_b: int             # in the junction
    frames: list             # the clip: pictures from 2 s before to 3 s after
    plate_frames: list       # the best views of its back
    plate: str | None = None
    plate_conf: float = 0.0
    plate_reads: list = field(default_factory=list)
    boxes: dict = field(default_factory=dict)   # picture -> its box


class RedLightAgent:
    """Feed it every picture (frame(k, t, rgb)); it returns the new cases (module doc). signals: the controller's
    SignalLog (cosim/enforcement.py); detector(rgb) -> [Det]; reader: a PlateReader or None."""

    def __init__(self, site: enf.Site, signals, detector, reader=None, grace_s: float = GRACE_S):
        self.site, self.signals, self.detect, self.reader = site, signals, detector, reader
        self.road = Road(site)
        self.cam = site.cam
        self.gz = site.cam.get("ground_z", 0.0)
        self.tracker = Tracker()
        self.grace_s = grace_s
        self.cases: list[Case] = []
        self.n_cases = 0
        self.crossings: list[dict] = []   # every vehicle seen crossing, red or not (for grading)

    def observe(self, k: int, t: float, dets: list[Det]) -> list[Obs]:
        obs = []
        for dt in dets:
            if dt.cls not in VEHICLE:
                continue
            x1, y1, x2, y2 = dt.box
            if y2 >= self.cam["height"] - 2:   # its back cut by the picture's edge: not placed
                continue
            g = enf.to_ground([((x1 + x2) / 2.0, y2)], self.cam, self.gz)[0]
            if np.isnan(g[0]):
                continue
            s, d = self.road.sd(g[0], g[1])
            if not -60.0 < s < 60.0 or not -6.0 < d < self.road.width + 6.0:
                continue
            obs.append(Obs(k, t, dt.box, dt.cls, dt.conf, s, d))
        return obs

    def frame(self, k: int, t: float, rgb: np.ndarray, dets: list[Det] | None = None) -> list[Case]:
        if dets is None:
            dets = self.detect(rgb)
        live = self.tracker.update(self.observe(k, t, dets))
        new = []
        for tr in self.tracker.tracks:
            if tr.case is not None or getattr(tr, "judged", False):
                continue
            c = self._judge(tr, t, final=tr.done)
            if c is not None:
                new.append(c)
        if self.reader is not None:
            for tr in live:
                self.reader.offer(tr, k, rgb)
        return new

    def _cross(self, tr: Track):
        """(time its front crossed the line, its speed m/s, the pictures before and after) from the two pictures
        either side; or, when the vehicle came into the picture with its front already over the line but its back
        not yet (the camera sees the road from a few metres before the line), from its first second in view at its
        speed then, at most EXTRAPOLATE_S back. None if it did not cross."""
        L = LENGTH_M.get(tr.kind(), 4.5)
        for a, b in zip(tr.obs, tr.obs[1:], strict=False):
            fa, fb = a.s + L, b.s + L
            if fa < 0.0 <= fb and b.t > a.t:
                v = (b.s - a.s) / (b.t - a.t)
                return a.t + (b.t - a.t) * (-fa) / (fb - fa), v, a, b
        first = tr.obs[0]
        if first.s + L >= 0.0 and first.s < 1.0:
            early = [o for o in tr.obs if o.t <= first.t + 1.0]
            if len(early) >= 3 and early[-1].t > first.t:
                ts = np.array([o.t for o in early])
                ss = np.array([o.s for o in early])
                v, _ = np.polyfit(ts - first.t, ss, 1)
                if v > 1.0:
                    back = (first.s + L) / v
                    if back <= EXTRAPOLATE_S:
                        return first.t - back, float(v), None, first
        return None

    def _judge(self, tr: Track, now: float, final: bool) -> Case | None:
        cr = self._cross(tr)
        if cr is None:
            if final:
                tr.judged = True
            return None
        t_c, v, a, b = cr
        a = a or b
        lane = self.road.lane((a.d + b.d) / 2.0) or self.road.lanes[0]
        links = self.site.links.get(lane, [])
        if not hasattr(tr, "logged"):
            red = self.signals.red_since(t_c, links)
            st = self.signals.state_at(t_c) or ""
            tr.logged = {"track": tr.id, "t_cross": round(t_c, 2), "lane": lane, "kind": tr.kind(),
                         "signal": "".join(st[i] for i in links if i < len(st)),
                         "red_s": None if red is None else round(red, 2), "speed_kmh": round(v * 3.6, 1)}
            self.crossings.append(tr.logged)
        red = tr.logged["red_s"]
        if red is None or red < self.grace_s:
            tr.judged = True
            return None
        # it went on into the junction?
        on = [o for o in tr.obs if o.t >= t_c and o.s >= CONFIRM_M]
        if not on:
            if final or now - t_c > CONFIRM_S:
                tr.judged = True
                tr.logged["outcome"] = "stopped past the line (not confirmed)"
            return None
        if tr.emergency_on():
            tr.judged = True
            tr.logged["outcome"] = "exempt: emergency vehicle with its lights on"
            return None
        before = [o for o in tr.obs if o.t < t_c] or [o for o in tr.obs if o.s < 0.0][:1]
        if not before:
            tr.judged = True
            tr.logged["outcome"] = "not seen at the line"
            return None
        self.n_cases += 1
        fa = before[-1]
        fb = on[0]
        clip = [o.k for o in tr.obs if t_c - 2.0 <= o.t <= t_c + 3.0]
        c = Case(id=f"{self.site.name}-{int(t_c * 10):07d}", site=self.site.name, track=tr.id, kind=tr.kind(),
                 lane=lane, t_cross=round(t_c, 2), red_s=round(red, 2), signal=tr.logged["signal"],
                 speed_kmh=round(v * 3.6, 1), frame_a=fa.k, frame_b=fb.k, frames=clip, plate_frames=[],
                 boxes={o.k: [round(x, 1) for x in o.box] for o in tr.obs})
        tr.case = c
        tr.logged["outcome"] = "case"
        tr.logged["case"] = c.id
        self.cases.append(c)
        return c


# ------------------------------------------------------------------ the plate
PLATE_S = (-14.0, 6.0)   # back of the vehicle this far from the line (m): close enough to the camera to read
PLATE_KEEP = 8           # the best views kept per vehicle
VOTE_MIN = 0.15          # a view's reading votes only when at least this sure


class PlateStore:
    """The plate camera's last seconds, kept as the JPEG it came as (decoded only when a plate is read from it)."""

    def __init__(self, keep_s: float = 15.0):
        self.keep_s = keep_s
        self.frames: dict[int, tuple] = {}   # picture -> (time, jpeg)

    def add(self, k: int, t: float, jpeg: bytes) -> None:
        self.frames[k] = (t, jpeg)
        for kk in [kk for kk, (tt, _) in self.frames.items() if tt < t - self.keep_s]:
            self.frames.pop(kk, None)

    def between(self, t0: float, t1: float) -> list[tuple[int, float]]:
        return sorted(((k, t) for k, (t, _) in self.frames.items() if t0 <= t <= t1), key=lambda x: x[1])

    def get(self, k: int):
        import cv2

        f = self.frames.get(k)
        if f is None:
            return None
        img = cv2.imdecode(np.frombuffer(f[1], np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else img[:, :, ::-1]


PLATE_CAM_S = (-9.0, 2.5)   # the back this far from the line: in the plate camera's picture
PLATE_CAM_VIEWS = 8         # its pictures read per vehicle
PLATE_W_M = 0.52            # a rear plate's width
_DEBUG_CHECKS = None        # (tools: a list here collects every check's measures)
PLATE_SIZE_OK = (0.85, 1.35)   # a plate read is a whole plate when its width is this share of the expected one
PLATE_WHERE_OK = (1.3, 1.2)    # ... and this vehicle's when its centre is within this many plate widths of where
# this vehicle's plate can be (across, up and down). Measured on recording 5 (1 Nov, 191 right readings): widths
# 0.93-1.23 of the expected, centres within 1.23 across and 0.79 up or down; a fragment read as a plate was 0.67 wide,
# another vehicle's plate 3.15 plate widths too high.


class PlateReader:
    """Reads each vehicle's plate (cosim/anpr.py), one reading per view; the plate is the reading the views agree on
    most, weighed by their confidence. The views: the plate camera's pictures of the vehicle's back (found in them
    from where the overview camera has the vehicle on the road at that moment: both cameras calibrated), and the
    overview camera's best views of its back (largest, whole in the picture, near the line)."""

    def __init__(self, site: enf.Site | None = None, store: PlateStore | None = None):
        self.views: dict[int, list] = {}   # track -> [(size, picture, crop)] (overview camera)
        self.site, self.pstore = site, store
        self.road = Road(site) if site is not None else None
        self.pboxes: dict[tuple, tuple] = {}   # (track, "p<picture>") -> its box in the plate camera's picture

    def expected(self, cam: dict, s_: float, d_: float) -> tuple[float, float, float] | None:
        """Where a plate on the back of a vehicle at (s, d) is in a camera's picture, and how wide it is there
        (centre x, centre y, width in pixels): the cameras' calibration."""
        gz = cam.get("ground_z", 0.0)
        pts = [[*self.road.xy(s_, d_ + dd), gz + enf.PLATE_Z_M] for dd in (0.0, -PLATE_W_M / 2, PLATE_W_M / 2)]
        px, depth = enf.project(np.array(pts), cam)
        if (depth <= 0).any():
            return None
        return float(px[0, 0]), float(px[0, 1]), float(abs(px[2, 0] - px[1, 0]))

    def checked(self, r, cam: dict, s_: float, d_: float, ox: int, oy: int) -> tuple:
        """(text, conf, box) of a reading, the text dropped (kept in the record as 'rejected: ...') when the plate it
        was read from is not the size a whole plate has at that distance (a fragment, or something else) or not
        where this vehicle's plate can be (another vehicle's)."""
        if not r.text or r.box is None:
            return r.text, r.conf, r.box
        exp = self.expected(cam, s_, d_)
        if exp is None:
            return r.text, r.conf, r.box
        cx, cy, w_exp = exp
        bx1, by1, bx2, by2 = r.box
        w = bx2 - bx1
        if _DEBUG_CHECKS is not None:
            _DEBUG_CHECKS.append((r.text, round(r.conf, 3), round(w / w_exp, 2),
                                  round(((bx1 + bx2) / 2 + ox - cx) / w_exp, 2), round(((by1 + by2) / 2 + oy - cy) / w_exp, 2),
                                  "plate" if cam is self.site.plate_cam else "overview"))
        if not PLATE_SIZE_OK[0] * w_exp <= w <= PLATE_SIZE_OK[1] * w_exp:
            return None, 0.0, r.box  # (its size: not a whole plate)
        dx = (bx1 + bx2) / 2 + ox - cx
        dy = (by1 + by2) / 2 + oy - cy
        if abs(dx) > PLATE_WHERE_OK[0] * w_exp or abs(dy) > PLATE_WHERE_OK[1] * w_exp:
            return None, 0.0, r.box  # (its place: another vehicle's plate)
        return r.text, r.conf, r.box

    def plate_cam_views(self, tr: Track) -> list[tuple[int, tuple, tuple]]:
        """(plate camera picture, the vehicle's back in it: x1, y1, x2, y2, its (s, d) then) for up to
        PLATE_CAM_VIEWS pictures."""
        if self.road is None or self.pstore is None or not self.site.plate_cam or len(tr.obs) < 2:
            return []
        cam = self.site.plate_cam
        gz = cam.get("ground_z", 0.0)
        ts = np.array([o.t for o in tr.obs])
        ss = np.array([o.s for o in tr.obs])
        ds = np.array([o.d for o in tr.obs])
        out = []
        for k, t in self.pstore.between(ts[0], ts[-1]):
            s_ = float(np.interp(t, ts, ss))
            d_ = float(np.interp(t, ts, ds))
            if not PLATE_CAM_S[0] <= s_ <= PLATE_CAM_S[1]:
                continue
            pts = []
            for ds_ in (-1.25, 1.25):                    # the back of the vehicle, with room for where it really is
                for sa in (-0.6, 0.6):
                    x, y = self.road.xy(s_ + sa, d_ + ds_)
                    pts += [[x, y, gz - 0.1], [x, y, gz + 1.7]]
            for sa in (-0.6, 0.6):                       # where the plate can be (0.5 m either side, 0.2-0.8 m up)
                for ds_ in (-0.5, 0.5):
                    x, y = self.road.xy(s_ + sa, d_ + ds_)
                    pts += [[x, y, gz + 0.2], [x, y, gz + 0.8]]
            px, depth = enf.project(np.array(pts), cam)
            if (depth <= 0).any():
                continue
            W, H = cam["width"], cam["height"]
            pl = px[8:]
            if pl[:, 0].min() < 0 or pl[:, 0].max() > W or pl[:, 1].min() < 0 or pl[:, 1].max() > H:
                continue                                 # (the plate's place not in the picture)
            x1, y1 = px[:8].min(0)
            x2, y2 = px[:8].max(0)
            out.append((k, (int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))), (s_, d_)))
        if len(out) > PLATE_CAM_VIEWS:                   # spread over its time in view
            idx = np.linspace(0, len(out) - 1, PLATE_CAM_VIEWS).round().astype(int)
            out = [out[i] for i in sorted(set(idx))]
        return out

    def offer(self, tr: Track, k: int, rgb) -> None:
        o = tr.last
        if o.k != k or not PLATE_S[0] <= o.s <= PLATE_S[1]:
            return
        x1, y1, x2, y2 = (int(round(v)) for v in o.box)
        h, w = rgb.shape[:2]
        if x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2:
            return
        size = (x2 - x1) * (y2 - y1)
        vs = self.views.setdefault(tr.id, [])
        vs.append((size, k, rgb[y1:y2, x1:x2].copy(), (o.s, o.d, x1, y1)))
        vs.sort(key=lambda v: -v[0])
        del vs[PLATE_KEEP:]

    def read(self, tr: Track) -> tuple[str | None, float, list, list]:
        """(plate, confidence, [(picture, text, conf)], [picture]) for a vehicle."""
        from cosim import anpr

        reads = []
        for k, (x1, y1, x2, y2), (s_, d_) in self.plate_cam_views(tr):
            img = self.pstore.get(k)
            if img is None:
                continue
            r = anpr.read_vehicle(img[y1:y2, x1:x2])
            self.pboxes[(tr.id, f"p{k}")] = (x1, y1, x2, y2)
            reads.append((f"p{k}", *self.checked(r, self.site.plate_cam, s_, d_, x1, y1)))
        for size, k, crop, (s_, d_, x1, y1) in self.views.get(tr.id, []):
            r = anpr.read_vehicle(crop)
            got = self.checked(r, self.site.cam, s_, d_, x1, y1) if self.road is not None else (r.text, r.conf, r.box)
            reads.append((k, *got))
        votes: dict[str, float] = {}
        count: dict[str, int] = {}
        for _, text, conf, _ in reads:
            if text and conf >= VOTE_MIN:   # (an unsure reading is shown to the officer but does not vote)
                votes[text] = votes.get(text, 0.0) + conf
                count[text] = count.get(text, 0) + 1
        order = [k for k, *_ in sorted(reads, key=lambda r: (-(r[2] or 0.0), 0 if str(r[0]).startswith("p") else 1))]
        if not votes:
            return None, 0.0, reads, order
        best = max(votes, key=votes.get)
        agree = votes[best] / sum(votes.values())
        conf = max(c for _, t, c, _ in reads if t == best)
        # separate views read the same whole plate: the errors of one view are rarely those of another
        conf = min(1.0, conf * agree + 0.15 * (count[best] - 1) * agree)
        order = [k for k, t, *_ in sorted(reads, key=lambda r: (r[1] != best, -(r[2] or 0.0)))]
        return best, round(conf, 3), reads, order

    def crop(self, tr: Track, k):
        """The view of the vehicle's back read as picture k (an overview picture, or "p<picture>": the plate
        camera's)."""
        if isinstance(k, str) and k.startswith("p"):
            box = self.pboxes.get((tr.id, k))
            img = self.pstore.get(int(k[1:])) if self.pstore is not None and box is not None else None
            if img is None:
                return None
            x1, y1, x2, y2 = box
            return img[y1:y2, x1:x2].copy()
        for _, kk, crop, _ in self.views.get(tr.id, []):
            if kk == k:
                return crop
        return None

    def forget(self, tid: int) -> None:
        self.views.pop(tid, None)
        for key in [key for key in self.pboxes if key[0] == tid]:
            self.pboxes.pop(key, None)


# ------------------------------------------------------------------ the evidence
class FrameStore:
    """The last seconds of pictures, kept as JPEG (for a case's photos and clip)."""

    def __init__(self, keep_s: float = 14.0, quality: int = 92):
        self.keep_s, self.quality = keep_s, quality
        self.frames: dict[int, tuple] = {}   # picture -> (time, jpeg bytes)

    def add(self, k: int, t: float, rgb) -> None:
        import cv2

        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self.frames[k] = (t, buf.tobytes())
        old = [kk for kk, (tt, _) in self.frames.items() if tt < t - self.keep_s]
        for kk in old:
            self.frames.pop(kk, None)

    def get(self, k: int):
        import cv2

        f = self.frames.get(k)
        if f is None:
            return None
        return cv2.imdecode(np.frombuffer(f[1], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]

    def latest_t(self) -> float:
        return max((t for t, _ in self.frames.values()), default=-1e9)


def sha256(path) -> str:
    import hashlib

    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class CaseWriter:
    """Writes each case's evidence to out_dir/<case id>/ (module doc 5): photo_a.jpg, photo_b.jpg (with the data bar
    real cameras print: place, time, signal, time into red, speed, lane), vehicle.jpg and plate.png (its back and its
    plate, enlarged), clip.mp4 (2 s before to 3 s after), case.json (everything, with each file's SHA-256 so that it
    can be shown not to have been changed). clock(t) -> the town's date and time for SUMO time t."""

    def __init__(self, out_dir, site: enf.Site, clock=None, fps: float = 10.0):
        from pathlib import Path

        self.out = Path(out_dir)
        self.site = site
        self.clock = clock or (lambda t: f"t = {t:.2f} s")
        self.fps = fps

    def _bar(self, img, lines):
        import cv2

        h, w = img.shape[:2]
        bar = 34 * len(lines) + 12
        out = np.vstack([img, np.full((bar, w, 3), 18, np.uint8)])
        for i, ln in enumerate(lines):
            cv2.putText(out, ln, (14, h + 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
        return out

    def _mark(self, img, box, colour=(255, 60, 40)):
        import cv2

        img = np.ascontiguousarray(img.copy())
        line = enf.project([[*p, self.site.cam.get("ground_z", 0.0)] for p in self.site.line], self.site.cam)[0]
        cv2.line(img, tuple(int(v) for v in line[0]), tuple(int(v) for v in line[1]), (255, 210, 0), 2, cv2.LINE_AA)
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
        return img

    def write(self, case: Case, tr: Track, store: FrameStore, reader: PlateReader | None, signals) -> dict:
        import json

        import cv2

        base, n = case.id, 2
        while (self.out / case.id / "case.json").exists():   # (an earlier run's case with the same time: kept)
            case.id = f"{base}-{n}"
            n += 1
        d = self.out / case.id
        d.mkdir(parents=True, exist_ok=True)
        plate, conf, reads, pframes = (reader.read(tr) if reader is not None else (None, 0.0, [], []))
        case.plate, case.plate_conf = plate, conf
        case.plate_reads = [{"picture": k, "text": t, "conf": c} for k, t, c, _ in reads]
        case.plate_frames = pframes
        when = self.clock(case.t_cross)
        files = {}
        for name, k, what in (("photo_a.jpg", case.frame_a, "A: before the stop line"),
                              ("photo_b.jpg", case.frame_b, "B: in the junction")):
            img = store.get(k)
            if img is None:
                continue
            t_k = next((o.t for o in tr.obs if o.k == k), case.t_cross)
            st = signals.state_at(t_k) or ""
            links = self.site.links.get(case.lane, [])
            sig = "".join(st[i] for i in links if i < len(st))
            red = signals.red_since(t_k, links)
            img = self._mark(img, case.boxes.get(k))
            img = self._bar(img, [f"{what}   site {self.site.name} lane {case.lane}   {self.clock(t_k)}",
                                  f"signal {sig.upper() or '?'}   red for {red:.2f} s" if red is not None
                                  else f"signal {sig.upper() or '?'}",
                                  f"front crossed the stop line {when}, {case.red_s:.2f} s into red, "
                                  f"{case.speed_kmh:.0f} km/h"])
            cv2.imwrite(str(d / name), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            files[name] = sha256(d / name)
        if pframes and reader is not None:
            crop = reader.crop(tr, pframes[0])
            if crop is not None:
                cv2.imwrite(str(d / "vehicle.jpg"), cv2.resize(crop, None, fx=2.0, fy=2.0)[:, :, ::-1])
                files["vehicle.jpg"] = sha256(d / "vehicle.jpg")
                box = next((b for k, _, _, b in reads if k == pframes[0] and b is not None), None)
                if box is not None:
                    x1, y1, x2, y2 = box
                    pl = crop[y1:y2, x1:x2]
                    if pl.size:
                        cv2.imwrite(str(d / "plate.png"),
                                    cv2.resize(pl, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)[:, :, ::-1])
                        files["plate.png"] = sha256(d / "plate.png")
        frames = [k for k in case.frames if store.get(k) is not None]
        if frames:
            import imageio_ffmpeg

            first = store.get(frames[0])
            h, w = first.shape[:2]
            wr = imageio_ffmpeg.write_frames(str(d / "clip.mp4"), (w, h), fps=self.fps, codec="libx264",
                                             pix_fmt_in="rgb24", macro_block_size=2,
                                             output_params=["-crf", "22", "-preset", "veryfast", "-pix_fmt", "yuv420p"])
            wr.send(None)
            for k in frames:
                wr.send(np.ascontiguousarray(self._mark(store.get(k), case.boxes.get(k))).tobytes())
            wr.close()
            files["clip.mp4"] = sha256(d / "clip.mp4")
        links = self.site.links.get(case.lane, [])
        around = [(t, s) for t, s in signals.changes(links) if case.t_cross - 120 <= t <= case.t_cross + 10]
        rec = {"case": dict(vars(case), boxes={str(k): v for k, v in case.boxes.items()}),
               "when": when, "site": self.site.to_json(), "rules": RULES, "grace_s": GRACE_S,
               "signal_log": [{"t": t, "state": s} for t, s in around], "files": files,
               "status": "pending review", "made_by": "cosim/redlight.py (automatic proposal: an officer decides)"}
        (d / "case.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return rec


# ------------------------------------------------------------------ the whole agent at one site
class SiteAgent:
    """Detector, tracker, rule, plate reader and evidence together: give it every picture, it writes each case to
    out_dir/<case id>/ once the seconds after it are in, and returns the case records written."""

    def __init__(self, site: enf.Site, signals, detector, out_dir, clock=None, read_all: bool = False):
        self.pstore = PlateStore()
        self.reader = PlateReader(site, self.pstore)
        self.agent = RedLightAgent(site, signals, detector, self.reader)
        self.store = FrameStore()
        self.writer = CaseWriter(out_dir, site, clock)
        self.signals = signals
        self.pending: list[Case] = []
        self.read_all = read_all          # read every crossing vehicle's plate too (for grading the reader)
        self.plates: dict[int, tuple] = {}
        self.finished: set[int] = set()   # tracks whose plate business is over

    def plate_picture(self, k: int, t: float, jpeg: bytes) -> None:
        """A picture from the plate camera (JPEG as it came)."""
        self.pstore.add(k, t, jpeg)

    def frame(self, k: int, t: float, rgb, dets=None) -> list[dict]:
        self.store.add(k, t, rgb)
        self.pending += self.agent.frame(k, t, rgb, dets)
        out = []
        tracks = {tr.id: tr for tr in self.agent.tracker.tracks}
        for c in list(self.pending):
            tr = tracks.get(c.track)
            if tr is None or t >= c.t_cross + 3.2 or tr.done:
                try:
                    out.append(self.writer.write(c, tr, self.store, self.reader, self.signals))
                except Exception:  # noqa: BLE001 - one case's trouble never stops the camera
                    log.exception("writing case %s", c.id)
                self.pending.remove(c)
        for tr in self.agent.tracker.tracks:
            if not tr.done or tr.id in self.finished:
                continue
            if tr.case is None:
                if self.read_all and getattr(tr, "logged", None):
                    p, conf, _, _ = self.reader.read(tr)
                    tr.logged["plate"], tr.logged["plate_conf"] = p, conf
                self.reader.forget(tr.id)
                self.finished.add(tr.id)
            elif all(c.track != tr.id for c in self.pending):
                if getattr(tr, "logged", None) is not None:
                    tr.logged["plate"], tr.logged["plate_conf"] = tr.case.plate, tr.case.plate_conf
                self.reader.forget(tr.id)
                self.finished.add(tr.id)
        # finished tracks are dropped (their crossings are logged)
        self.agent.tracker.tracks = [tr for tr in self.agent.tracker.tracks if not tr.done or t - tr.last.t < 10.0]
        return out
