"""Step 2 of the multi-agent plan: every traffic-light camera on Town05 sends a short report.

A report says, every `period` seconds (2 s by default), what one camera sees on the lanes it is
responsible for:

    {"cam": "1126-2", "tls": "1126", "t": 812.0, "frames": 4,
     "lanes": {"<lane id>": [vehicles, queued], ...},   # only the lanes with someone on them
     "in": 7, "q_in": 5, "out": 2,                      # totals on the lanes into / out of its junction
     "parked": 3,                                        # stationary off the road (parking lots, driveways)
     "other_road": 1,                                    # on a road lane this camera does not count
     "stopped_long": 0,                                  # on a counted lane, still for over 90 s
     "ems": [{"lane": "...", "votes": 4, "of": 8, "confirmed": true}]}  # sirens seen

Parked cars vs queued cars. Every camera has two masks drawn from the SUMO network with its own
pinhole model (cosim/perception.py): the lanes it counts (the lanes max-pressure reads, each
owned by the one camera that sees most of it), and every drivable lane in view. A detection's
ground point (bottom middle of its box) then says where the vehicle stands:
  * on a counted lane       -> counted; queued when it has not moved for `queue_s`;
  * on another road lane    -> `other_road` (moving traffic elsewhere, never a queue);
  * off every road          -> `parked` once still for `park_s` (a parking lot is off the road:
                               Town05's SUMO network has no parking lanes), else ignored.
So a full parking lot next to a junction never adds pressure, whatever the detector says about
those cars. What the masks cannot know: buildings and trees hiding part of a lane (the masks are
drawn from geometry alone; `coverage` is therefore an upper bound), and a car parked on a lane
itself (reported as `stopped_long`, left for the junction agent to judge).

Emergency vehicles. The detector's siren-on classes are voted over the last `vote_frames`
frames of each tracked vehicle; `confirmed` needs `vote_need` of them, and then stays on while
the camera keeps tracking that vehicle (a vote of 3 in 8 at 34 % a picture would otherwise blink
on and off and one ambulance would look like several). Measured on Town05's
held-out junctions (out/detector/eval_town05.md, carla-yolo26n-t05): a siren-on vehicle is
found as siren-on in 34 % of single pictures, and a siren-off emergency vehicle is called
siren-on in 9 %. One picture is therefore not enough either way; the vote, and in the plan the
vehicle's own radio request (cosim.controllers.PriorityMaxPressure), decide.

Two sources of reports, same format:
  * `CameraReporter`: from detections on real CARLA images (tools/town05_cameras.py live);
  * `SimulatedCameras`: from SUMO's ground truth, only on the visible part of each owned lane,
    with the detector's measured miss and false-box rates. Running YOLO on all 53 cameras live
    is out of reach of one GTX 1650; this is what lets the junction agents be developed and
    compared on many seeds in SUMO alone.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cosim.perception import (
    MAX_RANGE_M,
    QUAD_M,
    CamSpec,
    LaneCamera,
    build_masks,
    ground_point_lane,
    plan_cameras,
    project,
)

SIREN = {"ambulance_lights", "police_lights"}
VEHICLES = {"car", "van", "bus", "truck", "ambulance", "ambulance_lights", "police", "police_lights"}
MAST = {"back_m": -10.0, "pitch": -22.0}  # the inspector's cameras (cosim/inspector.py)
RES, FOV = (960, 540), 75.0

# Detector error model. Counts: measured by tools/town05_cameras.py live on the held-out junctions
# 1126, 1882, 2296 (10 cameras, 3910 reports, x0.5, seed 1, 21 Oct): 2366 vehicles in view, 175 not
# reported, 204 reported that were not there; so 93 % found and 0.05 extra per picture. That is on
# the counted lanes only, within 90 m. Over whole pictures, every distance, the detector finds far less
# (5092 of 7984 = 64 %, 23 false boxes per 100 pictures: out/detector/eval_town05.md), mostly far away.
RECALL = 1 - 175 / 2366
FALSE_BOXES_PER_FRAME = 204 / 3910
SIREN_RECALL = 174 / 517          # siren-on found as siren-on, per picture
SIREN_FALSE = 36 / 410            # siren-off emergency vehicle called siren-on, per picture


def compass(yaw: float) -> str:
    """CARLA yaw (0 = +x, clockwise seen from above; +x east, +y south) -> N, NE, ..."""
    return ["E", "SE", "S", "SW", "W", "NW", "N", "NE"][int(((yaw % 360) + 22.5) // 45) % 8]


# ----------------------------------------------------------------------------- layout


@dataclass
class Layout:
    """The 53 cameras, what each counts, and where each lane is visible."""

    cams: list[LaneCamera]
    names: list[str]
    lanes: list[str]                              # every lane max-pressure reads (vehicle lanes)
    owner: dict[str, int]                         # lane -> camera index
    roles: dict[str, list[tuple[str, str]]]       # lane -> [(tls, "in" | "out")]
    visible: dict[str, list[tuple[float, float]]]  # lane -> visible stretches (from, to) along it, metres
    lengths: dict[str, float]
    road: list[np.ndarray] = field(default_factory=list)  # per camera (H, W) bool: any drivable lane

    def owned(self, c: int) -> list[str]:
        return [ln for ln, k in self.owner.items() if k == c]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"res": [self.cams[0].width, self.cams[0].height], "fov": self.cams[0].fov,
                "cameras": [dict(vars(c.spec), name=n) for c, n in zip(self.cams, self.names, strict=True)],
                "lanes": self.lanes, "owner": self.owner, "roles": self.roles,
                "visible": self.visible, "lengths": self.lengths}
        (folder / "layout.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        np.savez_compressed(folder / "masks.npz",
                            **{f"lane{i}": c.mask.astype(np.int16) for i, c in enumerate(self.cams)},
                            **{f"road{i}": r for i, r in enumerate(self.road)})

    @classmethod
    def load(cls, folder: Path) -> Layout:
        meta = json.loads((folder / "layout.json").read_text(encoding="utf-8"))
        w, h = meta["res"]
        cams = []
        npz = np.load(folder / "masks.npz")
        road = []
        for i, c in enumerate(meta["cameras"]):
            spec = CamSpec(**{k: v for k, v in c.items() if k != "name"})
            cam = LaneCamera(spec, w, h, meta["fov"], mask=npz[f"lane{i}"].astype(np.int32), lanes=meta["lanes"])
            cams.append(cam)
            road.append(npz[f"road{i}"])
        return cls(cams, [c["name"] for c in meta["cameras"]], meta["lanes"], meta["owner"],
                   {k: [tuple(x) for x in v] for k, v in meta["roles"].items()},
                   {k: [tuple(x) for x in v] for k, v in meta["visible"].items()}, meta["lengths"], road)


def vehicle_lanes(net) -> set[str]:
    return {ln.getID() for e in net.getEdges(withInternal=True) for ln in e.getLanes()
            if ln.allows("passenger") or ln.allows("bus") or ln.allows("truck")}


def lane_roles(policy) -> dict[str, list[tuple[str, str]]]:
    """lane -> [(tls, "in" | "out")] from a set-up max-pressure controller's own reach."""
    roles: dict[str, set[tuple[str, str]]] = {}
    for j in policy.junctions.values():
        for pairs in j.pairs:
            for a, b in pairs:
                for x in policy._reach(a, upstream=True):
                    roles.setdefault(x, set()).add((j.tls, "in"))
                for x in policy._reach(b, upstream=False):
                    roles.setdefault(x, set()).add((j.tls, "out"))
    return {k: sorted(v) for k, v in roles.items()}


def lane_pieces(net, lane_id: str, frame, ground_z) -> list[tuple[float, float, np.ndarray]]:
    """(from, to along the lane in metres, (4, 3) CARLA corners) for every QUAD_M piece."""
    lane = net.getLane(lane_id)
    shape, half = lane.getShape(), lane.getWidth() / 2.0
    pts, s = [], [0.0]
    for (x0, y0), (x1, y1) in zip(shape, shape[1:], strict=False):
        seg = math.hypot(x1 - x0, y1 - y0)
        k = max(1, int(math.ceil(seg / QUAD_M)))
        for j in range(k):
            pts.append((x0 + (x1 - x0) * j / k, y0 + (y1 - y0) * j / k))
    pts.append(shape[-1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        s.append(s[-1] + math.hypot(x1 - x0, y1 - y0))
    out = []
    for n, ((x0, y0), (x1, y1)) in enumerate(zip(pts, pts[1:], strict=False)):
        seg = s[n + 1] - s[n]
        if seg < 1e-6:
            continue
        nx, ny = -(y1 - y0) / seg * half, (x1 - x0) / seg * half
        corners = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
        cc = [frame.to_carla_xy(x, y) for x, y in corners]
        z = ground_z(*frame.to_carla_xy((x0 + x1) / 2, (y0 + y1) / 2))
        out.append((s[n], s[n + 1], np.array([[x, y, z] for x, y in cc])))
    return out


def _in_view(corners: np.ndarray, spec: CamSpec, w: int, h: int, fov: float) -> bool:
    px, depth = project(corners, spec, w, h, fov)
    if (depth < 1.0).any() or depth.mean() > MAX_RANGE_M:
        return False
    cx, cy = px[:, 0].mean(), px[:, 1].mean()
    return 0 <= cx < w and 0 <= cy < h


def build_layout(net, frame, policy, ground_z, res=RES, fov=FOV) -> Layout:
    """net: sumolib net read with internal lanes; policy: a max-pressure controller already set
    up on this network (its reach decides which lanes are counted)."""
    from PIL import Image, ImageDraw

    w, h = res
    specs = plan_cameras(net, frame, ground_z, MAST["back_m"], MAST["pitch"])
    cams = [LaneCamera(s, w, h, fov) for s in specs]
    names, per_tls = [], {}
    for c in specs:
        per_tls[c.tls] = per_tls.get(c.tls, 0) + 1
        names.append(f"{c.tls}-{per_tls[c.tls]} looking {compass(c.yaw)}")
    veh = vehicle_lanes(net)
    roles = {k: v for k, v in lane_roles(policy).items() if k in veh}
    lanes = sorted(roles)
    pieces = {ln: lane_pieces(net, ln, frame, ground_z) for ln in lanes}
    quads = [(i, corners) for i, ln in enumerate(lanes) for _, _, corners in pieces[ln]]
    owner = build_masks(cams, lanes, quads)
    visible, lengths = {}, {}
    for ln in lanes:
        lengths[ln] = round(net.getLane(ln).getLength(), 2)
        if ln not in owner:
            continue
        spec = cams[owner[ln]].spec
        runs: list[list[float]] = []
        for s0, s1, corners in pieces[ln]:
            if _in_view(corners, spec, w, h, fov):
                if runs and abs(runs[-1][1] - s0) < 1e-3:
                    runs[-1][1] = s1
                else:
                    runs.append([s0, s1])
        visible[ln] = [(round(a, 2), round(b, 2)) for a, b in runs]
    # every drivable lane in view, for telling "on another road" from "off the road"
    road = []
    all_lanes = sorted(veh)
    centre = {ln: frame.to_carla_xy(*net.getLane(ln).getShape()[len(net.getLane(ln).getShape()) // 2]) for ln in all_lanes}
    for cam in cams:
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        s = cam.spec
        for ln in all_lanes:
            if math.hypot(centre[ln][0] - s.x, centre[ln][1] - s.y) > MAX_RANGE_M + net.getLane(ln).getLength():
                continue
            for _, _, corners in lane_pieces(net, ln, frame, ground_z):
                px, depth = project(corners, s, w, h, fov)
                if (depth < 1.0).any() or depth.mean() > MAX_RANGE_M * 1.5:
                    continue
                if (px[:, 0] < -w).all() or (px[:, 0] > 2 * w).all() or (px[:, 1] > 2 * h).all():
                    continue
                draw.polygon([tuple(p) for p in px], fill=1)
        road.append(np.array(img, dtype=bool))
    return Layout(cams, names, lanes, owner, roles, visible, lengths, road)


def zones(policy, reach_m: float | None = None) -> dict[str, dict[str, dict[str, list[tuple[float, float]]]]]:
    """tls -> "in" | "out" -> lane -> stretches (from, to) within reach_m of the junction: the
    last reach_m before each stop line (queues stand there) and the first reach_m after it."""
    reach = reach_m or policy.reach_m
    out: dict = {}
    for j in policy.junctions.values():
        z = out.setdefault(j.tls, {"in": {}, "out": {}})
        for pairs in j.pairs:
            for a, b in pairs:
                for role, start, nbrs in (("in", a, policy._pred), ("out", b, policy._succ)):
                    frontier, seen = [(start, 0.0)], {start}
                    while frontier:
                        ln, covered = frontier.pop()
                        length = policy._len.get(ln, 0.0)
                        take = min(length, reach - covered)
                        if take <= 0:
                            continue
                        iv = (length - take, length) if role == "in" else (0.0, take)
                        z[role].setdefault(ln, []).append(iv)
                        for n in nbrs.get(ln, []):
                            if n not in seen:
                                seen.add(n)
                                frontier.append((n, covered + length))
    return out


def _union(ivs):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _overlap(ivs, vis) -> float:
    return sum(max(0.0, min(b, d) - max(a, c)) for a, b in ivs for c, d in vis)


def coverage(layout: Layout, policy) -> dict:
    """Per junction: how many metres of the stretches max-pressure cares about (reach_m before
    each stop line and after the junction) lie in a camera's picture. Vehicle lanes only.
    Geometry only (no buildings or trees), so an upper bound."""
    veh = set(layout.lanes)
    per = {}
    tot = {"in": [0.0, 0.0], "out": [0.0, 0.0]}
    for tls, z in zones(policy).items():
        acc = {"in": [0.0, 0.0], "out": [0.0, 0.0]}
        poor = set()
        for role in ("in", "out"):
            for ln, ivs in z[role].items():
                if ln not in veh:
                    continue
                u = _union(ivs)
                total = sum(b - a for a, b in u)
                seen = _overlap(u, layout.visible.get(ln, []))
                acc[role][0] += seen
                acc[role][1] += total
                if role == "in" and not ln.startswith(":") and total > 5.0 and seen < 0.5 * total:
                    poor.add(ln)
        for role in ("in", "out"):
            tot[role][0] += acc[role][0]
            tot[role][1] += acc[role][1]
        per[tls] = {"in_seen_m": round(acc["in"][0]), "in_total_m": round(acc["in"][1]),
                    "in_share": round(acc["in"][0] / acc["in"][1], 3) if acc["in"][1] else None,
                    "out_seen_m": round(acc["out"][0]), "out_total_m": round(acc["out"][1]),
                    "out_share": round(acc["out"][0] / acc["out"][1], 3) if acc["out"][1] else None,
                    "cameras": sum(1 for c in layout.cams if c.spec.tls == tls),
                    "incoming_lanes_under_half_seen": sorted(poor)}
    return {"cameras": len(layout.cams), "lanes_counted": len(layout.lanes), "lanes_owned": len(layout.owner),
            "reach_m": policy.reach_m,
            "in_share": round(tot["in"][0] / tot["in"][1], 3) if tot["in"][1] else None,
            "out_share": round(tot["out"][0] / tot["out"][1], 3) if tot["out"][1] else None, "junctions": per}


# ----------------------------------------------------------------------------- tracking


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    first: float
    last: float
    anchor: tuple[float, float]
    still_since: float
    names: deque = field(default_factory=lambda: deque(maxlen=8))
    siren: bool = False  # siren confirmed once on this track: kept until the track is lost

    @property
    def ground(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2.0, self.box[3]

    def still_for(self, t: float) -> float:
        return t - self.still_since


class Tracker:
    """Greedy IoU tracker for one fixed camera. A vehicle is "still" while its ground point
    stays within max(4 px, 6 % of its box height) of where it stopped."""

    def __init__(self, min_iou: float = 0.3, max_gap_s: float = 1.5, vote_frames: int = 8):
        self.min_iou, self.max_gap_s, self.vote_frames = min_iou, max_gap_s, vote_frames
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, t: float, boxes, names) -> list[Track]:
        dets = [(tuple(float(v) for v in b), n) for b, n in zip(boxes, names, strict=True) if n in VEHICLES]
        pairs = sorted(((iou(tr.box, d[0]), ti, di) for ti, tr in enumerate(self.tracks)
                        for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.min_iou:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self.tracks[ti]
            tr.box, tr.last = dets[di][0], t
            tr.names.append(dets[di][1])
            gx, gy = tr.ground
            tol = max(4.0, 0.06 * (tr.box[3] - tr.box[1]))
            if math.hypot(gx - tr.anchor[0], gy - tr.anchor[1]) > tol:
                tr.anchor, tr.still_since = (gx, gy), t
        for di, (box, name) in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(self._next, box, t, t, ((box[0] + box[2]) / 2.0, box[3]), t,
                       deque([name], maxlen=self.vote_frames))
            self._next += 1
            self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks if t - tr.last <= self.max_gap_s]
        return [tr for tr in self.tracks if tr.last == t]


def _on_mask(mask: np.ndarray, box) -> bool:
    """Like perception.ground_point_lane, for a boolean mask."""
    x1, y1, x2, y2 = box
    h, w = mask.shape
    cx = int(min(max((x1 + x2) / 2, 0), w - 1))
    bottom = int(min(max(y2 - 1, 0), h - 1))
    top = int(max(bottom - (y2 - y1) / 3, 0))
    return bool(mask[top:bottom + 1, cx].any())


@dataclass
class ReportRules:
    period_s: float = 2.0
    queue_s: float = 1.0        # still this long on a counted lane: queued
    park_s: float = 10.0        # still this long off the road: parked
    long_s: float = 90.0        # still this long on a counted lane: stopped_long
    vote_frames: int = 8
    vote_need: int = 3


class CameraReporter:
    """Reports of one camera from its detections (boxes in the mask's pixels, class names)."""

    def __init__(self, layout: Layout, c: int, rules: ReportRules | None = None):
        self.layout, self.c = layout, c
        self.cam = layout.cams[c]
        self.road = layout.road[c] if layout.road else np.zeros((self.cam.height, self.cam.width), bool)
        self.rules = rules or ReportRules()
        self.tracker = Tracker(vote_frames=self.rules.vote_frames)
        self.frames = 0

    def observe(self, t: float, boxes, names) -> None:
        self._last_t = t
        for tr in self.tracker.update(t, boxes, names):
            if not tr.siren and sum(1 for x in tr.names if x in SIREN) >= self.rules.vote_need:
                tr.siren = True
        self.frames += 1

    def classify(self, tr: Track, t: float) -> tuple[str, str | None]:
        v = ground_point_lane(self.cam.mask, tr.box)
        if v:
            return "lane", self.cam.lanes[v - 1]
        if _on_mask(self.road, tr.box):
            return "other_road", None
        return ("parked" if tr.still_for(t) >= self.rules.park_s else "off_road"), None

    def report(self, t: float) -> dict:
        r = self.rules
        lanes: dict[str, list[int]] = {}
        out = {"cam": self.layout.names[self.c].split(" ")[0], "tls": self.cam.spec.tls, "t": round(t, 1),
               "frames": self.frames, "in": 0, "q_in": 0, "out": 0, "parked": 0, "other_road": 0,
               "stopped_long": 0, "ems": []}
        for tr in self.tracker.tracks:
            # only what the latest picture shows: a track lost a moment ago lives on for max_gap_s, and
            # counting it too counted a vehicle twice whenever the tracker gave it a new id (first live
            # run, 21 Oct: 1.8 vehicles reported for every one there)
            if tr.last < getattr(self, "_last_t", t) - 1e-6:
                continue
            kind, lane = self.classify(tr, t)
            if kind in ("parked", "other_road"):
                out[kind] += 1
                continue
            if kind != "lane":
                continue
            still = tr.still_for(t)
            n = lanes.setdefault(lane, [0, 0])
            n[0] += 1
            if still >= r.queue_s:
                n[1] += 1
            if still >= r.long_s:
                out["stopped_long"] += 1
            votes = sum(1 for x in tr.names if x in SIREN)
            if votes or tr.siren:
                out["ems"].append({"lane": lane, "votes": votes, "of": len(tr.names), "confirmed": tr.siren})
        _totals(out, lanes, self.layout, self.cam.spec.tls)
        self.frames = 0
        return out


def _totals(out: dict, lanes: dict[str, list[int]], layout: Layout, tls: str) -> None:
    for ln, (n, q) in lanes.items():
        roles = {role for t, role in layout.roles.get(ln, []) if t == tls}
        if "in" in roles:
            out["in"] += n
            out["q_in"] += q
        elif "out" in roles:
            out["out"] += n
    out["lanes"] = {ln: v for ln, v in sorted(lanes.items()) if v[0]}


def in_view(layout: Layout, lane: str, pos: float, length: float) -> bool:
    """Is any part of a vehicle (front at `pos` along the lane, `length` long) on a visible stretch?
    The detector finds a car that is half in the picture; the truth has to count it too."""
    return any(pos >= a - 1.0 and pos - length <= b + 1.0 for a, b in layout.visible.get(lane, ()))


def observed(reports: list[dict]) -> dict[str, int]:
    """lane -> vehicles, for cosim.controllers.ObservedMaxPressure."""
    out: dict[str, int] = {}
    for rep in reports:
        for ln, (n, _) in rep.get("lanes", {}).items():
            out[ln] = out.get(ln, 0) + n
    return out


# ----------------------------------------------------------------------------- simulated


class SimulatedCameras:
    """Reports of every camera from SUMO's ground truth, as the detector would give them: only
    vehicles on the visible stretches of each camera's own lanes, each found with probability
    `recall` per picture, plus `false_per_frame` false boxes per picture (placed on one of its
    lanes), siren votes at the measured per-picture rates. Motorcycles are left out: the
    detector has no class for them. `parked`: camera name -> parked cars it sees (static props,
    measured in CARLA by tools/town05_cameras.py live; 0 when unknown)."""

    def __init__(self, layout: Layout, fps: float = 2.0, rules: ReportRules | None = None, seed: int = 1,
                 recall: float = RECALL, false_per_frame: float = FALSE_BOXES_PER_FRAME,
                 siren_recall: float = SIREN_RECALL, siren_false: float = SIREN_FALSE,
                 parked: dict[str, int] | None = None):
        self.layout, self.fps = layout, fps
        self.rules = rules or ReportRules()
        self.rng = random.Random(seed)
        self.recall, self.false_per_frame = recall, false_per_frame
        self.siren_recall, self.siren_false = siren_recall, siren_false
        self.parked = parked or {}
        self.by_cam = [layout.owned(c) for c in range(len(layout.cams))]
        self.dropped_lanes: list[str] = []  # in the layout, not in the network (frame)
        self._acc = [dict() for _ in layout.cams]  # lane -> [sum seen, sum queued]
        self._votes: dict[str, deque] = {}
        self._confirmed: set[str] = set()
        self._ems_lane: dict[str, tuple[int, str]] = {}
        self._moto: dict[str, bool] = {}
        self._lengths: dict[str, float] = {}
        self.frames = 0
        self._last_frame = -1e9
        self._last_report = -1e9

    def _visible(self, lane: str, pos: float, length: float = 4.5) -> bool:
        return in_view(self.layout, lane, pos, length)

    def _len(self, tc, vid: str) -> float:
        if vid not in self._lengths:
            self._lengths[vid] = tc.vehicle.getLength(vid)
        return self._lengths[vid]

    def frame(self, tc, t: float, emergency: dict[str, bool]) -> None:
        """One picture from every camera. emergency: vehicle id -> on a run (siren on)."""
        self.frames += 1
        seen_ems = set()
        for c, lanes in enumerate(self.by_cam):
            acc = self._acc[c]
            for ln in list(lanes):
                n = q = 0
                try:
                    ids = tc.lane.getLastStepVehicleIDs(ln)
                except Exception:  # noqa: BLE001 - a lane the layout has and the network no longer has: since
                    lanes.remove(ln)  # 27 Oct the parking lanes' movements at the signals are gone
                    self.dropped_lanes.append(ln)
                    continue
                for vid in ids:
                    moto = self._moto.get(vid)
                    if moto is None:
                        moto = self._moto[vid] = tc.vehicle.getVehicleClass(vid) == "motorcycle"
                    if moto:
                        continue
                    if not self._visible(ln, tc.vehicle.getLanePosition(vid), self._len(tc, vid)):
                        continue
                    if vid in emergency:
                        p = self.siren_recall if emergency[vid] else self.siren_false
                        v = self._votes.setdefault(vid, deque(maxlen=self.rules.vote_frames))
                        v.append(self.rng.random() < p)
                        if sum(v) >= self.rules.vote_need:
                            self._confirmed.add(vid)
                        self._ems_lane[vid] = (c, ln)
                        seen_ems.add(vid)
                    if self.rng.random() >= self.recall:
                        continue
                    n += 1
                    q += tc.vehicle.getSpeed(vid) < 0.1
                a = acc.setdefault(ln, [0, 0])
                a[0] += n
                a[1] += q
            k = _poisson(self.rng, self.false_per_frame)
            for _ in range(k):
                if lanes:
                    acc.setdefault(self.rng.choice(lanes), [0, 0])[0] += 1
        for vid in list(self._votes):
            if vid not in seen_ems:
                self._votes.pop(vid)
                self._confirmed.discard(vid)
                self._ems_lane.pop(vid, None)

    def step(self, tc, t: float, emergency: dict[str, bool]) -> list[dict] | None:
        """Call every SUMO step: takes the pictures when due, returns the reports when due."""
        if t - self._last_frame >= 1.0 / self.fps - 1e-6:
            self._last_frame = t
            self.frame(tc, t, emergency)
        if t - self._last_report >= self.rules.period_s - 1e-6 and self.frames:
            self._last_report = t
            return self.reports(t)
        return None

    def reports(self, t: float) -> list[dict]:
        out = []
        f = max(self.frames, 1)
        for c, cam in enumerate(self.layout.cams):
            name = self.layout.names[c].split(" ")[0]
            lanes = {ln: [int(round(s / f)), int(round(q / f))] for ln, (s, q) in self._acc[c].items()}
            rep = {"cam": name, "tls": cam.spec.tls, "t": round(t, 1), "frames": self.frames, "in": 0, "q_in": 0,
                   "out": 0, "parked": int(self.parked.get(name, 0)), "other_road": 0, "stopped_long": 0, "ems": []}
            for vid, votes in self._votes.items():
                if self._ems_lane.get(vid, (None,))[0] == c and (sum(votes) or vid in self._confirmed):
                    rep["ems"].append({"lane": self._ems_lane[vid][1], "votes": int(sum(votes)), "of": len(votes),
                                       "confirmed": vid in self._confirmed})
            _totals(rep, lanes, self.layout, cam.spec.tls)
            out.append(rep)
            self._acc[c] = {}
        self.frames = 0
        return out


def _poisson(rng: random.Random, lam: float) -> int:
    k, p, limit = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def junction_summary(reports: list[dict]) -> dict[str, dict]:
    """The reports of one moment folded per junction: what a junction agent (step 3) reads."""
    out: dict[str, dict] = {}
    for r in reports:
        j = out.setdefault(r["tls"], {"in": 0, "q_in": 0, "out": 0, "parked": 0, "ems_confirmed": 0, "cams": 0})
        j["cams"] += 1
        for k in ("in", "q_in", "out", "parked"):
            j[k] += r[k]
        j["ems_confirmed"] += sum(1 for e in r["ems"] if e["confirmed"])
    return out
