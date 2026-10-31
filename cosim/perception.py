"""Counting vehicles per lane from CARLA cameras, for signal control from what a camera sees.

The signal controllers otherwise read exact counts from SUMO. Here, each signalised junction
gets one camera per road, mounted 9 m up above the stop line and looking out along the road
(so it sees that road's queue coming in and the traffic leaving). Every lane the controller
counts is drawn into the camera images once, as a lane-id mask (`LaneCamera.mask`), by
projecting the SUMO lane shapes with the camera's own pinhole model; a lane belongs to the one
camera that sees most of it, so no vehicle is counted twice. A detection's ground point (the
bottom middle of its box) then gives its lane.

Geometry (CARLA / Unreal, left-handed): camera x forward, y right, z up; a world point P seen
from camera C with yaw/pitch: depth = (P-C).f, right = (P-C).r, up = (P-C).u, pixel
u = W/2 + F*right/depth, v = H/2 - F*up/depth, F = W / (2 tan(fov/2)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cosim.mapping import Frame

CAM_HEIGHT_M = 9.0
CAM_PITCH_DEG = -14.0
CAM_BACK_M = 1.0  # behind the stop line, over the road
MAX_RANGE_M = 90.0
QUAD_M = 2.0  # lane shapes are cut into pieces this long for drawing
VEHICLE_CLASSES = {"car", "van", "bus", "truck", "ambulance", "ambulance_lights", "police", "police_lights"}


@dataclass
class CamSpec:
    tls: str
    node: str
    road_to: str  # the node at the other end of the road
    x: float  # CARLA coordinates
    y: float
    z: float
    yaw: float
    pitch: float = CAM_PITCH_DEG


@dataclass
class LaneCamera:
    spec: CamSpec
    width: int
    height: int
    fov: float
    mask: np.ndarray | None = None  # (H, W) int32: 0 = no counted lane, else index+1 into lanes
    lanes: list[str] = field(default_factory=list)


def basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    r = np.array([-math.sin(y), math.cos(y), 0.0])
    u = np.array([-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), math.cos(p)])
    return f, r, u


def project(points: np.ndarray, cam: CamSpec, width: int, height: int, fov: float) -> tuple[np.ndarray, np.ndarray]:
    """points (N, 3) CARLA -> pixels (N, 2) and depth (N,)."""
    f, r, u = basis(cam.yaw, cam.pitch)
    d = points - np.array([cam.x, cam.y, cam.z])
    depth, right, up = d @ f, d @ r, d @ u
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.stack([width / 2 + focal * right / depth, height / 2 - focal * up / depth], axis=1)
    return px, depth


def _point_along(shape, dist: float, from_end: bool) -> tuple[float, float]:
    pts = list(reversed(shape)) if from_end else list(shape)
    left = dist
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg >= left and seg > 0:
            t = left / seg
            return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        left -= seg
    return pts[-1]


def plan_cameras(net, frame: Frame, ground_z, back_m: float = CAM_BACK_M, pitch: float = CAM_PITCH_DEG,
                 height: float = CAM_HEIGHT_M) -> list[CamSpec]:
    """One camera per road at each signalised junction, looking along the road away from the
    junction. net: sumolib net; ground_z(x, y) -> CARLA ground height at a CARLA point.
    back_m: where along the road from the junction's edge (negative: into the junction, as a
    camera on a mast arm across the junction would stand); pitch, height: its view."""
    cams = []
    for t in net.getTrafficLights():
        nodes = {c[0].getEdge().getToNode().getID() for c in t.getConnections()}
        for nid in sorted(nodes):
            node = net.getNode(nid)
            # a road: the other end's node and the direction it leaves this junction in. Both carriageways of the
            # ring road lead from 2014 to 2328 (it is a loop), one each way: keyed by the node alone they were one
            # road and one of them had no camera (27 Oct)
            roads: dict[tuple[str, int], tuple[object, bool]] = {}

            def out_angle(e, incoming: bool) -> float:
                shape = e.getShape()
                end = shape[-1] if incoming else shape[0]
                ahead = _point_along(shape, min(10.0, e.getLength() / 2), from_end=incoming)
                return math.degrees(math.atan2(ahead[1] - end[1], ahead[0] - end[0]))

            def key_of(other: str, ang: float) -> tuple[str, int]:
                for (o, a0) in roads:
                    if o == other and abs((ang - a0 + 180.0) % 360.0 - 180.0) < 120.0:  # same road unless opposite
                        return (o, a0)
                return (other, round(ang))

            for e in node.getIncoming():
                if e.getFunction() != "internal" and e.getFromNode().getID() not in nodes:
                    k = key_of(e.getFromNode().getID(), out_angle(e, True))
                    roads[k] = (e, True)
            for e in node.getOutgoing():
                other = e.getToNode().getID()
                if e.getFunction() != "internal" and other not in nodes:
                    k = key_of(other, out_angle(e, False))
                    roads.setdefault(k, (e, False))
            for (other, _a), (e, incoming) in sorted(roads.items()):
                shape = e.getShape()
                end = shape[-1] if incoming else shape[0]
                ahead = _point_along(shape, min(10.0, e.getLength() / 2), from_end=incoming)
                dx, dy = ahead[0] - end[0], ahead[1] - end[1]
                n = math.hypot(dx, dy) or 1.0
                sx, sy = end[0] + dx / n * back_m, end[1] + dy / n * back_m
                cx, cy = frame.to_carla_xy(sx, sy)
                ax, ay = frame.to_carla_xy(sx + dx / n, sy + dy / n)
                yaw = math.degrees(math.atan2(ay - cy, ax - cx))
                cams.append(CamSpec(t.getID(), nid, other, cx, cy, ground_z(cx, cy) + height, yaw, pitch))
    return cams


def lane_quads(net, lanes: list[str], frame: Frame, ground_z) -> list[tuple[int, np.ndarray]]:
    """(lane index, (4, 3) CARLA corners) for every QUAD_M piece of the given lanes."""
    out = []
    for i, lid in enumerate(lanes):
        lane = net.getLane(lid)
        shape, half = lane.getShape(), lane.getWidth() / 2.0
        pts = []
        for (x0, y0), (x1, y1) in zip(shape, shape[1:], strict=False):
            seg = math.hypot(x1 - x0, y1 - y0)
            k = max(1, int(math.ceil(seg / QUAD_M)))
            pts += [(x0 + (x1 - x0) * j / k, y0 + (y1 - y0) * j / k) for j in range(k)]
        pts.append(shape[-1])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-6:
                continue
            nx, ny = -(y1 - y0) / seg * half, (x1 - x0) / seg * half
            corners = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
            cc = [frame.to_carla_xy(x, y) for x, y in corners]
            mx, my = frame.to_carla_xy((x0 + x1) / 2, (y0 + y1) / 2)
            z = ground_z(mx, my)
            out.append((i, np.array([[x, y, z] for x, y in cc])))
    return out


def build_masks(cams: list[LaneCamera], lanes: list[str], quads) -> dict[str, int]:
    """Draw the lanes into every camera, then keep each lane only in the camera that sees most
    of it. Returns lane -> owning camera index (lanes no camera sees are left out)."""
    from PIL import Image, ImageDraw

    full = []
    for cam in cams:
        img = Image.new("I", (cam.width, cam.height), 0)
        draw = ImageDraw.Draw(img)
        items = []
        for i, corners in quads:
            px, depth = project(corners, cam.spec, cam.width, cam.height, cam.fov)
            if (depth < 1.0).any() or depth.mean() > MAX_RANGE_M:
                continue
            if (px[:, 0] < -cam.width).all() or (px[:, 0] > 2 * cam.width).all() or (px[:, 1] > 2 * cam.height).all():
                continue
            items.append((depth.mean(), i, [tuple(p) for p in px]))
        for _, i, poly in sorted(items, key=lambda t: -t[0]):  # far first: near pieces cover them
            draw.polygon(poly, fill=i + 1)
        full.append(np.array(img, dtype=np.int32))
    area = np.zeros((len(cams), len(lanes) + 1), dtype=np.int64)
    for c, m in enumerate(full):
        area[c] = np.bincount(m.ravel(), minlength=len(lanes) + 1)[: len(lanes) + 1]
    owner = {}
    for i, lid in enumerate(lanes):
        col = area[:, i + 1]
        if col.max() > 0:
            owner[lid] = int(col.argmax())
    for c, cam in enumerate(cams):
        keep = np.array([0] + [i + 1 for i, lid in enumerate(lanes) if owner.get(lid) == c], dtype=np.int32)
        cam.mask = np.where(np.isin(full[c], keep), full[c], 0)
        cam.lanes = lanes
    return owner


def ground_point_lane(mask: np.ndarray, box) -> int:
    """Lane index+1 under a detection: scanning up from the box's bottom middle, at most a
    third of the box height (the wheels touch the road at the bottom edge)."""
    x1, y1, x2, y2 = box
    h, w = mask.shape
    cx = int(min(max((x1 + x2) / 2, 0), w - 1))
    bottom = int(min(max(y2 - 1, 0), h - 1))
    top = int(max(bottom - (y2 - y1) / 3, 0))
    for y in range(bottom, top - 1, -1):
        v = int(mask[y, cx])
        if v:
            return v
    return 0


def count_lanes(cam: LaneCamera, boxes, names) -> dict[str, int]:
    """boxes: (N, 4) xyxy in the mask's pixels; names: class name per box."""
    counts: dict[str, int] = {}
    for box, name in zip(boxes, names, strict=True):
        if name not in VEHICLE_CLASSES:
            continue
        v = ground_point_lane(cam.mask, box)
        if v:
            lid = cam.lanes[v - 1]
            counts[lid] = counts.get(lid, 0) + 1
    return counts
