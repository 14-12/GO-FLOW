"""The moment of a crash on a camera's picture (3 Nov): does a crashing car stop dead at the hit, or ease into it?

From a --record-clips run (tools/record_incident_clips.bat): the crash clip, its labels (clips.json: the camera's pose,
the incident's place and t_contact) and the source recording's frame index (<camera>.frames.csv: the SUMO second of
every picture). The traffic-camera detector (models/carla-yolo26n-t05, ONNX) finds the vehicles in each picture
from 3 s before the contact to 3 s after; the wrecks are the boxes standing still at the place of the crash
(projected into the picture from the camera's pose) 20 s later, followed back picture by picture. For each wreck:
its speed in the picture (box centre, pixels per second) around the contact, and how many pictures it takes to stop.
A contact sheet of the pictures around the contact is written for the eye.

    python tools/check_impact_video.py out/incident_clips/<time> [--onnx out/detector_onnx/carla-yolo26n-t05.onnx]
    python tools/check_impact_video.py out/incident_clips/<time> --camera "1427-2 looking E" --incident 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["car", "van", "bus", "truck", "ambulance", "ambulance_lights", "police", "police_lights", "pedestrian"]
VEHICLES = {0, 1, 2, 3, 4, 6}
IMG = 960
STEP_S = 0.5


def detect(sess, frame: np.ndarray, conf: float = 0.3) -> list[tuple]:
    """[(x1, y1, x2, y2, score, class)] in the frame's pixels (letterboxed to 960x960, NMS)."""
    h, w = frame.shape[:2]
    s = IMG / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    top, left = (IMG - nh) // 2, (IMG - nw) // 2
    canvas = np.full((IMG, IMG, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]  # (13, N)
    boxes, scores, cls = out[:4].T, out[4:].max(0), out[4:].argmax(0)
    keep = scores >= conf
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    res = []
    if len(boxes):
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / s
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / s
        idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
                               scores.tolist(), conf, 0.5)
        for i in np.array(idx).flatten():
            if int(cls[i]) in VEHICLES:
                res.append((*map(float, xyxy[i]), float(scores[i]), int(cls[i])))
    return res


def project(pose: dict, fov: float, w: int, h: int, x: float, y: float, z: float = 0.5):
    """CARLA world point -> picture pixel (CARLA camera: x forward, y right, z up; UE rotation yaw, pitch)."""
    yaw, pitch = math.radians(pose["yaw_deg"]), math.radians(pose["pitch_deg"])
    dx, dy, dz = x - pose["x"], y - pose["y"], z - pose["z"]
    # into the camera's frame: undo yaw (about z), then pitch (about y)
    cx = dx * math.cos(yaw) + dy * math.sin(yaw)
    cy = -dx * math.sin(yaw) + dy * math.cos(yaw)
    fx = cx * math.cos(pitch) + dz * math.sin(pitch)
    fz = -cx * math.sin(pitch) + dz * math.cos(pitch)
    if fx <= 0.1:
        return None
    f = w / (2 * math.tan(math.radians(fov) / 2))
    return w / 2 + f * cy / fx, h / 2 - f * fz / fx


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--onnx", default="out/detector_onnx/carla-yolo26n-t05.onnx")
    ap.add_argument("--before", type=float, default=3.0)
    ap.add_argument("--after", type=float, default=3.0)
    ap.add_argument("--rest", type=float, default=20.0, help="the wrecks are found standing this long after")
    ap.add_argument("--camera", default=None, help="a camera's whole recording instead of crash.mp4 (with --incident)")
    ap.add_argument("--incident", type=int, default=None)
    a = ap.parse_args(argv)
    import onnxruntime as ort

    folder = Path(a.folder)
    if a.camera is None:
        clips = json.loads((folder / "clips.json").read_text(encoding="utf-8"))
        c = clips["clips"]["crash"] if "clips" in clips else clips["crash"]
        inc = c["incident"]
    else:  # straight from a camera's whole recording and incidents.json (no crash clip cut yet)
        incs = json.loads((folder / "incidents.json").read_text(encoding="utf-8"))["incidents"]
        inc = next(i for i in (incs if isinstance(incs, list) else incs.values()) if i["id"] == a.incident)
        lay = json.loads((Path(__file__).resolve().parents[1] / "out" / "town05_cameras" / "layout.json")
                         .read_text(encoding="utf-8"))
        spec = next(x for x in lay["cameras"] if x["name"] == a.camera)
        c = {"camera_id": a.camera, "fov_deg": lay["fov"], "incident": inc,
             "source_recording": a.camera.replace(" ", "_") + ".mp4",
             "camera_pose_carla": {"x": spec["x"], "y": spec["y"], "z": spec["z"], "yaw_deg": spec["yaw"],
                                   "pitch_deg": spec["pitch"]}}
    t_contact = inc.get("t_contact")
    if t_contact is None:
        print("no t_contact in clips.json (a recording from before 3 Nov)")
        return 1
    src = folder / c["source_recording"]
    rows = []
    with (folder / (src.stem + ".frames.csv")).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # sim_s is the SUMO step a picture belongs to (the step it ends on); within a step the pictures run on CARLA's
    # clock (carla_s, real time at --clock-speed 1): the SUMO time a picture shows is the step's start plus the CARLA
    # seconds since that step's first picture (at most the step's end)
    sims, first = [], {}
    for r in rows:
        lab, cs = float(r["sim_s"]), float(r["carla_s"])
        c0 = first.setdefault(lab, cs)
        sims.append(min(lab - STEP_S + (cs - c0), lab))
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: dict[int, np.ndarray] = {}

    def frame_at(i):
        if i not in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            frames[i] = fr if ok else None
        return frames[i]

    idx = [i for i, s in enumerate(sims) if t_contact - a.before - 1e-6 <= s <= t_contact + a.after + 1e-6 and i < n]
    i_rest = min(range(len(sims)), key=lambda i: abs(sims[i] - (t_contact + a.rest)))
    sess = ort.InferenceSession(a.onnx)
    h, w = frame_at(idx[0]).shape[:2]
    px = project(c["camera_pose_carla"], c["fov_deg"], w, h, *inc["carla_xy"])
    det = {i: detect(sess, frame_at(i)) for i in idx + [i_rest]}
    # the wrecks: standing at the rest picture near the crash's place, and in the last picture of the window
    rest = det[i_rest]
    last = det[idx[-1]]
    wrecks = []
    for b in rest:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        near = px is None or math.hypot(cx - px[0], cy - px[1]) < 0.25 * w
        same = max((iou(b, l) for l in last), default=0.0)
        if near and same > 0.6:
            wrecks.append(b)
    print(f"camera {c['camera_id']}, contact at SUMO {t_contact} s (clip {inc.get('contact_at_clip_s')} s), "
          f"step that saw it {inc['t0']}; crash place in the picture {None if px is None else tuple(round(v) for v in px)}; "
          f"{len(wrecks)} wreck box(es) found")
    report = {"camera": c["camera_id"], "t_contact": t_contact, "t0": inc["t0"], "kind": inc.get("kind"),
              "severity_kmh": inc.get("severity_kmh"), "wrecks": []}
    for wb in wrecks:
        track = []
        cur = wb
        for i in reversed(idx):
            best = max(det[i], key=lambda b: iou(b, cur), default=None)
            if best is None or iou(best, cur) < 0.2:
                break
            track.append((sims[i], best))
            cur = best
        track.reverse()
        speeds = []
        for (t1, b1), (t2, b2) in zip(track, track[1:], strict=False):
            c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
            c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
            speeds.append((round(t2 - t_contact, 2), round(math.hypot(c2[0] - c1[0], c2[1] - c1[1]) / (t2 - t1), 1)))
        before = [v for t, v in speeds if -1.0 <= t <= 0.0]
        v0 = max(before) if before else 0.0
        moving = v0 > 15.0  # px/s: a car standing at the hit is not followed
        stop_t = next((t for t, v in speeds if t > -1.0 and v < 0.15 * v0), None) if moving else None
        slow_t = max((t for t, v in speeds if t <= (stop_t if stop_t is not None else 9) and v >= 0.7 * v0),
                     default=None) if moving else None
        row = {"box_at_rest": [round(v) for v in wb[:4]], "class": CLASSES[wb[5]], "px_per_s": speeds,
               "speed_before_px_s": v0, "moving_at_hit": moving,
               "last_fast_picture_s": slow_t, "first_still_picture_s": stop_t,
               "stop_takes_s": None if slow_t is None or stop_t is None else round(stop_t - slow_t, 2)}
        report["wrecks"].append(row)
        print(f"  {row['class']} at {row['box_at_rest']}: moving {moving} ({v0} px/s); last picture at >=70 % speed "
              f"{slow_t} s, first still {stop_t} s -> the stop takes {row['stop_takes_s']} s")
        print("    px/s by picture (s from contact):", speeds)
    # contact sheet: the pictures from 1 s before to 1.5 s after, crash place marked, wrecks boxed
    sheet_idx = [i for i in idx if -1.0 - 1e-6 <= sims[i] - t_contact <= 1.5 + 1e-6]
    tiles = []
    for i in sheet_idx:
        fr = frame_at(i).copy()
        for b in det[i]:
            cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
        if px is not None:
            cv2.circle(fr, (int(px[0]), int(px[1])), 6, (0, 0, 255), 2)
        cv2.putText(fr, f"{sims[i] - t_contact:+.1f} s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        if px is not None:  # a crop around the crash
            x0 = int(min(max(px[0] - 240, 0), w - 480))
            y0 = int(min(max(px[1] - 135, 0), h - 270))
            fr = fr[y0:y0 + 270, x0:x0 + 480]
        tiles.append(fr)
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    out = folder / "impact_check.jpg"
    cv2.imwrite(str(out), sheet)
    (folder / "impact_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("sheet:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
