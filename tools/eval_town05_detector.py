"""Old against new traffic-camera detector, on the Town05 junctions held out of training.

    python tools/eval_town05_detector.py [--models models/carla-yolo26n.pt,models/carla-yolo26n-t05.pt]

On out/dataset_town05's test split (3 signals, 10 cameras, never seen in training or model selection):
* mAP50 and mAP50-95 (Ultralytics val, 960 px);
* at the inspector's threshold (conf 0.35), what the inspector would show:
  - found: share of labelled objects with a box of the right group on them (IoU >= 0.5); groups as the
    inspector counts them: vehicles (car, van, truck), public transport (bus), pedestrians,
    emergency SIREN ON (ambulance_lights, police_lights), emergency siren off (ambulance, police);
  - false boxes: a box on nothing labelled (IoU < 0.3 with every object), per 100 pictures; and the
    "emergency" ones among them, the facade boxes seen in the free-fly demo;
  - wrong group: a box on an object of another group (an ordinary car called an ambulance).
Writes out/detector/eval_town05.json and .md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

GROUP = {"car": "vehicles", "van": "vehicles", "truck": "vehicles", "bus": "public transport",
         "pedestrian": "pedestrians", "ambulance_lights": "emergency SIREN ON", "police_lights": "emergency SIREN ON",
         "ambulance": "emergency siren off", "police": "emergency siren off"}
GROUPS = ["vehicles", "public transport", "pedestrians", "emergency SIREN ON", "emergency siren off"]
CONF = 0.35


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of boxes a (N, 4) and b (M, 4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def gt_boxes(label: Path, w: int, h: int, names: list[str]):
    boxes, groups = [], []
    for ln in label.read_text(encoding="utf-8").split("\n"):
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = ln.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        groups.append(GROUP[names[int(c)]])
    return np.array(boxes).reshape(-1, 4), groups


def inspector_view(model, images: list[Path], names: list[str]) -> dict:
    found = {g: 0 for g in GROUPS}
    total = {g: 0 for g in GROUPS}
    false_boxes = {g: 0 for g in GROUPS}
    wrong = {}
    for k in range(0, len(images), 8):
        batch = images[k:k + 8]
        for img, r in zip(batch, model.predict([str(p) for p in batch], imgsz=960, conf=CONF, verbose=False,
                                                quantize=16), strict=True):
            h, w = r.orig_shape
            gb, gg = gt_boxes(img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt", w, h, names)
            pb = r.boxes.xyxy.cpu().numpy()
            pg = [GROUP[r.names[int(c)]] for c in r.boxes.cls.cpu().numpy()]
            m = iou(gb, pb)
            for i, g in enumerate(gg):
                total[g] += 1
                if any(m[i, j] >= 0.5 and pg[j] == g for j in range(len(pg))):
                    found[g] += 1
            for j, g in enumerate(pg):
                best = int(np.argmax(m[:, j])) if len(gb) else -1
                if best < 0 or m[best, j] < 0.3:
                    false_boxes[g] += 1
                elif gg[best] != g and m[best, j] >= 0.5:
                    key = f"{gg[best]} called {g}"
                    wrong[key] = wrong.get(key, 0) + 1
    n = len(images)
    return {"images": n, "found": {g: [found[g], total[g]] for g in GROUPS},
            "false_boxes_per_100_images": {g: round(100 * false_boxes[g] / n, 1) for g in GROUPS},
            "false_emergency_boxes_per_100_images": round(
                100 * (false_boxes["emergency SIREN ON"] + false_boxes["emergency siren off"]) / n, 1),
            "wrong_group": dict(sorted(wrong.items(), key=lambda kv: -kv[1]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/dataset_town05/dataset.yaml")
    ap.add_argument("--models", default="models/carla-yolo26n.pt,models/carla-yolo26n-t05.pt")
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    from ultralytics import YOLO

    from train_detector import resolved_yaml

    data = resolved_yaml((ROOT / a.data).resolve(), ROOT / "out" / "detector")
    images = sorted((ROOT / a.data).resolve().parent.joinpath("images", a.split).glob("*.jpg"))
    out = {"split": a.split, "data": a.data, "conf": CONF, "models": {}}
    for mp in a.models.split(","):
        model = YOLO(str(ROOT / mp))
        names = [model.names[i] for i in range(len(model.names))]
        v = model.val(data=str(data), split=a.split, imgsz=960, batch=4, quantize=16, plots=False, verbose=False,
                      project=str(ROOT / "out" / "detector"), name=f"eval_town05_{Path(mp).stem}", exist_ok=True)
        per_class = {names[c]: round(float(v.box.ap50[i]), 3) for i, c in enumerate(v.box.ap_class_index)}
        res = {"mAP50": round(float(v.box.map50), 3), "mAP50-95": round(float(v.box.map), 3), "AP50": per_class}
        res.update(inspector_view(model, images, names))
        out["models"][Path(mp).stem] = res
        print(Path(mp).stem, json.dumps(res, indent=1), flush=True)
    det = ROOT / "out" / "detector"
    (det / "eval_town05.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"### Town05 held-out junctions ({a.split} split, {len(images)} pictures, inspector cameras)", "",
          "| model | mAP50 | mAP50-95 | " + " | ".join(f"found: {g}" for g in GROUPS)
          + " | false boxes /100 pictures | false emergency boxes /100 pictures |",
          "|---|---|---|" + "---|" * (len(GROUPS) + 2)]
    for name, r in out["models"].items():
        f = r["found"]
        md.append(f"| {name} | {r['mAP50']} | {r['mAP50-95']} | "
                  + " | ".join(f"{f[g][0]}/{f[g][1]}" for g in GROUPS)
                  + f" | {sum(r['false_boxes_per_100_images'].values()):.1f} | "
                    f"{r['false_emergency_boxes_per_100_images']} |")
    md += ["", "Wrong group (box on an object of another group, IoU >= 0.5):", ""]
    for name, r in out["models"].items():
        md.append(f"* {name}: " + (", ".join(f"{k} {n}" for k, n in r["wrong_group"].items()) or "none"))
    (det / "eval_town05.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
