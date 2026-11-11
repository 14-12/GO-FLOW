"""The traffic-camera detector on the viewer's picture, one look against another (cosim/look.py).

    python tools/look_detector_check.py out/look_check/old out/look_check/town [--model models/carla-yolo26n-t05.pt]

Each folder is a set made by tools/make_dataset_town05.py --look old|town with the same seed, cameras and hours:
the same traffic in front of the same cameras, only the weather and the lights differ. At the inspector's
threshold (conf 0.35), per hour and in all: objects found by group (a box of the right group, IoU >= 0.5), false
boxes per 100 pictures, the mean confidence of the boxes that found an object. Writes look_detector_check.json
beside the first folder.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from eval_town05_detector import CONF, GROUP, GROUPS, gt_boxes, iou  # noqa: E402


def score(model, images: list[Path], names: list[str]) -> dict:
    found = {g: 0 for g in GROUPS}
    total = {g: 0 for g in GROUPS}
    false_n, confs = 0, []
    for k in range(0, len(images), 8):
        batch = images[k:k + 8]
        for img, r in zip(batch, model.predict([str(p) for p in batch], imgsz=960, conf=CONF, verbose=False,
                                                quantize=16), strict=True):
            h, w = r.orig_shape
            gb, gg = gt_boxes(img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt", w, h, names)
            pb = r.boxes.xyxy.cpu().numpy()
            pc = r.boxes.conf.cpu().numpy()
            pg = [GROUP[r.names[int(c)]] for c in r.boxes.cls.cpu().numpy()]
            m = iou(gb, pb)
            for i, g in enumerate(gg):
                total[g] += 1
                hit = [j for j in range(len(pg)) if m[i, j] >= 0.5 and pg[j] == g]
                if hit:
                    found[g] += 1
                    confs.append(float(max(pc[j] for j in hit)))
            for j in range(len(pg)):
                if not len(gb) or m[:, j].max() < 0.3:
                    false_n += 1
    n = max(len(images), 1)
    return {"pictures": len(images), "found": {g: [found[g], total[g]] for g in GROUPS},
            "found_share": round(sum(found.values()) / max(sum(total.values()), 1), 3),
            "false_boxes_per_100_pictures": round(100 * false_n / n, 1),
            "mean_confidence_of_found": round(float(np.mean(confs)), 3) if confs else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--model", default="models/carla-yolo26n-t05.pt")
    a = ap.parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(str(ROOT / a.model))
    names = [model.names[i] for i in range(len(model.names))]
    out = {}
    for d in a.sets:
        images = sorted((ROOT / d / "images").glob("*/*.jpg"))
        by_hour: dict[str, list[Path]] = {}
        for p in images:
            m = re.search(r"-([0-9.]+)h_", p.name)
            by_hour.setdefault(m.group(1) if m else "?", []).append(p)
        res = {"all": score(model, images, names)}
        for hour in sorted(by_hour, key=float):
            res[f"{hour} h"] = score(model, by_hour[hour], names)
        out[d] = res
        for k, v in res.items():
            print(d, k, json.dumps(v), flush=True)
    (ROOT / a.sets[0]).parent.joinpath("look_detector_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
