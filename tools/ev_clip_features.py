"""Per-picture numbers of each clip of tools/record_ev_clips.py, for the siren-light model (siren step 8).

For every picture of every clip: what the shipped detector says about the vehicle (the class of its box that
overlaps the vehicle's visible box most, IoU >= 0.3, conf 0.35, imgsz 960 as in the H window; "" when none), and
the light numbers of the vehicle's box (light_numbers: a flashing bar shows as them jumping from picture to
picture), and the box itself at 48 x 48.

    python tools/ev_clip_features.py --clips out/ev_clips        -> out/ev_clips/features.npz
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from cosim.siren_flash import FEATS, light_numbers  # noqa: E402

CROP = 48            # each picture's vehicle box (10 % margin) kept at 48 x 48, to work on the numbers elsewhere


def crop(rgb: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    mx, my = max(2, (x2 - x1) // 10), max(2, (y2 - y1) // 10)
    im = Image.fromarray(rgb[max(0, y1 - my):y2 + my + 1, max(0, x1 - mx):x2 + mx + 1])
    return np.asarray(im.resize((CROP, CROP), Image.BILINEAR))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="out/ev_clips")
    ap.add_argument("--model", default=str(ROOT / "models" / "carla-yolo26n-t05.pt"))
    ap.add_argument("--no-detector", action="store_true")
    a = ap.parse_args(argv)
    folder = ROOT / a.clips
    metas = [json.loads(line) for line in (folder / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    detect = None
    if not a.no_detector:
        import torch
        from ultralytics import YOLO

        dev = 0 if torch.cuda.is_available() else "cpu"
        prec = 16 if dev != "cpu" else None  # FP16 on the GPU (ultralytics 8.4: `quantize`, formerly `half`)
        model = YOLO(a.model)
        names = model.names

        def detect(bgr):
            r = model.predict(bgr, imgsz=960, conf=0.35, device=dev, verbose=False, quantize=prec)[0]
            return [(list(map(float, b)), names[int(k)], float(s)) for b, k, s in
                    zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    feats, dets, keep, crops = [], [], [], []
    for i, m in enumerate(metas):
        z = np.load(folder / "clips" / f"{m['clip']:05d}.npz", allow_pickle=True)
        frames, boxes = z["frames"], z["boxes"]
        f_clip, d_clip, c_clip = [], [], []
        for fr, bx in zip(frames, boxes):
            rgb = np.asarray(Image.open(io.BytesIO(fr.tobytes())).convert("RGB"))
            seen = bx[0] >= 0
            f_clip.append(light_numbers(rgb, bx) if seen else [np.nan] * len(FEATS))
            c_clip.append(crop(rgb, bx) if seen else np.zeros((CROP, CROP, 3), np.uint8))
            cls = ""
            if detect is not None and seen:
                best = 0.0
                for b, name, _ in detect(np.ascontiguousarray(rgb[:, :, ::-1])):
                    o = iou(b, [bx[0], bx[1], bx[2] + 1, bx[3] + 1])
                    if o >= 0.3 and o > best:
                        best, cls = o, name
            d_clip.append(cls)
        feats.append(f_clip)
        dets.append(d_clip)
        crops.append(np.stack(c_clip))
        keep.append(m)
        if i % 50 == 0:
            print(f"{i + 1}/{len(metas)}", flush=True)
    np.savez_compressed(folder / "features.npz", feats=np.array(feats, dtype=np.float32), feat_names=np.array(FEATS),
             crops=np.stack(crops), boxes=np.stack([np.load(folder / "clips" / f"{m['clip']:05d}.npz",
                                                            allow_pickle=True)["boxes"] for m in keep]),
             dets=np.array(dets, dtype=object), meta=np.array([json.dumps(m) for m in keep]))
    print(json.dumps({"clips": len(keep)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
