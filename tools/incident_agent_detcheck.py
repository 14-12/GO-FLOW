"""The live detector's batched path against one `track(persist=True)` per camera (11 Nov; cosim/incident_agent_proc).

Takes the pictures his agent keeps (5 fps) from two or more recorded cameras, interleaves them as they would arrive
live, and runs them (a) camera by camera through `YOLO.track(persist=True)` (his process_frames' way) and (b) in
batches of the waiting pictures with one ByteTrack per camera (the live default). Reports how many rows differ.

    .venv\\Scripts\\python.exe tools\\incident_agent_detcheck.py out\\incident_clips\\<run> [--pictures 600]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def one_raw(det, bgr):
    return det.model.predict(bgr, **det.opts())[0].boxes.xyxy.cpu().numpy()


def match(a, b):
    """Greedy nearest match of box centres: (pairs (i, j, distance px), unmatched in a, unmatched in b)."""
    import numpy as np

    if len(a) == 0 or len(b) == 0:
        return [], len(a), len(b)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    pairs, used_a, used_b = [], set(), set()
    for k in np.argsort(d, axis=None):
        i, j = divmod(int(k), d.shape[1])
        if i in used_a or j in used_b or d[i, j] > 20:
            continue
        pairs.append((i, j, float(d[i, j])))
        used_a.add(i)
        used_b.add(j)
    return pairs, len(a) - len(pairs), len(b) - len(pairs)


def compare_boxes(xa, xb):
    import numpy as np

    out = {"pictures": len(xa), "boxes": int(sum(len(x) for x in xa)), "unmatched": 0, "centre_px_max": 0.0,
           "centre_px_p99": 0.0}
    ds = []
    for a, b in zip(xa, xb):
        ca = np.c_[(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2] if len(a) else a
        cb = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2] if len(b) else b
        pairs, ua, ub = match(ca, cb)
        out["unmatched"] += ua + ub
        ds += [p[2] for p in pairs]
    if ds:
        ds.sort()
        out["centre_px_max"], out["centre_px_p99"] = round(ds[-1], 3), round(ds[int(0.99 * len(ds))], 3)
    return out


def compare_tracked(order, ref, got):
    """Rows of the two paths matched by box centre, ids compared as identities (a track's id in one path maps to
    one id in the other), not as numbers: Ultralytics numbers tracks from one counter for every tracker of a
    process and resets it whenever a tracker is made."""
    import numpy as np
    from collections import Counter, defaultdict

    out = {"rows_one_by_one": 0, "rows_batched": 0, "unmatched": 0, "class_differ": 0, "centre_px_max": 0.0,
           "same_identity_share": None}
    pair_ids = defaultdict(Counter)
    for (cam, _), (_, da), (_, db) in zip(order, ref, got):
        out["rows_one_by_one"] += len(da)
        out["rows_batched"] += len(db)
        ca = np.array([[d[4], d[5]] for d in da]).reshape(-1, 2)
        cb = np.array([[d[4], d[5]] for d in db]).reshape(-1, 2)
        pairs, ua, ub = match(ca, cb)
        out["unmatched"] += ua + ub
        for i, j, dist in pairs:
            out["centre_px_max"] = max(out["centre_px_max"], round(dist, 3))
            out["class_differ"] += da[i][1] != db[j][1]
            pair_ids[(cam, da[i][0])][db[j][0]] += 1
    total = sum(sum(c.values()) for c in pair_ids.values())
    same = sum(c.most_common(1)[0][1] for c in pair_ids.values())
    out["same_identity_share"] = round(same / total, 4) if total else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("--pictures", type=int, default=600, help="per camera")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    import cv2
    import torch

    from cosim.incident_agent_proc import CONFIG, WEIGHTS, Detector

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    device = 0 if torch.cuda.is_available() else "cpu"
    pics = {}
    for mp4 in sorted(Path(a.clips).glob("*.mp4"))[:4]:
        if mp4.stem in ("crash", "normal_traffic"):
            continue
        cap, k, got = cv2.VideoCapture(str(mp4)), 0, []
        while len(got) < a.pictures:
            ok, bgr = cap.read()
            if not ok:
                break
            k += 1
            if k % 2 == 0:
                got.append(bgr)
        cap.release()
        pics[mp4.stem] = got
    cams = list(pics)
    order = [(c, pics[c][i]) for i in range(min(len(v) for v in pics.values())) for c in cams]
    one = Detector(str(WEIGHTS), cfg, cams, device, batch=False)
    t1 = time.time()
    ref = one(order)
    t_one = time.time() - t1
    many = Detector(str(WEIGHTS), cfg, cams, device, batch=True)
    t1, got = time.time(), []
    for i in range(0, len(order), a.batch):
        got += many(order[i:i + a.batch])
    t_many = time.time() - t1
    raw_one = [one_raw(many, b) for _, b in order[:200]]
    raw_many = []
    for i in range(0, 200, a.batch):
        raw_many += [r.boxes.xyxy.cpu().numpy() for r in many.model.predict([b for _, b in order[i:i + a.batch]],
                                                                             **many.opts())]
    diff = {"pictures": len(order), "cameras": cams,
            "ms_per_picture_one_by_one": round(1000 * t_one / len(order), 1),
            "ms_per_picture_batched": round(1000 * t_many / len(order), 1),
            "raw_boxes": compare_boxes(raw_one, raw_many),
            "tracked": compare_tracked(order, ref, got)}
    print(json.dumps(diff), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(diff, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
