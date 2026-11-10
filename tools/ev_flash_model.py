"""A model that sees an emergency vehicle's lights flash, from a few pictures of it (siren step 8).

Input: out/ev_clips/features.npz (tools/ev_clip_features.py): for each clip of tools/record_ev_clips.py (one ambulance
or police car, lights on or off, in front of an inspector camera; 16 pictures at 10 a second), per picture the
light numbers of the vehicle's box (cosim.siren_flash.light_numbers) and what the shipped detector called it.

The model looks at how the light numbers move from picture to picture over a window (8 pictures at 5 a second, as
the H window takes them: 1.6 s), cosim.siren_flash.window_features, and gives the chance that the lights are on
(logistic regression, numpy only). Trained on the clips of the detector's train and val junctions, tested on those of
its 3 test junctions (make_dataset_town05.split_tls, seed 11), never seen in training. Compared, per class, with the
detector alone: one picture, and its 3-of-8 vote (the H window's and the camera reports' rule).

    python tools/ev_flash_model.py --features out/ev_clips/features.npz --out models/siren_flash.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from cosim.siren_flash import FEATURE_NAMES, SirenFlash, window_features  # noqa: E402

STEP = 2          # every 2nd picture of a clip: 5 a second, as the H window
WIN = 8           # pictures in a window (1.6 s)


def fit_logistic(X, y, l2=1e-2, iters=4000, lr=0.5):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="out/ev_clips/features.npz")
    ap.add_argument("--out", default="models/siren_flash.json")
    ap.add_argument("--report", default="out/ev_clips/report.json")
    a = ap.parse_args(argv)
    z = np.load(ROOT / a.features, allow_pickle=True)
    feats, dets = z["feats"], z["dets"]
    metas = [json.loads(str(m)) for m in z["meta"]]
    sel = list(range(0, feats.shape[1], STEP))[:WIN]
    X, y, split, ok = [], [], [], []
    for f, m in zip(feats, metas):
        fx = window_features(f[sel])
        ok.append(fx is not None)
        X.append(fx if fx is not None else np.zeros(len(FEATURE_NAMES)))
        y.append(1.0 if m["lights"] else 0.0)
        split.append("test" if m["split"] == "test" else "train")
    X, y, ok = np.array(X), np.array(y), np.array(ok)
    tr = np.array([s == "train" for s in split]) & ok
    te = np.array([s == "test" for s in split]) & ok
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    w, b = fit_logistic((X[tr] - mu) / sd, y[tr])
    model = {"version": "siren_flash_v1", "features": list(FEATURE_NAMES), "mean": mu.tolist(), "std": sd.tolist(),
             "w": w.tolist(), "b": float(b), "threshold": 0.5, "pictures": WIN, "pictures_per_s": 5.0,
             "trained_on": {"clips": int(tr.sum()), "split": "train+val junctions of out/dataset_town05"}}
    flash = SirenFlash(model)
    p = np.array([flash.prob_from_features(x) for x in X])

    def det_on(d):
        return [bool(n) and n.endswith("_lights") for n in d]

    def vote(d, need=3, win=8):
        on = det_on(d)
        return any(sum(on[max(0, i - win + 1):i + 1]) >= need for i in range(len(on)))

    rows = {}
    for i, m in enumerate(metas):
        if not te[i]:
            continue
        key = m["class"]
        cond = "night" if m["night"] else m["condition"].split("-")[0]
        d5 = [dets[i][k] for k in sel]
        for k in (key, f"{key} | {cond}", "all lights on" if m["lights"] else "all lights off"):
            r = rows.setdefault(k, {"clips": 0, "detector_one_picture": 0, "detector_3_of_8": 0, "flash": 0,
                                    "detector_sees_emergency": 0, "flash_and_emergency": 0})
            r["clips"] += 1
            mid = d5[len(d5) // 2]
            r["detector_one_picture"] += int(bool(mid) and mid.endswith("_lights"))
            r["detector_3_of_8"] += int(vote(d5))
            fl = p[i] >= model["threshold"]
            r["flash"] += int(fl)
            em = sum(1 for n in d5 if n in ("ambulance", "ambulance_lights", "police", "police_lights")) >= 3
            r["detector_sees_emergency"] += int(em)
            r["flash_and_emergency"] += int(fl and em)
    for r in rows.values():
        for k in list(r):
            if k != "clips":
                r[k + "_share"] = round(r[k] / r["clips"], 3)
    report = {"what": "share of test clips called SIREN ON (lights-on classes: higher is better; lights-off: lower)",
              "test_clips": int(te.sum()), "train_clips": int(tr.sum()), "window": f"{WIN} pictures at 5/s",
              "rows": dict(sorted(rows.items())),
              "train_accuracy": float(((p[tr] >= 0.5) == (y[tr] > 0.5)).mean()),
              "test_accuracy": float(((p[te] >= 0.5) == (y[te] > 0.5)).mean())}
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(model, indent=1), encoding="utf-8")
    (ROOT / a.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
