"""Does an emergency vehicle's light bar flash? From a few pictures of it (siren step 8, 10 Nov).

The detector reads one picture at a time: a siren-on vehicle is called siren-on in about a third of the pictures
(out/detector/eval_town05.md), because in one picture a flashing bar may be dark, or lit in a colour the detector
learnt for the other state. Over several pictures a bar that flashes shows as its colours jumping: red up, blue down,
then the other way. Here, per picture, a few light numbers of the vehicle's box (light_numbers), and over a window of
pictures how much they move (window_features); a logistic regression (models/siren_flash.json, tools/ev_flash_model.py,
trained on CARLA clips of tools/record_ev_clips.py) turns them into the chance that the lights are on. numpy only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "siren_flash.json"
FEATS = ("red", "blue", "bright", "luma")
FEATURE_NAMES = tuple(f"{k}_{s}" for k in ("red", "blue", "bright") for s in ("std", "range", "step")) + \
    ("red_blue_corr", "red_mean", "blue_mean", "luma_mean")


def light_numbers(rgb: np.ndarray, box) -> list[float]:
    """rgb: H x W x 3 uint8 (RGB); box: x1, y1, x2, y2 (inclusive). In the upper half of the box (where the bar is): how
    far the reddest and the bluest pixels stand out (95th percentile of R - (G+B)/2 and of B - (R+G)/2), the 99th
    percentile of brightness, and the mean brightness, all / 255."""
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    top = rgb[max(0, y1):max(y1 + 1, (y1 + y2) // 2 + 1), max(0, x1):x2 + 1].astype(np.float32)
    if top.size == 0:
        return [float("nan")] * len(FEATS)
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return [float(np.percentile(r - (g + b) / 2, 95)) / 255.0, float(np.percentile(b - (r + g) / 2, 95)) / 255.0,
            float(np.percentile(luma, 99)) / 255.0, float(luma.mean()) / 255.0]


def window_features(series) -> np.ndarray | None:
    """series: pictures x 4 light numbers (NaN where the vehicle was not seen). None with fewer than half seen."""
    s = np.asarray(series, dtype=np.float64)
    seen = ~np.isnan(s[:, 0])
    if seen.sum() < max(3, len(s) // 2):
        return None
    idx = np.arange(len(s))
    s = np.stack([np.interp(idx, idx[seen], s[seen, j]) for j in range(s.shape[1])], axis=1)
    out = []
    for j in range(3):
        x = s[:, j]
        out += [x.std(), x.max() - x.min(), np.abs(np.diff(x)).mean()]
    r, b = s[:, 0], s[:, 1]
    corr = float(np.corrcoef(r, b)[0, 1]) if r.std() > 1e-6 and b.std() > 1e-6 else 0.0
    out += [corr, r.mean(), b.mean(), s[:, 3].mean()]
    return np.array(out)


class SirenFlash:
    def __init__(self, model: dict | str | Path | None = None):
        if model is None or isinstance(model, (str, Path)):
            model = json.loads(Path(model or MODEL).read_text(encoding="utf-8"))
        self.m = model
        self.mean, self.std = np.array(model["mean"]), np.array(model["std"])
        self.w, self.b = np.array(model["w"]), float(model["b"])
        self.threshold = float(model.get("threshold", 0.5))
        self.pictures = int(model.get("pictures", 8))

    def prob_from_features(self, x) -> float:
        z = float(((np.asarray(x) - self.mean) / self.std) @ self.w + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def prob(self, series) -> float | None:
        """series: the last pictures' light numbers (oldest first); None with too few."""
        if len(series) < self.pictures:
            return None
        x = window_features(np.asarray(series)[-self.pictures:])
        return None if x is None else self.prob_from_features(x)
