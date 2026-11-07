"""Long queues coming (7 Nov): a classifier for "this approach starts a long queue within K steps".

Why. The first alert rule (agent.act before: an approach under its `low` now, forecast at its `full`) fired on queues
refilling at the next red and caught no build-up on the dev seeds, at any traffic (docs/WORKLOG.md, 7 Nov). A queue
measured every 30 s swings with the signal (cycle 115 s), so one step says little; what matters is a queue that stays.

The event. `c` = the approach's fill averaged over the last CYCLE_STEPS steps (about one signal cycle). Its `level` is
the Q-quantile of `c` on the train seeds (never under 3 queued vehicles). A long queue STARTS at step t when c[t] >=
level, it was under level for the CYCLE_STEPS steps before, and under LOW_FRAC x level CYCLE_STEPS steps before.

The model. LightGBM, binary, on the forecast's own features (features.build, the +120 s demand column) plus four of
the approach's state against its level (cycle mean, last step, rise over 2 steps, max over the cycle), for the rows
where the approach is not already over its level. Target: a long queue starts within K steps. The alert threshold is
chosen on the dev seeds (the test seeds stay closed).

    python -m goflow_prediction.congestion --dataset D3_mix.npz --model DIR --train-seeds 101-130 --dev-seeds 11-20

writes DIR/congestion_model.txt and DIR/congestion.json (levels, settings, dev scores) next to a forecast model, which
the agent then uses for its alerts (agent.ForecastAgent.act).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .features import Window, build

CYCLE_STEPS = 4      # 4 x 30 s: about one signal cycle (115 s)
Q = 0.95             # an approach's long queue: its cycle-mean fill exceeded 5 % of the time on the train seeds
LOW_FRAC = 0.6       # ... reached from under 0.6 x that
K = 6                # within 6 steps (3 minutes)
MIN_QUEUE_VEH = 3.0
THRESHOLD = 0.7      # alert at this probability (dev seeds: about half the alerts right, 2/3 of the starts caught)
EXTRA = ["c_rel", "fill_rel", "c_rise2_rel", "max4_rel"]


def cycle_means(fill: np.ndarray, run: np.ndarray) -> np.ndarray:
    """fill [T, N] -> c [T, N]: the mean of the last CYCLE_STEPS steps inside each run (NaN before; NaN fill = 0)."""
    T, N = fill.shape
    c = np.full((T, N), np.nan)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        cs = np.cumsum(np.vstack([np.zeros((1, N)), np.nan_to_num(fill[idx])]), 0)
        for i in range(CYCLE_STEPS - 1, len(idx)):
            c[idx[i]] = (cs[i + 1] - cs[i + 1 - CYCLE_STEPS]) / CYCLE_STEPS
    return c


def levels(c_train: np.ndarray, storage: np.ndarray) -> np.ndarray:
    return np.fmax(np.nanquantile(c_train, Q, axis=0), MIN_QUEUE_VEH / np.maximum(storage, 1.0))


def starts(c: np.ndarray, run: np.ndarray, level: np.ndarray) -> np.ndarray:
    """[T, N] True where a long queue starts (see the module doc)."""
    out = np.zeros(c.shape, bool)
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        cc = c[idx]
        for i in range(2 * CYCLE_STEPS, len(idx)):
            with np.errstate(invalid="ignore"):
                out[idx[i]] = ((cc[i] >= level) & (np.nanmax(cc[i - CYCLE_STEPS:i], 0) < level)
                               & (cc[i - CYCLE_STEPS] < LOW_FRAC * level))
    return out


def features(w: Window, storage, A_up, A_down, level: np.ndarray) -> np.ndarray:
    """The classifier's rows [S*N, d] from a window [S, L, N] (L >= CYCLE_STEPS + 2)."""
    X, _ = build(w, 1, storage, A_up, A_down, True, False)
    f = np.nan_to_num(w.fill)
    c_now = f[:, -CYCLE_STEPS:].mean(1)
    c_prev2 = f[:, -CYCLE_STEPS - 2:-2].mean(1)
    with np.errstate(invalid="ignore"):
        extra = np.stack([c_now / level, w.fill[:, -1] / level, (c_now - c_prev2) / level,
                          np.nanmax(w.fill[:, -CYCLE_STEPS:], 1) / level], -1)
    return np.hstack([X, extra.reshape(-1, len(EXTRA)).astype(np.float32)]), c_now


class Congestion:
    """A trained classifier (congestion_model.txt + congestion.json in a model folder)."""

    def __init__(self, folder):
        import lightgbm as lgb

        with open(os.path.join(folder, "congestion.json"), encoding="utf-8") as f:
            self.card = json.load(f)
        self.nodes = self.card["nodes"]
        self.level = np.array([self.card["level"][n] for n in self.nodes])
        self.k = int(self.card["k_steps"])
        self.threshold = float(self.card["threshold"])
        self.booster = lgb.Booster(model_file=os.path.join(folder, "congestion_model.txt"))

    def predict(self, w: Window, storage, A_up, A_down) -> tuple[np.ndarray, np.ndarray]:
        """-> (probability of a long queue starting within k steps [N], 0 where already over its level; cycle mean [N])
        for the last step of a one-window Window."""
        X, c_now = features(w, storage, A_up, A_down, self.level)
        p = self.booster.predict(X).reshape(c_now.shape)[0]
        return np.where(c_now[0] < self.level, p, 0.0), c_now[0]


# ---------------------------------------------------------------------------------------------- training
def score_alerts(prob_rows, t0, run, st_mask, k):
    """Alerts from probabilities (cooldown CYCLE_STEPS steps per approach) against the starts. prob_rows [R, N] for
    rows t0 [R]; st_mask [T, N] the starts. -> dict(alerts, right, starts, caught, median_lead_s)."""
    def evaluate(thr):
        last, alerts = {}, []
        for i, t in enumerate(t0):
            for j in np.flatnonzero(prob_rows[i] >= thr):
                if t - last.get((run[t], j), -99) < CYCLE_STEPS:
                    continue
                last[(run[t], j)] = t
                alerts.append((t, j))
        T = len(run)
        right = sum(any(t + d < T and run[t + d] == run[t] and st_mask[t + d, j] for d in range(1, k + 1))
                    for t, j in alerts)
        by = {}
        for t, j in alerts:
            by.setdefault(j, []).append(t)
        rows = set(int(t) for t in t0)
        on = [(t, j) for t in sorted(rows) for j in np.flatnonzero(st_mask[t])]
        caught, leads = 0, []
        for t, j in on:
            pre = [a for a in by.get(j, []) if t - k <= a < t and run[a] == run[t]]
            if pre:
                caught += 1
                leads.append((t - min(pre)) * 30)
        return {"threshold": thr, "alerts": len(alerts), "right": int(right), "starts": len(on), "caught": caught,
                "precision": round(right / len(alerts), 3) if alerts else None,
                "recall": round(caught / len(on), 3) if on else None,
                "median_lead_s": float(np.median(leads)) if leads else None}
    return evaluate


def train(dataset: str, model_dir: str, train_seeds, dev_seeds, n_in: int = 12, threshold: float = THRESHOLD):
    import lightgbm as lgb

    from .data import load_dataset
    from .features import gather, neighbor_mats

    ds = load_dataset(dataset)
    with np.load(dataset) as z:
        part = z["part"] if "part" in z else np.zeros(len(ds.t), int)
    adj = np.load(os.path.join(model_dir, "adjacency.npy"))
    A_up, A_down = neighbor_mats(adj)
    st = np.where(ds.storage >= 1.0, ds.storage, np.nan)
    fill = ds.x["halt"] / st
    c = cycle_means(fill, ds.run)
    train_m, dev_m = np.isin(ds.seed, list(train_seeds)), np.isin(ds.seed, list(dev_seeds))
    if set(train_seeds) & set(dev_seeds):
        raise ValueError("train and dev seeds overlap")
    level = levels(c[train_m], ds.storage)
    st_mask = starts(c, ds.run, level)
    T, N = fill.shape
    pos = np.zeros(T, int)
    for r in np.unique(ds.run):
        idx = np.flatnonzero(ds.run == r)
        pos[idx] = np.arange(len(idx))
    ok = (pos >= n_in - 1) & np.array([i + K < T and ds.run[i + K] == ds.run[i] for i in range(T)])
    t0 = np.flatnonzero(ok & (train_m | dev_m))
    X, c_now = features(gather(ds, t0, n_in), ds.storage, A_up, A_down, level)
    fut = np.zeros((len(t0), N), bool)
    for d in range(1, K + 1):
        fut |= st_mask[t0 + d]
    elig = c_now < level
    tr = (train_m[t0][:, None] & elig).reshape(-1)
    booster = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
                             bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, scale_pos_weight=5,
                             seed=1, verbose=-1),
                        lgb.Dataset(X[tr], fut.reshape(-1)[tr].astype(float)), num_boost_round=400)
    prob = np.where(elig, booster.predict(X).reshape(len(t0), N), 0.0)
    dev_scores = {}
    for p in np.unique(part):
        sel = dev_m[t0] & (part[t0] == p)
        ev = score_alerts(prob[sel], t0[sel], ds.run, st_mask, K)
        hours = float((dev_m & (part == p)).sum() * 30 / 3600)
        dev_scores[str(int(p))] = {"hours": hours, "by_threshold": [ev(x) for x in (0.5, threshold, 0.8)]}
    booster.save_model(os.path.join(model_dir, "congestion_model.txt"))
    card = {"what": "P(a long queue starts within k_steps) per approach (goflow_prediction/congestion.py)",
            "nodes": ds.nodes, "level": {n: round(float(v), 5) for n, v in zip(ds.nodes, level)},
            "cycle_steps": CYCLE_STEPS, "q": Q, "low_frac": LOW_FRAC, "k_steps": K, "dt_s": 30.0,
            "threshold": threshold, "train_seeds": sorted(set(int(s) for s in train_seeds)),
            "dev_seeds": sorted(set(int(s) for s in dev_seeds)), "dataset": os.path.abspath(dataset),
            "dataset_parts": [{k: m.get(k) for k in ("scale", "accidents", "clock_speed")}
                              for m in ds.meta.get("parts", [ds.meta])],
            "train_rows": int(tr.sum()), "train_positives": int(fut.reshape(-1)[tr].sum()),
            "dev_scores_by_part": dev_scores}
    with open(os.path.join(model_dir, "congestion.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1)
    return card


def main(argv=None):
    from .train_lgbm import seeds_arg

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="a forecast model folder (train_lgbm.py): written next to it")
    ap.add_argument("--train-seeds", required=True)
    ap.add_argument("--dev-seeds", required=True)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args(argv)
    card = train(a.dataset, a.model, seeds_arg(a.train_seeds), seeds_arg(a.dev_seeds), threshold=a.threshold)
    for p, s in card["dev_scores_by_part"].items():
        for ev in s["by_threshold"]:
            print(f"part {p} ({s['hours']:.1f} h) threshold {ev['threshold']}: alerts {ev['alerts']} right {ev['right']}"
                  f" ({ev['precision']}), starts {ev['starts']} caught {ev['caught']} ({ev['recall']}), "
                  f"median lead {ev['median_lead_s']} s")


if __name__ == "__main__":
    main()
