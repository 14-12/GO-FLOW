"""Test fixtures only: a toy six-approach corridor, NOT Town05 data and not a stand-in for any model.

Approaches a-1 -> a-2 -> ... -> f-1 (traffic goes left to right). Each run has a signal-cycle sawtooth on every queue,
a demand that follows a curve, and a few jams that start at the last approach and spill back upstream one approach
every 2 steps, as a queue does. Enough structure for a model to learn and for the protocol to be exercised.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goflow_prediction.data import Dataset, save_dataset  # noqa: E402

NODES = ["a-1", "b-1", "c-1", "d-1", "e-1", "f-1"]
DT = 30.0
HORIZONS = (60, 120, 300)


def toy_dataset(seeds, steps=160, goflow=False) -> Dataset:
    N = len(NODES)
    rows = {k: [] for k in ("t", "run", "seed", "tod", "dem", "dem_at", "halt", "veh", "speed", "occ", "green", "gf")}
    storage = np.full(N, 20.0)
    for r, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        tt = np.arange(steps) * DT + DT
        tod = (16 * 3600 + tt * 60) % 86400
        demand = 0.5 + 0.4 * np.sin(tt / 900.0 + seed)
        base = 0.15 + 0.35 * demand[:, None] + 0.1 * np.sin(2 * np.pi * tt / 115.0)[:, None] + np.zeros((1, N))
        jam = np.zeros((steps, N))
        for start in rng.choice(np.arange(20, steps - 40), size=3, replace=False):
            for j in range(N - 1, -1, -1):
                s0 = start + 2 * (N - 1 - j)
                jam[s0:s0 + 12, j] = 1.0
        fill = np.clip(base + 0.8 * jam + rng.normal(0, 0.03, (steps, N)), 0, 1.2)
        halt = fill * storage
        veh = halt + 3 + rng.normal(0, 0.5, (steps, N))
        speed = np.clip(12 * (1 - fill), 0, None)
        occ = np.clip(fill * 0.6, 0, 1)
        green = np.clip(0.5 + 0.2 * np.sin(2 * np.pi * tt / 115.0)[:, None] + np.zeros((1, N)), 0, 1)
        dem_at = np.stack([0.5 + 0.4 * np.sin((tt + h) / 900.0 + seed) for h in HORIZONS], 1)
        for k, v in (("t", tt), ("run", np.full(steps, r)), ("seed", np.full(steps, seed)), ("tod", tod),
                     ("dem", demand), ("dem_at", dem_at), ("halt", halt), ("veh", veh), ("speed", speed),
                     ("occ", occ), ("green", green), ("gf", np.roll(occ, -4, axis=0))):
            rows[k].append(v)
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    meta = dict(dt_s=DT, horizons_s=list(HORIZONS), source="exact", controller="toy", note="test fixture")
    return Dataset(nodes=list(NODES), storage=storage, t=cat["t"], run=cat["run"], seed=cat["seed"],
                   tod_s=cat["tod"], demand=cat["dem"], demand_at=cat["dem_at"],
                   x={c: cat[c] for c in ("halt", "veh", "speed", "occ", "green")}, meta=meta,
                   goflow=cat["gf"] if goflow else None)


def toy_graph(path) -> str:
    rows = [dict(**{"from": a, "to": b}, cost=150.0) for a, b in zip(NODES, NODES[1:])]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    ds = toy_dataset(list(range(101, 107)) + [11, 12] + [21])
    save_dataset(ds, d / "toy.npz")
    toy_graph(d / "distance.csv")
    return d


@pytest.fixture(scope="session")
def trained(toy, tmp_path_factory):
    from goflow_prediction.train_lgbm import train

    out = tmp_path_factory.mktemp("models") / "v-test"
    train(str(toy / "toy.npz"), str(toy / "distance.csv"), str(out), list(range(101, 107)), [11, 12],
          rounds=200, version="v-test", test_seeds=[21])
    return out


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
