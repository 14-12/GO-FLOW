"""Explicit no-leakage checks (from DSO4 src/leakage.py, now on seeds instead of dates). AssertionError if broken."""
from __future__ import annotations

import datetime
import os

import numpy as np

from .config import TimeBase


def check_seed_splits(train, dev, test) -> None:
    a, b, c = set(train), set(dev), set(test)
    assert not (a & b or a & c or b & c), "a seed is in two splits"
    print(f"[ok] seeds: train {sorted(a)}, dev {sorted(b)}, test {sorted(c)}, disjoint")


def check_windows(ds, t0: np.ndarray, tb: TimeBase, seeds) -> None:
    x_first = t0 - tb.n_in + 1
    y_last = t0 + tb.max_steps
    assert (ds.run[x_first] == ds.run[t0]).all() and (ds.run[y_last] == ds.run[t0]).all(), "window crosses two runs"
    assert np.isin(ds.seed[t0], list(seeds)).all(), "window outside its split"
    print(f"[ok] {len(t0)} windows, each inside one run of seeds {sorted(set(seeds))}; input strictly before target")


class TestGuard:
    """The test seeds open once: a lock file is written on first access."""

    __test__ = False  # not a pytest class

    def __init__(self, lock_path):
        self.lock_path = lock_path

    def open(self, reason: str) -> bool:
        if os.path.exists(self.lock_path):
            raise RuntimeError(f"test already opened (see {self.lock_path}); deleting the lock would be cheating")
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_path)), exist_ok=True)
        with open(self.lock_path, "w", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {reason}\n")
        return True
