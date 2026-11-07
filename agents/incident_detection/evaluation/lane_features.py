"""Stopped-in-lane evidence: a vehicle standing for a long time where traffic normally drives.

Town05 cameras see parked cars (or objects the detector takes for cars) standing in the same
place for most of a recording, so "stopped for a long time" alone is not a crash sign. A lane
map tells the two apart: a grid of cells, each counting the distinct *moving* vehicles that
have passed through it. A stationary spot counts only if its cells held at least LANE_TRACKS
moving vehicles *before the spot appeared*. The map is built causally from the recording itself,
plus an optional prior from other recordings of the same camera (training runs only).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from track_features import (MOVING_SPEED, SPOT_GAP_S, SPOT_RADIUS, STATIONARY_SPEED, WINDOW_SECONDS,
                            add_speed)

CELL_PX = 20
LANE_TRACKS = 3
LANE_FEATURES = ("lane_spot_age_s", "lane_spots_over_60s", "lane_spots_over_120s")


def cells_around(x, y, h):
    r = max(1, int(round(0.5 * h / CELL_PX)))
    cx, cy = int(x // CELL_PX), int(y // CELL_PX)
    return [(cx + i, cy + j) for i in range(-r, r + 1) for j in range(-r, r + 1)]


def lane_prior(detections_list):
    """Distinct moving tracks per cell, from other recordings of the same camera."""
    prior = defaultdict(int)
    for n, det in enumerate(detections_list):
        det = add_speed(det)
        moving = det[det.speed > MOVING_SPEED]
        seen = set()
        for x, y, track in zip(moving.center_x, moving.center_y, moving.track_id):
            seen.add(((int(x // CELL_PX), int(y // CELL_PX)), (n, track)))
        for cell, _ in seen:
            prior[cell] += 1
    return dict(prior)


def window_lane_features(det, duration, prior=None):
    starts = np.round(np.arange(0, duration, WINDOW_SECONDS), 3)
    rows = {s: {"window_start_sec": float(s), **{k: 0.0 for k in LANE_FEATURES}} for s in starts}
    if det.empty:
        return pd.DataFrame(rows.values())
    det = add_speed(det)
    prior = prior or {}
    online = defaultdict(set)           # cell -> moving track ids seen so far in this recording
    spots = []                          # [x, y, h, first_seen, last_seen, in_lane]
    per_window = defaultdict(list)
    for t, frame in det.groupby("source_time_sec", sort=True):
        spots = [s for s in spots if t - s[4] <= SPOT_GAP_S]
        x, y, h = frame.center_x.to_numpy(), frame.center_y.to_numpy(), frame.bbox_height.to_numpy()
        speed, tracks = frame.speed.to_numpy(), frame.track_id.to_numpy()
        for i in np.flatnonzero(speed < STATIONARY_SPEED):
            best, best_d = None, SPOT_RADIUS * h[i]
            for s in spots:
                d = np.hypot(s[0] - x[i], s[1] - y[i])
                if d <= best_d and s[4] < t:
                    best, best_d = s, d
            if best is None:
                cells = cells_around(x[i], y[i], h[i])
                traffic = len(set().union(*(online.get(c, set()) for c in cells)) - {tracks[i]})
                traffic += max((prior.get(c, 0) for c in cells), default=0)
                spots.append([x[i], y[i], h[i], t, t, traffic >= LANE_TRACKS])
            else:
                best[0], best[1], best[2], best[4] = 0.8 * best[0] + 0.2 * x[i], 0.8 * best[1] + 0.2 * y[i], h[i], t
        for i in np.flatnonzero(speed > MOVING_SPEED):
            online[(int(x[i] // CELL_PX), int(y[i] // CELL_PX))].add(tracks[i])
        ages = [s[4] - s[3] for s in spots if s[4] == t and s[5]]
        per_window[np.round(np.floor(t / WINDOW_SECONDS) * WINDOW_SECONDS, 3)].append(ages)
    for start, frames_ages in per_window.items():
        if start not in rows:
            continue
        flat = [a for ages in frames_ages for a in ages]
        last = frames_ages[-1]
        rows[start].update({"lane_spot_age_s": float(max(flat)) if flat else 0.0,
                            "lane_spots_over_60s": float(sum(a >= 60 for a in last)),
                            "lane_spots_over_120s": float(sum(a >= 120 for a in last))})
    return pd.DataFrame(rows.values())
