"""Accident-specific evidence from tracked detections, per five-second window.

Two signs that separate an accident from a traffic jam:
  crash moment  a vehicle goes from driving to standing within about a second while its box
                touches another vehicle's box (jam traffic slows down over several seconds);
  avoidance     moving vehicles pass close to a vehicle that has stood for a while, travelling
                the way it was travelling before it stopped (in a jam nobody gets past).
All features are causal. Speeds are in bbox heights per second, measured over about one second.
"""
from collections import deque

import numpy as np
import pandas as pd

from track_features import SPOT_GAP_S, SPOT_RADIUS, STATIONARY_SPEED, WINDOW_SECONDS

FPS = 5
HARD_FROM_SPEED = 0.8       # driving speed one second before the stop
MOVING_SPEED = 0.3
PASS_DISTANCE = 1.5         # bbox heights of the stopped vehicle
SAME_DIRECTION_COS = 0.7
AVOID_MIN_AGE_S = 20.0
CRASH_FEATURES = ("hard_stops", "contact_hard_stops", "max_speed_drop",
                  "crash_born_spot_age_s", "passing_same_direction", "passing_any_direction")


def add_motion(det):
    """Velocity over about one second (5 frames), its speed, and the speed one second earlier."""
    det = det.sort_values(["track_id", "source_time_sec"]).copy()
    g = det.groupby("track_id")
    dt = det.source_time_sec - g.source_time_sec.shift(FPS)
    h = det.bbox_height.clip(lower=1)
    det["vx"] = (det.center_x - g.center_x.shift(FPS)) / dt.where(dt > 0) / h
    det["vy"] = (det.center_y - g.center_y.shift(FPS)) / dt.where(dt > 0) / h
    det["speed"] = np.hypot(det.vx, det.vy)
    det["speed_1s_ago"] = det.groupby("track_id").speed.shift(FPS)
    det["width"] = det.bbox_area / h
    return det.sort_values(["source_time_sec", "track_id"]).reset_index(drop=True)


def touching(frame, i):
    """Whether detection i's box overlaps another detection's box in the same frame."""
    x, y = frame.center_x.to_numpy(), frame.center_y.to_numpy()
    w, h = frame.width.to_numpy(), frame.bbox_height.to_numpy()
    others = np.arange(len(frame)) != i
    overlap = (np.abs(x - x[i]) < (w + w[i]) / 2) & (np.abs(y - y[i]) < (h + h[i]) / 2)
    return bool((overlap & others).any())


def window_crash_features(det, duration):
    starts = np.round(np.arange(0, duration, WINDOW_SECONDS), 3)
    rows = {s: {"window_start_sec": float(s), **{k: 0.0 for k in CRASH_FEATURES}} for s in starts}
    if det.empty:
        return pd.DataFrame(rows.values())
    det = add_motion(det)
    heading = {}                 # track id -> last driving direction (unit vector)
    spots = []                   # [x, y, h, first_seen, last_seen, direction or None, crash_born]
    per_window = {}
    for t, frame in det.groupby("source_time_sec", sort=True):
        frame = frame.reset_index(drop=True)
        spots = [s for s in spots if t - s[4] <= SPOT_GAP_S]
        x, y, h = frame.center_x.to_numpy(), frame.center_y.to_numpy(), frame.bbox_height.to_numpy()
        vx, vy, speed = frame.vx.to_numpy(), frame.vy.to_numpy(), frame.speed.to_numpy()
        before, tracks = frame.speed_1s_ago.to_numpy(), frame.track_id.to_numpy()
        w = per_window.setdefault(np.round(np.floor(t / WINDOW_SECONDS) * WINDOW_SECONDS, 3), {
            "hard": set(), "contact": set(), "drop": 0.0, "age": 0.0, "same": set(), "any": set()})
        for i in range(len(frame)):
            if speed[i] > MOVING_SPEED and np.isfinite(speed[i]):
                heading[tracks[i]] = np.array([vx[i], vy[i]]) / speed[i]
            if np.isfinite(before[i]) and np.isfinite(speed[i]):
                w["drop"] = max(w["drop"], float(before[i] - speed[i]))
        hard = np.flatnonzero((speed < STATIONARY_SPEED) & (before > HARD_FROM_SPEED))
        contact = {tracks[i] for i in hard if touching(frame, i)}
        w["hard"].update(tracks[hard].tolist())
        w["contact"].update(contact)
        for i in np.flatnonzero(speed < STATIONARY_SPEED):
            best, best_d = None, SPOT_RADIUS * h[i]
            for s in spots:
                d = np.hypot(s[0] - x[i], s[1] - y[i])
                if d <= best_d and s[4] < t:
                    best, best_d = s, d
            if best is None:
                spots.append([x[i], y[i], h[i], t, t, heading.get(tracks[i]), tracks[i] in contact])
            else:
                best[0], best[1], best[2], best[4] = 0.8 * best[0] + 0.2 * x[i], 0.8 * best[1] + 0.2 * y[i], h[i], t
                best[6] = best[6] or tracks[i] in contact
                if best[5] is None:
                    best[5] = heading.get(tracks[i])
        moving = np.flatnonzero(speed > MOVING_SPEED)
        for s in spots:
            if s[4] != t:
                continue
            if s[6]:
                w["age"] = max(w["age"], t - s[3])
            if t - s[3] < AVOID_MIN_AGE_S:
                continue
            for i in moving:
                if np.hypot(x[i] - s[0], y[i] - s[1]) > PASS_DISTANCE * s[2]:
                    continue
                w["any"].add(tracks[i])
                if s[5] is not None and (vx[i] * s[5][0] + vy[i] * s[5][1]) / speed[i] > SAME_DIRECTION_COS:
                    w["same"].add(tracks[i])
    for start, w in per_window.items():
        if start in rows:
            rows[start].update({"hard_stops": float(len(w["hard"])), "contact_hard_stops": float(len(w["contact"])),
                                "max_speed_drop": w["drop"], "crash_born_spot_age_s": w["age"],
                                "passing_same_direction": float(len(w["same"])),
                                "passing_any_direction": float(len(w["any"]))})
    return pd.DataFrame(rows.values())
