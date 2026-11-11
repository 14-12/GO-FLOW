"""Stopped-vehicle evidence from tracked detections, per five-second window.

A crash leaves vehicles standing in one place for minutes while other traffic goes around them;
a red light stops a whole queue for well under a minute and then releases it. Track ids break
when a vehicle is hidden, so the evidence follows *spots*: places occupied by a stationary
vehicle, matched across frames by position and box size, regardless of track id.

All features are causal: a window only uses detections up to its own end.
"""
import numpy as np
import pandas as pd

WINDOW_SECONDS = 5.0
STATIONARY_SPEED = 0.1      # bbox heights per second, measured over about one second
MOVING_SPEED = 0.3
SPOT_RADIUS = 0.6           # bbox heights: a stationary detection this close continues a spot
SPOT_GAP_S = 4.0            # a spot survives this long unseen (occlusion by passing traffic)
PASS_RADIUS = 4.0           # bbox heights: a moving vehicle this close to a spot passes it
OLD_SPOT_S = (30, 60, 120)

TRACK_FEATURES = ("max_spot_age_s", "spots_over_30s", "spots_over_60s", "spots_over_120s",
                  "moving_tracks", "passing_old_spot", "stationary_tracks", "stationary_share")


def add_speed(det):
    """Speed over about one second (5 frames at 5 fps), in bbox heights per second."""
    det = det.sort_values(["track_id", "source_time_sec"]).copy()
    g = det.groupby("track_id")
    dx, dy = det.center_x - g.center_x.shift(5), det.center_y - g.center_y.shift(5)
    dt = det.source_time_sec - g.source_time_sec.shift(5)
    det["speed"] = np.sqrt(dx ** 2 + dy ** 2) / dt.where(dt > 0) / det.bbox_height.clip(lower=1)
    return det.sort_values(["source_time_sec", "track_id"]).reset_index(drop=True)


def window_track_features(det, duration):
    """One row per five-second window (same windows as aggregate_windows)."""
    starts = np.round(np.arange(0, duration, WINDOW_SECONDS), 3)
    rows = {s: {"window_start_sec": float(s), **{k: 0.0 for k in TRACK_FEATURES}} for s in starts}
    if det.empty:
        return pd.DataFrame(rows.values())
    det = add_speed(det)
    spots = []  # each: [x, y, h, first_seen, last_seen]
    per_window = {}
    for t, frame in det.groupby("source_time_sec", sort=True):
        spots = [s for s in spots if t - s[4] <= SPOT_GAP_S]
        x, y, h = frame.center_x.to_numpy(), frame.center_y.to_numpy(), frame.bbox_height.to_numpy()
        speed = frame.speed.to_numpy()
        for i in np.flatnonzero(speed < STATIONARY_SPEED):
            best, best_d = None, SPOT_RADIUS * h[i]
            for s in spots:
                d = np.hypot(s[0] - x[i], s[1] - y[i])
                if d <= best_d and s[4] < t:
                    best, best_d = s, d
            if best is None:
                spots.append([x[i], y[i], h[i], t, t])
            else:
                best[0], best[1], best[2], best[4] = 0.8 * best[0] + 0.2 * x[i], 0.8 * best[1] + 0.2 * y[i], h[i], t
        live = [s for s in spots if s[4] == t]
        ages = np.array([s[4] - s[3] for s in live]) if live else np.zeros(0)
        moving = speed > MOVING_SPEED
        passing = set()
        for s, age in zip(live, ages):
            if age >= OLD_SPOT_S[1]:
                near = np.hypot(x - s[0], y - s[1]) <= PASS_RADIUS * s[2]
                passing.update(frame.track_id.to_numpy()[moving & near].tolist())
        w = per_window.setdefault(np.round(np.floor(t / WINDOW_SECONDS) * WINDOW_SECONDS, 3),
                                  {"ages": [], "moving": set(), "passing": set(), "stationary": set(), "tracks": set()})
        w["ages"].append(ages)
        w["moving"].update(frame.track_id.to_numpy()[moving].tolist())
        w["passing"].update(passing)
        w["stationary"].update(frame.track_id.to_numpy()[speed < STATIONARY_SPEED].tolist())
        w["tracks"].update(frame.track_id.tolist())
    for start, w in per_window.items():
        if start not in rows:
            continue
        ages = np.concatenate(w["ages"]) if w["ages"] else np.zeros(0)
        last = w["ages"][-1] if w["ages"] else np.zeros(0)
        rows[start].update({
            "max_spot_age_s": float(ages.max()) if len(ages) else 0.0,
            **{f"spots_over_{a}s": float((last >= a).sum()) for a in OLD_SPOT_S},
            "moving_tracks": float(len(w["moving"])), "passing_old_spot": float(len(w["passing"])),
            "stationary_tracks": float(len(w["stationary"])),
            "stationary_share": len(w["stationary"]) / len(w["tracks"]) if w["tracks"] else 0.0,
        })
    return pd.DataFrame(rows.values())
