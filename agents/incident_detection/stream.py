"""Omar's accident agent on a live camera: one window out as soon as it closes (Town05 adapter, 11 Nov 2026).

His `process_frames` returns only when its frames run out, restarts the tracker and the clock on every call, and
counts picture times instead of reading them. In the town a camera never ends. `StreamingWindows` keeps one camera's
state for the whole run and, when a 5 s window closes, calls HIS functions, unchanged, on what that window needs:

    aggregate_windows        (src/video_processing.py)   the window's 13 statistics
    add_temporal_features    (src/feature_engineering.py) inside IncidentDetector.predict_windows: delta, roll3
    IncidentDetector         (src/inference.py)          his v1 Random Forest and mean3 >= threshold
    window_crash_features    (evaluation/crash_features.py)  the crash moment: hard stops, with contact

What it keeps so that each window comes out exactly as his batch path would give it for the whole stream:
  * every track's last HISTORY detections (his speeds look 1 row back, the crash moment 10 rows back);
  * the last KEEP windows' statistics (delta needs 1 window back, roll3 2, mean3 two more scores).
tests/test_stream.py checks this against his functions run once on the whole stream.

Times are the simulation's seconds of the picture shown (the caller's), windows are aligned to multiples of 5 s of
that clock, so windows of different cameras line up. Nothing here reads the simulation: pictures in, verdicts out.
"""
from __future__ import annotations

import hashlib
import math
import sys
from collections import deque
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))           # `agents.incident_detection.src`, as his code imports itself
sys.path.insert(0, str(HERE / "evaluation"))        # his crash_features imports track_features by name

from agents.incident_detection.src.feature_engineering import CURRENT_COLUMNS  # noqa: E402
from agents.incident_detection.src.inference import IncidentDetector  # noqa: E402
from agents.incident_detection.src.video_processing import CATEGORIES, WINDOW_SECONDS, aggregate_windows  # noqa: E402
from crash_features import window_crash_features  # noqa: E402

HISTORY = 10          # detections kept per track: crash_features' speed (5 rows) and speed one second before (5 more)
KEEP = 6              # windows kept per camera: mean3 of the last 3 scores, each needing roll3 (2 back) and delta
FORGET_S = 60.0       # a track not seen for this long is forgotten
IMPACT = ("hard_stops", "contact_hard_stops", "max_speed_drop")
DET_KEYS = ("track_id", "class_id", "category", "confidence", "center_x", "center_y", "bbox_height", "bbox_area")


def model_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def window_of(t: float) -> float:
    """The start of the 5 s window holding time t (rounded as his aggregate_windows rounds)."""
    return round(math.floor(round(t, 6) / WINDOW_SECONDS) * WINDOW_SECONDS, 3)


class StreamingWindows:
    """One camera. add() every processed picture in time order; it returns the windows that closed."""

    def __init__(self, camera: str, detector: IncidentDetector, model_version: str = "v1", detector_name: str = ""):
        self.camera, self.det, self.version, self.detector_name = camera, detector, model_version, detector_name
        self.t0: float | None = None          # the first window's start (sim s): his "video second" 0
        self.cur: float | None = None         # the open window's start (sim s)
        self.frames: list[dict] = []          # the open window's frame rows
        self.dets: list[dict] = []            # the open window's detection rows
        self.tracks: dict[int, deque] = {}    # track id -> its last HISTORY detections before the open window
        self.seen: dict[int, float] = {}      # track id -> last time seen
        self.rows: deque = deque(maxlen=KEEP)  # the last windows' statistics (his CURRENT_COLUMNS + times)
        self.stats = {"pictures": 0, "late": 0, "windows": 0, "empty_windows": 0}
        self.last_t: float | None = None

    def add(self, t: float, frame: dict, dets: list[dict]) -> list[dict]:
        """t: the picture's time (sim s); frame: vehicle_count, bbox_coverage, mean_detection_confidence (his
        result_rows' frame row without the time); dets: his result_rows' detection rows without the time."""
        if self.last_t is not None and t <= self.last_t:
            self.stats["late"] += 1           # out of order or repeated: dropped (his path never sees one)
            return []
        self.last_t = t
        self.stats["pictures"] += 1
        w = window_of(t)
        out = []
        if self.cur is None:
            self.t0 = self.cur = w
        while w > self.cur + 1e-9:            # the open window (and any window with no picture) closes
            out.append(self.close(self.cur + WINDOW_SECONDS))
            self.cur = round(self.cur + WINDOW_SECONDS, 3)
        rel = t - self.t0
        self.frames.append(dict(frame, source_time_sec=rel))
        self.dets.extend(dict(d, source_time_sec=rel) for d in dets)
        return out

    def flush(self, end_t: float | None = None) -> list[dict]:
        """The open window, closed now (end of the run). end_t: when it ended (sim s), default its last picture."""
        if self.cur is None or not self.frames:
            return []
        end = self.last_t if end_t is None else end_t
        out = [self.close(min(self.cur + WINDOW_SECONDS, max(end, self.cur)))]
        self.cur = round(self.cur + WINDOW_SECONDS, 3)
        return out

    # -- one window -------------------------------------------------------------------------------

    def close(self, end_sim: float) -> dict:
        start_rel = round(self.cur - self.t0, 3)
        base = start_rel - WINDOW_SECONDS     # his functions see [prev window][this window]: this one is [5, 10)
        history = [dict(d) for q in self.tracks.values() for d in q]
        frames = [{"source_time_sec": 0.0, "vehicle_count": 0, "bbox_coverage": 0.0,
                   "mean_detection_confidence": 0.0}]  # a picture in the window before, so his code always has one
        frames += [dict(f, source_time_sec=f["source_time_sec"] - base) for f in self.frames]
        dets = [dict(d, source_time_sec=d["source_time_sec"] - base) for d in history + self.dets]
        own = [dict(d, source_time_sec=d["source_time_sec"] - base) for d in self.dets]
        duration = round(end_sim - self.cur, 6) + WINDOW_SECONDS
        win = aggregate_windows(frames, own_and_prev(own, history, base), duration)
        row = win[win.window_start_sec == WINDOW_SECONDS].iloc[0].to_dict()
        impact = {k: 0.0 for k in IMPACT}
        if dets:
            cf = window_crash_features(pd.DataFrame(dets), duration)
            hit = cf[cf.window_start_sec == WINDOW_SECONDS]
            if len(hit):
                impact = {k: float(hit.iloc[0][k]) for k in IMPACT}
        row.update(window_start_sec=start_rel, window_end_sec=round(start_rel + duration - WINDOW_SECONDS, 3))
        self.rows.append({k: row[k] for k in ("window_start_sec", "window_end_sec") + CURRENT_COLUMNS})
        verdict = self.det.predict_windows(pd.DataFrame(list(self.rows)), self.camera)[-1]
        verdict.update(camera=self.camera, sim_start_s=round(self.cur, 3), sim_end_s=round(end_sim, 3),
                       pictures=len(self.frames), impact=impact,
                       impact_alarm=bool(impact["contact_hard_stops"] >= 1),
                       model={"version": self.version, "detector": self.detector_name})
        self.stats["windows"] += 1
        if not self.frames:
            self.stats["empty_windows"] += 1
        # this window's detections become history
        for d in self.dets:
            q = self.tracks.setdefault(d["track_id"], deque(maxlen=HISTORY))
            q.append(d)
            self.seen[d["track_id"]] = d["source_time_sec"]
        old = [k for k, s in self.seen.items() if start_rel + WINDOW_SECONDS - s > FORGET_S]
        for k in old:
            self.tracks.pop(k, None)
            self.seen.pop(k, None)
        self.frames, self.dets = [], []
        return verdict


def own_and_prev(own: list[dict], history: list[dict], base: float) -> list[dict]:
    """His aggregate_windows needs each track's previous detection (its speed looks one row back): the last one
    of each track's history, then the window's own."""
    last = {}
    for d in history:
        last[d["track_id"]] = d
    return [dict(d, source_time_sec=d["source_time_sec"] - base) for d in last.values()] + own


def categories_of(class_map: dict, names: dict) -> dict:
    """class id -> his category, as DetectorConfig.class_ids (kept here for the rows' check)."""
    lookup = {str(n).lower(): int(i) for i, n in dict(names).items()}
    return {lookup[n.lower()]: c for n, c in class_map.items() if n.lower() in lookup and c in CATEGORIES}
