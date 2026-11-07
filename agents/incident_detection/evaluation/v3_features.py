"""Per-window data set for the v3 recordings: v1 features + stopped-spot + crash + emergency-vehicle evidence.

The split, cameras and runs come from v3_split.json and results/v3_recording_progress.json, both fixed by the
recording step. Each window gets a target:
    1   a counted crash is in this camera's picture (near, impact on screen, after the first minute);
    0   normal traffic, jams included;
   -1   another wreck is in the picture (far away or from the simulator's first minute): an alarm there is
        neither right nor wrong, so these windows count neither as events nor as normal time.
All features are causal: a window only uses detections up to its own end.

    python agents/incident_detection/evaluation/v3_features.py train
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
AGENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(AGENT_DIR.parents[1]))
sys.path.insert(0, str(EVAL_DIR))
from agents.incident_detection.src.feature_engineering import FEATURE_COLUMNS, add_temporal_features  # noqa: E402
from crash_features import CRASH_FEATURES, window_crash_features  # noqa: E402
from track_features import TRACK_FEATURES, WINDOW_SECONDS, window_track_features  # noqa: E402

TRACKS = AGENT_DIR / "data" / "town05_v3" / "carla-yolo26n-t05-tracks"
CACHE = AGENT_DIR / "data" / "town05_v3" / "features"
RENDER_LAG_S = 2.2
DATA = "v3"
# v4 (recorded from 6 Nov): frames.csv already gives the SUMO moment each picture shows, so no lag to take off
LAGS = {"v3": 2.2, "v4": 0.0}


def configure(data):
    """Point the module at one recorded data set (v3 or v4)."""
    global TRACKS, CACHE, RENDER_LAG_S, DATA
    DATA, RENDER_LAG_S = data, LAGS[data]
    TRACKS = AGENT_DIR / "data" / f"town05_{data}" / "carla-yolo26n-t05-tracks"
    CACHE = AGENT_DIR / "data" / f"town05_{data}" / "features"

AMBULANCE_LIGHTS, POLICE_LIGHTS = 5, 7
EMERGENCY_CLASSES = (AMBULANCE_LIGHTS, POLICE_LIGHTS)
EMERGENCY_STOPPED_SPEED = 0.1      # bbox heights per second, over about one second
MIN_EMERGENCY_CONFIDENCE = 0.4
EMERGENCY_FEATURES = ("emergency_lights_frames", "emergency_lights_stopped_frames")
MEMORY_WINDOWS = {"60s": 12, "180s": 36}
MEMORY_OF = ("contact_hard_stops", "hard_stops", "passing_same_direction", "passing_any_direction",
             "emergency_lights_frames", "emergency_lights_stopped_frames")
MEMORY_FEATURES = tuple(f"{f}_sum_{name}" for f in MEMORY_OF for name in MEMORY_WINDOWS)
EVIDENCE_FEATURES = tuple(CRASH_FEATURES) + tuple(TRACK_FEATURES) + EMERGENCY_FEATURES + MEMORY_FEATURES
ALL_FEATURES = tuple(FEATURE_COLUMNS) + EVIDENCE_FEATURES


def runs_of(split):
    progress = json.loads((EVAL_DIR / "results" / f"{DATA}_recording_progress.json").read_text(encoding="utf-8"))
    return [r for r in progress["runs"] if r["split"] == split]


def emergency_features(det, duration):
    """Frames in each window showing a vehicle with its lights on, and those where it stands still."""
    starts = np.round(np.arange(0, duration, WINDOW_SECONDS), 3)
    out = pd.DataFrame({"window_start_sec": starts.astype(float)})
    for name in EMERGENCY_FEATURES:
        out[name] = 0.0
    lit = det[det.class_id.isin(EMERGENCY_CLASSES) & (det.confidence >= MIN_EMERGENCY_CONFIDENCE)].copy()
    if lit.empty:
        return out
    lit = lit.sort_values(["track_id", "source_time_sec"])
    g = lit.groupby("track_id")
    dt = lit.source_time_sec - g.source_time_sec.shift(5)
    moved = np.hypot(lit.center_x - g.center_x.shift(5), lit.center_y - g.center_y.shift(5))
    lit["stopped"] = (moved / dt.where(dt > 0) / lit.bbox_height.clip(lower=1)) < EMERGENCY_STOPPED_SPEED
    lit["window_start_sec"] = np.round(np.floor(lit.source_time_sec / WINDOW_SECONDS) * WINDOW_SECONDS, 3)
    counts = lit.groupby("window_start_sec").agg(frames=("source_time_sec", "nunique"),
                                                 stopped=("stopped", "sum"))
    out = out.set_index("window_start_sec")
    out.loc[counts.index.intersection(out.index), "emergency_lights_frames"] = counts.frames.astype(float)
    out.loc[counts.index.intersection(out.index), "emergency_lights_stopped_frames"] = counts.stopped.astype(float)
    return out.reset_index()


def add_memory(windows):
    """How much of the short-lived evidence the last minute or three held."""
    for feature in MEMORY_OF:
        for name, size in MEMORY_WINDOWS.items():
            windows[f"{feature}_sum_{name}"] = windows[feature].rolling(size, min_periods=1).sum()
    return windows


def targets(windows, camera, near, visible):
    """The target per window (see the module's docstring). near: incident ids counted for this camera."""
    shown = (windows.sim_start_s + windows.sim_end_s) / 2 - RENDER_LAG_S
    target = np.zeros(len(windows), dtype=int)
    incident = np.full(len(windows), -1)
    for v in visible:
        if v["camera"] != camera:
            continue
        inside = ((shown >= v["from_sim_s"]) & (shown <= v["to_sim_s"])).to_numpy()
        target[inside] = 1 if v["incident"] in near else -1
        incident[inside] = v["incident"]
    return target, incident


def recording_features(path, near, visible):
    windows = add_temporal_features(pd.read_csv(path))
    tracks = pd.read_csv(path.with_name(path.name.replace(".csv", ".tracks.csv.gz")))
    duration = float(windows.window_end_sec.max())
    for feats in (window_crash_features(tracks, duration), window_track_features(tracks, duration),
                  emergency_features(tracks, duration)):
        windows = windows.merge(feats, on="window_start_sec", how="left", validate="1:1")
    windows = add_memory(windows.sort_values("window_start_sec").reset_index(drop=True))
    windows["target"], windows["incident"] = targets(windows, windows.camera.iloc[0], near, visible)
    windows["stream"] = windows.run + "|" + windows.camera
    return windows


def build(split, incident_clips, refresh=False):
    """Every recording of the split's runs, cached per split."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{split}.csv.gz"
    if cache.is_file() and not refresh:
        return pd.read_csv(cache)
    frames = []
    for run in runs_of(split):
        data = json.loads((incident_clips / run["folder"] / "incidents.json").read_text(encoding="utf-8"))
        for path in sorted(TRACKS.glob(f"{run['folder']}__*.csv")):
            camera = path.name.split("__")[1].removesuffix(".csv").replace("_", " ")
            near = {c["incident"] for c in run["near_crashes"] if c["camera"] == camera}
            frames.append(recording_features(path, near, data["visible"]))
            print(path.name, len(frames[-1]), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    configure(sys.argv[3] if len(sys.argv) > 3 else "v3")
    clips = Path(sys.argv[2]) if len(sys.argv) > 2 else AGENT_DIR.parents[2] / "Urban Traffic CARLA/out/incident_clips"
    built = build(sys.argv[1], clips, refresh=True)
    print(built.groupby("target").size().to_dict(), built.stream.nunique(), "recordings")
