"""Extract five-second v1 windows from the full Town05 recordings, one CSV per recording.

Reads out/incident_clips/<run>/<camera>.mp4 with its <camera>.frames.csv, run.json and
incidents.json (Urban Traffic CARLA layout). Each window gets the run, the camera, its SUMO
seconds and a label: 1 while an accident is in the camera's picture (incidents.json "visible",
shifted by the render lag because the picture shows the simulation late). The YOLO pass is the
slow step, so finished recordings are skipped on a rerun.

    python agents/incident_detection/evaluation/extract_town05.py <incident_clips folder> \
        --detector agents/incident_detection/config/detector_v1.json [--out data/town05]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR.parents[1]))
from agents.incident_detection.src import load_detector_config, process_video  # noqa: E402

RENDER_LAG_S = 2.2


def recordings(root):
    """Full camera recordings: every <camera>.mp4 that has a .frames.csv beside it."""
    for run in sorted(p for p in root.iterdir() if (p / "run.json").is_file()):
        for frames in sorted(run.glob("*.frames.csv")):
            video = frames.with_name(frames.name.replace(".frames.csv", ".mp4"))
            if video.is_file():
                yield run, video, frames


def camera_name(video):
    return video.stem.replace("_", " ")


def label_windows(windows, visible, camera, lag=RENDER_LAG_S):
    """1 when the window's midpoint, in sim seconds as shown on screen, is inside a visible interval."""
    shown = (windows.sim_start_s + windows.sim_end_s) / 2 - lag
    label = np.zeros(len(windows), dtype=int)
    incident = np.full(len(windows), -1)
    for v in visible:
        if v["camera"] != camera:
            continue
        inside = (shown >= v["from_sim_s"]) & (shown <= v["to_sim_s"])
        label[inside] = 1
        incident[inside] = v["incident"]
    return label, incident


def extract(run, video, frames_csv, config, device, detections=None):
    frames = pd.read_csv(frames_csv)
    windows = process_video(video, config=config, device=device, source_id=video.stem,
                            detections_out=detections).reset_index(drop=True)
    sim = lambda t: np.interp(t, frames.video_s, frames.sim_s)  # noqa: E731
    windows.insert(0, "run", run.name)
    windows.insert(1, "camera", camera_name(video))
    windows.insert(4, "sim_start_s", sim(windows.window_start_sec))
    windows.insert(5, "sim_end_s", sim(windows.window_end_sec))
    incidents = json.loads((run / "incidents.json").read_text(encoding="utf-8"))
    windows["label"], windows["incident"] = label_windows(windows, incidents["visible"], camera_name(video))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    windows["seed"], windows["careless_share"], windows["clock_start"] = meta["seed"], meta["careless_share"], meta["clock_start"]
    windows["detector"] = config.name
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="incident_clips folder holding one folder per run")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=AGENT_DIR / "data" / "town05")
    parser.add_argument("--run", action="append", help="only these runs")
    parser.add_argument("--camera", action="append", help="only these cameras (file stem, e.g. 838-2_looking_N)")
    parser.add_argument("--list", type=Path, help="text file of <run>__<camera file stem>: only these, in this order")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tracks", action="store_true",
                        help="also save every tracked detection (<stem>.tracks.csv.gz) in <detector>-tracks/")
    args = parser.parse_args()

    config = load_detector_config(args.detector)
    out_dir = args.out / (f"{config.name}-tracks" if args.tracks else config.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recordings(args.root) if (not args.run or r[0].name in args.run)
            and (not args.camera or r[1].stem in args.camera)]
    if args.list:
        wanted = args.list.read_text(encoding="utf-8").split()
        found = {f"{r[0].name}__{r[1].stem}": r for r in todo}
        todo = [found[name] for name in wanted]
    for n, (run, video, frames_csv) in enumerate(todo, 1):
        out = out_dir / f"{run.name}__{video.stem}.csv"
        tracks_out = out.with_suffix(".tracks.csv.gz")
        if out.is_file() and (tracks_out.is_file() or not args.tracks):
            continue
        started = time.time()
        detections = [] if args.tracks else None
        windows = extract(run, video, frames_csv, config, args.device, detections)
        if args.tracks:
            frames = pd.read_csv(frames_csv)
            tracks = pd.DataFrame(detections, columns=["source_time_sec", "track_id", "class_id", "category", "confidence",
                                                       "center_x", "center_y", "bbox_height", "bbox_area"])
            tracks.insert(1, "sim_s", np.interp(tracks.source_time_sec, frames.video_s, frames.sim_s).round(2))
            tracks.round(3).to_csv(tracks_out, index=False)
        windows.to_csv(out, index=False)
        print(f"[{n}/{len(todo)}] {run.name} {camera_name(video):26s} {len(windows)} windows, "
              f"{int(windows.label.sum())} in view of an accident | {time.time() - started:.0f}s", flush=True)
    print("done", out_dir)


if __name__ == "__main__":
    main()
