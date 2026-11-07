"""Stage 1: run v1 unchanged on the Town05 clips and score each clip against its labels.

v1 (features, Random Forest, mean3, threshold 0.35) is not changed or re-tuned. Each clip is
run once per detector config; one JSON per clip and detector is written, naming the clip id,
the detector config and the model version, plus a summary over all clips.

    python agents/incident_detection/evaluation/run_stage1.py <stage1 folder> \
        --detector agents/incident_detection/config/detector_v1.json \
        --detector agents/incident_detection/config/detector_carla_yolo26n_t05.json \
        [--out <results folder>] [--clip ring3_2328-2_W_crash]

Scoring, in clip seconds on screen (the picture lags the simulation by render_lag_s):
- an accident is in view from max(0, crash_on_screen_clip_s) to the end of
  wreck_in_picture_clip_s + render_lag_s; a window is an incident window when its midpoint is
  in view;
- caught: an alarm on an incident window; time to detect = end of the first such window minus
  the on-screen crash second (0 when the crash is before frame 0);
- false alarm: a run of consecutive alarm windows with no incident window, counted per hour of
  normal windows.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR.parents[1]))
from agents.incident_detection.src import IncidentDetector, load_detector_config, process_video  # noqa: E402

MODEL_PATH = AGENT_DIR / "models" / "incident_detector.joblib"
MODEL_VERSION = "v1"


def model_fingerprint(path=MODEL_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def view_intervals(clip, lag):
    out = []
    for inc in clip["incidents_in_picture"]:
        start = max(0.0, float(inc["crash_on_screen_clip_s"]))
        end = min(float(clip["duration_s"]), float(inc["wreck_in_picture_clip_s"][1]) + lag)
        out.append({**inc, "view_start_clip_s": start, "view_end_clip_s": end})
    return out


def score_clip(windows, clip, lag):
    """windows: v1 result dicts of one clip. Returns the per-clip summary."""
    df = pd.DataFrame([{k: w[k] for k in ("window_start_sec", "window_end_sec", "incident_detected",
                                          "incident_score")} for w in windows])
    mid = (df.window_start_sec + df.window_end_sec) / 2
    intervals = view_intervals(clip, lag)
    df["incident_window"] = False
    accidents = []
    for inc in intervals:
        inside = (mid >= inc["view_start_clip_s"]) & (mid <= inc["view_end_clip_s"])
        df.loc[inside, "incident_window"] = True
        hits = df.loc[inside & df.incident_detected, "window_end_sec"]
        # Alarms in the minute before the accident is on screen: a catch then says little.
        before = (df.window_end_sec <= inc["view_start_clip_s"]) & (df.window_start_sec >= inc["view_start_clip_s"] - 60)
        accidents.append({
            "incident": inc["incident"], "kind": inc["kind"], "severity_label": inc["severity_label"],
            "crash_before_clip": inc["crash_before_clip"],
            "view_clip_s": [inc["view_start_clip_s"], inc["view_end_clip_s"]],
            "incident_windows": int(inside.sum()),
            "alarm_windows": int((inside & df.incident_detected).sum()),
            "caught": bool(len(hits)),
            "first_alarm_clip_s": float(hits.min()) if len(hits) else None,
            "time_to_detect_s": max(0.0, float(hits.min()) - inc["view_start_clip_s"]) if len(hits) else None,
            "max_score_in_view": round(float(df.loc[inside, "incident_score"].max()), 4) if inside.any() else None,
            "alarm_windows_in_60s_before_view": int((before & df.incident_detected).sum()) if before.any() else None,
            "windows_in_60s_before_view": int(before.sum()),
        })
    run_id = (df.incident_detected != df.incident_detected.shift()).cumsum()
    alarm = df[df.incident_detected]
    episodes = alarm.groupby(run_id[df.incident_detected])["incident_window"].max()
    false_episodes = [
        [float(g.window_start_sec.min()), float(g.window_end_sec.max())]
        for key, g in alarm.groupby(run_id[df.incident_detected]) if not episodes[key]]
    normal = df[~df.incident_window]
    normal_hours = float((normal.window_end_sec - normal.window_start_sec).sum()) / 3600
    return {
        "windows": int(len(df)),
        "alarm_windows": int(df.incident_detected.sum()),
        "incident_windows": int(df.incident_window.sum()),
        "normal_minutes": round(normal_hours * 60, 2),
        "false_alarm_episodes": len(false_episodes),
        "false_alarm_clip_s": false_episodes,
        "false_alarms_per_camera_hour": round(len(false_episodes) / normal_hours, 2) if normal_hours else None,
        "false_alarm_windows": int((df.incident_detected & ~df.incident_window).sum()),
        "normal_alarm_share": round(float(normal.incident_detected.mean()), 3) if len(normal) else None,
        "score_mean_normal": round(float(normal.incident_score.mean()), 4) if len(normal) else None,
        "score_p90_normal": round(float(normal.incident_score.quantile(0.9)), 4) if len(normal) else None,
        "accidents": accidents,
    }, df.incident_window.tolist()


def run_clip(clip, root, config, detector, device, lag):
    started = time.time()
    source_id = clip["id"]
    raw = process_video(root / clip["file"], config=config, device=device, source_id=source_id)
    windows = detector.predict_windows(raw, source_id)
    summary, labels = score_clip(windows, clip, lag)
    for window, is_incident in zip(windows, labels):
        # New optional fields (agreed time base): sim seconds of the clip's own run.
        window["run"] = clip["run"]
        window["window_start_time_s"] = round(clip["sim_s_frame0"] + window["window_start_sec"], 3)
        window["window_end_time_s"] = round(clip["sim_s_frame0"] + window["window_end_sec"], 3)
        window["label_incident_in_view"] = bool(is_incident)
    feature_means = raw.drop(columns=["window_start_sec", "window_end_sec"]).mean(numeric_only=True)
    return {
        "clip_id": source_id, "camera": clip["camera"], "group": clip["group"], "run": clip["run"],
        "sim_s_frame0": clip["sim_s_frame0"], "fps": clip["fps"],
        "detector_config": {"name": config.name, "weights": Path(config.weights).name,
                            "class_map": dict(config.class_map), "confidence": config.confidence,
                            "imgsz": config.imgsz, "tracker": config.tracker},
        "model": {"version": MODEL_VERSION, "file": MODEL_PATH.name, "sha256_12": model_fingerprint(),
                  "temporal_rule": detector.temporal_rule, "threshold": detector.threshold},
        "scoring": {"render_lag_s": lag, "rule": "incident window = window midpoint while the accident is "
                    "in view on screen; false alarm = run of alarm windows with no incident window"},
        "summary": summary,
        "feature_means": {k: round(float(v), 4) for k, v in feature_means.items()},
        "windows": windows,
        "seconds": round(time.time() - started, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage1", type=Path, help="delivery folder holding labels.json and clips/")
    parser.add_argument("--detector", type=Path, action="append", required=True, help="detector config JSON")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "stage1")
    parser.add_argument("--clip", action="append", help="only these clip ids")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rescore", action="store_true",
                        help="recompute the summaries from the saved windows in --out, without running YOLO")
    args = parser.parse_args()

    labels = json.loads((args.stage1 / "labels.json").read_text(encoding="utf-8"))
    lag = float(labels["time_base"]["render_lag_s"])
    clips = [c for c in labels["clips"] if not args.clip or c["id"] in args.clip]
    detector = IncidentDetector(MODEL_PATH)
    args.out.mkdir(parents=True, exist_ok=True)
    overview = []
    for config_path in args.detector:
        config = load_detector_config(config_path)
        if not args.rescore and not Path(config.weights).is_file():
            print(f"skip {config.name}: weights not found at {config.weights} (set GOFLOW_YOLO_WEIGHTS)")
            continue
        for clip in clips:
            out = args.out / f"{clip['id']}__{config.name}__{MODEL_VERSION}.json"
            if args.rescore:
                if not out.is_file():
                    continue
                result = json.loads(out.read_text(encoding="utf-8"))
                result["summary"], _ = score_clip(result["windows"], clip, lag)
            else:
                result = run_clip(clip, args.stage1, config, detector, args.device, lag)
            out.write_text(json.dumps(result, indent=1), encoding="utf-8")
            s = result["summary"]
            caught = ", ".join(f"#{a['incident']} {'caught' if a['caught'] else 'missed'}"
                               + (f" after {a['time_to_detect_s']:.0f}s" if a["caught"] else "")
                               for a in s["accidents"]) or "no accident"
            print(f"{config.name:18s} {clip['id']:32s} alarms {s['alarm_windows']:2d}/{s['windows']} | "
                  f"normal time in alarm {s['normal_alarm_share']:.0%} | FA {s['false_alarm_episodes']} "
                  f"({s['false_alarms_per_camera_hour']}/h) | {caught} | {result['seconds']}s", flush=True)
            overview.append({"detector": config.name, "clip_id": clip["id"], "group": clip["group"],
                             **{k: v for k, v in s.items() if k != "false_alarm_clip_s"}})
    if overview:
        summary = {"created": datetime.now().isoformat(timespec="seconds"), "model_version": MODEL_VERSION,
                   "model_sha256_12": model_fingerprint(), "render_lag_s": lag, "per_clip": overview,
                   "per_detector": {}}
        for name in dict.fromkeys(o["detector"] for o in overview):
            rows = [o for o in overview if o["detector"] == name]
            fa = sum(o["false_alarm_episodes"] for o in rows)
            minutes = sum(o["normal_minutes"] for o in rows)
            accidents = [a for o in rows for a in o["accidents"]]
            summary["per_detector"][name] = {
                "clips": len(rows), "false_alarm_episodes": fa, "normal_hours": round(minutes / 60, 3),
                "false_alarms_per_camera_hour": round(fa / (minutes / 60), 2) if minutes else None,
                "normal_alarm_share": round(sum(o["normal_alarm_share"] * o["normal_minutes"] for o in rows)
                                            / minutes, 3) if minutes else None,
                "accidents_caught": f"{sum(a['caught'] for a in accidents)}/{len(accidents)}"}
        path = args.out / f"stage1_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print("saved", path)


if __name__ == "__main__":
    main()
