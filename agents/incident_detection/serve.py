"""Omar's accident agent as a process of its own: JSON lines on stdin, JSON lines on stdout (Town05 adapter).

Started by cosim/incident_agent_proc.py with this package's own venv (agents/incident_detection/.venv: pandas,
scikit-learn 1.6.1, joblib; no GPU). One line per message:

in   {"op": "hello", "detector": "<name>"}
     {"op": "frame", "cam": "<camera>", "t": <sim s of the picture>,
      "frame": {"vehicle_count": n, "bbox_coverage": x, "mean_detection_confidence": x},
      "dets": [[track_id, class_id, category, confidence, center_x, center_y, bbox_height, bbox_area], ...]}
     {"op": "flush"}            close every camera's open window (end of the run)
     {"op": "ping"}             -> {"op": "pong"}
     {"op": "stop"}             -> {"op": "bye", "stats": {...}}, then exit
out  {"op": "ready", "model": {...}}  once loaded
     {"op": "window", ...}      one per camera and closed window: his v1 fields + camera, sim_start_s, sim_end_s,
                                pictures, impact, impact_alarm, model, took_ms
     {"op": "error", "text": ..., "cam": ...}   a bad message; the process goes on

stderr carries tracebacks only. The process never reads the simulation.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL = HERE / "models" / "incident_detector.joblib"


def say(obj) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    import sklearn

    import stream
    from agents.incident_detection.src.inference import IncidentDetector

    detector = IncidentDetector(MODEL)
    try:  # one core: the same trees and votes, 89 ms a window instead of 147 (thread start-up), beside CARLA
        detector.model.set_params(classifier__n_jobs=1)
    except ValueError:
        pass
    model = {"version": "v1", "file": MODEL.name, "sha256_12": stream.model_sha(MODEL),
             "threshold": detector.threshold, "rule": detector.temporal_rule, "sklearn": sklearn.__version__,
             "accepted_for_town05": False}
    cams: dict[str, stream.StreamingWindows] = {}
    detector_name = ""
    counts = {"frames": 0, "windows": 0, "errors": 0}
    ready = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cam = None
        try:
            msg = json.loads(line)
            op = msg.get("op")
            if op == "hello":
                detector_name = str(msg.get("detector", ""))
                if not ready:
                    say({"op": "ready", "model": model, "detector": detector_name})
                    ready = True
            elif op == "frame":
                cam = str(msg["cam"])
                t = float(msg["t"])
                s = cams.get(cam)
                if s is None:
                    s = cams[cam] = stream.StreamingWindows(cam, detector, "v1", detector_name)
                dets = [dict(zip(stream.DET_KEYS, d)) for d in msg.get("dets", ())]
                for d in dets:
                    d["track_id"], d["class_id"] = int(d["track_id"]), int(d["class_id"])
                t1 = time.perf_counter()
                out = s.add(t, dict(msg["frame"]), dets)
                counts["frames"] += 1
                for w in out:
                    w["took_ms"] = round((time.perf_counter() - t1) * 1000.0 / len(out), 1)
                    counts["windows"] += 1
                    say(dict(w, op="window"))
            elif op == "flush":
                for s in cams.values():
                    for w in s.flush():
                        counts["windows"] += 1
                        say(dict(w, op="window"))
            elif op == "ping":
                say({"op": "pong"})
            elif op == "stop":
                break
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as e:  # noqa: BLE001 - a bad message is answered, never fatal
            counts["errors"] += 1
            say({"op": "error", "cam": cam, "text": f"{type(e).__name__}: {e}"})
            traceback.print_exc(file=sys.stderr)
    say({"op": "bye", "stats": dict(counts, cameras={k: v.stats for k, v in cams.items()})})
    return 0


if __name__ == "__main__":
    sys.exit(main())
