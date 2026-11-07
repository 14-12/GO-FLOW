"""Writes the order the v3 recordings go through the detector, without looking at any video or result.

Running every training recording would take too long and adds little: the runs hold far more normal traffic
than crashes. So: training recordings with a counted crash; then other training recordings, cycling over the
cameras until --train-total, so that every training camera has normal traffic; then every test recording.
Which training recordings are used depends on the crash list only, never on the agent.

    python agents/incident_detection/evaluation/v3_extract_order.py <out.txt> [--train-total 40]
"""
import argparse
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def stem(camera):
    return camera.replace(" ", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--train-total", type=int, default=40)
    args = parser.parse_args()
    plan = json.loads((EVAL_DIR / "v3_split.json").read_text(encoding="utf-8"))
    progress = json.loads((EVAL_DIR / "results" / "v3_recording_progress.json").read_text(encoding="utf-8"))
    train = [r for r in progress["runs"] if r["split"] == "train"]
    test = [r for r in progress["runs"] if r["split"] == "test"]
    near = [f"{r['folder']}__{stem(c['camera'])}" for r in train for c in r["near_crashes"]]
    near = list(dict.fromkeys(near))
    order = list(near)
    full = [r for r in train if r["complete"]]
    cameras = plan["cameras"]["train"]
    i = 0
    while len(order) < args.train_total and i < len(cameras) * len(full):
        # one step along the runs and one along the cameras, so cameras and runs both spread out
        name = f"{full[i % len(full)]['folder']}__{stem(cameras[i % len(cameras)])}"
        if name not in order:
            order.append(name)
        i += 1
    train_count = len(order)
    order += [f"{r['folder']}__{stem(c)}" for r in test for c in plan["cameras"]["test"]]
    args.out.write_text("\n".join(order) + "\n", encoding="utf-8")
    print(f"{len(near)} training with a crash, {train_count - len(near)} other training, {len(order) - train_count} test")


if __name__ == "__main__":
    main()
