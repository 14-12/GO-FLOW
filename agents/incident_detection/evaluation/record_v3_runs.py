"""Record the v3 crash data set with the Urban Traffic simulator, one 35-minute run after another.

Run from the simulator's project folder with its own Python (it imports cosim for the camera geometry):
    .venv\\Scripts\\python.exe <this file> [--count-only FOLDER ...] [--max-runs N]

The plan (seed, cameras, time of day, train or test) is fixed in v3_split.json before anything is recorded and
a run's split never depends on what happens in it. Stops after a full train/train/test round once the counted
crashes reach the targets. Counting uses only the simulator's own incident log, never the agent's output.
Resumable: runs already in the progress file are skipped. Writes <evaluation>/results/v3_recording_progress.json.
"""

import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / "v3_split.json"
PROGRESS = HERE / "results" / "v3_recording_progress.json"
ROOT = Path.cwd()
CLIPS = ROOT / "out" / "incident_clips"
sys.path.insert(0, str(ROOT))

MIN_T0_S = 60.0         # crashes in the simulator's first minute have no normal traffic before them
NEAR_M = 35.0
ON_SCREEN_S = 10.0      # the wreck must be in the picture this soon after the impact, or the impact was not seen
SLACK_S = 900           # start-up and closing on top of the recording itself
CARLA_NAMES = ("CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe")


def camera_specs():
    from cosim.perception import CamSpec
    layout = json.loads((ROOT / "out/town05_cameras/layout.json").read_text(encoding="utf-8"))
    specs = {c["name"]: CamSpec(c["tls"], c["node"], c["road_to"], c["x"], c["y"], c["z"], c["yaw"],
                                c.get("pitch", -22.0)) for c in layout["cameras"]}
    return specs, layout["fov"]


def near_crashes(run_dir, specs, fov):
    """One row per (crash, camera) with the impact on screen, near and not at the picture's edge."""
    from cosim.clip_recorder import sees
    data = json.loads((run_dir / "incidents.json").read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in data["incidents"]}
    rows = []
    for v in data["visible"]:
        crash = by_id[v["incident"]]
        if crash["t0"] < MIN_T0_S or v["from_sim_s"] > crash["t0"] + ON_SCREEN_S or v["camera"] not in specs:
            continue
        seen, distance = sees(specs[v["camera"]], 960, 540, fov, *crash["carla_xy"])
        if seen and distance <= NEAR_M:
            rows.append({"camera": v["camera"], "incident": crash["id"], "t0": crash["t0"], "kind": crash["kind"],
                         "lane": crash.get("lane"), "distance_m": round(distance, 1),
                         "from_sim_s": v["from_sim_s"], "to_sim_s": v["to_sim_s"]})
    return rows


def recorded_seconds(run_dir):
    longest = 0.0
    for f in run_dir.glob("*.frames.csv"):
        with f.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            longest = max(longest, float(rows[-1]["wall_s"]) - float(rows[0]["wall_s"]))
    return longest


def carla_running():
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
    return any(n.lower() in out.lower() for n in CARLA_NAMES)


def stop_simulator():
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ROOT / "tools/stop_cosim.ps1")], capture_output=True)
    for n in CARLA_NAMES:
        subprocess.run(["taskkill", "/f", "/im", n], capture_output=True)


def record(run, plan):
    """Returns the run's folder, or None when nothing usable was written."""
    focus = f' --crash-focus {plan["crash_focus"]}' if plan.get("crash_focus") else ""
    args = (f'--demo 1 --record-clips {plan["seconds"]} --record-fps {plan["fps"]} '
            f'--record-cams "{";".join(plan["cameras"][run["cameras"]])}" --record-crash-match __never__ '
            f'--record-dynamic 0 --careless-share {plan["careless_share"]}{focus} --seed {run["seed"]} --clock-speed 1 '
            f'--set-time {run["time"]} --no-cameras --clock-file out/freefly/clips_clock.json')
    env = dict(os.environ, FREEFLY_ARGS=args, CARLA_RES="1280x720", FREEFLY_SPLASH="0")
    before = set(p.name for p in CLIPS.iterdir()) if CLIPS.exists() else set()
    proc = subprocess.Popen(["cmd", "/c", str(ROOT / "tools/town05_freefly.bat")], cwd=ROOT, env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started, seen_carla, gone = time.time(), False, 0
    while proc.poll() is None:
        time.sleep(20)
        running = carla_running()
        seen_carla = seen_carla or running
        gone = 0 if running else gone + 1
        crashed = seen_carla and gone >= 4      # the .bat itself closes CARLA at a run's end and returns soon after
        if crashed or time.time() - started > plan["seconds"] + SLACK_S:
            print("simulator stopped answering:", "CARLA gone" if crashed else "timeout", flush=True)
            subprocess.run(["taskkill", "/f", "/t", "/pid", str(proc.pid)], capture_output=True)
            stop_simulator()
            break
    new = sorted(set(p.name for p in CLIPS.iterdir()) - before)
    return CLIPS / new[-1] if new else None


def totals(progress):
    out = {"train": 0, "test": 0}
    for r in progress["runs"]:
        if r["complete"]:
            out[r["split"]] += len(r["near_crashes"])
    return out


def save(progress):
    PROGRESS.parent.mkdir(exist_ok=True)
    PROGRESS.write_text(json.dumps(progress, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", nargs="*", help="only print the counted crashes of these run folders")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--plan", default="v3_split.json", help="plan file in this folder, e.g. v4_split.json")
    a = ap.parse_args()
    global PLAN, PROGRESS
    PLAN = HERE / a.plan
    PROGRESS = HERE / "results" / f"{a.plan.split('_')[0]}_recording_progress.json"
    specs, fov = camera_specs()
    if a.count_only is not None:
        for d in a.count_only:
            print(d, json.dumps(near_crashes(Path(d), specs, fov)))
        return
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    progress = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {"runs": []}
    done = {r["plan_index"] for r in progress["runs"]}
    # keep the PC awake while this runs; Windows drops the request when the process ends
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    for index, run in enumerate(plan["runs"]):
        if a.max_runs is not None and index >= a.max_runs:
            break
        if index in done:
            continue
        count = totals(progress)
        if (index % 3 == 0 and count["train"] >= plan.get("target_train", float("inf"))
                and count["test"] >= plan.get("target_test", float("inf"))):
            print("targets reached:", count, flush=True)
            break
        print(time.strftime("%H:%M"), f"run {index + 1}/{len(plan['runs'])}", run, count, flush=True)
        folder = record(run, plan)
        seconds = recorded_seconds(folder) if folder else 0.0
        complete = seconds >= 0.95 * plan["seconds"]
        crashes = near_crashes(folder, specs, fov) if folder and (folder / "incidents.json").exists() else []
        progress["runs"].append({"plan_index": index, "split": run["split"], "seed": run["seed"],
                                 "time": run["time"], "cameras": run["cameras"],
                                 "folder": folder.name if folder else None, "recorded_s": round(seconds),
                                 "complete": complete, "near_crashes": crashes})
        save(progress)
        print(time.strftime("%H:%M"), "done:", folder.name if folder else None, round(seconds), "s,",
              len(crashes), "near crashes, complete" if complete else "near crashes, INCOMPLETE", flush=True)
    print("finished:", totals(progress), flush=True)


if __name__ == "__main__":
    main()
