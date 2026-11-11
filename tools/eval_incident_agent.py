"""Grade the accident agent's verdicts against the simulation's record (11 Nov; docs/INCIDENT_AGENT_AUDIT.md).

The verdicts (out/incident_agent/<time>/windows.jsonl from the viewer, or replay.py's files) never saw the record;
it is read only here. The record is the --record-clips run of the same cameras (out/incident_clips/<run>/:
incidents.json "visible", run.json cameras).

His rules (GO-FLOW agents/incident_detection/evaluation: record_v3_runs.near_crashes, v3_features.targets,
train_v3.score_alarms):
  counted crash   impact on screen (in the picture within 10 s of the crash), wreck within 35 m and wholly in the
                  picture, at least 60 s after the camera's first window
  window target   1: a counted crash in the picture at the window's middle; -1: another wreck there (neither right
                  nor wrong); 0: normal traffic, jams included
  caught          an alarm on a window of the crash's run of target-1 windows; time to detect = end of the first such
                  window - start of the run; we add the delay as lived: when the verdict reached the viewer - crash
  false alarm     a run of alarm windows that are all target 0, per hour of target-0 windows
Channels: v1 (his model, incident_detected), impact (his crash moment, contact_hard_stops >= 1), either.

    python tools\\eval_incident_agent.py WINDOWS.jsonl out\\incident_clips\\<run> [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEAR_M = 35.0
ON_SCREEN_S = 10.0
SETTLE_S = 60.0
CRITERIA = {"false_alarms_per_camera_hour": 1.0, "recall": 0.8, "median_detect_s": 30.0, "slowest_detect_s": 120.0}
CHANNELS = {"v1": lambda w: bool(w.get("incident_detected")), "impact": lambda w: bool(w.get("impact_alarm")),
            "either": lambda w: bool(w.get("incident_detected")) or bool(w.get("impact_alarm"))}


def counted_crashes(incidents: dict, specs: dict, fov: float, first_window: dict, sees) -> list[dict]:
    by_id = {i["id"]: i for i in incidents["incidents"]}
    rows = []
    for v in incidents["visible"]:
        cam, crash = v["camera"], by_id.get(v["incident"])
        if crash is None or cam not in specs or cam not in first_window:
            continue
        if crash["t0"] < first_window[cam] + SETTLE_S or v["from_sim_s"] > crash["t0"] + ON_SCREEN_S:
            continue
        seen, d = sees(specs[cam], 960, 540, fov, *crash["carla_xy"])
        if seen and d <= NEAR_M:
            rows.append({"camera": cam, "incident": crash["id"], "t0": crash["t0"], "kind": crash["kind"],
                         "distance_m": round(d, 1), "from_sim_s": v["from_sim_s"], "to_sim_s": v["to_sim_s"],
                         "crash_source": crash.get("crash_source")})
    return rows


def label(windows: list[dict], visible: list[dict], counted: list[dict]) -> None:
    near = {(c["camera"], c["incident"]) for c in counted}
    for w in windows:
        mid = (w["sim_start_s"] + w["sim_end_s"]) / 2
        w["target"], w["incident"] = 0, None
        for v in visible:
            if v["camera"] == w["camera"] and v["from_sim_s"] <= mid <= v["to_sim_s"]:
                key = (v["camera"], v["incident"])
                if key in near:
                    w["target"], w["incident"] = 1, v["incident"]
                    break
                w["target"], w["incident"] = -1, v["incident"]


def score(windows: list[dict], alarm, t0_of: dict) -> dict:
    events, fa, fa_list, hours, normal_on, normal_all = [], 0, [], 0.0, 0, 0
    by_cam = defaultdict(list)
    for w in windows:
        by_cam[w["camera"]].append(w)
    for cam, ws in by_cam.items():
        ws.sort(key=lambda w: w["sim_start_s"])
        on = [alarm(w) for w in ws]
        for w, a in zip(ws, on):
            if w["target"] == 0:
                hours += (w["sim_end_s"] - w["sim_start_s"]) / 3600
                normal_all += 1
                normal_on += a
        i = 0
        while i < len(ws):  # runs of target-1 windows
            if ws[i]["target"] != 1:
                i += 1
                continue
            j = i
            while j + 1 < len(ws) and ws[j + 1]["target"] == 1:
                j += 1
            hit = next((k for k in range(i, j + 1) if on[k]), None)
            inc = ws[i]["incident"]
            ev = {"camera": cam, "incident": inc, "start_s": ws[i]["sim_start_s"], "caught": hit is not None,
                  "time_to_detect_s": None if hit is None else round(ws[hit]["sim_end_s"] - ws[i]["sim_start_s"], 1)}
            if hit is not None and "received_sim_s" in ws[hit] and inc in t0_of:
                ev["lived_delay_s"] = round(ws[hit]["received_sim_s"] - t0_of[inc], 1)
            events.append(ev)
            i = j + 1
        i = 0
        while i < len(ws):  # alarm episodes
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(ws) and on[j + 1]:
                j += 1
            if all(ws[k]["target"] == 0 for k in range(i, j + 1)):
                fa += 1
                fa_list.append({"camera": cam, "from_s": ws[i]["sim_start_s"], "to_s": ws[j]["sim_end_s"]})
            i = j + 1
    ttd = sorted(e["time_to_detect_s"] for e in events if e["caught"])
    res = {"normal_hours": round(hours, 3), "false_alarms": fa,
           "false_alarms_per_camera_hour": round(fa / hours, 2) if hours else None,
           "normal_time_in_alarm": round(normal_on / normal_all, 4) if normal_all else None,
           "events": len(events), "caught": len(ttd), "recall": round(len(ttd) / len(events), 3) if events else None,
           "median_detect_s": ttd[len(ttd) // 2] if ttd else None, "slowest_detect_s": ttd[-1] if ttd else None,
           "per_event": events, "false_alarm_list": fa_list}
    res["criteria_met"] = {
        "false_alarms": res["false_alarms_per_camera_hour"] is not None
        and res["false_alarms_per_camera_hour"] <= CRITERIA["false_alarms_per_camera_hour"],
        "recall": res["recall"] is not None and res["recall"] >= CRITERIA["recall"],
        "median_detect": res["median_detect_s"] is not None and res["median_detect_s"] <= CRITERIA["median_detect_s"],
        "slowest_detect": res["slowest_detect_s"] is not None
        and res["slowest_detect_s"] <= CRITERIA["slowest_detect_s"]}
    return res


def delays(windows: list[dict]) -> dict | None:
    """Seconds from a window's end to its verdict reaching the viewer, on the pictures' clock (received_shown_s),
    in steady running: without each camera's first 3 windows (start-up) and the windows received in the last 60 s
    of the run (the viewer's shutdown flushes them all at once)."""
    live = [w for w in windows if w.get("received_shown_s") is not None]
    if not live:
        return None
    last = max(w["received_shown_s"] for w in live)
    first = {}
    for w in sorted(live, key=lambda w: w["sim_start_s"]):
        first.setdefault(w["camera"], []).append(w)
    keep = [w for ws in first.values() for w in ws[3:] if w["received_shown_s"] < last - 60.0]
    d = sorted(w["received_shown_s"] - w["sim_end_s"] for w in keep)
    if not d:
        return None
    return {"windows": len(d), "median": round(d[len(d) // 2], 2), "p95": round(d[int(0.95 * len(d))], 2),
            "max": round(d[-1], 2)}


def grade(windows: list[dict], clips: Path) -> dict:
    from cosim.clip_recorder import sees
    from cosim.perception import CamSpec

    inc = json.loads((clips / "incidents.json").read_text(encoding="utf-8"))
    run = json.loads((clips / "run.json").read_text(encoding="utf-8"))
    cams = {w["camera"] for w in windows}
    specs, fov = {}, 75.0
    for name, c in run["cameras"].items():
        if name in cams:
            specs[name] = CamSpec(c["tls"], c["node"], c["road_to"], c["x"], c["y"], c["z"], c["yaw"],
                                  c.get("pitch", -22.0))
            fov = c.get("fov", fov)
    first = {}
    for w in windows:
        first[w["camera"]] = min(first.get(w["camera"], 1e18), w["sim_start_s"])
    counted = counted_crashes(inc, specs, fov, first, sees)
    label(windows, inc["visible"], counted)
    t0_of = {i["id"]: i["t0"] for i in inc["incidents"]}
    out = {"clips": str(clips), "cameras": sorted(cams), "windows": len(windows), "verdict_delay_s": delays(windows),
           "targets": {k: sum(w["target"] == k for w in windows) for k in (1, 0, -1)},
           "crashes_in_record": len(inc["incidents"]), "counted_crashes": counted,
           "channels": {name: score(windows, f, t0_of) for name, f in CHANNELS.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("windows")
    ap.add_argument("clips")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    ws = [json.loads(x) for x in open(a.windows, encoding="utf-8") if x.strip()]
    res = grade(ws, Path(a.clips))
    print(f"{res['windows']} windows on {len(res['cameras'])} cameras, {len(res['counted_crashes'])} counted crashes "
          f"(of {res['crashes_in_record']} in the record), targets {res['targets']}")
    print("verdict delay (s):", res["verdict_delay_s"])
    for name, r in res["channels"].items():
        print(f"{name:7s} caught {r['caught']}/{r['events']}, median {r['median_detect_s']} s, slowest "
              f"{r['slowest_detect_s']} s, false alarms {r['false_alarms']} in {r['normal_hours']:.2f} h "
              f"= {r['false_alarms_per_camera_hour']}/h, normal time in alarm {r['normal_time_in_alarm']}, "
              f"criteria {r['criteria_met']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
