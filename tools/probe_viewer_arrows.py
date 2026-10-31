"""Arrow keys with Z Q S D in the running free-fly viewer, measured frame by frame (out/probe_viewer_arrows.json).

    python tools/probe_viewer_arrows.py --mode proc      (tools/probe_viewer_arrows.bat starts the viewer for each mode)

Waits for the viewer, brings CARLA to the front, holds keys with keybd_event and records the spectator in
every CARLA frame from a client of its own. Per test: speed (mean, sd), how much it changes from one frame
to the next, frames where the camera went backwards, turn achieved and how even it is. Then Esc quits.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import carla  # noqa: E402

from town05_fps_test import grab_window  # noqa: E402

LOG = ROOT / "tools" / "logs" / "freefly_town05.log"
TESTS = [("Z", [0x5A], 2.5), ("Z+RIGHT", [0x5A, 0x27], 2.5), ("Q+LEFT", [0x51, 0x25], 2.0),
         ("RIGHT", [0x27], 1.5), ("Z+UP", [0x5A, 0x26], 1.0), ("tapR1", [0x27], 0.1), ("tapR2", [0x27], 0.1),
         ("tapL1", [0x25], 0.1), ("tapL2", [0x25], 0.1)]
EXT = {0x25, 0x26, 0x27, 0x28}


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="proc")
    a = ap.parse_args()
    t0 = time.time()
    while time.time() - t0 < 400:  # the viewer's first 10 s report
        if LOG.exists() and LOG.stat().st_mtime > t0 - 5 and "fps  updates" in LOG.read_text(encoding="utf-8", errors="ignore"):
            break
        time.sleep(2)
    else:
        print("viewer not ready", flush=True)
        return 1
    time.sleep(5)
    c = carla.Client("127.0.0.1", 2000)
    c.set_timeout(60)
    world = c.get_world()
    sp = world.get_spectator()
    trace = []
    cb = world.on_tick(lambda snap: trace.append((snap.timestamp.elapsed_seconds, snap.find(sp.id).get_transform())))
    u = ctypes.windll.user32
    out = {"mode": a.mode}
    for name, keys, hold in TESTS:
        grab_window(ROOT / "out" / "probe_viewer_arrows.png")  # CARLA in front
        time.sleep(0.6)
        sp.set_transform(carla.Transform(carla.Location(-40.0, 10.0, 25.0), carla.Rotation(-15.0, 0.0, 0.0)))
        time.sleep(0.6)
        trace.clear()
        for vk in keys:
            u.keybd_event(vk, 0, 1 if vk in EXT else 0, 0)
        time.sleep(hold)
        for vk in keys:
            u.keybd_event(vk, 0, (1 if vk in EXT else 0) | 2, 0)
        time.sleep(0.6)
        tr = list(trace)
        sp_, yr, back = [], [], 0
        for (ta, fa), (tb, fb) in zip(tr, tr[1:]):
            dt = tb - ta
            if dt <= 0:
                continue
            d = fb.location - fa.location
            sp_.append(math.sqrt(d.x * d.x + d.y * d.y + d.z * d.z) / dt)
            yr.append(wrap(fb.rotation.yaw - fa.rotation.yaw) / dt)
            f = fa.get_forward_vector()
            if "Z" in name and d.x * f.x + d.y * f.y + d.z * f.z < -0.01:
                back += 1
        n = len(sp_)
        mid = sp_[n // 5: 4 * n // 5] if "tap" not in name else sp_
        ymid = yr[n // 5: 4 * n // 5] if "tap" not in name else yr
        ch = [abs(y - x) for x, y in zip(mid, mid[1:])]
        ych = [abs(y - x) for x, y in zip(ymid, ymid[1:])]
        res = {"frames": n, "turn_deg": round(sum(wrap(fb.rotation.yaw - fa.rotation.yaw) for (_, fa), (_, fb) in zip(tr, tr[1:])), 1),
               "speed_mean": round(statistics.fmean(mid), 2) if mid else None,
               "speed_sd": round(statistics.pstdev(mid), 2) if len(mid) > 1 else None,
               "speed_change_per_frame": round(statistics.fmean(ch), 2) if ch else None,
               "backward_frames": back,
               "turn_rate_sd": round(statistics.pstdev(ymid), 1) if len(ymid) > 1 else None,
               "turn_rate_change_per_frame": round(statistics.fmean(ych), 1) if ych else None,
               "turn_stall_frames": (sum(1 for y in ymid if abs(y) < 0.2 * abs(statistics.fmean(ymid)))
                                     if ymid and ("RIGHT" in name or "LEFT" in name) else None)}
        out[name] = res
        print(name, res, flush=True)
    world.remove_on_tick(cb)
    res_file = ROOT / "out" / "probe_viewer_arrows.json"
    allres = json.loads(res_file.read_text(encoding="utf-8")) if res_file.exists() else {}
    allres[a.mode] = out
    res_file.write_text(json.dumps(allres, indent=1), encoding="utf-8")
    grab_window(ROOT / "out" / "probe_viewer_arrows.png")
    time.sleep(0.5)
    u.keybd_event(0x1B, 0, 0, 0)  # Esc: the viewer quits
    time.sleep(0.1)
    u.keybd_event(0x1B, 0, 2, 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
