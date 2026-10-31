"""What each key does to CARLA's spectator, with the CARLA window in front (out/probe_keys.json).

Started by tools/probe_keys.bat (CARLA in a window). Each key is held 0.8 s with keybd_event; the
spectator's move (in its own frame: forward, right, up) and turn are measured from get_transform.
"""
import ctypes
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import carla  # noqa: E402

from town05_fps_test import grab_window  # noqa: E402

KEYS = [("Z", 0x5A, 0), ("UP", 0x26, 1)]

c = carla.Client("127.0.0.1", 2000)
c.set_timeout(120)
world = c.get_world()
sp = world.get_spectator()
sp.set_transform(carla.Transform(carla.Location(0.0, 0.0, 40.0), carla.Rotation(-10.0, 0.0, 0.0)))
time.sleep(1.0)
grab_window(ROOT / "out" / "probe_keys.png")  # brings CARLA to the front
time.sleep(1.0)
u = ctypes.windll.user32
out = {}
for name, vk, ext in KEYS:
    a = sp.get_transform()
    u.keybd_event(vk, 0, ext, 0)
    time.sleep(0.8)
    u.keybd_event(vk, 0, ext | 0x0002, 0)
    time.sleep(0.4)
    b = sp.get_transform()
    f, r, up = a.get_forward_vector(), a.get_right_vector(), a.get_up_vector()
    d = b.location - a.location
    out[name] = {"forward": round(d.x * f.x + d.y * f.y + d.z * f.z, 2), "right": round(d.x * r.x + d.y * r.y + d.z * r.z, 2),
                 "up": round(d.z, 2), "yaw": round((b.rotation.yaw - a.rotation.yaw + 180) % 360 - 180, 1),
                 "pitch": round(b.rotation.pitch - a.rotation.pitch, 1)}
    print(name, out[name], flush=True)
(ROOT / "out" / "probe_keys.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

res = {}
# the viewer's arrows (cosim/flycam.ArrowTurn in CARLA's tick callback, keys read by cosim/overlay.Overlay)
# with CARLA's own Z: smoothness per CARLA frame, and short taps
import statistics  # noqa: E402

from cosim.flycam import ArrowTurn  # noqa: E402
from cosim.overlay import Overlay  # noqa: E402

ov = Overlay(0)
ov.carla = ov._find_carla()
turn = ArrowTurn(sp)
trace = []


def tick(snap):
    turn.tick(snap, ov.held())
    a = snap.find(sp.id)
    if a is not None:
        trace.append((snap.timestamp.elapsed_seconds, a.get_transform()))


cb = world.on_tick(tick)
grab_window(ROOT / "out" / "probe_keys.png")
time.sleep(0.5)
for label, keys, hold in (("Z", [(0x5A, 0)], 2.0), ("Z+RIGHT", [(0x5A, 0), (0x27, 1)], 2.0),
                          ("Q+LEFT+UP", [(0x51, 0), (0x25, 1), (0x26, 1)], 1.5), ("RIGHT_tap1", [(0x27, 1)], 0.1),
                          ("RIGHT_tap2", [(0x27, 1)], 0.1), ("RIGHT_tap3", [(0x27, 1)], 0.1), ("LEFT_tap", [(0x25, 1)], 0.1)):
    a = sp.get_transform()
    trace.clear()
    for vk, ext in keys:
        u.keybd_event(vk, 0, ext, 0)
    time.sleep(hold)
    for vk, ext in keys:
        u.keybd_event(vk, 0, ext | 0x0002, 0)
    time.sleep(0.5)
    b = sp.get_transform()
    sp_ = [a0.location.distance(a1.location) / (t1 - t0) for (t0, a0), (t1, a1) in zip(trace, trace[1:]) if t1 > t0]
    mid = sp_[len(sp_) // 4: 3 * len(sp_) // 4]
    jerk = [abs(y - x) for x, y in zip(mid, mid[1:])]
    res[label] = {"yaw": round((b.rotation.yaw - a.rotation.yaw + 180) % 360 - 180, 1),
                  "pitch": round(b.rotation.pitch - a.rotation.pitch, 1), "moved_m": round(a.location.distance(b.location), 1),
                  "speed_mean": round(statistics.fmean(mid), 1) if mid else None,
                  "frame_to_frame_change": round(statistics.fmean(jerk), 2) if jerk else None}
    print(label, res[label], flush=True)
world.remove_on_tick(cb)
out["arrows"] = res
(ROOT / "out" / "probe_keys.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
