"""Town05 frame rate, measured on the CARLA window itself.

What is measured is what free-fly will feel like: CARLA in a window, asynchronous mode (the server
draws as fast as the GPU lets it, nobody waits for Python), no camera sensor and no image copied to
Python. 60 parked cars (physics off) stand in for the traffic.

    python tools/town05_fps_test.py --tag medium_dx11     # scripted flight: 15 s warm-up, 30 s high over
                                                          # the town, 30 s at street level; JSON + 2 screenshots
    python tools/town05_fps_test.py --fly                 # you fly (WASD + right mouse in the CARLA window),
                                                          # the frame rate is printed every second; Q here stops
    python tools/town05_fps_test.py --summary             # table of every scripted result (out/town05_fps)

Run by tools/town05_fps_test.bat, which restarts CARLA for each configuration.
In asynchronous mode a snapshot's delta_seconds is the real time since the previous frame.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "town05_fps"
LOGS = ROOT / "tools" / "logs"
MAP = "Town05"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="manual")
    ap.add_argument("--quality", default="?")
    ap.add_argument("--rhi", default="")
    ap.add_argument("--ini", default="?")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--wait", type=float, default=300.0, help="seconds to wait for the server to answer")
    ap.add_argument("--vehicles", type=int, default=60)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="per segment (high, street)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--fly", action="store_true")
    ap.add_argument("--summary", action="store_true")
    return ap.parse_args(argv)


# ------------------------------------------------------------------ server
def connect(host: str, port: int, wait: float):
    import carla

    t0 = time.time()
    while True:
        try:
            c = carla.Client(host, port)
            c.set_timeout(10.0)
            v = c.get_server_version()
            print(f"CARLA {v} answering after {time.time() - t0:.0f} s", flush=True)
            return c
        except RuntimeError:
            if time.time() - t0 > wait:
                raise SystemExit(f"no CARLA on {host}:{port} after {wait:.0f} s")
            time.sleep(3)


def prepare(client, n_vehicles: int, seed: int):
    """Town05 loaded, asynchronous mode, n parked cars. Returns (world, actor ids)."""
    import carla

    client.set_timeout(300.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != MAP:
        t = time.time()
        world = client.load_world(MAP)
        print(f"loaded {MAP} in {time.time() - t:.0f} s", flush=True)
    s = world.get_settings()
    s.synchronous_mode = False
    s.fixed_delta_seconds = None
    s.no_rendering_mode = False
    world.apply_settings(s)
    lib = world.get_blueprint_library()
    bps = []
    for b in lib.filter("vehicle.*"):
        try:
            if b.get_attribute("number_of_wheels").as_int() == 4:
                bps.append(b)
        except (IndexError, RuntimeError, ValueError):
            pass
    rng = random.Random(seed)
    pts = list(world.get_map().get_spawn_points())
    rng.shuffle(pts)
    batch = [carla.command.SpawnActor(rng.choice(bps), p).then(
        carla.command.SetSimulatePhysics(carla.command.FutureActor, False)) for p in pts[:n_vehicles]]
    ids = [r.actor_id for r in client.apply_batch_sync(batch, False) if not r.error]
    print(f"{len(ids)} parked cars", flush=True)
    return world, ids


def cleanup(client, ids) -> None:
    import carla

    try:
        client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], False)
    except RuntimeError:
        pass


# ------------------------------------------------------------------ camera paths
def town_frame(world):
    pts = world.get_map().get_spawn_points()
    xs = [p.location.x for p in pts]
    ys = [p.location.y for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return cx, cy, span


def look_at(px, py, pz, tx, ty, tz):
    import carla

    yaw = math.degrees(math.atan2(ty - py, tx - px))
    pitch = math.degrees(math.atan2(tz - pz, math.hypot(tx - px, ty - py)))
    return carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0))


def high_path(world, period: float = 30.0):
    """Circles the town centre 60 m up, one turn per `period` s."""
    cx, cy, span = town_frame(world)
    r = min(max(0.3 * span, 80.0), 220.0)

    def pose(t):
        a = 2 * math.pi * t / period
        return look_at(cx + r * math.cos(a), cy + r * math.sin(a), 60.0, cx, cy, 0.0)
    return pose


def street_path(world, speed: float = 12.0, seconds: float = 30.0):
    """Drives along the road from the waypoint nearest the centre, 3 m up, at `speed` m/s."""
    import carla

    cx, cy, _ = town_frame(world)
    cmap = world.get_map()
    wp = cmap.get_waypoint(carla.Location(cx, cy, 0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
    route = [wp]
    for _ in range(int(speed * seconds) + 50):
        nxt = wp.next(1.0)
        if not nxt:
            break
        wp = nxt[0]
        route.append(wp)

    def pose(t):
        w = route[int(t * speed) % len(route)].transform
        loc = carla.Location(w.location.x, w.location.y, w.location.z + 3.0)
        return carla.Transform(loc, carla.Rotation(-5.0, w.rotation.yaw, 0.0))
    return pose


# ------------------------------------------------------------------ measuring
def stats(d: list[float]) -> dict | None:
    d = sorted(x for x in d if x > 0)
    if not d:
        return None
    n, tot = len(d), sum(d)
    p99 = d[min(n - 1, int(0.99 * n))]
    return {"frames": n, "fps_avg": round(n / tot, 1), "fps_median": round(1.0 / d[n // 2], 1),
            "fps_1pct_low": round(1.0 / p99, 1), "ms_avg": round(1000 * tot / n, 1),
            "ms_worst": round(1000 * d[-1], 1)}


def grab_window(path: Path) -> bool:
    """Screenshot of the CARLA window's picture (what you see), with Pillow. False if not found."""
    try:
        import ctypes
        from ctypes import wintypes

        from PIL import ImageGrab

        user32 = ctypes.windll.user32
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(hwnd, _):
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            if user32.IsWindowVisible(hwnd) and cls.value == "UnrealWindow":
                found.append(hwnd)
            return True

        user32.EnumWindows(cb, 0)
        if not found:
            return False
        hwnd = found[0]
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.8)
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        ImageGrab.grab(bbox=(pt.x, pt.y, pt.x + rect.right, pt.y + rect.bottom), all_screens=True).save(path)
        return True
    except Exception as e:  # noqa: BLE001 - a screenshot must not fail the test
        print(f"window screenshot failed: {e}", flush=True)
        return False


def sensor_shot(world, tf, path: Path) -> None:
    """Fallback screenshot from an RGB camera at the same pose."""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "1280")
    bp.set_attribute("image_size_y", "720")
    cam = world.spawn_actor(bp, tf)
    q: queue.Queue = queue.Queue()
    cam.listen(q.put)
    try:
        img = None
        for _ in range(12):  # let the exposure settle
            img = q.get(timeout=30)
        img.save_to_disk(str(path))
    finally:
        cam.stop()
        cam.destroy()


def dpi_aware() -> None:
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:  # noqa: BLE001
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001 - not Windows
        pass


def scripted(a) -> int:
    dpi_aware()
    client = connect(a.host, a.port, a.wait)
    world, ids = prepare(client, a.vehicles, a.seed)
    spect = world.get_spectator()
    high, street = high_path(world), street_path(world, seconds=a.seconds)
    rec = {"seg": None, "high": [], "street": []}

    def on_tick(snap):
        seg = rec["seg"]
        if seg:
            rec[seg].append(snap.timestamp.delta_seconds)

    cb = world.on_tick(on_tick)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        def run(seg, pose, seconds):
            rec["seg"] = seg
            t0 = time.time()
            while (el := time.time() - t0) < seconds:
                spect.set_transform(pose(el))
                time.sleep(0.01)
            rec["seg"] = None

        print(f"warm-up {a.warmup:.0f} s", flush=True)
        run(None, high, a.warmup)
        t_start = time.time()
        print("measuring: high over the town", flush=True)
        run("high", high, a.seconds)
        print("measuring: street level", flush=True)
        run("street", street, a.seconds)
        t_end = time.time()
        world.remove_on_tick(cb)
        cb = None

        shots = {}
        for name, tf in (("high", high(4.0)), ("street", street(2.0))):
            spect.set_transform(tf)
            time.sleep(2.5)  # tiles, LODs and exposure settle
            p = OUT / f"{a.tag}_{name}.png"
            if not grab_window(p):
                p = OUT / f"{a.tag}_{name}_sensor.png"
                sensor_shot(world, tf, p)
            shots[name] = p.name

        res = {"mode": "test", "tag": a.tag, "quality": a.quality, "rhi": a.rhi or "default (DX12)", "ini": a.ini,
               "map": MAP, "vehicles": len(ids), "t_start": t_start, "t_end": t_end,
               "high": stats(rec["high"]), "street": stats(rec["street"]),
               "all": stats(rec["high"] + rec["street"]), "screenshots": shots}
        (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("tag", "high", "street", "all")}), flush=True)
    finally:
        if cb is not None:
            world.remove_on_tick(cb)
        cleanup(client, ids)
    return 0


def fly(a) -> int:
    import msvcrt

    client = connect(a.host, a.port, a.wait)
    world, ids = prepare(client, a.vehicles, a.seed)
    world.get_spectator().set_transform(high_path(world)(0.0))
    rec: list[float] = []
    cb = world.on_tick(lambda s: rec.append(s.timestamp.delta_seconds))
    print("\nFREE-FLY: click into the CARLA window, hold the RIGHT mouse button to look around,"
          "\n          W A S D to move, Q / E down / up. The frame rate is printed here every second."
          "\n          Press Q in THIS window to stop.\n", flush=True)
    last = 0
    try:
        while True:
            time.sleep(1.0)
            k = len(rec)
            d = rec[last:k]
            last = k
            if d:
                print(f"{len(d) / sum(d):5.1f} fps   worst frame {max(d) * 1000:4.0f} ms", flush=True)
            if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                break
    finally:
        world.remove_on_tick(cb)
        s = stats(rec)
        print(f"\nwhole session: {json.dumps(s)}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"fly_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(
            json.dumps({"mode": "fly", "ini": a.ini, "quality": a.quality, "rhi": a.rhi, "all": s}, indent=2),
            encoding="utf-8")
        cleanup(client, ids)
    return 0


# ------------------------------------------------------------------ summary
def _num(x: str) -> float:
    try:
        return float(x)
    except ValueError:  # "[N/A]" on some laptop GPUs
        return 0.0


def gpu_stats(csv_path: Path, t0: float, t1: float) -> dict:
    """nvidia-smi samples (timestamp, mem used, mem total, util, temp, power, clock) within [t0, t1]."""
    rows = []
    try:
        for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            try:
                t = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                if t0 <= t <= t1:
                    rows.append([_num(x) for x in f[1:7]])
            except ValueError:
                continue
    except OSError:
        return {}
    if not rows:
        return {}
    col = list(zip(*rows, strict=True))
    return {"mem_max_mb": int(max(col[0])), "mem_total_mb": int(col[1][0]),
            "util_avg": round(sum(col[2]) / len(rows)), "temp_max": int(max(col[3])),
            "power_avg_w": round(sum(col[4]) / len(rows), 1), "clock_avg_mhz": round(sum(col[5]) / len(rows))}


def log_facts(log_path: Path, ini_name: str) -> dict:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"log": "not found"}
    rhi = "DX11" if "LogD3D11RHI" in text else "DX12" if "LogD3D12RHI" in text else "?"
    wanted = set()
    ini = ROOT / "tools" / "fps_test" / f"engine_{ini_name}.ini"
    if ini.is_file():
        wanted = {ln.split("=")[0].strip().lower() for ln in ini.read_text().splitlines() if "=" in ln}
    applied = {m.group(1).lower() for m in re.finditer(r"Setting CVar \[\[([^:\]]+):", text)}
    return {"rhi_used": rhi, "ini_cvars": len(wanted), "ini_cvars_applied": len(wanted & applied)}


def summary() -> int:
    rows = []
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("mode") != "test":
            continue
        g = gpu_stats(LOGS / f"fps_{r['tag']}_gpu.csv", r["t_start"], r["t_end"])
        lf = log_facts(LOGS / f"fps_{r['tag']}_carla.log", r["ini"])
        rows.append((r, g, lf))
    order = ["epic_dx12_default", "epic_dx12_poolfix", "medium_dx12", "epic_dx11_poolfix", "medium_dx11", "low_dx11"]
    rows.sort(key=lambda x: order.index(x[0]["tag"]) if x[0]["tag"] in order else 99)
    lines = ["# Town05 frame rate (CARLA window, asynchronous, 1280x720, 60 parked cars)", "",
             "| config | quality | renderer | FPS high | FPS street | FPS all (1% low) | GPU memory max | GPU use | temp | ini cvars applied |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r, g, lf in rows:
        def fps(k, r=r):
            return f"{r[k]['fps_avg']}" if r.get(k) else "-"
        allv = r.get("all") or {}
        mem = f"{g['mem_max_mb']} / {g['mem_total_mb']} MB" if g else "-"
        util = f"{g['util_avg']} %" if g else "-"
        temp = f"{g['temp_max']} C" if g else "-"
        cv = f"{lf.get('ini_cvars_applied', '?')} / {lf.get('ini_cvars', '?')}" if r["ini"] != "none" else "-"
        lines.append(f"| {r['tag']} | {r['quality']} | {lf.get('rhi_used', '?')} | {fps('high')} | {fps('street')} | "
                     f"{allv.get('fps_avg', '-')} ({allv.get('fps_1pct_low', '-')}) | {mem} | {util} | {temp} | {cv} |")
    lines += ["", "Screenshots: out\\town05_fps\\<config>_high.png and _street.png."]
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.summary:
        return summary()
    if a.fly:
        return fly(a)
    return scripted(a)


if __name__ == "__main__":
    sys.exit(main())
