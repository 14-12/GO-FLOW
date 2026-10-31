"""A top-down video of one Town05 junction running its signals, from a SUMO run (no CARLA needed).

    python tools/video_junction.py                       # junction 1126, 3 minutes, out/videos/junction_1126.mp4
    python tools/video_junction.py --junction 53 --seconds 240 --controller actuated

What it shows, each frame drawn from the simulation's own state (scenarios/town05, the signals of
tools/build_town05_signals.py):
* the roads, crossings and pavements of the junction, every vehicle (its SUMO size and heading) and pedestrian;
* each movement's signal at its stop line: an arrow per lane and direction (left / straight / right), green, amber,
  red (a green with a white edge would be one that must give way: none since 27 Oct, cosim/signal_plan.py);
* each crossing's pedestrian signal at both ends: WALK (green), flashing DON'T WALK (the clearance: nobody starts,
  those on it get across), steady DON'T WALK;
* a panel: the phase, what runs in it, the time to the next switch, the cycle's phases.
Vehicles and people are added on the junction's own movements and crossings so every phase has something to show.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOWN05 = ROOT / "scenarios" / "town05"
OUT = ROOT / "out" / "videos"
DIR_NAME = {"s": "straight", "r": "right", "l": "left", "t": "U-turn"}
COL = {"G": "#22c55e", "g": "#22c55e", "y": "#f59e0b", "r": "#ef4444", "u": "#f59e0b", "o": "#9ca3af"}


def capture(junction: str, seconds: float, warmup: float, controller: str, seed: int, scale: float,
            veh_rate: float, ped_rate: float, port: int) -> dict:
    """Run SUMO; return the junction's geometry and one snapshot per step."""
    import sumolib

    from cosim.agents_actuated import make as make_agents
    from cosim.signal_audit import load
    from cosim.sumo_side import SumoSide

    sys.path.insert(0, str(ROOT / "tools"))
    from check_signal_scenarios import geometry

    net = sumolib.net.readNet(str(TOWN05 / "town05.net.xml"), withInternal=True, withPedestrianConnections=True)
    node = net.getNode(junction)
    cx, cy = node.getCoord()
    R = 48.0

    def near(shape):
        return any(math.hypot(x - cx, y - cy) < R + 20 for x, y in shape[:1] + shape[-1:]) or \
            any(math.hypot(x - cx, y - cy) < R for x, y in shape)

    lanes = []
    for e in net.getEdges(withInternal=True):
        for ln in e.getLanes():
            sh = [p[:2] for p in ln.getShape()]
            if not near(sh):
                continue
            fn = e.getFunction()
            kind = ("crossing" if fn == "crossing" else "walkingarea" if fn == "walkingarea" else
                    "sidewalk" if ln.allows("pedestrian") and not ln.allows("passenger") else
                    "none" if not ln.allows("passenger") and not ln.allows("pedestrian") else
                    "internal" if fn == "internal" else "road")
            lanes.append({"id": ln.getID(), "kind": kind, "shape": sh, "width": ln.getWidth()})
    jshape = [p[:2] for p in node.getShape()]
    jn = load(TOWN05 / "town05.net.xml")[junction]
    geo = geometry(junction)
    heads = []  # (link index, x, y, heading, dir): one arrow per lane and direction, one behind the other
    by_lane: dict[str, list] = {}
    for lk in jn.links:
        if lk.kind == "vehicle":
            by_lane.setdefault(lk.from_lane, []).append(lk)
    for lane_id, lks in by_lane.items():
        ln = net.getLane(lane_id)
        sh = [p[:2] for p in ln.getShape()]
        (x0, y0), (x1, y1) = sh[-2], sh[-1]
        h = math.atan2(y1 - y0, x1 - x0)
        for j, lk in enumerate(sorted(lks, key=lambda k: {"s": 0, "l": 1, "r": 2}.get(k.dir, 3))):
            back = 3.0 + 3.8 * j
            heads.append({"link": lk.index, "x": x1 - back * math.cos(h), "y": y1 - back * math.sin(h), "h": h,
                          "dir": lk.dir, "lane": lane_id, "approach": lk.approach})
    cross = []
    for lk in jn.links:
        if lk.kind != "crossing":
            continue
        sh = [p[:2] for p in net.getEdge(lk.crossing).getLanes()[0].getShape()]
        cross.append({"link": lk.index, "id": lk.crossing, "shape": sh, "length": lk.length,
                      "width": net.getEdge(lk.crossing).getLanes()[0].getWidth()})

    policy = make_agents(controller) if controller.startswith("agents") else None
    s = SumoSide(TOWN05, controller if policy is None else "agents", seed=seed, port=port, label="video",
                 policy=policy, extras=True, emergency_every=120.0,
                 extra_args=["--scale", f"{scale:g}", "--end", f"{warmup + seconds + 5:g}"])
    s.extras.config.pedestrian_trips = True
    if policy is not None:
        ex = s.extras
        policy.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    c = s.conn
    rng = random.Random(seed)
    veh = []
    for side, a in geo["approaches"].items():
        for d, out in a["out"].items():
            veh.append(([a["in"], out], veh_rate * {"s": 0.7, "r": 0.15, "l": 0.15}.get(d, 0.1)))
    for k, (edges, _) in enumerate(veh):
        c.route.add(f"vid_r{k}", edges)
    nv = np_ = 0
    frames = []
    logic = {lg.programID: lg for lg in c.trafficlight.getAllProgramLogics(junction)}
    try:
        while s.time < warmup + seconds:
            r = s.step()
            now = r.time
            dt = s.step_length
            if now > warmup - 120:
                for k, (edges, rate) in enumerate(veh):
                    if rng.random() < rate * dt / 3600.0:
                        nv += 1
                        c.vehicle.add(f"vid{nv}", f"vid_r{k}", typeID="pas_passenger", depart="now",
                                      departLane="best", departSpeed="max")
                for cr, (inc, outg, _, _) in geo["crossings"].items():
                    if rng.random() < ped_rate * dt / 3600.0:
                        for a, b in ((inc, outg), (outg, inc)) if rng.random() < 0.5 else ((inc, outg),):
                            np_ += 1
                            pid = f"vidp{np_}"
                            try:
                                ln = c.lane.getLength(f"{a}_0")
                                start = max(0.0, ln - rng.uniform(4, 15)) if a == inc else rng.uniform(4, 15)
                                c.person.add(pid, a, min(start, ln), typeID="twin_pedestrian")
                                c.person.appendWalkingStage(pid, [a, b], arrivalPos=8.0 if b == outg else
                                                            max(0.0, c.lane.getLength(f"{b}_0") - 8.0))
                            except Exception:  # noqa: BLE001
                                pass
            if now < warmup:
                continue
            vs = []
            for vid, v in r.vehicles.items():
                if math.hypot(v.x - cx, v.y - cy) > R + 15:
                    continue
                try:
                    w = c.vehicle.getWidth(vid)
                except Exception:  # noqa: BLE001
                    w = 1.8
                vs.append((vid, v.x, v.y, v.angle, v.length, w, v.vclass))
            ps = [(pid, p.x, p.y) for pid, p in r.persons.items() if math.hypot(p.x - cx, p.y - cy) < R + 10]
            ph = c.trafficlight.getPhase(junction)
            pid_ = c.trafficlight.getProgram(junction)
            frames.append({"t": now, "state": r.tls[junction], "phase": ph, "name": c.trafficlight.getPhaseName(junction),
                           "next": c.trafficlight.getNextSwitch(junction) - now, "veh": vs, "ped": ps,
                           "program": pid_})
        phases = [(p.state, p.duration, p.minDur, p.maxDur, getattr(p, "name", "")) for p in logic[frames[-1]["program"]].phases]
    finally:
        s.close()
    return {"junction": junction, "cx": cx, "cy": cy, "R": R, "lanes": lanes, "jshape": jshape, "heads": heads,
            "cross": cross, "phases": phases, "frames": frames, "jn": jn, "controller": controller, "scale": scale}


def phase_text(jn, state: str) -> list[str]:
    by_app: dict[str, list[str]] = {}
    for lk in jn.links:
        if lk.kind == "vehicle" and state[lk.index] in "Gg":
            tag = DIR_NAME.get(lk.dir, lk.dir) + (" (gives way)" if state[lk.index] == "g" else "")
            if tag not in by_app.setdefault(lk.approach, []):
                by_app[lk.approach].append(tag)
    names = {"N": "North", "S": "South", "E": "East", "W": "West"}
    out = [f"{names.get(a, a)}: " + ", ".join(sorted(v, key=lambda x: ("straight" not in x, x))) for a, v in sorted(by_app.items())]
    return out


def render(cap: dict, out: Path, sub: int = 2, fps: int = 12, max_frames: int | None = None) -> Path:
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, Polygon

    from cosim.ped_lights import ped_aspect

    cx, cy, R = cap["cx"], cap["cy"], cap["R"]
    jn = cap["jn"]
    frames = cap["frames"]
    phases = cap["phases"]
    clear_states: dict[int, set[str]] = {}
    for k, (st, *_rest, nm) in enumerate(phases):
        if nm == "walk clearance":
            was = phases[k - 1][0]
            for cr in cap["cross"]:
                i = cr["link"]
                if was[i] in "Gg" and st[i] not in "Gg":
                    clear_states.setdefault(i, set()).add(st)

    W, H, DPI = 1280, 720, 100
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#0f172a")
    ax = fig.add_axes([0.0, 0.0, 0.5625, 1.0])
    pan = fig.add_axes([0.5625, 0.0, 0.4375, 1.0])

    def lane_poly(sh, w):
        left, right = [], []
        for i, (x, y) in enumerate(sh):
            if i + 1 < len(sh):
                dx, dy = sh[i + 1][0] - x, sh[i + 1][1] - y
            else:
                dx, dy = x - sh[i - 1][0], y - sh[i - 1][1]
            d = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / d * w / 2, dx / d * w / 2
            left.append((x + nx, y + ny))
            right.append((x - nx, y - ny))
        return left + right[::-1]

    def draw_static():
        ax.set_facecolor("#1e3a2a")
        for ln in cap["lanes"]:
            if ln["kind"] in ("walkingarea",):
                if len(ln["shape"]) >= 3:
                    ax.add_patch(Polygon(ln["shape"], closed=True, fc="#cbd5e1", ec="none", zorder=1))
        for ln in cap["lanes"]:
            k = ln["kind"]
            if k == "road":
                ax.add_patch(Polygon(lane_poly(ln["shape"], ln["width"]), fc="#374151", ec="none", zorder=1))
            elif k == "sidewalk":
                ax.add_patch(Polygon(lane_poly(ln["shape"], ln["width"]), fc="#cbd5e1", ec="none", zorder=1))
            elif k == "none":
                ax.add_patch(Polygon(lane_poly(ln["shape"], ln["width"]), fc="#4b5563", ec="none", zorder=1))
        if len(cap["jshape"]) >= 3:
            ax.add_patch(Polygon(cap["jshape"], fc="#374151", ec="none", zorder=1.5))
        for ln in cap["lanes"]:  # lane edges
            if ln["kind"] == "road":
                p = lane_poly(ln["shape"], ln["width"])
                n = len(ln["shape"])
                xs, ys = zip(*p[:n])
                ax.plot(xs, ys, color="#e5e7eb", lw=0.6, ls=(0, (4, 4)), zorder=2, alpha=0.6)
        for cr in cap["cross"]:  # zebra
            sh = cr["shape"]
            (x0, y0), (x1, y1) = sh[0], sh[-1]
            L = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            n = int(L / 0.6)
            hw = cr["width"] / 2
            nx, ny = -uy, ux
            for i in range(1, n - 1, 2):  # bars across the crossing, 0.6 m wide, 0.6 m apart
                a0, a1 = i * 0.6, i * 0.6 + 0.6
                ax.add_patch(Polygon([(x0 + ux * a0 + nx * hw, y0 + uy * a0 + ny * hw),
                                      (x0 + ux * a1 + nx * hw, y0 + uy * a1 + ny * hw),
                                      (x0 + ux * a1 - nx * hw, y0 + uy * a1 - ny * hw),
                                      (x0 + ux * a0 - nx * hw, y0 + uy * a0 - ny * hw)],
                                     fc="white", ec="none", alpha=0.85, zorder=2.2))
        ax.set_xlim(cx - R, cx + R)
        ax.set_ylim(cy - R, cy + R)
        ax.set_aspect("equal")
        ax.axis("off")

    def veh_poly(x, y, ang, L, w):
        # SUMO: position = front bumper middle, angle: degrees clockwise from north
        a = math.radians(90.0 - ang)
        ux, uy = math.cos(a), math.sin(a)
        nx, ny = -uy, ux
        bx, by = x - ux * L, y - uy * L
        return [(x + nx * w / 2, y + ny * w / 2), (x - nx * w / 2, y - ny * w / 2),
                (bx - nx * w / 2, by - ny * w / 2), (bx + nx * w / 2, by + ny * w / 2)]

    def arrow(x, y, h, d, col, edge):
        L = 2.6
        if d == "s":
            dx, dy = math.cos(h) * L, math.sin(h) * L
            ox, oy = 0.0, 0.0
        else:
            turn = h + (math.pi / 2 if d == "l" else -math.pi / 2)
            dx, dy = math.cos(turn) * L * 0.8, math.sin(turn) * L * 0.8
            ox, oy = math.cos(h) * 0.9, math.sin(h) * 0.9
        return FancyArrow(x + ox - dx / 2, y + oy - dy / 2, dx, dy, width=0.55, head_width=1.5, head_length=1.0,
                          length_includes_head=True, fc=col, ec=edge, lw=1.2 if edge != col else 0.0, zorder=6)

    # timeline of the program
    cyc = [(k, st, d, nm) for k, (st, d, mn, mx, nm) in enumerate(phases)]

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess

    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    prev = {v[0]: v for v in frames[0]["veh"]}
    n_out = 0
    total = len(frames) * sub if max_frames is None else min(max_frames, len(frames) * sub)
    for fi in range(1, len(frames)):
        f0, f1 = frames[fi - 1], frames[fi]
        v0 = {v[0]: v for v in f0["veh"]}
        p0 = {p[0]: p for p in f0["ped"]}
        for k in range(sub):
            if n_out >= total:
                break
            a = k / sub
            t = f0["t"] + a * (f1["t"] - f0["t"])
            state = f0["state"] if a < 1.0 else f1["state"]
            ax.clear()
            pan.clear()
            draw_static()
            for vid, x, y, ang, L, w, vc in f1["veh"]:
                if vid in v0:
                    _, x0_, y0_, ang0, *_ = v0[vid]
                    da = ((ang - ang0 + 180) % 360) - 180
                    x, y, ang = x0_ + (x - x0_) * a, y0_ + (y - y0_) * a, ang0 + da * a
                col = {"emergency": "#f8fafc", "authority": "#1d4ed8", "bus": "#eab308", "truck": "#a16207",
                       "delivery": "#9a3412", "motorcycle": "#a855f7"}.get(vc, "#60a5fa")
                ax.add_patch(Polygon(veh_poly(x, y, ang, L, w), fc=col, ec="#0f172a", lw=0.6, zorder=4))
                if vc == "emergency":
                    ax.add_patch(Polygon(veh_poly(x, y, ang, L * 0.25, w), fc="#ef4444", ec="none", zorder=4.1))
            for pid, x, y in f1["ped"]:
                if pid in p0:
                    x = p0[pid][1] + (x - p0[pid][1]) * a
                    y = p0[pid][2] + (y - p0[pid][2]) * a
                ax.plot([x], [y], "o", ms=4.2, mfc="#fde047", mec="#111827", mew=0.8, zorder=5)
            # vehicle signals
            for hd in cap["heads"]:
                ch = state[hd["link"]]
                col = COL.get(ch, "#9ca3af")
                edge = "white" if ch == "g" else col
                ax.add_patch(arrow(hd["x"], hd["y"], hd["h"], hd["dir"], col, edge))
            # pedestrian signals at both ends of each crossing
            flash_on = int(t * 2) % 2 == 0
            for cr in cap["cross"]:
                asp = ped_aspect(state, cr["link"], clear_states.get(cr["link"]))
                sh = cr["shape"]
                (x0, y0), (x1, y1) = sh[0], sh[-1]
                L = math.hypot(x1 - x0, y1 - y0)
                ux, uy = (x1 - x0) / L, (y1 - y0) / L
                for (px, py), sgn in (((x0, y0), -1), ((x1, y1), 1)):
                    bx, by = px + ux * sgn * 2.2, py + uy * sgn * 2.2
                    if asp == "walk":
                        fc, txt = "#16a34a", "WALK"
                    elif asp == "flashing":
                        fc, txt = ("#dc2626" if flash_on else "#3f1d1d"), "DON'T\nWALK"
                    else:
                        fc, txt = "#991b1b", "DON'T\nWALK"
                    ax.add_patch(Polygon([(bx - 2.1, by - 1.3), (bx + 2.1, by - 1.3), (bx + 2.1, by + 1.3),
                                          (bx - 2.1, by + 1.3)], fc=fc, ec="white", lw=0.8, zorder=7))
                    ax.text(bx, by, txt, color="white", fontsize=5.2, ha="center", va="center", zorder=8,
                            fontweight="bold", linespacing=0.9)
            for lbl, (dx, dy) in {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}.items():
                ax.text(cx + dx * (R - 4), cy + dy * (R - 4), lbl, color="white", fontsize=14, ha="center",
                        va="center", fontweight="bold", zorder=9, alpha=0.8)
            # panel
            pan.set_facecolor("#0f172a")
            pan.set_xlim(0, 1)
            pan.set_ylim(0, 1)
            pan.axis("off")
            fph = f0["phase"]
            nm = f0["name"]
            veh_lines = phase_text(jn, state)
            walks = [cr["id"] for cr in cap["cross"] if ped_aspect(state, cr["link"], clear_states.get(cr["link"])) == "walk"]
            flash = [cr["id"] for cr in cap["cross"] if ped_aspect(state, cr["link"], clear_states.get(cr["link"])) == "flashing"]
            names = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST"}
            veh = [lk for lk in jn.links if lk.kind == "vehicle"]
            on = sorted({lk.approach for lk in veh if state[lk.index] in "Gg"}, key="NSEW".index)
            amber = sorted({lk.approach for lk in veh if state[lk.index] == "y"}, key="NSEW".index)
            part = any(state[lk.index] not in "Gg" for lk in veh if lk.approach in on)
            who = " + ".join(names[a] for a in on) + (" (right lanes)" if part and len(on) > 1 else "")
            if amber:
                title, sub_t = " + ".join(names[a] for a in amber) + " YELLOW", "the green ends: stop if you can"
            elif on:
                title, sub_t = f"{who} GREEN", "a free go: nothing crossing its path is green"
            elif walks:
                title, sub_t = "WALK", "every car red: people cross on all the crosswalks"
            elif flash:
                title, sub_t = "PEDESTRIAN CLEARANCE", "every car red until the crosswalks are clear"
            else:
                title, sub_t = "ALL RED", "the junction clears before the next green"
            pan.text(0.05, 0.95, f"Town05 junction {cap['junction']}", color="#94a3b8", fontsize=11, va="top")
            pan.text(0.05, 0.905, title, color="white", fontsize=19, fontweight="bold", va="top")
            pan.text(0.05, 0.845, sub_t, color="#e2e8f0", fontsize=11, va="top")
            rem = max(0.0, f0["next"] - (t - f0["t"]))
            pan.text(0.05, 0.795, f"next switch in {rem:4.1f} s   (sim time {t / 60:5.1f} min)", color="#cbd5e1",
                     fontsize=9.5, va="top", family="monospace")
            y = 0.745
            pan.text(0.05, y, "Cars allowed to go:", color="#94a3b8", fontsize=10, va="top")
            y -= 0.045
            for ln in veh_lines or ["none (all stop)"]:
                pan.text(0.08, y, ln, color="#22c55e" if veh_lines else "#ef4444", fontsize=10.5, va="top")
                y -= 0.045
            pan.text(0.05, y - 0.01, "Pedestrians:", color="#94a3b8", fontsize=10, va="top")
            y -= 0.055
            if walks:
                pan.text(0.08, y, f"WALK on {len(walks)} crossings, every car red", color="#22c55e", fontsize=10.5,
                         va="top")
            elif flash:
                pan.text(0.08, y, "flashing DON'T WALK: no one starts,\nthose on the crossing finish", color="#f87171",
                         fontsize=10.5, va="top")
                y -= 0.04
            else:
                pan.text(0.08, y, "DON'T WALK everywhere", color="#f87171", fontsize=10.5, va="top")
            # cycle strip
            y0 = 0.36
            cyc_txt = (f"The cycle: {sum(d for _, _, d, _ in cyc):g} s at every signal, kept in step (green waves)"
                       if cap["controller"] == "coordinated" else
                       "The cycle (greens 5-30 s on demand; the walk when someone waits)")
            pan.text(0.05, y0 + 0.05, cyc_txt, color="#94a3b8", fontsize=9.5, va="top")
            x = 0.05
            total_d = sum(min(max(d, 3.0), 20.0) for _, _, d, _ in cyc)
            for k, st, d, nmk in cyc:
                wdt = 0.9 * min(max(d, 3.0), 20.0) / total_d
                car_on = sorted({lk.approach for lk in jn.links if lk.kind == "vehicle" and st[lk.index] in "Gg"},
                                key="NSEW".index)
                kind = ("y" if "y" in st else "w" if not car_on and "G" in st else "f" if "u" in st
                        or (nmk == "walk clearance" and car_on) else "r" if not car_on else "g")
                colk = {"g": "#16a34a", "w": "#0ea5e9", "f": "#65a30d", "y": "#f59e0b", "r": "#dc2626"}[kind]
                pan.add_patch(Polygon([(x, y0 - 0.06), (x + wdt, y0 - 0.06), (x + wdt, y0), (x, y0)], fc=colk,
                                      ec="white" if k == fph else "#0f172a", lw=2.5 if k == fph else 0.5))
                lbl = {"g": "+".join(car_on) + "\ngreen", "w": "walk", "f": "flash\nDW", "y": "Y", "r": "AR"}[kind]
                pan.text(x + wdt / 2, y0 - 0.03, lbl, color="white", fontsize=7, ha="center", va="center")
                x += wdt
            # legend
            ly = 0.2
            items = [("#22c55e", "green arrow: go (nothing crossing your path\nis green: no waiting inside the junction)"),
                     ("#f59e0b", "amber: stop if you can"), ("#ef4444", "red: stop")]
            for i, (cl, tx) in enumerate(items):
                pan.add_patch(FancyArrow(0.06, ly - i * 0.05, 0.04, 0, width=0.012, head_width=0.03, head_length=0.015,
                                         fc=cl, ec="white" if "give way" in tx else cl, lw=1.2))
                pan.text(0.13, ly - i * 0.05, tx, color="#e2e8f0", fontsize=8.5, va="center")
            who = ("coordinated (one cycle, green waves)" if cap["controller"] == "coordinated"
                   else f"actuated + junction agents ({cap['controller']})")
            pan.text(0.05, 0.01, f"SUMO 1.27.1 · signals: {who} · "
                     f"{0.5 * fps / sub:g}x real time", color="#64748b", fontsize=7.5, va="bottom")
            fig.canvas.draw()
            proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
            n_out += 1
        if n_out >= total:
            break
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--junction", default="1126")
    ap.add_argument("--seconds", type=float, default=180.0, help="simulated seconds shown")
    ap.add_argument("--warmup", type=float, default=420.0)
    ap.add_argument("--controller", default="agents")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--veh-rate", type=float, default=250.0, help="vehicles/h added per approach")
    ap.add_argument("--ped-rate", type=float, default=90.0, help="people/h added per crossing")
    ap.add_argument("--speed", type=float, default=2.0, help="video speed (x real time)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--port", type=int, default=9400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cap = capture(a.junction, a.seconds, a.warmup, a.controller, a.seed, a.scale, a.veh_rate, a.ped_rate, a.port)
    # frames every 0.5 s of simulation; sub frames between them: 0.5 s * fps / sub = speed
    sub = max(1, round(0.5 * a.fps / a.speed))
    out = Path(a.out) if a.out else OUT / f"junction_{a.junction}.mp4"
    render(cap, out, sub=sub, fps=a.fps)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
