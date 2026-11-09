"""Several cars in the same car park at once (cosim/parking_motion.py, 9 Nov), from above: 8 cars sent to the Marché
one every 4 s (tools/check_parking_crowd.py), three moments side by side, each car in its own colour with the way it
has just come, the one waiting for its way to be clear marked.

    python tools/parking_crowd_figure.py            -> docs/img/town05_parking_crowd.png
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
OUT = ROOT / "docs" / "img" / "town05_parking_crowd.png"
FONTS = ROOT / "assets" / "fonts"
COLOURS = [(52, 211, 153), (249, 140, 34), (96, 165, 250), (244, 114, 182), (250, 204, 21), (167, 139, 250),
           (45, 212, 191), (251, 113, 133)]


def run(place="marche", cars=8, every=4.0, seed=5):
    """Every step: {car: (x, y, angle, on a drive?, waiting?)} (the car's middle)."""
    import sumolib
    import traci

    from cosim import parking_motion as pm
    from cosim import purposes as P
    from cosim.sumo_side import Vehicle

    scen = ROOT / "scenarios" / "town05"
    traci.start([sumolib.checkBinary("sumo"), "-n", str(scen / "town05.net.xml"), "-a", str(P.ADD), "--step-length",
                 "0.5", "--no-step-log", "--no-warnings", "--time-to-teleport", "-1", "--seed", str(seed)],
                label="crowd_fig")
    c = traci.getConnection("crowd_fig")
    frames = []
    try:
        d = P.PurposeDemand(c, clock_speed=240.0, seed=seed)
        d.rng = random.Random(seed)
        d.purpose_at = lambda hour, vclass: "shopping"
        d.places.choose = lambda rng, what, hour, exclude=None: place
        motion = pm.ParkingMotion(d, c)
        launched, t_next, outs = 0, 1.0, set()
        for _ in range(2400):
            c.simulationStep()
            t = c.simulation.getTime()
            if launched < cars and t >= t_next:
                if d.launch(f"crowd{launched}", "DEFAULT_VEHTYPE", "passenger", 10.0, (), ()):
                    launched += 1
                t_next = t + every
            vs = {}
            for vid in c.vehicle.getIDList():
                x, y = c.vehicle.getPosition(vid)
                vs[vid] = Vehicle(vid, "passenger", "t", c.vehicle.getLength(vid), x, y, c.vehicle.getAngle(vid),
                                  c.vehicle.getSpeed(vid), 0)
            d.step([])
            for vid, aid in d.parked.items():
                v = vs.get(vid)
                if v is not None and "spaces" in d.places.areas[aid]:
                    a = math.radians(v.angle)
                    v.x += math.sin(a) * v.length / 2
                    v.y += math.cos(a) * v.length / 2
            motion.step(t, vs)
            fr = {}
            for vid, v in vs.items():
                if not vid.startswith("crowd") or abs(v.x) > 1e5:
                    continue
                a = math.radians(v.angle)
                dv = motion.drives.get(vid)
                waiting = dv is not None and (dv.progress(t) < 0 or dv.rate < 0.05) and not dv.done(t)
                fr[vid] = (v.x - math.sin(a) * v.length / 2, v.y - math.cos(a) * v.length / 2, v.angle, dv is not None,
                           waiting)
            frames.append((t, fr))
            outs |= {vid for vid, dv in motion.drives.items() if dv.kind == "out"}
            if launched >= cars and len(outs) >= launched and not motion.drives:
                break
        spaces = d.places.areas[d.places.park_of[place][0]]["spaces"]
        return frames, spaces, dict(motion.stats)
    finally:
        c.close()


def main() -> int:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont

    frames, spaces, stats = run()
    busy = [(sum(1 for f in fr.values() if f[3]), i) for i, (_t, fr) in enumerate(frames)]
    # three moments: the busiest, and the busiest well before and well after it
    k1 = max(busy)[1]
    k0 = max((b for b in busy if b[1] < k1 - 40), default=busy[0])[1]
    k2 = max((b for b in busy if b[1] > k1 + 40), default=busy[-1])[1]
    card = json.loads((ROOT / "scenarios" / "town05" / "places.json").read_text(encoding="utf-8"))
    sat = card["sat"]
    (cx, cy), mpp, N = sat["centre"], sat["mpp"], sat["size"]
    base = Image.open(ROOT / "out" / "town05_places" / "sat_rgb.png").convert("RGB")
    xs = [p[0] for p in spaces]
    ys = [p[1] for p in spaces]
    x0, x1, y0, y1 = min(xs) - 14, max(xs) + 10, min(ys) - 14, max(ys) + 8

    def ipx(x, y):
        return (N / 2 + (x - cx) / mpp, N / 2 - (y - cy) / mpp)
    a, b = ipx(x0, y1), ipx(x1, y0)
    crop = base.crop((int(a[0]), int(a[1]), int(b[0]), int(b[1])))
    S = 2.6
    W, H = int(crop.width * S), int(crop.height * S)
    img = ImageEnhance.Brightness(ImageEnhance.Color(crop).enhance(0.55)).enhance(0.62).resize((W, H), Image.LANCZOS)
    head, gap = 150, 16
    canvas = Image.new("RGB", (3 * W + 2 * gap, H + head + 50), (10, 13, 22))
    dr = ImageDraw.Draw(canvas, "RGBA")
    f_big = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 46)
    f_mid = ImageFont.truetype(str(FONTS / "Barlow-Medium.ttf"), 22)
    f_sm = ImageFont.truetype(str(FONTS / "Barlow-SemiBold.ttf"), 18)
    WHITE, MUTED = (238, 242, 250), (150, 162, 190)
    order = sorted({vid for _t, fr in frames for vid in fr}, key=lambda v: int(v[5:]))
    col = {vid: COLOURS[i % len(COLOURS)] for i, vid in enumerate(order)}
    for n, k in enumerate((k0, k1, k2)):
        ox = n * (W + gap)
        canvas.paste(img, (ox, head))

        def px(x, y, ox=ox):
            p = ipx(x, y)
            return ((p[0] - int(a[0])) * S + ox, (p[1] - int(a[1])) * S + head)
        for sx, sy, _ang in spaces:
            q = px(sx, sy)
            dr.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=(56, 189, 248, 110))
        t, fr = frames[k]
        for vid, (x, y, ang, on, waiting) in fr.items():
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            trail = [frames[j][1][vid][:2] for j in range(max(0, k - 16), k + 1) if vid in frames[j][1]]
            if on and len(trail) > 1:
                dr.line([px(*p) for p in trail], fill=col[vid] + (200,), width=4)
            h = math.radians(ang)
            fx, fy = math.sin(h), math.cos(h)
            rx, ry = fy, -fx
            L, B = 2.3, 0.95
            pts = [px(x + fx * L + rx * B, y + fy * L + ry * B), px(x + fx * L - rx * B, y + fy * L - ry * B),
                   px(x - fx * L - rx * B, y - fy * L - ry * B), px(x - fx * L + rx * B, y - fy * L + ry * B)]
            c_ = col[vid] if on else (120, 130, 150)
            dr.polygon(pts, fill=c_ + (200 if on else 90,), outline=WHITE + (220 if on else 60,))
            if waiting:
                q = px(x, y)
                dr.text((q[0] + 12, q[1] - 12), "waits", font=f_sm, fill=WHITE)
        dr.text((ox + 16, head + 10), f"+{t - frames[k0][0]:.0f} s", font=f_mid, fill=WHITE)
    dr.text((28, 22), "SEVERAL CARS IN THE SAME CAR PARK", font=f_big, fill=WHITE)
    dr.text((30, 80), "Parking du Marché  ·  8 cars sent shopping, one every 4 s, a short stay  ·  SUMO's own run  ·  "
                      "each keeps to the right in the aisles, waits for its way to be clear", font=f_mid, fill=MUTED)
    dr.text((30, 110), f"{stats.get('started_later', 0)} drives started later for a clear way "
                       f"({stats.get('started_later_s', 0):.0f} s in all), {stats.get('gave_way', 0)} times one gave "
                       "way on its way; no two bodies overlapping (tools/check_parking_crowd.py)",
            font=f_sm, fill=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print("wrote", OUT, canvas.size, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
