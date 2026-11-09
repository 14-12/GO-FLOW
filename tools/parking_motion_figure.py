"""The drive into a space and out of it (cosim/parking_motion.py), from above: a car sent shopping at the Marché
Central, on SUMO's own run, its body drawn every second on the top-down picture of the Parking du Marché (in green
as it drives in, orange as it backs out and drives off), and SUMO's own jump for comparison.

    python tools/parking_motion_figure.py            -> docs/img/town05_parking.png
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "img" / "town05_parking.png"
FONTS = ROOT / "assets" / "fonts"


def run(area="pa_lot_marche", place="marche", seed=2):
    import sumolib
    import traci

    from cosim import parking_motion as pm
    from cosim import purposes as P
    from cosim.sumo_side import Vehicle

    scen = ROOT / "scenarios" / "town05"
    traci.start([sumolib.checkBinary("sumo"), "-n", str(scen / "town05.net.xml"), "-a", str(P.ADD), "--step-length",
                 "0.5", "--no-step-log", "--no-warnings", "--time-to-teleport", "-1"], label="parking_fig")
    c = traci.getConnection("parking_fig")
    shown, sumo_xy = [], []
    try:
        d = P.PurposeDemand(c, clock_speed=120.0, seed=3)
        d.rng = random.Random(seed)
        d.purpose_at = lambda hour, vclass: "shopping"
        d.places.choose = lambda rng, what, hour, exclude=None: place
        motion = pm.ParkingMotion(d, c)
        d.launch("car", "DEFAULT_VEHTYPE", "passenger", 10.0, (), ())
        seen_out = False
        for _ in range(3000):
            c.simulationStep()
            t = c.simulation.getTime()
            vs = {}
            for vid in c.vehicle.getIDList():
                x, y = c.vehicle.getPosition(vid)
                vs[vid] = Vehicle(vid, "passenger", "t", c.vehicle.getLength(vid), x, y, c.vehicle.getAngle(vid),
                                  c.vehicle.getSpeed(vid), 0)
            d.step([])
            if "car" in vs:
                sumo_xy.append((t, vs["car"].x, vs["car"].y, d.parked.get("car")))
            for vid, aid in d.parked.items():
                v = vs.get(vid)
                if v is not None and "spaces" in d.places.areas[aid]:
                    a = math.radians(v.angle)
                    v.x += math.sin(a) * v.length / 2
                    v.y += math.cos(a) * v.length / 2
            motion.step(t, vs)
            v = vs.get("car")
            if v is None:
                continue
            dv = motion.drives.get("car")
            h = math.radians(v.angle)
            shown.append((t, v.x - math.sin(h) * v.length / 2, v.y - math.cos(h) * v.length / 2, v.angle,
                          dv.kind if dv else None, v.signals))
            seen_out |= bool(dv and dv.kind == "out")
            if seen_out and dv is None and len(shown) > 12 and all(s[4] is None for s in shown[-12:]):
                break
        return shown, sumo_xy, d.places.areas[area], dict(motion.stats)
    finally:
        c.close()


def main() -> int:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont

    from cosim import parking_motion as pm

    shown, sumo_xy, area, stats = run()
    card = json.loads((ROOT / "scenarios" / "town05" / "places.json").read_text(encoding="utf-8"))
    sat = card["sat"]
    (cx, cy), mpp, N = sat["centre"], sat["mpp"], sat["size"]
    base = Image.open(ROOT / "out" / "town05_places" / "sat_rgb.png").convert("RGB")
    xs = [s[1] for s in shown if s[4]] + [p[0] for p in area["spaces"]]
    ys = [s[2] for s in shown if s[4]] + [p[1] for p in area["spaces"]]
    x0, x1, y0, y1 = min(xs) - 32, max(xs) + 32, min(ys) - 10, max(ys) + 12

    def ipx(x, y):
        return (N / 2 + (x - cx) / mpp, N / 2 - (y - cy) / mpp)
    a, b = ipx(x0, y1), ipx(x1, y0)
    crop = base.crop((int(a[0]), int(a[1]), int(b[0]), int(b[1])))
    S = 4.0                                               # output pixels per picture pixel
    W, H = int(crop.width * S), int(crop.height * S)
    img = ImageEnhance.Brightness(ImageEnhance.Color(crop).enhance(0.55)).enhance(0.62).resize((W, H), Image.LANCZOS)
    img = Image.blend(img, Image.new("RGB", img.size, (10, 16, 30)), 0.25)
    head = 150
    canvas = Image.new("RGB", (W, H + head), (10, 13, 22))
    canvas.paste(img, (0, head))
    dr = ImageDraw.Draw(canvas, "RGBA")

    def px(x, y):
        p = ipx(x, y)
        return ((p[0] - int(a[0])) * S, (p[1] - int(a[1])) * S + head)
    f_big = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 46)
    f_mid = ImageFont.truetype(str(FONTS / "Barlow-Medium.ttf"), 22)
    f_sm = ImageFont.truetype(str(FONTS / "Barlow-SemiBold.ttf"), 19)
    GREEN, ORANGE, RED, WHITE, MUTED = (52, 211, 153), (249, 140, 34), (244, 63, 94), (238, 242, 250), (150, 162, 190)

    for sx, sy, _ang in area["spaces"]:                    # the spaces, faint
        q = px(sx, sy)
        dr.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=(56, 189, 248, 120))

    # SUMO's own jump: its last place on the road and its space, dashed red
    road = [s for s in sumo_xy if s[3] is None]
    park_i = next((i for i, s in enumerate(sumo_xy) if s[3]), None)
    if park_i:
        p0, p1 = px(*sumo_xy[park_i - 1][1:3]), px(*sumo_xy[park_i][1:3])
        n = 16
        for k in range(0, n, 2):
            q0 = (p0[0] + (p1[0] - p0[0]) * k / n, p0[1] + (p1[1] - p0[1]) * k / n)
            q1 = (p0[0] + (p1[0] - p0[0]) * (k + 1) / n, p0[1] + (p1[1] - p0[1]) * (k + 1) / n)
            dr.line([q0, q1], fill=RED + (230,), width=4)
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dr.text((mx + 12, my - 10), "SUMO alone: one step, 42 m", font=f_sm, fill=RED)
    del road

    def body(s, col, alpha):
        _t, x, y, ang, *_ = s
        h = math.radians(ang)
        fx, fy = math.sin(h), math.cos(h)
        rx, ry = fy, -fx
        L, B = 2.3, 0.95
        pts = [px(x + fx * L + rx * B, y + fy * L + ry * B), px(x + fx * L - rx * B, y + fy * L - ry * B),
               px(x - fx * L - rx * B, y - fy * L - ry * B), px(x - fx * L + rx * B, y - fy * L + ry * B)]
        dr.polygon(pts, fill=col + (int(alpha * 0.55),), outline=col + (alpha,))
        nose = px(x + fx * L * 0.8, y + fy * L * 0.8)
        dr.ellipse([nose[0] - 3, nose[1] - 3, nose[0] + 3, nose[1] + 3], fill=WHITE + (alpha,))

    for kind, col in (("in", GREEN), ("out", ORANGE)):
        part = [s for s in shown if s[4] == kind]
        if not part:
            continue
        line = [px(s[1], s[2]) for s in part]
        dr.line(line, fill=col + (255,), width=5, joint="curve")
        for i, s in enumerate(part[::2]):
            body(s, col, 110 + int(120 * i / max(len(part[::2]) - 1, 1)))
        backing = [s for s in part if s[5] & pm.BACK]
        if backing and kind == "out":
            q = px(*backing[len(backing) // 2][1:3])
            dr.text((q[0] + 16, q[1] - 30), "backs out", font=f_sm, fill=ORANGE)
        moving = [s1[0] for s0, s1 in itertools.pairwise(part) if math.dist(s0[1:3], s1[1:3]) > 0.01]
        dur = (moving[-1] - part[0][0]) if moving else 0.0
        q = px(*part[0][1:3]) if kind == "in" else px(*part[-1][1:3])
        dy = -46 if kind == "in" else 26
        dr.text((q[0] - 30, q[1] + dy), f"{'IN' if kind == 'in' else 'OUT'}  {dur:.0f} s", font=f_sm, fill=col)

    dr.text((28, 22), "DRIVING INTO A SPACE, AND OUT OF IT", font=f_big, fill=WHITE)
    dr.text((30, 80), f"Parking du Marché  ·  a car sent shopping, SUMO's own run  ·  it joins the road "
                      f"{stats['seam_max_m']:.2f} m from where SUMO puts it back", font=f_mid, fill=MUTED)
    for i, (lab, col) in enumerate((("in: blinker, along the strip, down the aisle, into the space", GREEN),
                                    ("out: reversing lights, back into the aisle, then off to the road", ORANGE))):
        dr.rectangle([30, 118 + i * 0, 52, 130], fill=col) if i == 0 else dr.rectangle([30, H + head - 40, 52,
                                                                                         H + head - 28], fill=col)
        dr.text((60, 110) if i == 0 else (60, H + head - 48), lab, font=f_sm, fill=WHITE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print("wrote", OUT, canvas.size, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
