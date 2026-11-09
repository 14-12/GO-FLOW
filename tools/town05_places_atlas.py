"""The atlas of Town05's places: the town from above (tools/town05_places_probe.py), each place drawn over it with
its outline, its car park's spaces, where it is entered, its name and what it holds.

    python tools/town05_places_atlas.py      -> out/town05_places/atlas.png, docs/img/town05_places.jpg
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cosim.place_icons import STYLE, Painter, colour

PROBE = ROOT / "out" / "town05_places"
DOC = ROOT / "docs" / "img" / "town05_places.jpg"
K = 2                     # drawn at twice the size, then scaled down (smooth edges)
FONTS = ROOT / "assets" / "fonts"


def main() -> int:
    import sumolib

    card = json.loads((ROOT / "scenarios" / "town05" / "places.json").read_text(encoding="utf-8"))
    sat = card["sat"]
    (cx, cy), mpp, N = sat["centre"], sat["mpp"], sat["size"]

    def px(x, y):
        return ((N / 2 + (x - cx) / mpp) * K, (N / 2 - (y - cy) / mpp) * K)

    base = Image.open(PROBE / "sat_rgb.png").convert("RGB")
    base = ImageEnhance.Color(base).enhance(0.45)
    base = ImageEnhance.Brightness(base).enhance(0.5)
    tint = Image.new("RGB", base.size, (14, 22, 40))
    base = Image.blend(base, tint, 0.35).resize((N * K, N * K), Image.LANCZOS)
    W = N * K
    # vignette: the edges darker than the middle
    import numpy as np

    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32) / W - 0.5
    keep = np.clip(1.0 - (np.hypot(xx, yy) / 0.72) ** 3 * 0.85, 0.15, 1.0)
    arr = np.asarray(base).astype(np.float32) * keep[:, :, None] + np.array([6, 9, 16]) * (1 - keep[:, :, None])
    base = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    places = card["places"]
    areas = {a["id"]: a for a in card["parking_areas"]}

    fill = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fd, gd = ImageDraw.Draw(fill), ImageDraw.Draw(glow)
    order = sorted(places, key=lambda p: {"houses": 0, "kerb": 1}.get(p["kind"], 2))
    for p in order:
        col = colour(p["kind"])
        if p["kind"] == "kerb":
            for k in p["kerbs"]:
                sh = [px(*q) for q in net.getLane(k["lane"]).getShape()]
                gd.line(sh, fill=col + (230,), width=10 * K)
                fd.line(sh, fill=col + (255,), width=3 * K)
            continue
        if p.get("depot"):
            sh = [px(*q) for q in p["outline"]]
            gd.line(sh, fill=col + (230,), width=12 * K)
            fd.line(sh, fill=col + (255,), width=4 * K)
            continue
        if p.get("blocks"):                     # its buildings, each in its own footprint, in its outline
            poly = [px(*q) for q in p["outline"]]
            fd.polygon(poly, fill=col + (28,), outline=col + (120,), width=K)
            for b in p["blocks"]:
                poly = [px(*q) for q in b]
                gd.polygon(poly, fill=col + (150,))
                fd.polygon(poly, fill=col + (95,), outline=col + (255,), width=K)
            continue
        poly = [px(*q) for q in p["outline"]]
        fd.polygon(poly, fill=col + (60,))
        gd.line(poly + [poly[0]], fill=col + (255,), width=9 * K)
        fd.line(poly + [poly[0]], fill=col + (255,), width=2 * K)
    glow = glow.filter(ImageFilter.GaussianBlur(9 * K))
    img = Image.alpha_composite(base.convert("RGBA"), glow)
    img = Image.alpha_composite(img, fill)
    d = ImageDraw.Draw(img)

    # the car parks' spaces: a little box per space, on CARLA's painted bays
    pc = colour("parking")
    for a in areas.values():
        if a.get("hidden") or "spaces" not in a:
            continue
        for x, y, ang in a["spaces"]:
            t = math.radians(ang)
            fx, fy = math.sin(t), math.cos(t)          # the space's heading (SUMO angle: clockwise from north)
            rx, ry = fy, -fx
            hl, hw = 2.2, 1.05
            pts = [px(x + fx * sl * hl + rx * sw * hw, y + fy * sl * hl + ry * sw * hw)
                   for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            d.polygon(pts, outline=pc + (255,), fill=pc + (70,), width=2)
    # doors: a chevron on the road where the place is entered
    for p in places:
        door = p.get("door")
        if not door or p["kind"] == "kerb" or "parks_at" in p:
            continue
        from sumolib.geomhelper import positionAtShapeOffset
        ln = net.getLane(door["lane"])
        sh = ln.getShape()
        x0, y0 = positionAtShapeOffset(sh, door["pos"])
        x1, y1 = positionAtShapeOffset(sh, min(door["pos"] + 1.0, ln.getLength()))
        hx, hy = x1 - x0, y1 - y0
        n = math.hypot(hx, hy) or 1.0
        hx, hy = hx / n, hy / n
        ax, ay = px(x0, y0)
        col = colour(p["kind"])
        s = 9 * K
        f = (hx, -hy)
        r = (-f[1], f[0])
        d.polygon([(ax + f[0] * s, ay + f[1] * s), (ax - f[0] * s + r[0] * s, ay - f[1] * s + r[1] * s),
                   (ax - f[0] * s * 0.3, ay - f[1] * s * 0.3), (ax - f[0] * s - r[0] * s, ay - f[1] * s - r[1] * s)],
                  fill=col + (255,), outline=(10, 14, 24, 255))
        lx, ly = px(*p["label"])
        d.line([(ax, ay), (lx, ly)], fill=col + (150,), width=2 * K)

    # labels: a dark pill with the icon, the name, what it is and how many it holds; moved off each other
    f_name = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 26 * K)
    f_sub = ImageFont.truetype(str(FONTS / "Barlow-Medium.ttf"), 15 * K)
    paint = Painter(d, "pil")
    taken: list = []
    for p in sorted(places, key=lambda p: -len(p["name"])):
        if p["kind"] == "kerb" and p.get("capacity", 0) < 20:
            continue
        lx, ly = px(*p["label"])
        bits = [p.get("sub", "")]
        if p.get("capacity") and p["kind"] in ("parking", "kerb"):
            bits.append(f"{p['capacity']} spaces")
        elif p.get("homes"):
            bits.append(f"{p['homes']} homes")
        elif p.get("jobs"):
            bits.append(f"{p['jobs']} jobs")
        if p.get("hours") and p["hours"] != [0, 24]:
            bits.append(f"{p['hours'][0]}:00-{p['hours'][1]}:00")
        sub = "  ·  ".join(b for b in bits if b)
        wn = d.textlength(p["name"], font=f_name)
        ws = d.textlength(sub, font=f_sub)
        icon = 34 * K
        w = int(max(wn, ws) + icon + 30 * K)
        h = 56 * K
        for dy in (0, -70, 70, -140, 140, -210, 210):
            box = (lx - w / 2, ly - h / 2 + dy * K, lx + w / 2, ly + h / 2 + dy * K)
            if not any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in taken):
                break
        taken.append(box)
        col = colour(p["kind"])
        d.rounded_rectangle(box, radius=h // 2, fill=(10, 14, 24, 235), outline=col + (255,), width=2 * K)
        ix, iy = box[0] + h / 2, (box[1] + box[3]) / 2
        paint.icon(p["kind"], ix, iy, icon)
        tx = box[0] + h / 2 + icon / 2 + 10 * K
        d.text((tx, box[1] + 5 * K), p["name"], font=f_name, fill=(240, 244, 252, 255))
        d.text((tx, box[1] + 34 * K), sub, font=f_sub, fill=(170, 182, 205, 255))

    # title and legend
    f_t = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 64 * K)
    f_l = ImageFont.truetype(str(FONTS / "Barlow-Medium.ttf"), 18 * K)
    n_sp = sum(len(a.get("spaces", ())) for a in areas.values()) + sum(a.get("roadside", 0) for a in areas.values())
    homes = sum(p.get("homes", 0) for p in places)
    jobs = sum(p.get("jobs", 0) for p in places)
    d.rounded_rectangle((40 * K, 40 * K, 700 * K, 330 * K), radius=24 * K, fill=(8, 11, 20, 225),
                        outline=(60, 72, 96, 255), width=2 * K)
    d.text((70 * K, 58 * K), "TOWN05  ·  PLACES", font=f_t, fill=(240, 244, 252, 255))
    d.text((72 * K, 132 * K), f"{len([p for p in places if p['kind'] != 'kerb'])} places  ·  {homes} homes  ·  {jobs} jobs  "
                              f"·  {n_sp} parking spaces", font=f_l, fill=(170, 182, 205, 255))
    kinds = [k for k in STYLE if any(p["kind"] == k for p in places)]
    for i, k in enumerate(kinds):
        x = 72 * K + (i % 3) * 210 * K
        y = 180 * K + (i // 3) * 34 * K
        paint.icon(k, x + 12 * K, y + 12 * K, 24 * K)
        d.text((x + 32 * K, y), STYLE[k][2], font=f_l, fill=(220, 228, 240, 255))
    out = img.convert("RGB").resize((N, N), Image.LANCZOS)
    out.save(PROBE / "atlas.png")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    out.resize((1600, 1600), Image.LANCZOS).save(DOC, quality=88)
    print(f"atlas: {PROBE / 'atlas.png'}, {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
