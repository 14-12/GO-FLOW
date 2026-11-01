"""Reading a Tunisian number plate from a picture (31 Oct): find it on the back of a vehicle, straighten it, read it.

Classic number-plate recognition, as the first generation of real ANPR systems did it (and many still do for a
known national plate): a known plate design read with templates of its own font.

1. `find_plates`: candidate places on a vehicle's back: rows of character-sized strokes (light on dark or dark on
   light) side by side, plate-shaped blobs (the characters run into the border at small sizes), and bands where
   bright strokes crowd together.
2. `plate_quad`: around a candidate, the plate's panel (dark on a private plate, light on a state one) inside its
   thin border: its four corners, however the plate is tilted or seen at a slant.
3. `warp`: the panel straightened to the standard size (PLATE_W x PLATE_H).
4. `read_plate`: the characters, thresholded inside the panel; the Arabic word تونس in the middle splits the series
   (left, up to 3 digits) from the number (right, up to 4) - a state plate's dash likewise; each group is read by
   placing the plate font's own digits (Barlow Condensed SemiBold, assets/fonts, drawn at the group's height) along
   it, the best-matching sequence that covers it (so digits run together at small sizes are still read). The
   confidence comes from the weakest group's match and its margin over the next-best sequence: a reading the
   officer must check by eye when low. Over a vehicle's pictures, readings are voted (cosim/redlight.PlateReader).

The font and the layout are what a real ANPR system is built for (national plate specifications); nothing here
looks at which plate a car really carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLATE_W, PLATE_H = 240, 51      # the straightened plate (520 x 110 mm)
ASPECT = 520.0 / 110.0


@dataclass
class Reading:
    text: str | None
    conf: float                     # 0..1
    digits: list = field(default_factory=list)   # (digit, score, margin)
    box: tuple | None = None        # the plate in the crop it was read from (x1, y1, x2, y2)
    kind: str = "private"


# ------------------------------------------------------------------ the font's digits
@lru_cache(maxsize=1)
def templates() -> dict[str, np.ndarray]:
    """Each digit of the plate font, white on black, cut to its ink, as float arrays."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(ROOT / "assets" / "fonts" / "BarlowCondensed-SemiBold.ttf"), 120)
    out = {}
    for ch in "0123456789":
        im = Image.new("L", (160, 180), 0)
        ImageDraw.Draw(im).text((20, 10), ch, font=font, fill=255)
        a = np.asarray(im, dtype=np.float32) / 255.0
        ys, xs = np.nonzero(a > 0.5)
        out[ch] = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return out


def _match(glyph: np.ndarray) -> list[tuple[str, float]]:
    """Scores of a binary glyph (cut to its ink) against every digit: normalised correlation at the glyph's size."""
    import cv2

    h, w = glyph.shape
    g = glyph.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    out = []
    for ch, t in templates().items():
        # the digit at the glyph's height, its width as the font has it (a narrow '1' stays narrow), centred
        tw = max(1, int(round(t.shape[1] * h / t.shape[0])))
        r = cv2.resize(t, (tw, h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((h, max(w, tw)), np.float32)
        x0 = (canvas.shape[1] - tw) // 2
        canvas[:, x0:x0 + tw] = r
        gg = np.zeros_like(canvas)
        gx = (canvas.shape[1] - w) // 2
        gg[:, gx:gx + w] = g
        c = (canvas - canvas.mean()) / (canvas.std() + 1e-6)
        out.append((ch, float((c * gg).mean())))
    return sorted(out, key=lambda x: -x[1])


def _cuts(mask: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Column ranges splitting a mark into n glyphs, each cut at the thinnest ink near where it is expected."""
    w = mask.shape[1]
    if n <= 1:
        return [(0, w)]
    col = mask.sum(0)
    cuts = [0]
    for j in range(1, n):
        c = j * w / n
        lo, hi = int(max(cuts[-1] + 1, c - 0.25 * w / n)), int(min(w - 1, c + 0.25 * w / n))
        cuts.append(lo + int(np.argmin(col[lo:hi + 1])) if hi >= lo else int(c))
    cuts.append(w)
    return [(a, b) for a, b in zip(cuts, cuts[1:], strict=False) if b > a]


# ------------------------------------------------------------------ finding the plate
TEXT_BOX = (17 / 288, 17 / 61, 271 / 288, 51 / 61)   # the digits' extent on the plate (the plate renderer's layout)
INSET = (3.5 / 288, 3.5 / 61)                         # the panel inside the border
EDGE_X = 0.15                                         # a private plate's series starts (its number ends) this near
                                                      # the plate's edge, of its width


def _root_of(par: list[int]):
    """Union-find over `par`: the function giving an element's group."""
    def root(a: int) -> int:
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    return root


def find_plates(crop: np.ndarray, n: int = 8) -> list[tuple]:
    """Candidate plate boxes (x1, y1, x2, y2) in an RGB crop of a vehicle's back, likeliest first."""
    import cv2

    h, w = crop.shape[:2]
    if h < 20 or w < 30:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    y0 = int(h * 0.25)                                   # (the plate is in the lower part of the back)
    g = gray[y0:]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.05)) | 1,) * 2)
    rows, bands, blobs = [], [], []
    for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):    # light strokes on dark, dark on light
        m = cv2.morphologyEx(g, op, k)
        # rows of character-sized marks
        b = (m > max(30.0, float(np.percentile(m, 90)))).astype(np.uint8)
        nl, _, st, cen = cv2.connectedComponentsWithStats(b)
        cs = [(st[i], cen[i]) for i in range(1, nl)
              if 5 <= st[i][3] <= 0.12 * h + 4 and 0.12 < st[i][2] / st[i][3] < 3.2 and st[i][4] >= 8]
        par = list(range(len(cs)))
        root = _root_of(par)

        for i, (si, ci) in enumerate(cs):
            for j in range(i + 1, len(cs)):
                sj, cj = cs[j]
                if max(si[3], sj[3]) > 1.9 * min(si[3], sj[3]):
                    continue
                hh = (si[3] + sj[3]) / 2
                dx = abs(ci[0] - cj[0]) - (si[2] + sj[2]) / 2
                if dx > 1.6 * hh or abs(ci[1] - cj[1]) > 0.45 * hh + 0.35 * abs(ci[0] - cj[0]):
                    continue
                par[root(i)] = root(j)
        groups: dict[int, list[int]] = {}
        for i in range(len(cs)):
            groups.setdefault(root(i), []).append(i)
        for ids in groups.values():
            if len(ids) < 2:
                continue
            x1 = min(cs[i][0][0] for i in ids)
            x2 = max(cs[i][0][0] + cs[i][0][2] for i in ids)
            y1 = min(cs[i][0][1] for i in ids)
            y2 = max(cs[i][0][1] + cs[i][0][3] for i in ids)
            hh = float(np.median([cs[i][0][3] for i in ids]))
            if not 3.0 * hh <= x2 - x1 <= 12 * hh:
                continue
            ink = sum(cs[i][0][2] for i in ids) / (x2 - x1)
            rows.append((len(ids) * ink, (int(x1), int(y1 + y0), int(x2), int(y2 + y0))))
        for i in range(1, nl):                           # characters run into the plate's border: one blob
            x, y, bw, bh, a = st[i]
            if bh >= 5 and bw >= 0.12 * w and bw <= 0.7 * w and 3.0 <= bw / bh <= 9.0 and a / (bw * bh) >= 0.15:
                blobs.append((a / (bw * bh) * (1 - min(abs(bw / bh - 4.7) / 4.7, 0.8)),
                              (int(x), int(y + y0), int(x + bw), int(y + y0 + bh))))
        if op == cv2.MORPH_TOPHAT:
            # bands where bright strokes crowd together
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(w * 0.03)), 1)))
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            nl, _, st, _ = cv2.connectedComponentsWithStats(b)
            for i in range(1, nl):
                x, y, bw, bh, a = st[i]
                if bw < max(16, 0.08 * w) or bw > 0.7 * w or bh < 4 or not 2.0 < bw / bh < 10 or a / (bw * bh) < 0.2:
                    continue
                ar = bw / bh
                sc = a / (bw * bh) * (1 - min(abs(ar - 4.2) / 4.2, 0.8)) * (1 - abs((x + bw / 2) / w - 0.5))
                bands.append((sc, (int(x), int(y + y0), int(x + bw), int(y + y0 + bh))))
    out = []
    for lst in (rows, blobs, bands):
        lst.sort(key=lambda c: -c[0])
        out += [b for _, b in lst[:n // 2]]
    return out


def plate_quad(crop: np.ndarray, box: tuple, light_panel: bool = False) -> np.ndarray | None:
    """The plate's panel (inside its border) around a candidate box: its corners (top left, top right, bottom right,
    bottom left) in the crop, or None. The panel is dark on a private plate, light on a state one."""
    import cv2

    h, w = crop.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, max(1, y2 - y1)
    X1, X2 = max(0, int(x1 - 0.25 * bw)), min(w, int(x2 + 0.25 * bw))
    Y1, Y2 = max(0, int(y1 - bh)), min(h, int(y2 + bh))
    S = 4
    g = cv2.cvtColor(crop[Y1:Y2, X1:X2], cv2.COLOR_RGB2GRAY)
    if light_panel:
        g = 255 - g
    g = cv2.resize(g, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    cx, cy = ((x1 + x2) / 2 - X1) * S, ((y1 + y2) / 2 - Y1) * S
    inner = g[max(0, int(cy - bh * S / 2)):int(cy + bh * S / 2), max(0, int(cx - bw * S / 2)):int(cx + bw * S / 2)]
    if inner.size == 0:
        return None
    best = None
    for q in (30, 45, 60):
        m = (g <= np.percentile(inner, q)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gh, gw = g.shape
        for c in cs:
            bx, by, bw_, bh_ = cv2.boundingRect(c)
            if (bx <= 1 and X1 > 0) or (by <= 1 and Y1 > 0) or (bx + bw_ >= gw - 1 and X2 < w) or \
                    (by + bh_ >= gh - 1 and Y2 < h):
                continue                                 # (runs on into the dark around it: not the panel)
            (rx, ry), (rw, rh), _ = cv2.minAreaRect(c)
            if rw < rh:
                rw, rh = rh, rw
            if rh < 4 * S or not 3.5 < rw / rh < 6.5:     # (a whole plate: 4.7 wide for 1 high, give or take)
                continue
            if abs(rx - cx) > 0.5 * bw * S or abs(ry - cy) > 0.8 * bh * S:
                continue
            fill = cv2.contourArea(c) / (rw * rh)
            if fill < 0.6:
                continue
            sc = fill * (1 - abs(rw / rh - 4.55) / 4.55) * min(1.0, rw / (0.7 * bw * S))
            if best is None or sc > best[0]:
                best = (sc, c)
    if best is None:
        return None
    hull = cv2.convexHull(best[1])
    peri = cv2.arcLength(hull, True)
    pts = None
    for eps in (0.02, 0.04, 0.06):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            pts = ap[:, 0, :].astype(np.float32)
            break
    if pts is None:
        pts = cv2.boxPoints(cv2.minAreaRect(best[1])).astype(np.float32)
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    o = np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])
    return o / S + np.float32([X1, Y1])


def warp(crop: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """The plate straightened to PLATE_W x PLATE_H from its panel's corners."""
    import cv2

    ix, iy = INSET[0] * PLATE_W, INSET[1] * PLATE_H
    dst = np.float32([[ix, iy], [PLATE_W - ix, iy], [PLATE_W - ix, PLATE_H - iy], [ix, PLATE_H - iy]])
    m = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(crop, m, (PLATE_W, PLATE_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rectify(crop: np.ndarray, box: tuple) -> np.ndarray:
    """A box cut out and scaled to the standard size (no panel found)."""
    import cv2

    x1, y1, x2, y2 = box
    return cv2.resize(crop[y1:y2, x1:x2], (PLATE_W, PLATE_H), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ reading it
@lru_cache(maxsize=64)
def _templates_at(h: int) -> dict:
    """The font's digits at height h, with the font's gap either side (a '1' laid over part of an '8' sees ink in
    it), slightly blurred as a camera sees them: {digit: (template, pad)}."""
    import cv2

    out = {}
    for ch, t in templates().items():
        tw = max(2, int(round(t.shape[1] * h / t.shape[0])))
        r = cv2.resize(t, (tw, h), interpolation=cv2.INTER_AREA)
        pd = max(1, int(round(0.08 * h)))
        out[ch] = (cv2.GaussianBlur(np.pad(r, ((0, 0), (pd, pd))), (3, 3), 0.8).astype(np.float32), pd)
    return out


def _read_group(G: np.ndarray, xa: int, xb: int, ya: int, yb: int, maxn: int) -> list[tuple[float, str]]:
    """The digit sequences (mean match, digits) that cover columns xa..xb of rows ya..yb, best first."""
    import cv2

    hd = yb - ya
    if hd < 6:
        return []
    strip = G[max(0, ya - 2):yb + 3, :]
    scores = {}
    for ch, (t, pd) in _templates_at(hd).items():
        if t.shape[1] >= strip.shape[1] or t.shape[0] > strip.shape[0]:
            continue
        m = cv2.matchTemplate(strip, t, cv2.TM_CCOEFF_NORMED).max(0)
        scores[ch] = (np.concatenate([np.full(pd, -1.0, np.float32), m]), t.shape[1] - 2 * pd)  # at the ink's x
    if not scores:
        return []
    tol, gmax = max(2, int(0.12 * hd)), int(0.35 * hd) + 2
    best: list[dict] = [dict() for _ in range(maxn + 1)]   # best[n][end x] = (sum, digits)
    for x0 in range(max(0, xa - tol), xa + tol + 1):
        for ch, (s, w) in scores.items():
            if x0 < len(s) and (x0 + w not in best[1] or s[x0] > best[1][x0 + w][0]):
                best[1][x0 + w] = (float(s[x0]), ch)
    for n in range(2, maxn + 1):
        for e0, (v0, seq) in best[n - 1].items():
            for x0 in range(max(0, e0 - 1), e0 + gmax + 1):
                for ch, (s, w) in scores.items():
                    if x0 >= len(s) or x0 + w > xb + tol:
                        continue
                    v = v0 + float(s[x0])
                    if x0 + w not in best[n] or v > best[n][x0 + w][0]:
                        best[n][x0 + w] = (v, seq + ch)
    out = [(v / n, seq) for n in range(1, maxn + 1) for e, (v, seq) in best[n].items() if abs(e - xb) <= tol]
    return sorted(out, reverse=True)


def read_plate(plate: np.ndarray) -> Reading:
    """A straightened plate (PLATE_W x PLATE_H, RGB) -> its text 'SERIES TU NUMBER' (or a state plate's
    'CC - NNNNN'), with its confidence."""
    import cv2

    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    mx, my = int(0.03 * W), int(0.1 * H)
    inner = gray[my:H - my, mx:W - mx]
    light = float(np.median(inner)) > 140 and float(np.percentile(inner, 20)) > 110   # a state plate: dark on white
    kind = "state" if light else "private"
    G = (255 - gray if light else gray).astype(np.float32)
    gi = cv2.GaussianBlur(G[my:H - my, mx:W - mx].astype(np.uint8), (3, 3), 0)
    t, _ = cv2.threshold(gi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # (inside the panel: not the border)
    bw = np.zeros(gray.shape, np.uint8)
    bw[my:H - my, mx:W - mx] = G[my:H - my, mx:W - mx] > t
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    marks = [st[i] for i in range(1, n) if 0.3 * H <= st[i][3] <= 0.9 * H and st[i][4] >= 10]
    if not marks:
        return Reading(None, 0.0, kind=kind)
    if not light:
        # a private plate: the series starts at the plate's left edge and the number ends at its right edge (the
        # plate's layout); each group runs inwards over marks close together; the word تونس is what lies between (at
        # small sizes it is nearly as tall as the digits: not told apart by height)
        ms = sorted(marks, key=lambda m: m[0])
        if ms[0][0] > EDGE_X * W or ms[-1][0] + ms[-1][2] < (1 - EDGE_X) * W:
            return Reading(None, 0.0, kind=kind)
        groups = ([ms[0]], [ms[-1]])
        for grp, seq in ((groups[0], ms[1:]), (groups[1], ms[-2::-1])):
            h0 = grp[0][3]
            for m in seq:
                prev = grp[-1]
                gap = m[0] - (prev[0] + prev[2]) if grp is groups[0] else prev[0] - (m[0] + m[2])
                if gap > 0.45 * h0 or not 0.8 * h0 <= m[3] <= 1.2 * h0 or abs(m[0] + m[2] / 2 - W / 2) < 0.1 * W:
                    break
                grp.append(m)
        groups[1].sort(key=lambda m: m[0])
        if set(map(tuple, groups[0])) & set(map(tuple, groups[1])):
            return Reading(None, 0.0, kind=kind)
        hd = max(m[3] for g_ in groups for m in g_)
        if min(m[3] for g_ in groups for m in g_) < 0.75 * hd:
            return Reading(None, 0.0, kind=kind)
    else:
        hd = max(m[3] for m in marks)
        digits = sorted((m for m in marks if m[3] >= 0.75 * hd), key=lambda m: m[0])   # (the dash is lower)
        if len(digits) < 2:
            return Reading(None, 0.0, kind=kind)
        gaps = [(digits[i + 1][0] - (digits[i][0] + digits[i][2]), i) for i in range(len(digits) - 1)]
        gmax, gi_ = max(gaps)
        if gmax < 0.6 * hd:
            return Reading(None, 0.0, kind=kind)
        groups = (digits[:gi_ + 1], digits[gi_ + 1:])
    texts, scores, margins = [], [], []
    for grp, maxn in zip(groups, (2, 5) if light else (3, 4), strict=True):
        xa, xb = int(min(m[0] for m in grp)), int(max(m[0] + m[2] for m in grp))
        prof = bw[:, xa:xb].sum(1).astype(float)
        on = list(prof >= 0.18 * prof.max()) + [False]
        runs, start = [], None
        for y, v in enumerate(on):                       # the group's digit rows: its longest run of ink rows
            if v and start is None:
                start = y
            elif not v and start is not None:
                runs.append((y - start, start, y))
                start = None
        _, ya, yb = max(runs)
        res: list = []
        for d0, d1 in ((0, 0), (1, 0), (0, -1), (1, -1), (-1, 0), (0, 1)):
            r = _read_group(G, xa, xb, ya + d0, yb + d1, maxn)
            if r and (not res or r[0][0] > res[0][0]):
                res = r
        if not res:
            return Reading(None, 0.0, kind=kind)
        texts.append(res[0][1])
        scores.append(res[0][0])
        other = [v for v, s in res[1:] if s != res[0][1]]
        margins.append(res[0][0] - (other[0] if other else 0.0))
    conf = max(0.0, min(1.0, (min(scores) - 0.5) / 0.35)) * max(0.0, min(1.0, min(margins) / 0.08))
    if kind == "state":
        text = f"{texts[0]} - {texts[1]}"
    else:
        text = f"{int(texts[0])} TU {int(texts[1])}"
    digs = [(c, round(s, 3), round(m, 3)) for t_, s, m in zip(texts, scores, margins, strict=True) for c in t_]
    return Reading(text, round(conf, 3), digs, kind=kind)


def read_vehicle(crop: np.ndarray) -> Reading:
    """A vehicle's back (RGB crop of its box) -> the best reading of its plate (None if none could be read)."""
    best = Reading(None, 0.0)
    seen = []
    H, W = crop.shape[:2]
    for box in find_plates(crop):
        # the box itself as the plate (characters and border run together into one blob)
        x1, y1, x2, y2 = box
        if (x2 - x1) >= 3 * (y2 - y1):
            cy, hh, px = (y1 + y2) / 2, (x2 - x1) / ASPECT / 2, 0.02 * (x2 - x1)
            bb = (max(0, int(x1 - px)), max(0, int(cy - hh)), min(W, int(x2 + px)), min(H, int(round(cy + hh))))
            if bb[3] - bb[1] >= 6:
                r = read_plate(rectify(crop, bb))
                if r.text and (best.text is None or r.conf > best.conf):
                    r.box = bb
                    best = r
        for light_panel in (False, True):
            q = plate_quad(crop, box, light_panel)
            if q is None:
                continue
            key = tuple(np.round(q.mean(0) / 4).astype(int))
            if key in seen:
                continue
            seen.append(key)
            r = read_plate(warp(crop, q))
            if r.text and (best.text is None or r.conf > best.conf):
                r.box = tuple(int(v) for v in (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()))
                best = r
    return best


def find_plate(crop: np.ndarray) -> tuple | None:
    """(kept for callers that want one box) the first candidate."""
    c = find_plates(crop)
    return c[0] if c else None
