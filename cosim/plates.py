"""Tunisian number plates for the free-fly viewer's cars: the design, the numbers, the textures.

The design follows the Tunisian private-car plate: white characters on black, a thin white border, the series
(up to 3 digits) on the left, تونس in the middle, the vehicle's number in the series (up to 4 digits) on the
right; rear 520 x 110 mm, front 450 x 100 mm. Rental cars have the same layout white on blue. Series run up to
about 260 (2026), the newest the most common on the road.

The Arabic word is drawn once from Noto Naskh Arabic Bold with Pillow's complex-text layout (raqm) and kept as
assets/plates/tounes.png, so the viewer needs no Arabic shaping at run time (Pillow's Windows wheels may lack
FriBiDi). The digits use Barlow Condensed SemiBold (assets/fonts, OFL).

How a car gets its plates: cosim/plate_proc.py (two small plate actors per car, each given its texture through
CARLA's runtime texture API; see tools/plates/make_plate_mesh.py for why).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
REAR_MM, FRONT_MM = (520.0, 110.0), (450.0, 100.0)
TEX_W, TEX_H = 288, 61  # the texture's pixels (520:110): sharp up close, ~17 600 pixels set one by one
NEWEST_SERIES = 260
RENTAL_SHARE = 0.0  # rental cars' white-on-blue plates: none (24 Oct, asked); render() still draws one
# State vehicles (police cars, ambulances) carry the state plate, not the private one: red characters on white, the
# owning ministry's code on the left, the vehicle's number on the right ("15 - 12345"). The ministry codes of the
# Interior and of Health are not in the sources used (Wikipedia gives 03 Justice, 15 Transport): set them here.
STATE_CODES = {"vehicle.dodge.charger_police": 7, "vehicle.dodge.charger_police_2020": 7, "vehicle.ford.ambulance": 12,
               "vehicle.carlamotors.firetruck": 7}


@dataclass(frozen=True)
class Plate:
    series: int
    number: int
    kind: str = "private"  # private | rental | state (series = the ministry's code)

    @property
    def text(self) -> str:
        """As written in Latin script, the way people say it: '247 TU 3816' (series, TU, number); state: '15 - 12345'."""
        if self.kind == "state":
            return f"{self.series:02d} - {self.number:05d}"
        return f"{self.series} TU {self.number}"

    @property
    def arabic(self) -> str:
        if self.kind == "state":
            return self.text
        return f"{self.series} تونس {self.number}"


def random_plate(rng: random.Random, used: set | None = None) -> Plate:
    """A plausible plate: series weighted towards the newest (cars on the road are mostly recent), number 1-9999;
    a RENTAL_SHARE of them rentals (blue; none now). `used`: (series, number) already given, never repeated."""
    for _ in range(1000):
        age = rng.expovariate(1 / 38.0)  # series: about 38 a year back per e-fold of the fleet
        series = max(1, NEWEST_SERIES - int(age))
        number = rng.randint(1, 9999)
        if used is None or (series, number) not in used:
            break
    if used is not None:
        used.add((series, number))
    return Plate(series, number, "rental" if rng.random() < RENTAL_SHARE else "private")


def state_plate(rng: random.Random, code: int, used: set | None = None) -> Plate:
    """A state vehicle's plate: the ministry's code and a number 1-99999, never repeated."""
    for _ in range(1000):
        number = rng.randint(1, 99999)
        if used is None or (("state", code), number) not in used:
            break
    if used is not None:
        used.add((("state", code), number))
    return Plate(code, number, "state")


def plate_for(model: str, rng: random.Random, used: set | None = None) -> Plate:
    """The plate a car of this CARLA model gets: the state plate on police cars and ambulances, else a private one."""
    code = STATE_CODES.get(model)
    return state_plate(rng, code, used) if code is not None else random_plate(rng, used)


@lru_cache(maxsize=4)
def _fonts(h: int):
    from PIL import Image, ImageFont

    digits = ImageFont.truetype(str(ASSETS / "fonts" / "BarlowCondensed-SemiBold.ttf"), int(h * 0.80))
    tounes = Image.open(ASSETS / "plates" / "tounes.png").convert("L")
    return digits, tounes


def render(plate: Plate, w: int = TEX_W, h: int = TEX_H, scale: int = 3):
    """The plate as an RGB image, w x h (drawn at `scale` times and reduced, for clean edges)."""
    from PIL import Image, ImageDraw, ImageFilter

    W, H = w * scale, h * scale
    if plate.kind == "state":
        return _render_state(plate, w, h, scale)
    bg = (16, 17, 19) if plate.kind == "private" else (18, 52, 132)
    ink = (236, 236, 232)
    im = Image.new("RGB", (W, H), (60, 62, 66))  # the plate's metal edge
    d = ImageDraw.Draw(im)
    r = int(H * 0.10)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=bg)
    # a thin white border, inset as on the real plate (3 mm inset, 1.5 mm line at 110 mm high)
    ins, lw = H * 0.045, max(2, int(H * 0.022))
    d.rounded_rectangle([ins, ins, W - 1 - ins, H - 1 - ins], radius=int(r * 0.7), outline=ink, width=lw)
    digits, tounes = _fonts(H)
    cy = H * 0.52
    # the series on the left, the number on the right, تونس in the middle
    left, right = str(plate.series), str(plate.number)
    pad = W * 0.055
    for txt, anchor, x in ((left, "lm", pad), (right, "rm", W - pad)):
        d.text((x, cy), txt, font=digits, fill=ink, anchor=anchor)
    th = int(H * 0.50)
    tw = int(tounes.width * th / tounes.height)
    mask = tounes.resize((tw, th), Image.LANCZOS)
    lb = d.textbbox((pad, cy), left, font=digits, anchor="lm")[2]
    rb = d.textbbox((W - pad, cy), right, font=digits, anchor="rm")[0]
    mx = int((lb + rb) / 2 - tw / 2)
    im.paste(Image.new("RGB", mask.size, ink), (mx, int(cy - th * 0.55)), mask)
    # a little wear: a soft light from above and a darker lower edge, so it reads as metal, not a sticker
    shade = Image.linear_gradient("L").resize((W, H)).point(lambda v: int(18 * v / 255))
    im = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), im, shade)
    im = im.filter(ImageFilter.GaussianBlur(scale * 0.35))
    return im.resize((w, h), Image.LANCZOS)


def _render_state(plate: Plate, w: int, h: int, scale: int):
    """Red on white: the ministry's code on the left, a dash, the number on the right."""
    from PIL import Image, ImageDraw, ImageFilter

    W, H = w * scale, h * scale
    bg, ink = (238, 238, 234), (196, 28, 36)
    im = Image.new("RGB", (W, H), (150, 152, 156))
    d = ImageDraw.Draw(im)
    r = int(H * 0.10)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=bg)
    ins, lw = H * 0.045, max(2, int(H * 0.022))
    d.rounded_rectangle([ins, ins, W - 1 - ins, H - 1 - ins], radius=int(r * 0.7), outline=ink, width=lw)
    digits, _ = _fonts(H)
    cy = H * 0.52
    left, right = f"{plate.series:02d}", f"{plate.number:05d}"
    lw_, rw_ = d.textlength(left, font=digits), d.textlength(right, font=digits)
    dw, gap = H * 0.20, H * 0.16
    x = (W - (lw_ + gap + dw + gap + rw_)) / 2  # the whole "15 - 75617" centred, as on the plate
    d.text((x, cy), left, font=digits, fill=ink, anchor="lm")
    mx = x + lw_ + gap
    d.rectangle([mx, cy - H * 0.04, mx + dw, cy + H * 0.04], fill=ink)
    d.text((mx + dw + gap, cy), right, font=digits, fill=ink, anchor="lm")
    shade = Image.linear_gradient("L").resize((W, H)).point(lambda v: int(18 * v / 255))
    im = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), im, shade)
    im = im.filter(ImageFilter.GaussianBlur(scale * 0.35))
    return im.resize((w, h), Image.LANCZOS)


def to_texture(img):
    """A Pillow RGB image -> carla.TextureColor (CARLA has only a per-pixel setter)."""
    import carla

    w, h = img.size
    tex = carla.TextureColor(w, h)
    px = img.convert("RGB").load()
    Color, setp = carla.Color, tex.set
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            setp(x, y, Color(r, g, b, 255))
    return tex


def sheet(path: Path, n: int = 12, seed: int = 3) -> None:
    """A preview of n plates (README, and to look at the design)."""
    from PIL import Image

    rng = random.Random(seed)
    used: set = set()
    plates = [random_plate(rng, used) for _ in range(n)]
    plates[-2] = state_plate(rng, 15)
    w, h = TEX_W, TEX_H
    out = Image.new("RGB", (2 * w + 60, (n // 2) * (h + 20) + 20), (190, 192, 196))
    for i, p in enumerate(plates):
        out.paste(render(p), (20 + (i % 2) * (w + 20), 20 + (i // 2) * (h + 20)))
    out.save(path)


if __name__ == "__main__":
    import sys

    sheet(Path(sys.argv[1] if len(sys.argv) > 1 else "plates_preview.png"))
