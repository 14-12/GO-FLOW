"""Test pattern for CARLA's plate textures: which part of the texture a car's plate shows, and whether the
material cuts the plate by the texture's alpha. Each texture: its number, a grid every 1/8 with u/v labels,
coloured bands by v, and alpha 0 in the top-left quarter (if that quarter is cut away on the car, the plate
is a masked material)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import ue_texture as ut  # noqa: E402

BANDS = [(230, 60, 60), (240, 150, 40), (240, 230, 60), (80, 200, 80), (60, 200, 220), (70, 110, 240),
         (160, 80, 230), (230, 90, 180)]


def pattern(n: int, size: int) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    d = ImageDraw.Draw(im)
    k = size / 8
    for j in range(8):
        d.rectangle([0, j * k, size, (j + 1) * k], fill=BANDS[j] + (255,))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(8, int(size / 22)))
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size / 6))
    for i in range(9):
        x = i * k
        d.line([x, 0, x, size], fill=(0, 0, 0, 255), width=max(1, size // 128))
        d.line([0, x, size, x], fill=(0, 0, 0, 255), width=max(1, size // 128))
    for i in range(8):
        for j in range(8):
            d.text((i * k + 2, j * k + 2), f"{i}{j}", font=font, fill=(0, 0, 0, 255))
    d.text((size * 0.30, size * 0.40), f"#{n}", font=big, fill=(255, 255, 255, 255), stroke_width=3,
           stroke_fill=(0, 0, 0, 255))
    px = im.load()
    for y in range(size // 4):
        for x in range(size // 4):
            r, g, b, _ = px[x, y]
            px[x, y] = (r, g, b, 0)
    return im


def main(src: str, out: str) -> None:
    src_p, out_p = Path(src), Path(out)
    files = sorted(src_p.glob("Textures/*.uexp")) + [src_p / "T_LicensePlate_d.uexp"]
    n = 0
    for f in files:
        try:
            tex = ut.read(f.with_suffix(""))
        except ValueError:
            continue
        img = pattern(n, tex.w)
        rel = f.relative_to(src_p)
        dst = out_p / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tex.ubulk = f.with_suffix(".ubulk") if f.with_suffix(".ubulk").exists() else None
        data = ut.build_mips(tex, img)
        write_fresh(tex, data, dst, dst.with_suffix(".ubulk"))
        print(n, rel)
        n += 1


def write_fresh(tex, data, out_uexp, out_ubulk):
    """Like ue_texture.write_mips, but the .ubulk is built from scratch (its size is the external mips')."""
    ue = bytearray(tex.uexp.read_bytes())
    ext = [m for m in tex.mips if m.external]
    size = max((m.offset + m.size for m in ext), default=0)
    ub = bytearray(size)
    for m, d in zip(tex.mips, data, strict=True):
        if len(d) != m.size:
            raise ValueError(f"{out_uexp}: mip {m.w}x{m.h}: {len(d)} bytes, want {m.size}")
        if m.external:
            ub[m.offset:m.offset + m.size] = d
        else:
            ue[m.offset:m.offset + m.size] = d
    out_uexp.write_bytes(bytes(ue))
    if ext:
        out_ubulk.write_bytes(bytes(ub))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
