"""Read and rewrite the pixels of a cooked UE4 (4.26) Texture2D: CARLA 0.9.16 ships its content as loose
.uasset/.uexp/.ubulk files. The mip table (FTexturePlatformData) is in the .uexp, after the pixel format string;
large mips live in the .ubulk at the offsets the table gives, small ones inline in the .uexp."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

PAYLOAD_IN_SEPARATE_FILE = 0x10000  # BULKDATA_PayloadInSeperateFile


@dataclass
class Mip:
    flags: int
    count: int
    size: int
    offset: int       # in the .ubulk when external, else absolute offset in the .uexp of the data
    w: int
    h: int
    external: bool
    entry: int        # where this mip's bulk header starts in the .uexp


@dataclass
class Texture:
    uexp: Path
    ubulk: Path | None
    fmt: str
    w: int
    h: int
    mips: list[Mip]


def read(base: Path) -> Texture:
    """base: the asset path without extension."""
    uexp = base.with_suffix(".uexp")
    b = uexp.read_bytes()
    i = b.find(b"PF_")
    if i < 0:
        raise ValueError(f"{uexp}: no pixel format")
    n = struct.unpack_from("<i", b, i - 4)[0]
    fmt = b[i:i + n - 1].decode()
    w, h = struct.unpack_from("<ii", b, i - 16)
    j = i + n
    first, nmips = struct.unpack_from("<ii", b, j)
    j += 8
    mips = []
    for _ in range(nmips):
        entry = j
        j += 4  # bCooked
        flags, count, size, off = struct.unpack_from("<IiiQ", b, j)
        j += 20
        ext = bool(flags & PAYLOAD_IN_SEPARATE_FILE)
        if not ext:
            off = j  # the data follows the header
            j += size
        mw, mh, _mz = struct.unpack_from("<iii", b, j)
        j += 12
        mips.append(Mip(flags, count, size, off, mw, mh, ext, entry))
    ubulk = base.with_suffix(".ubulk")
    return Texture(uexp, ubulk if ubulk.exists() else None, fmt, w, h, mips)


def mip_bytes(tex: Texture, k: int) -> bytes:
    m = tex.mips[k]
    if m.external:
        return tex.ubulk.read_bytes()[m.offset:m.offset + m.size]
    return tex.uexp.read_bytes()[m.offset:m.offset + m.size]


def write_mips(tex: Texture, data: list[bytes], out_uexp: Path, out_ubulk: Path | None) -> None:
    """New pixel data for every mip, each exactly the size it had (same format and size)."""
    ue = bytearray(tex.uexp.read_bytes())
    ub = bytearray(tex.ubulk.read_bytes()) if tex.ubulk else None
    for m, d in zip(tex.mips, data, strict=True):
        if len(d) != m.size:
            raise ValueError(f"mip {m.w}x{m.h}: {len(d)} bytes, want {m.size}")
        if m.external:
            ub[m.offset:m.offset + m.size] = d
        else:
            ue[m.offset:m.offset + m.size] = d
    out_uexp.write_bytes(bytes(ue))
    if ub is not None and out_ubulk is not None:
        out_ubulk.write_bytes(bytes(ub))


def dds(fmt: str, w: int, h: int, data: bytes) -> bytes:
    """A minimal DDS file around DXT1 / DXT5 data (for Pillow to decode)."""
    four = {"PF_DXT1": b"DXT1", "PF_DXT5": b"DXT5"}[fmt]
    hdr = struct.pack("<4sIIIIIII44sII4sIIIIIIIIII", b"DDS ", 124, 0x1007 | 0x80000, h, w, len(data), 0, 1,
                      b"\0" * 44, 32, 0x4, four, 0, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
    return hdr + data


def to_image(tex: Texture, k: int = 0):
    import io

    from PIL import Image

    m = tex.mips[k]
    return Image.open(io.BytesIO(dds(tex.fmt, m.w, m.h, mip_bytes(tex, k)))).convert("RGBA")


def encode(fmt: str, img) -> bytes:
    """RGBA Pillow image (sides multiple of 4) -> DXT1 / DXT5 blocks."""
    import etcpak

    raw = img.convert("RGBA").tobytes()
    w, h = img.size
    return etcpak.compress_bc1(raw, w, h) if fmt == "PF_DXT1" else etcpak.compress_bc3(raw, w, h)


def build_mips(tex: Texture, img) -> list[bytes]:
    """All mips of `tex` from one full-size RGBA image (resized down, Lanczos)."""
    from PIL import Image

    out = []
    for m in tex.mips:
        w, h = max(m.w, 4), max(m.h, 4)
        out.append(encode(tex.fmt, img.resize((w, h), Image.LANCZOS)))
    return out
