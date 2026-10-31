"""Rename one name in a cooked UE4 (4.26) package's name map: .uasset (summary, names, imports, exports) next to
its .uexp (the exports' data). A name of another length shifts everything after the name map, so the
summary's offsets and every export's SerialOffset (counted as if .uasset and .uexp were one file) move by
the difference. The .uexp refers to names by their index, which stays.

Used to give CARLA's licence-plate material a texture parameter the runtime texture API sets
("LIcenseText" -> "BaseColor", tools/plates/install_plates.py), so each vehicle can get a plate of its own.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

TAG = 0x9E2A83C1


def _crc_table_deprecated() -> list[int]:
    # FCrc::CRCTable_DEPRECATED: CRC-32 (polynomial 0x04C11DB7), MSB first
    out = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if c & 0x80000000 else (c << 1)
        out.append(c & 0xFFFFFFFF)
    return out


_T_DEP = _crc_table_deprecated()


def strihash_deprecated(s: str) -> int:
    """FCrc::Strihash_DEPRECATED (ANSI names): the non-case-preserving hash (low 16 bits are stored)."""
    h = 0
    for ch in s.upper():
        h = ((h >> 8) & 0x00FFFFFF) ^ _T_DEP[(h ^ ord(ch)) & 0xFF]
    return h & 0xFFFFFFFF


def strcrc32(s: str) -> int:
    """FCrc::StrCrc32: CRC-32 over each character as 4 little-endian bytes."""
    return zlib.crc32(s.encode("utf-32-le")) & 0xFFFFFFFF


@dataclass
class Package:
    data: bytearray
    offsets: dict = field(default_factory=dict)  # summary field -> (position, value, size)
    names: list = field(default_factory=list)    # (position, text, hash1, hash2, length in bytes)
    exports: list = field(default_factory=list)  # position of each export's SerialOffset (int64)


def parse(uasset: Path) -> Package:
    b = bytearray(uasset.read_bytes())
    pk = Package(b)
    o = 0

    def i32():
        nonlocal o
        v = struct.unpack_from("<i", b, o)[0]
        o += 4
        return v

    def i64():
        nonlocal o
        v = struct.unpack_from("<q", b, o)[0]
        o += 8
        return v

    def fstr():
        nonlocal o
        n = i32()
        if n < 0:
            s = b[o:o - 2 * n].decode("utf-16-le")[:-1]
            o += -2 * n
        else:
            s = b[o:o + n].decode("latin-1")[:-1] if n else ""
            o += n
        return s

    def mark(name, size=4):
        pos = o
        v = i64() if size == 8 else i32()
        pk.offsets[name] = (pos, v, size)
        return v

    if struct.unpack_from("<I", b, 0)[0] != TAG:
        raise ValueError(f"{uasset}: not a UE package")
    o = 4
    legacy = i32()
    if legacy != -7:
        raise ValueError(f"{uasset}: legacy version {legacy}, expected -7 (UE 4.26)")
    i32()  # LegacyUE3Version
    ue4 = i32()
    i32()  # licensee
    n = i32()
    o += n * 20  # custom versions
    mark("TotalHeaderSize")
    fstr()
    i32()  # PackageFlags
    name_count = i32()
    mark("NameOffset")
    i32()
    mark("GatherableTextDataOffset")
    export_count = i32()
    mark("ExportOffset")
    i32()
    mark("ImportOffset")
    mark("DependsOffset")
    i32()
    mark("SoftPackageReferencesOffset")
    mark("SearchableNamesOffset")
    mark("ThumbnailTableOffset")
    o += 16  # Guid
    n = i32()
    o += n * 8  # generations
    for _ in range(2):  # saved by / compatible with engine version
        o += 2 * 3 + 4
        fstr()
    i32()  # compression flags
    if i32() != 0:
        raise ValueError("compressed chunks")
    i32()  # package source
    if i32() != 0:
        raise ValueError("additional packages to cook")
    mark("AssetRegistryDataOffset")
    mark("BulkDataStartOffset", 8)
    mark("WorldTileInfoDataOffset")
    n = i32()
    o += n * 4  # chunk ids
    i32()  # preload dependency count
    mark("PreloadDependencyOffset")
    # names
    o = pk.offsets["NameOffset"][1]
    for _ in range(name_count):
        pos = o
        s = fstr()
        h1, h2 = struct.unpack_from("<HH", b, o)
        o += 4
        pk.names.append((pos, s, h1, h2, o - pos))
    # exports
    o = pk.offsets["ExportOffset"][1]
    size = 4 * 4 + 8 + 4 + 8 + 8 + 4 * 3 + 16 + 4 + 4 * 2 + 4 * 5
    if 0 < ue4 < 507:  # 0: an unversioned (cooked) package, the engine's own version (4.26)
        raise ValueError("old export format")
    for _ in range(export_count):
        pk.exports.append(o + 4 * 4 + 8 + 4 + 8)  # after Class/Super/Template/Outer, ObjectName, Flags, SerialSize
        o += size
    return pk


def check_hashes(pk: Package) -> int:
    """How many names' stored hashes this module reproduces (all of them, or the hashing is wrong)."""
    ok = 0
    for _, s, h1, h2, _ in pk.names:
        if strihash_deprecated(s) & 0xFFFF == h1 and strcrc32(s) & 0xFFFF == h2:
            ok += 1
    return ok


def rename(uasset: Path, old: str, new: str, out: Path) -> int:
    """Writes the package with name `old` renamed `new` to `out` (.uasset; copy the .uexp alongside yourself).
    Returns the size change in bytes."""
    pk = parse(uasset)
    hit = [n for n in pk.names if n[1] == old]
    if not hit:
        raise ValueError(f"{uasset}: no name {old!r}")
    if any(n[1] == new for n in pk.names):
        raise ValueError(f"{uasset}: already has {new!r}")
    pos, _, _, _, length = hit[0]
    enc = new.encode("latin-1") + b"\0"
    entry = struct.pack("<i", len(enc)) + enc + struct.pack("<HH", strihash_deprecated(new) & 0xFFFF,
                                                           strcrc32(new) & 0xFFFF)
    delta = len(entry) - length
    b = pk.data
    nb = bytearray(b[:pos]) + entry + b[pos + length:]
    for name, (p, v, size) in pk.offsets.items():
        if name == "NameOffset" or v <= 0 or v <= pos:
            continue
        struct.pack_into("<q" if size == 8 else "<i", nb, p, v + delta)
    for p in pk.exports:  # positions after the name map moved too
        q = p + delta if p > pos else p
        v = struct.unpack_from("<q", nb, q)[0]
        struct.pack_into("<q", nb, q, v + delta)
    out.write_bytes(bytes(nb))
    return delta
