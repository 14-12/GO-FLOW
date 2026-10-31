"""A number-plate mesh for CARLA, made from one CARLA already ships: SM_TnPlate, a 52 x 11 cm plate.

CARLA's plates are a material slot of each car's skeletal mesh (SK_<model> -> MI_LicensePlate_<model>), and the
runtime texture API (World.apply_color_texture_to_object) only changes the static meshes of an actor: it can't
give a car a plate of its own (measured, tools/probe_plates_api.py: no change on any of 18 models). So each car
gets two small plate actors of its own (static.prop.mesh), attached front and back, whose material the API does
set: M_BasicMaterial's "BaseColor" texture.

CARLA's Static/Other/SM_Plane is a 20 x 1200 x 1000 cm box (24 vertices, one material: MI_Kiosk02Papers ->
M_BasicMaterial, "BaseColor" / "Normal" / "OcclussionRoughnessMetallic" textures). This writes a copy with:
* the package and object renamed SM_Plane -> SM_TnPlate (a new asset beside it; SM_Plane itself is untouched),
* its vertices scaled to a plate, X (the face normal) +-0.15 cm, Y +-26 cm, Z +-5.5 cm, and its render bounds to
  match (culling and level of detail work on them),
* its texture coordinates set so each big face shows the whole texture once, upright and unmirrored as seen from
  outside (SM_Plane tiles it 12 x 10 times, turned a quarter; measured with tools/probe_plates2.py); the four
  thin edges show the texture's top edge (the plate's metal rim),
* its collision profile BlockAll -> NoCollision: the cooked collision is still the 12 m box, and a car carrying
  it would have stopped CARLA's ground ray casts (pedestrians' feet) 6 m around it.

    python tools/plates/make_plate_mesh.py <Content/Carla/Static/Other> <out dir>
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ue_package as up  # noqa: E402

HALF = (0.15, 26.0, 5.5)  # cm: thickness, width, height (a Tunisian plate is 520 x 110 mm)
SRC_HALF = (10.0, 600.0, 500.0)


def find_positions(b: bytes) -> tuple[int, int]:
    """(offset of the first position, count): FPositionVertexBuffer = stride 12, n, then a bulk array (12, n)."""
    for i in range(0, len(b) - 16):
        s, n, es, c = struct.unpack_from("<IIii", b, i)
        if s == 12 and es == 12 and n == c and 0 < n < 1000:
            return i + 16, n
    raise ValueError("no position buffer")


def find_uvs(b: bytes, n: int) -> tuple[int, int, int]:
    """(offset of the texture coordinates, coordinates per vertex, bytes per coordinate): FStaticMeshVertexBuffer =
    NumTexCoords, NumVertices, bFullPrecisionUVs, bHighPrecisionTangents, tangents (bulk array), UVs (bulk array)."""
    for i in range(0, len(b) - 24):
        nt, nv, full, hp = struct.unpack_from("<IIII", b, i)
        if 1 <= nt <= 4 and nv == n and full in (0, 1) and hp in (0, 1):
            es, c = struct.unpack_from("<ii", b, i + 16)
            if c == n and es in (8, 16):
                o = i + 24 + es * c
                es2, c2 = struct.unpack_from("<ii", b, o)
                if c2 == n * nt:
                    return o + 8, nt, es2
    raise ValueError("no vertex buffer")


def plate_uv(x: float, y: float, z: float) -> tuple[float, float]:
    """The texture coordinate of a corner (source units): +X face seen from +X (its right is -Y), -X face from -X."""
    v = (SRC_HALF[2] - z) / (2 * SRC_HALF[2])
    if x > 0 and abs(x - SRC_HALF[0]) < 0.01:
        return (SRC_HALF[1] - y) / (2 * SRC_HALF[1]), v
    return (y + SRC_HALF[1]) / (2 * SRC_HALF[1]), v


def find_bounds(b: bytes) -> int:
    """Offset of the render bounds' extent (FBoxSphereBounds: origin, extent, radius) = the source half sizes."""
    for i in range(len(b) - 16, 0, -4):
        x, y, z = struct.unpack_from("<fff", b, i)
        if all(abs(abs(v) - w) < 0.01 for v, w in zip((x, y, z), SRC_HALF, strict=True)):
            r = struct.unpack_from("<f", b, i + 12)[0]
            if abs(r - (sum(w * w for w in SRC_HALF)) ** 0.5) < 0.5:
                return i
    raise ValueError("no bounds")


def make(src_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ua, ue = src_dir / "SM_Plane.uasset", src_dir / "SM_Plane.uexp"
    tmp1, tmp2, tmp3 = out_dir / "_a.uasset", out_dir / "_b.uasset", out_dir / "SM_TnPlate.uasset"
    d = up.rename(ua, "/Game/Carla/Static/Other/SM_Plane", "/Game/Carla/Static/Other/SM_TnPlate", tmp1)
    d += up.rename(tmp1, "SM_Plane", "SM_TnPlate", tmp2)
    d += up.rename(tmp2, "BlockAll", "NoCollision", tmp3)
    tmp1.unlink()
    tmp2.unlink()
    import numpy as np

    b = bytearray(ue.read_bytes())
    p0, n = find_positions(b)
    uo, nt, es = find_uvs(b, n)
    if es != 4:
        raise ValueError(f"UVs of {es} bytes: only half precision is handled")
    uv = np.frombuffer(bytes(b[uo:uo + n * nt * es]), dtype=np.float16).reshape(n, nt, 2).copy()
    for k in range(n):
        x, y, z = struct.unpack_from("<fff", b, p0 + 12 * k)
        big = abs(abs(x) - SRC_HALF[0]) < 0.01 and all(
            abs(abs(struct.unpack_from("<f", b, p0 + 12 * j)[0]) - SRC_HALF[0]) < 0.01 for j in range(k - k % 4, k - k % 4 + 4))
        uv[k, 0] = plate_uv(x, y, z) if big else (0.5, 0.004)
    b[uo:uo + n * nt * es] = uv.tobytes()
    for k in range(n):
        o = p0 + 12 * k
        v = struct.unpack_from("<fff", b, o)
        struct.pack_into("<fff", b, o, *(c / s * h for c, s, h in zip(v, SRC_HALF, HALF, strict=True)))
    bo = find_bounds(b)
    struct.pack_into("<ffff", b, bo, *HALF, sum(h * h for h in HALF) ** 0.5)
    (out_dir / "SM_TnPlate.uexp").write_bytes(bytes(b))
    pk = up.parse(tmp3)
    return {"vertices": n, "header_delta": d, "names_ok": up.check_hashes(pk) == len(pk.names)}


if __name__ == "__main__":
    print(make(Path(sys.argv[1]), Path(sys.argv[2])))
