"""The flatbed of the recovery truck (24 Oct): SM_TnDeck, a 6.4 x 2.36 m steel deck 14 cm thick, made like the plate
mesh (tools/plates/make_plate_mesh.py) from CARLA's Static/Other/SM_Plane, a 24-vertex box.

CARLA 0.9.16 has no tow truck. The recovery truck is the European HGV's tractor with this deck on its chassis
(cosim/towing.py), posed by the pose process with the truck and tilted back to the road to load a wreck, as a
rollback ("plateau") tow truck does. Its look is a texture given at run time (cosim/towing.deck_texture): steel
planks along the deck, yellow and black chevrons on the rear edge, a reflective red and white band on the sides.

Faces (SM_Plane's vertices come in fours, one face each): the top and bottom take the texture's deck area, u along
the deck (rear u=0, front u=1), v across it; the long sides the texture's side band (v near 0); the rear end the
first column (the chevrons), the front end the last.

    python tools/plates/make_deck_mesh.py <Content/Carla/Static/Other> <out dir>
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ue_package as up  # noqa: E402
from make_plate_mesh import SRC_HALF, find_bounds, find_positions, find_uvs  # noqa: E402

HALF = (7.0, 118.0, 320.0)  # cm: half thickness (X), half width (Y), half length (Z)
SIDE_V = 0.03               # the side band's row
DECK_V0 = 0.08              # the deck area starts below the side band
NAME = "SM_TnDeck"


def deck_uv(face_axis: int, sign: float, x: float, y: float, z: float) -> tuple[float, float]:
    """Texture coordinate of a corner (source units) of the face whose normal is `face_axis` (0 x, 1 y, 2 z).

    The deck's top is the -X face: in CARLA only that face shows a mesh's texture coordinates with this material
    (measured, tools/probe_deck_uv.py: a test pattern on SM_TnDeck and on SM_TnPlate scaled up, the other five faces
    one colour), as for the plate. So the deck is built standing, X its thickness and Z its length, and turned to lie
    flat, -X up, when posed (cosim/towing.MESH_ROT)."""
    u = (z + SRC_HALF[2]) / (2 * SRC_HALF[2])
    w = (y + SRC_HALF[1]) / (2 * SRC_HALF[1])
    if face_axis == 0:          # top and bottom: the deck
        return u, DECK_V0 + (1.0 - DECK_V0) * w
    if face_axis == 1:          # the long sides: the side band
        return u, SIDE_V
    return (0.004 if sign < 0 else 0.996), DECK_V0 + (1.0 - DECK_V0) * w  # the ends: first / last column


def make(src_dir: Path, out_dir: Path) -> dict:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    ua, ue = src_dir / "SM_Plane.uasset", src_dir / "SM_Plane.uexp"
    tmp1, tmp2, tmp3 = out_dir / "_a.uasset", out_dir / "_b.uasset", out_dir / f"{NAME}.uasset"
    d = up.rename(ua, "/Game/Carla/Static/Other/SM_Plane", f"/Game/Carla/Static/Other/{NAME}", tmp1)
    d += up.rename(tmp1, "SM_Plane", NAME, tmp2)
    d += up.rename(tmp2, "BlockAll", "NoCollision", tmp3)
    tmp1.unlink()
    tmp2.unlink()
    b = bytearray(ue.read_bytes())
    p0, n = find_positions(b)
    uo, nt, es = find_uvs(b, n)
    if es != 4:
        raise ValueError(f"UVs of {es} bytes: only half precision is handled")
    uv = np.frombuffer(bytes(b[uo:uo + n * nt * es]), dtype=np.float16).reshape(n, nt, 2).copy()
    pos = [struct.unpack_from("<fff", b, p0 + 12 * k) for k in range(n)]
    faces = 0
    for g in range(0, n, 4):
        quad = pos[g:g + 4]
        axis = next(a for a in range(3) if all(abs(abs(p[a]) - SRC_HALF[a]) < 0.01 for p in quad)
                    and max(p[a] for p in quad) - min(p[a] for p in quad) < 0.01)
        sign = 1.0 if quad[0][axis] > 0 else -1.0
        for k in range(g, g + 4):
            uv[k, 0] = deck_uv(axis, sign, *pos[k])
        faces += 1
    b[uo:uo + n * nt * es] = uv.tobytes()
    for k in range(n):
        struct.pack_into("<fff", b, p0 + 12 * k, *(c / s * h for c, s, h in zip(pos[k], SRC_HALF, HALF, strict=True)))
    bo = find_bounds(b)
    struct.pack_into("<ffff", b, bo, *HALF, sum(h * h for h in HALF) ** 0.5)
    (out_dir / f"{NAME}.uexp").write_bytes(bytes(b))
    pk = up.parse(tmp3)
    return {"vertices": n, "faces": faces, "header_delta": d, "names_ok": up.check_hashes(pk) == len(pk.names)}


if __name__ == "__main__":
    print(make(Path(sys.argv[1]), Path(sys.argv[2])))
