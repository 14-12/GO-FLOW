"""Drivers at the wheel of the cars near the free-fly camera (25 Oct).

CARLA's vehicles drive empty. A person is seated at the wheel of the vehicles close to the camera: a walker from
a small pool, its legs and arms posed sitting (bones, as the officer's gestures in cosim/pose_proc.SceneOps), put at
the driver's seat of the model and moved with the vehicle every picture.

The pose (tools/probe_drivers.py, run 1): a bone's override is a change of its rest rotation (pitch, yaw, roll);
thigh roll -90 swings a leg forward, leg (knee) roll + bends the knee, the same signs on both sides; the upper arm's
pitch brings it forward (-80 right, +80 left, from the officer's probe).

The seat: CARLA's vehicles stand on their origin (the box's bottom at z 0) and face +x; the driver sits on the left
(-y). For cars the hip is about at the box's centre lengthwise (the ruler pictures of run 1), 0.4 of the half width
left of the middle, at 0.83 of the half height (run 2: the Charger 2020, door open, hip moved +-0.2 m: on the
cushion, back to the backrest) but at least HEAD_ROOM under the roof (run 3: heads through the roof of the low
cars); vans, trucks and the bus have their own (run 3's pictures). Legs: thighs a little up, knees bent 45 degrees,
so the feet rest about 0.26 m under the hip, on the floor (with the knees at 55, feet showed under the cars).
"""

from __future__ import annotations

# bone -> (pitch, yaw, roll) added to its rest rotation
POSE: dict[str, tuple[float, float, float]] = {
    "crl_thigh__L": (0.0, 0.0, -95.0),
    "crl_thigh__R": (0.0, 0.0, -95.0),
    "crl_leg__L": (0.0, 0.0, 45.0),
    "crl_leg__R": (0.0, 0.0, 45.0),
    "crl_arm__R": (-68.0, 0.0, 0.0),
    "crl_arm__L": (68.0, 0.0, 0.0),
}
HIP_ABOVE_ORIGIN = 0.04  # m: a walker's hip joint above its actor's origin (bones.json: thighs at z 0.037)

# model -> hip (x from the front of the box, y as a share of the half width, z above the vehicle's origin)
TALL: dict[str, tuple[float, float, float]] = {
    "carlamotors.carlacola": (1.15, 0.35, 1.35),
    "carlamotors.european_hgv": (0.95, 0.5, 2.05),
    "carlamotors.firetruck": (1.0, 0.5, 1.85),
    "mercedes.sprinter": (1.9, 0.42, 1.2),
    "ford.ambulance": (1.8, 0.45, 1.05),
    "volkswagen.t2": (0.9, 0.45, 0.85),
    "volkswagen.t2_2021": (1.25, 0.45, 0.85),
    "mitsubishi.fusorosa": (1.8, 0.3, 2.0),  # its box includes the mirrors: 3.9 m wide
}
# cars whose roof is lower than their box says (a taxi sign on top): hip height
HIP_Z: dict[str, float] = {"ford.crown": 0.55}
SIDE: dict[str, float] = {"micro.microlino": 0.3}  # driver nearer the middle (share of the half width)
HEAD_ROOM = 0.92  # m: hip to the roof's top for a seated driver (head 0.8 above the hip, the roof's thickness)


TWO_WHEELERS = ("crossbike", "century", "omafiets", "low_rider", "ninja", "zx125", "yzf")  # CARLA seats their riders
POOL = 16  # walkers kept for drivers (pose_proc.DRIVER_M: the cars nearest the camera)


def has_seat(type_id: str, base_type: str = "") -> bool:
    """A vehicle that gets a driver: not a bicycle or a motorbike (CARLA's come with their rider)."""
    return base_type.lower() not in ("bicycle", "motorcycle") and not any(w in type_id for w in TWO_WHEELERS)


def seat_rel(type_id: str, info: dict) -> tuple | None:
    """The driver's place as a pose_proc.compose() rel (x, y, z, pitch, yaw, roll), from config/vehicle_models.json's
    entry for the model (its box's extent and centre); None for a two-wheeler."""
    if not has_seat(type_id, info.get("base_type", "")):
        return None
    x, y, z = seat_of(type_id, tuple(info["extent"]), (info["centre"][0], info["centre"][2]))
    return (round(x, 3), round(y, 3), round(z, 3), 0.0, 0.0, 0.0)


def seat_of(type_id: str, extent: tuple[float, float, float], center: tuple[float, float]) -> tuple[float, float, float]:
    """Where to put the driver walker's origin, in the vehicle's frame: (x, y, z)."""
    ex, ey, ez = extent
    cx, cz = center
    model = type_id.removeprefix("vehicle.")
    if model in TALL:
        back, side, hip = TALL[model]
        return cx + ex - back, -side * ey, hip - HIP_ABOVE_ORIGIN
    hip = HIP_Z.get(model, min(0.83 * ez, cz + ez - HEAD_ROOM))
    return cx - 0.05, -SIDE.get(model, 0.4) * ey, hip - HIP_ABOVE_ORIGIN
