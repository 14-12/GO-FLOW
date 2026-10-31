"""The pedestrians' ground heights, asked of CARLA by a process of its own (cosim/carla_side.CarlaSide._surface_z).

CARLA's `world.ground_projection` is a call to the server that waits for its next picture (~25 ms at 40 FPS),
and its Python binding keeps Python's lock all that time. Made by a thread of the free-fly viewer, it held
every other thread of the viewer up: profiled at rush hour (--profile), the viewer's threads together used
only 2 s of CPU in 10 s and still ran late everywhere (SUMO's steps ~270 ms instead of ~10, the overlay 7-14
redraws a second instead of 30), the sampler itself getting its turn every 20-35 ms instead of 5. Here
the waiting holds up no one.

    requests  (key, x, y, z_lane)   CARLA coordinates; key: the caller's cache key
    answers   (key, z)              the ground's height, or z_lane when what is under it is not ground
    ("stop",) ends it.
"""

from __future__ import annotations

SURFACES = {"Roads", "RoadLines", "Sidewalks", "Ground", "Terrain"}


def ground_z(world, carla, x: float, y: float, z_lane: float) -> float:
    """CARLA's ground projection from 2 m above the lane, if it hit ground within 1 m of the lane; else the lane."""
    try:
        hit = world.ground_projection(carla.Location(x, y, z_lane + 2.0), 5.0)
        if hit is not None and getattr(hit.label, "name", str(hit.label)) in SURFACES \
                and abs(hit.location.z - z_lane) < 1.0:
            return hit.location.z
    except Exception:  # noqa: BLE001 - older CARLA, or the server busy: the lane's height
        pass
    return z_lane


def run(host: str, port: int, q_in, q_out) -> None:
    import carla

    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.get_world()
    while True:
        msg = q_in.get()
        if msg[0] == "stop":
            return
        key, x, y, z_lane = msg
        q_out.put((key, ground_z(world, carla, x, y, z_lane)))
