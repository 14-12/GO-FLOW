"""Which CARLA vehicle model draws which SUMO vehicle.

Chosen from the blueprint library's own attributes (`base_type`, `special_type`,
`has_lights`), not from a hard-coded list of model names, so it keeps working across CARLA
releases. Models with working lights are preferred: the bridge drives brake lamps,
indicators and headlamps from SUMO's signal bits. (Measured on 0.9.16: 11 of the 20 plain
and electric cars have lights; preferring only generation 2, as a first version did, left 5
car models for every car in town.) Electric models are used only as cars (no Cybertruck for
SUMO's trucks).

The choice depends only on the SUMO vehicle id (a stable hash), so a vehicle keeps its model
and colour however many times the run is replayed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# SUMO vClass -> what to draw it as.
_CLASS_WANT = {
    "passenger": ("car", None),
    "taxi": ("car", "taxi"),
    "bus": ("bus", None),
    "coach": ("bus", None),
    "truck": ("truck", None),
    "trailer": ("truck", None),
    "delivery": ("van", None),
    "emergency": ("van", "emergency"),  # ambulance
    "authority": ("car", "emergency"),  # police
    "motorcycle": ("motorcycle", None),
    "moped": ("motorcycle", None),
    "bicycle": ("bicycle", None),
}


@dataclass(frozen=True)
class Candidate:
    id: str
    base_type: str
    special_type: str
    generation: int
    colours: tuple[str, ...]
    has_lights: bool = True


def attr_text(attribute) -> str:
    """An ActorAttribute's value as text. Attributes are typed and `as_str()` raises on the
    others (measured on CARLA 0.9.16: "generation: bad attribute cast: cannot convert to
    String"), so read each type with its own accessor."""
    import carla

    kind = attribute.type
    if kind == carla.ActorAttributeType.Int:
        return str(attribute.as_int())
    if kind == carla.ActorAttributeType.Bool:
        return str(attribute.as_bool()).lower()
    if kind == carla.ActorAttributeType.Float:
        return str(attribute.as_float())
    return attribute.as_str()


def candidates_from_library(library) -> list[Candidate]:
    """Read the attributes that matter from a carla.BlueprintLibrary."""
    out = []
    for bp in library.filter("vehicle.*"):
        def attr(name, default="", bp=bp):
            return attr_text(bp.get_attribute(name)) if bp.has_attribute(name) else default
        colours = tuple(bp.get_attribute("color").recommended_values) if bp.has_attribute("color") else ()
        gen = attr("generation", "1")
        out.append(Candidate(bp.id, attr("base_type").lower(), attr("special_type").lower(),
                             int(gen) if gen.isdigit() else 1, colours, attr("has_lights", "false") == "true"))
    return out


def _stable(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def choose(cands: list[Candidate], vclass: str, vehicle_id: str) -> tuple[str, str | None]:
    """(blueprint id, colour or None) for a SUMO vehicle."""
    base, special = _CLASS_WANT.get(vclass, ("car", None))
    plain = ("", "electric") if base == "car" else ("",)
    pool = [c for c in cands if c.base_type == base
            and (c.special_type == special if special else c.special_type in plain)]
    if not pool and special:  # e.g. no "taxi" model: any of the base type
        pool = [c for c in cands if c.base_type == base and c.special_type != "emergency"]
    if not pool:
        pool = [c for c in cands if c.base_type == "car" and c.special_type != "emergency"]
    lit = [c for c in pool if c.has_lights]
    pool = sorted(lit or pool, key=lambda c: c.id)
    h = _stable(vehicle_id)
    pick = pool[h % len(pool)]
    colour = pick.colours[(h >> 20) % len(pick.colours)] if pick.colours else None
    return pick.id, colour
