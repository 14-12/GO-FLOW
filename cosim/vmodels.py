"""Each SUMO vehicle is as long and as wide as the CARLA car that draws it (23 Oct).

The viewer drew SUMO's default-sized vehicles (a car 5.0 x 1.8 m) with CARLA models from 4.2 m (Audi TT) to
5.6 m (Nissan Patrol), a lorry of 7.5 m as an 7.9 m European HGV, a bus of 12 m as a 10.3 m Fusorosa: in a queue a
long model's bumper sat inside the next car, a short one left gaps. Now the model is chosen when the vehicle is
inserted (the same stable choice by its id as before, cosim/blueprints.choose), and the vehicle gets a vType
of that model: SUMO's base type copied, with the model's length and width (config/vehicle_models.json, measured
from every model's bounding box in CARLA 0.9.16, tools/probe_carla_assets.py). CARLA's pool of parked actors is
kept per model (cosim/freefly.pooled_carla_side), so the actor drawn is that model.

Widths: a CARLA bounding box includes the mirrors (Mini 2.10 m, Fusorosa 3.94 m); SUMO's width is the body's,
taken as 92 % of it and at most 2.55 m (the legal width), so a bus still fits its lane in the sublane model.

Police cars are always the Charger 2020 police model (it opens its doors: the officer gets out at an accident),
ambulances the Ford ambulance: the emergency types (scenarios/twin_extras.add.xml) are sized to them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_FILE = ROOT / "config" / "vehicle_models.json"
SEP = "@"  # SUMO type id = <base type>@<CARLA model>
FIXED = {"authority": "vehicle.dodge.charger_police_2020", "emergency": "vehicle.ford.ambulance"}
EMERGENCY_TYPES = {"twin_ambulance_run": "emergency", "twin_ambulance": "emergency",
                   "twin_police_run": "authority", "twin_police": "authority"}
# what every vehicle type gets in the free-fly viewer (sublane model, --lateral-resolution 0.8): no sideways
# motion while standing, and sideways speed at most 0.3 x the forward speed (a 17 deg slant at most)
# and 1 m of lateral gap to a neighbour (SUMO's 0.6 m left no room for a body slanted across its lane)
LANE_CHANGE_ATTRS = {"lcMaxSpeedLatStanding": "0", "lcMaxSpeedLatFactor": "0.3", "minGapLat": "1.0"}
if os.environ.get("VT_EXTRA"):  # an experiment's vType attributes (json)
    LANE_CHANGE_ATTRS.update(json.loads(os.environ["VT_EXTRA"]))


def load(path: Path = MODELS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sumo_dims(info: dict) -> tuple[float, float, float]:
    """(length, width, height) for SUMO from a model's CARLA bounding box extents."""
    ex, ey, ez = info["extent"]
    return round(2 * ex, 2), round(min(2 * ey * 0.92, 2.55), 2), round(2 * ez, 2)


def model_of(type_id: str) -> str | None:
    """The CARLA model in a type id made by Typer ("pas_passenger@vehicle.audi.tt", with any "~suffix" after)."""
    return type_id.split(SEP, 1)[1].split("~", 1)[0] if SEP in type_id else None


def with_attrs(vtype_xml: str, attrs: dict) -> str:
    """A <vType .../> line with attributes added (or replaced)."""
    import re

    for k, v in attrs.items():
        if re.search(rf'\b{k}="[^"]*"', vtype_xml):
            vtype_xml = re.sub(rf'\b{k}="[^"]*"', f'{k}="{v}"', vtype_xml)
        else:
            vtype_xml = vtype_xml.replace("<vType ", f'<vType {k}="{v}" ', 1)
    return vtype_xml


def extras_types_file(models: dict, out: Path, attrs: dict | None = None) -> Path:
    """A copy of scenarios/twin_extras.add.xml with the emergency types sized to their CARLA models and `attrs`
    added to every vehicle type."""
    import re

    from cosim.extras import EXTRAS_TYPES_FILE

    text = EXTRAS_TYPES_FILE.read_text(encoding="utf-8")

    def fix(m):
        line = m.group(0)
        tid = re.search(r'id="([^"]+)"', line).group(1)
        vc = EMERGENCY_TYPES.get(tid)
        if vc is None:
            return line
        length, width, height = sumo_dims(models[FIXED[vc]])
        return with_attrs(line, {"length": length, "width": width, "height": height, **(attrs or {})})

    text = re.sub(r"<vType\b[^>]*/>", fix, text, flags=re.S)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


class Typer:
    """Gives each vehicle inserted by the viewer the vType of the CARLA model that will draw it."""

    def __init__(self, conn, cands, models: dict, base_types: dict[str, str]):
        """base_types: SUMO type id -> vClass (the demand's types)."""
        from cosim.blueprints import choose

        self.conn, self.cands, self.models, self.choose = conn, cands, models, choose
        self.base = base_types
        self.made: set[str] = set()
        self.missing: set[str] = set()

    def model_for(self, vid: str, vclass: str) -> str:
        if vclass in FIXED:
            return FIXED[vclass]
        return self.choose(self.cands, vclass, vid)[0]

    def type_for(self, vid: str, base: str, model: str | None = None) -> str:
        """`base` sized to the CARLA model that will draw `vid` (chosen for its class, or `model`)."""
        vclass = self.base.get(base, "passenger")
        model = model or self.model_for(vid, vclass)
        if model not in self.models:
            self.missing.add(model)
            return base
        tid = f"{base}{SEP}{model}"
        if tid not in self.made:
            length, width, height = sumo_dims(self.models[model])
            vt = self.conn.vehicletype
            vt.copy(base, tid)
            vt.setLength(tid, length)
            vt.setWidth(tid, width)
            vt.setHeight(tid, height)
            self.made.add(tid)
        return tid
