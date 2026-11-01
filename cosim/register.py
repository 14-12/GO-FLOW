"""The town's vehicle register (31 Oct): every plate on the road, the car it is on and its registered owner.

In real life the police office looks a plate up in the national vehicle register (in Tunisia, the registration
certificate, "carte grise", kept by the ATTT): the car's make, model, colour and first registration, and its keeper's
name and address. Here the register is made from the town itself: each car of the viewer carries its own plates
(cosim/plate_proc.py), and its make, model and paint are those of its CARLA model; its owner is made up.

THE OWNERS ARE FICTIONAL. Names are drawn from lists of common Tunisian first names and family names, addresses from
made-up street names, and every identity number starts with "SIM-" so that it can never be taken for a real one. The
register says so in its own header, and so does every notice made from it.

The red-light agent never reads this (it only reads plates from pictures); the police office does, as in real life.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

FIRST = ["Mohamed", "Ahmed", "Ali", "Youssef", "Amine", "Hamza", "Oussama", "Skander", "Aziz", "Bilel", "Karim",
         "Mehdi", "Nizar", "Sami", "Walid", "Fatma", "Amira", "Sarra", "Ines", "Mariem", "Rania", "Nour", "Yasmine",
         "Salma", "Asma", "Hiba", "Emna", "Olfa", "Rim", "Sonia"]
LAST = ["Gharbi", "Jlassi", "Hammami", "Mansouri", "Ayari", "Ferchichi", "Dridi", "Saidi", "Khelifi", "Mejri",
        "Chaabane", "Riahi", "Sassi", "Zouari", "Baccouche", "Amri", "Belhaj", "Bouzid", "Chebbi", "Hadded",
        "Jaziri", "Kammoun", "Labidi", "Mabrouk", "Nasri", "Ouertani", "Rekik", "Selmi", "Tlili", "Zribi"]
STREETS = ["Rue des Jasmins", "Rue des Orangers", "Avenue des Palmiers", "Rue de l'Olivier", "Rue des Roses",
           "Avenue de la Mer", "Rue du Port", "Rue des Artisans", "Rue de la Medina", "Avenue des Pins",
           "Rue des Amandiers", "Rue de la Gare", "Rue du Marche", "Avenue du Lac", "Rue des Figuiers"]
TOWNS = [("Tunis", 1000), ("La Marsa", 2070), ("Ariana", 2080), ("Ben Arous", 2013), ("Le Bardo", 2000),
         ("La Goulette", 2060), ("Carthage", 2016), ("El Menzah", 1004), ("Manouba", 2010), ("Rades", 2040)]
VANS = ("sprinter", "volkswagen.t2", "carlacola", "ford.ambulance", "transit", "fusorosa")
COMPANIES = ["Societe de Location (fictive)", "Transports du Lac (fictive)", "Livraisons Express (fictive)"]

# CARLA paint colours (blueprint "color" attribute, "r,g,b") -> the colour written on the registration
COLOUR_NAMES = {"white": (235, 235, 235), "black": (20, 20, 20), "silver": (170, 170, 175), "grey": (100, 100, 105),
                "red": (180, 25, 25), "dark red": (100, 15, 20), "blue": (30, 60, 170), "dark blue": (20, 30, 80),
                "light blue": (110, 160, 210), "green": (30, 120, 50), "dark green": (20, 60, 30),
                "yellow": (230, 200, 30), "orange": (230, 120, 20), "brown": (110, 70, 40), "beige": (210, 190, 150)}

FICTIONAL = ("FICTIONAL: a simulated register for the Urban Traffic CARLA project. No person or address in it is "
             "real; identity numbers start with SIM-.")


def colour_name(rgb: str | None) -> str:
    if not rgb:
        return "unknown"
    try:
        r, g, b = (int(float(x)) for x in rgb.split(","))
    except ValueError:
        return "unknown"
    return min(COLOUR_NAMES, key=lambda n: sum((a - b_) ** 2 for a, b_ in zip(COLOUR_NAMES[n], (r, g, b), strict=False)))


def make_model(type_id: str | None) -> tuple[str, str]:
    """CARLA blueprint id -> (make, model): 'vehicle.audi.tt' -> ('Audi', 'TT')."""
    if not type_id or not type_id.startswith("vehicle."):
        return "unknown", "unknown"
    parts = type_id.split(".")
    make = parts[1].replace("_", " ").title()
    model = " ".join(parts[2:]).replace("_", " ").title() if len(parts) > 2 else ""
    fixes = {"Bmw": "BMW", "Gazelle": "Gazelle", "Carlamotors": "CarlaMotors", "Mini": "MINI"}
    return fixes.get(make, make), model.replace("Tt", "TT").replace("Mkz", "MKZ").replace("Etron", "e-tron")


def first_registered(plate: str) -> int | None:
    """About when a private plate's series was issued (series ~ 260 in 2026, about 12 a year before)."""
    try:
        series = int(plate.split()[0])
    except (ValueError, IndexError):
        return None
    if " TU " not in plate:
        return None
    return max(1995, 2026 - round((260 - series) / 12.0))


def _rng(plate: str) -> random.Random:
    return random.Random(int(hashlib.sha256(plate.encode()).hexdigest()[:12], 16))


def record_for(plate: str, type_id: str | None, rgb: str | None) -> dict:
    """The register's entry for a plate on a car of CARLA model `type_id` painted `rgb` (owner made up; module
    doc). The same plate always gives the same owner."""
    r = _rng(plate)
    make, model = make_model(type_id)
    state = " TU " not in plate
    if state:
        owner = {"kind": "state", "name": "State vehicle (ministry fleet)", "address": None, "id": None}
    elif any(k in (type_id or "") for k in VANS) and r.random() < 0.8:  # vans: mostly a company's
        owner = {"kind": "company", "name": r.choice(COMPANIES), "id": f"SIM-RC-{r.randrange(10 ** 6):06d}",
                 "address": _address(r)}
    else:
        owner = {"kind": "person", "name": f"{r.choice(FIRST)} {r.choice(LAST)}", "id": f"SIM-{r.randrange(10 ** 8):08d}",
                 "address": _address(r)}
    return {"plate": plate, "vehicle": {"make": make, "model": model, "colour": colour_name(rgb),
                                        "carla_model": type_id, "first_registered": first_registered(plate)},
            "owner": owner, "fictional": True}


def _address(r: random.Random) -> str:
    town, code = r.choice(TOWNS)
    return f"{r.randrange(1, 140)} {r.choice(STREETS)}, {code + r.randrange(0, 9)} {town}"


class Register:
    """The register as a file (out/enforcement/register.json): {plate: entry}, with its FICTIONAL header."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.is_file():
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = d.get("entries", {})

    def add(self, plate: str, type_id: str | None, rgb: str | None) -> bool:
        """A car of the town and its plate (as registered: once; True if new)."""
        if not plate or plate in self.entries:
            return False
        self.entries[plate] = record_for(plate, type_id, rgb)
        return True

    def lookup(self, plate: str) -> dict | None:
        return self.entries.get(plate)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"note": FICTIONAL, "entries": self.entries}, indent=1, ensure_ascii=False),
                             encoding="utf-8")
