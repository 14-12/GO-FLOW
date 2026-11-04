"""Générateur de scénarios SUMO (trafic ordinaire + véhicules prioritaires + événements).

- random_training_scenario : scénarios aléatoires sur le carrefour isolé (entraînement PPO).
- EVAL_SCENARIOS           : 6 scénarios d'évaluation sur le corridor (mêmes pour toutes les méthodes).

Chaque scénario est décrit par un dict sérialisable (`spec`) puis écrit en fichiers
.rou.xml + .sumocfg. Les événements (sirène coupée, perte de capteur) sont appliqués par le
runner pendant la simulation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
NET = ROOT / "network"

VTYPES = """  <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="16.67"/>
  <vType id="ambulance" vClass="emergency" accel="2.8" decel="5.0" sigma="0.2" length="6.5" maxSpeed="19.44"
         speedFactor="1.2" guiShape="emergency" color="1,1,1">
    <param key="priority_class" value="EMERGENCY_MEDICAL"/><param key="siren" value="on"/>
  </vType>
  <vType id="fire_truck" vClass="emergency" accel="1.8" decel="4.5" sigma="0.2" length="10" maxSpeed="16.67"
         speedFactor="1.1" guiShape="firebrigade" color="1,0,0">
    <param key="priority_class" value="FIRE"/><param key="siren" value="on"/>
  </vType>
  <vType id="police" vClass="authority" accel="3.0" decel="5.0" sigma="0.2" length="5" maxSpeed="22.22"
         speedFactor="1.2" guiShape="police" color="0,0,0.6">
    <param key="priority_class" value="POLICE"/><param key="siren" value="on"/>
  </vType>
  <vType id="bus_transit" vClass="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="13.89"
         guiShape="bus" color="0,0.6,1">
    <param key="priority_class" value="TRANSIT"/>
  </vType>
"""
# Aucun vType n'a de device bluelight : les urgences respectent les feux.

# --------------------------------------------------------------------------- demande
SINGLE_IN = {"N": "N2C", "E": "E2C", "S": "S2C", "W": "W2C"}
SINGLE_OUT = {"N": "C2N", "E": "C2E", "S": "C2S", "W": "C2W"}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT = {"N": "E", "E": "S", "S": "W", "W": "N"}    # depuis l'approche N, tourner à gauche -> sortie E
RIGHT = {"N": "W", "E": "N", "S": "E", "W": "S"}

CORRIDOR_ENTRIES = {"W0": "W02J1", "E0": "E02J3", "N1": "N12J1", "S1": "S12J1", "N2": "N22J2",
                    "S2": "S22J2", "N3": "N32J3", "S3": "S32J3"}
CORRIDOR_EXITS = {"W0": "J12W0", "E0": "J32E0", "N1": "J12N1", "S1": "J12S1", "N2": "J22N2",
                  "S2": "J22S2", "N3": "J32N3", "S3": "J32S3"}


def single_flows(demand: dict[str, float]) -> list[tuple[str, str, float]]:
    """demand : véhicules/h par approche -> liste (from, to, veh/h). 70 % tout droit, 15 % gauche, 15 % droite."""
    flows = []
    for a, q in demand.items():
        for out, share in ((OPPOSITE[a], 0.70), (LEFT[a], 0.15), (RIGHT[a], 0.15)):
            flows.append((SINGLE_IN[a], SINGLE_OUT[out], q * share))
    return flows


def corridor_flows(level: float) -> list[tuple[str, str, float]]:
    """Artère E-O chargée (500 veh/h par sens x level), rues transversales 200 veh/h x level."""
    flows = []
    sides = [k for k in CORRIDOR_ENTRIES if k not in ("W0", "E0")]
    for ent, opp in (("W0", "E0"), ("E0", "W0")):
        q = 500 * level
        flows.append((CORRIDOR_ENTRIES[ent], CORRIDOR_EXITS[opp], q * 0.7))
        for s in sides:
            flows.append((CORRIDOR_ENTRIES[ent], CORRIDOR_EXITS[s], q * 0.3 / len(sides)))
    for s in sides:
        q = 200 * level
        k = s[1]
        opp = ("S" if s[0] == "N" else "N") + k
        flows.append((CORRIDOR_ENTRIES[s], CORRIDOR_EXITS[opp], q * 0.6))
        flows.append((CORRIDOR_ENTRIES[s], CORRIDOR_EXITS["W0"], q * 0.2))
        flows.append((CORRIDOR_ENTRIES[s], CORRIDOR_EXITS["E0"], q * 0.2))
    return flows


# --------------------------------------------------------------------------- écriture
def write_scenario(spec: dict, out_dir: Path, tripinfo: bool = False, fcd: bool = False,
                   actuated: bool = False) -> Path:
    """Écrit .rou.xml et .sumocfg ; renvoie le chemin du .sumocfg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = spec["name"]
    lines = ["<routes>", VTYPES]
    for i, (fr, to, vph) in enumerate(spec["flows"]):
        if vph <= 0:
            continue
        lines.append(f'  <flow id="f{i}_{fr}_{to}" type="car" from="{fr}" to="{to}" begin="0" '
                     f'end="{spec["duration"]}" probability="{min(vph / 3600, 1):.5f}" '
                     f'departLane="best" departSpeed="max"/>')
    for b in spec.get("buses", []):
        lines.append(f'  <flow id="{b["id"]}" type="bus_transit" from="{b["from"]}" to="{b["to"]}" '
                     f'begin="{b["begin"]}" end="{spec["duration"]}" period="{b["period"]}" departLane="best">')
        lines.append(f'    <param key="late_s" value="{b.get("late_s", 0)}"/>')
        lines.append("  </flow>")
    for v in sorted(spec.get("priority", []), key=lambda x: x["depart"]):
        lines.append(f'  <trip id="{v["id"]}" type="{v["type"]}" from="{v["from"]}" to="{v["to"]}" '
                     f'depart="{v["depart"]:.1f}" departLane="best" departSpeed="max">')
        for k, val in v.get("params", {}).items():
            lines.append(f'    <param key="{k}" value="{val}"/>')
        lines.append("  </trip>")
    lines.append("</routes>")
    rou = out_dir / f"{name}.rou.xml"
    rou.write_text("\n".join(lines) + "\n")

    net_name = spec["network"] + ("_actuated" if actuated else "")
    net = (NET / f"{net_name}.net.xml").resolve()
    outputs = ""
    if tripinfo:
        outputs += f'    <tripinfo-output value="{name}.tripinfo.xml"/>\n'
    if fcd:
        outputs += f'    <fcd-output value="{name}.fcd.xml"/>\n'
    cfg = out_dir / f"{name}.sumocfg"
    cfg.write_text(f"""<configuration>
  <input>
    <net-file value="{net.as_posix()}"/>
    <route-files value="{rou.name}"/>
  </input>
  <time><begin value="0"/><end value="{spec['duration'] + spec.get('drain', 600)}"/><step-length value="1"/></time>
  <processing><time-to-teleport value="-1"/><ignore-junction-blocker value="-1"/></processing>
  <output>
{outputs}  </output>
  <report><no-step-log value="true"/><no-warnings value="true"/></report>
</configuration>
""")
    (out_dir / f"{name}.spec.json").write_text(json.dumps(spec, indent=1))
    return cfg


# --------------------------------------------------------------------------- entraînement
EMERGENCY_TYPES = ["ambulance", "fire_truck", "police"]


def random_training_scenario(seed: int, duration: int = 900) -> dict:
    rng = random.Random(seed)
    level = rng.uniform(0.35, 1.25)
    demand = {a: 450 * level * rng.uniform(0.5, 1.5) for a in "NESW"}
    priority, events = [], []
    for i in range(rng.choice([0, 1, 1, 2, 2, 3])):
        a = rng.choice("NESW")
        out = rng.choices([OPPOSITE[a], LEFT[a], RIGHT[a]], [0.7, 0.15, 0.15])[0]
        vid = f"emg_{i}"
        dep = rng.uniform(30, duration - 120)
        params = {}
        if rng.random() < 0.1:  # demande non confirmée (fausse ou en attente)
            params["confirmed"] = "false"
            if rng.random() < 0.5:
                params["confirm_at"] = f"{dep + rng.uniform(5, 20):.0f}"
        priority.append({"id": vid, "type": rng.choice(EMERGENCY_TYPES), "from": SINGLE_IN[a],
                         "to": SINGLE_OUT[out], "depart": dep, "params": params})
        if rng.random() < 0.1:
            events.append({"time": dep + rng.uniform(3, 15), "type": "siren_off", "vehicle": vid})
    buses = []
    if rng.random() < 0.7:
        a = rng.choice("NESW")
        buses.append({"id": "bus", "from": SINGLE_IN[a], "to": SINGLE_OUT[OPPOSITE[a]],
                      "begin": rng.uniform(0, 120), "period": rng.uniform(120, 300),
                      "late_s": rng.choice([0, 0, 90, 150])})
    return {"name": f"train_{seed}", "network": "single", "duration": duration, "drain": 0,
            "flows": single_flows(demand), "buses": buses, "priority": priority, "events": events,
            "seed": seed, "demand_level": level}


# --------------------------------------------------------------------------- évaluation (corridor)
def _jit(rng, x, d=15):
    return x + rng.uniform(-d, d)


def scenario_normal(seed):
    return {"desc": "Trafic normal, aucune urgence", "level": 1.0, "priority": [], "events": []}


def scenario_emergency(seed):
    rng = random.Random(seed)
    return {"desc": "Ambulance traversant les 3 carrefours de l'artère, puis police sur une transversale",
            "level": 1.0, "events": [],
            "priority": [
                {"id": "amb_0", "type": "ambulance", "from": "W02J1", "to": "J32E0", "depart": _jit(rng, 200)},
                {"id": "police_0", "type": "police", "from": "N32J3", "to": "J32S3", "depart": _jit(rng, 520)},
            ]}


def scenario_congestion(seed):
    rng = random.Random(seed)
    return {"desc": "Forte congestion (x1.35) avec deux urgences sur l'artère en sens opposés",
            "level": 1.35, "events": [],
            "priority": [
                {"id": "amb_0", "type": "ambulance", "from": "W02J1", "to": "J32E0", "depart": _jit(rng, 250)},
                {"id": "fire_0", "type": "fire_truck", "from": "E02J3", "to": "J12W0", "depart": _jit(rng, 560)},
            ]}


def scenario_concurrent(seed):
    rng = random.Random(seed)
    t = _jit(rng, 300, 10)
    return {"desc": "Demandes concurrentes : ambulance sur l'artère et pompiers sur la transversale de J2 "
                    "arrivant presque en même temps, puis police",
            "level": 1.0, "events": [],
            "priority": [
                {"id": "amb_0", "type": "ambulance", "from": "W02J1", "to": "J32E0", "depart": t},
                {"id": "fire_0", "type": "fire_truck", "from": "N22J2", "to": "J22S2", "depart": t + 28},
                {"id": "police_0", "type": "police", "from": "E02J3", "to": "J12W0", "depart": t + 40},
            ]}


def scenario_degraded(seed):
    rng = random.Random(seed)
    return {"desc": "Cas dégradés : fausse demande non confirmée, sirène coupée en route, "
                    "perte des capteurs de J2 pendant 60 s, puis vraie ambulance",
            "level": 1.0,
            "priority": [
                {"id": "fake_0", "type": "car", "from": "W02J1", "to": "J32E0", "depart": _jit(rng, 150),
                 "params": {"priority_class": "EMERGENCY_MEDICAL", "siren": "on", "confirmed": "false"}},
                {"id": "police_off", "type": "police", "from": "E02J3", "to": "J12W0", "depart": _jit(rng, 330)},
                {"id": "amb_0", "type": "ambulance", "from": "W02J1", "to": "J32E0", "depart": _jit(rng, 600)},
            ],
            "events": [{"time": 345, "type": "siren_off", "vehicle": "police_off"},
                       {"time": 450, "type": "sensor_dropout", "tls": "J2", "until": 510}]}


def scenario_transit(seed):
    rng = random.Random(seed)
    return {"desc": "Bus en retard sur l'artère (priorité conditionnelle) puis ambulance",
            "level": 1.0, "events": [],
            "buses": [{"id": "bus_late", "from": "W02J1", "to": "J32E0", "begin": 30, "period": 150, "late_s": 120},
                      {"id": "bus_ontime", "from": "E02J3", "to": "J12W0", "begin": 90, "period": 150, "late_s": 0}],
            "priority": [
                {"id": "amb_0", "type": "ambulance", "from": "E02J3", "to": "J12W0", "depart": _jit(rng, 480)},
            ]}


EVAL_SCENARIOS = {
    "S1_normal": scenario_normal,
    "S2_urgence": scenario_emergency,
    "S3_congestion": scenario_congestion,
    "S4_concurrentes": scenario_concurrent,
    "S5_degrade": scenario_degraded,
    "S6_transport": scenario_transit,
}


def scenario_dispatch(seed):
    """S7 : information non structurée. Deux ambulances arrivent à J2 par des directions en conflit,
    presque en même temps. Les règles servent la première signalée (amb_0, transfert non urgent).
    Seul un message texte du centre de régulation dit que amb_1 transporte un patient en arrêt cardiaque :
    un coordinateur à règles ne peut pas le lire, le LLM oui."""
    rng = random.Random(seed)
    t = _jit(rng, 300, 10)
    return {"desc": "Message du centre de régulation (texte libre) : deux ambulances en conflit à J2, "
                    "la seconde transporte un patient critique",
            "level": 1.0, "events": [], "critical": "amb_1",
            "priority": [
                {"id": "amb_0", "type": "ambulance", "from": "W02J1", "to": "J32E0", "depart": t},
                {"id": "amb_1", "type": "ambulance", "from": "N22J2", "to": "J22S2", "depart": t + 24},
            ],
            "dispatch": [
                {"time": 120, "text": "Info voirie : nettoyage de la rue N3 prévu demain matin, aucune action requise."},
                {"time": round(t + 24), "text": "Régulation SAMU : l'unité amb_1 (part maintenant du nord, traversera J2 vers le sud) "
                                                "transporte un patient en arrêt cardiaque, pronostic vital engagé. "
                                                "L'unité amb_0 sur l'artère effectue un transfert non urgent."},
            ]}


# Scénarios complémentaires (hors moyennes principales) : évalués par evaluation/dispatch_experiment.py
EXTRA_SCENARIOS = {"S7_dispatch": scenario_dispatch}
ALL_SCENARIOS = {**EVAL_SCENARIOS, **EXTRA_SCENARIOS}


def eval_scenario(name: str, seed: int, duration: int = 900) -> dict:
    s = ALL_SCENARIOS[name](seed)
    return {"name": f"{name}_{seed}", "scenario": name, "network": "corridor", "duration": duration,
            "drain": 600, "flows": corridor_flows(s["level"]), "buses": s.get("buses", []),
            "priority": s["priority"], "events": s["events"], "seed": seed, "desc": s["desc"],
            "demand_level": s["level"], "dispatch": s.get("dispatch", []), "critical": s.get("critical")}
