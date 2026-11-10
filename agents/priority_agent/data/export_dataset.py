"""Génère le dataset de Data Understanding à partir de SUMO (système actuel : feux à cycle fixe).

Tables produites dans data/raw/ :
  lanes.csv             mesures par voie toutes les 5 s (véhicules, arrêts, vitesse, occupation, attente)
  trips.csv             un trajet par véhicule (tripinfo SUMO : durée, attente, arrêts, temps perdu)
  priority_tracks.csv   position / vitesse / sirène des véhicules prioritaires chaque seconde
  priority_requests.csv messages V2I émis (classe, sirène, confirmation, approche, distance, ETA)
  signals.csv           phase des feux chaque seconde
  scenarios.csv         description de chaque scénario (réseau, demande, nb d'urgences)

Usage : python -m data.export_dataset --train-scenarios 30
"""
from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from environment.runtime import Runtime
from simulation.scenarios import EVAL_SCENARIOS, eval_scenario, random_training_scenario, write_scenario

OUT = Path(__file__).parent / "raw"


def run_fixed(spec: dict, tmp: Path) -> dict[str, list]:
    cfg = write_scenario(spec, tmp, tripinfo=True)
    rt = Runtime(cfg, seed=spec["seed"], events=spec["events"]).start()
    lanes, tracks, reqs, sig = [], [], [], []
    seen_req = set()
    c = rt.c
    while c.simulation.getMinExpectedNumber() > 0 and rt.t < spec["duration"] + spec.get("drain", 0):
        rt.step()
        t = rt.t
        for tls, inter in rt.inters.items():
            sig.append((spec["name"], t, tls, c.trafficlight.getPhase(tls), c.trafficlight.getRedYellowGreenState(tls)))
            if int(t) % 5 == 0:
                for a, lns in inter.incoming_lanes.items():
                    for ln in lns:
                        lanes.append((spec["name"], t, tls, a, ln, c.lane.getLastStepVehicleNumber(ln),
                                      c.lane.getLastStepHaltingNumber(ln), c.lane.getLastStepMeanSpeed(ln),
                                      c.lane.getLastStepOccupancy(ln), c.lane.getWaitingTime(ln)))
        for vid, pv in rt.tracker.vehicles.items():
            nxt = pv.upcoming[0] if pv.upcoming else None
            tracks.append((spec["name"], t, vid, pv.priority_class, pv.siren_on, pv.confirmed, pv.speed, pv.edge,
                           nxt.tls_id if nxt else None, nxt.distance_m if nxt else None))
            if nxt and nxt.distance_m <= 300 and (vid, nxt.tls_id) not in seen_req:
                seen_req.add((vid, nxt.tls_id))
                reqs.append((spec["name"], t, vid, pv.priority_class, pv.priority_level, pv.siren_on, pv.confirmed,
                             pv.late_s, nxt.tls_id, nxt.approach, nxt.distance_m, nxt.eta_s, pv.is_active))
    rt.close()
    trips = []
    for tr in ET.parse(tmp / f"{spec['name']}.tripinfo.xml").getroot().findall("tripinfo"):
        trips.append((spec["name"], tr.get("id"), tr.get("vType"), float(tr.get("depart")), float(tr.get("arrival")),
                      float(tr.get("duration")), float(tr.get("routeLength")), float(tr.get("waitingTime")),
                      int(tr.get("waitingCount")), float(tr.get("timeLoss"))))
    return {"lanes": lanes, "tracks": tracks, "reqs": reqs, "sig": sig, "trips": trips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-scenarios", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="adsp_data_"))
    specs = [random_training_scenario(90000 + i) for i in range(args.train_scenarios)]
    specs += [eval_scenario(n, 0) for n in EVAL_SCENARIOS]
    acc = {k: [] for k in ("lanes", "tracks", "reqs", "sig", "trips")}
    scen = []
    for spec in specs:
        res = run_fixed(spec, tmp)
        for k in acc:
            acc[k] += res[k]
        scen.append((spec["name"], spec["network"], round(spec["demand_level"], 3), len(spec["priority"]),
                     len(spec.get("buses", [])), len(spec["events"]), spec.get("desc", "aléatoire (entraînement)")))
        print(spec["name"], len(res["trips"]), "trajets")
    pd.DataFrame(acc["lanes"], columns=["scenario", "time", "tls_id", "approach", "lane", "vehicles", "halting",
                                        "mean_speed", "occupancy", "waiting_time"]).to_csv(OUT / "lanes.csv", index=False)
    pd.DataFrame(acc["trips"], columns=["scenario", "vehicle_id", "vtype", "depart", "arrival", "duration",
                                        "route_length", "waiting_time", "stops", "time_loss"]).to_csv(OUT / "trips.csv", index=False)
    pd.DataFrame(acc["tracks"], columns=["scenario", "time", "vehicle_id", "priority_class", "siren_on", "confirmed",
                                         "speed", "edge", "next_tls", "dist_to_tls"]).to_csv(OUT / "priority_tracks.csv", index=False)
    pd.DataFrame(acc["reqs"], columns=["scenario", "time", "vehicle_id", "priority_class", "priority_level", "siren_on",
                                       "confirmed", "late_s", "tls_id", "approach", "distance_m", "eta_s",
                                       "valid_request"]).to_csv(OUT / "priority_requests.csv", index=False)
    pd.DataFrame(acc["sig"], columns=["scenario", "time", "tls_id", "phase", "state"]).to_csv(OUT / "signals.csv", index=False)
    pd.DataFrame(scen, columns=["scenario", "network", "demand_level", "n_priority", "n_bus_lines", "n_events",
                                "description"]).to_csv(OUT / "scenarios.csv", index=False)
    print("dataset écrit dans", OUT)


if __name__ == "__main__":
    main()
