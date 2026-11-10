"""Étape 1 — Validation de la base technique SUMO + TraCI.

Vérifie :
  1. que la simulation démarre et que l'intersection est lisible ;
  2. qu'aucun véhicule d'urgence (ambulance, pompiers, police) n'a de device bluelight ;
  3. que les messages V2I sont émis à la bonne distance ;
  4. qu'une commande de phase est appliquée et vérifiable ;
  5. qu'une police en patrouille (sirène éteinte) ne déclenche aucune demande ;
  6. que la commande des feux a un effet mesurable sur les urgences avec sirène
     (comparaison cycle fixe vs préemption naïve), sur plusieurs graines.

Usage :  python run_step1.py [--gui] [--seeds 42 43 44]
"""
from __future__ import annotations

import argparse
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.sumo_interface import SIREN_REQUIRED, SumoSession

ROOT = Path(__file__).parent
CFG = ROOT / "simulation/scenarios/single_base.sumocfg"
OUT = ROOT / "outputs/step1"
TLS = "C"


def run_episode(mode: str, seed: int, gui: bool = False, verbose: bool = False) -> dict:
    """mode = 'fixed' (programme SUMO seul) ou 'preempt' (préemption naïve)."""
    OUT.mkdir(parents=True, exist_ok=True)
    tripinfo = OUT / f"tripinfo_{mode}_{seed}.xml"
    sess = SumoSession(CFG, label=f"{mode}_{seed}", seed=seed, gui=gui,
                       extra_args=["--tripinfo-output", str(tripinfo)])
    sess.start()

    checks = {"bluelight_absent": True, "v2i_first_distance": None, "command_verified": None,
              "patrol_no_request": True}
    logged_requests = set()
    while sess.is_running():
        sess.step()
        sess.apply_pending_targets()

        for vid in sess.conn.simulation.getDepartedIDList():
            if sess.get_priority_class(vid) in SIREN_REQUIRED and sess.has_bluelight_device(vid):
                checks["bluelight_absent"] = False

        requests = sess.get_priority_requests(TLS)
        for req in requests:
            if req.request_id not in logged_requests:
                logged_requests.add(req.request_id)
                if req.vehicle_id == "police_patrol":
                    checks["patrol_no_request"] = False
                if req.priority_class in SIREN_REQUIRED and checks["v2i_first_distance"] is None:
                    checks["v2i_first_distance"] = req.distance_m
                if verbose:
                    print(f"[t={sess.time:6.0f}] V2I {req.request_id} {req.priority_class} "
                          f"niveau={req.priority_level} sirène={'on' if req.siren_on else '-'} "
                          f"approche={req.approach_edge} dist={req.distance_m:.0f}m eta={req.eta_s:.0f}s")

        if mode == "preempt" and requests:
            # Urgences avec sirène uniquement (le bus n'est pas préempté à cette étape) ;
            # la liste est déjà triée par niveau de priorité, puis ordre d'arrivée, puis ETA.
            emergencies = [r for r in requests if r.priority_class in SIREN_REQUIRED and r.siren_on]
            if emergencies:
                target = sess.get_approach_green_phase(TLS, emergencies[0].approach_edge)
                res = sess.request_green_phase(TLS, target)
                if res["status"] == "kept" and checks["command_verified"] is None:
                    checks["command_verified"] = sess.verify_signal_state(TLS, target)
    sess.close()

    ambs = [t for t in sess.priority_tracks.values() if t.siren_ever_on and t.arrival]
    trips = ET.parse(tripinfo).getroot().findall("tripinfo")
    ordinary = [t for t in trips if t.get("vType") == "car"]
    return {
        "mode": mode, "seed": seed,
        "amb_travel_time": statistics.mean(t.arrival - t.depart for t in ambs),
        "amb_stops": sum(t.stops for t in ambs),
        "amb_waiting": statistics.mean(t.waiting_time for t in ambs),
        "cars_mean_waiting": statistics.mean(float(t.get("waitingTime")) for t in ordinary),
        "cars_mean_duration": statistics.mean(float(t.get("duration")) for t in ordinary),
        "n_cars": len(ordinary),
        "per_vehicle": {t.vehicle_id: t.arrival - t.depart for t in ambs},
        **checks,
    }


def smoke_test() -> None:
    """Lecture d'état + rejet d'une phase invalide + transition contrôlée."""
    with SumoSession(CFG, label="smoke", seed=42) as s:
        for _ in range(20):
            s.step()
        st = s.get_intersection_state(TLS)
        greens = s.get_green_phases(TLS)
        print(f"Phases vertes : {greens} | phase courante : {st['phase_index']} "
              f"({st['phase_state']}) | file totale : {st['total_queue']}")
        print("Commande phase jaune (1) :", s.request_green_phase(TLS, 1))
        print("Commande phase inexistante (9) :", s.request_green_phase(TLS, 9))
        target = [g for g in greens if g != st["phase_index"]][0]
        print(f"Commande vers phase {target} :", s.request_green_phase(TLS, target))
        for _ in range(5):
            s.step()
            s.apply_pending_targets()
        ok = s.verify_signal_state(TLS, target)
        print(f"Phase {target} active après transition jaune : {ok}")
        assert ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    print("=== Smoke test TraCI ===")
    smoke_test()

    print("\n=== Comparaison cycle fixe vs préemption naïve ===")
    rows = []
    for seed in args.seeds:
        for mode in ("fixed", "preempt"):
            rows.append(run_episode(mode, seed, gui=args.gui, verbose=(seed == args.seeds[0])))

    hdr = f"{'mode':8} {'seed':>4} {'urg_tt':>7} {'arrêts':>6} {'urg_att':>7} {'att_voit':>8} {'durée_voit':>10} {'bluelight_off':>13} {'v2i_dist':>8} {'cmd_ok':>6} {'patrouille_ok':>13}"
    print(hdr)
    for r in rows:
        print(f"{r['mode']:8} {r['seed']:>4} {r['amb_travel_time']:>7.1f} {r['amb_stops']:>6} "
              f"{r['amb_waiting']:>7.1f} {r['cars_mean_waiting']:>8.1f} {r['cars_mean_duration']:>10.1f} "
              f"{str(r['bluelight_absent']):>13} {str(r['v2i_first_distance']):>8} {str(r['command_verified']):>6} {str(r['patrol_no_request']):>13}")

    for mode in ("fixed", "preempt"):
        sub = [r for r in rows if r["mode"] == mode]
        tt = [r["amb_travel_time"] for r in sub]
        wt = [r["cars_mean_waiting"] for r in sub]
        per = {v: statistics.mean(r["per_vehicle"][v] for r in sub) for v in sub[0]["per_vehicle"]}
        print(f"\n{mode:8} par véhicule : " + ", ".join(f"{v}={t:.1f}s" for v, t in per.items()))
        print(f"{mode:8} urgences : {statistics.mean(tt):.1f} ± {statistics.pstdev(tt):.1f} s | "
              f"attente moyenne voitures : {statistics.mean(wt):.1f} ± {statistics.pstdev(wt):.1f} s")


if __name__ == "__main__":
    main()
