"""Enregistre une rediffusion (positions des véhicules, feux, modes) pour le dashboard.

Usage : python -m evaluation.record_replay --scenario S4_concurrentes --seed 1 \
            --configs B1_fixe B2P_preemption B4_ppo_regles [B5_ppo_llm --llm-model qwen2.5:7b]
Sortie : results/replay/<scenario>_<seed>__<config>.json + results/replay/network.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sumolib

from evaluation.experiments import load_model, ollama_available
from evaluation.runner import PPO_CONFIGS, emergency_ids, run_episode
from simulation.scenarios import NET, eval_scenario

OUT = Path(__file__).parents[1] / "results" / "replay"
VTYPE = {"car": 0, "bus_transit": 1, "ambulance": 2, "fire_truck": 3, "police": 4}


def export_network() -> None:
    net = sumolib.net.readNet(str(NET / "corridor.net.xml"))
    lanes = []
    for e in net.getEdges():
        for ln in e.getLanes():
            lanes.append([[round(x, 1), round(y, 1)] for x, y in ln.getShape()])
    from environment.intersection import _direction
    juncs = []
    for n in net.getNodes():
        if n.getType() != "traffic_light":
            continue
        x0, y0 = n.getCoord()
        heads = {}
        for tl in net.getTrafficLights():
            if tl.getID() != n.getID():
                continue
            for in_lane, _out, idx in tl.getConnections():
                fx, fy = in_lane.getEdge().getFromNode().getCoord()
                a = _direction(fx - x0, fy - y0)
                end = in_lane.getShape()[-1]
                h = heads.setdefault(a, {"links": [], "pts": []})
                h["links"].append(idx)
                h["pts"].append(end)
        for a, h in heads.items():
            xs = [p[0] for p in h["pts"]]
            ys = [p[1] for p in h["pts"]]
            h["x"], h["y"] = round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1)
            h["links"] = sorted(set(h["links"]))
            del h["pts"]
        juncs.append({"id": n.getID(), "shape": [[round(x, 1), round(y, 1)] for x, y in n.getShape()],
                      "x": x0, "y": y0, "heads": heads})
    (OUT / "network.json").write_text(json.dumps({"lanes": lanes, "junctions": juncs}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S4_concurrentes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--configs", nargs="+", default=["B1_fixe", "B2AP_actuated_preemption", "B3L_ppo_local", "B4_ppo_regles"])
    ap.add_argument("--llm-model", default="qwen2.5:3b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    export_network()
    model = load_model() if set(a.configs) & PPO_CONFIGS else None
    llm = None
    if "B5_ppo_llm" in a.configs:
        if ollama_available(a.ollama_url):
            from coordination.llm_agent import make_ollama, warmup
            llm = make_ollama(a.llm_model, a.ollama_url)
            warmup(llm, a.llm_model)
        else:
            print("Ollama injoignable : B5 ignoré")
            a.configs.remove("B5_ppo_llm")
    spec = eval_scenario(a.scenario, a.seed)
    deps = [p["depart"] for p in spec["priority"]]
    t0, t1 = (min(deps) - 15, max(deps) + 150) if deps else (100, 300)
    for cfg in a.configs:
        out = run_episode(cfg, spec, model=model, llm=llm, llm_name=a.llm_model, record=True)
        frames = [f for f in out["extra"]["frames"] if t0 <= f["t"] <= t1]
        emg = emergency_ids(spec)
        trips = {tr["id"]: tr for tr in out["extra"]["trips"]}
        comp = {
            "config": cfg, "scenario": a.scenario, "seed": a.seed, "desc": spec["desc"],
            "metrics": {k: v for k, v in out["metrics"].items() if not isinstance(v, dict)},
            "emergency": {v: {"time_loss": trips[v]["time_loss"], "stops": trips[v]["stops"]} for v in emg if v in trips},
            "frames": [{"t": f["t"],
                        "v": [[x[0], x[1], x[2], x[3], VTYPE.get(x[4], 0)] for x in f["v"]],
                        "s": f["s"], "m": f["m"]} for f in frames],
            "decisions": out["extra"].get("decisions", [])[:80],
            "llm_transcripts": out["extra"].get("llm_transcripts", [])[:20],
        }
        p = OUT / f"{a.scenario}_{a.seed}__{cfg}.json"
        p.write_text(json.dumps(comp, separators=(",", ":")))
        print(cfg, len(comp["frames"]), "images,", round(p.stat().st_size / 1e3), "ko", comp["emergency"])


if __name__ == "__main__":
    main()
