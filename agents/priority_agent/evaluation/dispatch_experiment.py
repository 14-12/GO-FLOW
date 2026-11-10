"""S7 : l'information non structurée (message du centre de régulation) — là où le LLM apporte quelque chose.

Deux ambulances arrivent à J2 par des directions en conflit. Les règles (B4) servent la première signalée
(amb_0, transfert non urgent). Un message texte du SAMU indique que amb_1 transporte un patient en arrêt
cardiaque : seul l'agent LLM (B5) peut le lire et relever sa priorité (outil escalate_priority, vérifié par
les garde-fous : véhicule d'urgence actif et réellement mentionné par un message reçu).

Usage : python -m evaluation.dispatch_experiment --llm-model qwen2.5:7b
Sortie : results/eval_dispatch.csv + tableau récapitulatif.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.experiments import load_model, ollama_available
from evaluation.runner import run_episode
from simulation.scenarios import eval_scenario

RES = Path(__file__).parents[1] / "results"
CONFIGS = ["B2AP_actuated_preemption", "B4_ppo_regles", "B5_ppo_llm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    ap.add_argument("--llm-timeout", type=float, default=90.0)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    a = ap.parse_args()
    configs = list(a.configs)
    llm = None
    if "B5_ppo_llm" in configs:
        if not ollama_available(a.ollama_url):
            print("Ollama injoignable : B5 ignoré")
            configs.remove("B5_ppo_llm")
        else:
            from coordination.llm_agent import make_ollama, warmup
            llm = make_ollama(a.llm_model, a.ollama_url, timeout_s=a.llm_timeout)
            warmup(llm, a.llm_model)
    model = load_model()
    rows = []
    for seed in a.seeds:
        spec = eval_scenario("S7_dispatch", seed)
        for cfg in configs:
            print(f"--- {cfg} / S7_dispatch / graine {seed} ...", flush=True)
            m = run_episode(cfg, spec, model=model, llm=llm, llm_name=a.llm_model, llm_timeout_s=a.llm_timeout)["metrics"]
            m["routine_time_loss"] = m["emg_detail"].get("amb_0")
            rows.append(m)
            print(f"{cfg:26} graine={seed} patient critique (amb_1)={m['critical_time_loss']} s  "
                  f"transfert (amb_0)={m['routine_time_loss']} s  voitures={m['car_time_loss']} s  "
                  f"violations={m['safety_state_violations']}"
                  + (f"  escalades={m.get('llm_escalations')}" if cfg == "B5_ppo_llm" else ""), flush=True)
    df = pd.DataFrame(rows)
    df["emg_detail"] = df["emg_detail"].astype(str)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "eval_dispatch.csv", index=False)
    cols = ["critical_time_loss", "routine_time_loss", "emg_time_loss", "car_time_loss", "safety_state_violations"]
    print("\nMoyennes sur les graines (temps perdu en s) :")
    print(df.groupby("config")[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
