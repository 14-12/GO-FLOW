"""Campagne d'évaluation : configurations x scénarios x graines, mêmes conditions pour tous.

Usage :
  python -m evaluation.experiments                         # B1..B4 sur le corridor (6 scénarios x 5 graines)
  python -m evaluation.experiments --configs B5_ppo_llm --llm-model qwen2.5:7b   # architecture complète (Ollama)
  python -m evaluation.experiments --single                # évaluation du modèle PPO sur le carrefour isolé

Les résultats sont fusionnés dans results/eval_corridor.csv (une ligne par épisode) : relancer des
épisodes remplace seulement ces épisodes (configuration, scénario, graine). Le détail de chaque épisode (modes, décisions, rejets
de sécurité, transcriptions LLM) est dans results/episodes/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.runner import CONFIGS, PPO_CONFIGS, run_episode
from simulation.scenarios import EVAL_SCENARIOS, eval_scenario, random_training_scenario

ROOT = Path(__file__).parents[1]
RES = ROOT / "results"
MODEL = ROOT / "models" / "checkpoints" / "ppo_signal.zip"                    # carrefour isolé
CORRIDOR_MODEL = ROOT / "models" / "checkpoints" / "ppo_signal_corridor.zip"  # affiné sur le corridor


def load_model(path=None, corridor: bool = True):
    """Par défaut : le modèle affiné sur le corridor (repli sur le modèle isolé s'il manque)."""
    from sb3_contrib import MaskablePPO
    if path is None:
        path = CORRIDOR_MODEL if corridor and CORRIDOR_MODEL.exists() else MODEL
    return MaskablePPO.load(path, device="cpu")


def ollama_available(base_url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _save(rows: list[dict], csv: Path, configs: list[str], keys: list[str]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    if "emg_detail" in new:
        new["emg_detail"] = new["emg_detail"].apply(json.dumps)
    if csv.exists():
        old = pd.read_csv(csv)
        # ne remplace que les épisodes relancés (mêmes clés), pas toute la configuration
        k = [c for c in keys if c in old and c in new]
        if k:
            done = set(map(tuple, new[k].astype(str).values))
            old = old[[tuple(r) not in done for r in old[k].astype(str).values]]
        else:
            old = old[~old.config.isin(configs)]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(keys).to_csv(csv, index=False)
    return new


def corridor(configs, seeds, scenarios, llm_model, base_url, horizon=None, tag=None, model_path=None,
             llm_timeout=90.0):
    RES.mkdir(exist_ok=True)
    (RES / "episodes").mkdir(exist_ok=True)
    model = load_model(model_path) if set(configs) & PPO_CONFIGS else None
    llm = None
    if "B5_ppo_llm" in configs:
        if not ollama_available(base_url):
            print(f"Ollama injoignable sur {base_url} : B5 ignoré. Installez Ollama puis `ollama pull {llm_model}`.")
            configs = [c for c in configs if c != "B5_ppo_llm"]
        else:
            from coordination.llm_agent import make_ollama, warmup
            llm = make_ollama(llm_model, base_url, timeout_s=llm_timeout)
            warmup(llm, llm_model)
    rows = []
    for cfg in configs:
        for sc in scenarios:
            for seed in seeds:
                spec = eval_scenario(sc, seed)
                kw = {"horizon_s": horizon} if horizon else {}
                kw["llm_timeout_s"] = llm_timeout
                print(f"--- {cfg} / {sc} / graine {seed} ...", flush=True)
                out = run_episode(cfg, spec, model=model, llm=llm, llm_name=llm_model, **kw)
                rows.append(out["metrics"])
                if tag:
                    continue
                (RES / "episodes" / f"{cfg}__{sc}__{seed}.json").write_text(
                    json.dumps({"metrics": out["metrics"], **out["extra"]}, default=str))
                m = out["metrics"]
                print(f"{cfg:16} {sc:16} seed={seed} urg={m['emg_time_loss']} voit={m['car_time_loss']} "
                      f"rejets={m['safety_rejected_actions']}/{m['safety_rejected_modes']} viol={m['safety_state_violations']}"
                      + (f" llm={m.get('llm_calls')} appels {m.get('llm_latency_mean_s')} s" if cfg == "B5_ppo_llm" else ""),
                      flush=True)
    if rows:
        name = f"eval_corridor_{tag}.csv" if tag else "eval_corridor.csv"
        _save(rows, RES / name, configs, ["config", "scenario", "seed"])


def single(n: int):
    """Évaluation du contrôleur PPO seul sur 30 carrefours isolés jamais vus (graines 70000+)."""
    RES.mkdir(exist_ok=True)
    model = load_model(corridor=False)
    configs = ["B1_fixe", "B2_actuated", "B3_ppo", "B3L_ppo_local"]
    rows = []
    for i in range(n):
        spec = random_training_scenario(70000 + i)
        spec["drain"] = 300
        for cfg in configs:
            m = run_episode(cfg, spec, model=model)["metrics"]
            m["demand_level"] = spec["demand_level"]
            rows.append(m)
        print(f"scénario {i}: " + " | ".join(f"{r['config']}={r['car_time_loss']}" for r in rows[-4:]), flush=True)
    _save(rows, RES / "eval_single.csv", configs, ["config", "scenario"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=[c for c in CONFIGS if c != "B5_ppo_llm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--scenarios", nargs="+", default=list(EVAL_SCENARIOS))
    ap.add_argument("--llm-model", default="qwen2.5:3b")
    ap.add_argument("--llm-timeout", type=float, default=90.0, help="durée max d'un appel au LLM (s)")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--horizon", type=float, default=None, help="horizon d'anticipation de B4 (s)")
    ap.add_argument("--tag", default=None, help="suffixe du fichier de résultats (validation)")
    ap.add_argument("--model", default=None, help="checkpoint PPO (défaut : models/checkpoints/ppo_signal.zip)")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--n-single", type=int, default=30)
    a = ap.parse_args()
    if a.single:
        single(a.n_single)
    else:
        corridor(a.configs, a.seeds, a.scenarios, a.llm_model, a.ollama_url, a.horizon, a.tag, a.model, a.llm_timeout)


if __name__ == "__main__":
    main()
