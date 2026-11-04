"""Démo : le corridor J1-J2-J3 dans sumo-gui, piloté par l'architecture complète.

  python run_demo.py                          # PPO + coordinateur à règles (B4), scénario S4 (demandes concurrentes)
  python run_demo.py --config B1_fixe         # comparer avec les feux à cycle fixe
  python run_demo.py --config B5_ppo_llm --llm-model qwen2.5:7b   # avec l'agent LLM (Ollama requis)
  python run_demo.py --scenario S5_degrade    # fausse demande, sirène coupée, perte de capteurs
  python run_demo.py --scenario S7_dispatch --config B5_ppo_llm --llm-model qwen2.5:7b   # message du SAMU lu par le LLM

Dans sumo-gui : ambulance en blanc, pompiers en rouge, police en bleu foncé, bus en bleu clair, chacun entouré d'un cercle.
La simulation avance vite tant qu'il n'y a que des voitures, puis ralentit et la caméra suit chaque urgence.
Le terminal affiche en direct les changements de mode décidés par le coordinateur.
"""
import argparse

from evaluation.experiments import load_model, ollama_available
from evaluation.runner import CONFIGS, PPO_CONFIGS, run_episode
from simulation.scenarios import ALL_SCENARIOS, eval_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="B4_ppo_regles", choices=CONFIGS)
ap.add_argument("--scenario", default="S4_concurrentes", choices=list(ALL_SCENARIOS))
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--delay", type=int, default=250, help="ms par pas pendant le passage des urgences (avance rapide le reste du temps)")
ap.add_argument("--llm-model", default="qwen2.5:3b")
ap.add_argument("--llm-timeout", type=float, default=90.0)
ap.add_argument("--ollama-url", default="http://localhost:11434")
ap.add_argument("--no-gui", action="store_true")
a = ap.parse_args()

spec = eval_scenario(a.scenario, a.seed)
print(f"Scénario {a.scenario} (graine {a.seed}) : {spec['desc']}")
print(f"Configuration : {a.config}\n")
model = load_model() if a.config in PPO_CONFIGS else None
llm = None
if a.config == "B5_ppo_llm":
    if not ollama_available(a.ollama_url):
        raise SystemExit(f"Ollama injoignable sur {a.ollama_url}. Lancez `ollama serve` puis `ollama pull {a.llm_model}`.")
    from coordination.llm_agent import make_ollama, warmup
    llm = make_ollama(a.llm_model, a.ollama_url, timeout_s=a.llm_timeout)
    warmup(llm, a.llm_model)
out = run_episode(a.config, spec, model=model, llm=llm, llm_name=a.llm_model, gui=not a.no_gui,
                  backend="traci", verbose=True, gui_delay_ms=a.delay, llm_timeout_s=a.llm_timeout)
m = out["metrics"]
print("\nRésultats")
for k in ["emg_detail", "critical_time_loss", "emg_time_loss", "emg_stops", "car_time_loss", "car_waiting", "queue_mean",
          "safety_rejected_modes", "safety_state_violations", "safety_fallbacks"]:
    print(f"  {k:24} {m[k]}")
if a.config == "B5_ppo_llm":
    for tr in out["extra"]["llm_transcripts"]:
        print(f"\n[LLM t={tr['time']}s, {tr['latency_s']} s] " + " ; ".join(tr["events"]))
        for s in tr["steps"]:
            print(f"   -> {s['tool']}({s['args']})")
        print("   " + str(tr.get("final") or tr.get("error", "")))
