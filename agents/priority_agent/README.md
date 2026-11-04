# ADSP — Intelligent Multi-Agent Traffic Management

Gestion intelligente des feux pour faire passer les véhicules d'urgence (ambulance, pompiers,
police, **sirène active**) sans bloquer le reste du trafic. Architecture :

```
            ┌───────────────────────────────┐   modes + objectifs    ┌────────────────────┐
 V2I  ───▶  │ Coordinateur stratégique       │ ─────────────────────▶ │  PPO (par carrefour)│
 (SUMO)     │  B4 : règles   |  B5 : agent LLM│   validés par         │  choisit la phase   │
            │  (LangChain + Ollama, tools)    │   ModeSafety          └─────────┬──────────┘
            └───────────────────────────────┘                                  │ action
                                                                    ┌──────────▼──────────┐
                                                                    │  Safety Controller   │
                                                                    │  masque, vert min/max│
                                                                    │  jaune, fraîcheur    │
                                                                    └──────────┬──────────┘
                                                                               │ phase validée
                                                                    ┌──────────▼──────────┐
                                                                    │   SUMO (TraCI)       │
                                                                    └─────────────────────┘
```

Le LLM décide **quoi** et **quand** (corridor, arbitrage, cas dégradés, récupération). PPO décide
**comment** sur chaque carrefour. Le Safety Controller valide tout. Le LLM n'a aucun outil
d'exécution des feux.

## Résultats (corridor J1-J2-J3, 5 graines de test, 210 simulations)

| Configuration | Urgences : temps perdu (s) | Voitures : temps perdu (s) | Bus en retard (s) |
|---|---|---|---|
| B1 cycle fixe | 54,8 | 39,2 | 58,2 |
| B2 actionné | 19,3 | 15,3 | 20,0 |
| B2P cycle fixe + préemption | 9,3 | 38,6 | 62,2 |
| B2AP actionné + préemption | 9,0 | 15,4 | 20,1 |
| B3 PPO seul | 24,7 | 18,4 | 26,7 |
| **B3L PPO + priorité locale** | **7,9** | 18,2 | **3,4** |
| B4 PPO + coordinateur à règles | 8,0 | 18,3 | 5,7 |
| B5 PPO + agent LLM | à mesurer (Ollama) | | |

Zéro violation de sécurité sur 210 épisodes. Détail, tests statistiques et rediffusion :
`dashboard/dashboard.html`. Rapport complet : document « Rapport final — Gestion intelligente des feux (ADSP) ».

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS : source .venv/bin/activate
pip install -r requirements-research.txt   # tout le projet ADSP (PPO, dashboard, notebook)
```

Pour Town05 (projet Urban Traffic CARLA) : `tools\setup_goflow_priority.bat` à la racine du projet, qui
installe seulement `requirements.txt` (coordinateur). Voir `town05/README.md`.

`eclipse-sumo` embarque SUMO. `libsumo` accélère les simulations (×15) ; sans lui, TraCI est utilisé.

## Tout reproduire

| Étape | Commande | Durée indicative |
|---|---|---|
| Réseaux SUMO | `python -m simulation.build_networks` | secondes |
| Dataset (data understanding) | `python -m data.export_dataset` | 1 min |
| Notebook d'exploration | `python notebooks/build_01_data_understanding.py` | 1 min |
| Entraînement PPO | `python -m models.train_ppo --steps 1000000` | ~40 min (2 cœurs) |
| Affinage sur le corridor | `python -m models.finetune_corridor --steps 900000` | ~20 min |
| Tests statistiques | `python -m evaluation.stats` | secondes |
| Évaluation carrefour isolé | `python -m evaluation.experiments --single` | 3 min |
| Évaluation corridor B1–B4 | `python -m evaluation.experiments` | 10 min |
| Évaluation B5 (LLM) | `python -m evaluation.experiments --configs B5_ppo_llm --llm-model qwen2.5:7b` | selon le GPU |
| Rediffusion pour le dashboard | `python -m evaluation.record_replay` | 1 min |
| Dashboard | `python -m evaluation.build_dashboard` puis ouvrir `dashboard/dashboard.html` | secondes |
| Démo dans sumo-gui | `python run_demo.py` | 2 min |
| Tests | `python -m pytest -q tests` | 30 s |

Le modèle entraîné et les résultats sont fournis : on peut lancer directement la démo et le dashboard.

### Agent LLM (B5)

1. Installer [Ollama](https://ollama.com), puis `ollama pull qwen2.5:7b` (ou `llama3.1:8b`).
2. `python -m evaluation.experiments --configs B5_ppo_llm --llm-model qwen2.5:7b`
3. `python -m evaluation.stats`
4. `python -m evaluation.record_replay --configs B1_fixe B2AP_actuated_preemption B3L_ppo_local B4_ppo_regles B5_ppo_llm`
5. `python -m evaluation.build_dashboard`

Démo : `python run_demo.py --config B5_ppo_llm` affiche aussi chaque appel d'outil du LLM.

Les résultats B5 sont fusionnés avec les autres : toutes les vues du dashboard les intègrent.

## Organisation

| Dossier | Contenu |
|---|---|
| `simulation/` | réseaux (carrefour isolé, corridor J1-J2-J3, variantes actuated), générateur de scénarios |
| `data/` | export du dataset SUMO (`raw/*.csv`) |
| `notebooks/` | data understanding (exécuté, avec figures) |
| `environment/` | géométrie des carrefours, observation (35 valeurs), récompense, contrôleur de feux, env Gymnasium |
| `safety/` | Safety Controller (actions de feux + modes stratégiques) |
| `coordination/` | suivi V2I des priorités, coordinateurs local / règles / LLM |
| `models/` | entraînement MaskablePPO, checkpoint |
| `evaluation/` | runner, campagne d'expériences, rediffusion, dashboard |
| `results/` | résultats par épisode (`eval_corridor.csv`, `eval_single.csv`, `episodes/`, `replay/`) |
| `reports/figures/` | figures du rapport |
| `tests/` | 31 tests (sécurité, environnement, agent, étape 1) |
| `models/checkpoints/` | `ppo_signal.zip` (carrefour isolé), `ppo_signal_corridor.zip` (affiné, utilisé sur le corridor) |

## Véhicules prioritaires

| Classe | Véhicule | Niveau | Condition |
|---|---|---|---|
| `EMERGENCY_MEDICAL` | ambulance | 3 | sirène active et demande confirmée |
| `FIRE` | pompiers | 3 | idem |
| `POLICE` | police | 2 | idem |
| `TRANSIT` | bus | 1 | en retard de plus de 60 s |

Aucun véhicule n'a le device `bluelight` de SUMO : les urgences respectent les feux, et tout
gain de temps vient du contrôle des feux.
