# GO-FLOW priority agent on Town05 (Urban Traffic CARLA)

What was changed so that this agent can run in the Town05 project (4 Nov 2026). The code of the
`priority-agent` branch (commit `9024456`) is copied **unchanged** into `agents/priority_agent/`; everything
Town05 needs is **added** in `town05/`.

| File | What it does |
|---|---|
| `town05/conflicts.py` | `LinkMap`: which link indices of a signal one car green serves together (from the program the town sends). `TownJunction`: what `coordination.rules` reads from an `Intersection`, keyed by link instead of N/E/S/W. |
| `town05/tracker.py` | `TownTracker`: builds `PriorityVehicle` / `TLSStop` from the town's `state` message (vClass emergency → ambulance level 3, authority → police level 2, incident recovery truck → level 1; `siren_on` = on a run; `confirmed` = V2I or camera vote). |
| `town05/coordinator.py` | `TownRuleCoordinator(RuleCoordinator)`: the B4 rules unchanged; only the conflict test uses links, and the plan carries every link one green serves together. |
| `town05/serve.py` | The child process: JSON lines on stdin/stdout (`hello`→`ready`, `state`→`plan`, `bye`). Writes `out/goflow_priority/<run>/decisions.jsonl`, `summary.json`. |
| `town05/shadow_check.py` | Step 2: the unchanged code watching a Town05 run, logging only. |
| `town05/tests/` | 8 tests, no SUMO needed. |
| `requirements.txt` | Now the coordinator only (numpy, SUMO 1.27.1). The old full list is `requirements-research.txt`, SUMO pinned to 1.27.1; PPO and LLM extras in `requirements-ppo.txt`, `requirements-llm.txt`. |

Not used on Town05: the PPO policy (`environment/`, `models/`), `SignalSafety` and `SignalController` (the junction
agents keep their own safety rules), the LLM coordinator (offline later), bus priority (Town05 buses have no
timetable).

Run (from `agents/priority_agent`, with its venv):

    python -m pytest -q town05/tests
    python -m town05.shadow_check --seed 11 --scale 0.8 --end 1800
