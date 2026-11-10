# GO-FLOW priority agent on Town05 (Urban Traffic CARLA)

What was changed so that this agent can run in the Town05 project (4 Nov 2026). The code of the
`priority-agent` branch (commit `9024456`) is copied **unchanged** into `agents/priority_agent/`; everything
Town05 needs is **added** in `town05/`.

| File | What it does |
|---|---|
| `town05/conflicts.py` | `LinkMap`: which link indices of a signal one car green serves together (from the program the town sends). `TownJunction`: what `coordination.rules` reads from an `Intersection`, keyed by link instead of N/E/S/W. |
| `town05/tracker.py` | `TownTracker`: builds `PriorityVehicle` / `TLSStop` from the town's `state` message (vClass emergency → ambulance level 3, authority → police level 2, recovery trucks get no signal priority (they drive with hazard lights); `siren_on` = on a run; `confirmed` = V2I or camera vote). |
| `town05/coordinator.py` | `TownRuleCoordinator(RuleCoordinator)`: the B4 rules unchanged; only the conflict test uses links, and the plan carries every link one green serves together. |
| `town05/serve.py` | The child process: JSON lines on stdin/stdout (`hello`→`ready`, `state`→`plan`, `bye`). Writes `out/goflow_priority/<run>/decisions.jsonl`, `summary.json`. |
| `town05/llm_town.py` | `TownLLMAdvisor`: the LLM coordinator in a background thread, on events; its accepted decisions are laid over the rules' plan. Tools: `get_priority_requests`, `get_camera_sightings`, `get_network_state`, `plan_priority_corridor`, `set_intersection_mode`, `request_green_for_lane`, `escalate_priority`. Rules stay in charge when it is slow, down or refused. |
| `town05/shadow_check.py` | Step 2: the unchanged code watching a Town05 run, logging only. |
| `town05/tests/` | 16 tests, no SUMO or Ollama needed (the LLM tests use a scripted model and need `langchain-core`). |
| `requirements.txt` | Now the coordinator only (numpy, SUMO 1.27.1). The old full list is `requirements-research.txt`, SUMO pinned to 1.27.1; PPO and LLM extras in `requirements-ppo.txt`, `requirements-llm.txt`. |

Not used on Town05: the PPO policy (`environment/`, `models/`), `SignalSafety` and `SignalController` (the junction
agents keep their own safety rules), bus priority (Town05 buses have no
timetable).

Run (from `agents/priority_agent`, with its venv):

    python -m pytest -q town05/tests
    python -m town05.shadow_check --seed 11 --scale 0.8 --end 1800
    python -m town05.serve --llm --llm-model qwen2.5:3b     # what --goflow-llm starts

In the viewer: `set FREEFLY_ARGS=--goflow-llm qwen2.5:3b` before `tools\freefly_goflow_priority.bat`
(after `tools\setup_goflow_priority.bat llm`, Ollama installed and `ollama pull qwen2.5:3b`).
