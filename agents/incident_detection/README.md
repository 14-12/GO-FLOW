# incident_detection: Omar's accident agent in the Town05 viewer (shadow)

From GO-FLOW branch `feature/incident-carla-integration`, commit `03a7010` (7 Nov 2026, author 3moreeeee). The audit,
the decisions and the measurements are in `docs/INCIDENT_AGENT_AUDIT.md`.

**It is his work.** The files below are his, byte for byte (blob hashes equal to `03a7010`'s), and are not edited
here. What is ours is the adapter around them.

| His (unchanged) | What it is |
| --- | --- |
| `src/` (`detector_config.py`, `feature_engineering.py`, `inference.py`, `video_processing.py`) | the agent |
| `models/incident_detector.joblib` | v1: Random Forest, 27 features, mean3 >= 0.35 (trained on AI City, COCO YOLO11n) |
| `config/detector_carla_yolo26n_t05.json`, `config/detector_v1.json` | detector settings and class maps |
| `evaluation/crash_features.py`, `evaluation/track_features.py` | his crash-moment signal (hard stops with contact) |

| Ours | What it does |
| --- | --- |
| `stream.py` | `StreamingWindows`: his functions, unchanged, on a live camera; a window out as soon as it closes, identical to his batch path on the whole stream (`tests/test_stream.py`) |
| `serve.py` | the process: JSON lines in (rows per picture), JSON lines out (a verdict per camera and window) |
| `replay.py` | recorded rows, his batch way and streamed, compared |
| `requirements.txt` | inference only, pinned (his file pulled LangChain, XGBoost, torch) |
| `tests/` | streaming = batch; our rows = his `result_rows`; the protocol |

**Status: shadow.** No model meets the agreed criteria on Town05 (his reports; ours in the audit), so its verdicts
are shown and logged and never reach the police, the closures or the route alerts.

Setup: `tools\setup_incident_agent.bat` (Python 3.11 venv in `.venv`, then the tests). Use:
`python -m cosim.freefly --incident-agent "2014-2 looking W;2014-3 looking E"`.

Licence: the GO-FLOW repository has no licence file; this is our own team's code, in our own repository.
