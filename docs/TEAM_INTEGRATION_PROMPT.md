# Prompt: integrating your GO-FLOW work into Urban Traffic CARLA (Town05)

> **How to use this.** Paste everything below the line as the first message of your Claude session. Run that
> session with the GO-FLOW repository open: `main`, which now holds the Town05 project, plus your own branch.
> Claude will study both and come back with an integration plan. It will not change any code until you agree.

---

You are helping me integrate my work from the GO-FLOW repository (my branch: **`<write your branch name here>`**)
into **Urban Traffic CARLA**, our team's city traffic simulation. That project now sits on GO-FLOW's `main` branch.
The project owner wrote this brief. It explains the project, the data and hooks it offers, the rules of the
shared code, and what I need from you.

## 1. Your first job: understand, then propose (no code yet)

1. **Read the Town05 project on `main`.**
   - Start with `README.md`, especially section 5, "Architecture".
   - Then read the modules this brief names (sections 4 and 5).
   - `docs/WORKLOG.md` explains why things are the way they are. Read only the parts you need.
2. **Read my branch fully.** That means:
   - what my agent or model takes as input and what it outputs;
   - its time base and how often it runs;
   - its dependencies and model files;
   - how it was trained and evaluated;
   - its known weaknesses.
   If the branch has a handoff report or notebooks, read them.
3. **Write an integration plan** (section 9) and stop. I will review it with the project owner before any code is
   written.

Never guess about either codebase: open the file. If something this brief says does not match the code, the code
wins; tell me where they disagree.

## 2. The project in brief

**Urban Traffic CARLA** is a digital twin of a small city: CARLA's **Town05**.

- **The stack:**
  - SUMO 1.27.1 simulates every vehicle and pedestrian;
  - CARLA 0.9.16 draws them in 3D;
  - Python 3.11, on Windows 10/11 with an NVIDIA GPU (4 GB is enough).
- **SUMO is the master.** It runs the town at a 0.5 s step with its own demand and signal control. CARLA only draws
  what SUMO decided, so every picture shows a state the simulation really passed through.
- **The viewer.** You start it with `town05_freefly.bat` (it runs `python -m cosim.freefly`) and fly over the town
  while it runs.
- **The town:**
  - 15 signalised junctions, with coordinated signals on one 115 s cycle and green waves;
  - junction AI agents;
  - 55 fixed traffic cameras with a YOLO detector;
  - careless drivers who really crash, followed by police, lane closure and a recovery truck;
  - three bus lines, pedestrians on signalised crosswalks, a raised ring road;
  - ambulances and police on emergency runs, with signal pre-emption;
  - a day and night clock (1 real minute = 1 town hour) that drives the demand;
  - an LLM brief.
- **Main folders:**
  - `cosim/`: the co-simulation, a Python package;
  - `scenarios/town05/`: the SUMO network, demand and signal sets;
  - `tools/`: double-click `.bat` launchers and Python tools; logs go to `tools\logs\`;
  - `models/`: the detector weights;
  - `out/town05_cameras/`: the camera layout;
  - `tests/`: pytest, about 194 tests, run with `tools\test.bat`.

## 3. Architecture you must respect

| Part | Code | What it does |
| --- | --- | --- |
| Viewer main loop | `cosim/freefly.py` (`FreeFly`) | Interpolates vehicles between SUMO steps, spawns or parks CARLA actors, sets lights and weather, draws the overlay. Must never block. |
| SUMO thread | `freefly.py` (`SumoThread` → `FreeFly.produce()`), `cosim/sumo_side.py` (`SumoSide.step()`) | One SUMO step at a time, one or two steps ahead of the picture (a queue of 2). Runs the controller, demand, buses, accidents and emergency runs. |
| Helper processes | `pose_proc.py`, `fly_proc.py`, `plate_proc.py`, `ground_proc.py`, `inspector.py` | Anything slow lives in its own process. Example: the YOLO detector loads PyTorch only in its own process, and if loading fails it reports an error and the viewer carries on. |
| Signals | `signal_plan.py`, `coordination.py` (built offline); `controllers.py`, `CycleKeeper` in `sumo_side.py` (at run time) | Signal sets in `scenarios/town05/`; the controller is chosen with `--controller` |
| Junction agents | `cosim/agents_actuated.py` | One agent per junction on top of SUMO's actuated program: skips empty phases, ends wasted greens, balances queues, gives emergency vehicles priority, answers pedestrian calls |
| Cameras | `perception.py`, `camera_reports.py`, `inspector.py`, `cctv.py`, `out/town05_cameras/` | 55 cameras, 9 m up, pitch -22°, 960x540, 75° FOV. Each has a lane mask. |
| Accidents | `incidents.py`, `incident_scene.py`, `towing.py`, `depots.py` | Real collisions from distracted drivers, then the full response |
| Overlay | `overlay.py`, `hud.py`, `incident_hud.py`, `agent_hud.py` | What is drawn over the CARLA picture |
| Video recording | `cosim/clip_recorder.py`, `--record-*` options of `freefly.py` | MP4s from chosen cameras, with accidents labelled on the simulation clock |
| LLM | `cosim/llm.py`, `tools/town05_brief.py` | Free-tier LLM calls with provider failover; keys in `.env` (git-ignored) |

The rule this design follows: **nothing slow sits on the path to the next picture or the next SUMO step.** If
your work is slow, it belongs in its own process or in an offline run, connected by a queue, files or JSON lines.

## 4. Data the town can give your work

All times are **SUMO seconds** (`traci.simulation.getTime()`), counted from the start of a run. The town's clock
(hour of day) comes from `cosim/daily.py`.

| Data | Where | Shape |
| --- | --- | --- |
| Every vehicle, every step | `SumoSide.step()` → `StepResult` (`cosim/sumo_side.py`) | `vehicles: {id: Vehicle(id, vclass, type_id, length, x, y, angle, speed, signals)}`, `arrived`, `tls: {junction: state string}`, `tls_next`, `persons` |
| Anything in SUMO, live | the TraCI connection | Lane, edge and detector counts, speeds, occupancy, routes, signal programs. A callback on `SumoSide.pre_policy` gets the connection after each step, before the controller. It is a single slot, already used by the camera feed in `agents_observed` mode, so chain to it rather than replace it. |
| Accidents, ground truth | `IncidentManager.snapshot()` (`cosim/incidents.py`), also in each step's `Packet.incidents` | Per incident: `id, phase, kind` (rear-end / side impact), `label, severity` (impact speed), `t0, x, y, edge, lane, closed` lanes, `police`, `wrecks` (positions, damage, removed), `tows`, `events` |
| Camera reports (what the cameras "see") | `SimulatedCameras.step(tc, t, emergency)` → list of reports every 2 s (`cosim/camera_reports.py`) | Per camera: `cam, tls, t, in, q_in, out, parked, other_road, stopped_long` (vehicles still over 90 s), `ems` (sirens), `lanes: {lane: [count, queued]}`. They are simulated with the real detector's measured error rates, so they are realistic rather than perfect. |
| Camera geometry | `out/town05_cameras/layout.json`, `masks.npz` | Name (e.g. `"1427-1 looking N"`), junction, road, CARLA pose, size, FOV, lane mask per camera |
| Real camera video | `cosim/clip_recorder.py` with `--record-cams`, `--record-fps`, `--record-clips`… | One MP4 per camera (960x540, 10 fps, H.264), a `frames.csv` mapping each frame to its SUMO second, and `incidents.json` saying which accident is visible in which camera and when |
| Detections on real pictures | `models/carla-yolo26n-t05.pt` (YOLO26n, AGPL-3.0) | Classes `0 car, 1 van, 2 bus, 3 truck, 4 ambulance, 5 ambulance_lights, 6 police, 7 police_lights, 8 pedestrian`; imgsz 960; no motorcycle class |
| Long SUMO-only runs (no CARLA, fast) | `tools/eval_town05_*.bat`, `cosim/run.py` | Many seeds and demand levels (x0.5 / x0.8 / x1.0), for training data and fair evaluation |

**Rendering lag.** The pictures show the simulation about 2.2 s late. An event at SUMO second t is on screen at
about t + 2.2.

## 5. Where your work can plug in

Pick the hooks that fit what my branch actually does.

1. **Read live state.** Add a small observer called from `FreeFly.produce()` or chained on
   `SumoSide.pre_policy`. It passes the current state (vehicles, signals, incidents, camera reports) to your code
   through a queue. Your code runs in its own process and answers asynchronously; the SUMO thread never waits for
   it.
2. **Act on the town.** Every action path already exists:
   - **Signals.** A per-step controller has the interface `additional_files()`, `setup(conn)`, `step(conn)`. It is
     registered in `cosim/controllers.py` (`CONTROLLERS` / `make`) and chosen with `--controller`. Alternatively,
     feed the junction agents through `ActuatedAgents.observe(reports)`.
   - **Incidents.** The response (police, lane closure, towing) is driven by `cosim/incidents.py`. A detection
     coming from outside should be *confirmed* before it acts. Example: your alarm plus a camera report showing a
     vehicle still for more than 30 s in a lane whose light is green.
   - **Display.** Show results on the overlay through a small HUD class, like `agent_hud.py` or
     `incident_hud.py`.
   - **LLM brief.** Your outputs can feed the brief (`tools/town05_brief.py`).
3. **Offline.** Record labelled video or long SUMO-only runs, train and evaluate on them in your own environment,
   then bring back a model file and an adapter.

Recommended shape:
- your code under **`agents/<your_agent>/`**, with its own `requirements.txt`;
- a thin adapter in **`cosim/`** (for example `cosim/goflow_<your_agent>.py`) that starts your code as a separate
  process, using **your own Python environment**, and talks to it in JSON lines;
- a viewer flag to turn it on (for example `--goflow-<your_agent>`), **off by default**;
- errors are logged and shown, never fatal: the town keeps running if your process dies.

## 6. Notes per team branch

Use the part that matches my branch. If my branch is none of these, apply the same ideas.

- **Incident detection agent** (`feature/incident-detection-agent`, `feature/incident-carla-integration`). It takes
  an MP4 in and gives a verdict per 5 s window. We already agreed on:
  - a plan in five stages;
  - an interface (`source_id` = our camera name, absolute time in SUMO seconds, optional evidence of stopped
    tracks, errors reported as errors);
  - an evaluation per event, with time to detect and false alarms per camera-hour.

  Stage-1 clips and the Town05 detector were delivered. The recordings are no longer in the project folder, but
  `tools/record_incident_clips.bat`, `record_city_crash_clips.bat` and `record_v2_training.bat` make new ones the
  same way. Continue from the exchange documents (`INTEGRATION_RESPONSE_STAGE2.md`, `STAGE1_RESULTS_v1.md`, if
  they are in the branch) rather than starting over.
  - Ground truth comes from `IncidentManager.snapshot()` and from the recorder's per-camera visibility.
  - The live goal: watch 2 to 4 cameras. A confirmed alarm goes to the response chain.
  - Never fill severity from the simulator at run time.
- **Traffic prediction agent** (`feature/traffic-prediction-agent`).
  - **Inputs:** from SUMO (per lane or edge: counts, speed, occupancy, queue) or, more realistically, from the
    camera reports.
  - **Ground truth:** SUMO's own future state, available exactly.
  - **Training data:** SUMO-only runs over many seeds, demand levels and times of day.
  - **Evaluation:** on held-out seeds, against simple baselines (last value, time-of-day average). Report the error
    per horizon.
  - **Live uses:** show forecasts on the overlay; give them to the junction agents or the coordination (longer
    greens before a predicted queue); feed the LLM brief.
- **Incident → traffic prediction** (`integration/incident-to-traffic-prediction`). Incidents (lanes closed, phase,
  position) become inputs to the prediction of congestion around them. The simulation gives both the incident and
  the congestion it really caused, with as many accidents as needed: data runs can raise `--careless-share`.

## 7. How my work fits the Town05 project: strengths and weaknesses (points forts et points faibles)

The question here is not "is my work good" or "is the project good" on its own. It is **how well the two fit
together**:
- **Points forts:** where they already match or complete each other.
- **Points faibles:** where they do not fit yet (mismatches, gaps, conflicts).
- **Improvements:** what to change, on which side, so that the integration works, and how to check it.

### 7.1 What to compare

Go through each line for my branch:

| Aspect | Question |
| --- | --- |
| Input data | Does the town produce what my work needs (format, classes, units, rate, camera view)? Exactly, approximately, or not at all? |
| Output | Can the town use what my work produces: show it, act on it, evaluate it? Is there a hook for it (section 5)? |
| Time | My time base (video seconds, wall clock, windows of N s) against SUMO seconds, the 0.5 s step, 2 s camera reports and the 2.2 s render lag |
| Space | My notion of place (image, camera, road, lane, zone) against the town's (camera names, lanes, junctions, CARLA x/y) |
| Training domain | What my model was trained on (real video, highway, another city, another simulator) against Town05 (simulated city junctions and ring road) |
| Evaluation | Can my metrics be computed with the town's ground truth? Does the town allow a fairer test (more events, held-out seeds and cameras)? |
| Speed and hardware | Can it run live next to CARLA on a 4 GB GPU, or only offline on recordings? |
| Software | Python version, library conflicts with the project's `.venv` (CARLA client, SUMO, PyTorch 2.6 / CUDA 12.4), model file formats |
| Purpose | Does it serve the project's goal: multi-agent urban traffic management (signals, incidents, emergency response)? |

### 7.2 The incident detection agent: the fit as the project owner sees it

The project owner wrote this from the earlier review of the handoff report and notebooks. Check each line against
the current code and update it.

**Points forts (where they fit):**
- **The town supplies the agent's missing data.** Its weakest point is too few labelled incidents (the final test
  rests on 3). The town produces real collisions in any number, labelled to the second, with type, severity and
  lane, and with whether each wreck is visible in each camera.
- **Same kind of sensor.** Both work on fixed traffic cameras with a YOLO detector and tracking. The town's cameras
  are calibrated (height 9 m, pitch -22°, 75° FOV) and have lane masks, so the agent can get real speeds and
  per-lane counts.
- **Matching time scales.** The agent decides per 5 s window and the town's camera reports come every 2 s, so a
  window can be joined to both.
- **The town has the hard negatives the agent needs:** red-light queues, jams, buses at stops, parked cars.
- **The town can act on a detection:** police, lane closure, signals, overlay marker, LLM brief. The agent becomes
  part of a working system instead of a stand-alone classifier.
- **Signal state gives a strong extra clue.** A vehicle that stays still while its light is green is a likely
  incident.

**Points faibles (where they do not fit yet):**
- **Domain gap.** The agent was trained on real highway video (AI City Track 4). Town05 is simulated city traffic:
  junctions where stopping at red is normal. The two must not be mixed blindly, and results are reported apart.
- **Detector mismatch.** The agent uses COCO YOLO11n (classes 2, 3, 5, 7), which often misses small, distant
  vehicles. The town's detector is CARLA-trained YOLO26n with other classes, including police and ambulance.
- **Interface mismatch:**
  - the agent reads video files, while the town needs live frames;
  - its time is video-relative, while the town uses SUMO seconds;
  - it has no camera identity;
  - it gives no evidence of where the stopped vehicle is.
- **Its features depend on the camera.** Raw counts mean different things in different views; the town has 55
  different views.
- **Evaluation does not match the operational need.** The agent is judged per window; the town needs incidents
  caught, time to detect, and false alarms per camera-hour.
- **Software conflicts.** The agent pins scikit-learn and pulls in LangChain and XGBoost, which would change the
  project's working `.venv`.
- **GPU sharing.** CARLA leaves little GPU, so the agent can watch only 2 to 4 cameras live.
- **Severity.** The agent reports UNKNOWN. The town knows severity, but must never fill it in at run time.

**How to improve the fit:** use the five agreed stages and changes:
- run v1 unchanged on Town05 clips;
- honest baselines;
- a Town05 dataset;
- v2 trained on both datasets but tested apart;
- live use with a confirmation rule.

The changes on the agent's side: `process_frames`, a detector config, absolute time, an evidence output, features
relative to each camera, a background model for stalled vehicles, calibrated scores, per-event metrics and a lean
inference environment.

### 7.3 Other branches (traffic prediction, incident → prediction): what to check

- **Inputs.** Does it need exact counts and speeds (SUMO gives them) or realistic sensor data (the camera
  reports, noisier, every 2 s, only on the cameras' lanes)? A model trained on exact data may fail on camera
  data.
- **Places.** Does it predict per road, lane, junction or zone? Map its places to SUMO edges and lanes and to the
  15 signalised junctions.
- **Time.** Its horizon and step against the 0.5 s SUMO step, the 115 s signal cycle and the clock (1 real minute
  = 1 town hour). Demand changes fast in real seconds.
- **Domain.** What was it trained on? Real traffic data or another network transfers poorly to Town05's small grid
  and ring road. Plan to retrain on SUMO-only runs: many seeds, demand levels and times of day.
- **Use.** Who consumes the prediction? The junction agents or the coordination (signal timing), the overlay, or
  the LLM brief. A prediction nobody uses is a weak fit.
- **Incidents.** Does it cope with lane closures and the queues they cause? The town can generate as many as
  needed.
- **Evaluation.** Against SUMO's future state on held-out seeds, and against simple baselines (last value,
  time-of-day average), with the error reported per horizon.

### 7.4 What to write

A table with one row per aspect: **point fort**, **point faible**, **improvement** (and on which side: my code,
the project, or an adapter), and **the check** that proves the improvement works. Put the rows in order of impact
on the integration.

## 8. Rules of the shared project (please follow them)

- **Separate Python environments.** Never install your requirements into the project's `.venv`, which holds the
  CARLA client, SUMO and PyTorch 2.6 / CUDA 12.4. Your code runs in its own venv as a separate process.
- **Don't break what works.**
  - `tools\test.bat` must still pass.
  - `town05_freefly.bat` must still run with your feature **off**.
  - Keep changes to existing files small, and list each one in your plan.
- **Hands off the camera controls:** `cosim/fly_proc.py` and `tools/carla_config/DefaultInput.ini`.
- **The 4 GB GPU limit.** CARLA uses most of the GPU. Heavy processing runs on recorded clips with CARLA closed;
  live, watch only a few cameras.
- **Results are never overwritten or deleted.** Each run writes a new folder under `out/`, which git ignores. No
  video, datasets or large files in git.
- **Never commit secrets.** API keys go only in `.env`.
- **No number without a run.** Every result names its run, seed and model version. Don't call anything "validated"
  unless you ran it.
- **Windows details:** paths contain spaces, so quote them. `.bat` launchers are double-clicked and log to
  `tools\logs\`.
- **Licences.** The detector weights are AGPL-3.0; CARLA assets are CC-BY. Keep the attributions.
- **Git.** Never push to `main` directly.
  - Branch from `main`, e.g. `integration/<your_agent>-town05`.
  - Bring in your agent's code from your branch.
  - Open a pull request to `main` for review.

## 9. What I need from you: the integration plan

Write it as a document with these sections, then stop and wait:

1. **My work in one page:** inputs, outputs, time base, rate, dependencies, model files, how it was trained and
   evaluated, and its limits.
2. **How my work fits the project: points forts / points faibles**, from section 7: the table of section 7.4,
   improvements in order of impact on the integration.
3. **Mapping to the town:**
   - which data from section 4 it uses;
   - which hooks from section 5;
   - what it produces back into the town.
4. **Adapter design:**
   - process layout and the message format, with field names;
   - timing, and what happens when my process is slow or crashes;
   - the flag that turns it on.
5. **Files:** new files, and every change to an existing file, each with its reason.
6. **Data and evaluation:**
   - what to record or simulate, and the labels;
   - train/test splits, held out by seed and camera;
   - metrics and baselines;
   - real vs simulated data kept apart.
7. **Steps with a check after each one**, starting with the smallest end-to-end test: my code running unchanged on
   Town05 data.
8. **Risks and open questions** for the project owner.

Keep it concrete: file names, function names, field names, numbers. If anything in my branch is unclear or
missing, list it as a question instead of guessing.
