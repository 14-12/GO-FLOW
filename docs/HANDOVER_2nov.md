# Handover: Urban Traffic CARLA (2 Nov 2026)

Paste this into a new Claude chat to pick up the work. It covers:
- what the project is and what has been built;
- where everything lives;
- how it was measured;
- what is open.

The previous chat cannot be continued from another account (shared chats are read-only), so this file is the
context.

---

## 1. The project in one paragraph

**Urban Traffic CARLA** is a digital twin of CARLA's **Town05**:
- **SUMO** runs the traffic (signals, about 2900 vehicles an hour at the peak, buses, pedestrians, emergency vehicles);
- **CARLA 0.9.16** renders it;
- the user flies through it with a free-fly viewer (`cosim/freefly.py`).

On top of it:
- junction signal agents fed by traffic cameras with a YOLO26n detector trained on CARLA;
- accidents with a full police and recovery response;
- (since 31 Oct) a red-light enforcement camera with plate reading, a fictional vehicle register and a police office.

The owner is **Hela** (GitHub `14-12`, repository **https://github.com/14-12/GO-FLOW**, branch `main`). Teammates
(including Firas) build the **GO-FLOW accident-detection agent** on top of this project's camera recordings.

**Main rule kept throughout:** agents decide only from what a real system would have:
- the camera pictures;
- the controllers' signal logs.

The simulation's own truth (SUMO) is written beside each recording and used only to grade the agents, never shown to
them. Owners and personal data are always fictional and marked as such.

## 2. Where everything is

| What | Where |
| --- | --- |
| Project on the PC | `D:\Projects\Urban Traffic CARLA` (Windows; CARLA in `carla\CARLA_0.9.16\`, Python in `.venv\`) |
| GitHub | `14-12/GO-FLOW`, `main`. Last commits: `4b0e57b` incidents, `033a793` red-light camera. Working tree clean on 2 Nov. |
| Full documentation | `README.md` (how to run, every option, architecture in section 5; red-light camera is 5.7) |
| Dated work log with all measurements | `docs/WORKLOG.md` (latest sections: 31 Oct incidents; 31 Oct to 1 Nov red-light camera) |
| Prompt for teammates integrating their work | `docs/TEAM_INTEGRATION_PROMPT.md` |
| Tests | `tools\test.bat` (pytest; 213 passing on 1 Nov) |
| Logs | `tools\logs\` (viewer: `freefly_town05.log`) |
| Results | `out\` (not in git): `out\enforcement\` recordings and `cases\`, `out\detector\` detector evaluations |
| Backups made before each deployment | `out\_backup_*` |

**How to run:** double-click `town05_freefly.bat`. Keys: Z Q S D to move, E and A for up and down, H to inspect a
traffic camera, Esc to quit. The README has the rest.

## 3. What was built most recently

### 3.1 More kinds of incident (31 Oct): `cosim/incidents.py`, `incident_scene.py`, `incident_hud.py`

Accidents happen on their own: about 10 % of drivers are careless and sometimes look away for 1.5–3 s (on the phone,
or through a red light). A crash happens only if traffic is there.

Added on 31 Oct:
- **breakdowns** (`--breakdowns`);
- **speeders** (`--speeders`, 8 % of cars);
- **pile-ups on the ring road** (`--pileups`);
- **look-alikes that are not incidents**: double-parked delivery vans, long bus stops.

After a crash:
- **+6 s:** a police car is sent from 250–700 m away;
- **+10 s:** a recovery truck is sent;
- the lane is closed, cones go out, and the wrecks are towed away.

Demos: `tools\freefly_breakdown_demo.bat`, `tools\freefly_pileup_demo.bat`.

### 3.2 Red-light enforcement camera (31 Oct – 1 Nov), README section 5.7

**Site:**
- junction 1574, road -29 (`--enforce 1574:-29`);
- two cameras on one pole, 24 m before the stop line and 7 m up:
  - overview camera: 30°, pitch -17°;
  - plate camera: 17°, aimed at the last metres before the line.

**Files:**

| File | What it does |
| --- | --- |
| `cosim/enforcement.py` | Site, cameras, picture timing on SUMO's clock, signal log, recording |
| `cosim/redlight.py` | Detector, tracker, the rule, plate reading voted over views, evidence |
| `cosim/redlight_proc.py` | Runs the agent live beside the viewer |
| `cosim/anpr.py` | Classic Tunisian plate reading |
| `cosim/register.py` | Fictional vehicle register |
| `cosim/red_runners.py` | Drivers who run the end of the amber |
| `tools\police_office.py` / `.bat` | The officer's window: accept/reject, notice in HTML, SHA-256 integrity check |
| `tools\eval_redlight.py` / `.bat` | Grades a recording against SUMO's record |
| `tools\record_enforcement.bat` | Records 40 min with the live agent on |

**The rule:**
- the front crosses the line with the lane's signal red for at least 0.5 s;
- the vehicle is then seen going on into the junction;
- emergency vehicles with lights on are exempt.

**How plates are read:**
- the plate's panel is outlined and straightened;
- the series sits at the left edge and the number at the right edge;
- digits are matched against the plate font;
- a reading is kept only when its size and position match the cameras' calibration for that vehicle;
- readings are voted over several views;
- the officer confirms the plate against the photo before any notice.

**Measured** (recording 5, 1 Nov, 40 min, both cameras), graded against SUMO's record:

| Measure | Result |
| --- | --- |
| Red-light runs caught | 2 of 2 |
| False accusations | 0 |
| Crossing timing | within about 0.1 s |
| Plates on the two cases | right at confidence 1.00 (259 TU 7712, 255 TU 8086) |
| Plates of all crossing cars (31 found) | 22 right, 1 wrong (unsure, not counted), 8 unread |
| Viewer speed with both cameras | 20.2 FPS |

**Known limit:** a misread digit on the right plate, e.g. a tilted 4199 read as 4491, is not caught by the size and
position checks. It is left to voting across views and to the officer.

**Open idea from Hela:** if a plate can't be read, hand the car over to the next junction's camera.
- It can't pass a YOLO track ID: those exist only within one camera.
- It would pass a description (class, colour, appearance, exit, expected arrival time) and re-identify the car there.
- The officer would confirm the match before any notice.

Not built yet.

## 4. GO-FLOW accident agent: current state

The teammates' agent detects **accidents only, never jams**, from 5 s windows of camera video.

### 4.1 Their final test report (2 Nov): v2 not accepted

| Criterion | Needed | v2 | v1 |
| --- | --- | --- | --- |
| False alarms per camera-hour (normal time, jams included) | ≤ 1 | 0.60 ✅ | 18.81 |
| Clearly visible crashes caught | ≥ 80 % | 10/10 ✅ | 10/10 |
| Median time to detect | ≤ 30 s | 70 s ❌ | 5 s (v1 is in alarm 59 % of normal time) |
| Slowest detection | ≤ 120 s | 140 s ❌ | 15 s |

**Why v2 is slow:** it mostly reacts to stopped police lights, which appear 15–45 s after impact.
- These lights appeared in 91 % of test crash windows and 6.4 % of normal windows.
- All 12 of its false alarms had stopped emergency lights in view.
- The crash moment itself gives only a weak signal.

**Limits of the test:**
- only 10 test crashes;
- 9 of the 10 test crashes, and all 23 training crashes, are on the four ring-road cameras (2014-2, 2014-3, 2328-1,
  2328-2);
- the never-seen cameras had no crash in view.

The test runs are now used up. Their recommendation, which we agree with: keep the 30 s / 120 s limits and find a
stronger crash-moment signal, then test on new runs.

### 4.2 Our analysis: two of v2's problems come from the simulator

1. **The police arrive unrealistically fast.**
   - In `cosim/incidents.py` the police are dispatched at +6 s from 250–700 m away, so they arrive within about
     15–45 s. Real police take minutes.
   - This taught v2 the shortcut "stopped police lights = crash", which also caused its false alarms.
   - **Fix:** a realistic, configurable dispatch delay (minutes, with spread), plus a record of whether police
     reached the scene.
2. **The crash moment is weak on screen ("soft impact").**
   - The teammates saw cars go from driving to standing over 2–3 s, like hard braking. They asked whether the
     viewer's interpolation causes it.
   - **Not answered yet.** What is known:
     - SUMO runs 0.5 s steps;
     - a wreck is frozen with `setSpeedMode(0)` + `setSpeed(0)`, which should stop it at once;
     - the viewer draws motion between SUMO steps along Hermite curves (`cosim/interp.py`);
     - CARLA 0.9.16 can't deform cars, so damage shows as a pose push and turn (`cosim/incident_scene.py`), debris
       and hazard lights.
   - **To do:** measure frame by frame on a recording, comparing SUMO's positions with the camera's picture. Then fix
     any smoothing, and make the impact visible at its instant (push and turn, debris).

### 4.3 How to get enough clearly visible crashes (decision open)

**Their request:** at least 30 crashes clearly visible to a fixed camera:
- the wreck within about 30 m and fully in frame;
- the impact on screen, at least 2 min before and 5 min after;
- a mix of ring-road and junction cameras;
- new seeds;
- split into `train/` (about 2/3) and `test/` (about 1/3), with at least 3 test crashes on cameras unseen in
  training.

At the normal 10 % careless share this needs about 100 runs (about 60 h of PC time).

**Options discussed:**

| Option | What it means |
| --- | --- |
| A (recommended) | Keep 10 % careless town-wide, but make drivers heading into a recorded camera's view more often the careless ones, and have them look away on the approach. Whether a crash happens, and how, stays emergent. Disclose clearly to the team: crash frequency in front of cameras is then not a real-world rate. About 8–15 runs (estimate). |
| B | Raise the careless share town-wide (30–50 %). Simpler, but more crashes and jams everywhere, so the normal traffic is less normal. |
| C | Natural only. More cameras per run, accept fewer than 30 crashes. |

A short test run first would measure the real yield of option A.

Hela has not chosen yet ("let's just talk"), and has not answered the 4 decisions in their section 6.6:
1. their acceptance criteria;
2. v2 as a separate Town05 model, with v1 unchanged;
3. results as one JSON per clip in `out\goflow_exchange\results\`;
4. detections shown and logged only, never acting in the town, until the criteria are met.

**Other items in their request:**
- **detector:** `models\carla-yolo26n-t05.pt` + `.json`, the Ultralytics version, and per-class validation for
  `police_lights`, `ambulance_lights`, `truck`, `car` (precision, recall, mAP50);
- **labels:** `crashes.json` per delivery (fields listed in their request);
- **optional:** per-frame 2D boxes of wreck and emergency vehicles, and 5–10 look-alike events.

The existing evaluation is in `out\detector\eval_town05.md`. It reports per group, not per class, so the per-class
numbers still need computing.

Recording tools: `cosim/clip_recorder.py` (55 layout cameras, 960x540, 10 fps, `<camera>.frames.csv` with `sim_s`,
`incidents.json` with "visible"), `tools\record_incident_clips.bat`, `tools\record_v2_training.bat`,
`tools\record_city_crash_clips.bat`.

## 5. Suggested next steps (in order)

1. **Measure the soft impact** on a recording and fix it if it comes from the pipeline. This is quick and decides how
   much the rest matters.
2. **Make the police response time realistic and configurable.** Log whether police reached each scene, so "no
   police" cases can be tested.
3. **Hela chooses the crash-source option** (A, B or C) and answers the 4 decisions.
4. **Produce the new train/test delivery** to `out\goflow_exchange\stage2_<date>\` with a README:
   - the detector items;
   - the labels and `crashes.json`;
   - junction crashes and unseen cameras;
   - the disclosure of the crash-source method.
5. **Optional:** the plate handover to the next junction.

## 6. How the previous sessions worked (useful habits)

- **Cloud Claude and the PC:** a cloud Claude session reached the PC through the Claude desktop app's link. Through it
  it could use files, a shell in a Linux VM over the project folder, and screenshots and clicks.
  - Code was developed and tested in the cloud.
  - It was then copied to the PC after a backup to `out\_backup_<time>`, and each copy was checked by md5.
- **CARLA runs:** these need the Windows PC. They were started by double-clicking `.bat` files in `tools\`.
- **Honesty about measurements:** every number in the README and WORKLOG comes from a recording graded against
  SUMO's record. Failures are written down, not hidden.
- **Pushing:** Hela pushes to GitHub herself. Claude gives the commands; PowerShell on the PC.
- **Style used in the docs:** plain English, short sentences, measured numbers with their source and date.
