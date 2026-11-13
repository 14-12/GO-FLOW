# Work log: Town05

The project's dated notes: what was built, why, and what was measured, oldest topics first within each
section. The README is the entry point (setup, use, architecture); this file is the history behind it. File names and
numbers are as they were on the day; some tools named here were later removed with the other towns.

## Town05 free-fly: fly the town yourself, 30+ FPS (`town05_freefly.bat`, `cosim\freefly.py`)

Town05 is a mid-size CARLA town (15 signalised junctions, multi-lane grid, a raised ring road) that a laptop
with a GTX 1650 Max-Q draws fast enough to fly around in.

**Why it is fast.** Instead of rendering through a camera sensor in synchronous mode and copying every picture
to Python, CARLA runs asynchronously and draws straight to its own window. You fly with
CARLA's own controls (hold the right mouse button to look, Z Q S D to move, A / E down / up: AZERTY, set in carla/CARLA_0.9.16/CarlaUE4/Config/DefaultInput.ini, original kept as DefaultInput.qwerty.ini). Since 29 Oct, Shift held with Z Q S D / A / E flies twice as fast (40 m/s; `cosim/fly_proc.py` BOOST). SUMO runs in real
time beside it; the vehicles near you are moved 30 times a second along the Hermite curve between two SUMO
states, the traffic lights follow SUMO, and the clock (1 real minute = 1 game hour) is drawn over the
top-left corner of the CARLA picture (`cosim\overlay.py`: a see-through, click-through window that follows the
CARLA window and hides when another program is in front; an in-engine HUD would need the Unreal Editor).
Its keys are pressed while flying in CARLA: F5 pause, F6 / F7 one hour back / forward, F8 next junction,
F9 hide the clock, Esc quit. The arrows turn the view.

**Cars tipping over (fixed).** CARLA gives waypoint pitch wrapped: 360, 359.4, -360, 366.6 for roads within
a few degrees of level. The pitch of each car is blended between two SUMO steps, and a blend from 0 to 360
passed through 180: the car turned upside down for a moment. `CarlaSide.ground` now brings the pitch into
[-180, 180): on Town05's lanes it then stays within +-7.2 deg, and the free-fly demo counts 0 poses over
30 deg (it counts them, `steep_poses`). The fix applies to every scenario.

**Shaking in turns (fixed).** The body faced along the Hermite curve between two SUMO states; in turns and
lane changes SUMO's heading and the curve disagree and the car rocked left and right. Measured on 300 s of
Town05 traffic at 30 poses a second (`Poser`): the turn reversed 2827 times in 210 of 252 vehicles, with 271
jumps over 15 deg between two poses. The free-fly viewer now turns the body evenly from SUMO's heading at one
step to the next (`Poser(heading="turn")`): 127 reversals and no jump offline, 5 in the 70 s demo. Lane changes
take 2 s of sideways motion (`--lanechange.duration 2`) instead of one step's jump (193 in those 300 s).
The films keep the tangent heading.

**Cars vanishing at junctions (fixed).** SUMO ends a trip at the end of an edge, usually a stop line, so a car
disappeared in front of the camera. Now a vehicle in its last three edges gets a new destination
(`changeTarget`) when it is in view, or when the town has no more vehicles than the day's curve asks for; the
curve is followed by taking vehicles off only out of sight, and new trips start where the camera does not
look. Demo: 1 vehicle ended its trip in view in 70 s (4 before the check ran every second).

**Emergency vehicles.** `cosim/extras.py` (the main project's), ambulances and police every ~45 s
(`--emergency-every`), 60 % on a run with beacons on; no pedestrians (walkers cost frames).

**Public transport.** Three bus lines (`tools/build_town05_pt.py` writes `scenarios/town05/town05.pt.add.xml`
and `.pt.json`): closed loops on town streets (not the ring road), L1 west-east 1013 m with 4 stops, L2
north-south 556 m with 2, L3 south-west - north-east 1020 m with 4. Two buses a line from 06:00 to 22:00, one at
night; each stops 20 s at every stop and starts its next lap at the end of the loop (`BusLines`). CARLA's bus
shelter (`static.prop.busstop`) stands on the pavement at each stop. The random bus trips of the demand are left
out. SUMO alone, 15 min: the six buses kept running their laps, no vehicle stopped over 2 min.

**Smoother motion (SUMO in its own thread).** The main loop used to run each SUMO step itself (TraCI, the
population, trips, bus lines), so every half second the vehicles stood still for that long and then jumped.
SUMO now runs in a thread one or two steps ahead (`SumoThread`, a queue of 2) and the main loop only
interpolates and moves actors, 60 times a second: in the demo the median gap between two updates is 16.6 ms,
the 99th percentile 50 ms (was 33 ms and 250 ms), the turn reversed 1-3 times in 70 s, SUMO was never late.
New vehicles come from a pool of parked ones made while loading (a spawn while the picture ran stalled
everything: the first police car 12.6 s, buses and ambulances up to 98 s); more are spawned in a thread.

**Emergency vehicles on a run, all of them**: beacons on (CARLA's `Special1` light), 1.3x the limit, through
red at up to 20 km/h, and signal pre-emption: within 120 m of its next signal, a red for it cuts the current
green to 1 s so the signal's own program moves on through its yellow to the phase that serves it; a green
for it is held until it is through (at most 15 s). 31-35 interventions in a 70 s demo.

**Pedestrians** (`cosim/extras.py`, 8 per km of pavement, about 56 in Town05): CARLA walkers from a pool of
40 made while loading, physics off (the pose comes from SUMO; with physics the parked ones kept falling and
the frame rate was 24 instead of 32), drawn within 150 m of you. A walker is far costlier to draw than a car.

**Traffic cameras with YOLO26n** (`cosim/inspector.py`): 53 cameras, one per road at each of the 15
signalised junctions, the "mast" layout measured best on Ingolstadt7 (10 m into the junction, 9 m up,
looking down the road). They show as blue boxes on dark poles (blue, so they are not taken for traffic lights). Within 25 m of one, "H Inspect"
appears; H puts a 960x540 camera at its pose and runs the CARLA-trained detector (`models/carla-yolo26n.pt`:
car, van, truck, bus, pedestrian, ambulance and police with and without lights) on its pictures, in a process
of its own (in this one, loading PyTorch took the main loop from 60 to 6 updates a second). The picture with
the boxes and the counts (vehicles, public transport, pedestrians, emergency with siren on / off) appears in
the bottom-right corner; H closes it. The detector sees only what the camera's picture shows: a car behind a
building is not in the picture, so it is not counted. About 12-15 detections a second on the GPU. Its known
weak spots (README, detector section) apply: some false emergency boxes on Town05's facades were seen.

**Full screen and the overlay.** CARLA is made borderless full screen on its monitor (`--windowed` / `set
CARLA_RES=...` keeps a window) with the profile `medium_fs` (58 % resolution, upscaled). The overlay covers
the whole picture: clock top left (the dial names the time of day: dawn, sunrise, morning, midday, afternoon,
evening, sunset, dusk, night, midnight, late night; the pill names the traffic), the camera prompt and the
inspector. Only the parts that changed are sent to it. Arrow keys turn the view (left / right, up / down);
CARLA's own spectator does nothing with them (measured).

**Measured, full screen 1920x1080, evening rush** (`tools\town05_freefly_demo.bat`): 119 vehicles, 25
pedestrians drawn, 3 bus lines, emergency vehicles, 53 cameras: 33.3 FPS flying high, 34.7 at street level,
1 % low 26.1 / 28.4. Without pedestrians: 34.2 / 52.7.

**Render profile "medium"** (`tools\fps_test\engine_medium.ini`): CARLA's Epic quality with shadows kept
but cheaper (1024 maps, 2 cascades), no motion blur, screen-space reflections, depth of field or distance-field
AO, 70 % view and 1.5x LOD distance scale, half the foliage, 75 % resolution upscaled (TAAU), texture pool
1200 MB. It is a user Engine.ini put in place only while CARLA starts (`tools\town05_freefly.bat`).

**Measured on this laptop** (`tools\town05_fps_test.bat`, `town05_fps_dx12.bat`; CARLA window 1280x720,
asynchronous, 60 parked cars, 30 s circling 60 m up + 30 s at street level; `out\town05_fps\summary.md`):

| config | FPS high | FPS street | 1 % low | GPU memory |
|---|---|---|---|---|
| Epic, DirectX 12 (CARLA as shipped) | 46.1 | 72.0 | 37.4 | 3100 MB |
| Epic + texture pool 1200 MB | 46.4 | 71.7 | 37.2 | 2889 MB |
| **medium, DirectX 12** | **50.0** | **111.5** | **41.1** | 2666 MB |
| Epic, DirectX 11 + pool | 31.0 | 65.8 | 25.5 | 2188 MB |
| medium, DirectX 11 | 33.6 | 81.2 | 28.0 | 2036 MB |
| Low, DirectX 11 | crashed ("Fatal error") | | | |

DirectX 11 was slower: it left the GPU 40-60 % idle. **With the traffic** (`tools\town05_freefly_demo.bat`,
evening rush 17:30, 101-117 vehicles all drawn, 10 s warm-up, then 30 s high + 30 s street;
`out\town05_freefly_demo.json`): **41.3 FPS high, 57.5 at street level, 1 % low 32.8 / 40.2, worst frame
62 ms**. Two things it took: vehicles that leave are parked out of sight and reused rather than destroyed
and spawned again (a spawn stalls the server), and the Python side runs at below-normal priority at 30
updates a second (60 updates a second cost street-level FPS: 54.5 against 61.0).

**The scenario** (`scenarios\town05`, `tools\build_town05.bat`): built like Town10 and Town12 from the map's
OpenDRIVE; 56 of 256 movements put under a signal (`tlcomplete.py`). Demand scale 1.1 (about 2900 vehicles an
hour at the peak): SUMO alone, one hour, no teleporting, 1.4 gridlocks, 1.2 starts to (9 vehicles stopped over
2 min under actuated), 1.1 flows under actuated and Webster and with SUMO seeds 1-3 (2900-2907 trips, 0 stopped).
The day's vehicle curve uses Little's law with Town05's mean trip (138 s): at most ~115 vehicles at 17:00.
This build was made on Linux (same SUMO 1.27.1); `build_town05.bat` makes it again on Windows.

**Raised ring road.** SUMO's network is flat, and looking up the road height from z = 0 puts a car on the road
under a bridge: 58 of 3164 lane segments (the ring road, 10 m up) came out 10 m too low. The free-fly viewer
looks on every 5 m level between the lowest and highest road and keeps the lane nearest in plan that runs the
vehicle's way (`CarlaSide.use_road_levels`); the films and datasets keep the plain lookup.

## Free-fly work log (19-22 Oct)

### Traffic-camera detector retrained on Town05 (19 Oct)

The inspector (H) now uses `models/carla-yolo26n-t05.pt`, fine-tuned from `carla-yolo26n.pt` on 3,807 Town05
pictures taken through the inspector's own 53 cameras, plus each camera moved a few metres/degrees
(`tools/make_dataset_town05.bat`, 14 min). Labels come from CARLA's instance camera: a box covers only the
visible pixels, so hidden vehicles are never labelled. Sunrise to night, clear, cloudy and rain; 60 % of the
emergency vehicles on a run (siren on). Split by junction: 3 signals (10 cameras) test only, 2 val.
Training: `tools/train_detector_town05.bat`, 40 epochs at 960 px, 2 h 54 min on the GTX 1650.

Held-out junctions (711 pictures, inspector threshold 0.35, `out/detector/eval_town05.md`):

| model | mAP50 | vehicles found | buses | pedestrians | siren on | siren off | false boxes /100 pictures | false emergency boxes /100 |
|---|---|---|---|---|---|---|---|---|
| carla-yolo26n (before) | 0.218 | 51 % | 18 % | 22 % | 26 % | 9 % | 63.3 | 2.4 |
| carla-yolo26n-t05 | 0.496 | 64 % | 45 % | 44 % | 34 % | 27 % | 23.4 | 0.0 |

"Found" counts every labelled object, down to a few pixels far down the road. Ordinary cars boxed as
emergency vehicles: 161 before, 21 now. Still weak: siren on against siren off (109 mix-ups) and emergency
vehicles boxed as plain cars (173).

### Crosswalks with pedestrian signals (19 Oct)

Town05's SUMO network had sidewalks but no crossings, so pedestrians walked up and down one sidewalk.
`tools/build_town05_peds.py` (run once; originals kept as `town05.noped.*`) adds walking areas at every
junction and 55 crossings at the 15 signalised junctions, where CARLA paints its crosswalks (median 1.8 m
apart, 47 of 55 within 3 m). Each signal program gets a pedestrian phase at the end of the cycle: every car
red, every crossing walk 10 s, then 5 s all red. `cosim/sumo_side.WalkSkip` skips it when nobody waits there.
Pedestrians now walk routed trips (100-500 m) across these crossings and wait at red; one you can see who
reaches the end of a trip walks on to a new one. Free-fly: 20 pedestrians per km of pavement (140 in town,
was 56), the nearest 28 within 110 m drawn. People waiting for the walk signal are drawn on CARLA's pavement:
SUMO's walking areas reach past CARLA's kerbs, so they used to stand on the road's edge (SUMO alone: waiting
or walking off-crossing points on a shoulder or driving lane 868 -> 82 of 5 285 samples). The drawn walk/wait
lamps (`cosim/ped_lights.py`) were off at first: CARLA draws debug shapes as white squares in camera pictures.
They are on since 22 Oct (see below).
Camera markers are no longer drawn while a camera is open (they sat in front of its lens and filled the live
picture with green or red).

Why not crossings alongside the cars (walk while the parallel road is green, turning cars yielding): with
SUMO alone that locked the town within 20 minutes, cars stopped inside junctions behind walkers. The
exclusive phase, skipped when empty, keeps the viewer's evening rush flowing (SUMO alone, 118 vehicles:
mean speed 3.5 m/s against 4.1 without pedestrians, no vehicle stuck at the end). At the scenario's own
1-hour design demand (scale 1.1, about 140 vehicles) it still locks; that demand was already at the limit.

Checked: every one of 1598 crossing entries over 25 min started on walk; median crossing 11.5 s.
`tools/probe_crosswalk.bat` renders pedestrians on crossings. Demo (evening rush, full screen): 28.8-29.6 FPS
average, 1 % low 19-22 (before this change 33-35); the extra pedestrians are most of the cost.

### Smoother turns (19 Oct, second pass)

Cars still shook in turns with the even turn: its turning rate jumped at every SUMO step (0.5 s) and the
body slid sideways against its own motion. Now (`cosim/run.Poser` heading "trail", `cosim/interp.Trails`) the
front of each car follows a Catmull-Rom curve through SUMO's positions (the viewer runs one SUMO step behind
for the point after) and the body points along the path its front has taken, 0.7 x its length back, as a
car's body follows its front wheels. `tools/turn_smoothness.py`, 200 s of Town05 evening traffic at 60 poses
a second: turning reversals 141 -> 12, yaw acceleration p99.9 870 -> 251 deg/s2, turning rate capped at
150 deg/s.

### Arrow keys, camera markers, pedestrians' feet (19 Oct)

CARLA moves its spectator itself for Z Q S D and turns it only with the mouse (right button held); no key or
input axis turns it (`tools/probe_keys.bat`). Turning it from Python means `set_transform`, which also sends
a place, and that place is always a little old when CARLA applies it: with Z Q S D held the camera was
pulled back at every turn and shook. Tried: from the main loop, from CARLA's tick callback in the viewer's
process (backwards in 12 of 86 frames, `tools/probe_viewer_arrows.bat`) and in a process of its own (still
shook, in use). So the arrows now drive CARLA's own mouse look (`cosim/arrow_mouse.py`, `--arrows mouse`,
default): the right button is pressed for CARLA and the turn sent as small mouse moves from a thread, so
CARLA alone moves and turns the camera. Presses that did nothing are handled (no double-click, moves wait
until CARLA has the mouse) and moves are at most 2 px (bigger ones were sped up by pointer acceleration).
The other ways remain as `--arrows proc | tick | mouse-loop`.

The traffic-camera markers are blue boxes drawn once and for good, 0.8 m behind each lens: redrawn twice a
second they flashed, and red ones were taken for traffic lights.

Pedestrians stood in the pavement to the shins: the map's lanes are at road level and the pavement is a
kerb (0.15 m) higher. Their height is now CARLA's ground projection under them (cached per 0.5 m).

### Pedestrians on the pavement, the walk signal, smooth cars, the live camera (22 Oct)

**Pedestrians waiting in the street.** Measured offline with `tools/check_ped_pavement.py` (SUMO alone,
CARLA's own Town05 map built from the OpenDRIVE file, no server, 10 min at x0.5):

| | before | after |
|---|---|---|
| standing still, not on a sidewalk | 43 % (2 787 of 12 928 on a driving lane) | 0.8 % (none on a driving lane) |
| walking, off the crossings, not on a sidewalk | 9.1 % | 5.5 % (stepping between kerb and crossing) |
| steps where a pedestrian jumps over 1 m more than SUMO moved them | 798 | 0 |

* People waiting for the walk stood on the kerb line of a junction corner, 0 m from the driving lane.
  `CarlaSide._onto_pavement` left such a point alone because it lies on no lane at all (and
  `get_waypoint` sometimes answers "no lane" even well inside a sidewalk). It also kept the offset along
  the sidewalk, which at a sidewalk's end put the point past the end.
* Now every point off a crossing goes onto the nearest sidewalk within 8 m, 0.5 m inside its edges.
* The move onto the pavement glides, at most 0.6 m a step (`walker_pose(..., pid=)`). Without that,
  people stepping off a crossing jumped up to 10 m.

**The walk signal.** CARLA's lamps on the poles cannot be set from Python, so the pedestrian lamps
`cosim/ped_lights.py` draws are now on by default (`--no-ped-lights` turns them off).

* They are hidden while a traffic camera is open, because camera pictures show debug shapes as white
  squares.
* They are redrawn at once when a signal changes.
* The viewer's own emergency pre-emption (`FreeFly.preempt`) used to cut any phase to 1 s, the walk
  included. It now cuts only car greens, and only after their 5 s minimum. SUMO alone with an emergency
  vehicle every 45 s, 20 min: walks shorter than 9 s went from 10 of 177 to 0 of 206.

**Smooth cars.** Two causes, both fixed:

* *The poses were sent unevenly.* The main loop also drew the clock overlay (20-77 ms a redraw, 5 times
  a second) and made CARLA round trips. The log showed 13-47 vehicle updates a second while CARLA drew
  45-118 frames. The poses now have a thread of their own (`freefly.Mover`), at an even `--rate` (60),
  with Python's thread switch at 1 ms and Windows' timer at 1 ms. The main loop only handles the keys,
  the sun and the overlay, at 30 a second.
* *The curve passed SUMO's speed jumps on.* SUMO moves a car at one speed per 0.5 s step. Each step's
  Catmull-Rom curve is now driven at SUMO's speeds made continuous across steps (`interp.Timed`): at a
  step boundary the speed is the mean of the two steps either side, the path is walked by arc length,
  and the car is still exactly at SUMO's position at every step. `tools/turn_smoothness.py`, 120 s of
  evening traffic at 60 poses a second:

  | | before | after |
  |---|---|---|
  | speed jumps over 0.5 m/s between two poses | 4 842 | 222 |
  | acceleration p99.9 (m/s²) | 65 | 19 |
  | jerk p99 (m/s³) | 1 895 | 557 |
  | turning reversals | 5 | 7 |

**The live camera (H).** It opens full screen as a street camera feed (`cosim/cctv.py`).

* *Native* (default, `--cctv native`): CARLA's own view is put at the camera's pose, so the feed is
  CARLA's window itself, at its full frame rate and resolution. The overlay draws only the labels and
  the detector's boxes (2 ms a redraw, 30 a second).
  * A hidden sensor at the same pose and field of view (90 deg, `--cctv-fov`) feeds YOLO, 10 pictures a
    second.
  * The boxes are scaled to the window and carried forward along each vehicle's motion to the moment
    drawn (at most 0.3 s).
  * H again puts the view back where it was. Flying or turning the view off the camera closes it.
* *Sensor* (`--cctv sensor`, the first version): the sensor's own picture, graded, with scanlines,
  vignette and grain. It lagged: the picture only changed when a detection came back (5-12 a second),
  copied to Python and redrawn full screen.
* Labels: top left the camera, junction, direction, town and game clock; top right a blinking red dot
  with REC, and LIVE with the detector's rate; bottom left vehicles, buses, pedestrians, emergency
  vehicles with the siren on and off, and the detector; viewfinder corners.

Tests: `tests/test_freefly_motion.py` (pose thread rate and order, held when SUMO is late, the timed
curve, pavement snap and glide on the real Town05 map) and `tests/test_cctv.py` (the native overlay is
see-through except what it draws, and erases all it drew; boxes scaled and carried forward). Not yet checked in
CARLA itself.

### Smooth at rush hour: what held the viewer up (22 Oct, evening)

Measured with `tools\freefly_profile.bat`: a scripted 40 s flight at 08:00 with `--profile`, which logs every
10 s where each thread spends its time (`cosim/sampler.py`: stack samples and CPU time per thread), the
SUMO thread's parts, the calls the viewer makes to the CARLA server, and the pose process's figures.

**1. Python's lock held while waiting for CARLA.** The viewer's threads together used only about 2 s of CPU
in 10 s, yet everything ran late: SUMO's steps took ~270 ms instead of ~10, the overlay managed 4-14 redraws
a second. Some CARLA calls keep Python's lock while they wait for the server, and so stop every other
thread (checked with the CARLA client library: `apply_batch_sync` and `get_world` let another thread run
once in 1 s of waiting, `get_server_version` 884 times). The pedestrians' ground heights
(`world.ground_projection`, one server frame each) were asked almost constantly. They are now asked by a
process of its own (`cosim/ground_proc.py`, at most 24 waiting). Also:
* SUMO's per-step questions come back with the step (subscriptions: signal states and next switches,
  departures and arrivals, route index, a person's stages): 138 round trips a step down to 13
  (`tools/count_traci.py`, same traffic);
* the overlay's key and window checks keep the lock (`ctypes.PyDLL`), the camera's detector thread waits
  instead of polling.

| rush hour, camera closed | before | after |
|---|---|---|
| SUMO steps late, per 10 s | 42-131 | 0 |
| overlay redraws a second | 4-14 | 25-30 |
| SUMO thread busy, per 10 s | 8-10 s | ~1.2 s |

**2. The server handles about one batch a picture.** In async mode CARLA works through client calls for only
a moment each picture. The pose process sends one batch a picture, so every other costly call put the
poses a picture further behind, for good: 2 pictures behind after a second, more than 12 (cars 2-8 m
behind where they were sent) after 10 s, the delay changing from picture to picture. Fixed in three parts:
* the pose process skips a picture when two of its batches are still not shown (`pose_proc.Lag`);
* the vehicles' lights and the placing of reused walkers now go in the pose process's own batch, instead
  of two batches of their own per SUMO step;
* each walk-signal lamp is drawn once to last until its signal's next switch (at most 2 s), instead of four
  times a second: 1 500-2 350 draw calls per 10 s down to 380-540 (`PedLights.plan`).

Each pose is also timed by the server's own frame clock (`pose_proc.FrameClock`), not by when its callback
happens to run (0-15 ms late).

| near-camera judder, per 10 s of the flight | before | after |
|---|---|---|
| first 10 s (spawning) | 18 % | 6.5 % |
| next 10 s | 16 % | 2.4 % |
| next 10 s | 4.2 % | 0.7 % |
| batches shown exactly 2 pictures after sending | spread over 2 to 12+ | 99.7-100 % |

The demo moves the view with ~27 calls a second; flying yourself moves it inside CARLA, with none.

**3. The live camera (H).** Measured with `tools\freefly_camera_compare.bat` (`--demo-camera 20`: at 08:00 the
native feed open 20 s, the pose process's figures for exactly that time):

| detector sensor | CARLA FPS | uneven pictures | judder | boxes a picture |
|---|---|---|---|---|
| 960 px, every 0.2 s (the old default) | 23 | 57 % | 28 % | 4.6 |
| **640 px, every 0.2 s (now)** | **35** | **0 %** | **0.06 %** | 4.4 |
| 640 px, every picture | 28 | 0 % | 0.2 % | 4.5 |

The hidden sensor that feeds the detector rendered the town a second time at 960 px, and every such
picture made CARLA's own picture late: that was the camera's lag. YOLO works at 640 px anyway. Other sizes:
`--cctv-width`, `--cctv-tick`.

## Town05 agents, steps 1 and 2 (21 Oct)

Work towards making the traffic-light cameras into junction agents. Town05 only. Step 1 measures
what today's controllers do. Step 2 makes every camera send a short report and checks how much of
the road the 53 cameras actually see. A first LLM agent writes a brief from both, and it never
touches the signals.

**Step 1: baseline.** `tools/eval_town05_baseline.bat` (SUMO only) runs every controller on
Town05 at two demands (x0.5 and x0.8 of the scenario's routes), seeds 1-3, one simulated hour
each. Each run has the extras' pedestrians on routed trips over the crosswalks and an emergency
vehicle about every 120 s. Within a seed every controller meets the same emergency vehicles: the
extras now draw them from their own random generator. Results are in `out/baseline_town05/baseline.md`.

* `max_pressure_walk` (cosim/controllers.WalkAware) is max-pressure that also counts the
  pedestrians waiting at a crossing in the walk phase's pressure. Without that, nobody ever asks
  for the walk phase.
  * The walk lasts at least 10 s and is followed by a 5 s clearance.
  * Car transitions keep the program's 3 s all-red. Plain max-pressure would have used the 5 s
    clearance as the all-red of every change.
  * `tests/test_town05_agents.py` replays a run and checks the safety rules and the walk timing.
* Result: actuated is the controller to beat.
  * At x0.5, vehicles wait 79 s per trip under actuated and 147 s under max-pressure.
  * At x0.8, actuated holds (98 s, 2 vehicles stuck). Max-pressure, Webster and the fixed
    programs lock the town (950+ vehicles stuck).
  * Max-pressure switches about every 16 s per junction, losing 6 s of yellow and all-red each
    time. Town05's permissive left turns also limit it to the program's phase order.
* Pre-emption cuts ambulance time loss from 97 s to 64 s at x0.5. Actuated without pre-emption is
  at 51 s, because its queues are shorter.

**Step 2: camera reports.** The code is `cosim/camera_reports.py` and `tools/town05_cameras.py`.

* `town05_cameras_plan.bat` (no CARLA server) lays out the 53 cameras exactly as the inspector
  does. It draws two masks into each: the lanes it counts, and every drivable lane in view.
* A detection's ground point decides what it is:
  * on a counted lane: counted, and queued after 1 s standing;
  * on another road lane: `other_road`;
  * off every road and still for 10 s: `parked`.

  So a full parking lot never adds pressure. Town05's SUMO network has no parking lanes, so
  parking lots are off the road.
* Sirens are voted over 8 pictures per tracked vehicle, and 3 are needed. One picture finds a
  siren 34 % of the time and calls a siren-off vehicle siren-on 9 % of the time
  (eval_town05.md).
* Coverage (`out/town05_cameras/coverage.md`) is 76 % of the last 75 m before the stop lines. It
  is computed from geometry only, so it is an upper bound. Junctions 2014 and 2328 have 2 cameras
  each and are the weak spots.
* `town05_cameras_live.bat` (CARLA running) makes real reports at the held-out junctions 1126,
  1882 and 2296 and compares them with SUMO's truth. It writes `parked_static.json`.
  * Second run, 21 Oct: 10 cameras, 3910 reports, x0.5, seed 1, about 12 minutes.
  * Counts: 1.01 reported for every vehicle in view, an error of 0.1 per camera.
  * Queues: off by 0.02 vehicles.
  * Parked: camera 1126-2 sees about 10 parked cars in every report, and none of them is counted.
  * Traffic was light, at most 7 vehicles in view, so a heavier demand is still to test.
  * The first run reported 1.8 vehicles per real one. The tracker gave a car a new ID while the old
    one lingered, and a report counted both. A report now counts only the latest picture.
* Sirens from the camera alone are not reliable.
  * 11 confirmed, 10 missed and 15 false alarms.
  * 14 of the 15 false alarms had an emergency vehicle with its siren off in view.
  * So pre-emption is triggered by the vehicle's own request, and the camera only confirms it.
* `town05_cameras_sim.bat` runs max-pressure on simulated camera reports, with the visible lane
  stretches only and the detector's error rates as measured live (93 % found, 0.05 extra per
  picture).
  * At x0.5 vehicles wait 171 s, against 147 s on exact counts, and nothing locks up.
  * An earlier version used the error rates of whole pictures at every distance (64 % found, 0.23
    false boxes per picture). It locked one seed out of three. Those rates were too pessimistic for
    the counted lanes.

**The first AI agent: the brief.** `tools/town05_brief.bat` makes one LLM call, about 2k tokens.
It folds the step 1 and 2 results into three small tables and asks the model for a brief in a
fixed JSON shape. A junction the model names that is not in the data gets flagged.

* The client is `cosim/llm.py`, standard library only. Providers are tried in the order of
  `config/llm.json`: Groq gpt-oss-120b, then Gemini 3.5 Flash-Lite, then Groq gpt-oss-20b, then
  OpenRouter's free router.
* A provider that answers 429 cools down, and the same call goes to the next one. Daily counts are
  kept in `out/llm/usage.json`.
* The keys go in `.env`; copy `config/env.example`. It is git-ignored and never logged.
* One account per provider. Google's API terms forbid circumventing its limits and OpenRouter's
  forbid multiple accounts, so the failover goes across providers.
* `tools/llm_check.bat` shows which keys work.

## Town05 agents, steps 3 and 4: on the actuated program (22 Oct)

`cosim/agents_actuated.py`. Version 1 (`cosim/agents.py`, kept as `agents_v1`) replaced the actuated
program with max-pressure and lost to it. Version 2 lets SUMO's actuated program run every signal. One
agent per junction steps in only with commands that keep the program's own states and order: greens
end through their own yellow, jumps go from all red to all red, the walk is never cut. The features:
skip an empty phase, end a green that faces a full exit (spill), end a green kept alive by a trickle,
balance the fill of short links, and emergency pre-emption with a corridor to the next signal.
Two more are kept as options and stay off: batching the walk, and extending a full green past 60 s
(the ablation below shows why). `agents` reads SUMO's lanes; `agents_observed` reads the simulated camera
reports (visible stretches only, the detector's measured error rates).

**What fixed x1.0.** At x1.0 the town is over capacity: actuated locks up within the hour (about 900
vehicles standing at the end). The first version of the agents locked it up sooner. The ablation on
dev seeds 11/12 showed why (vehicles an hour, actuated 1580/1731):

* the queue features alone: 2499/1513;
* the emergency features alone: 1155/682.

In a jam, ambulances that could not move anyway kept cutting and holding greens. A request now counts
only while serving it can move the vehicle (`_can_pass`): the lane it leaves the junction by is not
full, and at most 8 vehicles stand ahead of it. The vehicle judges this itself, as part of its own
request, in both modes. The cameras see 27 % of the lanes' length, and judged by them the requests of
stuck vehicles got through. On dev seeds 11-16 at x1.0, mean throughput:

| | vehicles an hour | seeds better than actuated |
|---|---|---|
| actuated | 1510 | – |
| agents | 1571 | 5 of 6 |
| agents_observed, judged by the cameras | 1536 | 3 of 6 |
| agents_observed, judged by the vehicle | 1725 | 4 of 6 |

**Step 3, the evaluation** (`tools/eval_town05_agents.bat`, seeds 1-3, SUMO alone, 1 h, pedestrians,
an ambulance about every 120 s; full tables in `out/agents_town05/agents.md`). Δ is the paired difference
to actuated, mean over the seeds, with the number of seeds where the agents were better:

| demand | | waiting /trip | throughput /h | ambulance time loss | pedestrian time loss |
|---|---|---|---|---|---|
| x0.5 | actuated | 78.7 s | 1462 | 50.6 s | 90.2 s |
| | agents | Δ −7.5 s (3/3) | Δ −0.3 | Δ −31.6 s (3/3) | Δ −3.5 s (3/3) |
| | agents_observed | Δ −7.1 s (3/3) | Δ +1.3 | Δ −31.3 s (3/3) | Δ −3.9 s (3/3) |
| x0.8 | actuated | 97.5 s | 2298 | 75.9 s | 97.9 s |
| | agents | Δ +0.7 s (1/3) | Δ −20 (0/3) | Δ −49.6 s (3/3) | Δ +0.5 s |
| | agents_observed | Δ +1.9 s (1/3) | Δ −36 (1/3) | Δ −48.0 s (3/3) | Δ +1.0 s |
| x1.0 | actuated | 670.6 s | 1278 | 716.4 s | 157.8 s |
| | agents | Δ −284.4 s (2/3) | Δ +696 (2/3) | Δ −416.0 s (3/3) | Δ −14.9 s (2/3) |
| | agents_observed | Δ −233.3 s (3/3) | Δ +459 (3/3) | Δ −476.7 s (3/3) | Δ −11.0 s (2/3) |

So:
* At x0.5 the agents are better on everything, on cameras as on exact counts.
* At x0.8 they keep the cars where actuated has them (a 1-2 % throughput cost) and cut the ambulances'
  time loss by two thirds.
* At x1.0 they hold the town off gridlock longer. The spread is large: one seed of three still locked up
  under `agents`, as every seed did under actuated.

**Step 4, the ablation** (re-run 23 Oct without extend; `out/agents_town05_ablation/ablation.md` for
exact counts, `ablation_cameras.md` for the camera version: each feature off in turn, x0.8 and x1.0,
seeds 1-3; Δ against the full agents of the same kind):

| off | x0.8 waiting | x0.8 ambulance | x1.0 throughput /h | x1.0 waiting |
|---|---|---|---|---|
| skip | +4.6 s | −3.4 s | −682 | +270 s |
| spill | −1.9 s | −2.4 s | −713 | +287 s |
| trickle | −0.2 s | −1.6 s | −282 | +129 s |
| balance | −2.4 s | −1.8 s | −704 | +270 s |
| corridor | −0.3 s | +8.1 s | −715 | +302 s |
| preempt and corridor | +91 s (one seed locked up) | +176 s | −358 | +104 s |
| + extend (on) | 0 | 0 | −29 (0/3) | +7 s |
| + batch (walk on 2 people or 40 s) | 0 | +0.9 s | −517 | +184 s |

From the cameras (`agents_observed`):

| off | x0.8 waiting | x0.8 ambulance | x1.0 throughput /h | x1.0 waiting |
|---|---|---|---|---|
| skip | −0.4 s | +0.6 s | −339 | +160 s |
| spill | −0.6 s | +1.9 s | −194 | +54 s |
| trickle | −1.3 s | −1.8 s | −119 | +43 s |
| balance | −1.1 s | −1.8 s | +67 | +18 s |
| corridor | −5.2 s | +5.3 s | −239 | +107 s |
| preempt and corridor | −3.8 s | +52.6 s | −173 | +85 s |

* At x0.8 the queue features each move car waiting by a few seconds either way. The ambulance gain is
  pre-emption's, with the corridor adding 5-8 s of it.
* Without pre-emption one seed of three locked up at x0.8 on exact counts (607 vehicles stuck at the
  end): the queue features alone are not safe there.
* At x1.0 the town sits on the edge of gridlock and single runs swing by ±500 vehicles an hour. With
  exact counts, taking out any of skip, spill, balance or corridor locks up all three seeds, as actuated
  does; the full agents keep two of the three moving. From the cameras the effects are smaller and less
  regular (balance off is level), but every other feature still helps.
* Extend, turned back on, changes nothing at x0.8 and costs a little at x1.0 on every seed, so it stays
  off. Batching the walk costs cars at x1.0 and stays off.

## Free-fly: the loading screen and the first view (23 Oct)

**The loading screen, ARTERIA** (`cosim/splash.py`; "the living city, in perfect flow": a city's main roads are its
arteries). Its words are for anyone, not for us: "Raising the skyline", "Filling the streets", "Bringing people
outside", cars, traffic lights, cameras and people counted, the keys in plain words. `town05_freefly.bat` starts it
before CARLA, full screen and above everything, and it stays until the town is ready:
* on the left, Town05's own road network (SUMO's), drawn in light from the centre out as the load goes on, with
  traffic running along it (headlights, tail lights, buses), following the roads' real connections, and the
  signals' lights pulsing at the junctions;
* on the right, the stages of the load ticked off as they finish (the one under way with a spinner and its own
  progress or time), the overall progress, the counts as they become known (vehicles, signals, cameras,
  pedestrians), the game clock, and the keys, one at a time.

It is a process of its own (started with `pythonw`, no console). The viewer's Python is held for seconds at
a time while CARLA loads the map, because CARLA's calls keep Python's lock while they wait; drawn there, the
screen stood still exactly then. The viewer writes what it is doing to `out/freefly/loading.json` five times a
second (`splash.write_status`). When the town is ready the screen fades out and puts CARLA's window in front;
on an error it says so; without news it closes by itself. Esc closes it (the loading goes on). `set
FREEFLY_SPLASH=0` before the bat turns it off. About 6 ms a picture at 1920 x 1080. `tools\freefly_splash_check.bat`
saves a few of its pictures (`SPLASH_SHOTS=1`) and the first view.

**The first view.** The viewer used to start where CARLA puts its spectator, in front of a building, and F8's
junction views looked from a fixed diagonal, which could face a wall. Now every junction has views above each
of its approach roads, looking down the road at the junction (`FreeFly.approach_views`). A small depth camera
visits each of them while loading (about 4 s) and the one with the least close up in its picture is kept
(`_depth_probe`). CARLA's ray cast was tried first: it passed through some towers' meshes, and a tower beside
the view filled a third of the picture. The first view tries the five busiest junctions at 34 m and at 55 m
up: on the PC it starts 55 m above junction 1427, 7 % of the picture close up (of 40 views; the worst, 61 %).

## Free-fly: clean lane changes, accidents and their response, damage, Tunisian plates (23 Oct)

Run `tools\town05_freefly.bat` as before; `tools\freefly_incident_demo.bat` records an accident from crash to
clearance (screenshots `out\freefly\town05_incident_*.png`, timeline `out\town05_freefly_incident.json`).

**White squares at the walk signals.** They were the debug lamps of `cosim\ped_lights.py`. `--ped-lights` is now off
by default (`--ped-lights` turns them back on for checks).

**No car glitches.** SUMO's sublane model with `lcMaxSpeedLatStanding=0`, `lcMaxSpeedLatFactor=0.3`,
`minGapLat=1.0` on every type (a car no longer slides sideways while standing, a lane change is a slow steer),
every vehicle sized in SUMO to the CARLA model that draws it (`cosim\vmodels.py`, `config\vehicle_models.json`),
and each body kept along its lane (`cosim\lanegeo.py`; `interp.Trails`: at most 25 deg off the lane, 12 deg
below 3 m/s, no turning without rolling forward). `tools\check_overlaps.py`, 120 s at the evening rush:

| | before | after |
|---|---|---|
| overlap episodes | 16 (12 over 0.5 m, up to 1.8 m) | 4 (max 0.58 m) |
| sideways slides while standing | 100 | 12 |

**Accidents (not scripted).** `cosim\incidents.py`: 10 % of drivers are careless (`--careless-share`); now and then
one looks away (keeps speed, ignores the car ahead or a red) and SUMO's own collision check decides whether it
ends in a crash. A crash is kept (never removed while the camera is away), cars are put back bumper to bumper,
the lane is coned off, drivers behind are diverted, and the response follows: police car (an officer on foot
waves traffic past the scene, or holds it while the crane works), recovery trucks lift each wreck onto their deck
and drive it away, then the lane reopens. Severity: minor < 12.6 km/h, moderate < 29 km/h, severe above;
damage (turn, tilt, push, sinking, rollover on hard side impacts) grows with it. Marker: `cosim\incident_hud.py`
draws a sign over each accident with distance, phase and plates, pinned to the screen edge when off view.

Measured: SUMO alone, 30 min at the rush (`tools\check_accidents.py`): 2 incidents, both cleared, police on scene in
14-29 s, cleared in 110-230 s. PC demo (12:34 run): 2 crashes (35 km/h severe, 9 km/h minor), both cleared in
about 3 min of game time; police, officer, cones, two recovery trucks each.

**Number plates.** Every car of the pool gets its own Tunisian plate (series, تونس, number; white on black, rentals
white on blue; never repeated), front and rear, on a plate prop of our own (`cosim\plate_proc.py`,
`cosim\plates.py`; mesh SM_TnPlate installed by `tools\plates\install_plates.py`, md5-checked). Places and sizes
measured per model in CARLA (`tools\probe_plates2.py`): rear 40-52 cm, front 38-45 cm, motorcycles rear only.
Made while loading, 24 cars per batch: 187 cars in 10-11 s.

Viewer performance (incident demo, rush hour, GTX 1650): 26-33 FPS, judder 0.8-3 % (it was 72 % while plates
were made after loading in the first version).

Police cars and ambulances carry the state plate instead: red characters on white, the ministry's code, a dash, the
vehicle's number ("12 - 17612"). The codes used for the Interior and Health ministries (07, 12) are placeholders, not
from a source: set them in `cosim\plates.py` (`STATE_CODES`).

The model ends with no plate of their own (BMW Gran Tourer rear, both police Chargers' fronts behind the push bar, the
HGV's front, the Vespa's rear) were measured by `tools\probe_plates3.bat`: depth pictures of the end, the flattest
plate-sized patch of body in the plate band, the plate 8 mm proud of it (`out\_probe\plates3\D_*.png`). Every car,
van, truck and bus now has a front and a rear plate, motorcycles a rear one (the fire truck none).

Flight demo after all of this (`tools\town05_freefly_demo.bat`, evening rush, 70 s): 26.6 FPS high, 30.8 at street
level, 28.7 average, 1 % low 20.8, judder 2.2 %; the last measurement before (19 Oct, same demo) was 28.8-29.6 FPS,
1 % low 19-22, so plates, accidents and their scenes cost no measurable frame rate. Tests: 174 passed
(`tools\test.bat`; new `tests\test_plates.py`, `tests\test_incident_parts.py`).
Plate format: https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Tunisia

## Free-fly: the recovery truck, the scene's traffic, drivers on the pavement (24 Oct)

**No blue plates.** Rentals are gone (`RENTAL_SHARE = 0`): every private car has the white-on-black plate; police
cars and ambulances the state plate.

**The recovery truck (dépanneuse).** The big truck that lifted the wreck is replaced by a rollback: the HGV's chassis
with a flat deck of our own (`SM_TnDeck`, made from SM_Plane by `tools\plates\make_deck_mesh.py`, installed with the
plate mesh, aluminium planks, red/white side band, chevrons). `cosim\towing.py` measures it on the HGV
(`tools\probe_tow.bat`) and times the loading (about 43 s), as it is done on the street:
1. the truck passes the scene on the open lane, stops ahead of the wreck and backs in, reverse lights on (an S-curve,
   9 s); traffic behind it on that lane is held while it backs;
2. the deck slides back (3 s) and tilts 13 deg about the chassis' end until its tail touches the road (3.5 s);
3. the worker hooks the winch; the wreck is pulled up the deck at walking pace, its front wheels then its rear wheels
   following the deck's slope (no flying);
4. the deck tilts back and slides forward with the wreck on it, the worker straps it, climbs in, and it drives off.

**Drivers on the pavement.** After a crash each driver gets out by the door facing the pavement (the other one, and
round the back, when that door is against the other car), walks to the pavement and waits there; traffic on the lanes
they cross stops for them. Later they walk off along the pavement.

**Police on scene means an officer on scene.** The status turns to POLICE ON SCENE only when the officer has left his
car (by his own door, round the car) and stands at his post, behind the last wreck on the open side; he waves cars on
(or holds them while the truck backs in).

**Traffic past the scene.** Cones close the lane from 10 m behind the wrecks; SUMO keeps traffic out of it (only the
police and the trucks may change into it), the open lanes go at 30 km/h, cars behind the wrecks merge early, buses skip
a stop on the closed lane, and a car that needed the closed lane to turn takes a road its own side reaches (they used to
stand at the end of the open lane, blocking the tow truck: the PC run of 05:15). Measured
(`tools\check_accidents.py`, 3 seeds): standing share while waved on 0.01-0.17, all incidents cleared.

**A SUMO crash avoided.** SUMO 1.27 segfaults some steps after a route is looked up with a vehicle's own copy of its
type ("type@vehicle", made when one of its settings is changed) once that vehicle leaves; routes are now looked up with
the shared type (`Incidents._route`). Found with `TRACI_TRACE=<file>` (cosim\sumo_side.py).

Checks: `tools\check_incident_scene.py` (6 seeds, SUMO with the scene's logic, CARLA mocked): no walker inside a car,
no truck through another vehicle, wreck on the deck and truck where SUMO has it to 0.0 m, every incident cleared.
PC demo (05:43 run): 2 crashes, 4 recovery trucks, every wreck loaded and taken away, 0 steep poses, 26-29 FPS.
Tests: 178 passed (`tools\test.bat`).

## Free-fly: accidents run their course while you watch (24 Oct, evening)

Before, much of an accident's response waited for you to look away: the police car and the recovery trucks were only
put on the road out of sight, and a knot near the scene (a car waiting to turn into the closed lane, one stopped half
way out of it, the queue behind them) was only undone by taking cars off the road out of sight. Watching a scene, the
truck never came, or came and stood. Now everything goes on in sight as out of it:
* **Police and trucks come from further back.** When every spot near the scene is in sight they start out of sight up
  to 700 m back along the roads leading there and drive in (`_insert`; the log says from how far back).
* **Knots are undone in sight too.** A car standing in a junction 40 s edges through (8 s without waiting for those
  already in it); one standing 150 s in sight takes another way (out of sight it is still taken off); a car that needs
  the closed lane to turn and has no way on from its side goes somewhere its side reaches; one stopped half out of the
  closed lane, or half over it from the open one, moves over at a crawl (`_still_stuck`, `_recentre`); a car bound for
  a short closed road from the junction before it gets onto it and moves over (the lane is shut 10 m on there:
  `block_at`).
* **The cone line's stand-in no longer stops the open lane.** In SUMO's sublane model the cars on the open lane beside
  it, as cooperative drivers, stopped behind it to let it change lanes, which it never does: a car stood at the start
  of the open lane for 10 minutes, the truck behind it. It now wants no other lane (`_inert`), and it is as wide as a
  truck, so nobody slips by it inside the closed lane.
* **The truck backs in.** It waited for the road behind it to clear, which with a queue behind it never happened.
  Now only where its body sweeps counts (from it to the place, behind and ahead); who follows it stops short of that as
  it pulls up (`_pre_hold`); a car stopped there anyway is waved round it onto the next lane after 6 s
  (`_out_of_the_way`); a truck that has to be put on the road at a junction crash and finds no room tries another lane.
* **The loaded truck drives off at once.** The last one stood until something else moved it (the clearing phase never
  released it); one that loaded in a junction no longer waits there to give way (4 s, then it edges out); the signals
  let a leaving truck through.

Also from the screenshots: the cone line is 0.5 m inside the closed lane and cars on the open lane keep their own gaps
past the scene (they kept 30 cm to the next car, and to the cones, for the rest of their trip after pulling out);
the lane is shut where the taper begins (36 m back), so nobody drives through the cones; the officer never stands in
the open lane (a wreck pushed over the lane line put him there).

Checks (`tools\check_incident_scene.py`; `--watch 250`: whatever is within 250 m of an accident counts as in sight, so
police and trucks must come from out of sight and nothing near the scene is taken off): watched, seeds 7, 11, 23, 26,
31, 53, 11 accidents, every one cleared, lane reopened 4.8-15.7 minutes after the crash (the trucks come from up to
700 m away through the evening traffic); not watched, seeds 7, 11, 23, 26, 9 accidents, cleared in 2.7-4.2 minutes.
No walker inside a car, no truck through another vehicle, wreck on the deck and truck where SUMO has it to 0.0 m.
Before, watched, seed 7 kept a truck standing at the scene for 20 minutes and seed 11 a truck that never came.
PC (`tools\freefly_incident_demo.bat`, the camera on the scene the whole time): 2 crashes, police and 4 recovery
trucks came in sight, every wreck loaded and driven off 1-4.5 s after loading, lanes reopened 3.1 and 8.6 minutes after
the crashes, 20-31 FPS, 0 vehicles vanished in view. Tests: 185 passed (`tools\test.bat`).

## Free-fly: traffic on the ring road (24 Oct, night)

The ring road round Town05 (the "highway") looked empty. It is whole in SUMO: netconvert joined its OpenDRIVE
roads 38, 12, 34 and 35 into one edge per direction and 36 and 37 into another, so it is two loops (35 -> 37, 1.4 km;
-38 -> -36, 1.5 km), three lanes each way, meeting the town at two signalised junctions (2014, 2328). But it had the
town's 50 km/h (netconvert's type file gives every lane 13.89 m/s), and the town's random trips rarely use it:
about 10 cars on the whole ring at the evening peak.

`cosim\highway.py` (on by default in the viewer):
* the ring gets 80 km/h (`--highway-kmh`);
* cars of its own drive round it, 20 on each loop at the evening peak, following the day's curve
  (`--highway-cars N`, 0 for none); they start and are taken off only where the camera does not look, and never
  leave the ring (one sent the short way by an accident's detour goes round again: before, it ended its trip on the
  ring, in sight, 4-5 a run);
* at its two junctions a green for the ring alone ends after 10 s at most. With its own cars those greens were never
  empty, so the junction agents could no longer skip them; the town's roads into the ring junctions are 14-21 m long
  (two or three cars) and their queues filled the junctions behind them (2296, then 1292).

Measured (`tools\check_highway.py`, SUMO as the viewer runs it, evening peak, 30 minutes, the junction agents):

| | cars on the ring | mean speed there | town's cars standing | lock-ups |
|---|---|---|---|---|
| before (seeds 11, 23, 31, 41, 53) | 9-12 | 37-43 km/h | 47-51 % | none |
| 16 per loop, its greens up to 60 s (same seeds) | 35-38 | 54-61 km/h | 57-68 % | cars standing 5 min in 2 seeds |
| 20 per loop, its greens 10 s (seeds 23, 31, 53; the default) | 44-46 | 45-49 km/h | 51-63 % | none |
| 30 per loop, its greens 10 s (seeds 23, 31, 53) | 64-67 | 44-47 km/h | 54-64 % | none |

More ring cars than that queue on the ring itself (45 per loop: 31-34 km/h). The two signals make platoons: the cars
bunch up at them, so a stretch of the ring far from them is often empty for a while. On the PC
(`tools\freefly_highway_demo.bat`: six views along the ring, `out\freefly\town05_highway_*.png`,
`out\town05_freefly_highway.json`) at the evening rush: 18-19 cars on each loop, 139-160 vehicles drawn, 22-29 FPS
(30 a loop: 18-26 FPS), none vanished in view. Tests: 190 passed (`tools\test.bat`).

## Free-fly: the hospital, the police station and the recovery yard (25 Oct)

Ambulances, police cars and recovery trucks used to appear on the road out of sight and vanish again. Now they have
homes (`cosim\depots.py`, on by default in the viewer; `--no-depots` for the old way):

| depot | where (SUMO road, its parking lane) | sign | fleet |
|---|---|---|---|
| hospital | 49, lane 2 | HÔPITAL / URGENCES 24h/24 | 4 ambulances |
| police station | -52, lane 2 | POLICE / COMMISSARIAT | 4 police cars |
| recovery yard | -50, lane 2 (across from the police station) | DÉPANNAGE / REMORQUAGE 24h/24 | 3 recovery trucks |

* The fleets stand in slots along each road's parking lane, which is closed to the town's cars (only emergency and
  police vehicles may use it). The signs are boards on the pavement beside them, with text drawn at run time.
  `tools\probe_depots.bat` photographs them (`out\_probe\depots`).
  `tools\freefly_depots_demo.bat` runs the viewer and shows the three depots three times, a minute apart
  (`out\freefly\town05_depot_*.png`, `out\town05_freefly_depots.json`).
* **Emergency calls** (`EmergencyCalls`). These take the place of the ambulances and police cars that used to just drive
  through. At most one call every 75 s: a vehicle leaves its depot with beacons on and drives to a kerb at least
  250 m away. It stays there 20-45 s with beacons off, then drives home and parks in a free slot. The police station
  always keeps one car back for accidents. A call is called off if an accident happens on the road it was going to.
* **Accidents.** The police car and the recovery trucks come from the station and the yard when the trip is short
  (500 m at most) and does not pass another accident or a road where traffic is standing. They come by a road that
  leads straight onto the lane they stop on. Once the job is done they drive home; a wreck taken to the yard is
  unloaded there. If a depot can't send one, a vehicle is put on the road near the scene as before. A depot vehicle
  that gets held up on its way is replaced by one put on the road near the scene: out of sight after 60 s standing,
  or after 4 minutes en route (5 for a truck).

Found while testing, fixed for all runs, with or without the depots:
* No accident on a depot's road or in the junctions at its two ends (with a lane there shut, nothing could get
  past the parked fleet). No accident on a road shorter than 20 m (road 13 into the ring junction, 14 m: the wrecks
  stood in the junction and the ring backed up behind them). No accident within 60 m of one that is still going on
  (three in junction 1722 at once).
* Depot vehicles on their way home or to a call now avoid an accident's closed lane like any other car. If the only
  way home is through that lane, they drive round a 300-900 m loop and come back.
* A car standing at the end of a lane that doesn't lead to its next road takes a road its lane does lead to
  (two cars side by side, each waiting for the other's lane, stood at a green for good).
* A car that rolled into the place where a recovery truck loads, after it had been told to wait, is let go. Before,
  the truck waited for that car and the car waited for the truck.
* A car left half over the closed lane that doesn't move back by itself slides back half a metre at a time.

Measured (`tools\check_incident_scene.py`, 25 min of the evening rush, a share of careless drivers, seeds 7, 11,
23, 26, 31, 41, 53 and 61, with nobody watching and with everything within 250 m of an accident counted as watched):

| | accidents | cleared | stuck | median time to reopen |
|---|---|---|---|---|
| without the depots (`--no-depots`) | 34 | 33 | 1 (seed 41, watched) | 254-276 s |
| with the depots | 32 | 32 | 0 | 349-479 s |

It takes longer because the vehicles drive in from their depots instead of appearing close by. With the depots,
4 runs out of 16 had a pedestrian touching a vehicle for a moment (mostly a driver stepping out of their own wrecked
car), and 5 had a recovery truck overlapping a passing car while loading in a junction (25 frames at most);
without them, 4 and 1.

## Free-fly: traffic lights as at a real junction (25 Oct)

Until now each signalised junction ran CARLA's own cycle (from CARLA's `netconvert_carla.py`): one approach,
one pole, green at a time with all its movements, the three others red, then a 10 s walk on every crossing at once
with every car red. Signal manuals call this "split phasing" and keep it for odd junctions (FHWA Traffic Signal
Timing Manual, ch. 4); it serves one pole in four, so a car waits through three greens.

`tools/build_town05_signals.py` rebuilds the 15 programs the way most real junctions run (NEMA ring and barrier:
the two opposite approaches together):
* opposite approaches have green together: straight on and right turns; left turns go on the same green and give
  way to the oncoming traffic (a "permissive" left: Town05's left turns share their lane with the straight-on cars,
  and a separate left-arrow phase needs a lane of its own);
* pedestrians walk with the traffic parallel to them, and turning cars give way to them; after the walk comes its
  clearance (the "flashing don't walk": nobody starts, those on the crossing get over, the cars keep their green):
  crossing length / 1.2 m/s, 5-15 s, ended as soon as the crossings are empty (`cosim/sumo_side.WalkSkip`);
* yellow from the approach speed (ITE: 1 s reaction + v / (2 x 3.05 m/s2), rounded up to 0.5 s): 3.5 s in town,
  5 s for the ring road at 80 km/h; all red from the junction's width ((W + 6 m) / v): 1.5-3 s;
* SUMO's actuated program on top (greens 7-45 s on its loops), and the junction agents as before.

SUMO's netconvert builds the phases (`--tls.rebuild --tls.layout opposites`) and, which matters, works out again
who gives way to whom inside each junction for them: a permissive left turn must yield to the oncoming cars,
which CARLA's cycle never needed. Lanes, shapes and connections are unchanged (1027 lanes, no shape moved); the
CARLA poles' links are carried over (200 of 200), so in the viewer the two opposite poles now turn green together.
The originals stay as `scenarios\town05\town05.carla_cycle.*`.

Measured (`tools\check_highway.py`, SUMO as the viewer runs it, 20 min of the evening rush, seeds 7, 11, 23, 26, 31,
41, 53, 61):

| | town's cars standing | vehicles halting at the 15 junctions | ring road speed | lock-ups |
|---|---|---|---|---|
| CARLA's cycle, actuated | 58.5 % | 72 | 50 km/h | none |
| CARLA's cycle, junction agents (exact counts) | 55.5 % | 70 | 48 km/h | none |
| CARLA's cycle, junction agents (cameras, the viewer's) | 58.9 % | 71 | 50 km/h | none |
| opposite approaches together, actuated | 45.3 % | 50 | 58 km/h | none |
| opposite approaches together, junction agents (exact counts) | 43.7 % | 49 | 58 km/h | none |
| opposite approaches together, junction agents (cameras) | 51.0 % | 54 | 61 km/h | 1 of 8 (seed 61) |

The one lock-up is a knot in the triangle of junctions 1292, 1126 and 1574 (short roads; cars inside the junction
waiting for a full exit). The viewer undoes knots (`IncidentManager.unjam`: a car standing 150 s out of sight is
taken off); measured that way (`tools\check_highway.py --unjam`, cameras' agents, same 8 seeds) CARLA's cycle
54.2 % standing, 66 halting, 49 km/h on the ring, 92 cars taken off; opposite approaches together 44.3 %, 49,
60 km/h, 80 taken off.

In the viewer (`tools\freefly_signals_demo.bat`: each of the 15 junctions from 30 m up once some poles are green,
`out\freefly\town05_signals_*.png`, `out\town05_freefly_signals.json`): at 12 of the 15, two poles green at once
(the opposite approaches) and the others red; at the T-junctions 2240 and 2296 one approach has a green of its
own, and 2086 was caught in its yellow (two poles yellow). Tests: 194 passed, 1 skipped (`tools\test.bat`: the
walk-aware max-pressure test needs an exclusive walk phase, which no longer exists).

One agent rule no longer pays with two greens a cycle: "trickle" (ending a green kept alive by a few moving cars
when the other side has a queue) costs a whole switch each time; it is off by default now (`agents+trickle` turns
it back on). Without "balance" as well, one seed locked up, so that one stays.

## Free-fly: the accident marker, a recovery truck loading in a junction (25 Oct)

* The accident marker (`cosim/incident_hud.py`) sat at the edge of the screen pointing at the accident however far
  you flew ("it follows me"), and its notice was redrawn at a new place on every picture, which tore in the
  overlay. Now: off screen it shows only within 250 m (with its title alone), on screen within 900 m; the notice
  is made once per text (distance in 10 m steps) and moves only when it has moved 3 px or more.
* A recovery truck that loads a wreck in a junction is drawn where SUMO does not have it (SUMO cannot always put it
  there), so cars drove through it (seed 31: 26 frames). While it backs in and loads, the cars coming up to the ways
  through the junction that pass its body wait at their stop line (`IncidentManager._guard_place`); they go when it
  leaves. Seed 31: no overlap.

## Free-fly: a driver at the wheel (25 Oct)

CARLA's cars drove empty. Now the cars near the camera have a person at the wheel (`cosim/drivers.py`):
* a pool of 16 walkers (grown-ups in everyday clothes, not the police or the recovery crew) is made while the town
  loads; the pose process (`cosim/pose_proc.py`) seats one in each of the cars nearest the camera, within 45 m,
  and moves it with its car every picture, as it does the number plates; a car keeps its driver up to 52 m (no
  swapping at the edge), then the walker goes to another car;
* the pose: the walker's bones turned once (thighs forward, knees bent, arms out to the wheel), as the officer's
  gestures at an accident; which way each bone turns was tried in CARLA first (`tools\probe_drivers.py`);
* the seat: for cars, at the middle of the box lengthwise, 0.4 of the half width to the left, at 0.83 of the half
  height but at least 0.92 m under the roof (low cars: the Mustang, the Coupe, the Mini); vans, lorries and the bus
  have their own places (`TALL`), all 34 four-wheeled models checked from the side and the front in CARLA
  (`tools\probe_drivers.bat`, `out\_probe\drivers\`);
* no driver in a wreck (the drivers are out, at the scene), in a vehicle parked at its depot, or in a car whose door
  is open at an accident (the officer got out); bicycles and motorbikes come with their own rider.

`--no-drivers` turns it off. `tools\freefly_drivers_demo.bat` goes to 8 cars standing in traffic (different models)
and photographs each from above its driver's side and from the front (`out\freefly\town05_drivers_*.png`,
`out\town05_freefly_drivers.json`). On the PC: 6-12 drivers seated at a time at street level, the pose process
8-13 ms a picture as before, no errors.

## Town05 agents, step 5: final evaluation, the weak spots, the agents in the viewer (24 Oct)

Report (design, method, all results): the "Junction Agents on Town05" doc. Here the short version and how to run it.

**Fixed: an ambulance stuck in a queue made a junction loop.** `tools\diag_town05_agents.py <controller> <scale> <seed>`
watches one evaluation run and says where and when the town jams. On dev seed 14 (x1.0) it showed junction 207 giving
phase 0 65 greens in 15 minutes and its three other approaches 4: an ambulance stood in phase 0's queue, its exit full,
its request coming and going, and each time it came back the agent jumped straight to phase 0 again. Two rules in
`cosim\agents_actuated.py` end it: `ev_patience_s` (45 s: an emergency vehicle not through its signal by then asks no
more there) and `ev_starve_limit` (a phase passed over 3 times is served before an emergency jump). With both:
15/12/14/11 greens.

**Not a weak spot: x0.8 throughput.** The -20 vehicles an hour of seeds 1-3 was seed noise: on dev seeds 11-16 the agents
carry 2282 an hour against actuated's 2277, waiting 97.2 s against 98.1 s.

**Tried and dropped:** `+noflow` (end a green under which nobody moves, from the loops or the cameras: x0.8 waiting -2 s
but x1.0 1382 vehicles an hour against 1656, and a lock-up at x0.8 on a held-out seed) and an earlier "exit full" (70 %
instead of 85 %: no change). Settings can be tried by name: `agents+noflow@noflow_s=6@full_share=0.7`.

**Final evaluation, held-out seeds 21-30** (`tools\eval_town05_final.bat`, `out\agents_final\agents.md`). Lock-ups:
at x0.5 and x0.8, one run in ten locks up for each controller (actuated x0.5 seed 24; agents x0.5 seed 23 and x0.8 seed
26; agents_observed x0.8 seed 29). The diagnosis of seed 26: three cars standing inside junction 2086 for 10 minutes,
each waiting for another, their exit empty: a deadlock of SUMO's junction model, which any controller can meet. So the
evaluation was run again with SUMO's deadlock breaker (`--ignore-junction-blocker 60`, nobody removed;
`tools\eval_town05_final_jb.bat`, `out\agents_final_jb\agents.md`): no run locks up, and against actuated (mean of the
paired changes, seeds better of 10):

| demand | agents | waiting /trip | ambulance time loss | throughput /h | pedestrian time loss |
|---|---|---|---|---|---|
| x0.5 | exact counts | -6.7 s (10) | -37.9 s (10) | +3.1 (8) | -3.4 s (10) |
| x0.5 | cameras | -7.2 s (10) | -38.4 s (10) | +5.2 (9) | -3.6 s (10) |
| x0.8 | exact counts | -1.9 s (7) | -54.2 s (10) | -5.4 (5) | +1.0 s (2) |
| x0.8 | cameras | -2.3 s (7) | -54.4 s (10) | -1.8 (4) | -0.5 s (6) |

Actuated: 79.8 s and 98.4 s waiting, ambulances 57.9 s and 86.2 s. At x1.0 (standard setting) the town locks up in 9
runs of 10 under actuated and in all 10 under the agents; the exact-count agents still cut the ambulances' time loss
(median -212 s) but cars gain nothing. The ablation on seeds 21-25 (`out\agents_final_ablation_exact`, cloud;
`out\agents_final_ablation`, cameras): at x0.8 pre-emption is worth 53 s per ambulance and skip 5.2 s of car waiting;
at x1.0 skip off gave more throughput (+395 an hour, 4 of 5 seeds): skipping empty greens may hasten the lock-up of a
full town.

**The agents in the viewer.** `tools\town05_freefly.bat` now runs `--controller agents_observed` on Town05 (the
agents on the cameras' reports, as evaluated; `--controller actuated` for SUMO's program alone). The overlay
(`cosim\agent_hud.py`) shows each decision where it happens: over the junction for 10 s, a pill with what the agent did
("SKIPPED AN EMPTY GREEN", "GREEN HELD FOR AN AMBULANCE", "GREEN WAVE: GETTING READY FOR AN AMBULANCE", ...); at the
stop lines of that green, chevrons in the direction of travel, coloured by the kind of decision; top right, the
decisions near you of the last minute. The viewer's own pre-emption gives way to the agents'. Off by default since
24 Oct (too many notices next to the accidents'); `--agent-hud` shows it (`tools\freefly_agents_demo.bat` does).
`tools\freefly_agents_demo.bat` goes to junctions where the agents have just decided something, each kind in turn,
and saves a screenshot of each (`out\freefly\town05_agents_*.png`, `out\town05_freefly_agents.json`).
On the PC (GTX 1650, the demo at the evening rush): 28-37 FPS with the agents on, judder at most 6.5 %, the SUMO thread
never late; a simulated second costs it 180-260 ms (about 30 ms with actuated), mostly the camera simulation. Tests:
182 passed (`tools\test.bat`; new in `tests\test_town05_agents.py`: settings by name, noflow, the decision log, the
overlay).

## Town05 agents on the new signals (25 Oct)

Re-checked, corrected and measured again on 26 Oct. The full report is the "Junction Agents on Town05" doc, section
"Re-evaluation on the new signals"; the revised doc text is `notes\Junction_Agents_on_Town05_revised_26oct.md`.

**Why the old cycle was not enough.** Until 25 Oct every signal ran CARLA's own cycle (`netconvert_carla.py`; CARLA's
docs, core_actors: the lights of a junction take turns, one "set to green while the rest remain frozen in red"). One approach, one pole, was
green at a time with all its movements; the three others were red. Then came a 10 s walk on every crossing with every
car red, followed by 5 s all red. That is split phasing, which signal manuals keep for odd junctions (FHWA Traffic
Signal Timing Manual, ch. 4). A car waited through three greens. The 5 s after the walk could not clear a 14-24.5 m
crossing: SUMO alone, x0.5, it released 785 movements an hour over a crossing still occupied with less than its
clearance time (`out\signal_checks\town_carla_cycle`).

**What each signal does now** (`tools\build_town05_signals.py`; the full per-junction table, with each lane's
movements, the pole showing them, every conflict and who gives way, is `out\signal_audit\town05_signals.md`):
* **Opposite approaches go together** (NEMA ring and barrier): N+S, then E+W.
  * Lanes: on Town05's two-lane approaches the right lane is right + straight on and the left lane straight on +
    left. The three-lane approaches (N/S at 53, 979 and 1722; the ring road at 2014 and 2328) have one more
    straight-on lane. *(Corrected 27 Oct: at 53, 979 and 1722 that third lane is a parking lane; only the ring road has
    three driving lanes. See "Signals where no green gives way".)*
  * Every lane of an approach goes on its approach's green.
  * Each CARLA pole shows its approach: three heads over the lanes and one on the post
    (`tools\freefly_poles_demo.bat`, `out\town05_freefly_poles.json`). All four show the actor's one state, as the
    MUTCD's two or more faces for a through movement do. CARLA has no pedestrian head on these poles.
* **Left turns are permissive** (circular green, give way to the oncoming traffic).
  * They share their lane with the straight-on cars. A protected arrow from a shared lane can only be split phasing,
    which is the old cycle again.
  * The busiest left turn is 46 000 (veh/h)² at the design demand, against the 150 000 above which a left-turn phase
    is considered with two oncoming lanes (ADOT TGP 612).
* **Right turns** go on the green, giving way to people on the crossing they turn over (SUMO "g"). There is no right
  turn on red.
* **Pedestrians walk with the parallel traffic**, never on all four crossings at once.
  * **Walk:** with the car green, at least 7 s.
  * **Flashing don't walk** (the "walk clearance" phase): nobody starts, people already on the crossing get across,
    and the parallel cars keep their green.
  * **Steady don't walk:** the yellow and all red, before any conflicting traffic goes.
  * Flashing don't walk lasts crossing / 1.2 m/s. Together with the steady don't walk it covers at least the MUTCD's
    pedestrian clearance time, crossing / 3.5 ft/s (MUTCD 11th ed., 4I.06). That is 12 s over 14 m, 15 s over 20 m
    and 16.5 s over the ring road's 24.5 m.
  * The flashing don't walk ends early once the junction's crossings are empty (`cosim\sumo_side.WalkSkip`, like an
    on-crossing detector).
  * `--ped-lights` draws the lamps in the viewer, now flashing during the walk clearance.
* **Yellow** comes from approach speed (ITE): 3.5 s, or 5 s on the ring road. **All red** comes from junction width:
  1.5-3 s.
* **Timing:** SUMO's actuated program runs on top (greens of 7-45 s on its loops), with the junction agents above it.

**Fixed on 26 Oct**, found by the new audit (`cosim\signal_audit.py`, `tools\audit_town05_signals.py`: every program
against SUMO's own junction logic):
* A 15 s cap on the flashing don't walk left the 24.5 m crossings at 2014 and 2328 with 21.5 s, where 22.9 s are
  needed. It is now 16.5 s.
* Webster's split had cut every walk clearance to 5-8 s and the walks to 5-6 s. Now the walk clearances are the
  network's and the walks last at least 7 s.

All 45 programs now pass. `cosim\signal_monitor.py` checks the same rules step by step in a running simulation:

| check | what it found |
|---|---|
| Whole town, 4 simulated hours (actuated and agents, x0.5 and x0.8) | no conflicting greens, no green straight to red, no short intergreen, no crossing released before its clearance |
| 12 stress scenarios at junction 1126 (`tools\check_signal_scenarios.py`, `out\signal_checks\1126\summary.md`) | 36 runs (12 scenarios: heavy N-S, heavy E-W, all heavy, left turners, right turners over busy crossings, people at every crossing, people arriving at each switch, slow walkers at the end of the walk, light traffic, heavy traffic without pedestrians, an ambulance every 30 s, uneven demand; x 3 controllers, 20 min each): no rule broken anywhere in town; longest wait at 1126 at most 124 s for a vehicle (left turners, actuated) and 93 s on foot; people still on a crossing when it was released, after its full clearance time (slow walkers; turning cars give way to them): 0-42 releases a run |

**The pedestrian demand fix.**
* **Why:** the 25 Oct agents cost pedestrians 0.4-1.7 s per walk against actuated. The walk now runs with a car
  green, so skipping a green with no cars also skipped its walk.
* **The fix:** `ActuatedAgents._demand` counts people waiting at a crossing as demand for the green their walk runs
  with: a pedestrian call, like a push button.
  * It counts only people on a walking area whose next edge is that crossing.
  * Each person is counted once, for one crossing; nobody already on a crossing is counted.
  * Feature `pedcall`; `agents-pedcall` turns it off.
* **Checked** on seeds 21-25 (`out\agents_pedfix`, paired with the same agents without it,
  `out\agents_pedfix\pedfix_vs_v2.md`):
  * Pedestrians: time loss against actuated -0.1 / +0.4 s (exact counts / cameras) at x0.5, from +1.6 / +1.4 s on
    the same seeds before the fix; +0.2 / +0.2 s at x0.8, from +1.1 / +0.6 s.
  * The longest wait at 1126's crossings under light traffic fell from 76 s to 35 s.
  * Ambulances kept their gain.
  * Cars waited 0.4 / 1.2 s more at x0.5 and -0.6 / +0.5 s at x0.8.
* **Kept.**

**Final evaluation on the corrected signals** (`tools\eval_town05_final_v3.bat`, `out\agents_final_v3\agents.md`). The
protocol is that of `out\agents_final_v2`: actuated, agents and agents_observed; x0.5 to x1.2; held-out seeds 21-30;
1 h each; no deadlock breaker. Mean of the paired changes against actuated, with the seeds better:

| demand | seeds | actuated: waiting /trip | agents (exact) vs actuated: waiting, throughput /h, ambulance time loss, pedestrian time loss | agents (cameras) vs actuated: the same | vehicles stuck > 300 s at the end (all runs) |
|---|---|---|---|---|---|
| x0.5 | 21-30 | 29.1 s (ambulance 33.9 s, pedestrian 59.3 s) | -0.3s (8/10), -1.4 (3/10), -10.5s (10/10), +0.0s (4/10) | -0.1s (6/10), -1.4 (2/10), -10.7s (10/10), +0.0s (5/10) | 0 |
| x0.8 | 21-30 | 31.4 s (ambulance 41.4 s, pedestrian 59.4 s) | -0.0s (5/10), +0.3 (5/10), -13.0s (10/10), -0.0s (3/10) | -0.2s (7/10), +0.7 (6/10), -12.9s (10/10), +0.1s (5/10) | 0 |
| x1 | 21-30 | 35.0 s (ambulance 50.2 s, pedestrian 59.7 s) | +0.3s (4/10), +2.7 (6/10), -18.3s (10/10), +0.2s (4/10) | -0.2s (6/10), +1.5 (3/10), -18.1s (10/10), +0.2s (3/10) | 0 |
| x1.2 | 21-30 | 42.1 s (ambulance 62.7 s, pedestrian 60.8 s) | -0.3s (6/10), +3.8 (8/10), -26.2s (10/10), +0.1s (5/10) | -0.1s (6/10), +1.2 (8/10), -24.8s (10/10), -0.1s (7/10) | 0 |

The same on 25 Oct, before the two fixes (`out\agents_final_v2\agents.md`):

| demand | seeds | actuated: waiting /trip | agents (exact) vs actuated: waiting, throughput /h, ambulance time loss, pedestrian time loss | agents (cameras) vs actuated: the same | vehicles stuck > 300 s at the end (all runs) |
|---|---|---|---|---|---|
| x0.5 | 21-30 | 29.1 s (ambulance 34.0 s, pedestrian 59.2 s) | -0.7s (8/10), +1.9 (8/10), -10.5s (10/10), +1.4s (0/10) | -0.9s (8/10), +1.2 (6/10), -11.1s (10/10), +1.7s (0/10) | 0 |
| x0.8 | 21-30 | 31.4 s (ambulance 42.0 s, pedestrian 59.2 s) | +0.1s (4/10), +2.7 (7/10), -15.5s (10/10), +0.9s (0/10) | -0.1s (6/10), -1.1 (3/10), -14.0s (10/10), +0.7s (2/10) | 0 |
| x1 | 21-30 | 34.8 s (ambulance 51.1 s, pedestrian 59.3 s) | -0.5s (6/10), -0.8 (3/10), -19.8s (10/10), +0.7s (2/10) | -0.3s (5/10), -0.5 (3/10), -20.3s (10/10), +0.8s (0/10) | 0 |
| x1.2 | 21-30 | 41.5 s (ambulance 61.6 s, pedestrian 60.7 s) | +0.6s (3/10), -5.6 (5/10), -25.1s (10/10), +0.5s (1/10) | +0.4s (5/10), -0.1 (7/10), -24.3s (10/10), +0.4s (3/10) | 0 |

**Limits.**
* SUMO's junction model now has conflicting streams on one green (permissive lefts, turns over crosswalks). With
  `--collision.check-junctions` it reports a few contacts inside junctions:
  * Vehicles: 0-2 an hour town-wide.
  * Pedestrians: 7-16 an hour, mostly at walking pace. The old cycle had 7-9, faster.
  * The viewer's drawn overlaps per vehicle moved are about as before (`tools\check_overlaps.py --variant viewer`,
    `out\signal_checks\overlaps_*.json`).
  * A longer accepted gap for left turns (`jmTimegapMinor` 2.5 s) did not change them.
* Junction 2296's north approach has no CARLA pole on the map, so its signal exists in SUMO only.
* Pedestrian calls come from SUMO's people directly (a push button), in both agent modes.

## Signals where no green gives way (27 Oct)

**What was wrong.** The 25-26 Oct programs let turns go "permissive": a left turn on the same green as the oncoming
traffic, a right turn on the same green as the people on the crossing it turns over, each giving way (SUMO "g"). A
car entered the junction on its green and then stood in the middle of it, waiting. Measured
(`tools\check_junction_stops.py`, the junction agents, 30 min, seed 21): 438 / 700 / 871 vehicles stood inside a
junction for someone crossing their path at x0.5 / x0.8 / x1.0. And all four heads of a pole always showed one colour.
A second cause turned up on the way (`tools\diag_junction_stops.py`): SUMO's walkers stepped onto crosswalks at red
(15 min at x0.8: 174 of 321 crossings started at red, both directions), so even a car with nothing else in its way
stopped for them.

**What each signal does now** (`cosim\signal_plan.py`, built by `tools\build_town05_signals.py`; per junction in
`out\signal_audit\town05_signals.md`):
* **Every green is a free go.** No "g" anywhere, and no two links whose paths cross (cars or a crosswalk, by SUMO's own
  junction geometry) are ever green together. All 45 programs pass (`cosim\signal_audit.check_protected`,
  `tools\audit_town05_signals.py`); `tests\test_signal_programs.py` checks it.
* **The unit is the lane.** A lane's directions go together (the left-turner at the head of a straight-and-left lane
  would otherwise block the straight-on cars behind it). On Town05's two-lane approaches (left lane straight on +
  left, right lane straight on + right) every lane carries a direction that crosses another approach's path or a
  crosswalk, so the only fully protected way that keeps the town moving is **each approach on its own** with all its
  lanes, four car greens a cycle (FHWA STM ch. 4, split phasing). The T-junctions and the ring road (2014, 2240,
  2296, 2328, 1882) have three greens, where lanes of different approaches go together and the heads of a pole
  differ (the ring road's left-turn lane red while its straight-on lanes go).
* **Pedestrians** cross on a walk of their own with every car red (all crossings at once: a "scramble"), at least
  7 s, then the flashing DON'T WALK (the crossings show "u": red to SUMO's walkers, flashing on the lamps) for
  crossing / 1.2 m/s, then 2 s of steady DON'T WALK before any car moves (MUTCD 11th ed. 4I.06: flashing + buffer >=
  crossing / 3.5 ft/s). The walk is skipped when nobody waits; its clearance ends once the crossings are empty
  (`sumo_side.WalkSkip`, as on-crossing detectors do). **Nobody steps onto a crossing at DON'T WALK**: people on the
  corner whose next step is a red crossing are held there until WALK (`sumo_side.PedRedHold`; after it, 0 of 245
  crossings started at red).
* **Parking lanes** (N and S at 53, 979, 1722: OpenDRIVE type parking, 5 km/h) have no movement at the signals any more.
  They had a straight-on connection beside the right lane's right turn, which needed a fifth green. *Correction:* on
  25-26 Oct the README and the doc called those three approaches "three-lane"; they have two driving lanes and a
  parking lane. Only the ring road (E and W at 2014 and 2328) has three driving lanes.
* **Yellow and all red** as before (ITE: 3.5 s, 5 s on the ring road; all red from the junction's width). Actuated
  greens 5-30 s (FHWA STM tables 5-3 / 5-5), the junction agents on top.

**The heads in the viewer** (`cosim\signal_heads.py`; off by default since 29 Oct, see "Real signal lamps" below; `--signal-heads` to draw them). A CARLA
traffic light has one state for its four heads, so the poles' own lamps are switched off and each head is drawn lit
(CARLA debug drawing) at its red, amber or green lens with arrows: the three heads over the road are the approach's
lanes from the drivers' left, the left head the left lane (arrows left and straight on), the middle head the right
lane's straight on, the right head its right turn; on three-lane roads one head a lane. The head on the post is the
pedestrian signal of the crosswalk beside it (walk, flashing, steady). Checked in CARLA at 1126
(`tools\freefly_poles_demo.bat`, `out\town05_freefly_poles_27oct_b57.json`): 210 heads on 54 poles, each pole's heads
showing their lanes' states.

**What it costs.** Measured with the junction agents, 30 min, same traffic (`tools\check_junction_stops.py`,
`out\signal_checks\junction_stops_*.json`; x1.0 is the viewer's evening peak, most of the day runs at 0.5-0.8 of it):

| signals | x0.5 | x0.8 | x1.0 (rush hour) | cars standing inside a junction for a conflict (x0.8) |
|---|---|---|---|---|
| permissive (26 Oct) | 32 s, 728/774 trips | 37 s, 1158/1228 | 41 s, 1438/1531 | 700 |
| **split: each approach on its own (default)** | 93 s, 707/774 | 106 s, 1096/1228 (seed 22: 113 s) | 169 s, 1218/1514 | 4 |
| lead-lag: the left lanes held red while the opposite right lanes go | 102 s, 698/773 | 139 s, 1059/1222 | 324 s, 754/1348: jams | 8 |
| left-turn lanes (left arrow red while straight on goes) | 90 s | 280 s, 643/1096: jams | - | 6 |

Mean waiting per trip, trips completed / started. SUMO's own actuated program without the agents, split, x0.8: 106 s,
1106/1227 (the agents gain nothing yet on this cycle: they were tuned on the permissive one). The left-turn-lane row is
from a build before the parking-lane fix. Pedestrians lose 135-157 s a walk trip (83 s with the permissive
signals): they now wait for the scramble and never cross at red. Fully protected signals cost about a factor 2.5-4
in waiting on Town05: two lanes an approach, each carrying a turn, leave only one approach (or one pair of lanes) at a
time. Holding the left lane red while the right lanes go (what a head per direction would show) adds a switch per
road; giving the left turns a lane of their own leaves straight on one lane; both jam at the rush.

**Choosing the signals** (close the viewer first): `tools\signals_split.bat` (the default),
`tools\signals_leadlag.bat`, `tools\signals_permissive.bat` (the 26 Oct programs); `tools\use_signals.py` copies
`scenarios\town05\town05.<set>.*` over the files in use (it keeps a backup if they match no set).

## Coordinated signals, lamps on the lenses, no crosswalk over the ring road (27 Oct, afternoon)

**One cycle for the whole town, green waves** (`cosim\coordination.py`, built by `tools\build_town05_signals.py`
into `scenarios\town05\town05.coordinated.add.xml`; the viewer's default controller on Town05 is now
`coordinated`, `--controller agents_observed` for the junction agents). Run on their own, each signal kept its own
short cycle: greens of 4-6 s, and a car leaving one junction on green met red at the next, 23-65 m on.
* **The cycle:** 115 s at every signal: Webster's (1.5 L + 5) / (1 - Y) for the busiest junction at the design demand,
  at least the yellows, all reds, the walk and a minimum green for each green.
* **The greens:** each at least 12 s (7 s for a green of turns only), the rest shared by the junction's lane flows.
* **The offsets:** chosen so that the cars leaving one signal on green reach the next on green, weighted by how many
  make each step on their routes (coordinate search over the offsets): 42 % of the signal-to-signal steps arrive on
  green, against 31 % with every cycle starting together.
* **In step, whatever happens** (`sumo_side.CycleKeeper`), as coordinated-actuated controllers run: a green other
  than the first ends early when it has had its minimum and nobody is coming on its lanes, a walk nobody waits for is
  skipped, and what they leave goes to the first green (the junction's busiest, the one its wave is built on), which
  always ends at its planned second of the cycle. A pre-empting emergency vehicle that holds a green is caught up at
  the next cycle. Checked over 6 cycles (700 s) at every signal: the first green ended at the same second of the cycle every
  time; it lasts 26-36 s at the town's junctions and up to 75 s for the ring road at 2014.
* **Measured** (`tools\check_junction_stops.py`, 30 min, seed 21; mean waiting per trip, trips completed / started):

| controller | x0.5 | x0.8 | x1.0 (the viewer's evening rush) |
|---|---|---|---|
| junction agents (exact counts, adaptive, no common cycle) | 93 s | 100 s, 1094/1227 | 215 s, 974/1407 |
| **coordinated** | 106 s | 113 s, 1087/1228 (seed 22: 116 s) | **136 s, 1320/1531** |

  At the rush the coordinated signals move a third more trips with a third less waiting; off-peak the agents' short
  cycles wait a little less. Cars standing inside a junction for a conflict: 1-8 a run (all protected greens).
  `tools\green_wave_diagram.py` draws a corridor's time-space diagram (`out\signal_checks\green_wave_row200.png`).
* **Why the city's poles still turn green together:** on the town's two-lane approaches every lane carries a turn,
  so holding one lane red while the next goes (as the ring road does with its own left-turn lane) costs a switch per
  road; coordinated, the lead-lag set waited 201 s at x0.8 and the left-turn-lane set jammed (310 s, 603/1080). The
  ring road and the T-junctions (2014, 2328, 1882, 2240, 2296) keep heads that differ.

**The lamps** (`cosim\signal_heads.py`). The arrows sat 0.22 m in front of the head, bigger than a lens (0.3 m), and
glowed outside it. Now each arrow is 0.15 m, inside its 0.2 m lens, and drawn on the glass: a ray from in front of
each head finds the lens face inside its visor (210 of 210 heads in CARLA). Each direction's arrow has its own colour
on its own lens (red top, amber middle, green bottom). The low head on the post repeats the nearest lane for the
drivers at the line; the dot drawn on it as a pedestrian signal (a square, like the ones removed on 22 Oct) is gone.

**No crosswalk over the ring road.** Its four 24.5 m crossings at 2014 and 2328 (three lanes each way, 80 km/h) are
removed in the build (`left_lane_deletions`); the side road's crossings stay. Town05 has 51 signalised crossings.

**The ring road camera.** Both carriageways of the ring road lead from 2014 to 2328 (it is a loop), and the camera
plan (`cosim\perception.plan_cameras`) took them for one road: each of those junctions had a camera for one direction
only. 55 cameras now (53); the junction agents' camera view there rose from 55 % / 58 % to 82 % / 88 % of the incoming
lanes, enough for the agents to run them (`out\town05_cameras`, re-planned).

## Real signal lamps: one colour per pole, lit in CARLA's own lenses (29 Oct)

Asked: the drawn direction lamps (`cosim\signal_heads.py`) looked fake, not part of the signal; make them integrated
and as in real life, after research with examples.

**What real signals do.** A signal face shows a full circle when every movement from its lane(s) may go, and arrow
lenses only where one movement goes while another from the same approach is held (a protected left turn beside
moving straight-on traffic):
* US, MUTCD 2009 ch. 4D (figures 4D-9 to 4D-18; TxDOT's signal head requirements): a circular green over a lane
  releases every movement from it; green / yellow arrows are for a turn that runs on its own (protected mode); a
  shared lane whose movements all run together keeps circular indications.
* Europe, Vienna Convention on Road Signs and Signals (art. 23): the full green lets traffic go in every direction
  the road allows; an arrow (lit on a black lens) releases or stops one direction on its own. UK TSRGD 2016 schedule
  14 likewise (green arrow filters beside a full red).
* Lane control lights (a red X / green arrow over each lane) are for reversible lanes and tunnels, not junctions.

**What Town05 does now.** At the four-way junctions every approach already went alone with all its lanes (split), so
the real-life face there is the full circle, not an arrow. At the T-junctions and on the ring road (1882, 2014, 2240,
2296, 2328) one lane went while its neighbour waited, which no single CARLA pole can show. `cosim\signal_plan.py`'s
unit is now the approach (`SIGNAL_UNIT=approach`, the default): every approach goes as one everywhere, and in every
phase of every program each of the 54 CARLA poles has one colour (`test_every_carla_pole_shows_one_colour`; with the
27 Oct set 8 poles at those junctions mixed colours). The viewer no longer draws heads: CARLA's own lamps show
SUMO's colour (`--signal-heads` brings the 27 Oct drawing back). They are the traffic-light model's own emissive
lenses: lit in the glass, with their glow, seen by every camera and sensor (the debug drawing was only in the window).

**Traffic** (`tools\check_junction_stops.py`, 30 min, seed 21; waiting per trip, trips completed):

| controller, scale | 27 Oct (split) | 29 Oct (approach) |
|---|---|---|
| coordinated x0.8 | 112.5 s, 1087 | 108.6 s, 1103 |
| coordinated x1.0 (the rush) | 136.1 s, 1320 | 118.4 s, 1355 |
| junction agents x0.8 | 99.5 s, 1094 | 102.3 s, 1086 |

The coordinated offsets put 44.7 % of the signal-to-signal steps on green (42.2 % before). Signal sets:
`tools\signals_approach.bat` (default) / `signals_split.bat` (27 Oct) / `signals_leadlag.bat` /
`signals_permissive.bat`; `scenarios\town05\town05.approach.*` keep this one.

**Checked in CARLA** (`tools\freefly_poles_demo.bat`: junction 1126, each pole from 9 m in front of its heads, green,
amber and red; `out\freefly\town05_poles_1126_*.png`, `out\town05_freefly_poles.json`, contact sheet
`out\freefly_town05_poles_1126_29oct_sheet.jpg`): 12 of 12 shots show the colour SUMO gave, on all four heads of each
pole, the low one on the post too; no errors in the run's log. PC tests: 224 passed
(`tools\logs\test_29oct_b67.log`).

**Not possible from Python: an arrow inside a CARLA lens.** A CARLA traffic light is one actor with one state for its
four heads, and its lamps' look is in the Unreal asset. The texture API (`apply_float_color_texture_to_object`) puts
one texture on every material of every mesh of an object (CARLA 0.9.16 `ApplyTextureToActor`), not on one lens.
Arrow lenses would need new traffic-light blueprints made in the Unreal Editor (a CARLA source build), one head per
direction; with those the T-junctions could get their arrows back.

## More kinds of incident: breakdowns, speeders, pile-ups on the ring road, look-alikes (31 Oct)

Until today every incident was two cars that met: a careless driver on the phone (rear-end) or through a red light
(side impact). Added in `cosim/incidents.py` (module doc), each with its own option in the viewer:

* **Breakdowns** (`--breakdowns`, `--breakdown-every`, default one every 2400 s of simulated time on average while a
  new incident may start). A car puts its hazards on and rolls to a stop in its lane at 2.5 m/s2, never in a junction
  and never on a road shorter than 20 m; 30 % of them on the ring road (its long roads otherwise took nearly all). The
  usual response follows. No damage, no debris; the marker reads BREAKDOWN and "breakdown, 1 car". A car that runs
  into the stalled one turns it into a rear-end crash; a touch under 1 m/s is a near miss.
* **Speeders** (`--speeders`, 8 % of new cars): SUMO speed factor 1.25-1.5, three times as often careless. Wrecks carry
  `speeding`; the log names them.
* **Pile-ups on the ring road** (`--pileups`, `--ring-brake-every`, default 400 s). After a crash on the ring (or in a
  junction a ring road leads into), the next cars behind it in its lane may react late: the k-th with chance
  0.7 / 0.6 / 0.45 / 0.3, keeping their speed 0.8-1.8 s, then braking as hard as they can; whoever cannot stop joins
  the incident (up to 6 cars). Each car that joins is in turn watched by the one behind it. From three cars the kind
  is "pile-up". Ring crashes were nearly absent (none in 8 h of SUMO alone), so a ring car now and then brakes hard
  all of a sudden (an emergency stop past its normal deceleration), only with a car 15 m or less behind it, which
  then reacts late (1.2-2.2 s). The ring road's own cars are also three times as often careless.
* **Look-alikes, not incidents** (`--lookalikes`): a delivery van stops in its lane with hazards on for 60-180 s
  (multi-lane town roads only, at most 2 at a time), a line bus stands 70-150 s at 3 % of its stops. Logged
  (`IncidentManager.lookalike_snapshot`), nobody responds; the recorder writes them and the seconds each camera has
  them in its picture into `incidents.json` (`lookalikes`, `lookalikes_visible`).
* **CARLA side** (`cosim/incident_scene.py`): cars that join an incident later now get their damage and their own
  debris (before, a third car stood there undamaged); no debris for a breakdown.
* **Fix, all incidents**: a recovery truck waiting 240 s ahead of its wreck without getting in goes back and another
  comes another way; if that one cannot either, the wreck is lifted off where it stands. Seed 41 had a second wreck
  in junction 1930 whose truck waited behind it for 50 minutes, with 47 cars standing over 2 minutes; now cleared
  at 822 s.

**Measured, SUMO alone** (`tools/check_accidents.py --ring --buses --minutes 60`, evening rush, 8 seeds each, before
with `--no-new`):

| | before | after |
|---|---|---|
| incidents in 8 h | 11 (8 rear-end, 3 junction) | 38 (15 rear-end, 5 junction, 8 pile-ups, 10 breakdowns) |
| mean crash speed | 40 km/h | 45 km/h |
| look-alikes in 8 h | 0 | 25 double-parked vans, 73 long bus stops (then 0.05 a stop; now 0.03) |
| vehicles stopped over 2 min, not in an incident, per run | 7.4 | 13.1 (more incidents) |

Forced ring crashes (a car stopping dead, 4 seeds x 40 min): 3 of 7, 0 of 3, 1 of 2 and 3 of 7 became 3-car
pile-ups. Hard braking forced every step (4 seeds x 40 min, 638 brakings): 62 crashes, 10 of them pile-ups of 3-5
cars. The incident step costs 1.6 ms per SUMO step instead of 0.8 (SUMO alone, 10 min).

**In the viewer** (`tools/freefly_breakdown_demo.bat`, `tools/freefly_pileup_demo.bat`; screenshots kept in
`out/_demo_31oct_breakdown/` and `out/_demo_31oct_ring_crash/`): a breakdown on -39_3, hazards, BREAKDOWN marker,
officer, cones, driver on the pavement, cleared after 384 s; a rear-end crash on the ring road at 48 km/h after a
hard braking, cleared. `tools/town05_freefly_demo.bat`: 20.6 FPS with the new kinds and 20.7 with all of them off
(`out/_bak_31oct_incidents/`), so the new code does not cost frames; the PC itself gave 32.9 in the same demo that
morning (three CARLA runs had just ended). `tools\test.bat`: 202 passed.

## A red-light camera, its plate reader and the police office (31 Oct - 1 Nov)

**What was asked.** An agent that watches a camera at a signalised junction, catches the cars that cross on red,
reads their plates, looks the plate up in the town's vehicle register (owner and car) and sends the case to the
town's police office, where an officer reviews it and issues the fine notice; "like in real life, not synthetic".

**The rule kept throughout: the agent sees only what a real one would.** The pictures and the junction controller's
signal log. SUMO's own record of who entered on red (`cosim/red_runners.RedLightTruth`, with each car's real plate)
is written beside each recording and used only to grade the agent (`tools/eval_redlight.py`).

**Parts** (README section 5.7):
* `cosim/enforcement.py`: the site (stop line, lanes, the controller's links), the cameras, each picture timed on
  SUMO's clock (the step it shows), the signal log, the recording (video + picture times + log + record).
* `cosim/redlight.py`: detector (the Town05 YOLO26n, through Ultralytics on the GPU or ONNX), boxes placed on the
  road by the camera's geometry, tracker, the rule (front over the line with the lane red at least 0.5 s, then seen
  going on; emergency vehicles with lights on exempt), plate reading voted over views, the evidence (photos A/B with
  a data bar, clip, crops, case.json with SHA-256 of each file). `cosim/redlight_proc.py` runs it beside the viewer.
* `cosim/anpr.py`: classic number-plate reading for the Tunisian plate: candidates (rows of character strokes,
  plate-shaped blobs, bright bands), the plate's panel outlined and straightened (perspective), the series and the
  number found from the plate's layout (series at the left edge, number at the right edge, the word تونس between) and
  read by laying the plate font's own digits along each group (best-matching sequence, so digits run together are
  still read); a confidence from the match and its margin.
* `cosim/register.py`: every car with plates, make, model, colour and a fictional keeper.
* `tools/police_office.py`: the officer's window: photos, clip, readings, register entry; accept (with the plate as
  the officer reads it) or reject; a notice (HTML) for the keeper; files checked against their SHA-256.
* `cosim/red_runners.py`: drivers who go on through the end of the amber (dilemma zone) or see a fresh red too late.

**What the recordings showed, and what changed because of them** (all at junction 1574, road -29, evening rush;
`tools/record_enforcement.bat`):

| Recording | Camera | Truth on red | Agent | Plates |
| --- | --- | --- | --- | --- |
| 3 (30 min, red-runner share 0.6), graded offline with the final rule | one camera 18 m back, 6 m up, -11 deg | 1 | 1 caught, 0 false; 34 of 41 crossings found, crossing time median -0.04 s (p10 -0.98, p90 +0.29) | 6 right, 2 wrong, 26 unread of 34 crossing cars |
| 4 (40 min, share 1.0), live | one camera 24 m back, 7 m up, -17 deg | 2 | 2 caught, 0 false; crossing time and time into red within 0.03 s of SUMO's | 0 of 2 read (characters about 10 px high) |
| 5 (40 min, share 1.0), live | overview + plate camera (17 deg) on the same pole | 2 | 2 caught, 0 false (same drivers: the run is deterministic); 31 of 37 crossings found, crossing time median -0.07 s (p10 -0.50, p90 0.00) | live: 1 of 2 read (259 TU 7712, 1.00); offline with the final reader: both right (259 TU 7712, 255 TU 8086, 1.00); of the 31 crossing cars found, 22 right, 1 wrong (unsure: 0.19, not counted), 8 unread |

* Recording 3's camera saw a car only once its front was over the line: the crossing could only be timed by
  extrapolation. The pole went to 24 m back and the view down to -17 deg: the road from 9 m before the line.
* The overview camera's plates were too small to read (recording 4). Real installations have a separate plate
  camera; so does the site now: same pole, 17 deg instead of 30, aimed at the last metres before the line (both
  lanes), about twice the pixels on a plate. The agent finds a vehicle in its pictures from where the overview camera
  has it on the road at that moment (both cameras' poses are known), and reads several of them; the overview's own
  views still vote.
* The plate reader was first built and tested on the plate design alone (synthetic renders). On real CARLA crops it
  found almost nothing: the plate's characters run into its border at small sizes, plates are tilted and slanted,
  the panel is mid-grey in evening light, and at small sizes the word تونس is nearly as tall as the digits. Each was
  fixed on real crops (recording 2's 105 vehicle crops, recordings 4 and 5's red-runners): on recording 2's crops,
  4 then 6 right readings instead of 0, the only sure wrong one being a car on a tow truck whose real plate is not in
  the list compared against, and one misread digit (below).
* The traffic-camera markers (debug drawings: every camera sees them) showed as white poles in the enforcement
  picture; they are no longer drawn near an enforcement site.
* Recording 5 first gave three wrong readings: a partial plate read as a whole one ("8 TU 34" for 248 TU 3451,
  confidence 0.89), an emergency vehicle given another vehicle's plate (1.00), and an unsure one (0.19, not counted).
  Both sure ones are now caught by the cameras' calibration, as a real system would: where the vehicle is on the road
  says how wide its plate must look (0.52 m at that distance) and where in the picture it can be. Measured on the 191
  right readings: widths 0.93-1.23 of the expected, centres within 1.23 plate widths across and 0.79 up or down; the
  fragment was 0.67 wide, the other vehicle's plate 3.15 plate widths too high. A reading outside 0.85-1.35 wide, 1.3
  across or 1.2 up or down is dropped (`PLATE_SIZE_OK`, `PLATE_WHERE_OK` in `cosim/redlight.py`). Afterwards: 22
  right, 1 wrong (the unsure one), 8 unread; both cases unchanged. What this cannot catch is a misread digit on the
  right plate (on recording 2's crops: 4199 read 4491 at 0.66): the reading is voted over several views, and the
  officer confirms the plate against the photo before a notice.
* A case's id is the site and the crossing time; a later run at the same time (the simulation is deterministic)
  overwrote recording 4's cases. Such a case now gets a suffix (`-2`).

**The police office on real cases.** Recording 4's two cases, the plates entered by the officer as read on the
photos: integrity checked on every file, the register's entry found (case 1: Mercedes Coupe 2020, dark red, as in
the photos), notices SIM-2026-00001 and -00002 written, marked as a simulation with a fictional keeper.

**Cost.** With both cameras and the live agent, CARLA ran at 20.2 FPS on average in the viewer (recording 5's log,
239 readings); the agent handled 78 % of the overview's pictures (18,590, 5,242 skipped while it was busy), the
plate camera's are only kept until a plate is read.

**Not done yet.** Handing an unread car over to the next junction's camera (the user's idea, 1 Nov): the first
camera would send a description (class, colour, appearance, exit, expected arrival) and the next would re-identify
the car and read its plate, with the officer confirming the match.

## The moment of a crash, and the police in minutes (3 Nov)

The GO-FLOW team's accident agent v2 was not accepted (their report, 2 Nov): it caught 10 of 10 test crashes, but with
a median of 70 s to detect them. It mostly reacted to stopped police lights, which appeared 15 to 45 s after
impact. They also saw crashing cars go from driving to standing over 2 to 3 s, like hard braking, and asked whether
the viewer's smoothing causes this. Two of v2's problems came from the simulator.

**How it was measured.** `tools/measure_impact.py` runs the viewer's traffic and accidents in SUMO alone
(`tools/check_accidents.py`, no CARLA). It keeps each wreck's SUMO states from 8 s before the crash to 5 s after. It
then replays them through the viewer's own motion code (`interp.Timed`, as `pose_proc.FramePoser` uses it, 60
poses a second) and through the damage ramp. The run: 6 seeds (7, 11, 23, 31, 41, 53), 60 min each, the evening
rush, the ring road's cars on, careless share 0.3 (more crashes in less time; it does not change how one happens).
The same tool ran on the code before the change (git `033a793`) and after it. Results: `out\impact_3nov\`
(`impact_b2_*` before, `impact_a2_*` after, `impact_town_*` after without the ring road, `police_old_*` / `police_new_*`). What CARLA itself draws was not
recorded in this check: the pose process applies these poses one picture later and does not change their shape.

**What made the impact soft.** The viewer's smoothing between steps was not the cause: it spreads a stop over half
a step (0.5 s) at most. There were three causes:

1. **The push and turn came before the hit.** The damage ramp (`pose_proc.lay_wreck`) started on the wall clock
   when the scene first saw the crash. That is 0.7 s before the cars were drawn touching. The ramp started before
   contact for 64 of 64 wrecks, and was over before contact for 24 of them. At the real moment of contact nothing
   new happened. The debris and the hazard lights also appeared before the hit.
2. **The cars overlapped, then pulled apart.** SUMO notices a collision only at the end of the 0.5 s step it
   happened in. By then the cars are inside each other by up to a step's travel. The follower was drawn driving
   into the other car, then pulled back out over the next 0.5 s (median 0.75 m, up to 5.5 m; 17 of 33 colliders
   moved back more than 0.5 m). The drawn car took a median of 0.30 s to stop at the hit (90 % of its impact speed
   to under 0.5 m/s; up to 0.87 s), easing in like braking.
3. **Red-light runners braked inside the junction.** In 4 of 5 junction crashes, the car that ran the red braked
   at 4 to 4.5 m/s² for 2 to 3.5 s in the junction. It was then hit at 0 to 0.5 m/s in 3 of them. Its mode kept a
   safe gap, and the crossing car most likely counted as a car ahead (one braked while still distracted). A lapse that ran out inside the junction also turned it
   back into SUMO's careful driver. This is SUMO's own motion. On camera it looks like a car braking to a stop and
   then a soft crash.

Other braking before a crash is meant to be there and was left alone. On the ring road a car brakes hard for no
reason the others see, and the car behind reacts late, then brakes as hard as it can and still hits. A broken-down
car rolls to a stop at 2.5 m/s².

**What changed:**

* **Put back to the contact.** `contact_fraction` (`cosim/incidents.py`) finds when in the step the two bodies
  first touched: boxes of SUMO's lengths and widths, moved from the step before. `_to_contact` puts both cars back
  along their way to that point (`moveTo`; `moveToXY` inside a junction). `t_contact` / `t_hit` is that moment on the
  simulation's clock. `t0` stays the step that saw it, so earlier labels still line up.
* **A dead stop at the hit.** The viewer ends the crash step where the cars came to rest
  (`IncidentManager.hit_poses`, `FreeFly.produce`). It draws them going on at their speed and stopping dead there
  (`Timed(stop_at=...)`, `pose_proc.hit_fraction`). The step before still runs on towards where SUMO had them, so
  nothing brakes before the hit.
* **The impact starts at the hit.** The push and turn are timed from `t_hit` on the SUMO time the picture shows
  (`pose_proc.damage_share`). The debris goes down with the hit (`IncidentScene.hit_delay`). The hazard lights
  come on 2 s later (`HAZARDS_AFTER_S`).
* **A red-light runner goes on through the junction.** Once in the junction it goes on at its speed and brakes for
  nobody. Its lapse lasts until it is out, at most 4 s longer (`RED_THROUGH_MAX_S`).

**After** (the same seeds; the first crashes of the runs are the same ones, later ones differ):

| Measure | Before | After |
| --- | --- | --- |
| Damage ramp started before the drawn contact | 64 of 64 wrecks | 0 of 71 |
| Damage ramp over before the drawn contact | 24 of 64 | 0 of 71 |
| Colliders moving back more than 0.5 m after the hit | 17 of 33 (median 0.75 m, up to 5.5 m) | 0 of 36 |
| Drawn stop at the hit, 90 % of the impact speed to under 0.5 m/s | median 0.30 s, up to 0.87 s (40 wrecks) | median 0.03 s, up to 0.18 s (49) |
| Pairs drawn inside each other at rest (bodies shrunk 0.15 m) | 28 of 31 (where SUMO saw them, as drawn before; counted on the after runs) | 3 of 31 |
| Junction crashes: red-runner speed at the hit | 0.0, 0.1, 0.5, 4.7, 4.9 m/s | 5.5 to 8.8 m/s (8 crashes) |

The 6 town seeds without the ring road give the same picture: no move back after the hit, no ramp before contact,
and 1 of 18 pairs still inside each other. Those few pairs could not go back far enough because their car had come
onto its lane during the step. The red-runners still slow down *before* the junction (its safe gap, for a crossing
car ahead) and then cross at 5.5 to 8.8 m/s. A real distracted driver might not slow at all. This was left as it is.

**The police: minutes, not seconds.** Before, the police car was sent 6 s after every crash and was on scene 13 to
45 s after it. v2 learnt "stopped police lights = crash" from this. Now the dispatch delay is drawn per incident from
a log-normal: median 240 s, spread 0.5, kept within 60 to 900 s (`police_delay`, `POLICE_MEDIAN_S`, `POLICE_SPREAD`).
The drive there comes on top; the car starts out of sight close by, so this is 7 to 39 s. Viewer option
`--police-delay MEDIAN[:SPREAD]`; `6:0` gives the old response. Each incident records `police_record`: `due`, `sent`,
`on_scene` (empty if they never got there) and `how` ("car at the scene", "car stopped short in traffic, officer on
foot", "on foot (no way in for a car)"). It is in the snapshot, in `incidents.json` (`police`, `t_contact`,
`contact_at_clip_s`) and in `check_accidents.py`'s output.

Measured with `tools/check_accidents.py --ring --buses`, 3 seeds (7, 11, 23) × 60 min, normal careless share:

| Measure | `6:0` (before) | default (after) |
| --- | --- | --- |
| Incidents | 18 | 17 |
| Police on scene after the crash | median 14 s (13 to 45 s) | median 206 s (104 to 411 s) |
| Police not on scene when the run ended | 0 | 2 (still due) |
| Lane reopened after the crash | median 180 s | median 419 s (4 still open at the end) |
| Cars standing over 2 min, not part of an incident | 39 | 36 |

Slower police did not lock the town, and the number of incidents barely changed. Incidents now stay open longer, so
`MAX_ACTIVE` (2 at a time) is reached more often. This may matter for how many crashes a recording run yields.

**A bug on the way.** Re-sending a police car moved the incident's own `t0`. The event times written to
`incidents.json` (`t0` + each event's offset) then shifted for that incident. The re-send now moves only
`police_due`.

**Tests.** `tests/test_crash_moment.py` (15): contact geometry, the dead stop, the ramp from the hit, the debris
delay, the hazard delay, `hit_poses`, the police delay, its option and the snapshot fields. All 228 pass (213
before).

**Not yet checked in CARLA.** A recording with crashes in front of a camera should confirm it frame by frame on the
pictures: `tools\record_incident_clips.bat`, then compare `contact_at_clip_s` with the clip.

**Checked in CARLA (3 Nov, afternoon).** Two runs of `tools\record_incident_clips.bat` (careless share 0.5,
four cameras, 16:00, real time).

* *The camera pictures could not settle it.* Run 1 had 4 crashes. Two had their contact on film, on cameras 1427-2
  and 1126-1, but both were 70 to 90 m away: a car was 20 to 40 pixels long, partly hidden by poles and trees, and the
  detector's boxes jittered more than the stop itself (`tools/check_impact_video.py`, kept for closer crashes).
* *What CARLA drew instead.* The pose process now writes each wreck's actor, as the server's own snapshot has it in
  every picture, from the hit's step to 4 s after it (`pose_proc.CrashTrace`, `out\freefly\crash_trace.csv`;
  `tools/check_crash_trace.py`). This does not depend on any camera.
  * Run 1: hw28 at 15.5 m/s in the 0.3 s before its hit moved 0.23 m in all after it, in the picture of the hit, and
    stood. It did not move back.
  * Run 2: hw1 at 25.7 m/s stopped in the picture of the hit (0.14 m between +0.1 and +0.4 s). It was then pushed 1.45
    m and turned over the next 0.85 s: the damage ramp, timed from the hit as intended. The car that ran into the
    wrecks a second later (hw13) also stopped where it hit.
* *Two things the trace showed, fixed.*
  1. Run 1's struck car (hw13, standing) slid 1.5 m sideways in the 0.5 s *before* it was hit. That was its centring
     in the lane (`_fit_in_lane`), drawn in the hit's step. The hit's step now ends where it touched
     (`Wreck.hit_pose`); the centring is drawn in the next step, with the push of the impact. In run 2 the centring
     (0.8 m) came after the hit.
  2. A car far from the viewer's eye was posed only every 2nd picture (beyond 120 m) or every 4th (beyond 250 m)
     (`pose_proc.NEAR_M`, `FAR_M`). In run 1 the eye was at the first recorded camera, so a crash on another camera
     moved in steps of up to 0.45 s, 2 to 4 poses a second at about 9 pictures a second. This was also true of the
     earlier recordings (`record_v2_training`). Every recorded camera is now an eye for the pose process, as the
     red-light camera already was (`record_eyes` in `cosim/freefly.py`). In run 2 a crash on the ring road was posed
     in every picture. CARLA ran at 9 to 10 pictures a second with four cameras recording, as before.
* *Seen on the way, not changed.*
  * Run 1's incident 3: the police were sent 226 s after the crash and three cars in turn did not get through in
    the next 12 minutes, behind the queue another crash had left near the same junction. With the police seconds
    away, such a queue had no time to form. The police may need to come through the stopped traffic (as on a real
    run), or the queue be undone.
  * Only 2 incidents may be open at once (`MAX_ACTIVE`), and each now stays open 7 to 10 minutes. Run 1 had 3 crashes
    in its first 10 minutes and then none for 8. This matters for how many recording runs the 30-crash delivery needs.

## The police get through, and 3 incidents at a time (3 Nov, evening)

These are the two problems from the CARLA check. They were measured in SUMO alone at careless share 0.5, as in the
recording runs.

**The police stuck in a queue.** `tools/check_incident_scene.py` runs the scene code with the depots and someone
watching (`--watch 150`, so nobody is put on the road in sight). It reproduced the stuck police at 4 seeds × 40 min:
13 of 15 police cars got to their scene and 2 never did. One case, followed step by step (seed 11):

* the wrecks stood at the end of road 8 (31 m long) on lane 8_3, the only lane with the right turn onto -5;
* a car at the end of the open lane 8_4 that wanted that turn stood at a green light for 20 minutes, and the queue
  behind it, the police car (40 m from the scene) and then the recovery trucks with it.

`_keep_lane` did take such a car to a road its lane leads to. But a car standing 150 s in sight is rerouted by travel
times (`unjam`), and that put it back on the blocked turn, every time. With the police there within seconds, the
queue never got that long. Two changes:

* after that rerouting, `_keep_lane` runs again (`kept_lane_after_reroute`);
* a police car standing over 30 s in a queue within 60 m of its scene, on a road where it may not stop (before, only
  the closed lane or the wrecks' lane), lets the officer get out and walk (`ON_FOOT_AFTER_S`, `ON_FOOT_NEAR_M`).
  `police_record.how` says "officer on foot from the queue". The car keeps its place and comes on to its stop when the
  traffic moves. An officer whose car stands over 20 m from his post first walks up between the lanes, then across
  (`incident_scene.FAR_POST_M`).

Same 4 seeds × 40 min, after: 17 of 17 police got to their scene, median 74 s and at most 186 s after being sent. The
officer walked from the queue 4 times.

**Fewer crashes.** Only 2 incidents could be open at once (`MAX_ACTIVE`), and with the police minutes away each stays
open 7 to 10 minutes. `tools/check_accidents.py --ring --buses --share 0.5`, 3 seeds × 60 min:

| | Police at 6 s, 2 open | 4 min, 2 open | 4 min, 3 open | 4 min, 4 open |
| --- | --- | --- | --- | --- |
| Crashes (breakdowns not counted) | 13 | 8 | 14 | 14 |
| Cars standing over 2 min, not part of an incident | 35 | 52 | 57 | 50 |

`MAX_ACTIVE` is now 3. The viewer makes the scenes' people, police cars and recovery trucks ready for that many
(officers, crew and drivers, 1 + 1 + 3 per scene; option `--max-incidents`). 4 gave no more crashes: then the
cooldown between incidents (60 s) and the careless drivers set the pace.

**What is left around the scenes.** The same 4 seeds × 40 min, all on today's code, with the police at 6 s (2 open)
and at 4 min (3 open):

| | Police at 6 s, 2 open | 4 min, 3 open |
| --- | --- | --- |
| Incidents | 11 | 17 |
| Police on scene, sent to there | 11 of 11, median 52 s, at most 320 s | 17 of 17, median 74 s, at most 186 s |
| Recovery trucks held up on their way and replaced | 4 | 14 |
| Poses of a person inside a standing car (drivers to the pavement, officers to their post) | 159 | 155 |
| Poses of a recovery truck overlapping another vehicle | 831 | 427 |

So the people walking through cars and the trucks backing into others are not from the slower police; they were
there with the police at 6 s too (all of them in seed 31 then). With the police at 4 min they are spread over three
seeds (48, 53 and 54 poses), and seed 11 had a truck backing out of a junction across an ambulance in the queue. More trucks held up is from the slower police: a scene now has its queue before they come.
None of these stops an incident from clearing. They are the next things to fix if the crash clips must be clean in
the minutes after the police arrive.

**In CARLA (3 Nov, 16:19).** `tools\record_incident_clips.bat` with all of the above, 18 minutes (careless share 0.5,
four cameras):

* the scenes made ready for 3 incidents (walkers: 3 police, 3 crew, 9 drivers; 9 recovery-truck decks), and 3 were
  open at once from the start; 7 crashes in the 18 minutes;
* the police got to all 6 scenes they were due at, 7 to 134 s after being sent; at incident 5 the police car stood
  in a queue on another road and the officer walked; the 7th was not due yet when the run was stopped;
* every incident due to clear did (4);
* the wrecks, as CARLA drew them (`crash_trace.csv`): the 6 cars moving at their hit stopped within 0.16 to 0.59 s
  (the push of the impact included), none moved back more than 0.1 m;
* no errors in the viewer or the pose process (118 reports, 0 errors); CARLA 8.1 to 12.9 pictures a second, median
  9.3, as before;
* one recovery truck was held up on its way and replaced.

## Option A: crashes where the cameras see them (6 Nov)

The GO-FLOW team asked for 30 crashes or more clearly visible to a fixed camera (2 Nov). At the town's normal share of
careless drivers that would take about 100 recording runs. Of the three ways discussed, the project owner chose A:
the town's careless share stays the normal 10 %, and only where a recorded camera sees clearly are crashes made
likelier. The normal traffic elsewhere stays as it is.

**How.** `cosim/clip_recorder.Focus` says whether a car standing at a point would be whole in a camera's picture,
6 to 30 m ahead, at least 24 pixels from its edge (the team's "within about 30 m and fully in frame"). While the
cameras record (`--crash-focus P`), `IncidentManager._focus_step` looks at every car within 80 m of a camera, once a
step:

* it works out where the car would hit: the back of the car ahead if one is within 40 m, else where the car will be in
  1.5 and 3 s;
* if that point is in a camera's clear view, the car is made careless with chance P (decided once per car, only
  while a new incident may start, from a generator of its own so the town's own draws stay as they were);
* a careless driver there looks away 6 times as often (`FOCUS_BOOST`). A car made careless this way looks away only
  there; elsewhere it drives like everyone else.

Whether a crash follows, and how, is still the traffic's. Each incident says whether the focus had a hand in it
(`crash_source`: "camera focus" or "natural", in the snapshot, `incidents.json` and `tools/check_accidents.py`).
**Crashes in front of the cameras are far more frequent than in reality: never read a rate from these recordings.**

**Measured** (`tools/check_accidents.py --ring --buses --focus-cams "..." --crash-focus P`, SUMO alone, seeds 7 and 11,
60 min each; "seen clearly" = a wreck where it came to rest is in a camera's clear view):

| Cameras | P | Crashes | Seen clearly | Cars standing over 2 min, not part of an incident |
| --- | --- | --- | --- | --- |
| Ring road (2014-2, 2014-3, 2328-2, 2328-1) | 0 (natural) | 6 | 1 | 22 |
| Ring road | 0.5 | 30 | 24 | 123 |
| Ring road | 1 | 30 | 21 | 156 |
| Junctions (1126-1, 1427-3, 838-2, 979-4) | 0 | 6 | 0 | 22 |
| Junctions | 0.5 | 20 | 11 | 130 |
| Junctions | 1 | 31 | 16 | 111 |
| Unseen: 2014-1, 2328-3, 1292-1, 2086-1 | 0.5 | 14 | 6 | 76 |
| Unseen: 2240-1, 1882-1, 1126-2, 53-1 | 0.5 | 14 | 5 | 42 |

A first version left the cars made careless careless for their whole trip and looked only at where a car itself
would be: 40 crashes on the ring road at P = 0.5 but only 13 seen clearly. Aiming at where it would hit, and letting
it look away only in front of the camera, made it 24 of 30.

The cost: more incidents mean more queues around them (cars standing over 2 minutes: 22 → 42 to 156). The normal
traffic far from the cameras is as before, but the stretch before a crash may show another incident's queue. Clips of
normal traffic for the team are better taken from runs without the focus (as the v2 training runs of 31 Oct).

**The recordings.** `tools\record_stage2.bat`, P = 0.5, six 50-minute runs with four cameras each:
* train: the ring road at 09:00 and 17:30, junctions at 12:00;
* test: two sets of cameras never used before (15:00, 16:30), and the ring road at 10:30.

The estimate from the table is about 38 crashes seen clearly. `record_stage2.bat trial` records 25 minutes on the ring
road first, to check the yield in CARLA.

Tests: `tests/test_crash_focus.py` (5).

### The trial in CARLA, and the clips' clock (6 Nov)

**Trial** (`record_stage2.bat trial`, ring road cameras, 09:00, 25 min, `out\incident_clips\20261106_141418`): 6
crashes, 3 of them clearly in a camera's picture, about 7 an hour. SUMO alone at 09:00 gives 8 to 13 an hour (seeds 7
and 11), so the trial is at the low end, not out of line.

**More incidents at a time does not help.** `tools/check_accidents.py ... --crash-focus 0.5 --max-incidents M`, ring
road cameras, seeds 7 and 11, 60 min each, crashes seen clearly:

| Hour | M = 3 | M = 5 |
| --- | --- | --- |
| 09:00 | 8, 13 | 8, 9 |
| 16:00 | 14, 11 | 12, 7 |

More open incidents mean more queues, and fewer cars reach the cameras' views at speed. `record_stage2.bat` keeps 3.

**The labels were about 2 s early.** Projecting the trace of a wreck (`out/freefly/crash_trace.csv`, what CARLA drew)
into the 2328-2 video showed the car on film about 2 s after its `frames.csv` time. `sim_s` there was SUMO's latest
step when the picture arrived (`ff.sim_now`); the pictures show an older step (the viewer runs a step behind, the
frame feeder and the SUMO queue add more). `ClipRecorder` now times each picture as `cosim/enforcement.SiteCamera`
does: CARLA's own time of the picture, put on the wall clock by the lowest gap seen in `world.on_tick`, then
`FreeFly.depicted_at` (the SUMO time shown then). The old value stays in a new column, `sumo_s`. Every label taken
from `frames.csv` (`crash_at_clip_s`, `contact_at_clip_s`, the cut windows) follows. **Clips recorded before 6 Nov
(v1, v2) carry the early labels: their crash happens about 2 s after the labelled second.**

Tests: `tests/test_clip_recorder.py` (2 more).

## The GO-FLOW forecasters: merged, checked in CARLA, retrained, made visible (7 Nov)

Rayan's branch `integration/goflow-prediction-town05` (DSO4's queue forecaster and GO-FLOW's traffic-state agent, each
in its own process) was merged off by default. In CARLA on the project PC: 247 tests passed; both agents start from
the shipped models; a killed DSO4 came back in 7 s with no gap in its forecasts; F7 and Esc behave; the frame rate in
the 70 s demo was 26.9 / 27.0 fps on, 31.7 / 28.0 off.

**Windows line endings.** Git for Windows checked the LightGBM model files out with CRLF line endings, and LightGBM
cannot read those ("Model format error, expect a tree here"): DSO4 crashed at every start, then switched itself off.
`.gitattributes` now keeps `models/goflow/**` byte for byte.

**Traffic volume.** `--demand-scale X` multiplies the day's vehicle curve (as the dataset tool's `--scale`). SUMO alone,
crashes on, 4 seeds x 30 min per level, queue build-ups (an approach going from under DSO4's `low` to over its `full`):

| Traffic | Build-ups per hour | Gridlock |
| --- | --- | --- |
| x1 | about 1 | no |
| x1.5 | 4.5 | no |
| x2 | 3.5 | no |
| x3 | 10.5 | no |
| x4 | 11 | no |

**Retrained (v3).** `D3_x1_acc`, `D3_x2_acc`, `D3_x3_acc`: crashes on (43, 85 and 124 incidents in 40 runs), seeds
101-130 train, 11-20 dev, 21-30 not opened; merged with `goflow_prediction.data merge`. Queue fill error at +120 s on
the dev seeds (MAE):

| Traffic | v3 | shipped v2 (x1, no crashes) | Last value |
| --- | --- | --- | --- |
| x1 | 0.0211 | 0.0210 | 0.0275 |
| x2 | 0.0331 | 0.0392 | 0.0493 |
| x3 | 0.0432 | 0.0587 | 0.0670 |

GO-FLOW's occupancy at +120 s: 0.0240 against 0.0334 for the last value. v3 replaces v2 in `models/goflow/` (v2 is in
git history, fd5e798). The bundle is 37 MB (v2: 11 MB): more varied traffic, larger trees.

**The alerts do not work yet, at any traffic.** DSO4 alerts when an approach still under its `low` (Q90 of the train
fill) is forecast at its `full` (Q99) or more. On the dev seeds no build-up was caught in advance by v2 or v3. Lowering
the trigger (forecast at 0.5 to 0.9 of `full`) only adds false alerts: at x3, catching 3 of 6 build-ups costs about
230 alerts an hour, 90 % false. The build-ups happen within one 30 s step and the inputs do not announce them. That is
the agent's design (its alert rule, or what it is asked to predict), for DSO4's author.

**On screen.** `cosim/goflow_prediction_hud.py`:
* a panel bottom right whenever the forecasters run (before: one small line): their state, the time left to the first
  forecast while they warm up, the three approaches most likely to fill up in 2 minutes (queue now and then, share of
  full, GO-FLOW's state), the last alert;
* with `--goflow-hud`, chips on the roads near you for approaches GO-FLOW calls congested or severe, or DSO4 forecasts
  at least half full;
* F10 flies to the last alert's approach, else the one most at risk.

In CARLA at x2 (7 Nov, 18:30): the panel, the chips and F10 work; 22.7 fps with 270 vehicles. DSO4 raised one alert,
and it was right: 1722-1 at 20 % of its queue space, forecast at 50 % within 60 s, at 49 % (over its `full`, 47 %) 30 s
later; that is the queue filling again at the next red, not a queue spreading back. First seen: seven chips round one
junction, several for 1-3 queued cars, one cut by the screen's edge, so now at most 4 (the most at risk), a queue of 3
vehicles or more, inside the picture; and "329 % of full" in the panel, now "full".

At x2 in SUMO, GO-FLOW's states over 12 minutes: 350 free, 336 moderate, 28 congested, 1 severe (approach-steps); the
approach most at risk was forecast at 0.68 of full. Also fixed: after F6 / F7 the status said "DSO4 ok" while the
agents warmed up again; a hand-written GO-FLOW model card hid the clock speed of the export metadata.

Tests: `tests/test_goflow_prediction.py` (5 more).

### Long queues instead of spill-back alerts (7 Nov, evening)

The first alert rule caught no build-up (above). Its event was a jump within one 30 s step, which is mostly a queue
filling again at the next red. `agents/goflow_prediction/goflow_prediction/congestion.py` now defines a long queue on
the fill averaged over 4 steps (about one signal cycle): it starts when that mean reaches the approach's level (Q95 of
the mean on the train seeds, at least 3 vehicles) after a whole cycle under it, from under 0.6 x the level. A LightGBM
classifier (DSO4's features plus the approach against its level) gives the chance of a start within 3 minutes, and
DSO4 alerts at 0.7 ("LONG QUEUE LIKELY WITHIN 3 MIN"); every forecast carries each approach's chance, which the panel
and the chips now rank by. Dev seeds 11-20, 5 h per level:

| Traffic | Long queues | Caught before they start | Alerts right | Median warning |
| --- | --- | --- | --- | --- |
| x1 | 6 | 3 | 3 of 5 | 30 s |
| x2 | 40 | 27 | 27 of 50 | 60 s |
| x3 | 136 | 101 | 116 of 227 | 60 s |

A simpler trigger (the cycle mean over 0.8 x the level and rising) catches about as many at x2-x3 but with 3 to 10
times the false alerts and half the warning (x2: 219 false against 23; x3: 328 against 115). The live code path (agent.replay) gives the same alerts as the training
script's scoring (x2: 50 alerts, 27 right, 27 of 40 caught). The test seeds 21-30 stay closed.

**Frame rate.** The panel's text was rendered again for every picture (1.4 ms at 1920 x 1080); it is now rendered when
what it says changes, and chips and pills are reused by their text (0.04 ms per picture). Also: a child's restart
messages no longer nest ("exited with code ...: exited with code ...").

Tests: `agents/goflow_prediction/tests` (2 more), `tests/test_goflow_prediction.py` (3 more).

**In CARLA (7 Nov, 19:20).** Frame rate in the 70 s demo, back to back: 29.4 fps with the forecasters (the cached
panel; 26.9 and 27.0 before), 30.2 without (earlier runs without: 31.7 and 28.0), so the gap is under the run-to-run
noise. At x2 (`--demand-scale 2`, 11:00 into the evening rush), six "long queue likely within 3 min" alerts in the first
5 minutes of forecasts; of the five whose 3 minutes had passed, four queues reached their level in time (2086-1,
1722-3, 207-4, 1126-2), one did not (2086-4). A chip read "SEVERE · QUEUE 11 · LONG QUEUE IN 3 MIN: 0 %" over a queue
that was long already (the classifier gives 0 there): each forecast now marks those (`long_now`) and their chip says
"LONG QUEUE NOW". Tests: 1 more on each side.

## A rear-end crash drawn one car inside the other; vehicles parked half on the road (7 Nov, night)

**The crash.** In CARLA (ring road, 45 km/h): `live301` and `hw13` came to rest with their centres 3.1 m apart, in
line, a metre of one car inside the other. From `out/freefly/crash_trace.csv`: at the hit they touched (4.5 m apart);
then `hw13`, behind and doing 72 km/h, moved on 1.4 m while `live301`, ahead at 27 km/h, hardly moved. SUMO had named
`live301` the collider and `hw13` the victim, and the shove of the impact (`_damage`: the victim is pushed on 0.25 to
1.45 m) went to the car behind. `_touching` already put the cars bumper to bumper by their places on the lane; now
`IncidentManager._order_rear_end` gives the roles by place too (the car behind hit; along the lane, else along the
heading) and logs it when SUMO said the other way. 2 x 60 min SUMO alone: 13 rear-end crashes, none the wrong way
round, so it is rare; when it happens, the car ahead is shoved on and the one behind stays.

**Parking at the depots.** With the depots and the emergency calls (SUMO alone, 40 min from 16:30, seed 7; vehicles in
the depots' parking lanes sampled every 10 s, cars on the depot's road waiting over 20 s counted):

| | Parked sticking into the next lane | Deepest | Cars stuck beside a depot (samples) | Longest wait |
| --- | --- | --- | --- | --- |
| before | 103 of 1546 | 0.96 m (hospital), 0.39 m (police) | 688 (police 616) | 1298 s |
| straightened when parked | 0 | | 836 (police 779) | 2034 s |
| + turn in / put in | 0 | | 591 (police 347) | 920 s |
| + buses too | 1 of 1640 (caught before it was straightened) | | 120 (at the red light at the road's end) | 44 s |

Three causes, all the town's rule that no vehicle moves sideways while standing (cosim/vmodels.py):
1. a vehicle back from a call stopped in its slot before it had finished moving over (up to 1 m off the lane's
   middle): `depots.park_straight` sets it straight in the lane's middle once stopped (depot slots and call places);
2. on the police station's 61 m road a police car reached its slot still in the traffic lane and stood there 0.9 m
   across: on the last road to its spot a vehicle may now move sideways at 0.5 m/s while standing
   (`laneChangeModel.lcMaxSpeedLatStanding`, back to 0 once it is parked), and one standing beside its spot for 15 s is
   put in (`depots.turn_in`);
3. bus line L2's first stop is in that same parking lane: the bus stood beside it, then across it pulling out with a
   police car parked ahead: `BusLines.turn_in` lets a bus move sideways while standing on the roads of its stops, puts
   one standing beside its stop in, and sets one standing across two lanes straight in its lane.

**The clock card** said "1 MIN = 1 HOUR" at any `--clock-speed`; it now says how fast the day goes ("2 MIN = 1 HOUR"
at 30, "REAL TIME" at 1).

Tests: `tests/test_crash_moment.py` (2 more), `tests/test_depots.py` (2 more), `tests/test_hud_speed.py` (new).

## The clock at 2 real minutes a town hour; the forecasters retrained for it (7 Nov, night)

`cosim/daily.GAME_SPEED` is 30 (was 60): a town hour lasts 2 real minutes, a day 48. At 60 the evening rush went by in
2.5 minutes, the forecasters' 6-minute warm-up took 6 town hours, and the police "came" 4 town hours after a crash.
`--clock-speed` still sets any other speed for a run (the stage-2 recordings set 1, real time). The clock card shows it
("2 MIN = 1 HOUR").

The forecasters run only at the speed they were trained at (their demand inputs look ahead in town hours), so they
were retrained at 30 (v4): `D4_c30_x1_acc`, `D4_c30_x2_acc`, `D4_c30_x3_acc` (crashes on: 59, 88 and 137 incidents in
40 runs each, no gridlock; made on the project PC, 8 jobs, about 35 minutes), seeds 101-130 train, 11-20 dev, 21-30 not
opened, the same recipe as v3. Dev seeds, all levels together: DSO4's queue fill at +120 s, MAE 0.0419 against 0.0553
for the last value; GO-FLOW's occupancy at +120 s, 0.0280 against 0.0348. Long queues, alert at 0.7:

| Traffic | Long queues | Caught before they start | Alerts right | Median warning |
| --- | --- | --- | --- | --- |
| x1 | 5 | 2 | 2 of 2 | 45 s |
| x2 | 48 | 31 | 31 of 49 | 60 s |
| x3 | 134 | 102 | 107 of 203 | 60 s |

The connector's default clock is now `GAME_SPEED`; the dataset tool and the live check default to it too. v3 (clock
60) is in git history.

Tests: `tests/test_daily.py` (the clock's speed).

### A fast car into a standing one, drawn inside it (8 Nov)

In CARLA (ring road): `hw7` at 86 km/h into `live31`, standing. From the crash trace, `hw7` was drawn hitting with
the two centres 0.4 m apart. At 24 m/s a car goes 12 m in a SUMO step; SUMO saw the crash at the step's end, `hw7`'s
front past `live31`'s middle. `_to_contact` takes each car back along its way to where they touched, but no further
than its lane's start, and `hw7` had come onto the lane in that step: it stayed where it was, and so did its drawn
pose. By lane position it was then the car ahead: the roles (`_order_rear_end`) and the bumper-to-bumper order
(`_touching`) came out the wrong way.

Now: the drawn pose at the hit is always where they touched (SUMO goes as far back as its lane allows); who is ahead is
decided from where they touched, not from SUMO's lane positions; `_touching` sets the follower (the collider) behind
the leader, and when the follower's lane has no room behind it, the leader goes on instead.

### The forecasters on the test seeds, under the junction agents (8 Nov)

Before feeding the long-queue alerts to the junction agents: the shipped v4 forecasters (clock 30) scored once on the
closed test seeds 21-30 (`tools/goflow_prediction_test_score.py`, lock `out/goflow_prediction/test_v4/TEST_OPENED.lock`),
through the live code path (GO-FLOW step by step, its forecasts fed to DSO4, DSO4's own `p_long`). Same recipe as the
training data (30 min after 5 min warm-up from 06:00, crashes, buses, ring), 10 runs per row. The scorer was first
checked on the dev seeds: it gives the dev numbers again (x2: 49 alerts, 31 right, 31 caught).

Queue fill, MAE (agent / "stays as now"); GO-FLOW occupancy at +120 s, MAE (GO-FLOW / "stays as now"):

| Signals | Traffic | Fill +60 s | Fill +120 s | Fill +300 s | GO-FLOW +120 s |
|---|---|---|---|---|---|
| coordinated | x2 | 0.030 / 0.043 | 0.039 / 0.054 | 0.046 / 0.075 | 0.028 / 0.036 |
| coordinated | x3 | 0.038 / 0.056 | 0.048 / 0.068 | 0.059 / 0.099 | 0.034 / 0.044 |
| agents_observed | x1 | 0.036 / 0.035 | 0.040 / 0.039 | 0.048 / 0.041 | 0.042 / 0.039 |
| agents_observed | x2 | 0.077 / 0.075 | 0.083 / 0.084 | 0.106 / 0.097 | 0.065 / 0.070 |
| agents_observed | x3 | 0.110 / 0.111 | 0.118 / 0.125 | 0.155 / 0.155 | 0.083 / 0.089 |

Long-queue alerts at the shipped threshold 0.7 (right = a long queue starts within 3 min; caught = an alert in the
3 min before it):

| Signals | Traffic | Alerts right | Long queues caught | Median lead |
|---|---|---|---|---|
| coordinated | x2 | 36 of 58 (62 %) | 36 of 52 (69 %) | 30 s |
| coordinated | x3 | 107 of 209 (51 %) | 100 of 131 (76 %) | 60 s |
| agents_observed | x1 | 1 of 2 | 1 of 11 | 60 s |
| agents_observed | x2 | 60 of 122 (49 %) | 60 of 172 (35 %) | 60 s |
| agents_observed | x3 | 142 of 344 (41 %) | 138 of 236 (58 %) | 60 s |

How often a long queue really started, by `p_long` band (0.7-0.9 / 0.9-1.0): coordinated x2 60 % / 84 %, x3 49 % /
76 %; agents_observed x2 52 % / 57 %, x3 43 % / 67 %.

Reading. Under the signals they were trained on (coordinated, all training data), the forecasters hold on new seeds:
the dev numbers again. Under the junction agents they don't: queue forecasts no better than "it stays as now" (worse
at +300 s), alerts right less than half the time, and at x2 two long queues in three missed. The agents switch on
demand rather than in a 115 s cycle, so the queues swing differently, and the classifier's cycle features (4 steps)
mean less; long queues also start about three times as often (172 against 52 at x2). Not fed to the junction agents
yet. Next: datasets under agents_observed (train seeds 101-130, dev 11-20) in the training mix, retrain, and score
on new test seeds (31-40: 21-30 are now open).

### v5: the forecasters retrained under the junction agents too (8 Nov)

After the test above: 120 more SUMO runs under `agents_observed` (train seeds 101-130, dev 11-20, x1 / x2 / x3, the
same recipe: `D5_ao_x{1,2,3}_acc`), merged with v4's three coordinated datasets, and the v4 recipe again on all six
(DSO4 LightGBM, GO-FLOW XGBoost, GO-FLOW's out-of-fold export, DSO4 + GO-FLOW, the long-queue classifier). One set of
models for both controllers. Then scored once on new test seeds 31-40 (21-30 are open), v5 and the shipped v4 side by
side on the same runs (`out/goflow_prediction/test2_v5`, `test2_v4`).

Queue fill, MAE at +120 s / +300 s (v4, v5, "stays as now"); GO-FLOW occupancy +120 s (v4, v5, "stays as now"):

| Signals | Traffic | Fill +120 s | Fill +300 s | GO-FLOW +120 s |
|---|---|---|---|---|
| coordinated | x2 | 0.040 / 0.041 / 0.054 | 0.048 / 0.049 / 0.077 | 0.029 / 0.030 / 0.036 |
| coordinated | x3 | 0.049 / 0.051 / 0.068 | 0.060 / 0.061 / 0.098 | 0.034 / 0.036 / 0.044 |
| agents_observed | x1 | 0.039 / 0.029 / 0.038 | 0.047 / 0.029 / 0.041 | 0.041 / 0.031 / 0.038 |
| agents_observed | x2 | 0.081 / 0.063 / 0.083 | 0.103 / 0.064 / 0.097 | 0.064 / 0.053 / 0.069 |
| agents_observed | x3 | 0.117 / 0.094 / 0.124 | 0.153 / 0.098 / 0.151 | 0.083 / 0.070 / 0.090 |

Long-queue alerts at 0.7, v4 then v5 (each against its own levels: v5's are the Q95 of both controllers' traffic, so
the long queues counted differ a little):

| Signals | Traffic | Alerts right | Long queues caught |
|---|---|---|---|
| coordinated | x2 | 35 of 60 / 38 of 76 | 35 of 46 / 38 of 59 |
| coordinated | x3 | 110 of 204 / 109 of 216 | 103 of 133 / 101 of 129 |
| agents_observed | x2 | 62 of 115 / 87 of 213 | 62 of 173 / 86 of 116 |
| agents_observed | x3 | 133 of 346 / 269 of 610 | 126 of 247 / 240 of 285 |

At 0.8, v5 under agents_observed: x2 64 of 124 right, 63 of 116 caught; x3 225 of 385 right, 212 of 285 caught.
`p_long` runs high: of the v5 approaches given 0.7-0.9, 43-54 % started a long queue within 3 min; 0.9-1.0: 70-100 %.

Reading. Under the junction agents v5 forecasts the queues 25-35 % better than "stays as now" at every horizon (v4:
no better), and catches three long queues in four (v4: one in three to one in two), with a minute's warning; about 2
alerts in 5 are right at 0.7. Under the coordinated signals it is as good as v4. Shipped as v5 (`models/goflow`,
model_card.json has the test numbers); the live connector runs it (SUMO under agents_observed, both agents ready).
For the junction agents: a weight that grows with `p_long`, not a switch, since 0.7 means about 45 %.

### The dataset tool ran agents_observed blind; the agents read the forecasts (8 Nov)

**A bug in the datasets.** `tools/goflow_prediction_dataset.py` made the town as the viewer does but never connected
the junction agents: the viewer gives `agents_observed` the cameras' reports and the emergency vehicles' requests
(`FreeFly._wire_agents`), the dataset tool gave them nothing. Its `agents_observed` runs (`T_*`, `D5_*`, `T2_*` above)
were agents on no counts and no requests: the actuated program plus a skip from the loops. So v5 was trained and
tested on that, not on the agents the viewer runs. Now one function connects them for both
(`cosim/agents_actuated.wire`), and the dataset records `agents_wired` and the agents' counters per run. A check run
(dev seed 11, x2, 10 min) now shows them acting: 95 greens skipped, 34 ambulance jumps, 2 balance cuts. With the
cameras a run takes about three times as long.

**The agents read the forecasts** (`+forecast`, optional; `agents_observed+forecast`). `GoflowPrediction.risk` (and
`goflow_prediction.long_queue_risk`) gives per approach DSO4's `p_long`, 1 where the queue is long already, nothing
when the last forecast is older than 45 s. Each car green is matched to the approaches it serves (their lanes among
the green's in-lanes: 54 greens). The forecast acts only through two of the agent's own moves, as a weight:
* balance: a phase's fill counts `forecast_w` (0.5) x its risk more: a green whose road is about to queue is not ended
  for being under `balance_low`, another phase about to queue reaches `balance_high` sooner;
* at a green's maximum: with risk `forecast_extend_p` (0.8) or more, above every other green's, exits free and nobody
  waiting to walk elsewhere, 5 s more at a time to 90 s.
In the viewer: `--controller agents_observed+forecast --goflow-prediction --goflow-traffic`. A first check (dev seed 11,
x3, 10 min): 3 balance cuts from the forecast; a green reached its maximum with a high risk twice, both times with
another green as much at risk (not extended).

`tools/eval_agents_forecast.py` measures it: the forecasters' recipe, the agents wired, both forecasters in the same
process with the live agents' code (so two runs of a seed differ only by what the agents did), paired seeds; waiting
and time loss per trip, throughput, ambulance time loss, and long-queue approach-minutes.

### v6, and the agents with and without the forecasts (8 Nov)

**v6.** The v5 recipe again with the agents wired: 120 runs under `agents_observed` (`D6_ao_x{1,2,3}_acc`, every run
`agents_wired`), merged with v4's three coordinated datasets. Scored once on new test seeds 41-50
(`out/goflow_prediction/test3_v6`; 31-40 were v5's). Queue fill MAE (v6 / "stays as now") and long-queue alerts at 0.7:

| Signals | Traffic | Fill +120 s | Fill +300 s | GO-FLOW +120 s | Alerts right | Long queues caught |
|---|---|---|---|---|---|---|
| coordinated | x2 | 0.039 / 0.051 | 0.046 / 0.072 | 0.028 / 0.034 | 25 of 49 | 25 of 36 |
| coordinated | x3 | 0.047 / 0.063 | 0.057 / 0.092 | 0.033 / 0.040 | 75 of 162 | 70 of 89 |
| agents_observed | x1 | 0.031 / 0.041 | 0.031 / 0.043 | 0.033 / 0.040 | 0 of 2 | 0 of 3 |
| agents_observed | x2 | 0.066 / 0.088 | 0.068 / 0.101 | 0.055 / 0.071 | 101 of 235 | 100 of 146 |
| agents_observed | x3 | 0.097 / 0.126 | 0.104 / 0.151 | 0.072 / 0.091 | 285 of 622 | 252 of 293 |

Of the approaches given 0.7-0.9, 44-47 % started a long queue within 3 min; 0.9-1.0: 68-100 %. Shipped as v6
(`models/goflow`).

**The agents with the forecasts** (`tools/eval_agents_forecast.py`, SUMO only, the forecasters' recipe: 30 min after
5 min warm-up from 06:00, crashes, buses, ring road, clock 30; v6 in the loop). Agreed before the runs: kept on only if,
on held-out seeds, waiting and long-queue minutes are lower on at least 7 of 10 seeds with no loss for the ambulances.

Dev seeds 11-20 (`out/agents_forecast/dev`), paired against `agents_observed`, mean difference (seeds better):

| Traffic | Setting | Waiting /trip | Throughput /h | Ambulance time loss | Long-queue approach-min |
|---|---|---|---|---|---|
| x2 | w 0.5, extend from 0.8 | +1.5 s (2/10) | -27 (4/10) | -5.2 s (7/10) | +5.0 (5/10) |
| x2 | w 1.0, extend from 0.7 | +0.6 s (3/10) | -1 (3/10) | -2.6 s (6/10) | +0.5 (5/10) |
| x3 | w 0.5, extend from 0.8 | -1.6 s (5/10) | +57 (7/10) | -5.8 s (6/10) | -12.6 (8/10) |
| x3 | w 1.0, extend from 0.7 | -1.2 s (5/10) | +45 (5/10) | -1.7 s (6/10) | -4.4 (7/10) |

Kept the first setting. Final, new seeds 51-60 (`out/agents_forecast/final`), reference 88.1 s / 134.0 s waiting:

| Traffic | Waiting /trip | Time loss /trip | Throughput /h | Ambulance time loss | Long-queue approach-min |
|---|---|---|---|---|---|
| x2 | -0.2 s (6/10) | -0.2 s (6/10) | +3 (6/10) | -1.4 s (4/10) | -1.2 (5/10) |
| x3 | -4.2 s (6/10) | -5.2 s (6/10) | +112 (8/10) | -0.2 s (6/10) | -1.4 (6/10) |

What the agents did with it (final, per 35-min run): x2 36 greens extended, 4 greens ended for a road forecast to
queue; x3 71 and 11. A seed's waiting changes by -19 s to +12 s at x3 (sd 10 s), -6 s to +6 s at x2.

Reading. It does not meet the bar: off by default. At x2 it changes nothing. At x3 it goes the right way on both seed
sets (waiting -1.6 s and -4.2 s, throughput +2 % and +4 %, 15 of 20 seeds better on throughput), but by less than a
seed's spread. The agents' moves are few and gentle (they keep the actuated program's own order and limits), and a
warning a minute ahead, right about half the time, gives them little to act on that their counts do not already show.
Left in as an option (`agents_observed+forecast`) for heavy traffic, with the evaluation to repeat it.

## The route agent: reviewed, made to drive a car you follow, measured in SUMO (8 Nov)

**The branch.** `integration/route-recommendation-town05` (Emna, 1-8 Nov): agent 6, a route recommender. 15 new files
in `agents/route_recommendation`, nothing else changed in the project, merges with `main` without conflict. What it
was: a graph of Town05's junctions from `town05.net.xml`, A* and up to 3 different routes (less than 70 % of roads in
common), costs = free-flow time (length / limit) times the alerts' penalties (DSO4's forecast fill at +120 s over 0.3:
1 + 2 x fill; an accident: up to x50), a LangGraph pause for a person to choose, Qwen2.5-7B (Ollama) to explain, a
Streamlit page with test buttons. No model, nothing trained, its tests were scripts that print.

Measured on it (SUMO 1.27.1, Town05):
* one link per pair of junctions: of the two parallel ring-road roads 35 and 37 (DSO4 approaches 2328-1, 2014-2)
  only the other was kept, so an alert on them was put on the road next to them;
* the turns were not the network's: 9 of the 1518 routes it offered over all 506 pairs of junctions needed a turn
  the junction does not have; from a car's own first road to its last, 1065 of 2421 of its routes could not be
  given to the car;
* free-flow times leave out the signals: on 2326 car trips (coordinated signals, x1, seed 11), its estimate had a
  median of 24 s, the real trips 183 s (118 s of it waiting), 6.4 times;
* `step8_eval_grid.py` (52.8 % faster, 91/91) graded it on its own penalties with the alerts put on the route it
  avoids: not a measure.

**Her agent, fixed** (`agents/route_recommendation`):
* `router.py` (new): the graph's points are the roads (80) and its links the network's own lane connections for
  cars; Dijkstra, and Yen's algorithm for the 3 different routes; only sumolib, so the viewer imports it. Every best
  route between any two roads is drivable (6320 pairs, all reachable); the alert on 35 stays on 35.
* `adapter_traffic.py`: `nodes.json` from the working copy (it was read with `git show origin/main:`).
* `step5b_agent.py`, `step6_hitl.py`, `app.py` on `router.py`; the times said to be "without waiting at signals".
* `tests/test_router.py` (pytest, 9 tests) instead of the print scripts; `sumo_map.py`, `grid_map.py`,
  `step8_eval_grid.py` removed; requirements pinned (as tested, Python 3.11 and 3.13);
  `tools/setup_route_recommendation.bat`, `tools/route_recommendation_app.bat`.

**Typical times from past trips** (`tools/route_delay_table.py`, `models/route_guidance/movement_times.json`). What a
navigation service learns from its users' trips: the time a car spends on a road by the road it takes next, the wait
at the signal included. From the train seeds only (101-110, x1 / x2 / x3, the forecasters' recipe under
`agents_observed`, crashes on): 282 movements, 120 303 road times (`--vehroute-output.exit-times`; a trip's first road
and cars whose route was changed on the way left out). Their median is 6.9 times the free-flow time. Used by the
agent for its estimate, and (option) as its costs: `Router.set_typical`.

**Following a car in the viewer** (`cosim/route_guidance.py`, `route_minimap.py`, `pose_proc.ChaseCam`; keys F, V, M):
* F near a car (35 m, in front): the camera flies from where it is to behind the car (1.8 s, smootherstep, looking at
  the car all the way, on a raised arc) and follows it; V glides between three views (behind, roof, drone); a movement
  key or F lets go. The camera's pose is worked out in the pose process for the same picture as the car's and sent in
  the same batch.
* The agent (what a navigation service would have: map, the car's own trip, DSO4's forecasts, crashes once
  reported, the typical times) gives the car its route, plans again when the alerts change, switches for 5 s and 10 %
  better. `keep_driving` leaves the car alone until it arrives. Each trip goes to `out/route_guidance/followed.csv`.
* The mini-map, bottom right: flying, the town live (every car as a dot by its speed, the camera's view, the alerts,
  the car F would follow); following, the map eases to the trip, the camera's marker slides onto the car, the routes
  draw themselves, the text unfolds: A and B, its own route dashed and the agent's green, the next signals in its
  colour, time on the way, the agent's estimate, metres left, the decision and why, the options, what it sees; on
  arrival the real time beside the estimate. The forecast panel moves above it while flying.
* `--route-costs typical` (default, from the A/B below) or `free` (as built); `--route-explain`: a sentence from the
  local Qwen per decision in a thread of its own (off when Ollama does not answer: it did not on the PC today).

Checked in CARLA (`tools/freefly_follow_demo.bat`: 08:00, a town car followed, a breakdown staged on its way, the
views cycled, screenshots `out/freefly/town05_follow_*.png`, `out/town05_freefly_follow.json`), third run: live24
followed until it arrived, 2:44 real against 4:15 estimated (1:07 free flow); the camera against its place on the car,
every half second: median 0.18 m behind (269 samples), roof -0.01 m, drone 0.13 m (the largest differences are the
fly-in and the view changes); CARLA 36-47 fps while following. The staged breakdown (live70 on -0) did not change its
route: the agent kept it. First run found two bugs, fixed: the mini-map's view took in the car's old route (zoomed out
past the town) and a car picked right after an arrival read the last car's snapshot.

**SUMO A/B** (`tools/eval_route_guidance.py`): 30 % of the town's cars (a hash of id and seed, the same in every
variant) routed as they set off and, for the agent, again when the alerts change; the forecasters' recipe, junction
agents (`agents_observed`) wired, both forecasters in the loop in every variant. Marked cars' delay = trip time minus
the free-flow time of the shortest route between their first and last roads. Dev seeds 11-20, x3, paired against
`none` (mean difference, seeds better):

| Variant | Routes changed at start / later | Marked delay /trip | Marked trip | All: waiting /trip | All: throughput /h | Ambulance time loss | Long-queue approach-min |
|---|---|---|---|---|---|---|---|
| none | 0 / 0 | 156.1 s | 213.6 s | 140.8 s | 2561 | 107.4 s | 185.5 |
| shortest (free-flow, at start) | 1628 / 0 | -7.8 s (9/10) | -8.8 s (7/10) | -4.9 s (6/10) | +80 (7/10) | -9.9 s (5/10) | -14.6 (8/10) |
| agent, as built (free-flow x penalties) | 1698 / 326 | -4.8 s (8/10) | -5.7 s (6/10) | -1.5 s (4/10) | +36 (5/10) | +8.6 s (4/10) | -8.2 (7/10) |
| agent, typical times x penalties | 2945 / 866 | -27.9 s (10/10) | -24.5 s (10/10) | -13.9 s (10/10) | +177 (10/10) | -17.5 s (6/10) | -18.9 (9/10) |

The agent's estimate of a marked trip as it set off, against the trip (mean absolute error / mean real minus
estimate): free flow 176-181 s / +56 to +145 s; with the typical times 67-90 s / +9 to +38 s.

Reading. As built, the agent does less than plain shortest routes (its penalties multiply a free-flow time that is a
sixth of the trip, and the ambulances lose on 6 of 10 seeds). With the typical times it is better on every seed for
its own cars and for the whole town (waiting -10 %, throughput +7 %), and its estimate is two to three times closer.
Dev seeds only: the test (seeds 61-70, x2 and x3, opened once) waits for the bar to be agreed; proposed: marked delay
lower on at least 7 of 10 seeds, all-car waiting not worse, ambulance time loss not worse on the mean.

## Trips with real destinations: the town's places (8 Nov, evening)

Until now every car drove a random benchmark route from a town edge to another. Step 1 and 2 of giving trips a
meaning, and the start of step 3 (show it).

**Step 1, the places** (`tools/town05_places_probe.py` in CARLA, `tools/town05_places_build.py`,
`scenarios/town05/places.json`, `town05.places.add.xml`). From CARLA's own buildings and a top-down picture: 24 places
(a market, a mall, 2 office towers, a café, 2 restaurants, a park, 4 apartment blocks, 2 streets of houses, a hospital,
police, a garage, 3 car parks, 4 streets with kerb parking). 16 SUMO parking areas: 366 spaces on CARLA's painted bays
(cars park exactly on them: within 0.1 m, `tests/test_places.py`) and 138 kerb spaces. `check()`: lanes exist,
spaces off the driving lanes (3 m+), inside their outline, no overlap, none on a bus stop, every door reachable: 0
problems. There is no free ground for a restaurant's own car park: the Brasserie du Nord parks at the Parking du
Square across its street. The atlas: `docs/img/town05_places.jpg` (`tools/town05_places_atlas.py`).

**Step 2, trips with a purpose** (`cosim/purposes.py`). Most new trips go from a home to a place for a reason that
follows the hour (work, shopping, eating out, the park, the hospital, errands; visitors and deliveries from out of
town), park in a space there, stay (work about 8 town hours, shopping about 1) and drive home, where the car stays
until a trip from that home takes it out again. Closed places are not chosen (the market 7-14 h...). A full car park:
the nearest one with room. 20 % of new trips stay through traffic on the old routes. At a run's start the car parks
and homes are filled as a measured day has them at that hour (`scenarios/town05/parked_by_hour.json`, from seed 101,
a training seed: `tools/check_purpose_day.py --table`). Used by the viewer (`--trips purpose`, the default) and the
datasets (`one_run(trips="purpose")`).

Measured, SUMO alone, 24 town hours from 05:00, seed 101, same load (53.7 against 53.1 cars on the roads on
average), `tools/check_purpose_day.py`:

| | purpose | random |
| --- | --- | --- |
| trips ended | 672 (in a parking space) | 647 |
| km per trip | 0.44 | 0.56 |
| waiting, s per km driven | 131 | 108 |
| time lost, s per km | 179 | 148 |
| cars stuck over 5 min, gridlock | 0, no | 0, no |

Shorter trips between places in the centre: more of each km is spent at signals and pulling in and out of spaces.
The day: `docs/img/town05_day.png` (`tools/purpose_day_chart.py`): the morning to work, lunch, the afternoon's shops,
offices full at midday and empty at night, the homes full at night.

Fixed on the way: a car's home stop set at the start merged with its first passage (it parked at home for ever: the
home stop is now set once it has parked at its destination); `IncidentManager.unjam` took parked cars for knots (it
skips them now); the incidents' `type_for` marked cars that were never put in (11 184 "not known" errors a day, now
a handful, the old ones from the ring road); a car put in at home could be taken for a trip before SUMO had it in
its space (SUMO quit: it now waits for its space).

**Step 3, started: show it.** The route agent's mini-map draws the places (their buildings, an icon each, a car
park's ring green / amber / red by its free spaces, names when close), the trip's A and B as places, a header with
the purpose ("SHOPPING  Résidence du Lac -> Marché Central") and "PARKS AT Parking du Marché, 31 free of 43"; flying,
"N DRIVING · N STANDING · N PARKED · hh:mm" (parked cars are no longer red dots). Over the CARLA picture
(`cosim/place_hud.py`, key L): a label on a stem over each place near you, its icon and name, and closer what is
happening there (a car park's free spaces, OPEN until 14:00 · 9 here, CLOSED, cars home). Tests:
`tests/test_purposes.py` (6). Not yet checked in CARLA.

**Checked in CARLA (8 Nov, night).**
* The follow demo (`tools\freefly_follow_demo.bat`) followed live116 "TO WORK, Résidence Les Oliviers -> Tour
  Horizon": rerouted (45 s shorter), parked, arrived in 2:48 against 2:40 estimated; "PARKED AT Tour Horizon, 34 free
  of 40". The labels over the town are kept off the clock and the mini-map. 33-40 fps.
* Cars on the painted bays (`tools\freefly_places_demo.bat`, `--demo-places`: each car park seen from above at 12:30,
  and how far each parked car's CARLA body is from the middle of the nearest space). Before: 2.09 to 2.79 m off on
  the Parking du Marché and the Parking du Square (median 2.36 m, 13 cars): SUMO puts a parked car's front on the
  space's point, and the viewer drew the body half a length behind it. The viewer now moves a parked car forward by
  half its length: 0.0 m off for all 15 parked cars seen, and they stand in their bays in the screenshots.
* "78 standing out of 106" on the mini-map was not a jam. Counted by why every minute
  (`FreeFly.standing_why`, logged): at 12:40, 65 standing of 107: 44 before a red or amber signal, 9 police /
  ambulance / recovery vehicles waiting at their stations, 9 on the ring road, 2 buses, 1 queued: 41 % of the cars at
  a red signal. SUMO alone (`check_purpose_day.py`, seed 101, 12:10) has 28 of 66 cars standing (42 %, for any reason). The
  mini-map now says it: "MOVING · AT RED · QUEUED · PARKED" (the stations' vehicles counted as parked).

## Cars that drive into their space and out of it (8 Nov, night)

SUMO keeps a parked car in its space without driving it there. Traced on the Parking du Marché: the car stops on the
road beside the car park and is in its space at the next half-second step, 42 m away; when it leaves, it is back on
the road in one step. The viewer copied SUMO: cars vanished from the road and appeared in their bays.

`cosim/parking_motion.py` now drives them, in the viewer only (SUMO still decides when a car parks and leaves):

* **in**: from where SUMO stopped it, 9 m on along the road, a turn into the car park (blinker), along the strip
  inside its road side, down the aisle of its space, a turn into the space, brake lights as it stops. The aisle is
  found from the space itself (half a space and half an aisle behind it, running across it). About 25 s for the far
  spaces of the Parking du Marché;
* **out**: reversing lights, backing out into the aisle, a stop, then along the aisle and the strip to the road,
  where SUMO puts it back. When its stay ends the drive starts that much earlier; a car taken out of its home space
  for a trip waits in its space for its drive (`PurposeDemand.exit_delay`, `resume_at`). If SUMO cannot put it back
  yet it waits at the road; if SUMO puts it back first, SUMO's car waits (`setSpeed(0)`) and the drive's end moves to
  where SUMO's car is;
* **at a kerb**: past its place, then backing into it; out: a little back, then out onto the road;
* a stay shorter than the drive in and out: SUMO's stop is made that much longer (3 in the hour below); a car taken
  out again before its drive in was over turns back to SUMO's car (2);
* the heading is given, not worked out from the path (`interp.Trails.fixed`, `pose_proc`, `run.Poser`): a car backing
  out faces the right way, and a parked car keeps SUMO's heading;
* the spaces inside buildings are not driven to (the car would go through a wall): those cars still vanish at the
  door.

Measured, SUMO alone, as the viewer does it (`tools/check_parking_motion.py`, seed 101, 07:30, 1.5 town hours):
4 drives in and 7 out of the car parks with painted bays, 18 in and 11 out at kerbs; the largest move between two
half-second steps on a drive 2.84 m (none of SUMO's 42 m jumps); a drive out joins SUMO's car within 0.07 to 1.9 m
(median 0.57 m, 17 drives). `tests/test_parking_motion.py`: in and out of the Parking du Marché, no step over 2 m on
the drives, ending in its space (within 5 cm), back on the road where SUMO has it. The way in and out, from above:
`docs/img/town05_parking.png` (`tools/parking_motion_figure.py`).

In CARLA (`tools\freefly_parking_demo.bat`, `--demo-parking`: the camera over the car park of a drive that has just
started, three screenshots): cars backing out of the Parking du Square and the Parking du Marché and leaving; the
CARLA body follows the drive (its position read back from CARLA at each screenshot); drives out ended within 0.62 m
of SUMO's car. The camera's view is chosen with CARLA's ray casting (`place_view`): from 55 deg it had been inside the
building beside the Parking du Square.


## Cars on drives that see each other, barrier arms, and the car parks inside buildings (9 Nov, night)

What was left after the drives into the spaces: the cars on drives did not see each other, the car parks inside
buildings were not driven to, and the kerb drives, the turn-back and the heading at the end of a drive had not been
seen in CARLA.

**The cars see each other** (`cosim/parking_motion.py`: `reserve`, `give_way`, `_hold_followers`, `_merge_gap`):
* a new drive in a car park starts when its whole way is clear of the drives already on theirs (each as it will be at
  the same moment, half-second steps): a car going out waits in its space, one coming in on the road where SUMO
  stopped it (`started_later`);
* on its way a car looks 3 s ahead: a car in front of it (another drive held up, a car of the road) makes it slow,
  and stand if it is within a second; two drives each in the other's way: the first one goes. It gives up waiting
  (40 s) only on another drive that is not itself waiting on the road's traffic;
* along the strip and down the aisles cars keep 1.3 m to the right, so two meeting pass; a car backing out ends on
  the right half of the aisle for the way it will drive off;
* SUMO's cars on the lane behind a car turning in, or about to join the road, are stopped (`setSpeed(0)`), as behind
  a car waiting to turn into a car park; a car going out pulls onto the road in a gap (none within 10 m or 4 s
  behind, no queue standing where it joins);
* found on the way: the stop's duration SUMO gives for a car in its space is what is LEFT of it; `arrival + duration`
  put the end too early, the drive out reached the road before SUMO put the car back and waited there (14 of 18 in
  the first check). Now `now + duration`; a drive out held up makes SUMO's stop that much longer.
* the kerbs keep the drives as they were (a slow parking lane: holding its cars made it worse).

`tools/check_parking_crowd.py`: cars sent to one car park one after another, SUMO alone, how many half-second steps
two bodies (three circles each, 1.9 m wide) overlap by more than 0.2 m:

| Car park, cars, every | before | now |
| --- | --- | --- |
| Marché, 8, 4 s | 720 | 0 |
| Square, 6, 5 s | – | 0 |
| Square, 8, 4 s | – | 14 |
| Marché, 12, 2 s (more than the car park takes) | – | 8 |
| Square, 12, 2 s | – | 45 |

`tools/check_parking_motion.py` (the town, purpose trips) now also counts them, by kind:

| Run | before: car parks / kerbs | now: car parks / kerbs |
| --- | --- | --- |
| seed 101, 07:30, 1.5 h | 160 / 82 | 0 / 109 |
| seed 102, 12:00, 1.5 h | 748 / 8 | 85 / 4 |

What is left at noon: the Parking du Square's exit joins its road 13 m before a signal; cars leaving wait there
for the queue (up to 2 minutes), and a few drives that started while another was waiting cross it. Drive steps:
largest move between two steps 2.84 m; drives out join SUMO's car within 0 to 1.5 m (median 0.4 to 0.6 m).
`docs/img/town05_parking_crowd.png` (`tools/parking_crowd_figure.py`): 8 cars at the Marché, three moments.

**The Parking du Square's gate.** Its cars drove in over the hedge at its north-west corner. CARLA has a gate there
(two barrier arms, `SM_ParkingBarrier`, and the guard huts), on the road beside it: its SUMO stop moved before the
gate (lane 49_3, 27-39 m) and the car put back after it (`departPos` 61.5, a parkingArea attribute SUMO has: checked,
the car came back at 38.6 for 38.0 asked). In by the south arm, out by the north arm. The viewer opens a barrier arm
(and a building's door) while a car passes: the arm's mesh hidden (`world.enable_environment_objects`), shown again
after (`ParkingMotion.gates`, `door_states`, `FreeFly._doors`).

**The car parks inside buildings** (`tools/town05_entrance_probe.py`, `tools/town05_pillars_probe.py`,
`tools/town05_entrances_build.py`, `scenarios/town05/entrances.json`). Looked at in CARLA: around each building, rays
every 0.6 m, straight down (the ground, what covers it) and sideways at 0.5 m and 1.2 m over the ground (what stands
in a car's way: a vertical ray misses a wall under a roof), and photographs from the street. Found:
* **Parking Central** is a real multi-storey car park: an open ground floor between 25 square pillars, low walls on
  the south, east and west, open on the north to the Parking du Square's ground, whose gates are the way in. CARLA's
  ray casting does not meet its pillars (no collision): a depth camera under the first floor, 25 pictures, found them
  (`silo_pillars.json`, 5 x 5, about 12 m apart). Two jeeps CARLA keeps parked for good stood across the only gap
  between the guard hut and the pillars: the viewer takes them away (`entrances.json` "removed").
  The way in (26 m): from the road on its east (lane -47_3, the car park's SUMO stop moved there: 0-8.3 m, back on the
  road at 26.3 m), through the east gate's barrier arm, along the Square's ground, between the corner pillar and the
  next, 7 m under the first floor; there the car is out of sight and no longer drawn.
* **Galerie Atlas**: its metal shutters, hidden, show a blind behind them, no room (the rays meet it 0.8 m in): no way
  in. The towers (Horizon, Jumelles) and the apartment blocks (Oliviers, Lac): no opening at all. Their cars appear and
  vanish at the road beside them, as before; they are no longer drawn inside the walls.
The builder: a car's way (1.1 m from anything standing) from the rightmost lane of a road beside it, on the road for
10 m at most, to a covered place 7 m from open sky; doors with room behind them and barrier arms count as open.
Tests: `tests/test_entrances.py` (a made-up building: round to its open side; through a door; none).

**In CARLA** (`tools\freefly_parking_demo.bat`, 9 Nov 01:42-01:52, from 08:30; `out\town05_freefly_parking.json`,
screenshots `out\freefly\town05_parking_0*_*.png`): one drive of each kind seen.
* Into Parking Central (11:16, staged: half the new trips sent there until one came): the barrier arm hidden as the
  car passes, shown again after; the car turns in from the road, crosses the Square's ground where the jeeps were and
  drives in between the pillars; once inside it is not drawn (`drawn: false`). Out (12:53): a van appears between the
  pillars, drives out through the gate (open while it passes) onto the road.
* Two cars in the Parking du Square at once (10:04): out through the north barrier arm (open), the other car 6 to 28 m
  away all along.
* Into a space at the Marché (11:57): once in, its CARLA body is on the drive's last point (0.0 m) and its heading is
  the space's (0.0 deg off): the sharper heading at the end, seen.
* Kerb in (08:49) and out (10:21): pulls past, backs in; out a little back, then onto the road.
* The turn-back (09:17, staged: SUMO takes the car out again 40 % of the way into its kerb space): it went to SUMO's car
  in a straight curve, through the car parked behind (screenshot 02_turned_back_2). Changed: it drives back the way
  it came (`_retrace`), then to SUMO's car. Seen again (`tools\freefly_turnback_demo.bat`, 01:55): at a kerb (08:51)
  it backs out of its place along the way it came and joins SUMO's car; at the Marché (09:12) it backs along the strip
  it came by, clear of the parked cars (screenshots kept in `Claude outputs\run1` for the first run).
CARLA's body was 0 to 4 m from the drive's point while moving (the drive's point is SUMO's next step; the picture
shows the step before), 0.0 m when standing. Drives out joined SUMO's car within 0.2 to 0.65 m, one 4.3 m.

Left: two drives out at a kerb (of 20 in the two CARLA runs) ended 4.3 and 5.3 m from SUMO's car: SUMO had put the car
back before the drive's end and it rolled on while held (`setSpeed(0)`), after the drive's end had been fixed. The
kerbs keep their drives as before for the cars of the road (see above); the turn-back at a car park backs a long way
(it retraces its way in).

Found at 02:20 (seed 102, 12:00, which drive, which leg, against what; for the next session):
* **The back-out turn is too sharp.** `_out`'s back leg is `fillet([bay_c, P, back_end], r_max=3.5)`, and the fillet
  takes at most 45 % of the 3.5 m after P: a 1.6 m radius. The car's middle turns 90 deg round P almost on the spot,
  its corners sweep about 2.5 m round it: into the noses across a narrow aisle and into the neighbours (the Square's
  static jeep at 169.6, 264.8 and Seat Leon at 172.4, 264.4; its trucks at 186.9 and 191.2, 266.0). To do: drive the
  back-out and the turns as a car does (its rear axle on a path of at least about 4.5 m radius, the body following),
  backing straight until the nose clears the neighbours; in an aisle too narrow for that (the Square's, 4.3 m) a
  second move (forward a little, back again) instead of turning on the spot; test every drive's swept body against
  the parked cars and `parking_obstacles` before it is used, and pick another manoeuvre when it touches.
* **Poles.** A highway light's box (`BP_HighwayLight_simple2_SM_2`, 7.9 m long) is its post and its arm; the post is
  the end away from its lamp head (`..._SM_0` at 10.9 m): (172.5, 252.6) and (195.5, 252.6), in the barrier line,
  not in the aisle. `parking_obstacles.load` should put a post there (radius 0.25), not the arm's box; the kerbs'
  tall lamp posts the same (their boxes are 2 m wide).
* **The Square's gate**: a board on wheels (`SM_TrafficCones_03_611`, 162.0, 257.7) stands just inside the south
  arm, where the cars come in: in their way.
* **Kerbs**: drives pass pines (`TreePine_*`, trunk at the box's middle) and lamp posts on the pavement edge (Allée
  des Pins, Allée des Cyprès, Rue du Sud); check which kerb places have one beside them and keep the drive's swing
  off it (or leave that place empty).

## 9 Nov (3rd, 05:15-06:10): parking drives without going through anything

The user saw the cars glitch while parking. Measured first (`tools/check_parking_motion.py`, which now also counts
the steps a drive's body is inside a car standing in its space, `parked_hit_steps`; before, those were not looked at),
then changed `cosim/parking_motion.py`:
* **Turns a car can make.** `arcs()`: each corner a circular arc (a car's radius, 4.5 m where there is room); the
  fillet's curves were tighter than their size (the back-out turned on 1.2 m). Backing out: straight until the nose is
  out, a turn of 3 to 5 m, 0.8 m straight on along the aisle.
* **Every way tested before it is used** (`_sweep`, `_pick`): the body every half metre, 0.15 m kept, against
  CARLA's map near the car parks and kerbs (`scenarios/town05/parking_obstacles.json`, 1044 boxes from
  `cosim/obstacles.py`, written by `tools/town05_entrances_build.py`) and the cars standing in their spaces; the first
  of up to 11 ways (lane offset, radius) that touches nothing and turns no tighter than 3 m is taken.
* **The Parking du Square**: its own ways in and out (`places.json` "way_in"/"way_out"): in by the south gate, out by
  the north one, through the middle of each, between the barriers inside them; its aisle's middle from its two rows
  (4.7 m from a space, not 5.6). Three of its painted bays had a CARLA truck or container in them: dropped (14 left).
  Two of the Marché's moved 0.6 m back off a lamp post at their head.
* **CARLA objects taken away** (`entrances.json` "removed", hidden by the viewer): the Marché has a low fence (0.15 to
  1 m) all along its road and no opening there; its cars drove through it. Six pieces taken away: two openings where
  they go in and out. A group of cones just inside the Square's way in.
* **Kerbs, as one parks along a pavement.** SUMO stops the car on its lane beside its place and puts it back there.
  The viewer now shows the parked car 6.5 m further on (`kerb_fronts`: less near the lane's end, 6 m between two, a
  metre or two off a tree or a post): the car stops behind its place and pulls in forwards (an S), the cars behind it
  waiting while it is on the lane. To leave, SUMO's car is moved 8 m past the place (`vehicle.moveTo`) once that
  stretch of lane is clear, and held there; the shown car pulls out forwards (an S about its rear axle) and meets it
  there: no seam. SUMO puts a kerb car back only when its way out is clear (its stop made 2 s longer, or a trip's
  resume put off by 1 s, until it is).
* **Taken out on its way in** (`_then_out`): it goes on into its space, stands 2 s and drives out (SUMO keeps it
  parked meanwhile), instead of backing all the way along the way it came. At a kerb SUMO's car is held where SUMO put
  it until the car has parked, then out as any. The turn-back (still used when SUMO takes a car out by itself) ended
  with a loop: its facing was taken for its way (5.6 m off, seen at a kerb); fixed.
* A car SUMO had put back and this module holds was let go by the follower logic of another drive (`_hold_followers`);
  only a car wholly behind a car turning in is held (one beside it, or put inside it by SUMO, is let go: the drive
  waits for it).

Same three runs before (the code at 278a815) and after, 1.5 town hours each, steps:

| run | overlaps with moving cars | inside CARLA's objects | inside parked cars | largest step | worst seam |
|---|---|---|---|---|---|
| seed 101, 07:30 | 107 -> 42 | 94 -> 21 | 69 -> 7 | 2.8 -> 5.5 m | 5.6 -> 5.3 m |
| seed 102, 12:00 | 89 -> 4 | 220 -> 18 | 23 -> 1 | 2.3 -> 2.4 m | 1.5 -> 1.6 m |
| seed 103, 17:30 | 63 -> 85 | 153 -> 13 | 45 -> 1 | 2.7 -> 3.5 m | 3.3 -> 5.1 m |

(The "inside CARLA's objects" before counts the Marché's fence pieces and the Square's dropped bays.) 8 cars to the
Marché one every 4 s (`tools/check_parking_crowd.py`): 0 overlaps, as before; 16 of 18 ways touch nothing (the 2
others touch a fence piece's corner at its north-west for 1 m).

Left:
* The Square's aisle (4.7 m) cannot hold two cars side by side; with several drives at once (seed 103, 17:40) cars
  meeting in it overlapped for 62 steps (39 before): one should wait at the gate or in its space until the aisle is
  clear (reserve with the slipped timetables).
* The two shortest kerbs (Rue 4, 23 m; Rue 51, 32 m): SUMO puts the car back at the lane's end, moveTo can't take it
  on; seams of 5.1 and 5.3 m there (2 of 26 kerb drives out).
* At a kerb, SUMO inserts a car leaving its place on the lane where a car turning in still stands (20 steps, seed 101).

**In CARLA** (`tools\freefly_parking_demo.bat`, 9 Nov 11:45-11:55, from 08:30; `out\town05_freefly_parking.json`,
`out\freefly\town05_parking_0*_*.png`): all seven kinds seen. Kerb in (08:51): the car stops behind its place and
pulls into it forwards. Taken out again 3 s into its kerb drive (staged, 09:16): it went on into its place, then out
forwards to SUMO's car (0.16 m). Two cars in the Parking du Square (10:03): out by the north gate (open as it passed),
the other 7 to 30 m away. Into Parking Central (10:21) and out (12:33) as before (barrier arm open while it passes, not
drawn once inside). Kerb out (11:06): forwards onto its lane, met SUMO's car 0.19 m off. Into a space at the Square
(12:46), by the south gate (open only while the car passes): on its space's point (0.0 m), its heading the space's
(0.0 deg). Drives out joined SUMO's car within 0.02 to 0.19 m at the kerbs (5), 0.07 to 0.73 m at the car parks (7),
1.21 m out of Parking Central.

## 9 Nov (4th, 12:20-13:30): the three left from the morning

* **The Parking du Square, one car at a time** (`places.json` "one_at_a_time", `ParkingMotion.narrow`,
  `_aisle_busy`): its aisle (4.3 m between the rows' noses) cannot hold two cars. A drive there starts only when no
  other drive there is on its way (the one made first goes first); the car waits in its space or on the road beside
  the car park, the cars behind it on the road held. A car coming to park there on the road's other lane, changing
  onto it at the last moment, is held too (SUMO had stopped it where the first car still waited).
* **The shortest kerbs**: SUMO's car, held where the drive out ends, changed lanes towards its turn (two lanes over,
  5.3 m from the drive's end). A held car now keeps its lane (`_lock`: lane change mode 0 while held, its own back
  after).
* **A car SUMO puts on the lane from a kerb place where it had no place before** (parked from before the drives
  began): shown in its place as any and driven out from there (`kerb_untracked`), its way out clear of the cars on
  drives too (`_kerb_blocker`).
* Also: a drive waiting behind another that waits for the road (a queue out of the Marché) no longer gives up after
  40 s and drives into it (only in a ring, where it waits for itself); a kerb car leaving waits for the lane stretch it
  uses (the cars that could not stop before it), not for 10 s of traffic; out of a kerb place an S of two arcs about
  the rear axle (a hard turn first, its nose out past the car in front), backing 0.8 m first if that is clear; it does
  not back out through the car parked behind it to reach SUMO's car (it waits).

Same three runs, this morning's commit -> now (steps):

| run | overlaps with moving cars | inside CARLA's objects | inside parked cars | worst seam |
|---|---|---|---|---|
| seed 101, 07:30 | 42 -> 0 | 21 -> 22 | 7 -> 0 | 5.3 -> 1.5 m |
| seed 102, 12:00 | 4 -> 2 | 18 -> 9 | 1 -> 0 | 1.6 -> 1.5 m |
| seed 103, 17:30 | 85 -> 6 | 13 -> 33 | 1 -> 1 | 5.1 -> 0.6 m |

(From the start of the day, 278a815: overlaps 107/89/63, CARLA objects 94/220/153, parked cars 69/23/45.) The seams
of 1.5 m are at the Square: SUMO puts the car back already moving; the kerbs' are under 0.9 m. 8 cars to the Square
one every 4 s: 0 overlaps (they queue: 1263 s of waiting in all); to the Marché: 0, as before.

Left: the 2 to 6 overlap steps are cars SUMO drives passing a car waiting at the road (2 steps each); the 9 to 33 steps
inside CARLA's objects are kerb drives brushing a pine's trunk or a lamp post on the pavement edge (which ones depends
on which places are taken: TreePine_501 on Rue 19, Lamppost_tall99/56 on Rue 19 and 20).

**In CARLA** (`tools\freefly_parking_demo.bat`, 9 Nov 12:32-12:45, from 08:30): all seven kinds seen. Two cars in
the Square (10:33): the second waited 8.5 s for the first and stayed 5 to 11 m from it. A car taken out half way into a
Square space (staged, 11:34): it had waited 47 s at the road for the aisle, went on into its space, then out. Kerb in
(08:43) and out (09:19), Parking Central in (13:13) and out (14:13), into a Square space (12:07: on its space, heading
0.0 deg off). Drives out joined SUMO's car within 0.0 to 1.07 m (27 drives; kerbs 0.0 to 1.07, car parks 0.0 to 0.65);
355 s of waiting for the Square's aisle in the 6 town hours. A kerb car waiting for its way out now tries again every 2
s, not every step (5335 ways tried in that run).

## 9 Nov (5th, 12:50-15:30): pines, lamp posts, the hand-over, the 0.37 m

* **Kerb places clear of the pavement edge** (`tools/town05_kerb_places.py`, places.json "shown_fronts"): Town05's
  avenues have a pine or a lamp post on the pavement edge every 6 m or so, and 46 of the 138 kerb places as shown stood
  on one. Along each kerb, every 0.25 m, a car's body (5 m, 0.1 m kept, and the metre before and after it) beside its
  lane is tested against CARLA's map; the clear places are packed 6 m apart. The kerbs take that many cars now (SUMO's
  roadsideCapacity: Rue 19 33 -> 20 and 30 -> 18, Rue 20 33 -> 19 and 30 -> 27, Rue 51 4 -> 3 each; Rue 4 2 as
  before). Into a place close behind where SUMO stopped the car: on past it and back in on two arcs, or a tight S
  straight in if a car stands ahead.
* **The hand-over**: SUMO puts a car back from a car park 1.3 m beside its lane's line (SUMO's sublanes): the step out
  of the Square. SUMO's car is now moved to where the drive ends (vehicle.moveToXY) the step it is put back, and
  stands there that step.
* **The 0.37 m**: SUMO gives a kerb place to a car arriving the moment the car in it leaves; both were shown in it. The
  arriving car now waits on its lane until the other has pulled out (`_kerb_place_taken`). A kerb place kept from a
  car's stay before is worked out afresh when it parks.
* **Cars passing a car waiting at the road**: held behind it even when close (before: let through, it drove through
  it), and asked to pass it on the next lane; a car coming fast held while it can still stop (17 m/s into a car waiting
  at the Square's way out); a car still moving up beside a standing car's tail held too (SUMO's sublanes let it come up
  beside it in the lane); held cars keep their lane; every car coming to a car park's stop keeps to its lane's middle
  (two queued side by side in the Square's 3.5 m lane and stopped at the same point to park); a car coming to park in
  the Square waits before its stop while a car is on its way in it.
* A drive waits for a car SUMO drives that stands in its way (it gave up after 20 s and went through it). And when that
  car waits behind the drive's own car, which SUMO put back early and this module holds on the road (each waiting for
  the other: 2 minutes at the Marché's way out), SUMO's car is moved on 8 m and the drive's end with it (`_unring`).
* Cars coming to a car park's stop change lanes to reach it but no longer squeeze past the car in front within the
  lane (lane change mode 581: no speed gain, no sublane moves). (Mode 0 for them, tried first, kept them from reaching
  the stop's lane: in CARLA 1 drive into a car park in 9 town hours; with 581, 14.)

Same three runs, the commit before (a082ee3, the same town but 30 % more kerb places) -> now (steps):

| run | overlaps with moving cars | inside CARLA's objects | inside parked cars | worst seam |
|---|---|---|---|---|
| seed 101, 07:30 | 0 -> 0 | 22 -> 9 | 0 -> 0 | 1.5 -> 0.75 m |
| seed 102, 12:00 | 2 -> 0 | 9 -> 0 | 0 -> 6 | 1.5 -> 0.79 m |
| seed 103, 17:30 | 6 -> 0 | 33 -> 0 | 1 -> 0 | 0.6 -> 0.25 m |

8 cars to the Marché and 8 to the Square, one every 4 s: 0 overlaps, seams 0.0 m (3.5 m at the Marché before).

Left:
* Seed 101: a car turning into a kerb place on Rue 20 brushes a lamp post for 9 steps; seed 102: a car pulling out of
  the shortest kerb (Rue 51, 3 places 6 m apart) brushes the car parked in front of it (6 steps).
* Long waits instead of glitches: seed 101, a car turning into a kerb place on Rue 20 waits 99 s for a car standing in
  its way and the cars behind it wait with it (up to 152 s); seed 103, a car pulling out of a place on Rue 19 waits
  133 s; at noon the Square's cars going out wait in their spaces up to 3.5 min for its aisle (one at a time).

**In CARLA** (`tools\freefly_parking_demo.bat`, 9 Nov 14:04-14:19, from 08:30): all seven kinds seen, 14 drives into
car parks, 16 out, 19 into kerb places and 19 out, 3 into Parking Central, 4 out. Into a Square space (12:23): on its
space (0.0 m), its heading the space's (0.0 deg). Two cars in the Square (10:54): 5 to 24 m apart. Hand-overs to SUMO
(30 drives out): car parks 0.0 to 0.65 m; kerbs mostly under 0.8 m, four of 1.2 to 2.9 m (Rue 4 and Rue 19: still to
look at).

## 9 Nov (6th, 14:25-15:50): the waits, the last brushes, the hand-over

* **Kerb drive out started while SUMO still has the car parked** (`_kerb_start`): it pulls out up to KERB_EARLY_S
  before its stay ends, the lane behind it kept clear, and SUMO's car is put where the drive ends when SUMO lets it
  go (vehicle.moveToXY: a kerb's hand-over was 1.0 m off along the lane before, 1.2 to 2.9 m in CARLA). If SUMO lets
  it go but cannot put it back on its lane (a car leaving a kerb place waits for room), it is put there too and the
  cars behind go on (before: they were held 80 s for the car standing there, Rue 20). A drive out that would cut
  across a car broken down on the next lane is not started.
* **The 100 s ring at Rue 20**: SUMO stopped a car past its kerb place with a car close behind it; it went on and
  backed into its place, where the car behind stood, held for it. Now such a car is shown at a free place ahead of
  it and pulls in forwards (`_kerb_ahead`); a car held behind one backing into its place is asked to pass.
* **Two cars on one kerb place** (Rue 51: SUMO had two cars 0.5 m apart on one of its spaces; both shown on one
  place, the one pulling out went through the other): the second is shown at the nearest free place.
* **The lamp post on Rue 20**: a 5.4 m car's rear touched it (the places were tested for a 5 m car):
  `tools/town05_kerb_places.py` now tests a 5.5 m car too (the same 94 places; two on Rue 20 moved 0.75 m). Rue 51's
  and Rue 4's places 7 m apart (before 6 m: a car pulling out had 1 m).
* **The Marche's corner space**: backing out of it, a car's tail went 0.2 m into the fence behind it; ways out with
  no straight bit after the turn are tried too. A car pulling out of a kerb has ways that back 1.6 m first.
* In a one-at-a-time car park, two drives whose ways stay 3.5 m apart do not wait for each other; a car coming to
  park is not held for a car standing on the road past its stop.
* Tried and dropped: letting a car out of the Square close behind another (it ran into it at the road: 6 overlap
  steps in the 8-car check), holding the cars waiting to go out while a car waits to go in (the 8-car check took
  2.6 times as long).

Same three runs, the commit before (239cb7b) -> now (steps):

| run | overlaps with moving cars | inside CARLA's objects | inside parked cars | worst seam | longest wait of a drive | longest a road car is held (in all) |
|---|---|---|---|---|---|---|
| seed 101, 07:30 | 0 -> 0 | 9 -> 0 | 0 -> 0 | 0.75 -> 0.0 m | 101 s -> 20 s | 152 s -> 31 s |
| seed 102, 12:00 | 0 -> 0 | 0 -> 0 | 6 -> 0 | 0.79 -> 0.0 m | 218 s -> 218 s | 104 s -> 104 s |
| seed 103, 17:30 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0.25 -> 0.0 m | 133 s -> 47 s | 61 s -> 62 s |

8 cars to the Marche and 8 to the Square, one every 4 s: 0 overlaps, seams 0.0 m.

Left: seed 102 at noon, the Square's way out opens onto a queue standing at a red light (40 to 50 s each time); its
cars going out wait in their spaces up to 3.6 min, one at a time, and a car coming to park there waits up to 104 s on
the road before it. No glitch, a jam.

**In CARLA** (`tools\freefly_parking_demo.bat`, from 08:30), two runs:
* 15:13-15:27: all seven kinds seen; 21 hand-overs to SUMO, 19 at 0.0 m, one 0.78 m (a car that left the Marche
  before it had parked), one 1.59 m (a kerb car SUMO had put back beside its lane: moved to where the drive ends now,
  above).
* 15:30-15:46, with that: 27 hand-overs, all 0.05 m or less (car parks 0.0 m). 10 drives into car parks, 12 out, 13
  into kerb places, 15 out (12 started while SUMO still had the car parked). The building drives (Parking Central) did
  not come up in this run by 15:54 (the first run's at 13:18 and 14:48, unchanged code).


## GO-FLOW priority agent: integration, step 0-4 (4 Nov)

The priority agent of the GO-FLOW `priority-agent` branch (ADSP: priority tracking, rule coordinator, safety layer,
LLM coordinator, PPO) brought in under `agents/priority_agent/` (its own venv, `tools\setup_goflow_priority.bat`).
Only its coordinator runs here, as an advisor to the junction agents (`cosim/goflow_priority.py`,
`--controller agents_goflow` / `agents_observed_goflow`, viewer `--goflow-priority`, off by default). The PPO is not
used: it was trained for 2 greens, 4 approaches of 2 lanes and a 1 s step, and Town05 has 4-5 greens a signal.

* Shadow check (`agents\priority_agent\town05\shadow_check.py`, seed 11, x0.8, 900 s): the agent's unchanged code
  runs on Town05 without an error and changes no signal state (same SHA-256 of every signal state as a run without
  it); `Intersection.from_sumo` finds 4-5 greens at every signal (PPO needs 2) and maps the approaches wrongly at
  5 signals (2014 and 2328: the three roads fall into one direction; 1882, 2240, 2296: a walking area counted as a
  fourth approach). Hence the adapter speaks link indices, not N/E/S/W.
* Smoke run, one seed only (seed 11, x0.8, 1800 s, emergency every 120 s, `tools/eval_town05_baseline.py`; run in the cloud: Linux, Python 3.13, SUMO 1.27.1):
  agents vs agents_goflow: ambulances on a run 60.8 vs 45.3 s time loss (9 vehicles), car waiting 104.0 vs 110.3 s,
  pedestrian time loss 83.8 vs 85.0 s. One seed: not a result; the car cost is to be checked on dev seeds 11-16
  (horizon 20/30/45 s) before the held-out runs. Plan computed in 0.04 ms (p50), 0.10 ms (p95).
* Tests: `tests\test_goflow_priority.py` (names, the child process never fatal, stale plans ignored, the flag);
  `agents\priority_agent\town05\tests` (8, in the agent's venv).

## GO-FLOW priority agent: the LLM in the background, camera sightings (4 Nov)

`agents/priority_agent/town05/llm_town.py`: the agent's LLM coordinator (B5) reworked for a live city. The rules'
plan is answered at once on every state; events (new and cancelled requests, conflicts between emergency vehicles,
dispatch messages, and **flashing lights the cameras confirm on a lane where no vehicle sent a request**) go to a
worker thread that calls Ollama; what the LLM decides is laid over the rules' plan from the next state on. It never
blocks: before, the agent paused SUMO during each call (3.4 s mean on the corridor).

Guardrails, in code: ModeSafety on every mode; an emergency vehicle holding a junction is never displaced by a lower
one (class, then arrival); escalation only for a vehicle named in a dispatch message received; camera-only green
(`request_green_for_lane`) only with a confirmed sighting on that lane in the last 10 s, one per junction per 120 s,
20 s at most, the junction agents' own guards still applying; answers after the call's deadline (20 s) refused.
LLM down, slow or wrong: the rules' plan stays.

Checked (cloud, SUMO 1.27.1, no Ollama there): 8 tests with a scripted model (`town05/tests/test_town05_llm.py`:
the rules answer at once and the LLM overrides later; police never before an ambulance; LLM down = rules only; late
answers refused; camera green limits; escalation needs a dispatch message); `agents_observed_goflow` with
`--llm` and no Ollama, seed 11, x0.8, 900 s: runs on the rules, one `llm_unavailable` line; seed 12: 62 confirmed
camera sightings reached the agent in 900 s. Not yet measured: the LLM's real latency on the PC's CPU next to
CARLA, and whether its decisions help (needs Ollama on the PC).

## GO-FLOW priority agent: after the review (5 Nov)

* **A public hook instead of a subclass override.** `ActuatedAgents.request_source` (None by default) lets an outside
  planner say who should get which link; `_external_requests` runs every such request through the agents' own
  guards (next signal: patience, stopped on its green, `_can_pass`; further on: the exit's room). The safety rules
  stay in one place. `cosim/goflow_priority.py` no longer overrides `_requests` or copies the guards. With the hook
  unset nothing changes: `agents` and `agents_observed` give the same figures as on `main` (seed 11, x0.8, 1800 s:
  car waiting 104.02 / 107.78 s, emergency time loss 60.83 / 81.67 s), and their stats gain no key.
* **In charge or fallen back, in every result.** The hook counts `external_plan_steps` and
  `external_fallback_steps` in the agents' stats, which `tools/eval_town05_baseline.py` saves with every run
  (`agent_stats`), with `goflow_timeouts`, `goflow_errors`, `goflow_restarts`, `goflow_down`, `goflow_refused` and,
  with the LLM, `goflow_llm_calls / failures / late / refused / p95_latency_s`. Seed 11: GO-FLOW in charge on 3580
  of 3600 steps (the first 10 s before its first plan), no timeout, no restart.
* **No signal priority for recovery trucks** (they drive with hazard lights, not a siren): GO-FLOW plans police and
  ambulances only; a loaded truck waved off at the stop line keeps the agents' own rule.
* **Held-out seeds 31-40** in `tools\eval_town05_goflow.bat final`; seeds 21-30 only for comparison.
* Same figures as before the refactor (seed 11, x0.8, 1800 s): agents_goflow 45.26 s emergency time loss, cars
  110.27 s waiting; agents_observed_goflow 49.15 s, cars 102.79 s. One seed.

## GO-FLOW priority: a refused vehicle goes back to the agents' own rule (5 Nov)

Seen in the sumo-gui demo (`tools\goflow_priority_demo.bat`, an emergency vehicle every ~45 s): police cars waiting
at reds. GO-FLOW's safety layer refused them 94 times in 30 min ("3 pre-emptions in 300 s" at 1427, police are not
exempt), and since the plan still claimed those vehicles, the junction agents did not ask for them either: no
priority at all, worse than the agents alone. Now a vehicle refused in a plan is handed back to the agents' own rule
for that step (its accepted junctions stay in the plan); counted as `goflow_handed_back`.

Seed 11, x0.8, 1800 s, emergency every 45 s (`--no-gui`; mean time loss, s):

| | ambulances (19) | police (21) | police waiting | others waiting |
|---|---|---|---|---|
| agents (without GO-FLOW) | 114.3 | 73.0 | 46.4 | 120.8 |
| agents_goflow before the fix | 122.1 | 79.1 | 55.9 | 130.8 |
| agents_goflow after the fix | 103.9 | 68.0 | 44.8 | 128.7 |

One seed. The cost for the other vehicles (+8 s) is what the dev runs have to settle (horizon, rate limit).


## GO-FLOW priority agent: reviewed, merged, heavy traffic fixed, tested (9 Nov)

Yosr's branch `integration/priority-agent-town05` (`8e12ee8`) merged with `main` (`c90f93e`). Conflicts in
`README.md`, `cosim/freefly.py` (main moved the agents' wiring into `agents_actuated.wire()`: kept, only `goflow_role`
added) and this file. Her seed-11 figures come out the same on the merged code (car waiting 104.02 / 110.27 / 107.78 /
102.79 s).

**Heavy traffic.** Dev seeds 11-20, `tools/eval_town05_baseline.py`, 1800 s, paired against the agents alone: at x0.8
GO-FLOW cut the emergency vehicles' time loss (-15.4 s with exact counts, 9 of 10 seeds) for +2 s car waiting, but at
x1.0 it made the town worse (car waiting +24 s exact counts, +77 s cameras; 1549 vehicles stuck against 824, 2511
against 732; every 45 s: +166 s, 0 of 10 seeds better). The cause: a request for a signal beyond the vehicle's next one
had only the exit check (`_external_requests`). GO-FLOW keeps a signal it planned until the vehicle passes it, and
counts arrival times at 8 m/s or more, so an ambulance stuck in a queue kept the signals ahead holding green for it:
976 holds a run against the agents' own 122. The agents' own corridor rule has the guard for this (moving at 3 m/s or
more, due within 25 s at its real speed; see the top of `cosim/agents_actuated.py`). Now the planner's requests
beyond the next signal go through the same conditions (`_corridor_ok`, `external_guard`; 0 gives the old behaviour).

Dev seeds 11-20 after the fix (mean difference, seeds better; stuck: summed over the 10 runs, GO-FLOW / agents):

| Emergency every | Traffic | Signals on | Car waiting | Throughput /h | Emergency time loss | Stuck |
|---|---|---|---|---|---|---|
| 120 s | x0.8 | exact counts | +1.4 s (4/10) | -12 | -7.1 s (8/10) | 0 / 0 |
| 120 s | x0.8 | cameras | +0.6 s (4/10) | -4 | -1.9 s (5/10) | 0 / 0 |
| 120 s | x1.0 | exact counts | -6.0 s (7/10) | +33 | -4.0 s (5/10) | 958 / 824 |
| 120 s | x1.0 | cameras | +40.9 s (4/10) | -288 | +76.5 s (2/10) | 1809 / 732 |
| 45 s | x0.8 | exact counts | +2.5 s (2/10) | +4 | -1.3 s (7/10) | 0 / 0 |
| 45 s | x0.8 | cameras | +0.4 s (5/10) | +11 | +3.9 s (5/10) | 0 / 0 |
| 45 s | x1.0 | exact counts | +37.5 s (3/10) | -218 | +7.9 s (7/10) | 1918 / 1079 |
| 45 s | x1.0 | cameras | -41.2 s (7/10) | +267 | -39.9 s (8/10) | 1257 / 2318 |

Tried on the same dev seeds and not kept: arrival within 35 s (about the same), within 60 s (x1.0 worse again: +9.6 s,
1102 stuck against 824), and also limiting the next signal to 150 m as the agents do (no better). At x1.0 the town is
at the edge of gridlock: one seed's waiting goes from 139 s to 314 s on a small change in the signals, either way, so
10 seeds do not separate two versions there. Summed over the four x1.0 rows, the fix takes the extra waiting from
+82 s to +8 s a trip on average and the stuck vehicles from 11663 to 5942 (agents alone 4953). The price: most of the
ambulances' gain at x0.8 came from the same holds, and is gone.

Also fixed: the LLM's queue counts were SUMO's lanes even under `agents_observed`; now the cameras' reports
(`_gf_state`). `tools/eval_town05_goflow.bat` used test seeds 31-40 (opened for the forecasters' v5): now dev 11-20,
test 71-80 (61-70 are the route agent's), with `--open-test` (a `TEST_OPENED.lock` per folder, new in
`tools/eval_town05_baseline.py`). Tests: `tests/test_goflow_priority.py` (8), all 251 pass (3 need `cv2` / `carla`).

**The bar, set before the test** (seeds 71-80, opened once, 1800 s, each GO-FLOW controller against the same agents
alone, an emergency every 120 s and every 45 s):
* x0.8: emergency time loss lower on the mean and on at least 7 of 10 seeds; car waiting at most 2 s worse on the
  mean; no vehicles stuck where the agents alone have none;
* x1.0: car waiting at most 2 s worse on the mean; vehicles stuck (summed over the seeds) no more than the agents
  alone.
Met in every row: recommended with `--goflow-priority`. Not met: it stays an experiment, off by default. On the dev seeds
above it does not meet it (x0.8 emergency gains too small, x1.0 cameras at 120 s and exact counts at 45 s worse).

**The test** (seeds 71-80, opened once: `out/goflow_priority_eval/test_120`, `test_45`, `TEST_OPENED.lock`, run in the
cloud: Linux, Python 3.13, SUMO 1.27.1). Mean difference against the agents alone (seeds better; stuck: summed, GO-FLOW /
agents alone; in brackets the agents alone):

| Emergency every | Traffic | Signals on | Car waiting | Throughput /h | Emergency time loss | Stuck | Bar |
|---|---|---|---|---|---|---|---|
| 120 s | x0.8 | exact counts | +0.6 s (4/10) [103.0 s] | -2 | -1.7 s (6/10) [41.2 s] | 0 / 0 | no |
| 120 s | x0.8 | cameras | +0.0 s (5/10) [101.3 s] | +7 | -4.4 s (7/10) [44.7 s] | 0 / 0 | yes |
| 120 s | x1.0 | exact counts | +26.0 s (2/10) [158.5 s] | -227 | +46.7 s (2/10) [74.4 s] | 1128 / 337 | no |
| 120 s | x1.0 | cameras | +12.8 s (5/10) [162.8 s] | -97 | +10.6 s (5/10) [106.4 s] | 839 / 519 | no |
| 45 s | x0.8 | exact counts | +3.9 s (1/10) [107.8 s] | -1 | -1.8 s (6/10) [43.6 s] | 0 / 0 | no |
| 45 s | x0.8 | cameras | +0.6 s (5/10) [106.4 s] | +9 | +2.2 s (4/10) [40.4 s] | 0 / 0 | no |
| 45 s | x1.0 | exact counts | +4.1 s (5/10) [197.5 s] | +14 | +18.4 s (5/10) [122.4 s] | 1231 / 1408 | no |
| 45 s | x1.0 | cameras | +15.7 s (4/10) [178.0 s] | -83 | +16.3 s (3/10) [111.8 s] | 1138 / 889 | no |

GO-FLOW was in charge on 98-99 % of the steps (the rest: before its first plan), no timeout, no error, never down.

Reading. It does not meet the bar (1 row of 8): it stays off by default, an experiment. Once it may no longer hold
the signals ahead for a vehicle stuck in a queue, it gives the ambulances about what the agents' own rule gives
(-4 to +2 s at x0.8), and in heavy traffic it is still worse on most rows (+4 to +26 s waiting, more vehicles stuck on
3 rows of 4). What is left of its design over the agents' rule (signals planned up to 30 s ahead, arbitration
between two emergency vehicles) does not show here. The LLM was not run (no Ollama in the cloud). The next seeds
for any new version: 81 and above.


## The town audit, steps 1-3: skipped greens, cars taken off the road, the viewer's default (9 Nov)

Three screenshots (an endless queue behind an accident, cars standing mid-road, a police car read as "siren off")
and a town that felt wrong. The viewer's log said why: 85-93 % of the town's moving traffic standing at any moment.
It was running the coordinated signals (the default since 27 Oct) and, from the forecast launcher, twice the traffic.
Diagnosis runs (SUMO alone, seed 11, evening rush held at 17:30, 20 min) and the full report are in the session; the
three steps done here:

**1. A skipped walk skipped car greens.** `sumo_side.WalkSkip` skips a walk nobody waits for by going to phase 0.
That is right when the walk ends the cycle (the actuated programs), but `cosim/coordination.py` puts the busiest green
first, so in 12 of the 15 coordinated programs the walk is in the middle of the cycle, and every car green after it
was skipped too. Junction 1427 ran its phase 0 for 109 s and back to phase 0: its other three approaches never went
green unless someone waited to cross. Now the signal goes on to the next car green after the walk, its clearance and
buffer (`WalkSkip.after_walk`), and only those seconds go to the next first green. Tests: `tests/test_walk_skip.py`
(the SUMO one fails on the old code: car greens never shown at 12 signals).

**2. Cars taken off the road.** SUMO runs without teleporting; knots were undone by taking cars off, out of sight:
* `IncidentManager._relieve` took off any car standing 25 s on an accident's road or any road into it, even one
  waiting at a red light there. Now: after 45 s a new way (the accident's roads count as slow), from a lane that leads
  on; taken off only still standing 240 s on the accident's own road, out of sight.
* `IncidentManager.unjam` took off a car standing 150 s out of sight (in sight it got a new way). Now every car gets a
  new way first; taken off only still standing another 150 s, out of sight.
* Every run now reports what was taken off (`IncidentManager.removals()`, `removed` in
  `tools/goflow_prediction_dataset.py`, `eval_agents_forecast.py`, `eval_route_guidance.py`; the viewer logs
  `taken off the road` every SUMO minute). Tests: `tests/test_removals.py`.

**3. The viewer's default.** Dev seeds 11-20, a day as the viewer runs it (clock 30 from 07:00, 5 min warm-up then
30 min, the town's own traffic, trips with a purpose, accidents, buses, ring road), SUMO alone; per km driven (trip
time minus planned stops):

| Code, signals | Speed | Waiting | Standing | Taken off (accident + knots) | Taken off (population) |
|---|---|---|---|---|---|
| before, coordinated | 3.5 km/h | 934 s/km | 89 % | 369 a run | 162 a run |
| now, coordinated | 9.1 km/h | 283 s/km | 69 % | 7 | 313 |
| now, actuated | 11.9 km/h | 184 s/km | 59 % | 23 | 324 |
| now, junction agents (cameras) | 11.3 km/h | 210 s/km | 62 % | 21 | 388 |

The fix: waiting lower on 10 of 10 seeds. Coordinated, even fixed, is worse than the junction agents on 8 of 10
seeds; actuated and the agents do not differ (actuated lower on 4 of 10 for waiting, faster on 5 of 10). The viewer's
default is now the junction agents (`--controller agents_observed`); the HUD strip says who runs the signals and,
when it is not 1, the traffic level (`hud.setup_text`, also in the log at start).

Open, from the same audit:
* **The population model takes off about half as many cars as enter** (313-388 a run against 590-710 trips), out of
  sight, when the town holds more than the hour's target (`cosim/live.Population`; mostly the day's curve falling
  faster, at clock 30, than trips end). Counted now; changing the demand model (trips an hour instead of a count of
  cars) is a decision of its own.
* **The signals leave each approach about 15 % of a 115 s cycle** (each approach on its own, a walk of its own,
  roads of 23-65 m): even the best case keeps 59-62 % of cars standing. Next step of the plan.
* The forecasters' v6 was trained partly on coordinated runs made with the skipped greens: to be checked.


## Honest demand, the town's traffic level, and the signal design (9 Nov, night)

**Cars taken off to follow the curve.** `cosim/live.Population` kept the number of cars on the road on the day's curve
and took cars off, out of sight, whenever there were more: 313-388 a run against 590-710 trips (dev seeds 11-20, a
35-minute day). Two causes: at clock 30 the evening curve falls faster than trips end, and the count took in the ring
road's cars, the buses and the emergency vehicles in the dataset and evaluation tools (the viewer left the ring road
out), so at 17:30 only about 60 of a target of 116 were town cars.

**Now: trips at the curve's rate** (`Population(flow=True)`, the default): new trips start at curve / mean trip (140 s)
a second of simulation, the hour read on the town's clock; every car drives its trip to its end; how many are on the
road follows from how fast they get through. A jump of the clock (start, F6 / F7) fills the town at once, as before.
Trips SUMO cannot put on the road once its queue is half the curve are counted (`trips_not_started`), not added.
Tests: `tests/test_daily.py`.

**What the town carries.** With nothing taken off, the old curve locked the town every evening (one day, seed 11:
260 town cars on the road at 17:45 against a target of 116, 280 trips never started). Swept on dev seeds 11-13 (a day
at clock 30 from 07:00, SUMO's actuated signals), x the old curve:

| Traffic | Speed | Waiting | Taken off | Trips not started |
|---|---|---|---|---|
| 0.40 | 13.1 km/h | 149 s/km | 21 | 4 |
| 0.45 | 12.7 km/h | 167 s/km | 5 | 5 |
| 0.50 | 10.4 km/h | 304 s/km | 30 | 5 |
| 0.55 | 5.7 km/h | 526 s/km | 97 | 65 |
| 0.70 | 5.2 km/h | 593 s/km | 118 | 134 |

The town's own traffic (x1) is now 0.45 of the old curve (`daily.DEMAND_BASE`, chosen by the user): the evening rush
at what the town carries. An emergency run every 300 s instead of every 45 s changed little (x0.5: 317 s/km against 304).
`town05_freefly_forecast.bat` now runs the town's own traffic (`busy`: x1.15); its x2 would now lock the town.

**The signal design (step 4).** Tried on the same days, nothing better than today's:
* the permissive programs (opposite approaches together, turns giving way; `SIGNAL_LAYOUT=opposites
  tools/build_town05_signals.py`, rebuilt on today's network: the same as the 26 Oct set): x0.4 264 s/km against
  149, more cars standing inside junctions. Town05's approaches have two lanes and no turn bays: a car waiting to
  turn left holds up the lane behind it;
* longest green of the agents' programs (`PROTECTED_GREEN`), junction agents, dev seeds 11-15: 20 s 221 s/km, 30 s
  (today) 186 s/km, 45 s 300 s/km and three times the cars taken off.

**The controllers at the new traffic level** (dev seeds 11-20, the day as above):

| Signals | Speed | Waiting | Standing | Taken off (last resort) | Ambulance time loss |
|---|---|---|---|---|---|
| junction agents (the default) | 10.3 km/h | 260 s/km | 64 % | 27 a run | 74.8 s |
| actuated | 11.4 km/h | 219 s/km | 61 % | 23 | 84.6 s |
| coordinated (fixed) | 7.8 km/h | 374 s/km | 74 % | 24 | 129.6 s |

Agents and actuated: no clear difference for cars (5 of 10 seeds each way); the agents get the ambulances through
faster on 7 of 10. Coordinated is worse on 9 of 10. The default stays the junction agents.

Open: most of the cars still taken off are behind accidents and in knots as a last resort (about 25 a run);
the forecasters must be checked and retrained on this town (after steps 5-7).


## Cars taken off as a last resort, steps 5-7 of the audit (10 Nov)

Same days as before (dev seeds 11-20, a day at clock 30 from 07:00, the town's own traffic, SUMO alone), the junction
agents on the cameras, paired against the run before.

**Why cars were still taken off** (21 cars on dev seeds 11-12, each looked at when taken off):
* 6 were parked in a car park near an accident: `_relieve` took a parked car for one held up. Now parked cars, cars
  at a stop and cars off the road are never counted as held up.
* most of the others stood on the roads into junction 979, behind a crash inside it, until the 300 s last resort
  (the junction agents then rightly held back greens into those full roads). **Step 7:** drivers coming up to a
  junction where a wreck stands in the way of their movement now take another way out, one their own lane leads to
  and the wreck leaves free (`IncidentManager._reroute_junction`; a movement is blocked when its way through the
  junction passes within 2.5 m of a wreck, `_blocked_links`). Rerouting is the same whether you watch or not; only
  the last resort (taken off) is out of sight.

| | Taken off a run (behind accidents + knots) | Waiting | Ambulance time loss |
|---|---|---|---|
| before | 27.2 (12.9 + 14.3) | 260 s/km | 74.8 s |
| now | 18.1 (5.6 + 12.5) | 244 s/km (lower on 5 of 10) | 61.9 s (lower on 6 of 10) |

What is left (dev seeds 13-16, looked at): mostly one seed, two crashes near the centre at the rush and a jam over
five junctions (the town runs at what it carries; a crash there locks it, as in a real town), and cars behind a bus
standing at its stop, which a standing car cannot pass (no sideways move while standing).

**Step 5, spillback and neighbours.** Max-pressure (`max_pressure_walk`, which weighs each movement by its queue
minus the queue it feeds) against the junction agents, both on exact counts: waiting +44 s/km (lower on 4 of 10),
ambulances +18 s (it has no pre-emption here). The agents' own rule of no green into a full exit (`spill`) is worth
4 s/km (without it worse on 9 of 10). The agents stay as they are.

**Step 6, lane changes.** Measured across the whole road (seed 11, 12 min of the evening rush): 866 lane changes,
median 4.0 s (10 % under 2.5 s, 10 % over 5 s; about 4 s on real roads); 1548 small drifts inside a lane (0.5 s each,
700 m sideways in all); 1835 steps sideways at low speed steeper than the 12 deg the viewer turns a body; 415 steps
sideways while standing (the squeezes past wrecks, buses into their stop). Tried, not kept:
* sideways speed at most 0.2 x forward speed and 0.6 m sideways gap (`lcMaxSpeedLatFactor`, `minGapLat`): steep moves
  1835 -> 436, no overlap in `tools/check_overlaps.py` (1 before), but on the dev days waiting +44 s/km (lower on 4 of
  10) and 43 cars taken off a run against 18: lane changes too slow in queues;
* `lcSublane` 0.1: lane changes shorter (median 2.5 s);
* the viewer's low-speed limit 12 -> 17 deg (`interp.Trails`): no difference.
A short clip of the glitch seen in CARLA is needed to go further.

**Also:** the population counts simulation time, one SUMO step a call (a tool that holds the hour still,
`tools/check_overlaps.py`, got no new trips at all: 65 vehicles in 5 minutes, now 254); the same as before when the
clock moves. `tools/goflow_prediction_dataset.py --trips purpose` (the default, as the viewer; the forecasters' v6
data had random trips); `eval_agents_forecast.py` and `eval_route_guidance.py` use trips with a purpose too.


## The forecasters' v7: data and training on the new town (10 Nov, started)

The town changed under the forecasters (skipped greens fixed, trips at the day's rate with nobody taken off, the
town's own traffic 0.45 of the old curve, the junction agents by default, trips with a purpose, junction crashes
rerouted), so v6 is retrained once, after all the changes. `tools\goflow_v7_build.bat` makes four datasets on
seeds 101-130 and 11-20 (the junction agents at x1, x1.15 and x1.3, coordinated at x1; a day at clock 30 from 06:00,
accidents, no deadlock breaker, as the viewer) and the same on test seeds 81-90, then trains v7 with the v6 recipe
into `out\goflow_prediction\v7\models`. Not yet scored or shipped: `models\goflow` is still v6 (`docs/HANDOVER_10nov.md`).
A first try in the cloud (39 of 40 runs of the x1 dataset) was cut off by a restart of the cloud machine; the
build runs on the PC.

**10 Nov, 02:17: the build had not run; a crash that was not one.** Started on the PC, its log showed
`incident 1: TraCIException("Lane '' is not known")` every 2 s on several seeds. Cause (seed 101, reproduced in the
cloud): at t = 1 s SUMO reported a rear-end between two cars just put on the road at a kerb place, the follower
standing (0 m/s, the car ahead at 1.1 m/s: an overlap from being put on the road, not a hit). It became a crash; both
cars then went into their kerb place, off the road, and `_squeeze` asked SUMO for the cars on lane `''`, which failed
and cut the rest of the incident's step every 2 s. Fixed in `cosim/incidents.py`: a rear-end whose follower stands
still is no crash (counted as a near miss, `standing follower`); `_squeeze` skips a blocker off the road. Seed 101 run
again: no such error. Build stopped and restarted at 02:49 with the fix (the aborted log is kept outside git).


## Lane changes, step 6: the glitch is the body's heading, not SUMO's lane changes (10 Nov, night)

**How it was looked for.** `tools/check_lateral.py` runs the viewer's town in SUMO alone (one_run: trips with a
purpose, buses, the ring road, accidents, the junction agents), poses every vehicle 60 times a second with the
viewer's own code (Catmull-Rom along SUMO's front positions, `interp.Trails`), and follows the body's rear axle
(0.75 x length behind the front). A car's rear wheels roll, they do not slide: any motion of the rear axle across the
body is the glitch. An episode: over 0.2 m/s across the body and over 8 deg off its axis, for 0.25 s or more, put
down with what the car was doing and what was near it. Cars driven into or out of a parking space are left out
(the viewer poses them on their own, `cosim/parking_motion.py`). Dev seeds 11-13, 10 min of the morning rush each.

**The cause.** `Trails` pointed the body from the point 0.7 x length back *along the front's own path* to the
front: the rear was pinned to the path the front had drawn. A body is then a chord of that path, and its rear
slides sideways whenever the path bends: in every turn (on a 10 m radius ~9 deg off the way it rolls), in every lane
change and in every small drift SUMO's sublane model makes inside a lane. It also made the heading hunt (the turn
direction reversed ~180 times in 10 minutes). Near queues it shows most: slow lane changes into gaps and round a bus
at its stop, where the body is long relative to the motion.

**The fix** (`cosim/interp.py`, pose only): the body's rear point is dragged by the front and moves only along the
body (a tractrix, as a trailer behind its hitch, or a car's rear axle). The caps stay (2.5 deg a pose, 15 deg per
metre rolled: a standing body never turns) and so does the band round the lane's direction.

| Dev seed | Sliding, s per vehicle-hour (before -> now) | junctions | lane changes | drifts in a lane | standing | bodies cutting into each other (pairs) | yaw acc. p99.9 (deg/s2) | turn reversals |
|---|---|---|---|---|---|---|---|---|
| 11 | 61.0 -> 13.7 | 801 -> 144 s | 147 -> 43 s | 148 -> 2 s | 80 -> 74 s | 49 -> 41 | 284 -> 179 | 176 -> 21 |
| 12 | 57.1 -> 11.9 | 781 -> 150 s | 142 -> 34 s | 168 -> 7 s | 51 -> 48 s | 41 -> 39 | 287 -> 187 | 171 -> 20 |
| 13 | 56.4 -> 11.5 | 802 -> 148 s | 128 -> 42 s | 157 -> 11 s | 23 -> 26 s | 37 -> 38 | 281 -> 185 | 189 -> 29 |

The same at the body's centre (rear point 0.5 x length, seed 11): 709 -> 196 s. Traffic is not touched (the
change is in the viewer's poses only; SUMO runs exactly as before), so its speed and waiting are the same by
construction. Seen in CARLA (`town05_freefly.bat`, 13:00 town time, 25 fps): bodies straight in their lanes, turns
normal. Tests: `tests/test_trails_slide.py` (a 10 m turn: the rear slid 9.7 deg with the old code, under 3 now).

**What is left: standing sideways moves made on purpose.** 26-74 s of sliding while standing in each 10 minutes
(all vehicles together), from SUMO moving a stopped vehicle sideways where the town allows it:
cars squeezing past a wreck (`IncidentManager`, 0.3 m/s while standing) and buses turning into their stop
(`BusLines.turn_in`, 0.5 m/s). A body cannot turn there without rolling, so it slides. Taking these allowances away
locked the town before (cars behind a wreck, buses beside their stop); not changed.

**10 Nov, 03:40: the v7 build left the dev seeds out.** `goflow_v7_build.bat` passed `101-130,11-20` unquoted, and a
comma splits a batch file's arguments: the first dataset (D7_ao_x1) came out with seeds 101-130 only. The seeds are
quoted now, and `tools/goflow_prediction_dataset.py --resume` finishes a folder (keeps the runs its runs.json lists,
runs the seeds missing, makes dataset.npz again); the bat resumes any folder that exists instead of deleting it.
Build restarted at 03:43: D7_ao_x1 gets its 10 dev runs, then the others.


## Siren detection, step 8: the H window votes over each vehicle's track (10 Nov, night)

**Done: tracking and voting in the H window** (`cosim/inspector.TrackVote`). The window counted one picture at a
time. Its boxes now go through the cameras' own tracker (`camera_reports.Tracker`) and an emergency vehicle is
labelled by its track: SIREN ON once 3 of its last 8 pictures said so (then kept while the track lives), ambulance or
police by the majority of its emergency pictures; a car called an emergency vehicle in one picture stays a car. The
rule is the one of the cameras' reports, so the window shows what the reports say. Tests:
`tests/test_inspector_vote.py`.

What the vote gives, from the detector's measured per-picture rates (carla-yolo26n-t05, Town05 held-out junctions,
`out/detector/eval_town05.md`: siren on found as siren on in 174 of 517 pictures, siren off called siren on in 36 of
410), pictures taken as independent (they are not: the same vehicle at the same angle errs alike, so these are
upper bounds):

| Pictures of the vehicle (5 a second) | Siren on: confirmed | Siren off: called on |
|---|---|---|
| 8 (1.6 s) | 54 % | 2.5 % |
| 25 (5 s) | 96 % | 14 % |
| 50 (10 s) | 99.9 % | 29 % |

A siren-off emergency vehicle watched long enough gets called on (the latch). 4 of 8 would give 96 % / 5 % at 10 s
but needs 2.4 s instead of 1.4 s to confirm; not changed without clips to measure it on.

**Per class** (the same test set, AP50): ambulance 0.43, ambulance with lights 0.45, police 0.19, police with lights
0.42. Police without lights is the weak class; between the two states, 73 siren-on boxes were called siren off and
36 siren-off boxes siren on.

**Judged by emergency-vehicle delay: the camera cannot change it as the town is wired.** The junction agents'
pre-emption is driven by the vehicle's own request (`agents_actuated._requests`); the cameras' siren votes (`ems` in
the reports) reach no controller by default (only the GO-FLOW priority agent reads them, and it is off). So a better
detector changes the H window and the reports, not the delay. For the camera to count, it needs a role, for example
confirming a request before a signal is cut, or calling a signal for an emergency vehicle whose request is lost: the
user's decision.

**Not done yet: a model that sees the lights flash.** The Town05 detector dataset is single pictures 20 s apart
(`out/dataset_town05/meta.jsonl`): nothing to learn a flash from. It needs short clips (2 s at 10 pictures a second)
of emergency vehicles with the lights on and off, recorded in CARLA on the PC.


## Cars still taken off as a last resort, step 4 of the list (10 Nov, night)

Dev seeds 11-20, the viewer's day (clock 30 from 07:00, 5 min warm-up, 30 min, the junction agents, accidents),
SUMO alone, with the code of tonight; every car taken off looked at when taken off (what stood ahead of it, its lane,
its signal). 14.4 a run (4.0 behind accidents, 10.4 knots), 144 in all, but 81 of them on two seeds (14 and 18:
40 and 41; seed 14 also left 96 trips unstarted, the town locked) and 3-8 on the others.

What stood ahead of the car taken off: an ordinary car standing 43, a bus 38, an accident's vehicle 30, nothing within
50 m 28, a car moving 5. **None of the 38 buses was at its stop**: every one stood in the queue itself, 18 of them on
road -19 (whose kerb parking lane, SUMO lane type "parking", 2 m wide, 5 km/h, is open to buses) and most of the
others at a red light. So "cars behind a bus at its stop cannot pass" is not what takes cars off on these days; buses held in the
same knots are.

Tried, not kept: buses kept off the kerb parking lanes that have no stop of theirs (`BusLines`, lane permissions):
taken off 14.4 -> 22.0 a run (fewer on 5 of 10 seeds), waiting +25 s/km (lower on 5 of 10), ambulance time loss
+15 s. The runs are chaotic at this traffic level (one knot at the rush decides a seed: per seed 0 to 69 cars), so ten
seeds tell little about a change this small. On both bad seeds (14 and 18) the knot starts with an accident on road
-19 (206 m, the kerb places, buses of line L3): its queue fills roads 15 and 6 and the junctions 2296, 1292 and 1574
behind it. A fix of that (how drivers leave -19 round a crash, or the accident's lane closed further back) is the next
step.


## Forecasters v7: the bar, written before the test (10 Nov, 04:35)

Written before any v7 number is seen. The test opens once, on seeds 81-90 (T7_ao_x1, T7_ao_x1.15, T7_ao_x1.3, T7_c_x1),
`tools/goflow_prediction_test_score.py --seeds 81-90`, v7 and v6 scored on the same runs. v7 ships if all three hold:

1. DSO4's queue fill: MAE below "it stays as now" at +60, +120 and +300 s on each of the three junction-agent
   datasets (9 rows of 9);
2. DSO4's queue fill: MAE below v6's on at least 7 of the 12 rows (4 datasets x 3 horizons);
3. the long-queue alerts at the shipped threshold (0.7): at least 40 % of alerts right and at least half of the long
   queues caught, on T7_ao_x1.15 and T7_ao_x1.3.

GO-FLOW's occupancy at +120 s is reported against "it stays as now" and v6, not part of the bar.


## Forecasters v7: dev, test, shipped (10 Nov, morning)

**Data.** `tools\goflow_v7_build.bat` on the PC made D7_ao_x1, D7_ao_x1.15 and D7_ao_x1.3 (seeds 101-130 and 11-20, a
day at clock 30 from 06:00, accidents, trips with a purpose, no deadlock breaker) and trained the four models
(all steps EXIT 0, ALL DONE). To save time, D7_c_x1 and the four test sets T7_* (seeds 81-90) were made in the cloud
with the same code (SUMO 1.27.1) while the PC worked, checked by md5 after copying, and the bat took them as they were
(`--resume`: every seed there). A cloud copy of x1.3 (`D7_ao_x1.3_cloud`) was not used. The cloud runs carry no git
commit in their meta (the cloud copy is not a git checkout).

**Dev** (seeds 11-20, each dataset of D7_all on its own, `goflow_prediction.evaluate --part`, DSO4 with its GO-FLOW
model, the live code path): queue fill MAE, v7 / v6 / "it stays as now":

| Data | +60 s | +120 s | +300 s |
|---|---|---|---|
| agents x1 | 0.0370 / 0.0393 / 0.0421 | 0.0412 / 0.0419 / 0.0486 | 0.0463 / 0.0471 / 0.0550 |
| agents x1.15 | 0.0387 / 0.0407 / 0.0456 | 0.0419 / 0.0425 / 0.0514 | 0.0464 / 0.0476 / 0.0580 |
| agents x1.3 | 0.0474 / 0.0509 / 0.0546 | 0.0526 / 0.0542 / 0.0631 | 0.0606 / 0.0618 / 0.0738 |
| coordinated x1 | 0.0406 / 0.0551 / 0.0766 | 0.0472 / 0.0557 / 0.0554 | 0.0556 / 0.0735 / 0.0896 |

The long-queue classifier on dev at 0.7: x1.15 24 of 50 alerts right, 23 of 36 caught; x1.3 61 of 137, 57 of 74.

**Test, opened once** (seeds 81-90, `tools/goflow_prediction_test_score.py`, out/goflow_prediction/test4_v7; v6 on the
same runs in out/goflow_prediction/test4_v6):

| Data | Fill MAE +60 / +120 / +300 s: v7 | v6 | stays as now | Long queues at 0.7, v7: alerts right, caught | v6 | GO-FLOW occ. +120 s: v7 / v6 / now |
|---|---|---|---|---|---|---|
| agents x1 | 0.0371 / 0.0408 / 0.0462 | 0.0397 / 0.0438 / 0.0492 | 0.0428 / 0.0493 / 0.0558 | 32 of 62, 29 of 37 | 0 of 1, 0 of 28 | 0.0362 / 0.0398 / 0.0444 |
| agents x1.15 | 0.0413 / 0.0454 / 0.0524 | 0.0443 / 0.0481 / 0.0551 | 0.0481 / 0.0544 / 0.0637 | 43 of 80, 38 of 54 | 12 of 14, 12 of 48 | 0.0388 / 0.0426 / 0.0480 |
| agents x1.3 | 0.0455 / 0.0510 / 0.0603 | 0.0496 / 0.0559 / 0.0654 | 0.0519 / 0.0598 / 0.0713 | 56 of 126, 53 of 74 | 23 of 30, 23 of 80 | 0.0418 / 0.0469 / 0.0503 |
| coordinated x1 | 0.0399 / 0.0467 / 0.0551 | 0.0542 / 0.0549 / 0.0733 | 0.0709 / 0.0542 / 0.0848 | 72 of 133, 68 of 102 | 10 of 28, 10 of 91 | 0.0393 / 0.0485 / 0.0434 |

The bar: (1) below "stays as now" on 9 of 9 rows; (2) below v6 on 12 of 12; (3) at 0.7, x1.15 54 % of alerts right
and 70 % of long queues caught, x1.3 44 % and 72 %. **Passed: v7 shipped** in `models/goflow` (dso4, dso4_goflow,
goflow_xgb, model cards; `.gitattributes` keeps them byte for byte). v6 on the new town gave few alerts (its levels
were set on the old, heavier demand). `p_long` still runs high: given 0.7-0.9, 43-56 % started a long queue within
3 min (0.9-1.0: 84-88 %). Tests: tests/test_goflow_prediction.py 19 passed, the two agents' own 21 and 15 passed.
Not yet seen in CARLA with v7 (`town05_freefly_forecast.bat`): computer use on the PC timed out at night.

**10 Nov, morning: a second try, not kept either.** Most of the cars taken off behind the accident on road -19
(seeds 14 and 18) had just pulled out of its kerb places: a new trip from a kerb strip puts the car on the kerb lane
at the strip's door, and with road -19's one driving lane standing it could not merge. Tried: a driver whose door
gives on a standing queue (a car under 1 m/s within 25 m past the door on the lane it must join) stays in the space
for now (`purposes.door_blocked`). Measured on dev seeds 11-20 at three starts (07:00, 12:00, 17:00, 30 runs a
side): cars taken off 17.8 -> 19.8 a run (07:00 14.4 -> 15.8, 12:00 11.3 -> 17.7, 17:00 27.7 -> 25.8); seed 14 at
07:00 went 40 -> 18 as meant, but other seeds got other accidents and other knots (seed 19: 8 -> 38, a crash on road 6
that the first run did not have). Per run 0 to 88 cars: one knot at the rush decides a run, and a change that moves
when cars leave changes which crashes happen. Neither change kept. What would tell more: the removals per accident
of a kind (a road's only lane blocked) over many more runs than 10 dev seeds give.


## Siren detection, step 8 (continued): a model that sees the lights flash (10 Nov, morning)

**Clips.** `tools/record_ev_clips.bat` (CARLA off screen, no SUMO): through each of the 55 inspector cameras, at 08:00
clear, 15:00 rain and 23:30 night, one ambulance and one police car at a time, lights on and off, twice each, driving
along a lane 12-50 m in front of the camera (0-12 m/s, towards it or away) or standing; 16 pictures at 10 a second,
the vehicle's visible box from the instance segmentation. 1309 clips (`out/ev_clips`, 2.4 GB). The light bars do
flash in CARLA: an ambulance's bar alternates blue and red from one picture to the next (2-3 Hz), a police bar too,
fainter by day. `tools/ev_clip_features.py` then gives, per picture, what the shipped detector calls the vehicle and
its light numbers (`cosim/siren_flash.light_numbers`: in the box's upper half, how far the reddest and the bluest
pixels stand out, the brightest pixels, the mean).

**Model** (`cosim/siren_flash.py`, `models/siren_flash.json`, `tools/ev_flash_model.py`): over 8 pictures at 5 a second
(1.6 s, the H window's pace), how much the red, blue and bright numbers move (spread, range, step to step), how red
and blue move against each other, and their means; a logistic regression (numpy only). Trained on the clips of the
detector set's train and val junctions (1074), tested on those of its 3 test junctions (235), never seen.

Share of test clips called SIREN ON (lights on: higher is better; lights off: lower is better):

| Class | Clips | Detector, one picture | Detector, 3 of 8 pictures | Flash model |
|---|---|---|---|---|
| ambulance, lights on | 59 | 0.78 | 0.92 | 0.90 |
| police, lights on | 59 | 0.59 | 0.63 | 0.76 |
| ambulance, lights off | 59 | 0.24 | 0.25 | 0.12 |
| police, lights off | 58 | 0.24 | 0.22 | 0.00 |
| all lights on | 118 | 0.69 | 0.77 | 0.83 |
| all lights off | 117 | 0.24 | 0.24 | 0.06 |

By time of day, lights on, flash model / 3-of-8 vote: clear 0.78 / 0.78, rain 0.75 / 0.65, night 0.97 / 0.89; lights
off, called on: clear 0.08 / 0.33, rain 0.10 / 0.28, night 0.00 / 0.11 (out/ev_clips/report.json).

**In the H window** (`cosim/inspector.TrackVote`): the detector process gives each vehicle box its light numbers; a
track's siren state is the flash model's once the track has 8 pictures, the detector's 3-of-8 vote before that (and
without the model). The same test clips through that code (only the pictures where the detector found the vehicle):
siren-on called on 0.85 (ambulance) and 0.69 (police) against 0.92 and 0.63 with the vote alone; siren-off called on
0.08 and 0.00 against 0.25 and 0.22. So the model mostly removes false SIREN ON (24 % of lights-off clips -> 4 %) and
finds about as many real ones. Run on the PC in CARLA (`--demo-camera 40`, the native feed): camera open, 32 FPS,
detector 28 pictures a second of capacity at 5 taken a second, no error; tests: tests/test_siren_flash.py,
tests/test_inspector_vote.py, tests/test_cctv.py (40 passed on the PC with the trails and forecaster tests).

**Limits.** The clips have one emergency vehicle at a time and its exact box; in the town the detector's box is
looser and another vehicle may hide it. Night and day only as three times of day. And, as before: the junction agents'
pre-emption does not read the cameras, so this changes the H window and the reports, not emergency-vehicle delay
(the user's decision whether the camera should get a role).

## 10 Nov (afternoon): the cameras as a back-up for lost emergency requests

**The question** (user's choice): when a junction's camera sees a siren-on vehicle and no request came from it (lost on
the way, or a vehicle without the radio), should the junction give priority anyway? Judged by emergency-vehicle delay,
on the paired dev seeds 11-20 (07:00, 30 min after 5 min warm-up, x1.0, an emergency vehicle every 45 s, `agents_observed`).

**What changed.**
* `cosim/agents_actuated.py`, `+camback` (optional, off unless named: `agents_observed+camback`): a siren the junction's
  cameras confirm on one of its approaches (or a lane within 75 m upstream of it), and no request heard at that junction
  in this step, granted or refused, gives that approach's links the same pre-emption as a request. Only what a real
  junction has: the cameras' confirmed siren and its lane, the queue they count there, the exit's room, and the
  requests it received. The request's own limits stay: 45 s per approach (ev_patience_s), at most 8 queued, an exit
  with room; a request the junction heard and refused is not overruled by the cameras.
* `request_loss` (setting, 0 by default): the share of (vehicle, junction) requests that never arrive, fixed per pair
  by a hash so paired runs lose the same ones. Only for measuring: `agents_observed@request_loss=0.1`.
* `cosim/camera_reports.py`: the simulated cameras now call sirens as the H window does since this morning: the
  3-of-8 vote, then, after 1.6 s in view, the flash model's call once (per track: ambulance lights on 0.85, off 0.08;
  police on 0.69, off 0.00, measured on the test junctions' clips); its own random stream, so the counts are unchanged
  draw for draw (seed 11 identical). `@siren_flash=0` gives the old vote alone. Nothing else reads the sirens except
  `+camback` and the GO-FLOW priority agent's camera confirmation (`agents_observed_goflow`, not re-measured).

**Results** (mean of 10 dev seeds; in brackets the seeds better than the reference):

| Run | Waiting s/km | Speed km/h | Ambulance time loss s | All EV time loss s | Cars taken off |
|---|---|---|---|---|---|
| no requests lost (reference) | 219 | 11.2 | 57.0 | 57.7 | 14.4 |
| no requests lost, + camera back-up | 222 (4/10) | 11.1 | 63.9 (4/10) | 59.1 (4/10) | 14.3 |
| 10 % of requests lost | 267 | 10.0 | 71.8 | 63.5 | 25.3 |
| 10 % lost, + camera back-up | 188 (7/10 vs lost) | 12.0 | 53.9 (7/10 vs lost) | 55.2 (7/10) | 8.3 |

With 10 % of the requests lost (about 11 lost (vehicle, junction) pairs a run), the back-up gives the delay back: ambulance time loss
71.8 -> 53.9 s, the town's waiting 267 -> 188 s/km. With nothing lost it costs about nothing (waiting +3 s/km,
ambulances +6.9 s, 4 seeds of 10 better: within the seeds' spread). The town is chaotic: one run set to another moves
waiting by 50 s/km, so only the large differences above are claims.

**What did not work first** (kept here so it is not tried again): with the detector's 3-of-8 vote alone, a falsely
confirmed siren stays confirmed while the vehicle is in view, and on two dev seeds 60-70 % of the camera's greens went
to emergency vehicles with their lights off. With nothing lost the back-up then cost +53 s/km of waiting and +12 s of
ambulance delay (3/10 seeds better); keeping refused requests refused alone did not help (+69 s/km, +13 s). With the
flash model's rates: +3 s/km, +6.9 s. With 10 % lost, against the lost run: -56, -37 and -79 s/km for the three
versions.

**Status.** Kept as an option, off by default (`agents_observed+camback`): with no lost requests it does not help, and
whether to turn it on is the user's decision. Test: tests/test_town05_agents.py (camera back-up, flash calls).

**Live check of the flash model in CARLA.** The H window ran with the flash model for about 2 hours (cameras 1722-1
and 1126-2, 18-27 detections a second, no error), but no emergency vehicle passed those two cameras while watched;
the measured numbers remain the clip test above. After about 2 hours (clock x30: the evening peak, 491 vehicles and
485 pedestrians) the viewer's SUMO thread no longer kept up with real time: 12 steps in 10 s instead of 20 (SUMO's
step 4.0 s, the incidents 1.6 s, the population 1.2 s of those 10 s), CARLA at 7.7 fps, the poses held 1747 times
("SUMO late" counts held poses, not seconds), and the picture stood still. The clock's speed only sets the time of
day; SUMO steps in real time whatever it is. (Corrected the same day: this first said x30 was too fast for SUMO.) The console repeats ultralytics' "'half' is deprecated" warning (cosim/inspector.py, `half=` in predict).

## 10 Nov (afternoon): cars taken off, where they come from

The third try announced (close road -19 when an accident blocks its only lane) was not made: drivers whose way
crosses an accident's road are already sent around (cosim/incidents._divert: the road counts as 10 min), the cars
taken off on -19 are 45 of 534, and on -19 the queue usually had a bus of line L3 in it, or the accident was on
another road (seed 18: -5) and -19 was full of its queue. Over the 30 audit runs (dev seeds 11-20 at
07:00, 12:00, 17:00; 534 cars, 0-88 a run), by what stood in front of the car taken off:

| Cause | Cars | Share |
|---|---|---|
| in the queue behind an accident | 158 | 30 % |
| right behind an accident's vehicles | 125 | 23 % |
| a queue at a red light that did not clear in 5 min | 123 | 23 % |
| a queue away from signals (priority junction, merge) | 46 | 9 % |
| behind a bus that could not go on | 44 | 8 % |
| a queue at a green that could not move (exit full) | 38 | 7 % |

Half come from accidents: their drivers wait RELIEVE_REMOVE_S and are then taken off (cosim/incidents._relieve), by
design. Fewer would need traffic to get past an accident (police letting cars by on the other lane or the kerb lane),
which is a design decision, not a fix.

## 10 Nov (afternoon): the camera back-up on by default (the user's decision)

`camback` is now one of the junction agents' default features (`FEATURES`); `agents_observed-camback` runs without
it. With no requests lost, checked at the three audit hours on the paired dev seeds 11-20 (on against off):

| Start | Waiting s/km | Ambulance time loss s | All EV time loss s | Cars taken off |
|---|---|---|---|---|
| 07:00 | 219 -> 222 (4/10) | 57.0 -> 63.9 | 57.7 -> 59.1 | 14.4 -> 14.3 |
| 12:00 | 200 -> 229 (4/10) | 59.6 -> 64.3 | 63.1 -> 61.6 | 11.3 -> 16.7 |
| 17:00 | 263 -> 232 (6/10) | 73.3 -> 71.7 | 71.9 -> 65.9 | 27.7 -> 16.2 |

No direction across the three hours (the town moves this much from any change); with 10 % of requests lost it helps
(above). The 07:00 run with it on by default gives exactly the `+camback` runs above (the same trips).

**Getting past an accident.** Asked: drivers move over carefully before the police come, then pass on a free lane. That
is already how incidents work: from the crash on, drivers on the blocked lane are told to merge up to
150 m before the wrecks and on the road before it before the junction (`_merge`), those stopped right behind pull out
as they roll (`_squeeze`), and one lane stays open beside the wrecks at 30 km/h, the kerb lane where it is the only
one (as on -19). Nothing added. In one bad run (17:00, seed 18) the accidents were on the centre's 54-64 m two-lane
roads: the free lane stood in a queue to the next signal, and 946 drivers took another way during one of them. Cars
stopped right behind a wreck and taken off: 38 in the 30 audit runs (7 %), the share a zipper merge could still help.

## 10 Nov (evening): the viewer's SUMO thread at the evening peak

**Why.** After 2 hours live (evening peak, 491 vehicles, the H window open) the viewer's SUMO thread fell below real
time and the picture stood still (above). Measured: `town05_freefly.bat` with `--set-time 17:30 --clock-speed 1 --demo
900` (the town fills to the 17:30 level, ~490 vehicles, in 15 min), the thread's own timing every 10 s; and cProfile
of the same town in SUMO alone (17:30, seed 18, 20 min).

**What took the time.** The simulated cameras (agents_observed's reports, cosim/camera_reports.SimulatedCameras):
one TraCI call per camera lane every step (475 lanes) and two per vehicle seen, 45 ms of every 0.5 s step in SUMO
alone, half the run's time; in the viewer they sit in "sumo.step". Then the ground under each drawn vehicle
(cosim/run.Poser.begin_step: a CARLA map lookup on each road level, every vehicle, every step).

**Changed (both exact).**
* SimulatedCameras reads each camera lane's vehicles from a lane subscription (one answer with the step) and the
  vehicles' lane positions and speeds from the vehicles' subscription (`wire` adds the lane position to it); a lane
  or vehicle not answered yet is asked as before.
* Poser keeps the ground of a vehicle that has not moved since the last step (x, y, angle and length the same).
* Tried, not kept: reading only the routes of vehicles near an accident in `incidents._divert` (within 7 roads of
  it is almost the whole of Town05: 52 197 calls instead of 54 521).

**Same traffic.** Dev seed 11 at 07:00 and seed 18 at 17:00 (probe runs, 30 min): the same trips, every one
(tripinfo identical), in 56-59 s instead of 95-101 s. Tests: tests/test_pose_proc.py (the ground kept), the camera
tests in tests/test_town05_agents.py (39 passed on the PC).

**Viewer, same settings, before / after** (SUMO thread busy, ms in every 10 s; CARLA fps):

| Vehicles in town | Before | After | CARLA fps before / after |
|---|---|---|---|
| 200-300 | 4720 | 2709 | 31.5 / 32.1 |
| 300-400 | 6164 | 3533 | 25.5 / 25.4 |
| 400-500 | 8098 | 4481 | 21.2 / 21.6 |

At the peak the thread is now busy 45 % of the time instead of 81 %. The rest, at 490 vehicles: SUMO's own step and
the cameras 1.8 s, the incidents 1.2 s (mostly `_divert` reading every route every 5 s per accident), the population
0.8 s, the ground 0.4 s. CARLA's frame rate is the GPU's and did not change.

## 10 Nov (evening): the ring road carrying the town's trips, tried, not kept

**Why.** For every car to have a trip with an end (and so a trip-time estimate), the ring road's own cars (looping,
appearing and leaving out of sight, cosim/highway.py) would go, and the town's trips would use the ring where it is
quicker, as drivers would (`router_knows_ring`: the trips' route search knows the ring's 80 km/h; HighwayTraffic with
peak 0 has no cars of its own). Option `ring_trips` in tools/goflow_prediction_dataset.one_run (off by default).

**Measured.** 07:00, dev seeds 11-20, 30 min, agents_observed, before -> after:

| | Before (ring cars) | After (ring as real trips) |
|---|---|---|
| cars on the ring | 26.6 | 2.8 |
| town share standing | 0.63 | 0.69 |
| trips done | 323 | 289 |
| cars taken off | 14.3 | 31.6 |

Few trips are quicker by the ring (2.8 cars on it instead of 26.6), and the town's own roads take the rest: more
standing, fewer trips done, twice the cars taken off. Not kept: `ring_trips` stays off. Still the user's decision: give
the ring cars a planned trip along the ring (appear and leave out of sight as now) so they too have an estimate, or
leave them without one.

## 10 Nov (evening): every car's trip time estimated when it sets off

**What (the user's request).** Every car but buses and trucks has an estimate of its trip, from where it sets off to
where it ends, made once at the start as a navigation app would; the follow panel shows it. `cosim/trip_estimates.py`
(TripEstimates): a trip runs from a departure or pulling out of a parking space to parking, a stop at its route's end
or arrival. The estimate: each road's typical time by the road taken next (its signal's wait in it) from the table of
past trips, the first road from where the car is, the last at its limit up to the stop; a road the table does not know
at its limit. A new way keeps the estimate (noted); a new destination ends the trip ("extended") and starts another.
Never re-estimated on the way. Wired into the viewer (cosim/freefly.py, stepped right after SUMO's step) and the panel
(cosim/route_minimap.py: ITS TRIP: DRIVING / DONE IN, ESTIMATED AT START, EST. TIME LEFT / OVER ESTIMATE / OFF BY, from
X to Y · N m left; the agent's own tiles left out when a trip estimate exists, as asked). The agents decide nothing
from it; SUMO's real times only grade it (tools/eval_trip_estimates.py).

**A table from the viewer's own traffic.** tools/route_delay_table.py: --start-hours, --trips purpose, --ring-trips,
--junction-blocker, and `--road-times live`: each car's time on a road read from the run's own step (the road every
car is on, which SUMO sends with each step), filed by the road taken next, instead of SUMO's route output. Why: the
route output has only cars that finished (cars parked at home never "arrive": the samples leant on through traffic
and visitors), and with exit times SUMO quit twice on train runs ("vector too long", seed 105 from 12:00, seed 106
from 07:00; the same runs pass with the live reading). Left out: a road where the car parked or stopped, the first
road of each trip (it starts part way along it), a road left by teleporting, its last road. On seed 101 from 07:00 the
two readings agree where both have the movement (79 movements with 5+ samples: median difference 0.0 s) and the live
one has twice the samples (2195 against 1114). Built from TRAIN seeds 101-110, x1, from 07:00, 12:00, 17:00,
purpose trips, junction blocker 0 as the viewer: models/trip_estimates/movement_times.json, 207 movements, 49 495
samples. tests/test_route_delay_table.py (3).

**A fault found by the accuracy run.** SUMO says a car's parking ended one step before it drops that stop: the trip
opened at that step took the space it was leaving as its destination (same road, same spot, 0 s). 43 % of the trips
to a place on the dev seeds (4220 of 9820), real time ~112 s, estimate 0. Seen car by car in one dev run (the stop
list at the trip's start: the space left first, the place it goes to second). Fixed: while the car still stands at
its stop, that stop is skipped (`getStopState & 1`); tests/test_trip_estimates.py (5).

**Accuracy, dev seeds 11-20 from 07:00, 12:00, 17:00** (30 min each after 5 min warm-up, agents_observed; trips
that ended parked, at a stop or at their route's end; error = real - estimate; tools/eval_trip_estimates.py, the
same trips scored again with each table by `--rescore`, which gives a run's own estimates back exactly):

| | Trips | Median abs error | Median error | Within 20 % | Within 60 s |
|---|---|---|---|---|---|
| before the fix (new table, mean per road) | 13 009 | 55.5 s | +6.7 s | 0.22 | 0.53 |
| new table, mean per road | 14 968 | 33.1 s | -16.0 s | 0.36 | 0.74 |
| new table, median per road | 14 968 | 27.1 s | +5.0 s | 0.39 | 0.78 |
| old table (models/route_guidance), mean | 14 968 | 35.9 s | -23.1 s | 0.34 | 0.73 |
| **old table, median per road (kept)** | 14 968 | 27.2 s | -1.2 s | 0.41 | 0.80 |

The sum of each road's mean overestimates (a signal's wait is skewed: a few long waits); the median per road does not.
Old against new table, both with the median, paired by run: the old is better in 22 of 30 runs within 20 % and 26 of
30 within 60 s, at each start hour; the median error ties (15 of 30). The old table was made from random trips at
x1, x2 and x3 traffic: its slower roads likely make up for what the estimate leaves out (slowing into a space), so
the edge is partly luck; it is the one measured better. Kept: the old table, each road's median
(TripEstimates(stat="median_s"), the default); the new table stays for later work. Selected on dev; no test seeds
opened.

With it: trips to a place 0.43 within 20 % (median error +2.9 s), through traffic 0.38 (-10.1 s); trips that took
60-180 s 0.51 within 20 %, 0.91 within 60 s; under 60 s 0.17 within 20 % (one red light is more than 20 %); 3-10 min
0.34 (these are the slow ones: an accident, a long queue; +61 s). The town hour changes little (0.31-0.46 within
20 %): no table by hour needed. Trips given another way (709): 0.32 within 20 %, +40 s.

**keep_driving.** A car gets a new destination only when its trip would end in view (before: also whenever the town
had fewer cars than its curve). Viewer at the evening peak, `--set-time 17:30 --clock-speed 1 --demo 900`, before /
after: trips extended 710 -> 652, vanished in view 3 -> 1, vehicles in town at 17:45 503 -> 507 (the same fill minute
by minute). Most new destinations were given in view anyway: cars without a place to park (through traffic,
emergency vehicles) about to end in front of the camera; their trip ends "extended" and a new one starts with its own
estimate. Ring-road cars are not touched (routes of 200 laps, `_keep_round`). Tests on the PC: 51 passed
(test_trip_estimates, test_route_guidance, test_town05_agents, test_pose_proc).

**In CARLA** (tools/freefly_follow_demo.bat: a car followed, a breakdown staged on its way on purpose). A car to
Brasserie du Nord parked in 2:23 against 2:19 estimated (panel: DONE IN 2:23, ESTIMATED AT START 2:19, OFF BY
+0:04). Two panel faults seen and fixed: the trip named a shared car park's main place ("to Hôpital" for a car going
to Brasserie du Nord; now the place of the car's own trip plan, `plan_names`); after the town changed the followed
car's route (an accident's detour) the panel said "0 m left" (the route agent read SUMO's route index against its old
list; now its way left is SUMO's route, `RouteGuide._follow_sumo`; a SUMO test for it in tests/test_route_guidance.py).
At 08:00 a car estimated 2:51 drove over 4 min: a detour at 0:56, then the demo's staged breakdown on its next roads
and a severe crash beside them; the estimate is not made again on the way (as asked), the panel says OVER ESTIMATE.
Tests on the PC: 57 passed (test_trip_estimates, test_route_guidance with its SUMO tests, test_route_delay_table,
test_town05_agents, test_pose_proc).

Job runner: a job's `set` stays for the jobs after it (one window calls them all): the follow demo first ran at 17:30
with the peak demo's FREEFLY_ARGS. Jobs now start with `setlocal`.

## 10 Nov (night): the time left as a navigation app gives it, ring-road cars with planned trips, the panel's lag

**Research (the user asked for Google Maps' way).** Google sums road-by-road times; live times come from its users'
phones (the paper's real-time input: speeds of the last 2 minutes), historical ones from the same hour of past weeks,
a learned model combines them; in Derrow-Pinion et al., "ETA Prediction with Graph Neural Networks in Google Maps"
(CIKM 2021), the real-time sum beats the historical one near the start of a trip and loses to it an hour ahead, and
combining beats both. Uber (DeepETA) adds a learned correction to a routing-engine sum. In the app the arrival time
keeps updating on the way, its colour says the delay against usual traffic (green, orange, red), the route line is
coloured by traffic (green, orange, red, dark red). Google does not publish its thresholds: ours are in
cosim/live_eta.py.

**What a real system has here.** `cosim/live_eta.py`: a share of the town's and the ring's cars run the app (a hash of
the id, the same cars at any share); `ProbeTraffic` keeps, for each road, the times of app cars that left it in the
last 2 minutes (from the step's subscriptions; a road where the car parked or stopped is left out). A road's live time:
their median shrunk toward its usual time by how few they are, (n * median + k * usual) / (n + k). The time left
(`time_left`): from where the car is, each road ahead blends live and usual, weight exp(-tau / blend) for the live
one, tau the time until the car gets there; the last road to its stop at its limit, as the start estimate. SUMO's
truth only grades it. The start estimate stays as it was.

**Measured on dev seeds 11-20** (from 07:00, 12:00, 17:00; every 30 s of every trip, the time left by each method
against the real time left; tools/eval_trip_estimates.py, `time_left` in its report; 58 022 checkpoints):

| Time left | Mean error | Median error | Within 20 % | Within 30 s |
|---|---|---|---|---|
| the start estimate minus the time driven | 87.1 s | 30.7 s | 0.19 | 0.49 |
| usual times from where the car is | 44.7 s | 20.0 s | 0.295 | 0.637 |
| + live: 25 % of cars, blend 120 s, shrink 10 (kept) | 44.3 s | 19.9 s | 0.299 | 0.640 |
| + live, no shrink | 47.0 s | 21.7 s | 0.28 | 0.61 |
| + live, 10 % / 50 % of cars | 44.5 / 44.1 s | | | |

Estimating again from where the car is halves the error; live traffic adds a little (better than usual times in 25 of
30 runs on the mean error, 22 of 30 within 30 s), most with 3-10 min left (mean error 158.8 -> 156.3 s). Tried, not
kept: raising a road's live time to how long app cars standing on it have been there (even only past twice the usual
time and a minute more): on a dev run the mean error went from 54.3 to 63.7 s (the queues clear, or the cars behind
are sent around them). The large errors left are trips held behind accidents: nobody knows how long a crash stays.
Checked again at hours not used to choose (the same dev seeds from 09:00, 14:00, 19:00, 55 480 checkpoints): usual
times 68.8 s mean error, with live traffic 67.8 s, better in 27 of 30 runs (24 of 30 within 30 s); the start estimate
minus the time driven 121.6 s.

**The panel** (cosim/route_minimap.py), as a navigation app: TIME LEFT (from where the car is, live traffic in, every
second; green, orange or red by its delay against usual traffic: over 15 % and 20 s orange, over 40 % and 45 s red),
DISTANCE LEFT, ESTIMATED AT START (made when it set off, never changed); a line "driving 0:52 · usual traffic · live
traffic in 16 % of it" (or "+1:20 vs usual traffic"); the roads ahead on the map coloured by their live time against
their usual one (orange from 1.25 times, red from 2, dark red from 3.5). At the end: DONE IN, ESTIMATED AT START, OFF
BY, as before. The viewer steps the app's view with the town (FreeFly.live_eta_step) and works out the followed car's
time left once a second (cosim/live_eta.for_car). Seen in CARLA (tools/freefly_follow_demo.bat, 08:00): "TIME LEFT 2:49
· DISTANCE LEFT 286 m · ESTIMATED AT START 2:51".

**The mini-map's car ahead of the town's (the user saw it).** The mini-map drew the followed car at SUMO's newest
step; the picture shows the poses a step or two later (the SUMO thread runs ahead, the poses are smoothed between
steps). The pose process now puts the followed car's pose for the next picture in a shared array (pose_proc.run
`shown`), and the mini-map draws the car there (SUMO's position only when none came in the last 0.5 s). A first
version made `fsnap` local in the overlay's paint function and the viewer stopped at once (fixed before any run).

**Ring-road cars with planned trips (the user's request).** Each ring car (cosim/highway.py) now drives a planned
trip: from where it appears (out of sight, as before) 2-6 of the ring's roads (590-859 m each: 1-3 laps), leaving part
way along its last road (150-440 m, SUMO's arrivalPos), so it has a start, an end and an estimate like the town's. If
that end is in sight when the car is on its last two roads, it drives one more lap (the trip ends "extended" in the
estimates and another starts); a car given another way by an accident's detour goes round to a trip's end. The
viewer's keep_driving leaves ring cars alone. `HighwayTraffic(planned=False)` and `eval_trip_estimates.py
--ring-laps`: the old 200 laps. The ring's road times from its own cars (tools/route_delay_table.py --prefix hw, train
seeds 101-105 from 07:00, 12:00, 17:00: 4 movements, 5979 samples, models/trip_estimates/ring_movement_times.json;
the town's table had 0-21 samples on them, from town cars waiting to turn off): ring-to-ring movements in the
estimate come from it.

Ring trips' estimates, dev seeds 11-20: 3948 trips from 07:00, 12:00, 17:00, median error 24.0 s on 190 s trips, 71 %
within 20 %; 3545 trips from 09:00, 14:00, 19:00, 23.7 s, 69 %.

The town, the same 60 paired runs (dev seeds 11-20 from 07:00, 09:00, 12:00, 14:00, 17:00, 19:00), 200 laps against
planned trips:

| | 200 laps | Planned trips | Runs better |
|---|---|---|---|
| cars on the ring | 21.0 | 19.6 | |
| town cars standing (share) | 0.625 | 0.642 | 27 of 60 |
| trips parked | 373.7 | 365.0 | 23 of 60 |
| cars taken off (accidents, knots) | 16.9 | 21.5 | 26 of 60 |

A small cost on the means, near half the runs either way (the town moves this much from any change). No ring car went
onto the town's roads (counted every minute; seed 11 from 12:00, run on another SUMO build, so not the same run). What may cost: about three times more ring cars
appearing mid-road (a trip each 1-3 laps instead of 200 laps). Kept (the user's request); next to try: longer trips.

Tests on the PC: 66 passed (test_highway, test_trip_estimates, test_route_delay_table, test_route_guidance with its
SUMO and CARLA tests, test_town05_agents, test_pose_proc).

## 11 Nov (night): the open points checked in the viewer

**The mini-map's car against the town's (measured).** `--demo-follow` now logs every half second, during the follow,
the distance from the CARLA car (its actor's transform) to where the mini-map draws it, and to where it drew it before
(SUMO's newest step): `minimap_vs_car` in out/town05_freefly_follow.json (`map_gap_summary`). 08:00, 463 samples:

| | Median | 95 % |
|---|---|---|
| now, all | 0.0 m | 0.53 m |
| now, car moving (over 2 m/s, 102 samples) | 0.34 m | 0.75 m |
| before, car moving | 20.9 m | 30.4 m |

The 20 m the user saw is gone. Screenshots: out/freefly/town05_follow_0*.png (11 Nov 00:3x).

**The viewer at the evening peak with the app's live view** (`--set-time 17:30 --clock-speed 1 --demo 900`, SUMO thread
ms in every 10 s, against the two runs of 10 Nov with the same settings):

| Vehicles in town | 10 Nov (two runs) | 11 Nov | CARLA fps |
|---|---|---|---|
| 200-300 | 2709 / 2727 | 2647 | 31.7 |
| 300-400 | 3533 / 3373 | 3315 | 24.7 |
| 400-600 | 5106 / 4619 | 4468 | 21.0 |

No cost to see (`sumo.step`, which now holds the estimates and the app's view, 1804 ms against 1956-1997); SUMO never
late.

**Ring cars vanishing in view.** That peak run: 255 ring trips given one more lap (their end in sight), but 11 ring
cars vanished in view (none with 200 laps; 19 vehicles in all against 1-3 on 10 Nov). Three causes, found with a log
line for each ring car that vanished (its last check) and runs on both SUMO builds:
* One more lap was a new route given in SUMO (`setRoute`, a route with a loop in it, the car's `arrivalPos`): on the
  PC's SUMO (1.27.1, .venv) the car then left within 1-2 s, in view; on SUMO 1.28 it drove the lap (175-240 s,
  checked). Checking every SUMO step instead of every 2 s did not help (9 vanished). Now SUMO keeps every ring car's
  200 laps; the ring counts the roads each car drives (`passed`) and, on the last road of its trip past its end, takes
  it off if it is out of sight (as the ring's surplus always was) or moves its end one lap on (`HighwayTraffic._at_end`;
  in the viewer every step near the end, `check_ends`). The estimates read the roads left from `end_of`, see a moved
  end as a new trip, and get the trips that ended from `take_ended` (SUMO lists a car taken off by TraCI in no arrived
  list, whatever the reason given: the first dev re-run graded 1 ring trip).
* An accident on the ring (the same one each run: incident 2, 110 s, a rear-end crash on -38): the accident code sends
  the cars heading for it another way, to the last road of their route (`_divert`, and `_relieve` after 45 s): for a
  ring car a ring road close by; those standing at the end of -36 ended their trip in the same step, in view (7, then
  3 with the viewer putting them back on their laps every step). Ring cars have no other way than the ring: they are
  no longer sent another way (cosim/incidents.py; they pass on the lane left open, as before);
  tests/test_removals.py.
* The time left from where a ring car is took the nearest lap's copy of its last road (its route goes round 200
  times): 83 s short on the dev seeds. Now the roads ahead come from its plan (`live_eta.roads_ahead`).

Evening peak, all fixes (`--set-time 17:30 --clock-speed 1 --demo 900`): 99 ring trips, 72 one more lap, 0 ring cars
vanished in view (the ring accident happened again), 2 vehicles in all (a town car and a recovery truck), as on 10 Nov;
SUMO thread 4448 ms in every 10 s at 400-600 vehicles, CARLA 20.7 fps.

Dev seeds 11-20 from 07:00, 12:00, 17:00, final code (out/trip_estimates/dev_ringtrips4), against the same runs with
200 laps: 
| | 200 laps | Planned trips | Runs better |
|---|---|---|---|
| cars on the ring | 21.4 | 21.2 | |
| town cars standing (share) | 0.620 | 0.634 | 15 of 30 |
| trips parked | 396.0 | 392.7 | 16 of 30 |
| cars taken off (accidents, knots) | 16.2 | 23.6 | 15 of 30 |

Half the runs either way; the means a little worse (cars taken off: a few runs). Ring trips: 3790, median error
22.4 s on 202 s trips, 76 % within 20 %, 91 % within 60 s, no bias. Their time left (every 30 s, 22 424 checkpoints):
from the start estimate 32.9 s mean error, from where the car is 20.1 s, with live traffic 19.7 s (63 % within 20 %).
The town's trips are unchanged by these fixes (time left: start 64.3 s, usual 48.5 s, live 47.9 s, 58 462
checkpoints). Tests on the PC: 107 passed (with tests/test_live_eta.py, test_removals.py, the incident tests).
**Job runner.** Tried `< nul` so that a viewer stopping at "Press any key" could not hold the queue: the viewer then
quit after a minute (it reads keys from the console, nul ends at once) and `timeout` refused it. Instead
tools/town05_freefly.bat skips its pause on an error when FREEFLY_NOPAUSE is set (the jobs set it; a double-click keeps
the pause).

## 11 Nov (night): Omar's accident agent in the viewer, in shadow

Full audit, plan and numbers: `docs/INCIDENT_AGENT_AUDIT.md`. His agent (GO-FLOW `feature/incident-carla-integration`,
`03a7010`) is in `agents/incident_detection/`, his files byte-identical; ours around them: `stream.py` (his functions
on a live camera, a window out as soon as it closes), `serve.py` (his process, own venv:
`tools\setup_incident_agent.bat`), `cosim/incident_agent_proc.py` (YOLO + ByteTrack on the GPU),
`cosim/incident_agent.py` (CARLA cameras at 5 fps, picture time = the SUMO second shown). Viewer:
`--incident-agent "2328-2 looking W;2014-3 looking E"`. **Shadow:** verdicts shown and written to
`out/incident_agent/<time>/windows.jsonl`, never a report: no model meets the agreed criteria.

* Streaming = his batch path: 0 differences in 2 164 windows of real CARLA video (`replay.py`).
* Viewer 47.4 fps -> 41.6 with 2 cameras, 38-39 with 4; the GTX 1650 keeps 2 cameras at 5 fps beside CARLA (4: half
  the pictures lost; 93-137 ms a picture against 31 alone; batching did not help live, kept as an option).
* Graded run (seed 12, 40 min, crash focus 0.5 on the 2 cameras, 6 crashes counted): v1 6 / 6 but 48 false alarms
  per camera-hour, in alarm 56 % of normal time; his crash moment 3 / 6 within 6 s, no real false alarm in 0.37 h,
  and it changes between two cameras at the same pose. His process killed live: back in 10 s, one window lost.
* `lap` added to our venv (ByteTrack; `tools/setup_ml.bat`). Clips from before the evening of 6 Nov have `sim_s` in
  0.5 s steps: time them by `carla_s` (`tools/incident_agent_rows.py` does).

## 11 Nov (day): the viewer's picture and overlay (the user's request: make it look impressive, 20 fps kept)

Rules of the job: fps before and after every change on the same flight (`--demo 70`, full screen, forecasts on; one
run to the next differs by about 1 fps), screenshots of the running town, the detector checked on the new picture,
`tools\test.bat` passing. Everything is in `Claude outputs\vq\` (base, lab, t).

**Before** (`medium_fs` of 10 Nov, colour-key overlay), flying high / street level, fps (1 % low):

| town time | before | shipped |
|---|---|---|
| 12:00 | 43.5 / 52.2 (34.8 / 36.2) | 43.3 / 53.5 (35.6 / 37.8) |
| 17:30 | 34.1 / 39.1 (27.8 / 29.3) | 36.2 / 41.1 (30.4 / 32.8) |
| 22:00 | 29.8 / 50.3 (21.4 / 26.8) | 29.6 / 45.6 (22.3 / 26.0) |

Tests: 353 passed before, 358 after (5 new, `tests/test_overlay_alpha.py`).

**What the frame rate depends on** (measured, not assumed):
* With the town's traffic the viewer waits for the CPU, not the GPU: at 17:30 the same 34-35 / 39-40 fps at 58, 67,
  75, 88 and 100 % resolution; GPU 40-77 % busy at 58 %, 50-85 % at 88 %, 3.0-3.3 GB of the 4. Without traffic the
  same views give 100 / 160 fps (`tools/look_lab.py`).
* At night the 302 street lights' shadows are the cost: no traffic, 23:30, flying high 44 fps, with shadows off 80;
  their resolution changes nothing, their number does.

**Kept:**
* **Overlay with per-pixel alpha** (`cosim/overlay.AlphaPresenter`, `--overlay alpha`, the default; `key` = the
  old colour key, also the fallback). The clock card, mini-map, forecast panel, markers, chips and place labels are
  glass with rounded corners, a hairline border and a soft shadow (`cosim/ui.py`: `glassify` turns a panel drawn
  the old way into glass, so each panel keeps its drawing code; `glass_fast` for the mini-map, redrawn every
  picture). Same fps (35.6 / 40.0 against 34.7 / 40.0). Main loop: overlay 153-165 ms/s against 132-136 with the
  key, 30 loops a second both. The first version cost 563 ms/s (numpy): now pygame's own multiply, and the mini-map
  held 0.2 s while flying.
* **A bug caught by its screenshot, not by its numbers:** pygame 2.6.1's `premul_alpha` on a subsurface reads the
  wrong rows: stripes over the whole picture, the fps unchanged. A copy first; the test compares any rectangle with
  the plain formula.
* **Render profile** `tools/fps_test/engine_medium_fs.ini`: 80 % resolution instead of 58 % (1536x864 on this
  screen), Catmull-Rom filter for the temporal anti-aliasing, screen-space reflections on (glass towers, cars: 35.4 /
  40.2 with, 35.5-35.9 / 40.6-41.1 without), anisotropy 8. The viewer in a window uses it too; `medium` itself is
  unchanged (the datasets and films were made with it). `set FREEFLY_INI=name` starts with another profile.
* **The sky by the time of day** (`cosim/look.py`, `--look town`, the default; `old` = one clear weather): 28 %
  clouds and less Rayleigh scattering by day (a deeper blue), more scattering and fewer clouds at a low sun (orange),
  12 % clouds at night; CARLA's defaults for the fields the old weather left at 0. No cost (lab, 18 weathers, +-5 of
  100-160 fps).
* The prompt ("H Inspect ...", "F Follow ...") sits above the accident markers that covered it.
* An unattended job (`FREEFLY_NOPAUSE`) quits when loading fails (it waited for Esc and held the job runner).

**The detector on the new picture** (`tools/look_detector_check.bat` and `.py`: `make_dataset_town05.py --look`,
seed 11, 12 cameras, 6 hours from 08:00 to 23:30, 286 pictures, the same traffic in every set, labels from the
instance camera; `carla-yolo26n-t05` at the inspector's 0.35):

| set | objects found | vehicles | siren ON | siren off | false boxes / 100 pictures |
|---|---|---|---|---|---|
| before (twice) | 0.638, 0.638 | 2572, 2577 of 3854 | 62, 61 of 159 | 16 of 90 | 27.3, 26.6 |
| old profile + new sky | 0.643 | 2599 | 61 | 16 | 30.4 |
| with `r.Tonemapper.Sharpen=1.6` | 0.585 | 2365 | 45 | 15 | 36.8 |
| with the street lights' small shadows culled (twice) | 0.640, 0.638 | 2589, 2578 | 54, 56 | 18, 17 | 28.8, 28.8 |
| **shipped** | **0.637** | **2575** | **58** | **19 of 91** | **29.7** |

* Sharpening looked best and cost nothing, and it is out: it sharpens the cameras' pictures too and the detector
  loses 8 % of the vehicles.
* Culling the street lights' small shadows (`r.Shadow.TexelsPerPixelSpotlight/Pointlight=0.5`, `FadeResolution=256`,
  `MinResolution=128`) gives 44.9 / 55.7 fps at 22:00 (1 % low 32.0) and the same picture to the eye, but at night
  the cameras find fewer emergency vehicles with their lights on: 18-19 -> 12-13 of 60 at 21:45 and 23:30 (by day
  the same). Not the default; kept as `engine_medium_fs_night.ini` (`set FREEFLY_INI=medium_fs_night`): the user's
  decision.
* What is left against before: siren ON 58 against 61-62 of 159 (80 % resolution alone 58, reflections alone 58),
  7 more false boxes in 286 pictures, 3 more siren-off found. One run each; two runs of the same set differ by 1-2.

**Evening peak** (`--set-time 17:30 --clock-speed 1 --demo 900`, full screen, no forecasters; measured with the
profile of that hour, which also had the sharpening and the shadow culling: by day the lab gives the same fps with
and without them): 10 s windows by vehicles in town, new / before: 150-250: 35.7 / 34.5; 250-350: 28.9 / 27.8;
350-450: 25.6 / 23.6; 450-600: 20.7 / 20.1 (lowest window 18.1 / 17.5). **Above 450 vehicles the viewer is at the
20 fps limit and under it in some windows, before as now**; in the 70 s demo at 17:30 the town holds 216. GPU
3.63 GB against 3.57, 48 % busy against 43 %.

**Tried, not kept:** exposure (the road is almost white at midday): `r.ExposureOffset`, `r.TonemapperGamma`,
`r.Color.Mid`, `r.DefaultFeature.AutoExposure.Bias`, `r.EyeAdaptation.LensAttenuation` change nothing in the packaged
build and `r.EyeAdaptationQuality=0` whites the picture out; fog, wetness, dust, Mie scattering (nothing worth
having); the map's 82 "building" lights at night (no visible change); 3 shadow cascades at 2048 (-10 % without
traffic, little to see); view distance, LOD and foliage back to 1.0 (nothing to see in the flight); ambient
occlusion off (+10 % without traffic, not needed).

**Not done, and why:** cars, people, roads and buildings are CARLA's packaged meshes and materials: changing them
needs the Unreal Editor (and the vehicles' look feeds the detector). The CARLA window cannot be seen through
computer use (masked): every picture is the viewer's own window grab.

## 11-12 Nov: junction agents, forecast-driven gating (`+gate`): dev runs, and the bar written before the test

The user's order for the agents (11 Nov): 1 gating from the forecasts, 2 traffic management round accidents,
3 parking guidance; nothing on by default before it is tested.

**The rule** (`cosim/agents_actuated.py`, optional `+gate`, off by default): a green whose movements lead to an
approach at the next signal where DSO4 gives a long queue within 3 minutes a chance of `gate_p` or more (at least
`gate_share` of the green's movements lead there; `fed`, the approaches within 400 m going on with the traffic: 54
greens) is ended after `gate_min_s`, when another phase has demand, its own approach is under 60 % full and is not
itself forecast to queue. The spill rule does this once the exit is full; the gate before. Unit test:
`tests/test_town05_agents.py::test_gate_...`.

**Dev** (`tools/eval_agents_forecast.py`, seeds 11-20, the forecasters' recipe: 30 min after 5 min warm-up from
06:00 at clock 30, crashes, buses, ring road; v7 in the loop; `out/gate/dev`, 120 runs in `out\eval_venv` on the
PC). Paired against `agents_observed`, mean difference (seeds better of 10):

| Traffic | Setting | Greens cut a run | Waiting /trip | Time loss /trip | Throughput /h | Ambulance time loss | Long-queue approach-min |
|---|---|---|---|---|---|---|---|
| x1 | p 0.7, share 0.5, 10 s | 1.6 | +0.8 s (3) | -1.4 s (3) | +1 (2) | -1.0 s (4) | -1.5 (3) |
| x1 | **p 0.5, share 0.34, 5 s** | 16 | -3.3 s (6) | -3.9 s (8) | +10 (7) | -3.3 s (8) | -1.9 (7) |
| x1 | p 0.3, share 0.34, 5 s | 24 | -3.7 s (6) | -6.8 s (8) | +24 (8) | +1.5 s (8) | +1.5 (7) |
| x1.15 | p 0.7, share 0.5, 10 s | 2.2 | +1.9 s (4) | +1.8 s (3) | -4 (2) | +1.8 s (3) | +2.2 (2) |
| x1.15 | **p 0.5, share 0.34, 5 s** | 12 | +0.2 s (4) | -0.4 s (5) | -1 (4) | +2.1 s (3) | -3.3 (6) |
| x1.15 | p 0.3, share 0.34, 5 s | 28 | +6.8 s (6) | +10.5 s (2) | -16 (4) | +2.7 s (5) | +1.0 (4) |
| x1.3 | p 0.7, share 0.5, 10 s | 3.7 | +3.7 s (4) | +2.4 s (4) | -20 (2) | -5.8 s (5) | -1.0 (4) |
| x1.3 | **p 0.5, share 0.34, 5 s** | 41 | -2.1 s (4) | +2.9 s (5) | -4 (3) | +2.5 s (6) | -8.8 (7) |
| x1.3 | p 0.3, share 0.34, 5 s | 63 | -3.2 s (5) | -3.7 s (5) | -32 (2) | -6.8 s (4) | -9.8 (6) |

Reference: waiting 109.0 / 115.5 / 167.7 s, time loss 139.9 / 144.9 / 194.8 s. A seed's waiting changes by 8-16 s
(sd of the paired difference) at x1 and x1.15 and by 15-47 s at x1.3, also when the gate cut only 2 greens: the runs
are chaotic (accidents), so no row above is more than about one standard error from zero. The strict setting acts
too rarely (the strong warnings are rare: 1-9 minutes of green a run); the loose one hurts at x1.15. The middle
setting is the candidate: better at x1 on 8 of 10 seeds for time loss and ambulances, nothing at x1.15, fewer
long-queue minutes at all three.

**The bar, written before the test (12 Nov).** One setting only: `agents_observed+gate@gate_p=0.5@gate_share=0.34@
gate_min_s=5` against `agents_observed`. New seeds 91-120 (30, because of the spread), x1, x1.15 and x1.3, opened
once. The gate goes on by default only if all hold:
1. at x1 (the town's own traffic): waiting and time loss per trip lower on the mean and on at least 18 of 30 seeds;
2. at x1: ambulance time loss not higher on the mean by more than 2 s;
3. at x1.15 and x1.3: time loss per trip not higher on the mean by more than 5 %.
Otherwise it stays an option, off, with these numbers. Test seeds used after this: 21-120.

**The test (12 Nov, opened once, seeds 91-120, `out/gate/test`, 180 runs): the gate fails its bar and stays off.**
Paired against `agents_observed`, mean difference +- standard error (seeds where it was lower, of 30):

| Traffic | Waiting /trip | Time loss /trip | Throughput /h | Ambulance time loss | Long-queue approach-min | Greens cut a run |
|---|---|---|---|---|---|---|
| x1 | +1.1 +- 2.5 s (13) | +1.1 +- 2.3 s (13) | -4 +- 5 | +0.5 +- 3.4 s | +0.3 | 12 |
| x1.15 | +1.1 +- 6.1 s (15) | +2.8 +- 6.1 s (13) | -1 +- 8 | +2.1 +- 4.3 s | +3.8 | 21 |
| x1.3 | +3.0 +- 4.8 s (15) | +1.4 +- 4.5 s (14) | -4 +- 8 | +2.1 +- 5.8 s | +1.6 | 35 |

Reference: waiting 111.6 / 128.4 / 154.4 s. Bar 1 asked for lower means on 18 of 30 seeds at x1: 13 of 30 and a
mean slightly higher. No measurable effect anywhere, and no sign of one: what the dev seeds showed at x1 (8 of 10)
was their spread. Cars taken off (accidents, knots): 11.4 -> 13.2, 12.6 -> 10.9, 14.9 -> 17.8 a run. Reading: a
warning right about half the time, on 12-35 greens in 35 minutes, moves less than an accident does by chance. `+gate`
is left in as an option, off. Test seeds used: 21-120; anything new: 121 and above.

**The same gate for accidents (`+accgate`, 12 Nov): dev only, nothing to test.** A road with an open accident (the
roads the drivers' navigation counts as slow, `incidents._divert`) counts as a road at risk 1 for the greens that feed
it; no forecaster needed. Dev seeds 11-20 (`out/gate/dev_acc`), against `agents_observed`, mean +- standard error:

| Traffic | Setting | Greens cut a run | Waiting /trip | Time loss /trip | Throughput /h | Cars taken off (accidents, knots) |
|---|---|---|---|---|---|---|
| x1 | share 0.5, 10 s | 3 | -0.2 +- 1.3 s | -0.8 +- 2.1 s | -6 +- 6 | 11.2 -> 10.3 |
| x1 | share 0.34, 5 s | 23 | -2.7 +- 7.9 s | -5.2 +- 7.3 s | +24 +- 18 | 11.2 -> 11.1 |
| x1.15 | share 0.5, 10 s | 4 | +1.2 +- 4.0 s | 0.0 +- 3.8 s | -6 +- 9 | 4.2 -> 4.9 |
| x1.15 | share 0.34, 5 s | 44 | +5.0 +- 4.3 s | +5.4 +- 5.4 s | -13 +- 14 | 4.2 -> 7.4 |
| x1.3 | share 0.5, 10 s | 7 | -9.0 +- 14.4 s | -12.3 +- 13.8 s | -27 +- 9 | 23.3 -> 16.3 |
| x1.3 | share 0.34, 5 s | 46 | -6.1 +- 14.7 s | -5.7 +- 16.4 s | -29 +- 20 | 23.3 -> 18.5 |

Nothing beyond one standard error, signs mixed, throughput down more often than up. Not taken to a test (no seeds
spent). Left in as an option, off. Both gates say the same: the drivers are already sent round an accident and a
full exit already ends a green (spill); holding the feeding green earlier adds nothing that shows through the runs'
spread. **Parking guidance** (third on the list) is already what the town does: a full car park is not driven to,
the nearest other one with room is (`purposes.area_for`, counted as `redirected_full`).

## 12 Nov: bus priority at the signals (`+bus`): dev runs, and the bar written before the test

The user's choice after the gates failed. **The rule** (`cosim/agents_actuated.py`, optional `+bus`, off by default):
a line bus in service within 120 m of its signal, not at its stop and with no stop left before the line, asks for
itself (as the emergency vehicles do; `buses` gives the buses in service). At its green's end a bus about to get
through (moving, or at most one vehicle standing ahead) gets the green held, at most 10 s past the green's maximum.
With `bus_cut=1` the other car greens are ended after 10 s while it waits, in the program's order (nothing skipped,
walks never cut), unless their own road is 60 % full. A request counts only while the bus can move (exit lane not
full, at most 6 standing ahead) and for 60 s at a junction; an emergency request comes first. Unit test:
`tests/test_town05_agents.py::test_bus_priority_...`.

**Measured on the buses** (`tools/eval_agents_forecast.py`, new columns): time lost per km and speed between stops of
the line buses (SUMO's tripinfo of the `pt_*` vehicles, their laps written unfinished; time at stops not counted).

**Dev** (seeds 11-20, the forecasters' recipe, 35 min at clock 30 from 06:00; `out/bus/dev2`). Paired against
`agents_observed`, mean +- standard error (seeds better of 10). Reference: buses lose 210 / 214 / 251 s a km and
drive 11.9 / 11.7 / 10.5 km/h between stops; cars wait 109.0 / 115.5 / 167.7 s a trip.

| Traffic | Setting | Greens held, cut a run | Bus time loss /km | Bus speed | Cars: waiting /trip | Cars: time loss /trip | Throughput /h | Ambulance time loss |
|---|---|---|---|---|---|---|---|---|
| x1 | hold | 72, 0 | -18.2 +- 9.0 s (8) | +0.7 km/h (8) | -8.5 +- 9.8 s (6) | -10.1 +- 9.0 s (6) | +10 +- 15 | -0.9 +- 8.1 s |
| x1 | hold + cut | 64, 35 | -17.6 +- 4.8 s (8) | +0.7 km/h (8) | +1.2 +- 5.3 s (5) | -1.5 +- 5.0 s (6) | +4 +- 12 | -7.6 +- 5.0 s |
| x1.15 | hold | 68, 0 | +6.7 +- 15.4 s (5) | -0.1 km/h (5) | +44.8 +- 13.0 s (0) | +49.8 +- 13.4 s (1) | -58 +- 18 | +36.9 +- 11.3 s |
| x1.15 | hold + cut | 70, 38 | -17.0 +- 7.5 s (8) | +0.8 km/h (8) | +7.7 +- 8.2 s (3) | +5.7 +- 7.3 s (4) | -20 +- 11 | -4.3 +- 5.5 s |
| x1.3 | hold | 76, 0 | -42.7 +- 12.0 s (10) | +1.6 km/h (10) | -17.2 +- 13.9 s (7) | -18.6 +- 14.6 s (7) | -51 +- 12 | -31.6 +- 14.8 s |
| x1.3 | hold + cut | 66, 40 | -37.6 +- 15.5 s (8) | +1.4 km/h (8) | -24.9 +- 16.3 s (6) | -27.0 +- 16.6 s (6) | -67 +- 17 | -30.5 +- 13.8 s |

A first version kept every other rule off a green with a bus anywhere in its queue; changed (hold only at the
green's end, only for a bus about to get through) before these runs: the numbers hardly moved. Reading: the buses
gain with hold + cut at every level (8 of 10 seeds, about 8 % of their time loss at x1 and x1.15, 15 % at x1.3).
The cars: nothing clear at x1; at x1.15 hold alone is worse on every seed, which 70 held greens a run cannot explain
by themselves (the number of accidents in a run changes with any change: 2 -> 5 on seeds 12 and 15); throughput is
lower at x1.15 and x1.3 with both. Ten seeds cannot settle the cars' side (a seed's waiting moves by 15-50 s).

**The bar, written before the test (12 Nov).** New seeds 121-150 (30), x1, x1.15, x1.3, opened once, both settings
measured against `agents_observed`: `agents_observed+bus@bus_cut=1` is the candidate, `agents_observed+bus` (hold
only) is measured beside it. A setting goes on by default only if all hold:
1. buses: time loss per km lower on the mean and on at least 20 of 30 seeds, at x1 and at x1.15;
2. cars: time loss per trip not higher on the mean by more than 5 %, at x1, x1.15 and x1.3;
3. cars: trips completed an hour not lower on the mean by more than 3 %, at x1 and x1.15;
4. ambulances: time loss not higher on the mean by more than 5 s at x1.
If both pass, the one with the lower cars' time loss at x1.15. Otherwise `+bus` stays an option, off, with these
numbers. Test seeds used after this: 21-150.

**The test (12 Nov, opened once, seeds 121-150, `out/bus/test`, 270 runs): bus priority fails its bar and stays
off.** Paired against `agents_observed`, mean +- standard error (seeds better of 30). Reference: buses lose 205 /
216 / 244 s a km; cars lose 136.9 / 151.3 / 184.7 s a trip, 759 / 810 / 843 trips an hour.

| Traffic | Setting | Bus time loss /km | Cars: time loss /trip | Throughput /h | Ambulance time loss | Cars taken off a run |
|---|---|---|---|---|---|---|
| x1 | hold + cut | -0.6 +- 9.8 s (18) | +10.1 +- 8.3 s, +7.4 % (12) | +4 +- 7 | +5.8 +- 8.9 s | 8.0 -> 16.4 |
| x1 | hold | +0.9 +- 8.8 s (20) | +11.3 +- 9.2 s, +8.3 % (12) | +1 +- 9 | +10.7 +- 8.4 s | 8.0 -> 19.5 |
| x1.15 | hold + cut | -12.9 +- 8.8 s (22) | +14.7 +- 8.2 s, +9.7 % (15) | -6 +- 11 | +20.8 +- 8.7 s | 11.7 -> 16.2 |
| x1.15 | hold | -5.1 +- 10.4 s (20) | +11.5 +- 10.0 s, +7.6 % (13) | -6 +- 8 | +17.5 +- 11.3 s | 11.7 -> 16.0 |
| x1.3 | hold + cut | -12.8 +- 10.4 s (22) | +8.9 +- 9.9 s, +4.8 % (15) | +4 +- 12 | +6.9 +- 9.9 s | 16.3 -> 22.0 |
| x1.3 | hold | -6.5 +- 13.2 s (17) | +9.2 +- 9.4 s, +5.0 % (14) | +1 +- 10 | +17.3 +- 10.8 s | 16.3 -> 23.9 |

Against the bar: (1) buses lower on the mean and on 20 of 30 seeds at x1 and x1.15: hold + cut 18 of 30 at x1 with no
gain on the mean (-0.6 s), hold alone higher on the mean at x1: failed; (2) cars' time loss within 5 %: +7.4 to +9.7
% at x1 and x1.15: failed; (3) throughput within 3 %: held; (4) ambulances within 5 s at x1: +5.8 and +10.7 s:
failed. The buses' gain on the dev seeds at x1 (-18 s a km, 8 of 10) did not come again; what is left is 5-6 % at
x1.15 and x1.3 with the cut. The cars are worse in all six rows (each about one standard error, all the same way),
with 4-11 more cars a run taken off behind accidents and knots. `+bus` is left in as an option, off. Test seeds
used: 21-150; anything new: 151 and above.

**Three rules, three times the same lesson.** Forecast gating, accident gating and bus priority each looked right on
the 10 dev seeds somewhere and did nothing, or harm, on 30 new seeds. On this town a run's result moves by 8-50 s a
trip with any change to the signals, because accidents and the knots behind them decide it and a different green
changes which ones happen. Ten dev seeds cannot tell a 5 % effect from that; thirty can only just. Before another
signal rule: either a much larger effect is expected, or the evaluation needs less noise (accidents off or the same
accidents replayed in both runs).

## 12 Nov: the quiet evaluation (`eval_agents_forecast.py --no-accidents`)

The same town, traffic, buses, ring road and emergency runs, with no crash, breakdown, pile-up or look-alike: the
two runs of a seed then differ by what the signals did, not by which accidents happened. Dev seeds 11-20, x1, x1.15,
x1.3, five controllers, 150 runs (`out/quiet/dev`). Reference without accidents: cars wait 86.5 / 100.2 / 123.1 s a
trip (with: 109.0 / 115.5 / 167.7), buses lose 185 / 193 / 202 s a km (with: 210 / 214 / 251).

**How quiet.** The probe is the strict gate, which cuts about two greens a run. Spread between seeds (sd) of its
paired difference in waiting per trip:

| Traffic | With accidents | Quiet |
|---|---|---|
| x1 | 8.4 s | 0.8 s |
| x1.15 | 11.9 s | 4.1 s |
| x1.3 | 15.2 s | 10.1 s |

Ten times quieter at the town's own traffic, three times at x1.15; at x1.3 the congestion itself makes runs diverge.
With 10 seeds a difference of 1 s a trip shows at x1 (it took 30 seeds to see 5 s before).

**What it shows of the rules** (paired against `agents_observed`, mean +- standard error, seeds better of 10):

| Traffic | Rule | Bus time loss /km | Cars: waiting /trip | Cars: time loss /trip | Throughput /h |
|---|---|---|---|---|---|
| x1 | gate (p 0.5) | +0.5 +- 2.4 s (5) | +1.3 +- 1.5 s (4) | -0.3 +- 0.8 s (4) | +3 +- 6 |
| x1 | bus: hold + cut | -18.2 +- 3.2 s (10) | -2.7 +- 3.6 s (6) | -1.4 +- 3.3 s (7) | +8 +- 7 |
| x1 | bus: hold | -20.9 +- 3.9 s (9) | +2.0 +- 5.5 s (3) | +0.1 +- 4.3 s (4) | -7 +- 12 |
| x1.15 | gate (p 0.5) | -0.4 +- 3.7 s (5) | +3.2 +- 1.9 s (3) | +3.0 +- 2.3 s (2) | -9 +- 7 |
| x1.15 | bus: hold + cut | -15.5 +- 4.9 s (7) | +9.0 +- 4.8 s (2) | +5.9 +- 4.0 s (3) | -22 +- 10 |
| x1.15 | bus: hold | -13.4 +- 4.9 s (8) | +9.0 +- 2.7 s (1) | +7.8 +- 4.1 s (2) | -20 +- 11 |
| x1.3 | gate (p 0.5) | +1.4 +- 4.3 s (4) | +0.1 +- 2.4 s (5) | -0.4 +- 3.2 s (4) | +4 +- 7 |
| x1.3 | bus: hold + cut | -12.4 +- 7.6 s (8) | +4.0 +- 5.6 s (4) | +5.0 +- 6.0 s (5) | -7 +- 12 |
| x1.3 | bus: hold | -10.7 +- 7.1 s (7) | +4.6 +- 6.1 s (6) | +5.7 +- 7.1 s (5) | -3 +- 11 |

* The gate does nothing, now known to about a second a trip at x1; a little worse at x1.15.
* Bus priority is real for the buses: 10 % of their lost time at x1 (10 of 10 seeds with the cut), 8 % at x1.15, 6 %
  at x1.3. For the cars it costs nothing at x1 and about 6 s a trip (5 %) and 2.5 % of the trips at x1.15.
* With accidents the test (seeds 121-150) gave the buses -0.6 +- 9.8 s at x1 and the cars +10 +- 8 s: both within two
  standard errors of what the quiet runs show, and neither could have shown it.

These are dev seeds: they say the instrument works and what to expect, not that a rule is accepted. Bus priority
stays off (its test failed its bar). Judging a rule from now on: (1) the quiet evaluation on new seeds, for whether it
does what it is for; (2) with accidents on new seeds, for harm, with a bar that the spread allows.

**Bus priority backing off at a busy junction (12 Nov, quiet dev seeds, `out/quiet/dev`): no.** Two guards tried: no
hold while another road is half full (`bus_hold_other_max=0.5`), no cut of a green whose own road is 30 % full
(`bus_cut_fill=0.3`). At x1.15 the cars' cost stays (+7 to +10 s time loss a trip in every version, 1-3 seeds of 10
better) and the buses' gain shrinks (-15.5 -> -9 s a km); the guards hardly ever act (greens held 66-78, cut 34-44 a
run, as without). At x1 nothing changes (buses -17 to -18 s a km on 9-10 seeds of 10, cars within +-4 s). So the cost
at x1.15 is not a few holds at full junctions: just past what the town carries, any green moved to the buses' roads
is missed elsewhere. What stands: bus priority (hold + cut) is worth 10 % of the buses' lost time at the town's own
traffic at no cost to the cars, and costs the cars about 5 % at x1.15. Whether to run it at x1 only is the user's
decision; it would still need its tests on new seeds (151+).

**Bus priority at the town's own traffic only (the user's decision, 12 Nov): the bar, written before the tests.**
One setting, `agents_observed+bus@bus_cut=1` against `agents_observed`, at x1 only (with `busy`, x1.15, it would stay
off). Two tests on new seeds, each opened once:
* quiet (`--no-accidents`), seeds 151-180: (a) buses' time loss per km lower on the mean and on at least 20 of 30
  seeds; (b) cars' time loss per trip not higher on the mean by more than 3 %; (c) trips completed an hour not lower
  by more than 3 %;
* with accidents, seeds 181-270 (90: with 30 the cars' mean is known to +-6 %, with 90 to about +-3.5 %): (d) cars'
  time loss per trip not higher on the mean by more than 5 %; (e) trips completed an hour not lower by more than
  3 %; (f) ambulances' time loss not higher on the mean by more than 8 s; (g) buses' time loss per km not higher on
  the mean.
All seven, or it stays off. Test seeds used after this: 21-270.

**Quiet test (seeds 151-180, x1, `out/bus/test2_quiet`): passed.** Paired against `agents_observed`, mean +-
standard error: buses' time loss -15.2 +- 4.6 s a km (-8.0 %, better on 25 of 30: (a) passed); cars' time loss per
trip +1.6 +- 4.1 s (+1.3 %: (b) passed); trips an hour +0.3 +- 7.1 (0.0 %: (c) passed); waiting +1.4 +- 3.6 s;
ambulances +1.8 +- 3.2 s. 69 greens held and 39 cut a run. The test with accidents (seeds 181-270) follows.

**Test with accidents (seeds 181-270, x1, `out/bus/test2_acc`, 180 runs): passed.** Paired against `agents_observed`,
mean +- standard error: cars' time loss per trip +2.9 +- 4.8 s (+2.0 %: (d) passed), trips an hour -0.4 +- 4.9
(-0.1 %: (e) passed), ambulances +3.7 +- 4.8 s ((f) passed), buses' time loss -17.1 +- 5.9 s a km (-7.6 %, better on
64 of 90: (g) passed). Not in the bar: the cars' waiting +5.1 +- 4.8 s (+4.4 %), cars taken off 14.7 -> 16.7 a run.
**All seven held: bus priority is the viewer's default at the town's own traffic** (`cosim/freefly.parse_args`:
`agents_observed+bus@bus_cut=1` when `--demand-scale` is 1; `busy` and a chosen `--controller` go without). The
buses ask through `wire()` (cosim/agents_actuated.py), as the viewer and the tools connect the agents; the HUD strip
says "junction agents + bus priority"; with `--agent-hud` the decisions show over the junctions ("green held for a
bus", "green shortened: a bus is waiting", yellow). Test seeds used: 21-270; anything new: 271 and above.
Seen in CARLA (12 Nov, `town05_freefly_forecast.bat` demo with `--agent-hud`): the HUD strip says "junction agents +
bus priority"; the agents' panel lists "838 · green held for a bus"; in the 70 s demo at 17:30 the buses asked on 216
steps, 3 greens were held and 1 cut (the demo report now has the agents' counts, `agent_stats`). Frame rate as
before: 35.4-35.8 / 40.3-40.4 fps at 17:30 (before 36.2 / 41.1), 43.9 / 53.9 at 12:00 (43.3 / 53.5). Tests: 361
passed. The two guards tried on the quiet dev seeds (`bus_hold_other_max`, `bus_cut_fill`) stay as settings, unused.

## 12 Nov: the overlay redesigned (the user's request: panels as good as the best games')

**Research** (what the games do, as documented: rdr2-case-study.vercel.app, interfaceingame.com's Cyberpunk 2077
critique, medium.com/super-jump "A UX analysis of Cyberpunk 2077's HUD", uxdesign.cc on The Last of Us Part II,
thereisnosleep.bearblog.dev "The best HUDs in gaming", bugnet.io "How to design a readable HUD", a UI art director's
"7 beginner mistakes", gamedeveloper.com "6 examples of UI design"): show little and only while it matters (RDR2
hides its cores while safe, Cyberpunk shows health only when not full); one corner for the feed of events; one
colour per meaning, used for nothing else (Half-Life 2's single colour; Cyberpunk criticised for red as ornament);
two type families, titles in capitals; a dark plate or outline behind text over any scene; labels in the world on a
leader line, fading with distance; test over real footage. GTA V's mission banners and Forza Horizon's notices for
the shape of the big title and the feed.

**`cosim/hudkit.py`**, the overlay's design system, and what uses it (with per-pixel alpha; `--overlay key` keeps the
old drawing):
* **Notices** top right (`Feed`): slide in (0.22 s), stay 5-7 s with a bar running out, fade (0.35 s); newest on top,
  4 at most; a repeat refreshes the one on screen, news about the same thing replaces it. Fed by `Events`: accidents
  as they happen, change phase and are cleared; the junction agents' decisions for buses and ambulances within 700 m
  ("Green ended early / An ambulance is coming · Junction 1574 · 214 m"); the forecasters' alerts. The agents' list
  top right (with `--agent-hud`) is replaced by them.
* **Banner** (`Banner`): a new accident within 450 m: a tag line in red, the status in heavy condensed capitals, what
  happened under it, a rule opening from the middle; 3.4 s, one every 20 s at most; no place label under it.
* **Labels** (`tag`, `place_tag`): places, accidents and the agents' pills: the kind's icon on a badge, the name in
  condensed capitals, a line under it, glass with the kind's bar; the places' labels fade out from 230 to 420 m
  instead of popping.
* **Status chips** under the clock (`chips`), **key caps** for the prompt (`prompt`) and the key help (`keys`), the
  forecast panel's title.
* Colours: red emergencies and accidents, blue police, amber recovery, yellow buses, orange forecasts, cyan the agents,
  sky blue parking, green cleared. Fonts: Barlow Condensed Bold and ExtraBold and Barlow Bold and Regular added
  (`assets/fonts`, SIL OFL as the others; converted from the @fontsource packages' WOFF).
* The overlay redraws 20 times a second while a notice or the banner moves (else as before).

**Measured** (70 s demo, full screen, forecasts on; flying high / street fps): 17:30 35.7-37.1 / 41.0-42.2 (before,
with bus priority: 35.4-35.8 / 40.3-40.4); 22:00 29.4-30.0 / 45.6-46.1 (before 29.6 / 45.6); 12:00 43.6 / 53.9
(before 43.9 / 53.9). Overlay in the main loop 148-157 ms a second (before 153-165). Notices a 70 s demo: 9-11
(ambulances 7-8, buses 1-4), banners 1-2; a picture with a notice on screen is now taken in each demo
(`town05_demo_notice.png`). Tests: 366 passed (`tests/test_hudkit.py`).

## 12 Nov: the mini-map and the clock redesigned (the user's request: "another design, a whole other level")

**The radar** (`cosim/radar.py`, bottom right while flying, with per-pixel alpha; following a car keeps the trip's map,
`--overlay key` the old mini-map). Taken from the open-world games: GTA V's round radar turning with you (heading
up), a compass bezel whose N moves round the edge (The Witcher 3, Cyberpunk 2077), the zoom following the camera's
height (110 m round you at street level, up to 330 m high up; the Cyberpunk critique was a map zoomed too close for
the speed), things off the map as blips on its edge with their distance (GTA), the roads coloured by the traffic on
them (green moving, amber under 45 % of the limit, red standing; each car is put on its nearest road), the camera's
view as a cone. Above it the town's numbers as chips (moving, at red, queued, parked), the heading over it
("NW 327°"), a scale bar under it. The town's map (roads, the places' buildings in their kind's colour, a 50 m grid)
is drawn once at two scales; each picture crops the disc round the camera and turns and scales it in one step.

**The sky clock** (`cosim/hud.py`, `_sky_card`, with per-pixel alpha; same place and size): a window on the town's
sky: its colours at this time (darker up, lighter at the horizon, warm round a low sun), the sun across it by the hour
and up by the real sun elevation, the moon and stars at night, a skyline whose windows light up from dusk (fewer
after 22:30, few after 01:00), the time of day on a tag; the time in Barlow Condensed ExtraBold, the weekday, the
speed; the traffic now (dot, words, % of peak) and the day's traffic as one curve, the part gone filled in the
present traffic colour, a glowing point at now. Redrawn once a game minute (3-6 ms).

**Measured** (70 s demo, full screen, forecasts on, `--agent-hud`; flying high / street fps):
* first version: 17:30 31.0 / 38.2 (before 35.7-37.1 / 41.0-42.2), overlay in the main loop +50 ms a second. The
  radar drew in 10.6 ms (Linux VM): the nearest-road search per car, a crop 1.45 times the disc, the bezel drawn and
  scaled each time. Now: the road per 3 m square remembered, the disc's crop only, the bezel's 15 turns (its ticks
  repeat every 15 degrees) and the shadow drawn once: 4.3 ms. (The Claude window was over CARLA in that run: part of
  the drop may be that.)
* after: 17:30 35.9 / 39.8 (before 35.7-37.1 / 41.0-42.2); 22:00 29.5 / 45.0 (before 29.4-30.0 / 45.6-46.1); 12:00
  42.1 / 51.5 (before 43.6 / 53.9). Overlay in the main loop 112-210 ms a second at 17:30 (before 105-201).
  Before = the 12 Nov overlay redesign's runs, same day, same settings.
* The traffic cameras' detector (the demo's native camera check, the overlay is not in CARLA's pictures): boxes a
  frame 14.5 / 3.8 / 10.8 at 17:30 / 22:00 / 12:00 (before 14.8 / 3.9 / 11.0).
* Tests: 375 passed (`tests/test_radar.py`: the map and the cars turn alike, the zoom follows the height, the sky
  clock at every hour, the windows lit at night and not by day).
* Three runs before these failed in CARLA's loading (the world never came, 0 % GPU, `nvidia-smi` losing the driver
  for a few seconds) before any of this code ran; after a restart of the PC and with CARLA left in front they ran.

**Limits.** Accidents on the radar are the route agent's alerts (the ones it plans round), not every accident marker
on screen. The radar's per-frame cost is still about 4 ms, 5 times a second (about 20 ms a second of the main loop).

### 12 Nov, later: the radar made smooth (the user: "laggy, not smooth at all, like the one you replaced")

Why: the radar was drawn 5 times a second, as the old mini-map. The old one is north up: between its pictures only
the camera's small arrow jumped. The radar turns the whole map with the camera: every turn jumped 5 times a second.

What changed:
* `cosim/radar.py`: the traffic (roads coloured by their cars, the cars) is drawn 5 times a second on a layer
  north up, at the screen's scale, 1.6 times wider than the radar's view; each picture only crops it round the camera
  (to the sub-pixel), turns it heading-up and scales it by the zoom eased since (one `rotozoom` of ~350 px). The zoom
  eases by time, not by picture. The bezel with its shadow is drawn once for each of its 15 turns; the cone, the
  camera's marker and the shading are one picture. The radar keeps one size (500 px of room for the chips), so it
  can be redrawn alone.
* `cosim/freefly.py` (`radar_tick`): between the overlay's pictures, while the camera moves or the zoom eases, the
  radar's ring alone is drawn again on what lay under it and sent alone (its 398 x 408 px of the window). The radar
  is drawn with every overlay picture too (no more 0.2 s hold for it). The log says how many radar pictures a second
  ("radar: 24.2 pictures a second") and the ticks' cost ("radar 101 (12/s)").
* Cost of one radar picture: 2.9 ms drawing + 0.8 ms sending on the PC alone (`Claude outputs/vq/radar/prof_pc.py`);
  ~8 ms in the running town (the CPU shared with SUMO and the poses).
* A first try of the tick reached 11 pictures a second only: the radar's size changed with its chips' numbers (the
  tick then gave up) and the tick waited 1/30 s after itself in a loop run at 30 a second (every other loop skipped).

Measured (70 s demo, flying high / street fps; radar pictures a second while flying):

| | old mini-map | radar 5 / s | radar smooth |
|---|---|---|---|
| 17:30 | 35.7-37.1 / 41.0-42.2 | 35.9 / 39.8 | 33.7 / 38.8, radar 22-30 / s |
| 22:00 | 29.4-30.0 / 45.6-46.1 | 29.5 / 45.0 | 28.7 / 44.3, radar 17-27 / s |
| 12:00 | 43.6 / 53.9 | 42.1 / 51.5 | 41.0 / 49.9, radar 18-26 / s |

1 % lows: 28.4 / 31.6 (17:30), 21.9 / 25.5 (22:00), 34.4 / 36.1 (12:00). The smooth radar costs 1-2 fps against the
5-a-second one, 2-4 against the old mini-map. Tests: 376 passed (`tests/test_radar.py`: the shifted layer puts the
roads where a freshly drawn one does, within a pixel).

## 12 Nov, evening: the key bar's margin, the maps' cars on time, cars leaving a parking place without stopping

The user's three points.

**The key bar** (F5 pause ... Esc quit) touched the bottom of the screen: it was put 42 px from the bottom, and its
glass bar carries its shadow's margin (ui.MARGIN): its lower edge was on the screen's edge. Now 16 px off it, as the
other parts (`compose_overlay`).

**The maps' cars ahead of the town's.** SUMO steps every 0.5 s; the pose process shows each car between two steps
(segments); the maps drew SUMO's newest step, which the SUMO thread has well before the picture shows it. Now: each
segment carries its SUMO time (`t_sumo`), the pose process writes the SUMO time each picture shows (`shown_follow[4]`,
with when, `[5]`), the viewer keeps the last 8 steps' cars (`FreeFly.hist`) and the maps take every car between the
two steps around that time (`cars_at`). The radar draws the cars with every picture (they were in its 5-a-second
layer). Measured in the demo (`map_gap`: every second, 25 moving cars within 150 m, metres from the CARLA car's middle
to the map's car made into a middle the way the poses are):

| | now (cars_at) median / p95 | before (SUMO's newest) median / p95 |
|---|---|---|
| 17:30 | 0.45 / 3.26 m (n 1128) | 23.79 / 51.83 m |
| 22:00 | 0.69 / 3.46 m (n 417) | 26.05 / 54.67 m |
| 12:00 | 0.54 / 3.25 m (n 901) | 22.46 / 53.80 m |

(The newest step was 2-5 s ahead of the picture, not one or two steps: more than I had thought.)

**A car leaving its space stopped in the road.** `cosim/parking_motion.py`: every leg of a drive slows to standing at
its end, so the drive out stopped on the road; SUMO, holding the car there, then started it from standing (a step and
more standing). A kerb's drive out started up to KERB_EARLY_S (8 s) before SUMO put the car back and stood on the
lane for it (1.5 s here). Now:
* a drive out's last leg ends at JOIN_V = 2.2 m/s (`Leg(v_end=...)`), and SUMO takes the car over at that speed
  (`setPreviousSpeed`, held at it for one step, `_ramp`, then SUMO's own);
* a drive just over is shown rolling on along its end's heading until SUMO has the car (up to ROLL_S = 1.05 s), and
  SUMO's car is put where it will be by then (`_roll`, `_roll_on`); longer, it stands as before;
* a kerb's drive out waits in its place so that it reaches its lane when SUMO puts the car back.
Measured in SUMO alone, one car into and out of each car park and kerb area (`Claude outputs/vq/park/trace_out.py`,
as tests/test_parking_motion.py): steps standing around the hand-over, before 1 (car parks) and 4 (kerbs), now 0 in
all 11 areas; the moves around it 1.1-1.5 m a step (2.2-3 m/s). `tools/check_parking_motion.py --seed 101 --start
7.5 --hours 1`: as before (no step over 2 m on a drive, no static or parked hit, 1 overlap with a road car at a kerb
as before, seams 0.0 m). (Correction, same evening: the "3.2 m lane change in one step" after a kerb seen in that
trace was the trace's SUMO, run without the viewer's sublane model. Run as the viewer runs it (--lateral-resolution
0.8, vmodels.LANE_CHANGE_ATTRS: `Claude outputs/vq/park/trace_sub.py`), the car steers over 0.25-0.5 m a step,
2.5 s in all, and none of the 11 areas has a standing step at the hand-over.)
Tests: `test_a_car_leaving_rolls_onto_the_road_without_stopping` (a car park, a kerb), `test_the_maps_show...`.

**Frame rate** (70 s demo, flying high / street): 17:30 34.4 / 39.0 (before 33.7 / 38.8; a first run 32.4 / 33.0,
not seen again), 22:00 28.9 / 45.0 (28.7 / 44.3), 12:00 42.0 / 50.8 (41.0 / 49.9).

## 12 Nov, night: the side of the street a car parks on (the user's idea: a U-turn instead of a trip round the block)

**Looked at first.** Town05's network has no U-turn anywhere (0 turnaround connections); coming back along the
other side of a road costs 214 m round the blocks (median of the 58 roads with a twin; p90 596 m, max 1382 m). A
U-turn in the middle of a road is not something SUMO does (the viewer would have to drive it across the lanes coming
the other way, on roads of up to 4 lanes each way), and U-turns at junctions change the network: every signal
program's link numbering, the junction agents, the forecasters, and CARLA has no path for them. So not that. What
makes the trips round the blocks: a place with kerb parking has an area on each side of its street, and
`PurposeDemand.area_for` drew the side at random; all the town's homes are on two such streets (kerb_19, kerb_20),
whose sides are 40 m and 659 m apart by road (median over all starts).

**Measured (dev seeds 11-13, SUMO alone, the evaluation's run without accidents, `tools/check_park_side.py`):** 429
trips, 359 km planned; the side drawn against the shortest side from the same start: out 0.4 km more (2 trips over
50 m), back home 15.5 km more (39 trips over 50 m, 90 m a trip), 4.4 % of these trips' km.

**The change (`cosim/purposes.py`, PARK_SIDE "route"):** the destination's side and the home's side (the way back;
a new car also starts there) chosen together, the pair that makes the trip there and back shortest among the
place's areas with room; a car sent to another car park (its own full) keeps it; the random draw is still made, so a
seed's other draws do not move. Seed 11: home trips 1139 -> 862 m planned (-24 %), the side's extra 0.7 m a trip.

**The bar, written before the tests.** `agents_observed+bus@bus_cut=1` (the viewer's x1) with PARK_SIDE "route"
against "random", x1, each test opened once:
* quiet (`--no-accidents`), seeds 271-300: (a) cars' km per completed trip lower on the mean and on at least 20 of
  30 seeds; (b) cars' time loss per trip not higher on the mean by more than 3 %; (c) trips completed an hour not
  lower by more than 3 %; (d) buses' time loss per km not higher by more than 3 %;
* with accidents, seeds 301-390 (on the PC when it is free): (e) cars' time loss per trip not higher by more than
  5 %; (f) trips completed an hour not lower by more than 3 %; (g) ambulances' time loss not higher by more than 8 s;
  (h) buses' time loss per km not higher by more than 5 %.
All eight, or it goes back to "random". Then the forecasters' accuracy on the new traffic (they learnt the old).
Test seeds used after this: 21-390.

**Quiet test (seeds 271-300, x1, `out/park_side/test_quiet`, 60 runs): passed.** "route" paired against "random",
mean +- standard error (seeds where "route" was better): (a) km per completed trip -0.017 +- 0.008 km (-0.9 %, 20 of
30: passed, just); (b) cars' time loss per trip -7.8 +- 2.9 s (-6.4 %, 19); (c) trips completed an hour +11 +- 7
(+1.4 %, 16 of 30 not lower); (d) buses' time loss per km -3.8 +- 3.7 s (-2.2 %, 15). Also: waiting per trip -8.8
+- 2.9 s (-9.4 %, 21), all the cars' km -22 +- 6 km a run (-2.7 %, 21), ambulances -3.2 +- 2.6 s. The km per trip
moves less than the planned trips said (the town's trips include 20 % through traffic and trips to car parks with one
side); the time moves more: the trips round the blocks were made at the evening rush on roads 19 and 20.

**The forecasters on the new traffic: the bar, written before.** The shipped v7 learnt traffic with the random side.
New test seeds 391-400, the T7 recipe (agents_observed, a day at clock 30 from 06:00, accidents, junction blocker 0,
trips with a purpose) at x1, x1.15 and x1.3, each made twice on the same seeds: "random" (T8r_*) and "route" (T8_*);
all six scored in one opening (`tools/goflow_prediction_test_score.py --seeds 391-400`). The route side stays if:
1. DSO4's queue fill MAE on the "route" data below "it stays as now" at +60, +120 and +300 s (9 rows of 9);
2. on each of the 9 rows, the MAE on "route" at most 5 % above the MAE on "random" (the same seeds);
3. the long-queue alerts at 0.7 on "route" x1.15 and x1.3: at least 40 % of alerts right and at least half of the
   long queues caught (v7's own ship bar).
If 1 or 3 fails, the forecasters are retrained on the new traffic before the route side ships; if only 2, the same.
Test seeds used after this: 21-400.

**The forecasters on the new traffic (seeds 391-400, opened once, `out/park_side/fc/score`): bar 2 failed.** DSO4's
queue fill MAE, "route" / "it stays as now" / "random" (same seeds):

| Traffic | +60 s | +120 s | +300 s | Long queues at 0.7, route: right, caught | random |
|---|---|---|---|---|---|
| x1 | 0.0371 / 0.0426 / 0.0376 | 0.0408 / 0.0485 / 0.0412 | 0.0454 / 0.0547 / 0.0462 | 21 of 52, 20 of 36 | 29 of 58, 28 of 42 |
| x1.15 | 0.0385 / 0.0443 / 0.0418 | 0.0425 / 0.0503 / 0.0462 | 0.0484 / 0.0576 / 0.0527 | 29 of 66, 27 of 42 | 37 of 67, 37 of 49 |
| x1.3 | 0.0484 / 0.0572 / 0.0458 | 0.0534 / 0.0643 / 0.0501 | 0.0605 / 0.0741 / 0.0572 | 61 of 141, 56 of 80 | 61 of 115, 56 of 73 |

(1) below "stays as now" 9 of 9: passed. (2) at most 5 % above "random": x1 and x1.15 below it (0.92-0.99), x1.3
5.8-6.5 % above: failed. (3) at 0.7, x1.15 44 % right and 64 % caught, x1.3 43 % and 70 %: passed. Against "stays as
now" (skill = 1 - MAE / stays), the forecasters do a little less well on the new traffic on 8 of 9 rows (by 0.00 to
0.02) and their alerts are right less often (40-44 % against 50-55 %): they learnt the old traffic. As written:
they are retrained on the new traffic before "route" ships; PARK_SIDE is back to "random" until then.

**With accidents (seeds 301-390, x1, `out/park_side/test_acc`, 180 runs: 301-345 in the Linux VM, 346-390 on the
PC, each seed's pair on one machine): failed.** "route" paired against "random", mean +- standard error: (e) cars'
time loss per trip +0.7 +- 3.9 s (+0.5 %: passed); (f) trips completed an hour +9 +- 5 (+1.2 %: passed);
(g) ambulances' time loss +8.7 +- 4.4 s (+20 %, worse on 54 of 90: failed, the bar was 8 s); (h) buses' time loss
per km +15.9 +- 6.0 s (+8.2 %, worse on 54 of 90: failed, the bar was 5 %). Also all the cars' km -2.1 +- 1.1 %,
waiting +1.0 +- 4.2 s, accidents a run +0.14 +- 0.16.

**So PARK_SIDE stays "random".** Without accidents "route" saved the cars 6 % of their time and cost nobody; with
accidents the cars' gain is gone and the buses and ambulances lose. Not looked into yet (it needs dev seeds, then new
test seeds): which roads the buses lose their time on with "route" (more cars pulling in and out of the kerb lanes of
roads 19 and 20 on the side the buses use, near a crash?). The forecasters' check also asked for a retraining first.
The code stays as an option (`PurposeDemand.park_side`, `tools/eval_agents_forecast.py` variants "...|park=random",
`tools/goflow_prediction_dataset.py --park-side`, `tools/check_park_side.py`). Test seeds used after this: 21-400.

## 13 Nov, night: why the buses lost with the parking side by route (the user: find it, fix it, test again, then retrain)

**Where they lost it** (dev seeds 11-20, accidents, x1, `tools/check_bus_loss.py`: time lost road by road, a bus at
its stop left out; cars parking per road): with "route" the buses lost 215 s a run more (worse on 6 of 10), the
emergency vehicles 276 s more (8 of 10). The roads: -20 (buses +376 s, emergency +224 s) and -19 (+131 s, +112 s),
the sides of the two home streets that bus lines 1 and 3 drive along. Cars parking there a run: -20 24.7 -> 30.6
(its other side 20: 13 -> 0), -19 43.3 -> 47.2 (19: 24.9 -> 17.6); accidents a run 1.5 -> 2.5. Most cars reach the
home streets coming the way the buses go, so "the nearer side" is the bus's side: more cars slowing to turn in and
pulling out in front of the buses (and of the ambulances using the same roads), and more crashes there.

**The fix tried:** with "route", a side a bus line drives along counts BUS_SIDE_M metres further (`purposes.py`; the
bus lines' roads from town05.pt.json: of the kerb streets -19, -20, -51 and 4). Dev sweep: random, route, and route
with 150, 400 and 1000 m, accidents on dev seeds 11-20 and 401-430 (401-430 added as dev seeds: never a test seed),
quiet on 11-20 (`out/park_side/dev_sweep`, `out/park_side/dev_quiet`).

**Dev sweep (13 Nov, 03:17; 40 seeds with accidents, 10 quiet; paired against "random", mean +- standard error):**

| With accidents | buses' time loss /km | ambulances | cars' time loss /trip | waiting /trip | all cars' km |
|---|---|---|---|---|---|
| route | +1.0 +- 3.9 % | +3.1 +- 6.0 s | -4.8 % | -7.4 % | -1.3 % |
| route, bus side +150 m | +4.0 +- 3.6 % | +7.8 +- 6.0 s | -2.6 % | -4.3 % | -1.5 % |
| route, bus side +400 m | +4.9 +- 3.8 % | +9.5 +- 6.5 s | -2.0 % | -2.4 % | -1.3 % |
| route, bus side +1000 m | -2.5 +- 3.9 % | +10.7 +- 8.3 s | -0.1 % | +0.4 % | 0.0 % |

Quiet (10 seeds): buses within +-1 % for all; cars' time loss route 0.0 %, +150 m -6.7 %, +400 m -5.6 %, +1000 m
+1.9 %; all the cars' km -3.5 to -4.7 %.

**Read:** with accidents the buses' difference is within its noise for every version (+-4 %), so the penalty on the
bus sides shows no gain for them; the ambulances lose with every version, more with the penalty; the cars' gain
shrinks as the penalty grows and is gone at 1000 m. The parking on the bus sides (seen road by road on 10 seeds) is
real, but it is not shown to be what costs the buses and ambulances across 40 seeds, and keeping cars off those sides
does not help them. No version is expected to pass the bar on new seeds (the ambulances' +8 s alone), so no new test
is run (it would only be a new draw of seeds for the same idea), and no retraining (step 2 was for a version that
passes). **Stopped: PARK_SIDE stays "random", BUS_SIDE_M 0;** both stay as options. The idea's gain (cars -5 to -7 %
of their time at the town's own traffic without accidents) is real; its cost to the emergency vehicles with
accidents is not explained. What could be looked at next, if wanted: the ambulances' runs road by road (which runs
lose, near which crashes), not the buses.
