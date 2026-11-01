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
