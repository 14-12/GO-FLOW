# Next steps (written 25 Oct, ~02:00 local, before the session limit)

State: depots shipped and checked on the PC (bundles b31-b37, last b37: police slots either side of bus stop L2_S1 on -52_2,
buses allowed there, parked depot vehicles' lights off). Tests on PC: 195 passed (before b37's test change).
Cloud copy: /home/claude/proj. Transfer: bundle.sh NAME files -> commit to out\_xfer\in_NAME\NAME.tgz -> unpack.sh NAME md5.
Next bundle: b38. Rules: start .bat by double-clicking in Explorer (logs in tools\logs); never touch cosim/fly_proc.py or
DefaultInput.ini; new folder per transfer + md5 check.

## 1. Two vehicles overlapping + the glitching notification (user screenshot, 25 Oct ~01:50)
- Incident toast ("TOW TRUCK ON THE WAY · 35x m", warning triangle) drawn over a far, unrelated view, cut by black
  blocks carrying plate text ("231 TU 437", "8086"): the user had flown elsewhere and it "follows" him.
- Check cosim/incident_hud.py (marker / label placement, off-screen clamping, depth/occlusion) and whatever draws plate
  text in the overlay (cosim/overlay.py / labels / inspector "H Inspect traffic camera" prompts). The black blocks may be
  another overlay's boxes (two vehicles' labels at the same spot = two vehicles at one place = the collision he means).
- Find the two vehicles: likely the tow truck and another vehicle at the same pose (truck_off_sumo / deck / carried
  wreck), or two cars overlapping in SUMO. tools/check_overlaps.py, check_incident_scene overlap_detail.
- He insists: no car may ever overlap another car or leave its lane/limits.

## 2. Traffic lights like real life (he asked for a full research first)
Complaint: at each intersection only one pole is green at a time; the others all red. Wanted:
- opposing approaches' through movements green together; right turns with them; left turns handled (protected left
  phase or permissive left, yielding), i.e. NEMA dual-ring / 2-4 phase plans instead of one approach at a time (split
  phasing);
- the pole's 4th head is the pedestrian signal: walk concurrent with the parallel through movement, don't-walk otherwise;
- traffic as smooth as possible: actuated timing, gap-out/max-out, min green, yellow/all-red clearance (ITE formulas),
  coordination/green waves on arterials, cycle length (Webster), right-turn-on-red not used in Tunisia/France.
Research: WebSearch (FHWA Signal Timing Manual 2nd ed., NEMA ring-barrier, ITE yellow/red clearance, Webster, SUMO
netconvert tls options: --tls.layout opposites/incoming, --tls.left-green.time, --tls.guess.joining, tls.default-type
actuated; CARLA traffic light groups: each pole one approach, freeze/set_state, pedestrian head behaviour).
Then: inspect current programs (scenarios/town05/town05.net.xml tlLogic, town05.actuated.add.xml, cosim/tlcomplete.py,
cosim/tlmap.py (SUMO links -> CARLA poles), cosim/actuated.py, cosim/agents_actuated.py, cosim/ped_lights.py,
cosim/highway.cap_greens) and why they give one approach per phase (probably netconvert --tls.layout incoming or the
CARLA-side mapping, or the junction agents choosing single-approach phases).
Fix: build opposed-phase programs (through+right together for opposite approaches, then protected lefts if the
junction has left lanes, permissive otherwise), pedestrian walk with parallel throughs, keep the agents working on the
new phases (retrain/re-tune if their action space changes: tools/eval_town05_*, the Town05 agents docs), measure with
tools/check_highway.py-style runs and the agents' evaluation (waiting time, standing share, throughput) before/after,
check in the viewer at a few junctions (screenshots), README section.

## 3. Afterwards (agreed order): drivers ("real persons") in cars near the camera.

## Progress (25 Oct, ~03:10 local)
- DONE (cloud, not yet shipped): incident marker hidden beyond 250 m off screen / 900 m on screen, notice cached and
  moved only on >=3 px (cosim/incident_hud.py); loading tow trucks guard the junction ways passing their body
  (IncidentManager._guard_place, all internal lanes) -> s31 overlaps 26 -> 0.
- DONE (cloud): tools/build_town05_signals.py rebuilt Town05's programs (netconvert --tls.rebuild --tls.layout
  opposites, ITE yellow/all-red, walk clearance = crossing/1.2 m/s 5-15 s, linkSignalID carried by lanes: 200/200).
  scenarios/town05/town05.net.xml + actuated/webster add rewritten; originals town05.carla_cycle.*.
  sumo_side.WalkSkip._clearance ends a walk clearance when its crossings are empty; agents_actuated: "walk clearance"
  phases are kind "clear"; _jump keeps the remaining all red. Tests fixed (test_town05_agents).
- Measured (tools/check_highway.py 20 min 17:30, 8 seeds): standing share old actuated 0.585 / agents 0.555 ->
  new actuated 0.453 / agents 0.490; halting 72 -> 50; ring 49 -> 58 km/h; no lock-ups. Agent variants without
  "balance" locked seed 31. Testing agents-trickle / agents-skip-trickle (/tmp/sig_new_*), summary: python3
  /home/claude/bin/sigsum.py old new; old-signal copy of the project: /tmp/claude-0/projold.
- TODO: pick the viewer's default controller for the new layout, ship bundle b38 (incident_hud, incidents, sumo_side,
  agents_actuated, build_town05_signals, scenarios/town05/* (net, actuated, webster, carla_cycle copies,
  scenario.json), tests), PC tests + viewer screenshots of junctions, README section.

## Shipped (25 Oct, ~05:40 local): b38-b40
- b38: incident marker + truck guard + new signal programs (scenarios/town05/*, originals town05.carla_cycle.*),
  WalkSkip clearance, agents (trickle off by default, "walk clearance" = clear, _jump keeps all red), check_highway --unjam.
- b39: freefly --demo-signals + tools/freefly_signals_demo.bat. b40: README.
- PC: tests 194 passed, 1 skipped; agents demo ran; signals demo: 12/15 junctions two poles green together.
- NEXT: drivers ("real persons") in cars near the camera. Known weak spot: triangle 1292/1126/1574 can knot at the
  evening rush with the camera agents (1 of 8 seeds without unjam).

## Handoff (25 Oct, ~12:45 local) - weekly limit reached, shipped as b52 (out\_xfer\in_b52)
DONE and on the PC:
- Drivers at the wheel (b42-b51): cosim/drivers.py (POSE, seat_of, TALL, HIP_Z, SIDE, seat_rel, POOL 16),
  pose_proc.Driver.seat_drivers (DRIVER_M 45 / DRIVER_KEEP_M 52, "seats"/"drivers"/"door" messages, seg "empty"),
  SceneOps.seat (bones), freefly: --drivers (default on), prespawn_drivers, ask_plates sends seats, segment "empty"
  (wrecks, depot-parked), --demo-drivers + tools/freefly_drivers_demo.bat; tools/probe_drivers.py(+_some.bat);
  tests/test_drivers.py + test in tests/test_pose_proc.py. PC: tools\test.bat 198 passed, 1 skipped; demo screenshots
  out\freefly\town05_drivers_*.png show drivers in mini, tesla, impala, etron, police car, coupe, HGV, bus (probe).
  README section "Free-fly: a driver at the wheel (25 Oct)" written.
- Final agent eval on the new signals: out/agents_final_v2/agents.md (actuated / agents / agents_observed,
  x0.5/0.8/1.0/1.2, seeds 21-30, 1 h each). No lock-ups at any demand for any controller (BoxNudge + EvCaution in
  cosim/sumo_side.py). Agents vs actuated: ambulance time loss -10.5/-11.1 s (x0.5, from 34), -15.5/-14.0 (x0.8, from
  42), -19.8/-20.3 (x1.0, from 51), -25.1/-24.3 (x1.2, from 62), better on 10/10 seeds everywhere; car waiting within
  +-1 s of actuated (x0.5 -0.7/-0.9 s, 8/10 seeds); pedestrians +0.4..+1.7 s worse (x0.5 all seeds).
- Cause of the pedestrian cost: a skipped car green also skipped the walk that goes with it. FIX (in b52, NOT yet
  tested on the PC): ActuatedAgents._demand counts people waiting at the crossings of a car green as demand.
TODO:
1. PC: tools\test.bat (expect 198 passed).
2. Check the ped fix: cloud/PC run `python tools/eval_town05_baseline.py --controllers agents,agents_observed
   --scales 0.5,0.8 --seeds 21,22,23,24,25 --jobs 2 --out out/agents_pedfix` after copying the actuated runs of
   seeds 21-25 from out/agents_final_v2/runs, then `python tools/compare_town05_agents.py --out out/agents_pedfix
   --name agents`. Keep the fix if pedestrian time loss is no longer worse and ambulances/cars hold; then (ideally)
   rerun the full final eval into out/agents_final_v3.
3. README: section "Town05 agents on the new signals (25 Oct)" with the table above (and the ped fix result).
4. Doc "Junction Agents on Town05" (claude.ai artifact BN8p7mFLq5zS6XPvP9XNmm, project 53eed1ff-...): add a section
   "Re-evaluation on the new signals" before "Limits and next steps" and update the Summary (old figures were on
   CARLA's split-phase cycle: 79.8 s waiting at x0.5; now ~29 s; the over-capacity lock-ups are gone).

## Session 26 Oct (b53, b54) - the four handoff tasks, and a check of the signals themselves
1. PC tests: tools\test.bat 198 passed, 1 skipped on b52 (tools\logs\test_26oct_b52_before.log); after b53 213 passed,
   1 skipped (tools\logs\test_26oct_b53.log; new: tests\test_signal_programs.py, one in tests\test_ped_lamps.py).
2. Signals audited (cosim\signal_audit.py, tools\audit_town05_signals.py -> out\signal_audit\town05_signals.md: per
   junction the lanes and their movements, the CARLA pole, every conflict / give-way from SUMO's junction logic, the
   phases, the crossings, left-turn cross products). Two faults fixed in tools\build_town05_signals.py: the 15 s cap on
   the walk clearance (2014, 2328: 21.5 s from walk end to conflicting green, MUTCD 3.5 ft/s needs 22.9 s; now 16.5 s)
   and Webster cutting walk clearances to 5-8 s / walks to 5-6 s (webster_walks). All 45 programs pass.
   cosim\signal_monitor.py checks the same rules step by step in a running simulation; tools\check_signal_scenarios.py
   runs 12 stress scenarios at a junction (out\signal_checks\1126\summary.md: no rule broken in 36 runs).
   Ped lamps (--ped-lights) flash during the walk clearance (ped_lights.ped_aspect). Poles close up:
   tools\freefly_poles_demo.bat (out\town05_freefly_poles.json): 4 vehicle heads per pole (3 over the lanes, 1 on the
   post), no pedestrian head in CARLA.
3. Ped fix checked (out\agents_pedfix, pedfix_vs_v2.md): kept, as feature "pedcall" ("agents-pedcall" = without).
4. Final eval on the corrected signals: out\agents_final_v3 (tools\eval_town05_final_v3.bat; same protocol as v2).
5. README section "Town05 agents on the new signals (25 Oct)". The doc "Junction Agents on Town05" is not shared
   with the cloud session (read denied): its revised text is notes\Junction_Agents_on_Town05_revised_26oct.md, to paste
   over the doc (new section before "Limits and next steps", Summary/Setup/Limits updated, old-cycle figures labelled).
Open: SUMO reports a few vehicle contacts inside junctions with permissive lefts (0-2 an hour town-wide, none on the
old cycle; viewer overlaps per vehicle about as before; jmTimegapMinor 2.5 did not help) - out\signal_checks.
Possible next: leading pedestrian interval (3-7 s walk before the parallel green) at busy crossings.

## Session 27 Oct (b57-b59) - signals where no green gives way
Asked: no car may wait inside a junction for a "free to pass" (permissive) movement; each head its own lane and
arrows; pedestrians considered; study every case; smooth traffic.
1. cosim\signal_plan.py builds every program from SUMO's own conflicts, lane by lane: no "g", no two crossing links
   green together (signal_audit.check_protected; all 45 programs pass). Default set "split" (each approach on its own,
   all its lanes); the walk is a phase of its own (scramble), its clearance marked "u" (flashing DON'T WALK).
2. Two more causes of cars stopping inside, fixed: SUMO walkers stepping onto red crossings (sumo_side.PedRedHold:
   held on the corner until WALK) and the parking lanes at 53/979/1722 having a signal movement (removed; they are not
   a third driving lane, as the 25-26 Oct README said).
3. cosim\signal_heads.py: the poles' CARLA lamps off, each head drawn lit with lane arrows; checked in CARLA at 1126
   (out\freefly\poles_27oct_b59, out\town05_freefly_poles_27oct_b59.json). Fix on the way: camera_reports drops layout
   lanes the network no longer has (the viewer stopped at load on ':979_27_0').
4. Measured (tools\check_junction_stops.py; README "Signals where no green gives way"): no car stands inside for a
   conflict any more (4-10 against 700-871), but waiting per trip is 2.5-4x the permissive signals'; lead-lag (left
   lanes held red while the opposite right lanes go) and left-turn lanes jam at x1.0 / x0.8. Switch:
   tools\signals_split.bat (default) / signals_leadlag.bat / signals_permissive.bat.
5. tools\video_junction.py updated: out\videos\junction_1126_protected_27oct.mp4.
6. PC tests: 219 passed (tools\logs\test_27oct_b57.log).
Open: the agents were tuned on the permissive signals (out\agents_final_v3); a re-tune on the protected cycle (longer
greens at the rush, the walk only every other cycle when few wait) may win back some of the waiting. The doc
"Junction Agents on Town05" evaluations are on the permissive signals.

## Session 27 Oct afternoon (b62, b63) - coordination, lamps, ring road crosswalks, the missing camera
Asked: the city's signals coordinated and in step like the ring road's; greens longer than 4-5 s; the lamps on the
heads looked pasted on (outside the head); the square in front of the post head came back; no pedestrians across the
ring road; a camera missing on the ring road.
1. cosim\coordination.py: one 115 s cycle for every signal, greens >= 12 s (7 s turns only), offsets for green waves
   (42 % of signal-to-signal steps arrive on green, 31 % with offsets 0), town05.coordinated.add.xml in every signal
   set; sumo_side.CycleKeeper keeps them in step (first green ends at its planned second every cycle; other greens gap
   out after their minimum; skipped walks' time to the first green). Viewer default controller: coordinated.
   Measured (k_F runs): x1.0 136 s / 1320 trips against the agents' 215 s / 974; x0.8 113 s vs 100 s.
   Tried and dropped (all coordinated, x0.8): left-turn lanes (NEMA) 310 s jam, lead-lag 201 s, a signal per
   direction on shared lanes 167-209 s; SUMO's own coordinated-actuated mode (param coordinated) did not hold the
   first green to its earliestEnd here, so the keeper does it.
2. cosim\signal_heads.py: arrows 0.15 m on the lens glass (ray-cast face, 210/210 found), each direction its own
   colour and lens; the post head repeats the right lane (no point drawn: no square). CARLA shots
   out\freefly\town05_poles_1126_*.png (b62 run).
3. tools\build_town05_signals.py removes the 4 crosswalks over the ring road (2014, 2328): 51 crossings.
4. cosim\perception.plan_cameras: the ring road's two carriageways to the same junction were one road; 55 cameras,
   out\town05_cameras re-planned (2014/2328 incoming view 82 % / 88 %).
5. PC tests: 222 passed (tools\logs\test_27oct_b62.log). tools\green_wave_diagram.py: corridor time-space diagram.
Open: a time-of-day plan (a shorter cycle off-peak: the agents' short cycles wait less at x0.5-0.8); the offsets
search is greedy (a better optimiser could raise the 42 %).

## Session 29 Oct (b65-b69) - Shift faster, real signal lamps
Asked: Shift to fly faster (4x, then 2x: "too fast"); the drawn direction lamps looked fake: make them integrated and
as in real life, with research and examples.
1. cosim\fly_proc.py: hold Shift for 2x (b65, b66).
2. Research (README "Real signal lamps"): full circle when all of a lane's / approach's movements go (MUTCD 4D, Vienna
   Convention art. 23); arrows only for a movement released or held on its own.
3. cosim\signal_plan.py: SIGNAL_UNIT approach (default): every approach goes as one, each of the 54 CARLA poles one
   colour in every phase; the viewer uses CARLA's own lamps (--signal-heads off by default). New signal set
   "approach" (tools\signals_approach.bat, town05.approach.*). Coordinated x1.0: 118 s / 1355 trips (was 136 / 1320).
4. tools\freefly_poles_demo.bat: with CARLA's lamps, each pole from 9 m in front of its heads, green / amber / red.
5. PC tests: 224 passed (tools\logs\test_29oct_b67.log).
Open: arrow lenses need traffic-light blueprints made in the Unreal Editor (CARLA source build); with them the
T-junctions could run a lane at a time again with arrows.
