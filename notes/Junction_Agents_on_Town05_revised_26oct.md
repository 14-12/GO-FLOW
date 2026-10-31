# Junction Agents on Town05

Oct 24, 2026 · @firas · updated Oct 26, 2026 (re-evaluated on the new signals)

## Summary

The junction agents run Town05's signals from the camera reports, live in the 3D viewer. Since 25-26 Oct the signals themselves work as at a real junction. Opposite approaches share their green, left turns are permissive and give way, and pedestrians walk with the parallel traffic, with a flashing don't walk and a clearance long enough for each crossing. Before that they ran CARLA's one-approach-at-a-time cycle. Re-evaluated on the new signals (10 held-out seeds, `out/agents_final_v3`), the agents keep one clear gain, for ambulances, and are level with SUMO's actuated program for cars and pedestrians:

- **Ambulances:** time loss falls by 10.5-10.7 s at normal traffic (x0.5, from 33.9 s) and by 12.9-26.2 s at busier demands (x0.8-x1.2), on every seed at every demand.
- **Cars:** waiting per trip stays within ±0.3 s of actuated (29.1 s at x0.5, 31.4 s at x0.8), and throughput within ±4 vehicles an hour.
- **Pedestrians:** within ±0.2 s of actuated. That takes the pedestrian call added on 25 Oct; without it the agents cost pedestrians 0.4-1.7 s per walk on the new signals.
- **No lock-ups:** on the new signals no run locks up at any demand up to x1.2. That is not the signals' doing alone: the traffic model also changed between the two evaluations (see the re-evaluation section).

**On the old signal cycle** (the figures in the sections below, measured 24 Oct with SUMO's deadlock breaker on): car waiting fell by 7.2 s (from 79.8 s) and ambulance time loss by 38 s (from 58 s) at x0.5, on all 10 seeds. At x0.8 ambulances gained 54 s (from 86 s). At x1.0 every controller locked the town up. Those gains were measured against a cycle that served one approach in four. On the new signals most of the car gain is gone: there are few empty greens left to skip.

Two things came out of the weak-spot work. A stuck ambulance could make an agent loop on one useless green; two rules now stop it. And the rare lock-ups at normal traffic turned out to be deadlocks inside SUMO's junction model, which hit actuated too.

## Setup

All results come from SUMO 1.27.1 on CARLA's Town05 network, one simulated hour per run, at 0.5 s steps. CARLA 0.9.16 draws the same traffic in 3D for the viewer and the cameras.

| Item | Value |
| --- | --- |
| Network | Town05: 15 signalised junctions on a grid of short links, 53 traffic cameras in all |
| Demand | x0.5, x0.8 and x1.0 of the scenario's routes (x1.0 was over the town's capacity on the old cycle); x1.2 too on the new signals |
| Pedestrians | routed trips over the crosswalks. Old cycle: a 10 s walk phase and its 5 s clearance. New signals: walk with the parallel green, flashing don't walk, then the yellow and all red |
| Emergency vehicles | an ambulance about every 120 s, the same ones for every controller within a seed |
| Baseline | SUMO's actuated program. Old cycle: four single-approach greens of 5-60 s on induction loops, 3 s yellow, 3 s all red. New signals: two greens (opposite approaches together) of 7-45 s, yellow 3.5-5 s, all red 1.5-3 s |
| Agents | `agents`: they read SUMO's lanes exactly |
| Agents from cameras | `agents_observed`: they see only what the cameras report, with the detector's measured errors |

The measures are waiting time per trip, throughput (vehicles that finish their trip per hour), the ambulances' time loss and the pedestrians' time loss. Each agent run is compared with actuated on the same seed, so the difference is paired.

## How an agent works

Each signalised junction has one agent. It never replaces the actuated program: the program keeps running every signal, and the agent steps in only with commands that keep the program's own states and order. Greens end through their own yellow, jumps go from all red to all red, and the walk is never cut short.

A first version replaced the program with max-pressure and lost to it (125 s waiting against 77 s at x0.5, gridlock at x0.8). It is kept only for the comparison.

What an agent can do:

- **Skip** an empty phase: during the all red, it jumps to the all red before the next phase with demand. On the old cycle the walk phase was skipped when nobody waited. On the new signals the walk runs with a car green, and someone waiting at its crossing counts as demand for that green (the pedestrian call, since 25 Oct).
- **Spill**: it ends a green whose waiting cars all face a full exit, since that green moves nobody.
- **Trickle**: it ends a green kept alive by a few moving cars while another phase has a queue.
- **Balance**: it ends a green whose short approach is nearly empty when another phase's lane is filling up.
- **Pre-emption and corridor**: an ambulance on a run asks its next signal for green within 150 m, and that signal passes the request to the following one within 300 m. A request counts only while serving it can actually move the vehicle.
- **Fallback**: where the cameras see under 70 % of the approaches (junctions 2014 and 2328), the agent turns its queue features off and leaves only pre-emption and the corridor, which do not depend on the cameras.

Two more features were built and tested but stay off: extending a full green past 60 s, and batching the walk.

## What the agents see

The camera agents count only what Town05's 53 traffic-light cameras report. Every 2 s each camera reports the vehicles and the queue on each lane it sees. A detection's ground point decides whether it is counted, belongs to another road, or is a parked car (off every road and still for 10 s), so a full parking lot adds no pressure.

| Measure | Result |
| --- | --- |
| Coverage of the last 75 m before the stop lines | 76 % (from geometry, an upper bound) |
| Weak junctions | 2014 and 2328, 2 cameras each |
| Vehicles reported per vehicle in view | 1.01 (error 0.1 per camera) |
| Queue error | 0.02 vehicles |
| Siren seen in one picture | 34 % of the time; 9 % false alarms |
| Siren after voting over 8 pictures (3 needed) | 11 confirmed, 10 missed, 15 false alarms |

The live check ran at junctions 1126, 1882 and 2296, held out from tuning: 10 cameras, 3910 reports, compared with SUMO's truth. Traffic was light there (7 vehicles in view at most).

Cameras alone are not reliable enough for sirens: 14 of the 15 false alarms were emergency vehicles with the siren off. So pre-emption starts from the vehicle's own request, and the camera only confirms it.

## Fixing the weak spots

One real fault was found and fixed: a stuck ambulance could keep a junction looping back to the one green that moved nobody. The x0.8 throughput cost turned out to be seed noise. Tuning used the development seeds 11-16 only, never the seeds of the final evaluation.

**The x0.8 cost was noise.** On seeds 1-3 the agents had lost 20 vehicles an hour. On the six development seeds they lose nothing: 2282 vehicles an hour against actuated's 2277, waiting 97.2 s against 98.1 s.

**The livelock at x1.0.** A new tool (`tools/diag_town05_agents.py`) records every minute where vehicles stand, and when the town locks up. On seed 14 at x1.0 (with the noflow rule below switched on), it showed junction 207 serving phase 0 again and again. Between 1500 s and 2400 s, phase 0 got 65 greens and the three other approaches 4 in all.

1. An ambulance stood in the queue on phase 0's approach, whose exit was full.
2. Its request came and went as that exit filled and emptied by one car.
3. Each time it came back, the agent jumped from the all red straight to phase 0. Nobody moved, the green ended, and the loop began again.

Two rules end it:

- **Patience**: an emergency vehicle not through its signal 45 s after its first request asks no more at that signal.
- **No starvation**: a phase already passed over three times is served before an emergency jump.

The same run with both rules: 15, 12, 14 and 11 greens for the four approaches.

**Tried and rejected.**

| Change | x0.8 | x1.0 (6 dev seeds) | Kept |
| --- | --- | --- | --- |
| End a green under which nobody moves for 4 s (loops or cameras) | waiting 95.0 s against 97.2 s | 1382 vehicles an hour against 1656, better than actuated on 2 of 6 seeds | no, optional `+noflow` |
| Exit counted full at 70 % instead of 85 % | no change | no measurable change | no |

At x1.0 the town is over capacity, and every controller locks up within the hour. One feature changed can swing a single run by 500 vehicles an hour either way, so only means over many seeds are compared. With the fix, the agents averaged 1656 vehicles an hour on the development seeds against actuated's 1510, and did better on 4 of the 6 seeds.

## Final evaluation on held-out seeds (old signal cycle)

Measured on CARLA's one-approach-at-a-time cycle, before the signals were rebuilt (see "Re-evaluation on the new signals"). On 10 seeds never used for tuning (21-30), the agents beat actuated at x0.5 and x0.8 on most seeds, and most of all for ambulances. At x1.0 every controller locks up, so no gain is claimed there.

&#91;embedded content: out/agents\_final/runs · 10 held-out seeds, 1 simulated hour each\]

Each agent run is paired with the actuated run of the same seed: the same vehicles, pedestrians and ambulances. Medians are shown because one run in ten locks up (below), and a single lock-up moves a mean by tens of seconds.

| Demand | Agents | Throughput, median change (vehicles/h) | Pedestrian time loss, median change (s) | Runs locked up, of 10 |
| --- | --- | --- | --- | --- |
| x0.5 | exact counts | +4 | -3.5 | 1 (actuated: 1) |
| x0.5 | cameras | +5 | -3.6 | 0 |
| x0.8 | exact counts | +3 | +1.0 | 1 (actuated: 0) |
| x0.8 | cameras | -17 | -0.6 | 1 |
| x1.0 | exact counts | +1 | -5.4 | 10 (actuated: 9) |
| x1.0 | cameras | -24 | +8.6 | 10 |

**Lock-ups.** Those runs are actuated at x0.5 on seed 24, the exact-count agents at x0.5 on seed 23 and at x0.8 on seed 26, and the camera agents at x0.8 on seed 29. They pull the means: at x0.8 the exact-count agents' mean waiting is 17.8 s worse than actuated, while their median is 1.7 s better.

The diagnosis tool found the cause on seed 26. Three cars stood inside junction 2086 for 10 minutes, each waiting for another, with their exit road empty. That is a deadlock inside SUMO's junction model, which real drivers would sort out in seconds; it can catch any controller. With SUMO's deadlock breaker (`--ignore-junction-blocker 60`, no vehicle removed), the same run does not lock up: waiting 94.3 s against actuated's 97.4 s.

**The same evaluation with the deadlock breaker on** (all three controllers, x0.5 and x0.8, seeds 21-30): no run locks up, and the results become clear-cut. Means of the paired changes against actuated, with the seeds where the agents did better:

| Demand | Agents | Waiting per trip (s) | Ambulance time loss (s) | Throughput (vehicles/h) | Pedestrian time loss (s) |
| --- | --- | --- | --- | --- | --- |
| x0.5 | exact counts | -6.7 (10 of 10) | -37.9 (10 of 10) | +3.1 (8 of 10) | -3.4 (10 of 10) |
| x0.5 | cameras | -7.2 (10 of 10) | -38.4 (10 of 10) | +5.2 (9 of 10) | -3.6 (10 of 10) |
| x0.8 | exact counts | -1.9 (7 of 10) | -54.2 (10 of 10) | -5.4 (5 of 10) | +1.0 (2 of 10) |
| x0.8 | cameras | -2.3 (7 of 10) | -54.4 (10 of 10) | -1.8 (4 of 10) | -0.5 (6 of 10) |

Actuated alone waits 79.8 s per trip at x0.5 and 98.4 s at x0.8; its ambulances lose 57.9 s and 86.2 s. At x0.8 the agents' throughput is level with actuated: the -2 to -5 vehicles an hour are well inside the seeds' spread of ±20.

**x1.0.** The town locks up in 9 of 10 runs under actuated and in all 10 under both kinds of agent. The exact-count agents still cut the ambulances' time loss (median -212 s, better on 7 seeds of 10), but cars gain nothing.

## Ablation (old signal cycle)

Measured on the old cycle and not yet repeated on the new signals. At x0.8 the ambulance gain is almost all pre-emption's, and skipping empty greens is the feature that saves car waiting. Each row switches one feature off (or one on) and is compared with the full exact-count agents on the same seed. The seeds are held-out 21-25, in the standard setting without the deadlock breaker.

| Change | x0.8 waiting (s) | x0.8 ambulance time loss (s) | x0.8 throughput (vehicles/h) | x1.0 throughput (vehicles/h) |
| --- | --- | --- | --- | --- |
| skip off | +5.2 | -6.4 | -5.8 | +395 |
| spill off | +2.2 | -4.9 | -1.8 | +67 |
| trickle off | +4.6 | +0.5 | -26 | -219 |
| balance off | +0.5 | -5.6 | +8 | -135 |
| corridor off | -1.1 | -0.9 | +6 | +67 |
| pre-emption and corridor off | +0.7 | +53.1 | -4 | +213 |
| patience off | +0.1 | -4.6 | -8 | -207 |
| noflow on | +44.8 | +57.0 | -165 | -26 |

Mean changes over the 5 seeds; a positive number means more waiting or loss, or more throughput.

- **x0.8.** Without pre-emption, ambulances lose 53 s more (worse on all 5 seeds). Without skip, cars wait 5.2 s more (worse on 4 of 5). Trickle saves 4.6 s; spill and balance count for little at this demand. Noflow made one seed lock up.
- **x1.0.** A single run swings by ±400 vehicles an hour, so 5 seeds settle little. Patience, trickle and balance help. Switching skip off gave more throughput here (+395, better on 4 of 5), and +82 on the development seeds: skipping empty greens may speed up the lock-up once the town is full.

**From the cameras** (the same seeds, compared with the full camera agents) the picture holds at x0.8. Pre-emption is worth 50.7 s per ambulance (worse on all 5 seeds without it), the corridor another 6 s, and skip 4.7 s of car waiting. Spill, trickle and balance each move waiting by under 2 s here. At x1.0, as with exact counts, skip off gave more throughput (+158 vehicles an hour, 3 of 5 seeds); noflow cost 209.

| Change, from the cameras | x0.8 waiting (s) | x0.8 ambulance time loss (s) | x1.0 throughput (vehicles/h) |
| --- | --- | --- | --- |
| skip off | +4.7 | -2.1 | +158 |
| spill off | +0.1 | -2.4 | +85 |
| trickle off | +0.6 | -1.9 | -36 |
| balance off | +1.8 | -1.1 | +27 |
| corridor off | +1.3 | +6.0 | +49 |
| pre-emption and corridor off | -0.2 | +50.7 | +144 |
| noflow on | +2.3 | -2.0 | -209 |

## The agents live in the 3D viewer

The Town05 viewer now runs the camera agents by default, and every decision shows on screen where it happens. `tools\town05_freefly.bat` starts with `--controller agents_observed`; `--controller actuated` gives SUMO's program alone.

&#91;image: Junction 1427 holds its green for an ambulance at the evening rush: the pill over the junction says what the agent did, and the green chevrons mark the stop lines of that green\]

&#91;image: Junction 2086 has just skipped an empty green: blue chevrons on the approach it passed over; the accident markers of the same scene sit beside it\]

- **Over the junction**, for 10 s after a decision, a pill says what the agent did: "skipped an empty green", "green held for an ambulance", "green wave: getting ready for an ambulance", and so on.
- **At the stop lines** of the green it was about, chevrons point in the direction of travel. Each kind of decision has its colour: blue for a skip, green for an ambulance, red for a full exit, amber for a trickle, purple for balance.
- **Top right**, the decisions near you in the last minute, with how far away and how long ago.

The viewer's own pre-emption gives way to the agents'. The agents read the same simulated camera reports as in the evaluation, refreshed every 2 s. The YOLO detector itself stays on the camera you open with H.

On the PC (GTX 1650, evening rush in the demo, old signal cycle), the viewer ran at 28-37 FPS with the agents on, and judder stayed at or under 6.5 %. The SUMO thread never fell behind. A simulated second now costs it 180-260 ms of work, against about 30 ms with actuated: the camera simulation is most of that. In 5 minutes of the demo the agents made 124 decisions for ambulances, 37 green-wave requests and 30 skips. The demo's light traffic gave no full exits or trickles to show.

`tools\freefly_agents_demo.bat` goes to junctions where an agent has just decided something, one kind of decision after another, and saves a screenshot of each to `out\freefly\town05_agents_*.png`.

## Re-evaluation on the new signals

Every figure above this section was measured on CARLA's own signal cycle. That cycle gives one approach (one pole) green at a time with all its movements, then a 10 s walk on every crossing with every car red. On 25-26 Oct the 15 programs were rebuilt the way most real junctions run, and the agents were evaluated again on them. The earlier figures are kept as they were measured and are labelled as old-cycle results. Nothing above was re-run on the new signals.

**What the new signals do** (`tools/build_town05_signals.py`; every program checked against SUMO's own junction logic by `tools/audit_town05_signals.py`):

| Item | Old cycle (until 25 Oct) | New signals (25-26 Oct) |
| --- | --- | --- |
| Greens per cycle | 4: one approach at a time (split phasing) | 2: opposite approaches together (N+S, then E+W) |
| Left turns | alone with their approach | permissive: on the same green, giving way to oncoming traffic (they share their lane with straight-on traffic, so a protected arrow would mean split phasing again; the busiest left turn is far below the usual threshold for one) |
| Right turns | alone with their approach | on the green, giving way to people on the crossing they turn over; no right turn on red |
| Pedestrians | a 10 s walk on all four crossings, every car red, then 5 s all red | a walk on the two crossings parallel to the green, then a flashing don't walk while the parallel cars keep their green, then a steady don't walk (the yellow and all red) |
| Pedestrian clearance | 5 s for a 14-24.5 m crossing | at least crossing length / 3.5 ft/s (MUTCD), counting the steady don't walk: 12 s flashing over 14 m, 15 s over 20 m, 16.5 s over the ring road's 24.5 m |
| Yellow, all red | 3 s, 3 s | from approach speed and junction width (ITE): 3.5 s (5 s on the ring road), 1.5-3 s |
| Actuated greens | 5-60 s | 7-45 s (7 s minimum: the MUTCD walk) |

**Checks on the signals themselves.** All 45 programs (fixed, actuated and Webster) pass: no conflicting greens unless one side gives way, a yellow and an all red before any conflicting green, enough clearance for every crossing, and each CARLA pole's links always agree. Two faults were found and fixed on 26 Oct. A 15 s cap left the ring-road crossings 21.5 s where 22.9 s are needed. Webster's split had cut every walk clearance to 5-8 s. In 4 hours of simulation (actuated and agents, x0.5 and x0.8) the step-by-step monitor found no conflicting greens, no green straight to red, no short intergreens and no released crossing whose clearance had not run. On the old cycle it counted 728-785 releases an hour of a movement crossing a crosswalk that was still occupied and had had only 5 s of clearance. Twelve stress scenarios at junction 1126 (heavy north-south or east-west traffic, left turners, right turners over busy crossings, people at every crossing or arriving at each switch, slow walkers, an ambulance every 30 s, uneven demand) under each controller showed the same: no rule broken anywhere in town. The longest wait was 124 s for a vehicle and 93 s on foot (`out/signal_checks/1126`). In CARLA, every pole carries four vehicle heads (three over the lanes, one on the post) that all show its state. CARLA has no pedestrian head on these poles, so the walk is SUMO's crossing signal.

**The pedestrian call.** On the new signals the agents first cost pedestrians 0.4-1.7 s per walk against actuated. The cause: the walk now runs with a car green, so skipping a green with no cars also skipped its walk. Since 25 Oct, someone waiting at a crossing counts as demand for the green that serves it, just as a push button places a call. On seeds 21-25 (`out/agents_pedfix`, paired with the same agents without it) it brought pedestrian time loss back level with actuated (-0.1 to +0.4 s, from +0.6 to +1.6 s on the same seeds without it). At junction 1126 under light traffic the longest wait at a crossing fell from 76 s to 35 s. Ambulances kept their gain (-10 to -14 s against actuated). The cost is car waiting: +0.4 s (exact counts) and +1.2 s (cameras) per trip at x0.5, and -0.6 / +0.5 s at x0.8. The call is kept: serving a pedestrian who has called is what a real controller does, and the cost to cars is small. `agents-pedcall` turns it off for comparison.

**Final evaluation on the corrected signals** (26 Oct: the two clearance fixes and the pedestrian call; same protocol as before: actuated, agents and agents_observed, x0.5 to x1.2, held-out seeds 21-30, 1 h each, no deadlock breaker; `out/agents_final_v3/agents.md`). Mean of the paired changes against actuated, with the seeds where the agents did better:

| demand | seeds | actuated: waiting /trip | agents (exact) vs actuated: waiting, throughput /h, ambulance time loss, pedestrian time loss | agents (cameras) vs actuated: the same | vehicles stuck > 300 s at the end (all runs) |
| --- | --- | --- | --- | --- | --- |
| x0.5 | 21-30 | 29.1 s (ambulance 33.9 s, pedestrian 59.3 s) | -0.3s (8/10), -1.4 (3/10), -10.5s (10/10), +0.0s (4/10) | -0.1s (6/10), -1.4 (2/10), -10.7s (10/10), +0.0s (5/10) | 0 |
| x0.8 | 21-30 | 31.4 s (ambulance 41.4 s, pedestrian 59.4 s) | -0.0s (5/10), +0.3 (5/10), -13.0s (10/10), -0.0s (3/10) | -0.2s (7/10), +0.7 (6/10), -12.9s (10/10), +0.1s (5/10) | 0 |
| x1 | 21-30 | 35.0 s (ambulance 50.2 s, pedestrian 59.7 s) | +0.3s (4/10), +2.7 (6/10), -18.3s (10/10), +0.2s (4/10) | -0.2s (6/10), +1.5 (3/10), -18.1s (10/10), +0.2s (3/10) | 0 |
| x1.2 | 21-30 | 42.1 s (ambulance 62.7 s, pedestrian 60.8 s) | -0.3s (6/10), +3.8 (8/10), -26.2s (10/10), +0.1s (5/10) | -0.1s (6/10), +1.2 (8/10), -24.8s (10/10), -0.1s (7/10) | 0 |

No vehicle was stuck at the end of any of the 120 runs, and no run locked up, even at x1.2. On 25 Oct, before the two fixes (`out/agents_final_v2`), the agents were within about ±1 s of actuated for cars, cut ambulance time loss by 10.5-25.1 s on every seed, and cost pedestrians 0.4-1.7 s per walk. With the fixes, pedestrians are level (within ±0.2 s at every demand), cars stay within ±0.3 s of actuated, ambulances keep their gain (-10.5 to -26.2 s, every seed), and throughput is level (within ±4 vehicles an hour). Actuated itself barely moved with the longer ring-road clearance: 29.1 s waiting at x0.5 before and after.

**Before and after, in short.** Actuated alone waited 79.8 s per trip at x0.5 on the old cycle (with SUMO's deadlock breaker) and 29.1 s on the new signals. The town locked up at x1.0 under every controller then; now no run locks up at x0.5-x1.2. That change is not the signals' alone. Between the two evaluations the traffic model also changed: a stuck car inside a junction now edges through after 40 s instead of SUMO's deadlock breaker, and emergency vehicles on a red wait for a clear junction. So the old and new figures are not one paired comparison, and only the differences within each evaluation (agents against actuated, same seeds) are.

**What holds and what does not.**

- **Ambulances.** The agents' clearest gain holds on the new signals: -10 to -26 s of time loss against actuated, better on every seed at every demand. On the old cycle it was -38 to -54 s, but there actuated's ambulances also lost far more (58-86 s).
- **Cars.** The car-waiting gain (-6.7 to -7.2 s at x0.5 on the old cycle) is gone. With two greens a cycle there are few empty greens left to skip, and the agents are now within about ±1 s of actuated.
- **Pedestrians.** With the pedestrian call, pedestrians are level with actuated.
- **Not reproduced on the new signals:** the ablation, the camera-only check of sirens, and the viewer's frame rate with the agents on.

**Known limits of the new signals.**

- SUMO's junction model now sees vehicles in conflicting streams sharing a green (permissive lefts, turns across crosswalks). With `--collision.check-junctions` it reports a few contacts inside junctions. In the evaluation setting that was 0-2 vehicle contacts per simulated hour town-wide, two of them involving emergency vehicles on a run, and 7-16 vehicle-pedestrian contacts (50 in 4 hours). Most of these were at walking pace (median 1.3 m/s; 4 of the 50 at 4-8 m/s). In the viewer's setting it was 13 contact steps in 15 minutes, against none on the old cycle. The viewer's drawn overlaps were about the same as on the old cycle per vehicle moved (8 in junctions against 6, with 41 % more vehicles moving). Lengthening the left turners' accepted gap did not change them. Vehicle-pedestrian contacts were reported on the old cycle too: 7-9 an hour, faster (median 2.5 m/s, 7 of 16 at 4-11 m/s).
- Junction 2296's north approach has no CARLA pole on Town05's map, so the signal there exists in SUMO only.

## Note: protected signals (27 Oct)

Everything above was measured on the permissive signals of 25-26 Oct (turns giving way inside the junction). Since
27 Oct Town05 runs protected signals by default: no green ever has to give way, pedestrians cross on a walk phase of
their own, each approach on its own green (README "Signals where no green gives way"). With the agents at x0.8, cars
standing inside a junction for a conflict fell from 700 to 4 a run, and waiting per trip rose from 37 s to 106 s. The
three approaches at 53, 979 and 1722 called "three-lane" above have two driving lanes and a parking lane. The agents
have not been re-evaluated on the protected signals yet.

## Limits and next steps

- **Over capacity (old cycle).** At x1.0, demand was above what Town05's grid of short links (14-82 m, two car lanes) could carry under the old cycle, and every controller locked up within the hour. On the new signals, with the traffic-model changes of 25 Oct, no run locks up up to x1.2. Demand-side measures (rerouting, gating the entry roads) remain the next lever beyond that.
- **Skip in a full town.** Skipping empty greens saves waiting up to x0.8, but at x1.0 switching it off gave more throughput on both seed sets. A skip that stands down once the town fills up is the next feature to try.
- **Simulated cameras.** The camera agents run on `SimulatedCameras`: the cameras' geometry with the detector's measured miss and false-box rates. The YOLO detector itself was checked on three junctions under light traffic only, so heavier traffic is still to test live.
- **Fixed phase plan.** The agents keep the actuated program's phases and their order (now two greens, opposite approaches together, each with its walk and clearance). They never invent a new phase, which keeps them safe but limits what they can gain. With two greens a cycle there is little left to skip, so the car gain of the old cycle is gone.
- **Permissive conflicts in SUMO.** Permissive lefts and turns across crosswalks bring a few contacts inside junctions in SUMO's model: 0-2 vehicle contacts an hour town-wide, and vehicle-pedestrian contacts at walking pace. None of the vehicle contacts happened on the old cycle. A leading pedestrian interval (the walk starting 3-7 s before the parallel green) is a real-world measure to try at the busiest crosswalks.
- **Not yet repeated on the new signals:** the ablation, the siren check from the cameras, and the viewer's frame rate with the agents on.
- **One town.** All of this is on Town05. The real Soukra corridor and other networks are the next test of whether the agents generalise.
- **Next:** an LLM agent that reads the agents' decision log and explains, in plain words, what each junction did and why.
