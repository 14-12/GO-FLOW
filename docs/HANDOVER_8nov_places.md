# Handover: trips with real destinations (8 Nov 2026, evening)

Follows `docs/HANDOVER_8nov.md` (read its section 1, the working rules, first: they still hold). This file covers the
"places" work started after commit `6846efc`. None of it is committed yet: the files are on the PC, uncommitted.

## The plan the user agreed (do the steps in order; when a step is done and checked, go on to the next)

The user wants every step to look impressive ("i want something impressive in every step").

1. **Places inventory**: DONE (see `docs/WORKLOG.md`, "Trips with real destinations").
2. **Trips with a purpose** (home -> work in the morning, shops / deliveries midday, work -> home evening, ~20 %
   through traffic; cars park at the end and leave later): DONE in SUMO, measured (WORKLOG table, chart
   `docs/img/town05_day.png`).
3. **Show it**: code DONE, partly checked in CARLA (below):
   - `cosim/route_minimap.py`: places on the mini-map, A/B as places, purpose header, "PARKS AT" row, parked count;
   - `cosim/place_hud.py`: labels over the CARLA picture, key L (`cosim/overlay.py` VK "L", `cosim/freefly.py`);
   - `cosim/freefly.py`: `--trips purpose` (default), prefill of parked cars at start (`pop.prefill`), parked cars left
     out of the mini-map dots, of "F follow" candidates and of the over-target count; the follow demo prefers a car
     with a purpose.
   Checked in CARLA with `tools\freefly_follow_demo.bat` (8 Nov, 21:50): the demo followed live116 "TO WORK
   Résidence Les Oliviers -> Tour Horizon", rerouted (45 s shorter), parked: arrived in 2:48 against 2:40 estimated,
   "PARKED AT Tour Horizon 34 free of 40"; labels over the town shown, kept off the clock and the mini-map, 33-40 fps.
   Cars on the painted bays: fixed and measured (0.0 m off, `tools\freefly_places_demo.bat`); the mini-map's high
   "STANDING" was cars at red signals (now shown as "AT RED"). See the WORKLOG.
   Cars now drive into their space and out of it in the viewer (`cosim/parking_motion.py`, see the WORKLOG):
   check it with `tools\freefly_parking_demo.bat`; `tools/check_parking_motion.py` measures it without CARLA.
   **9 Nov, night** (see the WORKLOG, "Cars on drives that see each other..."): the cars on drives see each other
   (they wait for a clear way, give way, keep right in the aisles; SUMO's cars wait behind a car turning in); the
   Parking du Square's cars go through its gate (barrier arms opened by the viewer, its SUMO stop moved, `departPos`);
   the car parks inside buildings get a way in found in CARLA's geometry (`tools\freefly_parking_check.bat`: probe,
   `tools/town05_entrances_build.py --write`, then the demo). Measured without CARLA (`tools/check_parking_crowd.py`,
   `tools/check_parking_motion.py`); the CARLA run is the check of the ways in, the kerbs, the turn-back and the
   heading at the end.
4. **Re-measure on the new town**: remake the typical road times on train seeds 101-110 with purpose trips
   (`tools/route_delay_table.py` needs a trips option), check the forecasters on dev seeds 11-20 (retrain only if
   worse), rerun the route agent A/B on dev at x2 and x3 (`tools/eval_route_guidance.py` needs trips="purpose"; the
   agent's FleetGuide must respect purpose cars' parking stops).
5. **Test**: propose the bar to the user, WAIT for his agreement, then open test seeds 61-70 once (21-60 are used).

## Files (all in the PC folder)

New: `cosim/purposes.py`, `cosim/place_hud.py`, `cosim/place_icons.py`, `scenarios/town05/places.json`,
`scenarios/town05/town05.places.add.xml`, `scenarios/town05/parked_by_hour.json`, `tools/town05_places_probe.py`
(+ `.bat`), `tools/town05_places_build.py`, `tools/town05_places_atlas.py`, `tools/check_purpose_day.py`,
`tools/purpose_day_chart.py`, `tests/test_places.py`, `tests/test_purposes.py`, `docs/img/town05_places.jpg`,
`docs/img/town05_day.png`, this file.
Changed: `cosim/freefly.py`, `cosim/route_guidance.py`, `cosim/route_minimap.py`, `cosim/incidents.py`,
`cosim/overlay.py`, `tools/goflow_prediction_dataset.py`, `docs/WORKLOG.md`.
Tests: `python -m pytest -q tests` (281 passed on 8 Nov; 286 + 2 that need cv2 in the cloud on 9 Nov).
9 Nov, new: `tools/check_parking_crowd.py`, `tools/parking_crowd_figure.py`, `tools/town05_entrance_probe.py` (+ `.bat`),
`tools/town05_entrances_build.py`, `tools/freefly_parking_check.bat`, `scenarios/town05/entrances.json`,
`tests/test_entrances.py`, `docs/img/town05_parking_crowd.png`; changed: `cosim/parking_motion.py`, `cosim/freefly.py`
(doors, cars inside buildings not drawn, `--demo-parking-what`), `cosim/purposes.py`, `places.json` and
`town05.places.add.xml` (the Square's stop; the buildings' when the builder writes them).

## Things to know

- A parked car is a SUMO vehicle stopped in a `parkingArea` (off the road). `PurposeDemand.parked` (vehicle -> area),
  `.plans` (cars on a trip), `.idle` (cars parked at home), `.where` (car -> place), `.live(hour)` (what the viewer
  draws). The home stop is set only after the car has parked at its destination (setting it earlier merges stops).
- Keep `IncidentManager.parked` = `demand.parked`, or `unjam` removes parked cars.
- `matplotlib` is needed only for `tools/purpose_day_chart.py`.
- The user's own words on restaurants: add a restaurant car park only if the map has room; it has not, so the
  Brasserie du Nord parks at the Parking du Square across its street. Tell him plainly if asked.
