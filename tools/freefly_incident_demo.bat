@echo off
rem Unattended check of the accidents (cosim\incidents.py, cosim\incident_scene.py): at the evening rush (17:30), close-ups of
rem number plates first, then it waits (up to 15 min) for an accident and watches it from the road until it is cleared,
rem a screenshot at each stage (out\freefly\town05_incident_*.png), the timeline in out\town05_freefly_incident.json.
rem More careless drivers than usual (0.3 of new cars instead of 0.10), so the first accident comes sooner.
set FREEFLY_ARGS=--demo 1 --demo-incident 900 --careless-share 0.3 --set-time 17:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
