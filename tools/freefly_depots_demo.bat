@echo off
rem Unattended check of the depots (cosim\depots.py): at the evening rush (17:30), three rounds a minute apart of the
rem hospital, the police station and the recovery yard seen from across the road, a screenshot of each
rem (out\freefly\town05_depot_*.png); who was parked and out, and the emergency calls, in out\town05_freefly_depots.json.
set FREEFLY_ARGS=--demo 1 --demo-depots 3 --set-time 17:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
