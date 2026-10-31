@echo off
rem Unattended check of the ring road (cosim\highway.py): at the evening rush (17:30), six views along its loops, a
rem screenshot of each (out\freefly\town05_highway_*.png), the cars counted in out\town05_freefly_highway.json.
set FREEFLY_ARGS=--demo 1 --demo-highway 6 --set-time 17:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
