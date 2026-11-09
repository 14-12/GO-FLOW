@echo off
rem Unattended check of a car SUMO takes out of its space again on its way in (staged, cosim\freefly.demo_parking):
rem it drives back the way it came, then to SUMO's car; and the kerbs in and out (out\freefly\town05_parking_*.png,
rem out\town05_freefly_parking.json).
set FREEFLY_ARGS=--demo 1 --demo-parking 4 --demo-parking-what turned_back,turned_back,kerb_in,kerb_out --set-time 08:30 --clock-file out/freefly/demo_clock_parking.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
