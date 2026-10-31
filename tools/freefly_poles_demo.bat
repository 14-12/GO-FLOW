@echo off
rem Unattended close look at one junction's CARLA poles (default 1126): each pole from 8 m, once green and once
rem red, with the light boxes CARLA reports for it and the SUMO state then (what its own lamps show beside SUMO's
rem walk): out\freefly\town05_poles_*.png, out\town05_freefly_poles.json. Optional first argument: the junction.
set J=%1
if "%J%"=="" set J=1126
set FREEFLY_ARGS=--demo 1 --demo-signals 1 --demo-signals-close %J% --set-time 12:00 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat"
