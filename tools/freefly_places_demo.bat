@echo off
rem Unattended check of the town's car parks (cosim\purposes.py): at 12:30 each car park with painted bays and each
rem parking street is seen from 28 m up; a screenshot of each, and how far each parked car's CARLA body is from its
rem space. Screenshots out\freefly\town05_places_*.png, the numbers in out\town05_freefly_places.json.
set FREEFLY_ARGS=--demo 1 --demo-places 1 --set-time 12:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
