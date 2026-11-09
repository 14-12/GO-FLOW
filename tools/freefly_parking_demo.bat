@echo off
rem Unattended check of the cars driven into their space and out of it (cosim\parking_motion.py), from 08:30: one drive
rem of each kind in turn (cosim\freefly.demo_parking): into and out of a car park inside a building (by its way in,
rem scenarios\town05\entrances.json), two cars in the same car park at once, at a kerb in and out, a car SUMO takes
rem out again half way into its space (staged), one into its space (its heading there); each watched from above,
rem screenshots (out\freefly\town05_parking_*.png, out\town05_freefly_parking.json).
set FREEFLY_ARGS=--demo 1 --demo-parking 7 --demo-parking-what building_in,building_out,lot_crowd,kerb_in,kerb_out,turned_back,lot_in --set-time 08:30 --clock-file out/freefly/demo_clock_parking.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
