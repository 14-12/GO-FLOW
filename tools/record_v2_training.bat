@echo off
rem Double-click (31 Oct): ordinary-traffic recordings for training GO-FLOW's incident agent v2. Four runs one after
rem the other (about 2.5 hours; the PC is busy meanwhile), four cameras each, 35 minutes per run, the town's normal
rem share of careless drivers (10 %), a new random seed per run, a different time of day per run:
rem   A 09:00  ring road: 2014-2 W, 2014-3 E, 2328-2 W, 2328-1 E
rem   B 12:00  junctions of stage 1: 1126-1 E, 1427-3 S, 838-2 N, 979-4 W
rem   C 15:00  more junctions: 53-3 W, 1574-1 N, 1722-4 N, 207-2 W
rem   D 17:30  the four city overview cameras (tools\record_city_cams.json)
rem No clip is cut early (no crash clip is wanted): each run records its full 35 minutes. Accidents that still happen
rem are logged as always (incidents.json). Results: out\incident_clips\<time>\ per run. These runs are for training;
rem the four runs of 30 Oct stay held out for testing.
call :one "2014-2 looking W;2014-3 looking E;2328-2 looking W;2328-1 looking E" "" 09:00
call :one "1126-1 looking E;1427-3 looking S;838-2 looking N;979-4 looking W" "" 12:00
call :one "53-3 looking W;1574-1 looking N;1722-4 looking N;207-2 looking W" "" 15:00
call :one "" "--record-specs tools/record_city_cams.json" 17:30
echo All four training runs done.
exit /b 0

:one
set FREEFLY_ARGS=--demo 1 --record-clips 2100 --record-cams "%~1" %~2 --record-crash-match __never__ --record-dynamic 0 --careless-share 0.1 --seed %RANDOM% --clock-speed 1 --set-time %~3 --no-cameras --clock-file out/freefly/clips_clock.json
call "%~dp0town05_freefly.bat"
exit /b 0
