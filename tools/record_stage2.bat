@echo off
rem Double-click (6 Nov): the recordings for the GO-FLOW team's second delivery (their request of 2 Nov: 30 crashes or
rem more clearly visible to a fixed camera, a mix of ring-road and junction cameras, train and test, at least 3 test
rem crashes on cameras unseen in training). Option A (docs/WORKLOG.md, 6 Nov): the town's careless share stays the
rem normal 10 %%; a car heading into a place a recorded camera sees clearly (within 30 m, whole in the picture) becomes
rem careless with chance 0.5 and looks away there (--crash-focus 0.5). Whether a crash follows is still the traffic's;
rem each incident says whether the focus had a hand in it (crash_source in incidents.json). Crashes in front of the
rem cameras are therefore far more frequent than in reality: never read a rate from these recordings.
rem
rem Six runs one after the other, 50 minutes each (about 5.5 hours; the PC is busy meanwhile), four cameras each, a new
rem random seed per run, a different time of day per run. Each run records its full 50 minutes (no clip is cut early);
rem the delivery is cut from the recordings afterwards. Results: out\incident_clips\<time>\ per run.
rem   1 train  09:00  ring road: 2014-2 W, 2014-3 E, 2328-2 W, 2328-1 E
rem   2 train  12:00  junctions: 1126-1 E, 1427-3 S, 838-2 N, 979-4 W
rem   3 train  17:30  ring road again
rem   4 test   15:00  cameras never used before: 2014-1 S, 2328-3 N, 1292-1 W, 2086-1 N
rem   5 test   16:30  cameras never used before: 2240-1 N, 1882-1 E, 1126-2 S, 53-1 E
rem   6 test   10:30  ring road (the training cameras, another seed and time)
rem Expected (SUMO alone, tools/check_accidents.py --focus-cams ... --crash-focus 0.5, 2 seeds x 60 min per set): crashes
rem clearly seen per hour, ring 12, junctions 5.5, run 4's cameras 3, run 5's 2.5: about 38 in the six runs.
rem "record_stage2.bat trial": run 1's cameras for 25 minutes only, to see the yield before the six runs.
rem "record_stage2.bat 3": only run 3 (any of 1-6).
setlocal
if /i "%~1"=="trial" (call :one "2014-2 looking W;2014-3 looking E;2328-2 looking W;2328-1 looking E" 09:00 1500 & exit /b 0)
if "%~1"=="1" (call :run1 & exit /b 0)
if "%~1"=="2" (call :run2 & exit /b 0)
if "%~1"=="3" (call :run3 & exit /b 0)
if "%~1"=="4" (call :run4 & exit /b 0)
if "%~1"=="5" (call :run5 & exit /b 0)
if "%~1"=="6" (call :run6 & exit /b 0)
call :run1
call :run2
call :run3
call :run4
call :run5
call :run6
echo All six runs done.
exit /b 0

:run1
call :one "2014-2 looking W;2014-3 looking E;2328-2 looking W;2328-1 looking E" 09:00 3000
exit /b 0
:run2
call :one "1126-1 looking E;1427-3 looking S;838-2 looking N;979-4 looking W" 12:00 3000
exit /b 0
:run3
call :one "2014-2 looking W;2014-3 looking E;2328-2 looking W;2328-1 looking E" 17:30 3000
exit /b 0
:run4
call :one "2014-1 looking S;2328-3 looking N;1292-1 looking W;2086-1 looking N" 15:00 3000
exit /b 0
:run5
call :one "2240-1 looking N;1882-1 looking E;1126-2 looking S;53-1 looking E" 16:30 3000
exit /b 0
:run6
call :one "2014-2 looking W;2014-3 looking E;2328-2 looking W;2328-1 looking E" 10:30 3000
exit /b 0

:one
set FREEFLY_ARGS=--demo 1 --record-clips %~3 --record-cams "%~1" --record-crash-match __never__ --record-dynamic 0 --careless-share 0.1 --crash-focus 0.5 --seed %RANDOM% --clock-speed 1 --set-time %~2 --no-cameras --clock-file out/freefly/clips_clock.json
call "%~dp0town05_freefly.bat"
exit /b 0
