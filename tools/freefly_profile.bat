@echo off
rem Where the free-fly viewer spends its time: a scripted flight at 08:00 (rush hour) with --profile.
rem   freefly_profile.bat [extra freefly options], e.g. --no-ped-lights
rem Results: tools\logs\freefly_town05.log (lines "SUMO thread, ms", "CPU s", "profile <thread>", "poses per picture"),
rem out\town05_freefly_flight.json
call "%~dp0_env.bat"
set FREEFLY_ARGS=--demo 40 --demo-flight-only --set-time 08:00 --profile %*
call "%~dp0town05_freefly.bat"
echo profile run done
