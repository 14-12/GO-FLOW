@echo off
rem Checks the loading screen and the first view: a short scripted flight at 08:00 with the loading screen saving a
rem few of its own pictures. Results: out\freefly\splash_live_*.png, out\freefly\town05_first_view.png
call "%~dp0_env.bat"
set SPLASH_SHOTS=1
set FREEFLY_ARGS=--demo 20 --demo-flight-only --set-time 08:00
call "%~dp0town05_freefly.bat"
echo check done
