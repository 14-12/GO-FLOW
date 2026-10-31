@echo off
rem Unattended check of the free-fly viewer: a 70 s scripted flight (10 s warm-up) at the evening rush (17:30) with traffic.
rem Frame rate to out\town05_freefly_demo.json, two screenshots to out\freefly\. Its own clock file: the real one is kept.
set FREEFLY_ARGS=--demo 70 --set-time 17:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
