@echo off
rem Unattended check of following a car with the route agent (cosim\route_guidance.py, F in the viewer): at 08:00 a
rem town car with a long way to go is followed, a breakdown is staged on its way (police sent after 15 s, so the
rem agent hears of it), the views are cycled (chase, roof, drone). Screenshots out\freefly\town05_follow_*.png, the
rem camera against the car and the trip in out\town05_freefly_follow.json.
set FREEFLY_ARGS=--demo 1 --demo-follow 240 --set-time 08:00 --police-delay 15:0 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
