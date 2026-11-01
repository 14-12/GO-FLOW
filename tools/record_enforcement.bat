@echo off
rem Records the red-light camera site at junction 1574, road -29 (cosim\enforcement.py) for 40 minutes at the evening
rem rush, with every private car a driver who may go on through the end of the amber (1.0 instead of 0.10: a
rem 30-minute run at 0.6 had one car on red), so that some reach the line on red: the camera video, the picture times, the controller log and the simulation own
rem record of who entered on red, in out\enforcement\1574--29_<time>\; the red-light agent runs live too (its cases in
rem out\enforcement\cases\). The data to build and grade the red-light agent.
set FREEFLY_ARGS=--enforce 1574:-29 --enforce-record 2400 --red-runners 1.0 --set-time 17:30 --clock-speed 1 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
