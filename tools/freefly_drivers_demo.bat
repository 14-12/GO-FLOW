@echo off
rem Unattended check of the drivers (cosim\drivers.py): at 17:00, 8 cars standing in traffic, each seen close
rem from the driver's side and from the front, a screenshot of each (out\freefly\town05_drivers_*.png,
rem out\town05_freefly_drivers.json).
set FREEFLY_ARGS=--demo 1 --demo-drivers 8 --set-time 17:00 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
