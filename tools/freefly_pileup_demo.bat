@echo off
rem Unattended check of a crash on the ring road (31 Oct, cosim\incidents.py): nobody careless in town, no breakdowns,
rem a car on the ring road brakes hard about every 20 s of simulated time, so the first incident is a crash there, often
rem a pile-up (the cars behind it late to react). Screenshots and timeline as freefly_incident_demo.bat writes them.
set FREEFLY_ARGS=--demo 1 --demo-incident 900 --careless-share 0 --no-breakdowns --ring-brake-every 20 --set-time 17:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
