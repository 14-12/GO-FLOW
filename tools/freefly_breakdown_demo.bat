@echo off
rem Unattended check of a breakdown (31 Oct, cosim\incidents.py): as freefly_incident_demo.bat, but nobody is careless
rem and a car breaks down about every 40 s of simulated time, so the first incident is a breakdown: hazards on, no
rem debris, the BREAKDOWN marker, the police and a recovery truck. Screenshots out\freefly\town05_incident_*.png, the
rem timeline out\town05_freefly_incident.json (both as freefly_incident_demo.bat writes them).
set FREEFLY_ARGS=--demo 1 --demo-incident 600 --careless-share 0 --breakdown-every 40 --set-time 12:00 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
