@echo off
rem Unattended check of the traffic lights (tools\build_town05_signals.py): at the evening rush (17:30), each of
rem Town05's 15 signalised junctions from 30 m up once some of its poles are green, a screenshot of each
rem (out\freefly\town05_signals_*.png); how many poles were green, yellow, red in out\town05_freefly_signals.json.
set FREEFLY_ARGS=--demo 1 --demo-signals 15 --set-time 17:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
