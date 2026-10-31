@echo off
rem Town05 free-fly, live, starting over one signalised junction (default 1126) at midday, with the drawn
rem pedestrian lamps (--ped-lights: green = walk, flashing red = clearance, red = don't walk). Fly on as usual.
rem   tools\freefly_junction_live.bat [junction]
set J=%1
if "%J%"=="" set J=1126
set FREEFLY_ARGS=--start-junction %J% --ped-lights --set-time 12:00 %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat"
