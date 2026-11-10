@echo off
rem The viewer with the junction agents on the cameras' reports and the GO-FLOW priority agent planning the emergency
rem vehicles' requests (cosim\goflow_priority.py). Needs tools\setup_goflow_priority.bat once.
rem Extra viewer options: set FREEFLY_ARGS=... before running this file.
setlocal
set FREEFLY_ARGS=--controller agents_observed --goflow-priority --agent-hud %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
