@echo off
rem Double-click: Town05 uses the "approach" signals (29 Oct default: one colour per pole, CARLA's lamps lit) from now on (tools\use_signals.py; README "Signals where no
rem green gives way"). Close the viewer first. Log: tools\logs\signals_approach.log
call "%~dp0_env.bat"
"%PY%" tools\use_signals.py approach > "%LOGS%\signals_approach.log" 2>&1
type "%LOGS%\signals_approach.log"
pause
