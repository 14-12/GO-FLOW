@echo off
rem Double-click: Town05 uses the "leadlag" signals from now on (tools\use_signals.py; README "Signals where no
rem green gives way"). Close the viewer first. Log: tools\logs\signals_leadlag.log
call "%~dp0_env.bat"
"%PY%" tools\use_signals.py leadlag > "%LOGS%\signals_leadlag.log" 2>&1
type "%LOGS%\signals_leadlag.log"
pause
