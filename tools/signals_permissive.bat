@echo off
rem Double-click: Town05 uses the "permissive" signals from now on (tools\use_signals.py; README "Signals where no
rem green gives way"). Close the viewer first. Log: tools\logs\signals_permissive.log
call "%~dp0_env.bat"
"%PY%" tools\use_signals.py permissive > "%LOGS%\signals_permissive.log" 2>&1
type "%LOGS%\signals_permissive.log"
pause
