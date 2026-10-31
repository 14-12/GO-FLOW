@echo off
rem Double-click: Town05 uses the "split" signals from now on (tools\use_signals.py; README "Signals where no
rem green gives way"). Close the viewer first. Log: tools\logs\signals_split.log
call "%~dp0_env.bat"
"%PY%" tools\use_signals.py split > "%LOGS%\signals_split.log" 2>&1
type "%LOGS%\signals_split.log"
pause
