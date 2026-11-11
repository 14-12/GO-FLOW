@echo off
rem Creates agents\incident_detection\.venv (Python 3.11) for Omar's accident agent and installs its inference
rem requirements there, never in the project's .venv; then runs its tests (his functions against the adapter).
rem Log: tools\logs\setup_incident_agent.log. Unattended (FREEFLY_NOPAUSE=1): no pause at the end.
setlocal
call "%~dp0_env.bat"
set L=%LOGS%\setup_incident_agent.log
set V=%ROOT%\agents\incident_detection\.venv
echo [setup_incident_agent] %DATE% %TIME% > "%L%"
if not exist "%V%\Scripts\python.exe" py -3.11 -m venv "%V%" >> "%L%" 2>&1
"%V%\Scripts\python.exe" -m pip install -r "%ROOT%\agents\incident_detection\requirements.txt" >> "%L%" 2>&1
echo pip EXIT %errorlevel% >> "%L%"
rem ByteTrack (his tracker) needs lap in the project's .venv too, where YOLO runs (cosim/incident_agent_proc.py)
"%PY%" -m pip install "lap>=0.5.12" >> "%L%" 2>&1
echo lap EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%\agents\incident_detection"
"%V%\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests >> "%L%" 2>&1
echo tests EXIT %errorlevel% >> "%L%"
type "%L%"
if not defined FREEFLY_NOPAUSE pause
endlocal
