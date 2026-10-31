@echo off
rem The arrow keys with Z Q S D in the running free-fly viewer, for each way of turning (tools\probe_viewer_arrows.py).
rem Starts the viewer (full screen) per mode, measures, quits it. Results: out\probe_viewer_arrows.json, log tools\logs\probe_viewer_arrows.log
setlocal
call "%~dp0_env.bat"
set LOG=%LOGS%\probe_viewer_arrows.log
echo start %DATE% %TIME% > "%LOG%"
for %%M in (fly) do (
  echo mode %%M >> "%LOG%"
  set FREEFLY_ARGS=--arrows %%M --set-time 17:30 --clock-file out/freefly/probe_clock.json
  start "freefly %%M" cmd /c "%~dp0town05_freefly.bat"
  timeout /t 20 /nobreak >nul
  "%PY%" tools\probe_viewer_arrows.py --mode %%M >> "%LOG%" 2>&1
  timeout /t 25 /nobreak >nul
)
echo EXIT %DATE% %TIME% >> "%LOG%"
