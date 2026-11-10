@echo off
rem Clips of emergency vehicles, lights on and off, through the inspector's cameras (tools\record_ev_clips.py).
rem   record_ev_clips.bat          all cameras, 3 times of day -> out\ev_clips          (log: tools\logs\record_ev_clips.log)
rem   record_ev_clips.bat check    2 cameras, 1 time of day     -> out\ev_clips_check
rem CARLA is started off screen and stopped at the end; a co-simulation still running is stopped first.
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
if "%CARLA_EXE%"=="" (echo CarlaUE4.exe not found under %CARLA_DIR% & exit /b 1)
set LOG=%LOGS%\record_ev_clips.log
set ARGS=--out out\ev_clips --repeats 2
if /i "%~1"=="check" (
  set LOG=%LOGS%\record_ev_clips_check.log
  set ARGS=--out out\ev_clips_check --limit-cams 2 --conditions 1
)
set PYTHONUNBUFFERED=1
echo start %DATE% %TIME% > "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1" >nul 2>&1
call :stopcarla
"%PY%" tools\ini_profile.py install medium >> "%LOG%" 2>&1
start "CARLA" "%CARLA_EXE%" -quality-level=Epic -RenderOffScreen -carla-rpc-port=2000
"%PY%" tools\wait_carla.py --max 300 >> "%LOG%" 2>&1
set RC=%errorlevel%
"%PY%" tools\ini_profile.py remove >nul
if not "%RC%"=="0" (echo CARLA did not answer >> "%LOG%" & call :stopcarla & exit /b 1)
echo CARLA up %DATE% %TIME% >> "%LOG%"
"%PY%" tools\record_ev_clips.py %ARGS% >> "%LOG%" 2>&1
echo EXIT %errorlevel% %DATE% %TIME% >> "%LOG%"
call :stopcarla
"%PY%" tools\ini_profile.py remove >nul
exit /b 0

:stopcarla
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /f /im CarlaUE4.exe >nul 2>&1
set /a W=0
:gone
timeout /t 2 /nobreak >nul
set /a W+=1
tasklist /fi "imagename eq CarlaUE4-Win64-Shipping.exe" | find /i "CarlaUE4-Win64" >nul && if !W! lss 30 goto gone
exit /b 0
