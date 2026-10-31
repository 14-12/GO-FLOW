@echo off
rem Town05 training set for the traffic-camera detector (tools\make_dataset_town05.py), CARLA off screen.
rem   make_dataset_town05.bat              the full set, out\dataset_town05 (log: tools\logs\make_dataset_town05.log)
rem   make_dataset_town05.bat check        4 cameras, 2 conditions, 1 capture: out\dataset_town05_check
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
if "%CARLA_EXE%"=="" (echo CarlaUE4.exe not found under %CARLA_DIR% & pause & exit /b 1)
set LOG=%LOGS%\make_dataset_town05.log
set ARGS=
if /i "%~1"=="check" (
  set LOG=%LOGS%\make_dataset_town05_check.log
  set ARGS=--out out\dataset_town05_check --limit-cams 4 --conditions 2 --frames 1
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
"%PY%" tools\make_dataset_town05.py %ARGS% >> "%LOG%" 2>&1
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
