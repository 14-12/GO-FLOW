@echo off
rem Drivers in the cars tried in CARLA (tools\probe_drivers.py): seated drivers in each model. CARLA off screen.
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
if "%CARLA_EXE%"=="" (echo CarlaUE4.exe not found under %CARLA_DIR% & exit /b 1)
set LOG=%LOGS%\probe_drivers.log
set PYTHONUNBUFFERED=1
echo start %DATE% %TIME% > "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1" >nul 2>&1
call :stopcarla
start "CARLA" "%CARLA_EXE%" -quality-level=Epic -RenderOffScreen -carla-rpc-port=2000
"%PY%" tools\wait_carla.py --max 300 >> "%LOG%" 2>&1
if not "%errorlevel%"=="0" (echo CARLA did not answer >> "%LOG%" & call :stopcarla & exit /b 1)
echo CARLA up %DATE% %TIME% >> "%LOG%"
"%PY%" tools\probe_drivers.py >> "%LOG%" 2>&1
echo EXIT %errorlevel% %DATE% %TIME% >> "%LOG%"
call :stopcarla
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
