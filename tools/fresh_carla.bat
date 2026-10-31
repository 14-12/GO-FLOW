@echo off
rem Restarts the CARLA server cleanly: ends any co-simulation still running (stop_cosim.ps1) and any running or
rem hung server, starts it without a window (-RenderOffScreen,
rem cameras still render; saves the viewport's GPU memory on the 4 GB GPU) and waits until it answers.
rem Usage: fresh_carla.bat [Epic|Low]   (called by the large-map run scripts; can be run on its own)
setlocal
call "%~dp0_env.bat"
set Q=%1
if "%Q%"=="" set Q=Epic
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1"
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /f /im CarlaUE4.exe >nul 2>&1
rem wait until the old server has really gone (a hung one can take a while to die), at most ~60 s
set /a W=0
:gone
timeout /t 3 /nobreak >nul
set /a W+=1
tasklist /fi "imagename eq CarlaUE4-Win64-Shipping.exe" | find /i "CarlaUE4-Win64" >nul && if %W% lss 20 goto gone
start "CARLA" "%CARLA_EXE%" -quality-level=%Q% -RenderOffScreen -carla-rpc-port=2000
rem The wait has its own log: CARLA, started above, keeps open any file this script is redirected to.
"%PY%" tools\wait_carla.py --max 300 > "%LOGS%\fresh_carla.log" 2>&1
exit /b %errorlevel%
