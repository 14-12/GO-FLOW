@echo off
rem What each key does to CARLA's spectator (tools\probe_keys.py): CARLA in a 1280x720 window, log tools\logs\probe_keys.log
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
set LOG=%LOGS%\probe_keys.log
echo start %DATE% %TIME% > "%LOG%"
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
timeout /t 3 /nobreak >nul
start "CARLA" "%CARLA_EXE%" -quality-level=Epic -windowed -ResX=1280 -ResY=720 -carla-rpc-port=2000
"%PY%" tools\wait_carla.py --max 300 >> "%LOG%" 2>&1
"%PY%" tools\probe_keys.py >> "%LOG%" 2>&1
echo EXIT %errorlevel% >> "%LOG%"
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
"%PY%" tools\ini_profile.py remove >nul
