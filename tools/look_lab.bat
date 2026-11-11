@echo off
rem Look lab (tools\look_lab.py): CARLA with a render profile, full screen, no traffic.
rem   look_lab.bat PROFILE [arguments of look_lab.py]
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
set INI=%~1
if "%INI%"=="" set INI=medium_fs
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1" >nul 2>&1
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /f /im CarlaUE4.exe >nul 2>&1
ping -n 6 127.0.0.1 >nul
"%PY%" tools\ini_profile.py install %INI%
start "CARLA" "%CARLA_EXE%" -quality-level=Epic -windowed -ResX=1280 -ResY=720 -carla-rpc-port=2000
"%PY%" tools\wait_carla.py --max 300
set RC=%errorlevel%
"%PY%" tools\ini_profile.py remove >nul
if not "%RC%"=="0" (echo CARLA did not answer & exit /b 1)
"%PY%" tools\look_lab.py %2 %3 %4 %5 %6 %7 %8 %9
echo look_lab exit %errorlevel%
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /f /im CarlaUE4.exe >nul 2>&1
ping -n 6 127.0.0.1 >nul
"%PY%" tools\ini_profile.py remove >nul
exit /b 0
