@echo off
rem The detector on the viewer's picture, one look against another (tools\look_detector_check.py).
rem   look_detector_check.bat PROFILE LOOK OUTDIR     e.g.  look_detector_check.bat medium_fs town out\look_check\town
rem Renders the traffic cameras' pictures with labels (tools\make_dataset_town05.py --look), CARLA off screen with the
rem render profile PROFILE (tools\fps_test\engine_PROFILE.ini): 12 cameras, 6 hours, 2 captures each, seed 11.
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
set PYTHONUNBUFFERED=1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1" >nul 2>&1
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /f /im CarlaUE4.exe >nul 2>&1
ping -n 6 127.0.0.1 >nul
"%PY%" tools\ini_profile.py install %1
start "CARLA" "%CARLA_EXE%" -quality-level=Epic -RenderOffScreen -carla-rpc-port=2000
"%PY%" tools\wait_carla.py --max 300
set RC=%errorlevel%
"%PY%" tools\ini_profile.py remove >nul
if not "%RC%"=="0" (echo CARLA did not answer & exit /b 1)
"%PY%" tools\make_dataset_town05.py --out %3 --look %2 --limit-cams 12 --frames 2
echo make_dataset exit %errorlevel%
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /f /im CarlaUE4.exe >nul 2>&1
ping -n 6 127.0.0.1 >nul
"%PY%" tools\ini_profile.py remove >nul
exit /b 0
