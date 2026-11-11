@echo off
rem Town05 free-fly: fly the CARLA window yourself while SUMO's traffic drives the town (cosim\freefly.py).
rem   town05_freefly.bat [medium|epic]   render profile, default medium (measured on this PC, out\town05_fps\summary.md:
rem                                      medium 50 FPS flying high / 112 at street level, epic 46 / 72)
rem Extra options for cosim\freefly.py can be put in FREEFLY_ARGS, e.g.  set FREEFLY_ARGS=--set-time 18:30
rem CARLA goes borderless full screen on its monitor. To keep it in a window: set CARLA_RES=1600x900 (any size).
rem CARLA is restarted, because its render settings are only read when it starts. The profile
rem (tools\fps_test\engine_<profile>.ini) goes into CARLA's user Engine.ini for the start and is taken out again once
rem CARLA answers and after it closes (tools\ini_profile.py), so the other scripts keep CARLA's own settings.
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
if "%CARLA_EXE%"=="" (echo CarlaUE4.exe not found under %CARLA_DIR% & pause & exit /b 1)
set P=%~1
if "%P%"=="" set P=medium
set INI=medium
if /i "%P%"=="epic" set INI=none
set FULL=1
if not "%CARLA_RES%"=="" set FULL=0
if "%CARLA_RES%"=="" set CARLA_RES=1280x720
if "%FULL%"=="0" set FREEFLY_ARGS=--windowed %FREEFLY_ARGS%
if "%FULL%"=="1" if /i "%INI%"=="medium" set INI=medium_fs
for /f "tokens=1,2 delims=x" %%a in ("%CARLA_RES%") do (set RX=%%a& set RY=%%b)
"%PY%" -c "import pygame" >nul 2>&1 || "%PY%" -m pip install pygame==2.6.1

echo Town05 free-fly, profile %P%, window %RX%x%RY%. Starting CARLA (about a minute) ...
rem the loading screen (cosim\splash.py), from now until the town is ready; set FREEFLY_SPLASH=0 for none
if "%FREEFLY_SPLASH%"=="" set FREEFLY_SPLASH=1
set PYW=%PY:python.exe=pythonw.exe%
if not exist "%PYW%" set PYW=%PY%
if "%FREEFLY_SPLASH%"=="1" start "" "%PYW%" -m cosim.splash
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1" >nul 2>&1
call :stopcarla
rem the number plates' mesh, and the cars' own plates plain black (tools\plates\install_plates.py; checked by md5)
"%PY%" tools\plates\install_plates.py install >nul
rem CARLA's own W A S D movement off: the viewer flies the camera (tools\carla_keys.py; nothing done when already so)
"%PY%" tools\carla_keys.py install >nul
call :install %INI%
start "CARLA" "%CARLA_EXE%" -quality-level=Epic -windowed -ResX=%RX% -ResY=%RY% -carla-rpc-port=2000
"%PY%" tools\wait_carla.py --max 300
set RC=%errorlevel%
call :restore
if not "%RC%"=="0" ("%PY%" -c "from cosim.splash import write_status as w; w('CARLA did not answer', error='CARLA did not answer')" & echo CARLA did not answer. & pause & exit /b 1)
echo.
echo   In the CARLA window: hold the RIGHT mouse button to look, Z Q S D to move (hold Shift: 2x faster), E / A up / down, arrows to turn.
echo   Keys (press them in the CARLA picture): F5 pause, F6 / F7 one hour back / forward, F8 next junction, F9 hide clock,
echo   H inspect the traffic camera you are close to (a blue box on a pole), Esc quit.
echo.
"%PY%" -m cosim.freefly %FREEFLY_ARGS%
set FRC=%errorlevel%
echo free-fly ended with %FRC%  (log: tools\logs\freefly_town05.log)
rem on an error, keep this window open to read it (CARLA is stopped after); not for unattended jobs (FREEFLY_NOPAUSE=1)
if not "%FRC%"=="0" if not defined FREEFLY_NOPAUSE pause
call :stopcarla
rem CARLA writes its Engine.ini back when its window is closed: take the profile out again
call :restore
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

:install
rem the profile into CARLA's user Engine.ini (tools\ini_profile.py; "none": only take any profile out)
if /i "%~1"=="none" ("%PY%" tools\ini_profile.py remove) else ("%PY%" tools\ini_profile.py install %~1)
exit /b 0

:restore
"%PY%" tools\ini_profile.py remove >nul
exit /b 0
