@echo off
rem Town05 frame-rate test: what free-fly in the CARLA window would give, on this PC.
rem   town05_fps_test.bat                     4 configurations, about 12 minutes, unattended:
rem        epic_dx12_default   CARLA as it is today (Epic, DirectX 12)
rem        epic_dx11_poolfix   Epic, DirectX 11, texture pool fitted to the 4 GB GPU
rem        medium_dx11         Epic base + reduced shadows / effects / distance, 75 %% resolution upscaled
rem        low_dx11            CARLA's Low quality, DirectX 11
rem      Each: CARLA restarted in a 1280x720 window, Town05, 60 parked cars, 15 s warm-up, 30 s flying high
rem      over the town, 30 s at street level, 2 screenshots. DO NOT TOUCH THE CARLA WINDOW while it runs.
rem      Results: out\town05_fps\summary.md (+ screenshots), log tools\logs\town05_fps_test.log
rem   town05_fps_test.bat dx12                the two profiles again on DirectX 12 (town05_fps_dx12.bat)
rem   town05_fps_test.bat fly [medium^|epic^|low]   CARLA with that profile; YOU fly, FPS printed every second
rem   town05_fps_test.bat restore              takes any profile out of CARLA's Engine.ini (done automatically)
rem The profiles (tools\fps_test\engine_*.ini) go into CARLA's user Engine.ini for a run (tools\ini_profile.py).
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
if "%CARLA_EXE%"=="" (echo CarlaUE4.exe not found under %CARLA_DIR% & pause & exit /b 1)
set T=%ROOT%\tools\fps_test
set L=%LOGS%\town05_fps_test.log
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1" >nul 2>&1

if /i "%~1"=="restore" (call :restore & echo CARLA settings restored. & exit /b 0)
if /i "%~1"=="fly" goto fly
if /i "%~1"=="dx12" goto dx12

echo start %DATE% %TIME% > "%L%"
echo Town05 frame-rate test: 4 configurations, about 12 minutes. Do not touch the CARLA window.
call :run epic_dx12_default Epic "" none
call :run epic_dx11_poolfix Epic -dx11 poolfix
call :run medium_dx11 Epic -dx11 medium
call :run low_dx11 Low -dx11 poolfix
call :restore
"%PY%" tools\town05_fps_test.py --summary >> "%L%" 2>&1
echo end %DATE% %TIME% >> "%L%"
echo.
type "%ROOT%\out\town05_fps\summary.md"
echo.
echo Screenshots are in out\town05_fps. Full log: tools\logs\town05_fps_test.log
pause
exit /b 0

:dx12
rem the two profiles again on DirectX 12 (DirectX 11 left the GPU 40-60 %% idle in the first test)
echo ---- dx12 runs %DATE% %TIME% >> "%L%"
echo Town05 frame-rate test on DirectX 12: 2 configurations, about 6 minutes. Do not touch the CARLA window.
call :run epic_dx12_poolfix Epic "" poolfix
call :run medium_dx12 Epic "" medium
call :restore
"%PY%" tools\town05_fps_test.py --summary >> "%L%" 2>&1
type "%ROOT%\out\town05_fps\summary.md"
pause
exit /b 0

:fly
set P=%~2
if "%P%"=="" set P=medium
set Q=Epic
set INI=medium
if /i "%P%"=="epic" set INI=poolfix
if /i "%P%"=="low" (set Q=Low& set INI=poolfix)
echo Free-fly on Town05, profile %P% (quality !Q!, DirectX 11, ini !INI!). CARLA starts in about a minute.
call :stopcarla
call :install !INI!
start "CARLA" "%CARLA_EXE%" -quality-level=!Q! -windowed -ResX=1280 -ResY=720 -carla-rpc-port=2000 -dx11
"%PY%" tools\town05_fps_test.py --fly --quality !Q! --rhi=-dx11 --ini !INI!
call :stopcarla
call :restore
exit /b 0

:run
set TAG=%~1
set Q=%~2
set RHI=%~3
set INI=%~4
echo [%TIME%] %TAG% ...
echo ==== %TAG% quality=%Q% rhi=%RHI% ini=%INI% %TIME% >> "%L%"
call :stopcarla
call :install %INI%
start "CARLA" "%CARLA_EXE%" -quality-level=%Q% -windowed -ResX=1280 -ResY=720 -carla-rpc-port=2000 %RHI%
start "" /b nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.gr --format=csv,noheader,nounits -l 1 -f "%LOGS%\fps_%TAG%_gpu.csv"
"%PY%" tools\town05_fps_test.py --tag %TAG% --quality %Q% "--rhi=%RHI%" --ini %INI% >> "%L%" 2>&1
if errorlevel 1 echo    %TAG% failed, see tools\logs\town05_fps_test.log
taskkill /im nvidia-smi.exe /f >nul 2>&1
call :stopcarla
for %%F in ("%LOCALAPPDATA%\CarlaUE4\Saved\Logs\CarlaUE4.log" "%CARLA_DIR%\CarlaUE4\Saved\Logs\CarlaUE4.log") do if exist "%%~F" copy /y "%%~F" "%LOGS%\fps_%TAG%_carla.log" >nul
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
rem the profile into CARLA's user Engine.ini (tools\ini_profile.py; none: CARLA's own settings)
if /i "%~1"=="none" ("%PY%" tools\ini_profile.py remove >nul) else ("%PY%" tools\ini_profile.py install %~1 >nul)
exit /b 0

:restore
"%PY%" tools\ini_profile.py remove >nul
exit /b 0
