@echo off
rem Starts the CARLA server in a window. Usage: start_carla.bat [Epic|Low] (default Epic)
setlocal
call "%~dp0_env.bat"
set Q=%1
if "%Q%"=="" set Q=Epic
if "%CARLA_EXE%"=="" (echo CarlaUE4.exe not found under %CARLA_DIR% - run download_carla.bat first. & pause & exit /b 1)
echo [start_carla] %DATE% %TIME% quality=%Q% exe=%CARLA_EXE% > "%LOGS%\start_carla.log"
start "CARLA" "%CARLA_EXE%" -quality-level=%Q% -windowed -ResX=1280 -ResY=720 -carla-rpc-port=2000
echo CARLA is starting (quality %Q%). The first start takes a minute or two.
timeout /t 5
