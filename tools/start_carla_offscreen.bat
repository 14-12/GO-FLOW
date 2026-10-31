@echo off
rem Starts the CARLA server without drawing its own window (-RenderOffScreen): the cameras still render,
rem but the GPU memory of the viewport is saved. Needed for the large maps on a 4 GB GPU. Usage: [Epic|Low]
setlocal
call "%~dp0_env.bat"
set Q=%1
if "%Q%"=="" set Q=Epic
echo [start_carla_offscreen] %DATE% %TIME% quality=%Q% > "%LOGS%\start_carla.log"
start "CARLA" "%CARLA_EXE%" -quality-level=%Q% -RenderOffScreen -carla-rpc-port=2000
