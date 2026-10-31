@echo off
rem The live camera (H, native feed) at rush hour, three detector-sensor settings, 20 s each, measured:
rem   a  960 px, a picture every 0.2 s (the default)    b  640 px, every 0.2 s    c  640 px, every picture
rem Results: out\freefly\camera_compare_a.json, _b, _c ("poses_while_open": judder, CARLA pacing)
call "%~dp0_env.bat"
for %%v in (a b c) do call :one %%v
echo camera compare done
exit /b 0

:one
if "%1"=="a" set CAM=--cctv-width 960 --cctv-tick 0.2
if "%1"=="b" set CAM=--cctv-width 640 --cctv-tick 0.2
if "%1"=="c" set CAM=--cctv-width 640 --cctv-tick 0.0
set FREEFLY_ARGS=--demo 30 --demo-camera 20 --set-time 08:00 %CAM%
call "%~dp0town05_freefly.bat"
copy /y "%~dp0..\out\town05_freefly_camera.json" "%~dp0..\out\freefly\camera_compare_%1.json" >nul
exit /b 0
