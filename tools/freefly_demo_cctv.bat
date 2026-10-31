@echo off
rem The live traffic camera (H) checked without a keyboard: a scripted 40 s free-fly flight from 12:00, then the
rem close-ups, the camera the sensor way and the native way (CARLA's own view at the camera, 10 s, measured).
rem Results: out\town05_freefly_demo.json ("cctv_native"), out\freefly\town05_demo_cctv_native.png, tools\logs\freefly_town05.log
call "%~dp0_env.bat"
set FREEFLY_ARGS=--demo 40 --set-time 12:00
call "%~dp0town05_freefly.bat"
echo camera check done
