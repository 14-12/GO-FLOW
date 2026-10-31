@echo off
rem One scripted 60 s free-fly flight with the poses sent once per CARLA picture (--poses frame); judder measured.
rem Results: out\freefly\demo_poses_frame.json, log tools\logs\freefly_demo_poses_frame.log. Both ways: freefly_demo_compare.bat
call "%~dp0_env.bat"
set FREEFLY_ARGS=--demo 60 --poses frame
call "%~dp0town05_freefly.bat"
copy /y "%OUT%\town05_freefly_demo.json" "%OUT%\freefly\demo_poses_frame.json" >nul
copy /y "%LOGS%\freefly_town05.log" "%LOGS%\freefly_demo_poses_frame.log" >nul
echo frame check done
