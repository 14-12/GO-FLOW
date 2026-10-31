@echo off
rem Free-fly smoothness check (SUMO + CARLA, about 10 min): two scripted 60 s flights from 12:00, the poses first from a thread
rem of the viewer (--poses thread), then once per CARLA picture (--poses frame, cosim\pose_proc.py). The judder
rem (share of moving-vehicle pictures whose speed jumps) is measured from CARLA's own snapshots in both.
rem Results: out\freefly\demo_poses_thread.json and demo_poses_frame.json, logs in tools\logs.
call "%~dp0_env.bat"
set FREEFLY_ARGS=--demo 60 --set-time 12:00 --poses thread --measure-judder
call "%~dp0town05_freefly.bat"
copy /y "%OUT%\town05_freefly_demo.json" "%OUT%\freefly\demo_poses_thread.json" >nul
copy /y "%LOGS%\freefly_town05.log" "%LOGS%\freefly_demo_poses_thread.log" >nul
set FREEFLY_ARGS=--demo 60 --set-time 12:00 --poses frame
call "%~dp0town05_freefly.bat"
copy /y "%OUT%\town05_freefly_demo.json" "%OUT%\freefly\demo_poses_frame.json" >nul
copy /y "%LOGS%\freefly_town05.log" "%LOGS%\freefly_demo_poses_frame.log" >nul
echo smoothness check done
