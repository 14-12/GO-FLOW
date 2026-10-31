@echo off
rem Step 2: real camera reports at junctions 1126, 1882, 2296 vs SUMO's truth. Start CARLA first (start_carla.bat) and close the free-fly viewer. About 1 h.
call "%~dp0_env.bat"
"%PY%" tools\town05_cameras.py live
echo EXIT %errorlevel%
pause
