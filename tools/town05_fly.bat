@echo off
rem Town05 free-fly in the CARLA window, frame rate printed every second.
rem Usage: town05_fly.bat [medium|epic|low]   (default medium). Press Q in the console window to stop.
call "%~dp0town05_fps_test.bat" fly %*
