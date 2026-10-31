@echo off
rem Step 2: lay out the 53 Town05 cameras, their lane masks and the coverage table (no CARLA server needed).
call "%~dp0_env.bat"
"%PY%" tools\town05_cameras.py plan --previews
echo EXIT %errorlevel%
pause
