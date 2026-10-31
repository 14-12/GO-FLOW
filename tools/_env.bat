@echo off
rem Shared settings for the .bat files in this folder (called, not run on its own).
set ROOT=%~dp0..
for %%I in ("%ROOT%") do set ROOT=%%~fI
set PY=%ROOT%\.venv\Scripts\python.exe
set LOGS=%ROOT%\tools\logs
set OUT=%ROOT%\out
if not exist "%LOGS%" mkdir "%LOGS%"
if not exist "%OUT%" mkdir "%OUT%"
set CARLA_DIR=%ROOT%\carla\CARLA_0.9.16
set CARLA_EXE=
if exist "%CARLA_DIR%\CarlaUE4.exe" set CARLA_EXE=%CARLA_DIR%\CarlaUE4.exe
if exist "%CARLA_DIR%\WindowsNoEditor\CarlaUE4.exe" set CARLA_EXE=%CARLA_DIR%\WindowsNoEditor\CarlaUE4.exe
cd /d "%ROOT%"
rem Ultralytics must not pip-install packages by itself (it tried onnxruntime-gpu in the middle of a run, 15 Oct).
set YOLO_AUTOINSTALL=False
