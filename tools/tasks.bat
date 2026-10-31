@echo off
rem Snapshot of the processes that matter, into tools\logs\tasks.txt (see tasks.ps1).
call "%~dp0_env.bat"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tasks.ps1" > "%LOGS%\tasks.txt" 2>&1
