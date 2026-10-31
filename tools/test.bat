@echo off
setlocal
call "%~dp0_env.bat"
"%PY%" -m pytest -q > "%LOGS%\test.log" 2>&1
echo EXIT %errorlevel% >> "%LOGS%\test.log"
type "%LOGS%\test.log"
timeout /t 10
