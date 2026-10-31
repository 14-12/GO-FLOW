@echo off
call "%~dp0_env.bat"
"%PY%" tools\diag_tl.py out\diag_tl.json > "%LOGS%\diag_tl.log" 2>&1
echo EXIT %errorlevel% >> "%LOGS%\diag_tl.log"
