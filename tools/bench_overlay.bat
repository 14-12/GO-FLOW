@echo off
rem How fast the see-through overlay window updates on this PC (no CARLA, about 15 s): out\freefly\bench_overlay.json
call "%~dp0_env.bat"
"%PY%" tools\bench_overlay.py > "%LOGS%\bench_overlay.log" 2>&1
