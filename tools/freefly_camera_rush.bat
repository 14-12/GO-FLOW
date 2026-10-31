@echo off
rem The live traffic camera (H) checked at rush hour: the full scripted demo from 08:00 with --profile, ending with
rem the native camera measured for 10 s. Results: out\town05_freefly_demo.json ("cctv_native"),
rem out\freefly\town05_demo_cctv_native.png, tools\logs\freefly_town05.log
call "%~dp0_env.bat"
set FREEFLY_ARGS=--demo 40 --set-time 08:00 --profile
call "%~dp0town05_freefly.bat"
echo camera check done
