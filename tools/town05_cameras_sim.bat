@echo off
rem Step 2: max-pressure on simulated camera reports vs on exact counts (SUMO only). Needs town05_cameras_plan first.
call "%~dp0_env.bat"
"%PY%" tools\eval_town05_baseline.py --controllers max_pressure_walk,max_pressure_walk_observed --scales 0.5 --jobs 2 --skip-done
echo EXIT %errorlevel%
pause
