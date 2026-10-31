@echo off
rem Step 1: every signal controller on Town05, two demands, three seeds (SUMO only, no CARLA). About 30-60 min.
call "%~dp0_env.bat"
"%PY%" tools\eval_town05_baseline.py --jobs 2 --skip-done
echo EXIT %errorlevel%
pause
