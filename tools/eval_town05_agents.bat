@echo off
rem Step 3: junction agents v2 (on the actuated program) against actuated, x0.5/x0.8/x1.0, seeds 1-3, exact and camera counts.
rem SUMO only, no CARLA. 27 one-hour runs, about 45-90 min with 2 jobs. Results: out\agents_town05\agents.md
call "%~dp0_env.bat"
"%PY%" tools\eval_town05_baseline.py --controllers actuated,agents,agents_observed --scales 0.5,0.8,1.0 --seeds 1,2,3 --jobs 2 --skip-done --out out\agents_town05 --title "Town05 agents v2 against actuated"
"%PY%" tools\compare_town05_agents.py --out out\agents_town05
echo EXIT %errorlevel%
pause
