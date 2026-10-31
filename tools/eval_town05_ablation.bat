@echo off
rem Step 4: the agents v2 ablation, each feature off in turn (and the optional walk batch on), x0.8 and x1.0, seeds 1-3.
rem SUMO only, no CARLA. 60 one-hour runs, about 40-60 min with 2 jobs. Results: out\agents_town05_ablation\ablation.md
call "%~dp0_env.bat"
"%PY%" tools\eval_town05_baseline.py --controllers actuated,agents,agents-skip,agents-spill,agents-trickle,agents-balance,agents-extend,agents-preempt-corridor,agents-corridor,agents+batch --scales 0.8,1.0 --seeds 1,2,3 --jobs 2 --skip-done --out out\agents_town05_ablation --title "Town05 agents v2: ablation (one feature off at a time)"
"%PY%" tools\compare_town05_agents.py --out out\agents_town05_ablation --reference agents --name ablation --title "Town05: agents v2 ablation, one feature off at a time (delta against the full agents)"
echo EXIT %errorlevel%
pause
