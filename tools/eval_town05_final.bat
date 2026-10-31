@echo off
rem Final evaluation of the junction agents (24 Oct), on held-out seeds 21-30 never used for tuning:
rem   1. actuated, agents (exact counts), agents_observed (cameras), x0.5 / x0.8 / x1.0  -> out\agents_final\agents.md
rem   2. ablation from the cameras, one feature off at a time, x0.8 and x1.0, seeds 21-25 -> out\agents_final_ablation\
rem SUMO only, no CARLA. Several hours with 6 jobs. Log: tools\logs\eval_town05_final.log
call "%~dp0_env.bat"
set L=%LOGS%\eval_town05_final.log
"%PY%" tools\eval_town05_baseline.py --controllers actuated,agents,agents_observed --scales 0.5,0.8,1.0 --seeds 21,22,23,24,25,26,27,28,29,30 --jobs 6 --skip-done --out out\agents_final --title "Town05 junction agents, final evaluation (held-out seeds 21-30)" > "%L%" 2>&1
"%PY%" tools\compare_town05_agents.py --out out\agents_final --title "Town05: junction agents against actuated (held-out seeds 21-30)" >> "%L%" 2>&1
"%PY%" tools\eval_town05_baseline.py --controllers agents_observed,agents_observed-skip,agents_observed-spill,agents_observed-trickle,agents_observed-balance,agents_observed-corridor,agents_observed-preempt-corridor,agents_observed+noflow --scales 0.8,1.0 --seeds 21,22,23,24,25 --jobs 6 --skip-done --out out\agents_final_ablation --title "Town05 agents from the cameras: ablation (held-out seeds 21-25)" >> "%L%" 2>&1
"%PY%" tools\compare_town05_agents.py --out out\agents_final_ablation --reference agents_observed --name ablation_cameras --title "Town05 agents from the cameras: one feature off at a time (delta against the full agents)" >> "%L%" 2>&1
echo EXIT %errorlevel% >> "%L%"
