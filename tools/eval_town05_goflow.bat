@echo off
rem GO-FLOW priority agent against the junction agents, SUMO only (no CARLA). Integration plan, section 6.
rem   dev    : tuning, seeds 11-16        (this file with argument "dev")
rem   final  : held-out seeds 31-40 (the test), x0.5-x1.2, emergency every 120 s and 45 s; seeds 21-30 as well, for
rem            comparison only (they were used several times by the junction agents' own evaluation) (argument "final")
rem Each run's agent_stats says how often GO-FLOW was in charge (external_plan_steps) or had fallen back
rem (external_fallback_steps), and goflow_timeouts / goflow_errors / goflow_restarts / goflow_llm_*.
rem Results in out\goflow_priority_eval\<set>_<interval>\ (never overwritten: --skip-done). Log: tools\logs\eval_town05_goflow.log
setlocal
call "%~dp0_env.bat"
set L=%LOGS%\eval_town05_goflow.log
set CTL=actuated,agents,agents_observed,agents_goflow,agents_observed_goflow
if /i "%1"=="final" goto final
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.8,1.0 --seeds 11,12,13,14,15,16 --jobs 6 --skip-done --out "out\goflow_priority_eval\dev_120" --title "GO-FLOW priority: dev seeds 11-16" > "%L%" 2>&1
goto end
:final
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.5,0.8,1.0,1.2 --seeds 31,32,33,34,35,36,37,38,39,40 --jobs 6 --skip-done --out "out\goflow_priority_eval\test_120" --title "GO-FLOW priority: held-out seeds 31-40, emergency every 120 s" > "%L%" 2>&1
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.5,0.8,1.0,1.2 --seeds 31,32,33,34,35,36,37,38,39,40 --emergency-every 45 --jobs 6 --skip-done --out "out\goflow_priority_eval\test_45" --title "GO-FLOW priority: held-out seeds 31-40, emergency every 45 s" >> "%L%" 2>&1
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.8,1.0 --seeds 21,22,23,24,25,26,27,28,29,30 --jobs 6 --skip-done --out "out\goflow_priority_eval\seeds21-30_120" --title "GO-FLOW priority: seeds 21-30 (comparison only, not held out)" >> "%L%" 2>&1
:end
echo EXIT %errorlevel% >> "%L%"
type "%L%"
