@echo off
rem GO-FLOW priority agent against the junction agents, SUMO only (no CARLA), paired seeds.
rem   dev    : tuning, dev seeds 11-20, x0.8 and x1.0, an emergency every 120 s and every 45 s   (argument "dev")
rem   final  : the held-out test, seeds 71-80, same settings, opened once (TEST_OPENED.lock)       (argument "final")
rem Test seeds already used elsewhere: 21-30, 31-40, 41-50, 51-60 (forecasters, agents A/B), 61-70 (route agent).
rem Each run's agent_stats says how often GO-FLOW was in charge (external_plan_steps) or had fallen back
rem (external_fallback_steps), and goflow_timeouts / goflow_errors / goflow_restarts / goflow_llm_*.
rem Results in out\goflow_priority_eval\<set>_<interval>\ (never overwritten: --skip-done). Log: tools\logs\eval_town05_goflow.log
setlocal
call "%~dp0_env.bat"
set L=%LOGS%\eval_town05_goflow.log
set CTL=agents,agents_observed,agents_goflow,agents_observed_goflow
if /i "%1"=="final" goto final
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.8,1.0 --seeds 11,12,13,14,15,16,17,18,19,20 --end 1800 --jobs 6 --skip-done --out "out\goflow_priority_eval\dev_120" --title "GO-FLOW priority: dev seeds 11-20, emergency every 120 s" > "%L%" 2>&1
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.8,1.0 --seeds 11,12,13,14,15,16,17,18,19,20 --end 1800 --emergency-every 45 --jobs 6 --skip-done --out "out\goflow_priority_eval\dev_45" --title "GO-FLOW priority: dev seeds 11-20, emergency every 45 s" >> "%L%" 2>&1
goto end
:final
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.8,1.0 --seeds 71,72,73,74,75,76,77,78,79,80 --end 1800 --jobs 6 --out "out\goflow_priority_eval\test_120" --open-test "GO-FLOW priority, emergency every 120 s" --title "GO-FLOW priority: held-out seeds 71-80, emergency every 120 s" > "%L%" 2>&1
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.8,1.0 --seeds 71,72,73,74,75,76,77,78,79,80 --end 1800 --emergency-every 45 --jobs 6 --out "out\goflow_priority_eval\test_45" --open-test "GO-FLOW priority, emergency every 45 s" --title "GO-FLOW priority: held-out seeds 71-80, emergency every 45 s" >> "%L%" 2>&1
:end
echo EXIT %errorlevel% >> "%L%"
type "%L%"
