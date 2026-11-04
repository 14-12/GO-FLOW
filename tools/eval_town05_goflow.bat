@echo off
rem GO-FLOW priority agent against the junction agents, SUMO only (no CARLA). Integration plan, section 6.
rem   dev    : tuning, seeds 11-16        (this file with argument "dev")
rem   final  : held-out seeds 21-30 and 31-40, x0.5-x1.2, emergency every 120 s and 45 s  (argument "final")
rem Results in out\goflow_priority_eval\<set>_<interval>\ (never overwritten: --skip-done). Log: tools\logs\eval_town05_goflow.log
setlocal
call "%~dp0_env.bat"
set L=%LOGS%\eval_town05_goflow.log
set CTL=actuated,agents,agents_observed,agents_goflow,agents_observed_goflow
if /i "%1"=="final" goto final
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.8,1.0 --seeds 11,12,13,14,15,16 --jobs 6 --skip-done --out "out\goflow_priority_eval\dev_120" --title "GO-FLOW priority: dev seeds 11-16" > "%L%" 2>&1
goto end
:final
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.5,0.8,1.0,1.2 --seeds 21,22,23,24,25,26,27,28,29,30 --jobs 6 --skip-done --out "out\goflow_priority_eval\testA_120" --title "GO-FLOW priority: held-out seeds 21-30, emergency every 120 s" > "%L%" 2>&1
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.5,0.8,1.0,1.2 --seeds 21,22,23,24,25,26,27,28,29,30 --emergency-every 45 --jobs 6 --skip-done --out "out\goflow_priority_eval\testA_45" --title "GO-FLOW priority: held-out seeds 21-30, emergency every 45 s" >> "%L%" 2>&1
"%PY%" tools\eval_town05_baseline.py --controllers %CTL% --scales 0.5,0.8,1.0,1.2 --seeds 31,32,33,34,35,36,37,38,39,40 --jobs 6 --skip-done --out "out\goflow_priority_eval\testB_120" --title "GO-FLOW priority: held-out seeds 31-40" >> "%L%" 2>&1
:end
echo EXIT %errorlevel% >> "%L%"
type "%L%"
