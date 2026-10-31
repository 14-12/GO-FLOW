@echo off
rem Final evaluation of the junction agents on the corrected signals (26 Oct): the protocol of out\agents_final_v2
rem (actuated, agents, agents_observed; x0.5 / x0.8 / x1.0 / x1.2; held-out seeds 21-30; 1 h each; no deadlock
rem breaker) with the 26 Oct changes: walk clearances of the 24.5 m crossings at 2014 and 2328 16.5 s (was 15 s),
rem the pedestrian call in ActuatedAgents._demand. SUMO only, no CARLA.
rem Results: out\agents_final_v3\agents.md. Log: tools\logs\eval_town05_final_v3.log
call "%~dp0_env.bat"
set L=%LOGS%\eval_town05_final_v3.log
"%PY%" tools\eval_town05_baseline.py --controllers actuated,agents,agents_observed --scales 0.5,0.8,1.0,1.2 --seeds 21,22,23,24,25,26,27,28,29,30 --jobs 4 --skip-done --out out\agents_final_v3 --title "Town05 junction agents on the corrected signals, held-out seeds 21-30" > "%L%" 2>&1
"%PY%" tools\compare_town05_agents.py --out out\agents_final_v3 --title "Town05 junction agents on the corrected signals (26 Oct), held-out seeds 21-30" >> "%L%" 2>&1
echo EXIT %errorlevel% >> "%L%"
