@echo off
rem The final evaluation again with SUMO's deadlock breaker (--ignore-junction-blocker 60: a vehicle standing in a
rem junction for 60 s is no longer waited for; nobody is removed), x0.5 and x0.8, held-out seeds 21-30.
rem Results: out\agents_final_jb\agents.md. Log: tools\logs\eval_town05_final_jb.log
call "%~dp0_env.bat"
set L=%LOGS%\eval_town05_final_jb.log
"%PY%" tools\eval_town05_baseline.py --controllers actuated,agents,agents_observed --scales 0.5,0.8 --seeds 21,22,23,24,25,26,27,28,29,30 --jobs 6 --skip-done --ignore-junction-blocker 60 --out out\agents_final_jb --title "Town05 junction agents with SUMO's deadlock breaker (held-out seeds 21-30)" > "%L%" 2>&1
"%PY%" tools\compare_town05_agents.py --out out\agents_final_jb --title "Town05: junction agents against actuated with SUMO's deadlock breaker (held-out seeds 21-30)" >> "%L%" 2>&1
echo EXIT %errorlevel% >> "%L%"
