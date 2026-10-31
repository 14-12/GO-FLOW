@echo off
rem Tuning round 2 (24 Oct): the noflow rule (a green under which nobody moves is ended) and a lower "exit full"
rem share, against actuated and the agents, dev seeds 11-16, x0.8 and x1.0. Results: out\tune_v4\baseline.md.
call "%~dp0_env.bat"
set CTLS=actuated,agents,agents+noflow,agents+noflow@noflow_s=6,agents@full_share=0.7,agents+noflow@full_share=0.7,agents_observed,agents_observed+noflow
"%PY%" tools\eval_town05_baseline.py --controllers %CTLS% --scales 0.8,1.0 --seeds 11,12,13,14,15,16 --jobs 6 --skip-done --out out\tune_v4 --title "Town05 agents tuning, round 2 (dev seeds)" > "%LOGS%\tune2_town05_agents.log" 2>&1
echo EXIT %errorlevel% >> "%LOGS%\tune2_town05_agents.log"
