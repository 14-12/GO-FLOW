@echo off
rem Tuning the junction agents (24 Oct): new settings against the current agents and actuated on the dev seeds 11-16,
rem x0.8 and x1.0. SUMO only. Results: out\tune_v3\baseline.md. Edit CTLS to try other settings.
call "%~dp0_env.bat"
if "%CTLS%"=="" set CTLS=actuated,agents,agents+noflow,agents+noflow@noflow_s=6,agents@full_share=0.7,agents_observed,agents_observed+noflow
if "%TUNE_OUT%"=="" set TUNE_OUT=out\tune_v3
"%PY%" tools\eval_town05_baseline.py --controllers %CTLS% --scales 0.8,1.0 --seeds 11,12,13,14,15,16 --jobs 6 --skip-done --out %TUNE_OUT% --title "Town05 agents tuning (dev seeds)" > "%LOGS%\tune_town05_agents.log" 2>&1
echo EXIT %errorlevel% >> "%LOGS%\tune_town05_agents.log"
