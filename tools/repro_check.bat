@echo off
rem Reproducibility check: actuated and agents at x0.5, seed 1, here (SUMO only), to compare with the same runs
rem made elsewhere. Results: out\repro_check\runs\*.json (about 3 min)
call "%~dp0_env.bat"
"%PY%" tools\eval_town05_baseline.py --controllers actuated,agents --scales 0.5 --seeds 1 --jobs 2 --out out\repro_check --title "reproducibility check"
echo EXIT %errorlevel%
