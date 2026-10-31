@echo off
rem Runs the co-simulation with the given arguments (python -m cosim.run --help).
rem Log: tools\logs\%RUNLOG%.log (the run_*.bat wrappers set RUNLOG; default run.log).
setlocal
call "%~dp0_env.bat"
if "%RUNLOG%"=="" set RUNLOG=run
echo Running, log: tools\logs\%RUNLOG%.log
"%PY%" -m cosim.run %* > "%LOGS%\%RUNLOG%.log" 2>&1
echo EXIT %errorlevel% >> "%LOGS%\%RUNLOG%.log"
echo Done (exit %errorlevel%).
timeout /t 10
