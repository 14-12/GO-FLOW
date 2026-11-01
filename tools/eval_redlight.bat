@echo off
rem Grades the red-light agent on the newest recording of record_enforcement.bat (tools\eval_redlight.py, on the GPU).
rem Log: tools\logs\eval_redlight.log; the grading in that recordings eval.json, its cases in agent_cases\.
call "%~dp0_env.bat"
for /f "delims=" %%d in ('dir /b /ad /o-d "%OUT%\enforcement\1574--29_*"') do (set REC=%%d& goto found)
:found
"%PY%" tools\eval_redlight.py "%OUT%\enforcement\%REC%" --pt models\carla-yolo26n-t05.pt > "%LOGS%\eval_redlight.log" 2>&1
type "%LOGS%\eval_redlight.log"
pause
