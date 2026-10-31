@echo off
rem The AI brief on steps 1 and 2 (one LLM call). Needs the keys in .env and the step 1 and 2 results.
call "%~dp0_env.bat"
"%PY%" tools\town05_brief.py
echo EXIT %errorlevel%
pause
