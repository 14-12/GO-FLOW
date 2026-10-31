@echo off
rem Which free models answer with the keys in .env (a few tokens each).
call "%~dp0_env.bat"
"%PY%" tools\llm_check.py
echo EXIT %errorlevel%
pause
