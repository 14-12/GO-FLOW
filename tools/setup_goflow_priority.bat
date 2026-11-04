@echo off
rem Creates the GO-FLOW priority agent's own environment: agents\priority_agent\.venv (Python 3.11), installs
rem agents\priority_agent\requirements.txt there (never into the project's .venv), runs its tests.
rem Log: tools\logs\setup_goflow_priority.log
setlocal
call "%~dp0_env.bat"
set AG=%ROOT%\agents\priority_agent
set APY=%AG%\.venv\Scripts\python.exe
set LOG=%LOGS%\setup_goflow_priority.log
set PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
echo [setup goflow priority] %DATE% %TIME% > "%LOG%"
if not exist "%APY%" (
  if exist "%PY311%" ( "%PY311%" -m venv "%AG%\.venv" >> "%LOG%" 2>&1 ) else ( py -3.11 -m venv "%AG%\.venv" >> "%LOG%" 2>&1 )
)
if not exist "%APY%" (echo SETUP FAILED: no venv >> "%LOG%" & echo No Python 3.11 found. & pause & exit /b 1)
"%APY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%APY%" -m pip install -r "%AG%\requirements.txt" >> "%LOG%" 2>&1
if errorlevel 1 (echo SETUP FAILED: pip >> "%LOG%" & echo pip failed, see %LOG% & pause & exit /b 1)
rem "setup_goflow_priority.bat llm": also the LLM coordinator's packages (then install Ollama from ollama.com and run
rem "ollama pull qwen2.5:3b" once)
if /i "%1"=="llm" "%APY%" -m pip install -r "%AG%\requirements-llm.txt" >> "%LOG%" 2>&1
pushd "%AG%"
"%APY%" -m pytest -q town05\tests tests\test_safety.py tests\test_step1.py >> "%LOG%" 2>&1
set ERR=%errorlevel%
popd
if not "%ERR%"=="0" (echo SETUP FAILED: tests >> "%LOG%" & echo Tests failed, see %LOG% & pause & exit /b 1)
echo SETUP OK >> "%LOG%"
echo GO-FLOW priority agent ready.
timeout /t 5
