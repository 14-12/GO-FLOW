@echo off
rem Creates .venv (Python 3.11), installs requirements.txt, runs the tests. Log: tools\logs\setup.log
setlocal
call "%~dp0_env.bat"
set LOG=%LOGS%\setup.log
set PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
echo [setup] %DATE% %TIME% > "%LOG%"
if not exist "%PY%" (
  if exist "%PY311%" ( "%PY311%" -m venv .venv >> "%LOG%" 2>&1 ) else ( py -3.11 -m venv .venv >> "%LOG%" 2>&1 )
)
if not exist "%PY%" (echo SETUP FAILED: no venv >> "%LOG%" & echo No Python 3.11 found. & pause & exit /b 1)
echo Installing packages (a few minutes)...
"%PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%PY%" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 (echo SETUP FAILED: pip >> "%LOG%" & echo pip failed, see %LOG% & pause & exit /b 1)
"%PY%" -m pip freeze > "%LOGS%\pip-freeze.txt"
echo Running the tests...
"%PY%" -m pytest -q >> "%LOG%" 2>&1
if errorlevel 1 (echo SETUP FAILED: tests >> "%LOG%" & echo Tests failed, see %LOG% & pause & exit /b 1)
echo SETUP OK >> "%LOG%"
echo Setup OK.
timeout /t 5
