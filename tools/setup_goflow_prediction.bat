@echo off
rem Creates agents\goflow_prediction\.venv (Python 3.11) and installs its requirements there, never in the project's .venv.
rem Log: tools\logs\setup_goflow_prediction.log
call "%~dp0_env.bat"
set L=%LOGS%\setup_goflow_prediction.log
set V=%ROOT%\agents\goflow_prediction\.venv
echo [setup_goflow_prediction] %DATE% %TIME% > "%L%"
if not exist "%V%\Scripts\python.exe" py -3.11 -m venv "%V%" >> "%L%" 2>&1
"%V%\Scripts\python.exe" -m pip install -r "%ROOT%\agents\goflow_prediction\requirements.txt" >> "%L%" 2>&1
echo pip EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%\agents\goflow_prediction"
"%V%\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests >> "%L%" 2>&1
echo tests EXIT %errorlevel% >> "%L%"
type "%L%"
pause
