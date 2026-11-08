@echo off
rem Creates agents\route_recommendation\.venv (Python 3.11) and installs its requirements there, never in the project's .venv.
rem Then: ollama pull qwen2.5:7b, and tools\route_recommendation_app.bat for the Streamlit page.
rem Log: tools\logs\setup_route_recommendation.log
call "%~dp0_env.bat"
set L=%LOGS%\setup_route_recommendation.log
set V=%ROOT%\agents\route_recommendation\.venv
echo [setup_route_recommendation] %DATE% %TIME% > "%L%"
if not exist "%V%\Scripts\python.exe" py -3.11 -m venv "%V%" >> "%L%" 2>&1
"%V%\Scripts\python.exe" -m pip install -r "%ROOT%\agents\route_recommendation\requirements.txt" >> "%L%" 2>&1
echo pip EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%\agents\route_recommendation"
"%V%\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests >> "%L%" 2>&1
echo tests EXIT %errorlevel% >> "%L%"
type "%L%"
pause
