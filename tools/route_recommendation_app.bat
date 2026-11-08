@echo off
rem The route agent's Streamlit page (agents\route_recommendation\app.py). Needs tools\setup_route_recommendation.bat
rem once, and Ollama with qwen2.5:7b for the explanations.
call "%~dp0_env.bat"
cd /d "%ROOT%\agents\route_recommendation"
"%ROOT%\agents\route_recommendation\.venv\Scripts\python.exe" -m streamlit run app.py
pause
