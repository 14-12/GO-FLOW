@echo off
rem Town05 in the SUMO 2D window (no CARLA) with the GO-FLOW priority agent. The city runs fast, slows down and follows
rem each ambulance / police car on a run; the console prints what the junction agents and GO-FLOW decide.
rem   goflow_priority_demo.bat                  with GO-FLOW
rem   goflow_priority_demo.bat --without        the junction agents alone (before / after)
rem   goflow_priority_demo.bat --observed       the agents on the cameras' reports
rem   goflow_priority_demo.bat --llm qwen2.5:3b + the LLM in the background (Ollama running)
setlocal
call "%~dp0_env.bat"
"%PY%" tools\goflow_priority_demo.py %*
pause
