@echo off
rem Unattended check of the junction agents in the viewer (cosim\agents_actuated.py, cosim\agent_hud.py): at the
rem evening rush (17:30), it goes to junctions where the agents have just decided something (emergency vehicles first)
rem and takes a screenshot of each with the overlay (out\freefly\town05_agents_*.png, out\town05_freefly_agents.json).
set FREEFLY_ARGS=--demo 1 --demo-agents 300 --controller agents_observed --agent-hud --set-time 17:30 --clock-file out/freefly/demo_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %*
