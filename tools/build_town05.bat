@echo off
rem Builds scenarios\town05 (SUMO network, demand, signal programs) from Town05.xodr in the CARLA package. No server needed.
rem Town05: 15 signalised junctions (256 movements, 56 put under a signal by cosim\tlcomplete.py), multi-lane grid.
rem Demand scale 1.1 (about 2900 vehicles/h). Measured with SUMO alone, one hour, no teleporting: 1.4 gridlocks,
rem 1.2 starts to (9 vehicles stopped over 2 min under actuated); 1.1 flows under actuated and Webster, and under
rem actuated with SUMO seeds 1, 2, 3 (2900-2907 trips, 0 stopped over 2 min, at most 156 vehicles in the town).
setlocal
call "%~dp0_env.bat"
"%PY%" -m cosim.town --xodr "%CARLA_DIR%\CarlaUE4\Content\Carla\Maps\OpenDrive\Town05.xodr" --carla-map Town05 --name town05 --scale 1.1 > "%LOGS%\build_town05.log" 2>&1
echo EXIT %errorlevel% >> "%LOGS%\build_town05.log"
type "%LOGS%\build_town05.log"
