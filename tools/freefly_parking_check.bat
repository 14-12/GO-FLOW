@echo off
rem (9 Nov) The parking checks in CARLA, unattended, in one go (about 30 minutes; nothing to watch):
rem   1. where a car can drive into the car parks inside buildings: tools\town05_entrance_probe.bat (CARLA off screen,
rem      out\town05_places\entrances_probe.json and door_*.jpg);
rem   2. their ways in: tools\town05_entrances_build.py --write --figure (scenarios\town05\entrances.json, the stops
rem      moved in places.json and town05.places.add.xml, out\town05_places\entrances.png; log tools\logs\);
rem   3. the drives watched in the viewer: tools\freefly_parking_demo.bat (out\freefly\town05_parking_*.png,
rem      out\town05_freefly_parking.json).
setlocal
call "%~dp0town05_entrance_probe.bat"
call "%~dp0_env.bat"
"%PY%" tools\town05_entrances_build.py --write --figure > "%LOGS%\town05_entrances_build.log" 2>&1
call "%~dp0freefly_parking_demo.bat"
exit /b 0
