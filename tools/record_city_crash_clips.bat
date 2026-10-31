@echo off
rem Double-click: like record_incident_clips.bat, but only in the city (30 Oct, evening): four overview cameras of
rem their own (tools\record_city_cams.json), each 9 m up over a road 20-40 m from a junction and looking back at it,
rem so the stop lines and the junction itself are in the picture, where the town's crashes happen (1292, 838, 1722,
rem 1574: half of the city crashes of tools\check_accidents.py within 60 m of one of them). No camera on the ring road,
rem none started after a crash elsewhere. The crash clip: a crash at most 60 m from the camera, 45-60 s of ordinary
rem traffic before it, 3 min after. Up to 50 minutes. Results: out\incident_clips\<time>\ (clips.json).
set FREEFLY_ARGS=--demo 1 --record-clips 3000 --record-specs tools/record_city_cams.json --record-dynamic 0 --careless-share 0.5 --seed %RANDOM% --clock-speed 1 --set-time 16:00 --no-cameras --clock-file out/freefly/clips_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat"
