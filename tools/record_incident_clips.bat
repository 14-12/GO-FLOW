@echo off
rem Double-click: two labelled MP4 clips from Town05's fixed traffic cameras for the GO-FLOW incident agent
rem (cosim/clip_recorder.py, 30 Oct): normal_traffic.mp4 (4 min, no accident in the picture) and crash.mp4 (1 min
rem before a crash to 3 min after it). The viewer runs on its own: four cameras where crashes happen most
rem (tools/check_accidents.py, 38 crashes in 9.5 h), more careless drivers than usual (50 % instead of 10 %: the
rem crashes are still real collisions, only more frequent), the clock at real time from 16:00 so the light stays.
rem A new random seed each run (in clips.json). Up to 50 minutes; it stops by itself once it has both clips. Results: out\incident_clips\<time>\ (clips.json:
rem camera ids, simulation start times, the crash's second in the clip). Log: tools\logs\freefly_town05.log
set FREEFLY_ARGS=--demo 1 --record-clips 3000 --record-cams "1126-1 looking E;838-2 looking N;979-4 looking W;1427-3 looking S" --careless-share 0.5 --seed %RANDOM% --clock-speed 1 --set-time 16:00 --no-cameras --clock-file out/freefly/clips_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat"
