@echo off
rem Double-click (30 Oct, evening): the recording the GO-FLOW stage-1 clips still need. The fourth ring-road camera
rem ("2328-1 looking E") and the four city overview cameras of tools\record_city_cams.json, until a crash in the city
rem at most 60 m from one of the overview cameras (45-60 s of ordinary traffic before it, 3 min after), or 50 minutes.
rem Nothing on the ring road gives the crash clip. Results: out\incident_clips\<time>\ (clips.json, the full
rem recordings, incidents.json).
set FREEFLY_ARGS=--demo 1 --record-clips 3000 --record-cams "2328-1 looking E" --record-specs tools/record_city_cams.json --record-crash-match overview --record-dynamic 0 --careless-share 0.5 --seed %RANDOM% --clock-speed 1 --set-time 16:00 --no-cameras --clock-file out/freefly/clips_clock.json %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat"
