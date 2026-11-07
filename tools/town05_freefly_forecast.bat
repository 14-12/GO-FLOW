@echo off
rem Town05 free-fly with the GO-FLOW forecasters on (as town05_freefly.bat, plus the forecasts).
rem   DSO4 forecasts every road's queue and warns "LONG QUEUE LIKELY WITHIN 3 MIN"; GO-FLOW labels each road free,
rem   moderate, congested or severe. Bottom right: the forecast panel (roads most at risk, the last alert); on the
rem   roads near you: chips; over a junction: the alert. F10 flies to the latest alert or the road most at risk.
rem   The first forecasts come about 7 minutes after the traffic starts (they need 6 minutes of history).
rem   Traffic x2 (--demand-scale 2), so that queues form; for the town as it is, run with: town05_freefly_forecast.bat normal
rem   Other options can still go in FREEFLY_ARGS; a render profile (medium / epic) as with town05_freefly.bat.
setlocal
set GF=--goflow-prediction --goflow-traffic --goflow-hud
set PROFILE=%~1
if /i "%~1"=="normal" (set PROFILE=%~2) else (set GF=%GF% --demand-scale 2)
set FREEFLY_ARGS=%GF% %FREEFLY_ARGS%
call "%~dp0town05_freefly.bat" %PROFILE%
