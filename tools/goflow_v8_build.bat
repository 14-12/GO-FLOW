@echo off
rem GO-FLOW forecasters v8 (13 Nov): the v7 recipe (tools\goflow_v7_build.bat) on the town with the parking side by
rem route (cosim/purposes.PARK_SIDE "route", on since 13 Nov), then the test datasets on new seeds 431-440 and both
rem v8 and the shipped v7 scored on them (docs/WORKLOG.md, 13 Nov: the bar). SUMO only, no CARLA. Unattended (no pause).
rem Output: out\goflow_prediction\v8\ ; log: tools\logs\goflow_v8_build.log ; jobs: GFP_JOBS (default 10)
call "%~dp0_env.bat"
set L=%LOGS%\goflow_v8_build.log
set V=%OUT%\goflow_prediction\v8
set G=%OUT%\goflow_prediction\graph
set M=%V%\models
set PYP=%ROOT%\agents\goflow_prediction\.venv\Scripts\python.exe
set PYT=%ROOT%\agents\goflow_traffic\.venv\Scripts\python.exe
set TR=101-130
set DV=11-20
if "%GFP_JOBS%"=="" set GFP_JOBS=10
if not exist "%V%" mkdir "%V%"
if not exist "%M%" mkdir "%M%"
echo [goflow v8 build] %DATE% %TIME% jobs %GFP_JOBS% >> "%L%"

if not exist "%G%\distance.csv" "%PY%" tools\goflow_prediction_graph.py --out "%G%" >> "%L%" 2>&1
echo 1 graph EXIT %errorlevel% >> "%L%"

call :ds D8_ao_x1 agents_observed 1 "101-130,11-20"
call :ds D8_ao_x1.15 agents_observed 1.15 "101-130,11-20"
call :ds D8_ao_x1.3 agents_observed 1.3 "101-130,11-20"
call :ds D8_c_x1 coordinated 1 "101-130,11-20"

cd /d "%ROOT%\agents\goflow_prediction"
if not exist "%V%\D8_all.npz" "%PYP%" -m goflow_prediction.data merge "%V%\D8_all.npz" "%V%\D8_ao_x1\dataset.npz" "%V%\D8_ao_x1.15\dataset.npz" "%V%\D8_ao_x1.3\dataset.npz" "%V%\D8_c_x1\dataset.npz" >> "%L%" 2>&1
echo 3a merge EXIT %errorlevel% >> "%L%"
if not exist "%M%\dso4" "%PYP%" -m goflow_prediction.train_lgbm --dataset "%V%\D8_all.npz" --graph "%G%\distance.csv" --out "%M%\dso4" --version lgbm_c30_v8 --train-seeds %TR% --dev-seeds %DV% --test-seeds 431-440 >> "%L%" 2>&1
echo 3b DSO4 EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%\agents\goflow_traffic"
if not exist "%M%\goflow_xgb" "%PYT%" -m goflow_traffic.train --dataset "%V%\D8_all.npz" --graph "%G%\distance.csv" --out "%M%\goflow_xgb" --version xgb_c30_v8 --train-seeds %TR% --dev-seeds %DV% >> "%L%" 2>&1
echo 3c GO-FLOW EXIT %errorlevel% >> "%L%"
if not exist "%V%\D8_all_goflow.npz" "%PYT%" -m goflow_traffic.export --dataset "%V%\D8_all.npz" --graph "%G%\distance.csv" --model "%M%\goflow_xgb" --train-seeds %TR% --dev-seeds %DV% --out "%V%\D8_all_goflow.npz" >> "%L%" 2>&1
echo 3d GO-FLOW forecasts EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%\agents\goflow_prediction"
if not exist "%M%\dso4_goflow" "%PYP%" -m goflow_prediction.train_lgbm --dataset "%V%\D8_all_goflow.npz" --graph "%G%\distance.csv" --out "%M%\dso4_goflow" --version lgbm_gf_c30_v8 --goflow --train-seeds %TR% --dev-seeds %DV% --test-seeds 431-440 >> "%L%" 2>&1
echo 3e DSO4 + GO-FLOW EXIT %errorlevel% >> "%L%"
if not exist "%M%\dso4\congestion.json" "%PYP%" -m goflow_prediction.congestion --dataset "%V%\D8_all.npz" --model "%M%\dso4" --train-seeds %TR% --dev-seeds %DV% >> "%L%" 2>&1
echo 3f long-queue classifier EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%"

call :ds T9_ao_x1 agents_observed 1 431-440
call :ds T9_ao_x1.15 agents_observed 1.15 431-440
call :ds T9_ao_x1.3 agents_observed 1.3 431-440
call :ds T9_c_x1 coordinated 1 431-440
set PYE=%ROOT%\out\eval_venv\Scripts\python.exe
set TD="%V%\T9_ao_x1" "%V%\T9_ao_x1.15" "%V%\T9_ao_x1.3" "%V%\T9_c_x1"
if not exist "%V%\test_v8" "%PYE%" tools\goflow_prediction_test_score.py %TD% --seeds 431-440 --models "%M%" --out "%V%\test_v8" --reason "v8 against v7 on the parking side by route, 13 Nov bar" >> "%L%" 2>&1
echo 4a score v8 EXIT %errorlevel% >> "%L%"
if not exist "%V%\test_v7" "%PYE%" tools\goflow_prediction_test_score.py %TD% --seeds 431-440 --out "%V%\test_v7" --reason "v7 (shipped) on the same T9 runs, 13 Nov bar" >> "%L%" 2>&1
echo 4b score v7 EXIT %errorlevel% >> "%L%"
echo ALL DONE %DATE% %TIME% >> "%L%"
type "%L%" | findstr /C:"EXIT" /C:"DONE" /C:"[ok]"
echo Full log: %L%
exit /b 0

:ds
rem (10 Nov) the seeds come quoted: a comma splits a batch file's arguments, and before this the dev seeds 11-20
rem were left out of every dataset. A folder from a run cut off is finished (--resume), not made again.
set RS=
if exist "%V%\%1" set RS=--resume
cd /d "%ROOT%"
"%PY%" tools\goflow_prediction_dataset.py %RS% --seeds %~4 --minutes 30 --warmup 5 --start-hour 6 --controller %2 --scale %3 --accidents --junction-blocker 0 --trips purpose --park-side route --graph "%G%" --jobs %GFP_JOBS% --port 8930 --out "%V%\%1" >> "%L%" 2>&1
echo 2 dataset %1 EXIT %errorlevel% >> "%L%"
exit /b 0
