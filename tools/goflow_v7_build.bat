@echo off
rem GO-FLOW forecasters v7 (10 Nov): their training data on the town as it is since 9-10 Nov (the skipped greens fixed,
rem trips at the day's rate with nobody taken off, the town's own traffic = 0.45 of the old curve, the junction agents
rem by default, trips with a purpose, accidents rerouted), then the v6 recipe. SUMO only, no CARLA. Double-click and
rem leave it: several hours. Resumable: a dataset with its dataset.npz is kept, one cut off is made again; a model
rem folder that exists is kept. Nothing in models\goflow is touched (the next session ships v7 after testing it).
rem   1. graph                               out\goflow_prediction\graph
rem   2. datasets, seeds 101-130 (train) and 11-20 (dev), a day at clock 30 from 06:00:
rem        D7_ao_x1, D7_ao_x1.15, D7_ao_x1.3 (junction agents on the cameras), D7_c_x1 (coordinated)
rem   3. merged D7_all.npz; DSO4 (LightGBM); GO-FLOW (XGBoost); GO-FLOW's forecasts; DSO4 + GO-FLOW; long-queue classifier
rem   4. the same four on test seeds 81-90 (T7_*): made, never scored here (tools\goflow_prediction_test_score.py
rem      opens them once, in the next session)
rem Output: out\goflow_prediction\v7\ ; log: tools\logs\goflow_v7_build.log ; jobs: set GFP_JOBS=N first (default 4)
call "%~dp0_env.bat"
set L=%LOGS%\goflow_v7_build.log
set V=%OUT%\goflow_prediction\v7
set G=%OUT%\goflow_prediction\graph
set M=%V%\models
set PYP=%ROOT%\agents\goflow_prediction\.venv\Scripts\python.exe
set PYT=%ROOT%\agents\goflow_traffic\.venv\Scripts\python.exe
set TR=101-130
set DV=11-20
if "%GFP_JOBS%"=="" set GFP_JOBS=4
if not exist "%V%" mkdir "%V%"
if not exist "%M%" mkdir "%M%"
echo [goflow v7 build] %DATE% %TIME% jobs %GFP_JOBS% >> "%L%"

if not exist "%G%\distance.csv" "%PY%" tools\goflow_prediction_graph.py --out "%G%" >> "%L%" 2>&1
echo 1 graph EXIT %errorlevel% >> "%L%"

call :ds D7_ao_x1 agents_observed 1 101-130,11-20
call :ds D7_ao_x1.15 agents_observed 1.15 101-130,11-20
call :ds D7_ao_x1.3 agents_observed 1.3 101-130,11-20
call :ds D7_c_x1 coordinated 1 101-130,11-20

cd /d "%ROOT%\agents\goflow_prediction"
if not exist "%V%\D7_all.npz" "%PYP%" -m goflow_prediction.data merge "%V%\D7_all.npz" "%V%\D7_ao_x1\dataset.npz" "%V%\D7_ao_x1.15\dataset.npz" "%V%\D7_ao_x1.3\dataset.npz" "%V%\D7_c_x1\dataset.npz" >> "%L%" 2>&1
echo 3a merge EXIT %errorlevel% >> "%L%"
if not exist "%M%\dso4" "%PYP%" -m goflow_prediction.train_lgbm --dataset "%V%\D7_all.npz" --graph "%G%\distance.csv" --out "%M%\dso4" --version lgbm_c30_v7 --train-seeds %TR% --dev-seeds %DV% --test-seeds 81-90 >> "%L%" 2>&1
echo 3b DSO4 EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%\agents\goflow_traffic"
if not exist "%M%\goflow_xgb" "%PYT%" -m goflow_traffic.train --dataset "%V%\D7_all.npz" --graph "%G%\distance.csv" --out "%M%\goflow_xgb" --version xgb_c30_v7 --train-seeds %TR% --dev-seeds %DV% >> "%L%" 2>&1
echo 3c GO-FLOW EXIT %errorlevel% >> "%L%"
if not exist "%V%\D7_all_goflow.npz" "%PYT%" -m goflow_traffic.export --dataset "%V%\D7_all.npz" --graph "%G%\distance.csv" --model "%M%\goflow_xgb" --train-seeds %TR% --dev-seeds %DV% --out "%V%\D7_all_goflow.npz" >> "%L%" 2>&1
echo 3d GO-FLOW forecasts EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%\agents\goflow_prediction"
if not exist "%M%\dso4_goflow" "%PYP%" -m goflow_prediction.train_lgbm --dataset "%V%\D7_all_goflow.npz" --graph "%G%\distance.csv" --out "%M%\dso4_goflow" --version lgbm_gf_c30_v7 --goflow --train-seeds %TR% --dev-seeds %DV% --test-seeds 81-90 >> "%L%" 2>&1
echo 3e DSO4 + GO-FLOW EXIT %errorlevel% >> "%L%"
if not exist "%M%\dso4\congestion.json" "%PYP%" -m goflow_prediction.congestion --dataset "%V%\D7_all.npz" --model "%M%\dso4" --train-seeds %TR% --dev-seeds %DV% >> "%L%" 2>&1
echo 3f long-queue classifier EXIT %errorlevel% >> "%L%"
cd /d "%ROOT%"

call :ds T7_ao_x1 agents_observed 1 81-90
call :ds T7_ao_x1.15 agents_observed 1.15 81-90
call :ds T7_ao_x1.3 agents_observed 1.3 81-90
call :ds T7_c_x1 coordinated 1 81-90
echo ALL DONE %DATE% %TIME% >> "%L%"
type "%L%" | findstr /C:"EXIT" /C:"DONE" /C:"[ok]"
echo Full log: %L%
pause
exit /b 0

:ds
if exist "%V%\%1\dataset.npz" exit /b 0
if exist "%V%\%1" rmdir /s /q "%V%\%1"
cd /d "%ROOT%"
"%PY%" tools\goflow_prediction_dataset.py --seeds %4 --minutes 30 --warmup 5 --start-hour 6 --controller %2 --scale %3 --accidents --junction-blocker 0 --trips purpose --graph "%G%" --jobs %GFP_JOBS% --port 8930 --out "%V%\%1" >> "%L%" 2>&1
echo 2 dataset %1 EXIT %errorlevel% >> "%L%"
exit /b 0
