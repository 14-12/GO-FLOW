@echo off
rem Builds everything --goflow-prediction / --goflow-traffic need (SUMO only, no CARLA), into out\ (git-ignored):
rem   1. the 55 approaches and their graph            out\goflow_prediction\graph\
rem   2. the dataset (60 runs x 30 min by default)     out\goflow_prediction\datasets\%TAG%\
rem   3. DSO4's model                                  out\goflow_prediction\models\lgbm_real_v2
rem   4. GO-FLOW's model                               out\goflow_traffic\models\xgb_120s_v1
rem   5. GO-FLOW's forecasts on the dataset, DSO4 + GO-FLOW model    ...\dataset_goflow.npz, models\lgbm_gf_v2
rem   6. the live check (both agents, one killed)      out\goflow_prediction\live_check_*\report.json
rem Needs .venv (tools\setup_venv.bat) and both agents' venvs (tools\setup_goflow_prediction.bat,
rem tools\setup_goflow_traffic.bat). About 1 h with 4 jobs. A step whose output exists is skipped (nothing is
rem overwritten). Log: tools\logs\goflow_prediction_build.log
rem Smaller, for a quick try: set GFP_SEEDS=101-104,11-12& set GFP_TRAIN=101-104& set GFP_DEV=11-12& set GFP_TAG=try
call "%~dp0_env.bat"
set L=%LOGS%\goflow_prediction_build.log
if "%GFP_TAG%"=="" set GFP_TAG=D2_x1_coord_v1
if "%GFP_SEEDS%"=="" set GFP_SEEDS=101-140,11-30
if "%GFP_TRAIN%"=="" set GFP_TRAIN=101-140
if "%GFP_DEV%"=="" set GFP_DEV=11-20
if "%GFP_JOBS%"=="" set GFP_JOBS=4
set G=%OUT%\goflow_prediction\graph
set D=%OUT%\goflow_prediction\datasets\%GFP_TAG%
set M=%OUT%\goflow_prediction\models
set GF=%OUT%\goflow_traffic\models\xgb_120s_v1
set PYP=%ROOT%\agents\goflow_prediction\.venv\Scripts\python.exe
set PYT=%ROOT%\agents\goflow_traffic\.venv\Scripts\python.exe
echo [goflow_prediction_build] %DATE% %TIME% tag %GFP_TAG% seeds %GFP_SEEDS% > "%L%"

if not exist "%G%\nodes.json" "%PY%" tools\goflow_prediction_graph.py --out "%G%" >> "%L%" 2>&1
echo 1 graph EXIT %errorlevel% >> "%L%"
if not exist "%D%\dataset.npz" "%PY%" tools\goflow_prediction_dataset.py --seeds %GFP_SEEDS% --minutes 30 --warmup 5 --jobs %GFP_JOBS% --graph "%G%" --out "%D%" >> "%L%" 2>&1
echo 2 dataset EXIT %errorlevel% >> "%L%"

cd /d "%ROOT%\agents\goflow_prediction"
if not exist "%M%\lgbm_real_v2" "%PYP%" -m goflow_prediction.train_lgbm --dataset "%D%\dataset.npz" --graph "%D%\distance.csv" --out "%M%\lgbm_real_v2" --version lgbm_real_v2 --train-seeds %GFP_TRAIN% --dev-seeds %GFP_DEV% >> "%L%" 2>&1
echo 3 DSO4 model EXIT %errorlevel% >> "%L%"

cd /d "%ROOT%\agents\goflow_traffic"
if not exist "%GF%" "%PYT%" -m goflow_traffic.train --dataset "%D%\dataset.npz" --graph "%D%\distance.csv" --out "%GF%" --version xgb_120s_v1 --train-seeds %GFP_TRAIN% --dev-seeds %GFP_DEV% >> "%L%" 2>&1
echo 4 GO-FLOW model EXIT %errorlevel% >> "%L%"
if not exist "%D%\dataset_goflow.npz" "%PYT%" -m goflow_traffic.export --dataset "%D%\dataset.npz" --graph "%D%\distance.csv" --model "%GF%" --train-seeds %GFP_TRAIN% --dev-seeds %GFP_DEV% --out "%D%\dataset_goflow.npz" >> "%L%" 2>&1
echo 5a GO-FLOW forecasts EXIT %errorlevel% >> "%L%"

cd /d "%ROOT%\agents\goflow_prediction"
if not exist "%M%\lgbm_gf_v2" "%PYP%" -m goflow_prediction.train_lgbm --dataset "%D%\dataset_goflow.npz" --graph "%D%\distance.csv" --out "%M%\lgbm_gf_v2" --version lgbm_gf_v2 --goflow --train-seeds %GFP_TRAIN% --dev-seeds %GFP_DEV% >> "%L%" 2>&1
echo 5b DSO4 + GO-FLOW model EXIT %errorlevel% >> "%L%"

cd /d "%ROOT%"
"%PY%" tools\goflow_prediction_live_check.py --minutes 20 --kill-at 10 --goflow >> "%L%" 2>&1
echo 6 live check EXIT %errorlevel% >> "%L%"
type "%L%" | findstr /C:"EXIT" /C:"[ok]" /C:"identical" /C:"MAE"
echo Full log: %L%
pause
