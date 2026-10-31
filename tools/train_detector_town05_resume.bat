@echo off
rem Continues an interrupted tools\train_detector_town05.bat run from its last epoch, then evaluates.
call "%~dp0_env.bat"
set PYTHONUNBUFFERED=1
"%PY%" tools\train_detector.py --data out/dataset_town05/dataset.yaml --base carla-yolo26n.pt --name carla-yolo26n-t05 --epochs 40 --patience 12 --batch 4 --resume >> "%LOGS%\train_town05.log" 2>&1
echo train EXIT %errorlevel% %DATE% %TIME% >> "%LOGS%\train_town05.log"
"%PY%" tools\eval_town05_detector.py > "%LOGS%\eval_town05.log" 2>&1
echo eval EXIT %errorlevel% %DATE% %TIME% >> "%LOGS%\eval_town05.log"
