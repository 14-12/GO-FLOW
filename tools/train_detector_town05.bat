@echo off
rem Fine-tunes the traffic-camera detector on the Town05 set (out\dataset_town05, tools\make_dataset_town05.bat),
rem starting from models\carla-yolo26n.pt, then compares old and new on the held-out Town05 junctions.
rem   -> models\carla-yolo26n-t05.pt/.onnx/.json, out\detector\eval_town05.md   (logs: tools\logs\train_town05.log)
call "%~dp0_env.bat"
set PYTHONUNBUFFERED=1
"%PY%" tools\train_detector.py --data out/dataset_town05/dataset.yaml --base carla-yolo26n.pt --name carla-yolo26n-t05 --epochs 40 --patience 12 --batch 4 > "%LOGS%\train_town05.log" 2>&1
echo train EXIT %errorlevel% %DATE% %TIME% >> "%LOGS%\train_town05.log"
"%PY%" tools\eval_town05_detector.py > "%LOGS%\eval_town05.log" 2>&1
echo eval EXIT %errorlevel% %DATE% %TIME% >> "%LOGS%\eval_town05.log"
