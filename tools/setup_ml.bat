@echo off
rem Adds the detector-training packages to .venv: PyTorch with CUDA (about 2.5 GB), Ultralytics, ONNX Runtime.
rem Log: tools\logs\setup_ml.log
call "%~dp0_env.bat"
set L=%LOGS%\setup_ml.log
echo [setup_ml] %DATE% %TIME% > "%L%"
nvidia-smi >> "%L%" 2>&1
"%PY%" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 >> "%L%" 2>&1
echo torch EXIT %errorlevel% >> "%L%"
"%PY%" -m pip install "ultralytics>=8.4.100" onnx onnxslim onnxruntime >> "%L%" 2>&1
echo ultralytics EXIT %errorlevel% >> "%L%"
"%PY%" -c "import torch, ultralytics, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'ultralytics', ultralytics.__version__, 'numpy', numpy.__version__)" >> "%L%" 2>&1
"%PY%" -m pip freeze > "%LOGS%\pip-freeze-ml.txt"
echo DONE >> "%L%"
