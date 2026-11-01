@echo off
rem The traffic-camera detector as ONNX, in out\detector_onnx\ (tools\export_detector_onnx.py). Log: tools\logs\export_onnx.log
call "%~dp0_env.bat"
"%PY%" tools\export_detector_onnx.py > "%LOGS%\export_onnx.log" 2>&1
type "%LOGS%\export_onnx.log"
timeout /t 5
