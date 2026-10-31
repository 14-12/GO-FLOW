@echo off
rem Runs the already downloaded and signature-checked DirectX installer (see install_directx.bat).
setlocal
call "%~dp0_env.bat"
set LOG=%LOGS%\directx.log
echo Windows will ask for permission - click Yes.
"%ROOT%\.cache\directx\files\DXSETUP.exe" /silent
echo DXSETUP (run 2) exit %errorlevel% >> "%LOG%"
echo Done, exit %errorlevel%.
timeout /t 10
