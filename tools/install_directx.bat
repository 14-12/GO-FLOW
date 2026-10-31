@echo off
rem Installs the DirectX End-User Runtime (June 2010) that CarlaUE4.exe needs ("DirectX Runtime" error).
rem Source: https://www.microsoft.com/en-us/download/details.aspx?id=8109 (95.6 MB). Log: tools\logs\directx.log
setlocal
call "%~dp0_env.bat"
set LOG=%LOGS%\directx.log
set DX=%ROOT%\.cache\directx
if not exist "%DX%" mkdir "%DX%"
echo [directx] %DATE% %TIME% > "%LOG%"
curl.exe -L --fail --retry 5 -o "%DX%\directx_Jun2010_redist.exe" https://download.microsoft.com/download/8/4/a/84a35bf1-dafe-4ae8-82af-ad2ae20b6b14/directx_Jun2010_redist.exe >> "%LOG%" 2>&1
if errorlevel 1 (echo DOWNLOAD FAILED >> "%LOG%" & pause & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_sig.ps1" "%DX%\directx_Jun2010_redist.exe" >> "%LOG%" 2>&1
if errorlevel 1 (echo SIGNATURE NOT VALID - not installing >> "%LOG%" & echo Signature check failed. & pause & exit /b 1)
echo Extracting... >> "%LOG%"
"%DX%\directx_Jun2010_redist.exe" /Q /T:"%DX%\files"
echo extract exit %errorlevel% >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_sig.ps1" "%DX%\files\DXSETUP.exe" >> "%LOG%" 2>&1
if errorlevel 1 (echo DXSETUP SIGNATURE NOT VALID >> "%LOG%" & pause & exit /b 1)
echo Installing - Windows will ask for permission, click Yes.
"%DX%\files\DXSETUP.exe" /silent
echo DXSETUP exit %errorlevel% >> "%LOG%"
echo DIRECTX DONE >> "%LOG%"
echo Done.
timeout /t 10
