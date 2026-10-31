@echo off
rem Downloads CARLA 0.9.16 for Windows (official link from github.com/carla-simulator/carla/releases/tag/0.9.16)
rem into ..\carla\ and unzips it there. Re-running resumes an interrupted download.
setlocal
set ROOT=%~dp0..
set DEST=%ROOT%\carla
set ZIP=%DEST%\CARLA_0.9.16.zip
set URL=https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip
set LOG=%~dp0download.log
if not exist "%DEST%" mkdir "%DEST%"
echo Downloading %URL% > "%LOG%"
echo Downloading CARLA 0.9.16 (7.81 GB: 7812252810 bytes). Keep this window open.
curl.exe -L --fail --retry 20 --retry-delay 10 --retry-all-errors -C - -o "%ZIP%" "%URL%"
if errorlevel 1 (echo DOWNLOAD FAILED errorlevel %errorlevel% >> "%LOG%" & echo Download failed - run this file again to resume. & pause & exit /b 1)
echo DOWNLOAD OK >> "%LOG%"
echo Unzipping (this takes a while)...
if not exist "%DEST%\CARLA_0.9.16" mkdir "%DEST%\CARLA_0.9.16"
tar -xf "%ZIP%" -C "%DEST%\CARLA_0.9.16"
if errorlevel 1 (echo UNZIP FAILED errorlevel %errorlevel% >> "%LOG%" & echo Unzip failed. & pause & exit /b 1)
echo UNZIP OK >> "%LOG%"
echo Done. CARLA is in %DEST%\CARLA_0.9.16
pause
