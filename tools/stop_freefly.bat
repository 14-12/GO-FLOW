@echo off
rem Closes the Town05 free-fly viewer from outside (as Esc in the CARLA window does): ends the viewer and its
rem helpers (stop_cosim.ps1), CARLA, and the console window of the .bat that started it. Nothing else is touched.
call "%~dp0_env.bat"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosim.ps1"
taskkill /f /im CarlaUE4-Win64-Shipping.exe >nul 2>&1
taskkill /f /im CarlaUE4.exe >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -match '(town05_freefly|freefly_junction_live|freefly_[a-z]+_demo)\.bat' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
"%PY%" tools\ini_profile.py remove >nul 2>&1
echo free-fly closed.
timeout /t 3 >nul
