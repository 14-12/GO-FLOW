@echo off
rem The police office (tools\police_office.py): review the red-light cases of out\enforcement\cases\, accept (a notice
rem to the registered keeper, FICTIONAL) or reject each. Can stay open while the town runs: new cases appear.
call "%~dp0_env.bat"
start "" "%PY:python.exe=pythonw.exe%" tools\police_office.py
