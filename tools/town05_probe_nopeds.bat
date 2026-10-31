@echo off
rem probe: the demo without pedestrians (frame-rate cost of the walkers)
set FREEFLY_ARGS=--pedestrians-per-km 0
call "%~dp0town05_freefly_demo.bat"
