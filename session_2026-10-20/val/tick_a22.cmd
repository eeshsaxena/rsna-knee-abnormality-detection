@echo off
rem tick_a22.cmd (10-20, Claude): wrapper for ONE tick of val/tick_a22.py. NOT REGISTERED. Does nothing without val/A22_GO.txt + PREREG AMENDMENT 22.
cd /d C:\Users\eeshs\Downloads\rsna-knee
if not exist session_2026-10-20\val\a22 mkdir session_2026-10-20\val\a22
C:\Python314\python.exe session_2026-10-20\val\tick_a22.py >> session_2026-10-20\val\a22\tick_a22_task.out 2>&1
