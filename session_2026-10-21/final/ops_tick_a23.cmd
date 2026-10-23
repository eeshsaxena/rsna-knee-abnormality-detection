@echo off
rem ops_tick_a23.cmd (10-21, Claude, ops track): wrapper for ONE tick of final/ops_tick_a23.py. DRAFT, NOT REGISTERED.
rem Does nothing without final/A23_GO.txt + PREREG AMENDMENT 23. Do NOT add -I: kaggle lives in the user site-packages (import fails under -I).
cd /d C:\Users\eeshs\Downloads\rsna-knee
if not exist session_2026-10-21\final\a23 mkdir session_2026-10-21\final\a23
C:\Python314\python.exe session_2026-10-21\final\ops_tick_a23.py >> session_2026-10-21\final\a23\tick_a23_task.out 2>&1
