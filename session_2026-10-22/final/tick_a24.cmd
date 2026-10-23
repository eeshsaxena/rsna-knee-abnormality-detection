@echo off
rem tick_a24.cmd (10-22, Claude, private-entries track): wrapper for ONE tick of session_2026-10-22\final\tick_a24.py. DRAFT, NOT REGISTERED.
rem Does nothing without final\A24_GO.txt + PREREG AMENDMENT 24. Do NOT add -I: kaggle lives in the user site-packages (import fails under -I).
cd /d C:\Users\eeshs\Downloads\rsna-knee
if not exist session_2026-10-22\final\a24 mkdir session_2026-10-22\final\a24
C:\Python314\python.exe session_2026-10-22\final\tick_a24.py >> session_2026-10-22\final\a24\tick_a24_task.out 2>&1
