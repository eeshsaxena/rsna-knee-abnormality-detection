#!/bin/sh
cd /c/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/canary/bpick_work
export PYTHONDONTWRITEBYTECODE=1
for a in "tag=c4over c4shift=2" "tag=c4real c4shift=-2"; do echo "== $a"; python -B bpick_mc.py $a > /dev/null 2>&1; echo "rc $?"; done
echo ALLDONE
