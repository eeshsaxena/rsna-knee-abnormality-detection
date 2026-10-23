"""Read-only: pull public notebook sources (kernels pull -m) with backoff on HTTP 429.

Writes into research_1008/A_pulled/<user>_<slug>/. Never pushes or writes to Kaggle.
"""
import os
import subprocess
import sys
import time

K = r"C:/Users/eeshs/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe"
BASE = os.path.dirname(os.path.abspath(__file__))
REFS = [l.strip() for l in open(os.path.join(BASE, "_refs.txt"), encoding="utf-8") if l.strip()]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
log = open(os.path.join(BASE, "_pull.log"), "a", encoding="utf-8")
for ref in REFS:
    out = os.path.join(BASE, ref.replace("/", "_"))
    os.makedirs(out, exist_ok=True)
    if any(f.endswith((".ipynb", ".py")) for f in os.listdir(out)):
        log.write(f"SKIP {ref} (have)\n"); log.flush(); continue
    wait = 30
    for attempt in range(8):
        p = subprocess.run([K, "kernels", "pull", ref, "-p", out, "-m"], capture_output=True, text=True, env=env)
        txt = (p.stdout + p.stderr)
        txt = "\n".join(l for l in txt.splitlines() if "warn" not in l.lower())
        if "429" in txt:
            log.write(f"429 {ref} attempt {attempt} wait {wait}\n"); log.flush()
            time.sleep(wait); wait = min(wait * 2, 300); continue
        log.write(f"DONE {ref} rc={p.returncode} :: {txt.strip()[-200:]}\n"); log.flush()
        break
    time.sleep(8)
log.write("ALLDONE\n"); log.flush()
