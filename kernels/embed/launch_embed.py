#!/usr/bin/env python3
"""Push rsna-knee-embed (full pass) as soon as a GPU slot frees; then confirm RUNNING.
Does NOT watch to completion (the pass is ~3.7h) - just gets it launched robustly."""
import sys, types, re, unicodedata, subprocess, time, pathlib
_m = types.ModuleType('slugify')
def _s(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode()
    x = re.sub(r'[^\w\s-]', '', x).strip().lower(); return re.sub(r'[-\s]+', '-', x)
_m.slugify = _s; sys.modules['slugify'] = _m
from kaggle.api.kaggle_api_extended import KaggleApi
API = KaggleApi(); API.authenticate()
KID = "eeshsaxena/rsna-knee-embed"
HERE = str(pathlib.Path(__file__).parent)

def status():
    p = subprocess.run(["kaggle", "kernels", "status", KID], capture_output=True, text=True)
    m = re.search(r'KernelWorkerStatus\.(\w+)', (p.stdout or "") + (p.stderr or ""))
    return m.group(1) if m else "?"

for i in range(120):  # up to ~4h waiting
    try:
        r = API.kernels_push(HERE)
        err = str(getattr(r, "error", "") or "")
        if "Maximum batch" in err:
            print(f"[{i}] slot full, retry 120s", flush=True); time.sleep(120); continue
        time.sleep(8)
        st = status()
        if st in ("QUEUED", "RUNNING", "COMPLETE"):
            print(f"[{i}] LAUNCHED ({st})", flush=True); break
        print(f"[{i}] err={err[:70]} status={st}", flush=True); time.sleep(60)
    except Exception as e:
        if "Maximum batch" in str(e):
            print(f"[{i}] slot full (exc), retry", flush=True); time.sleep(120); continue
        print(f"[{i}] EXC {type(e).__name__}:{str(e)[:80]}", flush=True); time.sleep(120)
print("LAUNCH_DONE", status())
