#!/usr/bin/env python3
"""Retry-push rsna-knee-submit-frozen until a GPU slot frees; run to COMPLETE;
verify submission.csv; then submit it to the competition (code-comp -k submit).
Best-effort: needs the laptop online throughout. Restartable."""
import sys, types, re, unicodedata, subprocess, time, pathlib, glob, os
_m = types.ModuleType('slugify')
def _s(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode()
    x = re.sub(r'[^\w\s-]', '', x).strip().lower(); return re.sub(r'[-\s]+', '-', x)
_m.slugify = _s; sys.modules['slugify'] = _m
from kaggle.api.kaggle_api_extended import KaggleApi
API = KaggleApi(); API.authenticate()
KID = "eeshsaxena/rsna-knee-submit-frozen"
COMP = "rsna-knee-abnormality-detection"
HERE = str(pathlib.Path(__file__).parent)

def cli(a):
    p = subprocess.run(["kaggle"] + a, capture_output=True, text=True)
    return "\n".join(l for l in ((p.stdout or "")+(p.stderr or "")).splitlines()
                     if "RequestsDependency" not in l and "warnings.warn" not in l)
def status():
    m = re.search(r'KernelWorkerStatus\.(\w+)', cli(["kernels", "status", KID]))
    return m.group(1) if m else "?"

ver = None
for i in range(80):  # ~2.7h to catch a free slot
    try:
        r = API.kernels_push(HERE)
        err = str(getattr(r, "error", "") or "")
        if "Maximum batch" in err:
            print(f"[{i}] slot full, retry", flush=True); time.sleep(120); continue
        ver = getattr(r, "versionNumber", None)
        time.sleep(8)
        if status() in ("QUEUED", "RUNNING", "COMPLETE"):
            print(f"[{i}] LAUNCHED v{ver}", flush=True); break
        print(f"[{i}] err={err[:60]}", flush=True); time.sleep(60)
    except Exception as e:
        if "Maximum batch" in str(e): print(f"[{i}] slot full(exc)", flush=True); time.sleep(120); continue
        print(f"[{i}] EXC {str(e)[:70]}", flush=True); time.sleep(120)
if ver is None:
    print("never launched (slots stayed full)"); sys.exit(0)

for i in range(120):  # ~2h to run
    st = status(); print(f"run[{i}] {st}", flush=True)
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"): break
    time.sleep(60)
if status() != "COMPLETE":
    print("kernel not COMPLETE -> no submit"); sys.exit(0)

# verify output then submit
OUT = pathlib.Path(HERE) / "_out"; OUT.mkdir(exist_ok=True)
cli(["kernels", "output", KID, "-p", str(OUT)])
csv = OUT / "submission.csv"
if csv.exists() and sum(1 for _ in open(csv)) > 1:
    print("submission.csv OK -> submitting")
    print(cli(["competitions", "submit", COMP, "-k", KID, "-v", str(ver), "-f", "submission.csv",
               "-m", "frozen DINOv2-S + attention-MIL on soft rule-x-Qwen labels (autonomous v1)"]))
else:
    print("no valid submission.csv")
print("DONE", status())
