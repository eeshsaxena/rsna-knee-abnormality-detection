#!/usr/bin/env python3
"""Push rsna-knee-qwen-extract as soon as a Kaggle GPU slot frees (non-destructive:
waits for biohub det096 etc. to finish rather than cancelling). Then watch to COMPLETE
and dump the log (gold-58 macro AUC)."""
import sys, types, re, unicodedata, subprocess, time, pathlib, glob, os

_m = types.ModuleType('slugify')
def _s(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^\w\s-]', '', s).strip().lower(); return re.sub(r'[-\s]+', '-', s)
_m.slugify = _s; sys.modules['slugify'] = _m
from kaggle.api.kaggle_api_extended import KaggleApi
API = KaggleApi(); API.authenticate()

KID = "eeshsaxena/rsna-knee-qwen-extract"
HERE = pathlib.Path(__file__).parent
OUT = HERE / "_out"


def cli(a):
    p = subprocess.run(["kaggle"] + a, capture_output=True, text=True)
    o = (p.stdout or "") + (p.stderr or "")
    return p.returncode, "\n".join(l for l in o.splitlines()
                                   if "RequestsDependency" not in l and "warnings.warn" not in l)


def status():
    _, o = cli(["kernels", "status", KID])
    m = re.search(r'KernelWorkerStatus\.(\w+)', o)
    return m.group(1) if m else ("ABSENT" if "404" in o else "UNKNOWN")


# 1) push, retrying while GPU slots are full
pushed = False
for i in range(90):  # up to ~3h waiting for a slot
    try:
        r = API.kernels_push(str(HERE))
        err = str(getattr(r, "error", "") or "")
        if "Maximum batch" in err:
            print(f"[{i}] slot full, retry in 120s", flush=True); time.sleep(120); continue
        time.sleep(6)
        if status() in ("QUEUED", "RUNNING", "COMPLETE"):
            print(f"[{i}] PUSHED", flush=True); pushed = True; break
        print(f"[{i}] push err={err[:80]} status={status()}", flush=True)
    except Exception as e:
        if "Maximum batch" in str(e):
            print(f"[{i}] slot full (exc), retry", flush=True); time.sleep(120); continue
        print(f"[{i}] exc {type(e).__name__}: {str(e)[:100]}", flush=True); time.sleep(120)
if not pushed:
    print("could not push within window"); sys.exit(0)

# 2) watch to terminal
for i in range(180):  # ~1h at 20s (7B on 58 reports is quick once running)
    st = status()
    print(f"watch[{i}] {st}", flush=True)
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
        break
    time.sleep(20)

# 3) dump log
OUT.mkdir(exist_ok=True)
cli(["kernels", "output", KID, "-p", str(OUT)])
for lg in sorted(glob.glob(str(OUT / "*.log"))):
    txt = open(lg, encoding="utf-8", errors="replace").read()
    # write a clean ascii copy to avoid console encoding crashes
    (OUT / "clean.txt").write_text(txt.encode("ascii", "replace").decode("ascii"), encoding="utf-8")
    print("LOG saved ->", OUT / "clean.txt", f"({len(txt)} chars)")
print("DONE")
