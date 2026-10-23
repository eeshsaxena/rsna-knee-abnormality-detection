#!/usr/bin/env python3
"""Poll a Kaggle kernel to terminal, then fetch + print its log (ascii-safe).
Usage: python watch_kernel.py <owner/kernel-slug> [out_subdir]"""
import subprocess, re, time, sys, pathlib, glob, os, json

KID = sys.argv[1]
OUT = pathlib.Path(__file__).parent / (sys.argv[2] if len(sys.argv) > 2 else ('_out_' + KID.split('/')[-1]))


def cli(a):
    p = subprocess.run(["kaggle"] + a, capture_output=True, text=True)
    o = (p.stdout or "") + (p.stderr or "")
    return "\n".join(l for l in o.splitlines() if "RequestsDependency" not in l and "warnings.warn" not in l)


def status():
    o = cli(["kernels", "status", KID])
    m = re.search(r'KernelWorkerStatus\.(\w+)', o)
    return m.group(1) if m else ("ABSENT" if "404" in o else "UNKNOWN")


for i in range(240):  # ~40 min
    st = status()
    print(f"[{i}] {st}", flush=True)
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
        break
    time.sleep(12)

OUT.mkdir(exist_ok=True)
print(cli(["kernels", "output", KID, "-p", str(OUT)])[:200])
for lg in sorted(glob.glob(str(OUT / "*.log"))):
    txt = open(lg, encoding="utf-8", errors="replace").read()
    try:
        rows = json.loads(txt)
        lines = [f"[{r['stream_name'][:3]}] {r['data'].rstrip()}" for r in rows
                 if r.get('data', '').strip() and not any(s in r['data'] for s in
                 ['Debugger warning', 'frozen modules', 'PYDEVD', 'queued_call', 'MissingIDField', 'validate(nb)', 'NbConvert'])]
        out = "\n".join(lines)
    except Exception:
        out = txt
    (OUT / "clean.txt").write_text(out.encode("ascii", "replace").decode("ascii"), encoding="utf-8")
    print("LOG ->", OUT / "clean.txt", f"({len(txt)} chars)")
print("WATCH_DONE", status())
