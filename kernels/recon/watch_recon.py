#!/usr/bin/env python3
"""Poll rsna-knee-recon; when terminal, fetch output+logs and print them."""
import subprocess, re, time, pathlib, glob, os

KID = "eeshsaxena/rsna-knee-recon"
HERE = pathlib.Path(__file__).parent
OUT = HERE / "_out"


def cli(args):
    p = subprocess.run(["kaggle"] + args, capture_output=True, text=True)
    o = (p.stdout or "") + (p.stderr or "")
    return p.returncode, "\n".join(l for l in o.splitlines()
                                   if "RequestsDependency" not in l and "warnings.warn" not in l)


def status():
    _, o = cli(["kernels", "status", KID])
    m = re.search(r'KernelWorkerStatus\.(\w+)', o)
    return m.group(1) if m else ("ABSENT" if "404" in o else "UNKNOWN")


for i in range(150):  # ~30 min cap
    st = status()
    print(f"[{i}] {st}", flush=True)
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
        break
    time.sleep(12)

OUT.mkdir(exist_ok=True)
rc, o = cli(["kernels", "output", KID, "-p", str(OUT)])
print("OUTPUT FETCH:", o[:300])
# print any .log file (holds printed stdout for notebook kernels)
for lg in sorted(glob.glob(str(OUT / "*.log"))):
    print(f"\n===== LOG: {os.path.basename(lg)} =====")
    txt = open(lg, encoding="utf-8", errors="replace").read()
    print(txt[-12000:])
print("\nFILES IN OUTPUT:", [os.path.basename(f) for f in glob.glob(str(OUT / '*'))])
