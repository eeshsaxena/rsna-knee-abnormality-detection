#!/usr/bin/env python3
"""Stage the BIG (18GB, 12 slices/252px) shard on the box, in parallel with the 9GB chain.
Waits for rsna-knee-preprocess-big COMPLETE -> download -> scp to ~/rsna_shards_big/.
Does NOT train (box GPU only fits one training at a time); training is launched after the
9GB validation number. Detached daemon; writes progress to chain_big.log."""
import subprocess, re, time, glob, os, pathlib

KID = "eeshsaxena/rsna-knee-preprocess-big"
BOX = "training@192.168.10.30"
HERE = pathlib.Path(__file__).parent
OUT = HERE / "_out_preprocess_big"


def cli(a):
    p = subprocess.run(["kaggle"] + a, capture_output=True, text=True)
    return "\n".join(l for l in ((p.stdout or "")+(p.stderr or "")).splitlines()
                     if "RequestsDependency" not in l and "warnings.warn" not in l)
def status():
    m = re.search(r'KernelWorkerStatus\.(\w+)', cli(["kernels", "status", KID]))
    return m.group(1) if m else "?"
def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", BOX, cmd], capture_output=True, text=True)

for i in range(360):  # up to ~6h (big preprocess may be slow)
    st = status(); print(f"[{i}] big preprocess {st}", flush=True)
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"): break
    time.sleep(60)
if status() != "COMPLETE":
    print("big preprocess did not COMPLETE -> stop"); raise SystemExit

OUT.mkdir(exist_ok=True)
print(cli(["kernels", "output", KID, "-p", str(OUT)])[:200])
shards = sorted(glob.glob(str(OUT/"**"/"shard_*.npz"), recursive=True))
print("big shards downloaded:", len(shards), [os.path.basename(s) for s in shards])
if not shards:
    print("NO big shards -> stop"); raise SystemExit

ssh("mkdir -p ~/rsna_shards_big && rm -f ~/rsna_shards_big/shard_*.npz")
for s in shards:
    r = subprocess.run(["scp", "-o", "BatchMode=yes", s, f"{BOX}:rsna_shards_big/"], capture_output=True, text=True)
    print("  scp", os.path.basename(s), "rc", r.returncode, (r.stderr or "")[-100:], flush=True)
print("box big shards:", ssh("ls -la ~/rsna_shards_big/").stdout)
print(">>> BIG SHARD STAGED on box at ~/rsna_shards_big/ — ready to train after the 9GB number.")
print("BIG_STAGED_DONE")
