#!/usr/bin/env python3
"""Autonomous chain to the FIRST gold-58 number:
1) wait for preprocess kernel COMPLETE  2) download shards to laptop
3) scp shards -> box  4) run box_train.py on the box V100  5) print gold-58 AUC.
Runs on the laptop; needs the laptop online throughout (~1.5-2h)."""
import subprocess, re, time, glob, os, pathlib

KID = "eeshsaxena/rsna-knee-preprocess"
BOX = "training@192.168.10.30"
PY = "/home/training/miniconda3/envs/rsna/bin/python"
HERE = pathlib.Path(__file__).parent
OUT = HERE / "_out_preprocess_full"


def cli(a):
    p = subprocess.run(["kaggle"] + a, capture_output=True, text=True)
    return "\n".join(l for l in ((p.stdout or "")+(p.stderr or "")).splitlines()
                     if "RequestsDependency" not in l and "warnings.warn" not in l)
def status():
    m = re.search(r'KernelWorkerStatus\.(\w+)', cli(["kernels", "status", KID]))
    return m.group(1) if m else "?"
def ssh(cmd, cap=True):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", BOX, cmd], capture_output=cap, text=True)

# 1) wait for preprocess
for i in range(180):  # ~3h cap
    st = status(); print(f"[{i}] preprocess {st}", flush=True)
    if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"): break
    time.sleep(60)
if status() != "COMPLETE":
    print("preprocess did not COMPLETE -> stop"); raise SystemExit

# 2) download shards
OUT.mkdir(exist_ok=True)
print(cli(["kernels", "output", KID, "-p", str(OUT)])[:200])
shards = sorted(glob.glob(str(OUT/"**"/"shard_*.npz"), recursive=True))
print("downloaded shards:", len(shards), [os.path.basename(s) for s in shards])
if not shards:
    print("NO shards found in output -> stop"); raise SystemExit

# 3) scp shards to box
ssh("mkdir -p ~/rsna_shards && rm -f ~/rsna_shards/shard_*.npz")
for s in shards:
    r = subprocess.run(["scp", "-o", "BatchMode=yes", s, f"{BOX}:rsna_shards/"], capture_output=True, text=True)
    print("  scp", os.path.basename(s), "rc", r.returncode, (r.stderr or "")[-120:])
print("box shards:", ssh("ls -la ~/rsna_shards/").stdout)

# 4) launch box training DETACHED (nohup) so it survives the laptop leaving the LAN
print("=== launching box_train.py DETACHED on the V100 ===", flush=True)
r = ssh("cd ~ && nohup " + PY + " ~/rsna_train/box_train.py --shards ~/rsna_shards "
        "--model ~/dinov2_small --labels ~/rsna_train/pseudo_labels_v1.csv "
        "--lang ~/rsna_train/_reports_lang.csv --train ~/rsna_train/train.csv "
        "> ~/box_train.log 2>&1 </dev/null & echo LAUNCHED_PID:$!")
print(r.stdout.strip(), r.stderr.strip())
print("\n>>> BOX TRAINING LAUNCHED (detached). SAFE TO DISCONNECT / LEAVE THE LAB NOW.")
print(">>> The V100 trains on its own (~40 min) and writes ~/box_train.log with the gold-58 AUC.")
print(">>> On reconnect: ssh training@192.168.10.30 'tail -30 ~/box_train.log'")
print("ORCH_DONE")
