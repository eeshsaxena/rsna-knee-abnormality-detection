#!/usr/bin/env python3
"""Poll the MINI preprocess (~7min budget); when COMPLETE, download the small shard,
scp to box, and nohup-launch box_train on it -> fast validation gold-58 number.
Re-runnable: if mini not done in the window, exits; just launch again."""
import subprocess, re, time, glob, os, pathlib
KID = "eeshsaxena/rsna-knee-preprocess-mini"
BOX = "training@192.168.10.30"
PY = "/home/training/miniconda3/envs/rsna/bin/python"
HERE = pathlib.Path(__file__).parent
OUT = HERE / "_out_mini"

def cli(a):
    p = subprocess.run(["kaggle"] + a, capture_output=True, text=True)
    return "\n".join(l for l in ((p.stdout or "")+(p.stderr or "")).splitlines()
                     if "RequestsDependency" not in l and "warnings.warn" not in l)
def status():
    m = re.search(r'KernelWorkerStatus\.(\w+)', cli(["kernels", "status", KID])); return m.group(1) if m else "?"
def ssh(c): return subprocess.run(["ssh","-o","BatchMode=yes",BOX,c], capture_output=True, text=True)

st = "?"
for i in range(13):  # ~6.5 min
    st = status(); print(f"[{i}] mini {st}", flush=True)
    if st in ("COMPLETE","ERROR","CANCEL_ACKNOWLEDGED","CANCEL_REQUESTED"): break
    time.sleep(30)
if st != "COMPLETE":
    print("mini not done yet -> re-launch this script"); raise SystemExit

OUT.mkdir(exist_ok=True)
print(cli(["kernels","output",KID,"-p",str(OUT)])[:150])
shards = sorted(glob.glob(str(OUT/"**"/"shard_*.npz"), recursive=True))
print("mini shards:", [os.path.basename(s) for s in shards])
if not shards: print("NO mini shards -> stop"); raise SystemExit
ssh("mkdir -p ~/rsna_shards_mini && rm -f ~/rsna_shards_mini/*.npz")
for s in shards:
    r = subprocess.run(["scp","-o","BatchMode=yes",s,f"{BOX}:rsna_shards_mini/"], capture_output=True, text=True)
    print("  scp", os.path.basename(s), "rc", r.returncode)
r = ssh(f"cd ~ && nohup {PY} ~/rsna_train/box_train.py --shards ~/rsna_shards_mini --model ~/dinov2_small "
        f"--labels ~/rsna_train/pseudo_labels_v1.csv --lang ~/rsna_train/_reports_lang.csv "
        f"--train ~/rsna_train/train.csv --folds 4 --seeds 1 > ~/box_train_mini.log 2>&1 </dev/null & echo PID:$!")
print("box_train launched:", r.stdout.strip())
print(">>> MINI VALIDATION TRAINING LAUNCHED on the V100. Result -> ~/box_train_mini.log")
print("MINI_CHAIN_DONE")
