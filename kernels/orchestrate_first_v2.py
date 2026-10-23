#!/usr/bin/env python3
"""Autonomous chain to the first gold-58 AUC, emitting _status.json for the live
dashboard: preprocess -> download -> scp -> box embed+train -> AUC."""
import subprocess, re, time, glob, os, json, pathlib

KID = "eeshsaxena/rsna-knee-preprocess"
BOX = "training@192.168.10.30"
PY = "/home/training/miniconda3/envs/rsna/bin/python"
HERE = pathlib.Path(__file__).parent
OUT = HERE / "_out_preprocess_full"
STATUS = HERE / "_status.json"
T0 = time.time()
STAGES = [("preprocess", "Preprocess shards (Kaggle CPU)"),
          ("download", "Download shards to laptop"),
          ("scp", "Copy shards to box (LAN)"),
          ("boxtrain", "Embed + train on V100"),
          ("done", "Gold-58 AUC")]


def write(stage, pct, msg, auc=None):
    order = [s[0] for s in STAGES]
    cur = order.index(stage)
    st = {"updated": time.strftime("%H:%M:%S"), "elapsed": int(time.time()-T0),
          "stage": stage, "stage_pct": pct, "msg": msg, "auc": auc,
          "stages": [{"key": k, "label": l,
                      "state": "done" if i < cur else ("active" if i == cur else "pending")}
                     for i, (k, l) in enumerate(STAGES)]}
    STATUS.write_text(json.dumps(st))


def cli(a):
    p = subprocess.run(["kaggle"] + a, capture_output=True, text=True)
    return "\n".join(l for l in ((p.stdout or "")+(p.stderr or "")).splitlines()
                     if "RequestsDependency" not in l and "warnings.warn" not in l)
def kstatus():
    m = re.search(r'KernelWorkerStatus\.(\w+)', cli(["kernels", "status", KID])); return m.group(1) if m else "?"
def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", BOX, cmd], capture_output=True, text=True)

# 1) preprocess
for i in range(200):
    s = kstatus(); write("preprocess", -1, f"Kaggle kernel {s} ({i} min)")
    if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"): break
    time.sleep(60)
if kstatus() != "COMPLETE":
    write("preprocess", -1, f"stopped: {kstatus()}"); raise SystemExit

# 2) download
write("download", -1, "downloading shards from Kaggle...")
cli(["kernels", "output", KID, "-p", str(OUT)])
shards = sorted(glob.glob(str(OUT/"**"/"shard_*.npz"), recursive=True))
gb = round(sum(os.path.getsize(s) for s in shards)/1e9, 1)
write("download", 100, f"{len(shards)} shards, {gb} GB downloaded")
if not shards: write("download", 100, "NO shards found"); raise SystemExit

# 3) scp
ssh("mkdir -p ~/rsna_shards && rm -f ~/rsna_shards/shard_*.npz")
for j, s in enumerate(shards):
    write("scp", int(100*j/len(shards)), f"copying shard {j+1}/{len(shards)} to box")
    subprocess.run(["scp", "-o", "BatchMode=yes", s, f"{BOX}:rsna_shards/"], capture_output=True)
write("scp", 100, f"{len(shards)} shards on box")

# 4) box train (nohup + poll log)
ssh(f"nohup {PY} ~/rsna_train/box_train.py --shards ~/rsna_shards --model ~/dinov2_small "
    f"--labels ~/rsna_train/pseudo_labels_v1.csv --lang ~/rsna_train/_reports_lang.csv "
    f"--train ~/rsna_train/train.csv > ~/box_train.log 2>&1 & echo started")
auc = None
for i in range(240):  # ~60 min
    log = ssh("cat ~/box_train.log 2>/dev/null").stdout
    em = re.findall(r'embed (\d+)/(\d+)', log)
    folds = len(re.findall(r'gold-58 AUC', log))
    m = re.search(r'macro-AUC:\s*([\d.]+)', log)
    if m: auc = float(m.group(1)); write("done", 100, "complete", auc); break
    if em:
        d, t = map(int, em[-1]); pct = min(99, int(100*d/max(t, 1)))
        write("boxtrain", pct, f"embedding slices {d}/{t}" + (f" | {folds} folds trained" if folds else ""))
    elif folds:
        write("boxtrain", 100, f"training heads: {folds} folds done")
    else:
        write("boxtrain", -1, "loading DINOv2 + embedding on V100...")
    if "Traceback" in log or "Error" in log[-400:]:
        write("boxtrain", -1, "box error - see log"); break
    time.sleep(15)
print("ORCH_DONE auc=", auc)
