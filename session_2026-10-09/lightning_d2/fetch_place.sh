#!/bin/bash
# fetch_place.sh F [JOB]: BOX side, after the Lightning job ended. Downloads the job artifacts
# (teamspace drive jobs/<JOB>/ld2_out/fleet_runs/) into ~/lightning_d2/dl/, checks every md5 against the in-job
# manifest, the closing line, the file counts (1 raw + 1 ema best ckpt, 2 top-level oof csvs, 32 epoch oof csvs for
# epochs 6-21) and the ckpt args (fold F, seed 1044, ttr 224, epochs 21, save_epochs 6-21, frac75 train labels,
# steven_v4 selection labels, grad_ckpt False, model_sha f141aeed136dfc01, 16 train slices; every other arg identical
# to the Kaggle f0 s1044 ckpt, as in the Vast kit's verify), then moves the run into ~/fleet_runs/fF_s1044_ttr_frac75
# (only into an EMPTY placeholder or a new path) and writes ~/fleet_runs/lightning_d2_fF_s1044_ttr_frac75_receipt.json.
# CPU only (CUDA_VISIBLE_DEVICES= , nice 19); never touches the GPU, the queue or any other run.
set -u
F=$1; JOB=${2:-rsna-d2-f3-s1044}; N=f${F}_s1044_ttr_frac75
E=~/lightning_d2; DL=$E/dl/$JOB; FR=~/fleet_runs; D=$FR/$N; PY=~/miniconda3/envs/rsna/bin/python
fail(){ echo "PLACE_FAIL $N: $*"; exit 1; }
mkdir -p $DL
~/lightenv/bin/python - "$JOB" "$DL" <<'EOF' || fail "download failed"
import sys, os
os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")
import lightning_sdk as L
job, dl = sys.argv[1], sys.argv[2]
t = L.Teamspace(name="default-project", user="eeshsaxena")
t.download_folder("jobs/%s/ld2_out" % job, dl)
print("DOWNLOADED jobs/%s/ld2_out -> %s" % (job, dl))
EOF
A=$DL/fleet_runs
[ -f $A/$N.manifest.txt ] && [ -f $A/$N.log ] && [ -d $A/$N ] || fail "artifacts incomplete in $A"
[ -f $A/$N.DONE ] || fail "no $N.DONE marker (FAILED/ABORTED: $(cat $A/$N.FAILED 2>/dev/null))"
(cd $A && md5sum -c --quiet $N.manifest.txt) || fail "md5 of the downloaded files != in-job manifest"
tail -n 1 $A/$N.log | tr -d '\r' | grep -Eq '^\{"raw": [^,]+, "ema": [^}]+\}$' || fail "no closing line"
if [ -e $D ]; then [ -d $D ] && [ -z "$(ls -A $D)" ] || fail "$D exists and is not an EMPTY placeholder"; rmdir $D; fi
[ -e $D.log ] && fail "$D.log already exists"
mv -n $A/$N $D && mv -n $A/$N.log $D.log || fail "move into $FR failed"
cp $A/$N.manifest.txt $FR/lightning_d2_${N}_manifest.txt
cd ~ && CUDA_VISIBLE_DEVICES= nice -n 19 $PY - "$F" "$JOB" <<'PYEOF'
import sys, os, glob, json, hashlib, re, time, torch
F, JOB = int(sys.argv[1]), sys.argv[2]; N = f"f{F}_s1044_ttr_frac75"; FR = os.path.expanduser("~/fleet_runs"); D = f"{FR}/{N}"
man = {}
for ln in open(f"{FR}/lightning_d2_{N}_manifest.txt"):
    h, p = ln.split(None, 1); man[p.strip()] = h
bad = []
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
for rel, m in man.items():
    p = f"{FR}/{rel}"
    if not os.path.isfile(p) or md5(p) != m: bad.append("md5:" + rel)
have = sorted(os.path.relpath(p, FR) for p in glob.glob(f"{D}/**", recursive=True) if os.path.isfile(p)) + [f"{N}.log"]
bad += ["extra:" + e for e in sorted(set(have) - set(man))]
best_raw = glob.glob(f"{D}/f{F}_s1044_raw_*_best.pt"); best_ema = glob.glob(f"{D}/f{F}_s1044_ema_*_best.pt")
top_oof = glob.glob(f"{D}/oof_f{F}_s1044_*.csv"); ep_oof = glob.glob(f"{D}/epochs/oof_*.csv")
if len(best_raw) != 1 or len(best_ema) != 1: bad.append(f"best ckpts raw {len(best_raw)} ema {len(best_ema)}")
if len(top_oof) != 2: bad.append(f"top oof {len(top_oof)}")
if len(ep_oof) != 32: bad.append(f"epoch oof {len(ep_oof)} (want 32 = epochs 6-21 x raw/ema)")
refp = sorted(glob.glob(f"{FR}/f0_s1044_ttr_frac75/f0_s1044_raw_*_best.pt"))
ref = torch.load(refp[0], map_location="cpu", weights_only=False)["args"] if refp else None
SKIP = {"fold", "epochs", "save_epochs", "grad_ckpt"}          # the only intended differences vs the Kaggle fold-0 run
ck = []
for p in sorted(best_raw + best_ema):
    a = torch.load(p, map_location="cpu", weights_only=False); g = a["args"]
    want = dict(fold=F, seed=1044, ttr_size=224, epochs=21, save_epochs="6-21", train_labels="labels_frac75_logit.csv",
                sel_labels="llm_labels_v4_blend.csv", grad_ckpt=False, model_sha="f141aeed136dfc01", train_slices=16)
    mism = {k: g.get(k) for k, v in want.items() if g.get(k) != v}
    if ref is None: mism["kaggle_ref"] = "missing"
    else:
        diff = [k for k in sorted((set(g) & set(ref)) - SKIP) if g[k] != ref[k]]; only = sorted((set(g) ^ set(ref)) - SKIP)
        if diff: mism["diff_vs_kaggle_f0"] = diff
        if only: mism["one_sided_vs_kaggle_f0"] = only
    ck.append(dict(file=os.path.basename(p), epoch=a["epoch"] + 1, cv=round(float(a["cv"]), 4), args="OK" if not mism else mism))
    if mism: bad.append(f"ckpt args {os.path.basename(p)} {mism}")
L = open(f"{D}.log").read().rstrip().splitlines()
closing = L[-1]
if not re.match(r'^\{"raw": [^,]+, "ema": [^}]+\}$', closing): bad.append("closing line")
raw = [(int(m.group(1)), float(m.group(3)), float(m.group(2))) for m in
       (re.match(r"ep (\d+) raw \|.*gold58 ([0-9.]+) .*val AUC vs \S+ ([0-9.]+)", l) for l in L) if m]
best = max(raw, key=lambda r: r[1]) if raw else None
rec = dict(run=N, source="lightning job %s (Nebius L40S, studio rsna-d2-nb) artifacts ld2_out/fleet_runs/%s" % (JOB, N),
           fetched_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), files=len(man), best_raw=len(best_raw), best_ema=len(best_ema),
           top_oof=len(top_oof), epoch_oof=len(ep_oof), closing_line=closing, ckpts=ck,
           best_raw_val=dict(epoch=best[0] + 1, val_steven_v4=best[1], gold58=best[2]) if best else None,
           verify="OK" if not bad else bad)
json.dump(rec, open(f"{FR}/lightning_d2_{N}_receipt.json", "w"), indent=1)
print(("VERIFY_OK " if not bad else "VERIFY_FAIL ") + N + " | closing " + closing + " | best raw val %s" % rec["best_raw_val"] + " | " +
      " ; ".join(f"{c['file']}: ep {c['epoch']} cv {c['cv']} args {c['args']}" for c in ck))
sys.exit(1 if bad else 0)
PYEOF
