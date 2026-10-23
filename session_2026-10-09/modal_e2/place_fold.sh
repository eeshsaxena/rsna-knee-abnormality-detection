#!/bin/bash
# place_fold.sh F: BOX side, after `modal run e2_app.py::fetch --name fF_s1046_ttr288frac75 --dest ~/modal_e2/dl`.
# Moves ~/modal_e2/dl/fF_s1046_ttr288frac75{,.log} into ~/fleet_runs/ (only NEW paths: refuses if the target dir or
# log already exists), re-checks every md5 against the Modal manifest, checks the closing line, the file counts
# (1 raw + 1 ema best ckpt, 2 top-level oof csvs, 32 epoch oof csvs for epochs 6-21) and the ckpt args
# (seed 1046, fold F, ttr_size 288, epochs 21, frac75 train labels, steven_v4 selection labels, grad_ckpt False,
# model_sha f141aeed136dfc01), then writes ~/fleet_runs/modal_e2_fF_s1046_ttr288frac75_receipt.json.
# CPU only (CUDA_VISIBLE_DEVICES= , nice 19); never touches the GPU, the queue or any existing run.
set -u
F=$1; N=f${F}_s1046_ttr288frac75; SRC=~/modal_e2/dl; FR=~/fleet_runs; D=$FR/$N
PY=~/miniconda3/envs/rsna/bin/python
CLOSE_RE='^\{"raw": [^,]+, "ema": [^}]+\}$'
fail(){ echo "PLACE_FAIL $N: $*"; exit 1; }
[ -d "$SRC/$N" ] && [ -f "$SRC/$N.log" ] && [ -f "$SRC/$N.modal_manifest.json" ] || fail "fetch output missing in $SRC"
[ -e "$D" ] || [ -e "$D.log" ] && fail "$D or $D.log already exists (refusing to touch an existing run)"
tail -n 1 "$SRC/$N.log" | tr -d '\r' | grep -Eq "$CLOSE_RE" || fail "no closing line in the fetched log"
mv -n "$SRC/$N" "$D" && mv -n "$SRC/$N.log" "$D.log" || fail "move into $FR failed"
cp "$SRC/$N.modal_manifest.json" "$D/../.${N}.modal_manifest.json.tmp" && mv -f "$D/../.${N}.modal_manifest.json.tmp" "$FR/modal_e2_${N}_manifest.json"
cd ~ && CUDA_VISIBLE_DEVICES= nice -n 19 "$PY" - "$F" <<'PYEOF'
import sys, os, glob, json, hashlib, re, time, torch
F = int(sys.argv[1]); N = f"f{F}_s1046_ttr288frac75"; FR = os.path.expanduser("~/fleet_runs"); D = f"{FR}/{N}"
man = json.load(open(f"{FR}/modal_e2_{N}_manifest.json")); bad = []
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
for rel, (size, m) in man.items():
    p = f"{FR}/{rel}"
    if not os.path.isfile(p) or os.path.getsize(p) != size or md5(p) != m: bad.append("md5:" + rel)
have = sorted(os.path.relpath(p, FR) for p in glob.glob(f"{D}/**", recursive=True) if os.path.isfile(p)) + [f"{N}.log"]
extra = sorted(set(have) - set(man)); bad += ["extra:" + e for e in extra]
best_raw = glob.glob(f"{D}/f{F}_s1046_raw_*_best.pt"); best_ema = glob.glob(f"{D}/f{F}_s1046_ema_*_best.pt")
top_oof = glob.glob(f"{D}/oof_f{F}_s1046_*.csv"); ep_oof = glob.glob(f"{D}/epochs/oof_*.csv")
if len(best_raw) != 1 or len(best_ema) != 1: bad.append(f"best ckpts raw {len(best_raw)} ema {len(best_ema)}")
if len(top_oof) != 2: bad.append(f"top oof {len(top_oof)}")
if len(ep_oof) != 32: bad.append(f"epoch oof {len(ep_oof)} (want 32 = epochs 6-21 x raw/ema)")
ck = []
for p in sorted(best_raw + best_ema):
    a = torch.load(p, map_location="cpu", weights_only=False); g = a["args"]
    want = dict(fold=F, seed=1046, ttr_size=288, epochs=21, train_labels="labels_frac75_logit.csv",
                sel_labels="llm_labels_v4_blend.csv", grad_ckpt=False, model_sha="f141aeed136dfc01", aug="heavy",
                train_slices=16, drop_path=0.2, dropout=0.3, ema=0.9995, save_epochs="6-21")
    mism = {k: g.get(k) for k, v in want.items() if g.get(k) != v}
    ck.append(dict(file=os.path.basename(p), epoch=a["epoch"] + 1, cv=round(float(a["cv"]), 4), args="OK" if not mism else mism))
    if mism: bad.append(f"ckpt args {os.path.basename(p)} {mism}")
closing = open(f"{D}.log").read().rstrip().splitlines()[-1]
if not re.match(r'^\{"raw": [^,]+, "ema": [^}]+\}$', closing): bad.append("closing line")
rec = dict(run=N, source="modal volume rsna-e2 /runs/" + N, fetched_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           files=len(man), best_raw=len(best_raw), best_ema=len(best_ema), top_oof=len(top_oof), epoch_oof=len(ep_oof),
           closing_line=closing, ckpts=ck, verify="OK" if not bad else bad)
json.dump(rec, open(f"{FR}/modal_e2_{N}_receipt.json", "w"), indent=1)
print(("VERIFY_OK " if not bad else "VERIFY_FAIL ") + N + " | closing " + closing + " | " + " ; ".join(
    f"{c['file']}: ep {c['epoch']} cv {c['cv']} args {c['args']}" for c in ck))
sys.exit(1 if bad else 0)
PYEOF
