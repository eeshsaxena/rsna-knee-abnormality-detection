# rb_fold_check.py (r7teacher, 10-15 00:15 IST, runbook agent; BOX, CPU, read only): per-fold preflight of a student set BEFORE the
# student gate's 30-40 min CPU inference. One line per check, per fold; last line RB_FOLD_CHECK PASS|FAIL n/m.
# Checks = the union of student_gate.py G0 (file names, closing line, OOF ids, ckpt args incl. A1 FX4) and make_s_pack.py's
# per-fold checks (model_mod / model_sha / amp, recipe keys == the k11w D member, decode_probe, OAI-clean shape), plus: no stale
# gold74 predictions for the run names in any of the gate's GOLD_CACHES (gold_infer5.py and the gate reuse a cached file by
# RUN NAME, not by checkpoint sha), and the rental receipt for folds placed by the rental fetch loop (report only).
# It never writes anything. Prints names / args / counts only (no predictions, no ids).
#   ~/miniconda3/envs/rsna/bin/python rb_fold_check.py [--tpl 'f{F}_s1047_ttr_r7tb'] [--table-name labels_r7t_b.csv]
#        [--folds 0 1 2 3 4] [--runs ~/fleet_runs] [--allow-open]      (--allow-open: a run without its closing line is reported, not failed)
import argparse, glob, json, os, re
import pandas as pd, torch
H = os.path.expanduser('~')
ap = argparse.ArgumentParser()
ap.add_argument('--tpl', default='f{F}_s1047_ttr_r7tb'); ap.add_argument('--table-name', default='labels_r7t_b.csv')
ap.add_argument('--folds', nargs='*', type=int, default=[0, 1, 2, 3, 4]); ap.add_argument('--runs', default=f'{H}/fleet_runs')
ap.add_argument('--allow-open', action='store_true')
A = ap.parse_args()
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
     'Contusion', 'Fracture']
GOLD_CACHES = [f'{H}/novel/k9gold/preds', f'{H}/novel/k8gold/preds', f'{H}/novel/gold74_preds', f'{H}/k10_test/gold/preds',
               f'{H}/k11_test/gold/preds', f'{H}/k14_test/gold/preds', f'{H}/r7t_labels/gold_preds']       # == student_gate.py
KEYS = ('contract', 'views', 'backbone', 'dim', 'n_slices', 'n_views', 'n_contrast', 'store', 'crop_mm', 'src_slices', 'jpeg_fp', 'select_sha')
SEED = int(re.search(r'_s(\d+)_', A.tpl.format(F=0)).group(1))
R = []


def say(F, k, ok, info=''):
    R.append(bool(ok)); print(f'{"OK  " if ok else "FAIL"} f{F} {k} {info}', flush=True)


rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = rd(f'{H}/rsna_train/folds5.csv')['fold']; gs = set(rd(f'{H}/rsna_train/gold74_eval_set.csv').index)
pool = [s for s in folds.index if folds[s] >= 0 and s not in gs]
ref_ck = sorted(glob.glob(f'{H}/k11_test/a0/ds/rsna-knee-cnxt-fleet-k11w/f0_s1043f75_raw_ttr224-*_best.pt'))
ref = torch.load(ref_ck[0], map_location='cpu', weights_only=False)['args']
parity = {f[len('parity_'):-len('.npz')] for f in os.listdir(os.path.dirname(ref_ck[0])) if f.startswith('parity_')}
for F in A.folds:
    run = A.tpl.format(F=F); d = f'{A.runs}/{run}'
    stale = [c for c in GOLD_CACHES if glob.glob(f'{c}/{run}__*')]
    say(F, 'no cached gold74 prediction for this run name yet (else the gate would reuse it)', not stale, stale)
    if not os.path.isdir(d):
        say(F, 'run dir exists', False, d); continue
    tail = open(d + '.log').read().strip().splitlines()[-1] if os.path.isfile(d + '.log') else ''
    closed = bool(re.match(r'^\{"raw": [^,]+, "ema": [^}]+\}$', tail))
    if A.allow_open and not closed:
        print(f'INFO f{F} run not closed yet (last log line: {tail[:70]!r})', flush=True)
    else:
        say(F, 'log closing line {"raw": .., "ema": ..} (gate G0)', closed, tail[:80])
    raw = glob.glob(f'{d}/f{F}_s{SEED}_raw_*_best.pt')
    say(F, 'exactly one top-level raw-best checkpoint', len(raw) == 1, [os.path.basename(p) for p in raw])
    if len(raw) != 1:
        continue
    staged = os.path.basename(raw[0]).replace(f'f{F}_s{SEED}_', f'f{F}_s{SEED}r7tb_', 1)
    if SEED == 1047:
        say(F, 'staged name matches the make_s_pack / build_v224fs S_RE', bool(re.match(r'^f([0-4])_s1047r7tb_raw_ttr224-320x24_cv\d{4}_best\.pt$', staged)), staged)
    oof = f'{d}/oof_f{F}_s{SEED}_raw.csv'
    if os.path.isfile(oof):
        o = rd(oof)
        say(F, 'raw OOF ids == folds5 val ids minus gold74, 12 targets (gate G0)', sorted(o.index) == sorted(s for s in pool if folds[s] == F)
            and all(c in o.columns for c in T), f'{len(o)} rows')
    else:
        say(F, 'raw OOF present', False, oof)
    c = torch.load(raw[0], map_location='cpu', weights_only=False); a = c['args']
    want = dict(fold=F, seed=SEED, epochs=21, ttr_size=224, model_mod='fleet_model_ttr', model_sha='f141aeed136dfc01', amp='fp16',
                train_labels=A.table_name, sel_labels=A.table_name)
    mism = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
    say(F, 'ckpt args (gate G0 + make_s_pack)', not mism, mism or 'all 9 keys as wanted')
    km = [k for k in KEYS if a.get(k) != ref.get(k)]
    say(F, 'recipe keys == the k11w D member', not km, km)
    dp = a.get('decode_probe') or {}
    say(F, 'decode_probe <= 8 RSNA UIDs inside the k11w parity set', set(dp) <= parity and len(dp) <= 8
        and all(re.match(r'^1\.2\.826\.0\.1\.3680043\.', u) for u in dp), f'{len(dp)} keys')
    say(F, 'top-level keys {model, args, cv, epoch}, tensor-only state dict', set(c) == {'model', 'args', 'cv', 'epoch'}
        and all(isinstance(v, torch.Tensor) for v in c['model'].values()), sorted(c))
    txt = json.dumps({k: v for k, v in a.items() if k != 'decode_probe'}, default=str)
    say(F, 'no oai_<digit> string in args', not re.search(r'oai_\d', txt, re.I))
    rr = f'{A.runs}/rstud_{run}_receipt.json'
    if os.path.isfile(rr):
        print(f'INFO f{F} placed by the rental fetch loop: {rr}', flush=True)
print(f'RB_FOLD_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)} | tpl {A.tpl} | folds {A.folds}{" | allow-open" if A.allow_open else ""}')
