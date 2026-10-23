# score_frac75.py (k8/frac75, 2026-10-07): CPU gate scorer for the frac75 Fracture-target probe (NOVEL_PLAN_10-07 sec 2
# item 4, pre-registered in ~/novel/frac75/PREREG.md). Reads the FINISHED probe run, prints G1-G3 and PASS/FAIL.
# Read-only on ~/fleet_runs, ~/rsna_train, ~/jump2; writes only ~/novel/frac75/{gold74_preds,out}/. Never uses the GPU.
#   G1 no harm   : mean OOF AUC vs steven_v4 on the 11 NON-Fracture labels, best raw checkpoint   >= .8774 (J1 .8804 - .003)
#   G2 mechanism : gold74 Fracture AUC, NaN expert cells MASKED, best raw checkpoint, pass A       >= .912  (J1 .894 + .018)
#   G3 guard     : gold74 12-label macro AUC, masked, same predictions                            >= .9062 (J1 .9092 - .003)
#   report only  : gold58 Fracture sign vs J1, paired bootstraps, the task-fallback gate set, trainer-log parity
# Gold74 predictions of the probe come from ~/novel/gold_infer5.py (pass A = the trainer's eval input, fp32 CPU), run
# here as a subprocess if they do not exist yet (~10 min, 8 threads, nice 19). J1's come from ~/jump2/gold74_preds.
# usage: CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python ~/novel/frac75/score_frac75.py
#        [--run ~/fleet_runs/f0_s1043_ttr_frac75] [--gold_preds CSV] [--expect_labels NAME] [--dry] [--tag NAME]
# exit 0 = PASS, 1 = FAIL, 3 = probe not finished / not scorable, 2 = integrity error (bad pairing, wrong labels, ...).
import os, sys, re, json, glob, argparse, hashlib, subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

H = os.path.expanduser
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FR = T.index('Fracture')
# ---- pre-registered (PREREG.md, written before launch; do not edit after the probe starts)
BAR = {'G1': 0.8774, 'G2': 0.912, 'G3': 0.9062}
J1_REF = {'oof11': 0.8803, 'g74_frac': 0.894, 'g74_macro': 0.9092}       # masked convention, J1 best raw (ep 17, cv .8836)
# (the plan writes J1 OOF11 as .8804; measured .88031 here and .8803 in verify_perlabel vp.txt; the bar .8774 is the plan's, unchanged)
LABELS_NAME, LABELS_MD5 = 'labels_frac75_logit.csv', 'ad7582b817243f8f22e8bcfdd7e3f84a'
MODEL_SHA = 'f141aeed136dfc01'
J1_RUN = H('~/fleet_runs/f0_s1043_ttr_mix50')
J1_GOLD = H('~/jump2/gold74_preds/f0_s1043_ttr_mix50__raw.csv')
J1_GOLD_ALT = H('~/novel/gold74_preds/f0_s1043_ttr_mix50__raw__A.csv')
V4 = H('~/rsna_train/llm_labels_v4_blend.csv'); GOLD = H('~/rsna_train/gold74_eval_set.csv'); FOLDS = H('~/rsna_train/folds5.csv')
PY = H('~/miniconda3/envs/rsna/bin/python'); INFER = H('~/novel/gold_infer5.py')
ROOT = H('~/novel/frac75')

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=H('~/fleet_runs/f0_s1043_ttr_frac75'))
ap.add_argument('--gold_preds', default='')          # skip gold_infer5 and use this csv (selftest)
ap.add_argument('--expect_labels', default=LABELS_NAME)
ap.add_argument('--dry', action='store_true')        # tiny-store dry-run checkpoint: structural checks become warnings
ap.add_argument('--tag', default='')
ap.add_argument('--boot', type=int, default=2000)
A = ap.parse_args()
RUN = A.run.rstrip('/'); NAME = os.path.basename(RUN); TAG = A.tag or NAME
if A.dry and not TAG.startswith('DRY_'): TAG = 'DRY_' + TAG          # a dry/synthetic result can never unlock the folds
GPDIR = f'{ROOT}/dry/gold74_preds' if A.dry else f'{ROOT}/gold74_preds'   # dry preds never shadow the real probe's
os.makedirs(f'{ROOT}/out', exist_ok=True)
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); L.append(s); print(s, flush=True)
def die(code, msg):
    P(msg); open(f'{ROOT}/out/gate_{TAG}.txt', 'w').write('\n'.join(L) + '\n'); sys.exit(code)
WARN = []
def check(ok, msg):
    if ok: return
    if A.dry: WARN.append(msg); P('  WARN (dry):', msg)
    else: die(2, f'INTEGRITY ERROR: {msg}')
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def auc_masked(y, p):                                # == fleet_train.macro_auc per target (T9): NaN label/pred masked
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    ok = np.isfinite(y) & np.isfinite(p); yb = y[ok] >= 0.5
    return roc_auc_score(yb, p[ok]) if ok.sum() and 0 < yb.mean() < 1 else np.nan


def per_label(Pm, Y): return np.array([auc_masked(Y[:, j], Pm[:, j]) for j in range(12)])


P(f'== score_frac75 | run {RUN} | tag {TAG} | dry {A.dry}')
# ---- 1. finished?
log = RUN + '.log'
if not os.path.exists(log): die(3, f'NOT SCORABLE: no log {log}')
lines = open(log, errors='replace').read().splitlines()
closing = None
for ln in reversed(lines):
    if ln.startswith('{'):
        try: closing = json.loads(ln); break
        except Exception: pass
if closing is None or 'raw' not in closing: die(3, f'NOT FINISHED: no closing {{"raw": .., "ema": ..}} line in {log}')
P(f'closing line {closing}')
ck = sorted(glob.glob(f'{RUN}/f0_s1043_raw_*_best.pt'))
if len(ck) != 1: die(3, f'NOT SCORABLE: want exactly 1 raw best ckpt, got {ck}')
ck = ck[0]

# ---- 2. pairing / identity (ckpt args vs J1's; only train_labels may differ)
import torch
c = torch.load(ck, map_location='cpu', weights_only=False); a = c['args']
j1ck = glob.glob(f'{J1_RUN}/f0_s1043_raw_*_best.pt'); assert len(j1ck) == 1, j1ck
cj = torch.load(j1ck[0], map_location='cpu', weights_only=False); aj = cj['args']
ep0 = int(c['epoch'])
P(f'probe ckpt {os.path.basename(ck)} | epoch(1-based) {ep0 + 1} | cv {c["cv"]:.6f}')
P(f'J1    ckpt {os.path.basename(j1ck[0])} | epoch(1-based) {cj["epoch"] + 1} | cv {cj["cv"]:.6f}')
eps = [int(m.group(1)) for m in (re.match(r'^ep (\d+) raw \|', l) for l in lines) if m]
if not A.dry and (not eps or max(eps) + 1 != int(a.get('epochs', -1))):
    die(3, f'NOT SCORABLE: log has raw evals for {max(eps) + 1 if eps else 0} epochs, args epochs {a.get("epochs")} (crash / early stop)')
P(f'epochs evaluated {max(eps) + 1 if eps else 0} of {a.get("epochs")}')
check(abs(c['cv'] - closing['raw']) < 1e-12, f'ckpt cv {c["cv"]} != closing raw {closing["raw"]}')
check(a.get('train_labels') == A.expect_labels, f'train_labels {a.get("train_labels")} != {A.expect_labels}')
check(a.get('sel_labels') == 'llm_labels_v4_blend.csv', f'sel_labels {a.get("sel_labels")}')
check(a.get('model_sha') == MODEL_SHA, f'model_sha {a.get("model_sha")} != {MODEL_SHA}')
check((a.get('fold'), a.get('seed'), a.get('model_mod'), a.get('ttr_size')) == (0, 1043, 'fleet_model_ttr', 224),
      f'fold/seed/model {(a.get("fold"), a.get("seed"), a.get("model_mod"), a.get("ttr_size"))}')
diff = sorted(k for k in set(a) | set(aj) if k not in ('train_labels',) and repr(a.get(k)) != repr(aj.get(k)))
if A.expect_labels == aj.get('train_labels') and a.get('train_labels') == aj.get('train_labels'):
    P('pairing: (selftest on J1 itself)')
check(not diff, f'ckpt args differ from J1 beyond train_labels: {diff}')
P(f'pairing vs J1 ckpt args: differ only in train_labels ({aj.get("train_labels")} -> {a.get("train_labels")})'
  if not diff else f'pairing diff keys: {diff}')
lab_path = H(f'~/novel/frac75/{LABELS_NAME}')
if A.expect_labels == LABELS_NAME and os.path.exists(lab_path):
    m = hashlib.md5(open(lab_path, 'rb').read()).hexdigest()
    check(m == LABELS_MD5, f'{lab_path} md5 {m} != pre-registered {LABELS_MD5} (label file changed after prereg)')
    P(f'label file md5 {m} == pre-registered')
del c, cj

# ---- 3. G1: OOF of the best raw checkpoint vs steven_v4
v4 = rd(V4); fo = rd(FOLDS)['fold']; gold = rd(GOLD)
held = set(fo.index[fo < 0]) | set(gold.index)
f0ids = sorted(s for s in fo.index if fo[s] == 0 and s not in held)
def oof_scores(run):
    o = rd(f'{run}/oof_f0_s1043_raw.csv')[T].astype(float)
    Y = v4.loc[o.index, T].to_numpy(np.float64)
    pl = per_label(o.to_numpy(np.float64), Y)
    return o, pl
o, pl = oof_scores(RUN); oj, plj = oof_scores(J1_RUN)
check(len(o) == 867 and sorted(o.index) == f0ids, f'OOF ids: n {len(o)} (want the 867 fold-0 ids)')
check(not o.isna().any().any(), 'OOF has NaN')
m12 = np.nanmean(pl); m12j = np.nanmean(plj)
check(abs(m12 - closing['raw']) < 1e-9, f'OOF macro {m12:.6f} != ckpt/closing raw cv {closing["raw"]:.6f} (OOF file is not the selected epoch)')
g1 = float(np.nanmean(np.delete(pl, FR))); g1j = float(np.nanmean(np.delete(plj, FR)))
P(f'OOF vs v4 (n {len(o)}): 12-label macro probe {m12:.4f} J1 {m12j:.4f} (d {m12 - m12j:+.4f}) | '
  f'Fracture probe {pl[FR]:.4f} J1 {plj[FR]:.4f} | 11 non-Fracture probe {g1:.4f} J1 {g1j:.4f} (d {g1 - g1j:+.4f})')
if abs(g1j - J1_REF['oof11']) > 0.00015: P(f'  WARN: recomputed J1 OOF11 {g1j:.5f} != plan ref {J1_REF["oof11"]} (bar stays fixed)')

# ---- 4. gold74 predictions (pass A, CPU)
ids74 = list(gold.index); g58 = [s for s in ids74 if gold.loc[s, 'source'] == 'original_gold58']
gp = A.gold_preds or f'{GPDIR}/{NAME}__raw__A.csv'
if not os.path.exists(gp):
    P(f'gold preds missing -> running gold_infer5 (CPU, 8 threads, nice 19) on {os.path.basename(ck)}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NTHR='8', OMP_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run(['nice', '-n', '19', PY, INFER, '--passes', 'A', '--out', GPDIR, ck],
                       env=env, capture_output=True, text=True)
    for ln in (r.stdout + r.stderr).splitlines(): P('  [gold_infer5]', ln)
    if r.returncode != 0 or not os.path.exists(gp): die(2, f'gold_infer5 failed rc {r.returncode}')
check(A.gold_preds or os.path.getmtime(gp) >= os.path.getmtime(ck), f'{gp} is older than {ck} (stale preds from another run?)')
G = rd(gp).reindex(ids74)[T].astype(float); GJ = rd(J1_GOLD).reindex(ids74)[T].astype(float)
check(not G.isna().any().any(), f'probe gold preds have NaN / missing ids ({gp})')
if os.path.exists(J1_GOLD_ALT):
    dj = float(np.abs(rd(J1_GOLD_ALT).reindex(ids74)[T].astype(float).values - GJ.values).max())
    P(f'J1 gold preds {J1_GOLD} vs {J1_GOLD_ALT}: max|diff| {dj:.2e}')
Y74 = gold.loc[ids74, T].to_numpy(np.float64); i58 = [ids74.index(s) for s in g58]
a74 = per_label(G.values, Y74); a74j = per_label(GJ.values, Y74)
a58 = per_label(G.values[i58], Y74[i58]); a58j = per_label(GJ.values[i58], Y74[i58])
g2, g3 = float(a74[FR]), float(np.nanmean(a74)); g2j, g3j = float(a74j[FR]), float(np.nanmean(a74j))
for k, v, ref in (('g74_frac', g2j, J1_REF['g74_frac']), ('g74_macro', g3j, J1_REF['g74_macro'])):
    if abs(v - ref) > 0.0005: P(f'  WARN: recomputed J1 {k} {v:.4f} != plan ref {ref} (bar stays fixed)')
# trainer-log parity (fp16 GPU eval vs fp32 CPU pass A): gold74 macro at the selected epoch
pat = re.compile(r'^ep (\d+) raw \| gold74 ([0-9.]+) \(\d+ labels\) \| gold58 ([0-9.]+)')
logged = {int(m.group(1)): (float(m.group(2)), float(m.group(3))) for m in (pat.match(l) for l in lines) if m}
if ep0 in logged:
    P(f'trainer-logged gold74/gold58 at ep {ep0 + 1}: {logged[ep0][0]:.4f}/{logged[ep0][1]:.4f} | CPU pass A {g3:.4f}/{np.nanmean(a58):.4f}'
      f' (|d| gold74 {abs(logged[ep0][0] - g3):.4f})')
    check(abs(logged[ep0][0] - g3) < 0.003 or A.dry, 'CPU gold74 macro differs from the trainer log by >= .003')
SH = ['ACL', 'MCL', 'MedMen', 'LatMen', 'MedOA', 'LatOA', 'PFOA', 'Eff', 'Syn', 'Bak', 'Cont', 'Frac']
P('per-label (masked)          ' + ' '.join(f'{s:>6s}' for s in SH) + '   macro')
for nm, v in (('gold74 probe', a74), ('gold74 J1', a74j), ('gold58 probe', a58), ('gold58 J1', a58j),
              ('OOF-vs-v4 probe', pl), ('OOF-vs-v4 J1', plj)):
    P(f'{nm:27s} ' + ' '.join(f'{x:6.3f}' for x in v) + f'  {np.nanmean(v):.4f}')

# ---- 5. report-only: paired study bootstrap (masked), gold58 Fracture sign, the task-fallback gate set
def boot(Pa, Pb, Y, cols, B, seed=101):
    rng = np.random.default_rng(seed); n = len(Y); D = np.full((B, len(cols)), np.nan)
    for b in range(B):
        k = rng.integers(0, n, n)
        for i, j in enumerate(cols):
            D[b, i] = auc_masked(Y[k, j], Pa[k, j]) - auc_masked(Y[k, j], Pb[k, j])
    return D
d74 = boot(G.values, GJ.values, Y74, [FR], A.boot)[:, 0]
d58 = boot(G.values[i58], GJ.values[i58], Y74[i58], [FR], A.boot)[:, 0]
P(f'Fracture probe - J1: gold74 {g2 - g2j:+.4f} (boot SD {np.nanstd(d74):.4f}, P>0 {np.nanmean(d74 > 0):.2f}) | '
  f'gold58 {a58[FR] - a58j[FR]:+.4f} (SD {np.nanstd(d58):.4f}, P>0 {np.nanmean(d58 > 0):.2f}) -> gold58 Fracture sign '
  f'{"+" if a58[FR] - a58j[FR] > 0 else ("-" if a58[FR] - a58j[FR] < 0 else "0")}')
fb = {'Fb1 gold74 Frac >= J1+.01': g2 >= g2j + 0.01, 'Fb2 gold58 macro >= J1-.003': np.nanmean(a58) >= np.nanmean(a58j) - 0.003,
      'Fb3 OOF12 vs v4 >= J1-.003': m12 >= m12j - 0.003}
P('task-fallback gate set (REPORT ONLY, not the decision): ' + ' | '.join(f'{k} {"pass" if v else "FAIL"}' for k, v in fb.items()))

# ---- 6. verdict (pre-registered G1-G3 only)
G1, G2, G3 = g1 >= BAR['G1'], g2 >= BAR['G2'], g3 >= BAR['G3']
P(f'G1 no harm   : OOF 11 non-Fracture labels vs v4 {g1:.4f} >= {BAR["G1"]}  -> {"PASS" if G1 else "FAIL"}  (J1 {g1j:.4f})')
P(f'G2 mechanism : gold74 Fracture (masked)        {g2:.4f} >= {BAR["G2"]}   -> {"PASS" if G2 else "FAIL"}  (J1 {g2j:.4f})')
P(f'G3 guard     : gold74 macro (masked)           {g3:.4f} >= {BAR["G3"]}  -> {"PASS" if G3 else "FAIL"}  (J1 {g3j:.4f})')
ok = G1 and G2 and G3
fails = [k for k, v in (('G1', G1), ('G2', G2), ('G3', G3)) if not v]
P(f'FRAC75 GATE: {"PASS" if ok else "FAIL (" + ",".join(fails) + ")"}' + ('  [DRY RUN, NOT A DECISION]' if A.dry else ''))
P('decision: ' + ('PASS -> folds 1-4: bash ~/novel/frac75/launch_frac75.sh folds (21 ep, gc0, s1043); K9 needs per-set train_labels + key tag'
                  if ok else 'FAIL -> close target re-mixing; Synovitis/Fracture wait on image-aware relabelling (Eesh)'))
if WARN: P(f'{len(WARN)} dry-run warning(s): {WARN}')
open(f'{ROOT}/out/gate_{TAG}.txt', 'w').write('\n'.join(L) + '\n')
json.dump(dict(run=RUN, tag=TAG, dry=A.dry, G1=g1, G2=g2, G3=g3, J1=dict(oof11=g1j, g74_frac=g2j, g74_macro=g3j),
               bars=BAR, pass_=bool(ok and not A.dry), fails=fails, gold58_frac_delta=float(a58[FR] - a58j[FR]),
               oof12=m12, oof12_j1=m12j, gold_preds=gp, fallback={k: bool(v) for k, v in fb.items()}),
          open(f'{ROOT}/out/gate_{TAG}.json', 'w'), indent=1)
sys.exit(0 if ok else 1)
