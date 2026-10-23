# dose_gold58.py (writeup figure 5 data, 10-23). Recomputes the gold58 E-OA dose curves (sym and pair-weighted) on the c4 output.
# Same code path as session_2026-10-20/a21/gate_a21_offline.py: the c4 notebook's own cell-18 _s6f_blend runs on the gold58 serve
# predictions, then the OA trio is decorrelated with the independent recompute that the gates matched to 2.8e-8.
# Output: AGGREGATES ONLY (macro AUC delta vs c4 per dose) -> writeup/figs/data/dose_gold58.json. No per-study value is written.
# Self-check: every dose that was already logged (gate_a20/a21/val/shape logs) must reproduce to 1e-5, else the script fails.
import json, os, sys, hashlib, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata

R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
OUTD = R + '/writeup/figs/data'
TMP = R + '/writeup/figs/src/_tmp_dose'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
V = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); V[:, j] = w * rr + (1 - w) * pct(R6[:, j])
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()

# run the c4 notebook's exact cell 18 in a scratch dir under writeup/ (removed afterwards)
NB = R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb'
nb = json.load(open(NB, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]
os.makedirs(TMP, exist_ok=True); W = TMP
assert fn.count("W = '/kaggle/working'") == 1
fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
os.makedirs(W + '/s6f/B', exist_ok=True)
pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](setup, run)
c4o = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
shutil.rmtree(TMP, ignore_errors=True)

Wp = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}


def eoa(mode, a):
    o = c4o.copy(); U = {c: pct(c4o[c].values.astype(np.float64)) for c in OA}
    for c in OA:
        oth = [k for k in OA if k != c]
        if mode == 'sym': e = pct(U[c] - a * (U[oth[0]] + U[oth[1]]) / 2.0)
        else: e = pct(U[c] - a / 2.0 * sum(Wp[tuple(sorted((c, k), key=OA.index))] * U[k] for k in oth))
        o[c] = e.astype(np.float32)
    return o


mac = lambda o: float(np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)]))
base = mac(c4o)
A = [round(x, 2) for x in np.arange(0.0, 0.6001, 0.05)]
curve = {m: [{'a': a, 'macro': round(mac(eoa(m, a)), 6), 'delta': round(mac(eoa(m, a)) - base, 6)} for a in A] for m in ('sym', 'pw')}

# self-check against every dose already logged on the laptop (kernel code path, base .94113)
LOGGED = {('sym', .1): .00142, ('sym', .2): .00247, ('sym', .25): .00210, ('sym', .3): .00201,
          ('pw', .2): .00265, ('pw', .25): .00255, ('pw', .3): .00223, ('pw', .35): .00256, ('pw', .4): .00190, ('pw', .45): .00157}
ok = abs(base - .94113) < 1e-5
print('c4 gold58 macro %.5f (logged .94113) %s' % (base, 'OK' if ok else 'FAIL'))
for (m, a), v in LOGGED.items():
    got = [r['delta'] for r in curve[m] if abs(r['a'] - a) < 1e-9][0]
    good = abs(got - v) < 1e-5; ok = ok and good
    print('%-3s a %.2f delta %+.5f (logged %+.5f) %s' % (m, a, got, v, 'OK' if good else 'FAIL'))
for m in ('sym', 'pw'):
    print(m, ' '.join('%.2f:%+.5f' % (r['a'], r['delta']) for r in curve[m]))
json.dump({'what': 'gold58 (n 58 expert-labelled studies) macro-12 AUC of c4 + E-OA, delta vs c4; info only, never a ruler',
           'base_c4_macro': round(base, 6), 'n_studies': n, 'curve': curve, 'self_check': 'PASS' if ok else 'FAIL',
           'script': 'writeup/figs/src/dose_gold58.py'}, open(OUTD + '/dose_gold58.json', 'w'), indent=1)
print('DOSE_SELF_CHECK', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
