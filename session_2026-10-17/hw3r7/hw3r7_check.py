#!/usr/bin/env python
# hw3r7_check.py (2026-10-17, Claude): = r7_check.py for rsna-sub-v224r6hw3r7j (hw3 + R7 late leg; REF = hw3 v1, final .7 / .3).
#  (a) everything before the R7 cell == the HW entry v1 output (model files, stage 3, S6fall B, and submission_pre_r7 == HW submission)
#  (b) the R7 leg ran clean: used, coverage 1.0, no flip arg, R7 probs finite in [0, 1] and different from R6
#  (c) independent recompute: ours_r67 == mean(R6, R7); stage 3 (heliosli weights) on ours_r67 == submission_v224r67;
#      final == .8 pct(v224r67) + .2 pct(S6fall B)
# Usage: python r7_check.py OUT [REF]   (REF default = the HW entry v1 output)
import os, sys, json, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'session_2026-10-16', 'hw3', 'out_v224r6hws6f3j_v1')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
W_HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
def rd(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str, 'index': str})
    if 'StudyInstanceUID' not in d.columns and 'index' in d.columns:   # the kernel's ours_r67 dump names the id column 'index'
        d = d.rename(columns={'index': 'StudyInstanceUID'})
    return d.set_index('StudyInstanceUID')
jl = lambda p: json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}
def maxdiff(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)): return float('inf')
    X, Y = rd(a), rd(b); cols = [c for c in LAB if c in Y.columns and c in X.columns]
    if sorted(X.index) != sorted(Y.index) or not cols: return float('inf')
    return float(np.abs(X.loc[Y.index, cols].to_numpy(float) - Y[cols].to_numpy(float)).max())
def pct(v): return pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any(): r[ok] = pct(np.asarray(v, np.float64)[ok])
    return r
logs = [p for p in glob.glob(os.path.join(OUT, '*.log')) if 'cufile' not in p]
text = ''
if logs:
    try: text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception: text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text)); chk('no Traceback', 'Traceback (most recent call last)' not in text)
sr, cr, s6 = jl(os.path.join(OUT, 'submission_receipt.json')), jl(os.path.join(OUT, 'canary_receipt.json')), jl(os.path.join(OUT, 's6f_receipt.json'))
chk('(a) R6 bundle, fallback 0', sr.get('bundle_id') == 'fe055406fad6a12f' and not sr.get('fallback_studies'))
chk('(a) V224 canary none fired, flip forced, r224 used', cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('R224_OK') is True)
chk('(a) S6F mode S6F, fired []', s6.get('mode') == 'S6F' and s6.get('fired') == [])
for f, tol in (('raptor_sub.csv', 1e-6), ('raptor224_sub.csv', 1e-6), ('ours_sub.csv', 1e-3), ('raptor_raw.csv', 1e-3), ('submission_v224flip.csv', 1e-6)):
    md = maxdiff(os.path.join(OUT, f), os.path.join(REF, f)); chk(f'(a) {f} == hw3 v1 (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
md = maxdiff(os.path.join(OUT, 's6f', 'B', 'B_probs.csv'), os.path.join(REF, 's6f', 'B', 'B_probs.csv')); chk('(a) S6fall B == hw3 v1', md <= 1e-6, f'maxdiff {md:.2e}')
md = maxdiff(os.path.join(OUT, 'submission_pre_r7.csv'), os.path.join(REF, 'submission.csv')); chk('(a) submission_pre_r7 == hw3 v1 submission', md <= 1e-6, f'maxdiff {md:.2e}')
r7 = jl(os.path.join(OUT, 'r7late_receipt.json'))
chk('(b) R7 leg used, coverage 1.0', r7.get('used') is True and r7.get('coverage') == 1.0, {k: r7.get(k) for k in ('used', 'why', 'coverage', 'seconds', 'rc')})
chk('(b) R7 worker args: no --flip-tta, R7-style manifest args', '--flip-tta' not in (r7.get('worker_args') or ['--flip-tta']) and '--state-stacker' in (r7.get('worker_args') or []))
o6, o7 = rd(os.path.join(OUT, 'ours_sub.csv')), rd(os.path.join(OUT, 'ours_r7_sub.csv'))
ids = list(rd(os.path.join(OUT, 'submission_v224flip.csv')).index)
v7 = o7.loc[ids, LAB].to_numpy(float)
chk('(b) R7 probs finite in [0, 1]', np.isfinite(v7).all() and (v7 >= 0).all() and (v7 <= 1).all())
d67 = float(np.abs(v7 - o6.loc[ids, LAB].to_numpy(float)).max()); chk('(b) R7 differs from R6', d67 > 1e-3, f'maxdiff {d67:.4f}')
avg = 0.5 * (o6.loc[ids, LAB].to_numpy(float) + v7)
ma = float(np.abs(avg - rd(os.path.join(OUT, 'ours_r67_sub.csv')).loc[ids, LAB].to_numpy(float)).max()); chk('(c) ours_r67 == mean(R6, R7)', ma < 1e-6, f'maxdiff {ma:.2e}')
raw, r224 = rd(os.path.join(OUT, 'raptor_raw.csv')).loc[ids], rd(os.path.join(OUT, 'raptor224_sub.csv')).loc[ids]
rap_fail = (raw[LAB].values == 0.5).all(axis=1); ok = np.ones(len(ids), bool)
out = np.zeros((len(ids), len(LAB)))
for j, c in enumerate(LAB):
    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(avg[:, j], ok)
    rr = 0.65 * rr + 0.35 * pct_on(r224[c].values, ~rap_fail)
    wr = W_HW.get(c, 0.5); b = wr * rr + (1 - wr) * ro; b[rap_fail] = ro[rap_fail]; out[:, j] = b
V = rd(os.path.join(OUT, 'submission_v224r67.csv'))
mv = float(np.abs(out - V.loc[ids, LAB].to_numpy(float)).max()); chk('(c) stage 3 on ours_r67 (heliosli weights) == submission_v224r67', mv < 1e-6, f'maxdiff {mv:.2e}')
B, S = rd(os.path.join(OUT, 's6f', 'B', 'B_probs.csv')), rd(os.path.join(OUT, 'submission.csv'))
e = np.stack([.7 * pct(V.loc[ids, c].values) + .3 * pct(B.loc[ids, c].values) for c in LAB], 1)
mf = float(np.abs(e - S.loc[ids, LAB].to_numpy(float)).max()); chk('(c) final == .7 pct(v224r67) + .3 pct(S6fall)', mf < 1e-6, f'maxdiff {mf:.2e}')
mh = maxdiff(os.path.join(OUT, 'submission.csv'), os.path.join(REF, 'submission.csv')); print(f'  info: final vs HW v1 maxdiff {mh:.4f}')
v = S[LAB].to_numpy(float); chk('format: finite, [0, 1], 12 labels', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() and list(S.columns) == LAB)
print(f'HW3R7_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
