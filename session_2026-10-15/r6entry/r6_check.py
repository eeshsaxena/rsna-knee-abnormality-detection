#!/usr/bin/env python
# r6_check.py (2026-10-15, Claude): commit gate for rsna-sub-v224r6flips6fj (R6 entry = J twin of the best with the R7
# bundle swapped for R6 fe055406). Usage: python r6_check.py OUT_DIR  (fetched commit output). Prints R6_CHECK PASS/FAIL n/m.
#  (a) the R6 bundle was used (receipt bundle_id fe055406fad6a12f), FLIP forced, no canary fired, J DEC cell ran
#  (b) the V224-FLIP stage with R6 == heliosli 0-9576 v1 commit output (submission / ours_sub / raptor csvs, <= 1e-6)
#  (c) S6fall B probs == the S6F v1 and J twin commit B probs (S6fall is independent of the bundle)
#  (d) final submission == .8 pct(V224-FLIP) + .2 pct(S6fall) recomputed independently
import os, sys, json, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'ref_h9576')
S6F1 = os.path.join(HERE, '..', 's6f', 'out_v224flips6f_v1')
J1 = os.path.join(HERE, '..', 'jpeg', 'overlay', 'r2', 'out_v224flips6fj_v1')
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
def maxdiff(a, b):
    A, B = rd(a), rd(b)
    if sorted(A.index) != sorted(B.index) or list(A.columns) != list(B.columns):
        return float('inf')
    return float(np.abs(A.loc[B.index].to_numpy(float) - B.to_numpy(float)).max())

logs = [p for p in glob.glob(os.path.join(OUT, '*.log')) if 'cufile' not in p]
text = ''
if logs:
    try:
        text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text))
chk('no Traceback', 'Traceback (most recent call last)' not in text)
chk('no 502 page in outputs', not any('Server Error' in open(p, encoding='utf-8', errors='replace').read()[:2000]
                                      for p in glob.glob(os.path.join(OUT, '*.csv')) + glob.glob(os.path.join(OUT, '*.json'))))
sr = json.load(open(os.path.join(OUT, 'submission_receipt.json')))
chk('(a) bundle_id == fe055406fad6a12f (R6)', sr.get('bundle_id') == 'fe055406fad6a12f', sr.get('bundle_id'))
chk('(a) R6 stage fallback_studies == 0', not sr.get('fallback_studies'), sr.get('fallback_studies'))
cr = json.load(open(os.path.join(OUT, 'canary_receipt.json')))
chk('(a) V224 canary: none fired, flip forced, r224 used', cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('R224_OK') is True, cr)
s6 = json.load(open(os.path.join(OUT, 's6f_receipt.json')))
chk('(a) S6F mode S6F, fired []', s6.get('mode') == 'S6F' and s6.get('fired') == [], (s6.get('mode'), s6.get('fired')))
chk('(a) J DEC receipt present', os.path.isfile(os.path.join(OUT, 'dec_receipt.json')))
chk('(a) FLIP_TTA_READY printed by the worker', 'FLIP_TTA_READY' in text or 'FLIP_TTA_READY' in ''.join(
    open(p, encoding='utf-8', errors='replace').read() for p in glob.glob(os.path.join(OUT, 'worker*_log.txt'))))
for ours, ref in (('submission_v224flip.csv', 'submission.csv'), ('ours_sub.csv', 'ours_sub.csv'), ('raptor_raw.csv', 'raptor_raw.csv'),
                  ('raptor_sub.csv', 'raptor_sub.csv'), ('raptor224_sub.csv', 'raptor224_sub.csv')):
    md = maxdiff(os.path.join(OUT, ours), os.path.join(REF, ref))
    tol = 1e-3 if ours in ('ours_sub.csv', 'raptor_raw.csv') else 1e-6   # raw probs: GPU run-to-run noise (R7 J vs heliosli raptor_raw 6.3e-05); rank files exact
    chk(f'(b) {ours} == heliosli 0-9576 {ref} (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
bo = os.path.join(OUT, 's6f', 'B', 'B_probs.csv')
for nm, d in (('S6F v1', S6F1), ('J twin v1', J1)):
    md = maxdiff(bo, os.path.join(d, 's6f', 'B', 'B_probs.csv'))
    chk(f'(c) S6fall B_probs == {nm}', md <= 1e-6, f'maxdiff {md:.2e}')
base, B, S = rd(os.path.join(OUT, 'submission_v224flip.csv')), rd(bo), rd(os.path.join(OUT, 'submission.csv'))
lab = list(S.columns)
B = B.loc[base.index, lab]; S = S.loc[base.index]
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
exp = np.stack([.8 * pct(base[c].values) + .2 * pct(B[c].values) for c in lab], 1)
md = float(np.abs(exp - S.to_numpy(float)).max())
chk('(d) submission == .8 pct(V224-FLIP-R6) + .2 pct(S6fall)', md < 1e-6, f'maxdiff {md:.2e}')
v = S.to_numpy(float)
chk('submission finite in [0,1], 12 labels', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() and len(lab) == 12)
diff_vs_best = maxdiff(os.path.join(OUT, 'submission_v224flip.csv'), os.path.join(J1, 'submission_v224flip.csv'))
print(f'  info: V224-FLIP stage R6 vs R7 (J twin) maxdiff {diff_vs_best:.4f} (must be > 0: the bundle really changed)')
chk('R6 stage differs from the R7 stage (the swap took effect)', diff_vs_best > 1e-4)
print(f'R6_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
