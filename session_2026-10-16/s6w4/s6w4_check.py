#!/usr/bin/env python
# s6w4_check.py (2026-10-16, Claude): = s6w_check.py for rsna-sub-v224r6s6f4j (S6F weight .4).
# (a) R6 bundle, canary none fired / flip / r224, S6F mode with w_s6f .3, J DEC; (b) every model output AND the V224 stage-3 output ==
# the R6 entry v1; (c) final == .7 pct(V224) + .3 pct(S6fall) recomputed, and != the R6 entry's final (the change took effect).
import os, sys, json, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'session_2026-10-15', 'r6entry', 'out_v224r6flips6fj_v1')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
jl = lambda p: json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}
def maxdiff(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)): return float('inf')
    X, Y = rd(a), rd(b); cols = [c for c in Y.columns if c in LAB]
    if sorted(X.index) != sorted(Y.index): return float('inf')
    return float(np.abs(X.loc[Y.index, cols].to_numpy(float) - Y[cols].to_numpy(float)).max())
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
logs = [p for p in glob.glob(os.path.join(OUT, '*.log')) if 'cufile' not in p]
text = ''
if logs:
    try: text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception: text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text)); chk('no Traceback', 'Traceback (most recent call last)' not in text)
sr = jl(os.path.join(OUT, 'submission_receipt.json'))
chk('(a) bundle_id fe055406 (R6), fallback 0', sr.get('bundle_id') == 'fe055406fad6a12f' and not sr.get('fallback_studies'))
cr = jl(os.path.join(OUT, 'canary_receipt.json'))
chk('(a) V224 canary none fired, flip forced, r224 used', cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('R224_OK') is True)
s6 = jl(os.path.join(OUT, 's6f_receipt.json'))
chk('(a) S6F mode S6F, fired [], w_s6f .4', s6.get('mode') == 'S6F' and s6.get('fired') == [] and abs(((s6.get('blend') or {}).get('w_s6f') or 0) - 0.4) < 1e-12, (s6.get('mode'), (s6.get('blend') or {}).get('w_s6f')))
chk('(a) J DEC receipt', os.path.isfile(os.path.join(OUT, 'dec_receipt.json')))
for f, tol in (('raptor_sub.csv', 1e-6), ('raptor224_sub.csv', 1e-6), ('ours_sub.csv', 1e-3), ('raptor_raw.csv', 1e-3), ('submission_v224flip.csv', 1e-6)):
    md = maxdiff(os.path.join(OUT, f), os.path.join(REF, f)); chk(f'(b) {f} == R6 entry v1 (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
md = maxdiff(os.path.join(OUT, 's6f', 'B', 'B_probs.csv'), os.path.join(REF, 's6f', 'B', 'B_probs.csv'))
chk('(b) S6fall B_probs == R6 entry v1', md <= 1e-6, f'maxdiff {md:.2e}')
V, B, S = rd(os.path.join(OUT, 'submission_v224flip.csv')), rd(os.path.join(OUT, 's6f', 'B', 'B_probs.csv')), rd(os.path.join(OUT, 'submission.csv'))
ids = list(V.index)
e = np.stack([.6 * pct(V.loc[ids, c].values) + .4 * pct(B.loc[ids, c].values) for c in LAB], 1)
md = float(np.abs(e - S.loc[ids, LAB].to_numpy(float)).max()); chk('(c) final == .6 pct(V224) + .4 pct(S6fall)', md < 1e-6, f'maxdiff {md:.2e}')
mr = maxdiff(os.path.join(OUT, 'submission.csv'), os.path.join(REF, 'submission.csv')); chk('(c) final differs from the R6 entry (w took effect)', mr > 1e-4, f'maxdiff {mr:.4f}')
v = S[LAB].to_numpy(float); chk('format: finite, [0, 1], 12 labels', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() and list(S.columns) == LAB)
print(f'S6W4_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
