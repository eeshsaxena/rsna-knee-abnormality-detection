#!/usr/bin/env python
# hw4_check.py (2026-10-16, Claude): = hw_check.py for rsna-sub-v224r6hws6f4j (heliosli weights + S6fall .4): final at .6 / .4.
#  (a) R6 bundle used, canary none fired / flip forced / r224 used, S6F mode, J DEC receipt
#  (b) every model output == the R6 entry v1 (raptor_raw / raptor_sub / raptor224_sub / ours_sub / S6fall B_probs)
#  (c) an INDEPENDENT stage-3 recompute: first reproduces the R6 entry's own submission_v224flip.csv with the uniform weights
#      (proves the recompute), then must equal this run's submission_v224flip.csv with the heliosli weights
#  (d) only Lateral OA / PF OA / Synovitis / Baker's columns differ from the R6 entry's stage 3; (e) final = .8 pct(V224) + .2 pct(B)
# Usage: python hw_check.py OUT [REF_OUT]   (REF_OUT default = the R6 entry v1 output; use REF_OUT=OUT of R6 for the self-test)
import os, sys, json, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'session_2026-10-15', 'r6entry', 'out_v224r6flips6fj_v1')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
W_UNI = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}
W_HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
jl = lambda p: json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}
def maxdiff(a, b, cols=None):
    if not (os.path.isfile(a) and os.path.isfile(b)): return float('inf')
    X, Y = rd(a), rd(b); cols = cols or [c for c in Y.columns if c in LAB]
    if sorted(X.index) != sorted(Y.index): return float('inf')
    return float(np.abs(X.loc[Y.index, cols].to_numpy(float) - Y[cols].to_numpy(float)).max())
def pct(v): return pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any(): r[ok] = pct(np.asarray(v, np.float64)[ok])
    return r
def stage3(d, W, ids):
    raw, ours, r224 = rd(os.path.join(d, 'raptor_raw.csv')).loc[ids], rd(os.path.join(d, 'ours_sub.csv')).loc[ids], rd(os.path.join(d, 'raptor224_sub.csv')).loc[ids]
    rap_fail = (raw[LAB].values == 0.5).all(axis=1)
    our_fail = np.zeros(len(ids), bool)              # the receipts below require 0 fallback studies, so every row is scored
    out = np.zeros((len(ids), len(LAB)))
    for j, c in enumerate(LAB):
        rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)
        rr = 0.65 * rr + 0.35 * pct_on(r224[c].values, ~rap_fail)
        wr = W.get(c, 0.5); b = wr * rr + (1 - wr) * ro
        b[rap_fail] = ro[rap_fail]; out[:, j] = b
    return pd.DataFrame(out, index=ids, columns=LAB)
logs = [p for p in glob.glob(os.path.join(OUT, '*.log')) if 'cufile' not in p]
text = ''
if logs:
    try: text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception: text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text)); chk('no Traceback', 'Traceback (most recent call last)' not in text)
sr = jl(os.path.join(OUT, 'submission_receipt.json'))
chk('(a) bundle_id fe055406 (R6), fallback 0', sr.get('bundle_id') == 'fe055406fad6a12f' and not sr.get('fallback_studies'), (sr.get('bundle_id'), sr.get('fallback_studies')))
cr = jl(os.path.join(OUT, 'canary_receipt.json'))
chk('(a) V224 canary none fired, flip forced, r224 used', cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('R224_OK') is True)
s6 = jl(os.path.join(OUT, 's6f_receipt.json'))
chk('(a) S6F mode S6F, fired []', s6.get('mode') == 'S6F' and s6.get('fired') == [])
chk('(a) J DEC receipt', os.path.isfile(os.path.join(OUT, 'dec_receipt.json')))
br = jl(os.path.join(OUT, 'blend_receipt.json'))
chk('(a) blend receipt weights == heliosli table', (br.get('weights') or {}).get('raptor_per_class') == W_HW, (br.get('weights') or {}).get('raptor_per_class'))
for f, tol in (('raptor_sub.csv', 1e-6), ('raptor224_sub.csv', 1e-6), ('ours_sub.csv', 1e-3), ('raptor_raw.csv', 1e-3)):
    md = maxdiff(os.path.join(OUT, f), os.path.join(REF, f)); chk(f'(b) {f} == R6 entry v1 (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
md = maxdiff(os.path.join(OUT, 's6f', 'B', 'B_probs.csv'), os.path.join(REF, 's6f', 'B', 'B_probs.csv'))
chk('(b) S6fall B_probs == R6 entry v1', md <= 1e-6, f'maxdiff {md:.2e}')
refv = rd(os.path.join(REF, 'submission_v224flip.csv')); ids = list(refv.index)
mu = float(np.abs(stage3(REF, W_UNI, ids).to_numpy() - refv[LAB].to_numpy(float)).max())
chk('(c) recompute reproduces the R6 entry stage 3 with the uniform weights', mu < 1e-6, f'maxdiff {mu:.2e}')
ov = rd(os.path.join(OUT, 'submission_v224flip.csv'))
mh = float(np.abs(stage3(OUT, W_HW, ids).to_numpy() - ov.loc[ids, LAB].to_numpy(float)).max())
chk('(c) this stage 3 == recompute with the heliosli weights', mh < 1e-6, f'maxdiff {mh:.2e}')
dcols = [c for c in LAB if float(np.abs(ov.loc[ids, c].to_numpy(float) - refv[c].to_numpy(float)).max()) > 1e-9]
chk('(d) stage-3 columns that differ from R6 == the reweighted ones (subset)', set(dcols) <= {'Lateral OA', 'PF OA', 'Synovitis', "Baker's"}, dcols)
B, S = rd(os.path.join(OUT, 's6f', 'B', 'B_probs.csv')), rd(os.path.join(OUT, 'submission.csv'))
e = np.stack([.6 * pct(ov.loc[ids, c].values) + .4 * pct(B.loc[ids, c].values) for c in LAB], 1)
md = float(np.abs(e - S.loc[ids, LAB].to_numpy(float)).max()); chk('(e) final == .6 pct(V224-hw) + .4 pct(S6fall)', md < 1e-6, f'maxdiff {md:.2e}')
v = S[LAB].to_numpy(float); chk('(e) format: finite, [0, 1], 12 labels', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() and list(S.columns) == LAB)
print(f'HW4_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
