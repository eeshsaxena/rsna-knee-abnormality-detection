#!/usr/bin/env python
# combo_check.py (2026-10-16, Claude): gate for a COMBINATION kernel (build_combo.py): every stage must equal the output of the
# already-gated kernel that owns that stage, and both blends are recomputed independently.
#   r6kc: V224 stage (R6)  == R6 entry v1 output (r6entry/out_v224r6flips6fj_v1; R6_CHECK 19/19)
#         S6F stage        == R6 entry v1 (B_probs, and the frozen S6F blend == R6 entry v1 submission.csv)
#         KC stage         == KC1 v1 output (kcstage/out_v224s6fkcj_v1; V224S6FKC_CHECK 30/30) kc_probs
# Usage: python combo_check.py OUT r6kc
import os, sys, json, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT, K = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
S15 = os.path.join(HERE, '..', '..', 'session_2026-10-15')
REFS = {'r6kc': dict(base=os.path.join(S15, 'r6entry', 'out_v224r6flips6fj_v1'), kc=os.path.join(S15, 'kcstage', 'out_v224s6fkcj_v1'),
                     bundle='fe055406fad6a12f')}
RF = REFS[K]
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']
def maxdiff(a, b, cols=None):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return float('inf')
    X, Y = rd(a), rd(b)
    cols = cols or list(Y.columns)
    if sorted(X.index) != sorted(Y.index) or any(c not in X.columns for c in cols):
        return float('inf')
    return float(np.abs(X.loc[Y.index, cols].to_numpy(float) - Y[cols].to_numpy(float)).max())
jl = lambda p: json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}

logs = [p for p in glob.glob(os.path.join(OUT, '*.log')) if 'cufile' not in p]
text = ''
if logs:
    try:
        text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text))
chk('no Traceback in the kernel log', 'Traceback (most recent call last)' not in text)
chk('no 502 page in outputs', not any('Server Error' in open(p, encoding='utf-8', errors='replace').read()[:2000]
                                      for p in glob.glob(os.path.join(OUT, '*.csv')) + glob.glob(os.path.join(OUT, '*.json'))))
sr = jl(os.path.join(OUT, 'submission_receipt.json'))
chk('V224 stage bundle_id == %s' % RF['bundle'], sr.get('bundle_id') == RF['bundle'], sr.get('bundle_id'))
chk('V224 stage fallback_studies == 0', not sr.get('fallback_studies'), sr.get('fallback_studies'))
cr = jl(os.path.join(OUT, 'canary_receipt.json'))
chk('V224 canary: none fired, flip forced, r224 used', cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('R224_OK') is True)
chk('J DEC receipt present', os.path.isfile(os.path.join(OUT, 'dec_receipt.json')))
for f, tol in (('submission_v224flip.csv', 1e-6), ('raptor_sub.csv', 1e-6), ('raptor224_sub.csv', 1e-6), ('ours_sub.csv', 1e-3),
               ('raptor_raw.csv', 1e-3)):
    md = maxdiff(os.path.join(OUT, f), os.path.join(RF['base'], f))
    chk(f'V224 stage {f} == base-stage owner (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
s6 = jl(os.path.join(OUT, 's6f_receipt.json'))
chk('S6F mode S6F, fired []', s6.get('mode') == 'S6F' and s6.get('fired') == [], (s6.get('mode'), s6.get('fired')))
md = maxdiff(os.path.join(OUT, 's6f', 'B', 'B_probs.csv'), os.path.join(RF['base'], 's6f', 'B', 'B_probs.csv'))
chk('S6F B_probs == base-stage owner (tol 1e-6)', md <= 1e-6, f'maxdiff {md:.2e}')
md = maxdiff(os.path.join(OUT, 'submission_s6f.csv'), os.path.join(RF['base'], 'submission.csv'))
chk('frozen S6F blend (submission_s6f.csv) == base-stage owner submission.csv (tol 1e-6)', md <= 1e-6, f'maxdiff {md:.2e}')
kr = jl(os.path.join(OUT, 'kc_receipt.json'))
chk('KC mode KC, fired []', kr.get('mode') == 'KC' and kr.get('fired') == [], (kr.get('mode'), kr.get('fired')))
md = maxdiff(os.path.join(OUT, 'kc', 'kc_probs.csv'), os.path.join(RF['kc'], 'kc', 'kc_probs.csv'), LAB)
chk('KC probs == KC1 v1 kc_probs (12 labels, tol 1e-3)', md <= 1e-3, f'maxdiff {md:.2e}')
pct = lambda x: pd.Series(np.asarray(x, np.float64)).rank(method='average', pct=True).values
base, B = rd(os.path.join(OUT, 'submission_v224flip.csv')), rd(os.path.join(OUT, 's6f', 'B', 'B_probs.csv'))
S6 = rd(os.path.join(OUT, 'submission_s6f.csv'))
e1 = np.stack([.8 * pct(base[c].values) + .2 * pct(B.loc[base.index, c].values) for c in LAB], 1)
chk('S6F blend recomputed: .8 pct(V224) + .2 pct(B)', float(np.abs(e1 - S6.loc[base.index, LAB].to_numpy(float)).max()) < 1e-6)
KC = rd(os.path.join(OUT, 'kc', 'kc_probs.csv'))
S = rd(os.path.join(OUT, 'submission.csv'))
e2 = np.stack([.8 * pct(S6[c].values) + .2 * pct(KC.loc[S6.index, c].values) for c in LAB], 1)
md2 = float(np.abs(e2 - S.loc[S6.index, LAB].to_numpy(float)).max())
chk('final recomputed: .8 pct(S6F blend) + .2 pct(KC)', md2 < 1e-6, f'maxdiff {md2:.2e}')
v = S[LAB].to_numpy(float)
chk('submission finite in [0, 1], 12 labels, ids == base', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
    and list(S.columns) == LAB and sorted(S.index) == sorted(base.index))
print(f'COMBO_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)} {K}')
sys.exit(0 if all(R) else 1)
