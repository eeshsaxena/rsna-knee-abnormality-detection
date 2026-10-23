#!/usr/bin/env python
# s6m_check.py (2026-10-15, Claude): commit gate for an S6 entry built by patch_s6_model.py (PREREG_10-15 Amendment 7).
# Usage: python s6m_check.py OUT --tag TAG --base-out BASE_OUT [--control]
#   OUT       fetched commit output of rsna-sub-TAG
#   BASE_OUT  fetched commit output of the base kernel the patch was built on (e.g. the S6F J twin v1, or KC1 v1)
#   --control the model trio is the SAME file as the base's (nartaa s42): then B probs and submission must equal the base's
# Checks: (a) the swap took effect (driver sha == build, B_MODEL == build, model read from our mount, nartaa code tree unchanged);
# (b) every stage before S6F == the base output (V224-FLIP rank files exact, raw files <= 1e-3);
# (c) B probs: control -> == base (<= 1e-6); real -> finite, in [0, 1], all rows, and different from the base's (> 1e-4);
# (d) S6F blend == .8 pct(V224-FLIP) + .2 pct(B) recomputed; with a KC stage: KC probs == base (<= 1e-3) and
#     submission == .8 pct(submission_s6f) + .2 pct(KC) recomputed; (e) format.
import os, sys, json, glob, argparse
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--tag', required=True)
ap.add_argument('--base-out', required=True)
ap.add_argument('--control', action='store_true')
A = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT, BO = A.out, A.base_out
rec = json.load(open(os.path.join(HERE, 'build_receipt_%s.json' % A.tag)))
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
def maxdiff(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return float('inf')
    X, Y = rd(a), rd(b)
    if sorted(X.index) != sorted(Y.index) or list(X.columns) != list(Y.columns):
        return float('inf')
    return float(np.abs(X.loc[Y.index].to_numpy(float) - Y.to_numpy(float)).max())
def jl(p):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return {}

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
# (a) the swap
s6 = jl(os.path.join(OUT, 's6f_receipt.json'))
chk('(a) S6F mode S6F, fired []', s6.get('mode') == 'S6F' and s6.get('fired') == [], (s6.get('mode'), s6.get('fired')))
chk('(a) setup driver sha == build (patched driver ran)', (s6.get('setup') or {}).get('driver_sha256') == rec['driver_new_sha256'],
    ((s6.get('setup') or {}).get('driver_sha256', '')[:12], rec['driver_new_sha256'][:12]))
dr = jl(os.path.join(OUT, 's6f', 's6f_driver_receipt.json'))
inp = dr.get('inputs') or {}
chk('(a) driver ok', dr.get('ok') is True)
chk('(a) B_MODEL in the driver receipt == build', inp.get('model') == rec['b_model_new'], inp.get('model', {}).get('sha256', '')[:12])
chk('(a) model trio read from our mount', rec['s6m_slug'] in str(inp.get('dsm', '')), inp.get('dsm'))
chk('(a) nartaa dataset still supplies the code tree', 'rsna-knee-s6fall-weights' in str(inp.get('ds', '')), inp.get('ds'))
bdr = jl(os.path.join(BO, 's6f', 's6f_driver_receipt.json'))
chk('(a) code manifest sha == base (code tree byte-identical)', inp.get('manifest_sha256') == (bdr.get('inputs') or {}).get('manifest_sha256'),
    str(inp.get('manifest_sha256'))[:12])
if os.path.isfile(os.path.join(BO, 'dec_receipt.json')):
    chk('(a) J DEC receipt present (base had it)', os.path.isfile(os.path.join(OUT, 'dec_receipt.json')))
# (b) stages before S6F
for f, tol in (('submission_v224flip.csv', 1e-6), ('raptor_sub.csv', 1e-6), ('raptor224_sub.csv', 1e-6),
               ('ours_sub.csv', 1e-3), ('raptor_raw.csv', 1e-3)):
    md = maxdiff(os.path.join(OUT, f), os.path.join(BO, f))
    chk(f'(b) {f} == base (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
# (c) B probs
bo, bb = os.path.join(OUT, 's6f', 'B', 'B_probs.csv'), os.path.join(BO, 's6f', 'B', 'B_probs.csv')
md = maxdiff(bo, bb)
if A.control:
    chk('(c) CONTROL: B_probs == base B_probs (same weights via our mount)', md <= 1e-6, f'maxdiff {md:.2e}')
else:
    B = rd(bo)
    v = B.to_numpy(float)
    chk('(c) B_probs finite, in [0, 1], every base row', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
        and sorted(B.index) == sorted(rd(bb).index))
    chk('(c) B_probs differ from the base model (the swap changed the predictions)', md > 1e-4 and md != float('inf'), f'maxdiff {md:.4f}')
# (d) blends
pct = lambda x: pd.Series(np.asarray(x, np.float64)).rank(method='average', pct=True).values
base = rd(os.path.join(OUT, 'submission_v224flip.csv'))
B = rd(bo)
kc = os.path.isfile(os.path.join(OUT, 'kc', 'kc_probs.csv'))
s6sub = rd(os.path.join(OUT, 'submission_s6f.csv' if kc else 'submission.csv'))
lab = list(s6sub.columns)
Bx = B.loc[base.index, lab]
exp = np.stack([.8 * pct(base[c].values) + .2 * pct(Bx[c].values) for c in lab], 1)
md = float(np.abs(exp - s6sub.loc[base.index].to_numpy(float)).max())
chk('(d) S6F blend == .8 pct(V224-FLIP) + .2 pct(B)', md < 1e-6, f'maxdiff {md:.2e}')
if kc:
    K = rd(os.path.join(OUT, 'kc', 'kc_probs.csv'))
    KB = rd(os.path.join(BO, 'kc', 'kc_probs.csv'))
    kl = [c for c in lab if c in K.columns]          # kc_probs also carries kc_side / kc_rule / ... text columns
    mdk = (float(np.abs(K.loc[KB.index, kl].to_numpy(float) - KB[kl].to_numpy(float)).max())
           if sorted(K.index) == sorted(KB.index) and kl else float('inf'))
    chk('(d) KC probs == base KC probs (tol 1e-3)', mdk <= 1e-3, f'maxdiff {mdk:.2e}')
    S = rd(os.path.join(OUT, 'submission.csv'))
    kcols = [c for c in lab if c in K.columns]
    Kx = K.loc[s6sub.index, kcols]
    exp2 = np.stack([.8 * pct(s6sub[c].values) + .2 * pct(Kx[c].values) if c in kcols else s6sub[c].values for c in lab], 1)
    md2 = float(np.abs(exp2 - S.loc[s6sub.index, lab].to_numpy(float)).max())
    chk('(d) submission == .8 pct(S6F blend) + .2 pct(KC)', md2 < 1e-6, f'maxdiff {md2:.2e}')
S = rd(os.path.join(OUT, 'submission.csv'))
v = S.to_numpy(float)
chk('(e) submission finite in [0, 1], 12 labels, ids == base', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
    and S.shape[1] == 12 and sorted(S.index) == sorted(rd(os.path.join(BO, 'submission.csv')).index))
mds = maxdiff(os.path.join(OUT, 'submission.csv'), os.path.join(BO, 'submission.csv'))
if A.control:
    chk('(e) CONTROL: submission == base submission', mds <= 1e-6, f'maxdiff {mds:.2e}')
else:
    print(f'  info: submission vs base maxdiff {mds:.4f}')
print(f'S6M_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)} tag {A.tag}{" control" if A.control else ""}')
sys.exit(0 if all(R) else 1)
