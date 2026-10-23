#!/usr/bin/env python
# probe_check.py (2026-10-17, Claude): gate for the A16 probes. Usage: python probe_check.py OUT --wr W --a224 A --ws6 S [--ref REF]
#  (a) R6 bundle, canary clean, S6F mode S6F, J DEC; (b) every MODEL output == the HW v1 output (raptor_raw / raptor_sub / raptor224_sub /
#  ours_sub / S6fall B); (c) an independent stage-3 recompute (heliosli table + W_RAPTOR W elsewhere + A224) that first reproduces the HW v1
#  stage 3 with W .5 / A224 .35, then equals this run's submission_v224flip; (d) final == (1 - S) pct(V224) + S pct(B); (e) format.
import os, sys, json, glob, argparse
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--wr', type=float, required=True)
ap.add_argument('--a224', type=float, required=True); ap.add_argument('--ws6', type=float, required=True)
ap.add_argument('--ref', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'session_2026-10-16', 'hw', 'out_v224r6hws6fj_v1'))
A = ap.parse_args(); OUT, REF = A.out, A.ref
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
TABLE = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
jl = lambda p: json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}
def maxdiff(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)): return float('inf')
    X, Y = rd(a), rd(b); cols = [c for c in LAB if c in Y.columns and c in X.columns]
    if sorted(X.index) != sorted(Y.index) or not cols: return float('inf')
    return float(np.abs(X.loc[Y.index, cols].to_numpy(float) - Y[cols].to_numpy(float)).max())
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any(): r[ok] = pct(np.asarray(v, np.float64)[ok])
    return r
def stage3(d, wr_other, a224, ids):
    raw, ours, r224 = (rd(os.path.join(d, f)).loc[ids] for f in ('raptor_raw.csv', 'ours_sub.csv', 'raptor224_sub.csv'))
    rf = (raw[LAB].values == 0.5).all(axis=1); out = np.zeros((len(ids), len(LAB)))
    for j, c in enumerate(LAB):
        rr, ro = pct_on(raw[c].values, ~rf), pct_on(ours[c].values, np.ones(len(ids), bool))
        rr = (1 - a224) * rr + a224 * pct_on(r224[c].values, ~rf)
        w = TABLE.get(c, wr_other); b = w * rr + (1 - w) * ro; b[rf] = ro[rf]; out[:, j] = b
    return out
logs = [p for p in glob.glob(os.path.join(OUT, '*.log')) if 'cufile' not in p]
text = ''
if logs:
    try: text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception: text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text)); chk('no Traceback', 'Traceback (most recent call last)' not in text)
sr, cr, s6 = jl(os.path.join(OUT, 'submission_receipt.json')), jl(os.path.join(OUT, 'canary_receipt.json')), jl(os.path.join(OUT, 's6f_receipt.json'))
chk('(a) R6 bundle, fallback 0', sr.get('bundle_id') == 'fe055406fad6a12f' and not sr.get('fallback_studies'))
chk('(a) V224 canary none fired, flip forced, r224 used', cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('R224_OK') is True)
chk('(a) S6F mode S6F, fired [], w_s6f == %g' % A.ws6, s6.get('mode') == 'S6F' and s6.get('fired') == [] and abs(((s6.get('blend') or {}).get('w_s6f') or -1) - A.ws6) < 1e-12)
chk('(a) J DEC receipt', os.path.isfile(os.path.join(OUT, 'dec_receipt.json')))
bw = (jl(os.path.join(OUT, 'blend_receipt.json')).get('weights') or {})
chk('(a) blend receipt: raptor %g, table == heliosli' % A.wr, abs((bw.get('raptor') or -1) - A.wr) < 1e-12 and bw.get('raptor_per_class') == TABLE, (bw.get('raptor'), bw.get('ours')))
for f, tol in (('raptor_sub.csv', 1e-6), ('raptor224_sub.csv', 1e-6), ('ours_sub.csv', 1e-3), ('raptor_raw.csv', 1e-3)):
    md = maxdiff(os.path.join(OUT, f), os.path.join(REF, f)); chk(f'(b) {f} == HW v1 (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
md = maxdiff(os.path.join(OUT, 's6f', 'B', 'B_probs.csv'), os.path.join(REF, 's6f', 'B', 'B_probs.csv')); chk('(b) S6fall B == HW v1', md <= 1e-6, f'maxdiff {md:.2e}')
Vr = rd(os.path.join(REF, 'submission_v224flip.csv')); ids = list(Vr.index)
m0 = float(np.abs(stage3(REF, 0.5, 0.35, ids) - Vr[LAB].to_numpy(float)).max()); chk('(c) recompute reproduces the HW v1 stage 3 (W .5, A224 .35)', m0 < 1e-6, f'maxdiff {m0:.2e}')
V = rd(os.path.join(OUT, 'submission_v224flip.csv'))
m1 = float(np.abs(stage3(OUT, A.wr, A.a224, ids) - V.loc[ids, LAB].to_numpy(float)).max()); chk('(c) this stage 3 == recompute (W %g, A224 %g)' % (A.wr, A.a224), m1 < 1e-6, f'maxdiff {m1:.2e}')
B, S = rd(os.path.join(OUT, 's6f', 'B', 'B_probs.csv')), rd(os.path.join(OUT, 'submission.csv'))
e = np.stack([(1 - A.ws6) * pct(V.loc[ids, c].values) + A.ws6 * pct(B.loc[ids, c].values) for c in LAB], 1)
m2 = float(np.abs(e - S.loc[ids, LAB].to_numpy(float)).max()); chk('(d) final == %.2f pct(V224) + %.2f pct(S6fall)' % (1 - A.ws6, A.ws6), m2 < 1e-6, f'maxdiff {m2:.2e}')
v = S[LAB].to_numpy(float); chk('(e) format', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() and list(S.columns) == LAB)
print('  info: stage 3 vs HW v1 maxdiff %.4f | final vs HW v1 maxdiff %.4f' % (maxdiff(os.path.join(OUT, 'submission_v224flip.csv'), os.path.join(REF, 'submission_v224flip.csv')), maxdiff(os.path.join(OUT, 'submission.csv'), os.path.join(REF, 'submission.csv'))))
print(f'PROBE_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
