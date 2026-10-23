#!/usr/bin/env python
# syn_probe_check.py (2026-10-20 night, Claude): copy of a21/probe_check_eoa21.py + --asyn (E2s: after E-OA, Synovitis = pct(pct(Syn) + asyn pct(Eff)))
# and check (d3) on the receipt a_syn + SYN_E2_APPLIED in the log. With --asyn 0 it requires NO a_syn and NO SYN_E2_APPLIED. a21 is untouched.
# probe_check_eoa21.py (2026-10-20, Claude): probe_check_eoa.py + --mode {sym,pw} (A21 pair weights). Origin: probe_check_eoa.py (2026-10-19, Claude): probe_check_t.py with check (d) extended for the A20 E-OA step (OA trio decorrelated on the final output). Original: gate for the A17 table probes = probe_check.py (A16) with the per-finding table as a parameter.
# Usage: python probe_check_t.py OUT --table JSON [--wr .5 --a224 .35 --ws6 .2] [--ref REF]
#  (a) R6 bundle, canary clean, S6F mode S6F, J DEC; (b) every MODEL output == the HW v1 output (raptor_raw / raptor_sub / raptor224_sub /
#  ours_sub / S6fall B); (c) an independent stage-3 recompute (heliosli table + W_RAPTOR W elsewhere + A224) that first reproduces the HW v1
#  stage 3 with W .5 / A224 .35, then equals this run's submission_v224flip; (d) final == (1 - S) pct(V224) + S pct(B); (e) format.
import os, sys, json, glob, argparse
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--wr', type=float, default=0.5); ap.add_argument('--table', required=True)
ap.add_argument('--a224', type=float, default=0.35); ap.add_argument('--ws6', type=float, default=0.2); ap.add_argument('--aoa', type=float, default=0.2); ap.add_argument('--mode', default='sym', choices=['sym', 'pw']); ap.add_argument('--asyn', type=float, default=0.0)
ap.add_argument('--ref', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'session_2026-10-16', 'hw', 'out_v224r6hws6fj_v1'))
A = ap.parse_args(); OUT, REF = A.out, A.ref; PT = json.loads(A.table)
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
def stage3(d, wr_other, a224, ids, table=TABLE):
    raw, ours, r224 = (rd(os.path.join(d, f)).loc[ids] for f in ('raptor_raw.csv', 'ours_sub.csv', 'raptor224_sub.csv'))
    rf = (raw[LAB].values == 0.5).all(axis=1); out = np.zeros((len(ids), len(LAB)))
    for j, c in enumerate(LAB):
        rr, ro = pct_on(raw[c].values, ~rf), pct_on(ours[c].values, np.ones(len(ids), bool))
        rr = (1 - a224) * rr + a224 * pct_on(r224[c].values, ~rf)
        w = table.get(c, wr_other); b = w * rr + (1 - w) * ro; b[rf] = ro[rf]; out[:, j] = b
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
chk('(a) blend receipt: raptor %g, table == %s' % (A.wr, PT), abs((bw.get('raptor') or -1) - A.wr) < 1e-12 and bw.get('raptor_per_class') == PT, (bw.get('raptor'), bw.get('raptor_per_class')))
for f, tol in (('raptor_sub.csv', 1e-6), ('raptor224_sub.csv', 1e-6), ('ours_sub.csv', 1e-3), ('raptor_raw.csv', 1e-3)):
    md = maxdiff(os.path.join(OUT, f), os.path.join(REF, f)); chk(f'(b) {f} == HW v1 (tol {tol:g})', md <= tol, f'maxdiff {md:.2e}')
md = maxdiff(os.path.join(OUT, 's6f', 'B', 'B_probs.csv'), os.path.join(REF, 's6f', 'B', 'B_probs.csv')); chk('(b) S6fall B == HW v1', md <= 1e-6, f'maxdiff {md:.2e}')
Vr = rd(os.path.join(REF, 'submission_v224flip.csv')); ids = list(Vr.index)
m0 = float(np.abs(stage3(REF, 0.5, 0.35, ids) - Vr[LAB].to_numpy(float)).max()); chk('(c) recompute reproduces the HW v1 stage 3 (W .5, A224 .35)', m0 < 1e-6, f'maxdiff {m0:.2e}')
V = rd(os.path.join(OUT, 'submission_v224flip.csv'))
m1 = float(np.abs(stage3(OUT, A.wr, A.a224, ids, PT) - V.loc[ids, LAB].to_numpy(float)).max()); chk('(c) this stage 3 == recompute (W %g, A224 %g, probe table)' % (A.wr, A.a224), m1 < 1e-6, f'maxdiff {m1:.2e}')
B, S = rd(os.path.join(OUT, 's6f', 'B', 'B_probs.csv')), rd(os.path.join(OUT, 'submission.csv'))
e = np.stack([(1 - A.ws6) * pct(V.loc[ids, c].values) + A.ws6 * pct(B.loc[ids, c].values) for c in LAB], 1)
e = e.astype(np.float32).astype(np.float64)                       # the kernel writes the .8/.2 blend as float32, then E-OA re-ranks it
OAI = [LAB.index(c) for c in ('Medial OA', 'Lateral OA', 'PF OA')]; U = {t: pct(e[:, t]) for t in OAI}
for t in OAI:
    o = [m for m in OAI if m != t]
    if A.mode == 'sym':
        e[:, t] = pct(U[t] - A.aoa * (U[o[0]] + U[o[1]]) / 2.0)
    else:
        PW = {(4, 5): 0.876, (4, 6): 0.708, (5, 6): 1.416}       # LAB index pairs: Medial OA 4, Lateral OA 5, PF OA 6
        e[:, t] = pct(U[t] - A.aoa / 2.0 * sum(PW[tuple(sorted((t, m)))] * U[m] for m in o))
e = e.astype(np.float32).astype(np.float64)                       # the kernel writes the E-OA columns as float32 before E2s reads them
if A.asyn:
    js, je = LAB.index('Synovitis'), LAB.index('Effusion')
    e[:, js] = pct(pct(e[:, js]) + A.asyn * pct(e[:, je]))
m2 = float(np.abs(e - S.loc[ids, LAB].to_numpy(float)).max()); chk('(d) final == %.2f pct(V224) + %.2f pct(S6fall), then E-OA a %g on the OA trio, then E2s a_syn %g' % (1 - A.ws6, A.ws6, A.aoa, A.asyn), m2 < 1e-6, f'maxdiff {m2:.2e}')
_as = (s6.get('blend') or {}).get('a_syn')
chk('(d3) S6F receipt a_syn == %g and SYN_E2_APPLIED in the log (none of either when 0)' % A.asyn, (_as == A.asyn and 'SYN_E2_APPLIED' in text) if A.asyn else (_as is None and 'SYN_E2_APPLIED' not in text))
chk('(d2) S6F receipt a_oa == %g, oa_mode == %s and EOA_APPLIED in the log' % (A.aoa, A.mode), (s6.get('blend') or {}).get('a_oa') == A.aoa and (s6.get('blend') or {}).get('oa_mode') == A.mode and 'EOA_APPLIED' in text)
v = S[LAB].to_numpy(float); chk('(e) format', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() and list(S.columns) == LAB)
print('  info: stage 3 vs HW v1 maxdiff %.4f | final vs HW v1 maxdiff %.4f' % (maxdiff(os.path.join(OUT, 'submission_v224flip.csv'), os.path.join(REF, 'submission_v224flip.csv')), maxdiff(os.path.join(OUT, 'submission.csv'), os.path.join(REF, 'submission.csv'))))
print(f'PROBE_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
