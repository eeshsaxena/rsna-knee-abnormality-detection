#!/usr/bin/env python
# stack_commit_gate.py (10-20 ~23:18 IST, Claude, STACK BUILD round). Commit gate for the stack/kernels builds, run on the FETCHED committed
# output dir of a Kaggle commit. Lineage: a21/probe_check_eoa21.py -> night/c12_probe_check.py (+ --aloamm) -> stack/stack_probe_check.py
# (+ --amclmm) -> this file. The 19 checks of stack/stack_probe_check.py are kept (same arithmetic for (a)-(e)); (d2) / (d4) / (d5) are
# made STRICTER (the logged dose value is parsed: every copy of a marker must carry the flag's value; Kaggle logs repeat lines, so a
# repeated identical marker is tolerated, a conflicting one fails), and 7 checks are added:
#   (d-oa) (d-loa) (d-mcl) (d-rest): the (d) recompute split by component, so a failure names the component; (d-rest) = the 8 columns no
#          stack component writes must equal the plain .8 / .2 blend;
#   (d6) marker presence + order (first occurrences) in the kernel log: EOA_APPLIED < LOAMM_APPLIED < MCLMM_APPLIED < S6F_BLEND_OK < S6F_DONE;
#   (d7) S6F blend receipt keys == the scored pw30 key set + exactly the stack keys that are on (a_loamm / a_mclmm), nothing else;
#   (f)  receipts bound to these files: blend submission_sha256 == sha256(submission.csv), setup base_sha256 == sha256(submission_v224flip.csv),
#        blend b_probs_sha256 == run B probs_sha256 == sha256(s6f/B/B_probs.csv).
# 26 checks; a tick must key on 'PROBE_CHECK PASS 26/26'. INFO lines (not counted) say whether the visible rows can tell each component
# from its absence (the 3 public test rows usually cannot; dose identity then rests on (d2) / (d4) / (d5) + the registry sha guard).
# Usage: py -3.12 -I stack_commit_gate.py OUT --table JSON --a224 .25 --aoa {.3|.25} --mode pw --aloamm {0|.05} --amclmm {0|.1} [--ref REF]
import os, sys, json, glob, argparse, re, hashlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--wr', type=float, default=0.5); ap.add_argument('--table', required=True)
ap.add_argument('--a224', type=float, default=0.35); ap.add_argument('--ws6', type=float, default=0.2); ap.add_argument('--aoa', type=float, default=0.2)
ap.add_argument('--mode', default='sym', choices=['sym', 'pw']); ap.add_argument('--aloamm', type=float, default=0.0); ap.add_argument('--amclmm', type=float, default=0.0)
ap.add_argument('--ref', default=os.path.abspath(os.path.join(HERE, '..', '..', '..', 'session_2026-10-16', 'hw', 'out_v224r6hws6fj_v1')))
A = ap.parse_args(); OUT, REF = A.out, A.ref; PT = json.loads(A.table)
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
TABLE = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
# keys of the S6F blend receipt of the SCORED pw30 run (a21/out_v224r6c4pw30j_v1/s6f_receipt.json, MEASURED 10-20)
PW30_KEYS = {'w_s6f', 'cov_min', 'canary_col', 'mode', 'fired', 'base_source', 'n', 'base_blanked', 'b_probs_sha256', 'coverage', 'n_scored',
             'n_unscored', 'rho_base_s6f', 'rho_mean', 'a_oa', 'oa_mode', 'submission_sha256'}
R = []
def chk(n, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {n} {info}', flush=True)
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
jl = lambda p: json.load(open(p, encoding='utf-8')) if os.path.isfile(p) else {}
fsha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.isfile(p) else None
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
text, entries = '', []
if logs:
    try:
        entries = [e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8'))]; text = '\n'.join(entries)
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read(); entries = text.splitlines()
chk('kernel log present', bool(text)); chk('no Traceback', 'Traceback (most recent call last)' not in text)
sr, cr, s6 = jl(os.path.join(OUT, 'submission_receipt.json')), jl(os.path.join(OUT, 'canary_receipt.json')), jl(os.path.join(OUT, 's6f_receipt.json'))
bl = s6.get('blend') or {}
chk('(a) R6 bundle, fallback 0', sr.get('bundle_id') == 'fe055406fad6a12f' and not sr.get('fallback_studies'))
chk('(a) V224 canary none fired, flip forced, r224 used', cr.get('fired') == [] and cr.get('flip_tta_forced') is True and cr.get('R224_OK') is True)
chk('(a) S6F mode S6F, fired [], w_s6f == %g' % A.ws6, s6.get('mode') == 'S6F' and s6.get('fired') == [] and abs((bl.get('w_s6f') or -1) - A.ws6) < 1e-12)
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
OAI = [LAB.index(c) for c in ('Medial OA', 'Lateral OA', 'PF OA')]; JLO, JMM, JMC = LAB.index('Lateral OA'), LAB.index('Medial Meniscus'), LAB.index('MCL')
def expect(aoa, al, am):
    e = np.stack([(1 - A.ws6) * pct(V.loc[ids, c].values) + A.ws6 * pct(B.loc[ids, c].values) for c in LAB], 1)
    e = e.astype(np.float32).astype(np.float64)                   # the kernel writes the .8/.2 blend as float32, then E-OA re-ranks it
    U = {t: pct(e[:, t]) for t in OAI}
    if aoa:
        for t in OAI:
            o = [m for m in OAI if m != t]
            if A.mode == 'sym':
                e[:, t] = pct(U[t] - aoa * (U[o[0]] + U[o[1]]) / 2.0)
            else:
                PW = {(4, 5): 0.876, (4, 6): 0.708, (5, 6): 1.416}   # LAB index pairs: Medial OA 4, Lateral OA 5, PF OA 6
                e[:, t] = pct(U[t] - aoa / 2.0 * sum(PW[tuple(sorted((t, m)))] * U[m] for m in o))
    e = e.astype(np.float32).astype(np.float64)
    if al:
        e[:, JLO] = pct(pct(e[:, JLO]) - al * pct(e[:, JMM])); e[:, JLO] = e[:, JLO].astype(np.float32).astype(np.float64)
    if am:
        e[:, JMC] = pct(pct(e[:, JMC]) - am * pct(e[:, JMM]))
    return e
e = expect(A.aoa, A.aloamm, A.amclmm); Sv = S.loc[ids, LAB].to_numpy(float)
m2 = float(np.abs(e - Sv).max()); chk('(d) final == %.2f pct(V224) + %.2f pct(S6fall), then E-OA a %g on the OA trio, then C12p LOA <- MM a %g, then MCL <- MM a %g' % (1 - A.ws6, A.ws6, A.aoa, A.aloamm, A.amclmm), m2 < 1e-6, f'maxdiff {m2:.2e}')
cmd = lambda js: float(np.abs(e[:, js] - Sv[:, js]).max())
OTHER = [j for j in range(12) if j not in OAI + [JMC]]
chk('(d-oa) Medial OA + PF OA == E-OA %s a %g recompute' % (A.mode, A.aoa), cmd([OAI[0], OAI[2]]) < 1e-6, f'maxdiff {cmd([OAI[0], OAI[2]]):.2e}')
chk('(d-loa) Lateral OA == E-OA a %g then LOA <- MM a %g recompute' % (A.aoa, A.aloamm), cmd([JLO]) < 1e-6, f'maxdiff {cmd([JLO]):.2e}')
chk('(d-mcl) MCL == pct(pct(MCL) - %g pct(MM)) recompute' % A.amclmm, cmd([JMC]) < 1e-6, f'maxdiff {cmd([JMC]):.2e}')
chk('(d-rest) the 8 columns no stack component writes == plain %.2f / %.2f blend' % (1 - A.ws6, A.ws6), cmd(OTHER) < 1e-6, f'maxdiff {cmd(OTHER):.2e}')
# strict dose markers (value parsed, exactly one each)
fl = lambda x: float(x)
mk_eoa = re.findall(r'EOA_APPLIED a_oa ([0-9.eE+-]+) mode (\w+)', text); mk_lo = re.findall(r'LOAMM_APPLIED a_loamm ([0-9.eE+-]+)', text)
mk_mc = re.findall(r'MCLMM_APPLIED a_mclmm ([0-9.eE+-]+)', text)
_al, _am = bl.get('a_loamm'), bl.get('a_mclmm')
# Kaggle logs repeat some lines (82 distinct lines appear 2-4 times in the scored pw30 log, all from worker / driver output; the cell-18
# prints appeared once). So a marker may repeat, but every copy must carry the flag's value: a CONFLICTING value fails, a copy does not.
chk('(d5) S6F receipt a_mclmm == %g and MCLMM_APPLIED in the log, every copy a_mclmm %g (none of either when 0)' % (A.amclmm, A.amclmm),
    (_am == A.amclmm and len(mk_mc) >= 1 and all(fl(x) == A.amclmm for x in mk_mc)) if A.amclmm else (_am is None and not mk_mc), f'receipt {_am} log {mk_mc}')
chk('(d4) S6F receipt a_loamm == %g and LOAMM_APPLIED in the log, every copy a_loamm %g (none of either when 0)' % (A.aloamm, A.aloamm),
    (_al == A.aloamm and len(mk_lo) >= 1 and all(fl(x) == A.aloamm for x in mk_lo)) if A.aloamm else (_al is None and not mk_lo), f'receipt {_al} log {mk_lo}')
chk('(d2) S6F receipt a_oa == %g, oa_mode == %s and EOA_APPLIED in the log, every copy a_oa %g mode %s' % (A.aoa, A.mode, A.aoa, A.mode),
    bl.get('a_oa') == A.aoa and bl.get('oa_mode') == A.mode and len(mk_eoa) >= 1 and all(fl(x) == A.aoa and m == A.mode for x, m in mk_eoa),
    f'receipt {bl.get("a_oa")} {bl.get("oa_mode")} log {mk_eoa}')
def first(tok):
    k = [i for i, d in enumerate(entries) if tok in d]; return (k[0], len(k)) if k else (None, 0)
pos = {t: first(t) for t in ('EOA_APPLIED', 'LOAMM_APPLIED', 'MCLMM_APPLIED', 'S6F_BLEND_OK', 'S6F_DONE')}
want = ['EOA_APPLIED'] + (['LOAMM_APPLIED'] if A.aloamm else []) + (['MCLMM_APPLIED'] if A.amclmm else []) + ['S6F_BLEND_OK', 'S6F_DONE']
seq = [pos[t][0] for t in want]
chk('(d6) log markers %s all present, first occurrences in this order' % ' < '.join(want), None not in seq and seq == sorted(seq), str({t: pos[t] for t in pos}))
wkeys = PW30_KEYS | ({'a_loamm'} if A.aloamm else set()) | ({'a_mclmm'} if A.amclmm else set())
chk('(d7) S6F blend receipt keys == scored pw30 keys + %s (no other transform recorded)' % sorted(wkeys - PW30_KEYS), set(bl) == wkeys,
    'extra %s missing %s' % (sorted(set(bl) - wkeys), sorted(wkeys - set(bl))))
_b = ((s6.get('run') or {}).get('driver') or {}).get('B') or {}
chk('(f) receipts bound to these files: submission_sha256, setup base_sha256, B probs sha256',
    bl.get('submission_sha256') == fsha(os.path.join(OUT, 'submission.csv')) and (s6.get('setup') or {}).get('base_sha256') == fsha(os.path.join(OUT, 'submission_v224flip.csv'))
    and bl.get('b_probs_sha256') == _b.get('probs_sha256') == fsha(os.path.join(OUT, 's6f', 'B', 'B_probs.csv')))
v = S[LAB].to_numpy(float); chk('(e) format', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() and list(S.columns) == LAB)
# INFO (not counted): can the visible rows tell each component from its absence?
for nm, alt in (('dose (a_oa %g vs .3)' % A.aoa, (0.3, A.aloamm, A.amclmm)), ('LOAMM (a %g vs 0)' % A.aloamm, (A.aoa, 0.0, A.amclmm)), ('MCLMM (a %g vs 0)' % A.amclmm, (A.aoa, A.aloamm, 0.0))):
    if alt == (A.aoa, A.aloamm, A.amclmm):
        continue
    d = float(np.abs(expect(*alt) - e).max())
    print('  info: visible rows (n %d) %s: %s (maxdiff %.2e)' % (len(ids), nm, 'DISTINGUISHABLE' if d > 1e-6 else 'NOT distinguishable, dose identity rests on (d2)/(d4)/(d5) + registry sha', d))
print('  info: stage 3 vs HW v1 maxdiff %.4f | final vs HW v1 maxdiff %.4f' % (maxdiff(os.path.join(OUT, 'submission_v224flip.csv'), os.path.join(REF, 'submission_v224flip.csv')), maxdiff(os.path.join(OUT, 'submission.csv'), os.path.join(REF, 'submission.csv'))))
print(f'PROBE_CHECK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
sys.exit(0 if all(R) else 1)
