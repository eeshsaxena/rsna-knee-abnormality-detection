#!/usr/bin/env python
# fallback_g58.py (10-21 final red-team, Claude). LAPTOP ONLY, RSNA gold58 data only (competition labels; no OAI). Read-only on
# every source; writes only fallback_g58.log / .json next to this file.
# Question: what does a SILENT partial failure of the R6 ("ours") leg cost? In the A22 kernels (cell 8 / stage 3) a study that the
# R6 controller did not score (worker crash mid-run, OOM, decode error) takes Raptor's rank (b[our_fail] = rr[our_fail]); nothing
# blanks a column, so no canary fires (up to 50 % of rows). The S6F blend, E-OA and the stack transforms then run as normal.
# This replays that path on the gold58 serves for swide (c4 table, A224 .25, pw E-OA .25, lomm .05, MCLmm .1), pw25, c4, R6
# (uniform W .5, A224 .35) and r6pw20 (R6 + pw E-OA .2), with a random fraction f of rows losing the R6 leg.
# Also: S6F partial coverage in [.98, 1) (uncovered rows take the base rank in the S6fall slot), and Raptor per-study failures <= 3 %.
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; OUT = R + '/session_2026-10-21/final'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
Wp = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
d = np.load(R + '/session_2026-10-21/lastday/g58_payload.npz')
ids = list(d['ids']); R6s, S6, R384, R224 = d['R6s'], d['S6'], d['R384'].astype(np.float64), d['R224'].astype(np.float64)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L]
Y = g.values; n = len(ids)
C4T = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
R6T = {"Medial OA": .5, "Lateral OA": .5, "PF OA": .5}
CFG = {'swide': (C4T, .25, .25, .05, .10), 'pw25': (C4T, .25, .25, 0, 0), 'c4': (C4T, .25, 0, 0, 0),
       'R6': (R6T, .35, 0, 0, 0), 'r6pw20': (R6T, .35, .20, 0, 0)}


def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pct(np.asarray(v, np.float64)[ok])
    return r


def run(name, our_fail=None, rap_fail=None, s6_ok=None):
    T, A224, a, loamm, mclmm = CFG[name]
    our_fail = np.zeros(n, bool) if our_fail is None else our_fail
    rap_fail = np.zeros(n, bool) if rap_fail is None else rap_fail
    V = np.zeros((n, 12))
    for j, c in enumerate(L):                                   # cell 8 (stage 3), fail handling verbatim in logic
        rr = (1 - A224) * pct_on(R384[:, j], ~rap_fail) + A224 * pct_on(R224[:, j], ~rap_fail)
        ro = pct_on(R6s[:, j], ~our_fail)
        w = T.get(c, .5); b = w * rr + (1 - w) * ro
        b[rap_fail] = ro[rap_fail]; b[our_fail] = rr[our_fail]; b[rap_fail & our_fail] = .5
        V[:, j] = b
    V = V.astype(np.float32).astype(np.float64)
    ok = np.ones(n, bool) if s6_ok is None else s6_ok
    out = {}
    for j, c in enumerate(L):                                   # cell 18: .8 / .2 S6fall, unscored rows take the base rank
        rb = pct(V[:, j]); rs = np.full(n, np.nan); rs[ok] = pct(S6[ok, j]); rs[~ok] = rb[~ok]
        out[c] = (.8 * rb + .2 * rs).astype(np.float32).astype(np.float64)
    if a:
        U = {c: pct(out[c]) for c in OA}
        for c in OA:
            o = [k for k in OA if k != c]; w = [Wp[tuple(sorted((c, k), key=OA.index))] for k in o]
            out[c] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]])).astype(np.float32).astype(np.float64)
    if loamm:
        out['Lateral OA'] = pct(pct(out['Lateral OA']) - loamm * pct(out['Medial Meniscus'])).astype(np.float32).astype(np.float64)
    if mclmm:
        out['MCL'] = pct(pct(out['MCL']) - mclmm * pct(out['Medial Meniscus'])).astype(np.float32).astype(np.float64)
    P = np.stack([out[c] for c in L], 1)
    return np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12)])


rng = np.random.default_rng(20261021)
LOG = open(OUT + '/fallback_g58.log', 'w')


def say(s):
    print(s, flush=True); LOG.write(s + '\n')


base = {k: run(k) for k in CFG}
say('MEASURED (gold58 replay, n 58) clean macro AUC: ' + ', '.join('%s %.5f' % (k, v) for k, v in base.items()))
res = {'clean': base, 'r6_fail': {}, 's6_cov': {}, 'rap_fail': {}}
ND = 300
say('\n== R6-leg silent fallback: fraction f of rows take Raptor rank (mean delta e-4 over %d random row sets, [p10, p90]) ==' % ND)
for f in (.02, .05, .10, .20, .30, .50):
    k = max(1, int(round(f * n)))
    sel = [rng.choice(n, k, replace=False) for _ in range(ND)]
    row = {}
    for name in CFG:
        dl = []
        for s in sel:
            m = np.zeros(n, bool); m[s] = True
            dl.append((run(name, our_fail=m) - base[name]) * 1e4)
        dl = np.array(dl); row[name] = [float(dl.mean()), float(np.percentile(dl, 10)), float(np.percentile(dl, 90))]
    res['r6_fail'][f] = row
    say('  f %.2f (%2d rows): ' % (f, k) + ' | '.join('%s %+.1f [%+.1f, %+.1f]' % (nm, *row[nm]) for nm in CFG))
say('\n== S6F partial coverage (unscored rows take the base rank in the S6fall slot; passes the .98 gate) ==')
for f in (.02,):
    k = max(1, int(round(f * n))); dl = {nm: [] for nm in CFG}
    for _ in range(ND):
        ok = np.ones(n, bool); ok[rng.choice(n, k, replace=False)] = False
        for nm in CFG:
            dl[nm].append((run(nm, s6_ok=ok) - base[nm]) * 1e4)
    res['s6_cov'][f] = {nm: float(np.mean(v)) for nm, v in dl.items()}
    say('  uncovered %.2f (%d row): ' % (f, k) + ' | '.join('%s %+.2f' % (nm, np.mean(v)) for nm, v in dl.items()))
say('\n== Raptor per-study failures (<= 3 %% passes silently; failed rows take the R6 rank) ==')
for f in (.03,):
    k = max(1, int(round(f * n))); dl = {nm: [] for nm in CFG}
    for _ in range(ND):
        m = np.zeros(n, bool); m[rng.choice(n, k, replace=False)] = True
        for nm in CFG:
            dl[nm].append((run(nm, rap_fail=m) - base[nm]) * 1e4)
    res['rap_fail'][f] = {nm: float(np.mean(v)) for nm, v in dl.items()}
    say('  rap_fail %.2f (%d rows): ' % (f, k) + ' | '.join('%s %+.2f' % (nm, np.mean(v)) for nm, v in dl.items()))
json.dump(res, open(OUT + '/fallback_g58.json', 'w'), indent=1)
say('FALLBACK_G58 DONE')
LOG.close()
