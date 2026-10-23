# skep_stack6.py (10-20 night, Claude, SKEPTIC): placebo controls, run inline on 10-20 ~22:45 IST (this file records the code; logs
# skep_stack6.log = LOA <- X a .05 on pw30 under the 16-cell (MOA, LOA, PF, MM) raking; skep_stack6b.log = trio placebo E-OA under the
# 8-cell trio raking). Local CPU, no OAI. Reuses the helper definitions of skep_stack4.py / skep_stack3.py (everything above their data load).
import numpy as np
NIGHT = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
# --- part A (LOA <- X) ---
src = open(NIGHT + 'skep_stack4.py').read(); exec(src.split("D = CL.data(); C4 = D['C4']")[0].replace("skep_stack4.log", "skep_stack6.log"))
D = CL.data(); Y74 = D['Y74']
cols = OA + ['Medial Meniscus']; jc = [J[c] for c in cols]
ok = ~np.isnan(Y74[:, jc]).any(1); Yg = Y74[ok][:, jc].astype(int); T = np.zeros((2,) * 4)
for r in Yg: T[tuple(r)] += 1
for mn, (F, Y) in {'R7': (D['F7'], D['Yv2']), '3mean': (D['M3'], D['Y3v2'])}.items():
    m = ~np.isnan(Y[:, jc]).any(1); Fm = F[m]; Ym = Y[m][:, jc].astype(int); obs = np.zeros((2,) * 4)
    for r in Ym: obs[tuple(r)] += 1
    obs /= obs.sum(); tgt = rake(T + .5, Ym.mean(0)); w = (tgt / obs)[tuple(Ym.T)]; w = w / w.mean()
    B30 = pw(Fm); jl = J['Lateral OA']; yl = Ym[:, 1]; b0 = wauc(B30[:, jl], yl, w); out = []
    for X in ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'PF OA', 'ACL', 'MCL', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']:
        G = line(B30, 'Lateral OA', X, .05); out.append('%s %+.4f' % (X[:8], wauc(G[:, jl], yl, w) - b0))
    P('%-6s LOA <- X a .05 on pw30, (MOA,LOA,PF,MM) raked to gold74: %s' % (mn, ' | '.join(out)))
    w1 = np.ones(len(yl)); b1 = wauc(B30[:, jl], yl, w1); out = []
    for X in ['Medial Meniscus', 'ACL', "Baker's", 'Effusion', 'Contusion']:
        G = line(B30, 'Lateral OA', X, .05); out.append('%s %+.4f' % (X[:8], wauc(G[:, jl], yl, w1) - b1))
    P('%-6s same, raw report coupling: %s' % (mn, ' | '.join(out)))
# --- part B (trio placebo) ---
src = open(NIGHT + 'skep_stack3.py').read(); exec(src.split("D = CL.data()")[0].replace("skep_stack3.log", "skep_stack6b.log"))
D = CL.data(); Y74 = D['Y74'][:, jOA]; ok = ~np.isnan(Y74).any(1); Yg = Y74[ok].astype(int); cnt = np.zeros((2, 2, 2))
for r in Yg: cnt[r[0], r[1], r[2]] += 1
def plac(F, Xs, a=.3):
    G = F.copy()
    for c in OA: G[:, J[c]] = pct(pct(F[:, J[c]]) - a / 2.0 * (pct(F[:, J[Xs[0]]]) + pct(F[:, J[Xs[1]]])))
    return G
for mn, (F, Y) in {'R7': (D['F7'], D['Yv2']), '3mean': (D['M3'], D['Y3v2'])}.items():
    m = ~np.isnan(Y[:, jOA]).any(1); Fm = F[m]; Ym = Y[m][:, jOA].astype(int); obs = np.zeros((2, 2, 2))
    for r in Ym: obs[r[0], r[1], r[2]] += 1
    obs /= obs.sum(); tgt = rake(cnt + .5, Ym.mean(0)); w = (tgt / obs)[Ym[:, 0], Ym[:, 1], Ym[:, 2]]; w = w / w.mean()
    res = []
    for nm, G in (('pw30 (real)', eoa(Fm, .3, 'pw')), ('placebo ACL+Eff', plac(Fm, ['ACL', 'Effusion'])), ('placebo Cont+MCL', plac(Fm, ['Contusion', 'MCL'])),
                  ('placebo MM+LM', plac(Fm, ['Medial Meniscus', 'Lateral Meniscus'])), ('placebo Syn+Frac', plac(Fm, ['Synovitis', 'Fracture']))):
        res.append('%s %+.1f' % (nm, sum(wauc(G[:, J[c]], Ym[:, i], w) - wauc(Fm[:, J[c]], Ym[:, i], w) for i, c in enumerate(OA)) / 12 * 1e4))
    P('%-6s trio raked to gold74, macro e-4 vs raw: %s' % (mn, ' | '.join(res)))
