# box_sev.py (10-20 more, Claude). Read-only; prints AGGREGATES ONLY (n, prevalence, AUC, AUC deltas, sd, P).
# Never writes a file, never prints an id, a row or a per-knee value. Piped via ssh stdin into python -I.
#
# Question (expert regime II, "severity thresholds x own-label member disagreement"):
#   The rubric grades every "on the fence" finding negative and uses strict severity cut-offs (moderate+ effusion / Baker's,
#   high-grade cartilage loss, definite meniscal tear, high-grade ACL). If expert positives are the cases all members agree on,
#   an AND-like member combiner (penalise own-label member disagreement) should gain MORE as the cut-off gets stricter, and an
#   OR-like combiner (E1 pow2 family) should lose more. Ruler: MOAKS grades on OAI V00 knees, OAI-FREE report-trained members.
#   Pre-registered reading (written before any OAI number of this run was seen):
#     MECHANISM SUPPORT = for >= 3 of the 5 strict-label families (OA trio pooled, Effusion, menisci pooled, ACL, Baker's) the
#       AND-family delta at the rubric-like level is > 0 with P >= .90 AND exceeds the delta at the lenient level;
#     KILL = AND-family delta at the rubric-like level <= 0 or P < .5 in >= 3 families.  Else INCONCLUSIVE.
#   AND family = 'geo' (mean log u), 'p05' (mean sqrt u), 'msd-' (mean - .5 sd). OR family = 'pow2', 'invgeo', 'msd+'.
# Members: A = teacher D5 mean (s1043 frac75, 5-fold mean), B = s1044 mix50 f0, C = s1045 288 f0 (3 families, c4-like count);
#   and the 7-member set (5 s1043 folds + B + C).
import os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import rankdata
H = os.path.expanduser('~/oai_build')
BOOT = int(os.environ.get('BOOT', '300'))
L9 = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"]
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def rd(f):
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    return d[L9]


P = {'A': rd(H + '/teacher/teacher_D5_mean_v00.csv'), 'B': rd(H + '/teacher2/f0_s1044_ttr_mix50__raw__A__gpu.csv'),
     'C': rd(H + '/teacher2/f0_s1045_ttr288frac75__raw__A__gpu.csv')}
for k in range(5):
    fs = sorted(glob.glob(H + '/teacher/f%d_s1043_ttr_frac75__raw__A__shard*of4.csv' % k))
    d = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fs]).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
    P['f%d' % k] = d[L9]
S = pd.read_csv(H + '/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
T = pd.read_csv(H + '/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
common = T.index
for k in P: common = common.intersection(P[k].dropna().index)
S = S.reindex(common)
print('knees with all 8 prediction sets: %d (sources rows matched %d)' % (len(common), int(S.notna().any(axis=1).sum())))

# ---------------- ladders (lenient -> strict); '*' marks the rubric-like level ----------------
LAD = {}
a = S['m_acl_tear']
LAD['ACL'] = [('partial+', a >= 1), ('complete*', a >= 2)]
for side, nm in (('mm', 'Medial Meniscus'), ('lm', 'Lateral Meniscus')):
    M = S[['m_%sah_morph' % side, 'm_%sbody_morph' % side, 'm_%sph_morph' % side]]
    ok = M.notna().all(axis=1) & S['m_%spr_tear' % side].notna()
    mx = M.max(axis=1); root = S['m_%spr_tear' % side] >= 1
    LAD[nm] = [('signal+', ((mx >= 1) | root).where(ok)), ('tear*', ((mx >= 2) | root).where(ok)),
               ('complex/mac', ((mx >= 4) | root).where(ok))]
for c, regs in COMP.items():
    cc = S[['m_cm_' + r for r in regs]].astype(float).values
    fin = np.isfinite(cc).all(1)
    area = np.floor(cc + 1e-6); ft = np.round((cc - area) * 10)
    lv = {'area1+': (area >= 1).any(1), 'area2+': (area >= 2).any(1), 'FT*': (ft >= 1).any(1),
          'area2&FT': ((area >= 2) & (ft >= 1)).any(1), 'FT2+': (ft >= 2).any(1)}
    LAD[c] = [(k, pd.Series(np.where(fin, v.astype(float), np.nan), index=S.index)) for k, v in lv.items()]
e = S['m_eff_wk']
LAD['Effusion'] = [('grade1+', e >= 1), ('grade2+*', e >= 2), ('grade3', e >= 3)]
h = S['m_syic']
LAD['Synovitis'] = [('hoffa1+', h >= 1), ('hoffa2+', h >= 2)]
LAD["Baker's"] = [('m_popcys*', S['m_popcys'] >= 1), ('w_popcys1+', S['w_popcys'] >= 1), ('w_popcys2+', S['w_popcys'] >= 2)]
# restore NaN where the source is missing for the non-OA ladders
SRC = {'ACL': ['m_acl_tear'], 'Effusion': ['m_eff_wk'], 'Synovitis': ['m_syic']}
for c, cols in SRC.items():
    LAD[c] = [(k, v.astype(float).where(S[cols].notna().all(axis=1))) for k, v in LAD[c]]
LAD["Baker's"] = [(k, v.astype(float).where(S[col].notna())) for (k, v), col in zip(LAD["Baker's"], ['m_popcys', 'w_popcys', 'w_popcys'])]


# ---------------- operators ----------------
def comb(op, U):
    """U: k x n member percentiles (equal weights)."""
    if op == 'base': return U.mean(0)
    if op == 'pow2': return (U ** 2).mean(0)
    if op == 'invgeo': return -np.log(1 - U + .5 / U.shape[1]).mean(0)
    if op == 'msd+': return U.mean(0) + .5 * U.std(0)
    if op == 'geo': return np.log(U).mean(0)
    if op == 'p05': return np.sqrt(U).mean(0)
    if op == 'msd-': return U.mean(0) - .5 * U.std(0)
    if op == 'min': return U.min(0)
    if op == 'max': return U.max(0)
    raise ValueError(op)


OPS = ['pow2', 'invgeo', 'msd+', 'geo', 'p05', 'msd-', 'min', 'max']
SETS = {'3fam': ['A', 'B', 'C'], '7mem': ['f0', 'f1', 'f2', 'f3', 'f4', 'B', 'C']}
RES = []
print('label | level | n | prev | set | base AUC | ' + ' | '.join('d_%s (P)' % o for o in OPS))
for c in L9:
    for lvname, y in LAD[c]:
        y = y.astype(float); ok = y.notna().values; ids = common[ok]; yy = y.values[ok].astype(int)
        n = len(yy); npos = int(yy.sum())
        if npos < 15 or n - npos < 15:
            print('%s | %s | n %d pos %d: skipped (too few)' % (c, lvname, n, npos)); continue
        for sn, mem in SETS.items():
            U = np.stack([pct(P[m].loc[ids, c].values.astype(float)) for m in mem])
            S0 = comb('base', U); a0 = auc(S0, yy)
            outs = {o: comb(o, U) for o in OPS}
            d = {o: auc(outs[o], yy) - a0 for o in OPS}
            rng = np.random.default_rng(20)
            bd = {o: [] for o in OPS}
            for _ in range(BOOT):
                ii = rng.integers(0, n, n); yb = yy[ii]
                if yb.sum() < 3 or yb.sum() > n - 3: continue
                b0 = auc(S0[ii], yb)
                for o in OPS: bd[o].append(auc(outs[o][ii], yb) - b0)
            Pp = {o: float(np.mean(np.array(bd[o]) > 0)) for o in OPS}; sd = {o: float(np.std(bd[o])) for o in OPS}
            mdis = float(U.std(0).mean())
            RES.append(dict(label=c, level=lvname, n=n, prev=npos / n, set=sn, base=a0, mdis=mdis, **{'d_' + o: d[o] for o in OPS},
                            **{'P_' + o: Pp[o] for o in OPS}, **{'sd_' + o: sd[o] for o in OPS}))
            print('%s | %s | %d | %.3f | %s | %.4f | ' % (c, lvname, n, npos / n, sn, a0) +
                  ' | '.join('%+.4f (%.2f)' % (d[o], Pp[o]) for o in OPS))
            sys.stdout.flush()
R = pd.DataFrame(RES)
R['dAND'] = R[['d_geo', 'd_p05', 'd_msd-']].mean(1); R['dOR'] = R[['d_pow2', 'd_invgeo', 'd_msd+']].mean(1)
pd.set_option('display.width', 250)
print('\nSUMMARY (AND family mean = geo/p05/msd-, OR family mean = pow2/invgeo/msd+):')
print(R[['label', 'level', 'set', 'n', 'prev', 'base', 'dAND', 'dOR', 'd_geo', 'P_geo', 'd_pow2', 'P_pow2']].round(4).to_string(index=False))
# within-label strictness trend: Spearman of level index vs dAND (per label, per set)
print('\nTREND per label (level index lenient->strict): dAND by level | dOR by level')
for (c, sn), g in R.groupby(['label', 'set'], sort=False):
    print('  %-17s %-5s dAND %s | dOR %s' % (c, sn, ' '.join('%+.4f' % v for v in g.dAND), ' '.join('%+.4f' % v for v in g.dOR)))
