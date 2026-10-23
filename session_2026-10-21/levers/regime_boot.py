# regime_boot.py (10-21 levers, Claude, REGIME track step 5). RESEARCH ONLY, local CPU, NO OAI (OAI numbers = CLAIMED aggregates parsed
# by oai_claimed.py). Writes regime_boot.log + regime_boot.json.
# For every direction x a that regime_dlt_sweep.py put all-4-rulers-positive under EB (plus the in-flight references lomm05 = LOA <- MM .05
# and MCLmm = MCL <- MM .10), measure what the point estimates hide:
#  (1) TARGET uncertainty: B gold58 bootstrap draws of the same-knee D map (expert vs report-v2, all 66 pairs, so meanD and the trio move
#      too) -> EB* and RAW* targets -> delta per ruler. P(delta > 0) over draws, 3mean and worst ruler.
#  (2) SPECIFICITY: under EB, the partner's delta vs the 10 other partners of the same target (rank, and the placebo mean).
#  (3) c4 co-firing on the SAME gold58 images vs R7 OOF (is the ruler's model-side coupling representative of c4?).
#  (4) OAI (CLAIMED): raw / R2 / R2lo signs for the direction, where OAI has both labels.
#  (5) gold58 c4 pw30-path change (INFO ONLY; kappa .42 CLAIMED recal.md).
import json, itertools
import numpy as np
from rg_lib import (L, J, AB, OA, eoa, near_pd, copula_cells_mc as copula_cells, weights_from_target, load_all, pct, wauc_multi, tet_xy_fast as tet_xy, ns, LD)
Bd = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/levers/'
log = open(Bd + 'regime_boot.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


sw = json.load(open(Bd + 'regime_dlt_sweep.json')); cal = json.load(open(Bd + 'regime_cal.json')); dl = json.load(open(Bd + 'regime_dlt.json'))
oai = json.load(open(Bd + 'oai_claimed.json'))
AV = sw['AV']; MODELS = sw['MODELS']; REL = sw['REL']; GAM = sw['GAM']; Trep = np.array(cal['Trep'])
RES = sw['RES']
flag = []
a1 = json.load(open(Bd + 'regime_sweep.json'))['RES']
def allpos(src, k, ia, r): return all(src[k]['res'][m + '|' + r][ia] > 0 for m in MODELS)
hg_cand = []
for k, d in RES.items():
    for ia, a in enumerate(AV):
        ht = allpos(RES, k, ia, 'EB') and allpos(RES, k, ia, 'RAW') and allpos(RES, k, ia, 'EBg')
        hg = allpos(a1, k, ia, 'G74') and allpos(a1, k, ia, 'IPF')
        syn = 'Synovitis' in k
        if ht and hg: flag.append((k, a))                      # BOTH
        elif ht and not syn: flag.append((k, a))               # H_train only (non-Syn)
        elif hg and not syn and not allpos(RES, k, ia, 'EB'): hg_cand.append((a1[k]['res']['3mean|G74'][ia], k, a))
flag += [(k, a) for _, k, a in sorted(hg_cand, reverse=True)[:8]]   # top H_gold-only (non-Syn) by 3mean G74
P('flagged cells (EB all-4 > 0) + refs: %d' % len(flag))
D = load_all(); X, Y, c4 = D['X'], D['Y'], D['c4']
BASE = {m: eoa(F, .3, 'pw') for m, F in X.items()}
g, g58 = LD.gold(); ids58 = g58.StudyInstanceUID.tolist(); E58 = g58[L].values.astype(float)
R58 = LD.labels('v2').reindex(ids58)[L].values.astype(float)
r7 = LD.r7(); R7g58 = r7.loc[ids58, ['p|' + c for c in L]].values.astype(float)
trio = lambda i, j: L[i] in OA and L[j] in OA


def dmat(idx):
    Dm = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            Dm[i, j] = Dm[j, i] = tet_xy(E58[idx, i], E58[idx, j]) - tet_xy(R58[idx, i], R58[idx, j])
    return Dm


def eb_of(Dm):
    nt = [Dm[i, j] for i in range(12) for j in range(i + 1, 12) if not trio(i, j)]; mD = np.mean(nt)
    De = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            if i != j: De[i, j] = Dm[i, j] if trio(i, j) else mD + REL * (Dm[i, j] - mD)
    return De


NB = 100
rng = np.random.default_rng(21102)
P('building %d gold58 bootstrap D maps ...' % NB)
DRAWS = []
for b in range(NB):
    idx = rng.integers(0, 58, 58); Dm = dmat(idx); DRAWS.append((Dm, eb_of(Dm)))
P('done')
Dpt = np.zeros((12, 12))
for kk, e in dl['pairs'].items():
    a_, b_ = kk.split(' | '); Dpt[J[a_], J[b_]] = Dpt[J[b_], J[a_]] = e['delta']
c4pw = eoa(c4, .3, 'pw'); Y58 = D['Y58']
rho = lambda F, i, j: float(np.corrcoef(ns(F[:, i]), ns(F[:, j]))[0, 1])


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


OUTJ = {}
WCACHE = {}
P('')
P('per flagged cell: label AUC x1e4 (macro = /12). EB pt = 3mean/min ruler; boot P(>0) EB* and RAW* (3mean | worst ruler); placebo rank of o among'
  ' 11 partners under EB (3mean) and placebo mean; c4 vs R7 gold58 co-firing; OAI raw / R2 / R2lo at a .1 (+ = 3/3 > 0); gold58 c4 path')
for k, a in flag:
    t, o = k.split(' <- '); ia = AV.index(a)
    cols = list(dict.fromkeys(([c for c in OA] if (t in OA or o in OA) else []) + [t, o]))
    jc = [J[c] for c in cols]; m_ = ~np.isnan(Y[:, jc]).any(1); Ysub = Y[m_][:, jc].astype(int); yt = Ysub[:, cols.index(t)]
    # bootstrap target draws: weights per draw (EB*, RAW*)
    ck = tuple(cols)
    if ck not in WCACHE:
        Wd = []
        for Dm, De in DRAWS:
            for M in (De, Dm):
                R = near_pd(Trep[np.ix_(jc, jc)] + M[np.ix_(jc, jc)]); Wd.append(weights_from_target(Ysub, copula_cells(Ysub.mean(0), R)))
        WCACHE[ck] = Wd
    Wd = WCACHE[ck]
    res = {}
    for mn in MODELS:
        Bm = BASE[mn][m_]; bt = Bm[:, J[t]]
        b0 = wauc_multi(bt, yt, Wd); b1 = wauc_multi(pct(pct(bt) - a * pct(Bm[:, J[o]])), yt, Wd)
        dd = (b1 - b0).reshape(NB, 2); res[mn] = dd
    pe = np.array([res[m][:, 0].mean() for m in MODELS]); pr = np.array([res[m][:, 1].mean() for m in MODELS])
    Pe = np.array([(res[m][:, 0] > 0).mean() for m in MODELS]); Pr = np.array([(res[m][:, 1] > 0).mean() for m in MODELS])
    # placebo under EB point (from sweep json)
    oth = [x for x in L if x not in (t, o)]
    ebv = lambda oo: RES[t + ' <- ' + oo]['res']['3mean|EB'][ia]
    me = ebv(o); pl = np.array([ebv(x) for x in oth]); rank = 1 + int((pl > me).sum())
    # c4 vs R7 gold58 co-firing
    i, j = J[t], J[o]
    rc4 = rho(c4pw, i, j); rr7 = rho(eoa(R7g58, .3, 'pw'), i, j); rexp = tet_xy(E58[:, i], E58[:, j])
    # OAI claimed
    ok = AB[t] + '<-' + AB[o]; os_ = ''
    if ok in oai:
        s = lambda v: '+' if all(x > 0 for x in v) else ('-' if all(x < 0 for x in v) else '0')
        r_ = oai[ok]; os_ = 'raw%s R2%s R2lo%s' % (s(r_['raw.1'] if a > 0 else r_['raw-.1']), s(r_['R2.1']) if a > 0 else '?', s(r_['R2lo.1']) if a > 0 else '?')
    # gold58 c4 path (info)
    g = pct(pct(c4pw[:, i]) - a * pct(c4pw[:, j])); dg = auc(g, Y58[:, i]) - auc(c4pw[:, i], Y58[:, i])
    ebpt = np.array([RES[k]['res'][m + '|EB'][ia] for m in MODELS])
    P('%-12s %+5.2f | EB pt %+6.1f/%+6.1f | boot EB* %+6.1f P %.2f|%.2f  RAW* %+6.1f P %.2f|%.2f | placebo rank %2d/11 (pl mean %+5.1f) | '
      'rho g58 c4 %+.2f R7 %+.2f exp %+.2f | OAI %-15s | g58 c4 %+.4f' % (
          AB[t] + '<-' + AB[o], a, ebpt[3] * 1e4, ebpt.min() * 1e4, pe[3] * 1e4, Pe[3], Pe.min(), pr[3] * 1e4, Pr[3], Pr.min(), rank, pl.mean() * 1e4,
          rc4, rr7, rexp, os_, dg))
    OUTJ[k + ' | %.2f' % a] = dict(eb=ebpt.tolist(), boot_eb=pe.tolist(), P_eb=Pe.tolist(), boot_raw=pr.tolist(), P_raw=Pr.tolist(), rank=rank,
                                   placebo_mean=float(pl.mean()), rho_c4=rc4, rho_r7=rr7, tet_exp=rexp, oai=os_, g58=dg)
json.dump(OUTJ, open(Bd + 'regime_boot.json', 'w'), indent=1)
P('REGIME_BOOT DONE')
