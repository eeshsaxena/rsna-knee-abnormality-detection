# recal_reliab.py (10-20 night, Claude, TRACK 1). RESEARCH ONLY, local CPU, gold58 + c4 / eoa20 rebuild (no OAI). Writes recal_reliab.log.
# How much of a gold58 directional coupling gap (rho - q, c4) is signal? recal_transfer.py's EB with bootstrap-sd se gave tau 0, but the
# bootstrap sd of a ratio estimator is outlier-inflated. Two robust checks:
#   (1) EB with robust se ((p84 - p16) / 2 of the bootstrap) ;
#   (2) split-half reliability: correlation ACROSS directions of the gap measured on two random halves (400 splits), Spearman-Brown to n 58.
#       Also by subgroup (MCL-target, acute targets, non-OA all) and for the OA trio (sym partner mean) as the positive control.
import sys
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata, spearmanr
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import L, AB
OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
log = open(OUTD + 'recal_reliab.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(7)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
D = CL.data(); CUR = D['CUR']; Y = D['Y58']; n = len(Y); J = {c: i for i, c in enumerate(L)}
OA = ['Medial OA', 'Lateral OA', 'PF OA']
Z = np.column_stack([ns(CUR[:, j]) for j in range(12)])
DIRS = [(t, o) for t in L for o in L if o != t and not (t in OA and o in OA)]


def gaps(ii):
    out = np.full(len(DIRS), np.nan)
    for k, (t, o) in enumerate(DIRS):
        y = Y[ii, J[t]].astype(float)
        if y.min() == y.max(): continue
        zt, zo = Z[ii, J[t]], Z[ii, J[o]]
        ct = np.cov(zt, y)[0, 1]
        if abs(ct) < 1e-9: continue
        out[k] = np.corrcoef(zt, zo)[0, 1] - np.cov(zo, y)[0, 1] / ct
    return out


g_full = gaps(np.arange(n))
B = np.array([gaps(rng.integers(0, n, n)) for _ in range(1000)])
se_sd = np.nanstd(B, 0); se_rob = (np.nanpercentile(B, 84, 0) - np.nanpercentile(B, 16, 0)) / 2
mu = np.mean(g_full)
for nm, se in (('bootstrap sd', se_sd), ('robust (p84-p16)/2', se_rob)):
    from scipy.optimize import minimize_scalar
    nll = lambda lt: .5 * np.sum(np.log(np.exp(2 * lt) + se ** 2) + (g_full - mu) ** 2 / (np.exp(2 * lt) + se ** 2))
    tau = float(np.exp(minimize_scalar(nll, bounds=(-8, 0), method='bounded').x))
    P('EB (%s): n dirs %d, obs sd %.3f, rms se %.3f, median se %.3f -> tau ML %.3f, median shrink weight %.2f' % (
        nm, len(DIRS), g_full.std(), np.sqrt(np.mean(se ** 2)), np.median(se), tau, np.median(tau ** 2 / (tau ** 2 + se ** 2))))

# split-half reliability across directions
groups = {'all non-OA-internal': np.ones(len(DIRS), bool),
          'MCL target': np.array([t == 'MCL' for t, o in DIRS]),
          'acute target (MCL, Frac, Cont)': np.array([t in ('MCL', 'Fracture', 'Contusion') for t, o in DIRS]),
          'OA target, non-OA partner': np.array([t in OA for t, o in DIRS]),
          'non-OA target': np.array([t not in OA for t, o in DIRS])}
R = {k: [] for k in groups}; RS = {k: [] for k in groups}
for s in range(400):
    p = rng.permutation(n); ga = gaps(p[:29]); gb = gaps(p[29:])
    for k, m in groups.items():
        ok = m & ~np.isnan(ga) & ~np.isnan(gb)
        if ok.sum() > 5:
            R[k].append(np.corrcoef(ga[ok], gb[ok])[0, 1]); RS[k].append(spearmanr(ga[ok], gb[ok]).correlation)
for k in groups:
    r = np.nanmean(R[k]); rs = np.nanmean(RS[k]); sb = 2 * r / (1 + r) if r > -1 else np.nan
    P('split-half reliability %-32s n %3d | Pearson r_half %+.3f (p10 %+.2f, p90 %+.2f) Spearman %+.3f | Spearman-Brown full-n reliability %+.3f' % (
        k, groups[k].sum(), r, np.nanpercentile(R[k], 10), np.nanpercentile(R[k], 90), rs, sb))

# positive control: OA trio sym partner-mean gaps, split-half stability (3 numbers only: report the per-target half-to-half sd)
def trio_gap(ii):
    out = []
    for t in OA:
        o = [k for k in OA if k != t]; y = Y[ii, J[t]].astype(float); zt = Z[ii, J[t]]; m = .5 * (Z[ii, J[o[0]]] + Z[ii, J[o[1]]])
        out.append(np.cov(zt, m)[0, 1] / np.var(zt) - np.cov(m, y)[0, 1] / np.cov(zt, y)[0, 1])
    return np.array(out)
TG = np.array([trio_gap(rng.integers(0, n, n)) for _ in range(1000)])
P('OA trio (eoa20 base!) sym gap full %s | boot sd %s | mean-of-3 %.3f sd %.3f (robust %.3f)' % (
    np.round(trio_gap(np.arange(n)), 3), np.round(TG.std(0), 3), trio_gap(np.arange(n)).mean(), TG.mean(1).std(),
    (np.percentile(TG.mean(1), 84) - np.percentile(TG.mean(1), 16)) / 2))
# report the candidate directions
Df = pd.DataFrame(dict(dir=['%s <- %s' % (AB[t], AB[o]) for t, o in DIRS], gap=g_full, se_sd=se_sd, se_rob=se_rob))
for d in ('MCL <- MM', 'Frac <- MOA', 'LOA <- Eff', 'MCL <- Syn', 'MCL <- Cont', 'MCL <- MOA', 'LOA <- MM', 'LM <- MM', 'MM <- LM', 'Syn <- Eff',
          'Eff <- Syn', 'LOA <- LM', 'MOA <- MM', 'ACL <- LM', 'MCL <- Frac', 'Cont <- Frac', 'Frac <- Cont'):
    r = Df[Df.dir == d].iloc[0]
    P('  %-13s gap %+.3f  se(sd) %.3f  se(robust) %.3f  z_rob %+.2f' % (d, r.gap, r.se_sd, r.se_rob, (r.gap - mu) / r.se_rob))
Df.round(4).to_csv(OUTD + 'recal_reliab_dirs.csv', index=False)
P('RECAL_RELIAB DONE')
