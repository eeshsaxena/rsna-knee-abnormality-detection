# skeptic_g58.py (10-20 push, skeptic pass, Claude). RESEARCH ONLY, local CPU, competition files only (gold58 expert labels +
# the c4 / eoa20 cell-18 rebuild from cbase.py). No OAI data. Prints aggregates only.
# Checks:
#  A. 2x2 tables behind the five gold58 gaps beyond 2 bootstrap sd (zero cells break the bootstrap sd of a tetrachoric).
#  B. Parametric null for all 66 pairs: latent normal with the c4 normal-score correlation, gold58 marginals, n 58 ->
#     per-pair p of the observed tetrachoric; count of |p| < .0455; JOINT null (all 12 columns at once) for the count
#     and for "k exceedances sharing one label".
#  C. Pair-flip arithmetic of the census candidates (d x npos x nneg).
#  D. MCL <- MM influence: per-positive contribution, then drop the carrying positives and recompute MCL gaps and d.
import sys, itertools
import numpy as np
from scipy.stats import rankdata, norm
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
from census_lib import tet, auc, spear, ns, pct, L, AB, load
rng = np.random.default_rng(1020)
d = load(); F = d['eoa20']; C4 = d['c4']; Y = d['Y58']; n = len(Y); J = {c: j for j, c in enumerate(L)}
ab = [AB[c] for c in L]
print('n', n, 'prevalence', {AB[c]: int(Y[:, j].sum()) for j, c in enumerate(L)})
print('c4 gold58 AUC', {AB[c]: round(auc(C4[:, j], Y[:, j]), 4) for j, c in enumerate(L)})

# ---------- A. 2x2 tables
print('\n== A. 2x2 tables (c00 c01 c10 c11) and gold58 tet / c4 rho')
for a_, b_ in [('MCL', 'Baker\'s'), ('MCL', 'Medial OA'), ('MCL', 'Medial Meniscus'), ('MCL', 'Effusion'), ('Contusion', 'Fracture'),
               ('MCL', 'Synovitis'), ('MCL', 'Contusion'), ('Medial OA', 'Fracture'), ('Lateral OA', 'Effusion')]:
    x, y = Y[:, J[a_]], Y[:, J[b_]]
    c = (int(((x == 0) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()), int(((x == 1) & (y == 1)).sum()))
    print('  %-5s-%-5s table %s tet %+.2f c4 rho %+.2f' % (AB[a_], AB[b_], c, tet(x, y), spear(C4[:, J[a_]], C4[:, J[b_]])))

# bootstrap behaviour of tet for a zero cell (does resampling ever create a co-occurrence? no)
x, y = Y[:, J['MCL']], Y[:, J["Baker's"]]
bt = []
for _ in range(1000):
    ii = rng.integers(0, n, n); bt.append(tet(x[ii], y[ii]))
bt = np.array(bt)
print('  MCL-Bak bootstrap tet: mean %+.3f sd %.3f p05/p95 %+.2f/%+.2f (the c11 cell stays 0 in every resample)' % (bt.mean(), bt.std(), *np.quantile(bt, [.05, .95])))

# ---------- B. parametric null
Z = np.column_stack([ns(C4[:, j]) for j in range(12)]); R = np.corrcoef(Z.T)
prev = Y.mean(0); th = norm.ppf(1 - prev)
pairs = list(itertools.combinations(range(12), 2))
NS = 4000
Lc = np.linalg.cholesky(R + 1e-9 * np.eye(12))
TS = np.zeros((NS, len(pairs)))
for s in range(NS):
    U = rng.standard_normal((n, 12)) @ Lc.T
    # keep the gold58 marginals exactly: label the top-k of each latent column positive
    Yb = np.zeros((n, 12), int)
    for j in range(12):
        k = int(Y[:, j].sum()); Yb[np.argsort(-U[:, j])[:k], j] = 1
    for p_, (i, j) in enumerate(pairs):
        TS[s, p_] = tet(Yb[:, i], Yb[:, j])
obs = np.array([tet(Y[:, i], Y[:, j]) for i, j in pairs]); rho = np.array([spear(C4[:, i], C4[:, j]) for i, j in pairs])
# two-sided parametric p: excess side = P(tet_null <= obs), deficit side = P(tet_null >= obs)
plo = (TS <= obs[None, :] + 1e-12).mean(0); phi_ = (TS >= obs[None, :] - 1e-12).mean(0)
p2 = np.minimum(1, 2 * np.minimum(plo, phi_))
print('\n== B. parametric null (latent corr = c4 normal-score corr, gold58 marginals fixed, n 58, %d sims)' % NS)
o = np.argsort(p2)
for k in o[:10]:
    i, j = pairs[k]
    print('  %-5s-%-5s obs tet %+.2f | c4 rho %+.2f | null tet mean %+.2f sd %.2f | side %s | two-sided p %.4f' % (
        ab[i], ab[j], obs[k], rho[k], TS[:, k].mean(), TS[:, k].std(), 'excess' if plo[k] < phi_[k] else 'deficit', p2[k]))
nobs = int((p2 < .0455).sum())
print('  observed count p < .0455: %d of 66 (excess %d)' % (nobs, int(((p2 < .0455) & (plo < phi_)).sum())))
# joint null for the count and for the per-label max count (each sim scored against the per-pair null CDF)
srt = np.sort(TS, 0)
def pv(v, k):
    lo = np.searchsorted(srt[:, k], v + 1e-12, 'right') / NS; hi = 1 - np.searchsorted(srt[:, k], v - 1e-12, 'left') / NS
    return np.minimum(1, 2 * np.minimum(lo, hi))
PS = np.column_stack([pv(TS[:, k], k) for k in range(len(pairs))])
cnt = (PS < .0455).sum(1)
lab_cnt = np.zeros((NS, 12), int)
for k, (i, j) in enumerate(pairs):
    hit = PS[:, k] < .0455; lab_cnt[:, i] += hit; lab_cnt[:, j] += hit
mx = lab_cnt.max(1)
obs_lab = np.zeros(12, int)
for k, (i, j) in enumerate(pairs):
    if p2[k] < .0455: obs_lab[i] += 1; obs_lab[j] += 1
print('  joint null: count mean %.2f sd %.2f | P(count >= %d) %.3f | P(count >= 5) %.3f' % (cnt.mean(), cnt.std(), nobs, (cnt >= nobs).mean(), (cnt >= 5).mean()))
print('  joint null: max exceedances sharing one label: mean %.2f | P(max >= %d) %.3f | P(max >= 4) %.3f | observed per label %s' % (
    mx.mean(), obs_lab.max(), (mx >= obs_lab.max()).mean(), (mx >= 4).mean(), {ab[j]: int(obs_lab[j]) for j in range(12) if obs_lab[j]}))
from scipy.stats import poisson
print('  independent-test reference: Poisson(3.0) P(>=5) %.3f' % poisson.sf(4, 3.0))

# ---------- C. pair-flip arithmetic
print('\n== C. pair flips behind the gold58 single-label d (1 discordant pair = 1/(npos*nneg))')
for t, o_, a, dd in [('MCL', 'Medial Meniscus', .1, .0091), ('Fracture', 'Medial OA', .1, .0118), ('Lateral OA', 'Effusion', -.1, .0116),
                     ('MCL', None, .1, .0023)]:
    y = Y[:, J[t]]; p = int(y.sum()); q = n - p
    base = auc(F[:, J[t]], y)
    if o_ is None:
        oth = np.mean([pct(F[:, j]) for j in range(12) if j != J[t]], 0); new = pct(pct(F[:, J[t]]) - a * pct(oth))
    else:
        new = pct(pct(F[:, J[t]]) - a * pct(F[:, J[o_]]))
    dn = auc(new, y) - base
    print('  %-4s <- %-5s a %+.1f | npos %d nneg %d | d %+.4f (census %+.4f) = %.1f pair flips of %d | base AUC %.4f' % (
        AB[t], AB[o_] if o_ else 'mean11', a, p, q, dn, dd, dn * p * q, p * q, base))

# ---------- D. MCL <- MM influence
t, o_ = J['MCL'], J['Medial Meniscus']; y = Y[:, t]
base = F[:, t]; new = pct(pct(F[:, t]) - .1 * pct(F[:, o_]))
pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
def below(s, i): return ((s[neg] < s[i]).sum() + .5 * (s[neg] == s[i]).sum())
contrib = np.array([below(new, i) - below(base, i) for i in pos])
print('\n== D. MCL <- MM a .1: per-positive change in negatives ranked below (sum = %.1f pairs)' % contrib.sum())
for i, cc in sorted(zip(pos, contrib), key=lambda z: -z[1]):
    print('   pos %2d | MM label %d | MOA %d | Bak %d | Eff %d | c4 MCL pct %.2f | MM pct %.2f | delta pairs %+.1f' % (
        i, Y[i, o_], Y[i, J['Medial OA']], Y[i, J["Baker's"]], Y[i, J['Effusion']], pct(F[:, t])[i], pct(F[:, o_])[i], cc))
top = [i for i, cc in sorted(zip(pos, contrib), key=lambda z: -z[1]) if cc > 0]
print('  MCL positives with MM label 1: %d of %d' % (int(Y[pos, o_].sum()), len(pos)))
for drop in (top[:1], top[:2], top[:3]):
    keep = np.setdiff1d(np.arange(n), drop)
    Yk, Fk, Ck = Y[keep], F[keep], C4[keep]
    gaps = []
    for oo in ('Medial Meniscus', "Baker's", 'Medial OA', 'Effusion', 'Synovitis', 'Contusion'):
        gaps.append('%s %+.2f' % (AB[oo], spear(Ck[:, t], Ck[:, J[oo]]) - tet(Yk[:, t], Yk[:, J[oo]])))
    dk = auc(pct(pct(Fk[:, t]) - .1 * pct(Fk[:, o_])), Yk[:, t]) - auc(Fk[:, t], Yk[:, t])
    print('  drop %d carrying positives -> MCL gaps (c4 rho - tet): %s | MCL <- MM d %+.4f' % (len(drop), ' '.join(gaps), dk))
