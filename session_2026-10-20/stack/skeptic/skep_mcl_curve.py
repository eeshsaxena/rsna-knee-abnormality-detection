# skep_mcl_curve.py (10-21 ~00:10 IST, Claude, STACK SKEPTIC round). RESEARCH ONLY, local CPU, NO OAI, NO box.
# Independent skeptic of MCLmm (MCL = pct(pct(MCL) - .1 pct(MM)) on pw30). The inventory's RAK reads rake train report labels to the
# gold74 EXPERT (MCL, MM) 2x2, i.e. they evaluate MCLmm AT the gold74 coupling (expert log OR ~ -0.95), the coupling of the very sample
# that nominated MM (census, gold58 = 58 of the 74). This script:
#   (1) the MCL partner table on gold74 (expert) vs train / gold report-v2 labels, with a bootstrap covariance of the 11 expert log ORs
#       (they share the same 11 MCL+ knees), and an Empirical-Bayes (REML) partial-pooling estimate of the MM gap (selection correction:
#       MM was the max-gap partner);
#   (2) the RAK gain of MCLmm as a FUNCTION of the target expert log OR theta (3 c4-shaped nests, train population), so any posterior on
#       the hidden coupling can be mapped to a gain, and the break-even theta found;
#   (3) the same curve on a GOLD-LIKE population: train rows importance-weighted by a density ratio (gold74 vs train, logistic on R7 OOF
#       logits), checking that the weighted model co-firing moves to gold's;
#   (4) the MCLmm population prior that results (x RAK->public factor), by posterior and population.
# Writes stack/skeptic/skep_mcl_curve.log. Aggregates only.
import sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
T0 = time.time()
SK = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/'
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_local.py', encoding='utf-8').read()
ns_ = {}
exec(src.split('summary = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open('%s_tmp_stack_local.log', 'w')" % SK), ns_)
pipeline, wauc, rake, BASE = ns_['pipeline'], ns_['wauc'], ns_['rake'], ns_['BASE']
log = open(SK + ('skep_mcl_curve_gold58.log' if sys.argv[1:2] == ['gold58'] else 'skep_mcl_curve.log'), 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


L = LD.L; J = {c: i for i, c in enumerate(L)}; G = LD.G; pct = lambda v: rankdata(v) / len(v)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); Y74 = g[L].values.astype(float); allg = set(g.StudyInstanceUID)
gid = g.StudyInstanceUID.tolist()
r7 = LD.r7(); s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]; tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
     's1045': s45.loc[tr3, L].values.astype(float)}
Y = lv2.loc[tr3, L].values.astype(float); n = len(tr3); YF = np.nan_to_num(Y, nan=0.)
jm, jmm = J['MCL'], J['Medial Meniscus']
P('rows: train tr3 %d | gold74 %d (gold studies with report v2: %d)' % (n, len(gid), sum(i in lv2.index for i in gid)))
YG_rep = None


def lor2(a, b, eps=.5):
    ok = ~(np.isnan(a) | np.isnan(b)); a = a[ok].astype(int); b = b[ok].astype(int)
    T = np.zeros((2, 2))
    for x, y in zip(a, b): T[x, y] += 1
    Tq = T + eps
    return np.log(Tq[1, 1] * Tq[0, 0] / (Tq[1, 0] * Tq[0, 1])), np.sqrt((1 / Tq).sum()), T


# ---------------- (1) partner table + EB pooling ----------------
partners = [c for c in L if c != 'MCL']
GSET = sys.argv[1] if len(sys.argv) > 1 else 'gold74'
if GSET == 'gold58':   # HANDOFF 1795: the 16 extra gold74 studies were labelled by READING REPORTS; only gold58 is expert-labelled
    keep = (g.source == 'original_gold58').values; g = g[keep].reset_index(drop=True); Y74 = Y74[keep]; gid = g.StudyInstanceUID.tolist()
P('GOLD SET %s: %d studies, MCL+ %d' % (GSET, len(gid), int(np.nansum(Y74[:, jm]))))
YG_rep = lv2.reindex(gid)[L].values.astype(float)
P('\n(1) MCL partner table: expert %s log OR (+.5) [se], its report-v2 log OR, train report-v2 log OR, gap = train report - expert' % GSET)
ex, se, trl, grl = {}, {}, {}, {}
for c in partners:
    ex[c], se[c], T = lor2(Y74[:, jm], Y74[:, J[c]]); trl[c], _, _ = lor2(Y[:, jm], Y[:, J[c]]); grl[c], _, _ = lor2(YG_rep[:, jm], YG_rep[:, J[c]])
    P('  %-17s expert %+.2f [%.2f] (MCL+X+ %d, MCL+X- %d, MCL-X+ %d, MCL-X- %d) | gold report %+.2f | train report %+.2f | gap %+.2f' % (
        c, ex[c], se[c], T[1, 1], T[1, 0], T[0, 1], T[0, 0], grl[c], trl[c], trl[c] - ex[c]))
gap = np.array([trl[c] - ex[c] for c in partners])
rng = np.random.default_rng(2110)
B = 4000; EXB = np.zeros((B, len(partners)))
for b in range(B):
    ii = rng.integers(0, len(gid), len(gid)); Yb = Y74[ii]
    EXB[b] = [lor2(Yb[:, jm], Yb[:, J[c]])[0] for c in partners]
S = np.cov(EXB.T)   # covariance of the expert log ORs (= of the gaps; train report lor is ~fixed, n 3,467)
sd = np.sqrt(np.diag(S)); Rr = S / np.outer(sd, sd)
off = Rr[~np.eye(len(partners), dtype=bool)]
P('  bootstrap (B %d over the 74 gold studies): mean pairwise corr of the 11 expert log ORs %.2f [min %.2f, max %.2f]; corr(MM, others) mean %.2f' % (
    B, off.mean(), off.min(), off.max(), np.delete(Rr[partners.index('Medial Meniscus')], partners.index('Medial Meniscus')).mean()))


def reml(gv, Sv, grid=np.linspace(0, 1.5, 301)):
    k = len(gv); one = np.ones(k); out = []
    for t2 in grid:
        V = Sv + t2 ** 2 * np.eye(k); Vi = np.linalg.inv(V); w = one @ Vi @ one; c = (one @ Vi @ gv) / w; r = gv - c
        ll = -.5 * (np.linalg.slogdet(V)[1] + np.log(w) + r @ Vi @ r); out.append((ll, t2, c))
    ll = np.array([o[0] for o in out]); post = np.exp(ll - ll.max()); post /= post.sum()
    return grid, post, np.array([o[2] for o in out])


def pooled(sel, tag):
    idx = [partners.index(c) for c in sel]; gv = gap[idx]; Sv = S[np.ix_(idx, idx)]; k = len(idx); one = np.ones(k)
    grid, post, cs = reml(gv, Sv); tml = grid[post.argmax()]; im = sel.index('Medial Meniscus')
    # posterior of the MM gap, integrating tau over its (flat-prior) REML posterior; c ~ GLS with its variance
    mm_m, mm_v = [], []
    for t2, p in zip(grid, post):
        V = Sv + t2 ** 2 * np.eye(k); Vi = np.linalg.inv(V); w = one @ Vi @ one; c = (one @ Vi @ gv) / w; vc = 1 / w
        # BLUP of the MM true gap = c + d_MM, d | data ~ N(t2 Vi (g - c), t2 I - t2^2 Vi) (conditional on c), + var(c)
        dm = t2 ** 2 * (Vi @ (gv - c))[im]; dv = t2 ** 2 - t2 ** 4 * Vi[im, im]
        lam = t2 ** 2 * (Vi @ one)[im]   # d depends on c too: d_MM = lam*(-c) + ... ; total = c(1 - lam) + t2^2 (Vi g)_MM
        mm_m.append(c + dm); mm_v.append(max(dv, 0) + (1 - lam) ** 2 * vc)
    mm_m, mm_v = np.array(mm_m), np.array(mm_v)
    m = (post * mm_m).sum(); v = (post * (mm_v + mm_m ** 2)).sum() - m ** 2
    P('  EB %-28s k %2d | raw MM gap %+.2f | pooled mean c(ML tau) %+.2f | tau ML %.2f, P(tau < .2) %.2f | MM true gap posterior %+.2f (sd %.2f)' % (
        tag, k, gap[partners.index('Medial Meniscus')], cs[post.argmax()], tml, post[grid < .2].sum(), m, np.sqrt(v)))
    return m, np.sqrt(v)


P('\n  EB partial pooling of the gaps (REML over tau, bootstrap covariance; selection correction for picking the max-gap partner):')
EB_all = pooled(partners, 'all 11 partners')
DEG = ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's"]
EB_deg = pooled(DEG, 'degenerative 8 (MM, LM, OA x3, Eff, Syn, Bak)')
mmx = partners.index('Medial Meniscus')
P('  MM: expert %+.2f [%.2f], train report %+.2f -> theta (hidden expert lor) as measured %+.2f; EB-all %+.2f (sd %.2f); EB-deg %+.2f (sd %.2f)' % (
    ex['Medial Meniscus'], sd[mmx], trl['Medial Meniscus'], ex['Medial Meniscus'], trl['Medial Meniscus'] - EB_all[0], EB_all[1],
    trl['Medial Meniscus'] - EB_deg[0], EB_deg[1]))
# extra16 alone (outside the nominating gold58)
e16 = (g.source != 'original_gold58').values
if e16.any():
    l16, s16, T16 = lor2(Y74[e16, jm], Y74[e16, jmm]); l58, s58, T58 = lor2(Y74[~e16, jm], Y74[~e16, jmm])
    P('  split: gold58 expert MCL~MM lor %+.2f [%.2f] table %s | extra16 (REPORT-read labels, HANDOFF 1795) %+.2f [%.2f] table %s (MCL+ in extra16: %d)' % (
        l58, s58, T58.astype(int).tolist(), l16, s16, T16.astype(int).tolist(), int(np.nansum(Y74[e16, jm]))))

# ---------------- (2) RAK gain curve vs theta ----------------
NESTS = {'N_A': ('R7', 's1044', 's1045'), 'N_B': ('s1044', 'R7', 's1045'), 'N_C': ('s1045', 's1044', 'R7')}
cols = ['MCL', 'Medial Meniscus']; mrow = ~np.isnan(Y[:, [jm, jmm]]).any(1); Ys = Y[mrow][:, [jm, jmm]].astype(int)
THETAS = [-1.5, -1.25, -0.95, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.91]


def wts_theta(theta, base_w):
    obs = np.zeros((2, 2))
    for (a, b), w in zip(Ys, base_w): obs[a, b] += w
    obs /= obs.sum(); marg = np.array([(base_w * Ys[:, 0]).sum(), (base_w * Ys[:, 1]).sum()]) / base_w.sum()
    T = np.array([[1., 1.], [1., np.exp(theta)]]); tgt = rake(T / T.sum(), marg, it=300)
    w = base_w * (tgt / obs)[tuple(Ys.T)]
    return w / w.mean()


OUTS = {}
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); B0 = pipeline(M, dict(BASE)); c = dict(BASE); c['mclmm'] = .1; B1 = pipeline(M, c)
    OUTS[nn] = (B0, B1)


def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    return np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w))


def curve(base_w, tag):
    P('\n(2%s) RAK gain of MCLmm a .1 vs target expert (MCL, MM) log OR theta, macro e-4 (label / 12); population: %s' % ('' if tag == 'train' else 'g', tag))
    res = {}
    for nn, (B0, B1) in OUTS.items():
        row = []
        for th in THETAS:
            w = wts_theta(th, base_w)
            d = (wauc(B1[mrow, jm], YF[mrow, jm], w) - wauc(B0[mrow, jm], YF[mrow, jm], w)) / 12 * 1e4
            res[(nn, th)] = d; row.append('%+.2f:%+.2f' % (th, d))
        cof = wcorr(pct(B0[mrow, jm]), pct(B0[mrow, jmm]), base_w)
        P('  %s (co-firing MCL~MM %.3f): %s' % (nn, cof, ' | '.join(row)))
    mean = {th: np.mean([res[(nn, th)] for nn in OUTS]) for th in THETAS}
    # break-even theta by linear interpolation of the nest-mean curve
    ths = np.array(THETAS); ms = np.array([mean[t] for t in THETAS]); be = np.interp(0., ms[::-1], ths[::-1])
    P('  nest mean: ' + ' | '.join('%+.2f:%+.2f' % (t, mean[t]) for t in THETAS) + ' | break-even theta %+.2f' % be)
    return ths, ms


ths, ms_train = curve(np.ones(mrow.sum()), 'train')

# ---------------- (3) gold-like population by density ratio on R7 OOF logits ----------------
from sklearn.linear_model import LogisticRegression
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
Ftr = lg(r7.loc[tr3, ['p|' + c for c in L]].values.astype(float)); Fg = lg(r7.loc[gid, ['p|' + c for c in L]].values.astype(float))
mu_, sg_ = Ftr.mean(0), Ftr.std(0)
Xd = np.vstack([(Ftr - mu_) / sg_, (Fg - mu_) / sg_]); yd = np.r_[np.zeros(len(Ftr)), np.ones(len(Fg))]
RESG = {}
for C in (.05, .3):
    clf = LogisticRegression(C=C, max_iter=2000).fit(Xd, yd, sample_weight=np.r_[np.ones(len(Ftr)) * len(Fg) / len(Ftr), np.ones(len(Fg))])
    p = clf.predict_proba((Ftr - mu_) / sg_)[:, 1]; wdr = p / (1 - p); wdr = np.minimum(wdr, np.percentile(wdr, 99.5)); wdr /= wdr.mean()
    ess = wdr.sum() ** 2 / (wdr ** 2).sum()
    r7g = np.corrcoef(pct(Fg[:, jm]), pct(Fg[:, jmm]))[0, 1]; r7t = np.corrcoef(pct(Ftr[:, jm]), pct(Ftr[:, jmm]))[0, 1]
    r7w = wcorr(pct(Ftr[:, jm]), pct(Ftr[:, jmm]), wdr)
    # report-label coupling under the gold-like weights vs gold74 reports
    lw = lambda a, b, w: np.log(((w * (a == 1) * (b == 1)).sum() + .5) * ((w * (a == 0) * (b == 0)).sum() + .5) / (((w * (a == 1) * (b == 0)).sum() + .5) * ((w * (a == 0) * (b == 1)).sum() + .5)))
    okr = ~np.isnan(Y[:, [jm, jmm]]).any(1)
    P('\n(3) density ratio C %.2f: ESS %.0f of %d | R7 OOF label-free MCL~MM (rank corr): train %.3f, gold74 %.3f, train reweighted %.3f | '
      'report-v2 MCL~MM lor: train %+.2f, reweighted %+.2f, gold74 reports %+.2f | c4 gold58 (stack_rak2) .186' % (
          C, ess, len(wdr), r7t, r7g, r7w, trl['Medial Meniscus'], lw(Y[okr, jm], Y[okr, jmm], wdr[okr] * len(okr) / okr.sum()) if True else 0, grl['Medial Meniscus']))
    RESG[C] = curve(wdr[mrow], 'gold-like (density ratio C %.2f, ESS %.0f)' % (C, ess))[1]

# ---------------- (4) MCLmm population prior by posterior on theta ----------------
P('\n(4) MCLmm population effect (macro e-4) = RAK curve(theta) x public/RAK factor F ~ U(.45, .85) (RAK ladder .63-.78, RAK-u p10 .35);'
  ' theta posteriors as listed; Monte Carlo 200k')
rr = np.random.default_rng(99)
POST = {'as measured (%s lor)' % GSET: (ex['Medial Meniscus'], sd[mmx]),
        'EB pooled, degenerative 8': (trl['Medial Meniscus'] - EB_deg[0], EB_deg[1]),
        'EB pooled, all 11': (trl['Medial Meniscus'] - EB_all[0], EB_all[1]),
        'recal reliability .01 (gap kept 10 %)': (trl['Medial Meniscus'] - .1 * gap[mmx], .6)}
POPS = {'train-like': ms_train, 'gold-like C .05': RESG[.05], 'gold-like C .30': RESG[.3]}
for pn, (m, s) in POST.items():
    th = rr.normal(m, s, 200000); F = rr.uniform(.45, .85, 200000); row = []
    for popn, ms in POPS.items():
        gth = np.interp(th, ths, ms)
        lo, hi = th < ths[0], th > ths[-1]
        gth[lo] = ms[0] + (th[lo] - ths[0]) * (ms[1] - ms[0]) / (ths[1] - ths[0])
        gth[hi] = ms[-1] + (th[hi] - ths[-1]) * (ms[-1] - ms[-2]) / (ths[-1] - ths[-2])
        v = gth * F; row.append('%s %+.2f [%+.2f, %+.2f] P>0 %.2f' % (popn, v.mean(), np.percentile(v, 10), np.percentile(v, 90), (v > 0).mean()))
    P('  theta ~ N(%+.2f, %.2f) %-38s | %s' % (m, s, pn, ' | '.join(row)))
P('SKEP_MCL_CURVE DONE %.0f s' % (time.time() - T0))
