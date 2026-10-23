# skep_mcl_relabel.py (10-21 ~00:35 IST, Claude, STACK SKEPTIC round). RESEARCH ONLY, local CPU, NO OAI, NO box.
# RAK (inventory) REWEIGHTS train knees to the gold (MCL, MM) 2x2. The claimed mechanism is different: experts DROP report-MCL+ calls
# (chronic / low-grade), and the drops sit in MM+ knees. A report-trained model keeps scoring the dropped knees high, so they become
# high-scoring NEGATIVES on the hidden labels, which MCLmm pushes down. This script emulates that RELABEL mechanism directly:
#   pseudo-expert MCL = report MCL with each report-MCL+ knee dropped with prob d1 (MM+) or d0 (MM-), overall drop fraction f,
#   drop log-odds-ratio psi = logit(d1) - logit(d0).  Expected AUC: every report-MCL+ knee enters twice, as a positive with weight (1-d)
#   and as a negative with weight d.  MCLmm gain (a .1, macro e-4) as a function of psi, on the 3 c4-shaped nests, train population and
#   gold-like population (density ratio as skep_mcl_curve.py).  Then psi is estimated from the gold58 (EXPERT) studies: report-v2 MCL+
#   knees, confirmed or not, by MM status; and the population effect is mapped through the curve.
# Writes stack/skeptic/skep_mcl_relabel.log. Aggregates only.
import sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.optimize import brentq
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import load as LD
T0 = time.time()
SK = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/'
src = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/stack_local.py', encoding='utf-8').read()
ns_ = {}
exec(src.split('summary = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open('%s_tmp_stack_local2.log', 'w')" % SK), ns_)
pipeline, wauc, BASE = ns_['pipeline'], ns_['wauc'], ns_['BASE']
log = open(SK + 'skep_mcl_relabel.log', 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


L = LD.L; J = {c: i for i, c in enumerate(L)}; G = LD.G; pct = lambda v: rankdata(v) / len(v)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); allg = set(g.StudyInstanceUID)
r7 = LD.r7(); s44, _ = LD.s6seed(1044); s45, _ = LD.s6seed(1045); lv2 = LD.labels('v2')
tr = [i for i in r7.index if i not in allg and i in lv2.index]; tr3 = [i for i in tr if i in s44.index and i in s45.index]
X = {'R7': r7.loc[tr3, ['p|' + c for c in L]].values.astype(float), 's1044': s44.loc[tr3, L].values.astype(float),
     's1045': s45.loc[tr3, L].values.astype(float)}
Y = lv2.loc[tr3, L].values.astype(float); n = len(tr3)
jm, jmm = J['MCL'], J['Medial Meniscus']
m = ~np.isnan(Y[:, [jm, jmm]]).any(1); ymcl = Y[m, jm].astype(int); ymm = Y[m, jmm].astype(int)
P('train rows with report MCL + MM: %d | report MCL+ %d (MM+ share among them %.2f)' % (m.sum(), ymcl.sum(), ymm[ymcl == 1].mean()))

# ---- gold58 (EXPERT) and gold74 crosstabs: report-v2 MCL+ knees, expert-confirmed or dropped, by expert / report MM ----
P('\n(1) gold crosstabs: among report-v2 MCL+ knees, expert MCL+ (confirmed) vs expert MCL- (dropped), by MM status')
PSI = {}
for gs in ('gold58', 'gold74'):
    gg = g if gs == 'gold74' else g[g.source == 'original_gold58']
    ids = gg.StudyInstanceUID.tolist(); E = gg[L].values.astype(float); Rr = lv2.reindex(ids)[L].values.astype(float)
    ok = ~np.isnan(E[:, jm]) & ~np.isnan(Rr[:, jm])
    P('  %s: n %d | expert MCL+ %d | report MCL+ %d | expert+ & report- %d | expert- & report+ %d' % (
        gs, ok.sum(), int(E[ok, jm].sum()), int(Rr[ok, jm].sum()), int(((E[ok, jm] == 1) & (Rr[ok, jm] == 0)).sum()), int(((E[ok, jm] == 0) & (Rr[ok, jm] == 1)).sum())))
    for mmsrc, MMv in (('expert MM', E[:, jmm]), ('report MM', Rr[:, jmm])):
        rp = ok & (Rr[:, jm] == 1) & ~np.isnan(MMv)
        a = int(((MMv == 1) & rp & (E[:, jm] == 0)).sum()); b = int(((MMv == 1) & rp & (E[:, jm] == 1)).sum())
        c = int(((MMv == 0) & rp & (E[:, jm] == 0)).sum()); d = int(((MMv == 0) & rp & (E[:, jm] == 1)).sum())
        psi = np.log((a + .5) * (d + .5) / ((b + .5) * (c + .5))); sps = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
        f = (a + c) / max(a + b + c + d, 1)
        PSI[(gs, mmsrc)] = (psi, sps, f)
        P('    by %-9s: MM+ dropped %d / confirmed %d | MM- dropped %d / confirmed %d | drop fraction %.2f | psi (drop log OR MM+ vs MM-) %+.2f [se %.2f]' % (
            mmsrc, a, b, c, d, f, psi, sps))

# ---- relabel curve ----
NESTS = {'N_A': ('R7', 's1044', 's1045'), 'N_B': ('s1044', 'R7', 's1045'), 'N_C': ('s1045', 's1044', 'R7')}
OUTS = {}
for nn, (ra, r6, s6) in NESTS.items():
    M = dict(rap=[(1., X[ra])], r6=X[r6], s6=X[s6]); B0 = pipeline(M, dict(BASE)); c = dict(BASE); c['mclmm'] = .1; B1 = pipeline(M, c)
    OUTS[nn] = (B0[m, jm], B1[m, jm])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ilg = lambda z: 1 / (1 + np.exp(-z))


def drops(psi, f, w):
    """d1 (MM+), d0 (MM-) with logit(d1) - logit(d0) = psi and weighted overall drop fraction f among report-MCL+."""
    pos = ymcl == 1; s1 = (w * pos * (ymm == 1)).sum() / (w * pos).sum()
    z0 = brentq(lambda z: s1 * ilg(z + psi) + (1 - s1) * ilg(z) - f, -20, 20)
    return ilg(z0 + psi), ilg(z0)


def gain(psi, f, w):
    d1, d0 = drops(psi, f, w); dvec = np.where(ymm == 1, d1, d0) * (ymcl == 1)
    # duplicate report-MCL+ rows: positive with weight (1 - d), negative with weight d; report-MCL- rows stay negative
    pos = ymcl == 1; out = []
    for nn, (b0, b1) in OUTS.items():
        s0 = np.r_[b0, b0[pos]]; s1_ = np.r_[b1, b1[pos]]
        y = np.r_[ymcl, np.zeros(pos.sum())]; ww = np.r_[w * np.where(pos, 1 - dvec, 1.), w[pos] * dvec[pos]]
        keep = ww > 0
        out.append((wauc(s1_[keep], y[keep], ww[keep]) - wauc(s0[keep], y[keep], ww[keep])) / 12 * 1e4)
    return np.mean(out), out, (d1, d0)


def coupling(psi, f, w):
    """implied expert-label (MCL, MM) log OR after the drops (report MM as the partner), weighted."""
    d1, d0 = drops(psi, f, w); pos = ymcl == 1
    e11 = (w * pos * (ymm == 1) * (1 - d1)).sum(); e10 = (w * pos * (ymm == 0) * (1 - d0)).sum()
    e01 = (w * (ymm == 1)).sum() - e11; e00 = (w * (ymm == 0)).sum() - e10
    return np.log(e11 * e00 / (e10 * e01)), (e11 + e10) / w.sum()


from sklearn.linear_model import LogisticRegression
gid58 = g[g.source == 'original_gold58'].StudyInstanceUID.tolist()
Ftr = lg(r7.loc[tr3, ['p|' + c for c in L]].values.astype(float)); Fg = lg(r7.loc[gid58, ['p|' + c for c in L]].values.astype(float))
mu_, sg_ = Ftr.mean(0), Ftr.std(0)
clf = LogisticRegression(C=.05, max_iter=2000).fit(np.vstack([(Ftr - mu_) / sg_, (Fg - mu_) / sg_]), np.r_[np.zeros(len(Ftr)), np.ones(len(Fg))],
                                                     sample_weight=np.r_[np.ones(len(Ftr)) * len(Fg) / len(Ftr), np.ones(len(Fg))])
p = clf.predict_proba((Ftr - mu_) / sg_)[:, 1]; wdr = p / (1 - p); wdr = np.minimum(wdr, np.percentile(wdr, 99.5)); wdr = wdr[m] / wdr[m].mean()
POPS = {'train-like': np.ones(m.sum()), 'gold58-like (density ratio C .05)': wdr}
PSIS = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
CURVE = {}
P('\n(2) relabel curve: MCLmm a .1 gain (macro e-4, nest mean [N_A, N_B, N_C]) vs drop log OR psi; drop fraction f .5 (sens .35 / .65);'
  ' implied expert (MCL, MM) log OR theta and expert MCL prevalence shown')
for popn, w in POPS.items():
    ess = w.sum() ** 2 / (w ** 2).sum()
    for f in (.5, .35, .65):
        row = []; ms = []
        for psi in PSIS:
            gm, gl, (d1, d0) = gain(psi, f, w); th, prev = coupling(psi, f, w); ms.append(gm)
            row.append('psi %+.1f: %+.2f (th %+.2f, d1 %.2f d0 %.2f)' % (psi, gm, th, d1, d0))
        CURVE[(popn, f)] = np.array(ms)
        be = np.interp(0., np.array(ms), np.array(PSIS)) if ms[0] < 0 < ms[-1] else float('nan')
        P('  %s (ESS %.0f) f %.2f | break-even psi %+.2f\n    %s' % (popn, ess, f, be, '\n    '.join(row)))

# ---- population effect under psi posteriors ----
P('\n(3) MCLmm population effect (macro e-4) = relabel curve(psi) x F ~ U(.45, .85); psi posteriors (normal) as listed; f .5; MC 200k')
rr = np.random.default_rng(5)
ps58, se58, _ = PSI[('gold58', 'expert MM')]
POST = {'gold58 as measured': (ps58, se58),
        'mechanism prior N(+1.0, 1.0) x gold58 data': ((1.0 / 1.0 + ps58 / se58 ** 2) / (1 / 1.0 + 1 / se58 ** 2), (1 / (1 / 1.0 + 1 / se58 ** 2)) ** .5),
        'skeptic prior N(0, 0.7) x gold58 data': ((0. + ps58 / se58 ** 2) / (1 / .49 + 1 / se58 ** 2), (1 / (1 / .49 + 1 / se58 ** 2)) ** .5),
        'null (drops random w.r.t. MM)': (0.0, .3)}
ARR = np.array(PSIS)
for pn, (mm_, ss_) in POST.items():
    psi = rr.normal(mm_, ss_, 200000); F = rr.uniform(.45, .85, 200000); row = []
    for popn in POPS:
        cv = CURVE[(popn, .5)]; gp = np.interp(psi, ARR, cv)
        lo, hi = psi < ARR[0], psi > ARR[-1]
        gp[lo] = cv[0] + (psi[lo] - ARR[0]) * (cv[1] - cv[0]) / (ARR[1] - ARR[0]); gp[hi] = cv[-1] + (psi[hi] - ARR[-1]) * (cv[-1] - cv[-2]) / (ARR[-1] - ARR[-2])
        v = gp * F; row.append('%s %+.2f [%+.2f, %+.2f] P>0 %.2f' % (popn, v.mean(), np.percentile(v, 10), np.percentile(v, 90), (v > 0).mean()))
    P('  psi ~ N(%+.2f, %.2f) %-44s | %s' % (mm_, ss_, pn, ' | '.join(row)))
P('SKEP_MCL_RELABEL DONE %.0f s' % (time.time() - T0))
