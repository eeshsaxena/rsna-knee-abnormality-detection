# o1_lm_oai.py (10-20 night, Track 3 LM / Fracture, Claude). BOX ONLY, read-only, CPU, prints AGGREGATES ONLY
# (n, prevalence, means of percentile ranks, AUC, AUC deltas, sd, P, phi, Spearman). Never prints an id, a row or a per-knee value;
# writes nothing. Run: ssh training@192.168.10.30 'nice -n 5 ~/miniconda3/envs/rsna/bin/python -I -' < o1_lm_oai.py
# Inputs: R6 and S6fall served on the store-rebuilt OAI V00 DICOMs (~/r5_oai, r6_oai.md), D5 OAI-free fleet reference,
# MOAKS meniscal morphology (~/oai_build/oai_label_sources.csv), T123 labels (~/oai_build/oai_labels_T123.csv).
# Questions (Track 3): (1) which LM morphologies do the c4 OAI-free members miss / over-call; is intrasubstance signal ranked
# like a tear (rubric: degeneration not reaching the surface is NEGATIVE)? (2) is LM less reliable in trauma-context knees?
# (3) candidate output-only LM fixes on this independent expert ruler (pre-registered below, written before any output).
# PRE-REGISTRATION (rule written before the run): a candidate is SUPPORTED on OAI only if, on the primary label B1, its
# LM AUC delta is > 0 with paired-bootstrap P>0 >= .90 for R6, S6fall AND the c4-shaped proxy (D5 in the Raptor slot);
# KILLED if <= 0 or P>0 < .5 for 2+ of the 3. Candidates: C1 trauma-context shrink k .2 / .4; C2 ACL-context shrink k .4;
# C3 LM <- ACL decor a .1; C4 LM <- LOA a .1 and its borrow a -.1; C5 S6fall share for LM (R6/S6 only; descriptive curve).
import os, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
H = os.path.expanduser('~')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
B = int(os.environ.get('BOOT', '1000'))
pct = lambda v: rankdata(v) / len(v)


def auc(s, y, w=None):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if w is None:
        r = rankdata(s); p = y.sum(); q = len(y) - p
        return np.nan if p == 0 or q == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]   # weighted Mann-Whitney with ties
    _, inv = np.unique(s, return_inverse=True)
    wn = np.bincount(inv, w * (1 - y)); wp = np.bincount(inv, w * y)
    cn = np.cumsum(wn) - wn
    return float((wp * (cn + .5 * wn)).sum() / (wp.sum() * wn.sum()))


def rd(p):
    d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    return d.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[LAB].astype(float)


R6 = rd(H + '/r5_oai/run_r6/worker0/predictions.csv'); S6 = rd(H + '/r5_oai/s6_r5/out/predictions.csv')
D5 = rd(H + '/oai_build/teacher/teacher_D5_mean_v00.csv')
oai = lambda P: P[P.index.str.startswith('oai_')]
R6, S6 = oai(R6), oai(S6)
src = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
T = pd.read_csv(H + '/oai_build/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = R6.dropna().index.intersection(S6.dropna().index).intersection(D5.dropna().index).intersection(src.index).intersection(T.index)
lmm = src.loc[ids, ['m_lmah_morph', 'm_lmbody_morph', 'm_lmph_morph']].astype(float)
root = src.loc[ids, 'm_lmpr_tear'].astype(float)
ok = lmm.notna().any(axis=1).values
ids = ids[ok]; lmm = lmm.loc[ids]; root = root.loc[ids]
n = len(ids)
mx = lmm.max(axis=1).values
tear = (((lmm.values >= 2) & (lmm.values <= 8)).any(1) | (root.values == 1))
sig_only = (~tear) & (mx == 1)
normal = (~tear) & (mx == 0)
yB1 = tear.astype(int)
print('OAI knees with R6 + S6fall + D5 + LM morph read: n %d | B1 tear prev %.3f | signal-only %.3f | normal %.3f' % (n, yB1.mean(), sig_only.mean(), normal.mean()))
M = {'R6': R6.loc[ids], 'S6fall': S6.loc[ids], 'D5ref': D5.loc[ids]}
P = {k: {c: pct(v[c].values) for c in LAB} for k, v in M.items()}
# c4-shaped proxy for LM / any label: .8 pct(w_rap * pct(D5) + (1 - w_rap) * pct(R6)) + .2 pct(S6); LM w_rap .4 (Raptor slot -> D5)
c4w = {'Lateral Meniscus': .4, 'Lateral OA': .2, 'PF OA': .5, "Baker's": .5, 'Synovitis': .8, 'Fracture': .65}
P['c4D5'] = {c: .8 * pct(c4w.get(c, .5) * P['D5ref'][c] + (1 - c4w.get(c, .5)) * P['R6'][c]) + .2 * P['S6fall'][c] for c in LAB}
P['R6.8S6.2'] = {c: .8 * P['R6'][c] + .2 * P['S6fall'][c] for c in LAB}
MODELS = ['R6', 'S6fall', 'R6.8S6.2', 'c4D5', 'D5ref']

# ---------------- (1) morphology table
cats = {'normal (0)': normal, 'signal only (1)': sig_only}
anyc = lambda codes: ((np.isin(lmm.values, codes)).any(1))
cats['tear code 2 (any region)'] = anyc([2]); cats['tear code 3 horizontal'] = anyc([3]); cats['tear code 4'] = anyc([4])
cats['tear code 5'] = anyc([5]); cats['maceration 6-8'] = anyc([6, 7, 8]); cats['root tear'] = root.values == 1
cats['tear, no maceration'] = tear & ~anyc([6, 7, 8])
ph = lmm['m_lmph_morph'].values; ah = lmm['m_lmah_morph'].values; bd = lmm['m_lmbody_morph'].values
tr = lambda v: (v >= 2) & (v <= 8)
cats['tear in posterior horn'] = tr(ph); cats['tear anterior horn only'] = tr(ah) & ~tr(ph) & ~tr(bd); cats['tear body, not PH'] = tr(bd) & ~tr(ph)
print('\n(1) LM output by MOAKS morphology: n | mean pct of the LM output per model | AUC of category vs NORMAL knees')
print('%-28s %5s ' % ('category', 'n') + ' '.join('%17s' % m for m in MODELS))
for cn, m in cats.items():
    s = '%-28s %5d ' % (cn, m.sum())
    for k in MODELS:
        v = P[k]['Lateral Meniscus']
        a = auc(np.r_[v[m], v[normal]], np.r_[np.ones(m.sum()), np.zeros(normal.sum())]) if cn != 'normal (0)' and m.sum() else np.nan
        s += '   %.3f / %5s' % (v[m].mean(), ('%.3f' % a) if a == a else '  -  ')
    print(s)
print('Key AUCs per model (B1 = tear vs all; TvS = tear vs signal-only; TvN = tear vs normal; SvN = signal-only vs normal):')
for k in MODELS:
    v = P[k]['Lateral Meniscus']
    tv = lambda a, b: auc(np.r_[v[a], v[b]], np.r_[np.ones(a.sum()), np.zeros(b.sum())])
    print('  %-9s B1 %.4f | TvS %.4f | TvN %.4f | SvN %.4f | tear-no-mac vs signal %.4f' % (k, auc(v, yB1), tv(tear, sig_only), tv(tear, normal), tv(sig_only, normal), tv(cats['tear, no maceration'], sig_only)))

# ---------------- (2) strata
lft = src.loc[ids, ['m_cm_clf', 'm_cm_plf', 'm_cm_alt', 'm_cm_clt', 'm_cm_plt']].astype(float).values
area = np.floor(lft + 1e-6); ft = np.round((lft - area) * 10)
loa = ((ft >= 1) & np.isfinite(lft)).any(1)
acl = src.loc[ids, 'm_acl_tear'].astype(float).values
print('\n(2) LM B1 AUC within strata (expert strata unless noted)')
strata = {'LOA P_FT yes': loa, 'LOA P_FT no': ~loa, 'ACL tear (m_acl_tear>=1)': acl >= 1, 'ACL read, no tear': acl == 0}
for k in ('R6', 'S6fall', 'c4D5'):
    ctx = np.mean([P[k][c] for c in ('ACL', 'Contusion', 'Fracture')], 0)
    q1, q2 = np.quantile(ctx, [1 / 3, 2 / 3])
    st = dict(strata); st['%s trauma-ctx output top third' % k] = ctx >= q2; st['%s trauma-ctx output bottom third' % k] = ctx <= q1
    for sn, m in st.items():
        print('  %-7s %-40s n %4d prev %.3f AUC %.4f' % (k, sn, m.sum(), yB1[m].mean(), auc(P[k]['Lateral Meniscus'][m], yB1[m])))
ya = (acl >= 1).astype(float); okA = np.isfinite(acl)
print('  coupling ACL-LM: expert phi %.3f (n %d, ACL prev %.3f) | P(LM|ACL) %.3f vs P(LM|no ACL) %.3f' % (
    np.corrcoef(ya[okA], yB1[okA])[0, 1], okA.sum(), ya[okA].mean(), yB1[okA & (ya == 1)].mean(), yB1[okA & (ya == 0)].mean()))
for k in MODELS:
    print('    model Spearman ACL-LM %-9s %.3f | LOA-LM %.3f | MM-LM %.3f | Cont-LM %.3f' % (k, spearmanr(P[k]['ACL'], P[k]['Lateral Meniscus']).correlation,
          spearmanr(P[k]['Lateral OA'], P[k]['Lateral Meniscus']).correlation, spearmanr(P[k]['Medial Meniscus'], P[k]['Lateral Meniscus']).correlation,
          spearmanr(P[k]['Contusion'], P[k]['Lateral Meniscus']).correlation))
print('  expert phi LOA-LM %.3f' % np.corrcoef(loa.astype(float), yB1)[0, 1])

# ---------------- (3) candidates (paired bootstrap; R2 tilt for the ACL-LM pair: OAI marginals, gold74 log OR)
def cand(k):
    f = P[k]['Lateral Meniscus']; out = {}
    trc = np.mean([P[k][c] for c in ('ACL', 'Contusion', 'Fracture')], 0)
    for kk in (.2, .4):
        out['C1 trauma-ctx shrink k%.1f' % kk] = .5 + (f - .5) * (1 - kk * trc)
    out['C2 ACL-ctx shrink k0.4'] = .5 + (f - .5) * (1 - .4 * P[k]['ACL'])
    out['C3 LM <- ACL a.1'] = f - .1 * P[k]['ACL']
    out['C4 LM <- LOA a.1'] = f - .1 * P[k]['Lateral OA']
    out['C4b LM += LOA a-.1'] = f + .1 * P[k]['Lateral OA']
    out['placebo shrink k.4 ctx(Bak,MOA,PF)'] = .5 + (f - .5) * (1 - .4 * np.mean([P[k][c] for c in ("Baker's", 'Medial OA', 'PF OA')], 0))
    return f, out


def tilt_weights(a, y, logor_t):
    # w = exp(th * a * y + b1 a + b2 y), keep marginals of a, y, set log OR to logor_t
    th = logor_t - np.log(((a * y).sum() * ((1 - a) * (1 - y)).sum()) / max(((a * (1 - y)).sum() * ((1 - a) * y).sum()), 1e-9))
    b = np.zeros(2); tm = np.array([a.mean(), y.mean()])
    for _ in range(300):
        w = np.exp(th * a * y + b[0] * a + b[1] * y); cur = np.array([(w * a).sum(), (w * y).sum()]) / w.sum()
        b += np.log(tm / (1 - tm)) - np.log(cur / (1 - cur))
    return w / w.mean()


G74_LOGOR = np.log((11 * 26) / (18 * 19))   # gold74 ACL x LM 2x2 (aggregate counts, local g3 log): ACL+ 11/29 LM+, ACL- 19/45 LM+
rng = np.random.default_rng(1020); BI = [rng.integers(0, n, n) for _ in range(B)]
print('\n(3) candidates: LM AUC delta on B1 (paired bootstrap B %d) | also on the tear-vs-signal subset (TvS) | R2 = ACL-LM tilt to gold74 log OR %.2f (ACL-read knees)' % (B, G74_LOGOR))
wR2 = np.ones(n); wR2[okA] = tilt_weights(ya[okA], yB1[okA].astype(float), G74_LOGOR)
print('  R2 ESS %.0f of %d ACL-read knees' % (wR2[okA].sum() ** 2 / (wR2[okA] ** 2).sum(), okA.sum()))
tsub = tear | sig_only
for k in ('R6', 'S6fall', 'c4D5', 'R6.8S6.2', 'D5ref'):
    f, out = cand(k); b0 = auc(f, yB1); bts = [auc(f[ix], yB1[ix]) for ix in BI]
    b0s = auc(f[tsub], yB1[tsub]); b0r = auc(f[okA], yB1[okA], wR2[okA])
    for cn, o in out.items():
        d = auc(o, yB1) - b0; ds = np.array([auc(o[ix], yB1[ix]) for ix in BI]) - bts
        dS = auc(o[tsub], yB1[tsub]) - b0s; dR = auc(o[okA], yB1[okA], wR2[okA]) - b0r
        print('  %-9s %-36s B1 %+.5f (sd %.5f, P>0 %.3f) | TvS %+.5f | R2(ACL-read) %+.5f' % (k, cn, d, ds.std(), (ds > 0).mean(), dS, dR))
# C5 S6fall share curve (R6 / S6 only)
print('\n(4) S6fall share w in a pct(R6) / pct(S6fall) LM mix (c4 effective R6:S6 for LM about .71:.29 of the OAI-free part):')
f6, fs = P['R6']['Lateral Meniscus'], P['S6fall']['Lateral Meniscus']; base = .8 * f6 + .2 * fs; b0 = auc(base, yB1); bts = [auc(base[ix], yB1[ix]) for ix in BI]
for w in (0, .1, .2, .3, .4, .5, .7, 1.0):
    o = (1 - w) * f6 + w * fs; ds = np.array([auc(o[ix], yB1[ix]) for ix in BI]) - bts
    print('  w %.1f B1 AUC %.4f | vs w .2 %+.5f (sd %.5f, P>0 %.3f) | TvS %.4f' % (w, auc(o, yB1), auc(o, yB1) - b0, ds.std(), (ds > 0).mean(), auc(o[tsub], yB1[tsub])))
print('DONE (aggregates only)')
