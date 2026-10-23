# syn_more.py (10-20 night, Claude). LOCAL CPU ONLY, RSNA data only. Aggregates only.
#  A. Site / language prior in the report-trained Synovitis head (R7 OOF + our S6 seeds s1044 / s1045 OOF, non-gold train studies).
#  B. E2 on the c4 (= pw30) final Synovitis column: influence (leave-one-out), pair flips, split-half, exploratory post-hoc
#     variants (marked POST HOC), and the paired public / private read noise (390 / 910 studies).
#  C. Probe value under R1 (MODEL), mixture priors stated in the log.
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
import cbase
from load import s6seed, labels as replabels

R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = cbase.L; J = {c: i for i, c in enumerate(L)}
SYN, EFF = J['Synovitis'], J['Effusion']
pct = lambda v: rankdata(v) / len(v)
rng = np.random.default_rng(1021)


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return np.nan if p == 0 or q == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


d = cbase.load(); ids = d['ids']; Y = d['Y58']; ys, ye = Y[:, SYN], Y[:, EFF]; n = len(ids)
gold74 = set(d['ids74'])
lang = pd.read_csv(R + '/_data/_reports_lang.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['lang']
v2 = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')

print('=' * 30, 'A. Language (site proxy) prior in report-trained Syn / Eff heads (OOF, non-gold train)')
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
mods = {'R7': r7[['p|Synovitis', 'p|Effusion']].set_axis(['Synovitis', 'Effusion'], axis=1)}
for sd_ in (1044, 1045):
    try:
        x, _ = s6seed(sd_); mods['s%d' % sd_] = x[['Synovitis', 'Effusion']]
    except Exception as e:
        print('s6seed', sd_, 'unavailable', type(e).__name__)
for k, P in mods.items():
    P = P[~P.index.isin(gold74)].dropna()
    df = pd.DataFrame({'syn': pct(P.Synovitis.values), 'eff': pct(P.Effusion.values)}, index=P.index)
    df['lang'] = lang.reindex(df.index).values
    df['v2syn'] = (v2['Synovitis__verdict'].reindex(df.index) == 'YES').values
    df['v2eff'] = (v2['Effusion__verdict'].reindex(df.index) == 'YES').values
    g = df.groupby('lang').agg(n=('syn', 'size'), syn_pct=('syn', 'mean'), eff_pct=('eff', 'mean'), v2syn=('v2syn', 'mean'), v2eff=('v2eff', 'mean'))
    g['syn_minus_eff'] = g.syn_pct - g.eff_pct
    print('[%s] n %d' % (k, len(df)))
    print(g.round(3).to_string())
    print('[%s] across languages: Spearman(syn_minus_eff, v2 Syn YES rate) %.2f | Spearman(syn_pct, v2syn) %.2f | Spearman(eff_pct, v2eff) %.2f' % (
        k, spearmanr(g.syn_minus_eff, g.v2syn).correlation, spearmanr(g.syn_pct, g.v2syn).correlation, spearmanr(g.eff_pct, g.v2eff).correlation))
    # within reports that say effusion YES: does the Syn head still follow the language's synovitis-reporting habit?
    sub = df[df.v2eff]
    gg = sub.groupby('lang').agg(n=('syn', 'size'), syn_pct=('syn', 'mean'), v2syn=('v2syn', 'mean'))
    print('[%s] among v2 Eff YES: Spearman(syn_pct, v2syn) over languages %.2f | syn_pct en %.3f es %.3f de %.3f vs bg %.3f hr %.3f tr %.3f nl %.3f' % (
        k, spearmanr(gg.syn_pct, gg.v2syn).correlation, *[gg.syn_pct.get(l_, np.nan) for l_ in ('en', 'es', 'de', 'bg', 'hr', 'tr', 'nl')]))

print('=' * 30, 'B. E2 on the c4 / pw30 final Synovitis column (gold58)')
F = d['c4']; s, e = pct(F[:, SYN]), pct(F[:, EFF])
base = auc(s, ys)
print('base Syn AUC %.4f (27 pos x 31 neg = 837 pairs)' % base)
for a in (.1, .2, .3):
    o = s + a * e
    dd = auc(o, ys) - base
    loo = np.array([auc(np.delete(o, i), np.delete(ys, i)) - auc(np.delete(s, i), np.delete(ys, i)) for i in range(n)])
    infl = dd - loo                                  # contribution of each study
    top = np.sort(infl)[::-1]
    # pair flips: discordant -> concordant minus the reverse
    P_, N_ = np.where(ys == 1)[0], np.where(ys == 0)[0]
    before = (s[P_][:, None] > s[N_][None, :]); after = (o[P_][:, None] > o[N_][None, :])
    gained, lost = int((after & ~before).sum()), int((~after & before).sum())
    # split halves (random 29/29), sign agreement
    agree = []
    for _ in range(500):
        p = rng.permutation(n); h1, h2 = p[:29], p[29:]
        if min(ys[h1].sum(), ys[h2].sum()) < 3 or max(ys[h1].sum(), ys[h2].sum()) > 26:
            continue
        d1 = auc(o[h1], ys[h1]) - auc(s[h1], ys[h1]); d2 = auc(o[h2], ys[h2]) - auc(s[h2], ys[h2])
        agree.append(np.sign(d1) == np.sign(d2) if d1 != 0 and d2 != 0 else np.nan)
    agree = np.array(agree, float)
    print('a %.1f: d %+.4f (macro %+.5f) | pairs gained %d lost %d (net %d of 837) | top-3 studies carry %.0f%% of d | split-half same sign %.2f' % (
        a, dd, dd / 12, gained, lost, gained - lost, 100 * top[:3].sum() / dd if dd else np.nan, np.nanmean(agree)))
    # where do the flips happen: Syn+ (Eff+/Eff-) vs Syn- (Eff+/Eff-)
    rk_s, rk_o = rankdata(s), rankdata(o)
    mv = rk_o - rk_s
    print('       mean rank move: Syn+Eff+ %+.1f | Syn+Eff- %+.1f | Syn-Eff+ %+.1f | Syn-Eff- %+.1f' % tuple(
        mv[(ys == a_) & (ye == b_)].mean() for a_, b_ in ((1, 1), (1, 0), (0, 1), (0, 0))))
# exploratory, POST HOC (gold58 picked the idea): borrow Effusion but not for acute-trauma effusions
acl, cont, frac = (pct(F[:, J[c]]) for c in ('ACL', 'Contusion', 'Fracture'))
acute = (acl + cont + frac) / 3
for nm, o in (('E2 a .2', s + .2 * e), ('POST HOC E2 a .2 x (1 - acute)', s + .2 * e * (1 - acute)), ('POST HOC E2 a .2 - .2 ACL', s + .2 * e - .2 * acl),
              ('POST HOC E2 a .2 x (1 - ACL)', s + .2 * e * (1 - acl))):
    print('%-34s d %+.4f' % (nm, auc(o, ys) - base))

# paired read noise of E2 on 390 / 910 studies: (i) gold58 bootstrap sd scaled by sqrt(58 / m); (ii) R7 OOF (report regime)
bs = {}
for a in (.1, .2):
    o = s + a * e; v = []
    for _ in range(3000):
        ii = rng.integers(0, n, n)
        if 0 < ys[ii].sum() < n:
            v.append(auc(o[ii], ys[ii]) - auc(s[ii], ys[ii]))
    bs[a] = np.std(v)
    print('E2 a %.1f gold58 bootstrap sd (label AUC) %.4f -> macro sd on 390 %.5f, on 910 %.5f (sqrt scaling, MODEL)' % (
        a, bs[a], bs[a] * np.sqrt(58 / 390) / 12, bs[a] * np.sqrt(58 / 910) / 12))
lab = replabels('v4')['Synovitis']
P7 = r7[~r7.index.isin(gold74)]
okk = lab.reindex(P7.index).notna().values
X7s, X7e, Y7 = P7['p|Synovitis'].values[okk], P7['p|Effusion'].values[okk], lab.reindex(P7.index).values[okk].astype(int)
for a in (.1, .2):
    for m in (390, 910):
        v = []
        for _ in range(400):
            ix = rng.choice(len(Y7), m, replace=False)
            us, ue = pct(X7s[ix]), pct(X7e[ix])
            v.append((auc(us + a * ue, Y7[ix]) - auc(us, Y7[ix])) / 12)
        print('E2 a %.1f R7 OOF report regime (v4 >= .5, prev %.3f) paired macro sd on %d: %.5f (mean %+.5f)' % (a, Y7.mean(), m, np.std(v), np.mean(v)))

print('=' * 30, 'C. Probe value under R1 (MODEL): delta ~ mixture prior (macro); e_pub ~ N(0, sp); adopt iff delta + e_pub > 0')
def mc(p_real, m_real, m_null, s_true, sp, N=400000):
    real = rng.random(N) < p_real
    t = np.where(real, rng.normal(m_real, s_true, N), rng.normal(m_null, s_true, N))
    pub = t + rng.normal(0, sp, N)
    adopt = pub > 0
    return t.mean(), (t > 0).mean(), adopt.mean(), (t * adopt).mean(), ((t * adopt) < 0).mean()
for a, sp in ((.1, .00020), (.2, .00029)):
    for p_real in (.35, .5, .65):
        # real effect: gold58 per-label d shrunk by the E-OA gold58 -> public ratio (~.4), macro; null: OAI-like small loss
        m_real = (.0036 if a == .1 else .0084) * .4 / 12; m_null = (-.0005 if a == .1 else -.0012) / 12 * 2
        out = mc(p_real, m_real, m_null, abs(m_real) * .6 + .00004, sp)
        print('a %.1f p_real %.2f: E[delta] %+.6f P(delta>0) %.2f P(adopt) %.2f E[gain probe+R1] %+.6f P(adopted and harmful) %.2f' % ((a, p_real) + out))
