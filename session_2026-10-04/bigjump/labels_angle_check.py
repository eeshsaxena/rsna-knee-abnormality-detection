# CPU-only, read-only. Training-target angle checks on gold58 (target level) + binormal lift table.
# 1) noise structure: P(gold=1 | report verdict) per label (is any verdict order non-monotone vs v4?)
# 2) site (report-language) offset removal at target level: offsets estimated on the 4349 NON-gold studies only
# 3) within-class error correlation of each target with the parent's Raptor arm on gold58 (rho cost of teacher mixing)
# 4) binormal lift table with the measured gold-rho -> effective-rho shift (+0.10, calib_rho.py)
import io, sys
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
S = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
D = "C:/Users/eeshs/Downloads/rsna-knee/_data"
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
       'Synovitis', "Baker's", 'Contusion', 'Fracture']
rd = lambda p: pd.read_csv(p).set_index('StudyInstanceUID')
lang = pd.read_csv(D + '/_reports_lang.csv', usecols=['StudyInstanceUID', 'lang']).set_index('StudyInstanceUID')['lang']
g = pd.read_csv(S + '/kneelabels/jev/gold74_eval_set.csv')
g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')[LAB].astype(int)
v4 = rd(S + '/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv')[LAB]
mix = rd(S + '/labels_mix50_logit.csv')[LAB]
msa = rd(S + '/labels_mixSA_logit.csv')[LAB]
rap = rd(S + '/teacher_v2_out/raptor_proxy_v5v8.csv')[LAB]
v8 = rd(S + '/teacher_v2_out/teacher_v2_native384_v8_shard0.csv')[LAB]
wd = rd(S + '/teacher_out/teacher_train_probs.csv')[LAB]
pk = rd(S + '/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv')
G = g.index
NG = v4.index.difference(G)
print('gold58 n', len(G), '| lang of gold58:', lang.reindex(G).value_counts().to_dict())


def mauc(P, idx=G):
    return float(np.mean([roc_auc_score(g.loc[idx, l], P.loc[idx, l]) for l in LAB]))


def boot(Pa, Pb, B=2000, seed=0):
    rs = np.random.RandomState(seed); d = []
    ga, pa, pb = g.loc[G].values, Pa.loc[G].values, Pb.loc[G].values
    for _ in range(B):
        i = rs.randint(0, len(G), len(G)); s = []
        for k in range(12):
            y = ga[i, k]
            if y.min() == y.max(): continue
            s.append(roc_auc_score(y, pa[i, k]) - roc_auc_score(y, pb[i, k]))
        d.append(np.mean(s))
    d = np.array(d); return d.mean(), d.std(), (d > 0).mean()


# ---------- 1) noise structure by report verdict
print('\n[1] P(gold=1 | pilkwang verdict) on gold58, per label  (n in brackets); v4 mean per verdict')
rows = []
for l in LAB:
    vv = pk.reindex(G)[l + '__verdict']
    r = {'label': l}
    for v in ('YES', 'UNK', 'NO'):
        m = vv == v
        r[v] = f"{g.loc[G, l][m].mean():.2f} [{int(m.sum())}]" if m.sum() else '-'
        r['v4_' + v] = round(float(v4.loc[G, l][m].mean()), 2) if m.sum() else np.nan
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
allv = pd.concat([pd.DataFrame({'v': pk.reindex(G)[l + '__verdict'].values, 'y': g[l].values}) for l in LAB])
print('pooled over labels:', allv.groupby('v').y.agg(['mean', 'size']).round(3).to_dict())

# ---------- 2) language offset removal (offsets from NON-gold train only)
print('\n[2] target-level gold58 macro AUC, per-language logit offsets removed (lambda), offsets fit on 4349 non-gold')
lg = lang.reindex(v4.index).fillna('xx')
big = lg.reindex(NG).value_counts(); big = big[big >= 100].index


def debias(P, lam):
    Z = np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4)))
    glob = Z.loc[NG].mean()
    off = Z.loc[NG].groupby(lg.reindex(NG)).mean() - glob
    off = off.loc[off.index.isin(big)]
    out = Z.copy()
    for la, o in off.iterrows():
        m = lg == la
        out.loc[m] = Z.loc[m] - lam * o.values
    return 1 / (1 + np.exp(-out)), off


for name, P in (('steven_v4', v4), ('mix50_logit', mix), ('raptor_proxy(parent arm)', rap), ('v8 arm', v8)):
    base = mauc(P)
    line = f'{name:26s} base {base:.4f}'
    for lam in (0.5, 1.0):
        Pd, off = debias(P, lam)
        m, sd, pp = boot(Pd, P)
        line += f' | lam {lam}: {mauc(Pd):.4f} (d {m:+.4f} sd {sd:.4f} P>0 {pp:.2f})'
    print(line)
_, offv4 = debias(v4, 1.0); _, offr = debias(rap, 1.0)
print('between-language SD of logit offsets per label (v4 labels vs parent raptor arm):')
print(pd.DataFrame({'v4': offv4.std(), 'raptor': offr.std(),
                    'corr_v4_vs_raptor_offsets': [np.corrcoef(offv4[l], offr[l])[0, 1] for l in LAB]}).round(3).to_string())

# ---------- 3) within-class correlation with the parent arm on gold58 (Spearman of residual ranks within gold class)
print('\n[3] within-class Spearman vs parent raptor arm on gold58 (mean over labels & both classes) + gold58 macro AUC')


def wc_rho(A, Bm):
    r = []
    for l in LAB:
        for c in (0, 1):
            m = g[l] == c
            if m.sum() >= 6:
                x, y = A.loc[G, l][m], Bm.loc[G, l][m]
                if x.nunique() > 2 and y.nunique() > 2:
                    r.append(spearmanr(x, y).correlation)
    return float(np.nanmean(r))


for name, P in (('steven_v4', v4), ('mix50_logit', mix), ('mixSA_logit', msa), ('widedense teacher', wd)):
    print(f'{name:20s} gold58 {mauc(P):.4f} | wc-rho vs raptor arm {wc_rho(P, rap):.3f} | vs v8 arm {wc_rho(P, v8):.3f}')
# error correlation: does the parent arm share the report labels' errors? sign-agnostic: corr of (pred - gold) residual ranks
res = lambda P: P.loc[G].rank(pct=True) - g.loc[G]
print('mean per-label corr(residual v4 label, residual raptor arm) on gold58:',
      round(float(np.mean([np.corrcoef(res(v4)[l], res(rap)[l])[0, 1] for l in LAB])), 3))

# ---------- 4) binormal table
mu = lambda a: np.sqrt(2) * norm.ppf(a)


def best(a1, a2, rho):
    ws = np.linspace(0, 1, 101); m = (1 - ws) * mu(a1) + ws * mu(a2)
    v = (1 - ws) ** 2 + ws ** 2 + 2 * ws * (1 - ws) * rho; b = norm.cdf(m / np.sqrt(2 * v)); i = b.argmax()
    return ws[i], b[i] - a1


print('\n[4] best-W lift over parent .943; effective rho = gold-rho + 0.10 (calib_rho.py: v09r .835 -> .935, crop .81 -> .91)')
for lab, a2, gr in (('Tian v09a LLM-only', .918, .742), ('Tian v09r Raptor-mix', .927, .835),
                    ('fleet v4 if .930', .930, .78), ('fleet mix if .939', .939, .84),
                    ('fleet v4 .935', .935, .78), ('fleet mix .944', .944, .84),
                    ('own .945 @ .78', .945, .78), ('own .950 @ .75', .950, .75), ('own .950 @ .85', .950, .85)):
    w0, l0 = best(.943, a2, min(gr, .995)); w1, l1 = best(.943, a2, min(gr + .10, .995))
    print(f'{lab:24s} solo {a2:.3f} gold-rho {gr:.3f}: lift@gold-rho {l0:+.4f} (W {w0:.2f}) | lift@eff-rho {l1:+.4f} (W {w1:.2f})')
