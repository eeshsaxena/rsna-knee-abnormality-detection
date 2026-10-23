# CPU-only. Archit-style "fill the report-silent (UNK) cells with OUR OWN out-of-sample image predictions" priced at
# target level on gold58, against the same fill with the parent's Raptor arm, and its within-class rho to the parent arm.
# Own teacher analog = rank-mean of our v4-label crop members (cropaug_f1, cropaug_f2, crop_f1). Their gold58 preds are
# out-of-sample (gold58 never trained on) but checkpoint-selected on gold58 -> optimistic (stated in the .md).
import io, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
S = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
       'Synovitis', "Baker's", 'Contusion', 'Fracture']
rd = lambda p: pd.read_csv(p).set_index('StudyInstanceUID')
g = pd.read_csv(S + '/kneelabels/jev/gold74_eval_set.csv')
g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')[LAB].astype(int); G = g.index
v4 = rd(S + '/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv')[LAB].loc[G]
rap = rd(S + '/teacher_v2_out/raptor_proxy_v5v8.csv')[LAB].loc[G]
v8 = rd(S + '/teacher_v2_out/teacher_v2_native384_v8_shard0.csv')[LAB].loc[G]
pk = rd(S + '/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv').reindex(G)
V = pd.DataFrame({l: pk[l + '__verdict'] for l in LAB}).fillna('NA')
z = np.load(S + '/laneA/gold74_preds_laneA.npz', allow_pickle=True)
ids = [str(x) for x in z['ids']]; zl = [str(x) for x in z['labels']]
mem = lambda k: pd.DataFrame(z[k], index=ids, columns=zl)[LAB].loc[G]
own_members = ['finetune_cropaug_f1_radimagenet', 'finetune_cropaug_f2_radimagenet', 'finetune_crop_f1_radimagenet']
own = sum(mem(k).rank(pct=True) for k in own_members) / 3
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def mauc(P): return float(np.mean([roc_auc_score(g[l], P[l]) for l in LAB]))


def wc_rho(A, B):
    r = []
    for l in LAB:
        for c in (0, 1):
            m = g[l] == c
            if m.sum() >= 6 and A[l][m].nunique() > 2 and B[l][m].nunique() > 2:
                r.append(spearmanr(A[l][m], B[l][m]).correlation)
    return float(np.nanmean(r))


def match(T):  # rank -> logit with v4's per-label logit mean/std (as build_mixSA 'logit')
    T = lg(T.clip(1e-3, 1 - 1e-3)); return (T - T.mean()) / T.std() * lg(v4).std() + lg(v4).mean()


def fill(T, strata, alpha):
    out = lg(v4).copy(); M = match(T)
    for l in LAB:
        m = V[l].isin(strata)
        out.loc[m, l] = (1 - alpha) * lg(v4)[l][m] + alpha * M[l][m]
    return out


print('own teacher (rank-mean of 3 v4 crop members) gold58 %.4f | raptor arm %.4f | v8 arm %.4f | v4 labels %.4f'
      % (mauc(own), mauc(rap), mauc(v8), mauc(v4)))
print('within-class rho to parent raptor arm: own %.3f | v4 %.3f' % (wc_rho(own, rap), wc_rho(v4, rap)))
print('\n%-34s %8s %10s %10s' % ('target', 'gold58', 'wc-rho rap', 'wc-rho v8'))
rows = [('v4 (LLM only)', lg(v4))]
for name, T in (('OWN', own), ('RAPTOR', rap)):
    for strata, tag in ((['UNK'], 'UNK'), (['UNK', 'NO'], 'UNK+NO'), (['YES', 'UNK', 'NO', 'NA'], 'ALL')):
        for a in (0.5, 0.75, 1.0):
            if tag == 'ALL' and a == 1.0: continue
            rows.append((f'{name} fill {tag} a={a}', fill(T, strata, a)))
for n, P in rows:
    print('%-34s %8.4f %10.3f %10.3f' % (n, mauc(P), wc_rho(P, rap), wc_rho(P, v8)))
# per-label for the headline own-UNK fill
P = fill(own, ['UNK'], 0.75)
print('\nper-label gold58 AUC, v4 -> OWN UNK a=.75 -> RAPTOR UNK a=.75 (UNK cells / gold positives among them):')
Pr = fill(rap, ['UNK'], 0.75)
for l in LAB:
    m = V[l] == 'UNK'
    print(f'  {l:17s} {roc_auc_score(g[l], v4[l]):.3f} -> {roc_auc_score(g[l], P[l]):.3f} -> {roc_auc_score(g[l], Pr[l]):.3f}'
          f'   ({int(m.sum())} UNK, {int(g[l][m].sum())} pos)')
# paired bootstrap own-UNK .75 vs v4
rs = np.random.RandomState(0); d = []
for _ in range(2000):
    i = rs.randint(0, len(G), len(G)); s = []
    for l in LAB:
        y = g[l].values[i]
        if y.min() == y.max(): continue
        s.append(roc_auc_score(y, P[l].values[i]) - roc_auc_score(y, lg(v4)[l].values[i]))
    d.append(np.mean(s))
d = np.array(d); print(f'\nOWN UNK a=.75 minus v4: {d.mean():+.4f} sd {d.std():.4f} P>0 {(d > 0).mean():.3f}')
