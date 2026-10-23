# CPU-only. Where does the teacher-mix gain come from on gold58? Swap the image-teacher ranking into ONE verdict
# stratum at a time (YES / UNK / NO cells), keep v4 elsewhere, and score target-level macro AUC on gold58.
# Also: within-stratum AUC of v4, raptor, mix (pooled over labels via per-label AUC where both classes exist),
# and an oracle bound (perfect ranking inside YES only) = how much a perfect "which report-YES are real" signal is worth.
import io, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
S = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
       'Synovitis', "Baker's", 'Contusion', 'Fracture']
rd = lambda p: pd.read_csv(p).set_index('StudyInstanceUID')
g = pd.read_csv(S + '/kneelabels/jev/gold74_eval_set.csv')
g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')[LAB].astype(int); G = g.index
v4 = rd(S + '/kneelabels/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv')[LAB].loc[G]
mix = rd(S + '/labels_mix50_logit.csv')[LAB].loc[G]
rap = rd(S + '/teacher_v2_out/raptor_proxy_v5v8.csv')[LAB].loc[G]
pk = rd(S + '/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv').reindex(G)
V = pd.DataFrame({l: pk[l + '__verdict'] for l in LAB}).fillna('NA')
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def mauc(P): return float(np.mean([roc_auc_score(g[l], P[l]) for l in LAB]))


def swap(stratum, alpha=0.5):
    """target = v4 outside the stratum; inside, logit-mix of v4 and rank-matched raptor (alpha = teacher weight)."""
    out = lg(v4).copy(); T = lg(rap)
    T = (T - T.mean()) / T.std() * lg(v4).std() + lg(v4).mean()      # same logit mean/std match as build_mixSA logit
    for l in LAB:
        m = V[l].isin(stratum)
        out.loc[m, l] = (1 - alpha) * lg(v4)[l][m] + alpha * T[l][m]
    return out


print('gold58 target-level macro AUC: v4 %.4f | mix50_logit %.4f | raptor proxy %.4f' % (mauc(v4), mauc(mix), mauc(rap)))
for st in (['YES'], ['UNK'], ['NO'], ['UNK', 'NO'], ['YES', 'UNK', 'NO']):
    print(f'teacher 0.5 only inside {"+".join(st):12s}: {mauc(swap(st)):.4f}   | teacher 1.0 inside: {mauc(swap(st, 1.0)):.4f}')


def oracle_inside(stratum):
    out = lg(v4).copy()
    for l in LAB:
        m = V[l].isin(stratum)
        out.loc[m, l] = lg(v4)[l][m] + 0.01 * (g[l][m] - 0.5)          # break ties inside the stratum by truth
        # keep stratum position: tiny move only reorders within equal/near-equal v4 values
    return out


def oracle_full(stratum):
    """perfect ranking inside the stratum, stratum keeps its v4 value range (min..max)."""
    out = lg(v4).copy()
    for l in LAB:
        m = V[l].isin(stratum)
        if m.sum() < 2: continue
        lo, hi = lg(v4)[l][m].min(), lg(v4)[l][m].max()
        if hi - lo < 1e-6: hi = lo + 1.0
        out.loc[m, l] = np.where(g[l][m] == 1, hi, lo)
    return out


print('\noracle bounds (truth used inside one stratum, v4 elsewhere):')
for st in (['YES'], ['UNK'], ['NO']):
    print(f'  inside {st[0]:4s}: tie-break by truth {mauc(oracle_inside(st)):.4f} | perfect within-range {mauc(oracle_full(st)):.4f}')

print('\nwithin-stratum AUC vs gold (mean over labels with both classes, n labels):')
for st in ('YES', 'UNK', 'NO'):
    r = {}
    for name, P in (('v4', v4), ('pilk_conf', pd.DataFrame({l: pk[l + '__conf'] for l in LAB}).fillna(0.5)), ('raptor', rap), ('mix50', mix)):
        a = []
        for l in LAB:
            m = V[l] == st
            y = g[l][m]
            if m.sum() >= 4 and 0 < y.sum() < len(y) and P[l][m].nunique() > 1:
                a.append(roc_auc_score(y, P[l][m]))
        r[name] = (round(float(np.mean(a)), 3), len(a)) if a else None
    print(' ', st, r)
