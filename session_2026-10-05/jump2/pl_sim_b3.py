# pl_sim_b3.py (jump2, 2026-10-05): is there an own-OOF target ABOVE the plain v4 <-> mix50 line? CPU, read-only.
# Frontier = (1-b) v4 + b mix50 (b = Dread share x2). Compare own-OOF targets at matched parent-likeness (ws_rap on 867).
# Also: student ensembles on gold58 (what 'self-distillation = ensembling' could at most hand a student).
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.special import logit, expit
H = os.path.expanduser
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')[T]; mix50 = rd('~/rsna_train/labels_mix50_logit.csv')[T]
rap = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')[T]
r2 = rd('~/rsna_train/report_labels_v2.csv'); VER = pd.DataFrame({l: r2[l + '__verdict'] for l in T}).reindex(v4.index).fillna('UNK')
gold = rd('~/rsna_train/gold74_eval_set.csv'); G58 = [s for s in gold.index if gold.loc[s, 'source'] == 'original_gold58']; Y = gold.loc[G58, T].astype(float).values
OOF = {'F1mix': '~/fleet_runs/f0_s42/oof_f0_s42_raw.csv', 'F1v4': '~/fleet_runs/f0_s42_v4/oof_f0_s42_raw.csv',
       'F1gated': '~/fleet_runs/f0_s42_gated/oof_f0_s42_raw.csv', 'TTRv4': '~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_ema.csv',
       'TTRmix': '~/fleet_runs/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv'}
GP = {'F1mix': 'f0_s42__raw', 'F1v4': 'f0_s42_v4__raw', 'F1gated': 'f0_s42_gated__raw', 'TTRv4': 'f0_s1042_ttr__ema', 'TTRmix': 'f0_s1043_ttr_mix50__raw'}
O = {k: rd(v)[T] for k, v in OOF.items()}; V867 = sorted(set.intersection(*[set(o.index) for o in O.values()]))
ROWS = V867 + G58; n = len(V867)
ST = {k: pd.concat([O[k].loc[V867], rd(f'~/jump2/gold74_preds/{GP[k]}.csv')[T].loc[G58]]) for k in O}
lg = lambda a: logit(np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4))
LV = lg(v4.loc[V867].values)
def own(keys):
    zs = []
    for k in keys:
        L = lg(ST[k].values); zs.append((L - L[:n].mean(0)) / L[:n].std(0))
    z = np.mean(zs, 0); z = (z - z[:n].mean(0)) / z[:n].std(0)
    return expit(z * LV.std(0) + LV.mean(0))
B4 = v4.loc[ROWS].values; BM = mix50.loc[ROWS].values; VV = VER.loc[ROWS].values
def fauc(y, s):
    r = rankdata(s); y = y > .5; n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def g58(X): return float(np.mean([fauc(Y[:, j], X[n:, j]) for j in range(12)]))
V8 = VER.loc[V867]; R8 = rap.loc[V867].values
def ws(X):
    num = den = 0.0
    for j, l in enumerate(T):
        for v in ['YES', 'UNK', 'NO']:
            m = (V8[l] == v).values
            if m.sum() >= 30 and np.unique(X[:n, j][m]).size > 3:
                r = spearmanr(X[:n, j][m], R8[:, j][m]).correlation
                if np.isfinite(r): num += r * m.sum(); den += m.sum()
    return num / den
def rho(X): return float(np.mean([spearmanr(X[:n, j], R8[:, j]).correlation for j in range(12)]))
W = lambda s, no, y: np.where(VV == 'UNK', s, np.where(VV == 'NO', no, y)).astype(float)
rows = []
for b in [0, .25, .5, .625, .75, .875, 1.0]:
    X = (1 - b) * B4 + b * BM; rows.append(('frontier v4->mix50 b=%.3f (Dread %.2f)' % (b, b / 2), g58(X), ws(X), rho(X)))
for b in [1.25, 1.5]:  # extrapolated Dread share via logit-space mix (mixSA-like heavier teacher), sanity only
    X = expit((1 - b) * lg(B4) + b * lg(BM)); rows.append(('frontier extrap logit b=%.2f' % b, g58(X), ws(X), rho(X)))
for nm, keys in [('Ev4', ['TTRv4', 'F1v4']), ('Eall', ['TTRv4', 'F1v4', 'TTRmix', 'F1mix']), ('Emix', ['TTRmix', 'F1mix'])]:
    S = own(keys)
    for (s, no, y) in [(.5, .5, .5), (.9, .5, .25), (.75, .5, .5), (.75, .75, .75)]:
        w = W(s, no, y); X = (1 - w) * B4 + w * S
        rows.append((f'{nm} on v4 UNK{s} NO{no} YES{y}', g58(X), ws(X), rho(X)))
    for (s, no, y) in [(.5, .5, .5), (.75, .5, .25)]:  # own source mixed INTO mix50 (3-way: LLM + Dread + own)
        w = W(s, no, y); X = (1 - w) * BM + w * S
        rows.append((f'{nm} into mix50 UNK{s} NO{no} YES{y}', g58(X), ws(X), rho(X)))
# students and student ensembles (rank mean on 867 + gold58 jointly; rank within the 867, gold mapped by the 867 ECDF)
def rankmean(keys):
    out = []
    for k in keys:
        L = lg(ST[k].values); out.append((L - L[:n].mean(0)) / L[:n].std(0))
    return np.mean(out, 0)
for keys in [['F1v4'], ['TTRv4'], ['F1gated'], ['F1mix'], ['TTRmix'], ['TTRv4', 'F1v4'], ['TTRmix', 'F1mix'],
             ['TTRv4', 'F1v4', 'TTRmix', 'F1mix'], ['TTRv4', 'TTRmix'], ['TTRv4', 'F1v4', 'F1gated', 'TTRmix', 'F1mix']]:
    X = rankmean(keys); rows.append(('[students] ' + '+'.join(keys), g58(X), ws(X), rho(X)))
X = rap.loc[ROWS].values; rows.append(('[Raptor proxy itself]', g58(X), 1.0, 1.0))
D = pd.DataFrame(rows, columns=['target', 'gold58', 'ws_rap', 'rho_rap']).set_index('target')
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(D.round(4).to_string()); D.to_csv(H('~/jump2/pl_sim_out/B3_frontier.csv'))
# frontier interpolation: gold58 of the v4->mix50 line at each own target's ws
F = D[D.index.str.startswith('frontier v4')].sort_values('ws_rap')
print('\nown targets vs the frontier at equal ws_rap (gold58 above(+)/below(-) the line):')
for t in D.index:
    if ' on v4 ' in t or ' into mix50 ' in t:
        g_line = np.interp(D.loc[t, 'ws_rap'], F.ws_rap.values, F.gold58.values)
        print(f'  {t:42s} ws {D.loc[t, "ws_rap"]:.3f} gold58 {D.loc[t, "gold58"]:.4f} line {g_line:.4f} diff {D.loc[t, "gold58"] - g_line:+.4f}')
