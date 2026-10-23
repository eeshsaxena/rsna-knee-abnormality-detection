# vpl_check.py (verify-pseudolabel, 2026-10-05): independent re-computation of the pseudolabel track's key numbers.
# CPU only, read-only inputs, prints to stdout (redirected to ~/jump2/vpl/vpl_check.out). Own AUC / mapping code,
# written without reusing pl_sim_b2.py functions.
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.special import logit, expit

H = os.path.expanduser
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
lg = lambda a: logit(np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4))

folds = rd('~/rsna_train/folds5.csv')['fold']
gold = rd('~/rsna_train/gold74_eval_set.csv')
G58 = [s for s in gold.index if gold.loc[s, 'source'] == 'original_gold58']
g58file = set(pd.read_csv(H('~/kmimic/ids/gold58.csv'), dtype=str)['StudyInstanceUID'])
print('== 0) split hygiene')
print('gold74 n', len(gold), '| gold58 n', len(G58), '| gold58 == ~/kmimic/ids/gold58.csv:', set(G58) == g58file)
print('folds5 value of gold74 rows:', folds.reindex(gold.index).value_counts(dropna=False).to_dict())
print('folds5 sizes:', folds.value_counts().sort_index().to_dict())
Y = gold.loc[G58, T].astype(float)
print('gold58 label values:', sorted(set(np.unique(Y.values)).__iter__()), '| positives per label:', Y.sum().astype(int).to_dict())


def auc(y, s):
    y = np.asarray(y) > .5; r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def m58(X, rows=None):
    X = X.reindex(G58)
    if X[T].isna().any().any():
        return float('nan')
    return float(np.mean([auc(Y[l].values, X[l].values) for l in T]))


v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')[T]
mix50 = rd('~/rsna_train/labels_mix50_logit.csv')[T]
gated = rd('~/rsna_train/labels_gated75_logit.csv')[T]
dread = rd('~/rsna_train/teacher_train_probs.csv')[T]
rap = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')[T]
tabs = {'steven_v4': v4, 'mix50_logit': mix50, 'gated75_logit': gated, 'Dread teacher_train_probs': dread,
        'Raptor proxy (teacher_v2 shard0)': rap}
for extra in ['labels_mixSA_ecdf.csv', 'labels_mixSA_logit.csv']:
    tabs[extra] = rd('~/rsna_train/' + extra)[T]
print('\n== 1) gold58 macro AUC of label tables (target level)')
for k, X in tabs.items():
    cov = X.index.isin(G58).sum()
    print(f'  {k:34s} gold58 {m58(X):.4f}  (gold58 rows in table: {cov})')

# mix50 composition check: is mix50 = logit-space 0.5 v4 + 0.5 matched teacher? correlate on train rows
tr = folds.index[folds >= 0]
c = np.mean([spearmanr(mix50.loc[tr, l], dread.reindex(tr)[l]).correlation for l in T])
print(f'  Spearman(mix50, Dread) on train rows {c:.3f}; Spearman(mix50, v4) '
      f'{np.mean([spearmanr(mix50.loc[tr, l], v4.loc[tr, l]).correlation for l in T]):.3f}')

print('\n== 2) students: gold58 from gold74_preds (fold-0 checkpoints, gold rows never trained on)')
GP = {'F1mix': 'f0_s42__raw', 'F1v4': 'f0_s42_v4__raw', 'F1gated': 'f0_s42_gated__raw', 'TTRv4': 'f0_s1042_ttr__ema',
      'TTRmix': 'f0_s1043_ttr_mix50__raw', 'TTRv4_raw': 'f0_s1042_ttr__raw', 'TTRmix_ema': 'f0_s1043_ttr_mix50__ema',
      'F1mix_ema': 'f0_s42__ema'}
gp = {k: rd(f'~/jump2/gold74_preds/{f}.csv')[T] for k, f in GP.items() if os.path.exists(H(f'~/jump2/gold74_preds/{f}.csv'))}
TRAINED = {'F1mix': mix50, 'F1v4': v4, 'F1gated': gated, 'TTRv4': v4, 'TTRmix': mix50}
for k, P in gp.items():
    s = f'  {k:11s} gold58 {m58(P):.4f}'
    if k in TRAINED:
        s += f' | its target {m58(TRAINED[k]):.4f} | student - target {m58(P) - m58(TRAINED[k]):+.4f} | student - v4 {m58(P) - m58(v4):+.4f}'
    print(s)

print('\n== 3) parent-likeness on the fold-0 val rows (mean per-label Spearman vs Raptor proxy) + within-verdict')
OOF = {'F1mix': '~/fleet_runs/f0_s42/oof_f0_s42_raw.csv', 'F1v4': '~/fleet_runs/f0_s42_v4/oof_f0_s42_raw.csv',
       'F1gated': '~/fleet_runs/f0_s42_gated/oof_f0_s42_raw.csv', 'TTRv4': '~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_ema.csv',
       'TTRmix': '~/fleet_runs/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv'}
O = {k: rd(v)[T] for k, v in OOF.items()}
V0 = sorted(set.intersection(*[set(o.index) for o in O.values()]))
print('  867 rows?', len(V0), '| all fold 0:', bool((folds.reindex(V0) == 0).all()), '| overlap gold74:', len(set(V0) & set(gold.index)))
r2 = rd('~/rsna_train/report_labels_v2.csv')
VER = pd.DataFrame({l: r2[l + '__verdict'] for l in T}).reindex(v4.index).fillna('UNK')
print('  verdict coverage of v4 rows (fraction with a report_labels_v2 row):', round(float(v4.index.isin(r2.index).mean()), 3))


def rho(X):
    return float(np.mean([spearmanr(X.loc[V0, l], rap.loc[V0, l]).correlation for l in T]))


def ws(X):
    num = den = 0.
    for l in T:
        for v in ['YES', 'UNK', 'NO']:
            m = (VER.loc[V0, l] == v).values
            a = X.loc[V0, l].values[m]
            if m.sum() >= 30 and np.unique(a).size > 3:
                r = spearmanr(a, rap.loc[V0, l].values[m]).correlation
                if np.isfinite(r): num += r * m.sum(); den += m.sum()
    return num / den


for k in O:
    print(f'  {k:8s} rho_rap {rho(O[k]):.4f} ws_rap {ws(O[k]):.4f} | AUCvsV4 '
          f'{np.mean([auc(v4.loc[V0, l] >= .5, O[k].loc[V0, l]) for l in T]):.4f}')
for k, X in [('v4', v4), ('mix50', mix50), ('gated75', gated)]:
    print(f'  target {k:8s} rho_rap {rho(X):.4f} ws_rap {ws(X):.4f}')
print('  member-to-member Spearman:', {f'{a}-{b}': round(float(np.mean([spearmanr(O[a].loc[V0, l], O[b].loc[V0, l]).correlation for l in T])), 3)
                                      for i, a in enumerate(O) for b in list(O)[i + 1:]})

print('\n== 4) key candidate targets, own mapping (students z-scored on the 867, mapped to v4 logit mean/std on the 867)')
ROWS = V0 + G58
MU, SD = lg(v4.loc[V0].values).mean(0), lg(v4.loc[V0].values).std(0)


def src(keys):
    zs = []
    for k in keys:
        A = pd.concat([O[k].loc[V0], gp[k].loc[G58]])
        L = lg(A.values); ref = L[:len(V0)]
        zs.append((L - ref.mean(0)) / ref.std(0))
    z = np.mean(zs, 0); ref = z[:len(V0)]; z = (z - ref.mean(0)) / ref.std(0)
    return pd.DataFrame(expit(z * SD + MU), index=ROWS, columns=T)


VV = VER.loc[ROWS].values; B = v4.loc[ROWS].values


def mixw(S, wu, wn, wy):
    w = np.where(VV == 'UNK', wu, np.where(VV == 'NO', wn, wy))
    return pd.DataFrame((1 - w) * B + w * S.values, index=ROWS, columns=T)


C = {'v4': v4.loc[ROWS], 'mix50': mix50.loc[ROWS]}
for b in [.25, .5, .75, 1.0]:
    C[f'line b={b}'] = pd.DataFrame((1 - b) * B + b * mix50.loc[ROWS].values, index=ROWS, columns=T)
Ev4 = src(['TTRv4', 'F1v4']); TTm = src(['TTRmix']); Em = src(['TTRmix', 'F1mix'])
C['Ev4 ALL .5 (v09t analog)'] = mixw(Ev4, .5, .5, .5)
C['Ev4 UNK+NO .75 (DSD-own)'] = mixw(Ev4, .75, .75, 0)
C['Ev4 UNK.9 NO.5 YES.25'] = mixw(Ev4, .9, .5, .25)
C['TTRmix UNK.75 NO.5 YES.5 (pl_tm5 f0)'] = mixw(TTm, .75, .5, .5)
C['Emix UNK.75 NO.5 YES.5'] = mixw(Em, .75, .5, .5)
C['TTRmix ALL .5 (r2 a50 f0 analog)'] = mixw(TTm, .5, .5, .5)
# DSD-combo with Raptor proxy on YES .25 (Raptor mapped the same way)
Rz = lg(rap.loc[ROWS].values); Rz = (Rz - Rz[:len(V0)].mean(0)) / Rz[:len(V0)].std(0)
RS = pd.DataFrame(expit(Rz * SD + MU), index=ROWS, columns=T)
dsd = mixw(Ev4, .75, .75, 0).values
dsd = np.where(VV == 'YES', .75 * B + .25 * RS.values, dsd)
C['Ev4 DSD-combo (+Raptor YES .25)'] = pd.DataFrame(dsd, index=ROWS, columns=T)
rng = np.random.default_rng(1); BS = [rng.integers(0, 58, 58) for _ in range(1000)]
Yv = Y.values


def boot(X, Z):
    Xa, Za = X.loc[G58].values, Z.loc[G58].values; d = []
    for i in BS:
        v = [auc(Yv[i, j], Xa[i, j]) - auc(Yv[i, j], Za[i, j]) for j in range(12) if 0 < Yv[i, j].mean() < 1]
        d.append(np.mean(v))
    d = np.array(d); return d.mean(), d.std(), (d > 0).mean()


for k, X in C.items():
    line = f'  {k:40s} gold58 {m58(X):.4f} rho {rho(X):.4f} ws {ws(X):.4f}'
    if k.startswith(('Ev4', 'TTRmix', 'Emix')):
        a = boot(X, C['mix50']); b = boot(X, C['line b=0.75'])
        line += f' | vs mix50 {a[0]:+.4f} sd {a[1]:.4f} P {a[2]:.2f} | vs line b=.75 {b[0]:+.4f} sd {b[1]:.4f} P {b[2]:.2f}'
    print(line)

print('\n== 5) within-verdict AUC on gold58 (cells pooled over labels, label-wise ranks)')
Vg = VER.loc[G58]
for st in ['YES', 'UNK', 'NO']:
    yy = np.concatenate([Y[l].values[(Vg[l] == st).values] for l in T]); s = f'  {st:3s} n {len(yy)} P1 {yy.mean():.3f} |'
    for nm, X in [('v4', v4), ('mix50', mix50), ('TTRv4', gp['TTRv4']), ('F1v4', gp['F1v4']), ('TTRmix', gp['TTRmix']), ('F1mix', gp['F1mix'])]:
        ss = np.concatenate([((rankdata(X.loc[G58, l]) - .5) / 58)[(Vg[l] == st).values] for l in T])
        s += f' {nm} {auc(yy, ss):.3f}'
    print(s)

print('\n== 6) gate S sanity: would existing members trip PROMOTE / KILL? (PROMOTE: gold58>=.913 AND (rho<=.886 OR ws<=.826); '
      'KILL: gold58<.903 OR rho>.900 OR ws>.840)')
for k in O:
    g, r, w = m58(gp[k]), rho(O[k]), ws(O[k])
    kill = g < .903 or r > .900 or w > .840; prom = (not kill) and g >= .913 and (r <= .886 or w <= .826)
    print(f'  {k:8s} gold58 {g:.4f} rho {r:.4f} ws {w:.5f} -> {"KILL" if kill else "PROMOTE" if prom else "HOLD"}')

print('\n== 7) per-fold v4 prevalence (folds5): is per-fold z-scoring safe?')
pv = pd.DataFrame({f: v4.reindex(folds.index[folds == f]).mean() for f in range(5)}).T
print(pv.round(3).to_string())
print('  max abs fold-mean deviation from the all-fold mean, per label:', (pv - pv.mean()).abs().max().round(3).to_dict())
