# pl_sim_b2.py (jump2, 2026-10-05): pseudo-label TARGET simulation, part 2. CPU only, read-only inputs, writes ~/jump2/pl_sim_out/.
# Needs ~/jump2/gold74_preds/<run>__<tag>.csv from gold_infer.py (fold-0 checkpoints on the 74 gold studies; gold74 is
# fold -1 in folds5.csv and --exclude'd from every fleet run, so those predictions have OOF status, like the 867 val rows).
#  1) TRANSFER table (MEASURED): each measured student vs the target it was trained on: gold58 AUC, parent-likeness
#     (mean per-label Spearman vs the Raptor proxy on the 867 fold-0 val rows) and within-verdict Spearman (ws).
#  2) CANDIDATE targets (MEASURED at target level): gold58 AUC with paired bootstrap vs v4 / mix50 / the flat-0.5
#     self-distillation analog (Tian v09t: measured LB null), and parent-likeness on the 867.
#  3) PROJECTION student (MODEL): per label, CV ridge from the 4 measured students' standardised logits to logit(target)
#     on the 867; a crude 'student = image-explainable part of the target'. Validated on v4 / mix50 / gated.
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

H = os.path.expanduser
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
OUT = H('~/jump2/pl_sim_out'); os.makedirs(OUT, exist_ok=True)
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')[T]
mix50 = rd('~/rsna_train/labels_mix50_logit.csv')[T]
gated = rd('~/rsna_train/labels_gated75_logit.csv')[T]
rap = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')[T]
r2 = rd('~/rsna_train/report_labels_v2.csv')
VER = pd.DataFrame({l: r2[l + '__verdict'] for l in T}).reindex(v4.index).fillna('UNK')
gold = rd('~/rsna_train/gold74_eval_set.csv')
G58 = [s for s in gold.index if gold.loc[s, 'source'] == 'original_gold58']
Y58 = gold.loc[G58, T].astype(float)
OOF = {'F1mix': '~/fleet_runs/f0_s42/oof_f0_s42_raw.csv', 'F1v4': '~/fleet_runs/f0_s42_v4/oof_f0_s42_raw.csv',
       'F1gated': '~/fleet_runs/f0_s42_gated/oof_f0_s42_raw.csv', 'TTRv4': '~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_ema.csv',
       'TTRmix': '~/fleet_runs/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv'}
GP = {'F1mix': 'f0_s42__raw', 'F1v4': 'f0_s42_v4__raw', 'F1gated': 'f0_s42_gated__raw', 'TTRv4': 'f0_s1042_ttr__ema',
      'TTRmix': 'f0_s1043_ttr_mix50__raw'}
TRAINED_ON = {'F1mix': 'mix50', 'F1v4': 'v4', 'F1gated': 'gated75', 'TTRv4': 'v4', 'TTRmix': 'mix50'}
O = {k: rd(v)[T] for k, v in OOF.items()}
V867 = sorted(set.intersection(*[set(o.index) for o in O.values()]))
assert not set(V867) & set(gold.index)
gp = {k: rd(f'~/jump2/gold74_preds/{f}.csv')[T] for k, f in GP.items() if os.path.exists(H(f'~/jump2/gold74_preds/{f}.csv'))}
assert len(gp) == len(GP), f'gold preds missing: {set(GP) - set(gp)}'
ROWS = V867 + G58
lg = lambda a: logit(np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4))


def stud(k):  # OOF-status prediction for every row in ROWS
    return pd.concat([O[k].loc[V867], gp[k].loc[G58]])


ST = {k: stud(k) for k in O}
LV = lg(v4.loc[V867].values); MU_V, SD_V = LV.mean(0), LV.std(0)


def own_src(keys, mode='logit'):
    """mean of the members' logits, each standardised on the 867; then mapped onto v4's logit mean/std (mode logit)
    or left as the mean member probability (mode raw)."""
    if mode == 'raw':
        return pd.DataFrame(np.mean([ST[k].values for k in keys], 0), index=ROWS, columns=T)
    zs = []
    for k in keys:
        L = lg(ST[k].values); ref = L[:len(V867)]; zs.append((L - ref.mean(0)) / ref.std(0))
    z = np.mean(zs, 0); ref = z[:len(V867)]; z = (z - ref.mean(0)) / ref.std(0)
    return pd.DataFrame(expit(z * SD_V + MU_V), index=ROWS, columns=T)


SRC = {'TTRv4': ['TTRv4'], 'Ev4': ['TTRv4', 'F1v4'], 'TTRmix': ['TTRmix'], 'Emix': ['TTRmix', 'F1mix'],
       'Eall': ['TTRv4', 'F1v4', 'TTRmix', 'F1mix']}
B4 = v4.loc[ROWS].values; VV = VER.loc[ROWS].values


def mixw(S, w):  # per-cell weight w on the own source, probability space (as build_gated / pl_sim)
    return pd.DataFrame((1 - w) * B4 + w * S.values, index=ROWS, columns=T)


def W_strata(sil, no, yes):
    return np.where(VV == 'UNK', sil, np.where(VV == 'NO', no, yes)).astype(float)


def W_unc(amax, cap=0.9):  # uncertainty-weighted: weight grows as v4 nears 0.5 (UNK cells sit at ~0.25 -> u .5)
    return np.minimum(cap, amax * (1 - np.abs(2 * B4 - 1)))


C = {'v4': v4.loc[ROWS], 'mix50 (file)': mix50.loc[ROWS], 'gated75 (file)': gated.loc[ROWS]}
for s in SRC:
    S = own_src(SRC[s])
    C[f'{s} ALL .50 (v09t analog)'] = mixw(S, W_strata(.5, .5, .5))
    C[f'{s} ALL .75'] = mixw(S, W_strata(.75, .75, .75))
    C[f'{s} UNK .75'] = mixw(S, W_strata(.75, 0, 0))
    C[f'{s} UNK+NO .75'] = mixw(S, W_strata(.75, .75, 0))
    C[f'{s} UNK .9 NO .5 YES .25'] = mixw(S, W_strata(.9, .5, .25))
    C[f'{s} UNK .75 NO .5 YES .5'] = mixw(S, W_strata(.75, .5, .5))
    C[f'{s} UNC a1.5 cap.9'] = mixw(S, W_unc(1.5))
for s in ['Ev4', 'Eall']:
    C[f'{s} RAW UNK .9 NO .5 YES .25'] = mixw(own_src(SRC[s], 'raw'), W_strata(.9, .5, .25))
# two-teacher: own v4-ensemble on silent cells + mix50 elsewhere (keeps the Dread share only where the report speaks)
C['mix50 + Ev4 UNK .75'] = pd.DataFrame(np.where(VV == 'UNK', .25 * B4 + .75 * own_src(SRC['Ev4']).values,
                                                 mix50.loc[ROWS].values), index=ROWS, columns=T)


def per58(X):
    return np.array([roc_auc_score(Y58[l], X.loc[G58, l]) for l in T])


def mrho(X, R, rows):
    return float(np.nanmean([spearmanr(X.loc[rows, l], R.loc[rows, l]).correlation for l in T]))


VER867 = VER.loc[V867]


def ws(X, R):  # within (label, verdict) Spearman on the 867, weighted by cell count
    num = den = 0.0
    for l in T:
        for v in ['YES', 'UNK', 'NO']:
            m = (VER867[l] == v).values
            if m.sum() >= 30:
                a = X.loc[V867, l].values[m]; b = R.loc[V867, l].values[m]
                if np.unique(a).size > 3:
                    r = spearmanr(a, b).correlation
                    if np.isfinite(r): num += r * m.sum(); den += m.sum()
    return num / den if den else float('nan')


rng = np.random.default_rng(0); BOOT = [rng.integers(0, len(G58), len(G58)) for _ in range(1000)]
Yv = Y58.values


def fauc(y, s):  # Mann-Whitney AUC with midranks (ties count half), y in {0,1}
    r = rankdata(s); y = y > .5; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bdelta(X, Z):
    Xa, Za = X.loc[G58].values, Z.loc[G58].values; d = []
    for i in BOOT:
        v = [fauc(Yv[i, j], Xa[i, j]) - fauc(Yv[i, j], Za[i, j]) for j in range(12) if 0 < Yv[i, j].mean() < 1]
        d.append(np.mean(v))
    d = np.array(d); return d.mean(), d.std(), (d > 0).mean()


pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
# ---------------- 1) transfer table ----------------
tgt = {'v4': v4, 'mix50': mix50, 'gated75': gated}
rec = []
for k in O:
    t = tgt[TRAINED_ON[k]].loc[ROWS]
    rec.append({'student': k, 'target': TRAINED_ON[k], 'tgt_gold58': per58(t).mean(), 'stu_gold58': per58(ST[k]).mean(),
                'tgt_rho_rap': mrho(t, rap, V867), 'stu_rho_rap': mrho(ST[k], rap, V867),
                'tgt_ws_rap': ws(t, rap), 'stu_ws_rap': ws(ST[k], rap),
                'stu_auc_v4_867': float(np.mean([roc_auc_score(v4.loc[V867, l] >= .5, O[k].loc[V867, l]) for l in T]))})
rap58 = rap.reindex(G58)
rec.append({'student': '[Raptor proxy]', 'target': '-', 'stu_gold58': per58(rap.reindex(ROWS)).mean() if rap58.notna().all().all() else np.nan,
            'stu_rho_rap': 1.0})
TR = pd.DataFrame(rec).set_index('student'); TR.to_csv(f'{OUT}/B2_transfer.csv')
print('== 1) TRANSFER (MEASURED): target vs the student trained on it (gold58 macro AUC; Spearman vs Raptor proxy on 867)')
print(TR.round(4).to_string())

# ---------------- 3) projection simulator ----------------
BASIS = ['F1v4', 'TTRv4', 'F1mix', 'TTRmix']


def project(X, basis=BASIS, alpha=10.0):
    out = np.zeros((len(ROWS), 12))
    for j, l in enumerate(T):
        F = np.stack([lg(ST[k][l].values) for k in basis], 1)
        F = (F - F[:len(V867)].mean(0)) / F[:len(V867)].std(0)
        y = lg(X[l].values)[:len(V867)]
        Ftr = F[:len(V867)]; pred = np.zeros(len(V867))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Ftr):
            pred[te] = Ridge(alpha=alpha).fit(Ftr[tr], y[tr]).predict(Ftr[te])
        out[:len(V867), j] = pred
        out[len(V867):, j] = Ridge(alpha=alpha).fit(Ftr, y).predict(F[len(V867):])
    return pd.DataFrame(expit(out), index=ROWS, columns=T)


# ---------------- 2) candidates ----------------
base_v09t = C['Ev4 ALL .50 (v09t analog)']
recC = []
for k, X in C.items():
    p = per58(X); dv = bdelta(X, C['v4']); dm = bdelta(X, C['mix50 (file)']); dt = bdelta(X, base_v09t)
    P = project(X)
    recC.append({'target': k, 'gold58': p.mean(), 'dv4': dv[0], 'P>v4': dv[2], 'dmix': dm[0], 'P>mix': dm[2],
                 'dv09t': dt[0], 'sd_v09t': dt[1], 'P>v09t': dt[2],
                 'rho_rap': mrho(X, rap, V867), 'ws_rap': ws(X, rap),
                 'proj_gold58': per58(P).mean(), 'proj_rho_rap': mrho(P, rap, V867), 'proj_ws_rap': ws(P, rap),
                 **{f'a_{l[:6]}': p[j] for j, l in enumerate(T)}})
CT = pd.DataFrame(recC).set_index('target'); CT.to_csv(f'{OUT}/B2_candidates.csv')
print('\n== 2) CANDIDATE TARGETS: gold58 (MEASURED, target level) | paired bootstrap deltas | parent-likeness on 867 '
      '| projection student (MODEL)')
print(CT[['gold58', 'dv4', 'P>v4', 'dmix', 'P>mix', 'dv09t', 'sd_v09t', 'P>v09t', 'rho_rap', 'ws_rap', 'proj_gold58',
          'proj_rho_rap', 'proj_ws_rap']].round(4).to_string())
print('\nper-label gold58 AUC (selected):')
sel = ['v4', 'mix50 (file)', 'gated75 (file)', 'Ev4 ALL .50 (v09t analog)', 'Ev4 UNK .75', 'Ev4 UNK+NO .75',
       'Ev4 UNK .9 NO .5 YES .25', 'Ev4 UNC a1.5 cap.9', 'Emix UNK .9 NO .5 YES .25', 'Eall UNK .9 NO .5 YES .25',
       'mix50 + Ev4 UNK .75']
print(CT.loc[[s for s in sel if s in CT.index], [c for c in CT.columns if c.startswith('a_')]].round(3).to_string())
# students' own gold58 per label, for reference
print('\nstudents per-label gold58:')
print(pd.DataFrame({k: per58(ST[k]) for k in ST}, index=[l[:6] for l in T]).T.round(3).to_string())
# projection validation: projected v4 / mix50 / gated vs the real students
print('\nprojection validation (MODEL vs MEASURED students):')
for t, real in [('v4', ['F1v4', 'TTRv4']), ('mix50 (file)', ['F1mix', 'TTRmix']), ('gated75 (file)', ['F1gated'])]:
    r = CT.loc[t]
    print(f'  {t:15s} proj gold58 {r.proj_gold58:.4f} rho {r.proj_rho_rap:.3f} ws {r.proj_ws_rap:.3f} | real: ' +
          ' '.join(f'{k} {TR.loc[k, "stu_gold58"]:.4f}/{TR.loc[k, "stu_rho_rap"]:.3f}/{TR.loc[k, "stu_ws_rap"]:.3f}' for k in real))
# verdict-stratum view on gold58: P(gold=1) and AUC inside each stratum (label-wise rank-normalised scores)
print('\ngold58 within-verdict AUC (cells pooled, label-wise ranks):')
Vg = VER.loc[G58]
for st in ['YES', 'UNK', 'NO']:
    yy = np.concatenate([Y58[l].values[(Vg[l] == st).values] for l in T]); line = f'  {st:3s} n {len(yy):3d} P1 {yy.mean():.3f} |'
    for nm, X in [('v4', C['v4']), ('mix50', C['mix50 (file)']), ('TTRv4', ST['TTRv4']), ('F1v4', ST['F1v4']),
                  ('Ev4', own_src(SRC['Ev4'])), ('TTRmix', ST['TTRmix']), ('F1mix', ST['F1mix']), ('Eall', own_src(SRC['Eall']))]:
        ss = np.concatenate([((rankdata(X.loc[G58, l]) - .5) / 58)[(Vg[l] == st).values] for l in T])
        line += f' {nm} {roc_auc_score(yy, ss):.3f}'
    print(line)
print('\nwrote', OUT)
