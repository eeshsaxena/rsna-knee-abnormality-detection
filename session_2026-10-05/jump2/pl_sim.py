# pl_sim.py (jump2, 2026-10-05): pseudo-label / self-distillation TARGET simulation, CPU only, read-only inputs.
# A) fold-0 val (867 studies, every fleet OOF is out-of-sample there): parent-likeness of candidate targets
#    (mean per-label Spearman vs the parent's Raptor proxy, maxspan_v5, native384_v8 and the widedense teacher).
# B) gold58 (radiologist image labels; gold74 is --exclude'd from every fleet run, so the fold-0 checkpoints'
#    predictions on these rows (~/jump2/gold74_preds, gold_infer.py) have the same out-of-sample status as an OOF):
#    target AUC vs gold58 + within-class Spearman vs the Raptor proxy (labels_ownfill.py convention) + paired bootstrap.
# C) student-transfer calibration: target metrics vs the measured fleet students trained on them (F1 v4 / mix50 / gated).
# Writes ~/jump2/pl_sim_out/*.csv and prints tables.
import os, sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

H = os.path.expanduser
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
OUT = H('~/jump2/pl_sim_out'); os.makedirs(OUT, exist_ok=True)
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')[T]
mix50 = rd('~/rsna_train/labels_mix50_logit.csv')[T]
gated = rd('~/rsna_train/labels_gated75_logit.csv')[T]
wide = rd('~/rsna_train/teacher_train_probs.csv')[T]
rap = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')[T]
v5 = rd('~/fleet/teacher_v2_maxspan_v5_shard0.csv')[T]
v8 = rd('~/fleet/teacher_v2_native384_v8_shard0.csv')[T]
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
O = {k: rd(v)[T] for k, v in OOF.items()}
V867 = sorted(set.intersection(*[set(o.index) for o in O.values()]))
assert not set(V867) & set(gold.index), 'gold rows inside the fold-0 val set?'
gp = {}
for k, f in GP.items():
    p = H(f'~/jump2/gold74_preds/{f}.csv')
    if os.path.exists(p): gp[k] = rd(p)[T]
HAVE_GOLD = len(gp) == len(GP)
print(f'fold-0 val n={len(V867)} | gold58 n={len(G58)} | gold preds for {sorted(gp)} (all={HAVE_GOLD})')


def lg(a): return logit(np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4))
def rk(a): return (rankdata(a) - 0.5) / len(a)


# own sources: one OOF-status prediction per row (867 val rows from the OOF csv, gold rows from the fold-0 ckpt)
def own(k, rows):
    out = []
    for s in rows:
        out.append(O[k].loc[s].values if s in O[k].index else gp[k].loc[s].values)
    return pd.DataFrame(out, index=rows, columns=T)


SRC = {'TTRv4': ['TTRv4'], 'F1v4': ['F1v4'], 'TTRmix': ['TTRmix'], 'F1mix': ['F1mix'],
       'Ev4': ['TTRv4', 'F1v4'], 'Emix': ['TTRmix', 'F1mix'], 'Eall': ['TTRv4', 'F1v4', 'TTRmix', 'F1mix']}


def own_src(name, rows):
    # ensemble = mean of per-model logits standardised on the 867 val distribution (no rank: keeps gold rows mappable)
    zs = []
    for k in SRC[name]:
        ref = lg(O[k].loc[V867].values); mu, sd = ref.mean(0), ref.std(0)
        zs.append((lg(own(k, rows).values) - mu) / sd)
    z = np.mean(zs, 0)
    # standardise the ensemble z on the 867 too, then map to v4's logit mean/std (build_mixSA 'logit' match)
    zr = np.mean([(lg(O[k].loc[V867].values) - lg(O[k].loc[V867].values).mean(0)) / lg(O[k].loc[V867].values).std(0)
                  for k in SRC[name]], 0)
    z = (z - zr.mean(0)) / zr.std(0)
    lv = lg(v4.loc[V867].values)
    return pd.DataFrame(expit(z * lv.std(0) + lv.mean(0)), index=rows, columns=T)


def matched_teacher(tab, rows):  # logit mean/std match onto v4 over ALL 4407 (as build_mixSA / build_gated)
    la = lg(tab.loc[v4.index].values); lv = lg(v4.values)
    z = (lg(tab.loc[rows].values) - la.mean(0)) / la.std(0)
    return pd.DataFrame(expit(z * lv.std(0) + lv.mean(0)), index=rows, columns=T)


def fill(rows, src, strata, a, yes_teacher=0.0, space='prob'):
    """target = v4 except cells whose report verdict is in strata: (1-a) v4 + a own; report-YES cells optionally
    (1-yes_teacher) v4 + yes_teacher Raptor (matched). strata 'ALL' = every cell (flat soft bootstrap)."""
    b = v4.loc[rows].values.copy(); S = own_src(src, rows).values; V = VER.loc[rows].values
    m = np.ones_like(b, bool) if strata == 'ALL' else np.isin(V, strata)
    if space == 'prob':
        b[m] = (1 - a) * b[m] + a * S[m]
    else:
        bl = lg(b); bl[m] = (1 - a) * bl[m] + a * lg(S)[m]; b = expit(bl)
    if yes_teacher > 0:
        R = matched_teacher(rap, rows).values; my = (V == 'YES') & ~m
        b[my] = (1 - yes_teacher) * b[my] + yes_teacher * R[my]
    return pd.DataFrame(b, index=rows, columns=T)


def cands(rows):
    C = {'v4': v4.loc[rows], 'mix50(file)': mix50.loc[rows], 'gated75(file)': gated.loc[rows]}
    for src in ['TTRv4', 'Ev4', 'TTRmix', 'Emix', 'Eall']:
        for st, nm in [(['UNK'], 'UNK'), (['UNK', 'NO'], 'UNK+NO'), ('ALL', 'ALL')]:
            for a in ([0.5, 0.75, 1.0] if nm != 'ALL' else [0.25, 0.5, 0.75]):
                C[f'{src} fill {nm} a{a}'] = fill(rows, src, st, a)
        C[f'{src} DSD UNK+NO a.75 +RapYES .25'] = fill(rows, src, ['UNK', 'NO'], 0.75, 0.25)
        C[f'{src} UNK+NO a.75 + own YES .25'] = _two_level(rows, src, 0.75, 0.25)
    return C


def _two_level(rows, src, a_sil, a_addr):  # own weight a_sil on UNK+NO, a_addr on YES (no Raptor anywhere)
    b = v4.loc[rows].values.copy(); S = own_src(src, rows).values; V = VER.loc[rows].values
    w = np.where(np.isin(V, ['UNK', 'NO']), a_sil, a_addr)
    return pd.DataFrame((1 - w) * b + w * S, index=rows, columns=T)


def mrho(A, B):  # mean per-label Spearman
    return float(np.nanmean([spearmanr(A[l], B[l]).correlation for l in T]))


# ---------------- A) fold-0 val: parent-likeness of targets and of the measured students ----------------
rowsA = V867
CA = cands(rowsA)
refs = {'rap': rap, 'v5': v5, 'v8': v8, 'wide': wide}
recA = []
for k, X in CA.items():
    r = {'target': k}
    for rn, R in refs.items(): r[f'rho_{rn}'] = mrho(X, R.loc[rowsA])
    r['rho_v4'] = mrho(X, v4.loc[rowsA]); recA.append(r)
for k in O:  # the students themselves (OOF)
    X = O[k].loc[rowsA]; r = {'target': f'[student OOF] {k}'}
    for rn, R in refs.items(): r[f'rho_{rn}'] = mrho(X, R.loc[rowsA])
    r['rho_v4'] = mrho(X, v4.loc[rowsA]); recA.append(r)
A = pd.DataFrame(recA).set_index('target'); A.to_csv(f'{OUT}/A_fold0val_parentlikeness.csv')
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print('\n== A) fold-0 val 867: mean per-label Spearman of TARGET (or student OOF) vs parent arms / v4')
print(A.round(3).to_string())

# ---------------- B) gold58: target quality + within-class rho to the Raptor proxy ----------------
if HAVE_GOLD:
    CB = cands(G58)
    def per(X): return np.array([roc_auc_score(Y58[l], X[l]) for l in T])
    def wc(X, R):
        r = []
        for l in T:
            for c in (0, 1):
                m = (Y58[l] == c).values
                if m.sum() >= 6 and pd.Series(X[l].values[m]).nunique() > 2:
                    r.append(spearmanr(X[l].values[m], R[l].values[m]).correlation)
        return float(np.nanmean(r))
    R58 = rap.loc[G58]
    rng = np.random.default_rng(0); B = [rng.integers(0, len(G58), len(G58)) for _ in range(1000)]
    def boot_delta(X, Z):
        d = []
        for i in B:
            v = [roc_auc_score(Y58[l].values[i], X[l].values[i]) - roc_auc_score(Y58[l].values[i], Z[l].values[i])
                 for l in T if 0 < Y58[l].values[i].mean() < 1]
            d.append(np.mean(v))
        d = np.array(d); return d.mean(), d.std(), (d > 0).mean()
    recB = []
    for k, X in CB.items():
        p = per(X); dm, ds, pp = boot_delta(X, CB['mix50(file)']); dv, dvs, pv = boot_delta(X, CB['v4'])
        recB.append({'target': k, 'gold58': p.mean(), 'wc_rho_rap': wc(X, R58), 'd_vs_mix50': dm, 'sd': ds, 'P>mix50': pp,
                     'd_vs_v4': dv, 'P>v4': pv, **{f'auc_{l}': p[j] for j, l in enumerate(T)}})
    for k in gp:  # the students (single fold-0 model each) on gold58
        X = gp[k].loc[G58]; p = per(X)
        recB.append({'target': f'[student f0] {k}', 'gold58': p.mean(), 'wc_rho_rap': wc(X, R58),
                     **{f'auc_{l}': p[j] for j, l in enumerate(T)}})
    for nm, R in [('[parent arm] raptor proxy', rap), ('[parent arm] widedense', wide)]:
        X = R.loc[G58]; p = per(X)
        recB.append({'target': nm, 'gold58': p.mean(), 'wc_rho_rap': wc(X, R58), **{f'auc_{l}': p[j] for j, l in enumerate(T)}})
    Bt = pd.DataFrame(recB).set_index('target'); Bt.to_csv(f'{OUT}/B_gold58_targets.csv')
    print('\n== B) gold58: target AUC, within-class Spearman vs Raptor proxy, paired bootstrap vs mix50 / v4')
    print(Bt[['gold58', 'wc_rho_rap', 'd_vs_mix50', 'sd', 'P>mix50', 'd_vs_v4', 'P>v4']].round(4).to_string())
    # per-label for the headline rows
    hl = ['v4', 'mix50(file)', 'gated75(file)', 'TTRv4 fill UNK+NO a0.75', 'Ev4 fill UNK+NO a0.75',
          'Ev4 DSD UNK+NO a.75 +RapYES .25', 'Emix fill UNK+NO a0.75', 'Eall fill UNK+NO a0.75', 'Ev4 fill ALL a0.5',
          '[student f0] TTRv4', '[student f0] TTRmix', '[student f0] F1mix']
    print('\nper-label gold58 AUC (headline rows):')
    print(Bt.loc[[h for h in hl if h in Bt.index], [f'auc_{l}' for l in T]].round(3).rename(columns=lambda c: c[4:12]).to_string())
    # report-verdict strata check on the 58: P(gold=1) and student AUC inside UNK/NO cells
    print('\nwithin-stratum AUC on gold58 (cells pooled over labels, label-wise rank-normalised scores):')
    for st in ['YES', 'UNK', 'NO']:
        line = f'  {st:3s}'
        Vg = VER.loc[G58]
        yy = np.concatenate([Y58[l].values[(Vg[l] == st).values] for l in T])
        line += f' cells {len(yy):3d} P(gold=1) {yy.mean():.3f} |'
        for nm, X in [('v4', v4.loc[G58]), ('TTRv4', gp['TTRv4'].loc[G58]), ('Ev4', own_src('Ev4', G58)),
                      ('TTRmix', gp['TTRmix'].loc[G58]), ('rap', rap.loc[G58])]:
            ss = np.concatenate([rk(X[l].values)[(Vg[l] == st).values] for l in T])
            line += f' {nm} {roc_auc_score(yy, ss) if 0 < yy.mean() < 1 else float("nan"):.3f}'
        print(line)
else:
    print('\n(gold preds incomplete: section B skipped)')
print('\nwrote', OUT)
