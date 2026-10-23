# target_menu.py (research_1008 dimension C, 2026-10-08). Light CPU, laptop, read-only on its inputs.
# Per-label TARGET quality menu vs expert gold, with paired bootstrap SEs and an Efron optimism estimate for
# choosing the per-label teacher weight on the same gold58 studies.
# Teacher (logit-matched Dread widedense, the mix50 teacher) is recovered exactly as tm = 2*mix50 - v4
# (mix50 = .5 v4 + .5 tm in prob space; verify_perlabel row 11 / build_frac75 step 1).
import os, sys, json, hashlib
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

R = 'C:/Users/eeshs/Downloads/rsna-knee'
OLD = 'C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad'
DS = R + '/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
P = dict(gold=DS + '/gold74_eval_set.csv', mix50=DS + '/labels_mix50_logit.csv', v4=DS + '/llm_labels_v4_blend.csv',
         folds=DS + '/folds5.csv', frac75=R + '/session_2026-10-08/k8/frac75/labels_frac75_logit.csv',
         gated75=R + '/session_2026-10-04/bigjump/labels_gated75_logit.csv', mixSA=OLD + '/labels_mixSA_logit.csv',
         raptor=OLD + '/teacher_v2_out/raptor_proxy_v5v8.csv', rep=OLD + '/kneelabels/rsna-knee-llm-labels/report_labels_v2.csv')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
SH = ['ACL', 'MCL', 'MedMen', 'LatMen', 'MedOA', 'LatOA', 'PFOA', 'Eff', 'Syn', 'Bak', 'Cont', 'Frac']
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
for k, p in P.items():
    print(f'input {k:8s} md5 {md5(p)}')
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
mix = rd(P['mix50']); ids = mix.index
v4 = rd(P['v4']).reindex(ids)[T]; tm = 2 * mix[T] - v4
assert (tm.values > -1e-9).all() and (tm.values < 1 + 1e-9).all()
f75 = rd(P['frac75']).reindex(ids)[T]
assert np.abs((.25 * v4['Fracture'] + .75 * tm['Fracture']) - f75['Fracture']).max() < 1e-9   # recovery proof
rap = rd(P['raptor']).reindex(ids)[T]
rep = rd(P['rep']).reindex(ids)
gated = rd(P['gated75']).reindex(ids)[T]; msa = rd(P['mixSA']).reindex(ids)[T]
gold = rd(P['gold']); g58 = gold.index[gold.source == 'original_gold58']; g16 = gold.index[gold.source != 'original_gold58']
print('teacher recovered from mix50/v4; frac75 Fracture == .25 v4 + .75 tm (max diff < 1e-9)')
print(f'gold58 n {len(g58)} | gold16 n {len(g16)} (report-read labels, text-biased; info only)')


def mix_w(w, lab):
    return (1 - w) * v4[lab] + w * tm[lab]


def gated_mix(lab, w_sil, w_addr, teacher=None):
    t = tm[lab] if teacher is None else teacher[lab]
    ver = rep[lab + '__verdict'].fillna('UNK').values
    w = np.where(ver == 'UNK', w_sil, w_addr)
    return (1 - w) * v4[lab] + w * t


def auc_vec(y, S):
    """y (n,) bool; S (n,k) scores -> AUC per column (Mann-Whitney with ties)."""
    n1 = y.sum(); n0 = len(y) - n1
    rk = np.apply_along_axis(rankdata, 0, S)
    return (rk[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def recipes(lab):
    """menu of candidate targets for one label -> dict name -> Series over all ids"""
    m = {f'w{int(w * 100):03d}': mix_w(w, lab) for w in (0, .25, .5, .75, 1.0)}
    m['gate_s75_a50 (mixSA-like)'] = gated_mix(lab, .75, .5)
    m['gate_s75_a25'] = gated_mix(lab, .75, .25)
    m['gate_s100_a50'] = gated_mix(lab, 1.0, .5)
    m['gate_s50_a25'] = gated_mix(lab, .5, .25)
    m['gate_s50_a00'] = gated_mix(lab, .5, 0.0)
    m['gated75 file (raptor sil .75, addr 0)'] = gated[lab]
    m['raptor_rank'] = rap[lab]
    # 3-way: .5 v4 + .25 tm + .25 raptor (prob space; raptor logit-matched onto v4)
    from scipy.special import logit, expit
    lr = logit(np.clip(v4[lab], 1e-4, 1 - 1e-4)); la = logit(np.clip(rap[lab], 1e-4, 1 - 1e-4))
    rm = pd.Series(expit((la - la.mean()) / la.std() * lr.std() + lr.mean()), index=ids)
    m['v4 .5 + tm .25 + rap .25'] = .5 * v4[lab] + .25 * tm[lab] + .25 * rm
    m['v4 .5 + rap .5'] = .5 * v4[lab] + .5 * rm
    return m


rng = np.random.default_rng(0)
B = 2000
n58 = len(g58)
boot = rng.integers(0, n58, size=(B, n58))
out = {}
lines = []
for lab, sh in zip(T, SH):
    y58 = gold.loc[g58, lab].values
    ok = np.isfinite(y58)
    yb = (y58 >= .5)
    M = recipes(lab)
    names = list(M)
    S58 = np.column_stack([M[k].reindex(g58).values for k in names])
    a58 = auc_vec(yb[ok], S58[ok])
    y16 = gold.loc[g16, lab].values; ok16 = np.isfinite(y16); yb16 = y16[ok16] >= .5
    S16 = np.column_stack([M[k].reindex(g16).values for k in names])[ok16]
    a16 = auc_vec(yb16, S16) if 0 < yb16.mean() < 1 else np.full(len(names), np.nan)
    # paired bootstrap of every recipe minus mix50 (w050)
    j50 = names.index('w050')
    D = np.zeros((B, len(names)))
    for b in range(B):
        idx = boot[b]
        yy = yb[idx]; okb = ok[idx]
        if not (0 < yy[okb].mean() < 1):
            D[b] = np.nan; continue
        ab = auc_vec(yy[okb], S58[idx][okb])
        D[b] = ab - ab[j50]
    se = np.nanstd(D, 0); pgt = np.nanmean(D > 0, 0)
    # train-pool parent-likeness (Spearman vs raptor proxy) and Spearman vs mix50, on non-gold studies
    pool = ids.difference(gold.index)
    rho_r = [spearmanr(M[k].reindex(pool), rap[lab].reindex(pool)).correlation for k in names]
    rho_m = [spearmanr(M[k].reindex(pool), mix[lab].reindex(pool)).correlation for k in names]
    out[lab] = dict(names=names, gold58=a58.tolist(), gold16=a16.tolist(), d_vs_mix50=(a58 - a58[j50]).tolist(),
                    se=se.tolist(), p_gt0=pgt.tolist(), rho_raptor=rho_r, rho_mix50=rho_m,
                    npos58=int(yb[ok].sum()), n58=int(ok.sum()))
    lines.append(f'\n== {lab} (gold58 n {int(ok.sum())}, pos {int(yb[ok].sum())})')
    lines.append(f'{"recipe":40s} {"gold58":>7s} {"d_vs_mix50":>10s} {"bootSE":>7s} {"P>0":>5s} {"gold16":>7s} {"rhoRap":>7s} {"rhoMix":>7s}')
    for i, k in enumerate(names):
        lines.append(f'{k:40s} {a58[i]:7.4f} {a58[i] - a58[j50]:+10.4f} {se[i]:7.4f} {pgt[i]:5.2f} {a16[i]:7.4f} {rho_r[i]:7.3f} {rho_m[i]:7.3f}')
print('\n'.join(lines))

# ---------------- Efron optimism: choose the best of the 5 w-grid per label on a bootstrap sample, score on original
print('\n== Efron optimism of choosing the per-label teacher weight w on gold58 (grid w in {0,.25,.5,.75,1}; B=2000)')
print(f'{"label":10s} {"best_w":>6s} {"apparent":>9s} {"optimism":>9s} {"corrected":>10s} {"P(choose!=.5)":>13s} {"shrink":>7s}')
opt_tab = {}
for lab, sh in zip(T, SH):
    y58 = gold.loc[g58, lab].values; ok = np.isfinite(y58); yb = y58 >= .5
    S = np.column_stack([mix_w(w, lab).reindex(g58).values for w in (0, .25, .5, .75, 1.0)])
    a_orig = auc_vec(yb[ok], S[ok]); j_app = int(np.argmax(a_orig)); app = a_orig[j_app] - a_orig[2]
    opts = []; ch = []
    for b in range(B):
        idx = boot[b]; yy = yb[idx]; okb = ok[idx]
        if not (0 < yy[okb].mean() < 1):
            continue
        ab = auc_vec(yy[okb], S[idx][okb]); j = int(np.argmax(ab))
        in_gain = ab[j] - ab[2]; orig_gain = a_orig[j] - a_orig[2]
        opts.append(in_gain - orig_gain); ch.append(j != 2)
    o = float(np.mean(opts)); corr = app - o
    shrink = corr / app if app > 1e-9 else np.nan
    opt_tab[lab] = dict(best_w=[0, .25, .5, .75, 1.0][j_app], apparent=app, optimism=o, corrected=corr,
                        p_choose_other=float(np.mean(ch)))
    print(f'{sh:10s} {[0, .25, .5, .75, 1.0][j_app]:6.2f} {app:+9.4f} {o:9.4f} {corr:+10.4f} {np.mean(ch):13.2f} {shrink:7.2f}')

# ---------------- cross-label Synovitis: expert co-occurrence and Effusion-informed Synovitis targets
print('\n== Synovitis vs Effusion on gold58 (expert labels)')
ys = gold.loc[g58, 'Synovitis'].values >= .5; ye = gold.loc[g58, 'Effusion'].values >= .5
print(pd.crosstab(pd.Series(ye, name='Eff_gold'), pd.Series(ys, name='Syn_gold')))
cand = {'Syn mix50': mix['Synovitis'], 'Eff mix50 (as a Syn score)': mix['Effusion'],
        'rank .5 Syn mix50 + .5 Eff mix50': pd.Series(rankdata(mix['Synovitis']) + rankdata(mix['Effusion']), index=ids),
        'rank .75 Syn + .25 Eff': pd.Series(3 * rankdata(mix['Synovitis']) + rankdata(mix['Effusion']), index=ids),
        'Syn v4 (text)': v4['Synovitis'], 'Syn teacher (tm)': tm['Synovitis']}
Sx = np.column_stack([c.reindex(g58).values for c in cand.values()])
ax = auc_vec(ys, Sx)
Dx = np.array([auc_vec(ys[i], Sx[i]) - auc_vec(ys[i], Sx[i])[0] for i in boot])
for k, a, s, p in zip(cand, ax, Dx.std(0), (Dx > 0).mean(0)):
    print(f'{k:36s} gold58 AUC {a:.4f}  d_vs_mix50 {a - ax[0]:+.4f} SE {s:.4f} P>0 {p:.2f}')
json.dump(dict(menu=out, optimism=opt_tab), open(os.path.join(os.path.dirname(__file__), 'target_menu.json'), 'w'), indent=1)
print('\nwrote target_menu.json')
