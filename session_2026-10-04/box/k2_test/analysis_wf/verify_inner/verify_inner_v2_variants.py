"""Verifier step 2: independent recompute of the side-level and final-level injection numbers.
Own loaders for the Raptor/CoAt/final path (per-arm csvs in localLB_parts, not raptor_raw.npz), own AUC + bootstrap.
inject_common is used ONLY for the transformer-side stage2/stage3 functions (A5 + Rad + calibrator), which re-run
reproduces the stored stage-3 file to 1.1e-16 (rerun_reconstruct.log), plus its loader for the calibrator payload.
Three DINO held-out fold assignments are compared for the pseudo-OOS bracket: agent (label-loss argmax), oof.npz match,
label-free deviation. Rad held-out head = fold column of v52_oof.csv (agent's recovery matched it 58/58).
Also: bracket calibration against the measured LB anchors of the crop global blends."""
import sys, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/home/training/k2_test/analysis_wf')
import inject_common as ic

T = ic.T
H = '/home/training'; R = H + '/kmimic/runs/gold58_gpu'; P = R + '/localLB_parts'; W = R + '/kaggle/working'
ids = pd.read_csv(R + '/kaggle/input/rsna-knee-abnormality-detection/test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
n = len(ids)
tr = pd.read_csv(H + '/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
Y = tr.loc[ids, T].to_numpy(float)


def rd(path):
    d = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert set(ids) <= set(d.index), path
    return d.loc[ids, T].to_numpy(np.float64)


def rp(x):   # rank pct average ties, per column
    return pd.DataFrame(np.asarray(x, np.float64)).rank(method='average', pct=True).to_numpy()


def r01(x):
    return (pd.DataFrame(np.asarray(x, np.float64)).rank(method='average').to_numpy() - 1) / (len(x) - 1)


def auc12(X, w=None):
    return np.array([roc_auc_score(Y[:, j], X[:, j]) for j in range(12)])


def ma(X):
    return float(auc12(X).mean())


rng = np.random.default_rng(12345)
BOOT = [rng.integers(0, n, n) for _ in range(1000)]


def boot(Xn, Xo):
    d = []
    for b in BOOT:
        yb = Y[b]; v = []
        for j in range(12):
            if 0 < yb[:, j].sum() < n:
                v.append(roc_auc_score(yb[:, j], Xn[b, j]) - roc_auc_score(yb[:, j], Xo[b, j]))
        d.append(np.mean(v))
    d = np.array(d)
    return np.quantile(d, .05), np.quantile(d, .95), (d > 0).mean()


# ---------------- independent loads ----------------
F = rd(H + '/k2_test/runs/C_gold58/working/submission_fleetsolo.csv')
arm_names = ['maxspan-v5', 'native384dense-v10', 'maxspan-v5-reverse', 'native384-v8']
w_arm = np.array([.60, .10, .10, .20])
arms = [rd(P + f'/raptor_arm_{a}_prob.csv') for a in arm_names]
z = np.load(W + '/raptor_raw.npz'); pos = {str(s): i for i, s in enumerate(z['study_uids'])}
rr = z['raw_probabilities'][:, [pos[s] for s in ids]].astype(np.float64)
print('per-arm csv vs raptor_raw.npz max|diff|', max(np.abs(arms[k] - rr[k]).max() for k in range(4)))
coat = {m: rd(P + f'/coat_{m}_prob.csv') for m in ['resgated', 'global96', 'd4', 'repairv1']}
st3 = rd(P + '/pipeline_stage_3_transformer_side_dino_a5_rad.csv')
st4 = rd(P + '/pipeline_stage_4_final_outer_blend.csv')
pub_csv = rd(P + '/raptor_public_rank.csv'); fam_csv = rd(P + '/coat_family_rank.csv'); side_csv = rd(P + '/raptor_coat_side_rank.csv')
cw = np.array([{'ACL': .75, 'Medial Meniscus': .80, 'Lateral Meniscus': 1.0, 'Lateral OA': .75, 'Fracture': .75}.get(t, .60) for t in T])

pb0 = sum(w * np.clip(a, 0, 1) for w, a in zip(w_arm, arms)) / w_arm.sum()
pub0 = r01(pb0).astype(np.float32).astype(np.float64)
fam0 = rp(sum(coat.values()) / 4.0)
side0 = .6 * rp(pub0) + .4 * fam0
fin = lambda t, s: rp((1 - cw) * rp(t) + cw * s)
fin0 = fin(st3, side0)
print(f'pub0 vs stored {np.abs(pub0 - pub_csv).max():.1e} ranks equal {bool((rp(pub0) == rp(pub_csv)).all())}; fam0 vs stored {np.abs(fam0 - fam_csv).max():.1e}; '
      f'side0 vs stored {np.abs(side0 - side_csv).max():.1e}; final0 vs stored {np.abs(fin0 - st4).max():.4f} '
      f'({int((np.abs(fin0 - st4) > 1e-9).sum())} cells)')
print(f'gold58 AUC: pub {ma(pub0):.4f} fam {ma(fam0):.4f} side {ma(side0):.4f} st3 {ma(st3):.4f} final {ma(fin0):.4f} '
      f'stored final {ma(st4):.4f} fleet {ma(F):.4f}')
from scipy.stats import spearmanr
sp = lambda A, B: float(np.mean([spearmanr(A[:, j], B[:, j]).correlation for j in range(12)]))
print(f'Spearman mean: fleet~pub {sp(F, pub0):.3f} fleet~fam {sp(F, fam0):.3f} pub~fam {sp(pub0, fam0):.3f} fleet~side {sp(F, side0):.3f} '
      f'fleet~final {sp(F, st4):.3f}')
print('fleet prob std / Raptor prob-blend std (mean over labels):', round(float((F.std(0) / pb0.std(0)).mean()), 3))

# ---------------- pseudo-OOS transformer sides ----------------
D = ic.load_all()          # for the A5 / Rad / calibrator / protocol inputs only
fz = np.load(H + '/k2_test/analysis_wf/verify_inner/verify_folds.npz')
dino = D['dino'].astype(np.float64); folds = np.array(D['dino_fold']); runs = np.array(D['dino_run'])
# label-free fold: the held-out member deviates most from its 4 sibling-fold members of the same run
LF = []
for r in sorted(set(runs)):
    ix = np.where(runs == r)[0]
    dev = np.stack([np.abs(dino[i] - dino[[k for k in ix if k != i]].mean(0)).mean(1) for i in ix])
    LF.append(folds[ix][np.argmax(dev, 0)])
LF = np.stack(LF); lf_fold = np.array([Counter(LF[:, s].tolist()).most_common(1)[0][0] for s in range(n)])
print('label-free fold: all 4 runs agree on', int((LF == LF[0]).all(0).sum()), '| == agent fold', int((lf_fold == fz['agent_fold']).sum()),
      '| == oof.npz-match fold', int((lf_fold == fz['true_fold']).sum()), 'of', n)
rf = fz['rad_fold_v52']
proto = ic.protocol_features(D['tser'], ids)
rad_oof = {k: D[k][rf, np.arange(n)][None].astype(np.float32) for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw')}
Toof = {}
for nm, fo in [('agent', fz['agent_fold']), ('oofmatch', fz['true_fold']), ('labelfree', lf_fold)]:
    d1 = np.stack([dino[folds == fo[s], s].mean(0) for s in range(n)])
    s2 = ic.stage2_a5(rp(d1), D['a5_rank_mean'], D['a5_ok'])
    Toof[nm] = ic.stage3_rad(s2, rad_oof['rad_e10_raw'], rad_oof['rad_e13_raw'], rad_oof['rad_legacy_second_raw'], D['cal'], proto)
    print(f'T_oof[{nm:9s}] DINO-oof {ma(d1):.4f} T {ma(Toof[nm]):.4f} final(T_oof) {ma(fin(Toof[nm], side0)):.4f}')
# 4th: DINO stage 1 = the shipped oof.npz predictions
zo = np.load(H + '/kmimic/kaggle/input/rsna-knee-weights/oof.npz'); po = {str(s): i for i, s in enumerate(zo['ids'])}
oofp = zo['pred'][[po[s] for s in ids]].astype(np.float64)
s2 = ic.stage2_a5(rp(oofp), D['a5_rank_mean'], D['a5_ok'])
Toof['oofnpz'] = ic.stage3_rad(s2, rad_oof['rad_e10_raw'], rad_oof['rad_e13_raw'], rad_oof['rad_legacy_second_raw'], D['cal'], proto)
print(f'T_oof[oofnpz   ] DINO-oof {ma(oofp):.4f} T {ma(Toof["oofnpz"]):.4f} final(T_oof) {ma(fin(Toof["oofnpz"], side0)):.4f}')

# ---------------- variants ----------------
def k2g(f, X, Wg):
    return rp((1 - Wg) * rp(f) + Wg * rp(X))


V = {}
for f in (.10, .20, .33):
    pn = r01((1 - f) * rp(pb0) + f * rp(F)).astype(np.float32).astype(np.float64)
    V[f'5th arm rank f={f:.2f}'] = ('side', pn, fam0)
w5 = np.append(w_arm, .25 / .75)
pn = r01(sum(w * np.clip(a, 0, 1) for w, a in zip(w5, arms + [F])) / w5.sum()).astype(np.float32).astype(np.float64)
V['5th arm prob f=.25'] = ('side', pn, fam0)
wr = w_arm.copy(); wr[2] = 0; w5 = np.append(wr, .10)
pn = r01(sum(w * np.clip(a, 0, 1) for w, a in zip(w5, arms + [F])) / w5.sum()).astype(np.float32).astype(np.float64)
V['replace v5-reverse (prob)'] = ('side', pn, fam0)
V['coat add f=.20'] = ('side', pub0, rp(.8 * sum(coat.values()) / 4.0 + .2 * F))
rows = []
for name, (_, pn, fn) in V.items():
    sn = .6 * rp(pn) + .4 * fn
    r = dict(variant=name, dPub=ma(pn) - ma(pub0), dSide=ma(sn) - ma(side0), dFin_in=ma(fin(st3, sn)) - ma(fin0),
             MARS=float(np.abs(fin(st3, sn) - fin0).mean()))
    for k, Tk in Toof.items():
        r['dFin_pOOS_' + k] = ma(fin(Tk, sn)) - ma(fin(Tk, side0))
    rows.append(r)
for Wg in (.05, .10, .14, .15, .20, .25, .30, .40):
    r = dict(variant=f'global W={Wg:.2f}', dPub=0.0, dSide=0.0, dFin_in=ma(k2g(fin0, F, Wg)) - ma(fin0), MARS=float(np.abs(k2g(fin0, F, Wg) - fin0).mean()))
    for k, Tk in Toof.items():
        b = fin(Tk, side0); r['dFin_pOOS_' + k] = ma(k2g(b, F, Wg)) - ma(b)
    rows.append(r)
res = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(res.round(4).to_string(index=False))
res.to_csv('verify_inner_v2_variants.csv', index=False)

# bootstrap CIs for the headline numbers (1000 study resamples, own implementation)
pn20 = V['5th arm rank f=0.20'][1]; sn20 = .6 * rp(pn20) + .4 * fam0
for lab, Xn, Xo in [('5th arm rank .20: dPub', pn20, pub0), ('5th arm rank .20: dSide', sn20, side0),
                    ('5th arm rank .20: dFinal in-sample', fin(st3, sn20), fin0),
                    ('5th arm rank .20: dFinal pOOS[agent]', fin(Toof['agent'], sn20), fin(Toof['agent'], side0)),
                    ('5th arm rank .20: dFinal pOOS[labelfree]', fin(Toof['labelfree'], sn20), fin(Toof['labelfree'], side0)),
                    ('global W .30: dFinal in-sample', k2g(fin0, F, .30), fin0),
                    ('global W .30: dFinal pOOS[agent]', k2g(fin(Toof['agent'], side0), F, .30), fin(Toof['agent'], side0)),
                    ('global W .30: dFinal pOOS[labelfree]', k2g(fin(Toof['labelfree'], side0), F, .30), fin(Toof['labelfree'], side0)),
                    ('inner .20 minus global .14 (in-sample)', fin(st3, sn20), k2g(fin0, F, .14)),
                    ('inner .20 minus global .14 (pOOS agent)', fin(Toof['agent'], sn20), k2g(fin(Toof['agent'], side0), F, .14))]:
    lo, hi, p = boot(Xn, Xo)
    print(f'{lab:44s} point {ma(Xn) - ma(Xo):+.4f}  CI90 [{lo:+.4f},{hi:+.4f}]  P>0 {p:.2f}', flush=True)

# ---------------- bracket calibration vs measured LB anchors (crop global blends) ----------------
zc = np.load(H + '/k2_test/analysis_wf/proxy_crop_gold74_preds.npz', allow_pickle=True)
pc = {str(s): i for i, s in enumerate(zc['ids'])}; ci = [pc[s] for s in ids]
assert list(zc['labels']) == T
C = {k: zc[k][ci].astype(np.float64) for k in ('finetune_cropaug_f1_radimagenet', 'finetune_cropaug_f2_radimagenet',
                                                 'finetune_crop_f1_radimagenet', 'finetune_cropaug_mixSA_f1')}
C['K1_ens3'] = rp(rp(C['finetune_cropaug_f1_radimagenet']) + rp(C['finetune_cropaug_f2_radimagenet']) + rp(C['finetune_crop_f1_radimagenet']))
print('\nLB-anchor calibration (crop ckpts were gold58-SELECTED, so their gold58 brackets are optimistic):')
for nm, Wg, lb in [('finetune_cropaug_f1_radimagenet', .15, -.001), ('finetune_cropaug_f1_radimagenet', .25, -.002),
                   ('finetune_cropaug_mixSA_f1', .25, -.003), ('K1_ens3', .25, -.001)]:
    X = C[nm]; out = [f'{nm:32s} W {Wg:.2f} gold58 AUC {ma(X):.4f} | LB delta {lb:+.3f} | in-sample {ma(k2g(fin0, X, Wg)) - ma(fin0):+.4f}']
    for k, Tk in Toof.items():
        b = fin(Tk, side0); out.append(f'pOOS[{k}] {ma(k2g(b, X, Wg)) - ma(b):+.4f}')
    print(' | '.join(out))
for Wg in (.15, .25, .30):
    out = [f'{"fleet (for comparison)":32s} W {Wg:.2f} gold58 AUC {ma(F):.4f} | in-sample {ma(k2g(fin0, F, Wg)) - ma(fin0):+.4f}']
    for k, Tk in Toof.items():
        b = fin(Tk, side0); out.append(f'pOOS[{k}] {ma(k2g(b, F, Wg)) - ma(b):+.4f}')
    print(' | '.join(out))
