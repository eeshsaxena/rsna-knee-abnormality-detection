"""proxy_gold58.py : step 1 of the LB proxy model (box, CPU only, read-only inputs).

gold58 = the 58 gold-labelled train studies the parent's Raptor/CoAt arms never trained on (OOS), while its DINOv2 /
RadImageNet arms are IN-SAMPLE there. For every OOS arm, the fleet member, the crop members (laneA npz, gold-selected
checkpoints) and the Dread widedense teacher:
  * macro AUC (+ per label), bootstrap SE
  * Spearman rho to each other (mean over labels)
  * binormal-implied LATENT (within-class) correlation r, two estimators:
      rA = Monte-Carlo inversion of the mixture Spearman given the per-label AUCs and prevalence
      rB = direct within-class Spearman (pooled Fisher-z over negatives/positives), -> Pearson via 2 sin(pi rs/6)
  * validation of the equal-variance binormal two-reader blend model against MEASURED gold58 rank blends:
      the 4 context curves (fleet into raptor arm v5 / raptor_public / coat_family / raptor_coat_side) at a W grid,
      plus every ordered pair of a validation set at W .25/.50.
Writes ~/k2_test/analysis_wf/proxy_gold58_out.json and prints tables.
"""
import os, json, time, numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr
from sklearn.metrics import roc_auc_score

H = os.path.expanduser
OUT = H('~/k2_test/analysis_wf/proxy_gold58_out.json')
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
tr = pd.read_csv(H('~/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'),
                 dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R = H('~/kmimic/runs/gold58_gpu')
P = R + '/localLB_parts/'
F = pd.read_csv(H('~/k2_test/runs/C_gold58/working/submission_fleetsolo.csv'),
                dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [s for s in F.index if s in tr.index and not np.isnan(tr.loc[s, 'ACL'])]
assert len(ids) == 58, len(ids)
Y = tr.loc[ids, T].values.astype(float)
assert not np.isnan(Y).any()
N = len(ids)
prev = Y.mean(0)


def lc(f):
    d = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    return d.loc[ids, T].values.astype(float)


def rp(a): return np.column_stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])])


A, kind = {}, {}
for nm in ['raptor_arm_maxspan-v5_prob', 'raptor_arm_maxspan-v5-reverse_prob', 'raptor_arm_native384-v8_prob',
           'raptor_arm_native384dense-v10_prob', 'raptor_public_rank', 'raptor_coat_side_rank', 'coat_family_rank',
           'coat_d4_prob', 'coat_global96_prob', 'coat_repairv1_prob', 'coat_resgated_prob', 'a5_foldmean_prob']:
    A[nm] = lc(P + nm + '.csv'); kind[nm] = 'OOS parent arm'
for nm in ['pipeline_stage_1_dino_stack', 'diag_rad_e10_raw_foldmean', 'diag_rad_e13_raw_foldmean']:
    A[nm] = lc(P + nm + '.csv'); kind[nm] = 'IN-SAMPLE parent arm'
A['parent_final'] = lc(R + '/localLB_parent_preds.csv'); kind['parent_final'] = 'parent (partly in-sample)'
A['fleet_f0_mix50'] = F.loc[ids, T].values.astype(float); kind['fleet_f0_mix50'] = 'own fleet member (clean)'
z = np.load(H('~/k2_test/analysis_wf/proxy_crop_gold74_preds.npz'), allow_pickle=True)
zid = {str(s): i for i, s in enumerate(z['ids'])}
assert [str(x) for x in z['labels']] == T
zi = [zid[s] for s in ids]
for k, nm in [('finetune_cropaug_f1_radimagenet', 'crop_cropaug_f1'), ('finetune_cropaug_f2_radimagenet', 'crop_cropaug_f2'),
              ('finetune_crop_f1_radimagenet', 'crop_crop_f1'), ('finetune_cropaug_mixSA_f1', 'crop_mixSA_f1')]:
    A[nm] = z[k][zi].astype(float); kind[nm] = 'own crop member (gold58-selected ckpt)'
A['crop_K1_ens3'] = rp(rp(A['crop_cropaug_f1']) + rp(A['crop_cropaug_f2']) + rp(A['crop_crop_f1']))
kind['crop_K1_ens3'] = 'own crop 3-member rank mean (K1 analog)'
te = pd.read_csv(H('~/rsna_train/teacher_train_probs.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
A['dread_widedense'] = te.loc[ids, T].values.astype(float); kind['dread_widedense'] = 'Dread widedense single (OOS, LB .924)'
names = list(A)


def per(X, Yv=Y):
    return np.array([roc_auc_score(Yv[:, j], X[:, j]) for j in range(12)])


def fastauc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


PL = {n: per(A[n]) for n in names}
MAC = {n: float(PL[n].mean()) for n in names}
rng = np.random.default_rng(0)
BOOT = [rng.integers(0, N, N) for _ in range(600)]


def bmean(b, X, base=None):
    v = []
    for j in range(12):
        y = Y[b, j] > .5
        if 0 < y.sum() < len(y):
            v.append(fastauc(y, X[b, j]) - (fastauc(y, base[b, j]) if base is not None else 0))
    return np.mean(v)


def boot_macro(X):
    return np.array([bmean(b, X) for b in BOOT])


SE = {n: float(boot_macro(A[n]).std()) for n in ['fleet_f0_mix50', 'raptor_coat_side_rank', 'raptor_arm_maxspan-v5_prob', 'dread_widedense']}


def sp_lab(a, b): return np.array([spearmanr(a[:, j], b[:, j]).correlation for j in range(12)])


# ---------------- latent r estimators ----------------
NMC = 12000
mrng = np.random.default_rng(12345)
Z1, Z2, U = mrng.standard_normal(NMC), mrng.standard_normal(NMC), mrng.random(NMC)
RG = np.linspace(-0.5, 0.995, 46)


def dval(a): return np.sqrt(2) * norm.ppf(np.clip(a, 0.02, 0.998))


def rank01(x): return x.argsort().argsort().astype(float) / len(x)


_cache = {}


def spear_curve(d1, d2, p):
    key = (round(d1, 3), round(d2, 3), round(p, 4))
    if key in _cache: return _cache[key]
    y = U < p
    r1 = rank01(d1 * y + Z1)
    out = np.array([np.corrcoef(r1, rank01(d2 * y + r * Z1 + np.sqrt(1 - r * r) * Z2))[0, 1] for r in RG])
    _cache[key] = out
    return out


def rA_lab(a, b, pa, pb):
    """per-label latent r from the observed mixture Spearman (MC inversion)."""
    s = sp_lab(a, b); out = []
    for j in range(12):
        c = spear_curve(dval(pa[j]), dval(pb[j]), prev[j])
        c = np.maximum.accumulate(c)  # monotone
        out.append(float(np.interp(s[j], c, RG)))
    return np.array(out)


def rB_lab(a, b):
    out = []
    for j in range(12):
        zs, ws = [], []
        for cls in (0, 1):
            m = Y[:, j] == cls
            if m.sum() >= 5:
                rs = spearmanr(a[m, j], b[m, j]).correlation
                if np.isfinite(rs):
                    zs.append(np.arctanh(np.clip(rs, -.999, .999))); ws.append(m.sum() - 3)
        rs = np.tanh(np.average(zs, weights=ws)) if zs else np.nan
        out.append(2 * np.sin(np.pi * rs / 6))
    return np.array(out)


key_set = ['raptor_arm_maxspan-v5_prob', 'raptor_arm_maxspan-v5-reverse_prob', 'raptor_arm_native384-v8_prob',
           'raptor_arm_native384dense-v10_prob', 'raptor_public_rank', 'raptor_coat_side_rank', 'coat_family_rank',
           'coat_d4_prob', 'coat_global96_prob', 'coat_repairv1_prob', 'coat_resgated_prob', 'a5_foldmean_prob',
           'fleet_f0_mix50', 'crop_cropaug_f1', 'crop_cropaug_f2', 'crop_crop_f1', 'crop_mixSA_f1', 'crop_K1_ens3',
           'dread_widedense', 'parent_final', 'pipeline_stage_1_dino_stack', 'diag_rad_e13_raw_foldmean']
t0 = time.time()
SP, RA, RB, RAl, RBl = {}, {}, {}, {}, {}
for i, a in enumerate(key_set):
    for b in key_set[i + 1:]:
        s = sp_lab(A[a], A[b]); ra = rA_lab(A[a], A[b], PL[a], PL[b]); rb = rB_lab(A[a], A[b])
        for k in ((a, b), (b, a)):
            SP[k] = float(np.nanmean(s)); RA[k] = float(np.nanmean(ra)); RB[k] = float(np.nanmean(rb))
            RAl[k] = ra; RBl[k] = rb
print(f'latent r done in {time.time() - t0:.0f}s')


# ---------------- blend model ----------------
def model_pl(aB, aM, r, w):
    """per-label analytic equal-variance binormal, latent-score blend."""
    dB, dM = dval(aB), dval(aM)
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * np.clip(r, -0.99, 0.999)
    return norm.cdf(((1 - w) * dB + w * dM) / np.sqrt(2 * v))


def model_mc(aB, aM, r, w):
    """per-label Monte-Carlo, exact RANK blend under the binormal model."""
    out = []
    for j in range(12):
        p = prev[j]; y = U < p; rr = float(np.clip(r[j], -0.99, 0.995))
        x1 = dval(aB[j]) * y + Z1; x2 = dval(aM[j]) * y + rr * Z1 + np.sqrt(1 - rr * rr) * Z2
        s = (1 - w) * rank01(x1) + w * rank01(x2)
        out.append(fastauc(y, s) - fastauc(y, x1))
    return np.array(out)


def measured(b, m, w):
    X = rp((1 - w) * rp(A[b]) + w * rp(A[m]))
    return float(per(X).mean() - MAC[b]), X


def boot_delta(b, X):
    d = [bmean(i, X, A[b]) for i in BOOT[:400]]
    return float(np.std(d))


def predict_all(b, m, w, rmode='A'):
    rl = RAl[(b, m)] if rmode == 'A' else RBl[(b, m)]
    rl = np.where(np.isfinite(rl), rl, np.nanmean(rl))
    pl = float((model_pl(PL[b], PL[m], rl, w) - PL[b]).mean())
    # macro-level single binormal
    rbar = float(np.nanmean(rl))
    mac = float(model_pl(np.array([MAC[b]]), np.array([MAC[m]]), np.array([rbar]), w)[0] - MAC[b])
    return dict(pl=pl, mac=mac)


def fit_r(b, m, ws, meas):
    """macro-level r that best fits the measured curve (least squares)."""
    grid = np.linspace(0.0, 0.995, 400)
    sse = [sum((model_pl(np.array([MAC[b]]), np.array([MAC[m]]), np.array([g]), w)[0] - MAC[b] - d) ** 2
               for w, d in zip(ws, meas)) for g in grid]
    return float(grid[int(np.argmin(sse))])


res = dict(ids_n=N, prev=dict(zip(T, prev.round(4).tolist())), arms={}, pairs={}, curves={}, validation=[])
print('\n=== gold58 arms (macro AUC) ===')
for n in names:
    res['arms'][n] = dict(kind=kind[n], macro=MAC[n], per_label=dict(zip(T, PL[n].round(4).tolist())))
    print(f'{n:40s} {MAC[n]:.4f}  {kind[n]}')
print('prevalence:', dict(zip(T, (prev * N).astype(int).tolist())))
print('bootstrap SE of macro AUC:', {k: round(v, 4) for k, v in SE.items()})
res['boot_se_macro'] = SE

print('\n=== pairwise: Spearman (mean over labels) | latent rA (MC inversion) | latent rB (within-class) ===')
focus = ['fleet_f0_mix50', 'crop_K1_ens3', 'crop_cropaug_f1', 'crop_mixSA_f1', 'dread_widedense']
for f in focus:
    print(f'-- {f}')
    for o in key_set:
        if o == f: continue
        k = (f, o)
        print(f'   vs {o:38s} sp {SP[k]:.3f}  rA {RA[k]:.3f}  rB {RB[k]:.3f}')
for k in SP:
    if k[0] < k[1]:
        res['pairs'][f'{k[0]}|{k[1]}'] = dict(spearman=SP[k], rA=RA[k], rB=RB[k],
                                              rA_lab=np.round(RAl[k], 3).tolist(), rB_lab=np.round(RBl[k], 3).tolist())

print('\n=== the 4 context curves: fleet into OOS bases (delta macro AUC on gold58) ===')
WS = [0.10, 0.15, 0.25, 0.35, 0.50]
summary = []
for b in ['raptor_arm_maxspan-v5_prob', 'raptor_public_rank', 'coat_family_rank', 'raptor_coat_side_rank']:
    m = 'fleet_f0_mix50'
    meas, se, pa, pb, mc, mac = [], [], [], [], [], []
    for w in WS:
        d, X = measured(b, m, w); meas.append(d); se.append(boot_delta(b, X))
        q = predict_all(b, m, w, 'A'); pa.append(q['pl']); mac.append(q['mac'])
        pb.append(predict_all(b, m, w, 'B')['pl'])
        mc.append(float(model_mc(PL[b], PL[m], np.where(np.isfinite(RAl[(b, m)]), RAl[(b, m)], RA[(b, m)]), w).mean()))
    rfit = fit_r(b, m, WS, meas)
    res['curves'][b] = dict(W=WS, measured=meas, boot_se=se, pred_perlabel_rA=pa, pred_perlabel_rB=pb, pred_mc_rank_rA=mc,
                            pred_macro_rA=mac, rA=RA[(b, m)], rB=RB[(b, m)], r_fit_macro=rfit, base=MAC[b])
    print(f'{b} (base {MAC[b]:.4f}; rA {RA[(b, m)]:.3f} rB {RB[(b, m)]:.3f} r_fit(macro) {rfit:.3f})')
    for i, w in enumerate(WS):
        print(f'   W{w:.2f} meas {meas[i]:+.4f} (se {se[i]:.4f}) | pl-rA {pa[i]:+.4f} pl-rB {pb[i]:+.4f} mcRank-rA {mc[i]:+.4f} macro-rA {mac[i]:+.4f}')

print('\n=== broad validation: ordered pairs at W .25/.50 ===')
vset = ['raptor_arm_maxspan-v5_prob', 'raptor_arm_maxspan-v5-reverse_prob', 'raptor_arm_native384-v8_prob',
        'raptor_arm_native384dense-v10_prob', 'coat_d4_prob', 'coat_global96_prob', 'coat_repairv1_prob',
        'coat_resgated_prob', 'a5_foldmean_prob', 'raptor_public_rank', 'coat_family_rank', 'raptor_coat_side_rank',
        'fleet_f0_mix50', 'crop_K1_ens3', 'dread_widedense']
rows = []
for b in vset:
    for m in vset:
        if b == m: continue
        for w in (0.25, 0.50):
            d, _ = measured(b, m, w)
            qa = predict_all(b, m, w, 'A'); qb = predict_all(b, m, w, 'B')
            rows.append(dict(base=b, member=m, W=w, measured=d, pl_rA=qa['pl'], pl_rB=qb['pl'], macro_rA=qa['mac'], macro_rB=qb['mac']))
V = pd.DataFrame(rows)
res['validation'] = rows
for col in ['pl_rA', 'pl_rB', 'macro_rA', 'macro_rB']:
    e = V[col] - V.measured
    print(f'{col:9s} n={len(V)} MAE {e.abs().mean():.4f} bias {e.mean():+.4f} corr {np.corrcoef(V[col], V.measured)[0, 1]:.3f}'
          f' | sign agreement {(np.sign(V[col]) == np.sign(V.measured)).mean():.2f}')
    res[f'val_{col}'] = dict(mae=float(e.abs().mean()), bias=float(e.mean()), corr=float(np.corrcoef(V[col], V.measured)[0, 1]))
# typical measured-delta noise on gold58 (paired bootstrap), for scale
sub = V.sample(30, random_state=0)
nse = [boot_delta(r.base, rp((1 - r.W) * rp(A[r.base]) + r.W * rp(A[r.member]))) for r in sub.itertuples()]
print(f'median paired-bootstrap SE of a measured gold58 blend delta: {np.median(nse):.4f}')
res['median_boot_se_delta'] = float(np.median(nse))
json.dump(res, open(OUT, 'w'), indent=1)
print('wrote', OUT)
