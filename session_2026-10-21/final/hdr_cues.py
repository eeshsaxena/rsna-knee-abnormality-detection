# hdr_cues.py (10-21 FINAL critic track). CPU only, local RSNA competition files only (no OAI, no box, no network).
# Gap tested: DICOM protocol / sequence-content cues (SeriesDescription tokens, 3D / GRE / STIR, series count, field strength)
# as per-study label evidence BEYOND the image model. Prior checks covered only PatientSex, PatientAge, a contrast keyword and
# language-level site priors (THINK_1019 / TRICK_1019 critic).
# Inputs: session_2026-10-19/trick/box/dicom_series_tags.csv (458 train studies, header dump), .../box/lang.csv (site proxy),
# session_2026-10-18/next/g58/R7OOF.csv (image-model OOF stand-in for c4), report_labels_v2 (report regime), _data/train.csv
# (gold58 expert labels). Output: aggregates only, to stdout.
import re, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

rng = np.random.default_rng(20261021)
d = pd.read_csv('session_2026-10-19/trick/box/dicom_series_tags.csv', dtype=str)
lang = pd.read_csv('session_2026-10-19/trick/box/lang.csv', dtype=str).set_index('StudyInstanceUID')['lang']
tr = pd.read_csv('_data/train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
L = [c for c in tr.columns if c != 'Report']
gold = tr.dropna(subset=L)
r7 = pd.read_csv('session_2026-10-18/next/g58/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v2 = pd.read_csv('_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')

def toks(s):
    return set(t for t in re.split(r'[^A-Za-z0-9*]+', str(s).upper()) if t)

rows = {}
for sid, g in d.groupby('StudyInstanceUID'):
    T = set().union(*[toks(s) for s in g.SeriesDescription.fillna('')])
    acq = set(g.MRAcquisitionType.fillna('')); seq = '|'.join(g.ScanningSequence.fillna(''))
    fs = pd.to_numeric(g.MagneticFieldStrength, errors='coerce').max()
    desc = g.SeriesDescription.fillna('').str.upper().str.strip()
    rows[sid] = dict(
        acl_obl=float(bool(T & {'ACL', 'OBL', 'LCA', 'PCL', 'ACLT2W'})),
        patella=float('PATELLA' in T),
        three_d=float(('3D' in acq) or bool(T & {'DE3D', 'SPC', 'ISO', 'DESS', '3D', 'CUBE', 'VISTA'})),
        gre=float('GR' in seq),
        stir=float(('IR' in seq) or bool(T & {'STIR', 'TIRM', 'VSTIR'})),
        t3=float(fs >= 2.5) if np.isfinite(fs) else 0.0,
        nser=float(np.log(g.SeriesInstanceUID.nunique())),
        nfiles=float(np.log(pd.to_numeric(g.n_files, errors='coerce').sum())),
        repeat=float(bool(T & {'REPEAT', 'REPEATED'}) or (desc[desc != 'DUMMYSERIESDESC'].duplicated().any())),
    )
F = pd.DataFrame(rows).T
FEAT = list(F.columns)
print('studies with headers', len(F), '| feature prevalence', {c: round(F[c].mean(), 3) if c not in ('nser', 'nfiles') else None for c in FEAT})

def logit_fit(X, y, ridge=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); gr = X.T @ (y - p) - ridge * b
        b += np.linalg.solve(H, gr)
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 1e-12, None))
    return b, se

def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)

def pct(x):
    return rankdata(x) / len(x)

def zstd(x):
    s = x.std(); return (x - x.mean()) / (s if s > 0 else 1)

ng = F.index.difference(gold.index).intersection(r7.index).intersection(v2.index)
ng = ng[lang.reindex(ng).notna().values]
gg = F.index.intersection(gold.index).intersection(r7.index)
print(f'non-gold report-regime n {len(ng)} | gold58 expert n {len(gg)}')
langs = lang.reindex(ng)
top = langs.value_counts(); keep = top[top >= 15].index
LFE = pd.get_dummies(langs.where(langs.isin(keep), 'other')).astype(float).values[:, 1:]

# A. single-feature within-site z (lang FE + R7 z), with a within-language permutation null for max |z|
Z = np.zeros((len(L), len(FEAT))); usable = []
Xf = np.c_[[zstd(F.loc[ng, c].values.astype(float)) for c in FEAT]].T
def zmat(Xf_):
    out = np.full((len(L), len(FEAT)), np.nan)
    for i, c in enumerate(L):
        y = (v2.loc[ng, c].astype(float).values >= .5).astype(float)
        if y.sum() < 8 or y.sum() > len(y) - 8: continue
        z = r7.loc[ng, 'z|' + c].astype(float).values
        for j in range(len(FEAT)):
            X = np.c_[np.ones(len(y)), z, LFE, Xf_[:, j]]
            b, se = logit_fit(X, y); out[i, j] = b[-1] / se[-1]
    return out
Z = zmat(Xf)
obs_max = np.nanmax(np.abs(Z)); obs_chi = np.nansum(Z ** 2); ncell = np.isfinite(Z).sum()
print('\nA. within-site feature z (report v2 labels; logistic y ~ R7 z + language FE + feature), rows = labels')
print('   ' + ' '.join(f'{c[:7]:>7s}' for c in FEAT))
for i, c in enumerate(L):
    print(f'{c[:16]:16s} ' + ' '.join(f'{v:+7.2f}' if np.isfinite(v) else '      .' for v in Z[i]))
NPERM = 200
grp = langs.where(langs.isin(keep), 'other').values
mx = []; chi = []
for k in range(NPERM):
    perm = np.arange(len(ng))
    for gv in np.unique(grp):
        ix = np.where(grp == gv)[0]; perm[ix] = rng.permutation(ix)
    Zp = zmat(Xf[perm]); mx.append(np.nanmax(np.abs(Zp))); chi.append(np.nansum(Zp ** 2))
mx = np.array(mx); chi = np.array(chi)
print(f'   cells {ncell}; observed max|z| {obs_max:.2f} (perm P {np.mean(mx >= obs_max):.3f}, null p50 {np.median(mx):.2f} p95 {np.quantile(mx, .95):.2f});'
      f' sum z^2 {obs_chi:.1f} (perm P {np.mean(chi >= obs_chi):.3f}, null mean {chi.mean():.1f})')

# B. shippable form: header score fitted WITHOUT language (test has no report), 5-fold CV on non-gold, rank-mixed into R7 pct
print('\nB. shippable nudge final = pct(pct(R7) + b * pct(header score)), header score = 5-fold CV logistic on header features (no language)')
Xh = np.c_[np.ones(len(ng)), Xf]
res = {b: [] for b in (0.05, 0.1, 0.2)}; base_auc = []; lab_used = []
fold = rng.permutation(len(ng)) % 5
gold_res = {b: [] for b in (0.05, 0.1, 0.2)}
Xg = np.c_[np.ones(len(gg)), np.c_[[(F.loc[gg, c].values.astype(float) - F.loc[ng, c].values.astype(float).mean()) /
                                     (F.loc[ng, c].values.astype(float).std() or 1) for c in FEAT]].T]
boot = {b: [] for b in (0.05, 0.1, 0.2)}
for c in L:
    y = (v2.loc[ng, c].astype(float).values >= .5).astype(float)
    if y.sum() < 8 or y.sum() > len(y) - 8: continue
    lab_used.append(c)
    z = r7.loc[ng, 'z|' + c].astype(float).values
    h = np.zeros(len(y))
    for f in range(5):
        trn, tst = fold != f, fold == f
        b_, _ = logit_fit(Xh[trn], y[trn], ridge=1.0); h[tst] = Xh[tst] @ b_
    a0 = auc(z, y); base_auc.append(a0)
    for bb in res:
        res[bb].append(auc(pct(pct(z) + bb * pct(h)), y) - a0)
    # gold58 expert: header model fitted on ALL non-gold rows, applied to gold58
    b_all, _ = logit_fit(Xh, y, ridge=1.0)
    yg = (gold.loc[gg, c].astype(float).values >= .5).astype(float)
    if 2 <= yg.sum() <= len(yg) - 2:
        zg = r7.loc[gg, 'z|' + c].astype(float).values; hg = Xg @ b_all
        for bb in gold_res:
            gold_res[bb].append(auc(pct(pct(zg) + bb * pct(hg)), yg) - auc(zg, yg))
for bb in res:
    v = np.array(res[bb]); g5 = np.array(gold_res[bb])
    print(f'   b {bb:.2f}: report-OOF CV per-label dAUC mean {v.mean():+.5f} (labels > 0: {(v > 0).sum()}/{len(v)}; min {v.min():+.4f} max {v.max():+.4f})'
          f' -> macro-12 {v.sum() / 12:+.5f} | gold58 expert macro-12 {g5.sum() / 12:+.5f} ({(g5 > 0).sum()}/{len(g5)} labels > 0)')
print('   labels used', len(lab_used), '| R7 base report-OOF AUC mean', round(float(np.mean(base_auc)), 4))

# C. placebo: same pipeline with header features shuffled across studies (breaks study link, keeps marginals)
pl = []
for k in range(100):
    perm = rng.permutation(len(ng)); Xhp = Xh[perm]; v = []
    for c in lab_used:
        y = (v2.loc[ng, c].astype(float).values >= .5).astype(float); z = r7.loc[ng, 'z|' + c].astype(float).values
        h = np.zeros(len(y))
        for f in range(5):
            trn, tst = fold != f, fold == f
            b_, _ = logit_fit(Xhp[trn], y[trn], ridge=1.0); h[tst] = Xhp[tst] @ b_
        v.append(auc(pct(pct(z) + 0.1 * pct(h)), y) - auc(z, y))
    pl.append(np.sum(v) / 12)
pl = np.array(pl); obs = np.array(res[0.1]).sum() / 12
print(f'\nC. placebo (shuffled header rows), b .1 macro-12: mean {pl.mean():+.5f} sd {pl.std():.5f}; observed {obs:+.5f}; P(placebo >= observed) {np.mean(pl >= obs):.3f}')
