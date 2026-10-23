# build_frac75.py (k8/frac75, 2026-10-07): labels_frac75_logit.csv = labels_mix50_logit.csv with ONLY the Fracture
# column re-mixed as .25 x steven_v4 + .75 x teacher, the teacher logit-matched onto steven_v4 exactly as mix50 was built
# (SP/build_mixSA.py MAP=logit, logit_match; mix50 = the same with w .5 on every label and teacher_train_probs.csv).
# Step 1 reproduces ALL 12 mix50 columns from (v4, teacher) with that method and asserts max |diff| < 1e-12 (method
# proof). Step 2 writes the new file at the TEXT level: every line of mix50 is copied byte for byte except the last
# field (Fracture), so the other 11 columns, the header, the row order and the CRLF line ends are identical by
# construction (re-verified after writing). CPU, seconds. Read-only on its inputs.
# usage: python build_frac75.py --mix50 M --v4 V --teacher T --out O [--gold G --folds F --ref_frac75 R]
import argparse, hashlib, os, sys
import numpy as np, pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata, spearmanr

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
W_FRAC = 0.75                                    # teacher weight on Fracture (mix50: .5 everywhere)
ap = argparse.ArgumentParser()
ap.add_argument('--mix50', required=True); ap.add_argument('--v4', required=True); ap.add_argument('--teacher', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--gold', default=''); ap.add_argument('--folds', default=''); ap.add_argument('--ref_frac75', default='')
A = ap.parse_args()
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
for k in ('mix50', 'v4', 'teacher'):
    print(f'input {k:8s} md5 {md5(getattr(A, k))}  {getattr(A, k)}')

mix = rd(A.mix50); ids = mix.index
assert list(mix.columns) == T, list(mix.columns)
assert ids.is_unique and len(ids) == 4407 and not mix[T].isna().any().any()
v4 = rd(A.v4).reindex(ids); te = rd(A.teacher).reindex(ids)
assert not v4[T].isna().any().any() and not te[T].isna().any().any(), 'v4/teacher missing mix50 ids'


def logit_match(a, ref):                         # == SP/build_mixSA.py logit_match (MAP=logit), byte for byte in math
    la = logit(np.clip(a, 1e-4, 1 - 1e-4)); lr = logit(np.clip(ref, 1e-4, 1 - 1e-4))
    return expit((la - la.mean()) / la.std() * lr.std() + lr.mean())


# ---- 1. method proof: reproduce every mix50 column at w .5
worst = 0.0
for c in T:
    rep = 0.5 * v4[c].values + 0.5 * logit_match(te[c].values, v4[c].values)
    d = float(np.max(np.abs(rep - mix[c].values))); worst = max(worst, d)
    print(f'reproduce mix50 {c:17s} max|diff| {d:.2e}')
assert worst < 1e-12, f'mix50 NOT reproduced (worst {worst:.3e}): wrong method or inputs'
print(f'METHOD OK: all 12 mix50 columns reproduced, worst max|diff| {worst:.2e}')

# ---- 2. new Fracture column + text-level write
tm = logit_match(te['Fracture'].values, v4['Fracture'].values)
frac = (1 - W_FRAC) * v4['Fracture'].values + W_FRAC * tm
assert np.isfinite(frac).all() and (frac > 0).all() and (frac < 1).all()
raw = open(A.mix50, 'rb').read()
nl = b'\r\n' if b'\r\n' in raw else b'\n'
lines = raw.split(nl)
assert lines[-1] == b'' and len(lines) == len(ids) + 2, (len(lines), len(ids))
hdr, rows = lines[0], lines[1:-1]
assert hdr.decode().split(',') == ['StudyInstanceUID'] + T
pos = {s: i for i, s in enumerate(ids)}
out_rows = []
for k, ln in enumerate(rows):
    f = ln.split(b',')
    assert len(f) == 13 and f[0].decode() == ids[k], k
    out_rows.append(b','.join(f[:12] + [repr(float(frac[k])).encode()]))
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
with open(A.out, 'wb') as fh:
    fh.write(nl.join([hdr] + out_rows + [b'']))

# ---- 3. re-verify the written file
new_raw = open(A.out, 'rb').read(); nlines = new_raw.split(nl)
assert len(nlines) == len(lines) and nlines[0] == hdr and nlines[-1] == b''
same11 = all(a.rsplit(b',', 1)[0] == b.rsplit(b',', 1)[0] for a, b in zip(lines[1:-1], nlines[1:-1]))
assert same11, 'non-Fracture bytes differ'
new = rd(A.out)
assert list(new.columns) == T and (new.index == ids).all() and new.shape == (4407, 12) and not new.isna().any().any()
assert all((new[c].values == mix[c].values).all() for c in T[:-1])
back = np.array([float(ln.rsplit(b',', 1)[1]) for ln in nlines[1:-1]])     # exact text round trip (repr)
assert (back == frac).all(), float(np.max(np.abs(back - frac)))
dpd = float(np.max(np.abs(new['Fracture'].values - frac)))                    # pandas' default C float parser
print(f'Fracture text round trip exact; pandas default parser max|diff| {dpd:.1e} (1 ulp class, same parser as the trainer)')
ends = 'CRLF' if nl == b'\r\n' else 'LF'
print(f'WROTE {A.out}  md5 {md5(A.out)}  shape {new.shape}  NaN 0  line ends {ends}')
print('11 non-Fracture columns + id + header: BYTE-IDENTICAL to mix50 (per line, text before the last comma);'
      f' Fracture parsed back == computed exactly; range [{frac.min():.4g}, {frac.max():.4g}]')
print(f'Fracture: Spearman(mix50, frac75) all rows {spearmanr(mix["Fracture"], new["Fracture"]).correlation:.4f}; '
      f'changed rows {int((new["Fracture"].values != mix["Fracture"].values).sum())}/{len(ids)}')

# ---- 4. cross-check vs an earlier independent build (the perlabel track's ~/novel/labels_mix50_frac75.csv)
if A.ref_frac75 and os.path.exists(A.ref_frac75):
    ref = rd(A.ref_frac75).reindex(ids)
    d = np.abs(ref[T].values - new[T].values).max(0)
    print('vs ref', A.ref_frac75, 'max|diff| per column:', ' '.join(f'{x:.1e}' for x in d))

# ---- 5. target quality on expert gold (masked convention = trainer macro_auc: NaN expert cells dropped) + prevalence
if A.gold:
    from sklearn.metrics import roc_auc_score
    g = rd(A.gold); g58 = [s for s in g.index if g.loc[s, 'source'] == 'original_gold58']; g74 = list(g.index)
    src = {'v4': v4, 'mix50': mix, 'frac75': new, 'teacher': te}
    tlm = mix.copy(); tlm['Fracture'] = tm                    # the logit-matched teacher half actually mixed in
    src_prev = dict(src, teacher_lm=tlm)

    def auc_masked(ids_, c, P):
        y = g.loc[ids_, c].astype(float).values; p = P.loc[ids_, c].astype(float).values
        ok = np.isfinite(y) & np.isfinite(p)
        yb = y[ok] >= .5
        return roc_auc_score(yb, p[ok]) if 0 < yb.mean() < 1 else np.nan, int(ok.sum()), int(yb.sum())
    for nm, gi in (('gold58', g58), ('gold74', g74)):
        parts = []
        for k, P in src.items():
            a, n, npos = auc_masked(gi, 'Fracture', P); parts.append(f'{k} {a:.4f}')
        print(f'Fracture TARGET AUC vs experts, {nm} (n {n}, pos {npos}, masked): ' + ' | '.join(parts))
    for nm, gi in (('gold58', g58), ('gold74', g74)):
        m = {k: np.nanmean([auc_masked(gi, c, P)[0] for c in T]) for k, P in src.items()}
        print(f'12-label macro TARGET AUC, {nm} (masked): ' + ' | '.join(f'{k} {v:.4f}' for k, v in m.items()))
    y58 = g.loc[g58, 'Fracture'].astype(float); y74 = g.loc[g74, 'Fracture'].astype(float)
    print(f'expert Fracture prevalence: gold58 {(y58 >= .5).mean():.3f} ({int((y58 >= .5).sum())}/{len(y58)}) | '
          f'gold74 masked {(y74.dropna() >= .5).mean():.3f} ({int((y74.dropna() >= .5).sum())}/{int(y74.notna().sum())})')
    if A.folds:
        fo = rd(A.folds)['fold']
        tr = [s for s in ids if s in fo.index and fo[s] >= 0 and s not in set(g.index)]
        print(f'training-pool Fracture prevalence (n {len(tr)} = folds5 fold >= 0, gold74 excluded):')
        for k, P in src_prev.items():
            x = P.loc[tr, 'Fracture'].astype(float)
            print(f'  {k:10s} mean target {x.mean():.3f} | share >= .5 {(x >= .5).mean():.3f} | expert gold58 .310')
