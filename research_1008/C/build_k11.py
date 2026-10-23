# build_k11.py (research_1008 dimension C, 2026-10-08). CPU, seconds. Read-only on its inputs.
# K11 label set = frac75 (labels_frac75_logit.csv, md5 ad7582b8...) with ONLY the PF OA column re-mixed as
#   PF OA = .75 x steven_v4 + .25 x logit_match(teacher, steven_v4)
# i.e. the frac75 method (build_frac75.py) with W_PFOA = .25 instead of .5. The logit-matched teacher half is recovered
# exactly as tm = 2 x mix50 - v4 (mix50 = .5 v4 + .5 tm, proven to 1e-16 by build_frac75 step 1), so no teacher file
# is needed. Text-level write: every frac75 line is copied byte for byte except the PF OA field (7th label).
# Also prints a K11c variant (Contusion .25/.75 on top) for reference only; it is NOT written.
import hashlib, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

R = 'C:/Users/eeshs/Downloads/rsna-knee'
DS = R + '/session_2026-10-05/kaggle_train/ds/rsna-knee-fleet-train'
F75 = R + '/session_2026-10-08/k8/frac75/labels_frac75_logit.csv'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels_k11_frac75_pf25_logit.csv')
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
assert md5(F75) == 'ad7582b817243f8f22e8bcfdd7e3f84a', 'frac75 md5 mismatch'
assert md5(DS + '/labels_mix50_logit.csv') == '0ec64d32e3d2538bee66c7d77e54380d'
assert md5(DS + '/llm_labels_v4_blend.csv') == '44874d8ded7699d356b935ca8d7e8df7'
f75 = rd(F75); ids = f75.index
mix = rd(DS + '/labels_mix50_logit.csv').reindex(ids); v4 = rd(DS + '/llm_labels_v4_blend.csv').reindex(ids)
tm = 2 * mix[T] - v4[T]
W_PF = 0.25
pf = (1 - W_PF) * v4['PF OA'].values + W_PF * tm['PF OA'].values
assert np.isfinite(pf).all() and (pf > 0).all() and (pf < 1).all()
raw = open(F75, 'rb').read(); nl = b'\r\n' if b'\r\n' in raw else b'\n'
lines = raw.split(nl); hdr, rows = lines[0], lines[1:-1]
assert lines[-1] == b'' and hdr.decode().split(',') == ['StudyInstanceUID'] + T
J = 1 + T.index('PF OA')
out = []
for k, ln in enumerate(rows):
    f = ln.split(b',')
    assert len(f) == 13 and f[0].decode() == ids[k]
    f[J] = repr(float(pf[k])).encode(); out.append(b','.join(f))
open(OUT, 'wb').write(nl.join([hdr] + out + [b'']))
new = rd(OUT)
assert (new.index == ids).all() and list(new.columns) == T and not new.isna().any().any()
for c in T:
    if c != 'PF OA':
        assert (new[c].values == f75[c].values).all(), c
assert np.abs(new['PF OA'].values - pf).max() < 1e-15
print(f'WROTE {OUT}\n  md5 {md5(OUT)} shape {new.shape} line ends {"CRLF" if nl == b"\r\n" else "LF"}')
print('  11 columns byte-identical to frac75; PF OA = .75 v4 + .25 tm')
gold = rd(DS + '/gold74_eval_set.csv'); g58 = gold.index[gold.source == 'original_gold58']
pool = ids.difference(gold.index)


def auc(y, p):
    ok = np.isfinite(y); yb = y[ok] >= .5; n1 = yb.sum(); n0 = len(yb) - n1; r = rankdata(p[ok])
    return (r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for nm, P in (('mix50', mix), ('frac75', f75), ('K11', new)):
    a = [auc(gold.loc[g58, c].values, P.loc[g58, c].values) for c in T]
    print(f'  {nm:7s} gold58 target macro {np.mean(a):.4f} | PF OA {a[6]:.4f} | Fracture {a[11]:.4f}')
x = new.loc[pool, 'PF OA']; m = mix.loc[pool, 'PF OA']
print(f'  PF OA training pool n {len(pool)}: mean target mix50 {m.mean():.3f} -> K11 {x.mean():.3f}; share >= .5 '
      f'{(m >= .5).mean():.3f} -> {(x >= .5).mean():.3f}; Spearman(K11, mix50) {spearmanr(x, m).correlation:.4f}; '
      f'distinct values mix50 {m.nunique()} K11 {x.nunique()}')
