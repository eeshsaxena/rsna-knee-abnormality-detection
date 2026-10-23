# build_dsd.py (jump2, 2026-10-05): build a full-train pseudo-label TARGET table from our own 5-fold OOF.
# DSD = "decorrelated silent-cell distillation": steven_v4 stays on report-YES cells (optionally + a small own share),
# report-silent (UNK) and report-NO cells become (1 - a) * v4 + a * own_OOF (own_OOF logit mean/std matched onto v4,
# per label, exactly like build_mixSA.py 'logit'). No Raptor / parent teacher anywhere unless --yes_rap > 0.
# Usage (box, CPU, read-only inputs, writes only under ~/jump2):
#   python build_dsd.py --oof '~/fleet_runs/f?_s1043_ttr_mix50/oof_f?_s1043_raw.csv' --a 0.75 --a_yes 0.25 \
#       --out ~/jump2/labels_dsd_ttrmix_a75y25.csv
# --oof may be given several times (several model sets): each set is per-row logit-standardised, then averaged.
# Rows without an own OOF (gold74, never trained on) keep v4 (they are --exclude'd from training anyway).
import os, glob, argparse
import numpy as np, pandas as pd
from scipy.special import logit, expit

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
H = os.path.expanduser
ap = argparse.ArgumentParser()
ap.add_argument('--oof', action='append', required=True, help='glob of per-fold oof csvs (one model set)')
ap.add_argument('--a', type=float, default=0.75, help='own weight on UNK and NO cells')
ap.add_argument('--a_yes', type=float, default=0.0, help='own weight on report-YES cells')
ap.add_argument('--yes_rap', type=float, default=0.0, help='Raptor-proxy weight on report-YES cells (parent import!)')
ap.add_argument('--strata', default='UNK,NO')
ap.add_argument('--out', required=True)
A = ap.parse_args()
assert H(A.out).startswith(H('~/jump2/')), 'write only under ~/jump2'
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')[T]
r2 = rd('~/rsna_train/report_labels_v2.csv')
VER = pd.DataFrame({l: r2[l + '__verdict'] for l in T}).reindex(v4.index).fillna('UNK')
gold = set(rd('~/rsna_train/gold74_eval_set.csv').index)
lg = lambda a: logit(np.clip(np.asarray(a, float), 1e-4, 1 - 1e-4))

Z = []
for pat in A.oof:
    fs = sorted(glob.glob(H(pat)))
    parts = [rd(f)[T] for f in fs]
    o = pd.concat(parts)
    assert not o.index.duplicated().any(), f'{pat}: a study appears in two OOF files'
    print(f'{pat}: {len(fs)} files, {len(o)} rows')
    z = lg(o.values); z = (z - z.mean(0)) / z.std(0)
    Z.append(pd.DataFrame(z, index=o.index, columns=T))
common = sorted(set.intersection(*[set(z.index) for z in Z]))
z = np.mean([zz.loc[common].values for zz in Z], 0); z = (z - z.mean(0)) / z.std(0)
lv = lg(v4.loc[common].values)
own = pd.DataFrame(expit(z * lv.std(0) + lv.mean(0)), index=common, columns=T)
assert not set(common) & gold, 'gold rows in the OOF?'
n_train = len([s for s in v4.index if s not in gold])
print(f'own OOF rows {len(common)} of {n_train} non-gold studies')

res = v4.copy()
st = A.strata.split(',')
V = VER.loc[common].values; b = v4.loc[common].values; S = own.values
w = np.where(np.isin(V, st), A.a, A.a_yes)
t = (1 - w) * b + w * S
if A.yes_rap > 0:
    rap = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')[T]
    la = lg(rap.loc[v4.index].values); lv_all = lg(v4.values)
    R = expit((lg(rap.loc[common].values) - la.mean(0)) / la.std(0) * lv_all.std(0) + lv_all.mean(0))
    my = V == 'YES'
    t[my] = (1 - A.yes_rap) * t[my] + A.yes_rap * R[my]
res.loc[common, T] = t
res.reset_index().to_csv(H(A.out), index=False)
chg = np.abs(res.loc[common].values - b)
print(f'wrote {A.out} | cells changed > .05: {(chg > .05).mean():.3f} | mean |delta| {chg.mean():.4f} | '
      f'UNK share {(V == "UNK").mean():.3f} NO {(V == "NO").mean():.3f} YES {(V == "YES").mean():.3f}')
