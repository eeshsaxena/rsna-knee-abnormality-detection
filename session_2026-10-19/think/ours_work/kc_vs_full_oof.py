# kc_vs_full_oof.py: same 867 fold-0 studies, report labels: is the knee-centred KC crop structurally stronger than the full-frame
# fleet / student models on soft-tissue labels (ACL, MCL, menisci)? Read-only on existing OOF files.
import sys, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
R = sys.argv[1]; BP = sys.argv[2]; G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L]
kc = (rd(BP + '/fleet_runs/f0_s2001_kc256_r7tb/oof_f0_s2001_raw.csv') + rd(BP + '/fleet_runs/f0_s2011_kc256_ev2s_r7tb/oof_f0_s2011_raw.csv')) / 2
M = {'KC_f0': kc, 'K14DT_f0': rd(glob.glob(BP + '/fleet_runs/f0_s1043_ttr_frac75oaiT123v00/oof_f0_*_raw.csv')[0]),
     'fleetD_f0': rd(glob.glob(BP + '/fleet_runs/f0_s1044_ttr_frac75/oof_f0_*_raw.csv')[0]),
     'Tb_f0': rd(glob.glob(BP + '/fleet_runs/f0_s1047_ttr_r7tb/oof_f0_*_raw.csv')[0])}
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ids = [u for u in kc.index if all(u in m.index for m in M.values()) and u in r7.index and u in lab.index]
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
pct = lambda v: rankdata(v) / len(v)
print('n', len(ids))
print('%-17s %7s %9s %9s %9s %9s | KC-Tb  | .2 KC on (R7+Tb) vs .2 Tb-only extra' % ('label', 'R7', 'KC', 'K14DT', 'fleetD', 'Tb'))
rng = np.random.default_rng(3)
for c in L:
    v = lab.loc[ids, c + '__verdict'].values; ok = np.isin(v, ['YES', 'NO']); y = (v[ok] == 'YES').astype(int); ii = np.array(ids)[ok]
    a = {k: m.loc[ii, c].values for k, m in M.items()}; a['R7'] = r7.loc[ii, 'p|' + c].values
    au = {k: auc(x, y) for k, x in a.items()}
    # bootstrap sd of KC - Tb
    d = []
    for _ in range(300):
        k = rng.integers(0, len(y), len(y))
        if 0 < y[k].sum() < len(k): d.append(auc(a['KC_f0'][k], y[k]) - auc(a['Tb_f0'][k], y[k]))
    # does KC add on top of a 2-model base R7+Tb (less inflated than R7 alone)?
    base = .5 * pct(a['R7']) + .5 * pct(a['Tb_f0'])
    gkc = auc(.8 * pct(base) + .2 * pct(a['KC_f0']), y) - auc(base, y)
    gdt = auc(.8 * pct(base) + .2 * pct(a['K14DT_f0']), y) - auc(base, y)
    print('%-17s %.4f   %.4f    %.4f    %.4f    %.4f | %+.4f (%.4f) | KC %+.4f  K14DT %+.4f' % (c, au['R7'], au['KC_f0'], au['K14DT_f0'], au['fleetD_f0'], au['Tb_f0'], au['KC_f0'] - au['Tb_f0'], np.std(d), gkc, gdt))
