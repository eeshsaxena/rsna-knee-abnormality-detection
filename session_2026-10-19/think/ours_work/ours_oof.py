# ours_oof.py (10-19 think, lens "our unused assets"; read-only numpy on existing OOF files, CPU only).
# Large-n check of the "structurally strong labels" story: on non-gold train studies, every model's prediction is out of fold
# (R7: heliosli's 4-fold split; ours: folds5). Labels = LLM report verdicts (YES / NO only), report_labels_v2.csv.
# Per member and label: AUC(member), AUC(R7), gain of .8 pct(R7) + .2 pct(member) and .9 / .1 over R7 alone (single-model base:
# inflated vs a c4 base, see wrongbase.py), bootstrap sd. Then: do the per-label gains on OOF agree with the per-label gains on
# gold58 for the same member on the same R7 base (rank correlation over 12 labels)?
import sys, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
R = sys.argv[1]; BP = sys.argv[2]; G = R + '/session_2026-10-18/next/g58'; OUT = sys.argv[3]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
gold = set(pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID)
lab = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')


def cat(pat):
    fs = sorted(glob.glob(BP + pat))
    d = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[L] for f in fs])
    return d[~d.index.duplicated()]


MEM = {
    'KC_f0 (s2001+s2011)': (cat('/fleet_runs/f0_s2001_kc256_r7tb/oof_f0_s2001_raw.csv') + cat('/fleet_runs/f0_s2011_kc256_ev2s_r7tb/oof_f0_s2011_raw.csv')) / 2,
    'K14DT': cat('/fleet_runs/f*_s1043_ttr_frac75oaiT123v00/oof_f*_raw.csv'),
    'fleetD': cat('/fleet_runs/f*_s1044_ttr_frac75/oof_f*_raw.csv'),
    'Tb': cat('/fleet_runs/f*_s1047_ttr_r7tb/oof_f*_raw.csv'),
    'Ta': cat('/fleet_runs/f*_s1048_ttr_r7ta/oof_f*_raw.csv'),
}
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


rng = np.random.default_rng(7)
res = {}
for name, m in MEM.items():
    idx = [u for u in m.index if u in r7.index and u not in gold and u in lab.index]
    res[name] = {'n': len(idx), 'labels': {}}
    print('== %s  n %d' % (name, len(idx)))
    print('  %-17s %6s %6s %6s | %8s %8s (sd) | %8s' % ('label', 'npos', 'AUC_m', 'AUC_R7', 'g.2', '', 'g.1'))
    for c in L:
        v = lab.loc[idx, c + '__verdict'].values; ok = np.isin(v, ['YES', 'NO'])
        y = (v[ok] == 'YES').astype(int); ii = np.array(idx)[ok]
        a = r7.loc[ii, 'p|' + c].values; b = m.loc[ii, c].values
        am, ar = auc(b, y), auc(a, y)
        g2 = auc(.8 * pct(a) + .2 * pct(b), y) - ar; g1 = auc(.9 * pct(a) + .1 * pct(b), y) - ar
        bs = []
        for _ in range(200):
            k = rng.integers(0, len(y), len(y))
            if 0 < y[k].sum() < len(k): bs.append(auc(.8 * pct(a[k]) + .2 * pct(b[k]), y[k]) - auc(a[k], y[k]))
        res[name]['labels'][c] = dict(npos=int(y.sum()), n=int(len(y)), auc_m=am, auc_r7=ar, gain20=g2, gain10=g1, sd20=float(np.std(bs)))
        print('  %-17s %6d %.4f %.4f | %+.4f (%.4f) | %+.4f' % (c, y.sum(), am, ar, g2, np.std(bs), g1))
    mm = np.mean([x['auc_m'] for x in res[name]['labels'].values()]); mr = np.mean([x['auc_r7'] for x in res[name]['labels'].values()])
    g2m = np.mean([x['gain20'] for x in res[name]['labels'].values()])
    res[name]['macro'] = dict(auc_m=mm, auc_r7=mr, gain20=g2m)
    print('  macro: member %.4f R7 %.4f gain .2 %+.4f' % (mm, mr, g2m))

# gold58 per-label gains of the same members on the same single-model base (R7 OOF), for the agreement check
g = pd.read_csv(G + '/gold74_eval_set.csv'); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int)
R7g = r7.loc[ids, ['p|' + c for c in L]].values
rd = lambda f: pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
gm = lambda pat: np.mean([rd(f) for f in sorted(glob.glob(BP + pat))], axis=0)
GM = {'KC_f0 (s2001+s2011)': (rd(BP + '/fleet_runs/f0_s2001_kc256_r7tb/epochs/gold_f0_s2001_raw_ep12.csv') +
                             rd(BP + '/fleet_runs/f0_s2011_kc256_ev2s_r7tb/epochs/gold_f0_s2011_raw_ep12.csv')) / 2,
      'K14DT': gm('/k14_test/gold/preds/f*_s1043_ttr_frac75oaiT123v00__raw__A.csv'),
      'fleetD': gm('/r7t_labels/gold_preds/f*_s1044_ttr_frac75__raw__A.csv'),
      'Tb': gm('/r7t_labels/gold_preds/f*_s1047_ttr_r7tb__raw__A.csv')}
res['agreement_oof_vs_gold58'] = {}
for name, P in GM.items():
    gg = [auc(.8 * pct(R7g[:, j]) + .2 * pct(P[:, j]), Y[:, j]) - auc(R7g[:, j], Y[:, j]) for j in range(12)]
    go = [res[name]['labels'][c]['gain20'] for c in L]
    rho = spearmanr(gg, go)[0]
    res['agreement_oof_vs_gold58'][name] = dict(rho=float(rho), gold58_gain20=dict(zip(L, np.round(gg, 4).tolist())), gold58_macro=float(np.mean(gg)))
    print('agreement %-20s Spearman(per-label gain OOF, per-label gain gold58) over 12 labels = %+.2f | gold58 macro gain .2 on R7 %+.4f' % (name, rho, np.mean(gg)))
json.dump(res, open(OUT, 'w'), indent=1)
