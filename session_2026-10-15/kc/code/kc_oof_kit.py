# kc_oof_kit.py -- BOX_PLAN_1015 C7 (frozen refs) and G2 (KC family sanity floor, PREREG_10-15 Amendment 6), ONE code
# path for both so the bars are computed exactly as the refs were. Box CPU only; reads RSNA files only.
#   refs:  python kc_oof_kit.py --refs-only --out ~/kc/gate/refs.json
#   G2:    python kc_oof_kit.py --g2 --refs ~/kc/gate/refs.json --kc_gold M1.csv M2.csv [M3.csv]
#                               --kc_oof <f0 OOF csv at E*> --out ~/kc/gate/kc_g2.txt
# Metrics (= ~/kcrop_probe/gold_ref.py, MEASURED earlier): gold74 macro AUC over the labels with both classes present
# (NaN masked); blend = per-label (1 - w) rank(R7 OOF) + w rank(X), w = .2, rank = rankdata / n; rho = per-label-mean
# Spearman on the fold-0 val studies of folds5 vs R7 OOF p| columns.
# G2 bars (A6 as amended by the adopted critique): leg gold74 >= D_gold74 - .010; blend gain over R7 alone >= 0 (was
# +.0024); rho(KC f0 OOF, R7) <= rho(T_b f0, R7) + .02. A sanity floor only; the LB read (R1) decides.
import os, sys, glob, json, hashlib, argparse, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

H = os.path.expanduser('~')
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
ap = argparse.ArgumentParser()
ap.add_argument('--refs-only', dest='refs_only', action='store_true')
ap.add_argument('--g2', action='store_true')
ap.add_argument('--refs', default=f'{H}/kc/gate/refs.json')
ap.add_argument('--kc_gold', nargs='*', default=[])
ap.add_argument('--kc_oof', default='')
ap.add_argument('--out', required=True)
ap.add_argument('--gold', default=f'{H}/rsna_train/gold74_eval_set.csv')
ap.add_argument('--r7', default=f'{H}/r7t_labels/real/R7OOF.csv')
ap.add_argument('--folds', default=f'{H}/rsna_train/folds5.csv')
ap.add_argument('--gold_preds', default=f'{H}/r7t_labels/gold_preds')
ap.add_argument('--tb_f0_oof', default=f'{H}/fleet_runs/f0_s1047_ttr_r7tb/oof_f0_s1047_raw.csv')
ap.add_argument('--w', type=float, default=0.2)
A = ap.parse_args()

md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
g = pd.read_csv(A.gold, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
r7 = pd.read_csv(A.r7, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
R7 = r7.loc[g.index, ['p|' + l for l in L]].values
g58 = np.where(g.source.values == 'original_gold58')[0]
fo = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']


def auc(P, idx=None):
    idx = np.arange(len(g)) if idx is None else idx
    out = []
    for j, l in enumerate(L):
        y = g[l].values[idx]; m = ~np.isnan(y)
        if m.sum() > 0 and len(set(y[m])) == 2:
            out.append(roc_auc_score(y[m], P[idx][m, j]))
    return float(np.mean(out)), len(out)


def rk(P):
    return np.column_stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])])


def gain(P, w=None):
    w = A.w if w is None else w
    B = (1 - w) * rk(R7) + w * rk(P)
    return auc(B)[0] - auc(R7)[0], auc(B, g58)[0] - auc(R7, g58)[0]


def fam(tag):
    fs = sorted(glob.glob(f'{A.gold_preds}/f*_{tag}__raw__A.csv'))
    a = [pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[g.index, L].values for f in fs]
    return np.mean(a, 0), fs


def rho_vs_r7(oof_csv):
    o = pd.read_csv(oof_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = [s for s in o.index if s in fo.index and fo[s] == 0 and s in r7.index]
    rs = [spearmanr(o.loc[ids, l].values, r7.loc[ids, 'p|' + l].values).correlation for l in L]
    return float(np.nanmean(rs)), len(ids), {l: round(float(r), 4) for l, r in zip(L, rs)}


code_md5 = md5(os.path.abspath(__file__))
if A.refs_only:
    S, sf = fam('s1047_ttr_r7tb')
    D, df_ = fam('s1044_ttr_frac75')
    rho, n_rho, rho_lab = rho_vs_r7(A.tb_f0_oof)
    refs = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), code_md5=code_md5, w=A.w,
                inputs={p: md5(p) for p in [A.gold, A.r7, A.folds, A.tb_f0_oof] + sf + df_},
                gold74=dict(R7=auc(R7)[0], S=auc(S)[0], D=auc(D)[0], n_labels=auc(R7)[1]),
                gold58=dict(R7=auc(R7, g58)[0], S=auc(S, g58)[0], D=auc(D, g58)[0]),
                gain_at_w=dict(S=gain(S)[0], D=gain(D)[0], S_gold58=gain(S)[1], D_gold58=gain(D)[1]),
                rho_tb_f0_r7=dict(rho=rho, n=n_rho, per_label=rho_lab, oof=A.tb_f0_oof))
    refs['g2_bars'] = dict(leg_gold74_min=refs['gold74']['D'] - 0.010, gain_min=0.0,
                           rho_max=rho + 0.02, note='A6 G2 as amended by the adopted critique (gain bar >= 0, was +.0024)')
    json.dump(refs, open(A.out, 'w'), indent=1)
    print(json.dumps({k: refs[k] for k in ('gold74', 'gain_at_w', 'g2_bars')}, indent=1))
    print(f"rho(T_b f0 OOF, R7 OOF) fold 0 = {rho:.4f} (n {n_rho}) | code md5 {code_md5}")
    sys.exit(0)

assert A.g2 and A.kc_gold and A.kc_oof, '--g2 needs --kc_gold (member gold74 csvs) and --kc_oof'
refs = json.load(open(A.refs))
assert refs['code_md5'] == code_md5, f'kc_oof_kit.py changed since the refs were frozen ({refs["code_md5"]} vs {code_md5})'
mem = [pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[g.index, L].values for f in A.kc_gold]
leg = np.mean(mem, 0)
a_leg = auc(leg)[0]; gn74, gn58 = gain(leg); rho, n_rho, rho_lab = rho_vs_r7(A.kc_oof)
b = refs['g2_bars']
chk = dict(leg_gold74=(a_leg, a_leg >= b['leg_gold74_min']), gain=(gn74, gn74 >= b['gain_min']),
           rho=(rho, rho <= b['rho_max']))
ok = all(v[1] for v in chk.values())
lines = [f"KC_G2 {'PASS' if ok else 'FAIL'} | leg gold74 {a_leg:.4f} (bar >= {b['leg_gold74_min']:.4f}) | gain .2 "
         f"{gn74:+.4f} (bar >= {b['gain_min']}) gold58 {gn58:+.4f} | rho(KC f0, R7) {rho:.4f} (bar <= {b['rho_max']:.4f}, n {n_rho})",
         f"# members {len(mem)}: " + ' '.join(os.path.basename(f) for f in A.kc_gold),
         f"# member gold74: " + ' '.join(f'{auc(m)[0]:.4f}' for m in mem),
         f"# rho per label: {rho_lab}", f"# refs {A.refs} | code md5 {code_md5} | {time.strftime('%Y-%m-%d %H:%M:%S')}"]
open(A.out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
