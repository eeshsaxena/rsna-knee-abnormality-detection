# oai_eoa_check.py (10-19, Claude, TRICK_1019 sec 4 G1). Read-only analysis; prints AGGREGATES ONLY (AUCs, deltas, P, phi,
# rho, n, prevalence). Never writes a file, never prints an id or a row. Meant to run on the box (OAI rows never leave it).
#
# Question: does "OA decorrelation" (E-OA) raise per-compartment AUC against EXPERT image reads when the model was trained on
# REPORT labels? E-OA on a prediction table f: for t in (Medial OA, Lateral OA, PF OA), with o1, o2 the other two,
#   out_t = pct(f_t) - a * (pct(f_o1) + pct(f_o2)) / 2.
# Expert labels on the box: OAI V00 MOAKS cartilage per subregion (oai_label_sources.csv, m_cm_* = "area.fullthickness").
# Compartments = build_oai_labels.py COMP cartilage regions (osteophytes and X-ray JSN deliberately NOT used: they add coupling
# that the RSNA rubric does not have). Proxies of the rubric "moderate or large area (~1 cm+) of > 50 % thickness loss":
#   P_FT   (PRIMARY): any subregion of the compartment with full-thickness grade >= 1
#   P_AREA          : any subregion with area grade >= 2 (10-75 % or > 75 % of the subregion)
#   P_BOTH          : any subregion with area >= 2 AND full-thickness >= 1 (strictest)
# A knee enters a proxy only if >= 1 cartilage subregion of EVERY compartment was read (so the 3 labels share one cohort).
#
# Pre-registered reading (written before any OAI number was seen), per a in {.1, .2}:
#   SUPPORT : teacher mean, primary proxy: mean-OA-AUC delta > 0 with paired-bootstrap P>0 >= .90, AND delta > 0 on >= 2
#             of the 3 proxies, AND the a = -.1 control (positive borrowing) has delta < 0 on the primary proxy.
#   KILL    : primary delta <= 0 (or P>0 < .5).
#   else    : INCONCLUSIVE (E-OA stays a coin flip; do not spend a slot on it on this evidence).
#   SHIP_A  : .2 if SUPPORT at .2 and delta(.2) > delta(.1); else .1 if SUPPORT at .1; else none.
#   The per-fold lines are a replication read (same recipe, 5 folds), not part of the rule.
# Known limit (MEASURED on synthetic data, TRICK_1019 sec 4): low label phi + high model rho is NOT enough for decorrelation to
# help; if the models' shared OA component is real shared disease, decorrelation loses. That is why the AUC delta, not the phi
# gap, is the gate.
#
# Usage (box):   ~/miniconda3/envs/rsna/bin/python -I oai_eoa_check.py --sources ~/oai_build/oai_label_sources.csv \
#                  --preds ~/oai_build/teacher/teacher_D5_mean_v00.csv --fold-glob '~/oai_build/teacher/f{k}_s1043_ttr_frac75__raw__A__shard*of4.csv'
# Dry run (laptop, gold58 parity): --labels-binary <csv with StudyInstanceUID + OA columns> --preds <c4 gold58 final csv>
import argparse, glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def decor(F, a):
    U = np.stack([pct(F[:, k]) for k in range(3)], 1); O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O


def proxies(src):
    s = pd.read_csv(src, dtype={'kid': str}).set_index('kid')
    out = {}
    for name in ('P_FT', 'P_AREA', 'P_BOTH'):
        Y = pd.DataFrame(index=s.index, columns=OA, dtype=float)
        for lab, regs in COMP.items():
            c = s[['m_cm_' + r for r in regs]].astype(float).values
            area = np.floor(c + 1e-6); ft = np.round((c - area) * 10)
            fin = np.isfinite(c)
            if name == 'P_FT':
                pos = (ft >= 1) & fin
            elif name == 'P_AREA':
                pos = (area >= 2) & fin
            else:
                pos = (area >= 2) & (ft >= 1) & fin
            y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan
            Y[lab] = y
        out[name] = Y.dropna()
    return out


def load_preds(p):
    d = pd.read_csv(os.path.expanduser(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    return d[~d.index.duplicated()][OA].dropna()


def run(F, Y, tag, B=2000, seed=0):
    n = len(Y); rng = np.random.default_rng(seed)
    if n < 20:
        print(f'[{tag}] only {n} matched rows, skipped'); return {}
    prev = Y.mean(0)
    phi = np.corrcoef(Y.T); rho = spearmanr(F).correlation
    print(f'[{tag}] n {n} | prevalence M/L/PF {np.round(prev, 3).tolist()} | label phi ML {phi[0,1]:.3f} MPF {phi[0,2]:.3f} '
          f'LPF {phi[1,2]:.3f} | model rho ML {rho[0,1]:.3f} MPF {rho[0,2]:.3f} LPF {rho[1,2]:.3f}')
    if (Y.sum(0) < 5).any() or ((n - Y.sum(0)) < 5).any():
        print(f'[{tag}] too few positives or negatives in some OA label, skipped'); return {}
    base = np.array([auc(F[:, k], Y[:, k]) for k in range(3)])
    res = {}
    idx = [rng.integers(0, n, n) for _ in range(B)]
    for a in (-0.1, 0.1, 0.2):
        O = decor(F, a)
        per = np.array([auc(O[:, k], Y[:, k]) for k in range(3)]) - base
        bd = []
        for ii in idx:
            y = Y[ii]
            if (y.sum(0) == 0).any() or (y.sum(0) == len(ii)).any():
                continue
            bd.append(np.mean([auc(O[ii, k], y[:, k]) - auc(F[ii, k], y[:, k]) for k in range(3)]))
        bd = np.array(bd)
        res[a] = (per.mean(), (bd > 0).mean())
        print(f'[{tag}] a {a:+.1f}: OA-mean AUC delta {per.mean():+.5f} (macro-12 equiv {per.mean() * 3 / 12:+.5f}) | per label '
              f'{np.round(per, 4).tolist()} | boot sd {bd.std():.5f} P>0 {(bd > 0).mean():.3f} | base AUC {np.round(base, 4).tolist()}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True)
    ap.add_argument('--sources')
    ap.add_argument('--labels-binary')
    ap.add_argument('--fold-glob')
    ap.add_argument('--boot', type=int, default=2000)
    A = ap.parse_args()
    if A.sources:
        Ys = proxies(os.path.expanduser(A.sources))
    else:
        g = pd.read_csv(A.labels_binary, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        Ys = {'BINARY': g[OA].astype(float).dropna()}
    P = load_preds(A.preds)
    verdict = {}
    for name, Yd in Ys.items():
        ids = Yd.index.intersection(P.index)
        res = run(P.loc[ids].values.astype(float), Yd.loc[ids].values.astype(int), f'mean {name}', A.boot)
        verdict[name] = res
    if A.fold_glob:
        for k in range(10):
            fs = sorted(glob.glob(os.path.expanduser(A.fold_glob.format(k=k))))
            if not fs:
                continue
            Pk = pd.concat([pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in fs]).drop_duplicates('StudyInstanceUID')
            Pk = Pk.set_index('StudyInstanceUID')[OA].dropna()
            name = 'P_FT' if 'P_FT' in Ys else list(Ys)[0]; Yd = Ys[name]
            ids = Yd.index.intersection(Pk.index)
            run(Pk.loc[ids].values.astype(float), Yd.loc[ids].values.astype(int), f'fold{k} {name}', 500, seed=k + 1)
    prim = 'P_FT' if 'P_FT' in verdict else list(verdict)[0]
    r = verdict.get(prim, {})
    if not r:
        print('VERDICT: NO DATA'); return
    dneg = r[-0.1][0]
    out = {}
    for a in (0.1, 0.2):
        d, p = r[a]
        others = sum(1 for v in verdict.values() if v and v[a][0] > 0)
        if d > 0 and p >= .90 and others >= min(2, len(verdict)) and dneg < 0:
            out[a] = 'SUPPORT'
        elif d <= 0 or p < .5:
            out[a] = 'KILL'
        else:
            out[a] = 'INCONCLUSIVE'
        print(f'VERDICT ({prim}, a {a:.1f}): {out[a]} | delta {d:+.5f} P>0 {p:.3f} | proxies with delta>0: {others}/{len(verdict)} '
              f'| a -.1 control {dneg:+.5f}')
    # pre-registered ship rule: a = .2 only if it is SUPPORT and beats a = .1 on the primary proxy; else a = .1 if SUPPORT; else none
    if out[0.2] == 'SUPPORT' and r[0.2][0] > r[0.1][0]:
        ship = '0.2'
    elif out[0.1] == 'SUPPORT':
        ship = '0.1'
    else:
        ship = 'NONE (E-OA not built)'
    print(f'SHIP_A: {ship}')


if __name__ == '__main__':
    main()
