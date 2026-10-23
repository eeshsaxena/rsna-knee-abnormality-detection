# r5_analyze.py (10-20 r5, Claude). Box only, read-only, prints AGGREGATES ONLY (AUCs, deltas, P, rho, phi, n, prevalence).
# Never prints an id or a row; writes nothing. Inputs: the R6 (and optionally S6fall) predictions of the store-rebuilt runs
# (gold58 approx ids = RSNA uids, OAI ids = oai_*), the exact gold58 serves, the gold58 expert labels, OAI MOAKS sources.
#  A  gold58 approx validation (RSNA, no OAI): per label AUC vs gold (exact vs approx), Spearman(exact, approx), OA rho.
#  B  OAI V00 expert reads: per label AUC (R6, S6fall, .8/.2 rank blend, D5 reference), M/L swap check, OA coupling,
#     E-OA delta (a -.1/.1/.2/.3) with paired bootstrap, and the skeptic coupling-matched reweighting (R1 gold74 8-cell,
#     R2 tilt 1.0 / 1.5) exactly as skeptic_eoa_work/box_reweight.py + skeptic_ladder/box_ladder_transfer.py.
# OAI binarisation: OA trio = oai_eoa_check proxies P_FT (primary) and P_AREA from the MOAKS cartilage grades (a knee enters
# only if every compartment has >= 1 read region); other labels = oai_labels_T123.csv (build_oai_labels.py) >= .5, i.e.
# menisci: tear / maceration (morph 2-8) or root tear; Effusion: MOAKS effusion grade >= 2 (moderate+); Baker's: popliteal
# cyst present; ACL: MOAKS ACL tear code >= 1 (graft = masked); Synovitis: Hoffa synovitis grade >= 2 (T123 = grade / 3).
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, norm

H = os.path.expanduser('~')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
       "Baker's", 'Contusion', 'Fracture']
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
G74 = {'000': 33, '001': 12, '010': 3, '011': 2, '100': 4, '101': 8, '110': 2, '111': 7}   # gold74 complete (71), skeptic
CELLS = [format(i, '03b') for i in range(8)]
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    r = rankdata(s); p = y.sum(); q = len(y) - p
    if p == 0 or q == 0:
        return np.nan
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def decor(F, a):
    U = np.stack([pct(F[:, k]) for k in range(3)], 1); O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O


def ns(v):
    return norm.ppf((rankdata(v) - .5) / len(v))


def rd(p):
    d = pd.read_csv(os.path.expanduser(p), dtype={'StudyInstanceUID': str})
    return d.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')[LAB].astype(float)


def blend(A, B, w=.2):
    ids = A.index.intersection(B.index)
    out = pd.DataFrame(index=ids, columns=LAB, dtype=float)
    for c in LAB:
        out[c] = (1 - w) * pct(A.loc[ids, c].values) + w * pct(B.loc[ids, c].values)
    return out


def oa_proxies():
    s = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
    out = {}
    for name in ('P_FT', 'P_AREA'):
        Y = pd.DataFrame(index=s.index, columns=OA, dtype=float)
        for lab, regs in COMP.items():
            c = s[['m_cm_' + r for r in regs]].astype(float).values
            area = np.floor(c + 1e-6); ft = np.round((c - area) * 10); fin = np.isfinite(c)
            pos = ((ft >= 1) & fin) if name == 'P_FT' else ((area >= 2) & fin)
            y = pos.any(1).astype(float); y[~fin.any(1)] = np.nan; Y[lab] = y
        out[name] = Y.dropna()
    return out


def eoa_boot(F, Y, B, seed=0, alphas=(-.1, .05, .1, .15, .2, .3)):
    n = len(Y); rng = np.random.default_rng(seed)
    base = np.array([auc(F[:, k], Y[:, k]) for k in range(3)])
    idx = [rng.integers(0, n, n) for _ in range(B)]
    res = {}
    for a in alphas:
        O = decor(F, a)
        per = np.array([auc(O[:, k], Y[:, k]) for k in range(3)]) - base
        bd = []
        for ii in idx:
            y = Y[ii]
            if (y.sum(0) == 0).any() or (y.sum(0) == n).any():
                continue
            bd.append(np.mean([auc(O[ii, k], y[:, k]) - auc(F[ii, k], y[:, k]) for k in range(3)]))
        bd = np.array(bd)
        res[a] = (per.mean(), bd.std(), (bd > 0).mean(), per)
    return base, res


def evalw(F, Y, w, B=300, seed=0, alphas=(.05, .1, .15, .2, .3)):
    rng = np.random.default_rng(seed); n = len(Y); p = w / w.sum(); ess = 1 / (p ** 2).sum()
    D = {a: [] for a in alphas}; PH = []; RH = []; Q = []
    for b in range(B):
        ii = rng.choice(n, n, p=p); y = Y[ii]; f = F[ii]
        if (y.sum(0) < 5).any() or (y.sum(0) > n - 5).any():
            continue
        PH.append(np.corrcoef(y.T)[np.triu_indices(3, 1)].mean())
        Z = np.stack([ns(f[:, k]) for k in range(3)], 1); RH.append(np.corrcoef(Z.T)[np.triu_indices(3, 1)].mean())
        qq = []
        for k in range(3):
            o = [m for m in range(3) if m != k]; ck = np.corrcoef(Z[:, k], y[:, k])[0, 1]
            qq.append(np.mean([np.corrcoef(Z[:, j], y[:, k])[0, 1] for j in o]) / ck)
        Q.append(np.mean(qq))
        base = np.mean([auc(f[:, k], y[:, k]) for k in range(3)])
        for a in alphas:
            O = decor(f, a); D[a].append(np.mean([auc(O[:, k], y[:, k]) for k in range(3)]) - base)
    s = ' '.join('a%.2f %+.4f(sd %.4f,P %.2f)' % (a, np.mean(v), np.std(v), np.mean(np.array(v) > 0)) for a, v in D.items())
    return 'ESS %4.0f phi %.2f rho %.3f q %.3f gap %+.3f | %s' % (ess, np.mean(PH), np.mean(RH), np.mean(Q), np.mean(RH) - np.mean(Q), s)


def part_a(r6, s6):
    print('=' * 20, 'A. gold58 approx validation (RSNA gold58, expert labels; exact = real-DICOM serve)')
    g = pd.read_csv(H + '/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
    g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID')[LAB].astype(float)
    pairs = [('R6', H + '/r6g58/run_b/worker0/predictions.csv', r6)]
    if s6 is not None:
        pairs.append(('S6fall', H + '/s6seed/gold58_serve_gpu/nartaa_s42/out/predictions.csv', s6))
    for name, ex_p, ap in pairs:
        ex = rd(ex_p)
        ids = g.index.intersection(ex.index).intersection(ap.index)
        print(f'[{name}] gold58 studies matched {len(ids)}')
        if len(ids) < 20:
            continue
        rows = []
        for c in LAB:
            y = g.loc[ids, c].values; ok = np.isfinite(y)
            rows.append((c, auc(ex.loc[ids, c].values[ok], y[ok]), auc(ap.loc[ids, c].values[ok], y[ok]),
                         spearmanr(ex.loc[ids, c].values, ap.loc[ids, c].values).correlation, int(y[ok].sum()), int(ok.sum())))
        t = pd.DataFrame(rows, columns=['label', 'AUC_exact', 'AUC_approx', 'spearman_ex_ap', 'pos', 'n']).set_index('label')
        t['dAUC'] = t.AUC_approx - t.AUC_exact
        print(t.round(4).to_string())
        print(f'[{name}] macro-12 AUC exact {t.AUC_exact.mean():.4f} approx {t.AUC_approx.mean():.4f} | mean spearman '
              f'{t.spearman_ex_ap.mean():.3f} | OA-trio mean AUC exact {t.loc[OA].AUC_exact.mean():.4f} approx '
              f'{t.loc[OA].AUC_approx.mean():.4f}')
        for tag, P in (('exact', ex), ('approx', ap)):
            Z = np.stack([ns(P.loc[ids, c].values) for c in OA], 1)
            r = np.corrcoef(Z.T)[np.triu_indices(3, 1)]
            Yg = g.loc[ids, OA].dropna()
            F = P.loc[Yg.index, OA].values; Yv = Yg.values.astype(int)
            base = np.mean([auc(F[:, k], Yv[:, k]) for k in range(3)])
            d = {a: np.mean([auc(decor(F, a)[:, k], Yv[:, k]) for k in range(3)]) - base for a in (-.1, .1, .2)}
            print(f'[{name} {tag}] OA model rho(ns) ML/MPF/LPF {np.round(r, 3).tolist()} mean {r.mean():.3f} | gold58 E-OA '
                  f'OA-mean dAUC a-.1 {d[-.1]:+.4f} a.1 {d[.1]:+.4f} a.2 {d[.2]:+.4f} (n {len(Yg)})')


def part_b(models, boot):
    print('=' * 20, 'B. OAI V00 expert reads (MOAKS), store-rebuilt DICOM serve')
    T = pd.read_csv(H + '/oai_build/oai_labels_T123.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    prox = oa_proxies()
    common = None
    for k, P in models.items():
        idx = P.dropna().index
        common = idx if common is None else common.intersection(idx)
    print('knees scored by every model:', len(common), '|', {k: int(len(P.dropna())) for k, P in models.items()})
    # per label AUCs
    binz = {c: (T[c] >= .5).astype(float).where(T[c].notna()) for c in LAB if c in T.columns and T[c].notna().any()}
    binz['Effusion>=1'] = (T['Effusion'] >= .3).astype(float).where(T['Effusion'].notna())
    binz['Synovitis>=1'] = (T['Synovitis'] >= .3).astype(float).where(T['Synovitis'].notna())
    rows = []
    for c, y in binz.items():
        out = c if c in LAB else c.split('>=')[0]
        yy = y.reindex(common).dropna()
        r = {'label': c, 'n': len(yy), 'prev': round(float(yy.mean()), 3)}
        for k, P in models.items():
            r[k] = auc(P.loc[yy.index, out].values, yy.values.astype(int))
        rows.append(r)
    for name, Yd in prox.items():
        Y = Yd.reindex(common).dropna()
        for c in OA:
            r = {'label': f'{c} [{name}]', 'n': len(Y), 'prev': round(float(Y[c].mean()), 3)}
            for k, P in models.items():
                r[k] = auc(P.loc[Y.index, c].values, Y[c].values.astype(int))
            rows.append(r)
    t = pd.DataFrame(rows).set_index('label')
    print(t.round(4).to_string())
    # M / L swap check: medial label scored by the lateral output and vice versa (a mirrored geometry would invert this)
    Y = prox['P_FT'].reindex(common).dropna()
    for k, P in models.items():
        mm = auc(P.loc[Y.index, 'Medial OA'], Y['Medial OA'].astype(int)); ml = auc(P.loc[Y.index, 'Lateral OA'], Y['Medial OA'].astype(int))
        ll = auc(P.loc[Y.index, 'Lateral OA'], Y['Lateral OA'].astype(int)); lm = auc(P.loc[Y.index, 'Medial OA'], Y['Lateral OA'].astype(int))
        ym = binz['Medial Meniscus'].reindex(common).dropna(); yl = binz['Lateral Meniscus'].reindex(common).dropna()
        print(f'[{k}] swap check P_FT: MedOA<-Med {mm:.3f} <-Lat {ml:.3f} | LatOA<-Lat {ll:.3f} <-Med {lm:.3f} | MedMen<-Med '
              f'{auc(P.loc[ym.index, "Medial Meniscus"], ym.astype(int)):.3f} <-Lat {auc(P.loc[ym.index, "Lateral Meniscus"], ym.astype(int)):.3f}'
              f' | LatMen<-Lat {auc(P.loc[yl.index, "Lateral Meniscus"], yl.astype(int)):.3f} <-Med '
              f'{auc(P.loc[yl.index, "Medial Meniscus"], yl.astype(int)):.3f}')
    # E-OA
    for name, Yd in prox.items():
        Y = Yd.reindex(common).dropna(); Yv = Y.values.astype(int)
        phi = np.corrcoef(Yv.T)[np.triu_indices(3, 1)]
        print(f'-- E-OA [{name}] n {len(Y)} prevalence M/L/PF {np.round(Yv.mean(0), 3).tolist()} label phi ML/MPF/LPF '
              f'{np.round(phi, 3).tolist()}')
        for k, P in models.items():
            F = P.loc[Y.index, OA].values.astype(float)
            Z = np.stack([ns(F[:, j]) for j in range(3)], 1); rho = np.corrcoef(Z.T)[np.triu_indices(3, 1)]
            base, res = eoa_boot(F, Yv, boot)
            s = ' | '.join(f'a {a:+.2f} {v[0]:+.5f} (sd {v[1]:.5f}, P>0 {v[2]:.3f}) per {np.round(v[3], 4).tolist()}' for a, v in res.items())
            print(f'  [{k}] base OA AUC {np.round(base, 4).tolist()} | model rho(ns) {np.round(rho, 3).tolist()} mean {rho.mean():.3f}')
            print(f'  [{k}] OA-mean dAUC: {s}')
    # skeptic reweighting
    print('-- skeptic coupling-matched reweighting (R0 unweighted, R1 gold74 8-cell, R2 tilt theta 1.0 / 1.5; B 300)')
    pg = np.array([G74[c] + .5 for c in CELLS]); pg = pg / pg.sum()
    for name, Yd in prox.items():
        Y = Yd.reindex(common).dropna(); Yv = Y.values.astype(int)
        key = np.array([''.join(map(str, r)) for r in Yv])
        po = pd.Series(key).value_counts(normalize=True).reindex(CELLS).fillna(0).values
        wc = pg / np.maximum(po, 1e-9); w1 = wc[[CELLS.index(k) for k in key]]
        pairs = Yv[:, 0] * Yv[:, 1] + Yv[:, 0] * Yv[:, 2] + Yv[:, 1] * Yv[:, 2]
        tgt = Yv.mean(0); tilts = {}
        for th in (1.0, 1.5):
            beta = np.zeros(3)
            for _ in range(200):
                w = np.exp(th * pairs + Yv @ beta); cur = (w[:, None] * Yv).sum(0) / w.sum()
                beta += np.log(tgt / (1 - tgt)) - np.log(cur / (1 - cur))
            tilts[th] = w
        print(f'== {name} n {len(Y)} | OAI cell min frac {po.min():.4f}')
        for k, P in models.items():
            F = P.loc[Y.index, OA].values.astype(float)
            print(f'  {k:7s} R0      ', evalw(F, Yv, np.ones(len(Yv))))
            print(f'  {k:7s} R1g74   ', evalw(F, Yv, w1))
            for th, w in tilts.items():
                print(f'  {k:7s} tilt{th:.1f} ', evalw(F, Yv, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r6', required=True)
    ap.add_argument('--s6', default=None)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--parts', default='AB')
    ap.add_argument('--mix', action='store_true')
    a = ap.parse_args()
    r6 = rd(a.r6)
    s6 = rd(a.s6) if a.s6 else None
    if 'A' in a.parts:
        part_a(r6, s6)
    if 'B' in a.parts:
        oai = lambda P: P[P.index.str.startswith('oai_')]
        models = {'R6': oai(r6)}
        if s6 is not None:
            models['S6fall'] = oai(s6)
            models['R6.8S6.2'] = blend(models['R6'], models['S6fall'])
        d5 = rd(H + '/oai_build/teacher/teacher_D5_mean_v00.csv')
        models['D5ref'] = d5.loc[d5.index.intersection(models['R6'].index)]
        if a.mix:   # sensitivity: R6 mixed 50/50 (rank) with an OAI-free fleet model standing in for c4's other (OAI-trained) members
            models['R6+D5'] = blend(models['R6'], models['D5ref'], w=.5)
        part_b(models, a.boot)


if __name__ == '__main__':
    main()
