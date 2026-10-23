# local_fluid.py (10-20 ex, Claude). RESEARCH ONLY, local CPU. CANDIDATE fluid on the RSNA side:
#   FL3(a): Effusion, Synovitis, Baker's each out_t = pct(f_t) - a * mean(pct(f_o)) over the other two
#   ES2(a): Effusion <- Synovitis and Synovitis <- Effusion at a (Baker's untouched)
#   a in {-.1 (control), .1, .2}.
# (i)  gold58 c4 final (m14_gate rebuild, parity .94102), consistency only (gold58 is NOT a selection ruler); also R6,
#      S6fall, Raptor point reads; paired bootstrap for c4.
# (ii) R7 OOF (report-trained) on gold58 and gold74 expert labels.
# (iii) R7 OOF on train (minus gold74) vs report_labels_v2 YES/NO (UNK dropped per label; and UNK=0): the mechanism says
#      decorrelation should LOSE on report labels (reports carry the coupling the model learned).
# (iv) theory: q = corr(ns f_o, y_t) / corr(ns f_t, y_t) vs rho_ns(f_t, f_o); a small-a decorrelation gains iff q < rho.
# Prints aggregates only.
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm, spearmanr
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
FL = ['Effusion', 'Synovitis', "Baker's"]; JF = [L.index(c) for c in FL]
AS = (-.1, .1, .2)
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def transforms(F):
    U = np.stack([pct(F[:, i]) for i in range(3)], 1); out = {}
    for a in AS:
        O = U.copy()
        for i in range(3):
            o = [m for m in range(3) if m != i]; O[:, i] = U[:, i] - a * (U[:, o[0]] + U[:, o[1]]) / 2
        out['FL3 a%+.1f' % a] = O
    for a in AS:
        O = U.copy(); O[:, 0] = U[:, 0] - a * U[:, 1]; O[:, 1] = U[:, 1] - a * U[:, 0]
        out['ES2 a%+.1f' % a] = O
    return out


def evaluate(F, Y, tag, B=2000, seed=0, quiet=False):
    """F n x 3 predictions (E, S, B); Y n x 3 labels with NaN allowed (each label scored on its own coverage)."""
    TR = transforms(F); has = np.isfinite(Y)
    sc = lambda S, ii: np.array([auc(S[ii[has[ii, i]], i], Y[ii[has[ii, i]], i].astype(int)) for i in range(3)])
    al = np.arange(len(Y)); base = sc(F, al)
    pt = {t: sc(O, al) - base for t, O in TR.items()}
    BD = {t: [] for t in TR}
    if B:
        rng = np.random.default_rng(seed)
        for _ in range(B):
            ii = rng.integers(0, len(Y), len(Y)); ok = True
            for i in range(3):
                y = Y[ii[has[ii, i]], i]
                if y.sum() < 2 or len(y) - y.sum() < 2: ok = False
            if not ok: continue
            b0 = sc(F, ii)
            for t, O in TR.items(): BD[t].append(sc(O, ii) - b0)
    print('[%s] n per label %s prev %s | base AUC E/S/B %s' % (tag, has.sum(0).tolist(), np.round(np.nanmean(Y, 0), 3).tolist(),
                                                                 np.round(base, 4).tolist()))
    out = {}
    for t, d in pt.items():
        s3 = d.sum() / 12; s2 = d[:2].sum() / 12
        if B:
            A = np.array(BD[t]); P3 = np.mean(A.sum(1) > 0); P2 = np.mean(A[:, :2].sum(1) > 0); sd3 = A.sum(1).std() / 12
            extra = ' (P>0 %.2f, sd %.5f) ES-sum %+.5f (P>0 %.2f)' % (P3, sd3, s2, P2)
        else:
            extra = ' ES-sum %+.5f' % s2
        if not quiet:
            print('   %-10s macro FL3-sum %+.5f%s | per E/S/B %s' % (t, s3, extra, np.round(d, 4).tolist()))
        out[t] = (s3, s2, d)
    return out


def theory(F, Y, tag):
    Z = np.stack([ns(F[:, i]) for i in range(3)], 1); has = np.isfinite(Y); s = []
    for i in range(3):
        h = has[:, i]; ct = np.corrcoef(Z[h, i], Y[h, i])[0, 1]
        for o in range(3):
            if o == i: continue
            q = np.corrcoef(Z[h, o], Y[h, i])[0, 1] / ct; rho = np.corrcoef(Z[h, i], Z[h, o])[0, 1]
            s.append('%s<-%s q %.2f rho %.2f (rho-q %+.2f)' % (FL[i][:3], FL[o][:3], q, rho, rho - q))
    print('   theory [%s]: %s' % (tag, ' | '.join(s)))


# ---------------- gold58 c4 rebuild ----------------
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); n58 = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
C4 = np.zeros((n58, 12)); RAP = np.zeros((n58, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); RAP[:, j] = rr
    b = w * rr + (1 - w) * pct(R6[:, j]); C4[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
Y58all = g58[L].values.astype(int)
print('c4 rebuild gold58 macro %.5f (m14_gate.log: .94102)' % np.mean([auc(C4[:, j], Y58all[:, j]) for j in range(12)]))
Y58 = g58[FL].values.astype(float)
for nm, M in (('c4', C4), ('R6', R6), ('S6fall', S6), ('Raptor', RAP)):
    F = M[:, JF]
    print('gold58 %s rho E-S %+.2f E-B %+.2f S-B %+.2f' % (nm, *[spearmanr(F[:, i], F[:, j]).correlation for i, j in ((0, 1), (0, 2), (1, 2))]))
evaluate(C4[:, JF], Y58, 'gold58 c4 final', B=4000)
theory(C4[:, JF], Y58, 'gold58 c4')
for nm, M in (('R6', R6), ('S6fall', S6), ('Raptor', RAP)):
    evaluate(M[:, JF], Y58, 'gold58 ' + nm, B=0)
# leave-one-out sensitivity of the c4 ES2 / FL3 reads (how many single studies drive the sign)
F = C4[:, JF]; full = evaluate(F, Y58, 'gold58 c4 (quiet)', B=0, quiet=True)
for t in ('FL3 a+0.1', 'ES2 a+0.1'):
    vals = []
    for k in range(n58):
        m = np.arange(n58) != k; TR = transforms(F[m]); Ym = Y58[m]
        b0 = np.array([auc(F[m][:, i], Ym[:, i].astype(int)) for i in range(3)])
        d = np.array([auc(TR[t][:, i], Ym[:, i].astype(int)) for i in range(3)]) - b0
        vals.append(d.sum() / 12 if t.startswith('FL3') else d[:2].sum() / 12)
    vals = np.array(vals)
    print('   LOO c4 %s: full %+.5f | LOO min %+.5f max %+.5f | frac LOO > 0 %.2f' % (
        t, full[t][0] if t.startswith('FL3') else full[t][1], vals.min(), vals.max(), np.mean(vals > 0)))

# ---------------- R7 OOF on gold58 / gold74 ----------------
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in L]].set_axis(L, axis=1)
evaluate(F7.loc[ids, FL].values, Y58, 'gold58 R7 OOF', B=2000)
gi = g.StudyInstanceUID.tolist()
evaluate(F7.loc[gi, FL].values, g[FL].values.astype(float), 'gold74 R7 OOF', B=2000)
theory(F7.loc[gi, FL].values, g[FL].values.astype(float), 'gold74 R7')

# ---------------- R7 OOF on train vs report labels ----------------
F7t = F7[~F7.index.isin(g.StudyInstanceUID)]
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
cm = F7t.index.intersection(rep.index)
V = pd.DataFrame({c: np.where(rep.loc[cm, c + '__verdict'] == 'YES', 1.0, np.where(rep.loc[cm, c + '__verdict'] == 'NO', 0.0, np.nan)) for c in FL}, index=cm)
Ft = F7t.loc[cm, FL].values
print('train R7 rho E-S %+.2f E-B %+.2f S-B %+.2f (n %d)' % (*[spearmanr(Ft[:, i], Ft[:, j]).correlation for i, j in ((0, 1), (0, 2), (1, 2))], len(cm)))
evaluate(Ft, V.values, 'train R7 OOF vs report v2 (UNK dropped per label)', B=300)
theory(Ft, V.values, 'train R7 vs report')
evaluate(Ft, V.fillna(0.).values, 'train R7 OOF vs report v2 (UNK = 0)', B=300)
