# local_space.py (10-20 levers, Claude): RSNA-side checks for the output-transform space after E-OA. Competition data only.
# 1 label-free OA coupling (rho, normal scores) of each c4 member on gold58: which member is the low-coupling one (role check
#   for the OAI Raptor-weight interaction); q / gap with gold58 labels as INFO only.
# 2 report-label OOF ruler (R7 OOF, report labels v2, train minus gold74) as a MECHANISM CONTROL: linear vs convex / concave
#   subtraction on report-regime labels.
# 3 gold74 report-vs-expert: report false-positive rate by number of expert-positive OTHER compartments (tricompartmental
#   over-call shape).
# 4 gold58 c4 replay of lin / p2 (sym and pw) at a .2 / .25 / .3: CODE CHECK / INFO ONLY, not a selection ruler.
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58/'
LBL = 'C:/Users/eeshs/Downloads/rsna-knee/_labels/rsna-knee-llm-labels/report_labels_v2.csv'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']; IOA = [L.index(c) for c in OA]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
PW = {(0, 1): .876, (0, 2): .708, (1, 2): 1.416}
pct = lambda v: rankdata(v) / len(v)
pctc = lambda X: np.stack([pct(X[:, j]) for j in range(X.shape[1])], 1)
rng = np.random.default_rng(2011)


def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def coup(F, Y=None):
    Z = norm.ppf((rankdata(F, axis=0) - .5) / len(F))
    C = np.corrcoef(Z.T); rho = np.mean([C[0, 1], C[0, 2], C[1, 2]])
    if Y is None:
        return rho, [round(C[0, 1], 3), round(C[0, 2], 3), round(C[1, 2], 3)]
    q = np.mean([np.mean([np.corrcoef(Z[:, j], Y[:, k])[0, 1] for j in range(3) if j != k]) / np.corrcoef(Z[:, k], Y[:, k])[0, 1] for k in range(3)])
    return rho, q, rho - q


def eoah(F, a, p=1., pw=False):
    U = pctc(F); S = U ** p; out = np.empty_like(U)
    for t in range(3):
        o = [k for k in range(3) if k != t]
        ww = [PW[tuple(sorted((t, k)))] for k in o] if pw else [1., 1.]
        out[:, t] = U[:, t] - a * (ww[0] * S[:, o[0]] + ww[1] * S[:, o[1]]) / 2.
    return out


# ---------------- 1. gold58 coupling per member ----------------
g = pd.read_csv(D + 'gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g58 = g[g.source == 'original_gold58'].reset_index(drop=True); ids = g58.StudyInstanceUID.tolist(); Y58 = g58[OA].values.astype(int)
R6 = pd.read_csv(D + 'R6_gold58_serve.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R384 = np.load(D + 'gold58_r384_kernelserve_cpu.npy'); R224 = np.load(D + 'gold58_r224_kernelserve.npy')
S6 = pd.read_csv(D + 'n42gpu_gold58.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
assert pd.read_csv(D + 'gold58_r224_ids.csv', dtype=str).StudyInstanceUID.tolist() == ids
RAP = .75 * pctc(R384) + .25 * pctc(R224)


def c4final(m=1., rap=RAP, r6=R6, s6=S6):
    out = np.zeros((58, 12))
    for j, c in enumerate(L):
        w = C4.get(c, .5) * (m if c in OA else 1.)
        b = w * rap[:, j] + (1 - w) * pct(r6[:, j]); out[:, j] = .8 * pct(b) + .2 * pct(s6[:, j])
    return out


C4F = c4final()
print('== 1. gold58 OA coupling per member (rho label-free; q / gap use gold58 labels: INFO ONLY), n 58')
for nm, X in (('Raptor r384', R384), ('Raptor r224', R224), ('Raptor mix .75/.25', RAP), ('R6 exact', R6), ('S6fall n42gpu', S6),
              ('nonRaptor .67 R6 + .33 S6', (.4 / .6) * pctc(R6) + (.2 / .6) * pctc(S6)), ('c4 final', C4F)):
    F = X[:, IOA]; rho, pairs = coup(F); r, q, gp = coup(F, Y58)
    au = [round(auc(F[:, k], Y58[:, k]), 3) for k in range(3)]
    print('  %-26s rho %.3f pairs M-L/M-PF/L-PF %s | q %.3f gap %+.3f | AUC M/L/PF %s' % (nm, rho, pairs, q, gp, au))
# bootstrap: rho(Raptor) - rho(nonRaptor) (label-free) and gap difference
dr, dg = [], []
NR = (.4 / .6) * pctc(R6) + (.2 / .6) * pctc(S6)
for b in range(2000):
    ix = rng.integers(0, 58, 58)
    if any(Y58[ix, k].sum() in (0, 58) for k in range(3)):
        continue
    a1 = coup(RAP[ix][:, IOA], Y58[ix]); a2 = coup(NR[ix][:, IOA], Y58[ix])
    dr.append(a1[0] - a2[0]); dg.append(a1[2] - a2[2])
dr, dg = np.array(dr), np.array(dg)
print('  boot Raptor - nonRaptor: rho %+.3f (sd %.3f, P<0 %.2f) | gap %+.3f (sd %.3f, P<0 %.2f) | B %d' % (
    dr.mean(), dr.std(), (dr < 0).mean(), dg.mean(), dg.std(), (dg < 0).mean(), len(dr)))

# ---------------- 2. report-label OOF mechanism control ----------------
R7 = pd.read_csv(D + 'R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
RL = pd.read_csv(LBL, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
ix = R7.index.intersection(RL.index).difference(g.StudyInstanceUID)
F7 = R7.loc[ix, ['p|' + c for c in OA]].values.astype(float); YR = RL.loc[ix, OA].values.astype(float)
ok = np.isfinite(F7).all(1) & np.isfinite(YR).all(1); F7 = F7[ok]; YR = (YR[ok] >= .5).astype(int)
print('== 2. report-label OOF ruler (R7 OOF x report labels v2 >= .5, train minus gold74): n %d, prev %s, label phi %s' % (
    len(F7), np.round(YR.mean(0), 3).tolist(), [round(np.corrcoef(YR[:, i], YR[:, j])[0, 1], 3) for i, j in ((0, 1), (0, 2), (1, 2))]))
r, q, gp = coup(F7, YR); print('  R7 OOF rho %.3f q %.3f gap %+.3f' % (r, q, gp))
base = np.mean([auc(F7[:, k], YR[:, k]) for k in range(3)])
res = {}
for p in (1., 1.5, 2., .5):
    for a in (.1, .2, .3):
        S = eoah(F7, a, p); res[(p, a)] = np.mean([auc(S[:, k], YR[:, k]) for k in range(3)]) - base
print('  OA-mean dAUC on REPORT labels (expected < 0 for every E-OA form):')
for p in (1., 1.5, 2., .5):
    print('    h = u^%-3s a .1 / .2 / .3: %s' % (p, np.round([res[(p, a)] for a in (.1, .2, .3)], 5).tolist()))
# paired bootstrap p2 - lin at a .2 on report labels
dd = []
for b in range(300):
    ix2 = rng.integers(0, len(F7), len(F7)); Fb, Yb = F7[ix2], YR[ix2]
    b0 = [eoah(Fb, .2, 1.), eoah(Fb, .2, 2.)]
    dd.append(np.mean([auc(b0[1][:, k], Yb[:, k]) for k in range(3)]) - np.mean([auc(b0[0][:, k], Yb[:, k]) for k in range(3)]))
dd = np.array(dd); print('  report labels, p2 - lin at a .2: %+.5f (sd %.5f, P>0 %.2f)' % (dd.mean(), dd.std(), (dd > 0).mean()))
# where is report coupling concentrated? report P(Y_t | k other report-positive) vs gold74 expert (sec 3)
k = np.stack([YR[:, [o for o in range(3) if o != t]].sum(1) for t in range(3)], 1)
print('  report labels: P(Y_t = 1 | k other compartments report-positive), pooled over t: ' +
      ' | '.join('k %d: %.3f (n %d)' % (kk, YR[k == kk].mean(), (k == kk).sum()) for kk in range(3)))

# ---------------- 3. gold74 report vs expert on the same knees ----------------
gi = g.set_index('StudyInstanceUID'); idx = gi.index.intersection(RL.index)
E = gi.loc[idx, OA].astype(float); R = RL.loc[idx, OA].astype(float); okk = E.notna().all(axis=1) & R.notna().all(axis=1)
E = E[okk].values.astype(int); R = (R[okk].values >= .5).astype(int)
fp = {kk: [0, 0] for kk in range(3)}; fn = {kk: [0, 0] for kk in range(3)}; ep = {kk: [0, 0] for kk in range(3)}
for t in range(3):
    o = [x for x in range(3) if x != t]; kk_ = E[:, o].sum(1)
    for kk in range(3):
        m = kk_ == kk
        fp[kk][0] += int(((R[:, t] == 1) & (E[:, t] == 0) & m).sum()); fp[kk][1] += int(((E[:, t] == 0) & m).sum())
        fn[kk][0] += int(((R[:, t] == 0) & (E[:, t] == 1) & m).sum()); fn[kk][1] += int(((E[:, t] == 1) & m).sum())
        ep[kk][0] += int(((E[:, t] == 1) & m).sum()); ep[kk][1] += int(m.sum())
print('== 3. gold74 (n %d knees x 3 labels): report vs expert by number k of expert-positive OTHER compartments' % len(E))
for kk in range(3):
    print('  k %d: expert P(Y_t=1) %d/%d = %.2f | report FP %d/%d = %.2f | report FN %d/%d = %.2f' % (
        kk, ep[kk][0], ep[kk][1], ep[kk][0] / max(ep[kk][1], 1), fp[kk][0], fp[kk][1], fp[kk][0] / max(fp[kk][1], 1),
        fn[kk][0], fn[kk][1], fn[kk][0] / max(fn[kk][1], 1)))

# ---------------- 4. gold58 c4 replay (CODE CHECK / INFO ONLY) ----------------
b58 = np.mean([auc(C4F[:, j], g58[L[j]].values.astype(int)) for j in range(12)])
print('== 4. gold58 c4 replay macro %.5f (INFO ONLY, not a ruler); OA-trio deltas x 3/12 = macro' % b58)
for pw in (False, True):
    for p in (1., 1.5, 2.):
        row = []
        for a in (.2, .25, .3):
            S = eoah(C4F[:, IOA], a, p, pw); row.append(np.mean([auc(S[:, k], Y58[:, k]) for k in range(3)]) - np.mean([auc(C4F[:, IOA][:, k], Y58[:, k]) for k in range(3)]))
        print('  %s p %.1f a .2/.25/.3 macro %s' % ('pw ' if pw else 'sym', p, np.round(np.array(row) * 3 / 12, 5).tolist()))
for m in (.5, .75, 1., 1.25):
    F = c4final(m); row = []
    for a in (0., .2, .3):
        S = eoah(F[:, IOA], a) if a else F[:, IOA]
        row.append(np.mean([auc(S[:, k], Y58[:, k]) for k in range(3)]))
    print('  Raptor OA multiplier m %.2f: OA-mean at a 0 / .2 / .3 %s (INFO ONLY)' % (m, np.round(row, 4).tolist()))
