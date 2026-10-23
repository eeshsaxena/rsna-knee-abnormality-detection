# stack_ops.py (10-19, Claude, research only): blend-OPERATOR comparisons on gold58 at the c4 weights, plus a resolution model.
# CPU numpy only. Inputs: session_2026-10-18/next/g58 (gold58 preds of r384, r224, R6 serve + R6 dump logits, S6fall, R7 OOF).
# Every variant keeps the c4 effective weights; only the operator changes. Paired study bootstrap vs c4 (ranks over all 58 fixed).
import json, sys, numpy as np, pandas as pd
from scipy.stats import rankdata, norm

D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
A224, WS6 = 0.25, 0.2
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
R6 = rd(D + "/R6_gold58_serve.csv").loc[ids, L].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = rd(D + "/n42gpu_gold58.csv").loc[ids, L].values
R7 = rd(D + "/R7OOF.csv").loc[ids, ["p|" + l for l in L]].values
S43 = rd(D + "/s43_gold58.csv").loc[ids, L].values

# R6 full-precision deployed logit: fold-mean of the flip-averaged z; Baker's = state stacker z_out, Effusion = fluid stacker z_out
W0 = D + "/r6g58_box/run_b/worker0"
dump = json.load(open(W0 + "/dump.json"))
Z6 = np.array([np.mean(np.array(dump[u]["z"], np.float64), axis=0) for u in ids])
st = rd(W0 + "/state_stacker.csv"); fl = rd(W0 + "/fluid_stacker.csv")
Z6[:, L.index("Baker's")] = st.loc[ids, "Baker's|z_out"].values
Z6[:, L.index("Effusion")] = fl.loc[ids, "Effusion|z_out"].values
chk = np.corrcoef(rankdata(1 / (1 + np.exp(-Z6[:, 0]))), rankdata(R6[:, 0]))[0, 1]
P6z = 1 / (1 + np.exp(-Z6))

eps = 1e-7
lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
pct = lambda v: rankdata(v) / len(v)
nsc = lambda v: norm.ppf((rankdata(v) - 0.5) / len(v))
zst = lambda v: (v - v.mean()) / (v.std() + 1e-12)


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def ew(c):  # c4 effective flat weights (r384, r224, ours, s6)
    w = C4.get(c, .5)
    return np.array([(1 - WS6) * w * (1 - A224), (1 - WS6) * w * A224, (1 - WS6) * (1 - w), WS6])


def blend(op, ours=R6, s6=S6):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        X = [R384[:, j], R224[:, j], ours[:, j], s6[:, j]]
        w = C4.get(c, .5); e = ew(c)
        if op == 'c4':          # the kernel: nested rank with re-rank after stage 3
            rr = (1 - A224) * pct(X[0]) + A224 * pct(X[1]); b = w * rr + (1 - w) * pct(X[2])
            out[:, j] = (1 - WS6) * pct(b) + WS6 * pct(X[3])
        elif op == 'flat_rank':
            out[:, j] = sum(e[k] * pct(X[k]) for k in range(4))
        elif op == 'nested_nscore':
            rr = (1 - A224) * nsc(X[0]) + A224 * nsc(X[1]); b = w * rr + (1 - w) * nsc(X[2])
            out[:, j] = (1 - WS6) * nsc(b) + WS6 * nsc(X[3])
        elif op == 'flat_nscore':
            out[:, j] = sum(e[k] * nsc(X[k]) for k in range(4))
        elif op == 'flat_prob':
            out[:, j] = sum(e[k] * X[k] for k in range(4))
        elif op == 'flat_logit':
            out[:, j] = sum(e[k] * lg(X[k]) for k in range(4))
        elif op == 'flat_zlogit':
            out[:, j] = sum(e[k] * zst(lg(X[k])) for k in range(4))
        elif op == 'geo_rank':   # weighted geometric mean of percentile ranks (soft AND)
            out[:, j] = sum(e[k] * np.log(pct(X[k])) for k in range(4))
        elif op == 'pow2_rank':  # weighted quadratic mean of ranks (soft OR)
            out[:, j] = sum(e[k] * pct(X[k]) ** 2 for k in range(4))
        elif op == 'pow4_rank':
            out[:, j] = sum(e[k] * pct(X[k]) ** 4 for k in range(4))
        elif op == 'inv_geo':    # 1 - geomean(1 - r): soft OR on the other side
            out[:, j] = -sum(e[k] * np.log(1 - pct(X[k]) + 0.5 / n) for k in range(4))
        elif op == 'raptor_prob_slot':   # Raptor slot as a probability mean (same family), the rest as c4
            rr = pct((1 - A224) * X[0] + A224 * X[1]); b = w * rr + (1 - w) * pct(X[2])
            out[:, j] = (1 - WS6) * pct(b) + WS6 * pct(X[3])
        elif op == 'raptor_logit_slot':
            rr = pct((1 - A224) * lg(X[0]) + A224 * lg(X[1])); b = w * rr + (1 - w) * pct(X[2])
            out[:, j] = (1 - WS6) * pct(b) + WS6 * pct(X[3])
        elif op == 'eq4_rank':   # equal weight on the 4 members (crowd)
            out[:, j] = sum(pct(X[k]) for k in range(4)) / 4
        elif op == 'eq3fam_rank':  # equal weight on 3 families (Raptor slot, R6, S6)
            rr = (1 - A224) * pct(X[0]) + A224 * pct(X[1])
            out[:, j] = (pct(rr) + pct(X[2]) + pct(X[3])) / 3
        elif op == 'median4':
            out[:, j] = np.median(np.stack([pct(X[k]) for k in range(4)]), axis=0)
        elif op == 'trim_wmean':  # drop the member farthest from the weighted mean per study, renormalise
            R = np.stack([pct(X[k]) for k in range(4)]); m = (e[:, None] * R).sum(0)
            far = np.abs(R - m).argmax(0); M = np.ones_like(R); M[far, np.arange(n)] = 0
            out[:, j] = (e[:, None] * R * M).sum(0) / (e[:, None] * M).sum(0)
        else:
            raise ValueError(op)
    return out


def blend5(extra, we=0.15):  # c4 then a 5th member mixed in by rank at weight we (crowd test)
    P = blend('c4'); out = np.zeros_like(P)
    for j in range(12):
        out[:, j] = (1 - we) * pct(P[:, j]) + we * pct(extra[:, j])
    return out


rng = np.random.default_rng(11); B = 3000; idxs = []
while len(idxs) < B:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)


def boot(P):
    return np.array([[auc(P[ix, j], Y[ix, j]) for j in range(12)] for ix in idxs])


mac = lambda P: np.mean([auc(P[:, j], Y[:, j]) for j in range(12)])
P0 = blend('c4'); b0 = boot(P0); base = mac(P0)
res = {'base_c4': base, 'n': n, 'check_rank_corr_R6p_vs_dumpz': chk}
print('c4 gold58 macro %.5f (dump z vs serve p rank corr ACL %.6f)' % (base, chk))
VARS = ['flat_rank', 'nested_nscore', 'flat_nscore', 'flat_prob', 'flat_logit', 'flat_zlogit', 'geo_rank', 'pow2_rank', 'pow4_rank',
        'inv_geo', 'raptor_prob_slot', 'raptor_logit_slot', 'eq4_rank', 'eq3fam_rank', 'median4', 'trim_wmean']
rows = []
for v in VARS + ['c4_R6z', 'flat_logit_R6z', 'flat_zlogit_R6z', 'plus_R7_15', 'plus_S43_15', 'plus_R7_25']:
    if v == 'c4_R6z': P = blend('c4', ours=P6z)
    elif v == 'flat_logit_R6z': P = blend('flat_logit', ours=P6z)
    elif v == 'flat_zlogit_R6z': P = blend('flat_zlogit', ours=P6z)
    elif v == 'plus_R7_15': P = blend5(R7, .15)
    elif v == 'plus_R7_25': P = blend5(R7, .25)
    elif v == 'plus_S43_15': P = blend5(S43, .15)
    else: P = blend(v)
    d_pt = mac(P) - base
    bb = boot(P); d = np.nanmean(bb - b0, axis=1)
    per = np.array([auc(P[:, j], Y[:, j]) - auc(P0[:, j], Y[:, j]) for j in range(12)])
    sp = np.mean([pd.Series(P[:, j]).corr(pd.Series(P0[:, j]), method='spearman') for j in range(12)])
    rows.append((v, d_pt, d.std(), (d > 0).mean(), (d < 0).mean(), sp, per))
    print('%-18s d %+.5f  boot sd %.5f  P>0 %.3f P<0 %.3f  spearman vs c4 %.4f  top labels %s' % (
        v, d_pt, d.std(), (d > 0).mean(), (d < 0).mean(), sp,
        sorted([(L[k], round(per[k], 4)) for k in range(12) if abs(per[k]) > 0], key=lambda t: -abs(t[1]))[:3]))
    res[v] = {'d': d_pt, 'sd': float(d.std()), 'P>0': float((d > 0).mean()), 'P<0': float((d < 0).mean()), 'spearman_vs_c4': sp,
              'per_label': dict(zip(L, map(float, per)))}

# single-member AUCs for reference
res['single'] = {nm: {L[j]: auc(X[:, j], Y[:, j]) for j in range(12)} for nm, X in
                 [('r384', R384), ('r224', R224), ('R6', R6), ('R6z', P6z), ('S6', S6), ('R7', R7)]}
for nm in res['single']:
    print('single %-5s macro %.4f' % (nm, np.mean(list(res['single'][nm].values()))))
json.dump(res, open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack_ops.json', 'w'), indent=1, default=float)
