# final_picks_stress.py (10-19, Claude): final-pick stress test on gold58 as a proxy for the private split.
# Blend = c4 kernel form: rr = (1-A) pct(r384) + A pct(r224); b = w_c rr + (1-w_c) pct(ours=R6); final = (1-ws6) pct(b) + ws6 pct(S6fall).
# ours = R6_gold58_serve.csv (REAL R6 leg, gold_in_training 0). pct over the 58 (fixed predictions).
# Resampling (per replicate r):
#   stage 1 ("truth"): joint study bootstrap of the 58 (keeps cross-label correlation) = one plausible population;
#   stage 2 ("private"): per label, draw n_pos / n_neg private studies from that population's positives / negatives
#                         (report regime: Fracture 62 / 848 of 910; or gold58 regime); ("public") the same with 390.
# Shrink variants (deterministic per-config per-label AUC shifts applied to every replicate, computed on the full 58):
#   S1 = Raptor-up w gains halved (w_c > .5 and the w part of the gain vs w_c = .5 at the same A is positive);
#   S2 = every per-label delta vs R6 uniform halved (symmetric, "half of all gold58-visible tuning is noise");
#   S3 = Raptor-up w gains zeroed (extreme).
# LB fusion: stage-1 replicates reweighted by the likelihood of the public reads (MODEL magnitudes from density.md + this session).
# A15 rule simulation: A = public argmax among today's slate (c4 incumbent), scored on the independent private draw.
import sys, json, os, numpy as np, pandas as pd
from scipy.stats import rankdata, norm
D = sys.argv[1]; OUT = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000; SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 19
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(D + "/gold58_r224_ids.csv", dtype=str).StudyInstanceUID.tolist() == ids
R6 = pd.read_csv(D + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
assert R6.shape == R384.shape == R224.shape == S6.shape == (n, 12)
pct = lambda v: rankdata(v) / len(v)

def final(table, A, ws6=0.2):
    out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(R384[:, j]) + A * pct(R224[:, j]); w = table.get(c, 0.5)
        b = w * rr + (1 - w) * pct(R6[:, j]); out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(S6[:, j])
    return out

CFG = {  # name: (table, A224, ws6, kaggle id / note)
    "R6u":   ({}, 0.35, 0.2, "57248084 uniform + S6fall .2 (current B)"),
    "HW":    (HW, 0.35, 0.2, "57262115 heliosli table"),
    "a25":   (HW, 0.25, 0.2, "57316395 HW + A224 .25"),
    "c4":    (C4, 0.25, 0.2, "57343435 team best"),
    "fr100": ({**C4, "Fracture": 1.0}, 0.25, 0.2, "57382011"),
    "c2":    ({**C4, "Fracture": 0.9, "PF OA": 0.1}, 0.25, 0.2, "57382015"),
    "pf10":  ({**C4, "PF OA": 0.1}, 0.25, 0.2, "57382016"),
    "c2max": ({**C4, "Fracture": 1.0, "PF OA": 0.1}, 0.25, 0.2, "57382007"),
    "c3max": ({**C4, "Fracture": 1.0, "PF OA": 0.1, "Lateral OA": 0.1}, 0.25, 0.2, "pending gate"),
    "COPY":  (HW, 0.35, 0.0, "heliosli copy block: HW table, no S6fall"),
    "fr65":  ({**HW, "Fracture": 0.65}, 0.25, 0.2, "57343462 read < c4"),
    "a15":   (HW, 0.15, 0.2, "57344106 read < c4"),
    "c4a15": (C4, 0.15, 0.2, "57344133 read < c4"),
    "pf50":  ({**HW, "PF OA": 0.5}, 0.25, 0.2, "57344538 read < c4"),
}
NAMES = list(CFG); K = len(NAMES)
P = {k: final(t, A, w) for k, (t, A, w, _) in CFG.items()}

def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
PT = {k: np.array([auc(P[k][:, j], Y[:, j]) for j in range(12)]) for k in NAMES}   # point per-label AUC on the 58

# ---- shrink shifts (per config, per label) ----
def wpart(k, j):
    t, A, w6, _ = CFG[k]; c = L[j]; w = t.get(c, 0.5)
    if w <= 0.5: return 0.0
    t2 = {**t, c: 0.5}; P2 = final(t2, A, w6)
    return PT[k][j] - auc(P2[:, j], Y[:, j])
SH = {"S0": {k: np.zeros(12) for k in NAMES}}
SH["S1"] = {k: np.array([-0.5 * max(0.0, wpart(k, j)) for j in range(12)]) for k in NAMES}
SH["S3"] = {k: np.array([-1.0 * max(0.0, wpart(k, j)) for j in range(12)]) for k in NAMES}
SH["S2"] = {k: -0.5 * (PT[k] - PT["R6u"]) if k != "COPY" else np.zeros(12) for k in NAMES}
# COPY under S2: keep its own point (it is not a gold58-tuned entry of ours); under S1/S3 its HW Raptor-up gains are shrunk like HW's.
# S4 (mechanistic, symmetric): deflate Raptor itself. Gaussian noise on pct(r384) and pct(r224) (sigma calibrated so the Raptor slot
# rr = .75 pct(r384) + .25 pct(r224) loses DEFL macro AUC on gold58, averaged over draws); shift = mean noisy AUC - clean AUC per config/label.
DEFL = float(os.environ.get("DEFL", "0.008")); ND = 24
rngd = np.random.default_rng(777); EPS = rngd.standard_normal((ND, 2, n, 12))
p384 = np.column_stack([pct(R384[:, j]) for j in range(12)]); p224 = np.column_stack([pct(R224[:, j]) for j in range(12)])
def rr_macro(sig):
    v = []
    for d in range(ND):
        a, b = p384 + sig * EPS[d, 0], p224 + sig * EPS[d, 1]
        rr = np.column_stack([0.75 * pct(a[:, j]) + 0.25 * pct(b[:, j]) for j in range(12)])
        v.append(np.mean([auc(rr[:, j], Y[:, j]) for j in range(12)]))
    return float(np.mean(v))
rr0 = np.mean([auc(0.75 * p384[:, j] + 0.25 * p224[:, j], Y[:, j]) for j in range(12)])
lo, hi = 0.0, 1.0
for _ in range(25):
    mid = (lo + hi) / 2
    if rr0 - rr_macro(mid) < DEFL: lo = mid
    else: hi = mid
SIG = (lo + hi) / 2
def final_noisy(table, A, ws6, d):
    a, b = p384 + SIG * EPS[d, 0], p224 + SIG * EPS[d, 1]; out = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = (1 - A) * pct(a[:, j]) + A * pct(b[:, j]); w = table.get(c, 0.5)
        bb = w * rr + (1 - w) * pct(R6[:, j]); out[:, j] = (1 - ws6) * pct(bb) + ws6 * pct(S6[:, j])
    return out
SH["S4"] = {}
for k in NAMES:
    t, A, w6, _ = CFG[k]
    m = np.mean([[auc(final_noisy(t, A, w6, d)[:, j], Y[:, j]) for j in range(12)] for d in range(ND)], axis=0)
    SH["S4"][k] = m - PT[k]
S4INFO = dict(defl=DEFL, sigma=SIG, rr_clean=float(rr0), rr_noisy=rr_macro(SIG))

# ---- counts ----
REP = dict(zip(L, [.204, .147, .394, .151, .359, .257, .450, .593, .126, .247, .168, .068]))
def counts(N, regime):
    if regime == "report": pos = np.array([round(N * REP[c]) for c in L])
    else: pos = np.round(N * Y.mean(0)).astype(int)
    return pos.astype(int), (N - pos).astype(int)
PRIV = {"report": counts(910, "report"), "g58": counts(910, "g58")}
PUB = counts(390, "report")

# ---- comparison matrices ----
POS = [np.where(Y[:, j] == 1)[0] for j in range(12)]; NEG = [np.where(Y[:, j] == 0)[0] for j in range(12)]
CM = {}
for k in NAMES:
    for j in range(12):
        sp = P[k][POS[j], j][:, None]; sn = P[k][NEG[j], j][None, :]
        CM[k, j] = ((sp > sn) + 0.5 * (sp == sn)).astype(np.float64)

rng = np.random.default_rng(SEED)
c1 = []
while sum(len(x) for x in c1) < B:
    c = rng.multinomial(n, np.full(n, 1.0 / n), size=B)
    ok = np.ones(B, bool)
    for j in range(12):
        ok &= (c[:, POS[j]].sum(1) > 0) & (c[:, NEG[j]].sum(1) > 0)
    c1.append(c[ok])
c1 = np.concatenate(c1)[:B].astype(np.float64)

def stage2(wpos, wneg, npos, nneg):
    pp = wpos / wpos.sum(1, keepdims=True); pn = wneg / wneg.sum(1, keepdims=True)
    return rng.multinomial(npos, pp).astype(np.float64), rng.multinomial(nneg, pn).astype(np.float64)

def aucw(k, j, wp, wn):
    return ((wp @ CM[k, j]) * wn).sum(1) / (wp.sum(1) * wn.sum(1))

# AUC arrays: mode -> (B, K, 12)
MODES = ["truth", "priv_report", "priv_g58", "pub", "priv_cond"]
AU = {m: np.zeros((B, K, 12)) for m in MODES}
for j in range(12):
    wp1 = c1[:, POS[j]]; wn1 = c1[:, NEG[j]]
    W = {"truth": (wp1, wn1)}
    W["priv_report"] = stage2(wp1, wn1, PRIV["report"][0][j], PRIV["report"][1][j])
    W["priv_g58"] = stage2(wp1, wn1, PRIV["g58"][0][j], PRIV["g58"][1][j])
    W["pub"] = stage2(wp1, wn1, PUB[0][j], PUB[1][j])
    # conditional private (population = the full 58, i.e. private noise only)
    one_p = np.ones((B, len(POS[j]))); one_n = np.ones((B, len(NEG[j])))
    W["priv_cond"] = stage2(one_p, one_n, PRIV["report"][0][j], PRIV["report"][1][j])   # name kept short: = private noise only, truth = gold58
    for ki, k in enumerate(NAMES):
        for m in MODES:
            AU[m][:, ki, j] = aucw(k, j, *W[m])
ix = {k: i for i, k in enumerate(NAMES)}

def macro(mode, shrink):
    S = np.stack([SH[shrink][k] for k in NAMES])  # (K, 12)
    return (AU[mode] + S[None]).mean(2)  # (B, K)

def wstats(x, w):
    w = w / w.sum(); m = (w * x).sum(); return m, np.sqrt((w * (x - m) ** 2).sum())

A_SET = ["c4", "c2max", "c3max", "fr100", "c2", "pf10"]; B_SET = ["R6u", "HW", "a25", "c4"]; A4 = ["c4", "c2max", "c3max", "fr100"]
def wcorr(x, y, w):
    wn = w / w.sum(); mx, sx = wstats(x, w); my, sy = wstats(y, w); return float((wn * (x - mx) * (y - my)).sum() / (sx * sy))
def summarize(mode, shrink, w=None):
    M = macro(mode, shrink); w = np.ones(B) if w is None else w; wn = w / w.sum()
    res = {"mean": {}, "sd": {}, "p_pair": {}, "p_best": {}, "grid": {}}
    for k in NAMES:
        res["mean"][k], res["sd"][k] = map(float, wstats(M[:, ix[k]], w))
    allk = list(dict.fromkeys(A_SET + B_SET))
    for a in allk:
        for b in allk:
            if a != b: res["p_pair"][a + ">" + b] = float((wn * (M[:, ix[a]] > M[:, ix[b]])).sum())
    for nm_, S_ in (("A4", A4), ("B4", B_SET), ("all6", ["R6u", "HW", "a25", "c4", "c2max", "c3max"])):
        si = [ix[k] for k in S_]; am = np.array(si)[np.argmax(M[:, si], 1)]
        res["p_best"][nm_] = {NAMES[i]: float((wn * (am == i)).sum()) for i in si}
    cp = M[:, ix["COPY"]]
    for a in A_SET:
        for b in B_SET:
            if a == b: continue
            ma, mb = M[:, ix[a]], M[:, ix[b]]; d = mb - ma
            res["grid"][a + "|" + b] = dict(
                EA=float((wn * ma).sum()), Emax=float((wn * np.maximum(ma, mb)).sum()), hedge=float((wn * np.maximum(d, 0)).sum()),
                pB_gt_A=float((wn * (d > 0)).sum()), mean_BminusA=float((wn * d).sum()), sd_BminusA=float(wstats(d, w)[1]),
                corr_raw=wcorr(ma, mb, w), corr_margin_vs_copy=wcorr(ma - cp, mb - cp, w),
                pA_gt_copy=float((wn * (ma > cp)).sum()), pmax_gt_copy=float((wn * (np.maximum(ma, mb) > cp)).sum()))
    return res

OUTJ = {"n": n, "B": B, "seed": SEED, "s4": S4INFO,
        "configs": {k: dict(table={c: v for c, v in CFG[k][0].items()}, A224=CFG[k][1], ws6=CFG[k][2], note=CFG[k][3]) for k in NAMES},
        "priv_counts_report": dict(zip(L, PRIV["report"][0].tolist())), "priv_counts_g58": dict(zip(L, PRIV["g58"][0].tolist())),
        "pub_counts": dict(zip(L, PUB[0].tolist())),
        "point_macro": {k: float(PT[k].mean()) for k in NAMES},
        "point_per_label": {k: dict(zip(L, PT[k].round(5).tolist())) for k in NAMES},
        "point_per_label_vs_R6u": {k: dict(zip(L, (PT[k] - PT["R6u"]).round(5).tolist())) for k in NAMES},
        "point_per_label_vs_c4": {k: dict(zip(L, (PT[k] - PT["c4"]).round(5).tolist())) for k in NAMES},
        "shift_macro": {s: {k: float(SH[s][k].mean()) for k in NAMES} for s in SH},
        "shift_per_label": {s: {k: dict(zip(L, SH[s][k].round(5).tolist())) for k in NAMES} for s in SH}}
OUTJ["runs"] = {}
for mode in ["priv_report", "priv_g58", "priv_cond", "truth"]:
    for s in ["S0", "S1", "S2", "S3", "S4"]:
        if mode != "priv_report" and s not in ("S0", "S1"): continue
        OUTJ["runs"][mode + "|" + s] = summarize(mode, s)

# ---- LB fusion: reweight stage-1 "truth" replicates by the public-read likelihood (public noise marginalised as Gaussian) ----
def pair_pub_sd(a, b, shrink="S0"):
    Mp = macro("pub", shrink); Mt = macro("truth", shrink)
    return float(np.std((Mp[:, ix[a]] - Mp[:, ix[b]]) - (Mt[:, ix[a]] - Mt[:, ix[b]])))
CHAIN = [("HW", "R6u", 3.9e-4, 1.25e-4), ("a25", "HW", 1.5e-4, 0.4e-4), ("c4", "a25", 2.3e-4, 0.6e-4)]  # MODEL (density.md; c4 - a25 from the 10-19 frozen-team count 17/25)
CENS = [("c4", "fr65"), ("c4", "a15"), ("c4", "c4a15"), ("c4", "pf50")]   # MEASURED order: c4 took the best-id over these (parallel A18 reads)
COPYOBS = [("HW", "COPY", 5.0e-4, 1.5e-4)]; COPYCENS = [("R6u", "COPY")]   # MODEL / MEASURED (lb_private.md: B bracket above the copy block)
def lb_weights(shrink, hw_bias_frac=0.0, cens=True, copy=True):
    Mt = macro("truth", shrink); ll = np.zeros(B); info = []
    obs = CHAIN + (COPYOBS if copy else []); cc = (CENS if cens else []) + (COPYCENS if copy else [])
    for a, b, o, sdd in obs:
        sd = np.sqrt(pair_pub_sd(a, b, shrink) ** 2 + sdd ** 2)
        o = o * (1 - hw_bias_frac) if (a, b) == ("HW", "R6u") else o
        ll += norm.logpdf(o, Mt[:, ix[a]] - Mt[:, ix[b]], sd); info.append((a, b, o, round(sd, 6)))
    for a, b in cc:
        sd = pair_pub_sd(a, b, shrink); ll += norm.logcdf((Mt[:, ix[a]] - Mt[:, ix[b]]) / sd); info.append((a, b, ">0", round(sd, 6)))
    w = np.exp(ll - ll.max()); ess = w.sum() ** 2 / (w ** 2).sum()
    return w, float(ess), info
STEPS = [("HW", "R6u"), ("a25", "HW"), ("c4", "a25"), ("c4", "R6u"), ("HW", "COPY"), ("R6u", "COPY"), ("c4", "COPY"),
         ("fr100", "c4"), ("pf10", "c4"), ("c2max", "c4"), ("c3max", "c4"), ("c3max", "c2max")]
OUTJ["lb"] = {}; LBW = {}
for nm, s, hb, cen, cp in [("LB_main", "S0", 0.0, True, True), ("LB_hwover50", "S0", 0.5, True, True), ("LB_S1", "S1", 0.0, True, True),
                           ("LB_S2", "S2", 0.0, True, True), ("LB_S4", "S4", 0.0, True, True), ("LB_S1_hwover50", "S1", 0.5, True, True),
                           ("LB_chain_only", "S0", 0.0, False, False)]:
    w, ess, info = lb_weights(s, hb, cen, cp); LBW[nm] = (s, w)
    r = summarize("priv_report", s, w); r["ess"] = ess; r["terms"] = [list(map(str, t)) for t in info]
    Mt = macro("truth", s)
    r["posterior_truth_steps"] = {f"{a}-{b}": list(map(float, wstats(Mt[:, ix[a]] - Mt[:, ix[b]], w))) for a, b in STEPS}
    r["prior_truth_steps"] = {f"{a}-{b}": list(map(float, wstats(Mt[:, ix[a]] - Mt[:, ix[b]], np.ones(B)))) for a, b in STEPS}
    OUTJ["lb"][nm] = r

# ---- A15 rule simulation (today's slate): A = public argmax over {c4, c2max, fr100, c2, pf10, c3max}; scored on the private draw ----
# (fresh public noise given the truth replicate; the correlation with the already-observed public reads is ignored)
SLATE = ["c4", "c2max", "fr100", "c2", "pf10", "c3max"]
OUTJ["a15_rule"] = {}
for nm, (s, w) in [("S0", ("S0", np.ones(B))), ("S1", ("S1", np.ones(B))), ("S2", ("S2", np.ones(B))), ("S4", ("S4", np.ones(B))),
                   ("LB_main", LBW["LB_main"]), ("LB_S1", LBW["LB_S1"]), ("LB_S2", LBW["LB_S2"])]:
    wn = w / w.sum(); Mp = macro("pub", s); Mv = macro("priv_report", s)
    si = [ix[k] for k in SLATE]; pick = np.array(si)[np.argmax(Mp[:, si], 1)]
    pp = Mv[np.arange(B), pick]
    out = dict(E_rule=float((wn * pp).sum()), pick_freq={NAMES[i]: float((wn * (pick == i)).sum()) for i in si},
               E_fixed={k: float((wn * Mv[:, ix[k]]).sum()) for k in SLATE},
               p_rule_ge_c4=float((wn * (pp >= Mv[:, ix["c4"]])).sum()),
               E_oracle=float((wn * Mv[:, si].max(1)).sum()))
    for bname in B_SET:
        mb = Mv[:, ix[bname]]; out["Emax_with_" + bname] = float((wn * np.maximum(pp, mb)).sum())
        out["hedge_with_" + bname] = float((wn * np.maximum(mb - pp, 0)).sum()); out["pB_gt_A_" + bname] = float((wn * (mb > pp)).sum())
    OUTJ["a15_rule"][nm] = out
json.dump(OUTJ, open(OUT, "w"), indent=1)
print("done", OUT, "B", B, "S4", S4INFO)
