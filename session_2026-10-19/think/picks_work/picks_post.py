# picks_post.py (10-19, think/picks): posterior of the TRUE (population) macro-AUC margins of our LB entries over the heliosli copy
# block, conditioned on every public read we have, then a fresh PRIVATE draw (disjoint 910 studies) for each posterior sample.
# Noise: picks_cov.json (gold58 placement structure, MODEL scaling to 390 / 910). Means: NOT from gold58 (discredited); knob priors
# below are GUESSES (scenario flag). Public reads: MEASURED orderings (best-id moves, display bands) + MODEL magnitudes (density ruler).
import sys, json, numpy as np
SC = sys.argv[1] if len(sys.argv) > 1 else "base"; REG = sys.argv[2] if len(sys.argv) > 2 else "report"
MTOT = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000; SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
J = json.load(open("picks_cov.json")); NM = J["names"]; ix = {k: i for i, k in enumerate(NM)}
ENT = ["COPY", "R6u", "HW", "a25", "c4", "fr65", "pf50", "a15", "c4a15", "fr100", "pf10", "c2", "c2max", "c3max", "hw3", "s615", "wr45", "S6W", "c4noS6", "S6F", "FLIP"]
sel = [ix[x] for x in ENT]
CP = np.array(J["pub_" + REG]["cov"])[np.ix_(sel, sel)]; CV = np.array(J["priv_" + REG]["cov"])[np.ix_(sel, sel)]
NSCALE = float(SC.split("noise")[1]) if SC.startswith("noise") else 1.0      # e.g. noise1.5 / noise0.6 scale ALL sampling noise
CP = CP * NSCALE ** 2; CV = CV * NSCALE ** 2
KN = ["s6", "th", "a", "tcf", "tcp", "tcb", "tcl", "a2x", "f", "p", "l", "s3", "s15", "w", "s6wx", "r", "s6r"]
# prior (mean, sd) per knob, GUESS; units macro AUC
PR = {"s6": (3e-4, 3e-4), "th": (0, 3e-4), "a": (0, 1.5e-4), "tcf": (0, 1.5e-4), "tcp": (0, 1.5e-4), "tcb": (0, 1.5e-4), "tcl": (0, 1.5e-4),
      "a2x": (0, 0.5e-4), "f": (0, 3e-4), "p": (0, 3e-4), "l": (0, 2e-4), "s3": (0, 1.5e-4), "s15": (0, 1e-4), "w": (0, 1.5e-4), "s6wx": (0, 0.5e-4), "r": (0, 5e-4), "s6r": (0, 1e-4)}
if SC == "skepS6": PR["s6"] = (0, 3e-4)
if SC == "r6better": PR["r"] = (4e-4, 3e-4)       # R6 bundle genuinely better than R7 (heliosli's later release; gold58 R6 .937 vs R7 OOF .919)
if SC == "strongS6": PR["s6"] = (5e-4, 2e-4)
if SC == "tabReal": PR["th"] = (3e-4, 2e-4)
if SC == "tabOver": PR["th"] = (-1e-4, 2e-4)
if SC == "bigKnobs":
    for k in KN:
        if k not in ("s6", "th"): PR[k] = (PR[k][0], 2 * PR[k][1])
if SC == "smallKnobs":
    for k in KN:
        if k not in ("s6", "th"): PR[k] = (PR[k][0], 0.5 * PR[k][1])
def T(k):
    out = {n: 0.0 for n in KN}; out.update(k); return out
c4k = {"s6": 1, "a": 1, "tcf": 1, "tcp": 1, "tcb": 1, "tcl": 1}
def plus(d1, d2):
    o = dict(d1); o.update(d2); return o
TR = {"COPY": T({}), "HW": T({"s6": 1}), "R6u": T({"s6": 1, "th": -1}), "a25": T({"s6": 1, "a": 1}), "c4": T(c4k),
      "fr65": T({"s6": 1, "a": 1, "tcf": 1}), "pf50": T({"s6": 1, "a": 1, "tcp": 1}), "a15": T({"s6": 1, "a": 2, "a2x": 1}),
      "c4a15": T(plus(c4k, {"a": 2, "a2x": 1})),
      "fr100": T(plus(c4k, {"f": 1})), "pf10": T(plus(c4k, {"p": 1})), "c2": T(plus(c4k, {"f": 0.75, "p": 1})), "c2max": T(plus(c4k, {"f": 1, "p": 1})),
      "c3max": T(plus(c4k, {"f": 1, "p": 1, "l": 1})), "hw3": T({"s6": 1, "s3": 1}), "s615": T({"s6": 1, "s15": 1}), "wr45": T({"s6": 1, "w": 1}),
      "S6W": T({"s6": 1, "th": -1, "s3": 1, "s6wx": 1}),
      "c4noS6": T({"a": 1, "tcf": 1, "tcp": 1, "tcb": 1, "tcl": 1}),
      "S6F": T({"s6": 1, "th": -1, "r": -1}), "FLIP": T({"th": -1, "r": -1, "s6r": -1})}   # R6u = S6F + r (R7 -> R6 swap); FLIP = S6F - s6 - s6r   # hypothetical: no public read (unconstrained)
A = np.array([[TR[x][k] for k in KN] for x in ENT])          # (E, K)
mu = np.array([PR[k][0] for k in KN]); sd = np.array([PR[k][1] for k in KN])
LP = np.linalg.cholesky(CP + 1e-14 * np.eye(len(ENT))); LV = np.linalg.cholesky(CV + 1e-14 * np.eye(len(ENT)))
e = {k: i for i, k in enumerate(ENT)}
rng = np.random.default_rng(SEED)
def logsig(x, s): return -np.logaddexp(0, -x / s)
# Joint Gaussian X = [knobs (K), absolute public noise (E)]; public margin vs COPY = H X (linear). Condition EXACTLY on the
# MODEL magnitudes (linear-Gaussian), then importance-weight the MEASURED inequalities (soft logistic, scale hs).
nK, nE = len(KN), len(ENT)
m0 = np.concatenate([mu, np.zeros(nE)]); S0 = np.zeros((nK + nE, nK + nE)); S0[:nK, :nK] = np.diag(sd ** 2); S0[nK:, nK:] = CP
Dm = np.eye(nE) - np.eye(nE)[:, [e["COPY"]]]                   # x -> x - x_COPY
H = np.hstack([Dm @ A, Dm])                                     # (E, K+E): public margin vs COPY
OBS = [("HW", "R6u", 3.9e-4, 1.3e-4), ("a25", "HW", 1.5e-4, 0.5e-4), ("c4", "a25", 2.3e-4, 0.8e-4), ("R6u", "COPY", 1.0e-4, 1.0e-4)]
G = np.array([H[e[a_]] - H[e[b_]] for a_, b_, _, _ in OBS]); y = np.array([o for _, _, o, _ in OBS]); R = np.diag([s ** 2 for *_, s in OBS])
Kg = S0 @ G.T @ np.linalg.inv(G @ S0 @ G.T + R)
m1 = m0 + Kg @ (y - G @ m0); S1 = S0 - Kg @ G @ S0; S1 = (S1 + S1.T) / 2
L1 = np.linalg.cholesky(S1 + 1e-16 * np.eye(nK + nE))
CH = 200_000; keepK = []; keepT = []; keepW = []
hs = 0.2e-4
for c in range(max(1, MTOT // CH)):
    X = m1 + rng.standard_normal((CH, nK + nE)) @ L1.T
    K = X[:, :nK]; tru = K @ A.T; pub = X @ H.T
    d = lambda a_, b_: pub[:, e[a_]] - pub[:, e[b_]]
    lw = np.zeros(CH)
    lw += logsig(d("R6u", "COPY"), hs)                          # R6 bracket above the copy block (MEASURED 10-17)
    for x in ["fr65", "pf50", "a15", "c4a15", "fr100", "pf10", "c2", "c2max", "c3max"]: lw += logsig(d("c4", x), hs)   # c4 kept best-id
    for x in ["hw3", "s615", "wr45", "S6W"]: lw += logsig(d("HW", x), hs)                                               # HW kept best-id
    z = rng.uniform(0, 1.5e-4, CH)                              # .958 floor sits z below the copy block (copy runs reach the band floor)
    for x in ["fr100", "pf10"]: lw += logsig(pub[:, e[x]] + z, hs)          # displayed .958
    for x in ["c2", "c2max", "c3max"]: lw += logsig(-(pub[:, e[x]] + z), hs)  # displayed .957
    lw += logsig(-(pub[:, e["c4"]] + z - 1e-3), hs)              # c4 still displays .958 (below .959)
    lw += logsig(d("R6u", "S6F"), hs) + logsig(d("S6F", "FLIP"), hs)   # best-id moves 10-16 (R6) and 10-15 (S6F) (MEASURED)
    lw += logsig(-(pub[:, e["S6F"]] + z), hs)                    # S6F displayed .957
    lw += logsig(-(pub[:, e["FLIP"]] + z + 1e-3), hs)            # V224-FLIP displayed .956
    keepK.append(K.astype(np.float32)); keepT.append(tru.astype(np.float32)); keepW.append(lw)
K = np.concatenate(keepK); TRU = np.concatenate(keepT); LW = np.concatenate(keepW)
W = np.exp(LW - LW.max()); W /= W.sum(); ess = 1 / (W ** 2).sum()
NP = 200_000; idx = rng.choice(len(W), NP, p=W)
Kp = K[idx].astype(float); Tp = TRU[idx].astype(float)
pv = Tp + rng.standard_normal((NP, len(ENT))) @ LV.T; pv = pv - pv[:, [e["COPY"]]]   # private realized margin vs COPY
np.savez_compressed(f"post_{SC}_{REG}.npz", priv=pv.astype(np.float32), true=Tp.astype(np.float32), ent=np.array(ENT))
res = {"scenario": SC, "regime": REG, "M": MTOT, "ess": float(ess),
       "knob_prior": {k: PR[k] for k in KN},
       "knob_post": {k: [float(Kp[:, i].mean()), float(Kp[:, i].std())] for i, k in enumerate(KN)},
       "true_vs_copy": {x: [float(Tp[:, e[x]].mean()), float(Tp[:, e[x]].std())] for x in ENT},
       "priv_vs_copy": {x: [float(pv[:, e[x]].mean()), float(pv[:, e[x]].std()), float((pv[:, e[x]] > 0).mean())] for x in ENT}}
cands = ["R6u", "HW", "a25", "fr100", "pf10", "S6W", "hw3", "fr65", "a15", "c4a15", "pf50", "s615", "wr45", "c4noS6", "S6F"]
pa = pv[:, e["c4"]]; res["pairs"] = {}
for b in cands:
    pb = pv[:, e[b]]; dd = pb - pa
    res["pairs"][b] = dict(P_B_gt_A=float((dd > 0).mean()), mean_BminusA=float(dd.mean()), sd_BminusA=float(dd.std()),
                           hedge_Emax_minus_EA=float(np.maximum(dd, 0).mean()), corr=float(np.corrcoef(pa, pb)[0, 1]),
                           P_max_gt_copy=float((np.maximum(pa, pb) > 0).mean()), P_A_gt_copy=float((pa > 0).mean()))
res["true_steps"] = {f"{a}-{b}": [float((Tp[:, e[a]] - Tp[:, e[b]]).mean()), float((Tp[:, e[a]] - Tp[:, e[b]]).std()), float(((Tp[:, e[a]] - Tp[:, e[b]]) > 0).mean())]
                     for a, b in [("c4", "R6u"), ("c4", "HW"), ("c4", "a25"), ("HW", "R6u"), ("a25", "HW"), ("fr100", "c4"), ("pf10", "c4"), ("c2max", "c4")]}
json.dump(res, open(f"post_{SC}_{REG}.json", "w"), indent=1)
print(SC, REG, "ESS %.0f" % ess)
print(" knobs post (x1e-4 mean, sd):", {k: [round(v[0] * 1e4, 2), round(v[1] * 1e4, 2)] for k, v in res["knob_post"].items()})
print(" true steps (x1e-4 mean, sd, P>0):", {k: [round(v[0] * 1e4, 2), round(v[1] * 1e4, 2), round(v[2], 2)] for k, v in res["true_steps"].items()})
print(" priv vs copy (x1e-4 mean, sd, P>0):", {k: [round(v[0] * 1e4, 2), round(v[1] * 1e4, 2), round(v[2], 3)] for k, v in res["priv_vs_copy"].items() if k in ["c4", "R6u", "HW", "a25", "fr100", "pf10", "S6W"]})
for b, r in res["pairs"].items():
    print(f"  A=c4 B={b:6s} P(B>A) {r['P_B_gt_A']:.3f} mean(B-A) {r['mean_BminusA']*1e4:+.2f}e-4 sd {r['sd_BminusA']*1e4:.2f}e-4 hedge {r['hedge_Emax_minus_EA']*1e4:.2f}e-4 corr {r['corr']:.3f} P(max>copy) {r['P_max_gt_copy']:.3f} (A alone {r['P_A_gt_copy']:.3f})")
