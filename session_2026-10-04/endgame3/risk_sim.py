# Risk + final-pick simulation (CPU, small). Joint binormal two-reader model on ONE hidden test draw per world:
# parent P (macro AUC .943), crop member C (cropaug-type R50 crop), fleet member F (independent multi-view family).
# Every candidate = rank blend of P with C and/or F; public/private deltas come from the SAME study draw, so
# candidates sharing a member are correlated exactly as rank blends are. Displayed public = round(.943+p0+dpub,3).
# Private delta -> P(bronze)/P(silver) through round2/lens_rules Pcurves (field model, base and creep+overfit).
# Usage: python risk_sim.py [draws]
import sys, json, numpy as np
from scipy.stats import norm, rankdata
SP = "C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/scratchpad"
PC = np.load(SP + "/round2/lens_rules/Pcurves.npy"); XS = np.load(SP + "/round2/lens_rules/XS.npy")
def pmed(x, sc, col): return np.interp(x, XS, PC[sc][:, col], left=0.0, right=1.0)   # col 2 bronze, 1 silver
prev = np.array([.204, .147, .394, .151, .359, .257, .450, .593, .126, .247, .168, .068])
A_P = np.array([.985, .95, .93, .92, .955, .95, .93, .945, .905, .96, .94, .93]); A_P = A_P - (A_P.mean() - 0.943)
dP = np.sqrt(2) * norm.ppf(A_P); N, NPUB = 1300, 390
def kscale(t):
    lo, hi = 0.1, 1.5
    for _ in range(40):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if norm.cdf(k * dP / np.sqrt(2)).mean() < t else (lo, k)
    return (lo + hi) / 2
KG = {round(a, 3): kscale(a) for a in np.arange(0.900, 0.9505, 0.001)}
def kk(a): return KG[round(min(max(a, 0.900), 0.950), 3)]
def macro(S, Y):
    r = rankdata(S, axis=0); npos = Y.sum(0); nneg = len(Y) - npos
    return (((r * Y).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)).mean()
WC = [0.15, 0.20, 0.25, 0.35, 0.45]; WF = [0.30, 0.50]
def world(rng, m1, r1, m2, r2, r12):
    Y = (rng.random((N, 12)) < prev).astype(float)
    eP = rng.standard_normal((N, 12)); z1 = rng.standard_normal((N, 12)); z2 = rng.standard_normal((N, 12))
    eC = r1 * eP + np.sqrt(1 - r1 ** 2) * z1
    a = (r12 - r1 * r2) / np.sqrt(1 - r1 ** 2); b = np.sqrt(max(1 - r2 ** 2 - a ** 2, 1e-6))
    eF = r2 * eP + a * z1 + b * z2
    SPm = dP * Y + eP; SC = kk(m1) * dP * Y + eC; SF = kk(m2) * dP * Y + eF
    rP, rC, rF = (rankdata(S, axis=0) / N for S in (SPm, SC, SF))
    idx = rng.permutation(N); pu, pr = idx[:NPUB], idx[NPUB:]
    base = (macro(SPm[pu], Y[pu]), macro(SPm[pr], Y[pr]))
    cand = {}
    def add(name, S):
        cand[name] = (macro(S[pu], Y[pu]) - base[0], macro(S[pr], Y[pr]) - base[1])
    for w in WC: add("C%02d" % int(w * 100), (1 - w) * rP + w * rC)
    for w in WF: add("F%02d" % int(w * 100), (1 - w) * rP + w * rF)
    add("CF", 0.60 * rP + 0.15 * rC + 0.25 * rF)            # combined fleet leg (crop .15 + fleet .25)
    add("Csolo", rC); add("Fsolo", rF)
    return cand
def disp(p0, d): return np.round(0.943 + p0 + d, 3)
PRIORS = {  # (m1 lo,hi), (r1 lo,hi), (m2 lo,hi), (r2 lo,hi), (r12 lo,hi)
    "opt":  ((0.912, 0.930), (0.78, 0.90), (0.915, 0.935), (0.72, 0.86), (0.45, 0.75)),
    "cen":  ((0.912, 0.930), (0.80, 0.93), (0.915, 0.935), (0.76, 0.90), (0.50, 0.80)),
    "pes":  ((0.910, 0.926), (0.85, 0.94), (0.912, 0.928), (0.82, 0.92), (0.60, 0.85)),
}
if __name__ == "__main__":
    pri = sys.argv[1]; S = int(sys.argv[2]); seed = int(sys.argv[3])
    (a1, b1), (c1, d1), (a2, b2), (c2, d2), (e, f) = PRIORS[pri]
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(S):
        m1 = rng.uniform(a1, b1); r1 = rng.uniform(c1, d1)
        m2 = rng.uniform(a2, b2); r2 = rng.uniform(c2, d2); r12 = rng.uniform(e, f)
        p0 = rng.uniform(-0.0005, 0.0005)
        c = world(rng, m1, r1, m2, r2, r12)
        rows.append(dict(m1=m1, r1=r1, m2=m2, r2=r2, r12=r12, p0=p0, **{k: v for k, v in c.items()}))
    np.save(SP + "/endgame3/risk_draws_%s_%d.npy" % (pri, seed), np.array(rows, dtype=object), allow_pickle=True)
    print("saved", pri, seed, len(rows))
