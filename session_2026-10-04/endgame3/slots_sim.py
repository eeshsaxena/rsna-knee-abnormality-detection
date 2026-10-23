# Slot-lens helper (CPU, small). Two parts:
#  A) closed-form binormal decision tables (best W and lift for a member of solo AUC a2 and within-class rho vs
#     the 0.943 parent; and the 3-way parent+crop+fleet optimum). Same formula as fleet_g1b.best_lift.
#  B) a small rank-blend simulation (noise_calib geometry: 1300 studies, 390 public, 12 labels) that asks how many
#     LB slots a W-curve is worth: fixed-W vs "W30+solo" vs "argmax over 4 displayed W reads" vs "model fit".
# Usage: python slots_sim.py [worlds]
import sys, numpy as np
from scipy.stats import norm, rankdata
def mu(a): return np.sqrt(2) * norm.ppf(a)
def blend_auc(ws, aucs, R):
    ws = np.asarray(ws, float); m = ws @ mu(np.asarray(aucs)); v = ws @ R @ ws
    return norm.cdf(m / np.sqrt(2 * v))
def best2(a1, a2, rho, grid=np.linspace(0, 1, 201)):
    w = np.asarray(grid, float); m = (1 - w) * mu(a1) + w * mu(a2); v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * rho
    b = norm.cdf(m / np.sqrt(2 * v)); i = b.argmax(); return w[i], b[i] - a1, b
def tables():
    print("== A1: best W / lift over parent .943 (closed form) ==")
    rhos = [0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
    print("solo\\rho " + " ".join("%13.2f" % r for r in rhos))
    for a2 in [0.900, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940, 0.945]:
        cells = []
        for r in rhos:
            w, l, _ = best2(0.943, a2, r); cells.append("W%.2f %+.4f" % (w, l))
        print("%.3f    " % a2 + " ".join("%13s" % c for c in cells))
    print("\n== A2: lift at fixed W (rows) for solo .925 / .930 / .935, rho .78 / .82 / .86 ==")
    for a2 in [0.925, 0.930, 0.935]:
        for r in [0.78, 0.82, 0.86]:
            _, _, b = best2(0.943, a2, r, grid=np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]))
            print("a2 %.3f rho %.2f: " % (a2, r) + " ".join("%+.4f" % (x - 0.943) for x in b))
    print("   (W columns: .15 .20 .25 .30 .35 .40 .45 .50 .60)")
    print("\n== A3: 3-way parent + crop C + fleet F (closed form), optimum over (wC,wF) ==")
    g = np.linspace(0, 0.6, 61)
    for (aC, rPC, aF, rPF, rCF) in [(0.925, 0.80, 0.925, 0.82, 0.80), (0.925, 0.80, 0.930, 0.80, 0.80),
                                    (0.925, 0.80, 0.935, 0.78, 0.75), (0.925, 0.82, 0.935, 0.78, 0.85),
                                    (0.925, 0.86, 0.930, 0.80, 0.80), (0.920, 0.80, 0.940, 0.78, 0.78)]:
        R = np.array([[1, rPC, rPF], [rPC, 1, rCF], [rPF, rCF, 1]]); best = (-1, 0, 0)
        for wc in g:
            for wf in g:
                if wc + wf >= 0.95: continue
                b = blend_auc([1 - wc - wf, wc, wf], [0.943, aC, aF], R)
                if b > best[0]: best = (b, wc, wf)
        wC_only, lC, _ = best2(0.943, aC, rPC); wF_only, lF, _ = best2(0.943, aF, rPF)
        print("C %.3f/rho%.2f F %.3f/rho%.2f rhoCF %.2f: best wC %.2f wF %.2f lift %+.4f | C-only %+.4f (W%.2f) F-only %+.4f (W%.2f)"
              % (aC, rPC, aF, rPF, rCF, best[1], best[2], best[0] - 0.943, lC, wC_only, lF, wF_only))

# ---------------- B) simulation ----------------
prev = np.array([.204, .147, .394, .151, .359, .257, .450, .593, .126, .247, .168, .068])
A_P = np.array([.985, .95, .93, .92, .955, .95, .93, .945, .905, .96, .94, .93]); A_P = A_P - (A_P.mean() - 0.943)
dP = np.sqrt(2) * norm.ppf(A_P); N, NPUB = 1300, 390
def kscale(t):
    lo, hi = 0.1, 1.5
    for _ in range(40):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if norm.cdf(k * dP / np.sqrt(2)).mean() < t else (lo, k)
    return (lo + hi) / 2
def macro(S, Y):
    r = rankdata(S, axis=0); npos = Y.sum(0); nneg = len(Y) - npos
    return (((r * Y).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)).mean()
WG = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
READ4 = [0.20, 0.30, 0.40, 0.50]
def sim(nw, seed=11):
    rng = np.random.default_rng(seed); out = []
    for t in range(nw):
        a2 = rng.uniform(0.915, 0.940); rho = rng.uniform(0.72, 0.88); p0 = rng.uniform(-0.0005, 0.0005)
        Y = (rng.random((N, 12)) < prev).astype(float)
        eP = rng.standard_normal((N, 12)); z = rng.standard_normal((N, 12))
        SPm = dP * Y + eP; SF = kscale(a2) * dP * Y + rho * eP + np.sqrt(1 - rho ** 2) * z
        rP = rankdata(SPm, axis=0) / N; rF = rankdata(SF, axis=0) / N
        idx = rng.permutation(N); pu, pr = idx[:NPUB], idx[NPUB:]
        b0 = (macro(SPm[pu], Y[pu]), macro(SPm[pr], Y[pr]))
        pub = {}; prv = {}
        for w in WG:
            B = (1 - w) * rP + w * rF; pub[w] = macro(B[pu], Y[pu]) - b0[0]; prv[w] = macro(B[pr], Y[pr]) - b0[1]
        solo_pub = macro(SF[pu], Y[pu])
        disp = {w: np.round(0.943 + p0 + pub[w], 3) for w in WG}; solo_disp = np.round(solo_pub, 3)
        out.append((a2, rho, p0, pub, prv, disp, solo_disp))
        if (t + 1) % 100 == 0: print("world", t + 1, flush=True)
    return out
def evaluate(worlds):
    pols = {}
    def add(name, w, prv): pols.setdefault(name, []).append(prv[w])
    for (a2, rho, p0, pub, prv, disp, solo) in worlds:
        add("oracle best private W", max(WG, key=lambda w: prv[w]), prv)
        add("fixed W .30, no read", 0.30, prv)
        add("fixed W .25, no read", 0.25, prv)
        # solo read + offline rho (noise sd .03) -> closed-form best W, clipped to grid
        rho_off = rho + np.random.default_rng(int(a2 * 1e6)).normal(0, 0.03)
        w_cf, _, _ = best2(0.943, min(max(solo, 0.90), 0.95), min(max(rho_off, 0.6), 0.95))
        w_cf = min(WG, key=lambda w: abs(w - w_cf)); add("solo read + offline rho -> model W (2 slots)", w_cf, prv)
        # argmax displayed public over 4 reads; ties -> the lower W
        wa = sorted(READ4, key=lambda w: (-disp[w], w))[0]; add("argmax displayed over W .2/.3/.4/.5 (4 slots)", wa, prv)
        # model fit: grid search (a2', rho') minimising squared error of displayed deltas (+solo) -> model best W
        best = (1e9, 0.93, 0.8)
        dd = np.array([disp[w] - 0.943 for w in READ4])
        for ag in np.arange(0.905, 0.9455, 0.0025):
            for rg in np.arange(0.66, 0.95, 0.02):
                _, _, b = best2(0.943, ag, rg, grid=np.array(READ4))
                err = (((dd - (b - 0.943)) / 0.0009) ** 2).sum() + ((solo - ag) / 0.006) ** 2
                if err < best[0]: best = (err, ag, rg)
        wb, _, _ = best2(0.943, best[1], best[2])
        wm = min(WG, key=lambda w: abs(w - wb)); add("fit (a2,rho) to 4 W reads + solo -> model W (5 slots)", wm, prv)
        # gate: only blend if the W.30 read is not below parent display; else W 0 (parent, = 0 private delta)
        add("W.30 read, keep only if displayed >= .943 else drop (1 slot)", 0.30, prv) if disp[0.30] >= 0.943 else pols.setdefault("W.30 read, keep only if displayed >= .943 else drop (1 slot)", []).append(0.0)
    print("\n== B: expected PRIVATE delta of the chosen W (fleet solo U(.915,.940), rho U(.72,.88)), n=%d ==" % len(worlds))
    for k, v in pols.items():
        v = np.array(v); print("%-66s mean %+.5f  p10 %+.5f  P(>0) %.2f" % (k, v.mean(), np.percentile(v, 10), (v > 0).mean()))
    # how well does one W.30 read rank worlds? corr(pub,priv) of the delta at W.30
    pu = np.array([w[3][0.30] for w in worlds]); pr = np.array([w[4][0.30] for w in worlds])
    print("W.30: corr(public delta, private delta) %.2f | sd public %.5f private %.5f | sd(pub-priv) %.5f"
          % (np.corrcoef(pu, pr)[0, 1], pu.std(), pr.std(), (pu - pr).std()))
if __name__ == "__main__":
    tables()
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if nw: evaluate(sim(nw))
