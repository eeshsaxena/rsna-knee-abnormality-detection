# MODEL: private-LB rank of picks A (HW) and B (R6) vs the frozen field, all margins relative to the heliosli copy block C.
# Public structure MEASURED from api_lb_1017T_lens.json: A r171, B bracket r298-322 (use r310), C block r388-995 (608 teams,
# near-identical predictions), W zone r172-387 (216 iterators, distinct scores), V r142-170 (29), bands >= .959 and .957/.956.
# Scale and noise are GUESS/MODEL parameters (see lb_private.md).
import numpy as np, sys
rng = np.random.default_rng(1)
def run(D=5e-4, sS6=4e-4, mu0=4e-4, tau0=4e-4, selA=0.5e-4, sW=4e-4, bW=0.5e-4, sT=7e-4, bT=1e-4, Lmu=1e-4, Lsd=3e-4,
        wfrac=0.64, wshrink=0.25, wtau=2e-4, Nfinal=6300, nsim=20000, K=0, tf=0.5):
    f = 390 / 910
    u = D / 216.0                                   # score per W-zone rank unit (uniform density, MODEL)
    sC = 0.0
    mA = D
    # public margins over C
    V = mA + u * np.arange(1, 30)                    # r142-170
    W = mA - u * np.arange(1, 217)                   # r172-387 (B's slot sits inside)
    top = .9585 - (D + 29 * u)                       # .9585 boundary expressed relative to C score  -> C score = .9585 - D - 29u
    sCabs = .9585 - D - 29 * u
    bands = {.959: 48, .960: 29, .961: 18, .962: 17, .963: 9, .964: 8, .965: 3, .966: 4, .967: 1, .968: 3, .969: 1}
    T = np.concatenate([np.linspace(s - .0005, s + .0005, n + 2)[1:-1] - sCabs for s, n in bands.items()])
    sigT = np.where(T + sCabs >= .9595, 1.4e-3, 1.0e-3)
    B57 = np.linspace(.9565, .9575, 225)[1:-1] - sCabs   # .957 band 223
    B56 = np.linspace(.9555, .9565, 451)[1:-1] - sCabs   # .956 band 449
    sigW = np.sqrt(1 + f) * sW
    if K:
        idx = np.random.default_rng(7).choice(W.size, K, replace=False)
        newpub = mA + u * np.random.default_rng(8).uniform(0, 29, K)
        W = W.copy(); W[idx] = W[idx] + tf * (newpub - W[idx])
    nC = 608
    silver = Nfinal // 20; bronze = Nfinal // 10; gold = 10 + Nfinal // 500
    # A's S6 margin: posterior of true S6 effect (normal prior, public obs D - selA, sd sS6), plus private sampling noise
    obs = D - selA
    pv = 1 / (1 / tau0**2 + 1 / sS6**2); pm = pv * (mu0 / tau0**2 + obs / sS6**2)
    resA = []; resB = []; resAB = []; aboveC = 0; aboveCB = 0
    for _ in range(nsim):
        tA = rng.normal(pm, np.sqrt(pv)); A = tA + rng.normal(0, np.sqrt(f) * sS6)
        # weights effect (A - B): public wfrac*D, true heavily shrunk (public-probe tuned), private sampling noise
        Wpub = wfrac * D; Wt = rng.normal(wshrink * Wpub, wtau); Bm = A - (Wt + rng.normal(0, np.sqrt(f) * 2.5e-4))
        L = rng.normal(Lmu, Lsd)                      # other-lineage teams' common private shift vs our lineage
        pV = V - bW + rng.normal(0, sigW, V.size)
        pW = W - bW + rng.normal(0, sigW, W.size)
        pT = T - bT + L + rng.normal(0, sigT)
        p57 = B57 - bW + L + rng.normal(0, 1.0e-3, B57.size)
        p56 = B56 - bW + L + rng.normal(0, 1.0e-3, B56.size)
        field = np.concatenate([pV, pW, pT, p57, p56])
        def rank(x):
            return 1 + (field > x).sum() + (nC if 0.0 > x else 0)
        rA, rB = rank(A), rank(Bm); rAB = rank(max(A, Bm))
        resA.append(rA); resB.append(rB); resAB.append(rAB); aboveC += A > 0; aboveCB += max(A, Bm) > 0
    resA, resB, resAB = map(np.array, (resA, resB, resAB))
    q = lambda r: f"silver {np.mean(r <= silver):.2f} bronze+ {np.mean(r <= bronze):.2f} gold {np.mean(r <= gold):.3f} med r{int(np.median(r))} p10-p90 r{int(np.percentile(r,10))}-{int(np.percentile(r,90))}"
    return dict(A=q(resA), B=q(resB), AB=q(resAB), PA_aboveC=aboveC / nsim, PAB_aboveC=aboveCB / nsim,
                corr=np.corrcoef(resA, resB)[0, 1])
if __name__ == "__main__" and len(sys.argv) > 1:
    for k, v in [("drift K60 tf.5", dict(K=60)), ("drift K120 tf.5", dict(K=120)), ("drift K120 tf.8", dict(K=120, tf=.8)), ("drift K120 + weights overfit", dict(K=120, wshrink=-0.3)), ("drift K120 + weights real", dict(K=120, wshrink=1.0, wtau=1e-4)), ("drift K120 + skeptic S6", dict(K=120, mu0=0, tau0=3e-4))]:
        r = run(nsim=6000, **v)
        print(f"== {k}: P(A>C) {r['PA_aboveC']:.2f} P(max>C) {r['PAB_aboveC']:.2f} corr {r['corr']:.2f}"); print("   A  :", r["A"]); print("   B  :", r["B"]); print("   A+B:", r["AB"])
elif __name__ == "__main__":
    scen = {
        "base D5e-4": {},
        "D3e-4": dict(D=3e-4), "D8e-4": dict(D=8e-4),
        "noise x1.5 (sS6 6e-4, sW 6e-4)": dict(sS6=6e-4, sW=6e-4),
        "noise x0.6 (sS6 2.5e-4, sW 2.5e-4)": dict(sS6=2.5e-4, sW=2.5e-4),
        "skeptic prior S6 (mu0 0, tau 3e-4)": dict(mu0=0, tau0=3e-4),
        "weights real (wshrink 1.0)": dict(wshrink=1.0, wtau=1e-4),
        "weights overfit (wshrink -0.3)": dict(wshrink=-0.3),
        "lineage penalty (Lmu 3e-4)": dict(Lmu=3e-4),
        "no iterator bias (bW 0, bT 0, selA 0)": dict(bW=0, bT=0, selA=0),
    }
    for k, v in scen.items():
        r = run(nsim=6000, **v)
        print(f"== {k}: P(A>C) {r['PA_aboveC']:.2f} P(max(A,B)>C) {r['PAB_aboveC']:.2f} corr(rankA,rankB) {r['corr']:.2f}")
        print("   A  :", r["A"]); print("   B  :", r["B"]); print("   A+B:", r["AB"])
