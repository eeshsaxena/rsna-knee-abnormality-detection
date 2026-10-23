# Adversarial review numbers for BIG_JUMP_PLAN.md (closed-form binormal, MODEL only; laptop CPU, seconds).
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0)
def mu(a): return np.sqrt(2)*norm.ppf(a)
def nblend(aucs, ws, R):
    aucs = np.array(aucs, float); ws = np.array(ws, float)/np.sum(ws)
    return norm.cdf((ws@mu(aucs))/np.sqrt(2*(ws@R@ws)))
def blend(p, a, r, w): return nblend([p, a], [1-w, w], np.array([[1, r], [r, 1]]))
def best(p, a, r):
    ws = np.linspace(0, 1, 101); b = [blend(p, a, r, w) for w in ws]; i = int(np.argmax(b)); return ws[i], b[i]
def fams(groups, within, cross):
    aucs = []; ws = []; fam = []
    for gi, (a, n, w) in enumerate(groups):
        aucs += [a]*n; ws += [w/n]*n; fam += [gi]*n
    N = len(aucs); R = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            R[i, j] = 1 if i == j else (within[fam[i]] if fam[i] == fam[j] else cross)
    return nblend(aucs, ws, R)

print("1) Sanity: does the model reproduce aastik? 6 ConvNeXt-T 320x24 raw mean = 0.927 LB (MEASURED).")
for single in (0.918, 0.920, 0.922, 0.925):
    print(f"   single {single}: 6 members within .93 -> {fams([(single,6,1)],[0.93],0.9):.4f}; within .95 -> {fams([(single,6,1)],[0.95],0.9):.4f}")
print()
TTA = 0.0015
scen = {
 "PLAN KILL    (F1 .929 x6, EV2S .926 x3)": fams([(0.929,6,0.65),(0.926,3,0.35)],[0.93,0.95],0.90)+TTA,
 "ANALOG KILL  (F1 .922 x6, EV2S .917 x3)": fams([(0.922,6,0.65),(0.917,3,0.35)],[0.93,0.95],0.90)+TTA,
 "ANALOG KILL, no EV2S (F1 .922 x6)":       fams([(0.922,6,1.0)],[0.93],0.90)+TTA,
 "PLAN PASS-c  (TTR .934 x5, F1 .929 x4, EV2S .926 x3)": fams([(0.934,5,0.55),(0.929,4,0.25),(0.926,3,0.20)],[0.95,0.93,0.95],0.90)+TTA,
 "ANALOG PASS-c (TTR .928 x5, F1 .922 x4, EV2S .917 x3)": fams([(0.928,5,0.55),(0.922,4,0.25),(0.917,3,0.20)],[0.95,0.93,0.95],0.90)+TTA,
 "ANALOG PASS-up (TTR .932 x5, F1 .922 x4, EV2S .917 x3, cross .88)": fams([(0.932,5,0.6),(0.922,4,0.2),(0.917,3,0.2)],[0.95,0.93,0.95],0.88)+TTA,
}
print("2) Own-ensemble solo and best-W blend on 0.943 at rho_eff .80/.84/.88/.90")
for k, own in scen.items():
    row = []
    for r in (0.80, 0.84, 0.88, 0.90):
        w, b = best(0.943, own, r); row.append(f"{r:.2f}:{b:.4f}")
    print(f"   {k:62s} own {own:.4f} | " + " ".join(row))
print()
print("3) Monte Carlo P(displayed >= 0.950), i.e. public >= 0.9495 after read noise.")
print("   own ~ N(centre, sd), rho_eff ~ U(lo, hi), blend read noise SD .0012 (ENDGAME), best W chosen with known rho (optimistic).")
def p950(centre, sd, lo, hi, n=4000):
    own = rng.normal(centre, sd, n); r = rng.uniform(lo, hi, n); e = rng.normal(0, 0.0012, n)
    out = np.array([best(0.943, min(o, 0.975), rr)[1] for o, rr in zip(own, r)]) + e
    return (out >= 0.9495).mean(), np.median(out)
for name, c in (("PLAN KILL", scen["PLAN KILL    (F1 .929 x6, EV2S .926 x3)"]),
                ("PLAN PASS-c", scen["PLAN PASS-c  (TTR .934 x5, F1 .929 x4, EV2S .926 x3)"]),
                ("ANALOG KILL", scen["ANALOG KILL  (F1 .922 x6, EV2S .917 x3)"]),
                ("ANALOG PASS-c", scen["ANALOG PASS-c (TTR .928 x5, F1 .922 x4, EV2S .917 x3)"])):
    for lo, hi in ((0.80, 0.88), (0.84, 0.92)):
        p, m = p950(c, 0.005, lo, hi)
        print(f"   {name:14s} own {c:.4f} sd .005, rho U({lo},{hi}): P(>=.950) {p:.2f}, median public {m:.4f}")
print()
print("4) Own solo needed for displayed .950 at each rho_eff (plan table check)")
for r in (0.80, 0.84, 0.88, 0.90, 0.92):
    for a in np.arange(0.925, 0.975, 0.0005):
        if best(0.943, a, r)[1] >= 0.9495: print(f"   rho {r:.2f}: own >= {a:.4f}"); break
print()
print("5) Tian MEASURED check: c03 pair .932 + R34 family .921 at rho .86 read .932 (no add).")
for a2, rc in ((0.921, 0.86), (0.917, 0.90), (0.926, 0.90)):
    base = 0.932
    b = max(nblend([base, a2], [1-w, w], np.array([[1, rc], [rc, 1]])) for w in np.linspace(0, 0.5, 51))
    print(f"   main .932 + 2nd family {a2} at cross {rc}: best {b:.4f} (+{b-base:.4f})")
