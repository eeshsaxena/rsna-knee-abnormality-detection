"""b_cal2.py: r_PG from the goodpjw2008 P+G weight sweep (MEASURED third party); check W-curve shape; PSD bounds on r_FN
and r_FG given r_PF, r_PN, r_GN, r_PG. Laptop, light."""
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def b2(a, b, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(a) + W * d(b)) / np.sqrt(2 * v))
RG = np.linspace(.6, .9999, 8000)
reads = [(.15, .944, .944), (.30, .9435, .9445), (.45, .942, .942)]   # (W, lo display, hi display); W.30 = .944,.944,.943
ok = np.ones_like(RG, bool)
for W, lo, hi in reads:
    okW = np.zeros_like(RG, bool)
    for P in np.linspace(.9425, .94349, 7):
        for G in np.linspace(.9285, .92949, 7):
            m = b2(P, G, RG, W)
            okW |= (m >= lo - .0005) & (m < hi + .0005)
    ok &= okW
    print(f'W {W}: r consistent {RG[okW].min():.3f}-{RG[okW].max():.3f}' if okW.any() else f'W {W}: none')
print(f'JOINT r_PG {RG[ok].min():.3f}-{RG[ok].max():.3f}' if ok.any() else 'JOINT none')
for r in (.86, .88, .90, .92):
    print(f'  r_PG {r}: W curve ' + ' '.join(f'{W:.2f}:{b2(.943, .929, r, W):.4f}' for W in (0, .1, .15, .2, .3, .45, .6)))
# N+G check with per-label avg .162 for r_GN .80-.88
for r in (.80, .82, .84, .86, .88, .90):
    print(f'  r_GN {r}: N+G W curve ' + ' '.join(f'{W:.2f}:{b2(.949, .929, r, W):.4f}' for W in (0, .1, .162, .25, .3)))
# PSD bounds for r_FN given r_PF .945, r_PN .945 and for r_FG given r_PF, r_PG, r_GN, r_FN
def psd(R): return np.linalg.eigvalsh(R).min() >= -1e-9
for rPG in (.84, .88):
    for rFN in (.86, .90, .94):
        fg = [rFG for rFG in np.linspace(.5, .999, 500)
              if psd(np.array([[1, .945, .945, rPG], [.945, 1, rFN, rFG], [.945, rFN, 1, .838], [rPG, rFG, .838, 1]]))]
        print(f'PSD r_PG {rPG} r_FN {rFN}: r_FG feasible {min(fg):.3f}-{max(fg):.3f}' if fg else f'PSD r_PG {rPG} r_FN {rFN}: none')
