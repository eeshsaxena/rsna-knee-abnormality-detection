# noisecorr_g58.py: within-class (noise) rank correlation of each member with the c4 output on gold58, then a binormal
# MODEL of the public gain from adding the member at weight w at the end, using PUBLIC solo AUCs (CLAIMED notebook pages).
import sys, json, numpy as np
from scipy.stats import spearmanr, norm
exec(open('div_g58.py').read().split('base = c4()')[0])
base = c4()
def within(X):
    v = []
    for j in range(12):
        for cls in (0, 1):
            m = Y[:, j] == cls
            if m.sum() >= 8: v.append((spearmanr(X[m, j], base[m, j])[0], m.sum()))
    r = np.array(v); return float(np.average(r[:, 0], weights=r[:, 1]))
rw = {k: within(X) for k, X in M.items()}
for k, v in rw.items(): print("%-26s within-class rho vs c4 %.3f" % (k, v))
# binormal: AUC = Phi(d / sqrt 2); blend of standardised scores
def gain(aE, aX, rho, w):
    dE, dX = np.sqrt(2) * norm.ppf(aE), np.sqrt(2) * norm.ppf(aX)
    d = ((1 - w) * dE + w * dX) / np.sqrt((1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * rho)
    return norm.cdf(d / np.sqrt(2)) - aE
aE = .9588
print("\nMODEL public gain (binormal, c4 public .9588, macro treated as one label):")
for name, aX in [("Dread maxspan .928", .928), ("MaViT .935", .935), ("GP reader .929", .929), ("S6-class .950", .950), ("weak .91", .91)]:
    for rho in (.6, .7, .8):
        print("  %-20s rho %.1f  w.05 %+.5f  w.10 %+.5f  w.15 %+.5f  w.20 %+.5f" % (name, rho, *[gain(aE, aX, rho, w) for w in (.05, .1, .15, .2)]))
json.dump(rw, open('noisecorr_g58.json', 'w'), indent=1)
