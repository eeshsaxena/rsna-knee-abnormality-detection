# CPU-only. (a) combo target: own-OOF fill on UNK+NO + Raptor 0.5 on YES; (b) binormal scenario table at effective rho.
import io, sys, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
exec(open(__file__.replace('labels_combo.py', 'labels_ownfill.py'), encoding='utf-8').read().split("print('own teacher")[0])
def combo(a_own=0.75, a_rap=0.5):
    out = lg(v4).copy(); Mo = match(own); Mr = match(rap)
    for l in LAB:
        m = V[l].isin(['UNK', 'NO']); out.loc[m, l] = (1 - a_own) * lg(v4)[l][m] + a_own * Mo[l][m]
        y = V[l] == 'YES'; out.loc[y, l] = (1 - a_rap) * lg(v4)[l][y] + a_rap * Mr[l][y]
    return out
for ao, ar in ((0.75, 0.25), (0.75, 0.5), (1.0, 0.5)):
    P = combo(ao, ar); print(f'combo own UNK+NO a={ao} + raptor YES a={ar}: gold58 {mauc(P):.4f} wc-rho rap {wc_rho(P, rap):.3f}')
P = combo(0.75, 0.0); print(f'own UNK+NO a=.75 only (check): {mauc(P):.4f} wc-rho {wc_rho(P, rap):.3f}')
from scipy.stats import norm
mu = lambda a: np.sqrt(2) * norm.ppf(a)
def best(a1, a2, rho):
    ws = np.linspace(0, 1, 101); m = (1 - ws) * mu(a1) + ws * mu(a2)
    v = (1 - ws) ** 2 + ws ** 2 + 2 * ws * (1 - ws) * rho; b = norm.cdf(m / np.sqrt(2 * v)); i = b.argmax(); return ws[i], b[i] - a1
print('\nscenario (fleet ensemble solo, student gold-rho) -> best-W lift over .943 at gold-rho / eff-rho (+0.10)')
for name, s, r in (('v4 fleet ens', .935, .78), ('mix50 fleet ens (+.009, +.06 rho)', .944, .84),
                   ('DSD own UNK+NO (+.005, +.015 rho)', .940, .795), ('DSD optimistic (+.008, +.02)', .943, .80),
                   ('DSD + seeds/aug to .945', .945, .80), ('own .950 @ .80', .950, .80)):
    w0, l0 = best(.943, s, r); w1, l1 = best(.943, s, min(r + .10, .995))
    print(f'  {name:36s} {s:.3f} @ {r:.3f}: {l0:+.4f} (W {w0:.2f}) | eff {l1:+.4f} (W {w1:.2f})')
