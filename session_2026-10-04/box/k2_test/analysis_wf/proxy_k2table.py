"""proxy_k2table.py : K2 conditional table under three fleet r_LB regimes (box, CPU, tiny).
Regimes (effective r of the fleet member vs the parent on the public LB):
  LOW  .90 = gold58 "no transfer" (fleet r_fit vs the OOS parent sides .877-.918; parent's DINO half less fleet-like)
  MID  .94 = between, and >= the crop members' LB r (.88-.94) since the fleet is more parent-like than crop on gold58
  HIGH .97 = crop-anchored ratio transfer (proxy_lb.log section C, ratio form median)
"""
import os, json, numpy as np
from scipy.stats import norm
H = os.path.expanduser
L = json.load(open(H('~/k2_test/analysis_wf/proxy_lb_out.json')))
ratio = [r['fleet_c'] for r in L['fleet_rLB']['rows'] if r['form'] == 'ratio']
print(f'crop-anchored ratio transfer, central values: median {np.median(ratio):.3f}, min {min(ratio):.3f}, max {max(ratio):.3f}')


def dv(a): return np.sqrt(2) * norm.ppf(a)


def B(P, a, r, w):
    v = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * r
    return norm.cdf(((1 - w) * dv(P) + w * dv(a)) / np.sqrt(2 * v))


WG = np.linspace(0, 1, 201); P0 = 0.943


def best(a, r): b = B(P0, a, r, WG); i = int(np.argmax(b)); return float(WG[i]), float(b[i] - P0)


def ens(s, r_mp, k, r_mm):
    v = (1 + (k - 1) * r_mm) / k
    return float(norm.cdf(dv(s) / np.sqrt(2 * v))), float(min(0.999, r_mp / np.sqrt(v)))


SG = np.arange(0.900, 0.9801, 0.0005)


def need(r, tgt):
    for s in SG:
        if best(s, r)[1] >= tgt: return float(s)
    return float('nan')


out = []
for s in (0.905, 0.915, 0.925, 0.935):
    print(f'\nK2 SOLO {s:.3f}')
    for nm, r in (('LOW', 0.90), ('MID', 0.94), ('HIGH', 0.97)):
        b30 = float(B(P0, s, r, 0.30)); w, g = best(s, r)
        e10 = ens(s, r, 10, 0.95); w10, g10 = best(*e10)
        e10b = ens(s, r, 10, 0.93); w10b, g10b = best(*e10b)
        n = [need(e10[1], t) for t in (0.003, 0.005, 0.007)]
        out.append(dict(s=s, regime=nm, r=r, blend30=b30, W=w, lift=g, e10=e10, lift10=(w10, g10), e10b=e10b, lift10b=(w10b, g10b), need_at_r10=n))
        print(f'  {nm:4s} r {r:.2f}: BLEND W.30 {b30:.4f} (shows {round(b30 + 1e-9, 3):.3f}) | W* {w:.2f} lift {g:+.4f}'
              f' | 10 members (r_mm .95): solo {e10[0]:.4f} r {e10[1]:.3f} W* {w10:.2f} lift {g10:+.4f}'
              f' | (r_mm .93: {e10b[0]:.4f}, r {e10b[1]:.3f}, lift {g10b:+.4f})'
              f' | ensemble solo needed at r {e10[1]:.3f}: +.003 {n[0]:.4f} +.005 {n[1]:.4f} +.007 {n[2]:.4f}')
# max single-member r that still gives >= 0 at W .30, per solo
print('\nbreak-even r at W .30 (blend read = parent) and r needed for +.001 at W*:')
RG = np.linspace(0.5, 0.999, 1000)
for s in (0.905, 0.915, 0.925, 0.935):
    be = RG[np.where(B(P0, s, RG, 0.30) >= P0)[0]]
    r1 = [r for r in RG if best(s, r)[1] >= 0.001]
    print(f'  solo {s:.3f}: W.30 >= parent needs r <= {be.max() if len(be) else float("nan"):.3f}; +.001 at W* needs r <= {max(r1) if r1 else float("nan"):.3f}')
json.dump(out, open(H('~/k2_test/analysis_wf/proxy_k2table_out.json'), 'w'), indent=1)
