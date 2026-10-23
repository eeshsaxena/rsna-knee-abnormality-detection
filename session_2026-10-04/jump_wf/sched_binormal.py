# closed-form equal-variance binormal blend: parent A_p, member S, within-class score correlation rho, weight w
from statistics import NormalDist
N = NormalDist()
def d(a): return 2 ** 0.5 * N.inv_cdf(a)
def blend(ap, s, rho, w):
    num = (1 - w) * d(ap) + w * d(s)
    den = ((1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * rho) ** 0.5
    return N.cdf(num / den / 2 ** 0.5)
AP = 0.943
print('lift over .943 at W .30 | best W (lift)')
for rho in (0.80, 0.84, 0.88, 0.92):
    row = []
    for s in (0.915, 0.920, 0.925, 0.930, 0.935, 0.940):
        l30 = blend(AP, s, rho, 0.30) - AP
        bw = max((blend(AP, s, rho, w / 100) - AP, w / 100) for w in range(0, 71))
        row.append(f'S{s:.3f}: {l30:+.4f} | W*{bw[1]:.2f} {bw[0]:+.4f}')
    print(f'rho {rho:.2f}  ' + '  ;  '.join(row))
# ensemble strength from k members of solo s1 with pairwise rho_m (equal-variance binormal, equal weights)
print('\nensemble of k equal members, solo s1, pairwise rho_m')
for s1 in (0.915, 0.920):
    for rm in (0.90, 0.93, 0.95):
        out = []
        for k in (1, 2, 3, 5, 8):
            dk = d(s1) * k / (k + k * (k - 1) * rm) ** 0.5
            out.append(f'k{k} {N.cdf(dk / 2 ** 0.5):.4f}')
        print(f's1 {s1:.3f} rho_m {rm:.2f}: ' + ' '.join(out))

# K2 inversion: displayed blend read at W .30 given solo S1 -> implied rho_eff (bisection on the same model)
def implied_rho(s1, read, w=0.30):
    lo, hi = 0.0, 0.999
    f = lambda r: blend(AP, s1, r, w) - read
    if f(lo) < 0 or f(hi) > 0:
        return float('nan')
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if f(m) > 0 else (lo, m)
    return (lo + hi) / 2
print('\nK2 inversion: implied rho_eff for (solo S1, blend read at W .30); read = displayed value (+-.0005 rounding)')
for s1 in (0.905, 0.910, 0.915, 0.920, 0.925, 0.930):
    print(f'S1 {s1:.3f}: ' + '  '.join(f'{rd:.3f}->{implied_rho(s1, rd):.3f}' for rd in (0.939, 0.940, 0.941, 0.942, 0.943, 0.944, 0.945)))
