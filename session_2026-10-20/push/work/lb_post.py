# lb_post.py: MODEL posterior for the unrounded public scores of c4 / eoa10 / eoa20 from in-band positions (counts MEASURED by
# lbpos.py / frozen_multi.py). Teams near the .959 boundary are a Poisson process with log-linear density
# lambda(x) = A exp(-k x) above the .959 floor (x in band units, x = 0 at the floor) and A exp(k' |x|) below it (k' >= k).
import numpy as np, json
rng = np.random.default_rng(7); B = 200000
# counts: strip = teams in .958 above c4 (04:17); a/b/c = other .959 teams below eoa10 / between / above eoa20 (05:52-53);
# n960 = .960 band (05:53)
RUL = {'ALL':   dict(strip=23, a=29, b=32, c=4, n960=44),
       'F17':   dict(strip=4,  a=7,  b=7,  c=2, n960=6),
       'F18':   dict(strip=7,  a=10, b=11, c=3, n960=11),
       'F19a':  dict(strip=13, a=15, b=20, c=3, n960=27),
       'F19b':  dict(strip=18, a=23, b=26, c=3, n960=34)}
def run(r, kmode='counts', kpr=(0.0, 1.5)):
    n959 = r['a'] + r['b'] + r['c']
    m959 = rng.gamma(n959 + 1, size=B); m960 = rng.gamma(r['n960'] + 1, size=B)
    if kmode == 'counts': k = np.log(m959 / m960)
    elif kmode == 'flat': k = np.full(B, 1e-6)
    else: k = rng.uniform(*kmode, size=B)
    k = np.maximum(k, 1e-6)
    D = rng.dirichlet([r['a'] + 1, r['b'] + 1, r['c'] + 1], size=B); F10 = D[:, 0]; F20 = D[:, 0] + D[:, 1]
    inv = lambda F: -np.log(1 - F * (1 - np.exp(-k))) / k
    x10 = inv(F10); x20 = inv(F20)
    lam0 = m959 * k / (1 - np.exp(-k))
    kp = k + rng.uniform(*kpr, size=B) if kmode != 'flat' else np.full(B, 1e-6)
    ms = rng.gamma(r['strip'] + 1, size=B) if r['strip'] > 0 else rng.gamma(1, size=B) * 0 + rng.gamma(1, size=B)
    w = np.log1p(kp * ms / lam0) / kp
    return dict(w=w, x10=x10, x20=x20, G10=w + x10, G20=w + x20, D2010=x20 - x10, gap960=1 - x20)
q = lambda v: '%.2f [%.2f, %.2f]' % (np.median(v), np.quantile(v, .1), np.quantile(v, .9))
OUT = {}
for nm, r in RUL.items():
    for km in ('counts', 'flat', (0.2, 1.2)):
        s = run(r, km); tag = nm + ' k=' + (km if isinstance(km, str) else 'U(.2,1.2)')
        OUT[tag] = {k2: [float(np.median(v)), float(np.quantile(v, .1)), float(np.quantile(v, .9))] for k2, v in s.items()}
        print('%-18s 1-u_c4 %s | x10 %s | x20 %s | G10 %s | G20 %s | G20-G10 %s | to .960 %s  (band units = 1e-3)' % (
            tag, q(s['w']), q(s['x10']), q(s['x20']), q(s['G10']), q(s['G20']), q(s['D2010']), q(s['gap960'])))
# pooled (ALL ruler counts-k, F18 counts-k, F19b counts-k): equal-weight mixture
mix = {k2: np.concatenate([run(RUL[n], 'counts')[k2][:B // 3] for n in ('ALL', 'F18', 'F19b')]) for k2 in ('w', 'x10', 'x20', 'G10', 'G20', 'D2010', 'gap960')}
print('POOLED            ', ' | '.join('%s %s' % (k2, q(v)) for k2, v in mix.items()))
np.savez('lb_post_pooled.npz', **mix)
u = 1 - mix['w']
print('absolute (truncation display): c4 %s | eoa10 %s | eoa20 %s' % tuple(q(.958 * 1e3 + v) for v in (u, 1 + mix['x10'], 1 + mix['x20'])))
print('absolute (rounding display):   c4 %s | eoa10 %s | eoa20 %s' % tuple(q(.9575 * 1e3 + v) for v in (u, 1 + mix['x10'], 1 + mix['x20'])))
json.dump(OUT, open('lb_post.json', 'w'), indent=1)
