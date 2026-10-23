# abc_update.py (10-20 push): READ-ONLY post-read update of the public E-OA curve after the A21 reads, for choosing 10-21 entries.
# Inputs come from a public LB API snapshot (fetch_api_lb.py, read-only) + our own submissions list (3-decimal displays).
# Usage (example):
#   python abc_update.py --snap ../../../session_2026-10-14/research960/lbprobe/api_lb_1020_a21read.json --best pw30 \
#          --disp eoa30=0.960,pw20=0.959,pw30=0.960
# --best = which entry the API best-id maps to (eoa20 / eoa30 / pw20 / pw30); --disp = display score of each A21 entry.
# Optional --rank R --band_lo L --band_n N to override the snapshot (testing).
# Method: our best's in-band position x_new (exp density, Dirichlet on others above / below, as lb_post.py) is turned into
# delta = x_new + band offset - x20 (x20 from lb_post_pooled.npz, the same fixed .959 floor), then abc_local2.run sims are
# re-weighted by N(inc_best; delta, sd), argmax consistency and the display constraints. Prints posterior predictive for the
# untested shapes (S25, S35, PW25, PW35, PW40, PW50) vs the new best, P(shows .960 / .961) and population (private) increments.
import argparse, json, collections, numpy as np, abc_local2 as A
ap = argparse.ArgumentParser(); ap.add_argument('--snap'); ap.add_argument('--best', required=True); ap.add_argument('--disp', default='')
ap.add_argument('--rank', type=int); ap.add_argument('--band_lo', type=int); ap.add_argument('--band_n', type=int); ap.add_argument('--band', default=None)
ap.add_argument('--prox', default='PFT'); a = ap.parse_args()
ME = 16799346; MAP = {'eoa20': 'S20', 'eoa30': 'S30', 'pw20': 'PW20', 'pw30': 'PW30'}
if a.rank is None:
    L = sorted(json.load(open(a.snap))['lb'], key=lambda r: r['rank']); me = [r for r in L if r['teamId'] == ME][0]
    band = [r for r in L if r['displayScore'] == me['displayScore']]; rank, lo, n, disp = me['rank'], band[0]['rank'], len(band), me['displayScore']
    nxt = [r for r in L if r['displayScore'] == '%.3f' % (float(disp) + .001)]; n_up = len(nxt)
else:
    rank, lo, n, disp, n_up = a.rank, a.band_lo, a.band_n, a.band, int(round(a.band_n * .67))
above = rank - lo; below = n - 1 - above; off = {'0.958': -1, '0.959': 0, '0.960': 1, '0.961': 2}[disp]
rng = np.random.default_rng(4); B = 100000
k = np.maximum(np.log(rng.gamma(n, size=B) / rng.gamma(max(n_up, 1) + 1, size=B)), 1e-3)
F = rng.beta(below + 1, above + 1, size=B); xnew = off - np.log(1 - F * (1 - np.exp(-k))) / k
LB = np.load('lb_post_pooled.npz'); x20 = rng.choice(LB['x20'], B); dl = (xnew - x20) * 1e-3
print('best %s at r%d (%s band r%d, n %d, above %d): x_new %.2f [%.2f, %.2f] | delta vs eoa20 %+.2fe-4 [%+.2f, %+.2f]' % (
    a.best, rank, disp, lo, n, above, np.median(xnew), np.quantile(xnew, .1), np.quantile(xnew, .9), np.median(dl) * 1e4, np.quantile(dl, .1) * 1e4, np.quantile(dl, .9) * 1e4))
o = A.run(a.prox, 'D5', 'N01', 'logN', quiet=True); r = o['_arr']; G, w, gap, iv = r['G'], r['w'].copy(), r['gap'], r['iv']
inc = {e: G[:, iv[v]] - G[:, iv['S20']] for e, v in MAP.items()}
mu, sd = np.mean(dl), np.std(dl) + 2e-5
if a.best != 'eoa20':
    w *= np.exp(-.5 * ((inc[a.best] - mu) / sd) ** 2)
stack = np.stack([inc[e] for e in MAP], 1); w *= (np.argmax(stack, 1) == list(MAP).index(a.best))
x20s = 1 - gap * 1e3
for kv in [x for x in a.disp.split(',') if x]:
    e, d_ = kv.split('='); xe = x20s + inc[e] * 1e3; band_e = np.floor(xe).astype(int)
    w *= (band_e == {'0.958': -1, '0.959': 0, '0.960': 1, '0.961': 2}[d_])
t = w.sum(); print('effective weight %.3g (ESS %.0f)' % (t, t ** 2 / (w ** 2).sum()) if t > 0 else 'NO SIM CONSISTENT WITH THE READ: model rejected, re-check inputs')
if t > 0:
    best_inc = inc[a.best]
    for v in ('S25', 'S30', 'S35', 'S40', 'PW20', 'PW25', 'PW30', 'PW35', 'PW40', 'PW50'):
        dv = G[:, iv[v]] - G[:, iv['S20']] - best_inc; xe = x20s + (G[:, iv[v]] - G[:, iv['S20']]) * 1e3
        P = lambda m: float((w * m).sum() / t)
        print('   %-5s vs best: public %+.2fe-4 | P(> best) %.2f | P(shows .960+) %.2f | P(shows .961) %.2f' % (
            v, P(dv) * 1e4, P(dv > 0), P(xe >= 1), P(xe >= 2)))
    Pop = r['Pop']
    for v in ('S20', 'S25', 'S30', 'PW20', 'PW25', 'PW30', 'PW35'):
        print('   POP (private-relevant) %-5s vs c4 %+.2fe-4 | vs eoa20 %+.2fe-4' % (v, (w * Pop[:, iv[v]]).sum() / t * 1e4, (w * (Pop[:, iv[v]] - Pop[:, iv['S20']])).sum() / t * 1e4))
