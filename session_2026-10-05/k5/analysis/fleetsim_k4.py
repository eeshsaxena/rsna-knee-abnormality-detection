"""fleetsim_k4.py (k4, 2026-10-05): the prior and rejection block (fleetsim.py lines 7-83) is a copy of jump2/fleetsim.py (same priors, same K2+K3 rejection,
same seed); the tail below asks the K4 interim question. MODEL on MEASURED reads; every prior is tagged in the copied part.
Writes fleetsim_k4.out next to this file. Light CPU (~200k draws)."""
import sys, numpy as np
from scipy.stats import norm
rng = np.random.default_rng(20261005)
PSD = (sys.argv[1] if len(sys.argv) > 1 else 'psd') == 'psd'   # 'nopsd' skips the full-matrix PSD rejection
N0 = 200000 if PSD else 60000

def d(a): return np.sqrt(2) * norm.ppf(a)
def A(dd): return norm.cdf(dd / np.sqrt(2))
def blend(P, s, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(P) + W * d(s)) / np.sqrt(2 * v))
U = lambda a, b: rng.uniform(a, b, N0)
Nn = lambda m, s: rng.normal(m, s, N0)

# ---------------- priors ----------------
P = U(.9425, .9435)                         # parent displayed .943 (MEASURED)
s = {}; r = {}
s['F1m'] = U(.9315, .9325)                  # K2 SOLO .932 (MEASURED)
r['F1m'] = U(.85, .99)                      # flat prior, K2 BLEND rejects below
delta = r['F1m'] - .911                      # OOF reff vs raptorproxy -> LB offset (F1m OOF reff .911 MEASURED oofx)
s['T1v'] = U(.9225, .9235)                  # K3 SOLO TTR v4 .923 (MEASURED)
# OOF effective r vs the parent's Raptor proxy (MEASURED oofx.log): F1m .911, TTRmix50 .916, TTRv4 .868, F1v4 .857, gated .887
r['Tm'] = .916 + delta + Nn(0, .010)         # mapping error GUESS sd .010 (same labels as anchor)
r['T1v'] = .868 + delta + Nn(0, .015)        # GUESS sd .015 (different labels from anchor)
r['F1v'] = .857 + delta + Nn(0, .015)
r['F1g'] = .887 + delta + Nn(0, .012)
# solos not yet read: GUESS from OOF + gold58 (see blendmath.md sec 3)
s['Tm'] = Nn(.9325, .003)                   # TTR mix50 fold model: OOF .8836 vs F1m .8811, gold58 .913 vs .914
s['F1v'] = Nn(.917, .004)                   # F1 v4: OOF .8692 vs TTR v4 .8764 -> ~.006 below TTR v4 on LB
s['F1g'] = Nn(.924, .004)                   # gated: OOF between F1v4 and F1m
# second backbone family on v4 labels (EffNetV2-S / ResNet-50), only for 10-12 options: GUESS
s['F2'] = Nn(.921, .005)
r['F2'] = .868 + delta - .02 + Nn(0, .025)
# own-OOF silent-fill v4 labels (Archit/Raymond style, teacher = OUR 5-fold OOF, not Dread): GUESS
s['Tv2'] = Nn(.927, .004)
r['Tv2'] = .868 + delta + .015 + Nn(0, .02)
# 'what we need' reference member: GUESS-free target point (fixed)
s['H'] = np.full(N0, .930); r['H'] = np.full(N0, .86)
# within-type fold/seed r_mm: GUESS U[.94,.965] (bounded by cross-recipe same-label .948 and raw/ema .995 MEASURED; Tian seeds .941)
rho_s = U(.94, .965)
# cross-type effective r_mm (MEASURED oofx.log reff, + GUESS noise .01)
RM = {('F1m', 'Tm'): .948, ('F1m', 'T1v'): .920, ('F1m', 'F1v'): .927, ('F1m', 'F1g'): .947, ('Tm', 'T1v'): .934,
      ('Tm', 'F1v'): .900, ('Tm', 'F1g'): .915, ('T1v', 'F1v'): .917, ('T1v', 'F1g'): .913, ('F1v', 'F1g'): .942}
rm = {k: v + Nn(0, .01) for k, v in RM.items()}
f2v = U(.88, .92)                            # GUESS family-2 vs ConvNeXt on the same v4 labels
rm[('F2', 'T1v')] = f2v; rm[('F2', 'F1v')] = f2v; rm[('F2', 'Tm')] = f2v - .02; rm[('F2', 'F1m')] = f2v - .02; rm[('F2', 'F1g')] = f2v - .01
rm[('Tv2', 'T1v')] = rm[('T1v', 'F1v')] + .02; rm[('Tv2', 'Tm')] = rm[('Tm', 'T1v')]; rm[('Tv2', 'F1m')] = rm[('F1m', 'T1v')]
rm[('Tv2', 'F1v')] = rm[('T1v', 'F1v')]; rm[('Tv2', 'F1g')] = rm[('T1v', 'F1g')]; rm[('Tv2', 'F2')] = f2v
for t in ['F1m', 'Tm', 'T1v', 'F1v', 'F1g', 'F2', 'Tv2']: rm[('H', t)] = np.full(N0, .90)
TYPES = ['F1m', 'Tm', 'T1v', 'F1v', 'F1g', 'F2', 'Tv2', 'H']
def RMM(a, b):
    if a == b: return rho_s
    return rm[(a, b)] if (a, b) in rm else rm[(b, a)]

# ---------------- rejection on the measured reads ----------------
ok = np.ones(N0, bool)
b2 = blend(P, s['F1m'], r['F1m'], .30); ok &= (b2 >= .9425) & (b2 < .9435)          # K2 BLEND W .30 = .943
v2 = (1 + RMM('F1m', 'T1v')) / 2
e2 = A((d(s['F1m']) + d(s['T1v'])) / 2 / np.sqrt(v2)); re2 = (r['F1m'] + r['T1v']) / 2 / np.sqrt(v2)
b3 = blend(P, e2, np.minimum(re2, .9999), .25); ok &= (b3 >= .9425) & (b3 < .9435) & (re2 < 1)   # K3 STRICT W .25 = .943
for t in TYPES:
    ok &= (r[t] < np.sqrt(np.minimum(rho_s, 1))) & (r[t] > 0)
# PSD of parent + one type-level block with rho_s diagonal (= the infinite-copies limit)
M = np.zeros((N0, len(TYPES) + 1, len(TYPES) + 1)); M[:, 0, 0] = 1
for i, a in enumerate(TYPES):
    M[:, 0, i + 1] = M[:, i + 1, 0] = r[a]
    for j, b in enumerate(TYPES):
        M[:, i + 1, j + 1] = RMM(a, b)
if PSD:
    ok &= np.linalg.eigvalsh(M[:, :7, :7])[:, 0] > -1e-9   # first 6 real types + parent only
idx = np.nonzero(ok)[0]
print(f'accepted {len(idx)} of {N0}')
def take(x): return x[idx] if np.ndim(x) else x
P = take(P); s = {k: take(v) for k, v in s.items()}; r = {k: take(v) for k, v in r.items()}
rho_s = take(rho_s); rm = {k: take(v) for k, v in rm.items()}; delta = take(delta)
N = len(idx)


# ======================= K4 interim tail (new) =======================
out = []
def pr(x=''):
    print(x); out.append(x)
def q(x, ps=(10, 50, 90)): return '/'.join(f'{np.percentile(x, p):.4f}' for p in ps)
def disp(x): return np.round(x - 1e-12, 3)

def fleet_k(k, m1=False):
    """k TTR mix50 fold members (+ M1 = F1 mix50): equal-weight raw mean, same algebra as fleetsim.fleet()."""
    counts = {'Tm': k}
    if m1: counts['F1m'] = 1
    n = sum(counts.values()); ts = list(counts)
    v = np.zeros(N); dd = np.zeros(N); rr = np.zeros(N)
    for a in ts:
        ca = counts[a]; dd += ca * d(s[a]); rr += ca * r[a]; v += ca * (1 + (ca - 1) * rho_s)
        for b in ts:
            if b != a: v += ca * counts[b] * RMM(a, b)
    v /= n * n; dd /= n; rr /= n
    return A(dd / np.sqrt(v)), rr / np.sqrt(v)

pr(f'posterior N={N}; TTR mix50 member solo {q(s["Tm"])}, r_LB {q(r["Tm"])}; fold-fold rho_s {q(rho_s)}; parent {q(P)}')
pr('\nset | fleet solo 10/50/90 | fleet r 10/50/90 | W | true blend 10/50/90 | true lift median | '
   'display P(<=.942) P(.943) P(.944) P(>=.945) | E[display]')
RES = {}
for k, m1 in [(1, False), (2, False), (3, False), (4, False), (5, False), (5, True)]:
    se, re = fleet_k(k, m1)
    for W in (None, .20, .50):
        if W is None:
            x = se; tag = 'SOLO'
        else:
            x = blend(P, se, re, W); tag = f'W{W:.2f}'
        ds = disp(x); RES[(k, m1, W)] = (se, re, x, ds)
        pr(f'{k}f TTRm{" + M1" if m1 else ""} | {q(se)} | {q(re)} | {tag} | {q(x)} | '
           f'{"-" if W is None else f"{np.median(x - P):+.4f}"} | {(ds <= .942).mean():.2f} {np.isclose(ds, .943).mean():.2f} '
           f'{np.isclose(ds, .944).mean():.2f} {(ds >= .945).mean():.2f} | {ds.mean():.4f}')

# value of an INTERIM read (2 folds, today) for the FINAL K4 (5 folds, 10-06): how much does seeing the interim
# display shrink the uncertainty of the final W .20 true lift and of the TTR mix50 member (s, r)?
pr('\nInterim read -> what it tells about the final 5-fold K4 (posterior sd prior -> after observing the interim display)')
fin = RES[(5, False, .20)][2] - P
for name, key in [('interim 2f SOLO', (2, False, None)), ('interim 2f W .20', (2, False, .20)),
                  ('interim 2f W .50', (2, False, .50)), ('final 5f SOLO (for reference)', (5, False, None)),
                  ('final 5f W .50 (for reference)', (5, False, .50))]:
    obs = RES[key][3]
    vs = {'final W.20 lift': fin, 'member solo s_Tm': s['Tm'], 'member r_Tm': r['Tm']}
    post = {kk: 0.0 for kk in vs}
    for o in np.unique(obs):
        m = np.isclose(obs, o); pm = m.mean()
        for kk, vv in vs.items(): post[kk] += pm * vv[m].var()
    pr(f'  {name:32s} | ' + ' | '.join(f'{kk} sd {vv.std():.4f} -> {np.sqrt(post[kk]):.4f}' for kk, vv in vs.items()))

# does a 2f W .20 read predict the 5f W .20 read? (display agreement)
a2, a5 = RES[(2, False, .20)][3], RES[(5, False, .20)][3]
pr(f'\nP(final 5f W.20 display == interim 2f W.20 display) = {np.isclose(a2, a5).mean():.2f}; '
   f'P(final > interim) = {(a5 > a2 + 1e-9).mean():.2f}; P(final < interim) = {(a5 < a2 - 1e-9).mean():.2f}')
pr(f'true-lift gain of waiting (5f minus 2f at W .20): {q(RES[(5, False, .20)][2] - RES[(2, False, .20)][2])}')

# pre-registered read tables: observed display -> posterior of the quantity that read measures (MODEL)
pr('')
pr('Pre-registered reads (posterior 10/50/90 given the displayed read; P = probability of that display)')
for k in (1, 2, 5):
    se, re = RES[(k, False, None)][0], RES[(k, False, None)][1]
    for W, what in ((None, 'SOLO'), (.50, 'W .50'), (.20, 'W .20')):
        ds = RES[(k, False, W)][3]
        for o in np.unique(ds):
            m = np.isclose(ds, o)
            if m.mean() < .02: continue
            pr(f'  {k}f {what:5s} reads {o:.3f} (P {m.mean():.2f}): fleet solo {q(se[m])} | fleet r {q(re[m])} | '
               f'member s_Tm {q(s["Tm"][m])} | 5f W.20 true lift {q(fin[m])}')
NL = chr(10)
open(__file__.replace('fleetsim_k4.py', 'fleetsim_k4.out'), 'w').write(NL.join(out) + NL)
