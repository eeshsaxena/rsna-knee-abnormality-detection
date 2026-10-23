"""fleetsim.py (jump2, 2026-10-05): fleet composition -> predicted solo / r / W* / public read, by rejection-sampled
binormal model conditioned on our MEASURED LB reads (K2, K3) and the MEASURED fold-0 OOF structure (~/jump2/oofx.log).
Light CPU (vectorised numpy, ~40k draws). Writes fleetsim.out next to this file.

Tags: every input below is marked MEASURED (source) or GUESS.
"""
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

out = []
def pr(x=''):
    print(x); out.append(x)
def q(x, ps=(10, 50, 90)): return '/'.join(f'{np.percentile(x, p):.4f}' for p in ps)
pr(f'posterior after K2+K3 rejection (PSD check {PSD}): N={N}')
for t in TYPES:
    pr(f'  {t:4s} solo {q(s[t])}  r_LB {q(r[t])}   (10/50/90 pct)')
pr(f'  parent true {q(P)}  rho_s {q(rho_s)}')

def fleet(counts, w=None):
    """counts {type: n}; equal weight per member (rank mean). Returns fleet solo, r with parent (vectors)."""
    n = sum(counts.values()); ts = [t for t in counts if counts[t] > 0]
    v = np.zeros(N); dd = np.zeros(N); rr = np.zeros(N)
    for a in ts:
        ca = counts[a]; dd += ca * d(s[a]); rr += ca * r[a]
        v += ca * (1 + (ca - 1) * rho_s)
        for b in ts:
            if b != a: v += ca * counts[b] * RMM(a, b)
    v /= n * n; dd /= n; rr /= n
    return A(dd / np.sqrt(v)), rr / np.sqrt(v)

WG = np.round(np.arange(0, 0.801, 0.05), 2)
def best_fixed_W(se, re):
    """W maximising the posterior-expected TRUE blend (W must be chosen before reading)."""
    m = [blend(P, se, re, w).mean() for w in WG]; i = int(np.argmax(m)); return WG[i]
def disp(x): return np.round(x - 1e-12, 3)

COMPS = [
    ('C0 F1 mix50 f0 (K2 member, sanity)', {'F1m': 1}, '10-04'),
    ('A1 5-fold TTR mix50', {'Tm': 5}, '10-06'),
    ('A2 5f TTR mix50 + F1 mix50', {'Tm': 5, 'F1m': 1}, '10-06'),
    ('A3 5f TTR mix50 + TTR v4 f0', {'Tm': 5, 'T1v': 1}, '10-06'),
    ('A4 5f TTRm + F1m + TTRv4 + F1v4', {'Tm': 5, 'F1m': 1, 'T1v': 1, 'F1v': 1}, '10-06'),
    ('A5 all 9 (A4 + gated)', {'Tm': 5, 'F1m': 1, 'T1v': 1, 'F1v': 1, 'F1g': 1}, '10-06'),
    ('A6 v4-only pair TTRv4 + F1v4', {'T1v': 1, 'F1v': 1}, '10-06'),
    ('A7 TTRv4 f0 alone', {'T1v': 1}, '10-04'),
    ('B1 5f TTRm + 5f TTRv4', {'Tm': 5, 'T1v': 5}, '10-08'),
    ('B2 5f TTRm + 5f TTRv4 + F1m + F1v4', {'Tm': 5, 'T1v': 5, 'F1m': 1, 'F1v': 1}, '10-08'),
    ('B3 v4-only: 10 TTRv4 (2 seeds x 5f)', {'T1v': 10}, '10-08'),
    ('B4 5f TTRm + 10 TTRv4', {'Tm': 5, 'T1v': 10}, '10-08'),
    ('B5 mix50-only: 10 TTRm (2 seeds x 5f) + F1m', {'Tm': 10, 'F1m': 1}, '10-08'),
    ('C1 5f TTRm + 5f TTRv4 + 5f fam2-v4', {'Tm': 5, 'T1v': 5, 'F2': 5}, '10-12'),
    ('C2 v4-only: 5f TTRv4 + 5f fam2 + 5 TTRv4 s2', {'T1v': 10, 'F2': 5}, '10-12'),
    ('C3 5f TTRv4 + 5f fam2-v4', {'T1v': 5, 'F2': 5}, '10-12'),
    ('C4 C1 + F1m', {'Tm': 5, 'T1v': 5, 'F2': 5, 'F1m': 1}, '10-12'),
    ('C5 mix50-heavy: 15 TTRm (3 seeds x 5f)', {'Tm': 15}, '10-12'),
    ('C6 5f TTRm + 10 TTRv4-ownOOF-label (Tv2)', {'Tm': 5, 'Tv2': 10}, '10-12'),
    ('C7 10 Tv2 + 5 fam2', {'Tv2': 10, 'F2': 5}, '10-12'),
    ('H1 NEED: 10 members of (.930, r .86), r_mm .96', {'H': 10}, 'target'),
    ('H2 NEED: 5 H + 5f TTRm', {'H': 5, 'Tm': 5}, 'target'),
]
pr('\nfleet | ready | solo 10/50/90 | r 10/50/90 | W* (fixed, chosen ex ante) | true blend at W* 10/50/90 | display at W*: P(<=.942) P(.943) P(.944) P(>=.945) | E[display] | oracle-W lift median')
RES = {}
for name, cnt, ready in COMPS:
    se, re = fleet(cnt)
    W = best_fixed_W(se, re); bt = blend(P, se, re, W); ds = disp(bt)
    # oracle W per draw (upper bound of what an informed W could do)
    ob = np.max(np.stack([blend(P, se, re, w) for w in WG]), axis=0)
    RES[name] = dict(se=se, re=re, W=W, bt=bt, ds=ds)
    pr(f'{name:44s} | {ready} | {q(se)} | {q(re, (10, 50, 90))} | {W:.2f} | {q(bt)} | '
       f'{(ds <= .942).mean():.2f} {(np.isclose(ds, .943)).mean():.2f} {(np.isclose(ds, .944)).mean():.2f} {(ds >= .945).mean():.2f} | {ds.mean():.4f} | {np.median(ob - P):+.4f}')

# ---------------- K4 candidates for 10-06: expected public read and value of information ----------------
pr('\nK4 candidates (10-06). Decision informed: which production fleet to build/ship by 10-12, among')
DEC = ['C1 5f TTRm + 5f TTRv4 + 5f fam2-v4', 'C2 v4-only: 5f TTRv4 + 5f fam2 + 5 TTRv4 s2', 'C5 mix50-heavy: 15 TTRm (3 seeds x 5f)', 'B4 5f TTRm + 10 TTRv4']
pr('  ' + ' | '.join(DEC))
L = np.stack([RES[k]['bt'] - P for k in DEC])          # true lift of each final option, per draw (W fixed ex ante per option)
prior_best = L.mean(1).max()
pr(f'  prior expected lift of each: ' + ', '.join(f'{L[i].mean():+.4f}' for i in range(len(DEC))) + f'  -> best {prior_best:+.4f}')
CAND = {
    'K4a SOLO 5f TTR mix50 (A1)': ('solo', 'A1 5-fold TTR mix50', None),
    'K4b SOLO all 9 (A5)': ('solo', 'A5 all 9 (A4 + gated)', None),
    'K4c P + A1 at W*': ('blend', 'A1 5-fold TTR mix50', None),
    'K4d P + A4 at W*': ('blend', 'A4 5f TTRm + F1m + TTRv4 + F1v4', None),
    'K4e P + A5 at W*': ('blend', 'A5 all 9 (A4 + gated)', None),
    'K4f P + v4-only pair (A6) at W .40': ('blend', 'A6 v4-only pair TTRv4 + F1v4', .40),
    'K4g P + TTRv4 alone at W .40': ('blend', 'A7 TTRv4 f0 alone', .40),
    'K4h P + A1 at W .50 (aastik-style)': ('blend', 'A1 5-fold TTR mix50', .50),
    'K4i P + A5 at W .50 (aastik-style)': ('blend', 'A5 all 9 (A4 + gated)', .50),
}
pr('candidate | W | display 10/50/90 | E[display] | P(>=.944) | EVOI for the 10-12 fleet choice | sd(r_TTRv4) prior->post | sd(best final lift) prior->post')
sd_r0 = r['T1v'].std(); bestL = L.max(0); sd_b0 = bestL.std()
for cname, (kind, comp, Wfix) in CAND.items():
    se, re = RES[comp]['se'], RES[comp]['re']
    if kind == 'solo':
        obs = disp(se); W = None; shown = obs
    else:
        W = Wfix if Wfix is not None else RES[comp]['W']
        obs = disp(blend(P, se, re, W)); shown = obs
    # value of information: posterior mean lift per option within each observed display bin
    evoi = 0.0; post_sd_r = 0.0; post_sd_b = 0.0
    for o in np.unique(obs):
        m = np.isclose(obs, o); pm = m.mean()
        evoi += pm * L[:, m].mean(1).max()
        post_sd_r += pm * r['T1v'][m].var(); post_sd_b += pm * bestL[m].var()
    evoi -= prior_best
    pr(f'{cname:38s} | {"solo" if W is None else f"{W:.2f}"} | {q(shown)} | {shown.mean():.4f} | {(shown >= .944).mean():.2f} | {evoi:+.5f} | '
       f'{sd_r0:.3f}->{np.sqrt(post_sd_r):.3f} | {sd_b0:.4f}->{np.sqrt(post_sd_b):.4f}')

# ---------------- what fleet do we NEED: (solo, r) for +.002/+.003 at W* ----------------
pr('\nRequirement (MODEL, parent .943): fleet solo needed for a TRUE lift at W*')
for rr_ in (.84, .86, .88, .90, .92, .94):
    row = []
    for tgt in (.0015, .0025, .0035):
        ss = np.linspace(.915, .96, 901)
        bb = np.max(np.stack([blend(.943, ss, rr_, w) for w in np.arange(0, 1.001, .02)]), axis=0)
        j = np.argmax(bb >= .943 + tgt); row.append(f'{ss[j]:.4f}' if bb[j] >= .943 + tgt else ' n/a ')
    pr(f'  r {rr_:.2f}: +.0015 {row[0]} | +.0025 {row[1]} | +.0035 {row[2]}')
open(__file__.replace('fleetsim.py', 'fleetsim_' + ('psd' if PSD else 'nopsd') + '.out'), 'w').write('\n'.join(out) + '\n')
