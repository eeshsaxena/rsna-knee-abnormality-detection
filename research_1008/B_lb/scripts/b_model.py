#!/usr/bin/env python3
"""b_model.py (2026-10-08 ~21:30 IST, research_1008 dimension B). Laptop, light grids only.

Recalibrated macro single-d binormal rank-blend model (same functional form as SP/jump2/lbmath.py and
SP/k9n/k9n_weights.py, validated on 11 OOF pairs: error +.0000 to +.0009, one-signed optimistic).
AUC = Phi(d/sqrt2); rank blend with weights w over legs with correlation R:
    AUC_b = Phi( sum w_i d_i / sqrt(2 w'Rw) ).
Every displayed public read x is the interval [x-.0005, x+.0005).

NEW calibration inputs vs k9n_weights.py (all public LB reads):
  MEASURED (ours, kaggle competitions submissions): P .943; K5 SOLO .943, K5 W.25 .945, K5 W.50 .946;
     K6-i7 SOLO .942, W.50 .945; K7i SOLO .944, W.50 .946; K4i SOLO .939, W.20 .945.
  MEASURED third party (goodpjw2008 notebook rsna-knee-0-949-coatnet-stack-blend-lb-0-949, its own table, 10-08):
     N alone .949; 0.4 P(.943 community stack = our parent) + 0.6 N = .949;
     N + their own readers (Raptor 4f .933 solo, ConvNeXt 3f .929 solo) at 30% = .948.
  MEASURED third party (matterhorn3838 V2 / evgendvorkin Versia-5 / hitarthjain0 Tier-2 cell, LB cluster of 293 teams):
     N + G (goodpjw2008 ConvNeXt 3-fold reader, .929 solo) per-label weights mean .162 = .950.
"""
import itertools
import numpy as np
from scipy.stats import norm

np.set_printoptions(linewidth=160)
out = []


def pr(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    out.append(s)


def d(a): return np.sqrt(2) * norm.ppf(a)
def iv(x): return (x - .0005, x + .0005 - 1e-9)


def blend2(a, b, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(a) + W * d(b)) / np.sqrt(2 * v))


def blendk(s, R, w):
    w = np.asarray(w, float)
    dv = np.array([d(x) for x in s])
    return float(norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w))))


def psd(R): return np.linalg.eigvalsh(R).min() >= -1e-9


RG = np.linspace(.60, .9999, 8000)


def r_iv(a_rd, b_rd, blend_rd, W, n=9, shift=0.0):
    """r interval consistent with displayed reads a, b and blend at weight W on b.
    shift: subtract this optimism from the model before matching (model over-predicts)."""
    ok = np.zeros_like(RG, bool)
    for a in np.linspace(*iv(a_rd), n):
        for b in np.linspace(*iv(b_rd), n):
            m = blend2(a, b, RG, W) - shift
            ok |= (m >= blend_rd - .0005) & (m < blend_rd + .0005)
    if not ok.any():
        return None
    return RG[ok].min(), RG[ok].max()


def r_pt(a, b, x, W, shift=0.0):
    return RG[np.argmin(np.abs(blend2(a, b, RG, W) - shift - x))]


pr('=' * 100)
pr('A. CALIBRATION (inversions of displayed reads). shift 0 = raw model; shift .0005 = model optimism removed')
pr('=' * 100)
cal = [
    ('K5   P .943 + F5 .943 @W.50 = .946', .943, .943, .946, .50),
    ('K5   P .943 + F5 .943 @W.25 = .945', .943, .943, .945, .25),
    ('K6   P .943 + F7 .942 @W.50 = .945', .943, .942, .945, .50),
    ('K7i  P .943 + F10 .944 @W.50 = .946', .943, .944, .946, .50),
    ('K4i  P .943 + J1 .939 @W.20 = .945', .943, .939, .945, .20),
    ('gp   P .943 + N .949 @W.60 = .949 (goodpjw2008)', .943, .949, .949, .60),
    ('mh   N .949 + G .929 @W.162 = .950 (N+G cluster)', .949, .929, .950, .162),
    ('gp2  N .949 + R(.933/.929 readers ~.936) @W.30 = .948', .949, .936, .948, .30),
]
calres = {}
for name, a, b, x, W in cal:
    r0 = r_iv(a, b, x, W)
    r5 = r_iv(a, b, x, W, shift=.0005)
    p0 = r_pt(a, b, x, W)
    calres[name.split()[0]] = (r0, r5, p0)
    f = lambda t: 'none' if t is None else f'{t[0]:.3f}-{t[1]:.3f}'
    pr(f'{name:52s} r(raw) {f(r0):12s} point {p0:.3f} | r(optimism .0005 removed) {f(r5)}')

pr('\nNotes: (i) mh assumes the per-label weights act like one global .162; if the per-label routing itself added'
   ' signal, the true r_GN is HIGHER than shown (G adds less than it seems). (ii) gp2 reader strength .936 is a GUESS'
   ' for the 50/50 mix of a .933 and a .929 model.')

# central parameter choices (MODEL)
rPF_c = calres['K7i'][2]
rPN_lo, rPN_hi = calres['gp'][0]
rPN_c = calres['gp'][2]
rGN_c = calres['mh'][2]
pr(f'\nCentral: r_PF {rPF_c:.3f} (K7i point), r_PN {rPN_c:.3f} (range {rPN_lo:.3f}-{rPN_hi:.3f}), r_GN {rGN_c:.3f}')
pr('OLD K9N model used r_PN .925 (GUESS, from the 92-team cluster read as P+N). The goodpjw2008 measurement says P+N'
   ' at .4/.6 displays .949, so r_PN is much higher than .925 and P adds ~nothing on top of N.')

# What the OLD assumption would have predicted for goodpjw's P+N .4/.6
pr(f'Check: old r_PN .925 predicts P+N@.6 = {blend2(.943, .949, .925, .6):.4f} (would display .950-.951); measured .949.')

pr('\n' + '=' * 100)
pr('B. CANDIDATE BLENDS: modelled public AUC (raw model; subtract ~.0002-.0009, typically ~.0005, for a display read)')
pr('=' * 100)
# legs: P, F, N, G
F_vals = (.944, .945, .946)            # K9 SOLO unread; K7i SOLO .944 MEASURED; K9 expected .944-.946 MODEL
NT_vals = (.949, .948)                 # N true on public = read; .948 = winner's-curse/rounding low case (GUESS)
rFN_vals = (.86, .88, .90, .92, .94)   # NOT measured (GUESS band); anchor r_GN
rPN_vals = (rPN_lo, rPN_c, rPN_hi)
rFG_c, rPG_c = .91, .90                # GUESS (both ConvNeXt-T 2.5D MIL; P vs G unread)

CANDS = {
    'K9 main  P.50 F.50':            (.50, .50, 0, 0),
    'K9N-a    P.10 F.30 N.60':        (.10, .30, .60, 0),
    'K9N-b    P.30 F.30 N.40':        (.30, .30, .40, 0),
    'P.05 F.35 N.60':                 (.05, .35, .60, 0),
    'P.10 F.35 N.55':                 (.10, .35, .55, 0),
    'P.10 F.40 N.50':                 (.10, .40, .50, 0),
    'P.00 F.35 N.65 (F+N)':           (0, .35, .65, 0),
    'P.00 F.40 N.60 (F+N)':           (0, .40, .60, 0),
    'P.00 F.45 N.55 (F+N)':           (0, .45, .55, 0),
    'P.00 F.50 N.50 (F+N)':           (0, .50, .50, 0),
    'P.20 F.30 N.50':                 (.20, .30, .50, 0),
    'P.40 F.00 N.60 (goodpjw P+N)':   (.40, 0, .60, 0),
    'N+G cluster N.838 G.162':        (0, 0, .838, .162),
    'P.05 F.30 N.55 G.10':            (.05, .30, .55, .10),
    'P.00 F.35 N.55 G.10':            (0, .35, .55, .10),
}


def Rmat(rPF, rPN, rFN, rPG=rPG_c, rFG=rFG_c, rGN=None):
    rGN = rGN_c if rGN is None else rGN
    return np.array([[1, rPF, rPN, rPG], [rPF, 1, rFN, rFG], [rPN, rFN, 1, rGN], [rPG, rFG, rGN, 1]])


scen = []
for NT, F, rFN, rPN in itertools.product(NT_vals, F_vals, rFN_vals, rPN_vals):
    for rPF in (calres['K7i'][0][0], rPF_c, calres['K7i'][0][1]):
        R = Rmat(rPF, rPN, rFN)
        if psd(R):
            scen.append(dict(NT=NT, F=F, rFN=rFN, rPN=rPN, rPF=rPF, R=R))
pr(f'{len(scen)} PSD scenarios (N_true x F x r_FN x r_PN x r_PF); G legs at r_GN {rGN_c:.3f}, r_FG {rFG_c}, r_PG {rPG_c} (GUESS)')


def val(w, sc):
    return blendk((.943, sc['F'], sc['NT'], .929), sc['R'], w)


# central table by r_FN (NT .949, F .944, rPN central, rPF central)
pr('\nCentral (N .949, F .944, r_PN central, r_PF central), columns = r_FN')
pr(f'{"candidate":34s} ' + ' '.join(f'{x:>7.2f}' for x in rFN_vals) + '   | min-all   mean-all  max-all')
grid_simplex = [(wp, wf, round(1 - wp - wf, 2), 0) for wp in np.arange(0, .51, .05) for wf in np.arange(0, 1.0001 - wp, .05)
                if 1 - wp - wf >= -1e-9]
best_by_scen = []
for sc in scen:
    vals = [val(w, sc) for w in grid_simplex]
    i = int(np.argmax(vals))
    best_by_scen.append((grid_simplex[i], vals[i]))
summary = {}
for name, w in CANDS.items():
    row = []
    for rFN in rFN_vals:
        R = Rmat(rPF_c, rPN_c, rFN)
        row.append(blendk((.943, .944, .949, .929), R, w))
    allv = np.array([val(w, sc) for sc in scen])
    reg = np.array([b[1] for b in best_by_scen]) - allv
    summary[name] = (allv, reg)
    pr(f'{name:34s} ' + ' '.join(f'{x:7.4f}' for x in row) + f'   | {allv.min():.4f}   {allv.mean():.4f}   {allv.max():.4f}'
       f'  maxregret {reg.max():.4f}')

bw = np.array([b[0] for b in best_by_scen])
pr(f'\nPer-scenario best (P,F,N) on a .05 grid: P {bw[:, 0].min():.2f}-{bw[:, 0].max():.2f} (median {np.median(bw[:, 0]):.2f}), '
   f'F {bw[:, 1].min():.2f}-{bw[:, 1].max():.2f} (median {np.median(bw[:, 1]):.2f}), N {bw[:, 2].min():.2f}-{bw[:, 2].max():.2f} '
   f'(median {np.median(bw[:, 2]):.2f})')
for rFN in rFN_vals:
    sel = [b for b, sc in zip(best_by_scen, scen) if sc['rFN'] == rFN and sc['NT'] == .949 and sc['F'] == .944]
    ws = np.array([s[0] for s in sel])
    pr(f'  r_FN {rFN:.2f}: best P {np.median(ws[:, 0]):.2f} F {np.median(ws[:, 1]):.2f} N {np.median(ws[:, 2]):.2f}, '
       f'best AUC {np.median([s[1] for s in sel]):.4f}')

# minimax regret on finer grid (P,F,N only)
fine = [(wp, wf, round(1 - wp - wf, 3), 0) for wp in np.arange(0, .41, .02) for wf in np.arange(0, .81, .02) if 1 - wp - wf >= .1]
best_f = []
for sc in scen:
    v = [val(w, sc) for w in fine]
    best_f.append(max(v))
best_f = np.array(best_f)
mm = None
for w in fine:
    v = np.array([val(w, sc) for sc in scen])
    r = (best_f - v).max()
    if mm is None or r < mm[1]:
        mm = (w, r, v.min(), v.mean())
pr(f'\nMinimax-regret (P,F,N) on a .02 grid over all {len(scen)} scenarios: {tuple(round(x, 2) for x in mm[0][:3])}, '
   f'max regret {mm[1]:.4f}, min {mm[2]:.4f}, mean {mm[3]:.4f}')

pr('\n' + '=' * 100)
pr('C. WHAT F (our fleet SOLO) IS NEEDED FOR A TARGET (best P/F/N weights, N .949, r_PN central, r_PF central)')
pr('=' * 100)
pr(f'{"r_FN":>6s} ' + ' '.join(f'{"F->" + str(t):>10s}' for t in (.950, .951, .952, .953, .955)))
for rFN in (.86, .88, .90, .92, .94):
    row = []
    for tgt in (.950, .951, .952, .953, .955):
        found = None
        for F in np.arange(.935, .962, .0005):
            R = Rmat(rPF_c, rPN_c, rFN)
            bb = max(blendk((.943, F, .949, .929), R, w) for w in grid_simplex)
            if bb >= tgt + .0005:   # +.0005 so the optimism-corrected value still reaches the target
                found = F
                break
        row.append(f'{found:.4f}' if found else '   >.962')
    pr(f'{rFN:6.2f} ' + ' '.join(f'{x:>10s}' for x in row))
pr('(Each cell = smallest F SOLO whose best blend reaches the target after removing .0005 model optimism.)')

pr('\n' + '=' * 100)
pr('D. K10N / own-OAI members: modelled public of P.10/F.30/N.60-type blends when F improves (r_FN shifts with OAI)')
pr('=' * 100)
for label, F, rFN in [('K9N-a (F .944, r_FN .90)', .944, .90), ('K10N 320 set (F .945, r .90)', .945, .90),
                      ('K10N 320 set (F .9455, r .905)', .9455, .905),
                      ('own-OAI F (.947, r .92)', .947, .92), ('own-OAI F (.948, r .93)', .948, .93),
                      ('own-OAI F (.950, r .93)', .950, .93)]:
    R = Rmat(rPF_c, rPN_c, rFN)
    v_a = blendk((.943, F, .949, .929), R, (.10, .30, .60, 0))
    bb = max((blendk((.943, F, .949, .929), R, w), w) for w in grid_simplex)
    pr(f'{label:34s} K9N-a weights {v_a:.4f} | best {bb[0]:.4f} at P/F/N {bb[1][:3]}')

open(__file__.replace('.py', '.out'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
