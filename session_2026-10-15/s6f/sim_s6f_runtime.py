#!/usr/bin/env python
# sim_s6f_runtime.py (10-15, Claude subagent): hidden-run wall of V224-FLIP + S6fall (build_v224s6f.py) and the choice of the S6F
# start guard G / budget / hard deadline. MODEL. Same session model and R7 worker rule as v224f/sim_ffs_runtime.py (copied, not
# imported: that file runs at import); the base is V224-FLIP with the flip FORCED (always on, no time-aware switch).
# MEASURED inputs:
#   - V224-FLIP 57174021 submit-to-COMPLETE 4.57-4.65 h (two pollers, 10-14); V224-exact <= 3.28 h; HR7 2.84-3.26 h
#   - HV 57093094 (r384 + heliosli S6 4-fold eager ConvNeXt-S 384/352, compact_infer family): 3.18-3.35 h submit-to-score
#   - S6fall commit (nartaa v2, 10-14 19:13 UTC, T4 x2, 3 studies): env 19 s, inputs 8 s, B probe 26 s, model ready 3.9 s,
#     per study (windows, decode s, render s, gpu s): (574, 31.7, 1.6, 9.2) (502, 25.5, 1.4, 7.3) (510, 23.4, 1.2, 14.8);
#     warm second study 14.6 ms GPU per window (1 model, crop 288) vs HV S6 commit 33 ms per window (4 folds, crop 352)
# MODEL per-study cost of S6fall on the hidden set (s / study / T4, one process per GPU, 2 loader workers):
#   GPU-bound end: the commit's warm GPU time ~7.3-7.7 s (incl. any fp32 re-run compact_infer does on a non-finite fp16 pass);
#   decode-bound end: the compact decode path (mainline native_cache.prepare + geometry + SKM-TEA anchors + px_cache) is the
#   same as R7 / HV-S6, whose hidden per-study walls are 12.7-15 s (R7 plain, HV-S6); S6fall cannot be slower than that floor
#   with a lighter GPU part, so c ~ U(6.5, 13.0) s (GUESS within the MEASURED bounds), x session speed s, x per-study noise.
#   usage: python sim_s6f_runtime.py [--reps 4000] [--seed 0]
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reps', type=int, default=4000)
ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()

PRE = 0.42           # pip wheels + r384 Raptor stage before R7 (as sim_ffs_runtime)
FLIP_TOTAL = 4.62    # MEASURED V224-FLIP submit-to-score (midpoint of 4.57-4.65)
N_HID = 1300         # hidden studies (claimed ~1,300; 1,284 in nartaa's PREREG)
BLEND_H = 0.01       # S6F blend cell + receipts (seconds on the commit) + V224 stage 3 / canary (already in r224 term)
HARD = 9.0
C_LO, C_HI = 6.5, 13.0          # S6fall s / study / T4 (MODEL range, see header)
SETUP_LO, SETUP_HI = 0.03, 0.08  # S6F setup (DEC ~6 s, inputs + 200 MB sha, B probe ~30 s, worker spawn + model + V-Net) h
CI_MARGIN_H = (20 * 60 + 4 * 60) / 3600   # compact_infer --deadline = stop_at - 20 min (nartaa's margin); stop_at = hard - 4 min


def r7_worker(rng, n, flip_h, pre_h):
    """sim_ffs_runtime.r7_worker with aware=False (V224-FLIP: flip forced on every study)."""
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n)).astype(int)
    noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    rf = (flip_h * 3600 - 250) / files.sum()
    return (pre_h * 3600 + 250 + (rf * files * noise).sum()) / 3600


def s6_part(rng, n, c, s):
    """one S6 GPU process: per-study cost c x s x file-count share x noise; returns cumulative end times (h, from its start)."""
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n))
    order = np.argsort(-files, kind='mergesort')            # split_parts: most series first (series count ~ files)
    order = np.concatenate([order[::2], order[1::2]])
    f = files[order]
    noise = rng.normal(1.0, 0.12, n).clip(0.6, 1.6)
    per = c * s * f / f.mean() * noise
    return np.cumsum(per) / 3600


def run(rng, s, q, r224, c, setup, cfg, n=N_HID):
    flip_h = FLIP_TOTAL - q - PRE - r224
    base_end = max(r7_worker(rng, n // 2, flip_h * s * n / 1300, PRE * s) for _ in range(2)) + r224 * s
    start = base_end + 0.002                                     # S6F setup cell starts right after the V224 canary
    if start > cfg['G']:
        return dict(base=base_end, start=start, end=start + BLEND_H, mode='SKIP', cov=0.0)
    hard = min(start + cfg['budget'], cfg['deadline'])
    ci_deadline = hard - CI_MARGIN_H                             # no NEW study after this (compact_infer checks after each study)
    t0 = start + setup * s
    done, ends = 0, []
    for _ in range(2):
        cum = t0 + s6_part(rng, n // 2, c, s)
        # compact_infer finishes the study in flight when its deadline passes, then stops
        k = int(np.searchsorted(cum, ci_deadline, side='right'))
        k_done = min(len(cum), k + (1 if k < len(cum) else 0))
        done += k_done
        ends.append(cum[k_done - 1] if k_done else t0)
    end = max(ends)
    killed = end > hard
    end = min(end, hard) + BLEND_H
    cov = done / n if not killed else min(done / n, 0.0)          # a kill at the hard deadline: parts without a receipt (MODEL: 0)
    mode = 'S6F' if cov >= 0.98 else 'CUT'
    return dict(base=base_end, start=start, end=end, mode=mode, cov=cov)


CFGS = {}
for G in (5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5):
    CFGS['G %.2f / budget 3.0 / deadline 8.2' % G] = dict(G=G, budget=3.0, deadline=8.2)
CFGS['G 5.75 / budget 2.6 / deadline 8.2'] = dict(G=5.75, budget=2.6, deadline=8.2)
REC = dict(G=5.75, budget=3.3, deadline=8.2)                      # the build default (build_v224s6f.py --g-start-h / --budget-h)
CFGS['RECOMMENDED G 5.75 / budget 3.3 / deadline 8.2'] = REC
CFGS['G 6.00 / budget 3.3 / deadline 8.2'] = dict(G=6.0, budget=3.3, deadline=8.2)
CFGS['no guard (G 99) / budget 99 / deadline 8.2'] = dict(G=99, budget=99, deadline=8.2)

rng = np.random.default_rng(A.seed)


def mc(cfg, reps, c_rng=(C_LO, C_HI), n=N_HID, s_tail=1.3, p_tail=0.05):
    E, M, CV, B, S = [], [], [], [], []
    for _ in range(reps):
        s = float(np.exp(rng.normal(0, 0.08)) * (s_tail if rng.random() < p_tail else 1.0))
        q = rng.uniform(0.0, 0.3); r224 = rng.uniform(0.15, 0.30)
        c = rng.uniform(*c_rng); setup = rng.uniform(SETUP_LO, SETUP_HI)
        r = run(rng, s, q, r224, c, setup, cfg, n)
        E.append(r['end']); M.append(r['mode']); CV.append(r['cov']); B.append(r['base']); S.append(r['end'] + q)
    E, B, S = np.array(E), np.array(B), np.array(S)
    return E, M, B, S


print('# A. deterministic grid (one draw per cell, s = session speed, c = S6fall s/study/T4); kernel h (age from NB_T0)')
print('#    V224-FLIP base end (always-on flip) | S6F start | S6F mode | coverage | kernel end; q .1, r224 .2')
for name in ('G 5.50 / budget 3.0 / deadline 8.2', 'RECOMMENDED G 5.75 / budget 3.3 / deadline 8.2', 'G 6.00 / budget 3.3 / deadline 8.2'):
    cfg = CFGS[name]
    print(' ', name)
    for s in (0.9, 1.0, 1.1, 1.2, 1.3, 1.45):
        row = []
        for c in (6.5, 9.0, 13.0):
            r = run(rng, s, 0.1, 0.2, c, 0.05, cfg)
            row.append('c %4.1f: base %.2f start %.2f %-4s cov %.2f end %.2f' % (c, r['base'], r['start'], r['mode'], r['cov'], r['end']))
        print('    s %.2f | ' % s + ' | '.join(row))

print('\n# B. Monte Carlo (%d sessions per config): s = exp(N(0, .08)) x (1.3 w.p. .05), q ~ U(0, .3), r224 ~ U(.15, .30),' % A.reps)
print('#    c ~ U(%.1f, %.1f) s/study/T4, setup ~ U(%.2f, %.2f) h x s, N %d. Kernel end h; submit-to-score = end + q.' % (C_LO, C_HI, SETUP_LO, SETUP_HI, N_HID))
for name, cfg in CFGS.items():
    E, M, B, S = mc(cfg, A.reps)
    print(f'  {name}: end median {np.median(E):.2f} p90 {np.percentile(E, 90):.2f} p99 {np.percentile(E, 99):.2f} max {E.max():.2f} '
          f'| P(S6F used) {M.count("S6F") / len(M):.3f} P(skip) {M.count("SKIP") / len(M):.3f} P(cut -> canary) {M.count("CUT") / len(M):.3f} '
          f'| P(end > 8.0) {np.mean(E > 8.0):.3f} P(end > 9) {np.mean(E > HARD):.3f} | submit-to-score median {np.median(S):.2f} p90 {np.percentile(S, 90):.2f}')

print('\n# C. sensitivity for the recommended G (rows: S6fall cost range, hidden N, slow-session tail)')
for G, bud in ((5.5, 3.0), (5.75, 3.3)):
    cfg = dict(G=G, budget=bud, deadline=8.2)
    for c_rng, n, st, pt in (((6.5, 13.0), 1300, 1.3, .05), ((6.5, 9.0), 1300, 1.3, .05), ((10.0, 13.0), 1300, 1.3, .05),
                             ((13.0, 16.0), 1300, 1.3, .05), ((6.5, 13.0), 1500, 1.3, .05), ((6.5, 13.0), 1300, 1.3, .15),
                             ((6.5, 13.0), 1300, 1.45, .05)):
        E, M, B, S = mc(cfg, A.reps, c_rng, n, st, pt)
        print(f'  G {G:.2f} budget {bud} c {c_rng} N {n} tail x{st} p {pt}: end median {np.median(E):.2f} p99 {np.percentile(E, 99):.2f} max {E.max():.2f} '
              f'| P(used) {M.count("S6F") / len(M):.3f} P(skip) {M.count("SKIP") / len(M):.3f} P(cut) {M.count("CUT") / len(M):.3f} '
              f'| P(end > 8.0) {np.mean(E > 8.0):.3f}')

print('\n# D. bounds (no sampling): the latest S6F start is G; the run is cut by min(G + budget, 8.2 h); compact_infer takes no new')
print('#    study after the hard deadline - 24 min; the cell kills every S6F process group at the hard deadline; blend + receipts')
print('#    ~1 min. So the kernel can never pass ~8.22 h because of S6F (9 h limit margin >= 0.75 h), whatever S6fall costs.')
for G in (5.5, 5.75, 6.0):
    for c in (9.0, 13.0, 16.0):
        dur = 0.08 + 650 * c / 3600
        print(f'  G {G:.2f}: c {c:4.1f} s -> S6F wall {dur:.2f} h (s 1.0), {dur * 1.3:.2f} h (s 1.3); a start AT G ends {G + dur:.2f} / {G + dur * 1.3:.2f} h '
              f'-> {"inside 8.0" if G + dur * 1.3 <= 8.0 else ("cut at 8.2 (canary unless >= 98 % done)" if G + dur * 1.3 > 8.2 - CI_MARGIN_H else "8.0-8.2")}')
