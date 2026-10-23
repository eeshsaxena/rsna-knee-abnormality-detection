# sim_ta.py (MODEL, 2026-10-15 dir): what the time-aware flip rule of compact_infer_ta.py does on a modelled hidden run of
# V224-F-FLIP, versus a plain always-on flip (V224-FLIP + fleet). Inputs are the 10-14 runtime review's MODEL numbers (V224-exact
# hidden wall 3.0 h central = pre-R7 0.45 h + R7 2.25 h + r224/stage 3 0.30 h; flip adds 1.0 / 1.4 / 2.0 h to R7) and the V224-F
# wall model (fleet 1.80 h central, 2.07 h slow), all scaled by a session-speed factor. Per worker ~640 studies with lognormal DICOM
# counts in probe_dual.py's order (big first, interleaved), per-file rate constant inside a run +-15 % per-study noise; compile
# check 250 s at the start of every child. The rule is the driver's: after study k (k >= 10) projected end = t_k + 1.05 x
# rate x files left > budget and remaining flip work > 600 s -> switch; plain child restarts (250 s) at the plain rate.
#   usage: python sim_ta.py [--budget 4.9]
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--budget', type=float, default=4.9)
A = ap.parse_args()
FLEET_START_GUARD, WALL, HARD = 5.5, 8.3, 9.0


def worker(rng, n, r7_plain_h, r7_flip_h, pre_h, budget_h, aware):
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n)).astype(int)
    order = np.argsort(-files, kind='mergesort')
    order = np.concatenate([order[::2], order[1::2]])
    f = files[order]
    noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    rf = (r7_flip_h * 3600 - 250) / f.sum()       # s per file with flip (a worker's whole R7 = its compile + its loop)
    rp = (r7_plain_h * 3600 - 250) / f.sum()
    t = pre_h * 3600 + 250
    t_rows = []
    for k in range(n):
        t += rf * f[k] * noise[k]
        t_rows.append(t)
        if not aware:
            continue
        kk = k + 1
        if kk >= n:
            break
        rem = f[kk:].sum()
        if kk >= 10:
            rate = (t_rows[-1] - t_rows[0]) / f[1:kk].sum()
            work = 1.05 * rate * rem
            trig = t + work > budget_h * 3600 and work > 600
        else:
            trig = t > budget_h * 3600
        if trig:
            t += 250 + (rp * f[kk:] * noise[kk:]).sum()
            return t / 3600, kk / n
    return t / 3600, 1.0


rng = np.random.default_rng(0)
print('budget %.2f h | speed x flip_extra_h | aware: R7 end, flipped share, fleet start, end | always-on flip: R7 end, fleet start, '
      'fleet run?' % A.budget)
for speed in (0.9, 1.0, 1.15, 1.3):
    for extra in (1.0, 1.4, 2.0):
        res = {}
        for aware in (True, False):
            ends, shares = [], []
            for rep in range(20):
                w = [worker(rng, 640, 2.25 * speed, (2.25 + extra) * speed, 0.45 * speed, A.budget, aware) for _ in range(2)]
                ends.append(max(e for e, _ in w)); shares.append(np.mean([s for _, s in w]))
            r7 = float(np.mean(ends)); fs = r7 + 0.30 * speed + 0.05
            runs = fs <= FLEET_START_GUARD
            fe = fs + (1.80 if speed <= 1.0 else 2.07) * speed if runs else fs
            res[aware] = (r7, float(np.mean(shares)), fs, fe, runs)
        a, b = res[True], res[False]
        print('x%.2f +%.1f h | aware R7 %.2f h, flip %3.0f %%, fleet start %.2f, end %.2f (%s) | always-on R7 %.2f, fleet start %.2f, %s'
              % (speed, extra, a[0], 100 * a[1], a[2], a[3], 'fleet ok' if a[4] and a[3] < WALL else 'FLEET SKIPPED/LATE',
                 b[0], b[2], 'fleet ok' if b[4] and b[3] < WALL else 'FLEET SKIPPED -> Synovitis canary'))

# ---- appendix: the alternative "latest switch" rule (NOT used; MODEL) -----------------------------------------------------------
# keep flip while one more flip study + a restart + the remainder WITHOUT flip (estimated as 0.75 x the measured flip rate) still
# end by the budget. More flip coverage in slow sessions, but it relies on a prior plain/flip wall ratio: if R7 is decode-bound
# (ratio ~1) the post-switch remainder runs longer than estimated and R7 can overshoot the budget by ~0.1 h. The driver keeps the
# task's rule (projected end WITH flip), which needs no ratio prior and keeps R7 end <= budget in the smooth case.
def worker_latest(rng, n, r7_plain_h, r7_flip_h, pre_h, budget_h, ratio=0.75):
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n)).astype(int)
    order = np.argsort(-files, kind='mergesort')
    order = np.concatenate([order[::2], order[1::2]])
    f = files[order]
    noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    rf = (r7_flip_h * 3600 - 250) / f.sum()
    rp = (r7_plain_h * 3600 - 250) / f.sum()
    t = pre_h * 3600 + 250
    rows = []
    for k in range(n):
        t += rf * f[k] * noise[k]
        rows.append(t)
        kk = k + 1
        if kk >= n:
            break
        rem = f[kk:].sum()
        if kk >= 10:
            rate = (rows[-1] - rows[0]) / f[1:kk].sum()
            trig = t + 1.05 * rate * f[kk] + 250 + 1.05 * ratio * rate * (rem - f[kk]) > budget_h * 3600 and 1.05 * rate * rem > 600
        else:
            trig = t > budget_h * 3600
        if trig:
            t += 250 + (rp * f[kk:] * noise[kk:]).sum()
            return t / 3600, kk / n
    return t / 3600, 1.0


print('\nappendix, alternative latest-switch rule (not used): R7 end / flip share')
for speed in (1.0, 1.15, 1.3):
    for extra in (1.0, 1.4, 2.0):
        e, s = [], []
        for rep in range(20):
            w = [worker_latest(rng, 640, 2.25 * speed, (2.25 + extra) * speed, 0.45 * speed, A.budget) for _ in range(2)]
            e.append(max(x for x, _ in w)); s.append(np.mean([y for _, y in w]))
        print('x%.2f +%.1f h | latest-switch R7 %.2f h (max %.2f), flip %3.0f %%' % (speed, extra, np.mean(e), max(e), 100 * np.mean(s)))
