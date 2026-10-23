# addon_mc.py (10-19, think/runtime lens). MODEL only, CPU numpy, no I/O besides stdout.
# Question: if we add compute X (hours at session speed 1) to the c4-class kernel, what happens to
#   (A) a GUARDED stage placed AFTER S6F (start only if projected end <= HARD; process-group kill at HARD; all-or-nothing; fallback = c4 output)
#   (B) compute placed INLINE before S6F (e.g. an extra R6 TTA pass inside the R6 worker): S6F start guard G, S6F hard deadline 8.2 h.
# Calibration (MEASURED submit-to-COMPLETE bounds of 10 identical-compute c4-class runs, see runtime.md sec 1):
RUNS = {  # (lo, hi) hours, latency = queue + kernel + scoring
    's615': (5.64, 5.70), 'a25': (5.65, 5.715), 'c3max': (5.75, 5.957), 'c4': (6.045, 6.106), 'fr65': (6.094, 6.226),
    'R6': (6.05, 6.25), 'pf10': (6.252, 6.449), 'wr45': (6.33, 6.39), 'c2max': (5.75, 6.254), 'fr100': (5.75, 6.254),
    'c2': (5.75, 6.254), 'HW': (5.9, 6.55)}
import numpy as np
rng = np.random.default_rng(0)
N = 200000
names = list(RUNS)
pick = rng.integers(0, len(names), N)
lo = np.array([RUNS[names[i]][0] for i in pick]); hi = np.array([RUNS[names[i]][1] for i in pick])
lat = rng.uniform(lo, hi)
ovh = rng.uniform(0.0, 0.10, N)                      # queue + scoring overhead (GUESS)
E1 = lat - ovh                                         # kernel end at the observed session speed
TAIL_P, TAIL_M = 0.05, 1.30                           # GUESS slow-session tail, same as sim_s6f_runtime / sim_ffs_runtime
m = np.where(rng.random(N) < TAIL_P, TAIL_M, 1.0)
S6 = rng.uniform(1.90, 2.10, N)                       # S6F stage hours at s=1 (MODEL: S6F 6.6 - V224-FLIP 4.62)
base1 = E1 - S6                                       # Raptor + R6 + r224 + setup end
G, BUD, DL = 5.75, 3.3, 8.2
def c4_end(base, s6, mult, G=G, DL=DL):
    b = base * mult
    start_ok = b <= G
    hard = np.minimum(b + BUD, DL)
    end_full = b + s6 * mult
    cut = start_ok & (end_full > hard)
    end = np.where(~start_ok, b + 0.01, np.minimum(end_full, hard) + 0.02)
    degraded = (~start_ok) | cut
    return end, degraded
E, deg0 = c4_end(base1, S6, m)
print('c4 as built: end median %.2f p95 %.2f p99 %.2f max %.2f | P(S6F skip or cut = degraded) %.3f | P(end>9) %.4f'
      % (np.median(E), np.quantile(E, .95), np.quantile(E, .99), E.max(), deg0.mean(), (E > 9).mean()))
print('c4 unused to 9 h: median %.2f h, p05 %.2f h, min %.2f h' % (9 - np.median(E), 9 - np.quantile(E, .95), 9 - E.max()))
print()
print('(A) guarded post-S6F stage, HARD = 8.3 h, start iff E + 1.15*C*m_est <= HARD (m_est = 1), all-or-nothing')
for C in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
    noise = rng.normal(1.0, 0.07, N).clip(0.8, 1.3)
    cost = C * m * noise
    start = (E + 1.15 * C <= 8.3) & ~deg0
    end_add = E + cost + 0.03
    done = start & (end_add <= 8.3)
    kernel_end = np.where(start, np.minimum(end_add, 8.3 + 0.05), E)
    print('  C %.2f h: P(add-on used) %.3f | P(skipped at start) %.3f | P(killed at 8.3 -> c4 fallback) %.3f | kernel end p99 %.2f max %.2f | P(>9 h) %.4f'
          % (C, done.mean(), (~start).mean(), (start & ~done).mean(), np.quantile(kernel_end, .99), kernel_end.max(), (kernel_end > 9).mean()))
print()
print('(A0) UNGUARDED post-S6F stage of C hours (no start guard, no kill): P(kernel > 9 h)')
for C in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
    noise = rng.normal(1.0, 0.07, N).clip(0.8, 1.3)
    ke = E + C * m * noise + 0.03
    print('  C %.2f h: end median %.2f p99 %.2f max %.2f | P(>9 h) %.4f | P(>8.75 h) %.4f' % (C, np.median(ke), np.quantile(ke, .99), ke.max(), (ke > 9).mean(), (ke > 8.75).mean()))
print()
print('(B) INLINE compute X before S6F (e.g. extra R6 TTA pass): degraded = S6F skipped (start > G) or cut at its deadline')
for Gx in (5.75, 6.25, 6.6):
    for X in (0.25, 0.5, 0.85, 1.0, 1.7, 2.0):
        noise = rng.normal(1.0, 0.07, N).clip(0.8, 1.3)
        base = base1 + X * noise
        e, d = c4_end(base, S6, m, G=Gx)
        print('  G %.2f X %.2f h: end median %.2f p99 %.2f max %.2f | P(degraded) %.3f | P(>9 h) %.4f'
              % (Gx, X, np.median(e), np.quantile(e, .99), e.max(), d.mean(), (e > 9).mean()))
