# crash_vs_timing.py (10-21 final red-team, Claude). MODEL only. If the 10-21 S6F canaries came from a GENERIC compact_infer worker
# crash hazard (constant per worker-hour), the R6 stage (same compact_infer.py, 2 workers, ~3.5 h, NO canary: crashed rows take
# Raptor rank) would have crashed too, silently, in most runs. 0 of 5 10-21 runs show a display anomaly (swide .960 best, pw25 .959,
# 3 canaries all exactly .916). Likelihood of that under the crash hypothesis vs the timing hypothesis (no crash hazard).
import math
out = []
for n_crash in (1.5, 3.0):                     # canaries attributed to crashes (cause.md .5 / all 3)
    for s6_wh, r6_wh in ((4.0, 7.0), (4.0, 5.0)):  # worker-hours exposed: S6F 2 x ~2 h; R6 2 x ~3.5 h (or 2.5 h)
        p_s6 = n_crash / 5; lam = -math.log(1 - p_s6) / s6_wh; p_r6 = 1 - math.exp(-lam * r6_wh)
        for vis in (.5, .7):                    # P(a silent R6 crash shows on the 3-dp display), f ~ U(0, .5)
            L = (1 - p_r6 * vis) ** 5
            out.append('crash-caused canaries %.1f, exposure S6F %.0f / R6 %.0f worker-h: P(R6 crash per run) %.2f | vis %.1f: P(0 anomalies in 5 | crash) %.3f, LR timing:crash %.1f' % (n_crash, s6_wh, r6_wh, p_r6, vis, L, 1 / L))
for prior in (.5,):
    for L in (.2, .05):
        out.append('prior P(crash path) %.1f, P(0 anomalies | crash) %.2f -> posterior P(crash path) %.2f' % (prior, L, prior * L / (prior * L + 1 - prior)))
print('\n'.join(out)); open('crash_vs_timing.log', 'w').write('\n'.join(out) + '\n')
