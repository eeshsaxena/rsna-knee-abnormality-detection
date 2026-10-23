#!/usr/bin/env python
# sim_room_newfam.py (10-15 ~11:45 IST, Claude subagent, box-edge-plan lens "kernel runtime room"). MODEL on MEASURED inputs.
# No em dashes. Laptop CPU, numpy only, read-only (writes nothing but stdout).
#
# Question: how many T4-seconds per study x remain for ONE new model family X (2 T4, each scores half of the hidden studies, so
# x = seconds of one T4 per study) on top of the AO candidate bases, under p99(kernel end) <= 8.0 h, and which component X should
# replace if room is short.
#
# Model pieces copied (not imported: those files run at import) from the reviewed sims, same constants:
#   - V224-FLIP (R7 flip forced) calibrated to its MEASURED 4.57-4.65 h submit-to-score (FLIP_TOTAL 4.62 = queue + kernel),
#     pre-R7 0.42 h, r224 + stage 3 U(.15, .30) h x s, queue U(0, .3) h: v224f/sim_ffs_runtime.py D / s6f2/sim_s6f2_runtime.py
#   - S6fall: c ~ U(6.5, 13) s / study / T4 (MODEL), setup U(.03, .08) h x s, 2 parts of n/2 studies with lognormal file shares and
#     per-study noise sd .12, blend .01 h: s6f/sim_s6f_runtime.py
#   - S (5 x TTR224): 1.5 min fixed + n/2 x 1.62 s x mf x s (MEASURED microbench) + pre-cells .01 + HF .005 h
#   - K14 fleet (20 x TTR224/288): n/2 x 9.86 s x mf x s (MEASURED slower-GPU microbench, V224-F commit), merged with S into one
#     fleet stage (25 members, 11.48 s) when both are present, as in build_v224fs.py --base ff
#   - session model B: s = exp(N(0, .08)) x 1.3 w.p. .05; STRESS: exp(N(0, .12)) x {1, 1.3, 1.5} w.p. {.88, .09, .03} (GUESS)
# New here:
#   - NATURAL ends (no start guards, no deadlines): the question is whether everything fits, so a guard that would fire counts as
#     "does not fit". p99 of the natural end <= 8.0 h <=> P(some stage would be cut or skipped by an 8.0 h line) <= 1 %.
#   - X = a separate stage after the base: 0.002 h + setup_x x s + x x s x (max over the 2 parts of the sum of per-study relative
#     costs, lognormal file share x noise sd .12) / 3600 + blend .01 h. setup_x ~ U(.03, .08) h (GUESS: env + model load + warm-up,
#     like S6F). Bisection on x per base, paired sessions (every base sees the same draws).
#   - optional J (DEC cell + preflight of compressed hidden files, capped at 600 s): + U(.005, .05) h typical (MODEL) or the .17 h cap.
#   - G2: X as a second model group INSIDE S6fall's compact_infer (`--models2`, "second architecture on the same windows (one decode)"):
#     S6F per-study cost becomes max(loader, g1 + g2), g1 = S6fall warm GPU 7.24-7.58 s (MEASURED, 2 commits), loader = c draw.
#   usage: py -3.12 -I sim_room_newfam.py [--reps 20000] [--seed 0]
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reps', type=int, default=20000)
ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()

PRE, FLIP_TOTAL, N_HID = 0.42, 4.62, 1300
FIX_FL, PRECELLS, S_S, K14_S, HF_H = 1.5 / 60, 0.01, 1.62, 9.86, 0.005
C_LO, C_HI, SETUP_LO, SETUP_HI, BLEND_H = 6.5, 13.0, 0.03, 0.08, 0.01
G1_LO, G1_HI = 7.24, 7.58          # S6fall warm GPU s / study (MEASURED: gate_v224flips6f_v1_rfix 7.58, gate_v224flips6fs20_v1m 7.24)
TARGET, PCT = 8.0, 99.0


def draw(rng, model, n, c_rng, ftot):
    if model == 'B':
        s = float(np.exp(rng.normal(0, 0.08)) * (1.3 if rng.random() < 0.05 else 1.0))
    else:
        s = float(np.exp(rng.normal(0, 0.12)) * rng.choice([1.0, 1.3, 1.5], p=[0.88, 0.09, 0.03]))
    q, r224, mf = rng.uniform(0.0, 0.3), rng.uniform(0.15, 0.30), rng.uniform(1.0, 1.15)
    flip_h = ftot - q - PRE - r224
    r7 = []
    for _ in range(2):                                   # r7_flip_ao (forced flip, no driver)
        files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n // 2)).astype(int)
        noise = rng.normal(1.0, 0.15, n // 2).clip(0.6, 1.6)
        fh = flip_h * s * n / 1300
        rf = (fh * 3600 - 250) / files.sum()
        r7.append((PRE * s * 3600 + 250 + (rf * files * noise).sum()) / 3600)
    base = max(r7) + r224 * s                            # natural: no R7 deadline cut, no r224 skip

    def rel_parts():                                     # max over the 2 parts of sum(file share x noise) (~ n/2)
        tot = []
        for _ in range(2):
            f = np.maximum(20, rng.lognormal(np.log(220), 0.45, n // 2))
            noise = rng.normal(1.0, 0.12, n // 2).clip(0.6, 1.6)
            tot.append((f / f.mean() * noise).sum())
        return max(tot)
    d = dict(s=s, q=q, mf=mf, n=n, base=base, c=rng.uniform(*c_rng), setup6=rng.uniform(SETUP_LO, SETUP_HI),
             k6=rel_parts(), kx=rel_parts(), setupx=rng.uniform(SETUP_LO, SETUP_HI), g1=rng.uniform(G1_LO, G1_HI),
             j=rng.uniform(0.005, 0.05))
    return d


def s6f_dur(d, c=None):
    c = d['c'] if c is None else c
    return 0.002 + d['setup6'] * d['s'] + c * d['s'] * d['k6'] / 3600 + BLEND_H


def fleet_dur(d, per):
    return PRECELLS + FIX_FL + (d['n'] / 2) * per * d['mf'] * d['s'] / 3600 + HF_H


def base_end(d, comps, j=0.0):
    """natural end of a base = FLIP + the listed components (S6F, S, K14; S + K14 merge into one fleet stage)."""
    e = d['base'] + j
    if 'S6F' in comps:
        e += s6f_dur(d)
    per = (S_S if 'S' in comps else 0.0) + (K14_S if 'K14' in comps else 0.0)
    if per:
        e += fleet_dur(d, per)
    return e


BASES = {
    'FLIP alone': (),
    'FLIP + S': ('S',),
    'FLIP + S6F': ('S6F',),
    'FLIP + S6F + S': ('S6F', 'S'),
    'FLIP + K14 (AO)': ('K14',),
    'FLIP + K14 + S (fleet 25)': ('K14', 'S'),
    'FLIP + S6F + K14': ('S6F', 'K14'),
    'FLIP + S6F + S + K14': ('S6F', 'S', 'K14'),
}


def solve_x(Aa, Bb, pct=PCT, target=TARGET):
    """largest x with percentile(Aa + x Bb, pct) <= target (x may be negative = deficit)."""
    lo, hi = -80.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.percentile(Aa + mid * Bb, pct) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def run_block(title, model='B', n=N_HID, c_rng=(C_LO, C_HI), ftot=FLIP_TOTAL, jmode=None, seed_off=0, reps=None,
              bases=BASES, extra=True):
    reps = reps or A.reps
    rng = np.random.default_rng(A.seed + seed_off)
    D = [draw(rng, model, n, c_rng, ftot) for _ in range(reps)]
    s = np.array([d['s'] for d in D])
    jj = np.array([0.0 if jmode is None else (d['j'] if jmode == 'typ' else 0.17) for d in D])
    xfix = np.array([0.002 + d['setupx'] * d['s'] + BLEND_H for d in D])
    xslope = np.array([d['s'] * d['kx'] / 3600 for d in D])
    print(f'\n# {title} (session {model}, N {n}, S6F c {c_rng}, FLIP {ftot:.2f} h, J {jmode or "none"}, {reps} paired sessions)')
    print('#   base natural end: median / p90 / p99, P(>8.0) | X budget x (T4-s / study) at p99 <= 8.0 [p95 <= 8.0, p99 <= 8.3]'
          ' | X wall at s 1 for that x (h)')
    out = {}
    for name, comps in bases.items():
        E = np.array([base_end(d, comps) for d in D]) + jj
        Aa = E + xfix
        x99 = solve_x(Aa, xslope)
        x95 = solve_x(Aa, xslope, pct=95.0)
        x99w = solve_x(Aa, xslope, target=8.3)
        wall1 = 0.002 + 0.055 + x99 * (n / 2) / 3600 + BLEND_H
        out[name] = x99
        flag = '' if x99 > 0 else '  <- NO ROOM: base + X setup alone is %.2f h over the line at p99' % (np.percentile(Aa, PCT) - TARGET)
        # marginal criterion for bases that already lean on guards: x at which X adds <= 1 / 2 points to P(natural end > 8.0)
        P0 = np.mean(E > TARGET)
        rel = []
        for dlt in (0.01, 0.02):
            lo, hi = 0.0, 200.0
            if np.mean(Aa > TARGET) - P0 > dlt:
                rel.append(0.0)
                continue
            for _ in range(50):
                mid = (lo + hi) / 2
                if np.mean(Aa + mid * xslope > TARGET) - P0 <= dlt:
                    lo = mid
                else:
                    hi = mid
            rel.append(lo)
        print(f'  {name:28s}: {np.median(E):.2f} / {np.percentile(E, 90):.2f} / {np.percentile(E, 99):.2f}, P(>8.0) {P0:.4f} '
              f'| x {x99:6.1f} [{x95:6.1f}, {x99w:6.1f}] | X wall {max(wall1, 0):.2f} h | marginal +1 / +2 pt of P(>8.0): '
              f'x {rel[0]:.1f} / {rel[1]:.1f}{flag}')
    if extra:
        # G2: X as compact_infer group 2 inside S6F, base FLIP + S6F + S: S6F per-study cost = max(c, g1 + g2)
        for gname, comps in (('in S6F (group 2), base FLIP + S6F + S', ('S6F', 'S')), ('in S6F (group 2), base FLIP + S6F', ('S6F',))):
            cc = np.array([d['c'] for d in D]); g1 = np.array([d['g1'] for d in D])
            fixed = np.array([d['base'] + 0.002 + d['setup6'] * d['s'] + BLEND_H + (fleet_dur(d, S_S) if 'S' in comps else 0.0)
                              for d in D]) + jj
            slope6 = np.array([d['s'] * d['k6'] / 3600 for d in D])

            def p99_g2(g2):
                return np.percentile(fixed + np.maximum(cc, g1 + g2) * slope6, PCT)

            def pgt_g2(g2):
                return np.mean(fixed + np.maximum(cc, g1 + g2) * slope6 > TARGET)

            def med_g2(g2):
                return np.median(fixed + np.maximum(cc, g1 + g2) * slope6)
            P0, M0 = pgt_g2(0.0), med_g2(0.0)
            res = []
            for crit in (lambda g: pgt_g2(g) - P0 <= 0.01, lambda g: pgt_g2(g) - P0 <= 0.02, lambda g: med_g2(g) - M0 <= 0.05):
                lo, hi = 0.0, 60.0
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if crit(mid):
                        lo = mid
                    else:
                        hi = mid
                res.append(lo)
            free = np.mean(np.maximum(0.0, cc - g1))
            print(f'  G2 {gname}: base P(>8.0) {P0:.4f}, median {M0:.2f} h | group-2 GPU g2 (s / study / T4) adding <= +1 pt / +2 pt '
                  f'to P(>8.0): {res[0]:.1f} / {res[1]:.1f}; adding <= .05 h to the median: {res[2]:.1f} | mean free GPU slack '
                  f'max(0, c - g1) {free:.1f} s')
            lo, hi = 0.0, 60.0
            if p99_g2(0.0) > TARGET:
                print(f'  G2 {gname}: strict p99 <= 8.0: NO ROOM even at g2 = 0 (p99 {p99_g2(0.0):.2f} h)')
                continue
            for _ in range(40):
                mid = (lo + hi) / 2
                if p99_g2(mid) <= TARGET:
                    lo = mid
                else:
                    hi = mid
            free = np.mean([max(0.0, d['c'] - d['g1']) for d in D])
            print(f'  G2 {gname}: group-2 GPU budget g2 <= {lo:.1f} s / study / T4 at p99 <= 8.0 (mean free GPU slack inside the '
                  f'S6F stage max(0, c - g1) = {free:.1f} s; a separate stage on the same base gets x = {out["FLIP + S6F + S" if "S" in comps else "FLIP + S6F"]:.1f})')
    return out


print('# sim_room_newfam.py MODEL; kernel h = age from NB_T0; X budget x = T4-seconds per hidden study for ONE new family')
print('# (each T4 scores n/2 studies; x includes X\'s own decode / crop / GPU / TTA; X setup U(.03,.08) h x s on top).')
main = run_block('MAIN: base session model', seed_off=0)
run_block('STRESS session model (heavier slow tail, GUESS)', model='STRESS', seed_off=1)
run_block('S6F loader-bound c U(11.5, 14.5) (MODEL: commit warm loader 24.0-26.7 s / worker / 2 workers; hidden 4-worker contention)',
          c_rng=(11.5, 14.5), seed_off=2)
run_block('J variants (final-pick form): + DEC + preflight U(.005, .05) h', jmode='typ', seed_off=3)
run_block('J at its 600 s cap (+0.17 h)', jmode='cap', seed_off=4, extra=False)
run_block('FLIP calibration 4.65 h (MEASURED window top)', ftot=4.65, seed_off=5, extra=False)
run_block('FLIP calibration 4.75 h (FLRM 57200578 MEASURED 4.72-4.80 h incl. RM .06-.16 MODEL: implies FLIP 4.56-4.74)',
          ftot=4.75, seed_off=6, extra=False)
run_block('hidden N 1,500 (+15 %)', n=1500, seed_off=7, extra=False)
run_block('pessimistic stack: loader-bound S6F c U(11.5, 14.5) + J typ + FLIP 4.65', c_rng=(11.5, 14.5), jmode='typ', ftot=4.65,
          seed_off=8)

# recalibration table for the S6F read (~13:45-14:30 IST): S6F stage wall at s ~ 1 is (S6F submit-to-score) - FLIP 4.62 h;
# c ~ (wall - 0.067) x 3600 / 650. Rows: fixed c -> x for the 3 requested bases.
print('\n# R. after the S6F hidden read: fixed S6F c (s / study / T4) -> X budget x at p99 <= 8.0 (session B, N 1300, no J)')
print('#    S6F wall at s 1 = 0.067 + 650 c / 3600 h -> expected S6F submit-to-score ~ 4.62 + that')
for c in (7.0, 9.0, 11.0, 12.5, 14.0, 16.0):
    o = run_block(f'S6F c fixed {c:.1f} (S6F wall {0.067 + 650 * c / 3600:.2f} h, S6F submit-to-score ~{4.62 + 0.067 + 650 * c / 3600:.2f} h)',
                  c_rng=(c, c), seed_off=20, reps=A.reps // 2,
                  bases={k: BASES[k] for k in ('FLIP + S6F', 'FLIP + S6F + S')}, extra=False)

print('\n# D. deterministic: X wall in h at s 1 for a budget x = 0.057 + 650 x / 3600 (+0.01 blend); 1 s / study = 0.18 h at N 1300.')
for x in (1, 2, 3, 5, 8, 10, 13, 15, 20):
    print(f'  x {x:4.1f} s -> X wall {0.002 + 0.055 + 650 * x / 3600 + BLEND_H:.2f} h (s 1), {0.002 + 0.055 * 1.3 + 650 * x * 1.3 / 3600 + BLEND_H:.2f} h (s 1.3)')
