#!/usr/bin/env python
# sim_s6fkc_runtime.py (10-15, Claude subagent, kcstage): hidden-run wall of V224-FLIP-AO + S6fall (the S6F kernel) + the APPENDED
# KC stage (build_v224s6fkc.py), and the choice of the KC start guard G_KC / budget so the kernel end p99 <= 8.0 h (A6 G3). MODEL.
# The base part is s6f/sim_s6f_runtime.py's session model and S6F stage, COPIED unchanged (that file runs at import), with the S6F
# policy of the pushed kernel (G 5.75, budget 3.3, deadline 8.2). KC starts right after the S6F BLEND cell (S6F used, cut or skipped).
# KC stage inputs:
#   MEASURED (box, 10-15, kcstage/box/measure_decode.sh, RSNA train studies, cores 28-31, warm page cache):
#     kc_serve decode (the table path == kc_worker rows on 24/24 studies) serial 2.57 s/study on 1 core; 2 processes x --procs 2 on
#     4 cores 0.54-0.65 s/study wall (0.92-2.04 s/study/process under other jobs' contention); A6: 5.9 s serial / 1.55 s at 4 procs
#     with the T_a loaders sharing the cores. Side pass (the side kernel's study(), header only): Kaggle 0.156 s/study wall at 4 procs
#     (MEASURED, rsna-kc-side v2 on 4,407 train studies), box 0.152 s/study (24 studies, PASS vs the Kaggle table on all 54 columns).
#   MODEL Kaggle: 4 vCPUs (2 physical cores, HT) for 4 decoders ~2.5 core-equivalents, per-core speed x1.0-1.5 of the box, cold
#     mounted-data reads x1.0-1.5 -> decode-bound KC cost c_kc ~ U(1.0, 2.2) s per study (effective, both T4 processes together);
#     GPU per member per study 1.64 TFLOP / 5.3-6.0 TFLOP/s (fleet microbench) = 0.27-0.31 s x (1 + both-canon ~6 %), M members,
#     2 T4 in parallel -> 0.30-0.50 s/study for M 2-3: never the bound unless c_kc < 0.5 s.
#   setup (manifest + member sha ~0.3-0.6 GB, code copy) + driver env probe + model load / CUDA init: 0.02-0.05 h; side 0.05-0.08 h;
#   merge + blend + receipts ~0.005 h.
#   usage: python sim_s6fkc_runtime.py [--reps 4000] [--seed 0]
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reps', type=int, default=4000)
ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()

# ---- copied from s6f/sim_s6f_runtime.py (unchanged) ----
PRE = 0.42
FLIP_TOTAL = 4.62
N_HID = 1300
BLEND_H = 0.01
HARD = 9.0
C_LO, C_HI = 6.5, 13.0
SETUP_LO, SETUP_HI = 0.03, 0.08
CI_MARGIN_H = (20 * 60 + 4 * 60) / 3600
S6F = dict(G=5.75, budget=3.3, deadline=8.2)                     # the pushed S6F kernel's policy


def r7_worker(rng, n, flip_h, pre_h):
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n)).astype(int)
    noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    rf = (flip_h * 3600 - 250) / files.sum()
    return (pre_h * 3600 + 250 + (rf * files * noise).sum()) / 3600


def s6_part(rng, n, c, s):
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n))
    order = np.argsort(-files, kind='mergesort')
    order = np.concatenate([order[::2], order[1::2]])
    f = files[order]
    noise = rng.normal(1.0, 0.12, n).clip(0.6, 1.6)
    per = c * s * f / f.mean() * noise
    return np.cumsum(per) / 3600


def run_s6f(rng, s, q, r224, c, setup, cfg, n=N_HID):
    flip_h = FLIP_TOTAL - q - PRE - r224
    base_end = max(r7_worker(rng, n // 2, flip_h * s * n / 1300, PRE * s) for _ in range(2)) + r224 * s
    start = base_end + 0.002
    if start > cfg['G']:
        return dict(base=base_end, start=start, end=start + BLEND_H, mode='SKIP', cov=0.0)
    hard = min(start + cfg['budget'], cfg['deadline'])
    ci_deadline = hard - CI_MARGIN_H
    t0 = start + setup * s
    done, ends = 0, []
    for _ in range(2):
        cum = t0 + s6_part(rng, n // 2, c, s)
        k = int(np.searchsorted(cum, ci_deadline, side='right'))
        k_done = min(len(cum), k + (1 if k < len(cum) else 0))
        done += k_done
        ends.append(cum[k_done - 1] if k_done else t0)
    end = max(ends)
    killed = end > hard
    end = min(end, hard) + BLEND_H
    cov = done / n if not killed else min(done / n, 0.0)
    mode = 'S6F' if cov >= 0.98 else 'CUT'
    return dict(base=base_end, start=start, end=end, mode=mode, cov=cov)
# ---- end of the copied part ----


KC_SETUP = (0.02, 0.05)         # h (setup cell sha + driver env + model load / CUDA init)
KC_SIDE = (0.05, 0.08)          # h for N_HID (0.156 s/study MEASURED on Kaggle -> 0.056 h; + contention)
KC_C = (1.0, 2.2)               # s / study, decode-bound, both T4 processes together (MODEL)
KC_TAIL_H = 0.005               # merge + blend + receipts
KC_STOP_MARGIN_H = 180 / 3600.0


def kc_wall(rng, s, c_kc, n=N_HID, members=2):
    gpu = members * rng.uniform(0.27, 0.31) * 1.06 / 2.0             # s / study effective over 2 T4
    per = max(c_kc, gpu)
    noise = float(np.clip(rng.normal(1.0, 0.05), 0.85, 1.2))         # whole-stage noise (per-study noise averages out over 650 / GPU)
    return (rng.uniform(*KC_SETUP) + rng.uniform(*KC_SIDE) * n / N_HID + n * per * noise / 3600.0) * s


def run_all(rng, s, q, r224, c, setup, c_kc, kcfg, n=N_HID, members=2):
    r = run_s6f(rng, s, q, r224, c, setup, S6F, n)
    start = r['end'] + 0.003                                          # KC SETUP cell right after S6F_DONE
    out = dict(s6f_mode=r['mode'], s6f_end=r['end'], kc_start=start)
    if start > kcfg['G']:
        return dict(out, kc_mode='SKIP', end=start + KC_TAIL_H)
    hard = min(start + kcfg['budget'], kcfg['deadline'])
    stop_at = hard - KC_STOP_MARGIN_H
    if stop_at - start < kcfg.get('min_run_h', 0.5):
        return dict(out, kc_mode='SKIP_NO_TIME', end=start + KC_TAIL_H)
    w = kc_wall(rng, s, c_kc, n, members)
    if start + w <= stop_at:
        return dict(out, kc_mode='KC', end=start + w + KC_TAIL_H, kc_wall=w)
    return dict(out, kc_mode='CUT', end=stop_at + 0.01 + KC_TAIL_H, kc_wall=w)    # driver kills its workers -> coverage < 98 % -> canary


rng = np.random.default_rng(A.seed)


def mc(kcfg, reps, kc_rng=KC_C, n=N_HID, s_tail=1.3, p_tail=0.05, members=2, c_rng=(C_LO, C_HI)):
    E, M, S6, ST = [], [], [], []
    for _ in range(reps):
        s = float(np.exp(rng.normal(0, 0.08)) * (s_tail if rng.random() < p_tail else 1.0))
        q = rng.uniform(0.0, 0.3); r224 = rng.uniform(0.15, 0.30)
        c = rng.uniform(*c_rng); setup = rng.uniform(SETUP_LO, SETUP_HI); c_kc = rng.uniform(*kc_rng)
        r = run_all(rng, s, q, r224, c, setup, c_kc, kcfg, n, members)
        E.append(r['end']); M.append(r['kc_mode']); S6.append(r['s6f_mode']); ST.append(r['end'] + q)
    return np.array(E), M, S6, np.array(ST)


def line(name, E, M, S6, ST):
    n = len(M)
    used = sum(1 for m, s6 in zip(M, S6) if m == 'KC')
    both = sum(1 for m, s6 in zip(M, S6) if m == 'KC' and s6 == 'S6F')
    return (f'  {name}: end median {np.median(E):.2f} p90 {np.percentile(E, 90):.2f} p99 {np.percentile(E, 99):.2f} max {E.max():.2f} | '
            f'P(KC used) {used / n:.3f} P(KC skip) {sum(1 for m in M if m.startswith("SKIP")) / n:.3f} P(KC cut) {M.count("CUT") / n:.3f} '
            f'| P(S6F and KC both used) {both / n:.3f} | P(end > 8.0) {np.mean(E > 8.0):.3f} P(end > 8.25) {np.mean(E > 8.25):.3f} '
            f'| submit-to-score median {np.median(ST):.2f} p90 {np.percentile(ST, 90):.2f}')


GRID = []
for G in (6.2, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2):
    for bud in (1.5,):
        GRID.append(('G_KC %.2f / budget %.1f / deadline 8.2' % (G, bud), dict(G=G, budget=bud, deadline=8.2)))
GRID.append(('no guard (G 99) / budget 99 / deadline 8.2', dict(G=99, budget=99, deadline=8.2)))
print('# A. Monte Carlo (%d sessions per config). S6F stage as sim_s6f_runtime.py (G 5.75 / 3.3 / 8.2); KC c_kc ~ U(%.1f, %.1f) s/study,'
      % (A.reps, *KC_C))
print('#    side %s h, setup %s h, M 2 members, N %d, s = exp(N(0, .08)) x (1.3 w.p. .05). Kernel end h (age from NB_T0).' % (KC_SIDE, KC_SETUP, N_HID))
res = {}
for name, cfg in GRID:
    E, M, S6, ST = mc(cfg, A.reps)
    res[name] = (E, M, S6, ST)
    print(line(name, E, M, S6, ST))
STRESS = dict(kc_rng=(1.6, 2.6), p_tail=0.10)          # pessimistic KC cost (box serial 2.57 s x slow core x cold I/O) + more slow sessions
print('\n# B. the same configs under STRESS: c_kc ~ U%s s/study, slow-session share %.2f' % (STRESS['kc_rng'], STRESS['p_tail']))
stress = {}
for name, cfg in GRID:
    E, M, S6, ST = mc(cfg, A.reps, kc_rng=STRESS['kc_rng'], p_tail=STRESS['p_tail'])
    stress[name] = float(np.percentile(E, 99))
    print(line('STRESS ' + name, E, M, S6, ST))
ok = [(name, cfg) for name, cfg in GRID if np.percentile(res[name][0], 99) <= 8.0 and stress[name] <= 8.0]
best = max(ok, key=lambda nc: (sum(1 for m in res[nc[0]][1] if m == 'KC'), -nc[1]['G'])) if ok else None
print('\n# B2. chosen: p99 <= 8.0 h in A AND under STRESS, then the highest P(KC used) in A:', best[0] if best else 'NONE')
REC = best[1] if best else None
if REC:
    print('\n# C. sensitivity of the chosen config (rows: KC cost range, members, hidden N, slow-session tail, S6fall cost)')
    for kc_rng, mem, n, st, pt, crng in (((1.0, 2.2), 2, 1300, 1.3, .05, (C_LO, C_HI)), ((0.6, 1.2), 2, 1300, 1.3, .05, (C_LO, C_HI)),
                                         ((1.5, 2.2), 2, 1300, 1.3, .05, (C_LO, C_HI)), ((2.2, 3.0), 2, 1300, 1.3, .05, (C_LO, C_HI)),
                                         ((1.0, 2.2), 3, 1300, 1.3, .05, (C_LO, C_HI)), ((1.0, 2.2), 2, 1500, 1.3, .05, (C_LO, C_HI)),
                                         ((1.0, 2.2), 2, 1300, 1.3, .15, (C_LO, C_HI)), ((1.0, 2.2), 2, 1300, 1.45, .05, (C_LO, C_HI)),
                                         ((1.0, 2.2), 2, 1300, 1.3, .05, (6.5, 9.0)), ((1.0, 2.2), 2, 1300, 1.3, .05, (10.0, 13.0))):
        E, M, S6, ST = mc(REC, A.reps, kc_rng, n, st, pt, mem, crng)
        print(line('c_kc %s M %d N %d tail x%.2f p %.2f S6 c %s' % (kc_rng, mem, n, st, pt, crng), E, M, S6, ST))
    print('\n# D. bounds (no sampling): KC starts only at age <= G_KC; the driver stops its serve processes at hard - 3 min and the RUN cell')
    print('#    kills every KC process group at hard = min(start + budget, NB_T0 + 8.2 h); so the kernel cannot pass ~8.2 h because of KC.')
    for c_kc in (1.0, 1.6, 2.2, 3.0):
        w = 0.035 + 0.065 + N_HID * c_kc / 3600
        print(f'  c_kc {c_kc:.1f} s: KC wall {w:.2f} h (s 1.0), {w * 1.3:.2f} h (s 1.3); a start AT G_KC {REC["G"]:.2f} ends '
              f'{REC["G"] + w:.2f} / {REC["G"] + w * 1.3:.2f} h')
