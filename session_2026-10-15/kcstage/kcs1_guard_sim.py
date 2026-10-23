#!/usr/bin/env python
# kcs1_guard_sim.py (10-15, Claude subagent, kcstage; verification of finding KCS1). MODEL only, laptop CPU, seconds.
# Re-uses sim_s6fkc_runtime.py's session model VERBATIM (its source up to the GRID section is exec'd: S6F stage, KC wall, mc(), line())
# and answers: how much of the visible Baker's canary does the start guard G_KC cause, what the 8.2 h hard wall already bounds, and what
# a SIGTERM-flush of partial serve shards (the finding's optional code fix) would rescue.
#   P(canary) = P(KC skip) + P(KC cut): both fire Baker's = 0.5 on every row = UNREAD under R1 (one slot spent, no read).
#   slots per readable KC result = 1 / P(KC used).
#   worst case (no sampling): KC starts only if stop_at - start >= min_run; the driver stops its workers at hard - 180 s; the RUN cell
#   SIGKILLs at hard = min(start + budget, NB_T0 + 8.2 h), waits <= 120 s; blend + receipts seconds -> end <= NB_T0 + 8.2 h + ~0.04 h,
#   whatever G_KC is; the S6F stage has its own NB_T0 + 8.2 h wall. The comp limit is 9 h (comp_pages.txt).
#   usage: py -3.12 -I kcs1_guard_sim.py [--reps 3000] [--seed 0]   (output: kcs1_guard_sim.txt)
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--reps', type=int, default=3000)
ap.add_argument('--seed', type=int, default=0)
A_ = ap.parse_args()
src = open(os.path.join(HERE, 'sim_s6fkc_runtime.py'), encoding='utf-8').read()
cut = src.index('\nGRID = []')
S = {'__name__': 'sim_s6fkc_runtime_defs'}
_argv = sys.argv
sys.argv = ['sim_s6fkc_runtime.py', '--reps', str(A_.reps), '--seed', str(A_.seed)]
exec(compile(src[:cut], 'sim_s6fkc_runtime.py', 'exec'), S)          # defines run_s6f, kc_wall, run_all, mc, line, rng (seeded)
sys.argv = _argv
mc, np_ = S['mc'], np
C_LO, C_HI = S['C_LO'], S['C_HI']


def stats(E, M, S6, ST):
    n = len(M)
    used = sum(1 for m in M if m == 'KC') / n
    skip = sum(1 for m in M if m.startswith('SKIP')) / n
    cutp = M.count('CUT') / n
    return dict(used=used, skip=skip, cut=cutp, canary=skip + cutp, p99=float(np.percentile(E, 99)), mx=float(E.max()),
                gt8=float(np.mean(E > 8.0)), st_med=float(np.median(ST)), slots=1.0 / used if used else float('inf'))


CFG = [('G_KC 6.9 / budget 1.5 (built)', dict(G=6.9, budget=1.5, deadline=8.2)),
       ('G_KC 7.0 / budget 1.5', dict(G=7.0, budget=1.5, deadline=8.2)),
       ('G_KC 7.2 / budget 1.5', dict(G=7.2, budget=1.5, deadline=8.2)),
       ('G_KC 7.4 / budget 1.5', dict(G=7.4, budget=1.5, deadline=8.2)),
       ('no guard (G 99 / budget 99), wall 8.2 only', dict(G=99, budget=99, deadline=8.2))]
REG = [('A     c_kc U(1.0,2.2) S6 c U(6.5,13)', dict(kc_rng=(1.0, 2.2), p_tail=0.05, c_rng=(C_LO, C_HI))),
       ('A-hi  c_kc U(1.0,2.2) S6 c U(10,13) ', dict(kc_rng=(1.0, 2.2), p_tail=0.05, c_rng=(10.0, 13.0))),
       ('STR   c_kc U(1.6,2.6) S6 c U(6.5,13) tail .10', dict(kc_rng=(1.6, 2.6), p_tail=0.10, c_rng=(C_LO, C_HI))),
       ('STR-hi c_kc U(1.6,2.6) S6 c U(10,13) tail .10', dict(kc_rng=(1.6, 2.6), p_tail=0.10, c_rng=(10.0, 13.0)))]
print('# KCS1 check (MODEL, %d sessions per cell, seed %d; session model = sim_s6fkc_runtime.py verbatim)' % (A_.reps, A_.seed))
print('# columns: P(KC used) | P(canary) = P(skip) + P(cut) | slots per readable KC result | kernel end p99 / max h | P(end > 8.0) |'
      ' submit-to-score median h')
T = {}
for rn, rk in REG:
    print('\n## regime %s' % rn)
    for cn, cfg in CFG:
        E, M, S6, ST = mc(cfg, A_.reps, kc_rng=rk['kc_rng'], p_tail=rk['p_tail'], c_rng=rk['c_rng'])
        s = stats(E, M, S6, ST)
        T[(rn, cn)] = s
        print('  %-44s used %.3f | canary %.3f (skip %.3f cut %.3f) | slots/read %.2f | p99 %.2f max %.2f | P(>8.0) %.3f | sts %.2f'
              % (cn, s['used'], s['canary'], s['skip'], s['cut'], s['slots'], s['p99'], s['mx'], s['gt8'], s['st_med']))

# ---- partial-shard flush (finding's optional fix): of the CUT runs, how many would reach >= 98 % coverage at the driver stop? ----
# explicit-component copy of kc_wall (same distributions) so the serve part is separable; MODEL.
rng2 = np.random.default_rng(A_.seed + 1)


def kc_parts(s, c_kc, n=1300, members=2):
    gpu = members * rng2.uniform(0.27, 0.31) * 1.06 / 2.0
    per = max(c_kc, gpu)
    noise = float(np.clip(rng2.normal(1.0, 0.05), 0.85, 1.2))
    pre = (rng2.uniform(0.02, 0.05) + rng2.uniform(0.05, 0.08)) * s
    return pre, n * per * noise / 3600.0 * s


print('\n## partial-shard flush value (MODEL, regime A, sessions drawn as in mc(), S6F stage = run_s6f): of all sessions, the KC runs that'
      ' start, the CUT ones, and those a SIGTERM-flush would rescue (>= 98 % of studies scored at the driver stop)')
for G, bud in ((6.9, 1.5), (7.2, 1.5), (99, 99)):
    cuts = resc = runs = 0
    for _ in range(A_.reps):
        s = float(np.exp(rng2.normal(0, 0.08)) * (1.3 if rng2.random() < 0.05 else 1.0))
        q = rng2.uniform(0.0, 0.3); r224 = rng2.uniform(0.15, 0.30)
        c = rng2.uniform(C_LO, C_HI); setup = rng2.uniform(S['SETUP_LO'], S['SETUP_HI']); c_kc = rng2.uniform(1.0, 2.2)
        a = S['run_s6f'](rng2, s, q, r224, c, setup, S['S6F'])['end'] + 0.003
        if a > G:
            continue
        hard = min(a + bud, 8.2)
        stop = hard - 180 / 3600.0
        if stop - a < 0.5:
            continue
        runs += 1
        pre, serve = kc_parts(s, c_kc)
        if a + pre + serve <= stop:
            continue
        cuts += 1
        resc += (stop - a - pre) / serve >= 0.98
    print('  G %-4s: started %d of %d, cut %d (%.4f of all), rescued by a flush %d (%.4f of all)' % (G, runs, A_.reps, cuts, cuts / A_.reps,
                                                                                               resc, resc / A_.reps))
print('\n## worst-case end (no sampling): hard wall NB_T0 + 8.2 h; RUN cell kill at hard + wait <= 120 s + blend ~0.01 h -> <= 8.25 h'
      ' from NB_T0, for every G_KC; + pre-NB_T0 time (J DEC preflight cap 600 s) -> <= ~8.42 h < 9 h comp limit.')
