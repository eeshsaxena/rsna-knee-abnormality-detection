#!/usr/bin/env python
# guard_paired_1015.py (2026-10-15 dir, 10-15, Claude fix agent; AO review finding F1). MODEL. Laptop CPU, numpy only, ~30 s.
# PAIRED comparison of the AO start guards G 6.0 / 6.5 / 7.0 with the runtime model of sim_ffs_runtime.py section D (its constants
# and run_ao / draw_b / draw_stress are loaded from that file unchanged, nothing re-implemented): every session is drawn ONCE and
# replayed under each guard from the same RNG state, so the guards differ only by the fleet start decision.
# A canary = the fleet SKIPPED (start guard) or HARD-STOPPED at the 8.3 h wall (both end in HF_FALLBACK, the Synovitis canary) or
# her r224 skip (V224 CANARY). Prints per base x session model x hidden study count: whether the base part is identical across
# guards, the count of sessions that canary at the higher guard but not at the lower one, and P(canary) / P(skip) / P(hard) /
# p99 / max / P(end > 8.0 h) / P(end > 9 h) per guard.
#   usage: python guard_paired_1015.py [--n 20000] [--seed 20261015]   (output: guard_paired_1015.txt)
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=20000)
ap.add_argument('--seed', type=int, default=20261015)
A = ap.parse_args()
SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_ffs_runtime.py')
src = open(SIM, encoding='utf-8').read()
ns = {'np': np}
exec(src[src.index('PRE = 0.42'):src.index('def r7_worker')], ns)          # constants (PRE, FLIP_TOTAL, N_FL, S_S, WALL, ...)
exec(src[src.index('# ---- D. AO 10-15'):src.index('def ao_mc')], ns)      # RM / HF / base guards, AO_CFGS, run_ao, draw_b, draw_stress
run_ao, draw_b, draw_stress = ns['run_ao'], ns['draw_b'], ns['draw_stress']
G = (6.0, 6.5, 7.0)
print('# guard_paired_1015 (MODEL; sim_ffs_runtime.py section D model, paired sessions, %d per row, seed %d)' % (A.n, A.seed))
print('# canary = fleet SKIPPED or HARDSTOP (HF_FALLBACK, Synovitis) or r224 skip (V224 CANARY); hi-not-lo = sessions that canary at')
print('# the higher guard but not at the lower one (0 = the higher guard never adds a canary)')
for rm in (False, True):
    for dname, draw in (('B', draw_b), ('STRESS', draw_stress)):
        for n in (1300, 1500):
            rng = np.random.default_rng(A.seed)
            R = {g: [] for g in G}
            for _ in range(A.n):
                s, q, r224, mf = draw(rng)
                st = rng.bit_generator.state
                for g in G:
                    rng.bit_generator.state = st
                    R[g].append(run_ao(rng, s, q, r224, mf, dict(guard=g, rm=rm), n=n))
            can = {g: np.array([r['fleet'] in ('SKIPPED', 'HARDSTOP') or r['bflag'] == 'R224_SKIP' for r in R[g]]) for g in G}
            same = all(R[6.0][i]['base'] == R[6.5][i]['base'] == R[7.0][i]['base'] for i in range(A.n))
            print(f"\n{'FLRM' if rm else 'FLIP'}-AO + S | session model {dname} | hidden N {n} | base part identical across guards: {same} | "
                  f"canary at 6.5 not at 6.0: {int(np.sum(can[6.5] & ~can[6.0]))} | canary at 7.0 not at 6.5: {int(np.sum(can[7.0] & ~can[6.5]))}")
            for g in G:
                e = np.array([r['end'] for r in R[g]]); fl = [r['fleet'] for r in R[g]]
                print(f"  G {g:.1f}: P(canary) {can[g].mean():.4f} P(skip) {fl.count('SKIPPED') / A.n:.4f} P(hard) {fl.count('HARDSTOP') / A.n:.4f} "
                      f"| kernel end median {np.median(e):.2f} p99 {np.percentile(e, 99):.2f} max {e.max():.2f} | P(end > 8.0 h) {np.mean(e > 8.0):.4f} "
                      f"P(end > 9 h) {np.mean(e > 9.0):.4f}")
