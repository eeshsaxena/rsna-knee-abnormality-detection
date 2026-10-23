#!/usr/bin/env python
# sim_probe_res.py OUT.json: resolution of ONE per-label-group probe (one change vs the current best), MODEL (sim_pl.py worlds).
# For each scenario, group and move: P(probe becomes team best), mean / sd of the public and population deltas, corr(pub, pop),
# P(pop > 0 | win), E[pop | win], E[pop | loss]. Gains in macro AUC.
import sys, json, argparse
import numpy as np
import sim_pl as S

S.S_GRID = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
S.F_GRID = np.array([0.0, 0.1, 0.2, 0.3])
MOVES = []
for g in S.GROUPS:
    for s in (0.35, 0.65, 0.2, 0.8):
        MOVES.append((g, 's', s, 0.0))
    for f in (0.1, 0.3):
        MOVES.append((g + ':f', 'f', 0.5, f))      # on the F20 base: f .2 -> .1 / .3 for the group


def one(args):
    seed, sc, npop = args
    rng = np.random.default_rng(seed)
    T = S.world_tables(rng, sc, npop)
    out = {}
    for name, kind, s, f in MOVES:
        g = name.split(':')[0]
        labs = S.GROUPS[g]
        fb = 0.2 if kind == 'f' else 0.0
        si0, fi0 = S.idx(S.S_GRID, 0.5), S.idx(S.F_GRID, fb)
        si1, fi1 = S.idx(S.S_GRID, s), S.idx(S.F_GRID, f if kind == 'f' else fb)
        d = {k: float(sum(T[k][c, si1, fi1] - T[k][c, si0, fi0] for c in labs) / 12) for k in T}
        out[f'{name}|{"s" if kind == "s" else "f"}={s if kind == "s" else f}'] = (d['pub'], d['prv'], d['pop'])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--worlds', type=int, default=300)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--npop', type=int, default=12000)
    ap.add_argument('--scen', default='null_tau0,tau005,tau010,tau020,struct010_tau005,antistruct010_tau005')
    A = ap.parse_args()
    jobs = [(777 + 10007 * i + w, S.SCEN[n], A.npop) for i, n in enumerate(A.scen.split(',')) for w in range(A.worlds)]
    from multiprocessing import Pool
    with Pool(A.procs) as P:
        res = P.map(one, jobs, chunksize=2)
    agg, k = {}, 0
    for n in A.scen.split(','):
        R = res[k:k + A.worlds]; k += A.worlds
        agg[n] = {}
        for key in R[0]:
            v = np.array([r[key] for r in R]); dp, dv, dq = v[:, 0], v[:, 1], v[:, 2]
            win = dp > 0
            agg[n][key] = {'P_win': float(win.mean()), 'pub': dp.mean(), 'pub_sd': dp.std(), 'pop': dq.mean(), 'pop_sd': dq.std(),
                           'prv_sd': dv.std(), 'corr_pub_pop': float(np.corrcoef(dp, dq)[0, 1]) if dq.std() > 0 else None,
                           'P_pop_gt0_win': float((dq[win] > 0).mean()) if win.any() else None,
                           'E_pop_win': float(dq[win].mean()) if win.any() else None,
                           'E_pop_loss': float(dq[~win].mean()) if (~win).any() else None,
                           'E_prv_win': float(dv[win].mean()) if win.any() else None}
            for th in (1e-4, 2e-4):     # magnitude-aware read: count a win only if it passed enough frozen teams (MODEL)
                w2 = dp > th
                agg[n][key]['P_win_th%d' % int(th * 1e4)] = float(w2.mean())
                agg[n][key]['E_pop_win_th%d' % int(th * 1e4)] = float(dq[w2].mean()) if w2.any() else None
                agg[n][key]['P_pop_gt0_win_th%d' % int(th * 1e4)] = float((dq[w2] > 0).mean()) if w2.any() else None
    json.dump({'worlds': A.worlds, 'agg': agg, 'scen': {n: S.SCEN[n] for n in A.scen.split(',')}}, open(A.out, 'w'), indent=1, default=float)
    for n, v in agg.items():
        print('\n###', n, json.dumps(S.SCEN[n]))
        print('%-22s %6s %7s %6s %7s %6s %6s %8s %8s %8s' % ('move', 'P(win)', 'pub', 'pub_sd', 'pop', 'pop_sd', 'corr', 'P(pop>0|w)',
                                                              'E[pop|w]', 'E[pop|l]'))
        for key, r in v.items():
            f = lambda x, m=1e4, fmt='%+8.2f': (fmt % (x * m)) if x is not None else '       -'
            print('%-22s %6.2f %+7.2f %6.2f %+7.2f %6.2f %6s %8s %s %s' % (key, r['P_win'], r['pub'] * 1e4, r['pub_sd'] * 1e4, r['pop'] * 1e4,
                  r['pop_sd'] * 1e4, '%.2f' % r['corr_pub_pop'] if r['corr_pub_pop'] is not None else '-',
                  '%.2f' % r['P_pop_gt0_win'] if r['P_pop_gt0_win'] is not None else '-', f(r['E_pop_win']), f(r['E_pop_loss'])))
