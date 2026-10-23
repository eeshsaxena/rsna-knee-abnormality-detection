#!/usr/bin/env python
# sim_pl.py (2026-10-14): MODEL of per-label LB probing (coordinate ascent on per-label blend weights, read with the API
# best-id reader) on the V224 structure. Extends research960/lbprobe/sim_noise.py (same latent binormal components, same error
# correlations, same 1300 / 390 / 910 test split) with PER-LABEL component strengths.
#
# Structure (rank space, per label c):  H_c = s_c * (.65 pct(r384) + .35 pct(r224)) + (1 - s_c) * pct(R7)        (V224, s = .5)
#                                        final_c = (1 - f_c) pct(H_c) + f_c pct(F_c)                             (F base f = .2)
# Per-label truth (GUESS, swept): A_R7,c - A_Raptor,c = mu_c + tau * z_c (z ~ N(0,1)); mu_c = +-delta_s on the prior groups
# (OA: Raptor better; Fracture / MCL / Contusion: R7 better) when the "structured" prior is TRUE, else 0. F per-label
# deviation tau_F * z'. Levels: component AUC = .949 + .5 (gold58 N_c - mean) (MEASURED gold58 profile, shrunk: GUESS).
# Prevalences: train report-label rates (MEASURED on train; GUESS for test). Hidden-rerun noise (MEASURED ~4e-6) ignored.
#
# Read model: a probe = current best state with ONE coordinate (a label group's s or f) moved one step; it becomes the team best
# iff its public macro AUC > the current best's (binary, the API best-id read). Accepted -> new current state.
# Reported per policy: public gain, private gain (910 studies, what counts), population gain (expected private), P(< 0).
# usage: python sim_pl.py OUT.json [--worlds 120] [--procs 4] [--seed 0] [--quick]
import sys, json, time, argparse, itertools
import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']
PREV = np.array([.278, .174, .533, .254, .367, .247, .452, .605, .132, .253, .294, .071])      # MEASURED train report rates
GOLD_N = np.array([.964, .943, .982, .896, .981, .776, .860, .973, .829, .976, .937, .958])    # MEASURED gold58 N (r384)
LEVEL = .949 + .5 * (GOLD_N - GOLD_N.mean())                                                    # GUESS
N, NPUB = 1300, 390                                                                             # MEASURED split sizes
GROUPS = {'OA': [4, 5, 6], 'TRAUMA': [1, 10, 11], 'MEN': [2, 3], 'FLUID': [7, 8, 9], 'ACL': [0]}
PRIOR_DIR = {'OA': +1, 'TRAUMA': -1, 'MEN': 0, 'FLUID': 0, 'ACL': 0}                            # +1 = Raptor share up
S_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
F_GRID = np.round(np.arange(0.0, 0.501, 0.05), 2)


RHO_RR, RHO_34 = .82, .93      # MODEL, calib_pl.py: R7 ~ Raptor errors .82 puts V224 at ~.956 from ~.949-.950 parts


def corr(rho_f):
    C = np.full((4, 4), RHO_RR)
    np.fill_diagonal(C, 1.0)
    C[1, 2] = C[2, 1] = RHO_34
    C[3, :3] = C[:3, 3] = rho_f
    return np.linalg.cholesky(C)


def auc_from_scores(s, y):
    r = rankdata(s)
    npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def pct(x):
    return rankdata(x) / len(x)


def world_tables(rng, sc, npop):
    """AUC tables [set][label, s_idx, f_idx] for set in pub / prv / pop."""
    Lc = corr(sc['rho_f'])
    z = rng.standard_normal(12)
    zf = rng.standard_normal(12)
    mu = np.zeros(12)
    if sc['structured']:
        for i in GROUPS['OA']:
            mu[i] = -sc['delta_s']          # R7 minus Raptor < 0: Raptor better on OA
        for i in GROUPS['TRAUMA']:
            mu[i] = +sc['delta_s']
    d = mu + sc['tau'] * z
    A = np.zeros((12, 4))
    A[:, 0] = LEVEL + .0005 + d / 2
    A[:, 1] = LEVEL - .0005 - d / 2
    A[:, 2] = A[:, 1] - .004          # r224 .945 vs r384 .949 (CLAIMED by their authors)
    A[:, 3] = LEVEL - .005 + sc['tau_f'] * zf
    A = np.clip(A, .60, .995)
    alpha = np.sqrt(2) * ndtri(A)
    perm = rng.permutation(N)
    pub, prv = perm[:NPUB], perm[NPUB:]
    T = {k: np.zeros((12, len(S_GRID), len(F_GRID))) for k in ('pub', 'prv', 'pop')}
    for c in range(12):
        for setname, n in (('test', N), ('pop', npop)):
            y = (rng.random(n) < PREV[c]).astype(int)
            if y.sum() < 2:
                y[:2] = 1
            S = alpha[c][None, :] * y[:, None] + rng.standard_normal((n, 4)) @ Lc.T
            P = np.column_stack([pct(S[:, m]) for m in range(4)])     # ranks over the whole scored set (test: all 1300)
            mix = .65 * P[:, 1] + .35 * P[:, 2]
            for si, s in enumerate(S_GRID):
                H = s * mix + (1 - s) * P[:, 0]
                pH = pct(H)
                for fi, f in enumerate(F_GRID):
                    fin = H if f == 0 else (1 - f) * pH + f * P[:, 3]
                    if setname == 'test':
                        T['pub'][c, si, fi] = auc_from_scores(fin[pub], y[pub])
                        T['prv'][c, si, fi] = auc_from_scores(fin[prv], y[prv])
                    else:
                        T['pop'][c, si, fi] = auc_from_scores(fin, y)
    return T


def idx(grid, v):
    return int(np.argmin(np.abs(grid - v)))


def macro(T, st):
    si = [idx(S_GRID, st[c][0]) for c in range(12)]
    fi = [idx(F_GRID, st[c][1]) for c in range(12)]
    return {k: float(np.mean([T[k][c, si[c], fi[c]] for c in range(12)])) for k in T}


def run_policy(T, base, coords, budget, steps=(0.15, 0.10, 0.05), fsteps=(0.10, 0.05)):
    """Greedy coordinate ascent with binary accept on public. coords: list of (name, labels, kind, prior_dir)."""
    st = {c: list(base[c]) for c in range(12)}
    cur = macro(T, st)
    used, acc, log = 0, 0, []
    for p in range(len(steps)):
        for name, labs, kind, pd_ in coords:
            step = (steps if kind == 's' else fsteps)[min(p, len((steps if kind == 's' else fsteps)) - 1)]
            dirs = [pd_, -pd_] if pd_ != 0 else [+1, -1]
            moved = False
            for dr in dirs:
                while used < budget:
                    k = 0 if kind == 's' else 1
                    grid = S_GRID if kind == 's' else F_GRID
                    cand = {c: list(v) for c, v in st.items()}
                    ok = True
                    for c in labs:
                        nv = round(cand[c][k] + dr * step, 2)
                        if nv < grid.min() - 1e-9 or nv > grid.max() + 1e-9:
                            ok = False
                        cand[c][k] = nv
                    if not ok:
                        break
                    m = macro(T, cand)
                    used += 1
                    win = m['pub'] > cur['pub']
                    log.append((name, kind, dr * step, win, m['pub'] - cur['pub'], m['pop'] - cur['pop']))
                    if win:
                        st, cur, moved, acc = cand, m, True, acc + 1
                    else:
                        break
                if moved or used >= budget:
                    break
            if used >= budget:
                break
        if used >= budget:
            break
    return st, cur, used, acc, log


def coords_for(policy, with_f):
    if policy == 'label':
        cs = [(LABELS[c], [c], 's', next((PRIOR_DIR[g] for g, l in GROUPS.items() if c in l), 0)) for c in range(12)]
        if with_f:
            cs += [(LABELS[c] + ':f', [c], 'f', 0) for c in range(12)]
        return cs
    order = ['OA', 'TRAUMA', 'MEN', 'FLUID', 'ACL'] if policy in ('group', 'group_noprior') else ['OA', 'TRAUMA']
    cs = [(g, GROUPS[g], 's', PRIOR_DIR[g] if policy != 'group_noprior' else 0) for g in order]
    if with_f and policy != 'prior2':
        cs += [(g + ':f', GROUPS[g], 'f', 0) for g in order]
    return cs


def one_world(args):
    seed, sc, npop, budgets = args
    rng = np.random.default_rng(seed)
    T = world_tables(rng, sc, npop)
    out = {}
    for base_name, fb in (('V224', 0.0), ('V224F20', 0.2)):
        base = {c: (0.5, fb) for c in range(12)}
        b = macro(T, base)
        with_f = fb > 0
        for policy in ('prior2', 'group', 'group_noprior', 'label'):
            for B in budgets:
                if policy == 'prior2' and B > 4:
                    continue
                st, cur, used, acc, log = run_policy(T, base, coords_for(policy, with_f), B)
                out[f'{base_name}|{policy}|{B}'] = (cur['pub'] - b['pub'], cur['prv'] - b['prv'], cur['pop'] - b['pop'], used, acc,
                                                    [(l[0], l[2], l[3], l[4], l[5]) for l in log[:3]])
                # shrink-to-half variant (final map = base + .5 (ascent - base)), snapped to the grid
                sh = {c: (round(round((0.5 + (st[c][0] - 0.5) / 2) * 20) / 20, 2), round(round((fb + (st[c][1] - fb) / 2) * 20) / 20, 2))
                      for c in range(12)}
                m = macro(T, sh)
                out[f'{base_name}|{policy}+shrink|{B}'] = (m['pub'] - b['pub'], m['prv'] - b['prv'], m['pop'] - b['pop'], used, acc, [])
        # bounds: per-label public argmax (infinite probing, max overfit) and population argmax (oracle)
        fsl = slice(None) if with_f else slice(0, 1)
        for kind in ('pubopt', 'oracle'):
            key = 'pub' if kind == 'pubopt' else 'pop'
            st = {}
            for c in range(12):
                sub = T[key][c][:, fsl]
                si, fi = np.unravel_index(np.argmax(sub), sub.shape)
                st[c] = (S_GRID[si], F_GRID[fsl][fi] if with_f else 0.0)
            m = macro(T, st)
            out[f'{base_name}|{kind}|-'] = (m['pub'] - b['pub'], m['prv'] - b['prv'], m['pop'] - b['pop'], 0, 0, [])
        out[f'{base_name}|base_level|-'] = (b['pub'], b['prv'], b['pop'], 0, 0, [])
    return out


# rho_f .85 (MODEL, calib_pl.py): F at .2 adds ~+2.5e-4 and F .3 < F .2 (MEASURED signs: HFR7-RR > HR7-RR; RR-F30 lost to it)
SCEN = {
    'null_tau0': dict(tau=0.0, tau_f=0.0, structured=False, delta_s=0.0, rho_f=.85),
    'tau005': dict(tau=0.005, tau_f=0.005, structured=False, delta_s=0.0, rho_f=.85),
    'tau010': dict(tau=0.010, tau_f=0.010, structured=False, delta_s=0.0, rho_f=.85),
    'tau020': dict(tau=0.020, tau_f=0.020, structured=False, delta_s=0.0, rho_f=.85),
    'struct010_tau005': dict(tau=0.005, tau_f=0.005, structured=True, delta_s=0.010, rho_f=.85),
    'struct020_tau005': dict(tau=0.005, tau_f=0.005, structured=True, delta_s=0.020, rho_f=.85),
    'antistruct010_tau005': dict(tau=0.005, tau_f=0.005, structured=True, delta_s=-0.010, rho_f=.85),
    'tau010_rhof83': dict(tau=0.010, tau_f=0.010, structured=False, delta_s=0.0, rho_f=.83),
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--worlds', type=int, default=120)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--npop', type=int, default=12000)
    ap.add_argument('--scen', default=','.join(SCEN))
    A = ap.parse_args()
    budgets = (2, 4, 6, 12, 24, 35)
    jobs = []
    for si_, name in enumerate(A.scen.split(',')):
        for w in range(A.worlds):
            jobs.append((A.seed * 1000003 + si_ * 10007 + w, SCEN[name], A.npop, budgets))
    t0 = time.time()
    if A.procs > 1:
        from multiprocessing import Pool
        with Pool(A.procs) as P:
            res = P.map(one_world, jobs, chunksize=1)
    else:
        res = [one_world(j) for j in jobs]
    agg = {}
    k = 0
    for name in A.scen.split(','):
        R = res[k:k + A.worlds]
        k += A.worlds
        agg[name] = {}
        for key in R[0]:
            v = np.array([r[key][:5] for r in R], dtype=float)
            agg[name][key] = {'pub': v[:, 0].mean(), 'pub_sd': v[:, 0].std(), 'prv': v[:, 1].mean(), 'prv_sd': v[:, 1].std(),
                              'pop': v[:, 2].mean(), 'pop_sd': v[:, 2].std(), 'P_prv_lt0': float((v[:, 1] < 0).mean()),
                              'P_pop_lt0': float((v[:, 2] < 0).mean()), 'P_pop_gt0': float((v[:, 2] > 0).mean()),
                              'used': v[:, 3].mean(), 'acc': v[:, 4].mean(),
                              'pub_q90': float(np.quantile(v[:, 0], .9)), 'prv_q10': float(np.quantile(v[:, 1], .1)),
                              'prv_q90': float(np.quantile(v[:, 1], .9)),
                              # display: our unrounded base U ~ Uniform[.9561, .9566] (MODEL, RESEARCH_960 sec 1)
                              'P_disp957': float(np.clip((v[:, 0] - .0004) / .0005, 0, 1).mean()),
                              'P_disp958': float(np.clip((v[:, 0] - .0014) / .0005, 0, 1).mean()),
                              'P_disp960': float(np.clip((v[:, 0] - .0034) / .0005, 0, 1).mean())}
        # first-probe diagnostics (prior2: OA up then TRAUMA down) for the V224 base
        fp = [r['V224|prior2|2'][5] for r in R]
        for j, g in enumerate(('OA', 'TRAUMA')):
            rows = [f[j] for f in fp if len(f) > j and f[j][0] == g]
            if rows:
                dpub = np.array([x[3] for x in rows]); dpop = np.array([x[4] for x in rows])
                agg[name][f'probe1_{g}'] = {'n': len(rows), 'P_win': float(np.mean([x[2] for x in rows])), 'dpub_mean': dpub.mean(),
                                            'dpub_sd': dpub.std(), 'dpop_mean': dpop.mean(), 'dpop_sd': dpop.std(),
                                            'P_pop_gt0_given_win': float(np.mean(dpop[[x[2] for x in rows]] > 0)) if any(x[2] for x in rows) else None,
                                            'P_pop_gt0_given_loss': float(np.mean(dpop[[not x[2] for x in rows]] > 0)) if not all(x[2] for x in rows) else None}
    json.dump({'scenarios': {n: SCEN[n] for n in A.scen.split(',')}, 'worlds': A.worlds, 'npop': A.npop, 'seconds': time.time() - t0,
               'agg': agg}, open(A.out, 'w'), indent=1, default=float)
    print('SIM_PL_DONE', A.out, 'seconds %.0f' % (time.time() - t0))
