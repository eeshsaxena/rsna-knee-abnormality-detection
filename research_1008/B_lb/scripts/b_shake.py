#!/usr/bin/env python3
"""b_shake.py (2026-10-08, research_1008 dimension B). Laptop, light vectorised numpy (~1 min).

Monte Carlo of the public/private split for a macro AUC over 12 labels. MODEL, with these inputs:
  - test size ~1,300 studies (GUESS from the data page / K9N runbook N 1,150-1,322); public 30% = 390, private 70% = 910
  - per-label prevalence = the 58 expert gold studies in train.csv (MEASURED)
  - per-label AUC spread sd ~.03 around each leg's macro (MEASURED per-label sd .033-.043 in 11 OOF pairs, verify_blendmath)
  - legs P (.943), F (.944), N (.949, also a .9475 'curse' case), G (.929); equal-variance binormal latent scores
    with noise correlation R from b_model.py central values (r_PF .945, r_PN .945, r_GN .838, r_FN scanned)
  - a rank blend is modelled as the same-weight blend of latent scores (the form validated in verify_blendmath)
Outputs: SD(public - private) for one submission; SD of the paired private difference between two blends;
P(blend A beats blend B on private); P(public order == private order).
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20261008)
PREV = np.array([.414, .155, .448, .397, .259, .190, .362, .603, .466, .207, .328, .310])  # gold58 MEASURED
L = len(PREV)
DELTA = np.array([+.9, -.2, -.3, -.6, +.2, -.4, -.3, +1.3, -1.0, +.2, +.1, -.7])  # per-label difficulty profile (GUESS)
N_TEST, N_PUB = 1300, 390


def d(a): return np.sqrt(2) * norm.ppf(a)


def label_d(macro):
    """per-label d_l = base + s*DELTA so the macro AUC equals `macro` and per-label sd ~.03."""
    s = 0.22
    lo, hi = -2.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        m = norm.cdf((mid + s * DELTA) / np.sqrt(2)).mean()
        lo, hi = (mid, hi) if m < macro else (lo, mid)
    return mid + s * DELTA


def auc_batch(scores, y):
    """scores (R, n, k), y (R, n) bool -> AUC (R, k)."""
    order = scores.argsort(axis=1)
    ranks = np.empty_like(order)
    Rn, n, k = scores.shape
    ar = np.arange(1, n + 1)[None, :, None]
    np.put_along_axis(ranks, order, np.broadcast_to(ar, order.shape), axis=1)
    n1 = y.sum(1)[:, None]
    n0 = n - n1
    sr = (ranks * y[:, :, None]).sum(1)
    return (sr - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)


def run(legs, R, blends, reps=1200, batch=200):
    """legs: dict name -> macro AUC; blends: dict name -> weight vector over legs. Returns pub, priv arrays (reps, B)."""
    names = list(legs)
    Lc = np.linalg.cholesky(R)
    D = np.stack([label_d(legs[nm]) for nm in names])           # (legs, L)
    W = np.stack([np.asarray(w, float) for w in blends.values()], 1)  # (legs, B)
    pub_all, priv_all = [], []
    for b0 in range(0, reps, batch):
        rb = min(batch, reps - b0)
        perm = np.argsort(rng.random((rb, N_TEST)), axis=1)
        pub_idx, priv_idx = perm[:, :N_PUB], perm[:, N_PUB:]
        mac_pub = np.zeros((rb, W.shape[1]))
        mac_priv = np.zeros((rb, W.shape[1]))
        for l in range(L):
            y = rng.random((rb, N_TEST)) < PREV[l]
            z = rng.standard_normal((rb, N_TEST, len(names))) @ Lc.T          # correlated noise
            s = z + (y[:, :, None] * (D[:, l] / np.sqrt(1.0))[None, None, :])  # binormal: shift positives by d
            # with unit-variance noise in each class, AUC = Phi(d/sqrt2)
            bl = s @ W                                                          # (rb, n, B)
            yp = np.take_along_axis(y, pub_idx, 1)
            yq = np.take_along_axis(y, priv_idx, 1)
            bp = np.take_along_axis(bl, pub_idx[:, :, None], 1)
            bq = np.take_along_axis(bl, priv_idx[:, :, None], 1)
            mac_pub += auc_batch(bp, yp) / L
            mac_priv += auc_batch(bq, yq) / L
        pub_all.append(mac_pub)
        priv_all.append(mac_priv)
    return np.concatenate(pub_all), np.concatenate(priv_all)


def report(tag, legs, R, blends, ref='N+G cluster', reps=1200):
    pub, priv = run(legs, R, blends, reps=reps)
    names = list(blends)
    j = names.index(ref)
    print(f'\n### {tag}')
    print(f'{"blend":30s} {"E pub":>7s} {"E priv":>7s} {"sd pub":>7s} {"sd priv":>7s} {"sd(pub-priv)":>12s} | vs {ref}: '
          f'{"E dpriv":>8s} {"sd dpriv":>8s} {"sd dpub":>8s} {"P(priv>)":>8s} {"P(order same)":>13s}')
    for i, nm in enumerate(names):
        dq = priv[:, i] - priv[:, j]
        dp = pub[:, i] - pub[:, j]
        same = np.mean(np.sign(dq) == np.sign(dp)) if i != j else np.nan
        print(f'{nm:30s} {pub[:, i].mean():7.4f} {priv[:, i].mean():7.4f} {pub[:, i].std():7.4f} {priv[:, i].std():7.4f} '
              f'{(pub[:, i] - priv[:, i]).std():12.4f} | {dq.mean():+8.4f} {dq.std():8.4f} {dp.std():8.4f} '
              f'{np.mean(dq > 0):8.3f} {same:13.3f}')
    return pub, priv


def _main():
    import sys
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    names = ['P', 'F', 'N', 'G']
    blends = {
        'N+G cluster': (0, 0, .838, .162),
        'N alone': (0, 0, 1, 0),
        'K9 main P.5 F.5': (.5, .5, 0, 0),
        'K9N-a P.10 F.30 N.60': (.10, .30, .60, 0),
        'K9N-b P.30 F.30 N.40': (.30, .30, .40, 0),
        'MR P.05 F.35 N.60': (.05, .35, .60, 0),
        'F+N F.40 N.60': (0, .40, .60, 0),
        'P.05 F.30 N.55 G.10': (.05, .30, .55, .10),
    }
    for rFN in (.86, .90, .94):
        R = np.array([[1, .945, .945, .90], [.945, 1, rFN, .91], [.945, rFN, 1, .838], [.90, .91, .838, 1]])
        assert np.linalg.eigvalsh(R).min() > 0, rFN
        report(f'r_FN {rFN}: N .949 (population), F .944, P .943, G .929', dict(P=.943, F=.944, N=.949, G=.929), R, blends,
               reps=reps)
    # winner's-curse case: N population strength below its public read; cluster weights tuned (no extra)
    R = np.array([[1, .945, .945, .90], [.945, 1, .90, .91], [.945, .90, 1, .838], [.90, .91, .838, 1]])
    report('CURSE: N population .9475 (public read .949 was selected), r_FN .90', dict(P=.943, F=.944, N=.9475, G=.929), R,
           blends, reps=reps)


def validate(reps=1500):
    """Calibration: two single models (.888 OOF-like, r .95-.96) on 867 studies, all used as 'private'.
    MEASURED anchor: paired OOF SE ~.001-.002 on 867 fold-0 studies (NOVEL_PLAN noise line; v_ens V3 SE .0014)."""
    global N_TEST, N_PUB
    keep = (N_TEST, N_PUB)
    N_TEST, N_PUB = 867 + 1, 1
    for r in (.93, .95, .96):
        R = np.array([[1, r], [r, 1]])
        pub, priv = run(dict(A=.888, B=.884), R, {'A': (1, 0), 'B': (0, 1)}, reps=reps)
        dq = priv[:, 0] - priv[:, 1]
        print(f'VALIDATION n=867, single models .888 vs .884 at r {r}: sd(paired diff) {dq.std():.4f}, '
              f'sd(single) {priv[:, 0].std():.4f}')
    N_TEST, N_PUB = keep


if __name__ == '__main__':
    import sys
    validate()
    _main()
