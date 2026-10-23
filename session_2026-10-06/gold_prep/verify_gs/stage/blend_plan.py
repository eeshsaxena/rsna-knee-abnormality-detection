#!/usr/bin/env python3
# blend_plan.py (2026-10-06 ~22:45 IST, Claude; gold_prep gate scorers). CPU ONLY (CUDA_VISIBLE_DEVICES forced '').
"""Planning read: does a candidate family add to J1 on the fold-0 OOF?

For the base run (default J1 = f0_s1043_ttr_mix50) and each candidate run (same fold), using each run's BEST-tag OOF
(as score_run.py picks it) on the common val studies:
  * solo OOF macro AUC vs steven_v4 (trainer's macro_auc, verbatim);
  * 2-member equal-weight RANK-mean {base, X}: macro AUC, delta vs base solo, paired bootstrap SE of that delta and
    P(delta > 0) over studies (B resamples), and the in-sample best weight on X (grid .05; optimistic, planning only);
  * 3-member equal-weight rank-mean {base, X, Y} for every candidate pair, delta vs base and vs the better 2-member.
Rank-mean = per label, each member's predictions turned into average ranks / n on the common studies, then averaged.

Caveat printed with the table: OOF is scored against steven_v4; members TRAINED on v4 labels are flattered on this
metric (TTR v4 looked ~.007 behind J1 on OOF but was ~.016 weaker on the LB). Read v4-trained rows as optimistic.

Usage (box): CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python ~/gold_prep/gates/blend_plan.py \
    f0_s42 f0_s1042_ttr f0_s142_f2v4 [--partial] [--boot 300]
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
import argparse, itertools, json, sys
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_run as S


def fast_auc_cols(Yb, M, valid):
    """Per-label AUC via average ranks (identical to roc_auc_score incl. ties). Yb bool (n,L), M (n,L)."""
    out = np.full(Yb.shape[1], np.nan)
    for j in range(Yb.shape[1]):
        if not valid[j]:
            continue
        y = Yb[:, j]; n1 = int(y.sum()); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(M[:, j])
        out[j] = (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return out


def rank01(P):
    return np.column_stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cands', nargs='+', help='candidate run names (same fold as the base)')
    ap.add_argument('--base', default=S.J1)
    ap.add_argument('--partial', action='store_true', help='allow still-training candidates (provisional)')
    ap.add_argument('--boot', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default='')
    for k, v in S.DEFAULTS.items():
        ap.add_argument(f'--{k}', default=v)
    a = ap.parse_args(argv)
    ctx = S.build_ctx(a)
    base = S.load_run(S.resolve(a.base, a.runs_root), ctx, partial=False)
    runs = [base]
    for c in a.cands:
        try:
            R = S.load_run(S.resolve(c, a.runs_root), ctx, partial=a.partial)
        except S.RunError as e:
            print(f'skip {c}: {e}'); continue
        if R['fold'] != base['fold']:
            print(f'skip {c}: fold {R["fold"]} != base fold {base["fold"]}'); continue
        if R['dir'] == base['dir']:
            continue
        runs.append(R)
    common = sorted(set.intersection(*[set(R['P'].index) for R in runs]))
    print(f'base {base["name"]} ({base["best_tag"]}), fold {base["fold"]}, common val studies {len(common)}, '
          f'eval labels {os.path.basename(a.eval_labels)}')
    v4 = ctx['v4']
    Y = v4.loc[common, S.TARGETS].to_numpy(np.float64)
    fin = np.isfinite(Y)
    if not fin.all():                    # the trainer masks NaN labels per target; keep that by dropping per label
        print(f'note: {int((~fin).sum())} NaN label cells in the eval labels (masked per label in the exact AUC)')
    Yb = Y >= 0.5
    P = {R['name']: R['P'].loc[common].to_numpy(np.float32) for R in runs}
    RK = {k: rank01(v.astype(np.float64)) for k, v in P.items()}
    names = [R['name'] for R in runs]
    info = {R['name']: R for R in runs}

    def exact(M):
        return S.macro_auc(M, common, v4)[0]

    solo = {k: exact(P[k]) for k in names}
    # sanity: rank-transform leaves the AUC unchanged, and fast_auc == sklearn on the full set
    valid = np.array([fin[:, j].all() and 0 < Yb[:, j].mean() < 1 for j in range(Y.shape[1])])
    chk = float(np.nanmean(fast_auc_cols(Yb, RK[base['name']], valid)))
    print(f'check: exact base solo {solo[base["name"]]:.6f} | fast rank AUC {chk:.6f} | rank-transformed exact '
          f'{exact(RK[base["name"]]):.6f}')
    rng = np.random.default_rng(a.seed)
    BI = [rng.integers(0, len(common), len(common)) for _ in range(a.boot)]

    def boot_delta(Mb, Ma):
        d = []
        for ix in BI:
            yb = Yb[ix]; ok = valid & (yb.sum(0) > 0) & (yb.sum(0) < len(ix))
            d.append(np.nanmean(fast_auc_cols(yb, Mb[ix], ok)) - np.nanmean(fast_auc_cols(yb, Ma[ix], ok)))
        d = np.array(d)
        return float(d.std(ddof=1)), float((d > 0).mean())

    b = base['name']; out = dict(base=b, n=len(common), solo=solo, pairs=[], triples=[])
    print('\nSOLO (best tag, common studies):')
    for k in names:
        R = info[k]
        rho = '' if k == b else (f'  rho vs base {np.mean(S.spearman_labels(R["P"], base["P"], common)):.3f}')
        tl = R['meta'].get('train_labels')
        print(f'  {k:28s} {R["best_tag"]:3s} {solo[k]:.4f}  train_labels {tl}{"" if R["finished"] else "  (PROVISIONAL)"}{rho}')
    print(f'\n2-MEMBER equal-weight rank-mean {{{b}, X}}  (dBase = blend - base solo; SE/P from {a.boot} paired '
          f'bootstrap resamples of studies; w* = in-sample best weight on X, optimistic)')
    print(f'  {"X":28s} {"soloX":>7s} {"blend":>7s} {"dBase":>7s} {"SE":>6s} {"P>0":>5s} {"dBest":>7s} {"w*":>5s} {"@w*":>7s}')
    pair_auc = {}
    for k in names[1:]:
        M = (RK[b] + RK[k]) / 2
        bl = exact(M); pair_auc[k] = bl
        se, pp = boot_delta(M, RK[b])
        ws = np.round(np.arange(0, 1.0001, 0.05), 2)
        gw = [exact((1 - w) * RK[b] + w * RK[k]) for w in ws]
        iw = int(np.argmax(gw))
        dbest = bl - max(solo[b], solo[k])
        print(f'  {k:28s} {solo[k]:.4f} {bl:.4f} {bl - solo[b]:+.4f} {se:.4f} {pp:5.2f} {dbest:+.4f} {ws[iw]:5.2f} {gw[iw]:.4f}')
        out['pairs'].append(dict(x=k, solo_x=solo[k], blend=bl, d_base=bl - solo[b], se=se, p_pos=pp, d_best=dbest,
                                 w_star=float(ws[iw]), auc_w_star=gw[iw], finished=info[k]['finished']))
    if len(names) >= 3:
        print(f'\n3-MEMBER equal-weight rank-mean {{{b}, X, Y}}  (dBase vs base solo; dPair vs the better of the two '
              f'2-member blends)')
        print(f'  {"X":28s} {"Y":28s} {"blend":>7s} {"dBase":>7s} {"SE":>6s} {"P>0":>5s} {"dPair":>7s}')
        for x, y in itertools.combinations(names[1:], 2):
            M = (RK[b] + RK[x] + RK[y]) / 3
            bl = exact(M); se, pp = boot_delta(M, RK[b])
            dp = bl - max(pair_auc[x], pair_auc[y])
            print(f'  {x:28s} {y:28s} {bl:.4f} {bl - solo[b]:+.4f} {se:.4f} {pp:5.2f} {dp:+.4f}')
            out['triples'].append(dict(x=x, y=y, blend=bl, d_base=bl - solo[b], se=se, p_pos=pp, d_pair=dp))
    print('\ncaveat: OOF is vs steven_v4 labels; v4-TRAINED members are flattered here (TTR v4: OOF ~.007 behind J1 but '
          '~.016 behind on the LB). Fold-0 only (867 studies): single-fold deltas under ~.002 are noise.')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
