# gt2_read.py -- gate G-T2 (BIG_JUMP_PLAN sec 4): the TTR recipe test. Read-only (logs, OOF csvs); CPU; no torch.
#   TTR-A fold-0 vs f0_s42_v4 (same fold 0, seed 42, steven_v4 targets and selection labels, gold74 held out).
#   score per run = best-epoch max(raw, EMA) OOF macro AUC vs steven_v4 on the fold-0 val studies, recomputed EXACTLY
#   from the run's best OOF files (oof_f0_s{seed}_{raw,ema}.csv = the predictions of each tag's best epoch), with the
#   trainer's macro_auc rule (12 labels, label >= .5, NaN masked)
#   PASS   TTR best >= ref best + .008 AND no label of TTR's best tag below the ref best tag's label AUC - .02 AND
#          TTR gold58 (its best epoch + tag, from the log) >= ref gold58 (direction only)    -> PASS branch
#   KILL   TTR best <= ref best                                                           -> KILL branch
#   MIDDLE anything between                                                               -> MIDDLE branch
#   PROVISIONAL is printed (and exit 3) while the TTR log has no closing {"raw": .., "ema": ..} line (still training or
#   saving); the numbers are then the best so far. Exit 0 = final read. Selection note (plan sec 4 review): TTR picks
#   its best of 48 (24 epochs x raw/EMA) vs the ref's 20, worth ~+.001..+.002 on the same 867 (GUESS); the +.008 bar
#   absorbs it, the MIDDLE band is partly selection noise.
# usage (box, CPU):
#   ~/miniconda3/envs/rsna/bin/python ~/bigjump_build/code/gt2_read.py --ttr ~/fleet_runs/f0_s42_ttr \
#       --ref ~/fleet_runs/f0_s42_v4 --eval_labels ~/rsna_train/llm_labels_v4_blend.csv
#   (log = <run dir>.log; override with --ttr_log / --ref_log)
import os, re, sys, json, glob, argparse
import numpy as np, pandas as pd

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
LINE = re.compile(r'^ep (\d+) (raw|ema)(?: \| gold74 ([\d.na]+) \(\d+ labels\))?(?: \| gold58 ([\d.na]+) \(\d+ labels\))?'
                  r' \| val AUC vs (\S+) ([\d.na]+)')


def _auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def label_aucs(P, Y):
    out = {}
    for j, t in enumerate(TARGETS):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        out[t] = float(_auc(yb, P[ok, j])) if ok.sum() and 0 < yb.mean() < 1 else float('nan')
    return out


def read_run(d, log, ev):
    rows, done, seed = [], False, None
    for l in open(log, errors='replace'):
        l = l.strip()
        if l.startswith('{"raw"'):
            try:
                done = isinstance(json.loads(l), dict)
            except ValueError:
                pass
        m = LINE.match(l)
        if m:
            f = lambda v: float(v) if v not in (None, 'nan') else float('nan')
            rows.append(dict(ep=int(m[1]) + 1, tag=m[2], gold74=f(m[3]), gold58=f(m[4]), sel=m[5], val=f(m[6])))
    df = pd.DataFrame(rows)
    assert len(df), f'no epoch lines in {log}'
    res = dict(dir=d, log=log, done=done, epochs_logged=int(df.ep.max()), sel_labels=sorted(set(df.sel)), curve=df, tags={})
    for tag in ('raw', 'ema'):
        fs = sorted(glob.glob(f'{d}/oof_f0_s*_{tag}.csv'))
        if not fs:
            continue
        assert len(fs) == 1, fs
        o = pd.read_csv(fs[0], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        Y = ev.loc[o.index, TARGETS].to_numpy(np.float64)
        la = label_aucs(o[TARGETS].to_numpy(np.float64), Y)
        macro = float(np.nanmean([v for v in la.values()]))
        t = df[df.tag == tag]
        b = t.loc[t.val.idxmax()]                                   # first max == the trainer's strict '>' save
        res['tags'][tag] = dict(oof=fs[0], n=len(o), val=macro, labels=la, best_ep=int(b.ep), log_val=float(b.val),
                                gold58=float(b.gold58), best_files=sorted(os.path.basename(p) for p in glob.glob(f'{d}/*_{tag}_*_best.pt')))
    assert res['tags'], f'no oof_f0_s*_raw/ema.csv in {d}'
    bt = max(res['tags'], key=lambda k: res['tags'][k]['val'])
    res['best_tag'] = bt
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ttr', required=True); ap.add_argument('--ref', required=True)
    ap.add_argument('--ttr_log', default=''); ap.add_argument('--ref_log', default='')
    ap.add_argument('--eval_labels', required=True)
    ap.add_argument('--pass_bar', type=float, default=0.008); ap.add_argument('--label_floor', type=float, default=-0.02)
    ap.add_argument('--json', default='')
    A = ap.parse_args(argv)
    ev = pd.read_csv(A.eval_labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    T = read_run(A.ttr, A.ttr_log or A.ttr.rstrip('/') + '.log', ev)
    R = read_run(A.ref, A.ref_log or A.ref.rstrip('/') + '.log', ev)
    for name, r in (('TTR', T), ('REF', R)):
        print(f'{name} {r["dir"]}: epochs logged {r["epochs_logged"]}, closing line {"yes" if r["done"] else "NO"}, '
              f'selection labels {r["sel_labels"]}')
        c = r['curve'].pivot_table(index='ep', columns='tag', values=['val', 'gold58'])
        print(c.round(4).to_string())
        for tag, t in r['tags'].items():
            print(f'  {name} {tag}: best ep {t["best_ep"]} | OOF macro AUC vs eval labels {t["val"]:.4f} (log {t["log_val"]:.4f}) '
                  f'| gold58 {t["gold58"]:.4f} | n {t["n"]} | {t["best_files"]}')
    tb, rb = T['tags'][T['best_tag']], R['tags'][R['best_tag']]
    same = set(pd.read_csv(tb['oof'], usecols=['StudyInstanceUID'], dtype=str).StudyInstanceUID) == \
        set(pd.read_csv(rb['oof'], usecols=['StudyInstanceUID'], dtype=str).StudyInstanceUID)
    d = tb['val'] - rb['val']
    dl = {t: tb['labels'][t] - rb['labels'][t] for t in TARGETS}
    worst = min((v, t) for t, v in dl.items() if np.isfinite(v))
    g_na = not (np.isfinite(tb['gold58']) and np.isfinite(rb['gold58']))      # no gold58 in a log: direction n/a
    g_ok = g_na or tb['gold58'] >= rb['gold58']
    if d <= 0:
        verdict = 'KILL'
    elif d >= A.pass_bar and worst[0] >= A.label_floor and g_ok:
        verdict = 'PASS'
    else:
        verdict = 'MIDDLE'
    print('\nper-label AUC delta (TTR best tag - ref best tag): ' + ', '.join(f'{t} {v:+.4f}' for t, v in dl.items()))
    print(f'same val study set: {same} ({tb["n"]} vs {rb["n"]})')
    print(f'G-T2: TTR {T["best_tag"]} ep {tb["best_ep"]} {tb["val"]:.4f} vs REF {R["best_tag"]} ep {rb["best_ep"]} '
          f'{rb["val"]:.4f} -> delta {d:+.4f} (PASS bar +{A.pass_bar}); worst label {worst[1]} {worst[0]:+.4f} '
          f'(floor {A.label_floor}); gold58 {tb["gold58"]:.4f} vs {rb["gold58"]:.4f} ({"n/a" if g_na else "ok" if g_ok else "LOWER"})')
    final = T['done'] and R['done'] and same
    print(f'G-T2 VERDICT: {verdict}{"" if final else "  [PROVISIONAL: a run is still training/saving or the val sets differ]"}')
    if A.json:
        json.dump(dict(verdict=verdict, final=final, delta=d, worst_label=worst, gold58=(tb['gold58'], rb['gold58']),
                       ttr={k: v for k, v in T.items() if k != 'curve'}, ref={k: v for k, v in R.items() if k != 'curve'}),
                  open(A.json, 'w'), indent=1, default=str)
    return 0 if final else 3


if __name__ == '__main__':
    sys.exit(main())
