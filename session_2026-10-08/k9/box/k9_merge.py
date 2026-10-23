#!/usr/bin/env python3
"""k9_merge.py (box, CPU, read-only except DEST/members.json; 2026-10-08, Claude; from k8_merge.py): merge the two K9
picker outputs.

  python k9_merge.py --dest ~/k9_test [--c_margin 0.002] [--d_margin 0.004] [--c_bars F:v,..] [--d_bars F:v,..]
                     [--allow_interim]

K9 = parent + [set C + set D], equal members, raw best checkpoints, TTA A+B (NOVEL_PLAN_10-07 sec 4 row 10-09):
  DEST/members_C.json  set C  f{0..4}_s1044_ttr288mix50   TTR288, mix50 labels   (box chain auto_after_ttr288.sh)
  DEST/members_D.json  set D  f{0..4}_s1043_ttr_frac75    TTR224, frac75 labels  (Kaggle: swapiy f01 E17, kmrsuman
                                                                                  f23 + f4x E~21; run receipts)
  set D REPLACES set A (the K5 mix50 s1043 members): a set A member is refused here and by build_k9.py; set B is dropped.
Acceptance: FINAL only when BOTH sets are 5/5 (every fold picked: closing line + every picker check) and every fold is at
or above its sanity bar. INTERIM (a set < 5/5, no failure): exit 3 unless --allow_interim, which writes members.json with
interim=True and do_not_submit=True (never submitted). FAIL (a picker check failed on a finished fold, or a fold below its
bar): exit 1, nothing written.
Per-fold sanity bars (they catch a broken / diverged fold; they are not gain tests). Reference = set A's picked raw cv on
the same fold (A_REF below = the K5 members, frozen since 10-05; values == the K8 collect of 10-08 04:35, MEASURED):
  set C: bar_F = A_F - .002 (== K8: ~1.4 x the paired fold SE .0014).
  set D: bar_F = A_F - .004. Why wider: the frac75 folds 0-1 are E 17 (Kaggle 8.4 h budget) instead of 24, and their
         selection metric is OOF vs v4 labels while the training target moved toward the image teacher on Fracture, so a
         healthy frac75 fold reads ~.000 to .002 below its 24-epoch mix50 twin (MEASURED: f0 .8838 vs .8836 = +.0002, f1
         .8910 vs .8926 = -.0016; the paired fold SE is ~.0014). .004 = the worst healthy gap seen (.0016) + ~1.7 SE: a
         normal E 17-21 fold passes, a diverged / mislabelled fold (typically -.01 or worse) fails.
  --c_bars / --d_bars F:v,... override single folds (the operator's explicit call, recorded in members.json).
Refuses duplicate (seed, fold, ttr_size, train_labels) / file names, any member outside C / D, ema picks, and a set D
member whose train_labels is not labels_frac75_logit.csv (or a set C member not on mix50).
Exit 0 = written (FINAL, or INTERIM with --allow_interim); 1 = check failed; 3 = a set not complete (no --allow_interim).
"""
import argparse, json, os, sys, time

A_REF = {0: 0.8835985889396399, 1: 0.8926246223939701, 2: 0.8907517323356613, 3: 0.8763090478076047,
         4: 0.8928147656091167}                       # set A raw cv per fold (f{F}_s1043_ttr_mix50, the K5 members)
SETS = {'C': dict(seed=1044, ttr=288, labels='labels_mix50_logit.csv', suffix='ttr288mix50', key='t288'),
        'D': dict(seed=1043, ttr=224, labels='labels_frac75_logit.csv', suffix='ttr_frac75', key='f75')}


def load(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def bars_for(margin, spec, folds):
    over = {}
    for x in [y.strip() for y in spec.split(',') if y.strip()]:
        f, _, v = x.partition(':')
        over[int(f)] = float(v)
    bars = {F: (over[F] if F in over else A_REF[F] - margin) for F in folds}   # unrounded (printed with 6 decimals)
    src = {F: ('operator override' if F in over else f'set A f{F} cv {A_REF[F]:.6f} - {margin}') for F in folds}
    return bars, src, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=os.path.expanduser('~/k9_test'))
    ap.add_argument('--c_margin', type=float, default=0.002)
    ap.add_argument('--d_margin', type=float, default=0.004)
    ap.add_argument('--c_bars', default='')
    ap.add_argument('--d_bars', default='')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--allow_interim', action='store_true')
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(',') if x.strip()]
    members, skipped, info, final, counts = [], [], {}, True, {}
    for s, margin, spec in (('C', a.c_margin, a.c_bars), ('D', a.d_margin, a.d_bars)):
        cfg = SETS[s]
        X = load(f'{a.dest}/members_{s}.json')
        bars, src, over = bars_for(margin, spec, folds)
        st = dict(margin=margin, bars=bars, bar_source=src, overrides=over, ref='A_REF (set A raw cv)')
        if not X:
            print(f'K9MERGE: set {s} picker output missing (picker rc 3: no fold of f{{0..4}}_s{cfg["seed"]}_{cfg["suffix"]} '
                  f'has its closing line?)')
            X = dict(members=[], skipped=[dict(fold=F, reason=f'no members_{s}.json') for F in folds], failed=[])
        if X.get('failed'):
            print(f'K9MERGE FAIL: set {s} picker check FAILED on a finished fold: {X["failed"]} (read the FAIL lines)')
            return 1
        bad = [(r['fold'], r.get('tag'), r.get('seed'), r.get('ttr_size'), r.get('train_labels')) for r in X['members']
               if r.get('tag') != 'raw' or int(r.get('seed', -1)) != cfg['seed'] or int(r.get('ttr_size', -1)) != cfg['ttr']
               or r.get('train_labels') != cfg['labels']]
        if bad:
            print(f'K9MERGE FAIL: set {s} picks not (raw, seed {cfg["seed"]}, ttr {cfg["ttr"]}, {cfg["labels"]}): {bad}')
            return 1
        cv = {r['fold']: r['cv'] for r in X['members']}
        low = {F: (v, bars[F]) for F, v in cv.items() if not v >= bars[F]}
        st['cv'] = cv
        st['vs_A'] = {F: round(cv[F] - A_REF[F], 6) for F in cv}
        for F in folds:
            stt = 'NOT FINISHED' if F not in cv else ('BELOW BAR' if F in low else 'OK')
            cvs = f'{cv[F]:.6f} ({cv[F] - A_REF[F]:+.4f} vs A)' if F in cv else '-'
            ep = {r['fold']: r.get('epochs') for r in X['members']}.get(F)
            print(f'  set {s} fold {F}: cv {cvs} E {ep} bar {bars[F]:.6f} [{src[F]}] -> {stt}')
        if low:
            st['status'] = f'FAIL: fold(s) below the sanity bar {low}'
            print(f'K9MERGE FAIL: set {s} fold(s) below their sanity bar {low}: a broken / weak fold. Do not ship; read the '
                  f'run log. To accept anyway (the operator\'s call): K9_{s}_BARS="F:<bar>" (k9_collect.sh)')
            return 1
        members += [dict(r, set=s) for r in X['members']]
        skipped += [dict(sk, seed=cfg['seed'], suffix=cfg['suffix'], set=s) for sk in X['skipped']]
        counts[s] = len(cv)
        if sorted(cv) == folds:
            st['status'] = f'FINAL: {len(cv)}/5 folds, every fold >= its bar'
        else:
            final = False
            st['status'] = f'INTERIM: {len(cv)}/5 folds (not finished: {[F for F in folds if F not in cv]})'
        info[s] = st
    if not final and not a.allow_interim:
        for sk in skipped:
            print(f'  NOT FINISHED set {sk["set"]}: f{sk["fold"]}_s{sk["seed"]}_{sk["suffix"]}: {sk["reason"][:110]}')
        print(f'K9MERGE NOT FINAL: set C {counts["C"]}/5, set D {counts["D"]}/5. K9 ships only at C 5/5 + D 5/5. '
              f'For a TEST build only: K9_INTERIM=1 (members.json is marked do_not_submit)')
        return 3
    ids = [(r['seed'], r['fold'], int(r['ttr_size']), r['train_labels']) for r in members]
    files = [r['file'] for r in members]                    # one DEST/ckpt dir: a file name may appear once
    if len(set(ids)) != len(ids) or len(set(files)) != len(files):
        print(f'K9MERGE FAIL: duplicate members {sorted(ids)} / {sorted(files)}'); return 1
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               set='K9 = C f{0..4}_s1044_ttr288mix50 (TTR288, mix50) + D f{0..4}_s1043_ttr_frac75 (TTR224, frac75); '
                   'set A replaced by D, set B dropped',
               members=members, skipped=skipped, failed=[], copied=True, set_c=info['C'], set_d=info['D'],
               interim=not final, do_not_submit=not final)
    tmp = f'{a.dest}/members.json.tmp'
    json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, f'{a.dest}/members.json')
    for r in members:
        print(f'  {r["set"]} s{r["seed"]} f{r["fold"]} ttr{r["ttr_size"]} {r["train_labels"]} {r["tag"]} E {r.get("epochs")} '
              f'cv {r["cv"]:.6f} {r["file"]} source {r["source"]} receipt {os.path.basename(r.get("kaggle_receipt") or "-")}')
    for sk in skipped:
        print(f'  NOT FINISHED set {sk["set"]}: f{sk["fold"]}_s{sk["seed"]}_{sk["suffix"]}: {sk["reason"][:110]}')
    print(f'  set C: {info["C"]["status"]} | set D: {info["D"]["status"]}')
    print(f'K9MERGE OK {len(members)} member(s): C {counts["C"]}/5, D {counts["D"]}/5 ' +
          ('(FINAL)' if final else '(INTERIM: do_not_submit, NEVER SUBMIT)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
