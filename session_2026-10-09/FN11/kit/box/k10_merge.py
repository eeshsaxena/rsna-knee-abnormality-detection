#!/usr/bin/env python3
"""k10_merge.py (box, CPU, read-only except DEST/members.json; 2026-10-09, Claude; from k9_merge.py): merge the three
K10 picker outputs into ONE members.json that every K10 build reads.

  python k10_merge.py --dest ~/k10_test [--e_margin 0.004] [--e_bars F:v,..] [--c_margin .002] [--d_margin .004]
                      [--c_bars ..] [--d_bars ..] [--allow_interim]

Sets (equal members, RAW best checkpoints, TTA A+B):
  DEST/members_E.json  set E  f{0..4}_s1045_ttr288frac75  TTR288, frac75 labels, 21 epochs (f0 box V100; f1-f4 Vast RTX
                                                        3090, copied into the box placeholders + vast320 receipts)
  DEST/members_D.json  set D  f{0..4}_s1043_ttr_frac75    TTR224, frac75 labels (Kaggle; == the K9 picks byte for byte)
  DEST/members_C.json  set C  f{0..4}_s1044_ttr288mix50   TTR288, mix50 labels  (box; == the K9 picks byte for byte)
K10 = parent + [E + D] (E REPLACES C); FN10 = [E + D] + N; FN10-3 = [C + D + E] + N (parent-free only). C and D must be
exactly the K9 members (sha256 below, frozen from SP/k9/members.json 10-08 09:52): a changed C / D pick FAILS.
Acceptance: FINAL only when E 5/5 AND D 5/5 AND C 5/5 (every fold picked, every picker check, every fold >= its bar).
INTERIM (a set < 5/5, no failure): exit 3 unless --allow_interim (members.json interim + do_not_submit: TEST builds only).

Per-fold sanity bars (catch a broken / diverged fold; NOT gain tests). Written 10-09 ~03:40 IST, before any E checkpoint
was picked (the E f0 trainer log had reached epoch 5 of 21; folds 1-4 had not been copied to the box):
  set C: bar_F = A_F - .002, set D: bar_F = A_F - .004  (unchanged K9 rules; A_REF = set A raw cv, the K5 members)
  set E: bar_F = C_F - .004  (C_REF = set C's picked raw cv on the same fold, the K9 members)
    Why C - .004 (MEASURED components, box closing lines 10-09 03:20):
      seed:    same recipe, other seed, fold-matched: B(s1044) - A(s1043) TTR224 mix50 = +.0028 / -.0009 / +.0003 / +.0003
               (f0-f3); s1044 - s1043 TTR224 frac75 f0 = +.0028. SD of a fold-matched seed swap ~.0016-.0020.
      labels:  frac75 vs mix50, same seed + ttr (D - A): +.0002 / -.0017 / +.0002 / +.0006 / +.0001 (worst -.0017).
      epochs:  21 vs 24 at TTR224 mix50 (the B runs are 21 ep vs A 24 ep): no measurable loss (mean +.0006).
      -> a healthy E fold sits at C_F - .0003 +- ~.0022 (MODEL from the above); .004 ~ 1.7 SD below that: a normal fold
         passes, a diverged / mislabelled / wrong-epoch fold (historically -.01 or worse) fails.
    KNOWN RISK (pre-registered, not a reason to move the bar): C f0 (.8909) and C f3 (.8822) are seed-lucky folds (C - A
      +.0073 / +.0059 vs +.0026 to +.0028 on f1 / f2 / f4; s1044 beats s1043 by +.0028 on f0 in two recipes), so a healthy
      E f0 / f3 can land below C_F - .004 (P ~ .2-.35 for f0, MODEL: the 320 probe below sits only .0008 above the f0 bar). The 320 frac75 s1045 21-ep probe (same seed, labels,
      epochs) read .8877 on f0 = C f0 - .0032.
    PRE-REGISTERED OVERRIDE (the operator's mechanical call; printed below for every E fold under its bar): accept the fold
      with K10_E_BARS="F:<its cv>" ONLY IF cv >= C_F - .007 (2 seed SD + the worst label effect) AND the fold is stable
      (its last-3-epoch raw val mean >= its best raw cv - .003, from its own log) AND its log shows the full E epochs.
      Anything lower: the fold is not shipped; K10 waits or falls back (K10_RUNBOOK sec 6).
  --e_bars / --c_bars / --d_bars F:v,... override single folds (recorded in members.json).
Refuses duplicates (seed, fold, ttr_size, train_labels), any member outside C / D / E, ema picks, wrong labels / seed /
ttr per set, a set E fold 1-4 without a Vast receipt (picker K10-VREC) unless K10_E_VAST_FOLDS says otherwise.
Exit 0 = written (FINAL, or INTERIM with --allow_interim); 1 = check failed; 3 = a set not complete (no --allow_interim).
"""
import argparse, json, os, re, sys, time

A_REF = {0: 0.8835985889396399, 1: 0.8926246223939701, 2: 0.8907517323356613, 3: 0.8763090478076047,
         4: 0.8928147656091167}                       # set A raw cv per fold (f{F}_s1043_ttr_mix50, the K5 members)
C_REF = {0: 0.8909202479807492, 1: 0.8952226680508857, 2: 0.8935590285907183, 3: 0.8821807711682387,
         4: 0.8955640334637714}                       # set C raw cv per fold (f{F}_s1044_ttr288mix50, the K9 members)
K9_SHA = {('C', 0): 'aa394cbf83b0084e', ('C', 1): '2aa098e78541e33c', ('C', 2): '47d6f35bf0f1a4ff',
          ('C', 3): 'a62a6be3750bc390', ('C', 4): '33078bd461d504d7', ('D', 0): 'c478a39d75b9657f',
          ('D', 1): 'fea35749550a275c', ('D', 2): 'ded00943d1b22a1a', ('D', 3): '71815926cc6f286b',
          ('D', 4): '79e301ed41d5cb12'}                # sha256[:16] of the K9 picks (SP/k9/members.json)
SETS = {'E': dict(seed=1045, ttr=288, labels='labels_frac75_logit.csv', suffix='ttr288frac75', key='t288f75'),
        'D': dict(seed=1043, ttr=224, labels='labels_frac75_logit.csv', suffix='ttr_frac75', key='f75'),
        'C': dict(seed=1044, ttr=288, labels='labels_mix50_logit.csv', suffix='ttr288mix50', key='t288')}
SOFT_E = 0.007                                        # pre-registered override floor (see header)


def load(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def bars_for(s, margin, spec, folds):
    ref, rn = (C_REF, 'set C') if s == 'E' else (A_REF, 'set A')
    over = {}
    for x in [y.strip() for y in spec.split(',') if y.strip()]:
        f, _, v = x.partition(':')
        over[int(f)] = float(v)
    bars = {F: (over[F] if F in over else ref[F] - margin) for F in folds}   # unrounded (printed with 6 decimals)
    src = {F: ('operator override' if F in over else f'{rn} f{F} cv {ref[F]:.6f} - {margin}') for F in folds}
    return bars, src, over, ref


def log_stats(run):
    """(E from the header, best raw val, mean of the last 3 epochs' raw val) from a trainer log (the override evidence)."""
    log = run + '.log'
    if not os.path.isfile(log):
        return None
    E, vals = None, {}
    for l in open(log, encoding='utf-8', errors='replace'):
        if l.startswith('BJ opt-in:'):
            m = re.search(r'[|] epochs (\d+) [|]', l)
            E = int(m.group(1)) if m else None
        m = re.match(r'ep (\d+) raw [|].*val AUC vs \S+ ([0-9.]+)', l)
        if m:
            vals[int(m.group(1))] = float(m.group(2))
    if not vals:
        return None
    last = [vals[k] for k in sorted(vals)[-3:]]
    return dict(E=E, n_ep=len(vals), best=max(vals.values()), last3=sum(last) / len(last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=os.path.expanduser('~/k10_test'))
    ap.add_argument('--e_margin', type=float, default=0.004)
    ap.add_argument('--c_margin', type=float, default=0.002)
    ap.add_argument('--d_margin', type=float, default=0.004)
    ap.add_argument('--e_bars', default=''); ap.add_argument('--c_bars', default=''); ap.add_argument('--d_bars', default='')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--e_vast_folds', default='1,2,3,4', help='folds of set E that must carry a Vast receipt')
    ap.add_argument('--allow_interim', action='store_true')
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(',') if x.strip()]
    vast = [int(x) for x in a.e_vast_folds.split(',') if x.strip()]
    members, skipped, info, final, counts = [], [], {}, True, {}
    for s, margin, spec in (('E', a.e_margin, a.e_bars), ('D', a.d_margin, a.d_bars), ('C', a.c_margin, a.c_bars)):
        cfg = SETS[s]
        X = load(f'{a.dest}/members_{s}.json')
        bars, src, over, ref = bars_for(s, margin, spec, folds)
        st = dict(margin=margin, bars=bars, bar_source=src, overrides=over, ref='C_REF (set C raw cv)' if s == 'E' else 'A_REF')
        if not X:
            print(f'K10MERGE: set {s} picker output missing (picker rc 3: no fold of f{{0..4}}_s{cfg["seed"]}_{cfg["suffix"]} '
                  f'has its closing line?)')
            X = dict(members=[], skipped=[dict(fold=F, reason=f'no members_{s}.json') for F in folds], failed=[])
        if X.get('failed'):
            print(f'K10MERGE FAIL: set {s} picker check FAILED on a finished fold: {X["failed"]} (read the FAIL lines)')
            return 1
        bad = [(r['fold'], r.get('tag'), r.get('seed'), r.get('ttr_size'), r.get('train_labels')) for r in X['members']
               if r.get('tag') != 'raw' or int(r.get('seed', -1)) != cfg['seed'] or int(r.get('ttr_size', -1)) != cfg['ttr']
               or r.get('train_labels') != cfg['labels'] or r.get('suffix') != cfg['suffix']]
        if bad:
            print(f'K10MERGE FAIL: set {s} picks not (raw, seed {cfg["seed"]}, ttr {cfg["ttr"]}, {cfg["labels"]}, '
                  f'{cfg["suffix"]}): {bad}')
            return 1
        if s in ('C', 'D'):
            moved = [(r['fold'], r['sha256'][:16]) for r in X['members'] if K9_SHA[(s, r['fold'])] != r['sha256'][:16]]
            if moved:
                print(f'K10MERGE FAIL: set {s} picks differ from the K9 members (sha256[:16]) {moved}: K10 reuses the K9 '
                      f'{s} checkpoints byte for byte')
                return 1
        else:
            nov = [r['fold'] for r in X['members'] if r['fold'] in vast and not r.get('vast_receipt')]
            box_with = [r['fold'] for r in X['members'] if r['fold'] not in vast and r.get('vast_receipt')]
            if nov or box_with:
                print(f'K10MERGE FAIL: set E Vast receipts: folds {nov} need one (K10_E_VAST_FOLDS {vast}); folds {box_with} '
                      f'carry one but are not listed')
                return 1
        cv = {r['fold']: r['cv'] for r in X['members']}
        low = {F: (v, bars[F]) for F, v in cv.items() if not v >= bars[F]}
        st['cv'] = cv
        st['vs_ref'] = {F: round(cv[F] - ref[F], 6) for F in cv}
        for F in folds:
            stt = 'NOT FINISHED' if F not in cv else ('BELOW BAR' if F in low else 'OK')
            cvs = f'{cv[F]:.6f} ({cv[F] - ref[F]:+.4f} vs {"C" if s == "E" else "A"})' if F in cv else '-'
            r = {x['fold']: x for x in X['members']}.get(F) or {}
            print(f'  set {s} fold {F}: cv {cvs} E {r.get("epochs")} source {r.get("source")} bar {bars[F]:.6f} [{src[F]}] '
                  f'-> {stt}')
        if low:
            st['status'] = f'FAIL: fold(s) below the sanity bar {low}'
            print(f'K10MERGE FAIL: set {s} fold(s) below their sanity bar {low}: a broken / weak fold. Do not ship; read the '
                  f'run log. To accept anyway (the operator\'s call): K10_{s}_BARS="F:<bar>" (k10_collect.sh)')
            if s == 'E':
                for F, (v, b) in low.items():
                    r = {x['fold']: x for x in X['members']}[F]
                    ls = log_stats(os.path.splitext(r['run'])[0] if r.get('run', '').endswith('.log') else r['run'])
                    floor = C_REF[F] - SOFT_E
                    ok1 = v >= floor
                    ok2 = bool(ls) and ls['last3'] >= ls['best'] - 0.003
                    ok3 = bool(ls) and ls['E'] == r.get('epochs') and ls['n_ep'] == ls['E']
                    print(f'  PRE-REGISTERED OVERRIDE CHECK E f{F}: cv {v:.6f} >= C - {SOFT_E} = {floor:.6f}: {ok1}; '
                          f'last-3 raw val mean {ls and round(ls["last3"], 6)} >= best {ls and round(ls["best"], 6)} - .003: '
                          f'{ok2}; all {ls and ls["E"]} epochs logged ({ls and ls["n_ep"]}): {ok3} -> '
                          + (f'ELIGIBLE: K10_E_BARS="{F}:{v:.6f}" may be set (record it in HANDOFF)' if ok1 and ok2 and ok3
                             else 'NOT ELIGIBLE: do not ship this fold'))
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
        print(f'K10MERGE NOT FINAL: set E {counts["E"]}/5, set D {counts["D"]}/5, set C {counts["C"]}/5. K10 ships only at '
              f'E 5/5 + D 5/5 (+ C 5/5 for FN10-3). For a TEST build only: K10_INTERIM=1 (members.json do_not_submit)')
        return 3
    ids = [(r['seed'], r['fold'], int(r['ttr_size']), r['train_labels']) for r in members]
    files = [r['file'] for r in members]                    # one DEST/ckpt dir: a file name may appear once
    if len(set(ids)) != len(ids) or len(set(files)) != len(files):
        print(f'K10MERGE FAIL: duplicate members {sorted(ids)} / {sorted(files)}'); return 1
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), k10=True,
               set='K10 = E f{0..4}_s1045_ttr288frac75 (TTR288, frac75) + D f{0..4}_s1043_ttr_frac75 (TTR224, frac75); '
                   'C f{0..4}_s1044_ttr288mix50 (TTR288, mix50) for FN10-3 only',
               members=members, skipped=skipped, failed=[], copied=True, set_e=info['E'], set_d=info['D'], set_c=info['C'],
               interim=not final, do_not_submit=not final)
    tmp = f'{a.dest}/members.json.tmp'
    json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, f'{a.dest}/members.json')
    for r in members:
        rc = os.path.basename(r.get('kaggle_receipt') or r.get('vast_receipt') or '-')
        print(f'  {r["set"]} s{r["seed"]} f{r["fold"]} ttr{r["ttr_size"]} {r["train_labels"]} {r["tag"]} E {r.get("epochs")} '
              f'cv {r["cv"]:.6f} {r["file"]} source {r["source"]} receipt {rc}')
    for sk in skipped:
        print(f'  NOT FINISHED set {sk["set"]}: f{sk["fold"]}_s{sk["seed"]}_{sk["suffix"]}: {sk["reason"][:110]}')
    print(f'  set E: {info["E"]["status"]} | set D: {info["D"]["status"]} | set C: {info["C"]["status"]}')
    print(f'K10MERGE OK {len(members)} member(s): E {counts["E"]}/5, D {counts["D"]}/5, C {counts["C"]}/5 ' +
          ('(FINAL)' if final else '(INTERIM: do_not_submit, NEVER SUBMIT)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
