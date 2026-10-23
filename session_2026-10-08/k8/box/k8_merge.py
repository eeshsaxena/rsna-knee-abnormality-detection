#!/usr/bin/env python3
"""k8_merge.py (box, CPU, read-only except DEST/members.json; 2026-10-07, Claude; from k7_merge.py): merge the two K8
picker outputs.

  python k8_merge.py --dest ~/k8_test [--c_margin 0.002] [--c_bars 0:.8876,3:.8740] [--allow_interim]

K8 = parent + [set A + set C], equal members, raw best checkpoints, TTA A+B (NOVEL_PLAN_10-07 sec 2 item 2):
  DEST/members_A.json  set A  f{0..4}_s1043_ttr_mix50      TTR224 (the K5 members; required 5/5, no failed fold)
  DEST/members_C.json  set C  f{0..4}_s1044_ttr288mix50    TTR288 (box chain auto_after_ttr288.sh)
  set B (s1044 TTR224, the teammate's folds) is DROPPED from K8 on: it is not read here, and build_k8.py refuses it.

Set C acceptance (the plan: "accept the 288 set only when all 5 folds have closing lines"):
  FINAL     every fold 0-4 picked (closing line + every picker check), and every fold's cv >= its sanity bar.
  INTERIM   some folds not finished yet (no failure, no fold below its bar): refused (exit 3) unless --allow_interim, which
            writes members.json with interim=True and do_not_submit=True (the plan forbids submitting interim 288
            subsets; build_k8.py writes a build_i<N> dir and k2_check.py refuses an interim build under --kaggle_commit).
  FAIL      a picker check failed on a FINISHED C fold, or a picked C fold is below its bar: exit 1, nothing written.
Per-fold sanity bar (replaces K7's single .8876 = fold 0 J1 + .004, which would wrongly drop the hard fold 3, where the 224
runs read .8763 / .8766): bar_F = (set A's picked cv on fold F) - --c_margin (default .002, ~1.4 x the paired fold SE
.0014). It catches a broken / diverged 288 fold, it is not a gain test. --c_bars F:v,... overrides single folds (the
operator's explicit call, recorded in members.json). Every bar, cv and verdict is printed and stored in set_c.
Output DEST/members.json = members (A then C) + skipped + set_c. Refuses duplicate (seed, fold, ttr_size) / file names.
Exit 0 = written (FINAL, or INTERIM with --allow_interim); 1 = check failed; 3 = set C not complete (no --allow_interim).
"""
import argparse, json, os, sys, time


def load(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=os.path.expanduser('~/k8_test'))
    ap.add_argument('--c_margin', type=float, default=0.002)
    ap.add_argument('--c_bars', default='', help='F:bar[,F:bar] explicit per-fold bars (override the A-relative rule)')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--allow_interim', action='store_true')
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(',') if x.strip()]
    A, C = load(f'{a.dest}/members_A.json'), load(f'{a.dest}/members_C.json')
    if not A or A.get('failed') or not A.get('members'):
        print(f'K8MERGE FAIL: set A missing or failed: {None if not A else A.get("failed")}'); return 1
    a_cv = {r['fold']: r['cv'] for r in A['members']}
    if sorted(a_cv) != folds:
        print(f'K8MERGE FAIL: set A has folds {sorted(a_cv)}, K8 needs all of {folds} (skipped: {A.get("skipped")})'); return 1
    if any(r.get('tag') != 'raw' for r in A['members']):
        print(f'K8MERGE FAIL: set A picks {[(r["fold"], r["tag"]) for r in A["members"]]}: K8 ships raw checkpoints'); return 1
    over = {}
    for x in [y.strip() for y in a.c_bars.split(',') if y.strip()]:
        f, _, v = x.partition(':')
        over[int(f)] = float(v)
    bars = {F: (over[F] if F in over else a_cv[F] - a.c_margin) for F in folds}   # FIX 10-08: unrounded (round(.., 6) could
    #                                     # move a bar up by <= 5e-7 and fail a fold sitting exactly on it); printed with 6 decimals
    bar_src = {F: ('operator --c_bars' if F in over else f'set A f{F} cv {a_cv[F]:.6f} - {a.c_margin}') for F in folds}
    members = [dict(r, set='A') for r in A['members']]
    skipped = [dict(s, seed=1043, suffix='ttr_mix50', set='A') for s in A['skipped']]
    set_c = dict(margin=a.c_margin, bars=bars, bar_source=bar_src, overrides=over)
    if not C:
        print('K8MERGE: set C picker output missing (picker rc 3: no TTR288 fold has its closing line?)')
        C = dict(members=[], skipped=[dict(fold=F, reason='no members_C.json') for F in folds], failed=[])
    if C.get('failed'):
        print(f'K8MERGE FAIL: set C (s1044 TTR288) picker check FAILED on a finished fold: {C["failed"]} (read the FAIL lines)')
        return 1
    c_cv = {r['fold']: r['cv'] for r in C['members']}
    low = {F: (cv, bars[F]) for F, cv in c_cv.items() if not cv >= bars[F]}
    set_c['cv'] = c_cv
    set_c['vs_A'] = {F: round(c_cv[F] - a_cv[F], 6) for F in c_cv}
    for F in folds:
        st = ('NOT FINISHED' if F not in c_cv else ('BELOW BAR' if F in low else 'OK'))
        cvs = f'{c_cv[F]:.6f} ({c_cv[F] - a_cv[F]:+.4f} vs A)' if F in c_cv else '-'
        print(f'  set C fold {F}: cv {cvs} bar {bars[F]:.6f} [{bar_src[F]}] -> {st}')
    if low:
        set_c['status'] = f'FAIL: fold(s) below the sanity bar {low}'
        print(f'K8MERGE FAIL: set C fold(s) below their sanity bar {low}: a broken / weak 288 fold. Do not ship; read the '
              f'run log. To accept anyway (the operator\'s call), re-run with K8_C_BARS="F:<bar>" (k8_collect.sh)')
        return 1
    if any(r.get('tag') != 'raw' for r in C['members']):
        print(f'K8MERGE FAIL: set C picks {[(r["fold"], r["tag"]) for r in C["members"]]}: K8 ships raw checkpoints'); return 1
    members += [dict(r, set='C') for r in C['members']]
    skipped += [dict(s, seed=1044, suffix='ttr288mix50', set='C') for s in C['skipped']]
    n_c = len(c_cv)
    final = sorted(c_cv) == folds
    if final:
        set_c['status'] = f'FINAL: {n_c}/5 folds, every fold >= its bar'
    else:
        missing = [F for F in folds if F not in c_cv]
        set_c['status'] = f'INTERIM: {n_c}/5 folds (not finished: {missing})'
        if not a.allow_interim:
            for s in skipped:
                print(f'  NOT FINISHED set {s["set"]}: f{s["fold"]}_s{s["seed"]}_{s["suffix"]}: {s["reason"][:110]}')
            print(f'K8MERGE NOT FINAL: set C {n_c}/5 (folds {missing} have no closing line yet). The plan accepts the 288 '
                  f'set only at 5/5. For a TEST build only: K8_INTERIM=1 (members.json is marked do_not_submit)')
            return 3
    ids = [(r['seed'], r['fold'], int(r['ttr_size'])) for r in members]
    files = [r['file'] for r in members]
    if len(set(ids)) != len(ids) or len(set(files)) != len(files):
        print(f'K8MERGE FAIL: duplicate members {sorted(ids)} / {sorted(files)}'); return 1
    if any((r['seed'], int(r['ttr_size'])) not in ((1043, 224), (1044, 288)) for r in members):
        print(f'K8MERGE FAIL: a member outside sets A (1043, 224) / C (1044, 288): {ids}'); return 1
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               set='K8 = A f{0..4}_s1043_ttr_mix50 (TTR224) + C f{0..4}_s1044_ttr288mix50 (TTR288); set B dropped',
               members=members, skipped=skipped, failed=[], copied=True, set_c=set_c, interim=not final,
               do_not_submit=not final)
    tmp = f'{a.dest}/members.json.tmp'
    json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, f'{a.dest}/members.json')
    for r in members:
        print(f'  {r["set"]} s{r["seed"]} f{r["fold"]} ttr{r["ttr_size"]} {r["tag"]} cv {r["cv"]:.6f} {r["file"]} source {r["source"]}')
    for s in skipped:
        print(f'  NOT FINISHED set {s["set"]}: f{s["fold"]}_s{s["seed"]}_{s["suffix"]}: {s["reason"][:110]}')
    print(f'  set C: {set_c["status"]}')
    print(f'K8MERGE OK {len(members)} member(s): A 5/5, C {n_c}/5 ' +
          ('(FINAL)' if final else '(INTERIM: do_not_submit, NEVER SUBMIT)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
