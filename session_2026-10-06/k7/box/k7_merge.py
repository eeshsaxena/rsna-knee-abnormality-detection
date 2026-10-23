#!/usr/bin/env python3
"""k7_merge.py (box, CPU, read-only except DEST/members.json; 2026-10-06, Claude): merge the three K7 picker outputs.

  python k7_merge.py --dest ~/k7_test [--bar 0.8876] [--with_c auto|0]

Inputs (written by k7_collect.sh, one k5_pick.py run per set, renamed after each run):
  DEST/members_A.json  set A  f{0..4}_s1043_ttr_mix50     (required, no failed fold)
  DEST/members_B.json  set B  f{0..4}_s1044_ttr_mix50     (optional: absent = no run final yet; a failed fold = STOP)
  DEST/members_C.json  set C  f0_s1044_ttr288mix50       (optional: included only when every picked C member has
                                                          cv >= --bar (.8876 = J1 .8836 + .004, the box gate T288);
                                                          a failed C check EXCLUDES set C loudly, it never ships)
Output DEST/members.json = members (A, B, C order) + skipped (with seed + suffix) + set_c verdict. Checks: no duplicate
(seed, fold, ttr_size) and no duplicate file name across the sets (K7-KEY: B f0 and C f0 share fold+seed but differ in
ttr_size and file name). Prints every run that is NOT finished, by name.
Exit 0 = written; 1 = set A missing / failed, set B failed, or a duplicate.
"""
import argparse, json, os, sys, time


def load(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=os.path.expanduser('~/k7_test'))
    ap.add_argument('--bar', type=float, default=0.8876)
    ap.add_argument('--with_c', default='auto', choices=('auto', '0'))
    a = ap.parse_args()
    A, B, C = (load(f'{a.dest}/members_{s}.json') for s in 'ABC')
    if not A or A.get('failed') or not A.get('members'):
        print(f'K7MERGE FAIL: set A missing or failed: {None if not A else A.get("failed")}'); return 1
    if B and B.get('failed'):
        print(f'K7MERGE FAIL: set B (s1044 TTR224) has failed fold(s): {B["failed"]} (do not ship; read the FAIL lines)'); return 1
    members = [dict(r, set='A') for r in A['members']]
    skipped = [dict(s, seed=1043, suffix='ttr_mix50', set='A') for s in A['skipped']]
    if B:
        members += [dict(r, set='B') for r in B['members']]
        skipped += [dict(s, seed=1044, suffix='ttr_mix50', set='B') for s in B['skipped']]
    else:
        skipped += [dict(fold=f, seed=1044, suffix='ttr_mix50', set='B', reason='no set B run final (picker rc 3)') for f in range(5)]
    set_c = dict(bar=a.bar, with_c=a.with_c)
    if a.with_c == '0':
        set_c.update(status='EXCLUDED by the operator (K7_WITH_C=0)')
    elif not C:
        set_c.update(status='NOT FINAL (picker rc 3: no TTR288 run has its closing line yet)')
    elif C.get('failed'):
        set_c.update(status=f'EXCLUDED: a picker check FAILED on a finished run: {C["failed"]}')
    elif not C.get('members'):
        set_c.update(status='NOT FINAL')
    else:
        low = [(r['fold'], r['cv']) for r in C['members'] if r['cv'] < a.bar]
        cvs = {r['fold']: r['cv'] for r in C['members']}
        if low:
            set_c.update(status=f'EXCLUDED: picked cv below the bar {a.bar}: {low}', cv=cvs)
        else:
            members += [dict(r, set='C') for r in C['members']]
            set_c.update(status=f'INCLUDED: cv {cvs} >= {a.bar}', cv=cvs)
    if C:
        skipped += [dict(s, seed=1044, suffix='ttr288mix50', set='C') for s in C['skipped']]
    ids = [(r['seed'], r['fold'], int(r['ttr_size'])) for r in members]
    files = [r['file'] for r in members]
    if len(set(ids)) != len(ids) or len(set(files)) != len(files):
        print(f'K7MERGE FAIL: duplicate members {sorted(ids)} / {sorted(files)}'); return 1
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               set='K7 = A f{0..4}_s1043_ttr_mix50 + B f{0..4}_s1044_ttr_mix50 (+ C f0_s1044_ttr288mix50 if >= bar)',
               members=members, skipped=skipped, failed=[], copied=True, set_c=set_c)
    tmp = f'{a.dest}/members.json.tmp'
    json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, f'{a.dest}/members.json')
    for r in members:
        print(f'  {r["set"]} s{r["seed"]} f{r["fold"]} ttr{r["ttr_size"]} {r["tag"]} cv {r["cv"]:.6f} {r["file"]} source {r["source"]}')
    for s in skipped:
        print(f'  NOT FINISHED set {s["set"]}: f{s["fold"]}_s{s["seed"]}_{s["suffix"]}: {s["reason"][:110]}')
    print(f'  set C: {set_c["status"]}')
    n = {k: sum(r['set'] == k for r in members) for k in 'ABC'}
    print(f'K7MERGE OK {len(members)} member(s): A {n["A"]}/5, B {n["B"]}/5, C {n["C"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
