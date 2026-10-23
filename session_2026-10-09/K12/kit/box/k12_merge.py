#!/usr/bin/env python3
"""k12_merge.py (box, CPU, read-only except DEST/members.json; 2026-10-09, Claude; from k11_merge.py): ONE members.json for
the K12 builds = the FINAL FN11 members C + D + E + E2 (~/k11_test/members.json, read only; every pick asserted == the
shipped k11w bytes) + set DO = D-OAI f{F}_s1043_ttr_frac75oaiT12v00 (picker output DEST/members_DO.json).

PRE-REGISTERED (K12/PREREG_K12.md sec 2-3, sha256 31884631..., written 10-09 ~22:40 IST before any D-OAI f1-f4 read):
  DO fold F ACCEPTED iff every picker check passed (k5_pick.py: train_labels labels_frac75oai_T12v00.csv, seed 1043,
     ttr 224, min epochs 14, box mtime rule) AND the K12-OAI-CLEAN receipt (box/k12_oai_clean.py) lists its checkpoint
     (== the picked src) and its OOF (vs set D's OOF of the fold) with 0 hits AND cv >= D_F - .005 (no override).
  f0 = the 17-epoch probe, used as is. K12 needs all 5 folds; fewer -> INTERIM (TEST builds only, --allow_interim).
  --sim_do JSON: TEST ONLY, DO records synthesized for not-yet-trained folds (marked simulated); members.json simulated.
  python k12_merge.py --dest ~/k12_test [--base ~/k11_test/members.json] [--clean ~/k12_test/clean/clean_DO.json]
                      [--allow_interim] [--sim_do ~/k12_test/sim/members_DO_sim.json]
Exit 0 = written; 1 = check failed; 3 = DO not 5/5 (no --allow_interim).
"""
import argparse, json, os, sys, time

H = os.path.expanduser('~')
D_REF = {0: 0.883777, 1: 0.890967, 2: 0.890903, 3: 0.876911, 4: 0.892889}          # PREREG_K12 sec 3 (6 dp)
E2_SHA16 = {0: '24f053e0a93f775e', 1: '0693ed10e3a130e1', 2: '120c97b20a415b6d', 3: '2d67fc874ccf7374',
            4: '6e751cabe98dfec0'}                                        # the FN11 E2 picks (k11/build_w11)
MARGIN = 0.005
DO = dict(seed=1043, ttr=224, labels='labels_frac75oai_T12v00.csv', suffix='ttr_frac75oaiT12v00')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=f'{H}/k12_test')
    ap.add_argument('--base', default=f'{H}/k11_test/members.json')
    ap.add_argument('--clean', default=f'{H}/k12_test/clean/clean_DO.json')
    ap.add_argument('--allow_interim', action='store_true')
    ap.add_argument('--sim_do', default=None)
    a = ap.parse_args()
    assert os.path.realpath(a.dest).startswith(os.path.realpath(f'{H}/k12_test')), 'k12_merge writes only under ~/k12_test'
    base = json.load(open(a.base))
    assert base.get('k11') is True and not base.get('interim') and not base.get('do_not_submit') and not base.get('failed')
    bm = base['members']
    cnt = {s: sorted(r['fold'] for r in bm if r['set'] == s) for s in ('C', 'D', 'E', 'E2')}
    assert cnt == {s: [0, 1, 2, 3, 4] for s in cnt} and len(bm) == 20, cnt
    for r in bm:
        if r['set'] == 'E2' and r['sha256'][:16] != E2_SHA16[r['fold']]:
            print(f'K12MERGE FAIL: base E2 f{r["fold"]} sha256 != the FN11 pick'); return 1
    dref = {r['fold']: r['cv'] for r in bm if r['set'] == 'D'}
    for F, v in D_REF.items():
        assert abs(dref[F] - v) < 6e-7, (F, dref[F], v)
    X = json.load(open(f'{a.dest}/members_DO.json')) if os.path.isfile(f'{a.dest}/members_DO.json') else \
        dict(members=[], skipped=[dict(fold=F, reason='no picker output') for F in range(5)], failed=[])
    sim = json.load(open(a.sim_do)) if a.sim_do else None
    clean = json.load(open(a.clean)) if os.path.isfile(a.clean) else dict(items=[])
    ok_ck = {os.path.realpath(i['ckpt']): i for i in clean['items'] if not i['hits']}
    acc, excl, skipped = [], [], []
    for F in range(5):
        r = {x['fold']: x for x in X['members']}.get(F)
        if r is None and sim:
            r = {x['fold']: x for x in sim['members']}.get(F)
        fl = {x['fold']: x for x in X.get('failed', [])}.get(F)
        if r is None:
            why = (fl or {}).get('reason') or ({x['fold']: x for x in X.get('skipped', [])}.get(F) or {}).get('reason', '?')
            (excl if fl else skipped).append(dict(fold=F, set='DO', reason=str(why)[:200]))
            print(f'  set DO fold {F}: {"FAILED A CHECK (excluded)" if fl else "NOT FINISHED"}: {str(why)[:120]}')
            continue
        bad = []
        if r.get('tag') != 'raw' or int(r.get('seed', -1)) != DO['seed'] or int(r.get('ttr_size', -1)) != DO['ttr'] \
                or r.get('train_labels') != DO['labels'] or r.get('suffix') != DO['suffix']:
            bad.append('identity (raw, seed 1043, ttr 224, oai T12v00 labels, suffix)')
        ci = ok_ck.get(os.path.realpath(r['src']))
        if ci is None:
            bad.append(f'no K12-OAI-CLEAN PASS item for {r["src"]}')
        elif not r.get('simulated') and not (ci.get('oof') and os.path.realpath(ci['oof']) == os.path.realpath(r['oof_src'])
                                             and ci.get('ref')):
            bad.append('the clean receipt did not scan this fold OOF against set D')
        ok_bar = r['cv'] >= D_REF[F] - MARGIN
        tag = 'SIMULATED (TEST ONLY)' if r.get('simulated') else 'real'
        print(f'  set DO fold {F} [{tag}]: cv {r["cv"]:.6f} ({r["cv"] - D_REF[F]:+.4f} vs D) E {r.get("epochs")} source '
              f'{r.get("source")} bar {D_REF[F] - MARGIN:.6f} clean {"PASS" if ci else "MISSING"} -> '
              + ('ACCEPTED' if ok_bar and not bad else f'EXCLUDED ({"; ".join(bad) or "below bar"})'))
        if ok_bar and not bad:
            acc.append(dict(r, set='DO', k12_clean=a.clean))
        else:
            excl.append(dict(fold=F, set='DO', reason='; '.join(bad) or f'cv {r["cv"]:.6f} below bar'))
    folds = [r['fold'] for r in acc]
    nsim = sum(bool(r.get('simulated')) for r in acc)
    final = folds == [0, 1, 2, 3, 4] and not nsim
    st = dict(margin=MARGIN, ref='set D picked raw cv', accepted=folds, excluded=excl, simulated_folds=nsim,
              status=('FINAL: 5/5 accepted' if final else f'INTERIM: {len(folds)}/5 accepted {folds}, {nsim} simulated'))
    if not final and not a.allow_interim:
        print(f'K12MERGE NOT FINAL: DO {len(folds)}/5 ({nsim} simulated). TEST builds only with --allow_interim'); return 3
    out = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), k10=True, k11=True, k12=True,
               set='K12 = C + D + E + E2 (the FINAL FN11 members) + DO f{0..4}_s1043_ttr_frac75oaiT12v00 (TTR224 frac75 + OAI T12 V00)',
               base=a.base, members=[dict(r) for r in bm] + acc, skipped=skipped, failed=[], set_do=st,
               simulated=bool(nsim), interim=not final, do_not_submit=not final)
    tmp = f'{a.dest}/members.json.tmp'
    json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, f'{a.dest}/members.json')
    print(f'K12MERGE OK {len(out["members"])} member(s): C 5/5, D 5/5, E 5/5, E2 5/5, DO {len(folds)}/5 '
          + ('(FINAL)' if final else f'(INTERIM{", SIMULATED " + str(nsim) if nsim else ""}: do_not_submit, NEVER SUBMIT)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
