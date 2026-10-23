#!/usr/bin/env python3
"""t_k9_pick.py (box, CPU; TEST ONLY; 2026-10-08, Claude; from t_k8_pick.py): the K9 picker (k5_pick.py K9 copy: K9-REC per
fold receipt seeds, K9-KEY --key_tag) + k9_merge.py on the simulation root ~/k9_test/sim/root (k9_sim_setup.py: D f0/f1 and
C f0-f3 REAL symlinks, D f2-f4 + C f4 simulated, kmrsuman-format receipts) and the real ~/fleet_runs (read only). Writes
only under ~/k9_test/sim/tpick.

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python t_k9_pick.py [--box ~/k9_test/simtest/k5/box]
                                                                         [--k8box ~/k8_test/simtest/k5/box]
T1  set D (sim): 5 raw picks, train_labels frac75, D f0/f1 vouched by the f01 receipt, f2/f3 by f23, f4 by f4x (sha ==
    receipt), OOF copies oof_f{F}_s1043f75_raw_cv<N>.csv, record key_tag f75.
T2  the K8 picker (unchanged copy) on the same set D -> crashes (TypeError: int() of the per-fold SEED map): the bug K9-REC fixes.
T3  receipt order [f4x, f23, f01]: D f0 is still vouched by f01 (the f4x map gives fold 0 seed 1044 -> skipped).
T4  --require_receipt with no frac75 receipt (only the default f34 one, whose runs name the mix50 runs) -> FAIL on every fold.
T5  a receipt with a wrong sha256 for f2 -> FAIL on f2 (sha != receipt).
T6  set D with the default --train_labels (mix50) -> FAIL (train_labels); set A runs offered as set D (suffix ttr_mix50,
    --train_labels frac75) -> FAIL (train_labels).
T7  --min_epochs 18 -> the E17 folds f0/f1 FAIL (E < min); 14 (the K9 default) and 16 pass.
T8  set C (sim): 5 raw TTR288 picks in the same dest, D picks untouched.
T9  merge FINAL (sim): 10 members C 5 + D 5, set_c + set_d FINAL, bars D = A - .004 and C = A - .002.
T10 merge bars: --d_margin 0 -> D f1 (.8910 < A .8926) below its bar -> rc 1, nothing written; --d_bars 1:0.85 override
    with --d_margin 0 -> still rc 1 only if another fold is below; checked on f1 alone.
T11 merge refuses a set A (mix50 s1043) member smuggled into members_D.json; and an ema pick; and a duplicate D member.
T12 merge on the REAL ~/fleet_runs (D 2/5, C 4/5 at 04:45 10-08): rc 3 without --allow_interim; with it interim +
    do_not_submit, unfinished D f2-f4 listed.
"""
import argparse, json, os, shutil, subprocess, sys

H = os.path.expanduser('~')
R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {extra}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--box', default=f'{H}/k9_test/simtest/k5/box')
    ap.add_argument('--k8box', default=f'{H}/k8_test/simtest/k5/box')
    a = ap.parse_args()
    PY = sys.executable
    PICK = [PY, '-W', 'ignore', f'{a.box}/k5_pick.py', '--copy']
    MERGE = [PY, '-W', 'ignore', f'{a.box}/k9_merge.py']
    SIM = f'{H}/k9_test/sim/root'
    T = f'{H}/k9_test/sim/tpick'
    shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
    REC = lambda root, order=('f01', 'f23', 'f4x'): sum((['--kaggle_receipt', f'{root}/kaggle_frac75_{t}_run_receipt.json']
                                                        for t in order), [])
    DARGS = ['--seed', '1043', '--suffix', 'ttr_frac75', '--ttr_size', '224', '--train_labels', 'labels_frac75_logit.csv',
             '--key_tag', 'f75', '--require_receipt', '--tag', 'raw', '--min_epochs', '14']
    CARGS = ['--seed', '1044', '--suffix', 'ttr288mix50', '--folds', '0,1,2,3,4', '--ttr_size', '288', '--no_kaggle_receipt',
             '--tag', 'raw']

    def pick(dest, *args, cmd=None):
        os.makedirs(f'{T}/{dest}', exist_ok=True)
        if os.path.isfile(f'{T}/{dest}/members.json'):
            os.remove(f'{T}/{dest}/members.json')
        r = subprocess.run((cmd or PICK) + ['--dest', f'{T}/{dest}', *args], capture_output=True, text=True)
        m = json.load(open(f'{T}/{dest}/members.json')) if os.path.isfile(f'{T}/{dest}/members.json') else None
        open(f'{T}/{dest}.log', 'a').write(r.stdout + r.stderr)
        return r.returncode, m, r.stdout + r.stderr

    def setfile(dest, name):
        if os.path.isfile(f'{T}/{dest}/members.json'):
            os.replace(f'{T}/{dest}/members.json', f'{T}/{dest}/members_{name}.json')

    def merge(dest, *args):
        if os.path.isfile(f'{T}/{dest}/members.json'):
            os.remove(f'{T}/{dest}/members.json')
        r = subprocess.run(MERGE + ['--dest', f'{T}/{dest}', *args], capture_output=True, text=True)
        open(f'{T}/{dest}.merge.log', 'a').write(r.stdout + r.stderr)
        m = json.load(open(f'{T}/{dest}/members.json')) if os.path.isfile(f'{T}/{dest}/members.json') else None
        return r.returncode, m, r.stdout + r.stderr

    last = lambda out: out.strip().splitlines()[-1][:160] if out.strip() else ''
    # T1
    rc, mD, out = pick('d1', '--root', SIM, *DARGS, *REC(SIM))
    want_rec = {0: 'f01', 1: 'f01', 2: 'f23', 3: 'f23', 4: 'f4x'}
    good = rc == 0 and mD and len(mD['members']) == 5 and all(
        x['tag'] == 'raw' and x['train_labels'] == 'labels_frac75_logit.csv' and x['source'] == 'kaggle'
        and os.path.basename(x['kaggle_receipt'] or '') == f'kaggle_frac75_{want_rec[x["fold"]]}_run_receipt.json'
        and os.path.basename(x['oof']).startswith(f'oof_f{x["fold"]}_s1043f75_raw_cv') and x['key_tag'] == 'f75'
        and abs(x['oof_rescored_cv'] - x['cv']) <= 1e-12 for x in mD['members'])
    ok('T1 set D sim: 5 raw frac75 picks, receipts f01/f01/f23/f23/f4x, f75 OOF names', good,
       f'rc {rc} {[(x["fold"], x["epochs"], os.path.basename(x["kaggle_receipt"] or "-"), os.path.basename(x["oof"])) for x in (mD or {}).get("members", [])]} {last(out)}')
    setfile('d1', 'D')
    # T2: the unchanged K8 picker on the same receipts
    k8 = [PY, '-W', 'ignore', f'{a.k8box}/k5_pick.py', '--copy']
    k8args = [x for x in DARGS if x not in ('--key_tag', 'f75')]
    rc, m, out = pick('d2', '--root', SIM, *k8args, *REC(SIM), cmd=k8)
    ok('T2 the K8 picker crashes on the per-fold SEED receipts (the bug K9-REC fixes)', rc != 0 and 'TypeError' in out,
       f'rc {rc} {last(out)}')
    # T3
    rc, m, out = pick('d3', '--root', SIM, *DARGS, *REC(SIM, ('f4x', 'f23', 'f01')))
    f0 = [x for x in (m or {}).get('members', []) if x['fold'] == 0]
    ok('T3 receipt order f4x first: D f0 still vouched by f01 (f4x maps fold 0 to seed 1044)', rc == 0 and len(m['members']) == 5
       and f0 and os.path.basename(f0[0]['kaggle_receipt']) == 'kaggle_frac75_f01_run_receipt.json',
       f'rc {rc} {[os.path.basename(x["kaggle_receipt"] or "-") for x in (m or {}).get("members", [])]}')
    # T4
    rc, m, out = pick('d4', '--root', SIM, *DARGS)
    ok('T4 --require_receipt without frac75 receipts (only f34) -> FAIL on every fold', rc == 1 and m is not None
       and sorted(f['fold'] for f in m['failed']) == [0, 1, 2, 3, 4] and all('no Kaggle run receipt' in f['reason'] for f in m['failed']),
       f'rc {rc} {last(out)}')
    # T5
    os.makedirs(f'{T}/rec5', exist_ok=True)
    for t in ('f01', 'f23', 'f4x'):
        shutil.copyfile(f'{SIM}/kaggle_frac75_{t}_run_receipt.json', f'{T}/rec5/kaggle_frac75_{t}_run_receipt.json')
    r23 = json.load(open(f'{T}/rec5/kaggle_frac75_f23_run_receipt.json'))
    for e in r23['train']['folds']['2']['files']:
        if '_raw_' in e['path'] and e['path'].endswith('_best.pt'):
            e['sha256'] = 'f' * 64
    json.dump(r23, open(f'{T}/rec5/kaggle_frac75_f23_run_receipt.json', 'w'))
    rc, m, out = pick('d5', '--root', SIM, *DARGS, *REC(f'{T}/rec5'))
    ok('T5 wrong receipt sha256 for D f2 -> FAIL on f2 only', rc == 1 and m is not None
       and [f['fold'] for f in m['failed']] == [2] and 'sha256 != the Kaggle run receipt' in m['failed'][0]['reason'],
       f'rc {rc} {[(f["fold"], f["reason"][:80]) for f in (m or {}).get("failed", [])]}')
    # T6
    a6 = [x for x in DARGS if x not in ('--train_labels', 'labels_frac75_logit.csv')]
    rc, m, out = pick('d6a', '--root', SIM, *a6, *REC(SIM))
    ok6a = rc == 1 and m is not None and len(m['failed']) == 5 and all('train_labels' in f['reason'] for f in m['failed'])
    a6b = ['--seed', '1043', '--suffix', 'ttr_mix50', '--ttr_size', '224', '--train_labels', 'labels_frac75_logit.csv', '--tag', 'raw']
    rc2, m2, out2 = pick('d6b', '--root', f'{H}/fleet_runs', *a6b)
    ok6b = rc2 == 1 and m2 is not None and len(m2['failed']) == 5 and all('train_labels' in f['reason'] for f in m2['failed'])
    ok('T6 set D picked with mix50 labels -> FAIL; set A runs offered as frac75 -> FAIL (train_labels)', ok6a and ok6b,
       f'rc {rc}/{rc2}')
    # T7
    a7 = [x for x in DARGS if x not in ('--min_epochs', '14')]
    rc, m, out = pick('d7a', '--root', SIM, *a7, '--min_epochs', '18', *REC(SIM))
    f7 = sorted(f['fold'] for f in (m or {}).get('failed', []))
    ok7 = rc == 1 and f7 == [0, 1] and all('E 17 < --min_epochs 18' in f['reason'] for f in m['failed'])
    rc16, m16, _ = pick('d7b', '--root', SIM, *a7, '--min_epochs', '16', *REC(SIM))
    ok('T7 --min_epochs 18 fails the E17 folds f0/f1; 16 and 14 pass', ok7 and rc16 == 0 and len(m16['members']) == 5,
       f'rc {rc} failed {f7} | rc16 {rc16}')
    # T8 + T9 (one dest)
    rcC, mC, out = pick('d1', '--root', SIM, *CARGS); setfile('d1', 'C')
    good = (rcC == 0 and len(mC['members']) == 5 and all(int(x['ttr_size']) == 288 and
            os.path.basename(x['oof']).startswith(f'oof_f{x["fold"]}_s1044t288_raw_cv') for x in mC['members'])
            and all(os.path.isfile(x['copy']) for x in mD['members']))
    ok('T8 set C sim: 5 raw TTR288 picks, t288 OOF names, D picks in the same dest untouched', good, f'rc {rcC} {last(out)}')
    rc, m, out = merge('d1')
    bars_ok = m is not None and all(abs(float(m['set_d']['bars'][str(F)] if str(F) in m['set_d']['bars'] else m['set_d']['bars'][F])
                                        - (A - .004)) < 1e-6 for F, A in enumerate(
        [0.8835985889396399, 0.8926246223939701, 0.8907517323356613, 0.8763090478076047, 0.8928147656091167]))
    ok('T9 merge FINAL (sim): 10 members C 5 + D 5, both FINAL, D bars = A - .004', rc == 0 and len(m['members']) == 10
       and m['set_c']['status'].startswith('FINAL') and m['set_d']['status'].startswith('FINAL') and m['interim'] is False
       and m['do_not_submit'] is False and bars_ok, f'rc {rc} {last(out)}')
    # T10
    rc, m, out = merge('d1', '--d_margin', '0')
    ok('T10 --d_margin 0: D f1 (.8910 < A .8926) below its bar -> rc 1, nothing written', rc == 1 and m is None
       and 'set D fold(s) below their sanity bar' in out and '1: (' in out, f'rc {rc} {last(out)}')
    rc, m, out = merge('d1', '--d_margin', '0', '--d_bars', '1:0.85')
    ok('T10b operator override 1:0.85 with --d_margin 0 -> rc 0, override recorded', rc == 0 and m is not None
       and m['set_d']['overrides'] in ({'1': 0.85}, {1: 0.85}), f'rc {rc} {last(out)}')
    # T11
    os.makedirs(f'{T}/d11', exist_ok=True)
    shutil.copyfile(f'{T}/d1/members_C.json', f'{T}/d11/members_C.json')
    d = json.load(open(f'{T}/d1/members_D.json'))
    bad = dict(d['members'][2], train_labels='labels_mix50_logit.csv')
    json.dump(dict(d, members=d['members'][:2] + [bad] + d['members'][3:]), open(f'{T}/d11/members_D.json', 'w'))
    rcA, mA, outA = merge('d11')
    json.dump(dict(d, members=d['members'][:2] + [dict(d['members'][2], tag='ema')] + d['members'][3:]),
              open(f'{T}/d11/members_D.json', 'w'))
    rcE, mE, outE = merge('d11')
    json.dump(dict(d, members=d['members'] + [d['members'][0]]), open(f'{T}/d11/members_D.json', 'w'))
    rcU, mU, outU = merge('d11')
    ok('T11 merge refuses a mix50 (set A) member in D, an ema pick, a duplicate D member', rcA == 1 and mA is None
       and rcE == 1 and mE is None and rcU == 1 and mU is None, f'rc {rcA}/{rcE}/{rcU} {last(outA)} | {last(outU)}')
    # T12 (real runs)
    rcC, mC, _ = pick('d12', '--root', f'{H}/fleet_runs', *CARGS); setfile('d12', 'C')
    rcD, mD2, outD = pick('d12', '--root', f'{H}/fleet_runs', *DARGS, *REC(f'{H}/fleet_runs')); setfile('d12', 'D')
    nC, nD = len((mC or {}).get('members', [])), len((mD2 or {}).get('members', []))
    rc3, m3, out3 = merge('d12')
    rci, mi, outi = merge('d12', '--allow_interim')
    fin_d = [x['fold'] for x in (mD2 or {}).get('members', [])]
    good = (rcC == 0 and rcD == 0 and 1 <= nD and (nC < 5 or nD < 5) and rc3 == 3 and m3 is None and rci == 0
            and mi['interim'] is True and mi['do_not_submit'] is True and len(mi['members']) == nC + nD
            and sorted(s['fold'] for s in mi['skipped'] if s['set'] == 'D') == [F for F in range(5) if F not in fin_d])
    ok(f'T12 real runs (C {nC}/5, D {nD}/5): rc 3 without --allow_interim; with it interim + do_not_submit, unfinished listed',
       good, f'rc {rcC}/{rcD} merge {rc3}/{rci} {last(outi)}')
    print(f'TK9PICK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
    return 0 if all(R) else 1


if __name__ == '__main__':
    sys.exit(main())
