#!/usr/bin/env python3
"""t_k7_pick.py (box, CPU; TEST ONLY; 2026-10-06, Claude): the K7 picker (k5_pick.py K7 copy) + k7_merge.py on the real
set A runs (read only) and the simulation root ~/k7_sim/root (k7_sim_setup.py). Writes only under ~/k7_sim/tpick.

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python t_k7_pick.py [--box ~/k7_simtest/k5/box]

T1 regression: set A with the K6 command line (no new flags) picks exactly what K6 picked (~/k6_test/members_mix50.json:
   file, sha256, tag, epoch, cv, source, receipt) and writes the same OOF copy names.
T2 set B (s1044) with the 3 receipts + --require_receipt: 5 picks, all source kaggle, fold -> receipt f01/f01/f23/f23/f4.
T3 set B without any receipt + --require_receipt: every fold FAILS naming the receipt (rc 1).
T4 set B with only the seed-1043 f34 receipt: folds 3-4 are NOT vouched by it (seed mismatch) -> FAIL (rc 1).
T5 set C with set B's receipts and no --no_kaggle_receipt: the fold-0 s1044 receipt names run f0_s1044_ttr_mix50, so it
   does not vouch for f0_s1044_ttr288mix50 -> C picked as a box run (mtime rule), kaggle_receipt None.
T6 set C with the default --ttr_size 224: FAIL on the ttr_size check.
T7 one dest, B then C then B again: both fold-0 s1044 picks stay (recipe-aware stale removal), OOF copies
   oof_f0_s1044_raw_cv8835.csv and oof_f0_s1044t288_raw_cv8890.csv both present; a planted stale TTR224 file of fold 0
   s1044 IS removed while the TTR288 pick stays.
T8 k7_merge: C == B (a true duplicate: same run twice) -> K7MERGE FAIL; C cv below the bar -> C EXCLUDED;
   --with_c 0 -> EXCLUDED; the normal merge -> 11 members, C INCLUDED.
"""
import argparse, glob, json, os, shutil, subprocess, sys

H = os.path.expanduser('~')
R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {extra}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--box', default=f'{H}/k7_simtest/k5/box')
    a = ap.parse_args()
    PY = sys.executable
    PICK = [PY, '-W', 'ignore', f'{a.box}/k5_pick.py', '--copy']
    SIM = f'{H}/k7_sim/root'
    T = f'{H}/k7_sim/tpick'
    shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
    REC = sum((['--kaggle_receipt', f'{SIM}/kaggle_s1044_{t}_run_receipt.json'] for t in ('f01', 'f23', 'f4')), [])

    def pick(dest, *args):
        r = subprocess.run(PICK + ['--dest', f'{T}/{dest}', *args], capture_output=True, text=True)
        m = json.load(open(f'{T}/{dest}/members.json')) if os.path.isfile(f'{T}/{dest}/members.json') else None
        open(f'{T}/{dest}.log', 'a').write(r.stdout + r.stderr)
        return r.returncode, m, r.stdout

    # T1
    rc, m, _ = pick('d1')
    k6 = json.load(open(f'{H}/k6_test/members_mix50.json'))
    keys = ('fold', 'tag', 'epoch', 'cv', 'file', 'sha256', 'source', 'kaggle_receipt')
    same = rc == 0 and [{k: x[k] for k in keys} for x in m['members']] == [{k: x[k] for k in keys} for x in k6['members']]
    oofn = sorted(os.path.basename(x['oof']) for x in m['members']) == sorted(os.path.basename(x['oof']) for x in k6['members'])
    ok('T1 set A K6 command line == K6 picks', same and oofn, f'rc {rc} files {[x["file"] for x in m["members"]]}')
    # T2
    rc, m, _ = pick('d2', '--root', SIM, '--seed', '1044', '--require_receipt', *REC)
    rec = [os.path.basename(x['kaggle_receipt'] or '') for x in m['members']]
    ok('T2 set B 5 picks via 3 receipts', rc == 0 and len(m['members']) == 5 and all(x['source'] == 'kaggle' for x in m['members'])
       and rec == ['kaggle_s1044_f01_run_receipt.json'] * 2 + ['kaggle_s1044_f23_run_receipt.json'] * 2 + ['kaggle_s1044_f4_run_receipt.json'],
       f'rc {rc} {rec}')
    # T3
    rc, m, _ = pick('d3', '--root', SIM, '--seed', '1044', '--require_receipt')
    ok('T3 set B without receipts + --require_receipt FAILS', rc == 1 and len(m['failed']) == 5
       and all('no Kaggle run receipt vouches' in f['reason'] for f in m['failed']), f'rc {rc}')
    # T4
    rc, m, _ = pick('d4', '--root', SIM, '--seed', '1044', '--folds', '3,4', '--require_receipt',
                    '--kaggle_receipt', f'{H}/fleet_runs/kaggle_f34_run_receipt.json')
    ok('T4 the seed-1043 f34 receipt never vouches for s1044 folds 3-4', rc == 1 and [f['fold'] for f in m['failed']] == [3, 4]
       and all('no Kaggle run receipt vouches' in f['reason'] for f in m['failed']), f'rc {rc}')
    # T5
    rc, m, _ = pick('d5', '--root', SIM, '--seed', '1044', '--suffix', 'ttr288mix50', '--folds', '0', '--ttr_size', '288', *REC)
    ok('T5 the s1044 fold-0 Kaggle receipt does not vouch for the TTR288 box run', rc == 0 and m['members'][0]['source'] == 'box'
       and m['members'][0]['kaggle_receipt'] is None, f'rc {rc}')
    # T6
    rc, m, _ = pick('d6', '--root', SIM, '--seed', '1044', '--suffix', 'ttr288mix50', '--folds', '0', '--no_kaggle_receipt')
    ok('T6 set C with the default --ttr_size 224 FAILS', rc == 1 and 'want --ttr_size 224' in m['failed'][0]['reason'], f'rc {rc}')
    # T7
    os.makedirs(f'{T}/d7/ckpt')
    rcb, mb, _ = pick('d7', '--root', SIM, '--seed', '1044', '--folds', '0', '--require_receipt', *REC)
    shutil.copy(f'{T}/d7/members.json', f'{T}/d7/members_B.json')
    rcc, mc, _ = pick('d7', '--root', SIM, '--seed', '1044', '--suffix', 'ttr288mix50', '--folds', '0', '--ttr_size', '288', '--no_kaggle_receipt')
    shutil.copy(f'{T}/d7/members.json', f'{T}/d7/members_C.json')
    fb, fc = mb['members'][0]['file'], mc['members'][0]['file']
    both = os.path.isfile(f'{T}/d7/ckpt/{fb}') and os.path.isfile(f'{T}/d7/ckpt/{fc}')
    oofs = sorted(os.path.basename(p) for p in glob.glob(f'{T}/d7/oof/*.csv'))
    stale = f'{T}/d7/ckpt/f0_s1044_ema_ttr224-320x24_cv0000_best.pt'
    open(stale, 'w').write('stale')
    rcb2, _, out = pick('d7', '--root', SIM, '--seed', '1044', '--folds', '0', '--require_receipt', *REC)
    left = sorted(os.path.basename(p) for p in glob.glob(f'{T}/d7/ckpt/*.pt'))
    ok('T7 B and C fold-0 s1044 picks coexist; stale TTR224 file removed, TTR288 pick kept',
       rcb == rcc == rcb2 == 0 and both and oofs == ['oof_f0_s1044_raw_cv8835.csv', 'oof_f0_s1044t288_raw_cv8890.csv']
       and not os.path.exists(stale) and left == sorted([fb, fc]), f'{left} {oofs}')
    # T8 merge
    MERGE = [PY, f'{a.box}/k7_merge.py']
    d8 = f'{T}/d8'; os.makedirs(d8)
    shutil.copy(f'{T}/d1/members.json', f'{d8}/members_A.json')
    shutil.copy(f'{T}/d2/members.json', f'{d8}/members_B.json')
    shutil.copy(f'{T}/d5/members.json', f'{d8}/members_C.json')
    r = subprocess.run(MERGE + ['--dest', d8], capture_output=True, text=True)
    mm = json.load(open(f'{d8}/members.json'))
    ok('T8a merge A + B + C: 11 members, C INCLUDED', r.returncode == 0 and len(mm['members']) == 11
       and mm['set_c']['status'].startswith('INCLUDED'), r.stdout.strip().splitlines()[-1])
    r = subprocess.run(MERGE + ['--dest', d8, '--bar', '0.95'], capture_output=True, text=True)
    mm = json.load(open(f'{d8}/members.json'))
    ok('T8b C below the bar -> EXCLUDED, 10 members', r.returncode == 0 and len(mm['members']) == 10
       and mm['set_c']['status'].startswith('EXCLUDED: picked cv below'), mm['set_c']['status'])
    r = subprocess.run(MERGE + ['--dest', d8, '--with_c', '0'], capture_output=True, text=True)
    mm = json.load(open(f'{d8}/members.json'))
    ok('T8c --with_c 0 -> EXCLUDED, 10 members', r.returncode == 0 and len(mm['members']) == 10, mm['set_c']['status'])
    b = json.load(open(f'{d8}/members_B.json'))
    json.dump(dict(b, members=b['members'][:1]), open(f'{d8}/members_C.json', 'w'))       # the same run twice
    os.remove(f'{d8}/members.json')
    r = subprocess.run(MERGE + ['--dest', d8, '--bar', '0.5'], capture_output=True, text=True)
    ok('T8d a true duplicate (B f0 offered again as C) -> K7MERGE FAIL, nothing written', r.returncode == 1
       and 'K7MERGE FAIL: duplicate' in r.stdout and not os.path.exists(f'{d8}/members.json'), r.stdout.strip()[-120:])
    n = sum(R)
    print(f'TK7PICK {"PASS" if n == len(R) else "FAIL"} {n}/{len(R)}')
    return 0 if n == len(R) else 1


if __name__ == '__main__':
    sys.exit(main())
