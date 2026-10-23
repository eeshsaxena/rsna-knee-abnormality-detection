#!/usr/bin/env python3
"""t_k8_pick.py (box, CPU; TEST ONLY; 2026-10-07, Claude): the K8 picker (k5_pick.py K8 copy, --tag) + k8_merge.py on the
real set A runs (read only), the real ~/fleet_runs TTR288 runs (read only) and the simulation root ~/k8_test/sim/root
(k8_sim_setup.py). Writes only under ~/k8_test/sim/tpick.

  CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/kmimic/bin/python t_k8_pick.py [--box ~/k8_test/simtest/k5/box]

T1 regression: set A with --tag auto (old rule) AND with --tag raw picks exactly what K7 picked for set A
   (~/k7_test/members.json set A records: file, sha256, tag, epoch, cv, source, receipt) and the same OOF copy names.
T2 --tag ema on set A: every fold picks its EMA file (all checks pass on the forced tag), record tag_rule ema.
T3 set C (sim root, 5 folds, --tag raw --ttr_size 288 --no_kaggle_receipt): 5 raw picks, ttr 288, OOF copies
   oof_f{F}_s1044t288_raw_cv<N>.csv, set A picks in the same dest untouched (recipe-aware stale removal).
T4 merge FINAL (sim): 10 members (A 5 + C 5), set_c FINAL, interim False, do_not_submit False.
T5 merge on the REAL ~/fleet_runs (C folds not finished yet): rc 3 without --allow_interim (nothing written);
   with --allow_interim: interim True, do_not_submit True, every unfinished C fold listed as skipped.
T6 per-fold sanity bar: --c_bars 2:0.95 -> rc 1 (below bar, nothing written); --c_margin -0.01 (bar = A + .01) -> rc 1;
   --c_bars 2:0.80 (operator override) -> rc 0 and the override is recorded.
T7 merge refuses a set A with ema picks (T2 output) -> rc 1.
T8 merge refuses a duplicate member (C fold 0 offered twice) -> rc 1.
T9 picker set C with the default --ttr_size 224 on the sim root -> rc 1 (ttr_size check).
T10 merge refuses a set-B member (s1044 TTR224) smuggled into members_C.json -> rc 1.
"""
import argparse, glob, json, os, shutil, subprocess, sys

H = os.path.expanduser('~')
R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {extra}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--box', default=f'{H}/k8_test/simtest/k5/box')
    a = ap.parse_args()
    PY = sys.executable
    PICK = [PY, '-W', 'ignore', f'{a.box}/k5_pick.py', '--copy']
    MERGE = [PY, '-W', 'ignore', f'{a.box}/k8_merge.py']
    SIM = f'{H}/k8_test/sim/root'
    T = f'{H}/k8_test/sim/tpick'
    shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
    CARGS = ['--seed', '1044', '--suffix', 'ttr288mix50', '--folds', '0,1,2,3,4', '--ttr_size', '288', '--no_kaggle_receipt']

    def pick(dest, *args):
        os.makedirs(f'{T}/{dest}', exist_ok=True)
        if os.path.isfile(f'{T}/{dest}/members.json'):
            os.remove(f'{T}/{dest}/members.json')
        r = subprocess.run(PICK + ['--dest', f'{T}/{dest}', *args], capture_output=True, text=True)
        m = json.load(open(f'{T}/{dest}/members.json')) if os.path.isfile(f'{T}/{dest}/members.json') else None
        open(f'{T}/{dest}.log', 'a').write(r.stdout + r.stderr)
        return r.returncode, m, r.stdout

    def setfile(dest, name):
        if os.path.isfile(f'{T}/{dest}/members.json'):
            os.replace(f'{T}/{dest}/members.json', f'{T}/{dest}/members_{name}.json')

    def merge(dest, *args):
        if os.path.isfile(f'{T}/{dest}/members.json'):
            os.remove(f'{T}/{dest}/members.json')
        r = subprocess.run(MERGE + ['--dest', f'{T}/{dest}', *args], capture_output=True, text=True)
        open(f'{T}/{dest}.merge.log', 'a').write(r.stdout + r.stderr)
        m = json.load(open(f'{T}/{dest}/members.json')) if os.path.isfile(f'{T}/{dest}/members.json') else None
        return r.returncode, m, r.stdout

    keys = ('fold', 'tag', 'epoch', 'cv', 'file', 'sha256', 'source', 'kaggle_receipt')
    k7 = json.load(open(f'{H}/k7_test/members.json'))
    k7a = [{k: x[k] for k in keys} for x in k7['members'] if x['set'] == 'A']
    # T1
    rc1, m1, _ = pick('d1', '--root', f'{H}/fleet_runs', '--ttr_size', '224')
    rc2, m2, _ = pick('d1raw', '--root', f'{H}/fleet_runs', '--ttr_size', '224', '--tag', 'raw')
    same = (rc1 == 0 and rc2 == 0 and [{k: x[k] for k in keys} for x in m1['members']] == k7a
            and [{k: x[k] for k in keys} for x in m2['members']] == k7a)
    oofn = sorted(os.path.basename(x['oof']) for x in m2['members']) == sorted(
        os.path.basename(x['oof']) for x in k7['members'] if x['set'] == 'A')
    ok('T1 set A --tag auto and --tag raw == the K7 set A picks', same and oofn,
       f'rc {rc1}/{rc2} files {[x["file"] for x in (m2 or {}).get("members", [])]}')
    # T2
    rc, m, _ = pick('d2', '--root', f'{H}/fleet_runs', '--ttr_size', '224', '--tag', 'ema')
    ok('T2 --tag ema forces the EMA file on every fold (checks pass)', rc == 0 and len(m['members']) == 5 and all(
        x['tag'] == 'ema' and '_ema_' in x['file'] and x['tag_rule'] == 'ema' and abs(x['oof_rescored_cv'] - x['cv']) <= 1e-12
        for x in m['members']), f'rc {rc} {[x["file"] for x in m["members"]] if m else None}')
    setfile('d2', 'A')
    # T3 + T4 (sim, one dest)
    rc, mA, _ = pick('d3', '--root', SIM, '--ttr_size', '224', '--tag', 'raw'); setfile('d3', 'A')
    rcC, mC, _ = pick('d3', '--root', SIM, '--tag', 'raw', *CARGS); setfile('d3', 'C')
    good = (rc == 0 and rcC == 0 and len(mC['members']) == 5 and all(
        x['tag'] == 'raw' and int(x['ttr_size']) == 288 and os.path.basename(x['oof']).startswith(f'oof_f{x["fold"]}_s1044t288_raw_cv')
        for x in mC['members']))
    a_left = all(os.path.isfile(x['copy']) for x in mA['members'])
    ok('T3 set C sim: 5 raw TTR288 picks, t288 OOF names, set A picks in the same dest untouched', good and a_left,
       f'rc {rc}/{rcC} {[os.path.basename(x["oof"]) for x in (mC or {}).get("members", [])]}')
    rc, m, out = merge('d3')
    ok('T4 merge FINAL (sim): 10 members A 5 + C 5, set_c FINAL, not interim', rc == 0 and len(m['members']) == 10 and
       m['set_c']['status'].startswith('FINAL') and m['interim'] is False and m['do_not_submit'] is False,
       f'rc {rc} {out.strip().splitlines()[-1] if out.strip() else ""}')
    # T6 (bars) on the same dest
    rc, m, out = merge('d3', '--c_bars', '2:0.95')
    ok('T6a fold 2 below an explicit bar .95 -> rc 1, nothing written', rc == 1 and m is None and 'below their sanity bar' in out, f'rc {rc}')
    rc, m, out = merge('d3', '--c_margin', '-0.01')
    ok('T6b bar = A + .01 (every sim fold == A) -> rc 1', rc == 1 and m is None, f'rc {rc}')
    rc, m, out = merge('d3', '--c_bars', '2:0.80')
    ok('T6c operator override 2:0.80 -> rc 0, override recorded', rc == 0 and m['set_c']['overrides'] == {'2': 0.8}
       or (rc == 0 and m['set_c']['overrides'] in ({2: 0.8}, {'2': 0.8})), f'rc {rc} {m["set_c"]["overrides"] if m else None}')
    # T5 (real runs: only the finished TTR288 folds)
    rc, _, _ = pick('d5', '--root', f'{H}/fleet_runs', '--ttr_size', '224', '--tag', 'raw'); setfile('d5', 'A')
    rcC, mC, _ = pick('d5', '--root', f'{H}/fleet_runs', '--tag', 'raw', *CARGS); setfile('d5', 'C')
    nfin = len((mC or {}).get('members', []))
    rc3, m3, out3 = merge('d5')
    rci, mi, outi = merge('d5', '--allow_interim')
    good = (rc == 0 and rcC == 0 and 1 <= nfin < 5 and rc3 == 3 and m3 is None and rci == 0 and mi['interim'] is True
            and mi['do_not_submit'] is True and len(mi['members']) == 5 + nfin
            and sorted(s['fold'] for s in mi['skipped'] if s['set'] == 'C') == [F for F in range(5) if F not in
                                                                                 [x['fold'] for x in mC['members']]])
    ok(f'T5 real runs (C {nfin}/5 finished): rc 3 without --allow_interim; with it interim + do_not_submit, unfinished listed',
       good, f'rc {rcC} merge {rc3}/{rci} {outi.strip().splitlines()[-1] if outi.strip() else ""}')
    # T7
    shutil.copyfile(f'{T}/d2/members_A.json', f'{T}/d7_A.json')
    os.makedirs(f'{T}/d7', exist_ok=True)
    shutil.copyfile(f'{T}/d2/members_A.json', f'{T}/d7/members_A.json'); shutil.copyfile(f'{T}/d3/members_C.json', f'{T}/d7/members_C.json')
    rc, m, out = merge('d7')
    ok('T7 merge refuses a set A of EMA picks', rc == 1 and m is None and 'raw' in out, f'rc {rc}')
    # T8
    os.makedirs(f'{T}/d8', exist_ok=True)
    shutil.copyfile(f'{T}/d3/members_A.json', f'{T}/d8/members_A.json')
    c = json.load(open(f'{T}/d3/members_C.json')); c['members'] = c['members'] + [c['members'][0]]
    json.dump(c, open(f'{T}/d8/members_C.json', 'w'))
    rc, m, out = merge('d8')
    ok('T8 merge refuses a duplicate C member', rc == 1 and m is None, f'rc {rc} {out.strip().splitlines()[-1] if out.strip() else ""}')
    # T9
    rc, m, out = pick('d9', '--root', SIM, '--seed', '1044', '--suffix', 'ttr288mix50', '--no_kaggle_receipt', '--tag', 'raw')
    ok('T9 set C picked with --ttr_size 224 -> FAIL on the ttr_size check', rc == 1 and 'ttr_size' in out, f'rc {rc}')
    # T10
    os.makedirs(f'{T}/d10', exist_ok=True)
    shutil.copyfile(f'{T}/d3/members_A.json', f'{T}/d10/members_A.json')
    c = json.load(open(f'{T}/d3/members_C.json'))
    b = dict(c['members'][1], ttr_size=224, file=c['members'][1]['file'].replace('_ttr288-', '_ttr224-'))
    c['members'] = c['members'][:1] + [b] + c['members'][2:]
    json.dump(c, open(f'{T}/d10/members_C.json', 'w'))
    rc, m, out = merge('d10')
    ok('T10 merge refuses a set-B (s1044 TTR224) member in members_C.json', rc == 1 and m is None, f'rc {rc}')
    print(f'TK8PICK {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
    return 0 if all(R) else 1


if __name__ == '__main__':
    sys.exit(main())
