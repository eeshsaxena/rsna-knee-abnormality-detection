#!/usr/bin/env python3
"""t_k10_pick.py (box, CPU; TEST ONLY; 2026-10-09, Claude): picker K10-VREC + k10_merge rules on the simulation roots.
Needs ~/k10_test/sim/root (valid receipts) and ~/k10_test/sim/negroot (--neg_receipts) from _dev/k10_sim_setup.py and the
sim collect's ~/k10_test/simtest/members_{C,D,E}.json. Writes only ~/k10_test/pickt. Last line TK10PICK PASS|FAIL n/m.
"""
import copy, json, os, shutil, subprocess, sys

H = os.path.expanduser('~')
T = f'{H}/k10_test/pickt'
PY = sys.executable
TOOLS = os.path.dirname(os.path.abspath(__file__))
PICK = [PY, '-W', 'ignore', f'{TOOLS}/k5_pick.py', '--tag', 'raw']
MERGE = [PY, f'{TOOLS}/k10_merge.py']
E_ARGS = ['--seed', '1045', '--suffix', 'ttr288frac75', '--ttr_size', '288', '--train_labels', 'labels_frac75_logit.csv',
          '--key_tag', 't288f75', '--no_kaggle_receipt']
res = []


def ok(name, cond, detail=''):
    res.append((name, bool(cond)))
    print(f'{"PASS" if cond else "FAIL"}  {name}  {str(detail)[:400]}', flush=True)


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', **(env or {})))
    return r.returncode, r.stdout + r.stderr


shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
# T1: valid sim root, E with Vast receipts f1-f4: 5 picks, sources box / vast x4, receipts recorded
rc, out = run(PICK + E_ARGS + ['--root', f'{H}/k10_test/sim/root', '--vast_receipt_folds', '1,2,3,4', '--dest', f'{T}/t1', '--copy'])
m = json.load(open(f'{T}/t1/members.json')) if os.path.isfile(f'{T}/t1/members.json') else {}
src = {r['fold']: (r['source'], bool(r.get('vast_receipt'))) for r in m.get('members', [])}
ok('T1 E valid receipts: rc 0, 5 picks, f0 box (mtime rule), f1-f4 vast with receipt', rc == 0 and src == {
    0: ('box', False), 1: ('vast', True), 2: ('vast', True), 3: ('vast', True), 4: ('vast', True)}, src)
ok('T1 E OOF copies named oof_f{F}_s1045t288f75_raw_*', all(os.path.basename(r['oof']).startswith(f'oof_f{r["fold"]}_s1045t288f75_raw_')
                                                           for r in m.get('members', [])), [r.get('oof') for r in m.get('members', [])][:2])
ok('T1 the mtime line is reported for the Vast folds', out.count('mtime rule reported, not enforced') == 4, '')
# T2: negative receipts: f1 missing, f2 verify, f3 closing_line, f4 run name -> each FAILS, f0 still picked, rc 1
rc, out = run(PICK + E_ARGS + ['--root', f'{H}/k10_test/sim/negroot', '--vast_receipt_folds', '1,2,3,4', '--dest', f'{T}/t2', '--copy'])
m = json.load(open(f'{T}/t2/members.json')) if os.path.isfile(f'{T}/t2/members.json') else {}
fails = {f['fold']: f['reason'] for f in m.get('failed', [])}
ok('T2 rc 1, f0 picked, f1-f4 FAIL', rc == 1 and [r['fold'] for r in m.get('members', [])] == [0] and sorted(fails) == [1, 2, 3, 4], fails)
ok('T2 f1 why: no Vast receipt', 'no Vast receipt' in fails.get(1, ''), fails.get(1))
ok('T2 f2 why: verify', "verify 'md5" in fails.get(2, ''), fails.get(2))
ok('T2 f3 why: closing_line', 'closing_line' in fails.get(3, ''), fails.get(3))
ok('T2 f4 why: run / source_run', 'source_run' in fails.get(4, ''), fails.get(4))
# T3: without --vast_receipt_folds the Vast-copied sim folds fall back to the box mtime rule (sim mtimes are box-like: pass)
rc, out = run(PICK + E_ARGS + ['--root', f'{H}/k10_test/sim/root', '--dest', f'{T}/t3', '--copy'])
m = json.load(open(f'{T}/t3/members.json')) if os.path.isfile(f'{T}/t3/members.json') else {}
ok('T3 no --vast_receipt_folds: 5 picks, all source box', rc == 0 and all(r['source'] == 'box' for r in m.get('members', [])) and len(m.get('members', [])) == 5)
# T4: --min_epochs 22 > E 21 fails every E fold
rc, out = run(PICK + E_ARGS + ['--root', f'{H}/k10_test/sim/root', '--vast_receipt_folds', '1,2,3,4', '--dest', f'{T}/t4', '--min_epochs', '22'])
ok('T4 --min_epochs 22: rc 1 (E 21 < 22)', rc == 1 and 'E 21 < --min_epochs 22' in out)
# merge tests on copies of the sim collect's set files
S = f'{H}/k10_test/simtest'


def mdir(name, mut=None):
    d = f'{T}/{name}'; os.makedirs(d, exist_ok=True)
    for s in 'CDE':
        x = json.load(open(f'{S}/members_{s}.json'))
        if mut:
            x = mut(s, copy.deepcopy(x))
        if x is not None:
            json.dump(x, open(f'{d}/members_{s}.json', 'w'))
    return d


d = mdir('m_ok')
rc, out = run(MERGE + ['--dest', d])
mm = json.load(open(f'{d}/members.json')) if os.path.isfile(f'{d}/members.json') else {}
ok('M1 merge FINAL 15 members, k10 true, not do_not_submit', rc == 0 and len(mm.get('members', [])) == 15 and mm.get('k10') and not mm.get('do_not_submit'), out.splitlines()[-1:])
ok('M1 E bars = C_REF - .004 (f0 bar .886920)', 'set E fold 0: cv 0.890920 (+0.0000 vs C)' in out and 'bar 0.886920' in out)
d = mdir('m_bar')
rc, out = run(MERGE + ['--dest', d, '--e_bars', '0:0.8915'])
ok('M2 E f0 below an overridden bar: rc 1 + the pre-registered override check printed', rc == 1 and 'PRE-REGISTERED OVERRIDE CHECK E f0' in out
   and 'K10MERGE FAIL: set E fold(s) below' in out, [l for l in out.splitlines() if 'OVERRIDE' in l][:1])
d = mdir('m_dsha', lambda s, x: (x.__setitem__('members', [dict(r, sha256='0' * 64) if r['fold'] == 2 else r for r in x['members']]) or x) if s == 'D' else x)
rc, out = run(MERGE + ['--dest', d])
ok('M3 a D pick with other bytes than K9 FAILS', rc == 1 and 'differ from the K9 members' in out)
d = mdir('m_vast', None)
rc, out = run(MERGE + ['--dest', d, '--e_vast_folds', '1,2,3'])
ok('M4 E f4 carries a Vast receipt but is not listed: FAIL', rc == 1 and 'carry one but are not listed' in out)
d = mdir('m_novr', lambda s, x: (x.__setitem__('members', [dict(r, vast_receipt=None) if r['fold'] == 3 else r for r in x['members']]) or x) if s == 'E' else x)
rc, out = run(MERGE + ['--dest', d])
ok('M5 E f3 listed as Vast but without receipt: FAIL', rc == 1 and 'folds [3] need one' in out)
d = mdir('m_lab', lambda s, x: (x.__setitem__('members', [dict(r, train_labels='labels_mix50_logit.csv') if r['fold'] == 1 else r for r in x['members']]) or x) if s == 'E' else x)
rc, out = run(MERGE + ['--dest', d])
ok('M6 E member with mix50 labels refused', rc == 1 and 'set E picks not' in out)
d = mdir('m_int', lambda s, x: (x.__setitem__('members', [r for r in x['members'] if r['fold'] != 4]) or x) if s == 'E' else x)
rc, out = run(MERGE + ['--dest', d])
ok('M7 E 4/5: NOT FINAL rc 3', rc == 3 and 'K10MERGE NOT FINAL' in out)
rc, out = run(MERGE + ['--dest', d, '--allow_interim'])
mm = json.load(open(f'{d}/members.json')) if os.path.isfile(f'{d}/members.json') else {}
ok('M8 --allow_interim: INTERIM, do_not_submit', rc == 0 and mm.get('do_not_submit') and mm.get('interim'))
d = mdir('m_fail', lambda s, x: (x.__setitem__('failed', [dict(fold=1, reason='x')]) or x) if s == 'E' else x)
rc, out = run(MERGE + ['--dest', d])
ok('M9 a failed E fold in the picker output FAILS the merge', rc == 1 and 'picker check FAILED' in out)
npass = sum(g for _, g in res)
print(f'TK10PICK {"PASS" if npass == len(res) else "FAIL"} {npass}/{len(res)}')
