# gate_neg_fix1015.py (v224f/tests, 10-15, Claude fix agent): F1 gate negatives on the regenerated mock FF-S output
# (run `python tests/mock_fs_output.py ff` first). Laptop, offline. Prints GATE_NEG_FIX1015 PASS n/m.
#   1 clean mock, fixed gate (default flip budget 4.9)            -> PASS
#   2 log FLIP_TTA_FORCED budget 4.4 (a kernel with the old cut)  -> FAIL
#   3 V224F_SAVED knobs start 5.0 / 198 vs the 5.1 / 192 build    -> FAIL
#   4 fixed gate with --flip-budget-h 4.4 vs the 4.9 build json   -> refused (assert)
#   5 pre-fix gate (v224ffs_check.py.pre_fix_1015, default 4.4)   -> refused (assert): the old gate could not pass a 4.9 build
import os, sys, json, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(HERE); ROOT = os.path.normpath(os.path.join(V, '..', '..'))
OUT = os.path.join(HERE, 'mock_out_v224ffs20'); BJ = os.path.join(HERE, 'standin_v224ffs20', 'BUILD_V224FS.json')
REFS = ['--ref-flip', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1'),
        '--ref-exact', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1'),
        '--ref-fleet', os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'out_v224ff20_v1')]
assert json.load(open(BJ))['flip_budget_h'] == 4.9


def gate(out, script='v224ffs_check.py', extra=()):
    r = subprocess.run([sys.executable, os.path.join(V, script), out, '--wf', '0.2', '--build', BJ] + REFS + list(extra),
                       capture_output=True, text=True, encoding='utf-8')
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else (r.stderr.strip().splitlines() or [''])[-1]
    return r.returncode, last


def log_sub(M, a, b):
    p = os.path.join(M, 'rsna-sub-v224ffs20.log'); E = json.load(open(p, encoding='utf-8'))
    hit = [i for i, e in enumerate(E) if a in e['data']]; assert hit, a
    E[hit[0]]['data'] = E[hit[0]]['data'].replace(a, b, 1); json.dump(E, open(p, 'w', encoding='utf-8'))


def mutated(fn):
    M = OUT + '_negfix'
    if os.path.isdir(M):
        shutil.rmtree(M)
    shutil.copytree(OUT, M); fn(M)
    return M


res = []


def rep(name, ok, last):
    res.append(ok)
    print(f'  {"OK  " if ok else "BAD "} {name} | {last[:160]}', flush=True)


rc, last = gate(OUT)
rep('1 clean mock, fixed gate -> PASS', rc == 0 and 'V224FFS_CHECK PASS' in last, last)
M = mutated(lambda M: log_sub(M, '"flip_budget_h": 4.9', '"flip_budget_h": 4.4'))
rc, last = gate(M)
rep('2 FLIP_TTA_FORCED budget 4.4 in the log -> FAIL', rc != 0 and 'V224FFS_CHECK FAIL' in last, last)
M = mutated(lambda M: log_sub(M, '"start_budget_h": 5.1, "wall_h": 8.3, "min_start_min": 192.0',
                              '"start_budget_h": 5.0, "wall_h": 8.3, "min_start_min": 198.0'))
rc, last = gate(M)
rep('3 V224F_SAVED knobs 5.0 / 198 vs the 5.1 / 192 build -> FAIL', rc != 0 and 'V224FFS_CHECK FAIL' in last, last)
shutil.rmtree(M, ignore_errors=True)
rc, last = gate(OUT, extra=['--flip-budget-h', '4.4'])
rep('4 --flip-budget-h 4.4 vs the 4.9 build json -> refused', rc != 0 and 'AssertionError' in last, last)
rc, last = gate(OUT, script='v224ffs_check.py.pre_fix_1015')
rep('5 pre-fix gate (default 4.4) on the 4.9 build -> refused', rc != 0 and 'AssertionError' in last, last)
print(f'GATE_NEG_FIX1015 {"PASS" if all(res) else "FAIL"} {sum(res)}/{len(res)}', flush=True)
sys.exit(0 if all(res) else 1)
