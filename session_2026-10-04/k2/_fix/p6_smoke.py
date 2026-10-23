import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_lib import patch
K2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(K2)
patch('tests/smoke_k2.py', [
("""STUB = r'''# STUB fleet worker (K2 smoke only). STUB=ok|fastfail|hang|valve|hardstop
""",
"""STUB = r'''# STUB fleet worker (K2 smoke only). STUB=ok|fastfail|hang|valve|hardstop|blank|views2
"""),
("""P[keep:] = np.nan
out = os.environ['FLEET_OUT']; tmp = out[:-4] + '.tmp.npz'
np.savez(tmp, sids=np.array(ids), P=P, NV=np.full(len(ids), 4), names=np.array(['stub_a', 'stub_b', 'stub_c'][:M]),
""",
"""P[keep:] = np.nan
NV = np.full(len(ids), 4)
if mode == 'blank' and len(ids):        # one failed study per shard: 2/24 = 8 % > the 2 % coverage gate
    P[:, 0] = np.nan; NV[0] = 0
if mode == 'views2':                    # every study decoded with 2 views: below the 3 views/study gate
    NV[:] = 2
out = os.environ['FLEET_OUT']; tmp = out[:-4] + '.tmp.npz'
np.savez(tmp, sids=np.array(ids), P=P, NV=NV, names=np.array(['stub_a', 'stub_b', 'stub_c'][:M]),
"""),
("""def worker_parity():
    import numpy as np
""",
"""SLOW_RUNNER = r'''# K2e test runner (smoke only): the shipped worker, with fleet_common.build_study sleeping 30 s on ONE study
import os, sys, time, runpy
sys.path.insert(0, os.environ['FLEET_DIR'])
import fleet_common as fc
_orig, SLOW = fc.build_study, os.environ['SLOW_SID']
def build_study(df, root, sid, *a, **k):
    if sid == SLOW:
        time.sleep(30)
    return _orig(df, root, sid, *a, **k)
fc.build_study = build_study
w = os.environ['FLEET_DIR'] + '/fleet_worker.py'
sys.argv = [w]
runpy.run_path(w, run_name='__main__')
'''


def worker_timeout(ref):
    \"\"\"K2e: one study whose decode hangs is skipped after the timeout; every other row equals the normal run.\"\"\"
    import numpy as np, pandas as pd
    open(f'{S}/slow_runner.py', 'w').write(SLOW_RUNNER)
    sids = pd.read_csv(f'{S}/comp_big/test.csv', dtype=str).StudyInstanceUID.tolist()
    slow = sids[5]
    e = dict(ENV, FLEET_ALLOW_CPU='1', CUDA_VISIBLE_DEVICES='-1', FLEET_SHARD='0', FLEET_NSHARD='1',
             PYTHONPATH=f'{DRY}/shim', FLEET_DIR=f'{S}/ds/fleet_real', FLEET_OUT=f'{S}/wp_k2_timeout.npz',
             RSNA_COMP_ROOT=f'{S}/comp_big', FLEET_MICROBENCH='0', FLEET_STUDY_TIMEOUT_S='1', SLOW_SID=slow)
    t0 = time.time()
    r = run([f'{S}/slow_runner.py'], env=e, log=f'{S}/log_worker_k2_timeout.txt')
    dt = time.time() - t0
    if r.returncode != 0:
        say(f'B K2e decode timeout: CRASH rc {r.returncode} {r.stderr[-400:]}'); return False
    z = np.load(f'{S}/wp_k2_timeout.npz'); st = json.loads(str(z['stats']))
    i = list(z['sids']).index(slow)
    others = [k for k in range(len(z['sids'])) if k != i]
    d = float(np.nanmax(np.abs(z['P'][:, others] - ref['P'][:, others])))
    ok = (st.get('decode_timeout') == 1 and int(z['n_fail']) == 1 and np.isnan(z['P'][:, i]).all() and int(z['NV'][i]) == -1
          and d == 0.0 and (z['NV'][others] == ref['NV'][others]).all() and 'decode timeout' in r.stdout and dt < 60)
    say(f'B K2e decode timeout: slow study skipped (decode_timeout {st.get("decode_timeout")}, n_fail {int(z["n_fail"])}, '
        f'row NaN, NV {int(z["NV"][i])}), other {len(others)} rows max|diff| {d} vs the normal run, {dt:.0f} s -> '
        f'{"PASS" if ok else "FAIL"}')
    return ok


def worker_parity():
    import numpy as np
"""),
("""    say(f'B K2 microbench (3 studies, 3 forwards, CPU): {b} -> {"PASS" if ok3 else "FAIL"}')
    return ok1 and ok2 and ok3
""",
"""    say(f'B K2 microbench (3 studies, 3 forwards, CPU): {b} -> {"PASS" if ok3 else "FAIL"}')
    ok4 = worker_timeout(zk)
    return ok1 and ok2 and ok3 and ok4
"""),
("""       'fleet_badparent', 'comb_normal', 'comb_stub_ok', 'comb_fleetfail', 'comb_nocrop', 'comb_croplate', 'comb_stale_cf']
""",
"""       'fleet_badparent', 'fleet_blank', 'fleet_views2', 'fleet_noclock',
       'comb_normal', 'comb_stub_ok', 'comb_fleetfail', 'comb_nocrop', 'comb_croplate', 'comb_stale_cf']
"""),
])
patch('tests/run_scenario.py', [
("""cfg = dict(nb='fleet', comp='big', fds='real', stub=None, crop=False, age_h=7.0, parent='normal', stale=False,
           write_fail=None, fast_clock=False, expect_fleet='clean', expect_crop='none')
""",
"""cfg = dict(nb='fleet', comp='big', fds='real', stub=None, crop=False, age_h=7.0, parent='normal', stale=False,
           write_fail=None, fast_clock=False, noclock=False, expect_fleet='clean', expect_crop='none')
"""),
("""    'fleet_badparent':  dict(parent='badcols', expect_fleet='fail'),
""",
"""    'fleet_badparent':  dict(parent='badcols', expect_fleet='fail'),
    'fleet_blank':      dict(fds='stub', stub='blank', expect_fleet='fail'),
    'fleet_views2':     dict(fds='stub', stub='views2', expect_fleet='fail'),
    'fleet_noclock':    dict(fds='stub', stub='ok', noclock=True),
"""),
("""g = {'__name__': '__main__', 'T0': time.time() - cfg['age_h'] * 3600, 'COMP': COMP, 'ROOT': COMP}
""",
"""g = {'__name__': '__main__', 'T0': time.time() - cfg['age_h'] * 3600, 'COMP': COMP, 'ROOT': COMP}
if cfg['noclock']:                                           # no parent T0 (Windows has no /proc either): no clock
    del g['T0']
"""),
("""               ('fleet_hang', 'after the wall'), ('fleet_nods', 'not attached'), ('fleet_writefail', 'No space left')):
""",
"""               ('fleet_hang', 'after the wall'), ('fleet_nods', 'not attached'), ('fleet_writefail', 'No space left'),
               ('fleet_blank', 'failed studies'), ('fleet_views2', 'views/study')):
"""),
("""if SC == 'comb_croplate':
""",
"""if SC == 'fleet_noclock':
    rec = g['_k2_receipt']
    extra.append(('noclock: no kernel age, stage given NOCLOCK_MIN (40) minutes',
                  rec.get('kernel_age_h') is None and 39.0 < float(rec.get('left_min') or 0) <= 40.01))
if SC in ('fleet_normal', 'fleet_stub_ok', 'fleet_commit3'):  # commit3: age 0.2 h, the old 150 min cap would show
    rec = g['_k2_receipt']                                   # P6: the wall is the only guard when a clock exists
    extra.append(('wall only: left_min == (8.6 h - kernel age) to 0.1 min',
                  abs(float(rec.get('left_min') or 0) - (8.6 - float(rec.get('kernel_age_h') or 0)) * 60) < 0.1))
if cfg['nb'] != 'fleet' and isinstance(g.get('_k2c_receipt'), dict) and cfg['crop']:
    sb = f'{WORK}/_k2_crop'
    extra.append(('C6: sandbox submission*.csv renamed k1out*.csv',
                  os.path.isdir(sb) and not [f for f in os.listdir(sb) if f.startswith('submission')]
                  and isinstance(g['_k2c_receipt'].get('sandbox_renamed'), list)))
if SC == 'comb_croplate':
"""),
])
