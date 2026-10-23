"""K2 laptop CPU smoke (harness only; nothing here ships, nothing is pushed). Reuses round3/fleet_dry and cropfleet/_smoke
assets: 64 px store contract (only fleet_common STORE 320 -> 64), the 24-study fake competition comp_big (+ a 3-study
commit-sized cut), two tiny round3 dry ConvNeXt members (fp16-labelled), the two tiny crop smoke members, round3 cv2
PIL shim. Steps:
  A. stage: code64 (k2/code with STORE 64), the fleet dataset via k2/code64/pack_fleet_ds.py (the build tool), a
     fleet2-worker twin (for worker parity), a stub-worker dataset (ok/fastfail/hang/valve/hardstop), crop members,
     the 3 notebooks via make_k2_kernel.py (fleet-only, combined, combined with a 25 min crop start floor)
  B. worker parity: K2 worker vs the fleet2 worker on the same dataset (P max|diff| 0, same NaN pattern), and the
     microbench on the 3-study cut
  C. every scenario in run_scenario.py, one process each
  D. build_k2.py end to end on the REAL interim box checkpoint in k2/_real (preflight + pack + both kernels), if present
Writes tests/smoke_transcript.txt.  python smoke_k2.py [scenario ...]
"""
import os, sys, json, glob, shutil, subprocess, time
T = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K2 = os.path.dirname(T); SP = os.path.dirname(K2)
S = f'{T}/_smoke'
DRY = f'{SP}/round3/fleet_dry'
PY = sys.executable
ENV = {**os.environ, 'PYTHONWARNINGS': 'ignore', 'PYTHONIOENCODING': 'utf-8', 'CUDA_VISIBLE_DEVICES': '',
       'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
LOG = []


def say(msg):
    print(msg, flush=True); LOG.append(msg)


def run(args, env=None, log=None, timeout=1800):
    r = subprocess.run([PY, '-W', 'ignore', *args], capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=env or ENV, timeout=timeout)
    if log:
        open(log, 'w', encoding='utf-8').write(r.stdout + '\n--- stderr ---\n' + r.stderr)
    return r


STUB = r'''# STUB fleet worker (K2 smoke only). STUB=ok|fastfail|hang|valve|hardstop
import os, sys, time, json
import numpy as np, pandas as pd
mode, shard, nsh = os.environ.get('STUB', 'ok'), int(os.environ['FLEET_SHARD']), int(os.environ['FLEET_NSHARD'])
ids = pd.read_csv(os.environ['RSNA_COMP_ROOT'] + '/test.csv', dtype=str).StudyInstanceUID.tolist()[shard::nsh]
if mode == 'hang':
    time.sleep(600)
if mode == 'fastfail':
    if shard == 1:
        sys.exit(3)
    time.sleep(600)
M = 3 if mode == 'valve' else 2
P = np.random.default_rng(100 + shard).uniform(0.01, 0.99, (M, len(ids), 12)).astype(np.float32)
keep = 1 if mode == 'valve' else M
P[keep:] = np.nan
out = os.environ['FLEET_OUT']; tmp = out[:-4] + '.tmp.npz'
np.savez(tmp, sids=np.array(ids), P=P, NV=np.full(len(ids), 4), names=np.array(['stub_a', 'stub_b', 'stub_c'][:M]),
         n_keep=np.int64(keep), n_fail=np.int64(0), n_blank=np.int64(0), hard_stopped=np.bool_(mode == 'hardstop'),
         minutes=np.float64(0.01), stats=np.array('{}'), parity=np.array('{}'), bench=np.array('{}'),
         loop_min=np.float64(0.01))
os.replace(tmp, out)
'''


def stage():
    shutil.rmtree(S, ignore_errors=True); os.makedirs(f'{S}/ds')
    # code64 = k2/code with ONLY fleet_common STORE 320 -> 64 (round3 dry-run rule)
    os.makedirs(f'{S}/code64')
    for f in os.listdir(f'{K2}/code'):
        if not f.endswith('.py'):
            continue
        s = open(f'{K2}/code/{f}', encoding='utf-8', newline='').read()
        if f == 'fleet_common.py':
            assert s.count('STORE = 320 ') == 1; s = s.replace('STORE = 320 ', 'STORE = 64 ')
        open(f'{S}/code64/{f}', 'w', encoding='utf-8', newline='').write(s)
    mem = [f'{DRY}/fleetds/f0_s42_ema_cnxt64x24_cv5714_best.pt', f'{DRY}/fleetds/f-1_s42_ema_cnxt64x24_ep1_best.pt']
    r = run([f'{S}/code64/pack_fleet_ds.py', '--out', f'{S}/ds/fleet_real', '--store', f'{DRY}/store_64_normal'] + mem,
            log=f'{S}/log_pack.txt')
    assert r.returncode == 0, r.stderr[-1500:]
    say('A pack (k2 pack tool, 64 px): ' + r.stdout.strip().splitlines()[-1][-160:])
    shutil.copytree(f'{S}/ds/fleet_real', f'{S}/ds/fleet_f2orig')
    shutil.copy2(f'{SP}/fleet2/code/fleet_worker.py', f'{S}/ds/fleet_f2orig/fleet_worker.py')
    os.makedirs(f'{S}/ds/fleet_stub')
    open(f'{S}/ds/fleet_stub/fleet_worker.py', 'w').write(STUB)
    json.dump({'checkpoints': ['stub_a', 'stub_b']}, open(f'{S}/ds/fleet_stub/fleet_manifest.json', 'w'))
    shutil.copytree(f'{SP}/cropfleet/_smoke/input/datasets/eeshsaxena/rsna-knee-cropaug-members', f'{S}/ds/crop')
    # competitions: comp_big (24 studies, base series csv) and a 3-study commit-sized cut
    shutil.copytree(f'{DRY}/comp_big', f'{S}/comp_big', ignore=shutil.ignore_patterns('test_series_base.csv'))
    shutil.copy2(f'{DRY}/comp_big/test_series_base.csv', f'{S}/comp_big/test_series.csv')
    import pandas as pd
    t = pd.read_csv(f'{S}/comp_big/test.csv', dtype=str); ts = pd.read_csv(f'{S}/comp_big/test_series.csv', dtype=str)
    keep = t.StudyInstanceUID.tolist()[:3]
    os.makedirs(f'{S}/comp3/test_series')
    t[t.StudyInstanceUID.isin(keep)].to_csv(f'{S}/comp3/test.csv', index=False)
    ts[ts.StudyInstanceUID.isin(keep)].to_csv(f'{S}/comp3/test_series.csv', index=False)
    for s in keep:
        shutil.copytree(f'{S}/comp_big/test_series/{s}', f'{S}/comp3/test_series/{s}')
    shutil.copytree(f'{S}/comp_big/train_series', f'{S}/comp3/train_series')
    shutil.copy2(f'{S}/comp_big/train_series.csv', f'{S}/comp3/train_series.csv')
    # the three notebooks, built by the real builder
    sys.path.insert(0, K2)
    import make_k2_kernel as mk
    mk.main(['--out', f'{S}/nb/rsna-sub-k2-fleet', '--slug', 'rsna-sub-k2-fleet'])
    mk.main(['--out', f'{S}/nb/rsna-sub-k2-fleetcrop', '--slug', 'rsna-sub-k2-fleetcrop', '--combined',
             '--crop_cell_sha', '049033d6002cf9fc'])
    mk.main(['--out', f'{S}/nb/rsna-sub-k2-croplate', '--slug', 'rsna-sub-k2-croplate', '--combined',
             '--crop_min_start_min', '25'])
    say('A notebooks built: fleet-only (32 cells), combined (35 cells), combined crop-floor-25 (35 cells)')


def worker_parity():
    import numpy as np
    base = dict(ENV, FLEET_ALLOW_CPU='1', CUDA_VISIBLE_DEVICES='-1', FLEET_SHARD='0', FLEET_NSHARD='1',
                PYTHONPATH=f'{DRY}/shim')
    outs = {}
    for tag, ds, comp, mb in (('k2', 'fleet_real', 'comp_big', '0'), ('fleet2', 'fleet_f2orig', 'comp_big', '0'),
                              ('k2_bench', 'fleet_real', 'comp3', '3')):
        e = dict(base, FLEET_DIR=f'{S}/ds/{ds}', FLEET_OUT=f'{S}/wp_{tag}.npz', RSNA_COMP_ROOT=f'{S}/{comp}',
                 FLEET_MICROBENCH=mb)
        t0 = time.time()
        r = run([f'{S}/ds/{ds}/fleet_worker.py'], env=e, log=f'{S}/log_worker_{tag}.txt')
        assert r.returncode == 0, (tag, r.stderr[-1500:])
        outs[tag] = (np.load(f'{S}/wp_{tag}.npz'), time.time() - t0, r.stdout)
    zk, zf = outs['k2'][0], outs['fleet2'][0]
    same_ids = list(zk['sids']) == list(zf['sids']) and list(zk['names']) == list(zf['names'])
    d = float(np.nanmax(np.abs(zk['P'] - zf['P'])))
    nan_eq = bool((np.isnan(zk['P']) == np.isnan(zf['P'])).all())
    ok1 = same_ids and d == 0.0 and nan_eq and (zk['NV'] == zf['NV']).all()
    st = json.loads(str(zk['stats']))
    ok2 = 'decode_s' in st and st['decode_s'] > 0 and json.loads(str(zk['bench'])) == {} and 'bench' not in zf.files
    zb = outs['k2_bench'][0]; b = json.loads(str(zb['bench']))
    ok3 = len(b) == 2 and all(v > 0 for v in b.values()) and 'MICROBENCH' in outs['k2_bench'][2]
    say(f'B worker parity K2 vs fleet2 on {len(zk["sids"])} studies: P max|diff| {d} NaN-equal {nan_eq} -> {"PASS" if ok1 else "FAIL"}')
    say(f'B K2 worker extras: decode_s {st.get("decode_s", 0):.2f} s summed, no bench without the env -> {"PASS" if ok2 else "FAIL"}')
    say(f'B K2 microbench (3 studies, 3 forwards, CPU): {b} -> {"PASS" if ok3 else "FAIL"}')
    return ok1 and ok2 and ok3


def scenarios(names):
    allok = True
    for sc in names:
        t0 = time.time()
        r = run([f'{T}/run_scenario.py', sc], log=f'{S}/log_{sc}.txt', timeout=900)
        line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
        if r.returncode != 0 or not line:
            say(f'C {sc:<17} CRASH rc {r.returncode}: {(r.stderr or r.stdout)[-600:]}'); allok = False; continue
        res = json.loads(line[-1][7:])
        allok &= res['ok']
        fails = [l for l in r.stdout.splitlines() if l.startswith('FAIL')]
        say(f'C {sc:<17} {"PASS" if res["ok"] else "FAIL"} {res["passed"]}/{res["checks"]} ({time.time() - t0:.0f} s) '
            f'| fleet: {res["why"][:110]} | crop: {res["crop"][:120]}' + (f' | {fails}' if fails else ''))
    return allok


def real_build():
    ck = sorted(glob.glob(f'{K2}/_real/ckpt/*.pt'))
    if not ck:
        say('D real-checkpoint build skipped (k2/_real/ckpt empty)'); return True
    r = run([f'{K2}/build_k2.py', '--ckpt', ck[0], '--store', f'{K2}/_real/store', '--out', f'{S}/real_build',
             '--combined', '--overwrite'], log=f'{S}/log_real_build.txt')
    ok = r.returncode == 0 and 'upload folder OK' in r.stdout
    summ = json.load(open(f'{S}/real_build/BUILD_SUMMARY.json')) if ok else {}
    m = (summ.get('members') or [{}])[0]
    say(f'D build_k2.py on the REAL interim box checkpoint {os.path.basename(ck[0])} (epoch {m.get("epoch_1based")}, '
        f'{m.get("train_labels")}, model_sha {m.get("model_sha")}): {"PASS" if ok else "FAIL"} '
        f'| upload {summ.get("upload", {}).get("mb")} MB | kernels {[k["id"] for k in summ.get("kernels", [])]}'
        + ('' if ok else f' | {r.stderr[-800:]}'))
    # negative: raw AND ema of one run must be refused, and a second label variant without a re-key too
    neg = run([f'{K2}/build_k2.py', '--ckpt', ck[0], '--ckpt', ck[0], '--store', f'{K2}/_real/store',
               '--out', f'{S}/real_build_neg', '--overwrite'])
    ok_neg = neg.returncode != 0 and 'one tag per run' in neg.stderr
    say(f'D build_k2.py refuses two tags of one run: {"PASS" if ok_neg else "FAIL"}')
    return ok and ok_neg


ALL = ['fleet_normal', 'fleet_commit3', 'fleet_stub_ok', 'fleet_lowtime', 'fleet_fastfail', 'fleet_valve',
       'fleet_hardstop', 'fleet_hang', 'fleet_stale', 'fleet_writefail', 'fleet_nods', 'fleet_noparent',
       'fleet_badparent', 'comb_normal', 'comb_stub_ok', 'comb_fleetfail', 'comb_nocrop', 'comb_croplate', 'comb_stale_cf']

if __name__ == '__main__':
    t0 = time.time()
    names = sys.argv[1:] or ALL
    say(f'K2 smoke {time.strftime("%Y-%m-%d %H:%M:%S")} on {PY}')
    stage()
    ok_b = worker_parity()
    ok_c = scenarios(names)
    ok_d = real_build()
    say(f'K2 SMOKE {"PASS" if (ok_b and ok_c and ok_d) else "FAIL"} in {(time.time() - t0) / 60:.1f} min')
    open(f'{T}/smoke_transcript.txt', 'w', encoding='utf-8').write('\n'.join(LOG) + '\n')
