"""K8 (scratchpad/k8, 2026-10-07): + K8-KF: a KF notebook (make_k2_kernel --outer_flat 0.6: K2_CFG OUTER_FLAT + the KF cell)
  and its refusals (--outer_flat with blend / solo), scenarios kf_ok / kf_perturb / kf_missing / kf_drift (run_scenario.py).
K5 laptop CPU smoke (scratchpad/k5) = the K3 smoke below, unchanged, run on the K5 kit (K5 worker, K5 cell, K5
build_k2 / pack), + K5:
  A5 the TTR pair packed with --tta A,AB (family 1 A, TTR AB); a twin with the K3 worker (pre-K5); a twin whose manifest
     asks an unknown tta mode
  B5 K5 worker A-only == the K3 / K4 worker on the TTR pair (P max|diff| 0); AB run: TTR pass A (PA) == the A-only run,
     P == (PA + PB) / 2 exactly, family-1 rows == the A-only run, TTA record (modes, rows, forwards); commit-sized
     microbench with AB: bench passes {f1: 1, ttr: 2}
  C5 scenarios strict_ttr_tta (clean, k2_check --expect_tta), strict_tta_oldworker (manifest asks AB, pre-K5 worker:
     fail loud), strict_tta_bad (unknown mode: fail loud), ttrsolo_tta (solo, TTR AB)
  D  the real build reads M1 + M2 from k3/ckpt (read only; k5/ckpt holds the K5 set) + build_k2.py --tta on them
K3 text follows.
K3 laptop CPU smoke (scratchpad/k3; harness only; nothing here ships, nothing is pushed). = the K2 smoke + K3:
  A+ a tiny REAL TTR member (KneeFleetNetTTR, ttr_size 48 on the 64 px store, seed 1042, model_sha of the shipped
     fleet_model_ttr.py) packed with a dry family-1 member by k3/code/pack_fleet_ds.py (bigjump pack); a stub dataset
     whose stub worker reads the (filtered) manifest; 5 more notebooks (strict, strict + 2 members, TTR solo, stub
     filter, unknown member)
  B+ K3 worker (bigjump) == K2 kit worker on family-1 (P max|diff| 0)
  C+ strict / member-filter / outer-handler scenarios, real workers on the TTR pair
  C2 cross-run: the TTR member's rows in the filtered TTR-solo run == its rows in the unfiltered 2-member run, and the
     family-1 member's rows == the family-1-only run (the filter and the extra family change no prediction)
  D  build_k2.py on the REAL M1 + M2 (k3/ckpt): strict kernel + --kernel_only TTR-solo kernel + refusals
K2 text follows.
K2 laptop CPU smoke (harness only; nothing here ships, nothing is pushed). Reuses round3/fleet_dry and cropfleet/_smoke
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


STUB = r'''# STUB fleet worker (K2 smoke only). STUB=ok|fastfail|hang|valve|hardstop|blank|views2|okman
import os, sys, time, json
import numpy as np, pandas as pd
mode, shard, nsh = os.environ.get('STUB', 'ok'), int(os.environ['FLEET_SHARD']), int(os.environ['FLEET_NSHARD'])
MAN = json.load(open(os.environ['FLEET_DIR'] + '/fleet_manifest.json'))['checkpoints']   # okman: names from the manifest
ids = pd.read_csv(os.environ['RSNA_COMP_ROOT'] + '/test.csv', dtype=str).StudyInstanceUID.tolist()[shard::nsh]
if mode == 'hang':
    time.sleep(600)
if mode == 'fastfail':
    if shard == 1:
        sys.exit(3)
    time.sleep(600)
M = 3 if mode == 'valve' else (len(MAN) if mode == 'okman' else 2)
P = np.random.default_rng(100 + shard).uniform(0.01, 0.99, (M, len(ids), 12)).astype(np.float32)
keep = 1 if mode == 'valve' else M
P[keep:] = np.nan
NV = np.full(len(ids), 4)
if mode == 'blank' and len(ids):        # one failed study per shard: 2/24 = 8 % > the 2 % coverage gate
    P[:, 0] = np.nan; NV[0] = 0
if mode == 'views2':                    # every study decoded with 2 views: below the 3 views/study gate
    NV[:] = 2
out = os.environ['FLEET_OUT']; tmp = out[:-4] + '.tmp.npz'
np.savez(tmp, sids=np.array(ids), P=P, NV=NV, names=np.array(MAN if mode == 'okman' else ['stub_a', 'stub_b', 'stub_c'][:M]),
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
    mk.main(['--out', f'{S}/nb/rsna-sub-k2-solo', '--slug', 'rsna-sub-k2-solo', '--main_mode', 'solo'])
    say('A notebooks built: fleet-only (32 cells), combined (35 cells), combined crop-floor-25 (35 cells)')
    stage_k3(mk)


MAKE_TTR = r'''# smoke only: a tiny REAL TTR member from the dry family-1 member (same weights, KneeFleetNetTTR at ttr_size 48)
import sys, os, copy, hashlib, inspect, torch
code, src, dst = sys.argv[1:4]
sys.path.insert(0, code)
import fleet_model as fm, fleet_model_ttr as tt
ck = torch.load(src, map_location='cpu', weights_only=False)
a = dict(ck['args'])
kw = dict(backbone=a['backbone'], n_views=a['n_views'], n_slices=a['n_slices'], dim=a['dim'], n_contrast=a['n_contrast'],
          contrast_emb=a['contrast_emb'], pretrained=False, drop_path=0.0, chunk=48)
m = tt.KneeFleetNetTTR(ttr_size=48, **kw)
miss, unexp = m.load_state_dict(ck['model'], strict=False)
assert list(miss) == ['ttr_size_t'] and not unexp, (miss, unexp)
sha = hashlib.sha256((chr(10).join(inspect.getsource(tt).splitlines()) + chr(10) +
                      chr(10).join(inspect.getsource(fm).splitlines())).encode()).hexdigest()[:16]
a.update(seed=1042, model_mod='fleet_model_ttr', ttr_size=48, model_sha=sha)
ck2 = dict(ck); ck2['args'] = a; ck2['model'] = m.state_dict()
torch.save(ck2, dst)
print('TTR', os.path.basename(dst), 'sha', sha, 'ttr_size_t', int(ck2['model']['ttr_size_t']))
'''


def stage_k3(mk):
    import pandas as pd
    os.makedirs(f'{S}/ttrmem')
    open(f'{S}/make_ttr.py', 'w').write(MAKE_TTR)
    f1 = f'{DRY}/fleetds/f0_s42_ema_cnxt64x24_cv5714_best.pt'
    ttr = f'{S}/ttrmem/f0_s1042_ema_ttr48-64x24_cv5714_best.pt'
    r = run([f'{S}/make_ttr.py', f'{S}/code64', f1, ttr], log=f'{S}/log_make_ttr.txt')
    assert r.returncode == 0, r.stderr[-1500:]
    ok_sha = 'sha f141aeed136dfc01' in r.stdout
    say(f'A+ tiny TTR member: {r.stdout.strip()[-120:]} -> {"PASS" if ok_sha else "FAIL (model_sha != f141aeed136dfc01)"}')
    assert ok_sha
    r = run([f'{S}/code64/pack_fleet_ds.py', '--out', f'{S}/ds/fleet_ttr', '--store', f'{DRY}/store_64_normal', f1, ttr],
            log=f'{S}/log_pack_ttr.txt')
    assert r.returncode == 0, r.stderr[-1500:]
    say('A+ pack (k3 = bigjump pack tool) family-1 + TTR: ' + r.stdout.strip().splitlines()[-1][-150:])
    names = json.load(open(f'{S}/ds/fleet_ttr/fleet_manifest.json'))['checkpoints']
    tm = dict(f1=os.path.basename(f1), ttr=os.path.basename(ttr))
    assert sorted(names) == sorted(tm.values()), names
    json.dump(tm, open(f'{S}/ttr_members.json', 'w'))
    # K5: the same pair with per-member TTA (family 1 A, TTR AB), a pre-K5 worker twin, an unknown-mode twin
    r = run([f'{S}/code64/pack_fleet_ds.py', '--out', f'{S}/ds/fleet_ttr_tta', '--store', f'{DRY}/store_64_normal',
             '--tta', 'A,AB', f1, ttr], log=f'{S}/log_pack_ttr_tta.txt')
    assert r.returncode == 0, r.stderr[-1500:]
    mt = json.load(open(f'{S}/ds/fleet_ttr_tta/fleet_manifest.json'))
    ok_t = {k: v.get('tta') for k, v in mt['sources'].items()} == {tm['f1']: 'A', tm['ttr']: 'AB'}
    say(f'A5 pack --tta A,AB: manifest tta {[(k[:14], v.get("tta")) for k, v in mt["sources"].items()]} -> {"PASS" if ok_t else "FAIL"}')
    assert ok_t
    shutil.copytree(f'{S}/ds/fleet_ttr_tta', f'{S}/ds/fleet_ttr_tta_k3w')
    shutil.copy2(f'{SP}/k3/code/fleet_worker.py', f'{S}/ds/fleet_ttr_tta_k3w/fleet_worker.py')
    shutil.copytree(f'{S}/ds/fleet_ttr_tta', f'{S}/ds/fleet_ttr_badtta')
    mb = json.load(open(f'{S}/ds/fleet_ttr_badtta/fleet_manifest.json'))
    mb['sources'][tm['ttr']]['tta'] = 'ABM'
    json.dump(mb, open(f'{S}/ds/fleet_ttr_badtta/fleet_manifest.json', 'w'), indent=1)
    shutil.copytree(f'{S}/ds/fleet_ttr', f'{S}/ds/fleet_ttr_k3w')          # K3 worker on the A-only TTR pair
    shutil.copy2(f'{SP}/k3/code/fleet_worker.py', f'{S}/ds/fleet_ttr_k3w/fleet_worker.py')
    # K2 kit worker twin of fleet_real (family-1 parity K3 worker vs K2 worker)
    shutil.copytree(f'{S}/ds/fleet_real', f'{S}/ds/fleet_k2w')
    shutil.copy2(f'{SP}/k2/code/fleet_worker.py', f'{S}/ds/fleet_k2w/fleet_worker.py')
    # stub dataset whose stub reads the manifest (.pt names, files present so the filtered view links them)
    os.makedirs(f'{S}/ds/fleet_stubman')
    open(f'{S}/ds/fleet_stubman/fleet_worker.py', 'w').write(STUB)
    for n in ('stub_a.pt', 'stub_b.pt'):
        open(f'{S}/ds/fleet_stubman/{n}', 'w').write('stub')
    json.dump({'checkpoints': ['stub_a.pt', 'stub_b.pt']}, open(f'{S}/ds/fleet_stubman/fleet_manifest.json', 'w'))
    W = ['--w_main', '0.25']
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-strict', '--slug', 'rsna-sub-k3-strict', '--main_mode', 'strict'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-strictttr', '--slug', 'rsna-sub-k3-strictttr', '--main_mode', 'strict',
             '--members', f'{tm["f1"]},{tm["ttr"]}'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-ttrsolo', '--slug', 'rsna-sub-k3-ttrsolo', '--main_mode', 'solo',
             '--members', tm['ttr']] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-stubfilter', '--slug', 'rsna-sub-k3-stubfilter', '--main_mode', 'strict',
             '--members', 'stub_b.pt'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-unknown', '--slug', 'rsna-sub-k3-unknown', '--main_mode', 'strict',
             '--members', 'nope.pt'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k5-strictexp', '--slug', 'rsna-sub-k5-strictexp', '--main_mode', 'strict',   # K5b
             '--members', f'{tm["f1"]},{tm["ttr"]}', '--tta', f'{tm["f1"]}=A,{tm["ttr"]}=AB'] + W)
    # the old notebooks' K2_CFG carries no MEMBERS key (K2 cells byte-identical apart from the shared cell source)
    for nm in ('rsna-sub-k2-fleet', 'rsna-sub-k2-solo'):
        nb = json.load(open(f'{S}/nb/{nm}/{nm.replace("-", "_")}.ipynb', encoding='utf-8'))
        assert 'MEMBERS' not in ''.join(nb['cells'][31]['source']).splitlines()[0], nm
    say('A+ K3 notebooks built: strict W.25, strict + 2-member filter, TTR solo, stub filter, unknown member')
    # K8-KF: the KF twin notebook + refusals
    mk.main(['--out', f'{S}/nb/rsna-sub-k8-kf', '--slug', 'rsna-sub-k8-kf', '--main_mode', 'strict',
             '--members', f'{tm["f1"]},{tm["ttr"]}', '--tta', f'{tm["f1"]}=A,{tm["ttr"]}=AB', '--outer_flat', '0.6'] + W)
    nbk = json.load(open(f'{S}/nb/rsna-sub-k8-kf/rsna_sub_k8_kf.ipynb', encoding='utf-8'))
    src = ''.join(nbk['cells'][31]['source'])
    kfc = open(f'{K2}/cells/k2_fleet_cell_kf.py', encoding='utf-8').read()
    assert src.split(chr(10), 1)[1] == kfc and "'OUTER_FLAT': 0.6" in src.split(chr(10), 1)[0], 'KF notebook cell 31'
    refused = []
    for mode in ('blend', 'solo'):
        try:
            mk.main(['--out', f'{S}/nb/rsna-sub-k8-kfbad', '--slug', 'rsna-sub-k8-kfbad', '--main_mode', mode,
                     '--outer_flat', '0.6'] + W)
        except AssertionError as e:
            refused.append('strict' in str(e))
    assert refused == [True, True], f'--outer_flat with blend / solo must be refused: {refused}'
    say('A8 K8-KF notebook built (KF cell + OUTER_FLAT 0.6, strict, TTR pair AB); --outer_flat with blend / solo refused')


SLOW_RUNNER = r'''# K2e test runner (smoke only): the shipped worker, with fleet_common.build_study sleeping 30 s on ONE study
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
    """K2e: one study whose decode hangs is skipped after the timeout; every other row equals the normal run."""
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
    base = dict(ENV, FLEET_ALLOW_CPU='1', CUDA_VISIBLE_DEVICES='-1', FLEET_SHARD='0', FLEET_NSHARD='1',
                PYTHONPATH=f'{DRY}/shim')
    outs = {}
    for tag, ds, comp, mb in (('k2', 'fleet_real', 'comp_big', '0'), ('fleet2', 'fleet_f2orig', 'comp_big', '0'),
                              ('k2_bench', 'fleet_real', 'comp3', '3'), ('k2kit', 'fleet_k2w', 'comp_big', '0'),
                              ('ttr', 'fleet_ttr', 'comp_big', '0'), ('ttr_k3w', 'fleet_ttr_k3w', 'comp_big', '0'),
                              ('ttr_tta', 'fleet_ttr_tta', 'comp_big', '0'), ('ttr_tta_bench', 'fleet_ttr_tta', 'comp3', '3')):
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
    say(f'B worker parity K3 (bigjump) vs fleet2 on {len(zk["sids"])} studies: P max|diff| {d} NaN-equal {nan_eq} -> {"PASS" if ok1 else "FAIL"}')
    zw = outs['k2kit'][0]
    d2 = float(np.nanmax(np.abs(zk['P'] - zw['P'])))
    ok5 = list(zk['names']) == list(zw['names']) and d2 == 0.0 and bool((np.isnan(zk['P']) == np.isnan(zw['P'])).all())
    say(f'B+ worker parity K3 (bigjump) vs the K2 kit worker (family 1, the proven K2 path): P max|diff| {d2} -> {"PASS" if ok5 else "FAIL"}')
    zt = outs['ttr'][0]; tn = [str(x) for x in zt['names']]
    tm = json.load(open(f'{S}/ttr_members.json'))
    i1, it = tn.index(tm['f1']), tn.index(tm['ttr'])
    j1 = [str(x) for x in zk['names']].index(tm['f1'])
    d3 = float(np.nanmax(np.abs(zt['P'][i1] - zk['P'][j1])))
    dt = float(np.nanmax(np.abs(zt['P'][it] - zt['P'][i1])))
    ok6 = (int(zt['n_keep']) == 2 and d3 == 0.0 and np.isfinite(zt['P']).all() and dt > 1e-4
           and '[KneeFleetNetTTR]' in outs['ttr'][2] and '[KneeFleetNet]' in outs['ttr'][2])
    say(f'B+ K3 worker serves family 1 + TTR: keep {int(zt["n_keep"])}/2, family-1 rows == the family-1-only run '
        f'(max|diff| {d3}), TTR differs from its family-1 twin by {dt:.3f} (ttr 48 px resize active) -> {"PASS" if ok6 else "FAIL"}')
    ok1 = ok1 and ok5 and ok6
    say(f'B K2 worker extras: decode_s {st.get("decode_s", 0):.2f} s summed, no bench without the env -> {"PASS" if ok2 else "FAIL"}')
    say(f'B K2 microbench (3 studies, 3 forwards, CPU): {b} -> {"PASS" if ok3 else "FAIL"}')
    ok4 = worker_timeout(zk)
    ok5 = worker_parity_k5(outs, tm)
    return ok1 and ok2 and ok3 and ok4 and ok5


def worker_parity_k5(outs, tm):
    """B5: the K5 worker's A path == the K3/K4 worker; the AB path == pass A + pass B as tta_bench builds them."""
    import numpy as np
    zt, zk3 = outs['ttr'][0], outs['ttr_k3w'][0]
    d0 = float(np.nanmax(np.abs(zt['P'] - zk3['P'])))
    ok0 = (list(zt['names']) == list(zk3['names']) and list(zt['sids']) == list(zk3['sids']) and d0 == 0.0
           and (zt['NV'] == zk3['NV']).all() and bool((np.isnan(zt['P']) == np.isnan(zk3['P'])).all())
           and 'tta' not in zk3.files and json.loads(str(zt['tta']))['modes'] == {tm['f1']: 'A', tm['ttr']: 'A'})
    say(f'B5 K5 worker A-only == K3/K4 worker on family 1 + TTR ({len(zt["sids"])} studies): P max|diff| {d0} -> '
        f'{"PASS" if ok0 else "FAIL"}')
    za = outs['ttr_tta'][0]; nm = [str(x) for x in za['names']]
    i1, it = nm.index(tm['f1']), nm.index(tm['ttr'])
    dA = float(np.nanmax(np.abs(za['PA'][it] - zt['P'][it])))
    d1 = float(np.nanmax(np.abs(za['P'][i1] - zt['P'][i1])))
    ident = bool(np.array_equal(za['P'][it], (za['PA'][it] + za['PB'][it]) / 2))
    dB = float(np.nanmax(np.abs(za['PB'][it] - za['PA'][it])))
    t = json.loads(str(za['tta'])); n = len(za['sids'])
    ok1 = (dA == 0.0 and d1 == 0.0 and ident and dB > 1e-4 and np.isnan(za['PB'][i1]).all()
           and t['modes'] == {tm['f1']: 'A', tm['ttr']: 'AB'} and t['rows'] == n
           and t['fwd'] == {tm['f1']: {'A': n, 'B': 0}, tm['ttr']: {'A': n, 'B': n}} and abs(t['half'] - 0.499) < 1e-12
           and 'TTA modes' in outs['ttr_tta'][2] and 'TTA forwards' in outs['ttr_tta'][2])
    say(f'B5 K5 worker f1 A + TTR AB: TTR pass A == A-only run ({dA}), f1 rows == A-only run ({d1}), P == (PA+PB)/2 '
        f'{ident}, max|PB-PA| {dB:.3f}, record rows {t["rows"]} fwd {t["fwd"]} -> {"PASS" if ok1 else "FAIL"}')
    zb = outs['ttr_tta_bench'][0]; b = json.loads(str(zb['bench'])); tb = json.loads(str(zb['tta']))
    ok2 = (set(b) == {tm['f1'], tm['ttr']} and all(v > 0 for v in b.values())
           and tb['bench_passes'] == {tm['f1']: 1, tm['ttr']: 2} and 'MICROBENCH' in outs['ttr_tta_bench'][2])
    say(f'B5 K5 microbench (3 studies, AB): {b} passes {tb["bench_passes"]} -> {"PASS" if ok2 else "FAIL"}')
    return ok0 and ok1 and ok2


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


def cross_runs():
    """C2: the member filter and the second family change no prediction (rows compared by member name)."""
    import numpy as np
    tm = json.load(open(f'{S}/ttr_members.json'))

    def rows(sc, name):
        out = {}
        for z in [np.load(f) for f in sorted(glob.glob(f'{S}/run_{sc}/working/_fleet_*.npz'))]:
            nm = [str(x) for x in z['names']]
            for k, sid in enumerate(z['sids']):
                out[str(sid)] = z['P'][nm.index(name), k]
        return out
    try:
        a, b = rows('ttrsolo_ok', tm['ttr']), rows('strict_ttr', tm['ttr'])
        c, d = rows('strict_ttr', tm['f1']), rows('strict_ttr_all', tm['f1'])
    except Exception as e:
        say(f'C2 cross-run check: needs ttrsolo_ok, strict_ttr, strict_ttr_all ({e!r}) -> FAIL'); return False
    d1 = max(float(np.abs(a[s] - b[s]).max()) for s in a)
    d2 = max(float(np.abs(c[s] - d[s]).max()) for s in c)
    ok = sorted(a) == sorted(b) and sorted(c) == sorted(d) and len(a) == 24 and d1 == 0.0 and d2 == 0.0
    say(f'C2 TTR rows filtered (solo) vs unfiltered 2-member run: max|diff| {d1}; family-1 rows filtered pair vs no '
        f'filter: {d2} ({len(a)} studies) -> {"PASS" if ok else "FAIL"}')
    return ok


def real_build():
    ck = sorted(glob.glob(f'{SP}/k3/ckpt/*.pt'))                         # K5: read only (k5/ckpt holds the K5 set)
    m1 = [c for c in ck if os.path.basename(c).startswith('f0_s42_')]
    m2 = [c for c in ck if os.path.basename(c).startswith('f0_s1042_')]
    if len(m1) != 1 or len(m2) != 1:
        say(f'D real build: k3/ckpt must hold exactly one f0_s42_ and one f0_s1042_ file, has {ck} -> FAIL'); return False
    m1, m2 = m1[0], m2[0]
    n1, n2 = os.path.basename(m1), os.path.basename(m2)
    out = f'{S}/real_build'
    r = run([f'{K2}/build_k2.py', '--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out', out,
             '--slug', 'rsna-sub-k3-fleet', '--main_mode', 'strict', '--w_main', '0.25', '--members', f'{n1},{n2}',
             '--overwrite'], log=f'{S}/log_real_build.txt')
    ok = r.returncode == 0 and 'upload folder OK' in r.stdout
    r2 = run([f'{K2}/build_k2.py', '--kernel_only', '--out', out, '--slug', 'rsna-sub-k3-ttrsolo', '--main_mode', 'solo',
              '--w_main', '0.25', '--members', n2], log=f'{S}/log_real_build_ko.txt')
    ok = ok and r2.returncode == 0
    summ = json.load(open(f'{out}/BUILD_SUMMARY.json')) if ok else {}
    ks = {k['slug']: k for k in summ.get('kernels', [])}
    ok = ok and set(ks) == {'rsna-sub-k3-fleet', 'rsna-sub-k3-ttrsolo'} \
        and ks['rsna-sub-k3-fleet']['K2_CFG'].get('MAIN_MODE') == 'strict' \
        and ks['rsna-sub-k3-fleet']['K2_CFG'].get('MEMBERS') == [n1, n2] \
        and ks['rsna-sub-k3-ttrsolo']['K2_CFG'].get('MAIN_MODE') == 'solo' \
        and ks['rsna-sub-k3-ttrsolo']['K2_CFG'].get('MEMBERS') == [n2] \
        and 'fleet_model_ttr.py' in summ['upload']['files']
    say(f'D build_k2.py on the REAL M1 + M2 ({n1}, {n2}): {"PASS" if ok else "FAIL"} | upload {summ.get("upload", {}).get("mb")} MB '
        f'{len(summ.get("upload", {}).get("files", []))} files | kernels {sorted(ks)}'
        + ('' if ok else f' | {r.stderr[-800:]} {r2.stderr[-800:]}'))
    # K5: build_k2.py --tta on the real pair (M1 A, M2 AB) writes the manifest tta; a partial map is refused
    out5 = f'{S}/real_build_tta'
    r5 = run([f'{K2}/build_k2.py', '--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out', out5,
              '--slug', 'rsna-sub-k5-x', '--main_mode', 'strict', '--w_main', '0.20', '--w_variants', '0.30,0.50', '--members', f'{n1},{n2}',
              '--tta', f'{n1}=A,{n2}=AB', '--overwrite'], log=f'{S}/log_real_build_tta.txt')
    ok5 = r5.returncode == 0
    if ok5:
        man5 = json.load(open(f'{out5}/upload/rsna-knee-cnxt-fleet/fleet_manifest.json'))
        ok5 = {k: v.get('tta') for k, v in man5['sources'].items()} == {n1: 'A', n2: 'AB'}
    r6 = run([f'{K2}/build_k2.py', '--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out', f'{S}/real_build_neg',
              '--overwrite', '--tta', f'{n2}=AB'])
    ok6 = r6.returncode != 0 and '--tta names' in r6.stderr
    say(f'D5 build_k2.py --tta on the REAL M1 + M2: manifest tta {"OK" if ok5 else "FAIL " + r5.stderr[-300:]}; '
        f'refuses a partial --tta map: {"PASS" if ok6 else "FAIL " + r6.stderr[-300:]}')
    ok = ok and ok5 and ok6
    neg = []
    # a DIFFERENT run with M1's fold/seed (same weights, other train_labels): the builder must refuse the duplicate key
    os.makedirs(f'{S}/dupfs', exist_ok=True)
    dup = f'{S}/dupfs/{n1}'
    open(f'{S}/make_dup.py', 'w').write(
        'import sys, torch\nck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)\n'
        'ck["args"] = dict(ck["args"], train_labels="other_labels.csv")\ntorch.save(ck, sys.argv[2])\n')
    rd = run([f'{S}/make_dup.py', m1, dup])
    assert rd.returncode == 0, rd.stderr[-800:]
    for nm, args, needle in (
            ('a second run with the same fold/seed', ['--ckpt', m1, '--ckpt', dup, '--store', f'{K2}/_real/store', '--out',
                                                      f'{S}/real_build_neg', '--overwrite'], 'duplicate member key f0_s42'),
            ('two tags of one run', ['--ckpt', m1, '--ckpt', m1, '--store', f'{K2}/_real/store', '--out', f'{S}/real_build_neg',
                                     '--overwrite'], 'one tag per run'),
            ('an unknown --members name', ['--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out',
                                           f'{S}/real_build_neg', '--overwrite', '--members', 'nope.pt'], 'not in the manifest'),
            ('--kernel_only with an unknown name', ['--kernel_only', '--out', out, '--slug', 'rsna-sub-k3-x',
                                                    '--members', 'f0_s1042_ema_x.pt'], 'not in the manifest'),
            ('--kernel_only on an existing slug', ['--kernel_only', '--out', out, '--slug', 'rsna-sub-k3-ttrsolo',
                                                   '--members', n2], 'already built')):
        rn = run([f'{K2}/build_k2.py'] + args)
        good = rn.returncode != 0 and needle in rn.stderr
        neg.append(good)
        say(f'D build_k2.py refuses {nm}: {"PASS" if good else "FAIL " + rn.stderr[-300:]}')
    return ok and all(neg)


def real_build_k2():
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
       'fleet_badparent', 'fleet_blank', 'fleet_views2', 'fleet_noclock',
       'comb_normal', 'comb_stub_ok', 'comb_fleetfail', 'comb_nocrop', 'comb_croplate', 'comb_stale_cf',
       'solo_ok', 'solo_fail',
       'strict_ok', 'strict_fail', 'strict_writefail', 'strict_lowtime', 'strict_nods', 'strict_outer', 'solo_outer',
       'blend_outer', 'strict_ttr', 'strict_ttr_all', 'ttrsolo_ok', 'members_stub_ok', 'members_mismatch',
       'members_unknown',
       'strict_ttr_tta', 'strict_tta_oldworker', 'strict_tta_bad', 'ttrsolo_tta',                     # K5
       'strict_tta_expect_ok', 'strict_tta_expect_k4ds',                                                # K5b
       'kf_ok', 'kf_perturb', 'kf_missing', 'kf_drift']                                                 # K8-KF

if __name__ == '__main__':
    t0 = time.time()
    names = sys.argv[1:] or ALL
    say(f'K5 smoke {time.strftime("%Y-%m-%d %H:%M:%S")} on {PY}')
    stage()
    ok_b = worker_parity()
    ok_c = scenarios(names)
    ok_c2 = cross_runs() if {'ttrsolo_ok', 'strict_ttr', 'strict_ttr_all'} <= set(names) else True
    ok_d = real_build() if os.environ.get('K3_SMOKE_SKIP_REAL') != '1' else True
    say(f'K5 SMOKE {"PASS" if (ok_b and ok_c and ok_c2 and ok_d) else "FAIL"} in {(time.time() - t0) / 60:.1f} min')
    open(f'{T}/smoke_transcript.txt', 'w', encoding='utf-8').write('\n'.join(LOG) + '\n')
