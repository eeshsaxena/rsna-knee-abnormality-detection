# dev helper: time the K2e slow-study run with the K3 worker vs the K5 worker on the same smoke dataset (laptop CPU)
import os, sys, shutil, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
import smoke_k2 as s
d3 = f'{s.S}/ds/fleet_real_k3w_t'
if not os.path.isdir(d3):
    shutil.copytree(f'{s.S}/ds/fleet_real', d3)
    shutil.copy2(f'{s.SP}/k3/code/fleet_worker.py', f'{d3}/fleet_worker.py')
sids = __import__('pandas').read_csv(f'{s.S}/comp_big/test.csv', dtype=str).StudyInstanceUID.tolist()
for tag, ds, slow in (('k5_slow', 'fleet_real', sids[5]), ('k3_slow', 'fleet_real_k3w_t', sids[5]), ('k3_normal', 'fleet_real_k3w_t', 'none')):
    e = dict(s.ENV, FLEET_ALLOW_CPU='1', CUDA_VISIBLE_DEVICES='-1', FLEET_SHARD='0', FLEET_NSHARD='1',
             PYTHONPATH=f'{s.DRY}/shim', FLEET_DIR=f'{s.S}/ds/{ds}', FLEET_OUT=f'{s.S}/cmp_{tag}.npz',
             RSNA_COMP_ROOT=f'{s.S}/comp_big', FLEET_MICROBENCH='0', FLEET_STUDY_TIMEOUT_S='1', SLOW_SID=slow)
    t0 = time.time()
    r = subprocess.run([s.PY, '-W', 'ignore', f'{s.S}/slow_runner.py'], env=e, capture_output=True, text=True)
    print(tag, 'rc', r.returncode, f'{time.time() - t0:.0f} s', [l[:90] for l in r.stdout.splitlines() if 'done' in l or 'timeout' in l], flush=True)
