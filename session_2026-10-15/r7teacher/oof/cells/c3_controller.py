# R7-OOF controller [eeshsaxena, not in the R7 notebook]. Every RSNA train study is scored ONLY by the R7 fold model that did
# not train on it. compact_train.py v9.8 (pinned above by sha256 + md5): pool_fold = sha1(uid)[:8] % 4 and fold k's model trained
# on pool_fold != k, saved as E_lrprior_s0_f{k}.pt. The worker is the R7 notebook's compact_infer.py (SOURCES above, sha-checked,
# unchanged) with the R7 WORKER_ARGS unchanged. Differences from the R7 test path, all outside the worker code: --split train,
# a model folder that holds one fold file (the worker globs <arm>_s*_f*.pt), --dump-logits / --dump-state (read-outs only).
# probe_dual.py is not used because it needs sample_submission ids (test only); this cell starts the same worker command per GPU.
import csv, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path
csv.field_size_limit(1 << 30)
C = R7OOF
OUTW = WORK / 'r7oof'                       # small per-job artifacts, copied here at the end
OUTW.mkdir(parents=True, exist_ok=True)
JOBS_DIR = TMP / 'r7oof_jobs'               # worker outputs + their tmp renders (as R7: under TMP, not /kaggle/working)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
DEADLINE = NB_T0 + float(C['time_budget_h']) * 3600      # each worker stops after the study in flight; the merge cell follows
LABELS12 = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
            "Baker's", "Contusion", "Fracture"]

# ---- the fold rule and the plain-pass contract, checked on the pinned sources / manifest actually loaded above
assert manifest['run_config']['labels'] == LABELS12
assert manifest['run_config']['folds'] == 4 and manifest['run_config']['holdout_fold'] is None
assert manifest['models'] == ['%s_s0_f%d.pt' % (ARM, k) for k in range(4)], manifest['models']
assert manifest['gold_in_training'] == 0
_TR = SOURCES['compact_train.py']
assert _TR.count("pool_fold = np.array([int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool])") == 1
assert _TR.count("pool_fold = ") == 1
assert "tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)" in _TR
assert not manifest.get('flip_tta') and '--flip-tta' not in WORKER_ARGS, 'plain pass only'
assert hashlib.sha256((SRC / WORKER).read_bytes()).hexdigest() == manifest['src_sha256']['compact_infer.py']


def r7_fold(u):
    """compact_train.py v9.8 pool_fold with K = 4"""
    return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4


def shard_of(u, n):
    """session shard: md5, independent of the sha1 fold rule"""
    return int(hashlib.md5(u.encode()).hexdigest()[:8], 16) % n


with open(DATA / 'train.csv', encoding='utf-8', newline='') as fh:
    TRAIN_IDS = [r['StudyInstanceUID'] for r in csv.DictReader(fh)]
assert len(TRAIN_IDS) == len(set(TRAIN_IDS)) == 4407, len(TRAIN_IDS)
NSER = {}
with open(DATA / 'train_series.csv', encoding='utf-8', newline='') as fh:
    for r in csv.DictReader(fh):
        NSER[r['StudyInstanceUID']] = NSER.get(r['StudyInstanceUID'], 0) + 1
if C['uids'] is not None:                   # explicit list (commit / proof modes; may be empty)
    TODO = list(C['uids'])
    assert set(TODO) <= set(TRAIN_IDS) and len(TODO) == len(set(TODO))
else:
    TODO = [u for u in TRAIN_IDS if shard_of(u, int(C['shard_count'])) == int(C['shard_index'])]
SKIP = set(C['skip'] or [])
TODO = [u for u in TODO if u not in SKIP]
print('R7OOF_TODO', json.dumps({'studies': len(TODO), 'skipped': len(SKIP), 'per_fold': [sum(r7_fold(u) == k for u in TODO) for k in range(4)],
                                'shard': [C['shard_index'], C['shard_count']], 'explicit_uids': bool(C['uids'])}), flush=True)

# ---- one-fold model folders (symlinks to the sha-checked bundle copies)
BF = {}
for k in range(4):
    d = TMP / ('bundle_f%d' % k)
    d.mkdir(exist_ok=True)
    name = '%s_s0_f%d.pt' % (ARM, k)
    if not (d / name).exists():
        os.symlink(BUNDLE / name, d / name)
    assert sorted(p.name for p in d.glob(ARM + '_s*_f*.pt')) == [name]
    BF[k] = d

# ---- jobs: per fold, chunks of at most C['chunk'] studies (big and small studies mixed); folds interleaved; proof jobs last
by_fold = {k: [] for k in range(4)}
for u in sorted(TODO, key=lambda u: (-NSER.get(u, 0), hashlib.md5(u.encode()).hexdigest())):
    by_fold[r7_fold(u)].append(u)
per_fold_jobs = {}
for k in range(4):
    L_ = by_fold[k]
    n_ = max(1, -(-len(L_) // int(C['chunk'])))
    per_fold_jobs[k] = [{'id': 'oof_f%d_c%02d' % (k, j), 'kind': 'oof', 'fold': k, 'uids': L_[j::n_], 'models': str(BF[k])}
                        for j in range(n_) if L_[j::n_]]
JOBS = []
for j in range(max(len(v) for v in per_fold_jobs.values())):
    for k in range(4):
        if j < len(per_fold_jobs[k]):
            JOBS.append(per_fold_jobs[k][j])
N_GPU = max(1, len([l for l in subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                                                text=True).stdout.splitlines() if l.strip()]))
PROOF = [u for u in (C['proof_uids'] or []) if u in set(TRAIN_IDS)]
for gi in range(N_GPU):                     # the unchanged 4-model R7 path on the proof studies (all four folds per study)
    part = PROOF[gi::N_GPU]
    if part:
        JOBS.append({'id': 'proof_g%d' % gi, 'kind': 'proof', 'fold': None, 'uids': part, 'models': str(BUNDLE)})
print('R7OOF_JOBS', json.dumps([[j['id'], len(j['uids'])] for j in JOBS]), flush=True)

IO_BENCH = None

LOCK = threading.Lock()


def worker_env(gi):
    return dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1',
                CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MALLOC_ARENA_MAX='2')


def run_job(job, gi):
    wdir = JOBS_DIR / job['id']
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / 'ids.json').write_text(json.dumps(job['uids']), encoding='utf-8')
    left = DEADLINE - time.time()
    if left < float(C['min_job_start_min']) * 60:
        job.update(status='not started: %.1f min left' % (left / 60), gpu=gi)
        return
    # the probe_dual.py worker command for the compact worker, + --dump-logits / --dump-state, + this job's --tmp
    cmd = [sys.executable, '-u', str(SRC / WORKER), '--data', str(DATA), '--split', 'train', '--code', str(CODE), '--arm', ARM,
           '--assignment', str(wdir / 'ids.json'), '--output', str(wdir), '--amp', 'auto', '--workers', str(int(C['loader_workers'])),
           '--deadline', str(DEADLINE)]
    cmd += ['--models', job['models']] + [str(x) for x in WORKER_ARGS]
    cmd += ['--dump-logits', str(wdir / 'dump.json')] + (['--dump-state'] if C['dump_state'] else [])
    t = time.time()
    with LOCK:
        job.update(gpu=gi, start_s=round(t - NB_T0, 1), status='running')
    with open(wdir / 'log.txt', 'w', encoding='utf-8') as log:
        rc = subprocess.run(cmd, env=worker_env(gi), stdout=log, stderr=subprocess.STDOUT).returncode
    with LOCK:
        job.update(rc=rc, seconds=round(time.time() - t, 1), end_s=round(time.time() - NB_T0, 1), status='done' if rc == 0 else 'worker exit %d' % rc)
    print('JOB_END', job['id'], 'gpu', gi, 'rc', rc, 'seconds %.0f' % (time.time() - t), flush=True)


QUEUE = list(JOBS)


def gpu_loop(gi):
    while True:
        with LOCK:
            if not QUEUE:
                return
            job = QUEUE.pop(0)
        run_job(job, gi)


def cpu_times():
    with open('/proc/stat') as fh:
        v = [int(x) for x in fh.readline().split()[1:]]
    return sum(v), v[3] + v[4]


SAMPLES = []
threads = [threading.Thread(target=gpu_loop, args=(gi,), daemon=True) for gi in range(N_GPU)]
for th in threads:
    th.start()
last, (tot0, idle0) = 0.0, cpu_times()
while any(th.is_alive() for th in threads):
    time.sleep(5)
    if time.time() - last >= 60:
        last = time.time()
        tot1, idle1 = cpu_times()
        busy = 1.0 - (idle1 - idle0) / max(1, tot1 - tot0)
        tot0, idle0 = tot1, idle1
        try:
            la = os.getloadavg()[0]
        except OSError:
            la = None
        SAMPLES.append({'t_s': round(time.time() - NB_T0), 'cpu_busy': round(busy, 3), 'load1': la})
        with LOCK:
            running = [j for j in JOBS if j.get('status') == 'running']
        tails = []
        for j in running:
            try:
                tl = [l for l in (JOBS_DIR / j['id'] / 'log.txt').read_text(encoding='utf-8', errors='replace').splitlines() if l.startswith('{')][-1:]
            except OSError:
                tl = []
            tails.append('%s g%s %s' % (j['id'], j.get('gpu'), tl[0] if tl else '(starting)'))
        print('PROGRESS t=%.0fs cpu_busy=%.2f load1=%s | %s' % (time.time() - NB_T0, busy, la, ' | '.join(tails)), flush=True)
for th in threads:
    th.join()
print('R7OOF_WORKERS_DONE seconds %.0f' % (time.time() - NB_T0), flush=True)
# ---- optional, after the workers (so it never competes with them): cold read speed of one train study's DICOM bytes
if C.get('io_bench') and time.time() < NB_T0 + float(C['io_bench_before_min']) * 60:
    cand = [u for u in TRAIN_IDS if u not in set(TODO) and u not in set(PROOF)]
    u_ = cand[int(hashlib.md5(b'io').hexdigest()[:8], 16) % len(cand)]
    t_ = time.perf_counter()
    nb_, nf_ = 0, 0
    for p_ in sorted((DATA / 'train_series' / u_).glob('*/*.dcm')):
        nb_ += len(p_.read_bytes())
        nf_ += 1
    dt_ = time.perf_counter() - t_
    IO_BENCH = {'uid': u_, 'files': nf_, 'MB': round(nb_ / 2 ** 20, 1), 'seconds': round(dt_, 2), 'MB_per_s': round(nb_ / 2 ** 20 / max(dt_, 1e-6), 1)}
    print('IO_BENCH', json.dumps(IO_BENCH), flush=True)
