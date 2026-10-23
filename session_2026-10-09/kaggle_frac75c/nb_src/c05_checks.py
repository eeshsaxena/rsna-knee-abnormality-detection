# [step 4] parity before any GPU work: (a) store/tensor contract, select_sha, jpeg_fp, model_sha == the box ckpt;
# (b) the box ckpt's decode probe (4 studies decoded in THIS cv2 build) == the box decode; (c) the model built here
# (timm + bundled pretrained bytes) has the box state_dict keys/shapes/dtypes and the pretrained weights loaded;
# (d) fold counts == the box arithmetic; (e) one-thread CPU cost of the heavy train path vs the box.
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import inspect
import fleet_common as fc
import fleet_model as _fm, fleet_model_ttr as _ftt
_a = BOX['ckpt_args']
_msha = hashlib.sha256((chr(10).join(inspect.getsource(_ftt).splitlines()) + chr(10) +
                        chr(10).join(inspect.getsource(_fm).splitlines())).encode()).hexdigest()[:16]
PAR = dict(contract=json.loads(json.dumps(fc.CONTRACT)), select_sha=fc.SELECT_SHA, jpeg_fp=fc.jpeg_fingerprint(),
           model_sha=_msha)
_bad = []
if PAR['contract'] != _a['contract']:
    _bad.append(f'contract {PAR["contract"]} != box {_a["contract"]}')
for _k in ('select_sha', 'jpeg_fp', 'model_sha'):
    if PAR[_k] != _a[_k]:
        _bad.append(f'{_k} {PAR[_k]} != box {_a[_k]}')
_dp = {}
for _s, _h in _a['decode_probe'].items():
    with np.load(f'{STORE}/{_s}.npz') as _z:
        fc.check_store_contract(_z, _s)
        _dp[_s] = fc.decode_sha(_z)
    if _dp[_s] != _h:
        _bad.append(f'decode probe {_s[-12:]}: {_dp[_s]} != box {_h}')
PAR['decode_probe_match'] = sum(_dp[s] == h for s, h in _a['decode_probe'].items())
if _bad:
    REC['parity'] = dict(PAR, problems=_bad)
    abort('parity with the box failed: ' + ' | '.join(_bad))
log(f'parity OK: pixel_sha {PAR["contract"]["pixel_sha"]} tensor_sha {PAR["contract"]["tensor_sha"]} select_sha '
    f'{PAR["select_sha"]} jpeg_fp {PAR["jpeg_fp"]} model_sha {PAR["model_sha"]} | decode probe '
    f'{PAR["decode_probe_match"]}/{len(_a["decode_probe"])}')

_pre = install_hf_cache(DS, EXP)
_mc = None
for _off in ('1', '0'):
    _r = subprocess.run([sys.executable, f'{CODE}/kt_modelcheck.py', CODE, f'{DS}/box_facts.json', _pre],
                        capture_output=True, text=True, cwd=CODE,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HOME=HF_HOME, HF_HUB_OFFLINE=_off,
                                 OMP_NUM_THREADS='2'))
    _line = [l for l in _r.stdout.splitlines() if l.startswith('KTMODELCHECK ')]
    if _r.returncode == 0 and _line:
        _mc = json.loads(_line[-1][len('KTMODELCHECK '):])
        HF_OFFLINE[0] = _off
        break
    warn(f'model check with HF_HUB_OFFLINE={_off} failed (rc {_r.returncode}): {_r.stderr[-1500:]}')
if _mc is None:
    abort('could not build KneeFleetNetTTR with pretrained weights')
if HF_OFFLINE[0] != '1':
    warn('pretrained weights came from the HF hub download, not the bundled box bytes (equality checked below)')
PAR['model_check'] = _mc
PAR['hf_offline'] = HF_OFFLINE[0]
if not (_mc['keys_match'] and _mc['pretrained_equal']):
    REC['parity'] = PAR
    abort(f'model check failed: {_mc}')
log(f'model OK: {_mc["n_keys"]} state_dict keys/shapes/dtypes == box ckpt; {_mc["cnn_keys"]} backbone tensors == the '
    f'box pretrained file (timm {_mc["timm"]}, torch {_mc["torch"]}, offline {HF_OFFLINE[0]})')

COUNTS = {}
for F in FOLDS:
    COUNTS[F] = expected_counts(LBL, STORE, F)
    _bx = BOX['expected_counts'][str(F)]
    if COUNTS[F] != _bx:
        abort(f'fold {F} counts {COUNTS[F]} != box {_bx}')
    log(f'fold {F}: {COUNTS[F]}')
PAR['counts'] = COUNTS

import cv2 as _cv2
from fleet_aug import heavy_study
_cv2.setNumThreads(0)
_nt = torch.get_num_threads()
torch.set_num_threads(1)
_ids = sorted(f[:-4] for f in os.listdir(STORE) if f.endswith('.npz'))
_bench = _ids[200:216] if len(_ids) >= 216 else _ids[:16]
np.random.seed(0)
_t = time.time()
for _s in _bench:
    with np.load(f'{STORE}/{_s}.npz') as _z:
        fc.check_store_contract(_z, _s)
        heavy_study(_z, _s, 16, [p for p, _ in fc.VIEWS])
_th = (time.time() - _t) / len(_bench)
_t = time.time()
for _s in _bench:
    with np.load(f'{STORE}/{_s}.npz') as _z:
        for _j in range(len(fc.VIEWS)):
            if f'v{_j}_blob' in _z.files:
                fc.sample_triplets(fc.decode_stack(_z[f'v{_j}_blob'], _z[f'v{_j}_offs']), fc.N_SLICES, 0.0)
_te = (time.time() - _t) / len(_bench)
torch.set_num_threads(_nt)
_bb = BOX['cpu_bench_box']
BENCH = dict(n=len(_bench), heavy_s_per_study=round(_th, 4), eval_s_per_study=round(_te, 4),
             box_heavy_s_per_study=_bb['heavy_s_per_study'], box_eval_s_per_study=_bb['eval_s_per_study'],
             cpu_bound_s_it_floor=round(2 * _th * len(FOLDS) / max(1, os.cpu_count()), 3))
PAR['cpu_bench'] = BENCH
REC['parity'] = PAR
stage('checks', ok=True)
log(f'CPU heavy train path {_th:.3f} s/study/thread (box {_bb["heavy_s_per_study"]}), eval path {_te:.3f} '
    f'(box {_bb["eval_s_per_study"]}); with {os.cpu_count()} vCPUs and {len(FOLDS)} trainers at bs 2 the decode-only '
    f'floor is ~{BENCH["cpu_bound_s_it_floor"]} s/it per trainer')
