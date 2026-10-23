# fleet_worker.py -- ships INSIDE the checkpoint dataset next to fleet_common.py, fleet_model.py,
# fleet_manifest.json, parity_<sid>.npz (flat: `kaggle datasets create` skips sub-folders) and the checkpoints. Run once per GPU by the blend cell
# (CUDA_VISIBLE_DEVICES set there).   (round3/fleet CONSOLIDATED)
# Env: FLEET_DIR, FLEET_SHARD, FLEET_NSHARD, FLEET_BUDGET_S, FLEET_DEADLINE (absolute epoch s),
#      FLEET_OUT (npz path), RSNA_COMP_ROOT (optional, exported by the parent).
# Writes per-MODEL probabilities (M, n_shard, 12); a study row is written for ALL kept models or
# none, NaN for anything not run, so the merger can use one ensemble for every study.
# round3 fixes (FLEET_REVIEW.md, IDs W#):
#   W1  checkpoints come from fleet_manifest.json (draft glob f*_best.pt also served the raw AND ema
#       twin of every run); duplicate (fold, seed) refused
#   W2  ckpt contract == this fleet_common's CONTRACT (constants + pixel/tensor code sha), fp16-trained,
#       never bf16 (aastik S14); select_sha and train-decode probe logged MATCH/MISMATCH
#   W3  GPU probed with a real matmul; no silent CPU fallback (aastik L156-175) -> exit 3
#   W4  zero decoded views -> row stays NaN (draft: all-zero input scored as a valid row); NV saved
#   W5  all-or-none rows per study (draft: an exception at model k left models 0..k-1 written,
#       and the merger then silently dropped models k..M-1 for EVERY study)
#   W6  hard deadline (FLEET_DEADLINE) + atomic save + os._exit: a slow shard can no longer run past
#       the notebook limit; whatever is unfinished stays NaN and the cell's coverage gate decides
#   W7  C7 parity: shard 0 re-encodes the shipped parity train studies from DICOM in THIS image and
#       exits 4 if > 1% of pixels differ by > 8 levels (train decode != serve decode)
#   W8  oom_score_adj 1000: under host-RAM pressure the kernel kills a worker (-> parent restored),
#       not the notebook (-> no score); RSNA_COMP_ROOT honoured; channels_last as in training
# fleet2 patch: serves family 1 (KneeFleetNet, ckpt args without model_mod: exactly the round3 path) AND family 2
# (args model_mod == 'fleet_model_f2': KneeFleetNetF2, frozen BN, norm restored from the state_dict).
#   W-F2a model_sha checked per family: fleet_model.py source (family 1, unchanged) or fleet_model_f2.py + fleet_model.py
#   W-F2b channels_last only when the member trained with it; unknown model_mod refused
# K2 patch (scratchpad/k2, additive; predictions unchanged, proven by k2/tests/smoke_k2.py worker parity):
#   K2a decode wall-seconds per study timed in prep() -> STATS['decode_s'] (summed; the cell divides by n)
#   K2b FLEET_MICROBENCH=R (set by the K2 cell only on the <= 3-study commit run): after the loop, R timed
#       forwards per kept member on the last decoded study -> 'bench' {member: ms/study} in the npz + log
#   K2c 'loop_min' (main loop only, before the bench) saved next to 'minutes'
import os, sys, glob, time, json
try:
    open('/proc/self/oom_score_adj', 'w').write('1000')                      # W8
except Exception:
    pass
import numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor

FLEET_DIR = os.environ['FLEET_DIR']
sys.path.insert(0, FLEET_DIR)
import fleet_common as fc                      # the SAME file the trainer used (contract-checked below)
from fleet_model import KneeFleetNet

SHARD, NSHARD = int(os.environ.get('FLEET_SHARD', 0)), int(os.environ.get('FLEET_NSHARD', 1))
BUDGET = float(os.environ.get('FLEET_BUDGET_S', 3 * 3600))
DEADLINE = float(os.environ.get('FLEET_DEADLINE', 'inf'))
OUT = os.environ['FLEET_OUT']
T0 = time.time()
TAG = f'[fleet {SHARD}]'

# direct competition paths -- never a recursive glob of /kaggle/input (aastik S57: 360 s)
ROOT = None
for c in (os.environ.get('RSNA_COMP_ROOT', ''), '/kaggle/input/competitions/rsna-knee-abnormality-detection',
          '/kaggle/input/rsna-knee-abnormality-detection'):
    if c and os.path.isfile(c + '/test.csv') and os.path.isfile(c + '/test_series.csv') and os.path.isdir(c + '/test_series'):
        ROOT = c; break
assert ROOT, 'competition root not found'
test = pd.read_csv(f'{ROOT}/test.csv', dtype={'StudyInstanceUID': str})
ts = pd.read_csv(f'{ROOT}/test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
sids = test.StudyInstanceUID.tolist()[SHARD::NSHARD]
by = {s: g for s, g in ts.groupby('StudyInstanceUID')}

# W3: never serve the fleet on CPU (a P100 / no-kernel-image card reports is_available() == True)
dev = 'cpu'
try:
    assert torch.cuda.is_available()
    float((torch.randn(8, 8, device='cuda') @ torch.randn(8, 8, device='cuda')).sum())
    dev = 'cuda'
except Exception as e:
    print(f'{TAG} GPU unusable ({type(e).__name__}: {str(e)[:80]})', flush=True)
    if os.environ.get('FLEET_ALLOW_CPU') != '1':
        sys.exit(3)

# W1: explicit member list
man = json.load(open(f'{FLEET_DIR}/fleet_manifest.json'))
cks = [f'{FLEET_DIR}/{n}' for n in man['checkpoints']]
assert cks and all(os.path.isfile(p) for p in cks), ('manifest names a missing ckpt', man['checkpoints'])
fs_key = ['_'.join(os.path.basename(p).split('_')[:2]) for p in cks]          # f{fold}_s{seed}
assert len(fs_key) == len(set(fs_key)), f'duplicate fold/seed members (raw AND ema?): {sorted(fs_key)}'
# fold-interleave so a time-valve prefix stays fold-balanced (aastik S16)
byf = {}
for p in cks:
    byf.setdefault(os.path.basename(p).split('_')[0], []).append(p)
order = [q for i in range(max(map(len, byf.values()))) for f in sorted(byf) for q in byf[f][i:i + 1]]
def _model_sha(mod='fleet_model'):
    import hashlib, inspect, fleet_model as _fm
    src = chr(10).join(inspect.getsource(_fm).splitlines())
    if mod == 'fleet_model_f2':                                              # W-F2a
        import fleet_model_f2 as _f2
        src = chr(10).join(inspect.getsource(_f2).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]
models, probes = [], []
for p in order:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck['args']
    assert a.get('contract') == fc.CONTRACT, (p, 'ckpt contract', a.get('contract'), '!= fleet_common', fc.CONTRACT)  # W2
    assert a.get('amp') == 'fp16' and not a.get('bf16', False), (p, 'trained in', a.get('amp'), 'bf16', a.get('bf16'))
    mod = a.get('model_mod', 'fleet_model')
    assert mod in ('fleet_model', 'fleet_model_f2'), (p, 'unknown model_mod', mod)                     # W-F2b
    assert a.get('model_sha') in (None, _model_sha(mod)), (p, f'{mod}.py changed since training', a.get('model_sha'), _model_sha(mod))  # N-B
    if a.get('select_sha') != fc.SELECT_SHA:
        print(f'{TAG} NOTE {os.path.basename(p)} select_sha {a.get("select_sha")} != {fc.SELECT_SHA} '
              f'(series selection hardened after training; train choices unchanged by design)', flush=True)
    kw = dict(backbone=a['backbone'], n_views=a['n_views'], n_slices=a['n_slices'], dim=a['dim'],
              n_contrast=a['n_contrast'], contrast_emb=a['contrast_emb'], pretrained=False, drop_path=0.0, chunk=48)
    if mod == 'fleet_model_f2':
        from fleet_model_f2 import KneeFleetNetF2
        m = KneeFleetNetF2(freeze_bn=a['freeze_bn'], channels_last=a.get('channels_last', True), **kw)
    else:
        m = KneeFleetNet(**kw)
    miss, unexp = m.load_state_dict(ck['model'], strict=False)
    assert not miss and not unexp, (p, miss[:3], unexp[:3])
    m = m.eval().to(dev)
    if a.get('channels_last', True):                                         # W-F2b (family 1: always)
        m = m.to(memory_format=torch.channels_last)
    models.append(m)                                                         # fp32 weights + fp16 autocast
    probes.append(a.get('decode_probe', {}))
    del ck
print(f'{TAG} members: ' + ', '.join(f'{os.path.basename(p)}[{type(m).__name__}]' for p, m in zip(order, models)), flush=True)
print(f'{TAG} {len(models)} models, {len(sids)} studies, dev {dev}, root {ROOT}, jpeg_fp {fc.jpeg_fingerprint()} '
      f'| numpy {np.__version__} torch {torch.__version__}', flush=True)

# W2/W7: train-decode probe (all shards, cheap) + DICOM re-encode parity (shard 0 only)
par = dict(decode_match=0, decode_mismatch=0, checked=0)
pfiles = sorted(glob.glob(f'{FLEET_DIR}/parity_*.npz'))
for f in pfiles:
    psid = os.path.basename(f)[len('parity_'):-4]
    with np.load(f) as z:
        dsha = fc.decode_sha(z)
        for pr_ in probes:
            if psid in pr_:
                par['decode_match' if pr_[psid] == dsha else 'decode_mismatch'] += 1
if SHARD == 0 and pfiles:
    if os.path.isfile(f'{ROOT}/train_series.csv') and os.path.isdir(f'{ROOT}/train_series'):
        trs = pd.read_csv(f'{ROOT}/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        n = g8 = d = mx = 0; mism = []
        for f in pfiles:
            psid = os.path.basename(f)[len('parity_'):-4]
            with np.load(f) as z:
                r = fc.parity_check(z, trs[trs.StudyInstanceUID == psid], f'{ROOT}/train_series', psid)
            n += r['n_pix']; g8 += r['frac_gt8'] * r['n_pix']; d += r['frac_diff'] * r['n_pix']
            mx = max(mx, r['maxdiff']); mism += [f'{psid[-6:]}:{x}' for x in r['mismatch']]
        par.update(checked=len(pfiles), frac_diff=d / max(n, 1), frac_gt8=g8 / max(n, 1), maxdiff=mx, mismatch=mism)
        print(f'{TAG} PARITY re-encode vs stored train bytes: {par}', flush=True)
        if mism or n == 0 or par['frac_gt8'] > 0.01:
            print(f'{TAG} PARITY FAIL -> exit 4 (parent will be restored)', flush=True)
            sys.stdout.flush(); os._exit(4)
    else:
        print(f'{TAG} PARITY re-encode skipped: no train_series in {ROOT}', flush=True)
print(f'{TAG} decode probe vs ckpt (train-side decode): match {par["decode_match"]} '
      f'mismatch {par["decode_mismatch"]}', flush=True)

M = len(models)
P = np.full((M, len(sids), 12), np.nan, np.float32)
NV = np.full(len(sids), -1, np.int64)        # decoded views per study (-1 = not reached / failed)
STATS = {}


def prep(i):
    stats = {}
    t_dec = time.perf_counter()                                              # K2a
    try:
        sid = sids[i]
        return i, fc.build_study(by.get(sid, ts.iloc[0:0]), f'{ROOT}/test_series', sid, stats=stats), stats
    except Exception:
        return i, None, stats
    finally:
        stats['decode_s'] = time.perf_counter() - t_dec                      # K2a (same dict object as returned)


n_keep, n_done, n_fail, n_blank, stopped = M, 0, 0, 0, False
_BENCH_IN = None                                                             # K2b last decoded study (device tensors)
pool = ThreadPoolExecutor(4)
CH = 32
futs = [pool.submit(prep, i) for i in range(min(CH, len(sids)))]
nxt = min(CH, len(sids))
with torch.no_grad():
    while futs:
        if time.time() > DEADLINE:                                              # W6 hard stop
            stopped = True
            print(f'{TAG} DEADLINE reached at {n_done}/{len(sids)} -> stop, rest stays NaN', flush=True)
            break
        cur, futs = futs, []
        if nxt < len(sids):
            futs = [pool.submit(prep, i) for i in range(nxt, min(nxt + CH, len(sids)))]
            nxt += CH
        for f in cur:
            i, st, stats = f.result()
            n_done += 1
            for k, v in stats.items():
                STATS[k] = STATS.get(k, 0) + v
            if st is None:
                n_fail += 1; continue
            try:
                vs, pr, ct = [], [], []
                for v in fc.VIEWS:
                    g = st[v]
                    if g is None:
                        vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
                    else:
                        vs.append(fc.sample_triplets(g[0])); pr.append(True); ct.append(g[1])
                NV[i] = int(sum(pr))
                if NV[i] == 0:                                                  # W4
                    n_blank += 1; continue
                x = torch.from_numpy(np.stack(vs)[None]).to(dev)
                pr_t = torch.tensor([pr], device=dev); ct_t = torch.tensor([ct], device=dev)
                _BENCH_IN = (x, pr_t, ct_t)                                     # K2b
                row = np.empty((n_keep, 12), np.float32)
                with torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                    for k in range(n_keep):
                        row[k] = torch.sigmoid(models[k](x, pr_t, ct_t).float()).cpu().numpy()[0]
                P[:n_keep, i] = row                                                 # W5 all kept models or none
            except Exception as e:
                n_fail += 1; print(f'{TAG} fail', i, str(e)[:120], flush=True)
        if n_done >= 64 and n_keep == M:
            proj = (time.time() - T0) / n_done * len(sids)
            if proj > BUDGET:
                n_keep = max(1, int(M * BUDGET / proj * 0.9))
                P[n_keep:] = np.nan          # dropped models are blanked for EVERY study
                print(f'{TAG} projected {proj/3600:.2f} h > budget -> keep {n_keep}/{M}', flush=True)
        if n_done % 128 < CH:
            print(f'{TAG} {n_done}/{len(sids)} fail {n_fail} blank {n_blank} {(time.time()-T0)/60:.1f} min', flush=True)
pool.shutdown(wait=False, cancel_futures=True)
LOOP_MIN = (time.time() - T0) / 60                                           # K2c
BENCH = {}
MB = int(os.environ.get('FLEET_MICROBENCH', '0') or 0)                       # K2b commit-only microbench
if MB > 0 and _BENCH_IN is not None and time.time() < DEADLINE:
    try:
        xb, pb, cb = _BENCH_IN
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
            for k in range(n_keep):
                models[k](xb, pb, cb)                                        # warm-up (cudnn algo pick)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                tb = time.perf_counter()
                for _ in range(MB):
                    models[k](xb, pb, cb)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                BENCH[os.path.basename(order[k])] = round((time.perf_counter() - tb) / MB * 1000.0, 2)
        print(f'{TAG} MICROBENCH {MB} forwards/member on {dev}: ms/member/study {BENCH}', flush=True)
    except Exception as e:
        BENCH = {'error': repr(e)[:200]}
        print(f'{TAG} MICROBENCH failed (ignored): {e!r}', flush=True)
tmp = OUT[:-4] + '.tmp.npz' if OUT.endswith('.npz') else OUT + '.tmp.npz'   # np.savez appends .npz
np.savez(tmp, sids=np.array(sids), P=P, NV=NV, names=np.array([os.path.basename(p) for p in order]),
         n_keep=np.int64(n_keep), n_fail=np.int64(n_fail), n_blank=np.int64(n_blank),
         hard_stopped=np.bool_(stopped), minutes=np.float64((time.time() - T0) / 60),
         stats=np.array(json.dumps(STATS)), parity=np.array(json.dumps(par)),
         bench=np.array(json.dumps(BENCH)), loop_min=np.float64(LOOP_MIN))                    # K2b/K2c
os.replace(tmp, OUT)                                                            # W6 atomic
vps = NV[NV >= 0].mean() if (NV >= 0).any() else 0.0
print(f'{TAG} done {(time.time()-T0)/60:.1f} min, keep {n_keep}/{M}, fail {n_fail}, blank {n_blank}, '
      f'hard-stopped {stopped}, views/study {vps:.2f}, decode {STATS.get("decode_s", 0.0) / max(n_done, 1):.2f} s/study '
      f'(per decode thread, 4 threads in parallel), selection {STATS}', flush=True)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)                                   # do not wait on a decode thread stuck in a bad file
