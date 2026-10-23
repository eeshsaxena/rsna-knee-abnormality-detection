# parity_unit.py -- FB-1 parity, grad_ckpt 1 vs 0 on 2 REAL store studies (heavy aug drawn ONCE, identical input to
# both arms), same seed, bs 2, accum 1, the fleet_train.py step code verbatim (fp16 autocast, GradScaler, unscale +
# clip 1.0, AdamW 2 groups, LambdaLR, EMA AveragedModel 0.9995), optimizer steps until 4 APPLIED steps (GradScaler skips
# are counted and reported). LR warmup is set to 1 step so the 4 steps move the weights at the full recipe LR (the
# schedule is not what is under test; the model math is). Reports per-step loss / grad-norm diffs and the max abs diff
# of the final raw + EMA weights, plus the weight update size for scale. Saves both final raw models in the trainer's
# ckpt format for the worker strict-load check.
#   --mode bench: cudnn.benchmark True (as the trainer) -> expect agreement to fp16 tolerance
#   --mode det:   cudnn deterministic + benchmark False + CUBLAS_WORKSPACE_CONFIG -> expect bitwise or near
import os, sys, json, math, time, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import fbcommon as fb

ap = argparse.ArgumentParser()
ap.add_argument('--code', default=os.path.expanduser('~/fast_build/code2'))
ap.add_argument('--store', default=os.path.expanduser('~/fleet_store'))
ap.add_argument('--labels', default=os.path.expanduser('~/rsna_train/labels_mix50_logit.csv'))
ap.add_argument('--folds', default=os.path.expanduser('~/rsna_train/folds5.csv'))
ap.add_argument('--exclude', default=os.path.expanduser('~/rsna_train/gold74_eval_set.csv'))
ap.add_argument('--j1', default=os.path.expanduser('~/fleet_runs/f0_s1043_ttr_mix50/f0_s1043_raw_ttr224-320x24_cv8835_best.pt'))
ap.add_argument('--mode', choices=('bench', 'det'), required=True)
ap.add_argument('--cap_gib', type=float, default=9.8); ap.add_argument('--need_mib', type=int, default=12000)
ap.add_argument('--applied', type=int, default=4); ap.add_argument('--max_steps', type=int, default=10)
ap.add_argument('--out', required=True)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
fb.gpu_gate(A.need_mib)
import numpy as np, pandas as pd, torch, torch.nn.functional as F
fb.cap(A.cap_gib)
if A.mode == 'det':
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'det mode needs CUBLAS_WORKSPACE_CONFIG=:4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
else:
    torch.backends.cudnn.benchmark = True
sys.path.insert(0, A.code)
import fleet_common as fc
from fleet_aug import heavy_study
from fleet_model_ttr import KneeFleetNetTTR
dev = 'cuda'
PLANES = [p for p, _ in fc.VIEWS]
SEED = 1043

# ---- 2 real studies of fold-2 TRAIN (fold != 2, not held), all 4 views present, heavy aug drawn once
lab = pd.read_csv(A.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
excl = set(pd.read_csv(A.exclude, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
cand = [s for s in lab.index if s in folds.index and folds[s] >= 0 and folds[s] != 2 and s not in excl
        and os.path.exists(f'{A.store}/{s}.npz')]
pick = []
for s in cand:
    with np.load(f'{A.store}/{s}.npz') as z:
        if all(f'v{j}_blob' in z.files for j in range(len(fc.VIEWS))):
            pick.append(s)
    if len(pick) == 2:
        break
np.random.seed(SEED); random.seed(SEED)
items = []
for sid in pick:
    with np.load(f'{A.store}/{sid}.npz') as z:
        fc.check_store_contract(z, sid)
        x, pr, ct, pos, _ = heavy_study(z, sid, 16, PLANES)
    y = lab.loc[sid, fb.TARGETS].to_numpy(np.float32)
    pos = torch.from_numpy(pos)
    items.append((x, torch.tensor(pr), torch.tensor(ct), pos[None].expand(len(fc.VIEWS), -1).clone(), torch.from_numpy(y)))
batch = [torch.stack([it[k] if torch.is_tensor(it[k]) else torch.as_tensor(it[k]) for it in items]) for k in range(5)]
print('studies', pick, '| x', tuple(batch[0].shape), batch[0].dtype, '| pos', batch[3][0, 0].tolist(), flush=True)


def run(gc):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = KneeFleetNetTTR(ttr_size=224, backbone='convnext_tiny.fb_in22k_ft_in1k', dim=256, contrast_emb=True,
                            n_views=len(fc.VIEWS), n_slices=fc.N_SLICES, n_contrast=fc.N_CONTRAST,
                            pretrained=True, drop_path=0.2, dropout=0.3, grad_ckpt=bool(gc)).to(dev)
    model = model.to(memory_format=torch.channels_last)
    ckflags = sorted({bool(getattr(m, 'grad_checkpointing', False)) for m in model.cnn.modules() if hasattr(m, 'grad_checkpointing')})
    init = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9995))
    bb = [p for n, p in model.named_parameters() if n.startswith('cnn.')]
    hd = [p for n, p in model.named_parameters() if not n.startswith('cnn.')]
    opt = torch.optim.AdamW([{'params': bb, 'lr': 1e-4}, {'params': hd, 'lr': 1e-3}], weight_decay=0.05)
    steps, warm = 8, 1
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    torch.cuda.reset_peak_memory_stats()
    model.train(); opt.zero_grad(set_to_none=True)
    rec, applied, it = [], 0, 0
    t0 = time.time()
    while applied < A.applied and it < A.max_steps:
        x, pr, ct, pos, y = (t.to(dev, non_blocking=True) for t in batch)
        with torch.autocast('cuda', dtype=torch.float16, enabled=True):
            logit = model(x, pr, ct, pos)
        msk = torch.isfinite(y).float(); yy = torch.nan_to_num(y)
        loss = (F.binary_cross_entropy_with_logits(logit.float(), yy, reduction='none') * msk).sum() / msk.sum().clamp(min=1)
        scaler.scale(loss / 1).backward()
        s_before = scaler.get_scale()
        scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        ema.update_parameters(model)
        ok = scaler.get_scale() >= s_before
        applied += int(ok)
        rec.append(dict(step=it, loss=float(loss.item()), grad_norm=float(gn.item()), scale=s_before, applied=bool(ok),
                        logit0=[round(float(v), 6) for v in logit[0, :4].float().tolist()]))
        it += 1
    torch.cuda.synchronize()
    out = dict(gc=gc, ckpt_flags=ckflags, steps=rec, seconds=round(time.time() - t0, 2),
               peak_alloc_MiB=round(torch.cuda.max_memory_allocated() / 2 ** 20),
               peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2 ** 20))
    raw = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
    em = {k: v.detach().float().cpu().clone() for k, v in ema.module.state_dict().items()}
    sd_save = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model, ema, opt, scaler, sched, bb, hd
    torch.cuda.empty_cache()
    return out, init, raw, em, sd_save


def maxdiff(a, b):
    assert list(a) == list(b), 'state_dict key order/set differs'
    md, where = 0.0, ''
    for k in a:
        d = float((a[k] - b[k]).abs().max()) if a[k].numel() else 0.0
        if d > md:
            md, where = d, k
    return md, where


res = {'mode': A.mode, 'studies': pick, 'torch': torch.__version__}
runs = {}
for gc in (1, 0):
    o, init, raw, em, sd = run(gc)
    runs[gc] = (init, raw, em, sd)
    res[f'gc{gc}'] = o
    print(json.dumps(o), flush=True)
i1, r1, e1, s1 = runs[1]; i0, r0, e0, s0 = runs[0]
res['init_maxdiff'] = maxdiff(i1, i0)
res['raw_maxdiff'] = maxdiff(r1, r0)
res['ema_maxdiff'] = maxdiff(e1, e0)
res['update_max_abs'] = max(float((r1[k] - i1[k]).abs().max()) for k in r1 if r1[k].numel())
res['raw_bitwise_equal'] = all(torch.equal(r1[k], r0[k]) for k in r1)
res['ema_bitwise_equal'] = all(torch.equal(e1[k], e0[k]) for k in e1)
st1, st0 = res['gc1']['steps'], res['gc0']['steps']
res['loss_maxdiff'] = max(abs(a['loss'] - b['loss']) for a, b in zip(st1, st0)) if len(st1) == len(st0) else None
res['gradnorm_max_reldiff'] = (max(abs(a['grad_norm'] - b['grad_norm']) / max(1e-12, abs(a['grad_norm'])) for a, b in zip(st1, st0))
                               if len(st1) == len(st0) else None)
res['same_applied_pattern'] = [a['applied'] for a in st1] == [b['applied'] for b in st0]
res['state_dict_keys_equal'] = list(s1) == list(s0)
# trainer-format checkpoints (args = J1's args with this fold; gc0 gains the informational key) for the worker check
j1a = torch.load(A.j1, map_location='cpu', weights_only=False)['args']
for gc, sd in ((1, s1), (0, s0)):
    a = dict(j1a); a['fold'] = 2
    if gc == 0:
        a['grad_ckpt'] = False
    torch.save({'model': sd, 'args': a, 'cv': float('nan'), 'epoch': 0},
               f'{A.out}/f2_s1043_raw_ttr224-320x24_gc{gc}{A.mode}_best.pt')
json.dump(res, open(f'{A.out}/parity_{A.mode}.json', 'w'), indent=1)
print('PARITY ' + json.dumps({k: res[k] for k in ('mode', 'init_maxdiff', 'raw_maxdiff', 'ema_maxdiff', 'update_max_abs',
                                                   'raw_bitwise_equal', 'ema_bitwise_equal', 'loss_maxdiff',
                                                   'gradnorm_max_reldiff', 'same_applied_pattern', 'state_dict_keys_equal')}),
      flush=True)
