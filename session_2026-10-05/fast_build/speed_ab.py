# speed_ab.py -- FB-1 speed A/B: grad_ckpt 1 (A) vs 0 (B) on the REAL TTR mix50 recipe step (bs 2, accum 4, heavy aug,
# train_slices 16, drop_path .2, dropout .3, EMA .9995, fp16 autocast + GradScaler, AdamW 2 groups, clip 1.0, LambdaLR
# with the fold-run schedule length), data from the store through the trainer's heavy Dataset path (10 workers,
# pin_memory, persistent). ONE model; A/B toggles model.cnn.set_grad_checkpointing (exactly what the constructor flag
# does). Segments = 30 optimizer steps (120 micro-steps) after an untimed 2-step re-warm, interleaved A B / B A / A B.
#   source 'cached': 16 real batches prefetched once and cycled (GPU step time only)
#   source 'loader': live DataLoader iterator (end to end; data-wait time recorded)
# Then an eval A/B (no_grad fp16 forward, eval shape B=2 x 4 views x 24 slices) since the trainer's per-epoch eval also
# runs through the checkpoint flag. Peak CUDA memory per segment (allocator) + nvidia-smi max (self / others / total).
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
ap.add_argument('--fold', type=int, default=2)
ap.add_argument('--cap_gib', type=float, default=9.8); ap.add_argument('--need_mib', type=int, default=12000)
ap.add_argument('--reps', type=int, default=3); ap.add_argument('--opt_steps', type=int, default=30)
ap.add_argument('--accum', type=int, default=4); ap.add_argument('--bs', type=int, default=2)
ap.add_argument('--workers', type=int, default=10)
ap.add_argument('--sources', default='cached,loader')
ap.add_argument('--eval_batches', type=int, default=20)
ap.add_argument('--log', required=True)
A = ap.parse_args()
fb.gpu_gate(A.need_mib)
import numpy as np, pandas as pd, torch, torch.nn.functional as F
fb.cap(A.cap_gib)
torch.backends.cudnn.benchmark = True
sys.path.insert(0, A.code)
import fleet_common as fc
from fleet_aug import heavy_study
from fleet_model_ttr import KneeFleetNetTTR
dev = 'cuda'
PLANES = [p for p, _ in fc.VIEWS]
torch.manual_seed(1043); np.random.seed(1043); random.seed(1043)
smp = fb.SmiSampler(1.0); smp.start()

lab = pd.read_csv(A.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
excl = set(pd.read_csv(A.exclude, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
held = set(folds.index[folds < 0]) | excl
ids = [s for s in lab.index if s in folds.index and os.path.exists(f'{A.store}/{s}.npz')]
tr_ids = [s for s in ids if s not in held and folds[s] != A.fold]
va_ids = [s for s in ids if s not in held and folds[s] == A.fold]
print(f'train {len(tr_ids)} val {len(va_ids)}', flush=True)


def _wif(_):
    try:
        import cv2; cv2.setNumThreads(0)
    except Exception:
        pass


class DS(torch.utils.data.Dataset):                       # = fleet_train.DS._getitem_heavy (train) / default eval path
    def __init__(self, ids, train):
        self.ids, self.train = ids, train
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        sid = self.ids[i]
        if self.train:
            with np.load(f'{A.store}/{sid}.npz') as z:
                fc.check_store_contract(z, sid)
                x, pr, ct, pos, _ = heavy_study(z, sid, 16, PLANES)
            y = lab.loc[sid, fb.TARGETS].to_numpy(np.float32)
            pos = torch.from_numpy(pos)
            return x, torch.tensor(pr), torch.tensor(ct), pos[None].expand(len(fc.VIEWS), -1).clone(), torch.from_numpy(y)
        vs, pr, ct = [], [], []
        with np.load(f'{A.store}/{sid}.npz') as z:
            fc.check_store_contract(z, sid)
            for j in range(len(fc.VIEWS)):
                if f'v{j}_blob' in z.files:
                    st = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
                    vs.append(fc.sample_triplets(st, fc.N_SLICES, 0.0)); pr.append(True); ct.append(int(z[f'v{j}_ctr']))
                else:
                    vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
        x = torch.from_numpy(np.stack(vs))
        pos = torch.arange(fc.N_SLICES)
        y = lab.loc[sid, fb.TARGETS].to_numpy(np.float32)
        return x, torch.tensor(pr), torch.tensor(ct), pos[None].expand(len(fc.VIEWS), -1).clone(), torch.from_numpy(y)


tr_dl = torch.utils.data.DataLoader(DS(tr_ids, True), batch_size=A.bs, shuffle=True, drop_last=True,
                                    num_workers=A.workers, pin_memory=True, worker_init_fn=_wif, persistent_workers=True)
model = KneeFleetNetTTR(ttr_size=224, backbone='convnext_tiny.fb_in22k_ft_in1k', dim=256, contrast_emb=True,
                        n_views=len(fc.VIEWS), n_slices=fc.N_SLICES, n_contrast=fc.N_CONTRAST,
                        pretrained=True, drop_path=0.2, dropout=0.3, grad_ckpt=True).to(dev)
model = model.to(memory_format=torch.channels_last)
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9995))
bb = [p for n, p in model.named_parameters() if n.startswith('cnn.')]
hd = [p for n, p in model.named_parameters() if not n.startswith('cnn.')]
opt = torch.optim.AdamW([{'params': bb, 'lr': 1e-4}, {'params': hd, 'lr': 1e-3}], weight_decay=0.05)
L = len(tr_dl)
steps = max(1, 24 * L // A.accum); warm = max(1, L // A.accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
scaler = torch.amp.GradScaler('cuda', enabled=True)
it_glob = 0
live = iter(tr_dl)


def next_live():
    global live
    try:
        return next(live)
    except StopIteration:
        live = iter(tr_dl)
        return next(live)


t0 = time.time()
cache = [next_live() for _ in range(16)]
print(f'prefetched 16 batches in {time.time() - t0:.1f}s', flush=True)


def micro(batch):                                        # fleet_train.py loop body, verbatim
    global it_glob
    x, pr, ct, pos, y = (t.to(dev, non_blocking=True) for t in batch)
    with torch.autocast('cuda', dtype=torch.float16, enabled=True):
        logit = model(x, pr, ct, pos)
    msk = torch.isfinite(y).float(); yy = torch.nan_to_num(y)
    loss = (F.binary_cross_entropy_with_logits(logit.float(), yy, reduction='none') * msk).sum() / msk.sum().clamp(min=1)
    scaler.scale(loss / A.accum).backward()
    if (it_glob + 1) % A.accum == 0:
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        ema.update_parameters(model)
    if it_glob % 200 == 0:
        _ = loss.item()
    it_glob += 1
    return loss


def set_gc(gc):
    model.cnn.set_grad_checkpointing(bool(gc))
    fl = {bool(getattr(m, 'grad_checkpointing', False)) for m in model.cnn.modules() if hasattr(m, 'grad_checkpointing')}
    assert fl == {bool(gc)}, fl


def segment(gc, source, rep):
    set_gc(gc); model.train()
    assert it_glob % A.accum == 0
    ci = 0
    for _ in range(2 * A.accum):                         # untimed re-warm (cudnn.benchmark + allocator for this mode)
        micro(cache[ci % 16] if source == 'cached' else next_live()); ci += 1
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); smp.reset()
    n = A.opt_steps * A.accum
    wait = 0.0
    t0 = time.time()
    for _ in range(n):
        if source == 'cached':
            b = cache[ci % 16]; ci += 1
        else:
            tw = time.time(); b = next_live(); wait += time.time() - tw
        micro(b)
    torch.cuda.synchronize()
    dt = time.time() - t0
    r = dict(kind='train', source=source, rep=rep, gc=gc, micro_steps=n, opt_steps=A.opt_steps, seconds=round(dt, 2),
             s_per_micro=round(dt / n, 4), s_per_opt=round(dt / A.opt_steps, 3), data_wait_s=round(wait, 2),
             peak_alloc_MiB=round(torch.cuda.max_memory_allocated() / 2 ** 20),
             peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2 ** 20),
             ts=time.strftime('%H:%M:%S'), **smp.snap())
    print('SEG ' + json.dumps(r), flush=True)
    with open(A.log, 'a') as f:
        f.write(json.dumps(r) + '\n')
    return r


res = []
for source in A.sources.split(','):
    for rep in range(A.reps):
        for gc in ((1, 0) if rep % 2 == 0 else (0, 1)):
            res.append(segment(gc, source, rep))

# eval A/B: the trainer's predict() (no_grad, fp16 autocast, pos=None, all 24 slices) on real val studies
ev_dl = torch.utils.data.DataLoader(DS(va_ids[:2 * A.eval_batches], False), batch_size=A.bs, shuffle=False,
                                    num_workers=4, pin_memory=True, worker_init_fn=_wif)
ev_cache = [b for b in ev_dl]
for rep in range(2):
    for gc in ((1, 0) if rep % 2 == 0 else (0, 1)):
        set_gc(gc)
        ema.module.cnn.set_grad_checkpointing(bool(gc))
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=True):
            for b in ev_cache[:3]:
                model(b[0].to(dev), b[1].to(dev), b[2].to(dev))
            torch.cuda.synchronize(); t0 = time.time()
            P = []
            for b in ev_cache:
                P.append(torch.sigmoid(model(b[0].to(dev), b[1].to(dev), b[2].to(dev)).float()).cpu().numpy())
            torch.cuda.synchronize(); dt = time.time() - t0
        r = dict(kind='eval', rep=rep, gc=gc, batches=len(ev_cache), seconds=round(dt, 2),
                 s_per_batch=round(dt / len(ev_cache), 4), ts=time.strftime('%H:%M:%S'))
        print('SEG ' + json.dumps(r), flush=True)
        with open(A.log, 'a') as f:
            f.write(json.dumps(r) + '\n')
        res.append(r)
model.train()


def ratio(kind, source=None):
    a = [r['seconds'] for r in res if r['kind'] == kind and r.get('source') == source and r['gc'] == 1]
    b = [r['seconds'] for r in res if r['kind'] == kind and r.get('source') == source and r['gc'] == 0]
    return dict(gc1_mean_s=round(float(np.mean(a)), 2), gc0_mean_s=round(float(np.mean(b)), 2),
                ratio_gc0_over_gc1=round(float(np.mean(b) / np.mean(a)), 4),
                per_rep_ratio=[round(y / x, 4) for x, y in zip(a, b)])


summ = {s: ratio('train', s) for s in A.sources.split(',')}
summ['eval'] = ratio('eval')
summ['gc0_peak_reserved_MiB'] = max(r['peak_reserved_MiB'] for r in res if r['kind'] == 'train' and r['gc'] == 0)
summ['gc0_peak_alloc_MiB'] = max(r['peak_alloc_MiB'] for r in res if r['kind'] == 'train' and r['gc'] == 0)
summ['gc1_peak_reserved_MiB'] = max(r['peak_reserved_MiB'] for r in res if r['kind'] == 'train' and r['gc'] == 1)
summ['gc0_smi_max_self_MiB'] = max(r['smi_max_self_MiB'] for r in res if r['kind'] == 'train' and r['gc'] == 0)
print('SUMMARY ' + json.dumps(summ), flush=True)
with open(A.log, 'a') as f:
    f.write(json.dumps(dict(kind='summary', **summ)) + '\n')
