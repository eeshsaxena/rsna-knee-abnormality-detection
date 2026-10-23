# train_raptor.py (box): our seed of nartaa's Raptor .949 recipe (Kaggle discussion 746792, training reply 10-13).
# Model / eval windowing come VERBATIM from raptor_ref.py (= the scored r384 source, HR7 cell 2): RaptorClassifier,
# _eval_centers. Recipe (746792): coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k, ImageNet init, native 384 FULL field, 12 random
# 3-slice windows per study per step, +-10 % intensity jitter per study (no flips / affine), batch 1 study x grad accumulation
# 16, AdamW wd .0588, LR backbone 4.01e-5 / head 1.95e-3, OneCycle 15 % warm-up over 16 epochs, grad-norm clip 3.0, no EMA,
# BCE-with-logits pos_weight (1-p)/p from RSNA prevalence (p clip .03-.7, weight clip 1-10), masked cells weight 0, the
# weighted cell losses SUMMED and divided by 12 x studies, gold58 per-epoch eval with 24 evenly spaced windows (full field),
# final = uniform average of the 3 best epochs by gold58 macro AUC incl. BN buffers (num_batches_tracked of the best).
# Deviations (logged): fp16 autocast + GradScaler on V100 (no bf16 tensor cores) with non-finite guards; our targets
# (labels_raptor_rsna_blendjev_sol.csv); OAI images derived from our 320 px store (build_oai_d96_approx.py).
# Outputs (--out DIR): train.log lines, ep{E}.pt (model fp32 state_dict + gold metrics), last.pt (resume state),
# swa.pt in N's dict format {'model','gold_auc','aucs','src','res','arch','lab','epoch','swa_over'} + summary.json.
import os, sys, time, json, math, argparse, hashlib, random
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--index', nargs='+', required=True, help='index csv(s): id,path,src,n_valid,mask')
ap.add_argument('--labels_rsna', required=True); ap.add_argument('--labels_oai', default='')
ap.add_argument('--gold', default='/home/training/rsna_train/gold74_eval_set.csv')
ap.add_argument('--out', required=True)
ap.add_argument('--arch', default='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'); ap.add_argument('--res', type=int, default=384)
ap.add_argument('--pretrained', type=int, default=1)
ap.add_argument('--epochs', type=int, default=16); ap.add_argument('--accum', type=int, default=16)
ap.add_argument('--windows', type=int, default=12); ap.add_argument('--eval_windows', type=int, default=24)
ap.add_argument('--lr_bb', type=float, default=4.01e-5); ap.add_argument('--lr_head', type=float, default=1.95e-3)
ap.add_argument('--wd', type=float, default=0.0588); ap.add_argument('--pct_start', type=float, default=0.15)
ap.add_argument('--clip', type=float, default=3.0); ap.add_argument('--jitter', type=float, default=0.10)
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--amp', default='fp16', choices=['fp16', 'bf16', 'off'])
ap.add_argument('--device', default='cuda'); ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--channels_last', type=int, default=1); ap.add_argument('--grad_ckpt', type=int, default=0)
ap.add_argument('--smoke_steps', type=int, default=0, help='stop the epoch loop after this many study-steps (smoke)')
ap.add_argument('--max_rows', type=int, default=0, help='dry run: use only this many train rows')
ap.add_argument('--max_gold', type=int, default=0, help='dry run: use only this many gold studies')
ap.add_argument('--top', type=int, default=3); ap.add_argument('--resume', type=int, default=1)
ap.add_argument('--nonfinite_abort', type=float, default=0.01)
ap.add_argument('--swa_only', type=int, default=0, help='skip training; SWA from the ep*.pt already in --out')
ap.add_argument('--init_scale', type=float, default=1024.0)
ap.add_argument('--forbid_oai', type=int, default=0, help='rental: fail loud if any OAI id / path / label file is in reach')
A = ap.parse_args()
import torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raptor_ref as R
LAB = R.LAB; D = 96
os.makedirs(A.out, exist_ok=True)
LOG = open(f'{A.out}/train.log', 'a')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed)
dev = torch.device(A.device)
P(f'=== train_raptor {time.strftime("%Y-%m-%d %H:%M:%S")} | args {json.dumps(vars(A))}')
P(f'code md5 {md5(__file__)} | raptor_ref md5 {md5(R.__file__)} | torch {torch.__version__} timm {timm.__version__}')

# ---------------- rows ----------------
idx = pd.concat([pd.read_csv(p, dtype={'id': str, 'mask': str}) for p in A.index], ignore_index=True)
assert idx.id.is_unique, 'duplicate ids across index files'
idx = idx.set_index('id')
if A.forbid_oai:                                   # RSNA-only rental (path b): OAI must never be in reach
    bad = [i for i in idx.index if 'oai' in str(i).lower()] + [p for p in idx.path if 'oai' in str(p).lower()]
    assert not bad and not A.labels_oai, f'FORBID_OAI: OAI ids / paths / labels present: {bad[:3]} {A.labels_oai!r}'
    assert not any('oai' in str(x).lower() for x in A.index + [A.labels_rsna]), 'FORBID_OAI: an OAI file name was passed'
gold = pd.read_csv(A.gold, dtype={'StudyInstanceUID': str})
gold = gold[gold.source == 'original_gold58'].set_index('StudyInstanceUID')[LAB].astype(float)
assert len(gold) == 58
lr = pd.read_csv(A.labels_rsna, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LAB].astype(float)
assert np.isfinite(lr.values).all()
rs_train = [s for s in lr.index if s not in gold.index and s in idx.index]
P(f'RSNA labels {len(lr)} md5 {md5(A.labels_rsna)} | gold58 excluded | RSNA train rows with a volume {len(rs_train)}')
rows = [(s, lr.loc[s].values.astype(np.float32)) for s in rs_train]
if A.labels_oai:
    lo = pd.read_csv(A.labels_oai, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[LAB].astype(float)
    oai = [s for s in lo.index if s in idx.index]
    rows += [(s, lo.loc[s].values.astype(np.float32)) for s in oai]
    fin = np.isfinite(lo.loc[oai].values)
    P(f'OAI labels {len(lo)} md5 {md5(A.labels_oai)} | with a volume {len(oai)} | finite cells per label '
      + ' '.join(f'{LAB[j][:6]}:{fin[:, j].sum()}' for j in range(12)))
if A.max_rows:
    rng0 = np.random.RandomState(A.seed); rows = [rows[i] for i in sorted(rng0.choice(len(rows), A.max_rows, replace=False))]
gold_ids = [s for s in gold.index if s in idx.index]
if A.max_gold:
    gold_ids = gold_ids[:A.max_gold]
P(f'train rows {len(rows)} | gold eval studies {len(gold_ids)}')
# pos_weight from the RSNA training rows only (mean soft target), p clip .03-.7, weight clip 1-10
p_rs = lr.loc[rs_train].values.mean(0)
pw = np.clip((1 - np.clip(p_rs, 0.03, 0.7)) / np.clip(p_rs, 0.03, 0.7), 1.0, 10.0).astype(np.float32)
P('prevalence ' + ' '.join(f'{x:.3f}' for x in p_rs) + ' | pos_weight ' + ' '.join(f'{x:.2f}' for x in pw))


def centers_all(mask):
    """the candidate centre set of R._eval_centers (valid-mask range, both neighbours inside)."""
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    return cs or [max(1, min((lo + hi) // 2, D - 2))]


def mask_of(sid):
    return np.frombuffer(idx.at[sid, 'mask'].encode(), np.uint8) - 48


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        sid, y = self.rows[i]
        rng = np.random.default_rng()                    # per-worker seeded by torch worker_init below
        vol = np.load(idx.at[sid, 'path'], mmap_mode='r')
        cs = centers_all(mask_of(sid))
        pick = np.sort(rng.choice(cs, A.windows, replace=len(cs) < A.windows))
        tri = np.stack([vol[[c - 1, c, c + 1]] for c in pick], 0)          # (K, 3, H, W) uint8
        jit = 1.0 + rng.uniform(-A.jitter, A.jitter)
        return torch.from_numpy(np.ascontiguousarray(tri)), torch.from_numpy(y), torch.tensor(jit, dtype=torch.float32)


def worker_init(w):
    s = (torch.initial_seed() + w) % 2**32
    np.random.seed(s); random.seed(s)


def eval_tensor(sid):
    vol = np.load(idx.at[sid, 'path'], mmap_mode='r')
    cs = R._eval_centers(mask_of(sid), D, A.eval_windows)
    tri = np.stack([vol[[max(1, min(c, D - 2)) + d for d in (-1, 0, 1)]] for c in cs], 0)
    return torch.from_numpy(np.ascontiguousarray(tri))


MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def to_input(tri_u8, jit=None):
    x = tri_u8.to(dev, non_blocking=True).float().div_(255.0)              # (K,3,H,W) in [0,1], = eval_windows /255
    if jit is not None:
        x = (x * jit.to(dev)).clamp_(0, 1)
    x = (x - MEAN) / STD
    if A.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x[None]                                                          # (1,K,3,H,W)


# ---------------- model ----------------
bb = timm.create_model(A.arch, pretrained=bool(A.pretrained), num_classes=0, in_chans=3, img_size=A.res, global_pool='avg')
if A.grad_ckpt:
    bb.set_grad_checkpointing(True)
model = R.RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
if A.channels_last:
    model = model.to(memory_format=torch.channels_last)
head_p = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
bb_p = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
P(f'model {A.arch} img_size {A.res} | params backbone {sum(p.numel() for p in bb_p)/1e6:.2f} M head {sum(p.numel() for p in head_p)/1e6:.3f} M')
opt = torch.optim.AdamW([{'params': bb_p, 'lr': A.lr_bb}, {'params': head_p, 'lr': A.lr_head}], weight_decay=A.wd)
upd_per_ep = len(rows) // A.accum
total_upd = upd_per_ep * A.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[A.lr_bb, A.lr_head], total_steps=max(1, total_upd),
                                            pct_start=A.pct_start)
use_amp = A.amp != 'off' and dev.type == 'cuda'
amp_dtype = torch.float16 if A.amp == 'fp16' else torch.bfloat16
scaler = torch.amp.GradScaler('cuda', init_scale=A.init_scale, enabled=(use_amp and A.amp == 'fp16'))
PW = torch.from_numpy(pw).to(dev)
P(f'updates/epoch {upd_per_ep} x {A.epochs} = {total_upd} | amp {A.amp if use_amp else "off"}')

start_ep, hist = 0, []
if A.resume and os.path.exists(f'{A.out}/last.pt'):
    st = torch.load(f'{A.out}/last.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    scaler.load_state_dict(st['scaler']); start_ep = st['epoch'] + 1; hist = st['hist']
    torch.set_rng_state(st['rng_torch']); np.random.set_state(st['rng_np'])
    P(f'RESUMED from last.pt after epoch {st["epoch"]}')


def evaluate():
    model.eval(); preds = []
    with torch.no_grad():
        for sid in gold_ids:
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                lg = model(to_input(eval_tensor(sid))).float()
            preds.append(torch.sigmoid(lg)[0].cpu().numpy())
    model.train()
    p = np.array(preds); y = gold.loc[gold_ids].values
    if not np.isfinite(p).all():
        P(f'ABORT: {int((~np.isfinite(p)).sum())} non-finite gold predictions (completed epochs kept)'); sys.exit(4)
    aucs = {LAB[j]: float(roc_auc_score(y[:, j], p[:, j])) for j in range(12) if len(set(y[:, j])) == 2}
    return float(np.mean(list(aucs.values()))), aucs, p


dl = torch.utils.data.DataLoader(TrainDS(rows), batch_size=1, shuffle=True, num_workers=A.workers, pin_memory=(dev.type == 'cuda'),
                                 persistent_workers=A.workers > 0, prefetch_factor=4 if A.workers > 0 else None,
                                 worker_init_fn=worker_init, drop_last=False, generator=torch.Generator().manual_seed(A.seed))
model.train()
n_nonfinite = n_steps_total = n_skip_total = n_upd_total = 0
if A.swa_only:
    import glob as _g
    hist = [(int(os.path.basename(f)[2:-3]), float(torch.load(f, map_location='cpu', weights_only=False)['gold_auc']))
            for f in _g.glob(f'{A.out}/ep*.pt')]
    P(f'SWA-ONLY over existing epochs {sorted(hist)}')
for ep in range(start_ep, 0 if A.swa_only else A.epochs):
    t_ep = time.time(); opt.zero_grad(set_to_none=True); run_loss = 0.0; n_upd = 0; t_win = time.time(); skipped = 0
    for it, (tri, y, jit) in enumerate(dl):
        if A.smoke_steps and it >= A.smoke_steps:
            break
        x = to_input(tri[0], jit[0]); y = y.to(dev)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        w = torch.isfinite(y).float(); y0 = torch.nan_to_num(y, nan=0.0)
        cell = F.binary_cross_entropy_with_logits(logits, y0, pos_weight=PW, reduction='none') * w
        loss = cell.sum() / (12 * logits.shape[0])
        n_steps_total += 1
        if not torch.isfinite(loss):
            n_nonfinite += 1
            if n_steps_total > 200 and n_nonfinite / n_steps_total > A.nonfinite_abort:
                P(f'ABORT: non-finite loss in {n_nonfinite}/{n_steps_total} steps'); sys.exit(3)
        else:
            scaler.scale(loss / A.accum).backward(); run_loss += float(loss)
        if (it + 1) % A.accum == 0:
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), A.clip)
            sc0 = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(opt); scaler.update()
            n_upd_total += 1
            if scaler.is_enabled() and scaler.get_scale() < sc0:
                skipped += 1; n_skip_total += 1                           # inf/nan grads: optimizer step skipped
                if n_upd_total > 20 and n_skip_total / n_upd_total > 0.05:
                    P(f'ABORT: {n_skip_total}/{n_upd_total} optimizer updates skipped for non-finite grads (completed epochs kept)')
                    sys.exit(5)
            opt.zero_grad(set_to_none=True)
            if sched.last_epoch < total_upd - 1:                         # OneCycleLR refuses steps past total_steps
                sched.step()
            n_upd += 1
        if (it + 1) % 200 == 0:
            el = time.time() - t_win; t_win = time.time()
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0
            P(f'ep {ep} it {it+1}/{len(dl)} loss {run_loss/200:.4f} {el/200:.3f}s/step | upd {n_upd} lr_bb {opt.param_groups[0]["lr"]:.2e} '
              f'lr_head {opt.param_groups[1]["lr"]:.2e} | scale {scaler.get_scale() if scaler.is_enabled() else 1:.0f} '
              f'skipped_upd {skipped} nonfinite_loss {n_nonfinite} | peak mem {mem:.2f} GB')
            run_loss = 0.0
    opt.zero_grad(set_to_none=True)                                       # drop the partial accumulation (= floor(N/16) updates)
    t_ev = time.time(); g, aucs, preds = evaluate()
    P(f'=== ep {ep} gold58 {g:.6f} | ' + ' '.join(f'{k[:6]}:{v:.3f}' for k, v in aucs.items())
      + f' | epoch {(time.time()-t_ep)/60:.1f} min (eval {time.time()-t_ev:.0f}s) | updates {n_upd} skipped {skipped}')
    torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'gold_auc': g, 'aucs': aucs,
                'epoch': ep}, f'{A.out}/ep{ep}.pt')
    np.save(f'{A.out}/gold58_ep{ep}.npy', preds)
    hist.append((ep, g))
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': ep, 'hist': hist, 'rng_torch': torch.get_rng_state(), 'rng_np': np.random.get_state()},
               f'{A.out}/last.pt.tmp')
    os.replace(f'{A.out}/last.pt.tmp', f'{A.out}/last.pt')
    if A.smoke_steps:
        break

# ---------------- selection: uniform average of the top-k epochs by gold58 (BN buffers averaged too) ----------------
hist_s = sorted(hist, key=lambda t: -t[1])
top = [e for e, _ in hist_s[:A.top]]
sds = [torch.load(f'{A.out}/ep{e}.pt', map_location='cpu', weights_only=False)['model'] for e in top]
avg = {}
for k, v in sds[0].items():
    avg[k] = (sum(sd[k].double() for sd in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else v.clone()
model.load_state_dict(avg)
g, aucs, preds = evaluate()
ck = {'model': avg, 'gold_auc': g, 'aucs': aucs, 'src': 'timm-pretrained' if A.pretrained else 'scratch', 'res': A.res,
      'arch': A.arch, 'lab': LAB, 'epoch': top, 'swa_over': top}
torch.save(ck, f'{A.out}/swa.pt'); np.save(f'{A.out}/gold58_swa.npy', preds)
sha = hashlib.sha256(open(f'{A.out}/swa.pt', 'rb').read()).hexdigest()
json.dump({'epochs_ranked': hist_s, 'swa_over': top, 'swa_gold58': g, 'swa_aucs': aucs, 'swa_sha256': sha,
           'nonfinite_loss_steps': n_nonfinite, 'steps': n_steps_total}, open(f'{A.out}/summary.json', 'w'), indent=1)
P(f'=== SWA over epochs {top}: gold58 {g:.6f} | ' + ' '.join(f'{k[:6]}:{v:.3f}' for k, v in aucs.items()) + f' | sha256 {sha}')
# strict reload into the SERVING geometry (img_size 320) exactly as the scored source builds it
bb2 = timm.create_model(A.arch, pretrained=False, num_classes=0, in_chans=3, img_size=320, global_pool='avg')
m2 = R.RaptorClassifier(bb2, F_dim=bb2.num_features); m2.load_state_dict(torch.load(f'{A.out}/swa.pt', map_location='cpu', weights_only=False)['model'], strict=True)
P('SERVE-LOAD OK (strict, img_size 320) | DONE')
