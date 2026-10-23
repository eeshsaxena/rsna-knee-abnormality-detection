# ttr_preflight.py -- optional 2-4 min V100 check BEFORE the TTR-A launch (memory + speed next to the running job).
# Writes nothing. Builds the models exactly as fleet_train.py does (pretrained timm weights, grad ckpt, channels_last):
#   f1  = KneeFleetNet, 24 slots, drop_path .1 (the family-1 reference, measured in the SAME contention window)
#   ttr = KneeFleetNetTTR(ttr_size), K slots through the trainer's pos path, drop_path .2, head dropout .3
# feeds bs real store studies (trainer decode, no aug; the TTR input is the K-slot subset), runs fp16 autocast +
# GradScaler + AdamW (backbone/head lr as the trainer) for --steps steps, then no-grad eval forwards at all 24 slots.
# Prints per case: peak_alloc_gb / peak_reserved_gb of THIS process (nvidia-smi adds ~0.4 GB of CUDA context), the GPU's
# free memory before start, s/step (median after 2 warm-up steps), eval s/batch, and the projected minutes per epoch:
# train = s/step * n_train / bs, eval = s/eval-batch * (n_val + n_gold) / bs * 2 tags. Measured while sharing the GPU,
# so it is an upper bound for "alone". The ratio ttr/f1 is the G-T0 prior.
# usage (box GPU, ~2-4 min, only when nvidia-smi shows >= 6 GB free):
#   cd ~/bigjump_build/code && ~/miniconda3/envs/rsna/bin/python ttr_preflight.py --store ~/fleet_store
# laptop plumbing (CPU): python ttr_preflight.py --store <npz dir> --device cpu --no_pretrained --bs 1 --steps 3
#   --backbone convnext_atto.d2_in1k
import os, sys, time, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import fleet_common as fc
from fleet_model import KneeFleetNet
from fleet_model_ttr import KneeFleetNetTTR

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True); ap.add_argument('--backbone', default='convnext_tiny.fb_in22k_ft_in1k')
ap.add_argument('--cases', default='f1,ttr'); ap.add_argument('--ttr_size', type=int, default=224)
ap.add_argument('--K', type=int, default=16); ap.add_argument('--bs', type=int, default=2)
ap.add_argument('--steps', type=int, default=8); ap.add_argument('--eval_batches', type=int, default=3)
ap.add_argument('--n_train', type=int, default=3466); ap.add_argument('--n_eval', type=int, default=867 + 74)
ap.add_argument('--device', default='cuda'); ap.add_argument('--no_pretrained', action='store_true')
A = ap.parse_args()
dev = A.device
if dev == 'cuda':
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
files = sorted(glob.glob(f'{A.store}/*.npz'))
pick = [files[i] for i in np.linspace(0, len(files) - 1, A.bs).round().astype(int)]
xs, prs, cts = [], [], []
for f in pick:
    vs, pr, ct = [], [], []
    with np.load(f) as z:
        fc.check_store_contract(z, os.path.basename(f))
        for j in range(len(fc.VIEWS)):
            if f'v{j}_blob' in z.files:
                vs.append(fc.sample_triplets(fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']), fc.N_SLICES))
                pr.append(True); ct.append(int(z[f'v{j}_ctr']))
            else:
                vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
    xs.append(np.stack(vs)); prs.append(pr); cts.append(ct)
X = torch.from_numpy(np.stack(xs)); PR = torch.tensor(prs); CT = torch.tensor(cts)
g = torch.Generator().manual_seed(0)
Y = (torch.rand((A.bs, 12), generator=g) < 0.3).float()
print(f'ttr_preflight | {A.bs} real studies | input {tuple(X.shape)} | device {dev} | torch {torch.__version__}', flush=True)
if dev == 'cuda':
    fr, tot = torch.cuda.mem_get_info()
    print(f'GPU {torch.cuda.get_device_name()} | free {fr / 2**30:.2f} / {tot / 2**30:.2f} GB before start', flush=True)


def sync():
    if dev == 'cuda':
        torch.cuda.synchronize()


out = {}
for case in A.cases.split(','):
    r = {'case': case}
    try:
        kw = dict(backbone=A.backbone, dim=256, contrast_emb=True, n_views=len(fc.VIEWS), n_slices=fc.N_SLICES,
                  n_contrast=fc.N_CONTRAST, pretrained=not A.no_pretrained, grad_ckpt=True)
        if case == 'ttr':
            m = KneeFleetNetTTR(ttr_size=A.ttr_size, drop_path=0.2, dropout=0.3, **kw)
            pos = torch.from_numpy(np.sort(np.random.default_rng(0).choice(fc.N_SLICES, A.K, replace=False)))
            x = X[:, :, pos]; posb = pos[None, None].expand(A.bs, len(fc.VIEWS), -1).contiguous().to(dev)
        else:
            m = KneeFleetNet(drop_path=0.1, **kw); x = X; posb = None
        m = m.to(dev).to(memory_format=torch.channels_last)
        x, pr, ct, y = x.to(dev), PR.to(dev), CT.to(dev), Y.to(dev)
        bb = [p for n, p in m.named_parameters() if n.startswith('cnn.')]
        hd = [p for n, p in m.named_parameters() if not n.startswith('cnn.')]
        opt = torch.optim.AdamW([{'params': bb, 'lr': 1e-4}, {'params': hd, 'lr': 1e-3}], weight_decay=0.05)
        sc = torch.amp.GradScaler(dev, enabled=(dev == 'cuda'))
        if dev == 'cuda':
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m.train(); times, losses = [], []
        for step in range(A.steps):
            sync(); t0 = time.time()
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
                lo = m(x, pr, ct, posb)
            loss = F.binary_cross_entropy_with_logits(lo.float(), y)
            sc.scale(loss).backward(); sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
            sync(); times.append(time.time() - t0); losses.append(round(float(loss), 4))
        m.eval(); et = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
            for _ in range(A.eval_batches + 1):
                sync(); t0 = time.time(); m(X.to(dev), pr, ct); sync(); et.append(time.time() - t0)
        tt = times[2:] if len(times) > 3 else times
        r.update(s_per_step=round(float(np.median(tt)), 3), eval_s_per_batch=round(float(np.median(et[1:] or et)), 3),
                 losses_finite=bool(np.isfinite(losses).all()))
        r['proj_train_min'] = round(r['s_per_step'] * A.n_train / A.bs / 60, 1)
        r['proj_eval_min'] = round(r['eval_s_per_batch'] * A.n_eval / A.bs * 2 / 60, 1)
        r['proj_epoch_min'] = round(r['proj_train_min'] + r['proj_eval_min'], 1)
        if dev == 'cuda':
            r['peak_alloc_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            r['peak_reserved_gb'] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        del m, opt
    except torch.cuda.OutOfMemoryError as e:
        r['error'] = 'OOM ' + str(e)[:160]
    if dev == 'cuda':
        torch.cuda.empty_cache()
    out[case] = r
    print(json.dumps(r), flush=True)
if 'f1' in out and 'ttr' in out and 's_per_step' in out['f1'] and 's_per_step' in out['ttr']:
    print(f'RATIO ttr/f1: train {out["ttr"]["s_per_step"] / out["f1"]["s_per_step"]:.2f}, '
          f'epoch {out["ttr"]["proj_epoch_min"] / max(out["f1"]["proj_epoch_min"], 1e-9):.2f} (plan MODEL x0.33 train)')
