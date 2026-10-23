# fleet_train.py -- V100 16GB trainer for the ConvNeXt-T 320x24x4-view fleet.   (round3/fleet CONSOLIDATED)
# Input store = per-study npz written by the Kaggle CPU prep kernel with fleet_common.study_npz_dict
# (keys contract, v{j}_blob, v{j}_offs, v{j}_ctr, v{j}_uid for present views j=0..3).
# Hyper-parameters marked GUESS are not recoverable from aastik's public files.
# round3 fixes (FLEET_REVIEW.md, IDs T#):
#   T1  --fold -1 = all-data: no validation, fixed-epoch save (draft: np.concatenate([]) crash after
#       epoch 1, or, without --exclude, selection on gold74); --stop_epoch keeps the --epochs cosine
#   T2  held-out = (fold < 0 in folds5.csv) | --exclude, asserted disjoint from train/val (draft:
#       gold74 trained in every fold run whenever --exclude was dropped)
#   T3  --eval_labels (steven_v4): checkpoint selection + G1 metric NOT on the mix50 training targets
#       (half of mix50 is the image teacher, so selecting on it rewards copying the teacher)
#   T4  --gold: gold58/gold74 AUC logged as a bug guard, never used to select
#   T5  store contract asserted on every npz (S14 drift class); contract recorded in the ckpt;
#       decode probe of 4 stored studies recorded so the worker can prove serve-decode == train-decode
#   T6  --train_slices K (G0 budget variant), model built from fc.N_SLICES / fc.N_CONTRAST / len(VIEWS)
#   T7  per-STUDY mirror coin inside the Dataset (draft: one coin per batch, on the GPU)
#   T8  persistent_workers only when workers > 0; warmup/steps >= 1; cudnn.benchmark; cv2 threads off
#       in loader workers; TRAIN-ONLY epoch minutes printed for the G0 gate
#   T9  eval: NaN labels/preds masked per target; non-finite predictions never selected
#   T10 missing store studies counted and asserted (>= 99% of usable labelled studies)
#   T11 ckpt name carries the real STORE x N_SLICES; OOF index named StudyInstanceUID
#   T12 --device (cuda default, asserted available; cpu only for the laptop dry run, AMP off there)
# fleet2 patch (family 2 = BatchNorm CNN backbones), OPT-IN via --f2 1. Without --f2 the run is the round3 trainer
# bit for bit: same model class, RNG stream, weights, OOF, ckpt name and ckpt args (fleet2/tests/t2_trainer_parity.py).
#   F2-T1 --f2 1: model = fleet_model_f2.KneeFleetNetF2 (tag norm in persistent buffers, frozen BN, optional init ckpt)
#   F2-T2 --freeze_bn (default 1) / --init_ckpt / --channels_last (default 1); only read when --f2 1
#   F2-T3 guards: a BatchNorm or non-ImageNet-norm backbone WITHOUT --f2 is refused (the round3 path would train BN
#         on single-study zero-padded chunks and feed it ImageNet stats); --f2 needs --seed >= 100 so the
#         f{fold}_s{seed} member key (worker de-dup, pack tool) never collides with family 1 (42, 43, ...)
#   F2-T4 --f2 ckpt name tag from the backbone (ev2s, r50, r50rad), not "cnxt"; args gain model_mod, freeze_bn,
#         init_ckpt, norm, channels_last, lr, head_lr; model_sha = sha(fleet_model_f2 source + fleet_model source),
#         recomputed by the fleet2 worker and pack tool
#   F2-T5 fp16 watchdog (--f2 only): abort after 25 CONSECUTIVE GradScaler-skipped optimizer steps (non-finite fp16
#         loss/grads every step). Frozen-BN BatchNorm nets have no batch-statistics safety net; without this a NaN
#         run would burn its full 8 h printing nan (laptop t6/t7: random-init EV2S overflowed fp16). Gate it first
#         with f2_preflight.py on the box. Family-1 path untouched (no extra sync, bit-identical).
import os, sys, math, time, json, argparse, random
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet_common as fc
from fleet_model import KneeFleetNet, mirror_views

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True); ap.add_argument('--labels', required=True)
ap.add_argument('--folds', required=True); ap.add_argument('--fold', type=int, required=True)   # -1 = all-data
ap.add_argument('--exclude', default='')            # gold74 ids -> never trained on
ap.add_argument('--eval_labels', default='')        # T3 steven_v4 csv: selection + G1 metric
ap.add_argument('--gold', default='')               # T4 gold74 csv WITH labels: logged only
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=12)
ap.add_argument('--stop_epoch', type=int, default=0)    # T1 stop + save after N epochs; schedule stays --epochs
ap.add_argument('--train_slices', type=int, default=0)  # T6 budget variant: K of N_SLICES triplets, TRAIN only
ap.add_argument('--bs', type=int, default=2); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--head_lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=0.05); ap.add_argument('--ema', type=float, default=0.999)
ap.add_argument('--backbone', default='convnext_tiny.fb_in22k_ft_in1k')
ap.add_argument('--dim', type=int, default=256); ap.add_argument('--contrast_emb', type=int, default=1)
ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--time_epochs', type=int, default=0)   # >0: stop after N epochs (timing probe)
ap.add_argument('--device', default='cuda')             # T12
ap.add_argument('--min_cover', type=float, default=0.99)   # T10 store coverage floor (lower ONLY after reading the prep receipts)
ap.add_argument('--f2', type=int, default=0)               # F2-T1 family-2 model (BatchNorm backbones)
ap.add_argument('--freeze_bn', type=int, default=1)        # F2-T2 (--f2 only) BN in eval mode while training
ap.add_argument('--init_ckpt', default='')                 # F2-T2 (--f2 only) e.g. ~/rsna_train/ResNet50_radimagenet.pt
ap.add_argument('--channels_last', type=int, default=1)    # F2-T2 (--f2 only) 0 only if G0 shows NHWC is slow
A = ap.parse_args()
if A.f2:                                                    # F2-T3
    assert A.seed >= 100, f'--f2 runs use seeds >= 100 (family 1 owns 42, 43, ...): got --seed {A.seed}'
else:
    assert not A.init_ckpt, '--init_ckpt needs --f2 1'
ALL_DATA = A.fold < 0
torch.manual_seed(A.seed); np.random.seed(A.seed); random.seed(A.seed)
dev = A.device
if dev == 'cuda':
    assert torch.cuda.is_available(), 'no CUDA -- refusing to train on CPU (pass --device cpu only for dry runs)'
    torch.backends.cudnn.benchmark = True                                 # T8 fixed shapes
AMP = dev == 'cuda'                                                       # V100: fp16 only (no bf16)

lab = pd.read_csv(A.labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
folds = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
excl = set(pd.read_csv(A.exclude, dtype={'StudyInstanceUID': str}).StudyInstanceUID) if A.exclude else set()
held = set(folds.index[folds < 0]) | excl                                 # T2
have = {s for s in lab.index if os.path.exists(f'{A.store}/{s}.npz')}
ids = [s for s in lab.index if s in folds.index and s in have]
tr_ids = [s for s in ids if s not in held and (ALL_DATA or folds[s] != A.fold)]
va_ids = [] if ALL_DATA else [s for s in ids if s not in held and folds[s] == A.fold]
n_want = sum(1 for s in lab.index if s in folds.index and s not in held)
print(f'labels {len(lab)} | store has {len(have)} | held-out (fold<0 or --exclude) {len(held)} | '
      f'train {len(tr_ids)} | val {len(va_ids)} | all_data {ALL_DATA} | device {dev}', flush=True)
assert not (set(tr_ids) | set(va_ids)) & held
assert len(tr_ids) + len(va_ids) >= A.min_cover * n_want, (                      # T10
    f'store covers only {len(tr_ids) + len(va_ids)}/{n_want} usable studies -- check the prep receipts')
sel_tab = (pd.read_csv(A.eval_labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
           if A.eval_labels else lab)                                     # T3
if A.eval_labels and va_ids:
    assert set(va_ids) <= set(sel_tab.index), 'eval_labels missing val studies'
gold_tab = (pd.read_csv(A.gold, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
            if A.gold else None)                                          # T4
gold_ids = [s for s in gold_tab.index if s in have and s in lab.index] if gold_tab is not None else []
gold58 = ([s for s in gold_ids if gold_tab.loc[s, 'source'] == 'original_gold58']
          if gold_tab is not None and 'source' in gold_tab.columns else [])
PLANES = [p for p, _ in fc.VIEWS]

# T5 contract: every npz checked in __getitem__; first 8 also decoded here (aastik S14 decodes a stored slice)
probe_ids = sorted(ids)[:4]
decode_probe = {}
for s in sorted(ids)[:8]:
    with np.load(f'{A.store}/{s}.npz') as z:
        fc.check_store_contract(z, s)
        for j in range(len(fc.VIEWS)):
            if f'v{j}_blob' in z.files:
                st0 = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
                assert st0.shape[1:] == (fc.STORE, fc.STORE) and 1 <= len(st0) <= fc.SRC_SLICES, st0.shape
        if s in probe_ids:
            decode_probe[s] = fc.decode_sha(z)
print('contract', fc.CONTRACT, '| select_sha', fc.SELECT_SHA, '| jpeg_fp', fc.jpeg_fingerprint(), flush=True)


def _wif(_):                                         # T8 no cv2 thread pool per DataLoader worker
    try:
        import cv2; cv2.setNumThreads(0)
    except Exception:
        pass


class DS(torch.utils.data.Dataset):
    def __init__(self, ids, train):
        self.ids, self.train = ids, train
    def __len__(self):
        return len(self.ids)
    def _aug_view(self, trip):                     # trip uint8 (S,3,H,W); one affine per view
        import cv2
        H = trip.shape[-1]
        ang = np.random.uniform(-10, 10); sc = np.random.uniform(0.9, 1.1)   # GUESS
        tx, ty = np.random.uniform(-0.05, 0.05, 2) * H
        Mx = cv2.getRotationMatrix2D((H / 2, H / 2), ang, sc); Mx[:, 2] += (tx, ty)
        out = np.empty_like(trip)
        for s in range(trip.shape[0]):
            for c in range(3):
                out[s, c] = cv2.warpAffine(trip[s, c], Mx, (H, H), flags=cv2.INTER_LINEAR, borderValue=0)
        g = np.random.uniform(0.9, 1.1); b = np.random.uniform(-10, 10)           # GUESS
        return np.clip(out.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
    def __getitem__(self, i):
        sid = self.ids[i]
        vs, pr, ct = [], [], []
        with np.load(f'{A.store}/{sid}.npz') as z:
            fc.check_store_contract(z, sid)                                       # T5
            for j in range(len(fc.VIEWS)):
                if f'v{j}_blob' in z.files:
                    st = fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])      # SAME decode as the worker
                    off = np.random.uniform(-0.5, 0.5) if self.train else 0.0     # slice jitter, GUESS
                    t = fc.sample_triplets(st, fc.N_SLICES, off)
                    if self.train:
                        t = self._aug_view(t)
                    vs.append(t); pr.append(True); ct.append(int(z[f'v{j}_ctr']))
                else:
                    vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
        if self.train and sum(pr) > 1 and np.random.rand() < 0.1:                 # view dropout, GUESS
            j = np.random.choice([k for k in range(len(pr)) if pr[k]]); pr[j] = False
            vs[j] = np.zeros_like(vs[j]); ct[j] = 3
        x = torch.from_numpy(np.stack(vs))
        if self.train and np.random.rand() < 0.5:                                 # T7 per-STUDY mirror,
            x = mirror_views(x[None], PLANES)[0]                                  # before any slice subset
        pos = torch.arange(fc.N_SLICES)
        if self.train and 0 < A.train_slices < fc.N_SLICES:                       # T6 budget variant
            pos = torch.from_numpy(np.sort(np.random.choice(fc.N_SLICES, A.train_slices, replace=False)))
            x = x[:, pos]
        y = lab.loc[sid, TARGETS].to_numpy(np.float32)
        return x, torch.tensor(pr), torch.tensor(ct), pos[None].expand(len(fc.VIEWS), -1).clone(), torch.from_numpy(y)


def dl(ids, tr):
    return torch.utils.data.DataLoader(DS(ids, tr), batch_size=A.bs, shuffle=tr, drop_last=tr,
                                       num_workers=A.workers, pin_memory=(dev == 'cuda'), worker_init_fn=_wif,
                                       persistent_workers=A.workers > 0)                  # T8
tr_dl = dl(tr_ids, True)
va_dl = dl(va_ids, False) if va_ids else None                                             # T1
gold_dl = dl(gold_ids, False) if gold_ids else None
if not A.f2:
    model = KneeFleetNet(backbone=A.backbone, dim=A.dim, contrast_emb=bool(A.contrast_emb), n_views=len(fc.VIEWS),
                         n_slices=fc.N_SLICES, n_contrast=fc.N_CONTRAST,                      # T6
                         pretrained=True, drop_path=0.1, grad_ckpt=True).to(dev)
    model = model.to(memory_format=torch.channels_last)
    _bn = any(isinstance(q, torch.nn.modules.batchnorm._BatchNorm) for q in model.cnn.modules())   # F2-T3 guard
    _cfg = model.cnn.pretrained_cfg
    _nm = [round(float(v), 3) for v in tuple(_cfg.get('mean', (0.485, 0.456, 0.406))) + tuple(_cfg.get('std', (0.229, 0.224, 0.225)))]
    assert not _bn and _nm == [0.485, 0.456, 0.406, 0.229, 0.224, 0.225], (
        f'{A.backbone}: BatchNorm={_bn} norm={_nm} -- the round3 path supports BN-free ImageNet-norm '
        f'backbones only; train it with --f2 1 (frozen BN, tag norm)')
    BBTAG = 'cnxt'
else:
    from fleet_model_f2 import KneeFleetNetF2                                                 # F2-T1
    model = KneeFleetNetF2(backbone=A.backbone, dim=A.dim, contrast_emb=bool(A.contrast_emb), n_views=len(fc.VIEWS),
                           n_slices=fc.N_SLICES, n_contrast=fc.N_CONTRAST,
                           pretrained=True, drop_path=0.1, grad_ckpt=True, freeze_bn=bool(A.freeze_bn),
                           init_ckpt=os.path.expanduser(A.init_ckpt), channels_last=bool(A.channels_last)).to(dev)
    if A.channels_last:
        model = model.to(memory_format=torch.channels_last)
    _bb = A.backbone.split('.')[0]
    BBTAG = {'tf_efficientnetv2_s': 'ev2s', 'resnet50': 'r50', 'resnet34': 'r34',
             'tf_efficientnet_b4': 'eb4'}.get(_bb, _bb.replace('_', '')) + ('rad' if A.init_ckpt else '') + '-'   # F2-T4
    print(f'F2 backbone {A.backbone} | name tag {BBTAG} | norm {model.norm_mean}/{model.norm_std} | freeze_bn '
          f'{A.freeze_bn} | init {A.init_ckpt or "timm"} | lr {A.lr}/{A.head_lr} | channels_last {A.channels_last}',
          flush=True)
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(A.ema))
bb = [p for n, p in model.named_parameters() if n.startswith('cnn.')]
hd = [p for n, p in model.named_parameters() if not n.startswith('cnn.')]
opt = torch.optim.AdamW([{'params': bb, 'lr': A.lr}, {'params': hd, 'lr': A.head_lr}], weight_decay=A.wd)
steps = max(1, A.epochs * len(tr_dl) // A.accum); warm = max(1, len(tr_dl) // A.accum)   # T8
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
scaler = torch.amp.GradScaler('cuda', enabled=AMP)


def predict(m, loader):
    m.eval(); P = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
        for x, pr, ct, _pos, _y in loader:                        # eval: pos=None -> all N_SLICES
            P.append(torch.sigmoid(m(x.to(dev), pr.to(dev), ct.to(dev)).float()).cpu().numpy())
    return np.concatenate(P)


def macro_auc(P, ids, tab):
    # T9 NaN labels/preds masked per target (draft: NaN label -> negative, NaN pred -> crash)
    Y = tab.loc[ids, TARGETS].to_numpy(np.float64); aucs = []
    for j in range(len(TARGETS)):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        if ok.sum() and 0 < yb.mean() < 1:
            aucs.append(roc_auc_score(yb, P[ok, j]))
    return (float(np.mean(aucs)) if aucs else float('nan')), len(aucs)


def save(m, tag, name_tail, cv, ep, P=None):
    for q in [q for q in os.listdir(A.out) if q.startswith(f'f{A.fold}_s{A.seed}_{tag}_')]:
        os.remove(f'{A.out}/{q}')
    torch.save({'model': m.state_dict(), 'args': args, 'cv': cv, 'epoch': ep},
               f'{A.out}/f{A.fold}_s{A.seed}_{tag}_{BBTAG}{fc.STORE}x{fc.N_SLICES}_{name_tail}_best.pt')   # T11, F2-T4
    if P is not None:
        pd.DataFrame(P, columns=TARGETS, index=pd.Index(va_ids, name='StudyInstanceUID')).to_csv(
            f'{A.out}/oof_f{A.fold}_s{A.seed}_{tag}.csv')


def _model_sha():
    import hashlib, inspect, fleet_model as _fm
    src = chr(10).join(inspect.getsource(_fm).splitlines())
    if A.f2:                                                    # F2-T4: f2 module + the base it inherits from
        import fleet_model_f2 as _f2
        src = chr(10).join(inspect.getsource(_f2).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]
os.makedirs(A.out, exist_ok=True)
best = {'raw': -1.0, 'ema': -1.0}
args = dict(backbone=A.backbone, dim=A.dim, contrast_emb=bool(A.contrast_emb), n_slices=fc.N_SLICES,
            n_views=len(fc.VIEWS), n_contrast=fc.N_CONTRAST, store=fc.STORE, crop_mm=fc.CROP_MM,
            src_slices=fc.SRC_SLICES, views=fc.VIEWS, fold=A.fold, seed=A.seed,
            contract=fc.CONTRACT, select_sha=fc.SELECT_SHA, jpeg_fp=fc.jpeg_fingerprint(),      # T5
            decode_probe=decode_probe, amp='fp16' if AMP else 'fp32', bf16=False,
            epochs=A.epochs, stop_epoch=A.stop_epoch, train_slices=A.train_slices,
            sel_labels=os.path.basename(A.eval_labels or A.labels), train_labels=os.path.basename(A.labels),
            mirror_aug=0.5, model_sha=_model_sha())   # N-B
if A.f2:                                                        # F2-T4 (family-1 args stay exactly as before)
    args.update(model_mod='fleet_model_f2', freeze_bn=bool(A.freeze_bn), init_ckpt=os.path.basename(A.init_ckpt),
                norm=[list(model.norm_mean), list(model.norm_std)], channels_last=bool(A.channels_last),
                lr=A.lr, head_lr=A.head_lr)
nskip = 0                                                       # F2-T5
for ep in range(A.epochs):
    model.train(); t0 = time.time(); opt.zero_grad(set_to_none=True)
    for it, (x, pr, ct, pos, y) in enumerate(tr_dl):
        x, pr, ct, pos, y = (t.to(dev, non_blocking=True) for t in (x, pr, ct, pos, y))
        with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
            logit = model(x, pr, ct, pos if A.train_slices else None)
        msk = torch.isfinite(y).float(); yy = torch.nan_to_num(y)
        loss = (F.binary_cross_entropy_with_logits(logit.float(), yy, reduction='none') * msk).sum() / msk.sum().clamp(min=1)
        scaler.scale(loss / A.accum).backward()
        if (it + 1) % A.accum == 0:
            s_before = scaler.get_scale() if (A.f2 and AMP) else None                             # F2-T5
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            ema.update_parameters(model)
            if s_before is not None:                                                              # F2-T5
                nskip = nskip + 1 if scaler.get_scale() < s_before else 0
                if nskip >= 25:
                    print(f'!! F2-T5 fp16 watchdog: {nskip} consecutive GradScaler-skipped steps at ep {ep} it {it} '
                          f'(loss {loss.item()}, scale {scaler.get_scale():.3g}): non-finite fp16 forward/grads. '
                          f'Aborting; run f2_preflight.py for this backbone.', flush=True)
                    sys.exit(2)
        if it % 200 == 0:
            print(f'ep {ep} it {it}/{len(tr_dl)} loss {loss.item():.4f} {(time.time()-t0)/(it+1):.2f}s/it', flush=True)
    print(f'ep {ep} TRAIN-ONLY {(time.time()-t0)/60:.1f} min (G0 timing)', flush=True)   # T8
    last = (ep + 1 == A.epochs or (A.stop_epoch and ep + 1 >= A.stop_epoch)
            or (A.time_epochs and ep + 1 >= A.time_epochs))
    for tag, m in (('raw', model), ('ema', ema.module)):
        msg = f'ep {ep} {tag}'
        if gold_dl is not None:                                    # T4 bug guard, never selects
            Pg = predict(m, gold_dl)
            g74, n74 = macro_auc(Pg, gold_ids, gold_tab)
            msg += f' | gold74 {g74:.4f} ({n74} labels)'
            if gold58:
                k = [gold_ids.index(s) for s in gold58]
                g58, n58 = macro_auc(Pg[k], gold58, gold_tab); msg += f' | gold58 {g58:.4f} ({n58} labels)'
        if ALL_DATA:                                               # T1 no OOF: fixed-epoch save
            if last:
                save(m, tag, f'ep{ep + 1}', float('nan'), ep)
            print(msg + (' | saved' if last else ''), flush=True)
            continue
        P = predict(m, va_dl)
        cv, n_t = macro_auc(P, va_ids, sel_tab)
        cv_lab, _ = macro_auc(P, va_ids, lab)
        print(f'{msg} | val AUC vs {args["sel_labels"]} {cv:.4f} ({n_t} labels) | vs train labels {cv_lab:.4f}'
              f' | epoch {(time.time()-t0)/60:.1f} min', flush=True)
        if not np.isfinite(P).all() or not np.isfinite(cv):
            print(f'!! {tag} produced non-finite predictions/AUC at ep {ep}; not selectable', flush=True)
            continue
        if cv > best[tag]:
            best[tag] = cv
            save(m, tag, f'cv{int(cv * 1e4)}', cv, ep, P)
    if last:
        break
print(json.dumps(best), flush=True)
