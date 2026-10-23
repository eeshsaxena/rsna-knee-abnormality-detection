# bench.py -- fast_build throughput bench for the fleet trainer (config T = TTR, config F = default fleet).
# Mirrors the fleet_train.py training step (fp16 autocast + GradScaler, channels_last, cudnn.benchmark, AdamW 2 groups,
# grad clip 1.0, LambdaLR, EMA AveragedModel, accum so that bs x accum = 8) on synthetic uint8 input of the real shapes,
# using the REAL model classes copied from ~/bigjump_build/code (fleet_model*.py are never edited; the NCHW variant is a
# bench-only subclass of encode()).
# GPU safety (box rules): one bench process at a time (flock), waits for >= --min_free MiB free on the GPU (10 min max),
# hard cap torch.cuda.set_per_process_memory_fraction(cap/16) BEFORE any allocation, run under nice -n 10 and an outer
# timeout by the runner, stops starting new segments after --budget seconds. Inductor compile workers are forced to 1
# thread (TORCHINDUCTOR_COMPILE_THREADS=1, set by the runner and asserted here) so no uncapped helper process can open a
# CUDA context.
# Variant spec: 'k=v,k=v' joined by ';'. Keys: ck (full|none|s0|s01|s012) cl (1|0) comp (none|cnn|cnn-mat|blk|blk-mat)
# fused (0|1) B S chunk.
import os, sys, time, json, argparse, gc, math, subprocess, fcntl, datetime, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', choices=('T', 'F'), required=True)
ap.add_argument('--mode', choices=('train', 'eval', 'mem', 'opt'), default='train')
ap.add_argument('--variants', default='base')
ap.add_argument('--reps', type=int, default=1)
ap.add_argument('--rot', type=int, default=0)           # initial rotation of the variant order (interleave across processes)
ap.add_argument('--warm', type=int, default=8)          # micro-steps (train) / batches (eval)
ap.add_argument('--timed', type=int, default=12)        # micro-steps (train) / batches (eval)
ap.add_argument('--tag', default='')
ap.add_argument('--log', default=os.path.join(HERE, 'logs', 'results.jsonl'))
ap.add_argument('--budget', type=float, default=300.0)  # s since start: no new segment after this
ap.add_argument('--cap', type=float, default=3.0)       # GB of 16
ap.add_argument('--min_free', type=int, default=4500)   # MiB free on the GPU before we touch it
A = ap.parse_args()
T_START = time.time()
import signal
signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(SystemExit('SIGTERM (outer timeout)')))

assert os.environ.get('TORCHINDUCTOR_COMPILE_THREADS') == '1', 'runner must set TORCHINDUCTOR_COMPILE_THREADS=1'

# ---------------------------------------------------------------- one GPU process of ours at a time
_lock = open(os.path.join(HERE, 'gpu.lock'), 'w')
try:
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print('another bench process holds ~/fast_build/gpu.lock: refusing to start', flush=True)
    sys.exit(3)


def smi(q):
    out = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=60).stdout
    return [x.strip() for x in out.strip().splitlines()[0].split(',')]


t_wait = time.time()
while True:
    free = int(smi('memory.free')[0])
    if free >= A.min_free:
        break
    if time.time() - t_wait > 600:
        print(json.dumps({'abort': f'GPU free {free} MiB < {A.min_free} for 10 min'}), flush=True)
        sys.exit(4)
    time.sleep(20)
print(f'GPU free {free} MiB >= {A.min_free}: starting', flush=True)

import numpy as np
import torch
import torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(A.cap / 16.0, 0)      # hard cap BEFORE any allocation
CAP_BYTES = torch.cuda.get_device_properties(0).total_memory * A.cap / 16.0
torch.backends.cudnn.benchmark = True
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import fleet_model as fm
import fleet_model_ttr as fmt

# pmon sampler (nvidia-smi only reads counters; it is not a CUDA process)
PMON_PATH = os.path.join(HERE, 'logs', f'pmon_{os.getpid()}.txt')
_pmon = subprocess.Popen(['nvidia-smi', 'pmon', '-s', 'u', '-d', '1', '-o', 'T'], stdout=open(PMON_PATH, 'w'),
                         stderr=subprocess.DEVNULL)
MYPID = os.getpid()


def pmon_share(t0, t1):
    """mean sm% per pid over wall window [t0, t1] (pmon -o T prints HH:MM:SS)."""
    day = datetime.date.today()
    acc = {}
    try:
        lines = open(PMON_PATH).read().splitlines()
    except OSError:
        return {}
    for ln in lines:
        if ln.startswith('#'):
            continue
        p = ln.split()
        if len(p) < 6 or p[4] == '-':
            continue
        try:
            hh, mm, ss = (int(v) for v in p[0].split(':'))
            ts = datetime.datetime.combine(day, datetime.time(hh, mm, ss)).timestamp()
            pid, sm = int(p[2]), float(p[4])
        except ValueError:
            continue
        if t0 - 0.5 <= ts <= t1 + 0.5:
            acc.setdefault(pid, []).append(sm)
    return {('me' if k == MYPID else str(k)): [round(float(np.mean(v)), 1), len(v)] for k, v in acc.items()}


# ---------------------------------------------------------------- bench-only NCHW encode (L2); model files untouched
class F_NCHW(fm.KneeFleetNet):
    def encode(self, x_u8):
        outs = []
        for i in range(0, x_u8.shape[0], self.chunk):
            x = x_u8[i:i + self.chunk].float().div_(255.0)
            x = ((x - self.m) / self.s).contiguous()
            outs.append(self.cnn(x))
        return torch.cat(outs)


class T_NCHW(fmt.KneeFleetNetTTR):
    def encode(self, x_u8):
        outs = []
        S = self.ttr_size
        for i in range(0, x_u8.shape[0], self.chunk):
            x = x_u8[i:i + self.chunk].float().div_(255.0)
            if x.shape[-2] != S or x.shape[-1] != S:
                x = F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False, antialias=True)
            x = ((x - self.m) / self.s).contiguous()
            outs.append(self.cnn(x))
        return torch.cat(outs)


DEF_S = {'T': 16, 'F': 24}                      # train slices per view fed by the trainer
DP, DO = {'T': (0.2, 0.3), 'F': (0.1, 0.2)}[A.cfg]
EMA_DECAY = {'T': 0.9995, 'F': 0.999}[A.cfg]
ITS_PER_EPOCH = 1733


def parse_variant(s):
    v = dict(ck='full', cl=1, comp='none', fused=0, B=2, S=DEF_S[A.cfg], chunk=32)
    if s and s != 'base':
        for kv in s.split(','):
            k, _, val = kv.partition('=')
            v[k] = val if k in ('ck', 'comp') else int(val)
    v['name'] = s or 'base'
    return v


def set_ckpt(model, ck):
    on = {'full': {0, 1, 2, 3}, 'none': set(), 's0': {0}, 's01': {0, 1}, 's012': {0, 1, 2}}[ck]
    for i, st in enumerate(model.cnn.stages):
        st.grad_checkpointing = i in on


def build(v):
    cls = {('T', 1): fmt.KneeFleetNetTTR, ('T', 0): T_NCHW, ('F', 1): fm.KneeFleetNet, ('F', 0): F_NCHW}[(A.cfg, v['cl'])]
    kw = dict(backbone='convnext_tiny.fb_in22k_ft_in1k', dim=256, contrast_emb=True, n_views=4, n_slices=24,
              n_contrast=4, pretrained=True, drop_path=DP, dropout=DO, grad_ckpt=True)   # trainer: grad_ckpt=True always
    if A.cfg == 'T':
        kw['ttr_size'] = 224
    model = cls(**kw).cuda()
    if v['cl']:
        model = model.to(memory_format=torch.channels_last)
    set_ckpt(model, v['ck'])
    model.chunk = v['chunk']
    return model


def compile_model(model, comp):
    """in-place nn.Module.compile -> state_dict keys unchanged (no _orig_mod. prefix)."""
    if comp == 'none':
        return {}
    keys0 = list(model.state_dict().keys())
    mode = 'max-autotune-no-cudagraphs' if comp.endswith('-mat') else None
    if comp.startswith('cnn'):
        model.cnn.compile(mode=mode, dynamic=False)
    elif comp.startswith('blk'):
        for st in model.cnn.stages:
            for b in st.blocks:
                b.compile(mode=mode, dynamic=False)
    else:
        raise ValueError(comp)
    return {'sd_keys_same': list(model.state_dict().keys()) == keys0}


def make_inputs(v, train=True):
    B, S, V = v['B'], v['S'], 4
    g = torch.Generator().manual_seed(1234)
    x = torch.randint(0, 256, (B, V, S, 3, 320, 320), dtype=torch.uint8, generator=g).pin_memory()
    pr = torch.ones(B, V, dtype=torch.bool).pin_memory()
    ct = torch.randint(0, 4, (B, V), generator=g).pin_memory()
    y = torch.randint(0, 2, (B, 12), generator=g).float().pin_memory()
    pos = None
    if train and (A.cfg == 'T' or S != 24):     # trainer passes pos only when --train_slices (T); F reduced-S needs it
        rows = [torch.from_numpy(np.sort(np.random.RandomState(b).choice(24, S, replace=False))) for b in range(B)]
        pos = torch.stack([r[None].expand(V, -1) for r in rows]).long().pin_memory()
    return x, pr, ct, pos, y


def sync():
    torch.cuda.synchronize()


def gpu_clock():
    try:
        c, t = smi('clocks.sm,temperature.gpu')
        return int(c), int(t)
    except Exception:
        return None, None


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def log(rec):
    rec.update(cfg=A.cfg, mode=A.mode, tag=A.tag, pid=MYPID, ts=datetime.datetime.now().isoformat(timespec='seconds'))
    os.makedirs(os.path.dirname(A.log), exist_ok=True)
    with open(A.log, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    short = {k: rec[k] for k in rec if k not in ('step_s', 'opt_s', 'batch_s')}
    print(json.dumps(short), flush=True)


# ---------------------------------------------------------------- train segment
def train_segment(v, warm, timed, mem_only=False):
    from torch._dynamo.utils import counters
    counters.clear()
    B = v['B']
    # optimizer step every 8 FULL studies' worth of images (bs 2 x accum 4 in the trainer), so a reduced-S run
    # amortises the optimizer exactly like the real run; mem: step every micro-step (Adam state present from step 2)
    accum = 1 if mem_only else max(1, (8 * 4 * DEF_S[A.cfg]) // (B * 4 * v['S']))
    rec = dict(variant=v['name'], ck=v['ck'], cl=v['cl'], comp=v['comp'], fused=v['fused'], B=B, S=v['S'],
               chunk=v['chunk'], accum=accum, imgs_per_sample=4 * v['S'])
    torch.manual_seed(0)
    model = opt = ema = scaler = sched = None
    try:
        tb = time.perf_counter()
        model = build(v)
        ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY))
        rec.update(compile_model(model, v['comp']))      # after the EMA deepcopy, like a trainer fast path would
        bb = [p for n, p in model.named_parameters() if n.startswith('cnn.')]
        hd = [p for n, p in model.named_parameters() if not n.startswith('cnn.')]
        okw = dict(fused=True) if v['fused'] else {}
        opt = torch.optim.AdamW([{'params': bb, 'lr': 1e-4}, {'params': hd, 'lr': 1e-3}], weight_decay=0.05, **okw)
        steps = 24 * ITS_PER_EPOCH // 4
        warmup = ITS_PER_EPOCH // 4
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
        scaler = torch.amp.GradScaler('cuda', init_scale=2.0 ** 10)
        xc, prc, ctc, posc, yc = make_inputs(v)
        model.train()
        rec['build_s'] = round(time.perf_counter() - tb, 2)
        rec['opt_impl'] = 'fused' if opt.defaults.get('fused') else ('foreach' if opt.defaults.get('foreach') in (None, True) else 'for')
        nskip = [0]

        def opt_phase():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            ema.update_parameters(model)

        def micro(it):
            x, pr, ct, y = (t.to('cuda', non_blocking=True) for t in (xc, prc, ctc, yc))
            pos = posc.to('cuda', non_blocking=True) if posc is not None else None
            with torch.autocast('cuda', dtype=torch.float16):
                logit = model(x, pr, ct, pos)
            msk = torch.isfinite(y).float()
            yy = torch.nan_to_num(y)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), yy, reduction='none') * msk).sum() / msk.sum().clamp(min=1)
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0:
                opt_phase()

        torch.cuda.reset_peak_memory_stats()
        sync()
        t = time.perf_counter()
        micro(0)
        sync()
        rec['first_step_s'] = round(time.perf_counter() - t, 2)          # includes cudnn benchmark / compile
        rec['peak_first_MiB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20)   # inflated by cudnn algo search
        torch.cuda.reset_peak_memory_stats()                              # steady-state peak from here on
        if mem_only:
            for it in range(1, 4):
                micro(it)
            sync()
            rec['peak_alloc_MiB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            rec['peak_reserved_MiB'] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
            rec['oom'] = False
            return rec
        t = time.perf_counter()
        for it in range(1, warm):
            micro(it)
        sync()
        rec['warm_rest_s'] = round(time.perf_counter() - t, 2)
        steps_s = []
        sc0 = scaler.get_scale()
        w0 = time.time()
        ts = time.perf_counter()
        for it in range(warm, warm + timed):
            t1 = time.perf_counter()
            micro(it)
            sync()
            steps_s.append(time.perf_counter() - t1)
        total = time.perf_counter() - ts
        w1 = time.time()
        rec['scale_start_end'] = [sc0, scaler.get_scale()]
        rec['timed_steps'] = timed
        rec['timed_s'] = round(total, 3)
        rec['per_sample_s'] = round(total / (timed * B), 4)
        rec['per_img_ms'] = round(1000 * total / (timed * B * 4 * v['S']), 3)
        rec['per_full_sample_s'] = round(total / (timed * B * 4 * v['S']) * 4 * DEF_S[A.cfg], 4)   # = per_sample_s at S=DEF_S
        rec['per_sample_med_s'] = round(statistics.median(steps_s) / B, 4)   # median micro-step (opt-steps included at 1/accum)
        rec['step_s'] = [round(s, 3) for s in steps_s]
        rec['peak_alloc_MiB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        rec['peak_reserved_MiB'] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
        # isolated optimizer phase (L4): grads from one more backward, then N opt phases (unscale+clip+step+update+ema)
        x, pr, ct, y = (t_.to('cuda') for t_ in (xc, prc, ctc, yc))
        pos = posc.to('cuda') if posc is not None else None
        with torch.autocast('cuda', dtype=torch.float16):
            loss = model(x, pr, ct, pos).float().mean()
        scaler.scale(loss).backward()
        ot = []
        for _ in range(8):
            sync()
            t1 = time.perf_counter()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update_parameters(model)
            sync()
            ot.append(time.perf_counter() - t1)
        opt.zero_grad(set_to_none=True)
        rec['opt_s'] = [round(s, 4) for s in ot]
        rec['opt_med_s'] = round(statistics.median(ot), 4)
        rec['sm_share'] = pmon_share(w0, w1)
        rec['clock_temp'] = gpu_clock()
        rec['graph_breaks'] = int(sum(counters['graph_break'].values()))
        rec['unique_graphs'] = int(counters['stats'].get('unique_graphs', 0))
        rec['oom'] = False
    except torch.cuda.OutOfMemoryError as e:
        rec['oom'] = True
        rec['peak_alloc_MiB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        rec['err'] = str(e).splitlines()[0][:200]
    finally:
        del model, opt, ema, scaler, sched
        if v['comp'] != 'none':
            torch._dynamo.reset()
        cleanup()
    return rec


# ---------------------------------------------------------------- eval segment (L5): trainer predict() on 2 studies x 24 slices
def eval_segment(v, warm, timed):
    v = dict(v)
    v['S'] = 24
    rec = dict(variant=v['name'], ck=v['ck'], cl=v['cl'], chunk=v['chunk'], B=v['B'], S=24)
    model = None
    try:
        model = build(v)
        model.eval()
        xc, prc, ctc, _, _ = make_inputs(v, train=False)
        torch.cuda.reset_peak_memory_stats()
        bt = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for i in range(warm + timed):
                t1 = time.perf_counter()
                if i == warm:
                    torch.cuda.reset_peak_memory_stats()                  # steady-state peak (after cudnn search)
                    w0 = time.time()
                P = torch.sigmoid(model(xc.to('cuda'), prc.to('cuda'), ctc.to('cuda')).float()).cpu().numpy()
                bt.append(time.perf_counter() - t1)
        w1 = time.time()
        tt = bt[warm:]
        rec['per_study_s'] = round(sum(tt) / (len(tt) * v['B']), 4)
        rec['per_study_med_s'] = round(statistics.median(tt) / v['B'], 4)
        rec['batch_s'] = [round(s, 3) for s in bt]
        rec['peak_alloc_MiB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        rec['sm_share'] = pmon_share(w0, w1)
        rec['clock_temp'] = gpu_clock()
        rec['oom'] = False
    except torch.cuda.OutOfMemoryError as e:
        rec['oom'] = True
        rec['peak_alloc_MiB'] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        rec['err'] = str(e).splitlines()[0][:200]
    finally:
        del model
        cleanup()
    return rec


# ---------------------------------------------------------------- optimizer-only segment (L4, precise): foreach vs fused
def opt_segment(n_rounds, n_per):
    v = parse_variant('base')
    model = build(v)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY))
    bb = [p for n, p in model.named_parameters() if n.startswith('cnn.')]
    hd = [p for n, p in model.named_parameters() if not n.startswith('cnn.')]
    opts = {'foreach': torch.optim.AdamW([{'params': bb, 'lr': 1e-4}, {'params': hd, 'lr': 1e-3}], weight_decay=0.05),
            'fused': torch.optim.AdamW([{'params': bb, 'lr': 1e-4}, {'params': hd, 'lr': 1e-3}], weight_decay=0.05, fused=True)}
    scalers = {k: torch.amp.GradScaler('cuda', init_scale=2.0 ** 10) for k in opts}
    vv = dict(v, B=1, S=2)
    xc, prc, ctc, posc, yc = make_inputs(vv)
    model.train()
    with torch.autocast('cuda', dtype=torch.float16):
        loss = model(xc.cuda(), prc.cuda(), ctc.cuda(), posc.cuda() if posc is not None else None).float().mean()
    loss.backward()                                              # plain grads; the scaler just multiplies/unscales
    res = {k: [] for k in opts}
    parts = {k: {'unscale_clip': [], 'step_update': [], 'ema': []} for k in opts}
    for r in range(n_rounds):
        for k in (('foreach', 'fused') if r % 2 == 0 else ('fused', 'foreach')):
            o, sc = opts[k], scalers[k]
            for i in range(n_per + (2 if r == 0 else 0)):
                sync(); t1 = time.perf_counter()
                sc.unscale_(o); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                sync(); t2 = time.perf_counter()
                sc.step(o); sc.update()
                sync(); t3 = time.perf_counter()
                ema.update_parameters(model)
                sync(); t4 = time.perf_counter()
                if r == 0 and i < 2:
                    continue                                       # state init
                res[k].append(t4 - t1)
                parts[k]['unscale_clip'].append(t2 - t1); parts[k]['step_update'].append(t3 - t2); parts[k]['ema'].append(t4 - t3)
    for k in opts:
        log(dict(variant=f'opt_{k}', n=len(res[k]), opt_med_s=round(statistics.median(res[k]), 5),
                 opt_p10_p90=[round(float(np.percentile(res[k], 10)), 5), round(float(np.percentile(res[k], 90)), 5)],
                 parts_med={p: round(statistics.median(q), 5) for p, q in parts[k].items()},
                 n_params=sum(p.numel() for p in model.parameters()), clock_temp=gpu_clock()))


# ---------------------------------------------------------------- main
print(f'cfg {A.cfg} mode {A.mode} cap {CAP_BYTES / 2**30:.2f} GiB torch {torch.__version__} '
      f'variants {A.variants} reps {A.reps}', flush=True)
try:
    if A.mode == 'opt':
        opt_segment(n_rounds=6, n_per=6)
    else:
        specs = [s for s in A.variants.split(';') if s]
        order_i = 0
        for rep in range(A.reps):
            k = (rep + A.rot) % len(specs)
            rot = specs[k:] + specs[:k]                                 # rotate the order every rep
            for pos_i, s in enumerate(rot):
                if time.time() - T_START > A.budget:
                    log(dict(variant=s, skipped='budget', rep=rep))
                    continue
                v = parse_variant(s)
                if A.mode == 'eval':
                    rec = eval_segment(v, A.warm, A.timed)
                else:
                    tm = A.timed
                    if A.mode == 'train':                              # >= 10 timed micro-steps, whole opt steps
                        acc = max(1, (8 * 4 * DEF_S[A.cfg]) // (v['B'] * 4 * v['S']))
                        tm = max(A.timed, 10)
                        tm = int(math.ceil(tm / acc) * acc)
                    rec = train_segment(v, A.warm, tm, mem_only=(A.mode == 'mem'))
                rec.update(rep=rep + A.rot, order=order_i, pos_in_rep=pos_i)
                order_i += 1
                log(rec)
finally:
    _pmon.terminate()
    print(f'done in {time.time() - T_START:.0f} s', flush=True)
