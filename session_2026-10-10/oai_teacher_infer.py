#!/usr/bin/env python
# oai_teacher_infer.py (2026-10-10, Claude): CPU teacher predictions of RSNA-only fleet fold checkpoints on the OAI V00
# knees (box only; OAI data and these predictions NEVER leave the box). = ~/novel/gold_infer5.py's input path and member
# build (tta_bench.load_member / study_stacks / make_input / forward, pass A = offset 0, no mirror), with a different
# store and id list. Purpose: soft targets for the OAI-blank labels (MCL, Contusion, Fracture) of a possible later
# T123+PL probe; no decision is taken from these predictions without a pre-registration.
# Sharding: --shard i --nshard n (each process writes its own csv). fp32 CPU, nice 19, CUDA hidden.
# usage: CUDA_VISIBLE_DEVICES= NTHR=5 nice -n 19 python oai_teacher_infer.py --ids ids.txt --shard 0 --nshard 4 ckpt1 ...
import os, sys, time, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
H = os.path.expanduser
sys.path.insert(0, H('~/novel/code'))
import numpy as np, pandas as pd, torch
torch.set_num_threads(int(os.environ.get('NTHR', '5')))
import tta_bench as tb

ap = argparse.ArgumentParser()
ap.add_argument('ckpts', nargs='+')
ap.add_argument('--store', default=H('~/oai_store'))
ap.add_argument('--ids', required=True)                      # one OAI id per line (e.g. oai_9000001_L)
ap.add_argument('--out', default=H('~/oai_build/teacher'))
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshard', type=int, default=1)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
ids = [l.strip() for l in open(A.ids) if l.strip()]
assert all(s.startswith('oai_') for s in ids), 'non-OAI id in --ids'
ids = [s for s in ids if os.path.exists(f'{A.store}/{s}.npz')][A.shard::A.nshard]
print(f'shard {A.shard}/{A.nshard}: {len(ids)} OAI studies, threads {torch.get_num_threads()}', flush=True)
t0 = time.time()
for p in A.ckpts:
    run = os.path.basename(os.path.dirname(p)); tag = 'ema' if '_ema_' in os.path.basename(p) else 'raw'
    dst = f'{A.out}/{run}__{tag}__A__shard{A.shard}of{A.nshard}.csv'
    if os.path.exists(dst):
        print('skip', dst, flush=True); continue
    m, info = tb.load_member(p, 'cpu')
    P = np.full((len(ids), 12), np.nan, np.float32); t1 = time.time()
    for k, s in enumerate(ids):
        st = tb.study_stacks(A.store, s)
        x, pr, ct, nv = tb.make_input(st, 0.0, False, 'cpu')
        if nv:
            P[k] = tb.forward(m, x, pr, ct, 'cpu')
        if k and k % 200 == 0:
            print(f'  {run} {k}/{len(ids)} {(time.time() - t1) / k:.2f}s/study', flush=True)
    tmp = dst + '.tmp'
    pd.DataFrame(P, columns=tb.TARGETS, index=pd.Index(ids, name='StudyInstanceUID')).to_csv(tmp); os.replace(tmp, dst)
    print(f'{run} {tag} ep{info["epoch_1based"]} cv {info["cv"]:.4f} | {len(ids)} studies {time.time() - t1:.0f}s -> {dst}', flush=True)
    del m
print(f'done {time.time() - t0:.0f}s', flush=True)
