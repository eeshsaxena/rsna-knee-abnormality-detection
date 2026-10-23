#!/usr/bin/env python
# rm_runtime.py (2026-10-14, Claude subagent): hidden-run cost of the r224 mirror TTA, from the commit's measured per-pass times.
#   usage: python rm_runtime.py OUT_RM_COMMIT [--n 1200 1300]
# Model (MODEL unless marked): her r224 stage = one process, 4 prep procs decode every study (her fastread; the full hidden decode
# is <= ~7 min, PUBLIC949_ASSESSMENT.md, INFERRED from the 7.2 min turnaround of nartaa's efficiency run with the same reader)
# streaming into 2 GPU consumer threads (one per T4, batch of 1 study). Stage wall ~= load + max(decode, GPU), GPU = N / 2 x
# per-study GPU seconds. Plain = 1 pass; RM = 2 passes (+ the mirrored window build). Per-pass seconds = the commit's WARM passes
# (every mirrored pass is warm: same shapes as the plain pass just before it; the first plain pass per GPU pays cudnn autotune).
# A cross-check comes from a FLOP scale of the MEASURED r384 rate (CoAtNet-2 384 px fp16 1.80-1.86 s per 94 windows on a T4,
# SP/parent_speed/PROFILE.md): 224 / 384 squared = .34 -> ~.62 s per 94-window pass.
import sys, json, os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--n', type=int, nargs='+', default=[1200, 1300])
A = ap.parse_args()
r = json.load(open(os.path.join(A.out, 'raptor224_tta_receipt.json')))
L = np.array(r['per_study_s_list'], dtype=np.float64)          # rows: [total, plain, mirror] per study (consumer order)
warm_mirror = L[:, 2]
warm_plain = L[1:, 1] if len(L) > 2 else L[:, 1]                # drop the first plain pass (autotune) when there are >= 3 rows
pass_s = float(np.median(np.concatenate([warm_mirror, warm_plain])))
build_s = float(np.median(np.maximum(L[:, 0] - L[:, 1] - L[:, 2], 0)))   # window build + bookkeeping, both passes
flop_pass = 1.83 * (224 / 384) ** 2                    # MEASURED r384-arch rate per 94 windows at 384 px, FLOP-scaled to 224 px
out = {'commit_per_study_s': L.round(3).tolist(), 'warm_pass_s_commit': round(pass_s, 3), 'build_s_commit': round(build_s, 3),
       'flop_scaled_pass_s': round(flop_pass, 3)}
rows = []
for n in A.n:
    for nm, ps in (('commit', pass_s), ('flop', flop_pass)):
        for slow in (1.0, 1.3):
            dec = 7.0 * 60 * n / 1250 * slow                     # decode wall (4 procs), upper-bound style
            gp = n / 2 * (ps + build_s / 2) * slow                   # plain: 1 pass per study, 2 GPUs
            gm = n / 2 * (2 * ps + build_s) * slow                   # RM: 2 passes per study
            plain = 20 + max(dec, gp); rm = 20 + max(dec, gm)
            rows.append({'n': n, 'pass_from': nm, 'slow': slow, 'plain_min': round(plain / 60, 1), 'rm_min': round(rm / 60, 1),
                         'extra_min': round((rm - plain) / 60, 1)})
out['rows'] = rows
ex = [x['extra_min'] for x in rows]
out['extra_min_range'] = [min(ex), max(ex)]
out['extra_h_central'] = round(float(np.median([x['extra_min'] for x in rows if x['slow'] == 1.0])) / 60, 3)
print('RM_RUNTIME', json.dumps(out, indent=1))
