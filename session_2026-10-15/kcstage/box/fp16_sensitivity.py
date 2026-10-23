# fp16_sensitivity.py (10-15, Claude subagent, kcstage): MODEL for the G3 parity bar (commit fp16 T4 vs box fp32 CPU, max |dp| <= 1e-3):
# the stand-in members on dumped e2e studies, fp32 vs CPU autocast in fp16 (or bf16 if fp16 CPU autocast is unavailable; bf16 has a
# coarser mantissa, so it is a pessimistic bound). Not a GPU measurement: the GPU commit + kc_box_parity.py are the real test.
#   python fp16_sensitivity.py --dump DIR --code CODE --members m1 m2 [--n 2]
import os, sys, glob, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dump', required=True); ap.add_argument('--code', required=True); ap.add_argument('--members', nargs='+', required=True)
ap.add_argument('--n', type=int, default=2); ap.add_argument('--threads', type=int, default=10)
A = ap.parse_args()
sys.path.insert(0, A.code)
import torch
torch.set_num_threads(A.threads)
import kc_common as kc
import kc_worker as kw
ms = kw.load_members(A.members, 'cpu')
res = []
for p in sorted(glob.glob(A.dump + '/*.npz'))[:A.n]:
    z = np.load(p)
    views = {n: (z['views__' + n] if 'views__' + n in z.files else None) for n in kc.VIEW_NAMES}
    x, pr, ct = kc.views_to_tensor(views)
    x = torch.from_numpy(x)[None]; pr = torch.from_numpy(pr)[None]; ct = torch.from_numpy(ct)[None]
    with torch.no_grad():
        p32 = np.mean([torch.sigmoid(m(x, pr, ct).float()).numpy()[0] for m in ms], 0)
        out = {}
        for dt in (torch.float16, torch.bfloat16):
            try:
                with torch.autocast('cpu', dtype=dt):
                    q = np.mean([torch.sigmoid(m(x, pr, ct).float()).numpy()[0] for m in ms], 0)
                out[str(dt).replace('torch.', '')] = float(np.abs(q - p32).max())
            except Exception as e:
                out[str(dt).replace('torch.', '')] = 'n/a %s: %s' % (type(e).__name__, str(e)[:80])
    res.append({'sid': os.path.basename(p)[:-4], 'p32_range': [float(p32.min()), float(p32.max())], 'max_dp': out})
    print(json.dumps(res[-1]), flush=True)
print('FP16_SENSITIVITY ' + json.dumps(res))
