# compare_tiny.py -- compare the end-to-end tiny trainer runs (same recipe flags, 16 train / 8 val studies, 2 epochs =
# 4 optimizer steps at accum 4): saved raw + EMA weights (epoch 1 and 2), OOF csvs, ckpt args, logged loss / val AUC.
#   ref    = pristine copy of ~/bigjump_build/code/fleet_train.py, default flags
#   c2def  = ~/fast_build/code2/fleet_train.py, default flags (must equal ref)
#   c2gc0  = ~/fast_build/code2/fleet_train.py --grad_ckpt 0 (model math must equal c2def up to kernel noise)
import os, sys, re, json, glob
import numpy as np, pandas as pd, torch
T = os.path.expanduser('~/fast_build/tiny')
pairs = [p.split(':') for p in sys.argv[1:]]          # e.g. ref_det:c2def_det c2def_det:c2gc0_det


def load(run, tag, ep):
    p = glob.glob(f'{T}/{run}/epochs/f2_s1043_{tag}_ttr224-320x24_ep{ep}_best.pt')
    assert len(p) == 1, (run, tag, ep, p)
    return torch.load(p[0], map_location='cpu', weights_only=False)


def logvals(run):
    txt = open(f'{T}/{run}.log').read()
    loss = [float(x) for x in re.findall(r' it 0/\d+ loss ([0-9.]+)', txt)]
    auc = [float(x) for x in re.findall(r'val AUC vs \S+ ([0-9.]+)', txt)]
    return loss, auc


out = []
for a, b in pairs:
    r = dict(pair=f'{a} vs {b}')
    for ep in (1, 2):
        for tag in ('raw', 'ema'):
            ca, cb = load(a, tag, ep), load(b, tag, ep)
            sa, sb = ca['model'], cb['model']
            assert list(sa) == list(sb)
            md = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa if sa[k].numel())
            r[f'ep{ep}_{tag}_maxdiff'] = md
            r[f'ep{ep}_{tag}_bitwise'] = all(torch.equal(sa[k], sb[k]) for k in sa)
            r[f'ep{ep}_{tag}_cv'] = (ca['cv'], cb['cv'])
            if ep == 2 and tag == 'raw':
                aa, ab = ca['args'], cb['args']
                r['args_diff'] = sorted(k for k in set(aa) | set(ab)
                                        if json.dumps(aa.get(k), default=str) != json.dumps(ab.get(k), default=str))
                r['model_sha'] = (aa.get('model_sha'), ab.get('model_sha'))
            oa = f'{T}/{a}/epochs/oof_f2_s1043_{tag}_ep{ep}.csv'; ob = f'{T}/{b}/epochs/oof_f2_s1043_{tag}_ep{ep}.csv'
            pa, pb = pd.read_csv(oa).iloc[:, 1:].to_numpy(), pd.read_csv(ob).iloc[:, 1:].to_numpy()
            r[f'ep{ep}_{tag}_oof_maxdiff'] = float(np.abs(pa - pb).max())
    la, ua = logvals(a); lb, ub = logvals(b)
    r['log_loss_it0'] = (la, lb); r['log_val_auc'] = (ua, ub)
    out.append(r)
    print('TINY ' + json.dumps(r), flush=True)
