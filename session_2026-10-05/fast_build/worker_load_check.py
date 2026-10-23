# worker_load_check.py -- strict-load checkpoints through the REAL loader block of ~/bigjump_build/code/fleet_worker.py
# (source text from "def _model_sha(mod='fleet_model'):" through "    del ck", exec'd verbatim on CPU with the worker's
# own module imports; no bytecode written). Then: loaded state_dict keys/shapes == J1 (fold-0 TTR mix50) keys/shapes,
# loaded tensors == ckpt tensors, and the ckpt args differ from J1's only in the expected keys.
import os, sys, json
sys.dont_write_bytecode = True
CODE = os.path.expanduser('~/bigjump_build/code')
sys.path.insert(0, CODE)
import numpy as np, torch
import fleet_common as fc
from fleet_model import KneeFleetNet

J1 = os.path.expanduser('~/fleet_runs/f0_s1043_ttr_mix50/f0_s1043_raw_ttr224-320x24_cv8835_best.pt')
cks = sys.argv[1:]
src = open(f'{CODE}/fleet_worker.py').read()
a0 = src.index("def _model_sha(mod='fleet_model'):")
a1 = src.index('    del ck\n', a0) + len('    del ck\n')
block = src[a0:a1]
assert 'load_state_dict' in block and 'assert not miss and not unexp' in block
ns = dict(torch=torch, os=os, np=np, fc=fc, KneeFleetNet=KneeFleetNet, dev='cpu', TAG='[wchk]', order=list(cks))
exec(compile(block, f'{CODE}/fleet_worker.py[loader block]', 'exec'), ns)
assert len(ns['models']) == len(cks)
j1 = torch.load(J1, map_location='cpu', weights_only=False)
j1sd, j1a = j1['model'], j1['args']
rep = []
for p, m in zip(cks, ns['models']):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = m.state_dict()
    same_keys_j1 = list(sd) == list(j1sd) and all(sd[k].shape == j1sd[k].shape for k in sd)
    same_keys_ck = list(ck['model']) == list(j1sd)
    exact = all(torch.equal(sd[k].cpu(), ck['model'][k].cpu().to(sd[k].dtype)) for k in sd)
    a = ck['args']
    diff_args = sorted(k for k in set(a) | set(j1a) if json.dumps(a.get(k), default=str) != json.dumps(j1a.get(k), default=str))
    rep.append(dict(ckpt=os.path.basename(p), cls=type(m).__name__, n_keys=len(sd), keys_shapes_eq_J1=same_keys_j1,
                    ckpt_keys_eq_J1=same_keys_ck, loaded_eq_ckpt=exact, args_diff_vs_J1=diff_args,
                    grad_ckpt_arg=a.get('grad_ckpt', 'absent'), model_sha=a.get('model_sha')))
    print('WCHK ' + json.dumps(rep[-1]), flush=True)
print('WCHK_ALL_OK', all(r['keys_shapes_eq_J1'] and r['ckpt_keys_eq_J1'] and r['loaded_eq_ckpt'] for r in rep), flush=True)
