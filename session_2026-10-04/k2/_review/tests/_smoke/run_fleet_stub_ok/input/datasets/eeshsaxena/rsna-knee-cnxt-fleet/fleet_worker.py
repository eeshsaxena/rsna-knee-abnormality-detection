# STUB fleet worker (K2 smoke only). STUB=ok|fastfail|hang|valve|hardstop
import os, sys, time, json
import numpy as np, pandas as pd
mode, shard, nsh = os.environ.get('STUB', 'ok'), int(os.environ['FLEET_SHARD']), int(os.environ['FLEET_NSHARD'])
ids = pd.read_csv(os.environ['RSNA_COMP_ROOT'] + '/test.csv', dtype=str).StudyInstanceUID.tolist()[shard::nsh]
if mode == 'hang':
    time.sleep(600)
if mode == 'fastfail':
    if shard == 1:
        sys.exit(3)
    time.sleep(600)
M = 3 if mode == 'valve' else 2
P = np.random.default_rng(100 + shard).uniform(0.01, 0.99, (M, len(ids), 12)).astype(np.float32)
keep = 1 if mode == 'valve' else M
P[keep:] = np.nan
NV = np.full(len(ids), 4)
if mode == 'blank' and len(ids):
    P[:, 0] = np.nan; NV[0] = 0
if mode == 'views2':
    NV[:] = 2
out = os.environ['FLEET_OUT']; tmp = out[:-4] + '.tmp.npz'
np.savez(tmp, sids=np.array(ids), P=P, NV=NV, names=np.array(['stub_a', 'stub_b', 'stub_c'][:M]),
         n_keep=np.int64(keep), n_fail=np.int64(0), n_blank=np.int64(0), hard_stopped=np.bool_(mode == 'hardstop'),
         minutes=np.float64(0.01), stats=np.array('{}'), parity=np.array('{}'), bench=np.array('{}'),
         loop_min=np.float64(0.01))
os.replace(tmp, out)
