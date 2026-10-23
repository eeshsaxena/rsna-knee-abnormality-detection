#!/usr/bin/env python3
# Adversarial verifier (box CPU, harness only): rebuild pass A / pass B for J1 on the 20 parity studies with an
# INDEPENDENT numpy implementation (own triplet sampler, own flips; no fleet_model.mirror_views, no tta_bench
# make_input) from the worker's own decode (fc.build_study on the DICOMs), forward J1 fp32 CPU, and compare with the
# K5 worker's npz (PA, PB, P). Also shows that ONLY "offset +HALF then mirror" matches PB (not mirror-only, offset-only,
# mirror-then-offset with +HALF, or exact .5).
import os, sys, json, numpy as np, pandas as pd, torch
R = sys.argv[1]                       # ~/k5_adv/parity/cpu
J1 = sys.argv[2]
torch.set_num_threads(4)
sys.path.insert(0, f'{R}/tb_code')
import fleet_common as fc
import tta_bench as tb                 # only load_member (== worker member build)
ids = json.load(open(f'{R}/ids.json'))['ids20']
comp = f'{R}/comp20'
ts = pd.read_csv(f'{comp}/test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
by = {s: g for s, g in ts.groupby('StudyInstanceUID')}
z = np.load(f'{R}/new20_j1ab.npz')
names = [str(x) for x in z['names']]; j = names.index(os.path.basename(J1))
row = {str(s): k for k, s in enumerate(z['sids'])}
m, info = tb.load_member(J1, 'cpu')
PLANES = [p for p, _ in fc.VIEWS]
NS, ST = fc.N_SLICES, fc.STORE
HALF = 0.5 - 1e-3

def my_sample(stack, off):
    n = stack.shape[0]
    c = np.clip(np.rint(np.linspace(0.0, n - 1.0, NS) + off).astype(np.int64), 0, n - 1)
    out = np.empty((NS, 3, stack.shape[1], stack.shape[2]), np.uint8)
    for s in range(NS):
        out[s, 0] = stack[max(c[s] - 1, 0)]; out[s, 1] = stack[c[s]]; out[s, 2] = stack[min(c[s] + 1, n - 1)]
    return out

def my_mirror(v, plane):
    if plane == 'Sagittal':
        return np.ascontiguousarray(v[::-1, ::-1])          # slice order AND triplet channel order reversed
    return np.ascontiguousarray(v[..., ::-1])              # left-right (W) flip

def build(st, off, mirror, mirror_first=False):
    vs, pr, ct = [], [], []
    for jv, g in enumerate(st):
        if g is None:
            vs.append(np.zeros((NS, 3, ST, ST), np.uint8)); pr.append(False); ct.append(3); continue
        stack = g[0]
        if mirror and mirror_first:                         # wrong-order variant: mirror the raw stack, then offset
            stack = stack[::-1] if PLANES[jv] == 'Sagittal' else stack[..., ::-1]
            v = my_sample(np.ascontiguousarray(stack), off)
        else:
            v = my_sample(stack, off)
            if mirror:
                v = my_mirror(v, PLANES[jv])
        vs.append(v); pr.append(True); ct.append(int(g[1]))
    return torch.from_numpy(np.stack(vs)[None]), torch.tensor([pr]), torch.tensor([ct])

def fwd(x, pr, ct):
    with torch.no_grad():
        return torch.sigmoid(m(x, pr, ct).float()).numpy()[0]

res = dict(dA=0.0, dB=0.0, dP=0.0, n=0, absent_views=0, n24=0, nviews=0, slices=[], alt={})
alts = {'mirror_only': (0.0, True, False), 'offset_only': (HALF, False, False), 'exact_half_then_mirror': (0.5, True, False),
        'mirror_then_offset': (HALF, True, True), 'neg_half_then_mirror': (-HALF, True, False)}
for k in alts: res['alt'][k] = 0.0
for sid in ids:
    st = fc.build_study(by.get(sid, ts.iloc[0:0]), f'{comp}/test_series', sid, stats={})
    st = [st[v] for v in fc.VIEWS]
    for g in st:
        if g is None: res['absent_views'] += 1
        else:
            res['nviews'] += 1; res['slices'].append(int(g[0].shape[0])); res['n24'] += int(g[0].shape[0] == 24)
    if not any(g is not None for g in st):
        continue
    pa = fwd(*build(st, 0.0, False)); pb = fwd(*build(st, HALF, True))
    k = row[sid]
    wA, wB, wP = z['PA'][j, k], z['PB'][j, k], z['P'][j, k]
    res['dA'] = max(res['dA'], float(np.abs(pa - wA).max())); res['dB'] = max(res['dB'], float(np.abs(pb - wB).max()))
    res['dP'] = max(res['dP'], float(np.abs((pa + pb) / 2 - wP).max()))
    for name, (off, mir, mf) in alts.items():
        res['alt'][name] = max(res['alt'][name], float(np.abs(fwd(*build(st, off, mir, mf)) - wB).max()))
    res['n'] += 1
    print(sid[-10:], 'views', [None if g is None else g[0].shape[0] for g in st], 'dA %.2g dB %.2g' % (
        float(np.abs(pa - wA).max()), float(np.abs(pb - wB).max())), flush=True)
res['slices'] = sorted(set(res['slices']))
print(json.dumps(res, indent=1))
ok = res['dA'] <= 1e-6 and res['dB'] <= 1e-6 and res['dP'] <= 1e-6 and all(v > 1e-3 for k_, v in res['alt'].items()
                                                                              if k_ != 'exact_half_then_mirror')
print('INDEP_PASSB', 'PASS' if ok else 'FAIL')
