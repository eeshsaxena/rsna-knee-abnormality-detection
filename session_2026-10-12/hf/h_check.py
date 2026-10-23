#!/usr/bin/env python
# h_check.py (2026-10-12, Claude): commit gate for the HV / HF kernels (PREREG_10-12.md). Reads a fetched kernel output
# dir (kaggle kernels output <kernel> -p DIR). Prints HCHECK PASS/FAIL n/m; exit 0 only on PASS.
#   usage: python -I h_check.py DIR --kind hv|hf [--wf 0.20] [--members 5]
import os, sys, json, glob, argparse
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('--kind', choices=['hv', 'hf'], required=True)
ap.add_argument('--wf', type=float, default=None); ap.add_argument('--members', type=int, default=None)
A = ap.parse_args()
sys.stdout.reconfigure(encoding='utf-8')
R = []
def chk(name, ok, info=''):
    R.append(bool(ok)); print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)

logs = [p for p in glob.glob(os.path.join(A.dir, '*.log')) if 'cufile' not in p]
text = ''
if logs:
    try:
        text = '\n'.join(e.get('data', '') for e in json.load(open(logs[0], encoding='utf-8')))
    except Exception:
        text = open(logs[0], encoding='utf-8', errors='replace').read()
chk('kernel log present', bool(text), logs[:1])
chk('NOTEBOOK_COMPLETE (H stage 3)', 'NOTEBOOK_COMPLETE' in text)
chk('no Traceback in log', 'Traceback (most recent call last)' not in text)
sp = os.path.join(A.dir, 'submission.csv')
chk('submission.csv exists', os.path.isfile(sp))
if os.path.isfile(sp):
    s = pd.read_csv(sp, dtype={'StudyInstanceUID': str}); lab = s.columns[1:]
    chk('13 columns', s.shape[1] == 13, s.shape)
    chk('rows >= 1, unique ids', len(s) >= 1 and s['StudyInstanceUID'].is_unique, len(s))
    v = s[lab].values.astype(float)
    chk('values finite in [0,1]', np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all())
br = os.path.join(A.dir, 'blend_receipt.json')
if os.path.isfile(br):
    b = json.load(open(br))
    chk('H receipt: raptor_failed 0, ours_failed 0', b.get('raptor_failed') == 0 and b.get('ours_failed') == 0,
        (b.get('raptor_failed'), b.get('ours_failed')))
    chk('H weights .5/.5', b.get('weights', {}).get('raptor') == 0.5 and b.get('weights', {}).get('ours') == 0.5)
else:
    chk('blend_receipt.json exists', False)
if A.kind == 'hf':
    chk('fleet stage clean (RESULT OK fleet stage clean)', 'RESULT OK fleet stage clean' in text)
    if A.members is not None:
        chk(f'fleet ran {A.members} members', f'RESULT OK fleet stage clean: {A.members} member' in text)
    chk('HF_ENV_RESTORED', 'HF_ENV_RESTORED' in text)
    chk('HF_BLEND_OK (no HF_FALLBACK)', 'HF_BLEND_OK' in text and 'HF_FALLBACK' not in text)
    hp = os.path.join(A.dir, 'hf_receipt.json')
    if os.path.isfile(hp):
        h = json.load(open(hp))
        chk('hf_receipt mode HF', h.get('mode') == 'HF', h.get('mode'))
        if A.wf is not None:
            chk(f'hf_receipt w_f == {A.wf}', abs(float(h.get('w_f', -1)) - A.wf) < 1e-9, h.get('w_f'))
        # independent recompute: submission == (1-w) pct(H) + w pct(F)
        H = pd.read_csv(os.path.join(A.dir, 'submission_H.csv'), dtype={'StudyInstanceUID': str})
        F = pd.read_csv(os.path.join(A.dir, 'submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[H['StudyInstanceUID']]
        S = pd.read_csv(sp, dtype={'StudyInstanceUID': str})
        lab = H.columns[1:]; w = float(h['w_f'])
        pct = lambda x: pd.Series(np.asarray(x, np.float64)).rank(method='average', pct=True).values
        exp = np.stack([(1 - w) * pct(H[c].values) + w * pct(F[c].values) for c in lab], 1)
        md = float(np.abs(exp - S[lab].values.astype(float)).max())
        chk('submission == independent rank blend', md < 1e-6, f'maxdiff {md:.2e}')
    else:
        chk('hf_receipt.json exists', False)
n = sum(R)
print(f'HCHECK {"PASS" if all(R) else "FAIL"} {n}/{len(R)} ({A.kind})')
sys.exit(0 if all(R) else 1)
