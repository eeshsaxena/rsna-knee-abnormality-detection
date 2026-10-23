#!/usr/bin/env python
# local_proxy_check.py (10-21 lastday, Claude). LAPTOP ONLY, RSNA gold58 data only (no OAI). Read-only on every source; writes only
# into lastday/. (1) Replays the REAL cell 18 of the c4, pw25 and swide notebooks on the gold58 serves (as val/ext_a22.py) and checks
# that the closed-form proxy used by box_oaihead.py (c4 stage-3 table -> .8/.2 S6fall -> pw E-OA -> lomm05 -> MCLmm) matches them.
# (2) Writes the gold58 serve payload (RSNA ids + R6 / S6fall / Raptor r384 / r224 arrays) as lastday/g58_payload.npz, which
# box_oaihead.py embeds (base64) so nothing is copied to the box as a file.
import json, os, sys, hashlib, tempfile, shutil, base64, io
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; OUT = R + '/session_2026-10-21/lastday'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6s = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4T = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
VC4 = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4T.get(c, .5); VC4[:, j] = w * (.75 * pct(R384[:, j]) + .25 * pct(R224[:, j])) + (1 - w) * pct(R6s[:, j])
NB = {'c4': R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb',
      'pw25': R + '/session_2026-10-20/val/kernels/kernel_v224r6c4pw25j/rsna_sub_v224r6c4pw25j.ipynb',
      'swide': R + '/session_2026-10-20/stack/kernels/kernel_v224r6c4pw25lomm05mcl10j/rsna_sub_v224r6c4pw25lomm05mcl10j.ipynb'}


def run_gold(p, Vb):
    nb = json.load(open(p, encoding='utf-8')); fn = src(nb['cells'][18]).split('\n\ntry:\n    _S6F_BLEND = ')[0]
    W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    pd.DataFrame(Vb.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
    import contextlib
    ns = {}; exec(fn, ns)
    with contextlib.redirect_stdout(io.StringIO()):
        ns['_s6f_blend'](setup, run)
    out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}); shutil.rmtree(W, ignore_errors=True)
    return out[L].values.astype(np.float64)


Wp = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}


def proxy(V, S, a=0.25, loamm=0.05, mclmm=0.1):
    V32 = V.astype(np.float32).astype(np.float64)    # the notebook reads the float32 csv
    out = {c: (.8 * pct(V32[:, j]) + .2 * pct(S[:, j])).astype(np.float32).astype(np.float64) for j, c in enumerate(L)}
    if a:
        U = {c: pct(out[c]) for c in OA}
        for c in OA:
            o = [k for k in OA if k != c]; w = [Wp[tuple(sorted((c, k), key=OA.index))] for k in o]
            out[c] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]])).astype(np.float32).astype(np.float64)
    if loamm:
        out['Lateral OA'] = pct(pct(out['Lateral OA']) - loamm * pct(out['Medial Meniscus'])).astype(np.float32).astype(np.float64)
    if mclmm:
        out['MCL'] = pct(pct(out['MCL']) - mclmm * pct(out['Medial Meniscus'])).astype(np.float32).astype(np.float64)
    return np.stack([out[c] for c in L], 1)


ok = True
for k, kw in (('c4', dict(a=0, loamm=0, mclmm=0)), ('pw25', dict(a=.25, loamm=0, mclmm=0)), ('swide', dict(a=.25, loamm=.05, mclmm=.1))):
    real = run_gold(NB[k], VC4); mine = proxy(VC4, S6, **kw)
    md = np.abs(real - mine).max(); rk = all((rankdata(real[:, j]) == rankdata(mine[:, j])).all() for j in range(12))
    print('%-6s notebook sha %s | cell-18 replay vs proxy: max |diff| %.2e, ranks identical in all 12 labels: %s' % (k, sha(NB[k])[:8], md, rk))
    ok = ok and md < 1e-6 and rk
print('PROXY_CHECK', 'PASS' if ok else 'FAIL')
buf = io.BytesIO(); np.savez_compressed(buf, ids=np.array(ids), R6s=R6s, S6=S6, R384=R384, R224=R224)
open(OUT + '/g58_payload.npz', 'wb').write(buf.getvalue())
print('payload bytes', len(buf.getvalue()), 'sha256', hashlib.sha256(buf.getvalue()).hexdigest()[:16])
