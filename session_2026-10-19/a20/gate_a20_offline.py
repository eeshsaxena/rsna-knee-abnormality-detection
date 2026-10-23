# gate_a20_offline.py (10-19, Claude): run the EXACT cell-18 _s6f_blend code of each built notebook (c4, eoa00, eoa20) on the gold58
# serve predictions (V224 stage 3 rebuilt as in m14_gate.py; S6fall = n42gpu) in a temp working dir; compare outputs and gold58 macro.
# Gates: G-a eoa00 submission.csv byte-identical to c4's; G-b eoa20 gold58 macro == .94341 (m14) and non-OA columns identical to c4.
import json, os, sys, hashlib, tempfile, shutil
import numpy as np, pandas as pd
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
from scipy.stats import rankdata
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
V = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); V[:, j] = w * rr + (1 - w) * pct(R6[:, j])
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
NB = {'c4': R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb',
      'eoa00': R + '/session_2026-10-19/a20/kernel_v224r6c4eoa00j/rsna_sub_v224r6c4eoa00j.ipynb',
      'eoa20': R + '/session_2026-10-19/a20/kernel_v224r6c4eoa20j/rsna_sub_v224r6c4eoa20j.ipynb',
      'eoa10': R + '/session_2026-10-19/a20/kernel_v224r6c4eoa10j/rsna_sub_v224r6c4eoa10j.ipynb'}
res = {}
for k, p in NB.items():
    nb = json.load(open(p, encoding='utf-8')); s = nb['cells'][18]['source']; s = ''.join(s) if isinstance(s, list) else s
    fn = s.split('\n\ntry:\n    _S6F_BLEND = ')[0]
    W = tempfile.mkdtemp().replace(chr(92), '/'); assert fn.count("W = '/kaggle/working'") == 1
    fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
    ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](setup, run)
    out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
    res[k] = dict(rec=rec, bytes=open(W + '/submission.csv', 'rb').read(), out=out)
    m = np.mean([auc(out[c].values, Y[:, j]) for j, c in enumerate(L)])
    print('%-6s mode %s a_oa %s | gold58 macro %.5f | OA AUCs %s' % (k, rec.get('mode'), rec.get('a_oa'), m,
          [round(auc(out[c].values, Y[:, L.index(c)]), 4) for c in ('Medial OA', 'Lateral OA', 'PF OA')]))
    shutil.rmtree(W, ignore_errors=True)
ok_a = res['eoa00']['bytes'] == res['c4']['bytes']
non = [c for c in L if c not in ('Medial OA', 'Lateral OA', 'PF OA')]
ok_c = (res['eoa20']['out'][non].values == res['c4']['out'][non].values).all()
m20 = np.mean([auc(res['eoa20']['out'][c].values, Y[:, j]) for j, c in enumerate(L)])
mc4 = np.mean([auc(res['c4']['out'][c].values, Y[:, j]) for j, c in enumerate(L)])
# kernel code casts to float32 (ties), so the c4 base is .94113 here vs m14's float64 .94102; gate on the DELTA (m14: +.00239)
ok_b = abs((m20 - mc4) - 0.00239) < 2e-4 and res['eoa20']['rec'].get('a_oa') == 0.2 and res['eoa20']['rec'].get('mode') == 'S6F'
print('G-a eoa00 byte-identical to c4:', ok_a)
print('G-b eoa20 gold58 macro %.5f, delta vs c4 (same code path) %+.5f ~= m14 +.00239 (+-.0002), mode S6F, a_oa .2:' % (m20, m20 - mc4), ok_b)
print('G-c eoa20 non-OA columns identical to c4:', ok_c)
m10 = np.mean([auc(res['eoa10']['out'][c].values, Y[:, j]) for j, c in enumerate(L)])
ok_d = abs((m10 - mc4) - 0.00142) < 2e-4 and res['eoa10']['rec'].get('a_oa') == 0.1 and (res['eoa10']['out'][non].values == res['c4']['out'][non].values).all()
print('G-d eoa10 gold58 macro %.5f, delta vs c4 %+.5f ~= m14 +.00142 (+-.0002), a_oa .1, non-OA identical:' % (m10, m10 - mc4), ok_d)
print('A20_OFFLINE_GATE', 'PASS' if ok_a and ok_b and ok_c and ok_d else 'FAIL')
