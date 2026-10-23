# gate_r6eoa_offline.py (10-20 push, Claude): offline gate for the LOCAL R6 + E-OA builds (nothing pushed). Runs the EXACT cell-18
# _s6f_blend code of each built notebook (R6, R6e00 control, R6e20) on the gold58 serves (stage 3 rebuilt with R6's cell-14 params:
# uniform W .5, A224 .35; S6fall = n42gpu), the method of a20/gate_a20_offline.py.
# Gates: G-a R6e00 submission.csv byte-identical to R6's; G-b R6e20 gold58 delta vs R6 == +.00218 (ex_r6eoa_gate.log) +-.0002, a_oa .2,
# mode S6F; G-c non-OA columns identical to R6; G-d R6e20 OA columns == pct transform of R6's OA columns computed independently here.
import json, os, hashlib, tempfile, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; P = R + '/session_2026-10-20/push'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6s = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
V = np.zeros((n, 12))
for j, c in enumerate(L):
    rr = .65 * pct(R384[:, j]) + .35 * pct(R224[:, j]); V[:, j] = .5 * rr + .5 * pct(R6s[:, j])
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
NB = {'R6': R + '/session_2026-10-15/r6entry/kernel_v224r6flips6fj/rsna_sub_v224r6flips6fj.ipynb',
      'R6e00': P + '/kernel_v224r6flips6feoa00j/rsna_sub_v224r6flips6feoa00j.ipynb',
      'R6e20': P + '/kernel_v224r6flips6feoa20j/rsna_sub_v224r6flips6feoa20j.ipynb'}
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
    res[k]['m'] = m
    print('%-6s mode %s a_oa %s | gold58 macro %.5f | OA AUCs %s' % (k, rec.get('mode'), rec.get('a_oa'), m,
          [round(auc(out[c].values, Y[:, L.index(c)]), 4) for c in OA]))
    shutil.rmtree(W, ignore_errors=True)
non = [c for c in L if c not in OA]
ok_a = res['R6e00']['bytes'] == res['R6']['bytes']
d = res['R6e20']['m'] - res['R6']['m']
ok_b = abs(d - 0.00218) < 2e-4 and res['R6e20']['rec'].get('a_oa') == 0.2 and res['R6e20']['rec'].get('mode') == 'S6F'
ok_c = (res['R6e20']['out'][non].values == res['R6']['out'][non].values).all()
U = {c: pct(res['R6']['out'][c].values.astype(np.float64)) for c in OA}
ind = {c: pct(U[c] - 0.2 * sum(U[o] for o in OA if o != c) / 2.0).astype(np.float32) for c in OA}
ok_d = all(np.allclose(res['R6e20']['out'][c].values, ind[c], atol=1e-6) for c in OA)
print('G-a R6e00 byte-identical to R6:', ok_a)
print('G-b R6e20 gold58 delta vs R6 %+.5f ~= +.00218 (+-.0002), a_oa .2, mode S6F:' % d, ok_b)
print('G-c R6e20 non-OA columns identical to R6:', ok_c)
print('G-d R6e20 OA columns == independent E-OA transform of R6 output:', ok_d)
print('R6EOA_OFFLINE_GATE', 'PASS' if ok_a and ok_b and ok_c and ok_d else 'FAIL')
