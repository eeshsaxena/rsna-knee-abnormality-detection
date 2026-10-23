# gate_r6pw_offline.py (10-20 val Track D, Claude; adapted from push/gate_r6eoa_offline.py): offline gate for the LOCAL R6 + pw E-OA builds (nothing pushed). Runs the EXACT cell-18
# _s6f_blend code of each built notebook (R6, R6pw00 control, R6pw20, R6pw30) on the gold58 serves (stage 3 rebuilt with R6's cell-14
# params: uniform W .5, A224 .35; S6fall = n42gpu), the method of a20/gate_a20_offline.py.
# Gates: G-a R6pw00 submission.csv byte-identical to R6's; G-b R6pwXX gold58 delta vs R6 == the post-hoc target in r6transfer_pw.log
# +-.0002, a_oa, oa_mode pw, mode S6F; G-c non-OA columns identical to R6; G-d OA columns == independent pw transform of R6's output.
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
B = R + '/session_2026-10-20/val/picks2_work/build'
NB = {'R6': R + '/session_2026-10-15/r6entry/kernel_v224r6flips6fj/rsna_sub_v224r6flips6fj.ipynb',
      'R6pw00': B + '/kernel_v224r6flips6fpw00j/rsna_sub_v224r6flips6fpw00j.ipynb',
      'R6pw20': B + '/kernel_v224r6flips6fpw20j/rsna_sub_v224r6flips6fpw20j.ipynb',
      'R6pw30': B + '/kernel_v224r6flips6fpw30j/rsna_sub_v224r6flips6fpw30j.ipynb'}
TGT = {'R6pw20': (0.2, 0.00215), 'R6pw30': (0.3, 0.00191)}   # gold58 targets, MEASURED val/picks2_work/r6transfer_pw.log (post-hoc transform)
WPW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
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
    m = np.mean([auc(out[c].values, Y[:, j]) for j, c in enumerate(L)]); res[k]['m'] = m
    print('%-7s mode %s a_oa %s oa_mode %s | gold58 macro %.5f | OA AUCs %s' % (k, rec.get('mode'), rec.get('a_oa'), rec.get('oa_mode'), m,
          [round(float(auc(out[c].values, Y[:, L.index(c)])), 4) for c in OA]))
    shutil.rmtree(W, ignore_errors=True)
non = [c for c in L if c not in OA]
ok = {}
ok['G-a R6pw00 byte-identical to R6'] = res['R6pw00']['bytes'] == res['R6']['bytes']
U = {c: pct(res['R6']['out'][c].values.astype(np.float64)) for c in OA}
for k, (a, tgt) in TGT.items():
    d = res[k]['m'] - res['R6']['m']
    ok['G-b %s gold58 delta vs R6 %+.5f ~= %+.5f (+-.0002), a_oa %.1f, oa_mode pw, mode S6F' % (k, d, tgt, a)] = (abs(d - tgt) < 2e-4 and res[k]['rec'].get('a_oa') == a
        and res[k]['rec'].get('oa_mode') == 'pw' and res[k]['rec'].get('mode') == 'S6F')
    ok['G-c %s non-OA columns identical to R6' % k] = bool((res[k]['out'][non].values == res['R6']['out'][non].values).all())
    ind = {}
    for c in OA:
        o = [q for q in OA if q != c]; w = [WPW[tuple(sorted((c, q), key=OA.index))] for q in o]
        ind[c] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]])).astype(np.float32)
    ok['G-d %s OA columns == independent pw transform of R6 output' % k] = all(np.allclose(res[k]['out'][c].values, ind[c], atol=1e-6) for c in OA)
for k, v in ok.items(): print(k + ':', v)
print('R6PW_OFFLINE_GATE', 'PASS' if all(ok.values()) else 'FAIL')
