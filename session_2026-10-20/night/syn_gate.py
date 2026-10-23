# syn_gate.py (2026-10-20 night, Claude). LOCAL offline gate for the unarmed E2s builds (pattern of a21/gate_a21_offline.py):
# exec the EXACT cell-18 _s6f_blend code of pw30 and of each syn build on the gold58 serve predictions (V224 stage 3 rebuilt,
# S6fall = n42gpu) in a temp dir. Gates:
#   G1 syn00 submission.csv byte-identical to pw30's (no-op control);
#   G2 syn20: only the Synovitis column differs from pw30; it equals an independent recompute pct(pct(Syn) + .2 pct(Eff)) of pw30's
#      own output (max abs diff < 1e-6); gold58 Synovitis AUC +.0084 vs pw30 (MEASURED syn_more.log on c4 = same Syn / Eff columns).
import json, os, hashlib, tempfile, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; HERE = R + '/session_2026-10-20/night'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
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
NB = {'pw30': R + '/session_2026-10-20/a21/kernel_v224r6c4pw30j/rsna_sub_v224r6c4pw30j.ipynb',
      'syn00': HERE + '/syn_kernel_v224r6c4pw30syn00j/rsna_sub_v224r6c4pw30syn00j.ipynb',
      'syn20': HERE + '/syn_kernel_v224r6c4pw30syn20j/rsna_sub_v224r6c4pw30syn20j.ipynb'}
res = {}
for k, p in NB.items():
    nb = json.load(open(p, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
    fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]
    W = tempfile.mkdtemp().replace(chr(92), '/'); assert fn.count("W = '/kaggle/working'") == 1
    fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
    ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](setup, run)
    res[k] = dict(rec=rec, sha=sha(W + '/submission.csv'),
                  out=pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64))
    shutil.rmtree(W, ignore_errors=True)
    print('%-6s mode %s a_oa %s a_syn %s sha %s gold58 macro %.5f Syn AUC %.4f' % (k, res[k]['rec'].get('mode'), res[k]['rec'].get('a_oa'), res[k]['rec'].get('a_syn'),
          res[k]['sha'][:12], np.mean([auc(res[k]['out'][:, j], Y[:, j]) for j in range(12)]), auc(res[k]['out'][:, 8], Y[:, 8])))
P0, P2 = res['pw30']['out'], res['syn20']['out']
g1 = res['syn00']['sha'] == res['pw30']['sha']
same = [L[j] for j in range(12) if np.array_equal(P0[:, j], P2[:, j])]
indep = pct(pct(P0[:, 8]) + .2 * pct(P0[:, 7]))
mx = float(np.abs(indep - P2[:, 8]).max())
dS = auc(P2[:, 8], Y[:, 8]) - auc(P0[:, 8], Y[:, 8])
g2 = (same == [c for c in L if c != 'Synovitis']) and mx < 1e-6 and abs(dS - .0084) < 6e-4
print('G1 syn00 byte-identical to pw30: %s' % g1)
print('G2 syn20 unchanged columns %d/11, Syn max abs diff vs independent recompute %.2e, Syn dAUC %+.4f, macro d %+.5f: %s' % (
    len(same), mx, dS, dS / 12, g2))
print('SYN_GATE %s' % ('PASS' if g1 and g2 else 'FAIL'))
