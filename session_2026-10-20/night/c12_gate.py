# c12_gate.py (2026-10-20 night, Claude). LOCAL offline gate for the unarmed C12' builds (pattern of a21/gate_a21_offline.py and
# night/syn_gate.py): exec the EXACT cell-18 _s6f_blend code of c4, pw30 and each c12 build on the gold58 serve predictions (V224 stage 3
# rebuilt, S6fall = n42gpu) in a temp dir. Gates:
#   G1 lomm00 submission.csv byte-identical to pw30's (no-op control);
#   G2 lomm05: only the Lateral OA column differs from pw30; it equals an independent recompute pct(pct(LOA) - .05 pct(MM)) of pw30's own
#      output (max abs diff < 1e-6); receipt a_loamm .05, a_oa .3, oa_mode pw;
#   G3 lomm05 from scratch: the OA trio recomputed from the c4 kernel output (A21 pw a .3, as gate_a21_offline.py), then the C12' line;
#      the 11 non-LOA columns equal pw30, and LOA equals that recompute;
#   G4 gold58 LOA AUC delta vs pw30 == +.0058 (MEASURED recal_oa_targets.log / skep_c12p.log; info, not a selection).
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
NB = {'c4': R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb',
      'pw30': R + '/session_2026-10-20/a21/kernel_v224r6c4pw30j/rsna_sub_v224r6c4pw30j.ipynb',
      'lomm00': HERE + '/c12_kernel_v224r6c4pw30lomm00j/rsna_sub_v224r6c4pw30lomm00j.ipynb',
      'lomm05': HERE + '/c12_kernel_v224r6c4pw30lomm05j/rsna_sub_v224r6c4pw30lomm05j.ipynb'}
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
    print('%-7s mode %s a_oa %s oa_mode %s a_loamm %s sha %s gold58 macro %.5f LOA AUC %.4f' % (
        k, rec.get('mode'), rec.get('a_oa'), rec.get('oa_mode'), rec.get('a_loamm'), res[k]['sha'][:12],
        np.mean([auc(res[k]['out'][:, j], Y[:, j]) for j in range(12)]), auc(res[k]['out'][:, 5], Y[:, 5])))
ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name)
jl, jm = L.index('Lateral OA'), L.index('Medial Meniscus')
P0, P5, C = res['pw30']['out'], res['lomm05']['out'], res['c4']['out']
chk('G1 lomm00 byte-identical to pw30 (sha %s)' % res['lomm00']['sha'][:12], res['lomm00']['sha'] == res['pw30']['sha'] and res['lomm00']['rec'].get('a_loamm') is None)
same = [L[j] for j in range(12) if np.array_equal(P0[:, j], P5[:, j])]
ind = pct(pct(P0[:, jl]) - .05 * pct(P0[:, jm])); md = float(np.abs(ind - P5[:, jl]).max())
r5 = res['lomm05']['rec']
chk('G2 lomm05: unchanged columns %d/11, LOA == pct(pct(LOA) - .05 pct(MM)) of pw30 (maxdiff %.1e), receipt a_loamm %s a_oa %s %s' % (
    len(same), md, r5.get('a_loamm'), r5.get('a_oa'), r5.get('oa_mode')),
    same == [c for c in L if c != 'Lateral OA'] and md < 1e-6 and r5.get('a_loamm') == .05 and r5.get('a_oa') == .3 and r5.get('oa_mode') == 'pw')
# G3 from scratch: c4 output -> A21 pw a .3 on the trio -> float32 -> C12' line
OA = ['Medial OA', 'Lateral OA', 'PF OA']; jO = [L.index(c) for c in OA]; Wp = {(4, 5): 0.876, (4, 6): 0.708, (5, 6): 1.416}
E = C.copy(); U = {t: pct(C[:, t]) for t in jO}
for t in jO:
    o = [m for m in jO if m != t]; E[:, t] = pct(U[t] - .3 / 2.0 * sum(Wp[tuple(sorted((t, m)))] * U[m] for m in o))
E = E.astype(np.float32).astype(np.float64)
E[:, jl] = pct(pct(E[:, jl]) - .05 * pct(E[:, jm]))
m3 = float(np.abs(E - P5).max()); non = [j for j in range(12) if j != jl]
chk('G3 lomm05 == from-scratch recompute (c4 -> pw a .3 -> C12\' a .05), all 12 columns (maxdiff %.1e); non-LOA == pw30 exactly' % m3,
    m3 < 1e-6 and all(np.array_equal(P5[:, j], P0[:, j]) for j in non))
dL = auc(P5[:, jl], Y[:, jl]) - auc(P0[:, jl], Y[:, jl])
chk('G4 gold58 LOA dAUC vs pw30 %+.4f (macro %+.5f) == +.0058 (info)' % (dL, dL / 12), abs(dL - .0058) < 2e-4)
print('C12_GATE', 'PASS' if ok else 'FAIL')
