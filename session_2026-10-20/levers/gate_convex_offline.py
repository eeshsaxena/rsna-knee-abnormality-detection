# gate_convex_offline.py (10-20 levers, Claude): copy of push/shape/gate_shape_offline.py for the levers/ convex builds (P_OA 2) and
# their P_OA 1 controls. Gates: P1 controls byte-identical to the A21 pw30 / scored A20 eoa20 outputs; P2 OA cols == independent recompute;
# non-OA identical to c4; receipts. gold58 macro printed as INFO ONLY.
# Origin: gate_a20_offline.py (10-19, Claude): run the EXACT cell-18 _s6f_blend code of each built notebook (c4, eoa00, eoa20) on the gold58
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

A20 = R + '/session_2026-10-19/a20/'; A21 = R + '/session_2026-10-20/a21/'; LV = R + '/session_2026-10-20/levers/'
nbp = lambda d, t: d + 'kernel_' + t + '/rsna_sub_' + t + '.ipynb'
NB = {'c4': R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb',
      'a20_eoa20': nbp(A20, 'v224r6c4eoa20j'), 'a21_pw30': nbp(A21, 'v224r6c4pw30j'), 'a21_pw20': nbp(A21, 'v224r6c4pw20j'),
      'eoa20p1': nbp(LV, 'v224r6c4eoa20p1j'), 'pw30p1': nbp(LV, 'v224r6c4pw30p1j'),
      'pw25p2': nbp(LV, 'v224r6c4pw25p2j'), 'pw20p2': nbp(LV, 'v224r6c4pw20p2j'), 'eoa20p2': nbp(LV, 'v224r6c4eoa20p2j')}
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
    out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
    res[k] = dict(rec=rec, bytes=open(W + '/submission.csv', 'rb').read(), out=out)
    shutil.rmtree(W, ignore_errors=True)
OA = ['Medial OA', 'Lateral OA', 'PF OA']; non = [c for c in L if c not in OA]
mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
# independent recompute of the OA columns from the c4 output
c4o = res['c4']['out']; Wp = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def expect(mode, a):
    U = {c: pct(c4o[c].values.astype(np.float64)) for c in OA}; e = {}
    for c in OA:
        o = [k for k in OA if k != c]
        if mode == 'sym': e[c] = pct(U[c] - a * (U[o[0]] + U[o[1]]) / 2.0)
        else: e[c] = pct(U[c] - a / 2.0 * sum(Wp[tuple(sorted((c, k), key=OA.index))] * U[k] for k in o))
    return e
ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name)
chk('eoa20p1 (P 1 control) byte-identical to the SCORED a20 eoa20 output', res['eoa20p1']['bytes'] == res['a20_eoa20']['bytes'])
chk('pw30p1 (P 1 control) byte-identical to the a21 pw30 output', res['pw30p1']['bytes'] == res['a21_pw30']['bytes'])
def expect2(mode, a, p):
    U = {c: pct(c4o[c].values.astype(np.float64)) for c in OA}; S = {c: U[c] ** p for c in OA}; e = {}
    for c in OA:
        o = [k for k in OA if k != c]
        if mode == 'sym': e[c] = pct(U[c] - a * (S[o[0]] + S[o[1]]) / 2.0)
        else: e[c] = pct(U[c] - a / 2.0 * sum(Wp[tuple(sorted((c, k), key=OA.index))] * S[k] for k in o))
    return e
for k, (mode, a, p) in {'pw25p2': ('pw', .25, 2.), 'pw20p2': ('pw', .2, 2.), 'eoa20p2': ('sym', .2, 2.)}.items():
    e = expect2(mode, a, p); o = res[k]['out']
    md = max(float(np.abs(o[c].values - e[c].astype(np.float32)).max()) for c in OA)
    chk('%s OA cols == independent recompute (maxdiff %.1e), non-OA identical to c4, receipt a_oa %g mode %s p_oa %g' % (k, md, a, mode, p),
        md < 1e-6 and (o[non].values == c4o[non].values).all() and res[k]['rec'].get('a_oa') == a and res[k]['rec'].get('oa_mode') == mode
        and res[k]['rec'].get('p_oa') == p)
    lt = expect(mode, a); nd = sum(int((np.abs(o[c].values - lt[c].astype(np.float32)) > 1e-6).sum()) for c in OA)
    chk('%s differs from its linear twin (same mode, a) in %d OA cells' % (k, nd), nd > 0)
base = mac(c4o)
for k in ('c4', 'a20_eoa20', 'a21_pw20', 'a21_pw30', 'pw20p2', 'pw25p2', 'eoa20p2'):
    o = res[k]['out']; print('%-10s gold58 macro %.5f (%+.5f vs c4) | OA AUCs %s  [INFO ONLY]' % (k, mac(o), mac(o) - base, [round(auc(o[c].values, Y[:, L.index(c)]), 4) for c in OA]))
import json as _j
for t, a, p in (('v224r6c4pw25p2j', .25, 2.), ('v224r6c4pw20p2j', .2, 2.), ('v224r6c4eoa20p2j', .2, 2.), ('v224r6c4pw30p1j', .3, 1.), ('v224r6c4eoa20p1j', .2, 1.)):
    rc = _j.load(open(LV + 'build_receipt_%s.json' % t)); nbsha = sha(nbp(LV, t))
    chk('%s receipt src = c4 b47c29c3, out sha == notebook on disk (%s), a_oa %g p_oa %g' % (t, nbsha[:8], a, p),
        rc['src_sha256'].startswith('b47c29c3') and rc['out_sha256'] == nbsha and rc['a_oa'] == a and rc['p_oa'] == p)
print('CONVEX_OFFLINE_GATE', 'PASS' if ok else 'FAIL')
