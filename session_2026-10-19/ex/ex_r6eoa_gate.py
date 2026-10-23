# ex_r6eoa_gate.py (10-20, Claude): RESEARCH ONLY, local CPU, gold58. Builds NOTHING, writes no kernel, touches nothing in a20.
# Question: what would E-OA do on pick B (R6 57248084, kernel session_2026-10-15/r6entry/kernel_v224r6flips6fj), and what are the
# exact gold58 replay targets for a contingency build (c4 or R6 base, a in {0, .1, .15, .2})?
# Method = gate_a20_offline.py: exec the notebook's OWN cell-18 _s6f_blend code (R6 and c4 cell 18 are identical, MEASURED) in a
# temp dir, with the 9 E-OA lines of build_a20.py inserted IN MEMORY at the same anchor; stage 3 rebuilt from the gold58 serves
# with each kernel's cell-14 parameters (R6: uniform W .5, A224 .35; c4: per-finding table, A224 .25). Temp dirs are deleted.
import json, os, hashlib, tempfile, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4T = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
BASES = {'c4': (C4T, .25, R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb'),
         'R6': ({}, .35, R + '/session_2026-10-15/r6entry/kernel_v224r6flips6fj/rsna_sub_v224r6flips6fj.ipynb')}
def stage3(table, a224):
    V = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = table.get(c, .5); rr = (1 - a224) * pct(R384[:, j]) + a224 * pct(R224[:, j]); V[:, j] = w * rr + (1 - w) * pct(R6[:, j])
    return V
def block(a):   # verbatim copy of build_a20.py BLOCK
    return ['        A_OA = %r   # [A20 E-OA] OA decorrelation on the final output (OAI expert check G1 SUPPORT, SHIP_A .2); 0 = no-op' % a,
            "        _OA = ['Medial OA', 'Lateral OA', 'PF OA']",
            '        if A_OA and set(_OA) <= set(labels) and not set(_OA) & set(blank):',
            '            _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}',
            '            for c in _OA:',
            '                _o = [k for k in _OA if k != c]',
            '                out[c] = pct(_U[c] - A_OA * (_U[_o[0]] + _U[_o[1]]) / 2.0).astype(np.float32)',
            "            rec['a_oa'] = A_OA",
            "            print('EOA_APPLIED a_oa %g on %s' % (A_OA, _OA), flush=True)"]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
cell18 = {k: src(json.load(open(p, encoding='utf-8'))['cells'][18]) for k, (_, _, p) in BASES.items()}
print('cell 18 identical c4 vs R6:', cell18['c4'] == cell18['R6'])
def run(base, a):
    table, a224, _ = BASES[base]; s = cell18[base]
    if a is not None:
        lines = s.split('\n'); i = [k for k, l in enumerate(lines) if l == '        for c in blank:']; assert len(i) == 1; i = i[0]
        assert lines[i + 1] == '            out[c] = np.float32(0.5)' and lines[i + 2].startswith('        v = out[labels].values.astype(np.float64)')
        lines[i + 2:i + 2] = block(a); s = '\n'.join(lines)
    fn = s.split('\n\ntry:\n    _S6F_BLEND = ')[0]
    W = tempfile.mkdtemp().replace(chr(92), '/'); assert fn.count("W = '/kaggle/working'") == 1
    fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    V = stage3(table, a224)
    pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
    rn = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
    ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](setup, rn)
    b = open(W + '/submission.csv', 'rb').read(); out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
    shutil.rmtree(W, ignore_errors=True)
    assert out.StudyInstanceUID.tolist() == ids
    return rec, b, out
rng = np.random.default_rng(20261020); BI = [rng.integers(0, n, n) for _ in range(2000)]
def boot_delta(o1, o0, cols):
    d = []
    for ix in BI:
        s = 0.
        for c in cols:
            j = L.index(c); y = Y[ix, j]
            if 0 < y.sum() < len(y): s += auc(o1[c].values[ix], y) - auc(o0[c].values[ix], y)
        d.append(s / 12)
    d = np.array(d); return d.std(), (d > 0).mean()
res = {}
for base in ('c4', 'R6'):
    rec0, b0, o0 = run(base, None)
    m0 = np.mean([auc(o0[c].values, Y[:, j]) for j, c in enumerate(L)])
    print('[%s] base (unpatched cell 18) mode %s | gold58 macro %.5f | OA AUCs %s' % (base, rec0.get('mode'), m0, [round(auc(o0[c].values, Y[:, L.index(c)]), 4) for c in OA]))
    for a in (0.0, 0.1, 0.15, 0.2):
        rec, b, o = run(base, a)
        m = np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
        non = [c for c in L if c not in OA]; same_non = (o[non].values == o0[non].values).all()
        sd, p = boot_delta(o, o0, OA) if a else (0., 0.)
        print('   a %.2f | byte-identical to base %s | non-OA identical %s | a_oa %s | gold58 macro %.5f delta %+.5f (boot sd %.5f, P>0 %.2f) | OA AUCs %s'
              % (a, b == b0, same_non, rec.get('a_oa'), m, m - m0, sd, p, [round(auc(o[c].values, Y[:, L.index(c)]), 4) for c in OA]))
        res[(base, a)] = o
# pick-pair view: R6 vs c4 on gold58 (consistency only)
mc4 = np.mean([auc(res[('c4', 0.0)][c].values, Y[:, j]) for j, c in enumerate(L)]); mr6 = np.mean([auc(res[('R6', 0.0)][c].values, Y[:, j]) for j, c in enumerate(L)])
print('gold58 R6 - c4 (both unpatched, kernel path) %+.5f' % (mr6 - mc4))
# OA coupling of each base on gold58 (normal-score Spearman proxy: plain Spearman of the final columns)
for base in ('c4', 'R6'):
    o = res[(base, 0.0)]; U = np.stack([pct(o[c].values) for c in OA], 1); C = np.corrcoef(U.T)
    print('[%s] gold58 OA Spearman ML %.3f MPF %.3f LPF %.3f mean %.3f' % (base, C[0, 1], C[0, 2], C[1, 2], (C[0, 1] + C[0, 2] + C[1, 2]) / 3))
