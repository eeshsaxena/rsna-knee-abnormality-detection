# ex_nearmiss_gate.py (10-20, Claude): RESEARCH ONLY, local CPU, gold58 consistency numbers. Builds NOTHING, writes no kernel.
# Kernel-path (the c4 notebook's own cell-18 code, exec'd in a temp dir, float32 output) gold58 replay of the NEAR-MISS add-ons on
# top of E-OA, for completeness only (none survived; EX_1019.md sec 3.3). Extra lines go inside the E-OA `if` block, after its loop:
#   LM_MM : Lateral OA <- Medial Meniscus decorrelation  LOA' = pct(pct(E_LOA) - b pct(MM))
#   BORL  : Lateral OA += Lateral Meniscus borrow         LOA' = pct(pct(E_LOA) + b pct(LM))
#   BOTH  : LOA' = pct(pct(E_LOA) - b pct(MM) + b pct(LM))
#   PW10  : gap-weighted E-OA (weights LOA-PF 1.416, MOA-LOA .876, MOA-PF .708), replaces the symmetric loop
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
NB = R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb'
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
CELL = src(json.load(open(NB, encoding='utf-8'))['cells'][18])
V = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4T.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); V[:, j] = w * rr + (1 - w) * pct(R6[:, j])
HEAD = ['        A_OA = %r',
        "        _OA = ['Medial OA', 'Lateral OA', 'PF OA']",
        '        if A_OA and set(_OA) <= set(labels) and not set(_OA) & set(blank):',
        '            _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}']
SYM = ['            for c in _OA:',
       '                _o = [k for k in _OA if k != c]',
       '                out[c] = pct(_U[c] - A_OA * (_U[_o[0]] + _U[_o[1]]) / 2.0).astype(np.float32)']
PW = ["            _W = {('Lateral OA', 'PF OA'): 1.416, ('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708}",
      '            for c in _OA:',
      '                _o = [k for k in _OA if k != c]',
      '                _w = [_W.get((c, k), _W.get((k, c))) for k in _o]',
      '                out[c] = pct(_U[c] - A_OA / 2.0 * (_w[0] * _U[_o[0]] + _w[1] * _U[_o[1]])).astype(np.float32)']
def extra(kind, b):
    if kind == 'none': return []
    g = "            if 'Medial Meniscus' in labels and 'Lateral Meniscus' in labels and not {'Medial Meniscus', 'Lateral Meniscus'} & set(blank):"
    e = "                _E = pct(out['Lateral OA'].values.astype(np.float64)); _MM = pct(out['Medial Meniscus'].values.astype(np.float64)); _LM = pct(out['Lateral Meniscus'].values.astype(np.float64))"
    f = {'LM_MM': '_E - %r * _MM' % b, 'BORL': '_E + %r * _LM' % b, 'BOTH': '_E - %r * _MM + %r * _LM' % (b, b)}[kind]
    return [g, e, "                out['Lateral OA'] = pct(%s).astype(np.float32)" % f]
TAIL = ["            rec['a_oa'] = A_OA"]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
def run(lines_block):
    s = CELL
    if lines_block is not None:
        lines = s.split('\n'); i = [k for k, l in enumerate(lines) if l == '        for c in blank:'][0]
        assert lines[i + 1] == '            out[c] = np.float32(0.5)' and lines[i + 2].startswith('        v = out[labels].values.astype(np.float64)')
        lines[i + 2:i + 2] = lines_block; s = '\n'.join(lines)
    fn = s.split('\n\ntry:\n    _S6F_BLEND = ')[0]
    W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
    rn = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
    ns = {}; exec(fn, ns); ns['_s6f_blend'](setup, rn)
    out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}); shutil.rmtree(W, ignore_errors=True); return out
base = run(None); m0 = np.mean([auc(base[c].values, Y[:, j]) for j, c in enumerate(L)])
ref = run([HEAD[0] % 0.1] + HEAD[1:] + SYM + TAIL)
mref = np.mean([auc(ref[c].values, Y[:, j]) for j, c in enumerate(L)])
print('c4 kernel path gold58 macro %.5f | E-OA .1 %.5f (%+.5f)' % (m0, mref, mref - m0))
rng = np.random.default_rng(7); BI = [rng.integers(0, n, n) for _ in range(2000)]
def inc_boot(o, r):
    d = []
    for ix in BI:
        s = 0.
        for c in OA:
            j = L.index(c); y = Y[ix, j]
            if 0 < y.sum() < len(y): s += auc(o[c].values[ix], y) - auc(r[c].values[ix], y)
        d.append(s / 12)
    d = np.array(d); return d.std(), (d > 0).mean()
for name, blk in [('E-OA .1 + LM_MM .1', SYM + extra('LM_MM', .1)), ('E-OA .1 + BORL .1', SYM + extra('BORL', .1)),
                  ('E-OA .1 + BOTH .1', SYM + extra('BOTH', .1)), ('PW10', PW), ('PW10 + LM_MM .1', PW + extra('LM_MM', .1))]:
    o = run([HEAD[0] % 0.1] + HEAD[1:] + blk + TAIL)
    m = np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
    non = [c for c in L if c not in OA]; sd, p = inc_boot(o, ref)
    print('%-22s gold58 macro %.5f | vs c4 %+.5f | vs E-OA .1 %+.5f (boot sd %.5f, P>0 %.2f) | non-OA identical %s | OA AUCs %s'
          % (name, m, m - m0, m - mref, sd, p, (o[non].values == base[non].values).all(), [round(auc(o[c].values, Y[:, L.index(c)]), 4) for c in OA]))
