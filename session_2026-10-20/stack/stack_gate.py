# stack_gate.py (2026-10-21 ~00:20 IST, Claude, STACK). LOCAL offline gate for the unarmed stack builds (pattern of night/c12_gate.py):
# exec the EXACT cell-18 _s6f_blend code of pw30 (scored), pw25 (val build), lomm05 (night build) and each stack build on the gold58
# serve predictions (V224 stage 3 rebuilt from the members, S6fall = n42gpu) in a temp dir. Gates:
#   G1 mcl00 (A_MCLMM 0) submission.csv byte-identical to pw30's, no a_mclmm in its receipt;
#   G2 each build vs its base (pw30 / pw25): exactly the expected columns change ({MCL} or {Lateral OA, MCL}); all others byte-equal;
#   G3 changed columns == independent recompute from the base's own output: MCL = pct(pct(MCL) - .1 pct(MM)); LOA = pct(pct(LOA) - .05 pct(MM));
#   G4 cross-build identities: pw30lomm05mcl10 LOA == night lomm05 LOA; every MCL column == pw30mcl10's MCL (MCL does not depend on A_OA);
#   G5 receipts: a_oa / oa_mode of the base, a_loamm .05 only in lomm variants, a_mclmm .1;
#   G6 from-scratch: stack/stack_local.py pipeline (independent numpy code from the member serve files) == each build, all 12 columns.
#   INFO: gold58 macro and per-label deltas (never a selection ruler).
import json, os, hashlib, tempfile, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; S = R + '/session_2026-10-20'; HERE = S + '/stack'
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
nbp = lambda d, t: d + '/kernel_' + t + '/rsna_sub_' + t + '.ipynb'
NB = {'pw30': nbp(S + '/a21', 'v224r6c4pw30j'), 'pw25': nbp(S + '/val/kernels', 'v224r6c4pw25j'),
      'lomm05': S + '/night/c12_kernel_v224r6c4pw30lomm05j/rsna_sub_v224r6c4pw30lomm05j.ipynb',
      'mcl00': nbp(HERE, 'v224r6c4pw30mcl00j'), 'pw30mcl10': nbp(HERE, 'v224r6c4pw30mcl10j'), 'pw25mcl10': nbp(HERE, 'v224r6c4pw25mcl10j'),
      'pw30lomm05mcl10': nbp(HERE, 'v224r6c4pw30lomm05mcl10j'), 'pw25lomm05mcl10': nbp(HERE, 'v224r6c4pw25lomm05mcl10j')}
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
    res[k] = dict(rec=rec, sha=sha(W + '/submission.csv'), nb_sha=sha(p),
                  out=pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64))
    shutil.rmtree(W, ignore_errors=True)
    print('%-16s nb %s | mode %s a_oa %s %s a_loamm %s a_mclmm %s | out sha %s | gold58 macro %.5f MCL %.4f LOA %.4f' % (
        k, res[k]['nb_sha'][:8], rec.get('mode'), rec.get('a_oa'), rec.get('oa_mode'), rec.get('a_loamm'), rec.get('a_mclmm'), res[k]['sha'][:12],
        np.mean([auc(res[k]['out'][:, j], Y[:, j]) for j in range(12)]), auc(res[k]['out'][:, 1], Y[:, 1]), auc(res[k]['out'][:, 5], Y[:, 5])))
ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name)
jM, jL, jMM = L.index('MCL'), L.index('Lateral OA'), L.index('Medial Meniscus')
chk('G0 bases: pw30 nb 5a903187, pw25 nb 64ed7eba, lomm05 nb c0dd7ec8', res['pw30']['nb_sha'].startswith('5a903187') and res['pw25']['nb_sha'].startswith('64ed7eba')
    and res['lomm05']['nb_sha'].startswith('c0dd7ec8'))
chk('G1 mcl00 byte-identical to pw30 (sha %s), no a_mclmm' % res['mcl00']['sha'][:12], res['mcl00']['sha'] == res['pw30']['sha'] and res['mcl00']['rec'].get('a_mclmm') is None)
SPEC = {'pw30mcl10': ('pw30', ['MCL'], None), 'pw25mcl10': ('pw25', ['MCL'], None),
        'pw30lomm05mcl10': ('pw30', ['MCL', 'Lateral OA'], .05), 'pw25lomm05mcl10': ('pw25', ['MCL', 'Lateral OA'], .05)}
for k, (b, cols, al) in SPEC.items():
    O, B0 = res[k]['out'], res[b]['out']
    chg = sorted(L[j] for j in range(12) if not np.array_equal(O[:, j], B0[:, j]))
    chk('G2 %-16s changed columns vs %s == %s (got %s)' % (k, b, sorted(cols), chg), chg == sorted(cols))
    m1 = float(np.abs(pct(pct(B0[:, jM]) - .1 * pct(B0[:, jMM])) - O[:, jM]).max())
    m2 = float(np.abs(pct(pct(B0[:, jL]) - .05 * pct(B0[:, jMM])) - O[:, jL]).max()) if al else 0.
    chk('G3 %-16s MCL == pct(pct(MCL) - .1 pct(MM)) of %s (maxdiff %.1e)%s' % (k, b, m1, (', LOA == pct(pct(LOA) - .05 pct(MM)) (maxdiff %.1e)' % m2) if al else ''), m1 < 1e-6 and m2 < 1e-6)
    r = res[k]['rec']; rb = res[b]['rec']
    chk('G5 %-16s receipt a_oa %s %s (base %s %s), a_loamm %s, a_mclmm %s' % (k, r.get('a_oa'), r.get('oa_mode'), rb.get('a_oa'), rb.get('oa_mode'), r.get('a_loamm'), r.get('a_mclmm')),
        r.get('a_oa') == rb.get('a_oa') and r.get('oa_mode') == 'pw' and r.get('a_mclmm') == .1 and r.get('a_loamm') == al)
chk('G4 pw30lomm05mcl10 LOA == night lomm05 LOA; its other OA / non-MCL columns == lomm05', np.array_equal(res['pw30lomm05mcl10']['out'][:, jL], res['lomm05']['out'][:, jL])
    and all(np.array_equal(res['pw30lomm05mcl10']['out'][:, j], res['lomm05']['out'][:, j]) for j in range(12) if j != jM))
chk('G4 MCL column identical across all 4 builds (MCL does not depend on A_OA or LOAMM)', all(np.array_equal(res[k]['out'][:, jM], res['pw30mcl10']['out'][:, jM]) for k in SPEC))
# G6 from scratch with the stack_local pipeline (exec its definitions only)
src = open(HERE + '/stack_local.py', encoding='utf-8').read(); ns_ = {}
exec(src.split('summary = {}')[0].replace("log = open(OUT + '.log', 'w')", "log = open(OUT + '_gate_tmp.log', 'w')"), ns_)
M58 = dict(rap=[(.75, R384.astype(float)), (.25, R224.astype(float))], r6=R6.astype(float), s6=S6.astype(float))
CF = {'pw30mcl10': dict(mclmm=.1), 'pw25mcl10': dict(a=.25, mclmm=.1), 'pw30lomm05mcl10': dict(loamm=.05, mclmm=.1), 'pw25lomm05mcl10': dict(a=.25, loamm=.05, mclmm=.1)}
for k, d in CF.items():
    c = dict(ns_['BASE']); c.update(d); E = ns_['pipeline'](M58, c); md = float(np.abs(E - res[k]['out']).max())
    chk('G6 %-16s == independent from-scratch pipeline (members -> c4 -> pw E-OA -> lines), 12 columns, maxdiff %.1e' % (k, md), md < 1e-6)
print('INFO gold58 (n 58; never a ruler): per-label delta vs pw30, then macro')
b0 = res['pw30']['out']
for k in ('pw25', 'lomm05') + tuple(SPEC):
    o = res[k]['out']; d = [auc(o[:, j], Y[:, j]) - auc(b0[:, j], Y[:, j]) for j in range(12)]
    print('  %-16s %s | macro %+.5f' % (k, ' '.join('%s %+.4f' % (L[j][:4], d[j]) for j in range(12) if abs(d[j]) > 0), np.mean(d)))
print('STACK_GATE', 'PASS' if ok else 'FAIL')
