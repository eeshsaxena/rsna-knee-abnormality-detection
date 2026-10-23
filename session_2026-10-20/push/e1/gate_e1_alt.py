# gate_e1_alt.py (10-20, Claude): offline gate for the contingency builds alt/kernel_<a21 base>e1j (build_e1_on.py). Same replay as
# gate_e1_offline.py (each notebook's own cells 14 / 15 / 16-freeze / 18 on the gold58 serve predictions). Per scored A21 base:
#   K0  base replay gold58 macro == the a21 gate log (sym30 .94313, pw20 .94378, pw30 .94336): harness parity
#   K1  E1 1 on base: submission.csv byte-identical to the base (the scored entry's code path)
#   K2  E1 2 on base: non-OA columns + stage 3 identical to c4 + E1 alone (kernel_v224r6c4eoa00e1j); OA trio == the base's E-OA
#       formula (mode, a) applied to the E1-alone output (independent recompute); receipts e1 2 / a_oa / oa_mode
import json, os, time, hashlib, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']; NON = [c for c in L if c not in OA]
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
frame = lambda A: pd.DataFrame(A, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L]


def replay(path):
    nb = json.load(open(path, encoding='utf-8')); c14, c15, c18 = (src(nb['cells'][k]) for k in (14, 15, 18))
    T = tempfile.mkdtemp().replace(chr(92), '/'); D, W, P = Path(T + '/data'), Path(T + '/work'), Path(T + '/pred')
    for d in (D, W, P / 'worker0', W / 's6f' / 'B'): d.mkdir(parents=True)
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(D / 'test.csv', index=False)
    frame(np.full((n, 12), .5)).iloc[:1].to_csv(D / 'sample_submission.csv', index=False)
    frame(R6).to_csv(W / 'ours_sub.csv', index=False); frame(R6).to_csv(P / 'worker0' / 'predictions.csv', index=False)
    frame(R384).to_csv(W / 'raptor_raw.csv', index=False); frame(R384).to_csv(W / 'raptor_sub.csv', index=False)
    frame(R224).to_csv(W / 'raptor224_sub.csv', index=False)
    ns = dict(DATA=D, WORK=W, pred=P, R224_OK=True, RAPTOR_SECONDS=0.0, NB_T0=time.time(), time=time, json=json,
              display=lambda *a, **k: None, FLIP_TTA_FORCED_ON=True)
    exec(compile(c14, path + '#cell14', 'exec'), ns); exec(compile(c15, path + '#cell15', 'exec'), ns)
    shutil.copyfile(W / 'submission.csv', W / 'submission_v224flip.csv')
    setup = {'base_sha256': sha(W / 'submission_v224flip.csv'), 'v224_canary': dict(ns['_canary'])}
    frame(S6).to_csv(W / 's6f' / 'B' / 'B_probs.csv', index=False)
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W / 's6f' / 'B' / 'B_probs.csv')}}}
    fn = c18.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
    ns18 = {}; exec(compile(fn.replace("W = '/kaggle/working'", 'W = %r' % str(W).replace(chr(92), '/')), path + '#cell18', 'exec'), ns18)
    rec = ns18['_s6f_blend'](setup, run)
    r = dict(rec=rec, blend=json.loads((W / 'blend_receipt.json').read_text()), st3_bytes=(W / 'submission_v224flip.csv').read_bytes(),
             bytes=(W / 'submission.csv').read_bytes(), out=pd.read_csv(W / 'submission.csv', dtype={'StudyInstanceUID': str}))
    shutil.rmtree(T, ignore_errors=True); return r


mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name, flush=True)
PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def eoa(o, mode, a):
    U = {c: pct(o[c].values.astype(np.float64)) for c in OA}; e = {}
    for c in OA:
        k = [x for x in OA if x != c]
        if mode == 'sym': e[c] = pct(U[c] - a * (U[k[0]] + U[k[1]]) / 2.0)
        else: e[c] = pct(U[c] - a / 2.0 * sum(PW[tuple(sorted((c, x), key=OA.index))] * U[x] for x in k))
    return e


e1a = replay(HERE + '/kernel_v224r6c4eoa00e1j/rsna_sub_v224r6c4eoa00e1j.ipynb')            # c4 + E1 alone
eoa20 = replay(R + '/session_2026-10-19/a20/kernel_v224r6c4eoa20j/rsna_sub_v224r6c4eoa20j.ipynb')
A21 = R + '/session_2026-10-20/a21'
rows = []
for base, mode, a, tgt in (('v224r6c4eoa30j', 'sym', .3, .94313), ('v224r6c4pw20j', 'pw', .2, .94378), ('v224r6c4pw30j', 'pw', .3, .94336)):
    b = replay(A21 + '/kernel_%s/rsna_sub_%s.ipynb' % (base, base))
    x1 = replay(HERE + '/alt/kernel_%se1x1j/rsna_sub_%se1x1j.ipynb' % (base[:-1], base[:-1]))
    x2 = replay(HERE + '/alt/kernel_%se1j/rsna_sub_%se1j.ipynb' % (base[:-1], base[:-1]))
    chk('K0 %s base replay gold58 macro %.5f == a21 gate log %.5f' % (base, mac(b['out']), tgt), abs(mac(b['out']) - tgt) < 5e-6)
    chk('K1 %s E1 1 control byte-identical to the scored base (final + stage 3)' % base, x1['bytes'] == b['bytes'] and x1['st3_bytes'] == b['st3_bytes'])
    e = eoa(e1a['out'], mode, a); md = max(float(np.abs(x2['out'][c].values - e[c].astype(np.float32)).max()) for c in OA)
    chk('K2 %s E1 2: stage 3 == E1 alone, non-OA == E1 alone, OA == %s a %g on E1 alone (maxdiff %.1e), receipts e1 2 / a_oa %g / oa_mode %s' % (
        base, mode, a, md, a, mode), x2['st3_bytes'] == e1a['st3_bytes'] and (x2['out'][NON].values == e1a['out'][NON].values).all() and md < 1e-6
        and x2['rec'].get('e1') == 2.0 and x2['blend'].get('e1') == 2.0 and x2['rec'].get('a_oa') == a and x2['rec'].get('oa_mode') == mode)
    rows.append((base, mac(b['out']), mac(x2['out'])))
print('== info: gold58 kernel path (code check only, never a selection ruler) ==')
print('a20 eoa20 base %.5f -> +E1 %.5f (%+.5f) [main gate]' % (mac(eoa20['out']), .94350, .94350 - mac(eoa20['out'])))
for base, mb, me in rows: print('%-15s base %.5f -> +E1 %.5f (%+.5f)' % (base, mb, me, me - mb))
print('E1_ALT_GATE', 'PASS' if ok else 'FAIL')
