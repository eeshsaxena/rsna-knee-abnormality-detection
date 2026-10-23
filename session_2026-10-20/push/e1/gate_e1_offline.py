# gate_e1_offline.py (10-20, Claude): offline gate for the LOCAL e1eoa builds (build_e1eoa.py). Unlike a21/gate_a21_offline.py
# (which rebuilt V224 stage 3 in numpy and ran only cell 18), this EXECUTES each notebook's own cell 14 (stage 3) and cell 15
# (V224 canary) on the gold58 serve predictions, emulates cell 16's freeze (submission.csv -> submission_v224flip.csv byte copy +
# sha), then executes cell 18's _s6f_blend. Inputs (the same files as m14_gate / gate_a20 / e1_stages / local_mech):
#   raptor_raw.csv = raptor_sub.csv = gold58_r384_kernelserve_cpu.npy, raptor224_sub.csv = gold58_r224_kernelserve.npy,
#   ours_sub.csv = worker0/predictions.csv = R6_gold58_serve.csv, s6f/B/B_probs.csv = n42gpu_gold58.csv (S6fall stand-in).
# gold58 is a code / consistency check here, never a selection ruler. No OAI data is touched.
import json, os, sys, time, hashlib, tempfile, shutil
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
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(G + '/gold58_r224_ids.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist() == ids   # npy row order
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
frame = lambda A: pd.DataFrame(A, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L]

nbp = lambda d, t: d + '/kernel_' + t + '/rsna_sub_' + t + '.ipynb'
NB = {'c4': nbp(R + '/session_2026-10-18/a18', 'v224r6a25c4j'),
      'a20_eoa20': nbp(R + '/session_2026-10-19/a20', 'v224r6c4eoa20j'),          # the SCORED eoa20 (.959 best-id)
      'e1x1_a00': nbp(HERE, 'v224r6c4eoa00e1x1j'), 'e1x1_a20': nbp(HERE, 'v224r6c4eoa20e1x1j'),
      'e1_a00': nbp(HERE, 'v224r6c4eoa00e1j'), 'e1_a20': nbp(HERE, 'v224r6c4eoa20e1j')}
SHA = {'c4': 'b47c29c3', 'a20_eoa20': 'd40f43a4'}
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']


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
    exec(compile(c14, path + '#cell14', 'exec'), ns)                  # stage 3 (writes WORK/submission.csv + blend_receipt.json)
    exec(compile(c15, path + '#cell15', 'exec'), ns)                  # V224 canary (clean path on gold58)
    shutil.copyfile(W / 'submission.csv', W / 'submission_v224flip.csv')   # cell 16 setup: frozen byte copy + sha
    setup = {'base_sha256': sha(W / 'submission_v224flip.csv'), 'v224_canary': dict(ns['_canary'])}
    frame(S6).to_csv(W / 's6f' / 'B' / 'B_probs.csv', index=False)
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W / 's6f' / 'B' / 'B_probs.csv')}}}
    fn = c18.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
    ns18 = {}; exec(compile(fn.replace("W = '/kaggle/working'", 'W = %r' % str(W).replace(chr(92), '/')), path + '#cell18', 'exec'), ns18)
    rec = ns18['_s6f_blend'](setup, run)
    r = dict(rec=rec, blend=json.loads((W / 'blend_receipt.json').read_text()), canary=dict(ns['_canary']),
             st3_bytes=(W / 'submission_v224flip.csv').read_bytes(), bytes=(W / 'submission.csv').read_bytes(),
             st3=pd.read_csv(W / 'submission_v224flip.csv', dtype={'StudyInstanceUID': str}),
             out=pd.read_csv(W / 'submission.csv', dtype={'StudyInstanceUID': str}), nb_sha=sha(path))
    shutil.rmtree(T, ignore_errors=True); return r


res = {k: replay(p) for k, p in NB.items()}
for k, s in SHA.items(): assert res[k]['nb_sha'].startswith(s), (k, res[k]['nb_sha'])
mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name, flush=True)


# ---- independent numpy recompute (does not use any notebook code) ----------------------------------------------------------
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
def expect(e1, a):
    V = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) ** e1 + .25 * pct(R224[:, j]) ** e1
        V[:, j] = w * rr + (1 - w) * pct(R6[:, j]) ** e1
    V = V.astype(np.float32)
    F = np.stack([(.8 * pct(V[:, j]) ** e1 + .2 * pct(S6[:, j]) ** e1) for j in range(12)], 1).astype(np.float32).astype(np.float64)
    if a:
        oi = [L.index(c) for c in OA]; U = {t: pct(F[:, t]) for t in oi}; F2 = F.copy()
        for t in oi:
            o = [m for m in oi if m != t]; F2[:, t] = pct(U[t] - a * (U[o[0]] + U[o[1]]) / 2.0)
        F = F2.astype(np.float32)
    return V.astype(np.float64), F.astype(np.float64)


print('== G-H harness parity (the replay reproduces the earlier gold58 gates) ==')
Vm = np.zeros((n, 12))                                        # the A20 / A21 gate construction of V224 stage 3 (numpy, float32)
for j, c in enumerate(L):
    w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); Vm[:, j] = w * rr + (1 - w) * pct(R6[:, j])
T = tempfile.mkdtemp(); frame(Vm.astype(np.float32)).to_csv(T + '/v.csv', index=False); vb = open(T + '/v.csv', 'rb').read(); shutil.rmtree(T)
chk('H1 c4 cell-14 replay stage 3 byte-identical to the A20/A21 gate numpy V224 construction', res['c4']['st3_bytes'] == vb)
# kernel-path (float32 stage 3 + float32 final) targets = a20/gate_a20_offline.log + a21/gate_a21_offline.log; the float64 numpy
# screens (m14, e1_stages, local_mech) read c4 .94102 / eoa20 .94341 because float64 keeps exact-rational near-ties apart.
chk('H2 c4 replay gold58 macro %.5f == .94113 (kernel path, gate_a20/a21 logs; float64 screens .94102)' % mac(res['c4']['out']), abs(mac(res['c4']['out']) - .94113) < 5e-6)
chk('H3 a20 eoa20 replay gold58 macro %.5f == .94360 (kernel path, gate_a20/a21 logs; float64 m14 .94341)' % mac(res['a20_eoa20']['out']), abs(mac(res['a20_eoa20']['out']) - .94360) < 5e-6)
for k in res:
    r = res[k]
    chk('H4 %-9s clean path: canary none fired, r224 used, S6F mode S6F, base = frozen file, fired []' % k,
        r['canary']['fired'] == [] and r['canary']['r224_used'] and r['rec'].get('mode') == 'S6F'
        and r['rec'].get('base_source') == 'submission_v224flip.csv' and r['rec'].get('fired') == [])

print('== G0 controls (exponent 1) ==')
chk('G0a e1x1_a00 (E1 1, A_OA 0) submission.csv byte-identical to c4 (stage 3 too)',
    res['e1x1_a00']['bytes'] == res['c4']['bytes'] and res['e1x1_a00']['st3_bytes'] == res['c4']['st3_bytes'])
chk('G0b e1x1_a20 (E1 1, A_OA .2) submission.csv byte-identical to the SCORED a20 eoa20 path (stage 3 == c4 too)',
    res['e1x1_a20']['bytes'] == res['a20_eoa20']['bytes'] and res['e1x1_a20']['st3_bytes'] == res['c4']['st3_bytes'])
chk('G0c control receipts: e1 == 1 in blend + s6f receipts; a_oa .2 recorded only for e1x1_a20',
    res['e1x1_a00']['blend'].get('e1') == 1.0 and res['e1x1_a00']['rec'].get('e1') == 1.0 and 'a_oa' not in res['e1x1_a00']['rec']
    and res['e1x1_a20']['rec'].get('e1') == 1.0 and res['e1x1_a20']['rec'].get('a_oa') == 0.2)

print('== G1 E1 parity + structure ==')
m_e1, m_c = mac(res['e1_a00']['out']), mac(res['e1_a20']['out'])
chk('G1a e1_a00 (E1 alone) gold58 macro %.5f == .94111 +- 1e-5 (THINK_1019 8.4 / local_mech l.2)' % m_e1, abs(m_e1 - .94111) < 1e-5)
chk('G1b e1_a20 (candidate) gold58 macro %.5f == .94350 +- 1e-5 (local_mech l.10, (2, FINAL, .2))' % m_c, abs(m_c - .94350) < 1e-5)
for k, (e1, a) in {'e1_a00': (2, 0.0), 'e1_a20': (2, 0.2), 'c4': (1, 0.0), 'a20_eoa20': (1, 0.2)}.items():
    V, F = expect(e1, a)
    m3 = float(np.abs(res[k]['st3'][L].values - V).max()); mf = float(np.abs(res[k]['out'][L].values - F).max())
    chk('G1c %-9s == independent numpy recompute (E1 %g, A_OA %g): stage 3 maxdiff %.1e, final maxdiff %.1e' % (k, e1, a, m3, mf), m3 < 1e-6 and mf < 1e-6)
c, e = res['e1_a20'], res['e1_a00']
chk('G1d candidate stage 3 byte-identical to E1-alone stage 3; non-OA columns identical to E1 alone', c['st3_bytes'] == e['st3_bytes']
    and (c['out'][NON].values == e['out'][NON].values).all() and (c['out'].StudyInstanceUID == e['out'].StudyInstanceUID).all())
U = {t: pct(e['out'][t].values.astype(np.float64)) for t in OA}
eo = {t: pct(U[t] - .2 * sum(U[o] for o in OA if o != t) / 2.0).astype(np.float32) for t in OA}
md = max(float(np.abs(c['out'][t].values - eo[t]).max()) for t in OA)
chk('G1e candidate OA trio == E-OA sym .2 applied to the E1-alone output (maxdiff %.1e)' % md, md < 1e-6)
chk('G1f candidate receipts: blend e1 2, s6f e1 2, a_oa .2, mode S6F, fired []', c['blend'].get('e1') == 2.0 and c['rec'].get('e1') == 2.0
    and c['rec'].get('a_oa') == 0.2 and c['rec'].get('mode') == 'S6F' and c['rec'].get('fired') == [])
nch = sum(int((rankdata(c['out'][x].values) != rankdata(res['a20_eoa20']['out'][x].values)).any()) for x in L)
chk('G1g E1 is live: candidate differs from a20 eoa20 in %d of 12 columns (rank order)' % nch, nch >= 6)

print('== info: gold58 (code check only, never a selection ruler) ==')
rng = np.random.default_rng(7); idxs = []
while len(idxs) < 2000:
    ix = rng.integers(0, n, n)
    if all(0 < Y[ix, j].sum() < n for j in range(12)): idxs.append(ix)
def boot(a_, b_):
    A_ = np.array([[auc(a_[x].values[ix], Y[ix, j]) for j, x in enumerate(L)] for ix in idxs]).mean(1)
    B_ = np.array([[auc(b_[x].values[ix], Y[ix, j]) for j, x in enumerate(L)] for ix in idxs]).mean(1); d = A_ - B_
    return d.mean(), d.std(), (d > 0).mean()
base = mac(res['c4']['out'])
for k in res:
    o = res[k]['out']; print('%-10s macro %.5f (%+.5f vs c4) | per-label vs c4 %s' % (k, mac(o), mac(o) - base,
          [float(round(auc(o[x].values, Y[:, j]) - auc(res['c4']['out'][x].values, Y[:, j]), 4)) for j, x in enumerate(L)]))
for a_, b_ in (('e1_a20', 'a20_eoa20'), ('e1_a00', 'c4'), ('a20_eoa20', 'c4'), ('e1_a20', 'c4')):
    d, s, p = boot(res[a_]['out'], res[b_]['out'])
    print('boot %-9s - %-9s %+.5f (sd %.5f, P>0 %.3f, 2000 resamples, seed 7)' % (a_, b_, d, s, p))
print('sha', {k: res[k]['nb_sha'][:16] for k in res})
print('E1_OFFLINE_GATE', 'PASS' if ok else 'FAIL')
