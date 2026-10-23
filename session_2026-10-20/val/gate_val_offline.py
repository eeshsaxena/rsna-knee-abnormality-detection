#!/usr/bin/env python
# gate_val_offline.py (10-20 ~19:20 IST, Claude, Track C): offline gate for the val/kernels builds. Nothing is pushed or downloaded.
# Method = a21/gate_a21_offline.py + push/gate_r6eoa_offline.py: run the EXACT cell-18 _s6f_blend code of each notebook on
#  (1) the gold58 serves (stage 3 rebuilt: c4 = C4 table + A224 .25; R6 = uniform W .5 + A224 .35; S6fall = n42gpu), and
#  (2) the COMMITTED Kaggle inputs already on disk (submission_v224flip.csv + s6f/B/B_probs.csv + the run's own setup / run dicts from
#      s6f_receipt.json) of c4 57343435 (a18/out_v224r6a25c4j_v1), scored pw30 57425970 (a21/out_v224r6c4pw30j_v1) and
#      R6 57248084 (r6entry/out_v224r6flips6fj_v1). Kaggle writes LF; pandas on Windows writes CRLF, so replay bytes are compared
#      after CRLF -> LF (the raw-byte compare is printed too).
# Gates: strength-0 controls byte-identical to c4 / R6 (gold58 AND committed replay); pw30 rebuild == scored pw30 (notebook sha,
# gold58 bytes, committed replay bytes == the downloaded scored submission.csv and its receipt sha); OA columns == independent
# recompute; non-OA identical; receipts a_oa / oa_mode. Also writes simulated commit outputs (control's real committed dir + the new
# cell-18 replay) under val/sim/ for the commit-gate self-test (selftest_commit_gate.py).
import json, os, sys, hashlib, tempfile, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding='utf-8')
R = 'C:/Users/eeshs/Downloads/rsna-knee'; G = R + '/session_2026-10-18/next/g58'; V_ = R + '/session_2026-10-20/val'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']; non = [c for c in L if c not in OA]
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
shb = lambda b: hashlib.sha256(b).hexdigest()
lf = lambda b: b.replace(b'\r\n', b'\n')
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); Y = g[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6s = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4T = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
VC4, VR6 = np.zeros((n, 12)), np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4T.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); VC4[:, j] = w * rr + (1 - w) * pct(R6s[:, j])
    rr = .65 * pct(R384[:, j]) + .35 * pct(R224[:, j]); VR6[:, j] = .5 * rr + .5 * pct(R6s[:, j])

nbp = lambda d, t: d + '/kernel_' + t + '/rsna_sub_' + t + '.ipynb'
K = V_ + '/kernels'; A21 = R + '/session_2026-10-20/a21'; SH = R + '/session_2026-10-20/push/shape'; PU = R + '/session_2026-10-20/push'
NB = {  # key: (notebook, base)
    'c4': (R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb', 'c4'),
    'pw00': (nbp(K, 'v224r6c4pw00j'), 'c4'), 'pw30': (nbp(K, 'v224r6c4pw30j'), 'c4'), 'a21_pw30': (nbp(A21, 'v224r6c4pw30j'), 'c4'),
    'pw35': (nbp(K, 'v224r6c4pw35j'), 'c4'), 'shape_pw35': (nbp(SH, 'v224r6c4pw35j'), 'c4'),
    'pw40': (nbp(K, 'v224r6c4pw40j'), 'c4'), 'pw45': (nbp(K, 'v224r6c4pw45j'), 'c4'),
    'R6': (R + '/session_2026-10-15/r6entry/kernel_v224r6flips6fj/rsna_sub_v224r6flips6fj.ipynb', 'r6'),
    'R6pw00': (nbp(K, 'v224r6flips6fpw00j'), 'r6'), 'R6pw30': (nbp(K, 'v224r6flips6fpw30j'), 'r6'),
    'R6e20': (nbp(PU, 'v224r6flips6feoa20j'), 'r6')}
COMMITTED = {'c4': R + '/session_2026-10-18/a18/out_v224r6a25c4j_v1', 'pw30': A21 + '/out_v224r6c4pw30j_v1',
             'r6': R + '/session_2026-10-15/r6entry/out_v224r6flips6fj_v1'}

def cell18(p):
    nb = json.load(open(p, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
    fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1; return fn

def run_gold(p, Vb):
    fn = cell18(p); W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    pd.DataFrame(Vb.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
    ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](setup, run)
    b = open(W + '/submission.csv', 'rb').read(); out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
    shutil.rmtree(W, ignore_errors=True); return dict(rec=rec, bytes=b, out=out)

def run_committed(p, cdir):
    """Replay the notebook's cell 18 on a committed run's own inputs and setup / run dicts."""
    fn = cell18(p); W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    shutil.copyfile(cdir + '/submission_v224flip.csv', W + '/submission_v224flip.csv'); os.makedirs(W + '/s6f/B')
    shutil.copyfile(cdir + '/s6f/B/B_probs.csv', W + '/s6f/B/B_probs.csv')
    rc = json.load(open(cdir + '/s6f_receipt.json', encoding='utf-8'))
    ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](rc['setup'], rc['run'])
    b = open(W + '/submission.csv', 'rb').read(); out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
    shutil.rmtree(W, ignore_errors=True); return dict(rec=rec, bytes=b, out=out, receipt=rc)

Wp = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def expect(o, mode, a):   # independent recompute (scipy rankdata, not the notebook's pandas rank) from a control output
    U = {c: pct(o[c].values.astype(np.float64)) for c in OA}; e = {}
    for c in OA:
        oo = [k for k in OA if k != c]
        if mode == 'sym': e[c] = pct(U[c] - a * (U[oo[0]] + U[oo[1]]) / 2.0)
        else: e[c] = pct(U[c] - a / 2.0 * sum(Wp[tuple(sorted((c, k), key=OA.index))] * U[k] for k in oo))
    return e

ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name, flush=True)

print('== gold58 replay (n %d) ==' % n)
GR = {k: run_gold(p, VC4 if b == 'c4' else VR6) for k, (p, b) in NB.items()}
chk('G1 c4 pw00 (control) gold58 submission bytes == c4', GR['pw00']['bytes'] == GR['c4']['bytes'])
chk('G2 R6 pw00 (control) gold58 submission bytes == R6', GR['R6pw00']['bytes'] == GR['R6']['bytes'])
chk('G3a val pw30 notebook sha == SCORED pw30 57425970 notebook a21/kernel_v224r6c4pw30j (5a903187), metadata identical',
    sha(NB['pw30'][0]) == sha(NB['a21_pw30'][0]) == '5a903187415505e9e3566ebd749cae1328b20df6b60bd6593e6443b17fc0c9fd'
    and sha(K + '/kernel_v224r6c4pw30j/kernel-metadata.json') == sha(A21 + '/kernel_v224r6c4pw30j/kernel-metadata.json'))
chk('G3b val pw30 gold58 bytes == scored pw30 notebook gold58 bytes', GR['pw30']['bytes'] == GR['a21_pw30']['bytes'])
chk('G4a val pw35 == push/shape pw35 (notebook sha e65c0f2b, metadata, gold58 bytes)', sha(NB['pw35'][0]) == sha(NB['shape_pw35'][0])
    and sha(NB['pw35'][0]).startswith('e65c0f2b') and GR['pw35']['bytes'] == GR['shape_pw35']['bytes']
    and sha(K + '/kernel_v224r6c4pw35j/kernel-metadata.json') == sha(SH + '/kernel_v224r6c4pw35j/kernel-metadata.json'))
ENT = {'pw30': ('c4', .3), 'pw35': ('c4', .35), 'pw40': ('c4', .4), 'pw45': ('c4', .45), 'R6pw30': ('R6', .3)}
for k, (ctl, a) in ENT.items():
    e = expect(GR[ctl]['out'], 'pw', a); o = GR[k]['out']; c0 = GR[ctl]['out']
    md = max(float(np.abs(o[c].values - e[c].astype(np.float32)).max()) for c in OA)
    chk('G5 %s gold58 OA cols == independent recompute from %s output (maxdiff %.1e), non-OA identical to %s, receipt a_oa %g mode pw, mode S6F'
        % (k, ctl, md, ctl, a), md < 1e-6 and (o[non].values == c0[non].values).all() and o.StudyInstanceUID.tolist() == c0.StudyInstanceUID.tolist()
        and GR[k]['rec'].get('a_oa') == a and GR[k]['rec'].get('oa_mode') == 'pw' and GR[k]['rec'].get('mode') == 'S6F')

print('== committed-input replay (the 3 public test rows, the runs\' own setup / run dicts) ==')
CR = {}
for k, cd in (('c4', 'c4'), ('pw00', 'c4'), ('pw30', 'pw30'), ('a21_pw30', 'pw30'), ('pw35', 'c4'), ('pw40', 'c4'), ('pw45', 'c4'),
              ('pw30_on_c4', 'c4'), ('R6', 'r6'), ('R6pw00', 'r6'), ('R6pw30', 'r6'), ('R6e20', 'r6')):
    CR[k] = run_committed(NB[k.replace('_on_c4', '')][0], COMMITTED[cd]); CR[k]['cdir'] = COMMITTED[cd]
    assert CR[k]['rec'].get('base_source') == 'submission_v224flip.csv' and CR[k]['rec'].get('mode') == 'S6F', (k, CR[k]['rec'])
for k in ('c4', 'R6'):
    cd = COMMITTED[k.lower()]; kb = open(cd + '/submission.csv', 'rb').read()
    chk('G6 base %s replay of its own cell 18 == its committed submission.csv (LF bytes; raw CRLF->LF %s; sha %s == receipt)'
        % (k, CR[k]['bytes'] != lf(CR[k]['bytes']), shb(lf(CR[k]['bytes']))[:8]), lf(CR[k]['bytes']) == kb and shb(kb) == CR[k]['receipt']['blend']['submission_sha256'])
chk('G6 c4 pw00 (control) committed replay == c4 57343435 committed submission.csv (LF, sha 4ad325dd)',
    lf(CR['pw00']['bytes']) == open(COMMITTED['c4'] + '/submission.csv', 'rb').read())
chk('G6 R6 pw00 (control) committed replay == R6 57248084 committed submission.csv (LF, sha e1324ecc)',
    lf(CR['R6pw00']['bytes']) == open(COMMITTED['r6'] + '/submission.csv', 'rb').read())
spw = open(COMMITTED['pw30'] + '/submission.csv', 'rb').read(); rpw = json.load(open(COMMITTED['pw30'] + '/s6f_receipt.json', encoding='utf-8'))['blend']
chk('G3c val pw30 replay on the SCORED pw30 run inputs == its downloaded submission.csv (LF bytes, sha %s == receipt %s)'
    % (shb(lf(CR['pw30']['bytes']))[:8], rpw['submission_sha256'][:8]), lf(CR['pw30']['bytes']) == spw and shb(spw) == rpw['submission_sha256'])
KEYS = ('w_s6f', 'mode', 'fired', 'base_source', 'n', 'base_blanked', 'b_probs_sha256', 'coverage', 'n_scored', 'rho_mean', 'a_oa', 'oa_mode')
chk('G3d val pw30 replay receipt == the scored run s6f_receipt blend on %d keys (a_oa .3, oa_mode pw, rho_mean, coverage ...)' % len(KEYS),
    all(CR['pw30']['rec'].get(x) == rpw.get(x) for x in KEYS))
chk('G3e c4 committed inputs == scored pw30 committed inputs (same stage-3 sha dcaabfb7 and B sha 7975394d), and pw30 replayed on the c4 inputs == scored pw30 output',
    sha(COMMITTED['c4'] + '/submission_v224flip.csv') == sha(COMMITTED['pw30'] + '/submission_v224flip.csv')
    and sha(COMMITTED['c4'] + '/s6f/B/B_probs.csv') == sha(COMMITTED['pw30'] + '/s6f/B/B_probs.csv') and lf(CR['pw30_on_c4']['bytes']) == spw)
for k, (ctl, a) in ENT.items():
    o, c0 = CR[k]['out'], CR[ctl]['out']; e = expect(c0, 'pw', a)
    md = max(float(np.abs(o[c].values - e[c].astype(np.float32)).max()) for c in OA)
    chk('G7 %s committed-input replay: OA == independent recompute from the %s committed output (maxdiff %.1e), non-OA identical, receipt a_oa %g'
        % (k, ctl, md, a), md < 1e-6 and (o[non].values == c0[non].values).all() and CR[k]['rec'].get('a_oa') == a and CR[k]['rec'].get('oa_mode') == 'pw')
# info: how many distinct OA outputs the 3 public rows can show across a (the commit gate's (d) check cannot tell a .35 from a .40 if equal)
sig = {k: tuple(np.round(CR[k]['out'][OA].values.ravel(), 6)) for k in ('c4', 'pw30', 'pw35', 'pw40', 'pw45')}
print('info: 3-row public OA outputs: distinct among c4/pw30/pw35/pw40/pw45 = %d; pw35==pw30 %s, pw40==pw30 %s, pw45==pw30 %s, pw40==pw35 %s'
      % (len(set(sig.values())), sig['pw35'] == sig['pw30'], sig['pw40'] == sig['pw30'], sig['pw45'] == sig['pw30'], sig['pw40'] == sig['pw35']))

# simulated commit outputs for the commit-gate self-test: the control's REAL committed dir + this notebook's replayed cell 18
SIM = V_ + '/sim'
def write_sim(key, tag, cdir, recd, outb):
    d = SIM + '/out_%s_sim' % tag; shutil.rmtree(d, ignore_errors=True); shutil.copytree(cdir, d, ignore=shutil.ignore_patterns('__pycache__'))
    open(d + '/submission.csv', 'wb').write(lf(outb))
    rc = json.load(open(cdir + '/s6f_receipt.json', encoding='utf-8')); rc['blend'] = recd; rc['mode'] = recd.get('mode'); rc['fired'] = list(recd.get('fired') or [])
    rc['SIMULATED'] = 'val/gate_val_offline.py: committed inputs of %s + cell 18 of %s replayed locally; NOT a Kaggle run' % (os.path.basename(cdir), tag)
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    lg = [p for p in os.listdir(d) if p.endswith('.log') and 'cufile' not in p]; assert len(lg) == 1, lg
    try:
        L_ = json.load(open(d + '/' + lg[0], encoding='utf-8'))
        if 'a_oa' in recd:
            L_.append({'stream_name': 'stdout', 'time': 0, 'data': 'EOA_APPLIED a_oa %g mode %s on %s [SIMULATED]\n' % (recd['a_oa'], recd.get('oa_mode', 'sym'), OA)})
        json.dump(L_, open(d + '/' + lg[0], 'w', encoding='utf-8'))
    except json.JSONDecodeError:
        if 'a_oa' in recd: open(d + '/' + lg[0], 'a', encoding='utf-8').write('\nEOA_APPLIED a_oa %g [SIMULATED]\n' % recd['a_oa'])
    return d
os.makedirs(SIM, exist_ok=True)
for k, tag in (('pw35', 'v224r6c4pw35j'), ('pw40', 'v224r6c4pw40j'), ('pw45', 'v224r6c4pw45j'), ('R6pw30', 'v224r6flips6fpw30j'),
               ('R6e20', 'v224r6flips6feoa20j'), ('pw30_on_c4', 'v224r6c4pw30j')):
    print('SIM written', write_sim(k, tag, CR[k]['cdir'], CR[k]['rec'], CR[k]['bytes']))

print('== gold58 macro (INFO ONLY; gold58 is not a ruler) ==')
mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
mc, mr = mac(GR['c4']['out']), mac(GR['R6']['out'])
CURVE = {.3: .00223, .35: .00256, .4: .00190, .45: .00157}   # the given pw curve (task brief / HANDOFF 18:55)
for k in ('c4', 'pw30', 'pw35', 'pw40', 'pw45', 'R6', 'R6e20', 'R6pw30'):
    o = GR[k]['out']; m = mac(o); ref = mc if k.startswith('pw') or k == 'c4' else mr
    print('%-7s gold58 macro %.5f (%+.5f vs %s) | OA AUCs %s' % (k, m, m - ref, 'c4' if ref is mc else 'R6', [round(float(auc(o[c].values, Y[:, L.index(c)])), 4) for c in OA]))
for k, (ctl, a) in ENT.items():
    if ctl == 'c4':
        d = mac(GR[k]['out']) - mc; chk('G8 %s gold58 delta %+.5f == given curve %+.5f (+-1e-5; consistency only)' % (k, d, CURVE[a]), abs(d - CURVE[a]) < 1e-5)
# paired bootstrap on gold58 (info): each new strength vs pw30 (the scored best-id) and R6pw30 vs R6 / R6e20
rng = np.random.default_rng(1020); B_ = 2000
def boot(k1, k2):
    o1, o2 = GR[k1]['out'], GR[k2]['out']; ds = []
    for _ in range(B_):
        ix = rng.integers(0, n, n)
        if any(Y[ix, j].min() == Y[ix, j].max() for j in range(12)): continue
        ds.append(np.mean([auc(o1[c].values[ix], Y[ix, j]) - auc(o2[c].values[ix], Y[ix, j]) for j, c in enumerate(L)]))
    ds = np.array(ds); return ds.mean(), ds.std(), (ds > 0).mean(), len(ds)
for k1, k2 in (('pw35', 'pw30'), ('pw40', 'pw30'), ('pw45', 'pw30'), ('R6pw30', 'R6'), ('R6pw30', 'R6e20')):
    m_, s_, p_, nb_ = boot(k1, k2); print('info boot %-7s - %-6s gold58 macro %+.5f sd %.5f P>0 %.2f (B %d)' % (k1, k2, m_, s_, p_, nb_))
print('VAL_OFFLINE_GATE', 'PASS' if ok else 'FAIL')
