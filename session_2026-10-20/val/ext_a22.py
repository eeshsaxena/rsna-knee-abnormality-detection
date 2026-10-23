#!/usr/bin/env python
# ext_a22.py (10-20 ~20:30 IST, Claude, VAL synthesis). OFFLINE ONLY: nothing pushed, committed, submitted, downloaded or registered.
# Extends the A22 slate registry (val/slate_a22.json) with the two entries the val tracks recommend but Track C did not register:
#   pw25   = c4 + A21 pw block at a .25   (val/kernels/kernel_v224r6c4pw25j  == push/shape build, sha 64ed7eba)
#   r6pw20 = R6 + A21 pw block at a .2    (val/kernels/kernel_v224r6flips6fpw20j == picks2_work/build, sha e4d92c0a)
# Same method as gate_val_offline.py: replay each notebook's OWN cell 18 on gold58 and on the COMMITTED Kaggle inputs already on disk,
# check OA == independent recompute, non-OA identical, receipt a_oa / oa_mode; write simulated commit outputs (control's REAL committed
# dir + replayed cell 18) under val/sim/; then run the exact commit-gate argv on positives and negatives with stated expectations.
# Positive control on a REAL Kaggle output for the a .2 pw code path: the a21 pw20 gate on the scored pw20 57425269 committed dir.
import json, os, sys, hashlib, tempfile, shutil, subprocess
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
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
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

K = V_ + '/kernels'; A21 = R + '/session_2026-10-20/a21'; SH = R + '/session_2026-10-20/push/shape'; P2 = V_ + '/picks2_work/build'
nbp = lambda d, t: d + '/kernel_' + t + '/rsna_sub_' + t + '.ipynb'
NB = {'c4': R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb',
      'R6': R + '/session_2026-10-15/r6entry/kernel_v224r6flips6fj/rsna_sub_v224r6flips6fj.ipynb',
      'pw20': nbp(A21, 'v224r6c4pw20j'), 'pw30': nbp(A21, 'v224r6c4pw30j'),
      'pw25': nbp(K, 'v224r6c4pw25j'), 'shape_pw25': nbp(SH, 'v224r6c4pw25j'),
      'R6pw20': nbp(K, 'v224r6flips6fpw20j'), 'p2_R6pw20': nbp(P2, 'v224r6flips6fpw20j')}
COMMITTED = {'c4': R + '/session_2026-10-18/a18/out_v224r6a25c4j_v1', 'r6': R + '/session_2026-10-15/r6entry/out_v224r6flips6fj_v1',
             'pw20': A21 + '/out_v224r6c4pw20j_v1', 'pw30': A21 + '/out_v224r6c4pw30j_v1'}

def cell18(p):
    nb = json.load(open(p, encoding='utf-8')); s_ = src(nb['cells'][18])
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
    fn = cell18(p); W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    shutil.copyfile(cdir + '/submission_v224flip.csv', W + '/submission_v224flip.csv'); os.makedirs(W + '/s6f/B')
    shutil.copyfile(cdir + '/s6f/B/B_probs.csv', W + '/s6f/B/B_probs.csv')
    rc = json.load(open(cdir + '/s6f_receipt.json', encoding='utf-8'))
    ns = {}; exec(fn, ns); rec = ns['_s6f_blend'](rc['setup'], rc['run'])
    b = open(W + '/submission.csv', 'rb').read(); out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})
    shutil.rmtree(W, ignore_errors=True); return dict(rec=rec, bytes=b, out=out, receipt=rc, cdir=cdir)

Wp = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
def expect(o, a):
    U = {c: pct(o[c].values.astype(np.float64)) for c in OA}; e = {}
    for c in OA:
        oo = [k for k in OA if k != c]
        e[c] = pct(U[c] - a / 2.0 * sum(Wp[tuple(sorted((c, k), key=OA.index))] * U[k] for k in oo))
    return e

ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name, flush=True)

print('== static checks ==')
chk('E1 pw25 val/kernels == push/shape (notebook sha 64ed7eba, metadata bytes)', sha(NB['pw25']) == sha(NB['shape_pw25'])
    and sha(NB['pw25']).startswith('64ed7eba') and sha(K + '/kernel_v224r6c4pw25j/kernel-metadata.json') == sha(SH + '/kernel_v224r6c4pw25j/kernel-metadata.json'))
chk('E2 R6pw20 val/kernels == picks2_work/build (notebook sha e4d92c0a, metadata bytes)', sha(NB['R6pw20']) == sha(NB['p2_R6pw20'])
    and sha(NB['R6pw20']).startswith('e4d92c0a') and sha(K + '/kernel_v224r6flips6fpw20j/kernel-metadata.json') == sha(P2 + '/kernel_v224r6flips6fpw20j/kernel-metadata.json'))
J = {k: json.load(open(p, encoding='utf-8')) for k, p in NB.items()}
dc = lambda a, b: [i for i, (x, y) in enumerate(zip(J[a]['cells'], J[b]['cells'])) if src(x) != src(y)]
l18 = lambda k: src(J[k]['cells'][18]).splitlines()
d25 = [(x, y) for x, y in zip(l18('pw25'), l18('pw30')) if x != y]
chk('E3 pw25 vs c4: only cell 18 differs (%s); pw25 vs scored pw30 57425970: only cell 18, one line (the A_OA constant): %s'
    % (dc('pw25', 'c4'), [x.strip()[:24] for x, _ in d25]), dc('pw25', 'c4') == [18] and dc('pw25', 'pw30') == [18] and len(d25) == 1
    and len(l18('pw25')) == len(l18('pw30')) and d25[0][0].strip().startswith('A_OA = 0.25') and d25[0][1].strip().startswith('A_OA = 0.3'))
chk('E4 R6pw20 vs scored pw20 57425269: only cell 14 differs (%s) == c4 vs R6 (%s); cell 18 byte-equal' % (dc('R6pw20', 'pw20'), dc('c4', 'R6')),
    dc('R6pw20', 'pw20') == [14] and dc('c4', 'R6') == [14] and src(J['R6pw20']['cells'][18]) == src(J['pw20']['cells'][18]))
M = lambda p: json.load(open(os.path.dirname(p) + '/kernel-metadata.json', encoding='utf-8'))
strip = lambda m: {k: v for k, v in m.items() if k not in ('id', 'title', 'code_file')}
chk('E5 metadata: pw25 == scored pw30 and R6pw20 == R6 except id / title / code_file; ids %s, %s' % (M(NB['pw25'])['id'], M(NB['R6pw20'])['id']),
    strip(M(NB['pw25'])) == strip(M(NB['pw30'])) and strip(M(NB['R6pw20'])) == strip(M(NB['R6']))
    and M(NB['pw25'])['id'] == 'eeshsaxena/rsna-sub-v224r6c4pw25j' and M(NB['R6pw20'])['id'] == 'eeshsaxena/rsna-sub-v224r6flips6fpw20j')

print('== gold58 replay (n %d; INFO + code check) ==' % n)
GR = {k: run_gold(NB[k], VC4 if b == 'c4' else VR6) for k, b in (('c4', 'c4'), ('pw25', 'c4'), ('pw30', 'c4'), ('pw20', 'c4'), ('R6', 'r6'), ('R6pw20', 'r6'))}
for k, ctl, a in (('pw25', 'c4', .25), ('R6pw20', 'R6', .2)):
    e = expect(GR[ctl]['out'], a); o = GR[k]['out']; c0 = GR[ctl]['out']
    md = max(float(np.abs(o[c].values - e[c].astype(np.float32)).max()) for c in OA)
    chk('G5 %s gold58 OA == independent recompute from %s (maxdiff %.1e), non-OA identical, receipt a_oa %g mode pw S6F' % (k, ctl, md, a),
        md < 1e-6 and (o[non].values == c0[non].values).all() and GR[k]['rec'].get('a_oa') == a and GR[k]['rec'].get('oa_mode') == 'pw' and GR[k]['rec'].get('mode') == 'S6F')
mac = lambda o: np.mean([auc(o[c].values, Y[:, j]) for j, c in enumerate(L)])
for k, ref in (('c4', 'c4'), ('pw20', 'c4'), ('pw25', 'c4'), ('pw30', 'c4'), ('R6', 'R6'), ('R6pw20', 'R6')):
    o = GR[k]['out']; print('info %-7s gold58 macro %.5f (%+.5f vs %s) | OA AUCs %s' % (k, mac(o), mac(o) - mac(GR[ref]['out']), ref,
                                                                                [round(float(auc(o[c].values, Y[:, L.index(c)])), 4) for c in OA]))

print('== committed-input replay (3 public rows, the runs\' own setup / run dicts) ==')
CR = {'c4': run_committed(NB['c4'], COMMITTED['c4']), 'R6': run_committed(NB['R6'], COMMITTED['r6']),
      'pw25': run_committed(NB['pw25'], COMMITTED['c4']), 'R6pw20': run_committed(NB['R6pw20'], COMMITTED['r6']),
      'pw20_own': run_committed(NB['pw20'], COMMITTED['pw20'])}
spw20 = open(COMMITTED['pw20'] + '/submission.csv', 'rb').read(); rpw20 = json.load(open(COMMITTED['pw20'] + '/s6f_receipt.json', encoding='utf-8'))['blend']
chk('G3c scored pw20 57425269 notebook replayed on its OWN committed inputs == its downloaded submission.csv (LF sha %s == receipt %s)'
    % (shb(lf(CR['pw20_own']['bytes']))[:8], rpw20['submission_sha256'][:8]), lf(CR['pw20_own']['bytes']) == spw20 and shb(spw20) == rpw20['submission_sha256'])
for k, ctl, a in (('pw25', 'c4', .25), ('R6pw20', 'R6', .2)):
    o, c0 = CR[k]['out'], CR[ctl]['out']; e = expect(c0, a)
    md = max(float(np.abs(o[c].values - e[c].astype(np.float32)).max()) for c in OA)
    chk('G7 %s committed-input replay: OA == independent recompute from the %s committed output (maxdiff %.1e), non-OA identical, receipt a_oa %g pw'
        % (k, ctl, md, a), md < 1e-6 and (o[non].values == c0[non].values).all() and CR[k]['rec'].get('a_oa') == a and CR[k]['rec'].get('oa_mode') == 'pw')
for k in ('c4', 'R6'):
    cd = COMMITTED[k.lower()]
    chk('G6 base %s replay == its committed submission.csv (LF)' % k, lf(CR[k]['bytes']) == open(cd + '/submission.csv', 'rb').read())
sig = lambda k: tuple(np.round(CR[k]['out'][OA].values.ravel(), 6))
spw30 = tuple(np.round(pd.read_csv(COMMITTED['pw30'] + '/submission.csv')[OA].values.ravel(), 6))
print('info: 3-row public OA output pw25 == scored pw30 output: %s; == c4: %s' % (sig('pw25') == spw30, sig('pw25') == sig('c4')))

SIM = V_ + '/sim'
def write_sim(tag, r):
    cdir, recd, outb = r['cdir'], r['rec'], r['bytes']
    d = SIM + '/out_%s_sim' % tag; shutil.rmtree(d, ignore_errors=True); shutil.copytree(cdir, d, ignore=shutil.ignore_patterns('__pycache__'))
    open(d + '/submission.csv', 'wb').write(lf(outb))
    rc = json.load(open(cdir + '/s6f_receipt.json', encoding='utf-8')); rc['blend'] = recd; rc['mode'] = recd.get('mode'); rc['fired'] = list(recd.get('fired') or [])
    rc['SIMULATED'] = 'val/ext_a22.py: committed inputs of %s + cell 18 of %s replayed locally; NOT a Kaggle run' % (os.path.basename(cdir), tag)
    json.dump(rc, open(d + '/s6f_receipt.json', 'w', encoding='utf-8'), indent=1, default=str)
    lg = [p for p in os.listdir(d) if p.endswith('.log') and 'cufile' not in p]; assert len(lg) == 1, lg
    try:
        L_ = json.load(open(d + '/' + lg[0], encoding='utf-8'))
        L_.append({'stream_name': 'stdout', 'time': 0, 'data': 'EOA_APPLIED a_oa %g mode %s on %s [SIMULATED]\n' % (recd['a_oa'], recd.get('oa_mode'), OA)})
        json.dump(L_, open(d + '/' + lg[0], 'w', encoding='utf-8'))
    except json.JSONDecodeError:
        open(d + '/' + lg[0], 'a', encoding='utf-8').write('\nEOA_APPLIED a_oa %g [SIMULATED]\n' % recd['a_oa'])
    print('SIM written', d); return d
write_sim('v224r6c4pw25j', CR['pw25']); write_sim('v224r6flips6fpw20j', CR['R6pw20'])

# registry extension (additive; existing five entries untouched)
SL = V_ + '/slate_a22.json'; S = json.load(open(SL, encoding='utf-8'))
C4S = json.dumps(C4T); R6S = json.dumps({"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5})
assert C4S == S['entries']['pw35']['gate_args'][3] and R6S == S['entries']['r6pw30']['gate_args'][3], (C4S, R6S)
kd = lambda t: 'session_2026-10-20/val/kernels/kernel_' + t
NEW = {
    'pw25': dict(tag='v224r6c4pw25j', slug='eeshsaxena/rsna-sub-v224r6c4pw25j', kernel_dir=kd('v224r6c4pw25j'),
                 nb_sha256=sha(NB['pw25']), meta_sha256=sha(K + '/kernel_v224r6c4pw25j/kernel-metadata.json'), base='c4', a_oa=0.25, oa_mode='pw',
                 gate_script='session_2026-10-20/a21/probe_check_eoa21.py', gate_args=['--a224', '0.25', '--table', C4S, '--aoa', '0.25', '--mode', 'pw'],
                 gate_pass='PASS 17/17', desc='A22-PW25: pw30 best 57425970 shape family, c4 + OA decorrelation pair-weighted a .25 (one change)'),
    'r6pw20': dict(tag='v224r6flips6fpw20j', slug='eeshsaxena/rsna-sub-v224r6flips6fpw20j', kernel_dir=kd('v224r6flips6fpw20j'),
                   nb_sha256=sha(NB['R6pw20']), meta_sha256=sha(K + '/kernel_v224r6flips6fpw20j/kernel-metadata.json'), base='r6', a_oa=0.2, oa_mode='pw',
                   gate_script='session_2026-10-20/a21/probe_check_eoa21.py', gate_args=['--a224', '0.35', '--table', R6S, '--aoa', '0.2', '--mode', 'pw'],
                   gate_pass='PASS 17/17', desc='A22-R6PW20: B hedge, R6 57248084 + OA decorrelation pair-weighted a .2 (cell 18 == scored pw20 57425269)')}
for k, e in NEW.items():
    if k in S['entries']: assert S['entries'][k] == e, ('registry entry differs', k)
    S['entries'][k] = e
S['note'] = S['note'] + ' | 10-20 ~20:30 IST: pw25 + r6pw20 added by val/ext_a22.py (additive; offline gate + sims + gate selftest in ext_a22.log).'
json.dump(S, open(SL, 'w', encoding='utf-8'), indent=1)
print('REGISTRY', SL, 'keys', list(S['entries']))

print('== commit-gate rows (exact argv; expectation stated first) ==')
PY = ['py', '-3.12', '-I']; sim = lambda t: SIM + '/out_%s_sim' % t
cmd = lambda key, out: PY + [R + '/' + S['entries'][key]['gate_script'], out] + S['entries'][key]['gate_args']
g21 = lambda out, a224, table, aoa: PY + [R + '/session_2026-10-20/a21/probe_check_eoa21.py', out, '--a224', str(a224), '--table', table, '--aoa', str(aoa), '--mode', 'pw']
ROWS = [('pw25 gate on pw25 sim', cmd('pw25', sim('v224r6c4pw25j')), 'PASS'),
        ('r6pw20 gate on R6pw20 sim', cmd('r6pw20', sim('v224r6flips6fpw20j')), 'PASS'),
        ('pw20 (c4 params, --aoa .2) gate on the REAL scored pw20 57425269 committed dir (same cell 18 as r6pw20)', g21(COMMITTED['pw20'], 0.25, C4S, 0.2), 'PASS'),
        ('NEG pw25 gate on the REAL scored pw30 committed dir (dose swap)', cmd('pw25', COMMITTED['pw30']), 'FAIL'),
        ('NEG pw25 gate on the REAL c4 committed dir (no E-OA)', cmd('pw25', COMMITTED['c4']), 'FAIL'),
        ('NEG r6pw20 gate on R6pw30 sim (dose swap)', cmd('r6pw20', sim('v224r6flips6fpw30j')), 'FAIL'),
        ('NEG r6pw20 gate on the REAL R6 committed dir (no E-OA)', cmd('r6pw20', COMMITTED['r6']), 'FAIL'),
        ('NEG r6pw20 gate on the REAL scored pw20 committed dir (base swap)', cmd('r6pw20', COMMITTED['pw20']), 'FAIL')]
for name, argv, exp in ROWS:
    p = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=R)
    last = (p.stdout.strip().splitlines() or [''])[-1]
    got = 'PASS' if 'PROBE_CHECK PASS' in last else ('FAIL' if 'PROBE_CHECK FAIL' in last else 'ERR')
    fails = [l.strip() for l in p.stdout.splitlines() if l.strip().startswith('FAIL')]
    good = got == exp and (exp == 'FAIL' or last.endswith('PASS 17/17'))
    chk('%s: expect %s got %s | %s%s' % (name, exp, got, last, (' | first FAIL: ' + fails[0][:100]) if fails else ''), good)
print('EXT_A22', 'PASS' if ok else 'FAIL')
