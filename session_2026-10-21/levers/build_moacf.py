#!/usr/bin/env python
# build_moacf.py (10-21 levers, Claude, REGIME track). LOCAL BUILD ONLY, UNARMED: nothing is pushed, committed, registered or submitted.
# Adds ONE block to cell 18 (_s6f_blend, after E-OA and after any LOAMM / MCLMM block, right before the 'v = out[labels]' line) of four
# already-built bases (read only): MOA = pct(pct(MOA) - .05 pct(Contusion)), then MOA = pct(pct(MOA) - .05 pct(Fracture)) on the final output
# ("MOACF", regime.md sec 3). Bases: pw30 scored 57425970 (a21), pw25 / spl / swide as built for A22 (val/kernels, stack/skeptic/kernels,
# stack/kernels). Asserts: base nb + meta sha, json round-trip byte-exact, only cell 18 differs, the cell-18 diff is exactly the block,
# metadata differs only in id / title / code_file. Offline gate (gold58, cbase exec route, no GPU): base vs new final output equal in the
# 11 other columns, Medial OA equal to the python reference within float32, MOACF_APPLIED printed; gold58 macro printed as INFO.
# Writes ONLY under session_2026-10-21/levers/kernels/. Usage: py -3.12 -I build_moacf.py
import json, hashlib, os, sys, difflib, datetime, tempfile, shutil, io, contextlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(HERE, 'kernels'); os.makedirs(OUTD, exist_ok=True)
R = 'C:/Users/eeshs/Downloads/rsna-knee'; S = R + '/session_2026-10-20'
BASES = {  # name: (kernel dir, tag, nb sha256 from the A21 receipt / A22 registry)
    'pw30': (S + '/a21/kernel_v224r6c4pw30j', 'v224r6c4pw30j', '5a903187415505e9e3566ebd749cae1328b20df6b60bd6593e6443b17fc0c9fd'),
    'pw25': (S + '/val/kernels/kernel_v224r6c4pw25j', 'v224r6c4pw25j', '64ed7eba5880d6efaa8555042d398b39f547baa23b91f5e2ba4f962938fa8697'),
    'spl': (S + '/stack/skeptic/kernels/kernel_v224r6c4pw25lomm05mcl00j', 'v224r6c4pw25lomm05mcl00j', 'e641e88fcaa5c41f'),
    'swide': (S + '/stack/kernels/kernel_v224r6c4pw25lomm05mcl10j', 'v224r6c4pw25lomm05mcl10j', 'c9801a80bbc64ecb')}
CELL = 18; A_C, A_F = 0.05, 0.05
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
sha = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
dump = lambda nb: json.dumps(nb, indent=1, ensure_ascii=False).replace('\n', '\r\n').encode('utf-8')
LOG = open(os.path.join(OUTD, 'build_moacf.log'), 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
BLOCK = [
    "        A_MOACF = (%r, %r)   # [levers regime MOACF] Medial OA decorrelated from Contusion then Fracture on the final output (session_2026-10-21/levers/regime.md); (0, 0) = no-op" % (A_C, A_F),
    "        if any(A_MOACF) and {'Medial OA', 'Contusion', 'Fracture'} <= set(labels) and not {'Medial OA', 'Contusion', 'Fracture'} & set(blank):",
    "            _moa = out['Medial OA'].values.astype(np.float64)",
    "            _moa = pct(pct(_moa) - A_MOACF[0] * pct(out['Contusion'].values.astype(np.float64)))",
    "            _moa = pct(pct(_moa) - A_MOACF[1] * pct(out['Fracture'].values.astype(np.float64)))",
    "            out['Medial OA'] = _moa.astype(np.float32)",
    "            rec['a_moacf'] = list(A_MOACF)",
    "            print('MOACF_APPLIED a_moacf %g %g' % A_MOACF, flush=True)"]
ANCHOR = '        v = out[labels].values.astype(np.float64)'
builder_sha = sha(open(os.path.abspath(__file__), 'rb').read())

# ---- gold58 inputs for the offline gate (same construction as session_2026-10-20/push/cbase.py)
G = R + '/session_2026-10-18/next/g58'
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g58 = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g58.StudyInstanceUID.tolist(); n = len(ids); Y58 = g58[L].values.astype(int)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
pctv = lambda v: rankdata(v) / len(v)
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
V = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4W.get(c, .5); V[:, j] = w * (.75 * pctv(R384[:, j]) + .25 * pctv(R224[:, j])) + (1 - w) * pctv(R6[:, j])


def run_cell18(nbb):
    nb = json.loads(nbb); s_ = src(nb['cells'][CELL]); fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]
    W = tempfile.mkdtemp().replace(chr(92), '/'); assert fn.count("W = '/kaggle/working'") == 1
    fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    shf = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    setup = {'base_sha256': shf(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': shf(W + '/s6f/B/B_probs.csv')}}}
    ns = {}; buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(fn, ns); ns['_s6f_blend'](setup, run)
    o = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64)
    shutil.rmtree(W, ignore_errors=True); return o, buf.getvalue()


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p; return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


receipts = {}
for name, (kd, tag, nsha) in BASES.items():
    meta0b = open(os.path.join(kd, 'kernel-metadata.json'), 'rb').read(); meta0 = json.loads(meta0b)
    raw = open(os.path.join(kd, meta0['code_file']), 'rb').read()
    assert sha(raw).startswith(nsha), (name, 'base nb sha changed', sha(raw)); assert dump(json.loads(raw)) == raw, (name, 'json round-trip not byte-exact')
    nb = json.loads(raw); lines = src(nb['cells'][CELL]).split('\n')
    e = [j for j, l in enumerate(lines) if l.startswith("            print('EOA_APPLIED a_oa")]; assert len(e) == 1, e
    k = [j for j, l in enumerate(lines) if l == ANCHOR and j > e[0]]; assert len(k) == 1, (name, k)
    assert 'A_MOACF' not in src(nb['cells'][CELL])
    lines[k[0]:k[0]] = BLOCK; nb['cells'][CELL]['source'] = '\n'.join(lines)
    ntag = tag[:-1] + 'moacf05j'; cf = 'rsna_sub_%s.ipynb' % ntag; od = os.path.join(OUTD, 'kernel_' + ntag); os.makedirs(od, exist_ok=True)
    newb = dump(nb); open(os.path.join(od, cf), 'wb').write(newb)
    meta = dict(meta0); meta.update(id='eeshsaxena/rsna-sub-' + ntag, title='rsna sub ' + ntag, code_file=cf)
    json.dump(meta, open(os.path.join(od, 'kernel-metadata.json'), 'w'), indent=2); metab = open(os.path.join(od, 'kernel-metadata.json'), 'rb').read()
    A_, B_ = json.loads(raw), json.loads(newb)
    assert len(A_['cells']) == len(B_['cells']) and A_['metadata'] == B_['metadata']
    dc = [j for j, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if x != y]; assert dc == [CELL], dc
    ud = [l for l in difflib.unified_diff(src(A_['cells'][CELL]).split('\n'), src(B_['cells'][CELL]).split('\n'), lineterm='', n=0)
          if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    assert ud == ['+' + l for l in BLOCK], ud
    m1 = json.loads(metab); assert {q for q in set(meta0) | set(m1) if meta0.get(q) != m1.get(q)} == {'id', 'title', 'code_file'}
    # offline gate on gold58
    ob, lb = run_cell18(raw); on, ln = run_cell18(newb)
    jm = L.index('Medial OA'); others = [j for j in range(12) if j != jm]
    g1 = bool(np.array_equal(ob[:, others], on[:, others]))
    ref = pctv(pctv(ob[:, jm]) - A_C * pctv(ob[:, L.index('Contusion')])); ref = pctv(pctv(ref) - A_F * pctv(ob[:, L.index('Fracture')]))
    g2 = float(np.abs(ref.astype(np.float32).astype(np.float64) - on[:, jm]).max()) < 1e-6
    g3 = 'MOACF_APPLIED a_moacf 0.05 0.05' in ln and 'MOACF_APPLIED' not in lb
    g4 = bool(np.isfinite(on).all()) and on.shape == (58, 12)
    mb = np.mean([auc(ob[:, j], Y58[:, j]) for j in range(12)]); mn = np.mean([auc(on[:, j], Y58[:, j]) for j in range(12)])
    ab, an = auc(ob[:, jm], Y58[:, jm]), auc(on[:, jm], Y58[:, jm])
    gate = all([g1, g2, g3, g4])
    rc = dict(probe='levers MOACF', base=name, base_kernel_dir=kd.replace(R + '/', ''), base_nb_sha256=sha(raw), base_meta_sha256=sha(meta0b), tag=ntag,
              slug='eeshsaxena/rsna-sub-' + ntag, a_moacf=[A_C, A_F], out_nb_sha256=sha(newb), meta_sha256=sha(metab), cell=CELL, cells_changed=dc,
              cell18_diff=ud, offline_gate=dict(others_identical=g1, moa_equals_reference=g2, applied_print=g3, finite_shape=g4, PASS=gate),
              gold58_INFO=dict(macro_base=mb, macro_new=mn, moa_auc_base=ab, moa_auc_new=an), builder_sha256=builder_sha,
              built_at_ist=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), armed=False,
              note='local build only; needs Eesh GO + PREREG amendment + the commit gate pattern (stack/kernels/stack_commit_gate.py) before any push')
    json.dump(rc, open(os.path.join(OUTD, 'build_receipt_%s.json' % ntag), 'w', encoding='utf-8'), indent=1); receipts[name] = rc
    P('MOACF_BUILT %-6s %-34s nb %s | gate %s (others identical %s, MOA = ref %s, print %s) | gold58 INFO macro %.5f -> %.5f, MOA %.4f -> %.4f' % (
        name, ntag, sha(newb)[:16], 'PASS' if gate else 'FAIL', g1, g2, g3, mb, mn, ab, an))
json.dump({k: dict(tag=v['tag'], nb_sha256=v['out_nb_sha256'], gate=v['offline_gate']['PASS']) for k, v in receipts.items()},
          open(os.path.join(OUTD, 'build_index_moacf.json'), 'w'), indent=1)
P('MOACF_BUILD DONE %d kernels, builder %s' % (len(receipts), builder_sha[:16])); LOG.close()
