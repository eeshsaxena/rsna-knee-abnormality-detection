#!/usr/bin/env python
# build_swide08.py (10-22 ~09:10 IST, Claude, PUBLIC-PUSH track). LOCAL BUILD ONLY, UNARMED: nothing pushed, committed, registered, submitted.
# swide08 = swide (A22 57464326, stack/kernels/kernel_v224r6c4pw25lomm05mcl10j, nb c9801a80) with ONE literal changed in cell 18:
#   A_LOAMM = 0.05 -> A_LOAMM = 0.08   (Lateral OA <- Medial Meniscus dose .05 -> .08; nothing else)
# Pattern of stack/kernels/build_stack_kernels.py + levers/build_moacf.py. Asserts: base nb + meta sha (full), json round-trip byte-exact,
# only cell 18 differs, the cell-18 diff is exactly that one line, metadata differs only in id / title / code_file, and the reverse literal
# substitution reproduces the swide notebook byte for byte.
# Offline FACTORIAL gate (gold58 serve inputs, cbase exec route of levers/build_moacf.py, no GPU), each notebook's OWN cell 18 replayed:
#   F1 all-off: swide08 with A_LOAMM .08 -> .05 gives submission.csv BYTES identical to swide's replay;
#   F2 swide08 vs swide: the 11 other columns identical, only Lateral OA differs;
#   F3 Lateral OA == python reference pct(pct(LOA_0) - .08 pct(MM)) with LOA_0 from the A_LOAMM -> 0 replay (and swide LOA == same at .05);
#   F4 stdout markers EOA_APPLIED, LOAMM_APPLIED a_loamm 0.08, MCLMM_APPLIED a_mclmm 0.1, S6F_BLEND_OK in that order; receipt differs from
#      swide's only in a_loamm (+ submission_sha256);
#   F5 lomm off in both (A_LOAMM -> 0.0): swide08 and swide replays byte-identical; F6 finite, shape 58 x 12. gold58 macro = INFO only.
# Writes ONLY under session_2026-10-22/final/kernels_pub/. Usage: py -3.12 -I build_swide08.py
import json, hashlib, os, sys, difflib, datetime, tempfile, shutil, io, contextlib
import numpy as np, pandas as pd
from scipy.stats import rankdata
os.linesep = '\n'
HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/'); OUTD = HERE + '/kernels_pub'; os.makedirs(OUTD, exist_ok=True)
R = 'C:/Users/eeshs/Downloads/rsna-knee'; S = R + '/session_2026-10-20'
BASE_DIR = S + '/stack/kernels/kernel_v224r6c4pw25lomm05mcl10j'
BASE_NB_SHA = 'c9801a80bbc64ecbd1aba15e9470657fca4631ee7ac091f68b7742d1e17d7478'
BASE_META_SHA = '6a2745e7f68841c50f200bdc40e1259e95452237163885c67ed73473ec7f8d1e'
CELL = 18; A_OLD, A_NEW = 0.05, 0.08
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
sha = lambda b: hashlib.sha256(b).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
dump = lambda nb: json.dumps(nb, indent=1, ensure_ascii=False).replace('\n', '\r\n').encode('utf-8')
LOG = open(OUTD + '/build_swide08.log', 'w', encoding='utf-8')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')


builder_sha = sha(open(os.path.abspath(__file__), 'rb').read())
PFX = "        A_LOAMM = "


def subst(nbb, old, new):
    nb = json.loads(nbb); lines = src(nb['cells'][CELL]).split('\n')
    k = [j for j, l in enumerate(lines) if l.startswith(PFX)]; assert len(k) == 1, k
    o = PFX + '%r   #' % old; assert lines[k[0]].startswith(o), lines[k[0]]
    lines[k[0]] = lines[k[0]].replace(o, PFX + '%r   #' % new, 1); nb['cells'][CELL]['source'] = '\n'.join(lines); return dump(nb)


meta0b = open(BASE_DIR + '/kernel-metadata.json', 'rb').read(); meta0 = json.loads(meta0b)
raw = open(BASE_DIR + '/' + meta0['code_file'], 'rb').read()
assert sha(raw) == BASE_NB_SHA, ('base nb sha changed', sha(raw)); assert sha(meta0b) == BASE_META_SHA, ('base meta sha changed', sha(meta0b))
assert dump(json.loads(raw)) == raw, 'base json round-trip not byte-exact'
assert meta0['id'] == 'eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10j'
P('BASE swide A22 57464326: nb %s | meta %s | json round-trip exact' % (BASE_NB_SHA, BASE_META_SHA))
newb = subst(raw, A_OLD, A_NEW)
assert subst(newb, A_NEW, A_OLD) == raw, 'reverse substitution does not reproduce swide'
tag = 'v224r6c4pw25lomm08mcl10j'; cf = 'rsna_sub_%s.ipynb' % tag; od = OUTD + '/kernel_' + tag; os.makedirs(od, exist_ok=True)
open(od + '/' + cf, 'wb').write(newb)
meta = dict(meta0); meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(od + '/kernel-metadata.json', 'w'), indent=2); metab = open(od + '/kernel-metadata.json', 'rb').read()
open(od + '/NOT_ARMED.txt', 'w').write('UNARMED local build (session_2026-10-22/final/build_swide08.py). Push only after Eesh GO + registry merge + commit gate PASS 26/26.\n')
A_, B_ = json.loads(raw), json.loads(newb)
assert len(A_['cells']) == len(B_['cells']) and A_['metadata'] == B_['metadata'] and A_.get('nbformat') == B_.get('nbformat')
dc = [j for j, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if x != y]; assert dc == [CELL], dc
cx, cy = dict(A_['cells'][CELL]), dict(B_['cells'][CELL]); cx.pop('source'); cy.pop('source'); assert cx == cy
ud = [l for l in difflib.unified_diff(src(A_['cells'][CELL]).split('\n'), src(B_['cells'][CELL]).split('\n'), lineterm='', n=0)
      if l[:1] in '+-' and not l.startswith(('+++', '---'))]
assert len(ud) == 2 and ud[0].startswith('-' + PFX + '0.05   #') and ud[1] == '+' + ud[0][1:].replace(PFX + '0.05   #', PFX + '0.08   #', 1), ud
m1 = json.loads(metab); assert {q for q in set(meta0) | set(m1) if meta0.get(q) != m1.get(q)} == {'id', 'title', 'code_file'}

# ---- offline factorial gate on gold58 (inputs exactly as levers/build_moacf.py)
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
        exec(fn, ns); rec = ns['_s6f_blend'](setup, run)
    b = open(W + '/submission.csv', 'rb').read()
    o = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64)
    shutil.rmtree(W, ignore_errors=True); return o, b, buf.getvalue(), dict(rec or {})


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p; return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


ob, bb, lb, rb = run_cell18(raw)                              # swide
on, bn, ln, rn = run_cell18(newb)                             # swide08
o1, b1, l1, r1 = run_cell18(subst(newb, A_NEW, A_OLD))        # swide08 with its one change switched off (-> .05)
o0, b0, l0, r0 = run_cell18(subst(newb, A_NEW, 0.0))          # swide08 lomm off
s0 = run_cell18(subst(raw, A_OLD, 0.0))                       # swide lomm off
jl, jm = L.index('Lateral OA'), L.index('Medial Meniscus'); others = [j for j in range(12) if j != jl]
F1 = b1 == bb
F2 = bool(np.array_equal(ob[:, others], on[:, others]))
ref = lambda a: pctv(pctv(o0[:, jl]) - a * pctv(o0[:, jm])).astype(np.float32).astype(np.float64)
F3 = float(np.abs(ref(A_NEW) - on[:, jl]).max()) < 1e-6 and float(np.abs(ref(A_OLD) - ob[:, jl]).max()) < 1e-6 and not np.array_equal(on[:, jl], ob[:, jl])
mk = [x for x in ln.splitlines() if x.startswith(('EOA_APPLIED', 'LOAMM_APPLIED', 'MCLMM_APPLIED', 'S6F_BLEND_OK'))]
F4 = ([x.split()[0] for x in mk] == ['EOA_APPLIED', 'LOAMM_APPLIED', 'MCLMM_APPLIED', 'S6F_BLEND_OK'] and mk[1] == 'LOAMM_APPLIED a_loamm 0.08'
      and mk[2] == 'MCLMM_APPLIED a_mclmm 0.1' and rn.get('a_loamm') == A_NEW and rb.get('a_loamm') == A_OLD
      and {k for k in set(rn) | set(rb) if rn.get(k) != rb.get(k)} <= {'a_loamm', 'submission_sha256'})
F5 = b0 == s0[1] and 'LOAMM_APPLIED' not in l0
F6 = bool(np.isfinite(on).all()) and on.shape == (58, 12)
gate = all([F1, F2, F3, F4, F5, F6])
mb = np.mean([auc(ob[:, j], Y58[:, j]) for j in range(12)]); mn = np.mean([auc(on[:, j], Y58[:, j]) for j in range(12)])
lb_, ln_ = auc(ob[:, jl], Y58[:, jl]), auc(on[:, jl], Y58[:, jl])
P('FACTORIAL gold58: F1 all-off (.08 -> .05) bytes == swide %s | F2 11 other columns identical %s | F3 LOA == reference (.08 and swide .05), LOA moved %s | '
  'F4 markers/receipt %s %s | F5 lomm-off swide08 == lomm-off swide %s | F6 finite 58x12 %s -> %s' % (F1, F2, F3, F4, mk[:3], F5, F6, 'PASS' if gate else 'FAIL'))
P('INFO gold58 (n 58, never a ruler): macro %.5f -> %.5f | Lateral OA AUC %.4f -> %.4f | LOA maxdiff vs swide %.4f' % (mb, mn, lb_, ln_, float(np.abs(on[:, jl] - ob[:, jl]).max())))
rc = dict(probe='public-push swide08', base='swide A22 57464326', base_kernel_dir=BASE_DIR.replace(R + '/', ''), base_nb_sha256=BASE_NB_SHA,
          base_meta_sha256=BASE_META_SHA, tag=tag, slug='eeshsaxena/rsna-sub-' + tag, a_oa=0.25, oa_mode='pw', a_loamm=A_NEW, a_mclmm=0.1,
          out_nb_sha256=sha(newb), meta_sha256=sha(metab), cell=CELL, cells_changed=dc, cell18_diff=ud,
          offline_factorial_gate=dict(F1_alloff_bytes_eq_swide=F1, F2_others_identical=F2, F3_loa_reference=F3, F4_markers_receipt=F4,
                                      F5_lomm_off_eq=F5, F6_finite=F6, PASS=gate),
          gold58_INFO=dict(macro_base=mb, macro_new=mn, loa_auc_base=lb_, loa_auc_new=ln_), builder_sha256=builder_sha,
          built_at_ist=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), armed=False,
          note='local build only; needs Eesh GO + registry merge + stack/kernels/stack_commit_gate.py PASS 26/26 (--aloamm 0.08) before any submit')
json.dump(rc, open(OUTD + '/build_receipt_%s.json' % tag, 'w', encoding='utf-8'), indent=1)
P('SWIDE08_BUILT %s nb %s | meta %s | offline factorial gate %s | builder %s' % (tag, sha(newb), sha(metab), 'PASS' if gate else 'FAIL', builder_sha[:16]))
LOG.close(); sys.exit(0 if gate else 1)
