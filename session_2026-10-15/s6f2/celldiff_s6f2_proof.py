#!/usr/bin/env python
# celldiff_s6f2_proof.py (10-15, Claude subagent): proof that an s6f2 kernel is EXACTLY its pushed base notebook with ONLY 3 code
# cells APPENDED, and that those 3 cells are the pushed S6F kernel's cells (s6f/kernel_v224flips6f, nb 9c0e2558, cells 15-17)
# except exactly the listed splices. Independent of build_v224s6f2.py (imports nothing from it; every expected splice text is
# written out here). No em dashes.
#   --kernel flrm     kernel_v224flrms6f    vs v224rm/kernel_v224flrm   (eeshsaxena/rsna-sub-v224flrm v1, f14f6acc, 17 cells)
#   --kernel flips20  kernel_v224flips6fs20 vs v224f/kernel_v224flips20 (eeshsaxena/rsna-sub-v224flips20 v1, d2066b8c, 24 cells)
#   usage: py -3.12 -I celldiff_s6f2_proof.py --kernel flrm|flips20 [--nb NOTEBOOK] [--meta kernel-metadata.json] [--build RECEIPT]
#          -> S6F2_CELLDIFF PASS/FAIL n/m
import os, sys, json, base64, hashlib, difflib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
ROOT = os.path.dirname(SESS)
K = {'flrm': dict(tag='v224flrms6f', n=17, sha='f14f6accc7c178536ff8a5d8d70a65da63ca05b2dd6d2a7ce9673100f9178ecf', G='6.0', last='V224RM_DONE',
                  nb=os.path.join(SESS, 'v224rm', 'kernel_v224flrm', 'rsna_sub_v224flrm.ipynb'),
                  meta=os.path.join(SESS, 'v224rm', 'kernel_v224flrm', 'kernel-metadata.json'),
                  pulled=os.path.join(SESS, 'v224rm', 'pulled_v224flrm', 'rsna-sub-v224flrm.ipynb'),
                  hdr=('# [s6f2 base flrm] The cells above are V224-FLIP-RM (eeshsaxena/rsna-sub-v224flrm v1) unchanged and in order, '
                       'ending with V224RM_DONE. In the S6F text below\n')),
     'flips20': dict(tag='v224flips6fs20', n=24, sha='d2066b8c9c5e3a77a20fa45fc5a0f88db8926566e205a66c49536022df2f17c5', G='6.25',
                     last='V224F_DONE', nb=os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'rsna_sub_v224flips20.ipynb'),
                     meta=os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'kernel-metadata.json'),
                     pulled=os.path.join(HERE, 'pulled_v224flips20', 'rsna-sub-v224flips20.ipynb'),
                     hdr=('# [s6f2 base flips20] The cells above are V224-FLIP-AO + S (flips20) (eeshsaxena/rsna-sub-v224flips20 v1) '
                          'unchanged and in order, ending with V224F_DONE. In the S6F text below\n'))}
HDR2 = '# "V224-FLIP" means the output of the cells above (frozen as submission_v224flip.csv); built by s6f2/build_v224s6f2.py.\n'
S6F_NB = os.path.join(SESS, 's6f', 'kernel_v224flips6f', 'rsna_sub_v224flips6f.ipynb')
S6F_SHA = '9c0e2558d51e4adb0cd7474fa6c53aa86e4a5f146f02af1acc8e98ec9323ac0e'
FLIP_NB = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'rsna_sub_v224flip.ipynb')
FLIP_SHA = '73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b'
FLIPS20_BUILD = os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'BUILD_V224FS.json')
S6F_RECEIPT = os.path.join(SESS, 's6f', 'build_receipt_v224flips6f.json')
# the splices, written out (the builder's text must match these exactly)
SETUP_ANCHOR = "        rec['v224_canary'] = dict(globals().get('_canary') or {})\n"
SETUP_ADD = ("        rec['base_fired'] = sorted(set(str(_x) for _g in ('_canary', '_gcr', '_v224f') if isinstance(globals().get(_g), dict)\n"
             "                                       for _x in (globals()[_g].get('fired') or [])))   # [s6f2] every earlier canary of the base\n")
BLEND_OLD = "    blank = ['Fracture'] if 'r224->Fracture' in ((setup or {}).get('v224_canary') or {}).get('fired', []) else []\n"
BLEND_NEW = ("    _fired = list(((setup or {}).get('v224_canary') or {}).get('fired', [])) + list((setup or {}).get('base_fired') or [])  # [s6f2]\n"
             "    blank = [c for c in labels if any(str(x).endswith('->' + c) for x in _fired)]        # [s6f2] V224 CANARY / RM_GUARD / HF fleet canary\n"
             "    blank += [c for c in labels if c not in blank and n >= 2 and base[c].nunique() == 1]  # [s6f2] any constant base column stays blank\n")
RUN_OLD = '    G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = 5.75, 3.3, 8.2, 240.0, 1200.0\n'
ap = argparse.ArgumentParser()
ap.add_argument('--kernel', required=True, choices=sorted(K))
ap.add_argument('--nb', default=None)
ap.add_argument('--meta', default=None)
ap.add_argument('--build', default=None)
A = ap.parse_args()
KK = K[A.kernel]
NB_NEW = A.nb or os.path.join(HERE, 'kernel_' + KK['tag'], 'rsna_sub_%s.ipynb' % KK['tag'])
BUILD = A.build or os.path.join(HERE, 'build_receipt_%s.json' % KK['tag'])
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


base_raw, new_raw = open(KK['nb'], 'rb').read(), open(NB_NEW, 'rb').read()
base, new = json.loads(base_raw), json.loads(new_raw)
N = KK['n']
print('S6F2_CELLDIFF kernel', A.kernel, 'base', os.path.relpath(KK['nb'], ROOT), 'new', os.path.relpath(NB_NEW, ROOT), flush=True)
chk('base file == the pushed base kernel (sha %s...)' % KK['sha'][:16], sha(base_raw) == KK['sha'], sha(base_raw)[:16])
pl = json.load(open(KK['pulled'], encoding='utf-8'))
chk('base cells == kaggle kernels pull of the base slug (type + source of all %d cells)' % N,
    [(c['cell_type'], src(c)) for c in pl['cells']] == [(c['cell_type'], src(c)) for c in base['cells']])
flip = json.loads(open(FLIP_NB, 'rb').read())
if A.kernel == 'flips20':
    bj = json.load(open(FLIPS20_BUILD, encoding='utf-8'))
    chk('base == the flips20 build that passed V224AOS_CHECK 29/29 (BUILD_V224FS.json notebook sha) and its cells 0-14 == V224-FLIP (73a9d562)',
        bj.get('notebook_sha256') == KK['sha'] and sha(open(FLIP_NB, 'rb').read()) == FLIP_SHA and base['cells'][:15] == flip['cells'])
else:
    chk('base minus RM (11) and RM_GUARD (16) == V224-FLIP (73a9d562) cell for cell',
        sha(open(FLIP_NB, 'rb').read()) == FLIP_SHA and [base['cells'][i] for i in list(range(11)) + list(range(12, 16))] == flip['cells'])
BC, NC = base['cells'], new['cells']
chk('cell counts: base %d, new %d' % (N, N + 3), (len(BC), len(NC)) == (N, N + 3), (len(BC), len(NC)))
sm = difflib.SequenceMatcher(a=[(c['cell_type'], src(c)) for c in BC], b=[(c['cell_type'], src(c)) for c in NC], autojunk=False)
ops = sm.get_opcodes()
for tag, i1, i2, j1, j2 in ops:
    print(f'    {tag:7s} base[{i1}:{i2}] -> new[{j1}:{j2}]', flush=True)
chk('alignment: base[0:%d] EQUAL new[0:%d], then one INSERT new[%d:%d] at the END (append only)' % (N, N, N, N + 3),
    ops == [('equal', 0, N, 0, N), ('insert', N, N, N, N + 3)], ops)
chk('cells 0-%d equal as whole cell dicts (metadata, outputs, execution_count included)' % (N - 1), all(NC[i] == BC[i] for i in range(min(N, len(NC)))))
back = json.loads(new_raw)
back['cells'] = back['cells'][:N]
chk('new notebook minus cells %d-%d re-serialises to the base FILE byte for byte' % (N, N + 2), dumps(back).encode('utf-8') == base_raw,
    sha(dumps(back).encode('utf-8'))[:16])
chk('notebook-level metadata / nbformat equal', new.get('metadata') == base.get('metadata') and
    (new.get('nbformat'), new.get('nbformat_minor')) == (base.get('nbformat'), base.get('nbformat_minor')))
chk('the base ENDS with its done marker (%s) and no base cell names an S6F symbol' % KK['last'],
    src(BC[-1]).rstrip().splitlines()[-1].startswith("print('%s'" % KK['last'])
    and not any(x in src(c) for c in BC for x in ('_S6F_', '/tmp/s6f', 's6f_driver', 'submission_v224flip', 'base_fired')))
if A.kernel == 'flips20':
    order_ok = (src(BC[15]).startswith('# [eeshsaxena V224-AO-S SAVE') and "print('HF_ENV_RESTORED', flush=True)" in src(BC[22])
                and src(BC[23]).startswith('# [eeshsaxena V224-AO-S HF BLEND;') and "'MIN_START_MIN': 78.0" in src(BC[21]))
    chk('S stage runs FIRST: SAVE-AO (15) .. ENV-RESTORE (22) .. HF-AO (23, V224F_DONE) all before the S6F cells (24-26); guard 7.0 h (MIN_START_MIN 78)', order_ok)
else:
    chk('RM (11) after her r224 writefile (10), RM_GUARD (16) last, both before the S6F cells (17-19)',
        src(BC[11]).startswith('# [eeshsaxena V224-RM; the ONE change') and src(BC[16]).startswith('# [eeshsaxena V224-RM GUARD;')
        and src(BC[10]).startswith('%%writefile /tmp/raptor224_run.py'))
# ---- the appended cells vs the pushed S6F kernel's cells ----------------------------------------------------------------------
s6_raw = open(S6F_NB, 'rb').read()
s6 = json.loads(s6_raw)
chk('reference: the pushed S6F kernel (s6f/kernel_v224flips6f, sha 9c0e2558, V224S6F_CHECK PASS 31/31)', sha(s6_raw) == S6F_SHA)
ref = [src(c) for c in s6['cells'][15:18]]
app = [src(c) for c in NC[N:N + 3]] if len(NC) == N + 3 else ['', '', '']
hdr = KK['hdr'] + HDR2
undone = []
for k, t in enumerate(app):
    ok = t.startswith(hdr)
    t = t[len(hdr):] if ok else t
    t = t.replace("'tag': '%s'" % KK['tag'], "'tag': 'v224flips6f'")
    # every splice must be PRESENT exactly once (an absent splice would otherwise 'undo' to the reference and pass; neg M5 / M6)
    if k == 0:
        ok = ok and t.count(SETUP_ANCHOR + SETUP_ADD) == 1
        t = t.replace(SETUP_ANCHOR + SETUP_ADD, SETUP_ANCHOR)
    if k == 1:
        g_line = '    G_START_H, BUDGET_H, DEADLINE_H, STOP_MARGIN_S, MIN_RUN_S = %s, 3.3, 8.2, 240.0, 1200.0\n' % KK['G']
        ok = ok and t.count(g_line) == 1
        t = t.replace(g_line, RUN_OLD)
    if k == 2:
        ok = ok and t.count(BLEND_NEW) == 1 and BLEND_OLD not in t
        t = t.replace(BLEND_NEW, BLEND_OLD)
    undone.append((ok, t))
for k, nm in enumerate(('SETUP', 'RUN', 'BLEND')):
    d = list(difflib.unified_diff(ref[k].splitlines(), app[k].splitlines(), lineterm='', n=0))
    print('    %s vs s6f: %d diff lines' % (nm, len([l for l in d if l[:1] in '+-' and not l.startswith(('+++', '---'))])), flush=True)
chk('appended cells == the pushed S6F cells after undoing exactly: the 2-line base header, the tag literal, SETUP + base_fired (2 lines), '
    'RUN G %s (FLIP 5.75), BLEND blank rule (1 line -> 3 lines); nothing else differs' % KK['G'],
    all(ok for ok, _ in undone) and [t for _, t in undone] == ref)
chk('appended cells are code cells without outputs, tag %s in SETUP and BLEND receipts' % KK['tag'],
    all(c['cell_type'] == 'code' and c.get('outputs') == [] for c in NC[N:N + 3])
    and app[0].count("'tag': '%s'" % KK['tag']) == 1 and app[2].count("'tag': '%s'" % KK['tag']) == 1)
s0 = app[0]
try:
    b64 = s0.split("DRIVER_B64 = '", 1)[1].split("'", 1)[0]
    dsha = s0.split("DRIVER_SHA = '", 1)[1].split("'", 1)[0]
    drv = base64.b64decode(b64)
except Exception:
    drv, dsha = b'', None
r6 = json.load(open(S6F_RECEIPT, encoding='utf-8'))
bld = json.load(open(BUILD, encoding='utf-8')) if os.path.isfile(BUILD) else {}
chk('driver embedded == its DRIVER_SHA == the pushed S6F kernel driver (a2acdd1d, nartaa v2 B path) == this build receipt',
    sha(drv) == dsha == r6.get('driver_sha256') == bld.get('driver_sha256') and str(dsha).startswith('a2acdd1d'), str(dsha)[:16])
chk('build receipt: notebook sha == this notebook, base sha, appended [%d, %d, %d], base_kind %s, splices listed' % (N, N + 1, N + 2, A.kernel),
    bld.get('notebook_sha256') == sha(new_raw) and bld.get('base_sha256') == KK['sha'] and bld.get('appended') == [N, N + 1, N + 2]
    and bld.get('base_kind') == A.kernel and len(bld.get('splices') or []) == 3, str(bld.get('notebook_sha256'))[:16])
bm = json.load(open(KK['meta'], encoding='utf-8'))
nm = json.load(open(A.meta or os.path.join(os.path.dirname(NB_NEW), 'kernel-metadata.json'), encoding='utf-8'))
diffk = sorted(k for k in set(bm) | set(nm) if bm.get(k) != nm.get(k))
chk('kernel-metadata == the base\'s except id / title / code_file / dataset_sources (= the base\'s + nartaa/rsna-knee-s6fall-weights + '
    'glorysing/rsna-knee-offline-codecs-r09); id eeshsaxena/rsna-sub-%s; docker pin, T4 x2, private, internet off' % KK['tag'],
    diffk == ['code_file', 'dataset_sources', 'id', 'title'] and nm['id'] == 'eeshsaxena/rsna-sub-' + KK['tag']
    and nm['dataset_sources'] == bm['dataset_sources'] + ['nartaa/rsna-knee-s6fall-weights', 'glorysing/rsna-knee-offline-codecs-r09']
    and nm['is_private'] is True and nm['enable_internet'] is False and nm['enable_gpu'] is True
    and nm['docker_image'].endswith('2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'), {k: (bm.get(k), nm.get(k)) for k in diffk if k != 'dataset_sources'})
print('    info new notebook sha256', sha(new_raw), flush=True)
print(f'S6F2_CELLDIFF {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}', flush=True)
sys.exit(0 if all(R) else 1)
