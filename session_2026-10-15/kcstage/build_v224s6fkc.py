#!/usr/bin/env python
# build_v224s6fkc.py (10-15, Claude subagent, kcstage): V224-FLIP-AO + S6fall .2 (the team best) + the KC1 leg .2
# (PREREG_10-15 Amendment 6 G3; BOX_PLAN_1015 sec 5). APPEND-ONLY on the S6F kernel (or its J twin):
#   base s6f  = session_2026-10-15/s6f/kernel_v224flips6f (pushed v1, sha 9c0e2558, 18 cells, gate V224S6F_CHECK PASS 31/31)
#   base s6fj = session_2026-10-15/jpeg/overlay/r2/kernel_v224flips6fj (pushed v1, sha 1868634e, 19 cells = the DEC cell at index 2
#               + the S6F kernel, gate J_CHECK PASS 16/16)
# Every base cell stays byte-identical and in order, + 3 APPENDED code cells:
#   KC SETUP  freeze submission.csv -> submission_s6f.csv (+ sha); find the private KC dataset by its manifest sha256, check every
#             file (members + frozen code) by sha256 / A6 md5; copy code to /tmp/kc/code; write kc_driver.py + kc_serve.py; KC env
#             (inherited + /tmp/kc/code + the decoders installed above: /tmp/s6f_dec if S6F's DEC passed, /tmp/_dec inherited in J)
#   KC RUN    start guard G_KC, hard deadline min(start + budget, NB_T0 + deadline), ONE driver subprocess, process-group kill
#   KC BLEND  per label .8 pct(base) + .2 pct(KC leg) (unscored -> base rank; constant base columns kept), or the KC canary
#             (Baker's = 0.5 on every row; A6 G3); receipts kc_setup_receipt.json / kc/kc_driver_receipt.json / kc_receipt.json
# The driver = kc_driver_src.py, the serve wrapper = kc_serve.py (both embedded base64 + sha256 in the SETUP cell).
# Proof: deleting the appended cells re-serialises to the base FILE byte for byte (also celldiff_v224s6fkc_proof.py, independent).
#   usage: python build_v224s6fkc.py --kc-dir DIR [--base s6f|s6fj] [--tag TAG] [--g-kc-h H --budget-h H --deadline-h 8.2] [--out DIR]
#   stand-in builds (never submit): a manifest of kind 'standin' needs a tag containing 'standin'; a real KC1 manifest forbids it.
#   control build (never submit): --g-kc-h 0 (start guard always skips -> KC canary; gate --expect-canary --control skip).
import os, re, ast, io, sys, json, copy, base64, hashlib, argparse, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
BASES = {
    's6f': dict(dir=os.path.join(SESS, 's6f', 'kernel_v224flips6f'), nb='rsna_sub_v224flips6f.ipynb',
                sha='9c0e2558d51e4adb0cd7474fa6c53aa86e4a5f146f02af1acc8e98ec9323ac0e', cells=18, slug='eeshsaxena/rsna-sub-v224flips6f',
                meta_style='crlf', name='V224-FLIP-AO + S6fall (eeshsaxena/rsna-sub-v224flips6f v1)', tag='v224s6fkc'),
    's6fj': dict(dir=os.path.join(SESS, 'jpeg', 'overlay', 'r2', 'kernel_v224flips6fj'), nb='rsna_sub_v224flips6fj.ipynb',
                 sha='1868634ea68ef131035a01a347f4b1c4471f0a3d7a58b8a22f9efec3a309b02a', cells=19, slug='eeshsaxena/rsna-sub-v224flips6fj',
                 meta_style='crlf', name='V224-FLIP-AO + S6fall, J form (eeshsaxena/rsna-sub-v224flips6fj v1)', tag='v224s6fkcj')}
CELLS = [os.path.join(HERE, n) for n in ('cell_kc_setup_tpl.py', 'cell_kc_run_tpl.py', 'cell_kc_blend_tpl.py')]
DRIVER_SRC = os.path.join(HERE, 'kc_driver_src.py')
SERVE_SRC = os.path.join(HERE, 'kc_serve.py')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']
OTHER_CANARIES = {'ACL': 'S6F', 'Fracture': 'V224 r224 / RM', 'Synovitis': 'fleet / S'}
CODE_MD5 = {'kc_worker.py': '4af60f6ff7b94b3acab9dceb52162a18', 'kc_common.py': '45255ae0e361ef787b290d7510554a9d',
            'fleet_model.py': '897f83b272fbb144a4d3deab773d9064', 'fleet_model_ttr.py': 'bcdf7d153f339962b97d2736d8baed08'}
CODE_SHA = {'d96_reader_src.py': '30aa72fd972e29cf42c752aa5394a6c88a175f17b98740b1f6b0e0f55553ac40',
            'kc_side_kernel.py': '2979c1915e0cecf2c4792526a5581f93a723fac368eacff18849661dc0bdcb62'}
OUR_NAMES = ['_kc_setup', '_KC_SETUP', '_KC_ENV', '_kc_kill', '_kc_run', '_KC_RUN', '_kc_blend', '_KC_BLEND', '_KC_RECEIPT', '_kcj', '_kct',
             'kc_driver', 'kc_serve', 'submission_s6f', '/tmp/kc', 'kc_receipt', 'KC_MANIFEST']
BASE_FILES = ('raptor_raw.csv', 'raptor_sub.csv', 'raptor224_sub.csv', 'ours_sub.csv', 'blend_receipt.json', 'canary_receipt.json',
              'submission_receipt.json', 'submission_v224flip.csv', 's6f_receipt.json', 's6f_setup_receipt.json', 'dec_receipt.json',
              'dec_envwatch.json', '/s6f/', '/tmp/_dec', 's6f_driver')

ap = argparse.ArgumentParser()
ap.add_argument('--kc-dir', required=True, help='local copy of the KC dataset dir (KC_MANIFEST.json + files), see make_kc_dataset.py')
ap.add_argument('--base', default='s6f', choices=sorted(BASES))
ap.add_argument('--tag', default=None)
ap.add_argument('--g-kc-h', type=float, default=6.9)
ap.add_argument('--budget-h', type=float, default=1.5)
ap.add_argument('--deadline-h', type=float, default=8.2)
ap.add_argument('--stop-margin-s', type=float, default=180.0)
ap.add_argument('--min-run-s', type=float, default=1800.0)
ap.add_argument('--w', type=float, default=0.2)
ap.add_argument('--cov-min', type=float, default=0.98)
ap.add_argument('--canary-col', default="Baker's")
ap.add_argument('--on-fail', default='canary', choices=('canary', 'base'))
ap.add_argument('--rho-floor', default='none', help="'none' (A6: no sanity floor) or 'MEAN,MIN' e.g. 0.5,0.2 (not pre-registered)")
ap.add_argument('--out', default=HERE)
A = ap.parse_args()
BA = BASES[A.base]
TAG = A.tag or BA['tag']
assert re.fullmatch(r'[a-z0-9]+', TAG), TAG
assert (A.base == 's6fj') == TAG.endswith('j'), 'a J-form tag ends in j (j_check reads the tag from the log name), a non-J tag does not'
assert 0 < A.deadline_h <= 8.2 and A.stop_margin_s >= 60 and A.budget_h > 0, 'deadline policy outside A6 G3 (hard wall NB_T0 + 8.2 h)'
assert A.budget_h * 3600.0 - A.stop_margin_s >= A.min_run_s, 'the RUN cell could never start the driver (budget - margin < min run)'
assert A.canary_col in LAB and A.canary_col not in OTHER_CANARIES, ('canary column must be a label no other kernel uses', A.canary_col)
CONTROL = 'skip' if A.g_kc_h <= 0 else None
SUBMIT_FORM = CONTROL is None and A.w == 0.2 and A.cov_min == 0.98 and A.canary_col == "Baker's" and A.on_fail == 'canary' \
    and A.rho_floor == 'none' and A.deadline_h <= 8.2
if A.rho_floor == 'none':
    RHO = (None, None)
else:
    RHO = tuple(float(x) for x in A.rho_floor.split(','))
    assert len(RHO) == 2 and 0 <= RHO[1] <= RHO[0] <= 1


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))     # the V224 family serialisation (compact)


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


def meta_bytes(meta, style):
    t = json.dumps(meta, indent=1)
    return (t.replace('\n', '\r\n') if style == 'crlf' else t).encode('utf-8')


# ---- 1. base ----------------------------------------------------------------------------------------------------------------
RAW = open(os.path.join(BA['dir'], BA['nb']), 'rb').read()
assert sha(RAW) == BA['sha'], ('base notebook changed', sha(RAW))
NB = json.loads(RAW.decode('utf-8'))
assert dumps(NB).encode('utf-8') == RAW, 'serialisation does not reproduce the base file'
C = NB['cells']
NBASE = len(C)
assert NBASE == BA['cells'], NBASE
META_RAW = open(os.path.join(BA['dir'], 'kernel-metadata.json'), 'rb').read()
META = json.loads(META_RAW.decode('utf-8'))
assert meta_bytes(META, BA['meta_style']) == META_RAW, 'kernel-metadata.json style not reproduced'
S = [src(c) for c in C]
idx = lambda pred: [i for i, s in enumerate(S) if pred(s)]
i_t0 = idx(lambda s: 'NB_T0 = time.time()' in s and "'/tmp/raptor_run.py'" in s)
i_canary = idx(lambda s: s.startswith('# [eeshsaxena CANARY') and s.rstrip().endswith("print('V224_DONE', _cj.dumps(_canary), flush=True)"))
i_s6 = [idx(lambda s, k=k: s.startswith('# [eeshsaxena V224-FLIP-S6F %d/3 ' % k)) for k in (1, 2, 3)]
assert len(i_t0) == 1 and len(i_canary) == 1 and all(len(x) == 1 for x in i_s6), (i_t0, i_canary, i_s6)
i_s6 = [x[0] for x in i_s6]
assert i_t0[0] < i_canary[0] < i_s6[0] < i_s6[1] < i_s6[2] == NBASE - 1, 'S6F BLEND must be the last base cell, after the V224 canary'
assert "print('S6F_DONE ' + _s6j.dumps(" in S[i_s6[2]] and "_S6F_RECEIPT = {'tag': " in S[i_s6[2]]
assert "'dec_ok': False" in S[i_s6[0]] and "env['PYTHONPATH'] = after" in S[i_s6[0]]
i_dec = idx(lambda s: 'DEC_OK' in s and '/tmp/_dec' in s and 'envwatch' in s)
assert (len(i_dec) == 1 and i_dec[0] < i_t0[0]) if A.base == 's6fj' else not i_dec, ('J DEC cell', i_dec)
for i, s in enumerate(S):
    for nm in OUR_NAMES:
        assert nm not in s, ('name collision with a base cell', nm, i)

# ---- 2. the KC dataset (local copy) -------------------------------------------------------------------------------------------
MAN_RAW = open(os.path.join(A.kc_dir, 'KC_MANIFEST.json'), 'rb').read()
MANIFEST_SHA = sha(MAN_RAW)
MAN = json.loads(MAN_RAW)
for f, h in MAN['files'].items():
    assert sha_file(os.path.join(A.kc_dir, f)) == h, ('dataset file differs from its manifest', f)
for f, h in CODE_MD5.items():
    assert hashlib.md5(open(os.path.join(A.kc_dir, f), 'rb').read()).hexdigest() == h, f
for f, h in CODE_SHA.items():
    assert MAN['files'][f] == h, f
MEMBERS = [m['file'] for m in MAN['members']]
assert MEMBERS and all(m in MAN['files'] for m in MEMBERS) and len(set(MEMBERS)) == len(MEMBERS)
assert all(m.get('checked') for m in MAN['members']), 'every member must have been opened and checked by make_kc_dataset.py'
KC_SLUG = MAN['slug']
STANDIN = bool(MAN.get('standin'))
assert STANDIN == ('standin' in TAG), 'a stand-in dataset needs a tag with "standin" (never submit); a real KC1 dataset forbids it'
assert MAN.get('kind') == ('standin' if STANDIN else 'kc1')
if not STANDIN:
    assert len(MEMBERS) >= 2, 'A6: KC1 ships only with >= 2 members that passed G1'
assert KC_SLUG not in META['dataset_sources']

# ---- 3. driver + serve wrapper ------------------------------------------------------------------------------------------------
DRIVER_B = open(DRIVER_SRC, 'rb').read()
SERVE_B = open(SERVE_SRC, 'rb').read()
for nm, b in (('kc_driver.py', DRIVER_B), ('kc_serve.py', SERVE_B)):
    t = b.decode('utf-8')
    compile(t, nm, 'exec')
    assert '—' not in t and '–' not in t, ('dash', nm)
    code_only = ' '.join(tok.string for tok in tokenize.generate_tokens(io.StringIO(t).readline)
                         if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT))
    assert not re.search(r'(?<![A-Za-z_])submission(_s6f)?\.csv', code_only), (nm, 'must never name submission.csv')
    for f, h in list(CODE_MD5.items()):
        assert ("'%s': '%s'" % (f, h)) in t, (nm, 'pin', f)
assert "kw.study_inputs = make_study_inputs(table, kc)" in SERVE_B.decode() and "exec(compile(main_src" in SERVE_B.decode()
DRIVER_SHA, SERVE_SHA = sha(DRIVER_B), sha(SERVE_B)

# ---- 4. the three cells -------------------------------------------------------------------------------------------------------
setup_t, run_t, blend_t = [open(p, encoding='utf-8').read() for p in CELLS]
SETUP = (setup_t.replace('__DRIVER_SHA__', DRIVER_SHA).replace('__SERVE_SHA__', SERVE_SHA)
         .replace('__DRIVER_B64__', base64.b64encode(DRIVER_B).decode('ascii')).replace('__SERVE_B64__', base64.b64encode(SERVE_B).decode('ascii'))
         .replace('__MANIFEST_SHA__', MANIFEST_SHA).replace('__KC_SLUG__', KC_SLUG).replace('__TAG__', TAG).replace('__BASE_NAME__', BA['name']))
RUN = (run_t.replace('__G_KC_H__', repr(float(A.g_kc_h))).replace('__BUDGET_H__', repr(float(A.budget_h)))
       .replace('__DEADLINE_H__', repr(float(A.deadline_h))).replace('__STOP_MARGIN_S__', repr(float(A.stop_margin_s)))
       .replace('__MIN_RUN_S__', repr(float(A.min_run_s))).replace('__TAG__', TAG))
rho_doc = ('' if RHO[0] is None else ', and (N scored >= 100) rank corr(base, KC) mean >= %.2f and min >= %.2f (NOT pre-registered in A6)'
           % RHO)
fail_doc = ("THE KC CANARY: the base output with %s = 0.5 for every row (one constant column; ACL = S6F, Fracture = V224 r224, "
            "Synovitis = fleet / S are the other kernels' canaries), so a skipped / failed / partial KC leg can never pass for a read."
            % A.canary_col) if A.on_fail == 'canary' else \
    "the base rows unchanged (FINAL-PICK form, A6: a J-form KC skip falls back to base rows, never to a canary)."
BLEND = (blend_t.replace('__W_KC__', repr(float(A.w))).replace('__COV_MIN__', repr(float(A.cov_min)))
         .replace('__CANARY_COL__', A.canary_col).replace('__ON_FAIL__', A.on_fail).replace('__RHO_MEAN__', repr(RHO[0]))
         .replace('__RHO_MIN__', repr(RHO[1])).replace('__RHO_DOC__', rho_doc).replace('__FAIL_DOC__', fail_doc).replace('__TAG__', TAG))
for name, t in (('SETUP', SETUP), ('RUN', RUN), ('BLEND', BLEND)):
    left = re.findall(r'__[A-Z0-9_]+__', t.replace('__name__', '').replace('__main__', '').replace('__file__', '').replace('__version__', ''))
    assert not left, (name, left[:5])
    compile(t, '<%s>' % name, 'exec')
    assert '—' not in t and '–' not in t, ('dash', name)
    assert 'os.environ[' not in t and 'os.environ.update' not in t and 'sys.path.insert' not in t and 'sys.path.append' not in t, name
    code_t = ' '.join(tok.string for tok in tokenize.generate_tokens(io.StringIO(t).readline) if tok.type != tokenize.COMMENT)
    for f in BASE_FILES:
        assert f not in code_t.replace("'/tmp/s6f_dec'", '').replace("'/tmp/_dec'", '').replace("'/tmp/s6f/'", ''), (name, 'touches a base file', f)
    assert "open(W + '/submission.csv', 'w" not in t
assert "env['PYTHONPATH'] = os.pathsep.join(pp)" in SETUP and 'start_new_session=True' in RUN and 'os.killpg(g, signal.SIGKILL)' in RUN
assert "out[CANARY_COL] = np.float32(0.5)" in BLEND and "os.replace(tmp, W + '/submission.csv')" in BLEND

# ---- 5. assemble: APPEND only -------------------------------------------------------------------------------------------------
nb = copy.deepcopy(NB)
for t in (SETUP, RUN, BLEND):
    nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': t, 'outputs': [], 'execution_count': None})
back = copy.deepcopy(nb)
back['cells'] = back['cells'][:NBASE]
assert dumps(back).encode('utf-8') == RAW, 'removing the appended cells does not give the base notebook'
assert all(nb['cells'][i] == C[i] for i in range(NBASE))
out_dir = os.path.join(A.out, 'kernel_' + TAG)
os.makedirs(out_dir, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % TAG
nb_raw = dumps(nb).encode('utf-8')
open(os.path.join(out_dir, code_file), 'wb').write(nb_raw)
meta = copy.deepcopy(META)
meta.update(id='eeshsaxena/rsna-sub-' + TAG, title='rsna sub ' + TAG, code_file=code_file)
meta['dataset_sources'] = META['dataset_sources'] + [KC_SLUG]
assert {k: v for k, v in meta.items() if k not in ('id', 'title', 'code_file', 'dataset_sources')} == \
    {k: v for k, v in META.items() if k not in ('id', 'title', 'code_file', 'dataset_sources')}
assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is False
open(os.path.join(out_dir, 'kernel-metadata.json'), 'wb').write(meta_bytes(meta, BA['meta_style']))
open(os.path.join(A.out, 'kc_driver_built_%s.py' % TAG), 'wb').write(DRIVER_B)
cell_sha = [sha(t.encode('utf-8')) for t in (SETUP, RUN, BLEND)]
receipt = {'tag': TAG, 'slug': meta['id'], 'base': A.base, 'base_dir': os.path.relpath(BA['dir'], os.path.dirname(SESS)).replace('\\', '/'),
           'base_slug': BA['slug'], 'base_sha256': BA['sha'], 'base_cells': NBASE, 'notebook_sha256': sha(nb_raw), 'cells': len(nb['cells']),
           'appended': [NBASE, NBASE + 1, NBASE + 2], 'cell_sha256': cell_sha, 'driver_sha256': DRIVER_SHA, 'serve_sha256': SERVE_SHA,
           'kc_dataset': {'slug': KC_SLUG, 'manifest_sha256': MANIFEST_SHA, 'kind': MAN.get('kind'), 'standin': STANDIN,
                          'members': MEMBERS, 'member_sha256': {m: MAN['files'][m] for m in MEMBERS}, 'files': MAN['files'],
                          'local_dir': os.path.abspath(A.kc_dir)},
           'code_md5': CODE_MD5, 'code_sha256': CODE_SHA,
           'policy': {'g_kc_h': A.g_kc_h, 'budget_h': A.budget_h, 'deadline_h': A.deadline_h, 'stop_margin_s': A.stop_margin_s,
                      'min_run_s': A.min_run_s, 'w_kc': A.w, 'cov_min': A.cov_min, 'canary_col': A.canary_col, 'on_fail': A.on_fail,
                      'rho_floor': list(RHO), 'workers': 'one per GPU', 'procs_each': 2, **({'control': CONTROL} if CONTROL else {})},
           'submit_form': SUBMIT_FORM and not STANDIN, 'never_submit': STANDIN or CONTROL is not None,
           'dataset_sources': meta['dataset_sources'], 'docker_image': meta['docker_image'], 'machine_shape': meta['machine_shape'],
           'proof': 'cells[0:%d] == base cells; deleting the appended cells re-serialises to the base notebook byte for byte' % NBASE}
json.dump(receipt, open(os.path.join(A.out, 'build_receipt_%s.json' % TAG), 'w', encoding='utf-8'), indent=1)
with open(os.path.join(A.out, 'celldiff_%s.txt' % TAG), 'w', encoding='utf-8', newline='\n') as f:
    f.write('# %s = %s (sha %s...) + 3 APPENDED code cells (%d, %d, %d); cells 0-%d byte-identical and in order\n'
            '# (deleting the appended cells gives the base file byte for byte). Metadata: + dataset %s.\n'
            % (TAG, receipt['base_dir'], BA['sha'][:16], NBASE, NBASE + 1, NBASE + 2, NBASE - 1, KC_SLUG))
    for k, t in zip(receipt['appended'], (SETUP, RUN, BLEND)):
        shown = t.replace(base64.b64encode(DRIVER_B).decode('ascii'), '<base64 of kc_driver.py, sha %s..., full text below>' % DRIVER_SHA[:16])
        shown = shown.replace(base64.b64encode(SERVE_B).decode('ascii'), '<base64 of kc_serve.py, sha %s..., full text below>' % SERVE_SHA[:16])
        f.write('\n#### appended cell %d (sha %s), full text:\n' % (k, sha(t.encode('utf-8'))[:16]))
        f.write(shown)
    f.write('\n#### /tmp/kc/kc_driver.py (%d bytes, sha %s)\n' % (len(DRIVER_B), DRIVER_SHA))
    f.write(DRIVER_B.decode('utf-8'))
    f.write('\n#### /tmp/kc/kc_serve.py (%d bytes, sha %s)\n' % (len(SERVE_B), SERVE_SHA))
    f.write(SERVE_B.decode('utf-8'))
print('BUILT', TAG, os.path.join(out_dir, code_file), 'sha256', sha(nb_raw), 'cells', len(nb['cells']), '(base %d)' % NBASE,
      'driver', DRIVER_SHA[:16], 'serve', SERVE_SHA[:16], 'manifest', MANIFEST_SHA[:16], 'members', MEMBERS,
      'policy G_KC %.2f budget %.2f deadline %.2f' % (A.g_kc_h, A.budget_h, A.deadline_h), 'standin' if STANDIN else 'REAL',
      'control ' + CONTROL if CONTROL else '', 'datasets', meta['dataset_sources'])
