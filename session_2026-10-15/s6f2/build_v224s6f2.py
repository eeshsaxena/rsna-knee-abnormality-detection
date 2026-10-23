#!/usr/bin/env python
# build_v224s6f2.py (10-15, Claude subagent): s6f/build_v224s6f.py generalised with a --base switch. No em dashes.
# The S6F part (driver = nartaa v2 B path + ours, the 3 cell templates, DEC wheels / fixtures, kill control) is read UNCHANGED from
# session_2026-10-15/s6f/ (nothing there is written). Every base cell is kept byte-identical and in order; the 3 S6F cells are
# APPENDED after the base's last cell (proof: deleting them re-serialises to the base FILE byte for byte).
#   --base flip     V224-FLIP-AO + S6fall (the s6f build): base session_2026-10-14/v224/kernel_v224flip (73a9d562, 15 cells).
#                   Rebuilds s6f/kernel_v224flips6f byte for byte (notebook 9c0e2558, metadata, receipt, driver, celldiff) at the
#                   defaults; no splice; nothing new in its receipt.
#   --base flrm     FLRM-AO + S6fall: base v224rm/kernel_v224flrm (= eeshsaxena/rsna-sub-v224flrm v1, f14f6acc, 17 cells: V224-FLIP
#                   + RM after her r224 writefile + RM_GUARD last). Tag v224flrms6f, slug eeshsaxena/rsna-sub-v224flrms6f, G 6.0.
#   --base flips20  FLIP-AO + S6fall + S: base v224f/kernel_v224flips20 (= eeshsaxena/rsna-sub-v224flips20 v1, d2066b8c, 24 cells:
#                   V224-FLIP + the 9 S stage cells ending in HF-AO). S runs FIRST, S6F after (sim_s6f2_runtime.py). Tag
#                   v224flips6fs20, slug eeshsaxena/rsna-sub-v224flips6fs20, G 6.25.
# Splices (non-flip bases only; each anchor exactly once; listed in the receipt):
#   all 3 cells  one header line naming the base ("V224-FLIP" in the S6F text then means the output of the cells above);
#   SETUP        records base_fired = every earlier canary of this base: _canary (V224 CANARY), _gcr (RM_GUARD's receipt dict,
#                flrm), _v224f (HF, flips20; 'fleet->Synovitis' on an S fallback);
#   BLEND        blank = every label some base canary fired on (x endswith '->' + label) + any constant base column; blanked
#                columns stay 0.5 after the S6fall blend. Why: RM_GUARD writes Fracture 0.5 but not into _canary, and the HF S
#                fallback writes Synovitis 0.5; the FLIP rule (Fracture only, from _canary) would let the S6fall blend un-blank both.
#   usage: py -3.12 -I build_v224s6f2.py --base flrm|flips20|flip [--tag T] [--g-start-h G] [--budget-h 3.3] [--deadline-h 8.2]
#                                        [--out DIR] [--kill-control]
import os, re, ast, sys, json, copy, base64, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
ROOT = os.path.dirname(SESS)
S6F = os.path.join(SESS, 's6f')
BASES = {
    'flip': dict(nb=os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'rsna_sub_v224flip.ipynb'),
                 meta=os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'kernel-metadata.json'),
                 sha='73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b', cells=15,
                 pulled=os.path.join(SESS, 'v224rm', 'pulled_v224flip', 'rsna-sub-v224flip.ipynb'),
                 tag='v224flips6f', g=5.75, name='V224-FLIP', slug='eeshsaxena/rsna-sub-v224flip', last='V224_DONE'),
    'flrm': dict(nb=os.path.join(SESS, 'v224rm', 'kernel_v224flrm', 'rsna_sub_v224flrm.ipynb'),
                 meta=os.path.join(SESS, 'v224rm', 'kernel_v224flrm', 'kernel-metadata.json'),
                 sha='f14f6accc7c178536ff8a5d8d70a65da63ca05b2dd6d2a7ce9673100f9178ecf', cells=17,
                 pulled=os.path.join(SESS, 'v224rm', 'pulled_v224flrm', 'rsna-sub-v224flrm.ipynb'),
                 tag='v224flrms6f', g=6.0, name='V224-FLIP-RM', slug='eeshsaxena/rsna-sub-v224flrm', last='V224RM_DONE'),
    'flips20': dict(nb=os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'rsna_sub_v224flips20.ipynb'),
                    meta=os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'kernel-metadata.json'),
                    sha='d2066b8c9c5e3a77a20fa45fc5a0f88db8926566e205a66c49536022df2f17c5', cells=24,
                    pulled=os.path.join(HERE, 'pulled_v224flips20', 'rsna-sub-v224flips20.ipynb'),
                    build=os.path.join(SESS, 'v224f', 'kernel_v224flips20', 'BUILD_V224FS.json'),
                    tag='v224flips6fs20', g=6.25, name='V224-FLIP-AO + S (flips20)', slug='eeshsaxena/rsna-sub-v224flips20',
                    last='V224F_DONE')}
SUBMIT_TAGS = {b['tag']: k for k, b in BASES.items()}
PUSHED_DIRS = [os.path.join(SESS, d) for d in ('s6f', 'v224f', 'v224rm', 'jpeg', 'v224ff')] + [os.path.join(ROOT, 'session_2026-10-14')]
NARTAA_V2 = os.path.join(S6F, 'pulled_v2', 'rsna-knee-s6fall-solo.py')
NARTAA_V2_SHA = '6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe'
NARTAA_V1 = os.path.join(S6F, 'pulled_v1', 'rsna-knee-s6fall-solo.py')
NARTAA_V1_SHA = '3d4b4501c333192462cf83876808ca502a66e7f437bb9f234cbe617eef23cbcd'
REF_RECEIPT = os.path.join(S6F, 'ref_output', 'k1', 'k1_receipt.json')
CODE_MANIFEST_SHA256 = 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22'      # nartaa's commit, inputs.manifest_sha256
MODEL_SHA256 = '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'             # E_prior_s42_fall.pt
S6_META = os.path.join(SESS, 'silver', 'pulled', 's6fall_meta')
DRIVER_SRC = os.path.join(S6F, 's6f_driver_src.py')
CELLS = [os.path.join(S6F, n) for n in ('cell_setup_tpl.py', 'cell_run_tpl.py', 'cell_blend_tpl.py')]
TEMPLATE_SHA = {'cell_setup_tpl.py': None, 'cell_run_tpl.py': None, 'cell_blend_tpl.py': None, 's6f_driver_src.py': None,
                'cell_run_killctl_tpl.py': None}                     # filled below and recorded in non-flip receipts
F20J = os.path.join(SESS, 'jpeg', 'f20j')
DEC_TEMPLATE = os.path.join(F20J, 'dec_cell_template.py')
FIXTURES = os.path.join(F20J, 'selftest_fixtures.json')
FIXTURES_SHA = '725bd7d8f528f022a201bb0e89031388f1a58ba0b7994747c92cf66fecfd6b34'
JC_RESULT = os.path.join(SESS, 'jpeg', 'out_jc_v2', 'jpeg_check_result.json')
PYPI_SHA = {                                                                                    # = build_v224f20j.PYPI_SHA
    'pylibjpeg-2.1.0-py3-none-any.whl': '25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e',
    'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl':
        'd2c8da12c79d1f46596b89df94091be01f932d739613e8af93d1d557ac3bab74',
    'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl':
        '3e8ee9e5bd3b62766ca98fc143cd340633e60bb14450162426f32688908ad9e8'}
S6F_DS, CODEC_DS = 'nartaa/rsna-knee-s6fall-weights', 'glorysing/rsna-knee-offline-codecs-r09'
NEVER = ['remove_submission', '_watchdog', 'arm_guard', 'dump_traceback_later', 'GUARD_S', 'A_B64', 'run_a', 'run_parity',
         'blend_and_write', 'write_submission', 'b_fail_cap', '_GUARD_STOP']
OUR_NAMES = ['_s6f_setup', '_S6F_SETUP', '_S6F_ENV', '_s6f_kill', '_s6f_run', '_S6F_RUN', '_s6f_blend', '_S6F_BLEND', '_S6F_RECEIPT', '_s6f_alive', '_S6F_KILL_STUB',
             '_s6j', '_s6t', 's6f_driver', 'submission_v224flip', '/tmp/s6f']
# ---- splices for the non-flip bases (anchors in the s6f templates; each must occur exactly once) ---------------------------------
SETUP_ANCHOR = "        rec['v224_canary'] = dict(globals().get('_canary') or {})\n"
SETUP_ADD = ("        rec['base_fired'] = sorted(set(str(_x) for _g in ('_canary', '_gcr', '_v224f') if isinstance(globals().get(_g), dict)\n"
             "                                       for _x in (globals()[_g].get('fired') or [])))   # [s6f2] every earlier canary of the base\n")
BLEND_ANCHOR = "    blank = ['Fracture'] if 'r224->Fracture' in ((setup or {}).get('v224_canary') or {}).get('fired', []) else []\n"
BLEND_NEW = ("    _fired = list(((setup or {}).get('v224_canary') or {}).get('fired', [])) + list((setup or {}).get('base_fired') or [])  # [s6f2]\n"
             "    blank = [c for c in labels if any(str(x).endswith('->' + c) for x in _fired)]        # [s6f2] V224 CANARY / RM_GUARD / HF fleet canary\n"
             "    blank += [c for c in labels if c not in blank and n >= 2 and base[c].nunique() == 1]  # [s6f2] any constant base column stays blank\n")

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True, choices=sorted(BASES))
ap.add_argument('--tag', default=None)
ap.add_argument('--g-start-h', type=float, default=None)
ap.add_argument('--budget-h', type=float, default=3.3)
ap.add_argument('--deadline-h', type=float, default=8.2)
ap.add_argument('--stop-margin-s', type=float, default=240.0)
ap.add_argument('--min-run-s', type=float, default=1200.0)
ap.add_argument('--w', type=float, default=0.2)
ap.add_argument('--cov-min', type=float, default=0.98)
ap.add_argument('--canary-col', default='ACL')
ap.add_argument('--out', default=HERE)
ap.add_argument('--kill-control', action='store_true', help='control build: RUN cell starts a stub tree and proves the hard kill')
A = ap.parse_args()
BB = BASES[A.base]
A.tag = A.tag or BB['tag']
A.g_start_h = BB['g'] if A.g_start_h is None else A.g_start_h
SUBMIT = A.tag in SUBMIT_TAGS
assert not SUBMIT or SUBMIT_TAGS[A.tag] == A.base, ('a submit tag belongs to its own base', A.tag, A.base)
assert not SUBMIT or (A.w == 0.2 and A.cov_min == 0.98 and A.canary_col == 'ACL' and A.g_start_h == BB['g'] and A.budget_h == 3.3
                      and A.deadline_h == 8.2 and A.stop_margin_s == 240.0 and A.min_run_s == 1200.0), \
    'a submit tag keeps the pre-registered .2 / 98 %% / ACL and the runtime-reviewed policy (G %.2f for base %s, budget 3.3, deadline 8.2)' % (BB['g'], A.base)
assert 0 < A.deadline_h <= 8.2 and A.stop_margin_s >= 120 and A.budget_h > 0, 'deadline policy outside R4 (iii)'
assert A.budget_h * 3600.0 - A.stop_margin_s >= A.min_run_s, (
    'the RUN cell would never start the driver: budget %.0f s - stop margin %.0f s < min run %.0f s (SKIPPED_NO_TIME before Popen); '
    'for a kill control use --kill-control --budget-h 0.01 --min-run-s -1000' % (A.budget_h * 3600.0, A.stop_margin_s, A.min_run_s))
assert not (A.kill_control and (SUBMIT or A.g_start_h <= 0)), 'kill control: never a submit tag, start guard must pass'
CONTROL = 'kill' if A.kill_control else ('skip' if A.g_start_h <= 0 else None)
assert CONTROL is None or not SUBMIT, 'a control build must not use a submit tag'
OUT = os.path.realpath(A.out)
for d in PUSHED_DIRS:                                                 # never write into a dir that holds a pushed kernel
    d = os.path.realpath(d)
    assert not (OUT == d or OUT.startswith(d + os.sep)), ('refusing to write into a pushed-kernel dir', OUT)
assert not os.path.exists(os.path.join(OUT, 'kernel_' + A.tag, '.pushed')), 'kernel dir marked pushed'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))     # the V224 family serialisation (varshitha's)


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


def statements(text):
    """top-level statement -> its exact source lines (lineno .. end_lineno), by every name it binds."""
    lines = text.splitlines(keepends=True)
    out = {}
    for node in ast.parse(text).body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Tuple):
                    names += [e.id for e in t.elts if isinstance(e, ast.Name)]
        seg = ''.join(lines[node.lineno - 1:node.end_lineno])
        for nm in names:
            out[nm] = (seg, isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    return out


# ---- 1. the base ----------------------------------------------------------------------------------------------------------
BASE_NB, BASE_SHA, NBASE = BB['nb'], BB['sha'], BB['cells']
RAW = open(BASE_NB, 'rb').read()
assert sha(RAW) == BASE_SHA, ('base notebook changed', sha(RAW))
NB = json.loads(RAW.decode('utf-8'))
assert dumps(NB).encode('utf-8') == RAW, 'serialisation does not reproduce the base file'
C = NB['cells']
assert len(C) == NBASE, len(C)
PL = json.load(open(BB['pulled'], encoding='utf-8'))
assert [(c['cell_type'], src(c)) for c in PL['cells']] == [(c['cell_type'], src(c)) for c in C], 'base != the kaggle pull of ' + BB['slug']
META = json.load(open(BB['meta'], encoding='utf-8'))
assert META['id'] == BB['slug'], META['id']
# the V224-FLIP cells (0-14 in flip / flips20; 0-10 + 12-15 in flrm, RM inserted at 11) by content
FI = {'nbt0': 4, 'pip': 6, 'flip': 7, 'r224': 11, 'stage3': 13, 'canary': 14} if A.base != 'flrm' else \
     {'nbt0': 4, 'pip': 6, 'flip': 7, 'r224': 12, 'stage3': 14, 'canary': 15}
assert "NB_T0 = time.time()" in src(C[FI['nbt0']]) and "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600)" in src(C[FI['nbt0']])
assert "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)" in src(C[FI['pip']])
assert "TMP = TMP / 'probe'" in src(C[FI['pip']]) and "TMP = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')" in src(C[FI['pip']])
assert 'FLIP_TTA_FORCED_ON = True' in src(C[FI['flip']]) and "WORKER_ARGS += ['--flip-tta']" in src(C[FI['flip']])
assert "subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=min(_left, 2.5 * 3600))" in src(C[FI['r224']])
assert "sub.to_csv(WORK / 'submission.csv', index=False)" in src(C[FI['stage3']])
assert src(C[FI['canary']]).startswith('# [eeshsaxena CANARY') and "_canary['fired'].append('r224->Fracture')" in src(C[FI['canary']]) \
    and src(C[FI['canary']]).rstrip().endswith("print('V224_DONE', _cj.dumps(_canary), flush=True)")
if A.base in ('flip', 'flips20'):
    FLIP = json.loads(open(BASES['flip']['nb'], 'rb').read().decode('utf-8'))
    assert sha(open(BASES['flip']['nb'], 'rb').read()) == BASES['flip']['sha']
    assert all(C[i] == FLIP['cells'][i] for i in range(15)), 'cells 0-14 are not the V224-FLIP cells'
if A.base == 'flrm':
    FLIP = json.loads(open(BASES['flip']['nb'], 'rb').read().decode('utf-8'))
    assert [C[i] for i in list(range(11)) + list(range(12, 16))] == FLIP['cells'], 'flrm minus RM / RM_GUARD is not V224-FLIP'
    assert src(C[11]).startswith('# [eeshsaxena V224-RM; the ONE change vs its base]') and 'R224_MIRROR_PATCHED' in src(C[11])
    assert src(C[16]).startswith('# [eeshsaxena V224-RM GUARD;') and "_gcr.setdefault('fired', []).append('r224 mirror->Fracture')" in src(C[16]) \
        and src(C[16]).rstrip().endswith("print('V224RM_DONE', _gj.dumps({k: _gcr.get(k) for k in ('fired', 'r224_used', 'flip_tta_forced', 'rm_on', 'w_f_global')}), flush=True)")
if A.base == 'flips20':
    BJ = json.load(open(BB['build'], encoding='utf-8'))
    assert BJ['notebook_sha256'] == BASE_SHA and BJ['base'] == 'flip' and BJ['w_f'] == 0.2 and BJ['s_share'] == 1.0 and BJ['replace'] is True \
        and BJ['k14'] == [] and len(BJ['s']) == 5 and BJ['start_budget_h'] == 7.0 and BJ['wall_h'] == 8.3 and BJ['appended_cells'] == list(range(15, 24)), BJ
    hd = ['# [eeshsaxena V224-AO-S SAVE', '# === HF ENV-SNAP', '# === FN-IDS', 'K12M_CFG = ', 'S_CFG = ', '# [eeshsaxena V224-F CLOCK]',
          'K2_CFG = ', '# === HF ENV-RESTORE', '# [eeshsaxena V224-AO-S HF BLEND;']
    assert all(src(C[15 + k]).startswith(h) for k, h in enumerate(hd)), [src(C[15 + k])[:40] for k in range(9)]
    assert "'MIN_START_MIN': 78.0" in src(C[21]) and "'WALL_H': 8.3" in src(C[21]) and "print('HF_ENV_RESTORED', flush=True)" in src(C[22])
    assert src(C[23]).rstrip().endswith("print('V224F_DONE', json.dumps(_v224f), flush=True)") and "W_F = 0.2\n" in src(C[23]) \
        and "(['fleet->Synovitis'] if _hf['mode'] != 'HF' else [])" in src(C[23]) and "S_SHARE = 1.0\n" in src(C[23])
for i, c in enumerate(C):
    s = src(c)
    for nm in OUR_NAMES:
        assert nm not in s, ('name collision with a base cell', nm, i)
    assert not re.search(r"environ\[['\"]PYTHONPATH['\"]\]\s*=", s), ('a base cell sets PYTHONPATH', i)
    assert not re.search(r"^\s*(_canary|_gcr|_v224f)\s*=", s, re.M) or i in (FI['canary'], 16 if A.base == 'flrm' else -1,
                                                                            23 if A.base == 'flips20' else -1), ('unexpected binder', i)
LAST = src(C[-1])
assert BB['last'] in LAST and LAST.rstrip().splitlines()[-1].startswith("print('%s'" % BB['last']), 'the base must END with its done marker'

# ---- 2. the S6F driver: our source + nartaa's v2 statements, verbatim (== s6f/build_v224s6f.py) -----------------------------
v2_raw = open(NARTAA_V2, 'rb').read()
assert sha(v2_raw) == NARTAA_V2_SHA, 'pulled nartaa v2 changed'
v2 = v2_raw.decode('utf-8')
V2 = statements(v2)
V1 = statements(open(NARTAA_V1, 'rb').read().decode('utf-8')) if os.path.isfile(NARTAA_V1) else {}
assert not V1 or sha(open(NARTAA_V1, 'rb').read()) == NARTAA_V1_SHA
ref = json.load(open(REF_RECEIPT, encoding='utf-8'))
assert ref['inputs']['manifest_sha256'] == CODE_MANIFEST_SHA256 and ref['self']['sha256'] == NARTAA_V2_SHA, 'reference commit is not v2'
assert ref['inputs']['model']['sha256'] == MODEL_SHA256
for fn, key in (('checkpoint-manifest.json', 'sha256'), ('E_prior_s42_fall.DONE.json', 'model_sha256')):
    assert json.load(open(os.path.join(S6_META, fn), encoding='utf-8'))[key] == MODEL_SHA256, fn
assert 'B_MODEL = {"file": "E_prior_s42_fall.pt", "source_file": "E_prior_s42_fall.pt", "sha256": "%s"' % MODEL_SHA256 in V2['B_MODEL'][0]
assert '"compact_infer.py": "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a"' in V2['SOURCE_SHA'][0]
assert V2['MODE'][0].startswith('MODE = "submit"'), V2['MODE'][0][:40]
dsrc = open(DRIVER_SRC, encoding='utf-8').read()
verb, out_lines = [], []
for line in dsrc.splitlines(keepends=True):
    m = re.fullmatch(r'# @@VERBATIM (\w+)\n', line)
    if not m:
        out_lines.append(line)
        continue
    nm = m.group(1)
    assert nm in V2, ('not a top-level statement of nartaa v2', nm)
    seg, is_def = V2[nm]
    out_lines.append(seg + ('\n\n' if is_def else ''))
    verb.append(nm)
DRIVER = ''.join(out_lines).replace('__CODE_MANIFEST_SHA256__', CODE_MANIFEST_SHA256)
assert '@@VERBATIM' not in DRIVER and '__' + 'CODE_MANIFEST' not in DRIVER
compile(DRIVER, 's6f_driver.py', 'exec')
dtree = ast.parse(DRIVER)
dnames = [n.name for n in dtree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
assert len(dnames) == len(set(dnames)), 'duplicate def in the driver'
import io, tokenize
CODE_ONLY = ' '.join(tok.string for tok in tokenize.generate_tokens(io.StringIO(DRIVER).readline)
                     if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT))
for nm in NEVER:
    assert not re.search(r'\b%s\b' % re.escape(nm), CODE_ONLY), ('excluded part of nartaa\'s kernel present', nm)
assert not re.search(r'(?<![A-Za-z_])submission\.csv', CODE_ONLY), 'the driver must never name submission.csv'
for nm in verb:
    assert DRIVER.count(V2[nm][0]) == 1, nm
assert {'prepare_inputs', 'probe_b_stack', 'collect_part', 'b_cmd', 'b_env', 'split_parts', 'LAUNCHER', 'B_PROBE', 'run_group'} <= set(verb)
assert 'run_b' not in verb and 'def run_b(' not in DRIVER, 'nartaa run_b is ported (run_b_s6f), never copied with its hard errors'
v1_diff = sorted(nm for nm in verb if V1 and V1.get(nm, (None,))[0] != V2[nm][0])
DRIVER_B = DRIVER.encode('utf-8')
DRIVER_SHA = sha(DRIVER_B)

# ---- 3. the three cells ---------------------------------------------------------------------------------------------------
fx_raw = open(FIXTURES, 'rb').read()
assert sha(fx_raw) == FIXTURES_SHA
FIX = json.loads(fx_raw)
jc = json.load(open(JC_RESULT, encoding='utf-8'))
TOP = jc['decoders']['cp313']['files']
assert jc['decoders']['cp313']['check'] == '3.0.2 2.1.0 2.4.0 2.6.0' and all(jc['decoders']['cp313_sha_match_pypi'].values()) and len(TOP) == 8
tpl_dec = open(DEC_TEMPLATE, encoding='utf-8').read()
i0 = tpl_dec.index("    TEST = r'''") + len("    TEST = r'''")
DEC_TEST = tpl_dec[i0:tpl_dec.index("'''", i0)]
assert DEC_TEST.startswith('\nimport os, sys, io, json, base64, hashlib\n') and "print('DEC_SELFTEST ' + json.dumps(r), flush=True)" in DEC_TEST
for n, h in PYPI_SHA.items():
    assert ("'%s'" % h) in open(os.path.join(F20J, 'build_v224f20j.py'), encoding='utf-8').read(), n
for fn in TEMPLATE_SHA:
    TEMPLATE_SHA[fn] = sha(open(os.path.join(S6F, fn), 'rb').read())
setup_t, run_t, blend_t = [open(p, encoding='utf-8').read() for p in CELLS]
SETUP = (setup_t.replace('__DRIVER_SHA__', DRIVER_SHA).replace('__DRIVER_B64__', base64.b64encode(DRIVER_B).decode('ascii'))
         .replace('__WHEELS__', repr(PYPI_SHA)).replace('__TOP__', repr(TOP)).replace('__FIX__', repr(FIX))
         .replace('__DEC_TEST__', DEC_TEST).replace('__TAG__', A.tag))
RUN = (run_t.replace('__G_START_H__', repr(float(A.g_start_h))).replace('__BUDGET_H__', repr(float(A.budget_h)))
       .replace('__DEADLINE_H__', repr(float(A.deadline_h))).replace('__STOP_MARGIN_S__', repr(float(A.stop_margin_s)))
       .replace('__MIN_RUN_S__', repr(float(A.min_run_s))))
if A.kill_control:                                  # [rfix S6F-R1] control builds only; the default RUN cell is untouched
    KC = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(os.path.join(S6F, 'cell_run_killctl_tpl.py'),
                                                                             encoding='utf-8').read()).body if isinstance(n, ast.Assign)}
    for anchor, new in (("def _s6f_run(setup, env):\n", KC['KILLCTL_DEFS'] + "def _s6f_run(setup, env):\n"),
                        ("        p = subprocess.Popen([sys.executable, '-u', ROOT + '/s6f_driver.py', ", KC['KILLCTL_LAUNCH']),
                        ("        rec['after_exit_sweep'] = _s6f_kill(p, ROOT)                 # an orphan of a finished driver must not hold a GPU\n",
                         "        rec['after_exit_sweep'] = _s6f_kill(p, ROOT)                 # an orphan of a finished driver must not hold a GPU\n"
                         + KC['KILLCTL_CHECK'])):
        assert RUN.count(anchor) == 1, ('kill-control anchor', anchor[:60])
        RUN = RUN.replace(anchor, new)
    RUN = KC['KILLCTL_HEADER'] + RUN
    assert "'/s6f_driver.py'" not in RUN and RUN.count("start_new_session=True") == 2 and '_s6f_alive(ROOT' in RUN
BLEND = (blend_t.replace('__W_S6F__', repr(float(A.w))).replace('__COV_MIN__', repr(float(A.cov_min)))
         .replace('__CANARY_COL__', A.canary_col).replace('__TAG__', A.tag))
SPLICES = []
if A.base != 'flip':                                # [s6f2] base header + the base-canary rule (see the file header)
    assert SETUP.count(SETUP_ANCHOR) == 1 and BLEND.count(BLEND_ANCHOR) == 1
    SETUP = SETUP.replace(SETUP_ANCHOR, SETUP_ANCHOR + SETUP_ADD)
    BLEND = BLEND.replace(BLEND_ANCHOR, BLEND_NEW)
    HDR = ('# [s6f2 base %s] The cells above are %s (%s v1) unchanged and in order, ending with %s. In the S6F text below\n'
           '# "V224-FLIP" means the output of the cells above (frozen as submission_v224flip.csv); built by s6f2/build_v224s6f2.py.\n'
           % (A.base, BB['name'], BB['slug'], BB['last']))
    SETUP, RUN, BLEND = HDR + SETUP, HDR + RUN, HDR + BLEND
    SPLICES = ['header line x3 (base %s)' % A.base, 'SETUP: + base_fired (after v224_canary)', 'BLEND: blank = base canaries + constant columns']
for name, t in (('SETUP', SETUP), ('RUN', RUN), ('BLEND', BLEND)):
    assert not re.search(r'__[A-Z0-9_]+__', t.replace('__name__', '').replace('__main__', '').replace('__file__', '')
                         .replace('__version__', '')), (name, re.findall(r'__[A-Z0-9_]+__', t)[:5])
    compile(t, '<%s>' % name, 'exec')
    assert chr(0x2014) not in t and chr(0x2013) not in t, ('dash', name)
assert chr(0x2014) not in DRIVER and chr(0x2013) not in DRIVER
assert A.canary_col in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
                        'Synovitis', "Baker's", 'Contusion', 'Fracture'] and A.canary_col not in ('Fracture', 'Synovitis')
for name, t in (('SETUP', SETUP), ('RUN', RUN), ('BLEND', BLEND)):
    assert 'os.environ[' not in t and 'os.environ.update' not in t and 'sys.path.insert' not in t, name
    for f in ('raptor_raw.csv', 'raptor_sub.csv', 'raptor224_sub.csv', 'ours_sub.csv', 'blend_receipt.json', 'canary_receipt.json',
              'submission_receipt.json', 'submission_V224.csv', 'submission_H.csv', 'hf_receipt.json', 'v224f_receipt.json',
              'fleet_receipt.json', 'raptor224_tta_receipt.json'):
        assert f not in t, (name, f)
assert "env['PYTHONPATH'] = after" in SETUP and "start_new_session=True" in RUN and "os.killpg(g, signal.SIGKILL)" in RUN

# ---- 4. assemble: APPEND only ---------------------------------------------------------------------------------------------
nb = copy.deepcopy(NB)
for t in (SETUP, RUN, BLEND):
    nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': t, 'outputs': [], 'execution_count': None})
back = copy.deepcopy(nb)
back['cells'] = back['cells'][:NBASE]
assert dumps(back).encode('utf-8') == RAW, 'removing the appended cells does not give the base notebook'
assert all(nb['cells'][i] == C[i] for i in range(NBASE))
APP = [NBASE, NBASE + 1, NBASE + 2]
out_dir = os.path.join(A.out, 'kernel_' + A.tag)
os.makedirs(out_dir, exist_ok=True)
code_file = 'rsna_sub_%s.ipynb' % A.tag
nb_raw = dumps(nb).encode('utf-8')
open(os.path.join(out_dir, code_file), 'wb').write(nb_raw)
meta = copy.deepcopy(META)
assert S6F_DS not in meta['dataset_sources'] and CODEC_DS not in meta['dataset_sources']
meta.update(id='eeshsaxena/rsna-sub-' + A.tag, title='rsna sub ' + A.tag, code_file=code_file, is_private=True,
            enable_gpu=True, enable_tpu=False, enable_internet=False)
meta['dataset_sources'] = META['dataset_sources'] + [S6F_DS, CODEC_DS]
assert {k: v for k, v in meta.items() if k not in ('id', 'title', 'code_file', 'dataset_sources')} == \
    {k: v for k, v in META.items() if k not in ('id', 'title', 'code_file', 'dataset_sources')}
with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)
open(os.path.join(A.out, 's6f_driver_built_%s.py' % A.tag), 'wb').write(DRIVER_B)
cell_sha = [sha(t.encode('utf-8')) for t in (SETUP, RUN, BLEND)]
receipt = {'tag': A.tag, 'slug': meta['id'], 'base': os.path.relpath(BASE_NB, ROOT).replace('\\', '/'), 'base_sha256': BASE_SHA,
           'notebook_sha256': sha(nb_raw), 'cells': len(nb['cells']), 'appended': APP, 'cell_sha256': cell_sha,
           'driver_sha256': DRIVER_SHA, 'driver_bytes': len(DRIVER_B), 'driver_verbatim_from_nartaa_v2': verb,
           'verbatim_blocks_differing_v1_v2': v1_diff, 'nartaa_v2_sha256': NARTAA_V2_SHA, 'nartaa_v1_sha256': NARTAA_V1_SHA,
           'code_manifest_sha256': CODE_MANIFEST_SHA256, 'model_sha256': MODEL_SHA256,
           'policy': {'g_start_h': A.g_start_h, 'budget_h': A.budget_h, 'deadline_h': A.deadline_h, 'stop_margin_s': A.stop_margin_s,
                      'min_run_s': A.min_run_s, 'w_s6f': A.w, 'cov_min': A.cov_min, 'canary_col': A.canary_col,
                      'ci_deadline': 'stop_at - B_DEADLINE_MARGIN_S (1200 s)', **({'control': CONTROL} if CONTROL else {})},
           'dec': {'wheels': PYPI_SHA, 'top': TOP, 'fixtures_sha256': FIXTURES_SHA, 'target': '/tmp/s6f_dec'},
           'dataset_sources': meta['dataset_sources'], 'docker_image': meta['docker_image'], 'machine_shape': meta['machine_shape'],
           'proof': 'cells[0:15] == base cells; deleting cells 15-17 re-serialises to the base notebook byte for byte'}
if A.base != 'flip':                                 # the flip receipt stays byte-identical to s6f's
    receipt['proof'] = ('cells[0:%d] == base cells; deleting cells %d-%d re-serialises to the base notebook byte for byte'
                        % (NBASE, APP[0], APP[-1]))
    receipt.update(base_kind=A.base, base_name=BB['name'], base_slug=BB['slug'], base_cells=NBASE, base_last_marker=BB['last'],
                   base_pulled=os.path.relpath(BB['pulled'], ROOT).replace('\\', '/'), splices=SPLICES,
                   s6f_templates_sha256=dict(TEMPLATE_SHA), builder='session_2026-10-15/s6f2/build_v224s6f2.py',
                   blend_order=('final = .8 pct(X) + .2 pct(S6fall), X = the output of the cells above (%s)' % BB['name']))
json.dump(receipt, open(os.path.join(A.out, 'build_receipt_%s.json' % A.tag), 'w', encoding='utf-8'), indent=1)
with open(os.path.join(A.out, 'celldiff_%s.txt' % A.tag), 'w', encoding='utf-8', newline='\n') as f:
    if A.base == 'flip':
        f.write('# %s = %s (sha %s..., the submitted V224-FLIP) + 3 APPENDED code cells (15, 16, 17); cells 0-14 byte-identical and in order\n'
                '# (deleting 15-17 gives the base file byte for byte). Metadata: + datasets %s, %s.\n' % (A.tag, receipt['base'], BASE_SHA[:16],
                                                                                                      S6F_DS, CODEC_DS))
    else:
        f.write('# %s = %s (sha %s..., %s = %s v1) + 3 APPENDED code cells (%d, %d, %d); cells 0-%d byte-identical and in order\n'
                '# (deleting %d-%d gives the base file byte for byte). Metadata: + datasets %s, %s. Splices vs the s6f cells: %s.\n'
                % (A.tag, receipt['base'], BASE_SHA[:16], BB['name'], BB['slug'], APP[0], APP[1], APP[2], NBASE - 1, APP[0], APP[2],
                   S6F_DS, CODEC_DS, '; '.join(SPLICES)))
    for k, t in zip(APP, (SETUP, RUN, BLEND)):
        shown = t.replace(base64.b64encode(DRIVER_B).decode('ascii'), '<base64 of s6f_driver.py, sha %s..., full text below>' % DRIVER_SHA[:16])
        shown = shown.replace(repr(FIX), '{...3 self-test DICOMs, selftest_fixtures.json sha %s...}' % FIXTURES_SHA[:16])
        f.write('\n#### appended cell %d (sha %s), full text:\n' % (k, sha(t.encode('utf-8'))[:16]))
        f.write(shown)
    f.write('\n#### /tmp/s6f/s6f_driver.py (written by cell %d; %d bytes, sha %s); VERBATIM nartaa v2 statements: %s\n'
            % (APP[0], len(DRIVER_B), DRIVER_SHA, ', '.join(verb)))
    f.write(DRIVER)
print('BUILT', A.tag, 'base', A.base, os.path.join(out_dir, code_file), 'sha256', sha(nb_raw), 'cells', len(nb['cells']), 'driver', DRIVER_SHA[:16],
      'verbatim', len(verb), 'v1/v2-differing verbatim blocks', v1_diff, 'policy G %.2f budget %.2f deadline %.2f' % (A.g_start_h, A.budget_h,
                                                                                                                A.deadline_h),
      'splices', SPLICES, 'datasets', meta['dataset_sources'])
