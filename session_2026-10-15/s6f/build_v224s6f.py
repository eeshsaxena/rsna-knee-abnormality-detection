#!/usr/bin/env python
# build_v224s6f.py (10-15, Claude subagent): V224-FLIP-AO + S6fall (PREREG_10-15 Amendment 3 R4; SILVER_PLAN sec 2 B1).
# = the SUBMITTED V224-FLIP notebook (eeshsaxena/rsna-sub-v224flip v1, ref 57174021; sha 73a9d562, R7 flip FORCED = always on)
# with every one of its 15 cells byte-identical and in order, + 3 APPENDED code cells:
#   15 S6F SETUP  freeze submission.csv -> submission_v224flip.csv; write the S6F driver; decoders into /tmp/s6f_dec; S6F env
#   16 S6F RUN    start guard G, hard deadline min(start + budget, NB_T0 + deadline), ONE driver subprocess, process-group kill
#   17 S6F BLEND  per label .8 pct(V224-FLIP) + .2 pct(S6fall) (unscored -> V224-FLIP rank), or the S6F canary (ACL = 0.5)
# The driver = s6f_driver_src.py with every "# @@VERBATIM NAME" line replaced by that top-level statement of nartaa's
# rsna-knee-s6fall-solo VERSION 2 source (pulled_v2/, sha 6e87350d, = the public commit whose output is our parity reference),
# byte for byte (ast statement lines). Never included: remove_submission, the hang guard, component A, the blend, parity mode.
# Base stage order unchanged (r384 -> R7 -> r224 -> stage 3 -> V224 canary); proof: deleting the appended cells re-serialises
# to the base FILE byte for byte. Metadata = V224-FLIP's + nartaa/rsna-knee-s6fall-weights + glorysing/rsna-knee-offline-codecs-r09.
#   usage: python build_v224s6f.py [--tag v224flips6f] [--g-start-h 5.75 --budget-h 3.3 --deadline-h 8.2] [--out DIR]
#   control builds (never submit; receipt policy.control tells the gate which path must have run):
#     --tag v224flips6fskip --g-start-h 0                                    start guard always skips -> canary   (control skip)
#     --tag v224flips6fkill --kill-control --budget-h 0.01 --min-run-s -1000 a stub process tree under /tmp/s6f (3 sessions, 2
#        grandchildren) is SIGKILLed by the RUN cell's /proc process-group kill at start + 36 s, then S6F_KILL_CHECK proves no
#        survivor -> canary (control kill). [rfix S6F-R1: the old "cut" flags --budget-h 0.03 --min-run-s 0 never started the
#        driver (stop_at = hard - 240 s < start -> SKIPPED_NO_TIME); the builder now refuses any build whose RUN cell cannot launch.]
import os, re, ast, sys, json, copy, base64, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.dirname(HERE)
ROOT = os.path.dirname(SESS)
BASE_NB = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'rsna_sub_v224flip.ipynb')
BASE_META = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'kernel-metadata.json')
BASE_SHA = '73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b'
PULLED_FLIP = os.path.join(SESS, 'v224rm', 'pulled_v224flip', 'rsna-sub-v224flip.ipynb')     # kaggle pull of the submitted kernel
NARTAA_V2 = os.path.join(HERE, 'pulled_v2', 'rsna-knee-s6fall-solo.py')
NARTAA_V2_SHA = '6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe'
NARTAA_V1 = os.path.join(HERE, 'pulled_v1', 'rsna-knee-s6fall-solo.py')
NARTAA_V1_SHA = '3d4b4501c333192462cf83876808ca502a66e7f437bb9f234cbe617eef23cbcd'
REF_RECEIPT = os.path.join(HERE, 'ref_output', 'k1', 'k1_receipt.json')
CODE_MANIFEST_SHA256 = 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22'      # nartaa's commit, inputs.manifest_sha256
MODEL_SHA256 = '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'             # E_prior_s42_fall.pt
S6_META = os.path.join(SESS, 'silver', 'pulled', 's6fall_meta')
DRIVER_SRC = os.path.join(HERE, 's6f_driver_src.py')
CELLS = [os.path.join(HERE, n) for n in ('cell_setup_tpl.py', 'cell_run_tpl.py', 'cell_blend_tpl.py')]
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

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='v224flips6f')
ap.add_argument('--g-start-h', type=float, default=5.75)
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
assert A.w == 0.2 and A.cov_min == 0.98 or A.tag != 'v224flips6f', 'the submit tag must keep the pre-registered .2 / 98 %'
assert 0 < A.deadline_h <= 8.2 and A.stop_margin_s >= 120 and A.budget_h > 0, 'deadline policy outside R4 (iii)'
# [rfix S6F-R1] the RUN cell skips before Popen when (hard - stop margin) - start < MIN_RUN_S; with hard = start + budget that is
# budget - margin < min_run: such a build can never start the driver (a "cut" control that is really a skip control)
assert A.budget_h * 3600.0 - A.stop_margin_s >= A.min_run_s, (
    'the RUN cell would never start the driver: budget %.0f s - stop margin %.0f s < min run %.0f s (SKIPPED_NO_TIME before Popen); '
    'for a kill control use --kill-control --budget-h 0.01 --min-run-s -1000' % (A.budget_h * 3600.0, A.stop_margin_s, A.min_run_s))
assert not (A.kill_control and (A.tag == 'v224flips6f' or A.g_start_h <= 0)), 'kill control: never the submit tag, start guard must pass'
CONTROL = 'kill' if A.kill_control else ('skip' if A.g_start_h <= 0 else None)
assert CONTROL is None or A.tag != 'v224flips6f', 'a control build must not use the submit tag'


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


# ---- 1. base = the submitted V224-FLIP -------------------------------------------------------------------------------------
RAW = open(BASE_NB, 'rb').read()
assert sha(RAW) == BASE_SHA, ('base notebook changed', sha(RAW))
NB = json.loads(RAW.decode('utf-8'))
assert dumps(NB).encode('utf-8') == RAW, 'serialisation does not reproduce the base file'
C = NB['cells']
assert len(C) == 15, len(C)
PL = json.load(open(PULLED_FLIP, encoding='utf-8'))
assert [(c['cell_type'], src(c)) for c in PL['cells']] == [(c['cell_type'], src(c)) for c in C], 'base != kaggle pull of rsna-sub-v224flip'
META = json.load(open(BASE_META, encoding='utf-8'))
# the base cells this build relies on (by content): NB_T0 (stage 1), the R7 global pip install (why the S6F env is post-R7),
# the forced flip, stage 3's in-memory `sub`, the V224 canary's `_canary` + V224_DONE as the LAST base cell
assert "NB_T0 = time.time()" in src(C[4]) and "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600)" in src(C[4])
assert "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)" in src(C[6])
assert "TMP = TMP / 'probe'" in src(C[6]) and "TMP = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')" in src(C[6])
assert 'FLIP_TTA_FORCED_ON = True' in src(C[7]) and "WORKER_ARGS += ['--flip-tta']" in src(C[7])
assert "subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=min(_left, 2.5 * 3600))" in src(C[11])
assert "sub.to_csv(WORK / 'submission.csv', index=False)" in src(C[13])
assert src(C[14]).startswith('# [eeshsaxena CANARY') and "_canary['fired'].append('r224->Fracture')" in src(C[14]) \
    and src(C[14]).rstrip().endswith("print('V224_DONE', _cj.dumps(_canary), flush=True)")
for i, c in enumerate(C):
    s = src(c)
    for nm in OUR_NAMES:
        assert nm not in s, ('name collision with a base cell', nm, i)
    assert not re.search(r"environ\[['\"]PYTHONPATH['\"]\]\s*=", s), ('a base cell sets PYTHONPATH', i)

# ---- 2. the S6F driver: our source + nartaa's v2 statements, verbatim -------------------------------------------------------
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
for nm in NEVER:                                   # code and strings (comments may name what is excluded)
    assert not re.search(r'\b%s\b' % re.escape(nm), CODE_ONLY), ('excluded part of nartaa\'s kernel present', nm)
assert not re.search(r'(?<![A-Za-z_])submission\.csv', CODE_ONLY), 'the driver must never name submission.csv'
# every verbatim block is byte-equal to the pulled source and appears exactly once in the driver
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
setup_t, run_t, blend_t = [open(p, encoding='utf-8').read() for p in CELLS]
SETUP = (setup_t.replace('__DRIVER_SHA__', DRIVER_SHA).replace('__DRIVER_B64__', base64.b64encode(DRIVER_B).decode('ascii'))
         .replace('__WHEELS__', repr(PYPI_SHA)).replace('__TOP__', repr(TOP)).replace('__FIX__', repr(FIX))
         .replace('__DEC_TEST__', DEC_TEST).replace('__TAG__', A.tag))
RUN = (run_t.replace('__G_START_H__', repr(float(A.g_start_h))).replace('__BUDGET_H__', repr(float(A.budget_h)))
       .replace('__DEADLINE_H__', repr(float(A.deadline_h))).replace('__STOP_MARGIN_S__', repr(float(A.stop_margin_s)))
       .replace('__MIN_RUN_S__', repr(float(A.min_run_s))))
if A.kill_control:                                  # [rfix S6F-R1] control builds only; the default RUN cell is untouched
    KC = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(os.path.join(HERE, 'cell_run_killctl_tpl.py'),
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
for name, t in (('SETUP', SETUP), ('RUN', RUN), ('BLEND', BLEND)):
    assert not re.search(r'__[A-Z0-9_]+__', t.replace('__name__', '').replace('__main__', '').replace('__file__', '')
                         .replace('__version__', '')), (name, re.findall(r'__[A-Z0-9_]+__', t)[:5])
    compile(t, '<%s>' % name, 'exec')
    assert '—' not in t and '–' not in t, ('dash', name)
assert '—' not in DRIVER and '–' not in DRIVER
assert A.canary_col in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
                        'Synovitis', "Baker's", 'Contusion', 'Fracture'] and A.canary_col not in ('Fracture', 'Synovitis')
# the cells change nothing the base uses: no assignment to os.environ / sys.path at module level, no write to a V224 file
for name, t in (('SETUP', SETUP), ('RUN', RUN), ('BLEND', BLEND)):
    assert 'os.environ[' not in t and 'os.environ.update' not in t and 'sys.path.insert' not in t, name
    for f in ('raptor_raw.csv', 'raptor_sub.csv', 'raptor224_sub.csv', 'ours_sub.csv', 'blend_receipt.json', 'canary_receipt.json',
              'submission_receipt.json'):
        assert f not in t, (name, f)
assert "env['PYTHONPATH'] = after" in SETUP and "start_new_session=True" in RUN and "os.killpg(g, signal.SIGKILL)" in RUN

# ---- 4. assemble: APPEND only ---------------------------------------------------------------------------------------------
nb = copy.deepcopy(NB)
for t in (SETUP, RUN, BLEND):
    nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': t, 'outputs': [], 'execution_count': None})
back = copy.deepcopy(nb)
back['cells'] = back['cells'][:15]
assert dumps(back).encode('utf-8') == RAW, 'removing the appended cells does not give the base notebook'
assert all(nb['cells'][i] == C[i] for i in range(15))
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
           'notebook_sha256': sha(nb_raw), 'cells': len(nb['cells']), 'appended': [15, 16, 17], 'cell_sha256': cell_sha,
           'driver_sha256': DRIVER_SHA, 'driver_bytes': len(DRIVER_B), 'driver_verbatim_from_nartaa_v2': verb,
           'verbatim_blocks_differing_v1_v2': v1_diff, 'nartaa_v2_sha256': NARTAA_V2_SHA, 'nartaa_v1_sha256': NARTAA_V1_SHA,
           'code_manifest_sha256': CODE_MANIFEST_SHA256, 'model_sha256': MODEL_SHA256,
           'policy': {'g_start_h': A.g_start_h, 'budget_h': A.budget_h, 'deadline_h': A.deadline_h, 'stop_margin_s': A.stop_margin_s,
                      'min_run_s': A.min_run_s, 'w_s6f': A.w, 'cov_min': A.cov_min, 'canary_col': A.canary_col,
                      'ci_deadline': 'stop_at - B_DEADLINE_MARGIN_S (1200 s)', **({'control': CONTROL} if CONTROL else {})},
           'dec': {'wheels': PYPI_SHA, 'top': TOP, 'fixtures_sha256': FIXTURES_SHA, 'target': '/tmp/s6f_dec'},
           'dataset_sources': meta['dataset_sources'], 'docker_image': meta['docker_image'], 'machine_shape': meta['machine_shape'],
           'proof': 'cells[0:15] == base cells; deleting cells 15-17 re-serialises to the base notebook byte for byte'}
json.dump(receipt, open(os.path.join(A.out, 'build_receipt_%s.json' % A.tag), 'w', encoding='utf-8'), indent=1)
with open(os.path.join(A.out, 'celldiff_%s.txt' % A.tag), 'w', encoding='utf-8', newline='\n') as f:
    f.write('# %s = %s (sha %s..., the submitted V224-FLIP) + 3 APPENDED code cells (15, 16, 17); cells 0-14 byte-identical and in order\n'
            '# (deleting 15-17 gives the base file byte for byte). Metadata: + datasets %s, %s.\n' % (A.tag, receipt['base'], BASE_SHA[:16],
                                                                                                  S6F_DS, CODEC_DS))
    for k, t in zip((15, 16, 17), (SETUP, RUN, BLEND)):
        shown = t.replace(base64.b64encode(DRIVER_B).decode('ascii'), '<base64 of s6f_driver.py, sha %s..., full text below>' % DRIVER_SHA[:16])
        shown = shown.replace(repr(FIX), '{...3 self-test DICOMs, selftest_fixtures.json sha %s...}' % FIXTURES_SHA[:16])
        f.write('\n#### appended cell %d (sha %s), full text:\n' % (k, sha(t.encode('utf-8'))[:16]))
        f.write(shown)
    f.write('\n#### /tmp/s6f/s6f_driver.py (written by cell 15; %d bytes, sha %s); VERBATIM nartaa v2 statements: %s\n'
            % (len(DRIVER_B), DRIVER_SHA, ', '.join(verb)))
    f.write(DRIVER)
print('BUILT', A.tag, os.path.join(out_dir, code_file), 'sha256', sha(nb_raw), 'cells', len(nb['cells']), 'driver', DRIVER_SHA[:16],
      'verbatim', len(verb), 'v1/v2-differing verbatim blocks', v1_diff, 'policy G %.2f budget %.2f deadline %.2f' % (A.g_start_h, A.budget_h,
                                                                                                                A.deadline_h),
      'datasets', meta['dataset_sources'])
