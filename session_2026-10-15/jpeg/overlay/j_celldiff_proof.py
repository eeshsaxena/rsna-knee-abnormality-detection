#!/usr/bin/env python
# j_celldiff_proof.py (2026-10-15, Claude subagent): independent proof that a J kernel built by make_j.py differs from its base ONLY
# by the inserted DEC cell + the kernel metadata (id / title / code_file / the codec dataset). It does not import make_j.py: the DEC
# text is re-rendered here from dec_cell_j_template.py + the self-test fixtures + the PyPI sha table of the rsna-jpeg-check driver.
#   usage: python j_celldiff_proof.py BASE_KERNEL_DIR J_KERNEL_DIR [--slug OWNER/rsna-sub-TAG] [--receipt j_build_receipt_TAG.json]
# Prints J_CELLDIFF PASS n/n (exit 0) or J_CELLDIFF FAIL k/n (exit 1).
import os, re, sys, json, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
JPEG = os.path.normpath(os.path.join(HERE, '..'))
CODEC = 'glorysing/rsna-knee-offline-codecs-r09'
HEADER = '# [eeshsaxena J DECODERS (jpeg/overlay/make_j.py)'
ap = argparse.ArgumentParser()
ap.add_argument('base')
ap.add_argument('j')
ap.add_argument('--slug', default=None)
ap.add_argument('--receipt', default=None)
A = ap.parse_args()
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def compact(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


bm_raw = open(os.path.join(A.base, 'kernel-metadata.json'), 'rb').read()
jm_raw = open(os.path.join(A.j, 'kernel-metadata.json'), 'rb').read()
bm, jm = json.loads(bm_raw), json.loads(jm_raw)
b_raw = open(os.path.join(A.base, bm['code_file']), 'rb').read()
j_raw = open(os.path.join(A.j, jm['code_file']), 'rb').read()
B, J = json.loads(b_raw), json.loads(j_raw)
slug = A.slug or jm.get('id')
m = re.fullmatch(r'[a-z0-9-]+/rsna-sub-([a-z0-9]+)', slug or '')
tag = m.group(1) if m else None
rc_path = A.receipt or os.path.join(os.path.dirname(os.path.abspath(A.j)), 'j_build_receipt_%s.json' % tag)
rc = json.load(open(rc_path, encoding='utf-8')) if os.path.isfile(rc_path) else {}
print('J_CELLDIFF base %s (%s) vs J %s (%s) tag %s' % (A.base, sha(b_raw)[:16], A.j, sha(j_raw)[:16], tag), flush=True)

# ---- files -----------------------------------------------------------------------------------------------------------------------
chk('J kernel dir holds exactly kernel-metadata.json + the notebook', sorted(os.listdir(A.j)) == sorted(['kernel-metadata.json', jm['code_file']]),
    sorted(os.listdir(A.j)))
chk('both notebooks are compact-serialised JSON (the V224 family form)', compact(B) == b_raw and compact(J) == j_raw)
chk('notebook-level fields equal (metadata, nbformat, nbformat_minor, key set)',
    set(B) == set(J) and all(B[k] == J[k] for k in B if k != 'cells'), sorted(set(B) ^ set(J)))

# ---- cells -----------------------------------------------------------------------------------------------------------------------
bc, jc = B['cells'], J['cells']
p = next((i for i in range(len(bc)) if bc[i] != jc[i]), len(bc))
chk('exactly one extra cell', len(jc) == len(bc) + 1, (len(bc), len(jc)))
chk('cells before the insert identical (%d) and cells after it identical and in order (%d)' % (p, len(bc) - p),
    jc[:p] == bc[:p] and jc[p + 1:] == bc[p:])
n_diff = sum(1 for i in range(len(bc)) if bc[i] != (jc[i] if i < p else jc[i + 1]))
chk('cell-by-cell: 0 base cells changed', n_diff == 0, n_diff)
back = dict(J)
back['cells'] = jc[:p] + jc[p + 1:]
chk('deleting the inserted cell re-serialises to the base notebook byte for byte', compact(back) == b_raw)
ins = jc[p] if p < len(jc) else {}
chk('inserted cell: code, metadata {}, no outputs, execution_count None, nothing else',
    ins == {'cell_type': 'code', 'metadata': {}, 'source': ins.get('source'), 'outputs': [], 'execution_count': None})
s = ins.get('source') if isinstance(ins.get('source'), str) else ''.join(ins.get('source') or [])
first_code = next(i for i, c in enumerate(bc) if c['cell_type'] == 'code')
clock_first = bool(re.search(r'(?m)^\s*\w*T0\w*\s*=\s*time\.time\(\)', ''.join(bc[first_code]['source'])) and len(
    ''.join(bc[first_code]['source'])) < 1500 and 'subprocess' not in ''.join(bc[first_code]['source']))
want_p = first_code + 1 if clock_first else first_code
while not clock_first and want_p > 0 and bc[want_p - 1]['cell_type'] == 'markdown' and ''.join(bc[want_p - 1]['source']).lstrip().startswith('## Stage'):
    want_p -= 1
chk('inserted as the FIRST code cell (before any stage cell; base cells before it are all markdown)',
    p == want_p and all(c['cell_type'] == 'markdown' for c in bc[:p]) and jc[p]['cell_type'] == 'code' and
    not any(c['cell_type'] == 'code' for c in jc[:p]), {'index': p, 'expected': want_p, 'clock cell first': clock_first})

# ---- the inserted text == the template rendered here --------------------------------------------------------------------------
tpl = open(os.path.join(HERE, 'dec_cell_j_template.py'), encoding='utf-8').read()
fx_raw = open(os.path.join(JPEG, 'f20j', 'selftest_fixtures.json'), 'rb').read()
FIX = json.loads(fx_raw)
drv = open(os.path.join(JPEG, 'kernel_src', 'driver_template.py'), encoding='utf-8').read()
pypi = {}
for n, h in re.findall(r"'(pylib[\w.\-]+\.whl)':\s*'([0-9a-f]{64})'", drv):
    pypi[n] = h
pypi = {n: h for n, h in pypi.items() if '313' in n or n.startswith('pylibjpeg-2.1.0')}
jcres = json.load(open(os.path.join(JPEG, 'out_jc_v2', 'jpeg_check_result.json'), encoding='utf-8'))
TOP = jcres['decoders']['cp313']['files']
order = ['pylibjpeg-2.1.0-py3-none-any.whl', 'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl',
         'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl']
WHEELS = {n: pypi.get(n) for n in order}
want = tpl.replace('__WHEELS__', repr(WHEELS)).replace('__TOP__', repr(TOP)).replace('__FIX__', repr(FIX)).replace('__TAG__', tag or '?')
chk('fixtures sha 725bd7d8..., 3 PyPI shas found in the jpeg-check driver, 8 target entries',
    sha(fx_raw).startswith('725bd7d8f528f022') and all(WHEELS.values()) and len(TOP) == 8, {k[:24]: (v or '')[:8] for k, v in WHEELS.items()})
chk('inserted cell text == dec_cell_j_template.py rendered here (tag %s)' % tag, s == want, 'sha %s' % sha(s.encode('utf-8'))[:16])
ok_compile = True
try:
    compile(s, '<dec>', 'exec')
except SyntaxError:
    ok_compile = False
chk('inserted cell compiles, starts with the J header, no em / en dash, single PYTHONPATH write + single sys.path insert',
    ok_compile and s.startswith(HEADER) and '\u2014' not in s and '\u2013' not in s and s.count("os.environ['PYTHONPATH'] = after") == 1
    and s.count('sys.path.insert(0, TARGET)') == 1)
chk('no base cell defines / uses the DEC names (_dec_main, _DEC_RECEIPT, _dec_e)',
    not any(re.search(r'(?<![\w])(_dec_main|_DEC_RECEIPT|_dec_e)(?![\w])', ''.join(c['source'])) for c in bc))

# ---- metadata ------------------------------------------------------------------------------------------------------------------
chk('metadata keys equal', set(bm) == set(jm), sorted(set(bm) ^ set(jm)))
same = {k: (bm.get(k), jm.get(k)) for k in bm if k not in ('id', 'title', 'code_file', 'dataset_sources') and bm.get(k) != jm.get(k)}
chk('metadata equal except id / title / code_file / dataset_sources', not same, same)
had = CODEC in bm.get('dataset_sources', [])
chk('dataset_sources == base + [%s] (unchanged if the base had it)' % CODEC,
    jm.get('dataset_sources') == (bm['dataset_sources'] if had else bm['dataset_sources'] + [CODEC]), {'base had it': had})
chk('id / title / code_file == the J slug', tag is not None and jm.get('id') == slug and jm.get('title') == 'rsna sub %s' % tag and
    jm.get('code_file') == 'rsna_sub_%s.ipynb' % tag and slug != bm.get('id'), (jm.get('id'), jm.get('title'), jm.get('code_file')))
chk('private, internet off, same docker pin and accelerator as the base',
    jm.get('is_private') is True and jm.get('enable_internet') is False and jm.get('docker_image') == bm.get('docker_image') and
    jm.get('machine_shape') == bm.get('machine_shape') and jm.get('enable_gpu') == bm.get('enable_gpu') and 'id_no' not in jm)

# ---- receipt ---------------------------------------------------------------------------------------------------------------------
chk('build receipt agrees (base sha, J notebook sha, metadata sha, DEC index, DEC sha)',
    rc.get('base_notebook_sha256') == sha(b_raw) and rc.get('notebook_sha256') == sha(j_raw) and rc.get('meta_sha256') == sha(jm_raw)
    and rc.get('dec_cell_index') == p and rc.get('dec_cell_sha256') == sha(s.encode('utf-8')), rc_path)
ok = all(R)
print('J_CELLDIFF %s %d/%d tag %s notebook %s dec cell %d' % ('PASS' if ok else 'FAIL', sum(R), len(R), tag, sha(j_raw)[:16], p), flush=True)
sys.exit(0 if ok else 1)
