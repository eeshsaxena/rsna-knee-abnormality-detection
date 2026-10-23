#!/usr/bin/env python
# build_v224.py (2026-10-14, Claude subagent): private copies of varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956
# ("V224": R7 .5 + Raptor_mix .5, Raptor_mix = .65 pct(r384) + .35 pct(r224 sub), NOT re-ranked; her stage-2b time guard; her cell
# order, so r224 runs AFTER R7's global pip install of timm 1.0.25 / pydicom 3.0.1).
#
# Source: src/varshitha_v224.ipynb = `kaggle kernels pull varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956 -m`
# (10-14 11:58 IST; byte-identical to the 10-13 15:42 pull; her last run 2026-10-13 08:39 UTC). Every one of her 13 cells is
# kept VERBATIM and in her order; the builder only INSERTS cells and asserts that deleting the inserted cells gives back her
# notebook file byte for byte (same JSON serialisation as hers: compact separators, ensure_ascii False).
#
# Targets (each writes kernel_<name>/<code>.ipynb + kernel-metadata.json; metadata = hers: same 3 datasets, same docker pin,
# same machine shape, internet off; plus private, our id/title/code_file; her id_no and keywords are dropped):
#   v224      eeshsaxena/rsna-sub-v224       V224-exact: her 13 cells + CANARY cell after stage 3 (USE_F False: no fleet cell,
#                                            FLIP False).
#   v224flip  eeshsaxena/rsna-sub-v224flip   V224-FLIP: V224-exact + FLIP cell between her R7 assets cell (6) and R7 run cell (7):
#                                            asserts the pinned manifest has flip_tta off, sets manifest['flip_tta'] = True in
#                                            memory and appends --flip-tta to WORKER_ARGS (compact_infer v4c honours the flag;
#                                            her cell 6 adds the same flag when the manifest has flip_tta), prints FLIP_TTA_FORCED.
#   v224ctl   eeshsaxena/rsna-sub-v224ctl    CONTROL (never submit): V224-exact + a cell after her r224 %%writefile (cell 9) that
#                                            replaces /tmp/raptor224_run.py by a script that exits 3, so her stage 2b reports
#                                            rc=3 ok=False, stage 3 blends without r224 and the canary must blank Fracture.
#   v224diag  eeshsaxena/rsna-sub-v224diag   wheel-order diagnostic (never submit): her r224 script (cell 9 verbatim) run 3 times
#                                            with a raw-probability dump: stock image (A), stock again (B, the GPU noise floor),
#                                            after her cell 6's pip install of R7's pinned wheels (C). Prints WHEEL_DIAG.
# Canary (v224, v224flip, v224ctl): her only silent fallback is "blend without r224" (R224_SKIPPED by the 9 h guard,
# R224_TIMEOUT, rc != 0, missing csv, R224_UNUSABLE); all of them leave r224 None in stage 3. The canary cell then sets
# Fracture to 0.5 for every study (about -.03 macro AUC), prints CANARY_FIRED and re-writes submission.csv from her in-memory
# `sub` (the other 11 columns serialise exactly as before). Clean path: prints CANARY_CLEAN, submission.csv untouched.
# Her other failure paths already raise (no submission) and her per-study fallbacks print BLEND_FALLBACK (kept as is).
#   usage: python build_v224.py [--target v224|v224flip|v224ctl|v224diag|all] [--out DIR]
import os, json, copy, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_NB = os.path.join(HERE, 'src', 'varshitha_v224.ipynb')
SRC_NB_SHA = '6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d'
SRC_META = os.path.join(HERE, 'src', 'varshitha_kernel-metadata.json')
CI_SHA = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'   # compact_infer.py v4c pinned in her cell 6
ap = argparse.ArgumentParser()
ap.add_argument('--target', choices=['v224', 'v224flip', 'v224ctl', 'v224diag', 'all'], default='all')
ap.add_argument('--out', default=HERE)
ARGS = ap.parse_args()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def dumps(nb):
    # her file's exact serialisation (verified: json.dumps of her parsed notebook with these options == her bytes)
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s


def code_cell(text):
    # same shape as her code cells: string source, no id, no outputs
    return {'cell_type': 'code', 'metadata': {}, 'source': text, 'outputs': [], 'execution_count': None}


def md_cell(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}


assert sha(SRC_NB) == SRC_NB_SHA, 'varshitha notebook copy changed'
RAW = open(SRC_NB, 'rb').read()
HER = json.loads(RAW.decode('utf-8'))
assert dumps(HER).encode('utf-8') == RAW, 'serialisation does not reproduce her file'
HM = json.load(open(SRC_META, encoding='utf-8'))
assert HM['id'] == 'varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956'
assert HM['dataset_sources'] == ['heliosli/rsna-knee-compact-r7cnxfsss', 'nartaa/rsna-knee-publication-swa-weights-20261007',
                                 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004'], HM['dataset_sources']
assert HM['docker_image'] == 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
assert HM['machine_shape'] == 'NvidiaTeslaT4' and HM['enable_gpu'] is True and HM['enable_internet'] is False
assert HM['competition_sources'] == ['rsna-knee-abnormality-detection'] and HM['kernel_sources'] == [] and HM['model_sources'] == []

C = HER['cells']
assert len(C) == 13 and [c['cell_type'] for c in C] == ['markdown'] * 3 + ['code', 'code', 'markdown', 'code', 'code', 'markdown',
                                                                           'code', 'code', 'markdown', 'code']
assert src(C[3]).startswith('%%writefile /tmp/raptor_run.py\n')
assert src(C[4]).startswith('# Stage 1: public Raptor model') and 'NB_T0 = time.time()' in src(C[4])
assert src(C[6]).startswith("ARM = 'E_lrprior'\n") and "if manifest.get('flip_tta'):\n    # v4c" in src(C[6])
assert "    WORKER_ARGS += ['--flip-tta']\n" in src(C[6]) and "print('ASSETS_READY'" in src(C[6])
assert src(C[7]).lstrip().startswith('deadline = T0 + DEADLINE_HOURS * 3600') and "'--worker-args', json.dumps(WORKER_ARGS)" in src(C[7])
assert src(C[9]).startswith('%%writefile /tmp/raptor224_run.py\n')
assert "'3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'" in src(C[9])
assert 'out = "/kaggle/working/raptor224_sub.csv"' in src(C[9])
assert src(C[10]).startswith("# Stage 2b [s5]") and "_left = 9 * 3600 - (time.time() - NB_T0) - 40 * 60" in src(C[10])
assert "subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py']" in src(C[10])
assert src(C[12]).startswith('# Stage 3: per-finding rank blend') and 'A224 = 0.35' in src(C[12])
assert "sub.to_csv(WORK / 'submission.csv', index=False)" in src(C[12]) and "r224 = None" in src(C[12])
# her cell 6 pins compact_infer v4c (the source that knows --flip-tta)
assert ("'compact_infer.py': '%s'" % CI_SHA) in src(C[6])

# ---- inserted cells --------------------------------------------------------------------------------------------------
CANARY = r'''# [eeshsaxena CANARY; not in the public notebook] V224's only silent fallback is "blend without the 224 model" (stage 2b
# skipped by the 9 h guard, timed out, failed, csv missing or unusable: r224 is None above). Make it visible on the
# leaderboard: Fracture = 0.5 for every study (one constant column, about -.03 macro AUC). Clean path: nothing changes.
import json as _cj
USE_F, FLIP_TTA_ON = False, bool(globals().get('FLIP_TTA_FORCED_ON', False))   # no fleet cell in this kernel
_canary = {'r224_used': r224 is not None, 'R224_OK': bool(R224_OK), 'use_f': USE_F, 'flip_tta_forced': FLIP_TTA_ON, 'fired': []}
if r224 is None:
    assert 'Fracture' in labels
    sub['Fracture'] = np.float32(0.5)
    assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
    sub.to_csv(WORK / 'submission.csv', index=False)
    _canary['fired'].append('r224->Fracture')
    print('CANARY_FIRED r224 not used (R224_OK=%s): Fracture set to 0.5 for all %d rows' % (R224_OK, n), flush=True)
else:
    print('CANARY_CLEAN r224 used, no column blanked', flush=True)
(WORK / 'canary_receipt.json').write_text(_cj.dumps(_canary, indent=1))
print('V224_DONE', _cj.dumps(_canary), flush=True)
'''

FLIP = r'''# [eeshsaxena V224-FLIP; the ONE change vs V224-exact] R7 anatomical flip TTA forced on. The pinned R7 manifest ships
# flip_tta false, so the cell above built WORKER_ARGS without --flip-tta. compact_infer v4c (sha 8401812c, pinned above) then
# also scores the mirrored windows per fold model, swaps Medial/Lateral Meniscus and Medial/Lateral OA, averages the two passes
# in logit space and keeps MCL from the plain pass; the worker prints FLIP_TTA_READY. Nothing is fitted.
assert SOURCE_SHA['compact_infer.py'] == '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
assert '"--flip-tta"' in SOURCES['compact_infer.py'] and 'FLIP_TTA_READY' in SOURCES['compact_infer.py']
assert not manifest.get('flip_tta'), 'the R7 manifest already has flip_tta on: nothing to force'
assert '--flip-tta' not in WORKER_ARGS, WORKER_ARGS
assert manifest['backbone'] != 'ofl_l', 'flip TTA is for timm backbones only'
manifest['flip_tta'] = True          # in memory only; the dataset file and every sha assert above are unchanged
WORKER_ARGS += ['--flip-tta']        # the same flag her cell above adds when the manifest says flip_tta
FLIP_TTA_FORCED_ON = True
print('FLIP_TTA_FORCED', json.dumps({'worker_args': WORKER_ARGS}), flush=True)
'''

CTL = r'''# [eeshsaxena CONTROL; never submit] force V224's r224 failure path: replace the 224 script by one that exits 3, so stage 2b
# prints R224_STAGE rc=3 ok=False, stage 3 blends without the 224 model, and the canary must set Fracture to 0.5 (and only it).
with open('/tmp/raptor224_run.py', 'w') as _f:
    _f.write("import sys\nprint('CONTROL_R224_FORCED_FAIL exit 3', flush=True)\nsys.exit(3)\n")
print('CONTROL_R224_REPLACED /tmp/raptor224_run.py now exits 3', flush=True)
'''

# ---- diagnostic notebook ---------------------------------------------------------------------------------------------
c6 = src(C[6])
FIND_MOUNT = c6[c6.index('def find_mount(slug, required):'):c6.index('CODE_SRC = find_mount(CODE_SLUG')]
WHEELS_BLOCK = c6[c6.index('# pinned wheels (same versions'):c6.index("SRC = TMP / 'probe_src'")]
assert "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)" in WHEELS_BLOCK
DIAG_MD = ('# rsna-sub-v224diag: wheel-order diagnostic (private, never submitted)\n\n'
           "varshitha's r224 stage (her cell 9, verbatim) runs after R7's `pip install --no-index --no-deps` of timm 1.0.25 / "
           'pydicom 3.0.1; our HR7-RR ran it before (image timm 1.0.29 / pydicom 3.0.2). This kernel dumps the r224 raw '
           'probabilities three times in one session: A stock, B stock again (GPU noise floor), C after the R7 wheels '
           "(her cell 6's wheel block, verbatim). Only the output file names and a raw dump are patched into a copy of the script.")
DIAG_PATCH = r'''# Patch a COPY of her r224 script: env-driven output names + raw probability dump (arm 0 = the only arm). Her file is untouched.
import os, sys, time, json, hashlib, subprocess
from pathlib import Path
_s = open('/tmp/raptor224_run.py', encoding='utf-8').read()
_old_out = 'out = "/kaggle/working/raptor224_sub.csv"'
_old_csv = '    sub.to_csv(out, index=False)\n'
assert _s.count(_old_out) == 1 and _s.count(_old_csv) == 1
_s = _s.replace(_old_out, 'out = os.environ["R224_SUB_OUT"]')
_s = _s.replace(_old_csv, _old_csv + '    _raw = pd.DataFrame(arm_probs[0].astype(np.float64), columns=LAB); _raw.insert(0, "StudyInstanceUID", test_ids)\n'
                '    _raw.to_csv(os.environ["R224_RAW_OUT"], index=False, float_format="%.9g")\n')
open('/tmp/r224_diag.py', 'w', encoding='utf-8').write(_s)
print('DIAG_SCRIPT sha256', hashlib.sha256(_s.encode('utf-8')).hexdigest()[:16], flush=True)
WORK = Path('/kaggle/working')


def versions(tag):
    r = subprocess.run([sys.executable, '-c', 'import timm, pydicom, torch; print(timm.__version__, pydicom.__version__, torch.__version__)'],
                       capture_output=True, text=True)
    print('DIAG_VERSIONS', tag, r.stdout.strip(), r.stderr.strip()[-300:], flush=True)
    return r.stdout.strip()


def run224(tag):
    env = dict(os.environ, R224_SUB_OUT=str(WORK / ('diag_%s_sub.csv' % tag)), R224_RAW_OUT=str(WORK / ('diag_%s_raw.csv' % tag)))
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', '/tmp/r224_diag.py'], env=env, timeout=1800).returncode
    print('DIAG_RUN', tag, 'rc', rc, 'seconds %.0f' % (time.time() - t), flush=True)
    assert rc == 0 and (WORK / ('diag_%s_raw.csv' % tag)).is_file(), tag


DIAG = {'A': versions('A')}
run224('A')
DIAG['B'] = versions('B')
run224('B')
'''
DIAG_WHEELS = ('# her cell 6: find_mount + the pinned-wheel block, verbatim (only what installs the wheels)\n'
               "import os, sys, json, shutil, subprocess, zipfile, time, hashlib\nfrom pathlib import Path\n"
               "INPUT = Path('/kaggle/input')\nTMP = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')\n"
               "TMP = TMP / 'probe'\nTMP.mkdir(parents=True, exist_ok=True)\nCODE_SLUG = 'rsna-orthofoundation-fold0-epoch7-20261004'\n\n"
               + FIND_MOUNT + "CODE_SRC = find_mount(CODE_SLUG, 'receipt.json')\n\n" + WHEELS_BLOCK +
               "DIAG['C'] = versions('C')\nrun224('C')\n")
DIAG_CMP = r'''# compare: A vs B = GPU noise floor; A vs C and B vs C = stock image vs R7 wheels
import numpy as np, pandas as pd
R = {t: pd.read_csv(WORK / ('diag_%s_raw.csv' % t), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID') for t in 'ABC'}
S = {t: pd.read_csv(WORK / ('diag_%s_sub.csv' % t), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID') for t in 'ABC'}
out = {'versions': DIAG}
for a_, b_ in (('A', 'B'), ('A', 'C'), ('B', 'C')):
    d = (R[a_] - R[b_].loc[R[a_].index, R[a_].columns]).abs()
    out[a_ + b_] = {'max_abs': float(d.values.max()), 'mean_abs': float(d.values.mean()),
                    'per_label_max': {c: float(d[c].max()) for c in d.columns},
                    'sub_equal': bool((S[a_].values == S[b_].loc[S[a_].index, S[a_].columns].values).all())}
floor = out['AB']['max_abs']
env = max(out['AC']['max_abs'], out['BC']['max_abs'])
out['verdict'] = 'ENV_EFFECT' if env > 1e-3 and env > 10 * max(floor, 1e-7) else 'NO_ENV_EFFECT'
(WORK / 'diag_receipt.json').write_text(json.dumps(out, indent=1))
print('WHEEL_DIAG', json.dumps({k: out[k] for k in ('versions', 'verdict')}), 'floor_AB %.3g' % floor, 'env_max %.3g' % env, flush=True)
print(json.dumps(out, indent=1))
'''


def assemble(target):
    if target == 'v224diag':
        return [md_cell(DIAG_MD), copy.deepcopy(C[9]), code_cell(DIAG_PATCH), code_cell(DIAG_WHEELS), code_cell(DIAG_CMP)], set()
    cells, ins = [], set()
    for i, c in enumerate(C):
        cells.append(copy.deepcopy(c))
        if i == 6 and target == 'v224flip':
            cells.append(code_cell(FLIP)); ins.add(len(cells) - 1)
        if i == 9 and target == 'v224ctl':
            cells.append(code_cell(CTL)); ins.add(len(cells) - 1)
        if i == 12:
            cells.append(code_cell(CANARY)); ins.add(len(cells) - 1)
    return cells, ins


for t in (CANARY, FLIP, CTL, DIAG_PATCH, DIAG_WHEELS, DIAG_CMP):
    compile(t, '<cell>', 'exec')

TARGETS = ['v224', 'v224flip', 'v224ctl', 'v224diag'] if ARGS.target == 'all' else [ARGS.target]
for target in TARGETS:
    cells, ins = assemble(target)
    nb = copy.deepcopy(HER)
    nb['cells'] = cells
    if target != 'v224diag':
        # verbatim proof: drop the inserted cells -> her notebook file, byte for byte
        back = copy.deepcopy(nb)
        back['cells'] = [c for i, c in enumerate(cells) if i not in ins]
        assert dumps(back).encode('utf-8') == RAW, (target, 'her cells are not verbatim')
        assert len(cells) == 13 + len(ins) and len(ins) == (1 if target == 'v224' else 2)
        # cell order: R7 wheels (her cell 6) strictly before the r224 run (her cell 10)
        pos6 = [i for i, c in enumerate(cells) if src(c) == src(C[6])][0]
        pos10 = [i for i, c in enumerate(cells) if src(c) == src(C[10])][0]
        assert pos6 < pos10
    name = 'rsna-sub-' + target
    code_file = 'rsna_sub_%s.ipynb' % target
    out_dir = os.path.join(ARGS.out, 'kernel_' + target)
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, code_file)
    with open(nb_path, 'w', encoding='utf-8', newline='') as f:
        f.write(dumps(nb))
    meta = {k: copy.deepcopy(v) for k, v in HM.items() if k not in ('id_no', 'keywords')}
    meta.update(id='eeshsaxena/' + name, title=name.replace('-', ' '), code_file=code_file, is_private=True,
                enable_gpu=True, enable_tpu=False, enable_internet=False)
    if target == 'v224diag':   # no R7 bundle needed: the r224 weights + R7's code dataset (wheels) + the competition
        meta['dataset_sources'] = ['nartaa/rsna-knee-publication-swa-weights-20261007', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']
    with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print('BUILT', target, nb_path, 'sha256', sha(nb_path), 'cells', len(cells), 'inserted', sorted(ins),
          'datasets', meta['dataset_sources'])
