#!/usr/bin/env python
# build_v224ff.py (2026-10-15 dir, built 10-14 by a Claude subagent): V224-F-FLIP = V224-F (parametric in w_f, fleet stage and
# its budget exactly as V224-F) + R7 flip TTA with a TIME-AWARE flip. ONE change vs V224-F{w}.
#
# Base: session_2026-10-15/v224f/build_v224f.py is RUN (subprocess, same --wf / --start-budget-h / --wall-h) into base/; the
# resulting V224-F notebook is asserted byte-identical to the V224-F builder's own kernel_v224f{w} file when that exists (the
# kernel V224-F commit-tested), and every one of its 22 cells is kept BYTE-IDENTICAL and in order. The builder only INSERTS:
#   FLIP-TA  after varshitha's R7 assets cell (her cell 6, V224-F cell 6), before her R7 run cell (cell 7):
#            the V224-FLIP asserts (compact_infer sha 8401812c = v4c with --flip-tta / FLIP_TTA_READY; the pinned manifest has
#            flip_tta off; WORKER_ARGS has no --flip-tta; timm backbone), then writes compact_infer_ta.py (this dir; sha pinned)
#            next to the pinned sources, sets manifest['flip_tta'] = True in memory, WORKER_ARGS += ['--flip-tta'] (V224-FLIP's
#            flag) + the driver's --ta-* arguments, WORKER = 'compact_infer_ta.py'; prints FLIP_TTA_FORCED. Her cell 7 is
#            unchanged: probe_dual.py launches the driver per GPU, the driver runs the PINNED compact_infer.py with --flip-tta and
#            disables flip for the rest of that worker's studies (pinned compact_infer.py without --flip-tta = the V224-exact
#            worker) as soon as the projected R7 end with flip > the flip budget (FLIP_TTA_DISABLED_AT_STUDY k).
#            Flip budget (hours after notebook start) = env V224FF_FLIP_BUDGET_H if set, else --flip-budget-h (default
#            start_budget_h - 0.6 = 4.9 h: R7 + r224 (~0.3 h hidden, x1.3 slow) + the fleet pre-cells must end before V224-F's
#            fleet start guard (5.5 h), else the fleet is skipped and its Synovitis canary fires).
#   REPORT   after her R7 run cell (7): reads pred/worker*/receipt.json -> time_aware_flip, prints FLIP_TA_REPORT (and one
#            FLIP_TTA_DISABLED_AT_STUDY line per worker that switched), writes /kaggle/working/flip_ta_receipt.json. Never raises.
#   CTL      (--control only, before FLIP-TA; never submit): forced time pressure, os.environ['V224FF_FLIP_BUDGET_H'] = notebook
#            age now + 1 min. The driver starts flip (budget not yet passed), the ~4 min compile check pushes the first study
#            past the budget, so the 2-study worker disables flip at study 1 and scores its 2nd study without flip.
#   usage: python build_v224ff.py --wf 0.20 [--control] [--flip-budget-h 4.9] [--start-budget-h 5.5] [--wall-h 8.3]
#   slug:  eeshsaxena/rsna-sub-v224ff{WF*100} (v224ff20 / v224ff25 / v224ff30); control eeshsaxena/rsna-sub-v224ffctl
import os, sys, json, copy, hashlib, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224F_BUILDER = os.path.join(ROOT, 'session_2026-10-15', 'v224f', 'build_v224f.py')
V224F_DIR = os.path.dirname(V224F_BUILDER)
HER_NB = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'src', 'varshitha_v224.ipynb')
HER_SHA = '6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d'
DRIVER = os.path.join(HERE, 'compact_infer_ta.py')
CI_SHA = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'

ap = argparse.ArgumentParser()
ap.add_argument('--wf', type=float, required=True)
ap.add_argument('--control', action='store_true', help='forced time pressure: flip disables itself mid-R7; never submit')
ap.add_argument('--start-budget-h', type=float, default=5.5, help="V224-F's fleet start budget (passed to build_v224f.py)")
ap.add_argument('--wall-h', type=float, default=8.3, help="V224-F's fleet wall (passed to build_v224f.py)")
ap.add_argument('--flip-budget-h', type=float, default=None, help='projected R7 end with flip must stay <= this (h after start); '
                'default start_budget_h - 0.6')
ap.add_argument('--ta-min-studies', type=int, default=10)
ap.add_argument('--ta-switch-min-s', type=float, default=600.0)
ap.add_argument('--ta-safety', type=float, default=1.05)
ap.add_argument('--out', default=HERE)
A = ap.parse_args()
FB = round(A.start_budget_h - 0.6, 4) if A.flip_budget_h is None else float(A.flip_budget_h)
assert 0.0 < A.wf < 1.0 and abs(A.wf * 100 - round(A.wf * 100)) < 1e-9, A.wf
assert 1.0 < FB <= A.start_budget_h - 0.3, ('flip budget must leave room for r224 before the fleet start guard', FB)
assert A.ta_min_studies >= 2 and A.ta_switch_min_s >= 0 and A.ta_safety >= 1.0


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s


def code_cell(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text, 'outputs': [], 'execution_count': None}


# ---- 1. the V224-F base at this w_f, from the V224-F builder itself --------------------------------------------------------------
WTAG = 'v224f%02d' % int(round(A.wf * 100))
base_out = os.path.join(A.out, 'base')
r = subprocess.run([sys.executable, V224F_BUILDER, '--wf', repr(A.wf), '--start-budget-h', repr(A.start_budget_h),
                    '--wall-h', repr(A.wall_h), '--out', base_out], capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
print(r.stdout.strip().splitlines()[-1][:300])
BASE_NB = os.path.join(base_out, 'kernel_' + WTAG, 'rsna_sub_%s.ipynb' % WTAG)
BASE_META = os.path.join(base_out, 'kernel_' + WTAG, 'kernel-metadata.json')
RAW = open(BASE_NB, 'rb').read()
theirs = os.path.join(V224F_DIR, 'kernel_' + WTAG, 'rsna_sub_%s.ipynb' % WTAG)
SAME_AS_V224F_KERNEL = os.path.isfile(theirs) and open(theirs, 'rb').read() == RAW
if os.path.isfile(theirs):
    assert SAME_AS_V224F_KERNEL, 'the V224-F builder now builds a different %s than its own kernel dir holds' % WTAG
BASE = json.loads(RAW.decode('utf-8'))
assert dumps(BASE).encode('utf-8') == RAW
BC = BASE['cells']
assert sha(HER_NB) == HER_SHA
HC = json.load(open(HER_NB, encoding='utf-8'))['cells']
assert len(BC) == 22 and all(src(BC[i]) == src(HC[i]) and BC[i]['cell_type'] == HC[i]['cell_type'] for i in range(13)), 'base != her 13 cells'
assert src(BC[13]).startswith('# [eeshsaxena CANARY') and "bool(globals().get('FLIP_TTA_FORCED_ON', False))" in src(BC[13])
assert src(BC[14]).startswith('# [eeshsaxena V224-F SAVE') and src(BC[21]).startswith('# [eeshsaxena V224-F HF BLEND')
C6, C7 = src(BC[6]), src(BC[7])
assert C6.startswith("ARM = 'E_lrprior'\n") and "WORKER = 'compact_infer.py'\n" in C6 and ("'compact_infer.py': '%s'" % CI_SHA) in C6
assert "if manifest.get('flip_tta'):\n    # v4c" in C6 and "    WORKER_ARGS += ['--flip-tta']\n" in C6 and 'NB_T0' in C6
assert "'--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS)" in C7 and "for log in sorted(pred.glob('worker*/log.txt')):" in C7
assert "print('OURS_STAGE_COMPLETE'" in C7
BMETA = json.load(open(BASE_META, encoding='utf-8'))
assert BMETA['id'] == 'eeshsaxena/rsna-sub-' + WTAG

# ---- 2. inserted cells --------------------------------------------------------------------------------------------------------
TA_TEXT = open(DRIVER, encoding='utf-8').read()
TA_SHA = hashlib.sha256(TA_TEXT.encode('utf-8')).hexdigest()
assert ('CHILD_SHA = %r' % CI_SHA) in TA_TEXT and chr(0x2014) not in TA_TEXT
compile(TA_TEXT, 'compact_infer_ta.py', 'exec')

FLIP_TA = r'''# [eeshsaxena V224-F-FLIP; the ONE change vs V224-F] R7 anatomical flip TTA forced on, TIME-AWARE. The pinned R7 manifest
# ships flip_tta false, so the cell above built WORKER_ARGS without --flip-tta. compact_infer v4c (sha 8401812c, pinned above) then
# also scores the mirrored windows per fold model, swaps Medial/Lateral Meniscus and Medial/Lateral OA, averages the two passes in
# logit space and keeps MCL from the plain pass (FLIP_TTA_READY). Nothing is fitted. Time awareness: probe_dual.py (cell below,
# unchanged) launches compact_infer_ta.py (written here, sha pinned) per GPU instead of compact_infer.py; it runs the PINNED
# compact_infer.py with --flip-tta, and after every study projects this worker's R7 end with flip from its measured rate
# (seconds per DICOM file, rows 2..k): once projected end > FLIP_TA_BUDGET_H after the notebook start, it kills that child, keeps
# its finished rows and scores the remaining studies with the pinned compact_infer.py WITHOUT --flip-tta (the V224-exact worker),
# printing FLIP_TTA_DISABLED_AT_STUDY k. The budget keeps R7 + r224 inside V224-F's fleet start guard (__SB__ h).
import os, json, time, hashlib
assert SOURCE_SHA['compact_infer.py'] == '__CI__'
assert '"--flip-tta"' in SOURCES['compact_infer.py'] and 'FLIP_TTA_READY' in SOURCES['compact_infer.py']
assert hashlib.sha256((SRC / 'compact_infer.py').read_bytes()).hexdigest() == '__CI__'
assert not manifest.get('flip_tta'), 'the R7 manifest already has flip_tta on: nothing to force'
assert '--flip-tta' not in WORKER_ARGS and not any(str(a).startswith('--ta-') for a in WORKER_ARGS), WORKER_ARGS
assert manifest['backbone'] != 'ofl_l', 'flip TTA is for timm backbones only'
assert WORKER == 'compact_infer.py', WORKER
FLIP_TA_SRC = __TASRC__
(SRC / 'compact_infer_ta.py').write_bytes(FLIP_TA_SRC.encode('utf-8'))
assert hashlib.sha256((SRC / 'compact_infer_ta.py').read_bytes()).hexdigest() == '__TASHA__'
_fta_env = os.environ.get('V224FF_FLIP_BUDGET_H')
FLIP_TA_BUDGET_H = float(_fta_env) if _fta_env else __FB__
FLIP_TA_BUDGET_SOURCE = 'env V224FF_FLIP_BUDGET_H' if _fta_env else 'built in'
assert 0.0 < FLIP_TA_BUDGET_H < DEADLINE_HOURS, FLIP_TA_BUDGET_H
manifest['flip_tta'] = True          # in memory only; the dataset file and every sha assert above are unchanged
WORKER_ARGS += ['--flip-tta']        # the same flag her cell above adds when the manifest says flip_tta (V224-FLIP)
WORKER_ARGS += ['--ta-budget-end', repr(NB_T0 + FLIP_TA_BUDGET_H * 3600), '--ta-nb-t0', repr(NB_T0), '--ta-min-studies', '__KMIN__',
                '--ta-switch-min-s', '__SWMIN__', '--ta-safety', '__SAFETY__']
WORKER = 'compact_infer_ta.py'
FLIP_TTA_FORCED_ON = True
print('FLIP_TTA_FORCED', json.dumps({'worker': WORKER, 'worker_args': WORKER_ARGS, 'flip_budget_h': FLIP_TA_BUDGET_H,
                                     'budget_source': FLIP_TA_BUDGET_SOURCE, 'elapsed_h': round((time.time() - NB_T0) / 3600, 4),
                                     'driver_sha256': '__TASHA__'[:16]}), flush=True)
'''.replace('__CI__', CI_SHA).replace('__SB__', repr(A.start_budget_h)).replace('__TASHA__', TA_SHA) \
   .replace('__FB__', repr(FB)).replace('__KMIN__', str(A.ta_min_studies)).replace('__SWMIN__', repr(float(A.ta_switch_min_s))) \
   .replace('__SAFETY__', repr(float(A.ta_safety))).replace('__TASRC__', repr(TA_TEXT))   # the driver text last: no token can hit it

REPORT = r'''# [eeshsaxena V224-F-FLIP REPORT] what the time-aware flip did on each R7 worker (compact_infer_ta.py's time_aware_flip block of
# pred/worker*/receipt.json): FLIP_TA_REPORT + one FLIP_TTA_DISABLED_AT_STUDY line per worker that switched; receipt copied to
# /kaggle/working/flip_ta_receipt.json. Read-only on the R7 outputs (the cell above already copied them); it never stops the run.
import json as _fj
_fta = {'budget_h': FLIP_TA_BUDGET_H, 'budget_source': FLIP_TA_BUDGET_SOURCE, 'workers': {}}
try:
    for _wd in sorted(pred.glob('worker*')):
        if not _wd.is_dir():
            continue
        try:
            _t = _fj.loads((_wd / 'receipt.json').read_text(encoding='utf-8')).get('time_aware_flip') or {}
        except Exception as _e:
            _t = {'decision': 'no receipt', 'error': repr(_e)[:200]}
        _fta['workers'][_wd.name] = _t
        if _t.get('decision') != 'none':
            print('FLIP_TTA_DISABLED_AT_STUDY %s %s' % (_t.get('disabled_at_study'), _fj.dumps(
                {'worker': _wd.name, **{k: _t.get(k) for k in ('decision', 'reason', 'rule', 'decided_at_study', 'decided_h',
                                                              'projected_end_h', 'budget_end_h', 'flipped', 'plain', 'studies',
                                                              'kill_rc', 'plain_rc', 'end_h')}})), flush=True)
    _ws = list(_fta['workers'].values())
    _fta.update(flipped=sum(int(w.get('flipped') or 0) for w in _ws), plain=sum(int(w.get('plain') or 0) for w in _ws),
                decisions=sorted(set(str(w.get('decision')) for w in _ws)),
                all_flipped=bool(_ws) and all(w.get('decision') == 'none' for w in _ws))
except Exception as _e:
    _fta['error'] = repr(_e)[:300]
    print('FLIP_TA_REPORT_ERROR', repr(_e)[:300], flush=True)
try:
    (WORK / 'flip_ta_receipt.json').write_text(_fj.dumps(_fta, indent=1))
except Exception as _e:
    print('FLIP_TA_REPORT_ERROR', repr(_e)[:300], flush=True)
print('FLIP_TA_REPORT', _fj.dumps({k: _fta.get(k) for k in ('budget_h', 'budget_source', 'flipped', 'plain', 'decisions', 'all_flipped')}),
      flush=True)
'''

CTL = r'''# [eeshsaxena V224-F-FLIP CONTROL; never submit] forced time pressure: the flip budget env var is set to the notebook age now
# + 1 minute. The driver starts flip (budget not yet passed at its start); the ~4 min compile check of the first study pushes the
# first finished study past the budget, so a worker with more than one study disables flip at study 1 and scores the rest
# without it (FLIP_TTA_DISABLED_AT_STUDY 1); a 1-study worker has nothing left to switch.
import os, time
os.environ['V224FF_FLIP_BUDGET_H'] = repr(round((time.time() - NB_T0) / 3600.0 + 1.0 / 60.0, 6))
print('CONTROL_FLIP_BUDGET_H', os.environ['V224FF_FLIP_BUDGET_H'], flush=True)
'''

for t in (FLIP_TA, REPORT, CTL):
    compile(t, '<cell>', 'exec')
for t in (FLIP_TA, REPORT, CTL):
    assert chr(0x2014) not in t

# ---- 3. assemble ----------------------------------------------------------------------------------------------------------------
cells, ins = [], set()
for i, c in enumerate(BC):
    cells.append(copy.deepcopy(c))
    if i == 6:
        if A.control:
            cells.append(code_cell(CTL)); ins.add(len(cells) - 1)
        cells.append(code_cell(FLIP_TA)); ins.add(len(cells) - 1)
    if i == 7:
        cells.append(code_cell(REPORT)); ins.add(len(cells) - 1)
nb = copy.deepcopy(BASE)
nb['cells'] = cells
back = copy.deepcopy(nb); back['cells'] = [c for i, c in enumerate(cells) if i not in ins]
assert dumps(back).encode('utf-8') == RAW, 'V224-F cells are not verbatim'
assert len(cells) == 22 + len(ins) and len(ins) == (3 if A.control else 2)
p6 = [i for i, c in enumerate(cells) if src(c) == C6][0]
p7 = [i for i, c in enumerate(cells) if src(c) == C7][0]
pf = [i for i, c in enumerate(cells) if src(c) == FLIP_TA][0]
pr = [i for i, c in enumerate(cells) if src(c) == REPORT][0]
assert p6 < pf < p7 < pr, (p6, pf, p7, pr)

TAG = 'v224ffctl' if A.control else 'v224ff%02d' % int(round(A.wf * 100))
name = 'rsna-sub-' + TAG
code_file = 'rsna_sub_%s.ipynb' % TAG
out_dir = os.path.join(A.out, 'kernel_' + TAG)
os.makedirs(out_dir, exist_ok=True)
nb_path = os.path.join(out_dir, code_file)
with open(nb_path, 'w', encoding='utf-8', newline='') as f:
    f.write(dumps(nb))
meta = copy.deepcopy(BMETA)
meta.update(id='eeshsaxena/' + name, title=name.replace('-', ' '), code_file=code_file, is_private=True,
            enable_gpu=True, enable_tpu=False, enable_internet=False)
assert meta['dataset_sources'] == BMETA['dataset_sources'] and len(meta['dataset_sources']) == 5
with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)
with open(os.path.join(A.out, 'celldiff_%s.txt' % TAG), 'w', encoding='utf-8', newline='\n') as f:
    f.write(f'# {name}: V224-F {WTAG} (base sha {hashlib.sha256(RAW).hexdigest()}, same bytes as v224f/kernel_{WTAG}: '
            f'{SAME_AS_V224F_KERNEL}) cells kept byte-identical; inserted cells {sorted(ins)} (full text below; the FLIP_TA_SRC '
            f'literal is compact_infer_ta.py sha {TA_SHA}). w_f {A.wf}, flip budget {FB} h, fleet start budget {A.start_budget_h} h, '
            f'wall {A.wall_h} h, min_studies {A.ta_min_studies}, switch_min_s {A.ta_switch_min_s}, safety {A.ta_safety}, control {A.control}\n')
    for i in sorted(ins):
        t = src(cells[i])
        if t == FLIP_TA:
            t = t.replace(repr(TA_TEXT), "<compact_infer_ta.py source, %d chars, sha256 %s>" % (len(TA_TEXT), TA_SHA))
        f.write(f'\n#### cell {i}\n{t}')
print('BUILT', TAG, nb_path, 'sha256', sha(nb_path), 'cells', len(cells), 'inserted', sorted(ins), 'base', WTAG,
      'base_sha', hashlib.sha256(RAW).hexdigest()[:16], 'same_as_v224f_kernel', SAME_AS_V224F_KERNEL, 'flip_budget_h', FB,
      'driver_sha', TA_SHA[:16])
