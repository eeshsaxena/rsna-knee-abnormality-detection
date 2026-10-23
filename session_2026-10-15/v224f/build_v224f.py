#!/usr/bin/env python
# build_v224f.py (2026-10-15 dir, built 10-14 by a Claude subagent): V224-F = V224-exact + our HF fleet stage, parametric in w_f.
#
# Base: session_2026-10-14/v224/kernel_v224/rsna_sub_v224.ipynb (V224-exact, sha 4259ec97...: varshitha's 13 cells verbatim +
# our CANARY cell). Its 14 cells are kept BYTE-IDENTICAL and in order; the builder only APPENDS 8 cells after the canary, and
# asserts that deleting them re-serialises to the V224-exact notebook file byte for byte (and, minus the canary, to her file).
# Fleet cells: session_2026-10-13/hr7rr/kernel_hfr7rr/rsna_sub_hfr7rr.ipynb (HFR7-RR, our best scored entry, sha ca3da2fb...):
# ENV-SNAP / FN-IDS / K12-MERGE / ENV-RESTORE verbatim; the K2 fleet cell verbatim except its first line (the K2_CFG literal),
# where only WALL_H and MIN_START_MIN change (the fleet's own tested P3 time guard = this kernel's fleet time budget).
#
# Appended cells (14-21):
#   14 V224F SAVE   (ours)  submission.csv (V224 output incl. the canary) -> submission_V224.csv, sha256 recorded; prints
#                           V224F_SAVED with the elapsed time and the budget decision the fleet guard will take.
#   15 ENV-SNAP     (HFR7-RR cell 7, verbatim)
#   16 FN-IDS       (HFR7-RR cell 8, verbatim; NOTE it rewrites submission.csv as a 0.5 id placeholder and sets T0 = now)
#   17 K12-MERGE    (HFR7-RR cell 9, verbatim)
#   18 V224F CLOCK  (ours)  T0 = min(T0, NB_T0): the fleet wall is min(/proc start, T0); FN-IDS just set T0 ~3 h into the run,
#                           so without this a missing /proc clock would move the wall past 9 h.
#   19 K2 FLEET     (HFR7-RR cell 10; K2_CFG literal: WALL_H 8.3 (was 8.6), MIN_START_MIN = (WALL_H - start budget) x 60)
#                           -> the fleet starts only if the kernel is <= START_BUDGET_H old, workers stop at the wall and are
#                           killed 300 s after it; any failure = no submission_fleetsolo.csv (fail-loud set) + RESULT ... NOT CLEAN.
#   20 ENV-RESTORE  (HFR7-RR cell 11, verbatim)
#   21 HF V224F     (ours; HFR7-RR HF cell semantics) submission.csv = per-label average-tie percentile rank
#                           (1 - W_F) * pct(V224) + W_F * pct(fleet solo), float32, NOT re-ranked; H read from
#                           submission_V224.csv (sha checked, else her in-memory `sub`). A column the V224 canary blanked
#                           (r224 not used -> Fracture) is set back to 0.5 after the blend so that canary stays visible.
#                           ANY fleet problem -> HF_FALLBACK: submission.csv = V224 output with Synovitis = 0.5 (the fleet
#                           canary). Receipts hf_receipt.json + v224f_receipt.json; prints HF_COMPLETE and V224F_DONE.
# Control (--control, never submit): MIN_START_MIN 600 (> the 498 min a fresh kernel has to the 8.3 h wall), so the fleet
#   guard refuses to start -> HF_FALLBACK -> Synovitis 0.5, every other column = the V224 output.
#
#   usage: python build_v224f.py --wf 0.20 [--control] [--start-budget-h 5.5] [--wall-h 8.3] [--out DIR]
#   slug:  eeshsaxena/rsna-sub-v224f{WF*100} (e.g. v224f20, v224f25, v224f30); control: eeshsaxena/rsna-sub-v224fctl
import os, json, copy, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
V224_DIR = os.path.join(ROOT, 'session_2026-10-14', 'v224')
V224_NB = os.path.join(V224_DIR, 'kernel_v224', 'rsna_sub_v224.ipynb')
V224_SHA = '4259ec9774167bb280b794574b73e1ba73203e8a1995e97116388011c4226877'
V224_META = os.path.join(V224_DIR, 'kernel_v224', 'kernel-metadata.json')
HER_NB = os.path.join(V224_DIR, 'src', 'varshitha_v224.ipynb')
HER_SHA = '6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d'
HFR7RR_NB = os.path.join(ROOT, 'session_2026-10-13', 'hr7rr', 'kernel_hfr7rr', 'rsna_sub_hfr7rr.ipynb')
HFR7RR_SHA = 'ca3da2fbfbf35816f2a34b8d3c8e137f0a5fada89b2ac3555fc55e47571f9b11'
HFR7RR_META = os.path.join(ROOT, 'session_2026-10-13', 'hr7rr', 'kernel_hfr7rr', 'kernel-metadata.json')
FLEET_DS = ['eeshsaxena/rsna-knee-cnxt-fleet-k11w', 'eeshsaxena/rsna-knee-cnxt-fleet-k14w']
N_MEMBERS = 20
CONTROL_MIN_START = 600.0

ap = argparse.ArgumentParser()
ap.add_argument('--wf', type=float, required=True, help='fleet weight w_f (HFR7-RR used .20)')
ap.add_argument('--control', action='store_true', help='forced fleet fallback (fleet guard never starts); never submit')
ap.add_argument('--start-budget-h', type=float, default=5.5, help='fleet starts only if the kernel is at most this old')
ap.add_argument('--wall-h', type=float, default=8.3, help='fleet wall from kernel start (workers stop, killed 300 s later)')
ap.add_argument('--out', default=HERE)
A = ap.parse_args()
assert 0.0 < A.wf < 1.0 and abs(A.wf * 100 - round(A.wf * 100)) < 1e-9, A.wf
assert 0.0 < A.start_budget_h < A.wall_h <= 8.3, (A.start_budget_h, A.wall_h)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def dumps(nb):
    # her file's exact serialisation (build_v224.py: compact separators, ensure_ascii False)
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s


def code_cell(text):
    # same shape as her code cells: string source, no id, no outputs
    return {'cell_type': 'code', 'metadata': {}, 'source': text, 'outputs': [], 'execution_count': None}


# ---- inputs and their asserts --------------------------------------------------------------------------------------------
assert sha(V224_NB) == V224_SHA, 'V224-exact notebook changed'
assert sha(HER_NB) == HER_SHA, 'varshitha notebook copy changed'
assert sha(HFR7RR_NB) == HFR7RR_SHA, 'HFR7-RR notebook changed'
RAW_V224 = open(V224_NB, 'rb').read()
V224 = json.loads(RAW_V224.decode('utf-8'))
assert dumps(V224).encode('utf-8') == RAW_V224, 'serialisation does not reproduce the V224-exact file'
HER_RAW = open(HER_NB, 'rb').read()
VC = V224['cells']
assert len(VC) == 14 and src(VC[13]).startswith('# [eeshsaxena CANARY; not in the public notebook]')
_her = copy.deepcopy(V224); _her['cells'] = VC[:13]
assert dumps(_her).encode('utf-8') == HER_RAW, 'V224-exact minus the canary is not her notebook'
assert src(VC[4]).startswith('# Stage 1: public Raptor model') and 'NB_T0 = time.time()' in src(VC[4])
assert src(VC[6]).startswith("ARM = 'E_lrprior'\n") and 'DEADLINE_HOURS = 8.3\n' in src(VC[6])
assert "_left = 9 * 3600 - (time.time() - NB_T0) - 40 * 60" in src(VC[10])
assert src(VC[12]).startswith('# Stage 3: per-finding rank blend') and "sub.to_csv(WORK / 'submission.csv', index=False)" in src(VC[12])
assert "_canary['fired'].append('r224->Fracture')" in src(VC[13]) and "print('V224_DONE'" in src(VC[13])
VMETA = json.load(open(V224_META, encoding='utf-8'))
assert VMETA['id'] == 'eeshsaxena/rsna-sub-v224' and VMETA['dataset_sources'] == [
    'heliosli/rsna-knee-compact-r7cnxfsss', 'nartaa/rsna-knee-publication-swa-weights-20261007',
    'heliosli/rsna-orthofoundation-fold0-epoch7-20261004'], VMETA
HMETA = json.load(open(HFR7RR_META, encoding='utf-8'))
assert HMETA['dataset_sources'] == VMETA['dataset_sources'] + FLEET_DS, HMETA['dataset_sources']
assert HMETA['docker_image'] == VMETA['docker_image'] and HMETA['machine_shape'] == VMETA['machine_shape'] == 'NvidiaTeslaT4'

HC = json.load(open(HFR7RR_NB, encoding='utf-8'))['cells']
assert len(HC) == 16
ENV_SNAP, FN_IDS, K12_MERGE, FLEET, ENV_RESTORE, HF_OLD = (src(HC[i]) for i in (7, 8, 9, 10, 11, 15))
assert ENV_SNAP.startswith('# === HF ENV-SNAP') and '_HF_ENV0 = dict(_hf_os.environ)' in ENV_SNAP
assert FN_IDS.startswith('# === FN-IDS') and 'T0 = time.time()' in FN_IDS and "_FN_FILES = ['submission.csv', '_fn_ids.csv', '_fn_ids.json']" in FN_IDS
assert K12_MERGE.startswith("K12M_CFG = {'base_ds': 'rsna-knee-cnxt-fleet-k11w', 'add_ds': 'rsna-knee-cnxt-fleet-k14w'")
assert FLEET.startswith('K2_CFG = {') and '_k2_receipt = _k2_entry()' in FLEET
assert "t0p = globals().get('T0')" in FLEET and "if left < float(K2['MIN_START_MIN']) * 60:" in FLEET
assert "'GRACE_S': 300" in FLEET and "deadline = k0 + float(K2['WALL_H']) * 3600" in FLEET
assert ENV_RESTORE.startswith('# === HF ENV-RESTORE') and "print('HF_ENV_RESTORED', flush=True)" in ENV_RESTORE
assert HF_OLD.startswith('# === HF BLEND') and 'W_F = 0.2\n' in HF_OLD
# the HFR7-RR HF blend formula this kernel keeps (per-label average-tie pct rank, float32, no re-rank)
assert "pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values" in HF_OLD
assert "out[c] = ((1.0 - W_F) * pct(H[c].values) + W_F * pct(F[c].values)).astype(np.float32)" in HF_OLD

# K2_CFG: same literal as HFR7-RR except WALL_H / MIN_START_MIN
nl = FLEET.index('\n')
K2_LINE, FLEET_BODY = FLEET[:nl], FLEET[nl:]
cfg = eval(K2_LINE[len('K2_CFG = '):K2_LINE.index('   # literal')], {})
assert cfg['WALL_H'] == 8.6 and cfg['MIN_START_MIN'] == 20.0 and cfg['MAIN_MODE'] == 'blend' and cfg['RHO_MIN_N'] == 1000000000
assert len(cfg['MEMBERS']) == N_MEMBERS and set(cfg['TTA'].values()) == {'AB'} and sorted(cfg['TTA']) == sorted(cfg['MEMBERS'])
min_start = CONTROL_MIN_START if A.control else round((A.wall_h - A.start_budget_h) * 60.0, 1)
assert min_start > 0
cfg_new = dict(cfg, WALL_H=float(A.wall_h), MIN_START_MIN=float(min_start))
assert [k for k in cfg if cfg[k] != cfg_new[k]] == ['WALL_H', 'MIN_START_MIN'] and list(cfg_new) == list(cfg)
K2_LINE_NEW = (f'K2_CFG = {cfg_new!r}   # literal, written by session_2026-10-15/v224f/build_v224f.py from the HFR7-RR fleet '
               f'cell (K14 R, 20 members): only WALL_H ({cfg["WALL_H"]} -> {A.wall_h}) and MIN_START_MIN ({cfg["MIN_START_MIN"]} -> '
               f'{min_start}) differ: ' + ('CONTROL, the fleet never starts (forced HF_FALLBACK)' if A.control else
                                           f'the fleet starts only if the kernel is <= {A.start_budget_h} h old'))
assert eval(K2_LINE_NEW[len('K2_CFG = '):K2_LINE_NEW.index('   # literal')], {}) == cfg_new

TAG = 'v224fctl' if A.control else 'v224f%02d' % int(round(A.wf * 100))

SAVE = r'''# [eeshsaxena V224-F SAVE; not in the public notebook] Everything above is V224-exact (her 13 cells verbatim + our canary). Keep
# its output: submission.csv -> submission_V224.csv (sha256 recorded). The fleet stage below runs AFTER R7, r224 and the blend,
# so it cannot delay them; FN-IDS below rewrites submission.csv as an id placeholder, and the HF cell at the end writes the
# final submission.csv from submission_V224.csv (+ the fleet) in every case.
import os, json, time, hashlib, shutil
_V224F = {'tag': '__TAG__', 'w_f': __WF__, 'start_budget_h': __SB__, 'wall_h': __WALL__, 'min_start_min': __MINSTART__,
          'members': __NM__, 'control': __CTL__}
try:
    shutil.copyfile('/kaggle/working/submission.csv', '/kaggle/working/submission_V224.csv.tmp')
    os.replace('/kaggle/working/submission_V224.csv.tmp', '/kaggle/working/submission_V224.csv')
    _V224F['sha256'] = hashlib.sha256(open('/kaggle/working/submission_V224.csv', 'rb').read()).hexdigest()
    assert _V224F['sha256'] == hashlib.sha256(open('/kaggle/working/submission.csv', 'rb').read()).hexdigest()
except Exception as _e:
    _V224F['save_error'] = repr(_e)          # the HF cell then rebuilds the V224 output from her in-memory `sub`
_V224F['elapsed_h_at_save'] = round((time.time() - NB_T0) / 3600, 4)
_V224F['fleet_may_start'] = (not _V224F['control']) and _V224F['elapsed_h_at_save'] * 60 <= (_V224F['wall_h'] * 60 - _V224F['min_start_min'])
print('V224F_SAVED', json.dumps(_V224F), flush=True)
'''.replace('__TAG__', TAG).replace('__WF__', repr(float(A.wf))).replace('__SB__', repr(float(A.start_budget_h))) \
   .replace('__WALL__', repr(float(A.wall_h))).replace('__MINSTART__', repr(float(min_start))).replace('__NM__', str(N_MEMBERS)) \
   .replace('__CTL__', 'True' if A.control else 'False')

CLOCK = r'''# [eeshsaxena V224-F CLOCK] FN-IDS above set T0 = time.time() (its own start, hours into this kernel). The fleet wall below is
# min(/proc start of this process, T0) + WALL_H; pin T0 back to her NB_T0 (kernel start) so the wall stays at the kernel clock
# even if /proc were unreadable. R7 (the only other user of T0) finished long ago.
T0 = min(float(T0), float(NB_T0))
print('V224F_CLOCK', json.dumps({'T0': T0, 'NB_T0': NB_T0, 'age_h': round((time.time() - T0) / 3600, 4)}), flush=True)
'''

HF = r'''# [eeshsaxena V224-F HF BLEND; HFR7-RR HF cell semantics] submission.csv = per-label average-tie percentile rank
# (1 - W_F) * pct(V224) + W_F * pct(F), float32, not re-ranked. V224 = submission_V224.csv saved above (the V224-exact output,
# sha checked; her in-memory `sub` if that file is unusable). F = our K14 R fleet's raw-prob member mean
# (submission_fleetsolo.csv, written by the fleet stage only when it was clean with all members).
# Canaries: a column the V224 canary blanked (r224 not used -> Fracture 0.5) is set back to 0.5 after the blend. ANY fleet
# problem (not started by its time guard, failed, late, files missing / inconsistent) -> HF_FALLBACK: submission.csv =
# the V224 output with Synovitis = 0.5 for every study (one constant column, a visible leaderboard drop).
import os, json, time, hashlib, shutil
import numpy as np, pandas as pd
W_F = __WF__
HF_TAG = '__TAG__'
_W = '/kaggle/working'
_hf = {'tag': HF_TAG, 'w_f': W_F, 'members_expected': __NM__}
_hf_lab = list(labels)
_hf_cols = ['StudyInstanceUID'] + _hf_lab
assert 'Synovitis' in _hf_lab and 'Fracture' in _hf_lab
_hf_pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
_hf_blank = ['Fracture'] if 'r224->Fracture' in (globals().get('_canary') or {}).get('fired', []) else []
# ---- H = the V224 output -------------------------------------------------------------------------------------------------
_H = None
try:
    if _V224F.get('save_error'):
        raise RuntimeError('save failed: ' + _V224F['save_error'])
    if hashlib.sha256(open(f'{_W}/submission_V224.csv', 'rb').read()).hexdigest() != _V224F['sha256']:
        raise RuntimeError('submission_V224.csv changed since the save')
    _H = pd.read_csv(f'{_W}/submission_V224.csv', dtype={'StudyInstanceUID': str})
    _hf['h_source'] = 'submission_V224.csv'
except Exception as _e:
    _H = sub.copy()
    _hf['h_source'] = 'in-memory sub (%r)' % (_e,)
    print('V224F_H_FROM_MEMORY', repr(_e), flush=True)
assert _H.columns.tolist() == _hf_cols and _H['StudyInstanceUID'].tolist() == test_ids
assert np.isfinite(_H[_hf_lab].values.astype(np.float64)).all()
_H[_hf_lab] = _H[_hf_lab].astype(np.float32)
_hf['h_sha256'] = _V224F.get('sha256')
# ---- F and the blend -----------------------------------------------------------------------------------------------------
try:
    _rec = globals().get('_k2_receipt')
    if not (isinstance(_rec, dict) and _rec.get('clean') is True):
        raise RuntimeError('fleet stage not clean: %s' % ((_rec or {}).get('why') if isinstance(_rec, dict) else 'no receipt'))
    if _rec.get('models') != __NM__:
        raise RuntimeError('fleet ran %r members, expected __NM__' % (_rec.get('models'),))
    if not os.path.isfile(f'{_W}/submission_fleetsolo.csv'):
        raise RuntimeError('no submission_fleetsolo.csv')
    _F = pd.read_csv(f'{_W}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str})
    if sorted(_F['StudyInstanceUID']) != sorted(test_ids) or not set(_hf_lab) <= set(_F.columns) or not _F['StudyInstanceUID'].is_unique:
        raise RuntimeError('fleet ids / columns differ from the V224 output')
    _F = _F.set_index('StudyInstanceUID').loc[test_ids, _hf_lab]
    if not np.isfinite(_F.values).all():
        raise RuntimeError('fleet has non-finite values')
    _out = _H.copy()
    _rho = {}
    for _c in _hf_lab:
        _out[_c] = ((1.0 - W_F) * _hf_pct(_H[_c].values) + W_F * _hf_pct(_F[_c].values)).astype(np.float32)
        if len(_H) >= 3:
            _rho[_c] = round(float(pd.Series(_H[_c].values).corr(pd.Series(_F[_c].values), method='spearman')), 4)
    for _c in _hf_blank:
        _out[_c] = np.float32(0.5)                           # keep the V224 canary visible through the blend
    _v = _out[_hf_lab].values
    assert _out['StudyInstanceUID'].tolist() == test_ids and np.isfinite(_v).all() and ((_v >= 0) & (_v <= 1)).all()
    if _hf['h_source'] == 'submission_V224.csv':           # submission_H.csv = the H this blend used (h_check reads it)
        shutil.copyfile(f'{_W}/submission_V224.csv', f'{_W}/submission_H.csv')
    else:
        _H.to_csv(f'{_W}/submission_H.csv', index=False)
    _tmp = f'{_W}/submission.csv.tmp'; _out.to_csv(_tmp, index=False); os.replace(_tmp, f'{_W}/submission.csv')
    _hf.update(mode='HF', rows=len(_out), spearman_H_vs_F=_rho, blanked=list(_hf_blank),
               fleet={k: _rec.get(k) for k in ('models', 'n_keep', 'hard_stopped', 'stage_min', 'workers_s', 'kernel_age_h', 'coverage')})
    print('HF_BLEND_OK', json.dumps(_hf), flush=True)
except Exception as _e:
    _hf_blank = sorted(set(_hf_blank) | {'Synovitis'})
    _out = _H.copy()
    for _c in _hf_blank:
        _out[_c] = np.float32(0.5)
    _tmp = f'{_W}/submission.csv.tmp'; _out.to_csv(_tmp, index=False); os.replace(_tmp, f'{_W}/submission.csv')
    _hf.update(mode='H_only', why=repr(_e), rows=len(_out), blanked=_hf_blank)
    print('HF_FALLBACK', json.dumps(_hf), flush=True)
    print('CANARY_FIRED fleet not used: Synovitis set to 0.5 for all %d rows (%r)' % (len(_out), _e), flush=True)
_hf['submission_sha256'] = hashlib.sha256(open(f'{_W}/submission.csv', 'rb').read()).hexdigest()
_hf['total_seconds'] = round(time.time() - NB_T0, 1)
json.dump(_hf, open(f'{_W}/hf_receipt.json', 'w'), indent=1)
print('HF_COMPLETE', json.dumps(_hf), flush=True)
_v224f = {'use_f': True, 'w_f': W_F, 'tag': HF_TAG, 'mode': _hf['mode'], 'blanked': _hf['blanked'],
          'fired': (['fleet->Synovitis'] if _hf['mode'] != 'HF' else []) + (['r224->Fracture'] if 'Fracture' in _hf['blanked'] else []),
          'r224_used': bool((globals().get('_canary') or {}).get('r224_used')), 'save': _V224F, 'total_seconds': _hf['total_seconds']}
json.dump(_v224f, open(f'{_W}/v224f_receipt.json', 'w'), indent=1)
print('V224F_DONE', json.dumps(_v224f), flush=True)
'''.replace('__WF__', repr(float(A.wf))).replace('__TAG__', TAG).replace('__NM__', str(N_MEMBERS))

for t in (SAVE, CLOCK, HF, K2_LINE_NEW + FLEET_BODY):
    compile(t, '<cell>', 'exec')

# ---- assemble -------------------------------------------------------------------------------------------------------------
NEW = [('SAVE', SAVE), ('ENV-SNAP', ENV_SNAP), ('FN-IDS', FN_IDS), ('K12-MERGE', K12_MERGE), ('CLOCK', CLOCK),
       ('FLEET', K2_LINE_NEW + FLEET_BODY), ('ENV-RESTORE', ENV_RESTORE), ('HF', HF)]
nb = copy.deepcopy(V224)
nb['cells'] = [copy.deepcopy(c) for c in VC] + [code_cell(t) for _, t in NEW]
assert len(nb['cells']) == 22
# proof 1: dropping the appended cells gives the V224-exact file byte for byte (and, minus the canary, her file: asserted above)
back = copy.deepcopy(nb); back['cells'] = back['cells'][:14]
assert dumps(back).encode('utf-8') == RAW_V224
# proof 2: the fleet cells are the HFR7-RR cells (FLEET: all but its first line)
assert [src(c) for c in nb['cells'][15:18]] == [ENV_SNAP, FN_IDS, K12_MERGE] and src(nb['cells'][20]) == ENV_RESTORE
assert src(nb['cells'][19]).split('\n', 1)[1] == FLEET.split('\n', 1)[1]
name = 'rsna-sub-' + TAG
code_file = 'rsna_sub_%s.ipynb' % TAG
out_dir = os.path.join(A.out, 'kernel_' + TAG)
os.makedirs(out_dir, exist_ok=True)
nb_path = os.path.join(out_dir, code_file)
with open(nb_path, 'w', encoding='utf-8', newline='') as f:
    f.write(dumps(nb))
meta = copy.deepcopy(VMETA)
meta.update(id='eeshsaxena/' + name, title=name.replace('-', ' '), code_file=code_file, is_private=True,
            enable_gpu=True, enable_tpu=False, enable_internet=False)
meta['dataset_sources'] = VMETA['dataset_sources'] + FLEET_DS
with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)
with open(os.path.join(A.out, 'celldiff_%s.txt' % TAG), 'w', encoding='utf-8', newline='\n') as f:
    f.write(f'# {name}: V224-exact cells 0-13 byte-identical; appended cells 14-21 (full text). w_f {A.wf}, control {A.control}, '
            f'start budget {A.start_budget_h} h, wall {A.wall_h} h, MIN_START_MIN {min_start}\n')
    for i, (k, t) in enumerate(NEW):
        tagline = {'ENV-SNAP': 'HFR7-RR cell 7 verbatim', 'FN-IDS': 'HFR7-RR cell 8 verbatim', 'K12-MERGE': 'HFR7-RR cell 9 verbatim',
                   'ENV-RESTORE': 'HFR7-RR cell 11 verbatim',
                   'FLEET': 'HFR7-RR cell 10: line 1 (K2_CFG literal) changed, the rest verbatim; only line 1 shown'}.get(k, 'NEW')
        f.write(f'\n#### cell {14 + i} {k} ({tagline})\n')
        f.write(K2_LINE_NEW + '\n# was: ' + K2_LINE[:K2_LINE.index('   # literal')].replace(repr(cfg['MEMBERS']), '[...20 members...]')
                .replace(repr(cfg['TTA']), '{...AB...}') + '\n' if k == 'FLEET' else
                (t if tagline == 'NEW' else t[:200] + '\n# ... (verbatim, %d chars)\n' % len(t)))
print('BUILT', TAG, nb_path, 'sha256', sha(nb_path), 'cells', len(nb['cells']), 'w_f', A.wf, 'control', A.control,
      'WALL_H', A.wall_h, 'MIN_START_MIN', min_start, 'datasets', meta['dataset_sources'])
