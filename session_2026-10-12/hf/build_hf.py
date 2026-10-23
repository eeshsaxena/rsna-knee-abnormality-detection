#!/usr/bin/env python
# build_hf.py (2026-10-12, Claude): HF kernels = HV (heliosli Raptor + S6 notebook, verbatim cells) + our fleet stage F
# inserted after the Raptor stage and before the S6 stage, + a final HF blend cell (PREREG_10-12.md).
#   cells: HV 0-4 (md, checks, raptor writefile, Raptor stage) | ENV-SNAP | FN-IDS (FN11 cell 1, verbatim) |
#          K2 fleet (FN11 cell 2, K2_CFG line replaced) | ENV-RESTORE | HV 5-7 (S6 assets, S6 run, H blend) | HF blend
# The H blend (HV cell 7) writes submission.csv = H exactly as in HV. The HF cell then reads it and the fleet's
# submission_fleetsolo.csv and writes submission.csv = per-label average-tie percentile rank (1 - wF) * H + wF * F.
# No F (fleet stage not clean / file missing / ids differ) -> submission.csv stays H, HF_FALLBACK printed.
# usage: python build_hf.py --wf 0.20 --members D [--tag hf20] [--out DIR]
import os, sys, json, copy, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
HV_NB = os.path.join(HERE, '..', 'hv', 'kernel', 'rsna_sub_hv.ipynb')
HV_SHA = 'f69144dce209c40f6a0b68b62afc9a738c5a21c1aae2d70a7e0b938b43a6061c'   # HV v3: heliosli V2 + compact-v1 slug patch (3 code strings)
FN11_NB = ('C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/'
           'scratchpad/k11/build_fn11/kernel/rsna-sub-k11fn/rsna_sub_k11fn.ipynb')
FN11_SHA = 'b19a4320c340fc11'          # prefix recorded in FN11_RESULT.md sec 3
K14_NB = ('C:/Users/eeshs/AppData/Local/Temp/claude/C--Users-eeshs-Downloads/405896d1-8e93-4c82-a842-70e7cf895bfa/'
          'scratchpad/k14/build_fn14_R/kernel/rsna-sub-k14fn/rsna_sub_k14fn.ipynb')   # K14 R build (K12-MERGE k11w + k14w)
SETS = {
    'DT': [f'f{F}_s1043f75oai_raw_ttr224-320x24_cv{c}_best.pt' for F, c in enumerate(['8876', '8934', '8921', '8817', '8918'])],
    'D': [f'f{F}_s1043f75_raw_ttr224-320x24_cv{c}_best.pt' for F, c in enumerate(['8837', '8909', '8909', '8769', '8928'])],
    'C': [f'f{F}_s1044t288_raw_ttr288-320x24_cv{c}_best.pt' for F, c in enumerate(['8909', '8952', '8935', '8821', '8955'])],
    'E': [f'f{F}_s1045t288f75_raw_ttr288-320x24_cv{c}_best.pt' for F, c in enumerate(['8893', '8922', '8916', '8788', '8913'])],
    'E2': [f'f{F}_s1046t288f75_raw_ttr288-320x24_cv{c}_best.pt' for F, c in enumerate(['8893', '8913', '8934', '8803', '8941'])],
}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
src = lambda c: ''.join(c['source'])


def code(s):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s}


ENV_SNAP = '''# === HF ENV-SNAP (build_hf.py): remember the environment before our fleet stage, restored after it, so the S6
# stage starts in exactly the environment it had in HV (FN-IDS sets OMP / MKL / HF-offline defaults for our workers).
import os as _hf_os
_HF_ENV0 = dict(_hf_os.environ)
'''

ENV_RESTORE = '''# === HF ENV-RESTORE (build_hf.py): drop every variable our FN-IDS / fleet cells added or changed; keep the rest.
for _k in list(_hf_os.environ):
    if _k not in _HF_ENV0:
        del _hf_os.environ[_k]
for _k, _v in _HF_ENV0.items():
    if _hf_os.environ.get(_k) != _v:
        _hf_os.environ[_k] = _v
print('HF_ENV_RESTORED', flush=True)
'''

HF_CELL = r'''# === HF BLEND (build_hf.py, PREREG_10-12.md): submission.csv = per-label percentile rank (1 - W_F) * H + W_F * F.
# H = the HV stage-3 output (Raptor .5 + S6 .5, written just above, unchanged); F = our fleet's raw-prob member mean
# (submission_fleetsolo.csv, written by the K2 fleet stage only when that stage was clean). Any problem with F ->
# submission.csv stays H and HF_FALLBACK is printed (never a silent partial blend).
import os, json, time, hashlib
import numpy as np, pandas as pd
W_F = __WF__
HF_TAG = '__TAG__'
_W = '/kaggle/working'
_hf = {'tag': HF_TAG, 'w_f': W_F}
H = pd.read_csv(f'{_W}/submission.csv', dtype={'StudyInstanceUID': str})
_cols = H.columns.tolist(); _lab = _cols[1:]
assert len(_lab) == 12 and H['StudyInstanceUID'].is_unique and np.isfinite(H[_lab].values).all()
_why = None
try:
    if not (isinstance(globals().get('_k2_receipt'), dict) and str(_k2_receipt.get('status', '')).startswith('ok')
            or os.path.isfile(f'{_W}/submission_fleetsolo.csv')):
        raise RuntimeError('fleet stage not clean (no submission_fleetsolo.csv)')
    F = pd.read_csv(f'{_W}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str})
    if sorted(F['StudyInstanceUID']) != sorted(H['StudyInstanceUID']) or not set(_lab) <= set(F.columns):
        raise RuntimeError('fleet ids / columns differ from H')
    F = F.set_index('StudyInstanceUID').loc[H['StudyInstanceUID'], _lab]
    if not np.isfinite(F.values).all():
        raise RuntimeError('fleet has non-finite values')
    pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
    out = H.copy()
    rho = {}
    for c in _lab:
        out[c] = ((1.0 - W_F) * pct(H[c].values) + W_F * pct(F[c].values)).astype(np.float32)
        if len(H) >= 3:
            rho[c] = round(float(pd.Series(H[c].values).corr(pd.Series(F[c].values), method='spearman')), 4)
    assert np.isfinite(out[_lab].values).all() and ((out[_lab].values >= 0) & (out[_lab].values <= 1)).all()
    os.replace(f'{_W}/submission.csv', f'{_W}/submission_H.csv')
    _tmp = f'{_W}/submission.csv.tmp'; out.to_csv(_tmp, index=False); os.replace(_tmp, f'{_W}/submission.csv')
    _hf.update(mode='HF', rows=len(out), spearman_H_vs_F=rho)
    print('HF_BLEND_OK', json.dumps(_hf), flush=True)
except Exception as e:
    _why = repr(e)
    _hf.update(mode='H_only', why=_why)
    print('HF_FALLBACK', json.dumps(_hf), flush=True)
_hf['submission_sha256'] = hashlib.sha256(open(f'{_W}/submission.csv', 'rb').read()).hexdigest()
_hf['total_seconds'] = round(time.time() - NB_T0, 1)
json.dump(_hf, open(f'{_W}/hf_receipt.json', 'w'), indent=1)
print('HF_COMPLETE', json.dumps(_hf), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wf', type=float, required=True)
    ap.add_argument('--members', default='D')            # e.g. D or D,E ; DT / R sets need --fleet k14
    ap.add_argument('--fleet', default='k11w', choices=['k11w', 'k14'])   # k14: FN-IDS + K12-MERGE + merged fleet cell of the K14 R build
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default=None)
    A = ap.parse_args()
    assert 0.0 < A.wf < 1.0
    tag = A.tag or f'hf{int(round(A.wf * 100)):02d}'
    out = A.out or os.path.join(HERE, 'build', tag)
    assert sha(HV_NB) == HV_SHA, 'HV notebook changed'
    hv = json.load(open(HV_NB, encoding='utf-8'))
    fn = json.load(open(FN11_NB, encoding='utf-8'))
    assert sha(FN11_NB).startswith(FN11_SHA), 'FN11 notebook changed'
    hc = hv['cells']; assert len(hc) == 8
    assert 'NB_T0 = time.time()' in src(hc[4]) and 'T0 = NB_T0' in src(hc[5]) and 'W_RAPTOR, W_OURS = 0.5, 0.5' in src(hc[7])
    merge_cell = None
    if A.fleet == 'k14':
        k14 = json.load(open(K14_NB, encoding='utf-8'))
        ids_cell, merge_cell, fleet_cell = k14['cells'][1], k14['cells'][2], k14['cells'][3]
        assert src(merge_cell).startswith('K12M_CFG = {') and "'add_ds': 'rsna-knee-cnxt-fleet-k14w'" in src(merge_cell)
    else:
        ids_cell, fleet_cell = fn['cells'][1], fn['cells'][2]
    assert src(ids_cell).startswith('# === FN-IDS') and src(fleet_cell).startswith('K2_CFG = {')
    SETS['R'] = SETS['C'] + SETS['DT'] + SETS['E'] + SETS['E2']
    members = [m for s in A.members.split(',') for m in SETS[s]]
    assert A.fleet == 'k14' or not any('f75oai' in m for m in members), 'DT members need --fleet k14'
    cfg = {'W_MAIN': 0.5, 'W_VARIANTS': [0.4, 0.6], 'WALL_H': 8.6, 'MIN_START_MIN': 20.0, 'MAIN_MODE': 'blend',
           'MEMBERS': members, 'TTA': {m: 'AB' for m in members}, 'RHO_MIN_N': 1000000000}
    fs = src(fleet_cell); nl = fs.index('\n')
    new_fleet = (f'K2_CFG = {cfg!r}   # literal, written by hf/build_hf.py: FN11 fleet cell, MAIN_MODE blend (the HF cell '
                 f'reads submission_fleetsolo.csv; a failed stage writes none), members {A.members}' + fs[nl:])
    md = copy.deepcopy(hc[0])
    md['source'] = (f'# rsna-sub-{tag}: heliosli Raptor + S6 blend (H, unchanged) + our ConvNeXt-T fleet F ({A.members}), '
                    f'per-label rank blend (1 - {A.wf:.2f}) H + {A.wf:.2f} F\n\n'
                    'H is the public notebook heliosli/blend-mine-in-private-lb-luck-to-all-upvoters (Apache 2.0), cells kept '
                    'verbatim; its credits and the OAI acknowledgement follow below. F is our own fleet (dataset '
                    'eeshsaxena/rsna-knee-cnxt-fleet-k11w, trained on RSNA data only).\n\n' + src(hc[0]))
    mid = [copy.deepcopy(ids_cell)] + ([copy.deepcopy(merge_cell)] if merge_cell else []) + [code(new_fleet)]
    cells = [md, hc[1], hc[2], hc[3], hc[4], code(ENV_SNAP)] + mid + [code(ENV_RESTORE),
             hc[5], hc[6], hc[7], code(HF_CELL.replace('__WF__', repr(float(A.wf))).replace('__TAG__', tag))]
    nb = copy.deepcopy(hv); nb['cells'] = [copy.deepcopy(c) for c in cells]
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f'rsna_sub_{tag}.ipynb')
    json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    meta = json.load(open(os.path.join(HERE, '..', 'hv', 'kernel', 'kernel-metadata.json'), encoding='utf-8'))
    meta.update(id=f'eeshsaxena/rsna-sub-{tag}', title=f'rsna sub {tag}', code_file=f'rsna_sub_{tag}.ipynb')
    meta['dataset_sources'] = meta['dataset_sources'] + ['eeshsaxena/rsna-knee-cnxt-fleet-k11w'] +         (['eeshsaxena/rsna-knee-cnxt-fleet-k14w'] if A.fleet == 'k14' else [])
    json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
    print(f'BUILT {p} sha256 {sha(p)} | cells {len(cells)} | members {len(members)} ({A.members}) | W_F {A.wf}')


if __name__ == '__main__':
    main()
