#!/usr/bin/env python
# build_pl.py (2026-10-14, Claude subagent, R7-teacher workflow job 3): PER-LABEL blend builder on the V224 base.
#
# What it builds: a private kernel = a V224 base notebook (default: the SUBMITTED V224-exact kernel, session_2026-10-14/v224/
# kernel_v224, = varshitha's 13 cells verbatim + our CANARY cell) with ONE inserted cell (PL-H) right after her stage-3 cell, and,
# only when the map gives our fleet F a share and the base has a fleet stage, ONE appended cell (PL-F) at the very end.
# Every base cell stays byte-identical: the builder asserts that deleting the inserted cells gives back the base notebook FILE
# byte for byte, and that varshitha's 13 cells appear verbatim and in her order.
#
# Blend (rank space, per label c; "nested" = the HFR7-RR HF form, so f = .2 everywhere reproduces the HF cell exactly):
#   H_c     = s_c * RaptorMix_c + (1 - s_c) * R7_c        her stage-3 arithmetic and fallbacks, Raptor share s_c (hers: .5)
#             RaptorMix_c = .65 pct(r384) + .35 pct(r224)  (A224 unchanged; r224 failed -> r384 only, canary blanks Fracture)
#   final_c = (1 - f_c) * pct(H_c) + f_c * pct(F_c)       only if f_c > 0 (PL-F cell); else final_c = H_c
#   implied component weights: Raptor_mix (1 - f) s, R7 (1 - f)(1 - s), F f.
# PL-H self-check: her own weights must reproduce her `out` / `sub` bit for bit, else it raises (no submission, never a silent
# different blend). Columns whose s_c equals hers are left untouched (bit-identical to V224-exact).
#
# Map file (JSON). Resolution order: default -> groups (in file order) -> labels. Each entry gives {"s", "f"} or a triple
# {"R", "R7", "F"} (converted: f = F, s = R / (R + R7)). Values must sit on a .05 grid (pass --offgrid to allow others).
#   {"name": "oa65", "note": "...", "default": {"s": 0.5, "f": 0.0},
#    "groups": {"OA": {"labels": ["Medial OA", "Lateral OA", "PF OA"], "s": 0.65}},
#    "labels": {"Fracture": {"s": 0.4}}}
#
# Bases:
#   --base v224      (default) kernel_v224 (V224-exact, submitted 10-14 ref 57171955): no fleet stage, so every f must be 0.
#   --base v224flip  kernel_v224flip (V224 + R7 flip TTA): no fleet stage, every f must be 0.
#   --base-nb NB --base-meta META   any notebook that contains her 13 cells verbatim + a stage writing
#                    /kaggle/working/submission_fleetsolo.csv (e.g. the V224-F builds in session_2026-10-15/v224f): PL-H after
#                    her stage 3, PL-F appended last. Canaries re-applied by PL-F: r224 not used -> Fracture 0.5; F needed
#                    but missing -> Synovitis 0.5 (and H for every label).
#
# usage: python build_pl.py --map maps/oa65.json [--base v224|v224flip | --base-nb NB --base-meta META] [--out DIR] [--offgrid]
# writes OUT/kernel_pl_<name>/{rsna_sub_pl_<name>.ipynb, kernel-metadata.json, pl_map_resolved.json}; kernel id
# eeshsaxena/rsna-sub-pl-<name> (private, same datasets / docker pin / machine as the base). Never pushes, never submits.
import os, sys, json, copy, hashlib, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
V224 = os.path.join(ROOT, 'session_2026-10-14', 'v224')
HER_NB = os.path.join(V224, 'src', 'varshitha_v224.ipynb')
HER_SHA = '6406e92300011e48f0534ce74a91f5b7a0629b3771168ab7275f5584f884f36d'
BASES = {
    'v224': (os.path.join(V224, 'kernel_v224', 'rsna_sub_v224.ipynb'),
             '4259ec9774167bb280b794574b73e1ba73203e8a1995e97116388011c4226877',
             os.path.join(V224, 'kernel_v224', 'kernel-metadata.json')),
    'v224flip': (os.path.join(V224, 'kernel_v224flip', 'rsna_sub_v224flip.ipynb'),
                 '73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b',
                 os.path.join(V224, 'kernel_v224flip', 'kernel-metadata.json')),
}
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']
HER_S = 0.5            # her W_RAPTOR and every W_RAPTOR_C entry (OA .5): asserted against her cell text below
F_MAX = 0.6


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def dumps(nb):
    # varshitha's / build_v224's serialisation (compact separators, ensure_ascii False)
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s


def code_cell(text):
    return {'cell_type': 'code', 'metadata': {}, 'source': text, 'outputs': [], 'execution_count': None}


def on_grid(x):
    return abs(x * 20 - round(x * 20)) < 1e-9


def resolve(m, offgrid=False):
    """map JSON -> {label: {'s': float, 'f': float}} in LABELS order (validated)."""
    def conv(e, where):
        e = dict(e)
        if any(k in e for k in ('R', 'R7', 'F')):
            assert all(k in e for k in ('R', 'R7', 'F')) and not ('s' in e or 'f' in e), (where, 'give s/f OR the full R/R7/F triple', e)
            R, R7, F = float(e['R']), float(e['R7']), float(e['F'])
            assert min(R, R7, F) >= 0 and abs(R + R7 + F - 1.0) < 1e-9 and R + R7 > 0, (where, 'R + R7 + F must be 1', e)
            return {'s': round(R / (R + R7), 12), 'f': F}
        out = {}
        for k in ('s', 'f'):
            if k in e:
                out[k] = float(e[k])
        extra = set(e) - {'s', 'f', 'labels'}
        assert not extra, (where, 'unknown keys', extra)
        return out
    allowed = {'name', 'note', 'default', 'groups', 'labels', 'base_expect', 'probe'}
    assert set(m) <= allowed, ('unknown top-level keys', set(m) - allowed)
    cur = {c: {'s': HER_S, 'f': 0.0} for c in LABELS}
    d = conv(m.get('default', {}), 'default')
    for c in LABELS:
        cur[c].update(d)
    for g, e in (m.get('groups') or {}).items():
        labs = e.get('labels')
        assert labs and all(l in LABELS for l in labs), (g, 'group labels must be competition labels', labs)
        v = conv({k: e[k] for k in e if k != 'labels'}, 'group ' + g)
        for l in labs:
            cur[l].update(v)
    for l, e in (m.get('labels') or {}).items():
        assert l in LABELS, ('unknown label', l)
        cur[l].update(conv(e, 'label ' + l))
    for c, v in cur.items():
        assert 0.0 <= v['s'] <= 1.0 and 0.0 <= v['f'] <= F_MAX, (c, v)
        if not offgrid:
            assert on_grid(v['s']) and on_grid(v['f']), (c, v, 'values must sit on the .05 grid (or pass --offgrid)')
    return {c: {'s': cur[c]['s'], 'f': cur[c]['f']} for c in LABELS}


PL_H = r'''# [eeshsaxena PL-H; not in the public notebook] per-label Raptor share, map "__NAME__" (build_pl.py, map sha __MAPSHA__).
# Her stage-3 cell above is unchanged. This cell re-runs her per-finding arithmetic (her pct_on, her A224 Raptor mix, her
# fallbacks) with a per-label Raptor share s_c (R7 = 1 - s_c) and replaces ONLY the columns whose share differs from hers.
# Self-check first: her own weights must reproduce her `out` and `sub` bit for bit, else this cell raises (no submission).
# The fleet share f (if any) is applied by the PL-F cell at the end of the notebook (absent when every f is 0).
import json as _plj
PL_NAME = '__NAME__'
PL_MAP_SHA = '__MAPSHA__'
PL_MAP = __MAP__
assert list(PL_MAP) == list(labels), ('PL map labels differ from the submission columns', list(PL_MAP), list(labels))


def _pl_col(c, wr):
    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)
    if r224 is not None:
        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)
    b = wr * rr + (1.0 - wr) * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    return b


for _j, _c in enumerate(labels):
    _b = _pl_col(_c, W_RAPTOR_C.get(_c, W_RAPTOR))
    if not (np.array_equal(_b, out[:, _j]) and np.array_equal(_b.astype(np.float32), sub[_c].values)):
        raise RuntimeError('PL_SELF_CHECK_FAIL her stage 3 not reproduced for %s' % _c)
_pl = {'name': PL_NAME, 'map_sha': PL_MAP_SHA, 'map': PL_MAP, 'her_s': {c: W_RAPTOR_C.get(c, W_RAPTOR) for c in labels},
       'changed_s': {}, 'f_labels': {c: v['f'] for c, v in PL_MAP.items() if v['f'] > 0}, 'r224_used': r224 is not None,
       'R224_OK': bool(R224_OK), 'rows': n, 'raptor_failed': int(rap_fail.sum()), 'ours_failed': int(our_fail.sum())}
for _c in labels:
    _s = float(PL_MAP[_c]['s'])
    assert 0.0 <= _s <= 1.0, (_c, _s)
    if _s != W_RAPTOR_C.get(_c, W_RAPTOR):
        sub[_c] = _pl_col(_c, _s).astype(np.float32)
        _pl['changed_s'][_c] = _s
assert sub['StudyInstanceUID'].tolist() == test_ids and list(sub.columns) == cols
assert np.isfinite(sub[labels].values).all() and ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
sub.to_csv(WORK / 'submission_PLH.csv', index=False)      # H for the PL-F cell (before any canary)
(WORK / 'pl_receipt.json').write_text(_plj.dumps(_pl, indent=1))
print('PL_H_OK', _plj.dumps({k: _pl[k] for k in ('name', 'map_sha', 'changed_s', 'f_labels', 'r224_used', 'rows')}), flush=True)
'''

PL_F = r'''# [eeshsaxena PL-F; not in the public notebook] per-label fleet share, map "__NAME__" (build_pl.py, map sha __MAPSHA__).
# final_c = (1 - f_c) * pct(H_c) + f_c * pct(F_c) for f_c > 0, else H_c (f_c = .2 everywhere == the HFR7-RR HF cell).
# H = submission_PLH.csv (PL-H output, before any canary); F = submission_fleetsolo.csv (written only by a clean fleet stage).
# Canaries, re-applied last: F needed but missing / invalid -> H for every label and Synovitis = 0.5; r224 not used by the
# blend -> Fracture = 0.5. Overwrites submission.csv (whatever the base's own HF / canary cells wrote before it).
import os as _po, json as _pj, hashlib as _ph
import numpy as _pn, pandas as _pd
PL_F = __FMAP__
_PW = '/kaggle/working'
_pr = _pj.load(open(f'{_PW}/pl_receipt.json'))
assert _pr['name'] == '__NAME__' and _pr['map_sha'] == '__MAPSHA__', 'pl_receipt.json is not from this map'
assert {c: v['f'] for c, v in _pr['map'].items()} == PL_F, 'PL-F map differs from the PL-H map'
_H = _pd.read_csv(f'{_PW}/submission_PLH.csv', dtype={'StudyInstanceUID': str})
_cols = _H.columns.tolist(); _lab = _cols[1:]
assert _lab == list(PL_F) and _H['StudyInstanceUID'].is_unique and _pn.isfinite(_H[_lab].values).all()
_pct = lambda v: _pd.Series(_pn.asarray(v, _pn.float64)).rank(method='average', pct=True).values
_rf = {'name': '__NAME__', 'map_sha': '__MAPSHA__', 'f': PL_F, 'fired': []}
_fin = _H.copy()
try:
    if not _po.path.isfile(f'{_PW}/submission_fleetsolo.csv'):
        raise RuntimeError('fleet stage not clean (no submission_fleetsolo.csv)')
    _F = _pd.read_csv(f'{_PW}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str})
    if sorted(_F['StudyInstanceUID']) != sorted(_H['StudyInstanceUID']) or not set(_lab) <= set(_F.columns):
        raise RuntimeError('fleet ids / columns differ from H')
    _F = _F.set_index('StudyInstanceUID').loc[_H['StudyInstanceUID'], _lab]
    if not _pn.isfinite(_F.values).all():
        raise RuntimeError('fleet has non-finite values')
    for _c in _lab:
        if PL_F[_c] > 0:
            _fin[_c] = ((1.0 - PL_F[_c]) * _pct(_H[_c].values) + PL_F[_c] * _pct(_F[_c].values)).astype(_pn.float32)
    _rf['mode'] = 'PL_F'
except Exception as _e:
    _fin = _H.copy()
    _fin['Synovitis'] = _pn.float32(0.5)
    _rf.update(mode='H_only', why=repr(_e))
    _rf['fired'].append('F->Synovitis')
    print('PL_F_CANARY F missing (%r): H kept, Synovitis set to 0.5' % (_e,), flush=True)
if not _pr['r224_used']:
    _fin['Fracture'] = _pn.float32(0.5)
    _rf['fired'].append('r224->Fracture')
    print('PL_F_CANARY r224 not used: Fracture set to 0.5', flush=True)
assert _pn.isfinite(_fin[_lab].values).all() and ((_fin[_lab].values >= 0) & (_fin[_lab].values <= 1)).all()
_tmp = f'{_PW}/submission.csv.tmp'; _fin.to_csv(_tmp, index=False); _po.replace(_tmp, f'{_PW}/submission.csv')
_rf['submission_sha256'] = _ph.sha256(open(f'{_PW}/submission.csv', 'rb').read()).hexdigest()
_pj.dump(_rf, open(f'{_PW}/pl_f_receipt.json', 'w'), indent=1)
print('PL_F_DONE', _pj.dumps(_rf), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--base', choices=sorted(BASES), default='v224')
    ap.add_argument('--base-nb', default=None)
    ap.add_argument('--base-meta', default=None)
    ap.add_argument('--out', default=HERE)
    ap.add_argument('--offgrid', action='store_true')
    A = ap.parse_args()
    m = json.load(open(A.map, encoding='utf-8'))
    name = m.get('name') or os.path.splitext(os.path.basename(A.map))[0]
    assert name.replace('-', '').isalnum() and name == name.lower() and len(name) <= 24, ('map name: lowercase alnum/-, <= 24', name)
    pl = resolve(m, A.offgrid)
    canon = json.dumps(pl, sort_keys=True, separators=(',', ':'))
    mapsha = hashlib.sha256(canon.encode('utf-8')).hexdigest()[:16]
    use_f = any(v['f'] > 0 for v in pl.values())

    # ---- base --------------------------------------------------------------------------------------------------------
    if A.base_nb:
        assert A.base_meta, '--base-nb needs --base-meta'
        base_path, base_sha, meta_path, base_tag = A.base_nb, sha(A.base_nb), A.base_meta, 'custom:' + os.path.basename(A.base_nb)
    else:
        base_path, base_sha, meta_path = BASES[A.base]
        base_tag = A.base
        assert sha(base_path) == base_sha, (A.base, 'base notebook changed', sha(base_path))
    raw = open(base_path, 'rb').read()
    base = json.loads(raw.decode('utf-8'))
    custom_ser = dumps(base).encode('utf-8') != raw
    if not A.base_nb:
        assert not custom_ser, 'base serialisation not reproduced'
    assert sha(HER_NB) == HER_SHA, 'varshitha notebook copy changed'
    her = json.load(open(HER_NB, encoding='utf-8'))['cells']
    st3 = src(her[12])
    assert st3.startswith('# Stage 3: per-finding rank blend') and 'W_RAPTOR, W_OURS = 0.5, 0.5' in st3
    assert "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}" in st3 and 'A224 = 0.35' in st3
    for needle in ('def pct_on(v, ok):', 'out = np.zeros((n, len(labels)))', 'rap_fail = (raw.values == 0.5).all(axis=1)',
                   'our_fail = np.array(', 'sub = pd.DataFrame(out.astype(np.float32), columns=labels)', "r224 = None"):
        assert needle in st3, needle
    bc = base['cells']
    # her 13 cells verbatim and in her order (as a subsequence of the base)
    k = 0
    for c in bc:
        if k < 13 and c == her[k]:
            k += 1
    assert k == 13, ('base does not contain her 13 cells verbatim in order', k)
    i3 = [i for i, c in enumerate(bc) if c == her[12]]
    assert len(i3) == 1, 'her stage-3 cell must occur exactly once'
    i3 = i3[0]
    has_fleet = any('submission_fleetsolo.csv' in src(c) for c in bc)
    if use_f:
        assert has_fleet, ('this map gives F a share, but the base has no fleet stage (no submission_fleetsolo.csv); '
                           'use --base-nb with a V224-F notebook, or set every f to 0')
    canary_after = i3 + 1 < len(bc) and src(bc[i3 + 1]).startswith('# [eeshsaxena CANARY')

    # ---- assemble ----------------------------------------------------------------------------------------------------
    fill = lambda t: t.replace('__NAME__', name).replace('__MAPSHA__', mapsha)
    plh = fill(PL_H).replace('__MAP__', repr(pl))
    plf = fill(PL_F).replace('__FMAP__', repr({c: v['f'] for c, v in pl.items()}))
    for t in (plh, plf):
        compile(t, '<cell>', 'exec')
    cells = [copy.deepcopy(c) for c in bc]
    cells.insert(i3 + 1, code_cell(plh))
    ins = {i3 + 1}
    if use_f:
        cells.append(code_cell(plf))
        ins.add(len(cells) - 1)
    nb = copy.deepcopy(base)
    nb['cells'] = cells
    back = copy.deepcopy(nb)
    back['cells'] = [c for i, c in enumerate(cells) if i not in ins]
    if custom_ser:
        assert back == base, 'inserted cells are not the only change'
    else:
        assert dumps(back).encode('utf-8') == raw, 'deleting the inserted cells does not give back the base file byte for byte'
    kname = 'rsna-sub-pl-' + name
    code_file = 'rsna_sub_pl_%s.ipynb' % name.replace('-', '_')
    out_dir = os.path.join(A.out, 'kernel_pl_' + name)
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, code_file)
    with open(nb_path, 'w', encoding='utf-8', newline='') as f:
        f.write(dumps(nb))
    meta = json.load(open(meta_path, encoding='utf-8'))
    meta = {k_: v for k_, v in meta.items() if k_ not in ('id_no', 'keywords')}
    meta.update(id='eeshsaxena/' + kname, title=kname.replace('-', ' '), code_file=code_file, is_private=True,
                enable_gpu=True, enable_tpu=False, enable_internet=False)
    with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    implied = {c: {'Raptor_mix': round((1 - v['f']) * v['s'], 4), 'R7': round((1 - v['f']) * (1 - v['s']), 4), 'F': v['f']}
               for c, v in pl.items()}
    rec = {'name': name, 'map_file': os.path.abspath(A.map), 'map_sha': mapsha, 'resolved': pl, 'implied_weights': implied,
           'changed_vs_v224': {c: v for c, v in pl.items() if v['s'] != HER_S or v['f'] > 0}, 'base': base_tag,
           'base_sha256': base_sha, 'base_has_fleet': has_fleet, 'canary_cell_after_PLH': canary_after, 'pl_f_cell': use_f,
           'kernel_id': meta['id'], 'notebook_sha256': sha(nb_path), 'cells': len(cells), 'inserted_at': sorted(ins),
           'datasets': meta['dataset_sources']}
    with open(os.path.join(out_dir, 'pl_map_resolved.json'), 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=1)
    print('BUILT', kname, nb_path, 'sha256', rec['notebook_sha256'], '| cells', len(cells), 'inserted', sorted(ins),
          '| base', base_tag, '| map sha', mapsha, '| changed', json.dumps(rec['changed_vs_v224']))


if __name__ == '__main__':
    main()
