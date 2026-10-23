#!/usr/bin/env python3
"""make_k2_kernel.py: build a LOCAL kernel folder = the 0.943 parent (repro_0946, 31 cells, sha-pinned) + the K2 cells.
Patched successor of round3/fleet2 make_parent_fleet_kernel.py. Nothing is pushed by this script.

Fleet-only (K2):   parent cells 0-30 + [31] k2_fleet_cell.py
Combined (--combined):            + [32] k2_crop_pre_cell.py + [33] crop_fleet_cell.py (K1 bytes) + [34] k2_crop_post_cell.py

Every knob is written into the cell as a literal (pushed kernels get no env vars):
  K2_CFG  = W_MAIN, W_VARIANTS, WALL_H, MIN_START_MIN, MAIN_MODE (blend|solo|strict) [, MEMBERS]   (fleet stage)
  K2C_CFG = C_W, CF_WC, CF_WF, MIN_START_MIN, WALL_H            (crop stage, combined only)

usage:
  python make_k2_kernel.py --out DIR --slug rsna-sub-k2-fleet [--w_main 0.30 --w_variants 0.20,0.40]
  python make_k2_kernel.py --out DIR --slug rsna-sub-k2-fleetcrop --combined [--c_w 0.15 --cf_wc 0.15 --cf_wf 0.25]
"""
import argparse, copy, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
PARENT_DIR = f'{SP}/kernels/repro_0946'
PARENT_SHA256 = '35547f8a28ed09fc68acefa6d23f12fa0bfd6a959f54fdadfdfc3dc3c32768f8'   # == boxmimic pinned parent
K1_NB = f'{SP}/cropfleet/kernel/rsna-sub-cropfleet/rsna_sub_cropfleet.ipynb'        # the committed + verified K1
K1_CELL_PY = f'{SP}/cropfleet/crop_fleet_cell.py'
FLEET_DS = 'eeshsaxena/rsna-knee-cnxt-fleet'
CROP_DS = 'eeshsaxena/rsna-knee-cropaug-members'
MARKERS = ('K2 FLEET STAGE', 'FLEET LEG', 'CROP FLEET CELL', 'K2 CROP STAGE', 'LIGHT DEPLOY BLEND CELL')


def sha16(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def code_cell(src, name):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {'k2': name}, 'outputs': [],
            'source': src.splitlines(keepends=True)}


def fnum(x):
    return float(f'{float(x):.4f}')


def build(a):
    w_vars = [fnum(x) for x in str(a.w_variants).split(',') if x.strip()]
    k2_cfg = {'W_MAIN': fnum(a.w_main), 'W_VARIANTS': w_vars, 'WALL_H': fnum(a.wall_h),
              'MIN_START_MIN': fnum(a.min_start_min), 'MAIN_MODE': getattr(a, 'main_mode', 'blend')}
    members = [m.strip() for m in str(getattr(a, 'members', None) or '').split(',') if m.strip()]
    if members:                                                          # K3b: only when given (old cells unchanged)
        assert len(set(members)) == len(members), f'duplicate --members {members}'
        assert all(m.endswith('.pt') and '/' not in m and chr(92) not in m for m in members), \
            f'--members takes checkpoint FILE names as in fleet_manifest.json: {members}'
        k2_cfg['MEMBERS'] = members
    assert k2_cfg['MAIN_MODE'] in ('blend', 'solo', 'strict'), k2_cfg['MAIN_MODE']
    assert not (a.combined and k2_cfg['MAIN_MODE'] == 'strict'), 'strict mode is fleet-only (crop cells not reviewed for it)'
    ws = [k2_cfg['W_MAIN']] + w_vars
    assert all(0.0 < w < 1.0 for w in ws), ws
    names = [f'submission_w{int(round(w * 100)):02d}.csv' for w in ws]
    assert len(set(names)) == len(names), f'two weights map to one file name: {names}'
    assert 0 < k2_cfg['WALL_H'] <= 8.75, 'wall must leave room inside the 9 h limit'
    pbytes = open(f'{PARENT_DIR}/repro0946.ipynb', 'rb').read()
    assert hashlib.sha256(pbytes).hexdigest() == PARENT_SHA256, 'parent notebook is not the pinned repro_0946'
    meta = json.load(open(f'{PARENT_DIR}/kernel-metadata.json', encoding='utf-8'))
    nb = json.loads(pbytes.decode('utf-8'))
    assert len(nb['cells']) == 31, len(nb['cells'])
    for c in nb['cells']:
        s = ''.join(c.get('source', ''))
        assert not any(m in s for m in MARKERS), 'parent already carries a blend cell'
    cells = []
    fleet_src = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    hdr = (f'K2_CFG = {k2_cfg!r}   # literal, written by k2/make_k2_kernel.py (pushed kernels get no env vars)\n')
    cells.append(('k2_fleet_stage', hdr + fleet_src))
    k2c_cfg = None
    if a.combined:
        k2c_cfg = {'C_W': fnum(a.c_w), 'CF_WC': fnum(a.cf_wc), 'CF_WF': fnum(a.cf_wf),
                   'MIN_START_MIN': fnum(a.crop_min_start_min), 'WALL_H': k2_cfg['WALL_H']}
        assert 0 < k2c_cfg['C_W'] < 1 and k2c_cfg['CF_WC'] > 0 and k2c_cfg['CF_WF'] > 0
        assert k2c_cfg['CF_WC'] + k2c_cfg['CF_WF'] < 1
        cnames = ['submission_cropsolo.csv', f'submission_c{int(round(k2c_cfg["C_W"] * 100)):02d}.csv', 'submission_cf.csv']
        assert not set(cnames) & set(names + ['submission.csv', 'submission_fleetsolo.csv'])
        pre = open(f'{HERE}/cells/k2_crop_pre_cell.py', encoding='utf-8').read()
        pre_lines = pre.splitlines(keepends=True)
        # the sandbox assignment must stay the first executable statement after the imports (C1)
        first = [l for l in pre_lines if l.strip() and not l.lstrip().startswith('#')]
        assert first[0].startswith('import ') and first[1].startswith("os.environ['CROP_FLEET_WORK']"), first[:2]
        k1 = json.load(open(K1_NB, encoding='utf-8'))
        crop_src = ''.join(k1['cells'][-1]['source'])
        crop_py = open(K1_CELL_PY, encoding='utf-8').read().rstrip('\n') + '\n'
        assert crop_src == crop_py, 'K1 notebook last cell != cropfleet/crop_fleet_cell.py: rebuild K1 or pin one'
        assert crop_src.rstrip().endswith('_cf_receipt = _cf_entry()')
        assert "CF_WORK = os.environ.get('CROP_FLEET_WORK', '/kaggle/working')" in crop_src
        assert "CF.update(json.loads(os.environ.get('CROP_FLEET_OVERRIDES', '{}') or '{}'))" in crop_src
        if a.crop_cell_sha:
            assert sha16(crop_src) == a.crop_cell_sha, (sha16(crop_src), a.crop_cell_sha)
        cells.append(('k2_crop_pre', f'K2C_CFG = {k2c_cfg!r}   # literal, written by k2/make_k2_kernel.py\n' + pre))
        cells.append(('k1_crop_fleet_cell', crop_src))
        cells.append(('k2_crop_post', open(f'{HERE}/cells/k2_crop_post_cell.py', encoding='utf-8').read()))
    for name, src in cells:
        compile(src, name, 'exec')
        bad = [l for l in src.splitlines() if '/kaggle/' in l and 'kaggle/input' not in l and 'kaggle/working' not in l
               and not l.lstrip().startswith('#')]
        assert not bad, (name, bad[:3])
    new = copy.deepcopy(nb)
    for name, src in cells:
        new['cells'].append(code_cell(src, name))
    out = a.out
    os.makedirs(out, exist_ok=True)
    code_file = a.slug.replace('-', '_') + '.ipynb'
    with open(f'{out}/{code_file}', 'w', encoding='utf-8') as fh:
        json.dump(new, fh, indent=1, ensure_ascii=False)
    m2 = dict(meta)
    m2.update(id=f'eeshsaxena/{a.slug}', title=a.slug.replace('-', ' '), code_file=code_file, is_private=True)
    ds = list(m2.get('dataset_sources', [])) + [FLEET_DS] + ([CROP_DS] if a.combined else [])
    m2['dataset_sources'] = list(dict.fromkeys(ds))
    assert m2.get('enable_gpu') and not m2.get('enable_internet') and m2.get('machine_shape') == 'NvidiaTeslaT4'
    with open(f'{out}/kernel-metadata.json', 'w', encoding='utf-8') as fh:
        json.dump(m2, fh, indent=2)
    # read back
    chk = json.load(open(f'{out}/{code_file}', encoding='utf-8'))
    assert len(chk['cells']) == 31 + len(cells)
    for i in range(31):
        assert chk['cells'][i] == nb['cells'][i], f'parent cell {i} changed'
    for k, (name, src) in enumerate(cells):
        assert ''.join(chk['cells'][31 + k]['source']) == src
    files = ['submission.csv'] + sorted(set(names)) + ['submission_fleetsolo.csv']
    if a.combined:
        files += cnames
    info = {'out': out, 'slug': a.slug, 'id': m2['id'], 'code_file': code_file, 'cells': len(chk['cells']),
            'appended': [{'cell': 31 + k, 'name': n, 'sha16': sha16(s), 'lines': s.count('\n')} for k, (n, s) in enumerate(cells)],
            'K2_CFG': k2_cfg, 'K2C_CFG': k2c_cfg, 'files': files, 'datasets_added': [d for d in m2['dataset_sources'] if d.startswith('eeshsaxena/')],
            'parent_sha256': PARENT_SHA256}
    print(json.dumps(info, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--slug', required=True)
    ap.add_argument('--w_main', type=float, default=0.30)
    ap.add_argument('--w_variants', default='0.20,0.40')
    ap.add_argument('--wall_h', type=float, default=8.6)
    ap.add_argument('--min_start_min', type=float, default=20.0)
    ap.add_argument('--main_mode', choices=('blend', 'solo', 'strict'), default='blend')
    ap.add_argument('--members', default=None, help='K3b: comma list of manifest checkpoint file names (default: all)')
    ap.add_argument('--combined', action='store_true')
    ap.add_argument('--c_w', type=float, default=0.15)
    ap.add_argument('--cf_wc', type=float, default=0.15)
    ap.add_argument('--cf_wf', type=float, default=0.25)
    ap.add_argument('--crop_min_start_min', type=float, default=10.0)
    ap.add_argument('--crop_cell_sha', default=None, help='pin the K1 crop cell sha256[:16] (049033d6002cf9fc on 10-04)')
    return build(ap.parse_args(argv))


if __name__ == '__main__':
    main()
