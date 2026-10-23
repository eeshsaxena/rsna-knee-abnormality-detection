#!/usr/bin/env python3
"""build_k3.py: both K3 kernels from ONE upload folder (= one new version of eeshsaxena/rsna-knee-cnxt-fleet), one
command. Wraps build_k2.py (K3 kit); local only, nothing is pushed, versioned or submitted.

  python build_k3.py [--m2 <TTR ckpt>] [--m1 <family-1 ckpt>] [--out k3/build] [--overwrite]

  K3-main    eeshsaxena/rsna-sub-k3-fleet    MAIN_MODE strict, W_MAIN .25 (+ w20 / w40 / fleetsolo), MEMBERS [M1, M2]
  K3-ttrsolo eeshsaxena/rsna-sub-k3-ttrsolo  MAIN_MODE solo,   W_MAIN .25,                          MEMBERS [M2]

Defaults: M1 = k3/ckpt/f0_s42_raw_cnxt320x24_cv8810_best.pt (the K2 member); M2 = the ONE f0_s1042_*_best.pt in k3/ckpt
(refuses 0 or 2+ so a stale TTR file can never ship by accident). Both kernels name their members explicitly, so a
commit that mounts any other dataset version (e.g. pushed before the new version is ready) fails loud instead of
silently running the old member set. After the build: both kernels' dataset_sources == K2's, cell 31 of both notebooks
== cells/k2_fleet_cell.py + its own K2_CFG line, parent cells 0-30 untouched (make_k2_kernel read-back).
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
SP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
M1_DEFAULT = f'{HERE}/ckpt/f0_s42_raw_cnxt320x24_cv8810_best.pt'
K2_META = f'{SP}/k2/build/kernel/rsna-sub-k2-fleet/kernel-metadata.json'      # the proven K2 kernel's sources
MAIN, SOLO = 'rsna-sub-k3-fleet', 'rsna-sub-k3-ttrsolo'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--m1', default=M1_DEFAULT)
    ap.add_argument('--m2', default=None, help='TTR member (default: the one f0_s1042_*_best.pt in k3/ckpt)')
    ap.add_argument('--store', default=f'{HERE}/_real/store')
    ap.add_argument('--out', default=f'{HERE}/build')
    ap.add_argument('--w_main', default='0.25')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    m2 = a.m2
    if not m2:
        c = sorted(glob.glob(f'{HERE}/ckpt/f0_s1042_*_best.pt'))
        assert len(c) == 1, f'k3/ckpt must hold exactly ONE f0_s1042_*_best.pt (the final TTR file), found {c}'
        m2 = c[0]
    assert os.path.isfile(a.m1) and os.path.isfile(m2), (a.m1, m2)
    n1, n2 = os.path.basename(a.m1), os.path.basename(m2)
    assert n1.startswith('f0_s42_') and n2.startswith('f0_s1042_'), (n1, n2)
    import build_k2 as bk
    common = ['--out', a.out, '--w_main', a.w_main]
    bk.main(['--ckpt', a.m1, '--ckpt', m2, '--store', a.store, '--slug', MAIN, '--main_mode', 'strict',
             '--members', f'{n1},{n2}'] + common + (['--overwrite'] if a.overwrite else []))
    summ = bk.main(['--kernel_only', '--slug', SOLO, '--main_mode', 'solo', '--members', n2] + common)
    # ---- post checks
    ks = {k['slug']: k for k in summ['kernels']}
    assert set(ks) == {MAIN, SOLO}, sorted(ks)
    assert ks[MAIN]['K2_CFG']['MAIN_MODE'] == 'strict' and ks[MAIN]['K2_CFG']['MEMBERS'] == [n1, n2]
    assert ks[SOLO]['K2_CFG']['MAIN_MODE'] == 'solo' and ks[SOLO]['K2_CFG']['MEMBERS'] == [n2]
    assert summ['upload'].get('manifest') == [n1, n2], summ['upload'].get('manifest')
    assert 'fleet_model_ttr.py' in summ['upload']['files'] and len(summ['upload']['files']) == 13, summ['upload']['files']
    cell = open(f'{HERE}/cells/k2_fleet_cell.py', encoding='utf-8').read()
    srcs = {}
    for slug, k in ks.items():
        nb = json.load(open(f'{k["out"]}/{k["code_file"]}', encoding='utf-8'))
        src = ''.join(nb['cells'][31]['source'])
        hdr, body = src.split('\n', 1)
        assert hdr.startswith('K2_CFG = ') and body == cell, f'{slug}: cell 31 != cells/k2_fleet_cell.py + K2_CFG line'
        srcs[slug] = body
        meta = json.load(open(f'{k["out"]}/kernel-metadata.json', encoding='utf-8'))
        if os.path.isfile(K2_META):
            k2m = json.load(open(K2_META, encoding='utf-8'))
            for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources', 'machine_shape',
                        'enable_gpu', 'enable_internet'):
                assert meta.get(key) == k2m.get(key), (slug, key, meta.get(key), k2m.get(key))
        assert meta['id'] == f'eeshsaxena/{slug}' and meta['is_private'] is True
        print(f'{slug}: {len(nb["cells"])} cells, K2_CFG {hdr[len("K2_CFG = "):hdr.index("}") + 1]}, '
              f'dataset_sources == K2 ({len(meta["dataset_sources"])}): {os.path.isfile(K2_META)}')
    assert srcs[MAIN] == srcs[SOLO]
    up = summ['upload']['path']
    print(f'\nK3 BUILD OK: upload {up} ({summ["upload"]["mb"]} MB, {len(summ["upload"]["files"])} files) | '
          f'M1 {n1} | M2 {n2} | kernels {MAIN} (strict W {a.w_main}), {SOLO} (solo)')
    return summ


if __name__ == '__main__':
    main()
