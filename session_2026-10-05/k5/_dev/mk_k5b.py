# dev helper (not shipped): K5b = the kernel's own TTA expectation (K2_CFG 'TTA') checked by the K5 cell, so a K5 kernel
# that mounts a dataset version without the K5 TTA manifest (e.g. the K4 version: same J1 bytes) fails loud in-kernel.
# Edits make_k2_kernel.py, build_k2.py, cells/k2_fleet_cell.py (inside the K5a block only), k2_check.py, build_k5.py.
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def edit(p, pairs):
    s = open(f'{HERE}/{p}', encoding='utf-8', newline='').read()
    assert 'K5b' not in s, f'{p} already patched'
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    if p.endswith('.py'):
        import ast
        ast.parse(s)
    open(f'{HERE}/{p}', 'w', encoding='utf-8', newline='').write(s)
    print('patched', p)


edit('make_k2_kernel.py', [
    ('''  K2_CFG  = W_MAIN, W_VARIANTS, WALL_H, MIN_START_MIN, MAIN_MODE (blend|solo|strict) [, MEMBERS]   (fleet stage)''',
     '''  K2_CFG  = W_MAIN, W_VARIANTS, WALL_H, MIN_START_MIN, MAIN_MODE (blend|solo|strict) [, MEMBERS] [, TTA]   (fleet stage)
  K5b: --tta NAME=A|AB[,..] writes K2_CFG['TTA'] (only when given, so older kernels are unchanged); the K5 cell then
       fails loud unless the attached manifest's per-member tta equals it (wrong / pre-K5 dataset version mounted)'''),
    ('''        k2_cfg['MEMBERS'] = members
''', '''        k2_cfg['MEMBERS'] = members
    tta = [x.strip() for x in str(getattr(a, 'tta', None) or '').split(',') if x.strip()]
    if tta:                                                              # K5b: only when given (old cells unchanged)
        tmap = dict(x.rsplit('=', 1) for x in tta)
        assert len(tmap) == len(tta) and all(v in ('A', 'AB') for v in tmap.values()), f'--tta {tta}'
        assert all(k.endswith('.pt') for k in tmap), f'--tta takes checkpoint FILE names: {tta}'
        assert not members or set(members) <= set(tmap), f'--tta {sorted(tmap)} does not cover --members {members}'
        k2_cfg['TTA'] = tmap
'''),
    ('''    ap.add_argument('--members', default=None, help='K3b: comma list of manifest checkpoint file names (default: all)')
''', '''    ap.add_argument('--members', default=None, help='K3b: comma list of manifest checkpoint file names (default: all)')
    ap.add_argument('--tta', default=None, help='K5b: NAME=A|AB[,..]: the kernel expects exactly this manifest tta')
'''),
])

edit('build_k2.py', [
    ('''              '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \\
             (['--members', ','.join(want)] if want else [])
''', '''              '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \\
             (['--members', ','.join(want)] if want else []) + \\
             (['--tta', ','.join(f'{n}={m}' for n, m in tta.items())] if tta else [])          # K5b
'''),
    ('''            '--wall_h', str(a.wall_h), '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \\
           (['--members', ','.join(want)] if want else [])
''', '''            '--wall_h', str(a.wall_h), '--min_start_min', str(a.min_start_min), '--main_mode', a.main_mode] + \\
           (['--members', ','.join(want)] if want else []) + \\
           (['--tta', ','.join(f'{n}={m}' for n, m in upinfo['tta'].items())] if upinfo.get('tta') else [])   # K5b
'''),
])

edit('cells/k2_fleet_cell.py', [
    ('''# Receipt: rec['tta'] (modes, half, rows, forwards per member per pass, microbench passes) + one 'TTA' log line.
''', '''# Receipt: rec['tta'] (modes, half, rows, forwards per member per pass, microbench passes) + one 'TTA' log line.
# K5b: a kernel built with K2_CFG['TTA'] (build_k5.py) also needs the manifest's tta == that map for every member that
# ran, so a K5 kernel that mounts a dataset version without the K5 manifest (the K4 one holds the same J1 bytes) fails.
'''),
    ('''    if any(v not in ('A', 'AB') for v in want.values()):
        raise _K2Fail(f'manifest tta {want}: each must be A or AB')
''', '''    if any(v not in ('A', 'AB') for v in want.values()):
        raise _K2Fail(f'manifest tta {want}: each must be A or AB')
    exp = K2.get('TTA')
    if exp is not None:
        exp_run = {n: exp.get(n) for n in names}
        rec['tta_kernel'] = exp_run
        if exp_run != want:
            raise _K2Fail(f'kernel expects TTA {exp_run} but the attached manifest says {want} '
                          f'(a dataset version without the K5 TTA manifest is mounted?)')
'''),
])

edit('k2_check.py', [
    ('''        if expect_tta is not None:                                                      # K5
            t = rec.get('tta') or {}
''', '''        if expect_tta is not None:                                                      # K5
            t = rec.get('tta') or {}
            if 'TTA' in cfg:                                                            # K5b
                ok('kernel K2_CFG TTA == --expect_tta', cfg['TTA'] == expect_tta, cfg['TTA'])
'''),
])

edit('build_k5.py', [
    ('''        assert cfg['MAIN_MODE'] == mode and cfg['MEMBERS'] == names, (slug, cfg)
''', '''        assert cfg['MAIN_MODE'] == mode and cfg['MEMBERS'] == names, (slug, cfg)
        assert cfg.get('TTA') == tta, (slug, 'K2_CFG TTA (K5b)', cfg.get('TTA'), tta)
'''),
    ('''        print(f'{slug}: {len(nb["cells"])} cells, K2_CFG {hdr[len("K2_CFG = "):hdr.index("}") + 1]}, \'''',
     '''        print(f'{slug}: {len(nb["cells"])} cells, K2_CFG {ks[slug]["K2_CFG"]}, \''''),
])
