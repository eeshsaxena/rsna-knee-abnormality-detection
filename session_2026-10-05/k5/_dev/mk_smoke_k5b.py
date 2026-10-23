# dev helper (not shipped): K5b smoke scenarios (kernel TTA expectation) in k5/tests
import os, ast
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def edit(p, pairs):
    s = open(f'{HERE}/{p}', encoding='utf-8', newline='').read()
    assert 'K5b' not in s, f'{p} already patched'
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    ast.parse(s)
    open(f'{HERE}/{p}', 'w', encoding='utf-8', newline='').write(s)
    print('patched', p)


edit('tests/smoke_k2.py', [
    ('''    mk.main(['--out', f'{S}/nb/rsna-sub-k3-unknown', '--slug', 'rsna-sub-k3-unknown', '--main_mode', 'strict',
             '--members', 'nope.pt'] + W)
''', '''    mk.main(['--out', f'{S}/nb/rsna-sub-k3-unknown', '--slug', 'rsna-sub-k3-unknown', '--main_mode', 'strict',
             '--members', 'nope.pt'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k5-strictexp', '--slug', 'rsna-sub-k5-strictexp', '--main_mode', 'strict',   # K5b
             '--members', f'{tm["f1"]},{tm["ttr"]}', '--tta', f'{tm["f1"]}=A,{tm["ttr"]}=AB'] + W)
'''),
    ('''       'strict_ttr_tta', 'strict_tta_oldworker', 'strict_tta_bad', 'ttrsolo_tta']                     # K5''',
     '''       'strict_ttr_tta', 'strict_tta_oldworker', 'strict_tta_bad', 'ttrsolo_tta',                     # K5
       'strict_tta_expect_ok', 'strict_tta_expect_k4ds']                                                # K5b'''),
])

edit('tests/run_scenario.py', [
    ('''    'ttrsolo_tta':      dict(nb='ttrsolo', fds='ttr_tta', members='ttr_only', tta='ttr'),''',
     '''    'ttrsolo_tta':      dict(nb='ttrsolo', fds='ttr_tta', members='ttr_only', tta='ttr'),
    # ---- K5b: the kernel's own TTA expectation (K2_CFG TTA) vs the attached manifest
    'strict_tta_expect_ok':   dict(nb='strict_exp', fds='ttr_tta', members='ttr_pair', tta='pair'),
    'strict_tta_expect_k4ds': dict(nb='strict_exp', fds='ttr', expect_fleet='fail'),'''),
    ('''      'unknown': f'{S}/nb/rsna-sub-k3-unknown/rsna_sub_k3_unknown.ipynb'}[cfg['nb']]''',
     '''      'unknown': f'{S}/nb/rsna-sub-k3-unknown/rsna_sub_k3_unknown.ipynb',
      'strict_exp': f'{S}/nb/rsna-sub-k5-strictexp/rsna_sub_k5_strictexp.ipynb'}[cfg['nb']]          # K5b'''),
    ('''if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown') and not cfg['outer']:''',
     '''if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown', 'strict_exp') and not cfg['outer']:   # K5b'''),
    ('''if SC == 'strict_tta_oldworker':''', '''if SC == 'strict_tta_expect_ok':                                         # K5b
    extra.append(('K5b kernel TTA expectation recorded and == the manifest', rec.get('tta_kernel') == EXPECT_TTA))
if SC == 'strict_tta_expect_k4ds':
    extra.append(('K5b manifest without tta under a K5 kernel: why names the kernel expectation',
                  'kernel expects TTA' in why))
if SC == 'strict_tta_oldworker':'''),
])
