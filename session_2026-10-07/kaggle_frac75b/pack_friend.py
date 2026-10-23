# pack_friend.py (kaggle_frac75b): assemble the teammate pack from the built kit. Run AFTER python build.py.
#   Downloads\rsna-knee\friend2_frac75b_pack\  README_TEAMMATE.md, TODO.md, RUNBOOK_EESH.md, the 2 notebooks,
#                                              cli\<slug>\{kernel-metadata.json, notebook} for both slugs
#   Downloads\rsna-knee\friend2_frac75b_pack.zip  (the same files)
import os, re, json, shutil, zipfile, hashlib

KT = os.path.dirname(os.path.abspath(__file__))
DST = r'C:\Users\eeshs\Downloads\rsna-knee\friend2_frac75b_pack'
ZIP = DST + '.zip'
SLUGS = ['rsna-train-ttr-frac75-f23', 'rsna-train-ttr-frac75-f4x']
PLANS = {'rsna-train-ttr-frac75-f23': [(2, 1043), (3, 1043)], 'rsna-train-ttr-frac75-f4x': [(4, 1043), (0, 1044)]}
files = {'README_TEAMMATE.md': f'{KT}/pack/README_TEAMMATE.md', 'TODO.md': f'{KT}/pack/TODO.md',
         'RUNBOOK_EESH.md': f'{KT}/RUNBOOK.md'}
for slug in SLUGS:
    nb = slug.replace('-', '_') + '.ipynb'
    kd = f'{KT}/kernel_teammate/{slug}'
    files[nb] = f'{kd}/{nb}'
    files[f'cli/{slug}/kernel-metadata.json'] = f'{kd}/kernel-metadata.json'
    files[f'cli/{slug}/{nb}'] = f'{kd}/{nb}'
    km = json.load(open(f'{kd}/kernel-metadata.json'))
    assert km['id'] == f'YOUR_KAGGLE_USERNAME/{slug}' and km['title'] == slug and km['code_file'] == nb
    assert km['machine_shape'] == 'NvidiaTeslaT4' and km['enable_gpu'] and km['enable_internet'] and km['is_private']
    assert km['dataset_sources'] == ['eeshsaxena/rsna-knee-fleet-train', 'eeshsaxena/rsna-knee-frac75-labels']
    assert km['kernel_sources'] == ['eeshsaxena/rsna-fleet-prep-s0', 'eeshsaxena/rsna-fleet-prep-s1']
    b = open(f'{kd}/{nb}', 'rb').read()
    assert b == open(f'{KT}/kernel/{slug}/{nb}', 'rb').read()
    src = ''.join(json.loads(b)['cells'][1]['source'])
    assert f"NB_PLAN = dict(kernel={slug!r}, trainers={PLANS[slug]!r})" in src, slug
    assert "BUDGET_H=_env('KT_BUDGET_H', 10.3)" in src and "WALL_H=_env('KT_WALL_H', 11.4)" in src
for rel, src in files.items():
    t = open(src, 'rb').read().decode('utf-8')
    assert '\u2014' not in t and '\u2013' not in t, f'em/en dash in {src}'
    if rel.endswith('.md'):
        assert 'DRYTEST_RESULT' not in t, f'unfilled placeholder in {src}'
        assert not re.search(r'(?<![-\w])([Hh]e|[Hh]is|[Hh]im|[Hh]imself)\b', t), f'pronoun he/his in {src}'
readme = open(files['README_TEAMMATE.md']).read()
assert 'For Eesh' not in readme and '192.168' not in readme, 'README_TEAMMATE.md must stay teammate-facing'
assert 'GO' in readme and '--timeout 43200' in readme and '24 h' in readme and 'kaggle quota' in readme
shutil.rmtree(DST, ignore_errors=True)
for rel, src in files.items():
    os.makedirs(os.path.dirname(os.path.join(DST, rel)), exist_ok=True)
    shutil.copyfile(src, os.path.join(DST, rel))
if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for rel in files:
        z.write(os.path.join(DST, rel), rel)
with zipfile.ZipFile(ZIP) as z:
    names = sorted(z.namelist())
    assert names == sorted(files), names
    shas = {n: hashlib.sha256(z.read(n)).hexdigest()[:16] for n in names if n.endswith('.ipynb') and '/' not in n}
print('pack:', DST, '| zip:', ZIP, os.path.getsize(ZIP), 'bytes |', len(names), 'files | notebooks', shas)
for n in names:
    print('  ', n)
