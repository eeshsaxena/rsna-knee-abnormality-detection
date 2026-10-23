# pack_friend.py (kaggle_frac75c): assemble the two teammate packs (ONE notebook each) from the built kit. Run AFTER
# python build.py.
#   Downloads\rsna-knee\friend_s1044_A\ (+ .zip)  notebook A rsna-train-ttr-frac75-s1044-f12 (folds 1+2 s1044), for swapiy
#   Downloads\rsna-knee\friend_s1044_B\ (+ .zip)  notebook B rsna-train-ttr-frac75-s1044-f34 (folds 3+4 s1044), for kmrsuman
# Each: README_TEAMMATE.md + TODO.md (rendered from pack/*.tmpl.md), RUNBOOK_EESH.md (= RUNBOOK.md), the notebook,
# cli/<slug>/{kernel-metadata.json, notebook}.
import os, re, json, shutil, zipfile, hashlib

KT = os.path.dirname(os.path.abspath(__file__))
OUT = r'C:\Users\eeshs\Downloads\rsna-knee'
PACKS = {
    'A': dict(slug='rsna-train-ttr-frac75-s1045-f01', trainers=[(0, 1045), (1, 1045)], mate='swapiy'),
    'B': dict(slug='rsna-train-ttr-frac75-s1045-f23', trainers=[(2, 1045), (3, 1045)], mate='swapiy'),
}
TMPL = {'README_TEAMMATE.md': f'{KT}/pack/README_TEAMMATE.tmpl.md', 'TODO.md': f'{KT}/pack/TODO.tmpl.md'}
runbook = open(f'{KT}/RUNBOOK.md', encoding='utf-8').read()
assert 'DRYTEST_RESULT' not in runbook, 'RUNBOOK.md sec 9 not filled in yet'
summary = []
for key, pk in PACKS.items():
    slug, tr = pk['slug'], pk['trainers']
    nb = slug.replace('-', '_') + '.ipynb'
    name = f'swapiy_s1045_{key}'
    dst, zp = os.path.join(OUT, name), os.path.join(OUT, name + '.zip')
    runs = [f'f{F}_s{s}_ttr_frac75' for F, s in tr]
    var = {'SLUG': slug, 'NB': nb, 'MATE': pk['mate'], 'ZIP': name + '.zip',
           'FOLDS_TXT': f'folds {tr[0][0]} + {tr[1][0]}',
           'RUNS_TXT': ' + '.join(f'`{r}/`' for r in runs),
           'PLACEHOLDERS_TXT': ' and '.join(f'`~/fleet_runs/{r}`' for r in runs),
           'PLACEHOLDERS_GLOB': f'~/fleet_runs/f[{tr[0][0]}{tr[1][0]}]_s1045_ttr_frac75',
           'RECEIPT': f'kaggle_frac75_s1045_f{tr[0][0]}{tr[1][0]}_run_receipt.json'}
    assert slug.endswith(f'-f{tr[0][0]}{tr[1][0]}')
    text = {}
    for rel, src in TMPL.items():
        t = open(src, encoding='utf-8').read()
        for k, v in var.items():
            t = t.replace('{{' + k + '}}', v)
        assert '{{' not in t and '}}' not in t, f'unfilled placeholder in {rel} ({key})'
        text[rel] = t
    text['RUNBOOK_EESH.md'] = runbook
    kd = f'{KT}/kernel_teammate/{slug}'
    files = {nb: f'{kd}/{nb}', f'cli/{slug}/kernel-metadata.json': f'{kd}/kernel-metadata.json', f'cli/{slug}/{nb}': f'{kd}/{nb}'}
    km = json.load(open(f'{kd}/kernel-metadata.json'))
    assert km['id'] == f'YOUR_KAGGLE_USERNAME/{slug}' and km['title'] == slug and km['code_file'] == nb
    assert km['machine_shape'] == 'NvidiaTeslaT4' and km['enable_gpu'] and km['enable_internet'] and km['is_private']
    assert km['dataset_sources'] == ['eeshsaxena/rsna-knee-fleet-train', 'eeshsaxena/rsna-knee-frac75-labels']
    assert km['kernel_sources'] == ['eeshsaxena/rsna-fleet-prep-s0', 'eeshsaxena/rsna-fleet-prep-s1']
    b = open(f'{kd}/{nb}', 'rb').read()
    assert b == open(f'{KT}/kernel/{slug}/{nb}', 'rb').read()
    src = ''.join(json.loads(b)['cells'][1]['source'])
    assert f"NB_PLAN = dict(kernel={slug!r}, trainers={tr!r})" in src, slug
    assert "BUDGET_H=_env('KT_BUDGET_H', 10.3)" in src and "WALL_H=_env('KT_WALL_H', 11.4)" in src
    # teammate-facing text: one notebook, the hand-off rules, no box internals, no he/his, no em/en dashes
    rd = text['README_TEAMMATE.md']
    other = [p['slug'] for k, p in PACKS.items() if k != key][0]
    assert other not in rd and other not in text['TODO.md'], f'pack {key} mentions the other notebook'
    assert 'For Eesh' not in rd and '192.168' not in rd and 'fleet_runs' not in rd, 'README must stay teammate-facing'
    for must in ('kaggle quota', '12 h', 'T4 x2', f'kaggle kernels push -p cli/{slug} --timeout 43200', 'keep the slug',
                 '`eeshsaxena`', 'Can view', 'do not stop', 'EMPTY log', 'Do not retry', 'GO'):
        assert must.lower() in rd.lower(), f'README {key} lacks {must!r}'
    for rel, t in list(text.items()) + [(n, open(p, encoding='utf-8').read()) for n, p in files.items()]:
        assert '\u2014' not in t and '\u2013' not in t, f'em/en dash in {rel} ({key})'
        if rel.endswith('.md'):
            assert not re.search(r'(?<![-\w])([Hh]e|[Hh]is|[Hh]im|[Hh]imself)\b', t), f'pronoun he/his in {rel} ({key})'
    shutil.rmtree(dst, ignore_errors=True)
    for rel, t in text.items():
        os.makedirs(os.path.dirname(os.path.join(dst, rel)) or dst, exist_ok=True)
        with open(os.path.join(dst, rel), 'w', encoding='utf-8', newline='\n') as f:
            f.write(t)
    for rel, s in files.items():
        os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
        shutil.copyfile(s, os.path.join(dst, rel))
    allrel = sorted(list(text) + list(files))
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in allrel:
            z.write(os.path.join(dst, rel), rel)
    with zipfile.ZipFile(zp) as z:
        names = sorted(z.namelist())
        assert names == allrel, names
        assert z.read(nb) == z.read(f'cli/{slug}/{nb}') == b
        nbsha = hashlib.sha256(z.read(nb)).hexdigest()[:16]
    summary.append((name, pk['mate'], slug, os.path.getsize(zp), len(names), nbsha, names))
for name, mate, slug, size, n, nbsha, names in summary:
    print(f'{name}.zip  for {mate}  {slug}  {size} bytes  {n} files  notebook sha256 {nbsha}')
    for x in names:
        print('   ', x)
