# pack_friend.py (kaggle_frac75): assemble the teammate pack from the built kit. Run AFTER python build.py.
#   Downloads\rsna-knee\friend_frac75_pack\  README_TEAMMATE.md, TODO.md, RUNBOOK_EESH.md, the notebook,
#                                            cli\rsna-train-ttr-frac75-f01\{kernel-metadata.json, notebook}
#   Downloads\rsna-knee\friend_frac75_pack.zip  (the same files)
import os, json, shutil, zipfile, hashlib

KT = os.path.dirname(os.path.abspath(__file__))
DST = r'C:\Users\eeshs\Downloads\rsna-knee\friend_frac75_pack'
ZIP = DST + '.zip'
SLUG, NB = 'rsna-train-ttr-frac75-f01', 'rsna_train_ttr_frac75_f01.ipynb'
SRC_KD = f'{KT}/kernel_teammate/{SLUG}'
files = {
    'README_TEAMMATE.md': f'{KT}/pack/README_TEAMMATE.md',
    'TODO.md': f'{KT}/pack/TODO.md',
    'RUNBOOK_EESH.md': f'{KT}/RUNBOOK.md',
    NB: f'{SRC_KD}/{NB}',
    f'cli/{SLUG}/kernel-metadata.json': f'{SRC_KD}/kernel-metadata.json',
    f'cli/{SLUG}/{NB}': f'{SRC_KD}/{NB}',
}
km = json.load(open(f'{SRC_KD}/kernel-metadata.json'))
assert km['id'] == f'YOUR_KAGGLE_USERNAME/{SLUG}' and km['machine_shape'] == 'NvidiaTeslaT4' and km['enable_gpu']
assert km['dataset_sources'] == ['eeshsaxena/rsna-knee-fleet-train', 'eeshsaxena/rsna-knee-frac75-labels']
assert km['kernel_sources'] == ['eeshsaxena/rsna-fleet-prep-s0', 'eeshsaxena/rsna-fleet-prep-s1']
assert open(f'{SRC_KD}/{NB}', 'rb').read() == open(f'{KT}/kernel/{SLUG}/{NB}', 'rb').read()
for rel, src in files.items():
    t = open(src, 'rb').read().decode('utf-8')
    assert '\u2014' not in t and '\u2013' not in t, f'em/en dash in {src}'
    if rel.endswith('.md'):
        assert 'DRYTEST_RESULT' not in t and 'SCORE_RESULT' not in t, f'unfilled placeholder in {src}'
readme = open(files['README_TEAMMATE.md']).read()
assert 'For Eesh' not in readme and '192.168' not in readme, 'README_TEAMMATE.md must stay teammate-facing'
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
    nbz = z.read(NB)
print('pack:', DST, '| zip:', ZIP, os.path.getsize(ZIP), 'bytes |', len(names), 'files | notebook sha256',
      hashlib.sha256(nbz).hexdigest()[:16])
for n in names:
    print('  ', n)
