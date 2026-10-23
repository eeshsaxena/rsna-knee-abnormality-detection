# build.py (kaggle_frac75, 2026-10-07): re-derive the code dataset's expectations + manifest + metadata (they must come
# out BYTE-IDENTICAL to the published eeshsaxena/rsna-knee-fleet-train: asserted, so no new version is ever needed),
# write the metadata of the NEW labels dataset eeshsaxena/rsna-knee-frac75-labels (ds_labels/), and assemble the
# kernel folders (notebook + kernel-metadata) from nb_src/. Idempotent. Run: python build.py
#   kernel/rsna-train-ttr-frac75-f01/           id eeshsaxena/... (only if Eesh ever runs it himself)
#   kernel_teammate/rsna-train-ttr-frac75-f01/  id YOUR_KAGGLE_USERNAME/... (the teammate pack copy)
import os, re, json, hashlib, glob

KT = os.path.dirname(os.path.abspath(__file__))
DS = f'{KT}/ds/rsna-knee-fleet-train'
DL = f'{KT}/ds_labels'
SLUG = 'rsna-train-ttr-frac75-f01'
NBNAME = 'rsna_train_ttr_frac75_f01.ipynb'
KD = f'{KT}/kernel/{SLUG}'
KDT = f'{KT}/kernel_teammate/{SLUG}'
PREP = os.path.abspath(f'{KT}/../fleet_prep_out')
FRAC = 'labels_frac75_logit.csv'
PREREG_MD5 = 'ad7582b817243f8f22e8bcfdd7e3f84a'          # ~/novel/frac75/PREREG.md sec 1


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


before = {f: open(f'{DS}/{f}', 'rb').read() for f in os.listdir(DS)}

# 1 expectations
parts = {}
for s in (0, 1):
    parts.update(json.load(open(f'{PREP}/prep_manifest_{s}.json'))['parts'])
assert len(parts) == 12
labels = ['labels_mix50_logit.csv', 'folds5.csv', 'gold74_eval_set.csv', 'llm_labels_v4_blend.csv']
hf_file = 'convnext_tiny.fb_in22k_ft_in1k.model.safetensors'
exp = dict(
    parts=parts,
    hf=dict(repo='timm/convnext_tiny.fb_in22k_ft_in1k', revision='bc48a87f119bae1b06418f5340bfa97ef75d4609',
            file=hf_file, sha256=sha(f'{DS}/{hf_file}'),
            source='box ~/.cache/huggingface/hub/models--timm--convnext_tiny.fb_in22k_ft_in1k (blob 77b619e1...)'),
    labels_sha256={f: sha(f'{DS}/{f}') for f in labels},
    timm_wheel='timm-1.0.29-py3-none-any.whl.bin',
    box_versions=dict(python='3.11.16', torch='2.5.1+cu121', timm='1.0.29', numpy='1.26.4', cv2='4.14.0',
                      sklearn='1.9.0', pandas='3.0.5'),
    box_recipe='cd ~/bigjump_build/code; python fleet_train.py --model ttr --aug heavy --train_slices 16 --drop_path 0.2 '
               '--dropout 0.3 --ema 0.9995 --epochs 24 --seed 1043 --save_epochs 6-24 --workers 10 --store ~/fleet_store '
               '--labels ~/rsna_train/labels_mix50_logit.csv --folds ~/rsna_train/folds5.csv --exclude '
               '~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv --gold '
               '~/rsna_train/gold74_eval_set.csv --fold F --out ~/fleet_runs/fF_s1043_ttr_mix50',
    box_timing='V100 alone, 10 workers: 1733 it/epoch at 0.35-0.37 s/it steady, TRAIN-ONLY 10.1-10.4 min, epoch incl. '
               'raw+ema eval 12.3-12.6 min (J1 ~/fleet_runs/f0_s1043_ttr_mix50.log)',
)
assert exp['hf']['sha256'] == '77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a'
json.dump(exp, open(f'{DS}/kt_expect.json', 'w', newline=chr(10)), indent=1)

# 2 box facts: + labelled_missing (box store has every labelled study: store_has == labels for every fold)
bf = json.load(open(f'{DS}/box_facts.json'))
c = bf['expected_counts']['0']
bf['labelled_missing'] = c['labels'] - c['store_has']
assert bf['labelled_missing'] == 0
json.dump(bf, open(f'{DS}/box_facts.json', 'w', newline=chr(10)), indent=1)

# 3 dataset metadata + manifest
json.dump({'title': 'rsna-knee-fleet-train', 'id': 'eeshsaxena/rsna-knee-fleet-train',
           'licenses': [{'name': 'CC0-1.0'}]}, open(f'{DS}/dataset-metadata.json', 'w', newline=chr(10)), indent=1)
files = sorted(f for f in os.listdir(DS) if f not in ('kt_manifest.json', 'dataset-metadata.json'))
assert not [f for f in files if os.path.isdir(f'{DS}/{f}')], 'dataset must stay flat'
json.dump({'files': {f: sha(f'{DS}/{f}') for f in files}}, open(f'{DS}/kt_manifest.json', 'w', newline=chr(10)), indent=1)
after = {f: open(f'{DS}/{f}', 'rb').read() for f in os.listdir(DS)}
assert after == before, ('the code dataset changed: it must stay byte-identical to the published '
                         f'eeshsaxena/rsna-knee-fleet-train ({sorted(f for f in after if after[f] != before.get(f))})')

# 3b the NEW labels dataset: the frac75 csv only (+ its metadata); sha256 == the value baked into nb_src/c01_lib.py
lib = open(f'{KT}/nb_src/c01_lib.py').read()
baked = re.search(r"LABELS_TRAIN_SHA256='([0-9a-f]{64})'", lib).group(1)
baked_md5 = re.search(r"LABELS_TRAIN_MD5='([0-9a-f]{32})'", lib).group(1)
fb = open(f'{DL}/{FRAC}', 'rb').read()
assert hashlib.sha256(fb).hexdigest() == baked, f'{FRAC} sha256 != the value baked into c01_lib.py'
assert hashlib.md5(fb).hexdigest() == PREREG_MD5 == baked_md5, f'{FRAC} md5 != PREREG md5 {PREREG_MD5}'
mix = open(f'{DS}/labels_mix50_logit.csv', 'rb').read()
a = [l.rsplit(b',', 1) for l in fb.split(b'\n') if l]
m = [l.rsplit(b',', 1) for l in mix.split(b'\n') if l]
assert len(a) == len(m) == 4408 and a[0] == m[0] and all(x[0] == y[0] for x, y in zip(a, m)), 'not mix50 + new Fracture'
json.dump({'title': 'rsna-knee-frac75-labels', 'id': 'eeshsaxena/rsna-knee-frac75-labels',
           'licenses': [{'name': 'CC0-1.0'}]}, open(f'{DL}/dataset-metadata.json', 'w', newline=chr(10)), indent=1)
assert sorted(os.listdir(DL)) == ['dataset-metadata.json', FRAC], sorted(os.listdir(DL))

# 4 notebook
md = f"""# {SLUG}: TTR224 frac75 (Fracture-target) folds 0 + 1, seed 1043, on Kaggle T4 x2

The box TTR224 mix50 recipe (J1 = `~/fleet_runs/f0_s1043_ttr_mix50`) with ONE change: `--labels labels_frac75_logit.csv`
(= mix50 with only the Fracture column re-mixed .25 v4 + .75 teacher; dataset `eeshsaxena/rsna-knee-frac75-labels`,
sha256 baked into cell 1). One fold per T4, in parallel. Sized for a ~10 h GPU quota: the session ends by ~9.3 h.

1. locate inputs (code dataset `eeshsaxena/rsna-knee-fleet-train`, labels dataset `eeshsaxena/rsna-knee-frac75-labels`,
   prep kernel outputs `rsna-fleet-prep-s0/-s1`), verify every file by sha256 (frac75 csv: baked sha256 + only-Fracture-differs)
2. environment report (timm < 1.0: bundled wheel)
3. extract the 12 store tars to `/tmp/fleet_store`, prove it equals the box `~/fleet_store` (names, 24 sampled files)
4. parity: contract pixel/tensor sha, select_sha, jpeg_fp, model_sha, decode probe, state_dict keys, pretrained weights, fold counts
5. timing probe: both trainers concurrently, steady s/it, CPU/GPU utilisation
6. `E = min(24, floor((8.4 h - elapsed) / epoch time))` (expected 17-18), launch, watchdog (hard wall 9.2 h), receipt
7. summary

Outputs: `f0_s1043_ttr_frac75/`, `f1_s1043_ttr_frac75/` (+ `.log`; `epochs/` = last 3 epochs), `run_receipt.json`, `probe/`.
Please do not stop it early: it stops itself before the time limit and keeps everything it trained.
"""
cells = [dict(cell_type='markdown', metadata={}, source=md.splitlines(True))]
for p in sorted(glob.glob(f'{KT}/nb_src/c0*.py')):
    src = open(p).read().rstrip('\n') + '\n'
    cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=src.splitlines(True)))
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=4)
for kd, owner in ((KD, 'eeshsaxena'), (KDT, 'YOUR_KAGGLE_USERNAME')):
    os.makedirs(kd, exist_ok=True)
    json.dump(nb, open(f'{kd}/{NBNAME}', 'w', newline=chr(10)), indent=1)
    json.dump({'id': f'{owner}/{SLUG}', 'title': SLUG, 'code_file': NBNAME,
               'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
               'enable_internet': True,
               'dataset_sources': ['eeshsaxena/rsna-knee-fleet-train', 'eeshsaxena/rsna-knee-frac75-labels'],
               'kernel_sources': ['eeshsaxena/rsna-fleet-prep-s0', 'eeshsaxena/rsna-fleet-prep-s1'],
               'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'},
              open(f'{kd}/kernel-metadata.json', 'w', newline=chr(10)), indent=1)
assert open(f'{KD}/{NBNAME}', 'rb').read() == open(f'{KDT}/{NBNAME}', 'rb').read()
for c in cells[1:]:
    compile(''.join(c['source']), 'cell', 'exec')
txt = open(f'{KD}/{NBNAME}', encoding='utf-8').read() + md
assert chr(0x2014) not in txt and chr(0x2013) not in txt, 'no em/en dashes'
print('code dataset: byte-identical to the published version (', len(files), 'files ) | labels dataset:', FRAC,
      'sha256', baked[:16], '| notebook cells:', len(cells), '| ok')
