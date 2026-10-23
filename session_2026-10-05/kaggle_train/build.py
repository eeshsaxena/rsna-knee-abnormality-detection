# build.py: assemble the Kaggle dataset folder (expectations + manifest + metadata) and the kernel folder (notebook +
# kernel-metadata) from nb_src/. Idempotent. Run: python build.py
import os, json, hashlib, glob

KT = os.path.dirname(os.path.abspath(__file__))
DS = f'{KT}/ds/rsna-knee-fleet-train'
KD = f'{KT}/kernel/rsna-train-ttr-f34'
PREP = os.path.abspath(f'{KT}/../fleet_prep_out')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


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

# 4 notebook
os.makedirs(KD, exist_ok=True)
md = """# rsna-train-ttr-f34: TTR mix50 folds 3 + 4 (seed 1043) on Kaggle T4 x2

Same trainer, data, labels and pretrained bytes as the V100 box runs (J1 fold 0, folds 1-2), one fold per T4, in parallel.

1. locate inputs (code dataset `eeshsaxena/rsna-knee-fleet-train`, prep kernel outputs `rsna-fleet-prep-s0/-s1`), verify every file by sha256
2. environment report (timm < 1.0: bundled wheel)
3. extract the 12 store tars to `/tmp/fleet_store`, prove it equals the box `~/fleet_store` (names, 24 sampled files)
4. parity: contract pixel/tensor sha, select_sha, jpeg_fp, model_sha, decode probe, state_dict keys, pretrained weights, fold counts
5. timing probe: both trainers concurrently, steady s/it, CPU/GPU utilisation
6. `E = min(24, floor(budget / epoch time))`, launch, watchdog (hard wall 11.4 h), receipt
7. summary

Outputs: `f3_s1043_ttr_mix50/`, `f4_s1043_ttr_mix50/` (+ `.log`), `run_receipt.json`, `probe/`.
"""
cells = [dict(cell_type='markdown', metadata={}, source=md.splitlines(True))]
for p in sorted(glob.glob(f'{KT}/nb_src/c0*.py')):
    src = open(p).read().rstrip('\n') + '\n'
    cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=src.splitlines(True)))
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=4)
json.dump(nb, open(f'{KD}/rsna_train_ttr_f34.ipynb', 'w', newline=chr(10)), indent=1)
json.dump({'id': 'eeshsaxena/rsna-train-ttr-f34', 'title': 'rsna-train-ttr-f34', 'code_file': 'rsna_train_ttr_f34.ipynb',
           'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
           'enable_internet': True, 'dataset_sources': ['eeshsaxena/rsna-knee-fleet-train'],
           'kernel_sources': ['eeshsaxena/rsna-fleet-prep-s0', 'eeshsaxena/rsna-fleet-prep-s1'],
           'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'},
          open(f'{KD}/kernel-metadata.json', 'w', newline=chr(10)), indent=1)
for c in cells[1:]:
    compile(''.join(c['source']), 'cell', 'exec')
print('dataset files:', len(files), '| notebook cells:', len(cells), '| ok')
