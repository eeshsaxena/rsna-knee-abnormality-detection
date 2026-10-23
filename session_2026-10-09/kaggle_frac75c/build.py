# build.py (kaggle_frac75c, 2026-10-09): assemble the two seed-1044 frac75 training notebooks from nb_src/ (one shared
# source = the kaggle_frac75b source, proven on Kaggle 10-07/08; build rewrites ONLY the NB_PLAN line of nb_src/c01_lib.py
# per notebook). Both input datasets are LIVE and unchanged: this script only READS their local copies and asserts they
# match what the notebooks verify (nothing is created/versioned).
#   A  rsna-train-ttr-frac75-s1044-f12  fold 1 s1044 (GPU0) + fold 2 s1044 (GPU1)   (teammate swapiy)
#   B  rsna-train-ttr-frac75-s1044-f34  fold 3 s1044 (GPU0) + fold 4 s1044 (GPU1)   (teammate kmrsuman)
# With ~/fleet_runs/f0_s1044_ttr_frac75 (rsna-train-ttr-frac75-f4x, 10-08) they complete a 5-fold TTR224 frac75 s1044 set.
# Output: kernel/<slug>/ (id eeshsaxena/..., only if Eesh runs it) and kernel_teammate/<slug>/ (id YOUR_KAGGLE_USERNAME/...).
# Extra guard: every code cell == the proven kaggle_frac75b notebook except comment lines and the NB_PLAN line.
# Run: python build.py
import os, re, json, hashlib, glob

KT = os.path.dirname(os.path.abspath(__file__))
SP = r'C:\Users\eeshs\AppData\Local\Temp\claude\C--Users-eeshs-Downloads\405896d1-8e93-4c82-a842-70e7cf895bfa\scratchpad'
DS = os.environ.get('KT_DS', f'{SP}/kaggle_frac75/ds/rsna-knee-fleet-train')    # local copy of the LIVE code dataset
DL = f'{KT}/ds_labels'                                                            # local copy of the LIVE labels dataset
FRAC = 'labels_frac75_logit.csv'
LIVE_SHA = '5036425cd0d64f27f08acfe6865d668ecf830e11562ed8921c63afa782c2c8c0'   # eeshsaxena/rsna-knee-frac75-labels
BASE_NB = os.path.join(os.path.dirname(os.path.dirname(KT)), 'session_2026-10-07', 'kaggle_frac75b', 'kernel',
                       'rsna-train-ttr-frac75-f23', 'rsna_train_ttr_frac75_f23.ipynb')
BASE_SHA = 'd6175d4a556f337d25970af25323a430a99a6d9792677295fe3ef617cc0ebd19'   # ran on Kaggle 10-07/08 (E 21)
PLANS = {
    'A': dict(kernel='rsna-train-ttr-frac75-s1044-f12', trainers=[(1, 1044), (2, 1044)]),
    'B': dict(kernel='rsna-train-ttr-frac75-s1044-f34', trainers=[(3, 1044), (4, 1044)]),
}
ROLE = {
    'A': 'folds 1 + 2, seed 1044: half of the second-seed TTR224 frac75 set (fold 0 seed 1044 already exists as '
         '`~/fleet_runs/f0_s1044_ttr_frac75`, folds 3 + 4 run in `rsna-train-ttr-frac75-s1044-f34`)',
    'B': 'folds 3 + 4, seed 1044: the other half of the second-seed TTR224 frac75 set (fold 0 seed 1044 already exists '
         'as `~/fleet_runs/f0_s1044_ttr_frac75`, folds 1 + 2 run in `rsna-train-ttr-frac75-s1044-f12`)',
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# 1 the code dataset copy == its own kt_manifest.json (the notebook checks every file against it on Kaggle)
man = json.load(open(f'{DS}/kt_manifest.json'))['files']
bad = [f for f, h in man.items() if sha(f'{DS}/{f}') != h]
assert not bad, f'code dataset copy differs from kt_manifest.json: {bad}'
assert sorted(man) == sorted(f for f in os.listdir(DS) if f not in ('kt_manifest.json', 'dataset-metadata.json'))
bf = json.load(open(f'{DS}/box_facts.json'))
assert all(bf['expected_counts'][str(F)]['iters_per_epoch'] == 1733 for F in range(5))

# 2 the labels dataset copy: sha256 == LIVE == the value baked into c01_lib.py; md5 == PREREG; only Fracture differs
lib = open(f'{KT}/nb_src/c01_lib.py').read()
baked = re.search(r"LABELS_TRAIN_SHA256='([0-9a-f]{64})'", lib).group(1)
baked_md5 = re.search(r"LABELS_TRAIN_MD5='([0-9a-f]{32})'", lib).group(1)
fb = open(f'{DL}/{FRAC}', 'rb').read()
assert hashlib.sha256(fb).hexdigest() == baked == LIVE_SHA, f'{FRAC} sha256 != baked/LIVE'
assert hashlib.md5(fb).hexdigest() == baked_md5 == 'ad7582b817243f8f22e8bcfdd7e3f84a'
mix = open(f'{DS}/labels_mix50_logit.csv', 'rb').read()
a = [l.rsplit(b',', 1) for l in fb.split(b'\n') if l]
m = [l.rsplit(b',', 1) for l in mix.split(b'\n') if l]
assert len(a) == len(m) == 4408 and a[0] == m[0] and all(x[0] == y[0] for x, y in zip(a, m)), 'not mix50 + new Fracture'

# 3 the proven base notebook (kaggle_frac75b A, ran on Kaggle): its code cells are the reference
assert sha(BASE_NB) == BASE_SHA, f'base notebook changed: {BASE_NB}'
base_cells = [''.join(c['source']) for c in json.load(open(BASE_NB, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']


def only_comments_or_plan_differ(new, old):
    """True if the two cell sources differ only in comment-only lines and the NB_PLAN line."""
    keep = lambda s: [l for l in s.splitlines() if not l.lstrip().startswith('#') and not l.startswith('NB_PLAN = ')]
    return keep(new) == keep(old)


# 4 notebooks
PLAN_RE = re.compile(r"^NB_PLAN = dict\(.*\)$", re.M)
assert len(PLAN_RE.findall(lib)) == 1
srcs = sorted(glob.glob(f'{KT}/nb_src/c0*.py'))
built = {}
for key, plan in PLANS.items():
    slug = plan['kernel']
    nbname = slug.replace('-', '_') + '.ipynb'
    tr = ' + '.join(f'fold {F} seed {s} (GPU{i})' for i, (F, s) in enumerate(plan['trainers']))
    outs = ', '.join(f'`f{F}_s{s}_ttr_frac75/`' for F, s in plan['trainers'])
    md = f"""# {slug}: TTR224 frac75 (Fracture-target), {tr}, on Kaggle T4 x2

{ROLE[key]}.

The box TTR224 recipe with ONE change: `--labels labels_frac75_logit.csv` (= mix50 with only the Fracture column re-mixed
.25 v4 + .75 teacher; dataset `eeshsaxena/rsna-knee-frac75-labels`, sha256 baked into cell 1). One (fold, seed) per T4,
in parallel. Full budget: training planned to end by 10.3 h, hard wall 11.4 h, session ends < 11.5 h (push with
`--timeout 43200`). Same code as `rsna-train-ttr-frac75-f23` / `-f4x` (Kaggle 10-07/08: E 21 / 22); only the plan
line in cell 1 differs.

1. locate inputs (code dataset `eeshsaxena/rsna-knee-fleet-train`, labels dataset `eeshsaxena/rsna-knee-frac75-labels`,
   prep kernel outputs `rsna-fleet-prep-s0/-s1`), verify every file by sha256 (frac75 csv: baked sha256 + only-Fracture-differs)
2. environment report (timm < 1.0: bundled wheel)
3. extract the 12 store tars to `/tmp/fleet_store`, prove it equals the box `~/fleet_store` (names, 24 sampled files)
4. parity: contract pixel/tensor sha, select_sha, jpeg_fp, model_sha, decode probe, state_dict keys, pretrained weights, fold counts
5. timing probe: both trainers concurrently, steady s/it, CPU/GPU utilisation
6. `E = min(24, floor((10.3 h - elapsed) / epoch time))` (expected 21; 22 at <= ~0.785 s/it), launch, watchdog (hard wall 11.4 h), receipt
7. summary

Outputs: {outs} (+ `.log`; `epochs/` = last 3 epochs), `run_receipt.json`, `probe/`.
Please do not stop it early: it stops itself before the time limit and keeps everything it trained.
"""
    cells = [dict(cell_type='markdown', metadata={}, source=md.splitlines(True))]
    for p in srcs:
        src = open(p).read().rstrip('\n') + '\n'
        if p.endswith('c01_lib.py'):
            src = PLAN_RE.sub(lambda _: f'NB_PLAN = dict(kernel={slug!r}, trainers={plan["trainers"]!r})', src)
        cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=src.splitlines(True)))
    code = [''.join(c['source']) for c in cells[1:]]
    assert len(code) == len(base_cells) == 8
    assert code[1:] == base_cells[1:], 'code cells 2-8 differ from the proven kaggle_frac75b notebook'
    assert only_comments_or_plan_differ(code[0], base_cells[0]), 'cell 1 differs beyond comments + NB_PLAN'
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                         language_info=dict(name='python')), nbformat=4, nbformat_minor=4)
    for sub, owner in (('kernel', 'eeshsaxena'), ('kernel_teammate', 'YOUR_KAGGLE_USERNAME')):
        kd = f'{KT}/{sub}/{slug}'
        os.makedirs(kd, exist_ok=True)
        json.dump(nb, open(f'{kd}/{nbname}', 'w', newline=chr(10)), indent=1)
        json.dump({'id': f'{owner}/{slug}', 'title': slug, 'code_file': nbname,
                   'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
                   'enable_tpu': False, 'enable_internet': True,
                   'dataset_sources': ['eeshsaxena/rsna-knee-fleet-train', 'eeshsaxena/rsna-knee-frac75-labels'],
                   'kernel_sources': ['eeshsaxena/rsna-fleet-prep-s0', 'eeshsaxena/rsna-fleet-prep-s1'],
                   'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'},
                  open(f'{kd}/kernel-metadata.json', 'w', newline=chr(10)), indent=1)
    assert open(f'{KT}/kernel/{slug}/{nbname}', 'rb').read() == open(f'{KT}/kernel_teammate/{slug}/{nbname}', 'rb').read()
    for c in cells[1:]:
        compile(''.join(c['source']), 'cell', 'exec')
    ns = {}                                     # the plan line as built: folds, seeds, run names
    exec(compile(PLAN_RE.search(''.join(cells[1]['source'])).group(0), 'plan', 'exec'), ns)
    assert ns['NB_PLAN'] == plan, ns['NB_PLAN']
    assert all(s == 1044 for _, s in plan['trainers']) and len(plan['trainers']) == 2
    txt = open(f'{KT}/kernel/{slug}/{nbname}', encoding='utf-8').read() + md
    assert chr(0x2014) not in txt and chr(0x2013) not in txt, 'no em/en dashes'
    built[key] = (slug, nbname, hashlib.sha256(open(f'{KT}/kernel/{slug}/{nbname}', 'rb').read()).hexdigest()[:16])
assert sorted(F for p in PLANS.values() for F, _ in p['trainers']) == [1, 2, 3, 4], 'folds 1-4, each once'
for f in glob.glob(f'{KT}/**/*.md', recursive=True) + glob.glob(f'{KT}/**/*.py', recursive=True):
    t = open(f, encoding='utf-8').read()
    assert chr(0x2014) not in t and chr(0x2013) not in t, f'em/en dash in {f}'
print('code dataset copy == kt_manifest.json (', len(man), 'files ) | labels csv sha256', baked[:16], '== LIVE | code '
      'cells == proven kaggle_frac75b (comments + NB_PLAN aside) | built:')
for k, v in built.items():
    print(f'  {k}: {v[0]}  {v[1]}  sha256 {v[2]}')
