# wk_stage.py (gold_prep/r50, CPU): stage a K5-style FLEET_DIR + mini competition root for the R50 dry-run checkpoint,
# so the UNCHANGED K5 fleet_worker.py (shipped bytes, md5 47e0f161...) can load and serve it on CPU.
# The ONLY edit to the checkpoint copy: args['amp'] 'fp32' -> 'fp16' (a --device cpu run records fp32; the real V100
# run records fp16 and the worker refuses anything else). Model weights, model_sha, contract, every other arg untouched.
import glob, json, os, shutil, sys, hashlib
import pandas as pd, torch
H = os.path.expanduser('~'); G = f'{H}/gold_prep/r50'; RUN = sys.argv[1]; TAG = sys.argv[2]   # one tag per run (worker W1 refuses raw+ema twins)
WK = f'{G}/wk'; DS = f'{WK}/ds'; COMP = f'{WK}/comp'
MIR = f'{H}/kmimic/kaggle/input/rsna-knee-abnormality-detection'
shutil.rmtree(WK, ignore_errors=True); os.makedirs(DS); os.makedirs(f'{COMP}/test_series')
for f in glob.glob(f'{G}/k5code/*'):
    shutil.copy2(f, DS)
cks = sorted(glob.glob(f'{RUN}/f0_s243_{TAG}_*_best.pt'))
assert cks, f'no best ckpt in {RUN}'
names = []
for p in cks:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck['args']
    print(os.path.basename(p), '| amp', a['amp'], '| model_mod', a.get('model_mod'), '| model_sha', a.get('model_sha'),
          '| freeze_bn', a.get('freeze_bn'), '| channels_last', a.get('channels_last'), '| norm', a.get('norm'),
          '| epochs', a.get('epochs'), '| save_epochs', a.get('save_epochs'), '| drop_path', a.get('drop_path'),
          '| train_labels', a.get('train_labels'), '| sel', a.get('sel_labels'), '| seed', a.get('seed'), '| fold', a.get('fold'),
          '| cv', ck['cv'], '| epoch', ck['epoch'], '| keys', len(ck['model']))
    a['amp'] = 'fp16'
    n = os.path.basename(p)
    torch.save(ck, f'{DS}/{n}'); names.append(n)
man = {'checkpoints': names, 'sources': {n: {'original': n, 'run_dir': os.path.basename(RUN), 'model_mod': 'fleet_model_f2',
                                              'tta': 'AB'} for n in names}}
json.dump(man, open(f'{DS}/fleet_manifest.json', 'w'), indent=1)
va = pd.read_csv(f'{G}/pick_val.csv', dtype={'StudyInstanceUID': str})
ids = va.StudyInstanceUID[va.in_mirror].tolist()
trs = pd.read_csv(f'{MIR}/train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
pd.DataFrame({'StudyInstanceUID': ids}).to_csv(f'{COMP}/test.csv', index=False)
trs[trs.StudyInstanceUID.isin(ids)].to_csv(f'{COMP}/test_series.csv', index=False)
for s in ids:
    os.symlink(f'{MIR}/train_series/{s}', f'{COMP}/test_series/{s}')
# no train_series in COMP: W7 parity re-encode is skipped exactly as in the K5 A0 box test (the box mirror lacks DICOMs for
# parity study ...399585; the Kaggle commit run gates W7 on the real data)
print('staged', names, '| test studies', [s[-6:] for s in ids])
