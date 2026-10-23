# guards1020 setup (10-20, Claude): gold58 (train studies, no OAI) runs of c4's Raptor r224 / r384 (GPU, kernel cells 4 / 11 byte-exact
# apart from make_runs.py's path / weights_only / raw-save edits; r384 stays on CUDA here) and R6 (run_b's exact command) with the
# series CSV flags either as shipped (BASE) or as the data-page DEFINITION would set them (DEF: PD/T2 non-FS -> Fluid 1 / Fat 0,
# T1 FS -> Fluid 0 / Fat 1; derived from DICOM headers, g2_def_flags.csv). Writes only under ~/guards1020.
import os, re, json, pandas as pd
H = os.path.expanduser('~/guards1020'); SRC = '/home/training/kmimic/kaggle/input/rsna-knee-abnormality-detection'
F = pd.read_csv(H + '/g2_def_flags.csv', dtype=str).set_index('SeriesInstanceUID')
ser = pd.read_csv(SRC + '/train_series.csv', dtype=str)
m = ser.SeriesInstanceUID.isin(F.index)
assert (ser.loc[m, 'Fluid_Sensitive'].values == F.loc[ser.loc[m, 'SeriesInstanceUID'], 'F_base'].values).all()
dser = ser.copy()
dser.loc[m, 'Fluid_Sensitive'] = F.loc[ser.loc[m, 'SeriesInstanceUID'], 'F_def'].values
dser.loc[m, 'Fat_Suppression'] = F.loc[ser.loc[m, 'SeriesInstanceUID'], 'Fat_def'].values
print('DEF rows changed:', int((dser.Fluid_Sensitive != ser.Fluid_Sensitive).sum() + (dser.Fat_Suppression != ser.Fat_Suppression).sum()))
# R6 shadow data dir
D = H + '/data_def'; os.makedirs(D, exist_ok=True)
for f in ('train.csv', 'train_series'):
    if not os.path.islink(D + '/' + f): os.symlink(SRC + '/' + f, D + '/' + f)
dser.to_csv(D + '/train_series.csv', index=False)
g = pd.read_csv('/home/training/rsna_train/gold74_eval_set.csv'); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); assert len(ids) == 58
for var, S in (('base', ser), ('def', dser)):
    for tag in ('r224', 'r384'):
        RUN = f'{H}/rap_{var}_{tag}'
        comp = f'{RUN}/kaggle/input/rsna-knee-abnormality-detection'; os.makedirs(comp + '/test_series', exist_ok=True)
        os.makedirs(f'{RUN}/kaggle/working', exist_ok=True)
        pd.DataFrame({'StudyInstanceUID': ids}).to_csv(comp + '/test.csv', index=False)
        s = S[S.StudyInstanceUID.isin(ids)]; assert s.StudyInstanceUID.nunique() == 58
        s.to_csv(comp + '/test_series.csv', index=False)
        for u in ids:
            d = comp + '/test_series/' + u
            if not os.path.islink(d): os.symlink(SRC + '/train_series/' + u, d)
        wd = f'{RUN}/kaggle/input/rsna-knee-publication-swa-weights-20261007'; os.makedirs(wd, exist_ok=True)
        for f, p in (('raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt', '/home/training/r224g58/wts/'),
                     ('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt', '/home/training/kfn70_test/wts/')):
            if not os.path.islink(wd + '/' + f): os.symlink(p + f, wd + '/' + f)
        src = open(f'/home/training/r224g58/cell_{tag}.py', encoding='utf-8').read().split('\n', 1)[1]
        src = re.sub(r'(?<![\w.])/kaggle(?=/)', RUN + '/kaggle', src)
        assert src.count('weights_only=False') == 1; src = src.replace('weights_only=False', 'weights_only=True')
        anchor = '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n'
        assert src.count(anchor) == 1
        src = src.replace(anchor, anchor + f'    np.save("{RUN}/kaggle/working/{tag}_raw.npy", arm_probs[0]); pd.DataFrame({{"StudyInstanceUID": test_ids}}).to_csv("{RUN}/kaggle/working/{tag}_ids.csv", index=False)\n')
        open(f'{RUN}/run_{tag}.py', 'w', encoding='utf-8').write(src)
        print(var, tag, 'built', RUN, 'series rows', len(s))
# R6 command = run_b's, with data / assignment / output / deadline / dump replaced
cmd = json.load(open('/home/training/r6g58/run_b/cmd.json'))
rec = json.load(open('/home/training/r6g58/run_b/r6g58_receipt.json'))
W = H + '/r6_def/worker0'; os.makedirs(W, exist_ok=True)
import shutil; shutil.copyfile('/home/training/r6g58/run_b/worker0/ids.json', W + '/ids.json')
rep = {'--data': D, '--assignment': W + '/ids.json', '--output': W, '--dump-logits': W + '/dump.json'}
for i, a in enumerate(cmd[:-1]):
    if a in rep: cmd[i + 1] = rep[a]
env = dict(rec['env_overrides']); env['TORCHINDUCTOR_CACHE_DIR'] = H + '/inductor_cache'
json.dump({'cmd': cmd, 'env': env}, open(H + '/r6_def/run.json', 'w'), indent=1)
print('R6 cmd ready; data', D)
