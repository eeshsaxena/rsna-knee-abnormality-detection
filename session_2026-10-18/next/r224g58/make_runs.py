# Builds two kaggle-mimic run dirs on the box for the 58 gold studies and writes the kernel's r224 / r384 scoring sources
# with ONLY these edits: (1) drop the %%writefile line, (2) /kaggle -> RUN/kaggle, (3) torch.load weights_only=True (same
# tensors; no pickle code execution), (4) also save the raw per-study probabilities, (5) r384 only: CPU instead of CUDA
# (fp32 instead of fp16 autocast; serving geometry, k=94 windows and the mirror TTA unchanged).
import os, re, sys, pandas as pd
H = os.path.expanduser('~/r224g58'); SRC = '/home/training/kmimic/kaggle/input/rsna-knee-abnormality-detection'
g = pd.read_csv('/home/training/rsna_train/gold74_eval_set.csv'); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); assert len(ids) == 58
ser = pd.read_csv(SRC + '/train_series.csv', dtype=str)
for tag in ('r224', 'r384'):
    RUN = f'{H}/run_{tag}'
    comp = f'{RUN}/kaggle/input/rsna-knee-abnormality-detection'; os.makedirs(comp + '/test_series', exist_ok=True)
    os.makedirs(f'{RUN}/kaggle/working', exist_ok=True)
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(comp + '/test.csv', index=False)
    s = ser[ser.StudyInstanceUID.isin(ids)]; assert s.StudyInstanceUID.nunique() == 58
    s.to_csv(comp + '/test_series.csv', index=False)
    for u in ids:
        d = comp + '/test_series/' + u
        if not os.path.islink(d): os.symlink(SRC + '/train_series/' + u, d)
    wd = f'{RUN}/kaggle/input/rsna-knee-publication-swa-weights-20261007'; os.makedirs(wd, exist_ok=True)
    for f, p in (('raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt', H + '/wts/'),
                 ('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt', '/home/training/kfn70_test/wts/')):
        if not os.path.islink(wd + '/' + f): os.symlink(p + f, wd + '/' + f)
    src = open(f'{H}/cell_{tag}.py', encoding='utf-8').read().split('\n', 1)[1]
    n0 = len(re.findall(r'(?<![\w.])/kaggle(?=/)', src))
    src = re.sub(r'(?<![\w.])/kaggle(?=/)', RUN + '/kaggle', src)
    assert src.count('weights_only=False') == 1; src = src.replace('weights_only=False', 'weights_only=True')
    anchor = '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n'
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + f'    np.save("{RUN}/kaggle/working/{tag}_raw.npy", arm_probs[0]); pd.DataFrame({{"StudyInstanceUID": test_ids}}).to_csv("{RUN}/kaggle/working/{tag}_ids.csv", index=False)\n')
    if tag == 'r384':
        chk = 'if not torch.cuda.is_available():\n    raise RuntimeError("This scored pipeline requires CUDA; CPU inference is not enabled.")\n'
        assert src.count(chk) == 1
        src = src.replace(chk, 'torch.set_num_threads(int(os.environ.get("NTHR", "24")))   # [g58 CPU run]\n'.replace('os.environ', '__import__("os").environ'))
        a = '    assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"\n'
        assert src.count(a) == 1
        src = src.replace(a, '    if not str(device).startswith("cuda"):   # [g58 CPU run] fp32 on CPU, same model and windows\n'
                             '        return torch.sigmoid(model(xwins.unsqueeze(0)).float())[0].numpy()\n')
    open(f'{RUN}/run_{tag}.py', 'w', encoding='utf-8').write(src)
    print(tag, 'paths rewritten', n0, '->', RUN)
