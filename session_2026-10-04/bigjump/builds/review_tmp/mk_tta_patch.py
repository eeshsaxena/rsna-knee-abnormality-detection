p = 'review_tmp/tta_bench.py'
s = open('code/tta_bench.py', encoding='utf-8', newline='').read().replace('\r\n', '\n')
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)
rep("""import numpy as np, pandas as pd, torch
import fleet_common as fc""", """import numpy as np, pandas as pd, torch
try:                                                  # R-1 == fleet_worker K2d: output identical, and no 40-thread cv2
    import cv2; cv2.setNumThreads(1)                  #     pools competing with the training DataLoaders
except Exception:
    pass
import fleet_common as fc""")
rep('''    """R: {pass: (n, 12)} of one member -> {variant: (n, 12)}."""
    v = {'none': R['A'], 'mirror': (R['A'] + R['M']) / 2, 'offset': (R['A'] + R['P'] + R['N']) / 3,
         'A+B': (R['A'] + R['B']) / 2}
    lm = v['A+B'].copy()
    for t in LESION:
        j = TARGETS.index(t); lm[:, j] = np.maximum(R['A'][:, j], R['B'][:, j])
    v['A+B lesion-max'] = lm
    v['all5'] = sum(R[k] for k in PASSES) / len(PASSES)
    return v''', '''    """R: {pass: (n, 12)} of one member -> {variant: (n, 12)}; R-2: only the variants whose passes were run."""
    v = {'none': R['A']}
    if 'M' in R:
        v['mirror'] = (R['A'] + R['M']) / 2
    if 'P' in R and 'N' in R:
        v['offset'] = (R['A'] + R['P'] + R['N']) / 3
    if 'B' in R:
        v['A+B'] = (R['A'] + R['B']) / 2
        lm = v['A+B'].copy()
        for t in LESION:
            j = TARGETS.index(t); lm[:, j] = np.maximum(R['A'][:, j], R['B'][:, j])
        v['A+B lesion-max'] = lm
    if set(PASSES) <= set(R):
        v['all5'] = sum(R[k] for k in PASSES) / len(PASSES)
    return v''')
rep("""    ap.add_argument('--out', required=True)
    A = ap.parse_args(argv)""", """    ap.add_argument('--out', required=True)
    ap.add_argument('--min_free_gb', type=float, default=4.0, help='R-3: refuse to start below this free GPU memory')
    ap.add_argument('--max_gpu_gb', type=float, default=3.0, help='R-3: hard cap on THIS process (it OOMs, not a trainer)')
    ap.add_argument('--max_minutes', type=float, default=0.0, help='R-4: stop the pass loop after N min, score what is done')
    A = ap.parse_args(argv)""")
rep("""    if dev == 'cuda':
        assert torch.cuda.is_available(), 'no CUDA (pass --device cpu for a smoke run)'
    passes""", """    if dev == 'cuda':
        assert torch.cuda.is_available(), 'no CUDA (pass --device cpu for a smoke run)'
        fr, tot = torch.cuda.mem_get_info()                                          # R-3
        assert fr / 2**30 >= A.min_free_gb, (f'GPU free {fr / 2**30:.1f} GB < --min_free_gb {A.min_free_gb}: refusing '
                                             f'(the running trainers need the headroom)')
        torch.cuda.set_per_process_memory_fraction(min(1.0, A.max_gpu_gb * 2**30 / tot))
        print(f'GPU free {fr / 2**30:.1f} / {tot / 2**30:.1f} GB at start; this process capped at {A.max_gpu_gb} GB', flush=True)
    passes""")
rep("""    sids = list(dict.fromkeys(val + g58))""", """    inter = [s for pair in zip(val, g58) for s in pair] + val[len(g58):] + g58[len(val):]   # R-4: a time cut keeps both sets
    sids = list(dict.fromkeys(inter))""")
rep("""    for k, s in enumerate(sids):
        st = futs.pop(k).result()""", """    n_run = len(sids)
    for k, s in enumerate(sids):
        if A.max_minutes and (time.time() - t0) / 60 > A.max_minutes:                  # R-4
            n_run = k
            print(f'!! --max_minutes {A.max_minutes} reached after {k}/{len(sids)} studies: scoring those only', flush=True)
            break
        st = futs.pop(k).result()""")
rep("""    pool.shutdown()
""", """    pool.shutdown(wait=False, cancel_futures=True)
    if n_run < len(sids):                                                              # R-4
        done = set(sids[:n_run]); val = [s for s in val if s in done]; g58 = [s for s in g58 if s in done]
""")
rep("""    full = set(passes) == set(PASSES)
    combos = {}
    for n, d in R.items():
        combos[n] = variants(d) if full else {'none': d['A']}
    if len(members) >= 2:
        names = [i['name'] for _, i in members]
        combos['raw+EMA'] = {'none': sum(R[n]['A'] for n in names) / len(names)}
        if full:""", """    combos = {}
    for n, d in R.items():
        combos[n] = variants(d)                                                        # R-2
    if len(members) >= 2:
        names = [i['name'] for _, i in members]
        combos['raw+EMA'] = {'none': sum(R[n]['A'] for n in names) / len(names)}
        if 'B' in passes:""")
open(p, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
open('review_tmp/tta_bench_orig_lf.py', 'w', encoding='utf-8', newline='').write(open('code/tta_bench.py', encoding='utf-8', newline='').read().replace('\r\n', '\n'))
open('review_tmp/tta_bench_new_lf.py', 'w', encoding='utf-8', newline='').write(s)
print('patched ok')
