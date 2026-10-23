# tta_bench.py -- gate G-TTA (BIG_JUMP_PLAN sec 4, build item B5): does 2-pass TTA (pass A + pass B = mirror_views +
# slice offset +0.5) beat pass A alone, measured through the WORKER path on fleet checkpoints?
# Worker path = the member is built exactly as fleet_worker.py builds it (ckpt args -> KneeFleetNet / KneeFleetNetF2 /
# KneeFleetNetTTR, pretrained False, drop_path 0, chunk 48, strict load, eval, channels_last when trained so, fp32
# weights + fp16 autocast on CUDA, one study per forward) and fed exactly the worker's input (fleet_common.decode_stack
# of the stored q92 JPEG bytes == the worker's in-memory decode of the same bytes; sample_triplets; all-zero absent
# views, contrast 3). Pass A here == the worker's prediction (tests/t_b5_tta_bench.py: max |diff| 0 vs the worker on a
# store built from the same DICOMs). Studies come from the training store, so no DICOMs are needed on the box.
# Passes per member (one decode per study): A off 0 | M mirror | P off +.5 | N off -.5 | B mirror + off +.5
# Variants (probability means, as the worker/cell average inside a leg):
#   none = A | mirror = (A+M)/2 | offset = (A+P+N)/3 | A+B = (A+B)/2 | A+B lesion-max = A+B with max(A,B) on
#   ACL, Medial Meniscus, Lateral Meniscus, Contusion, Fracture | all5 = mean of the 5 passes
#   and, with >= 2 members (raw + EMA of one run): raw+EMA = mean of the members' A, raw+EMA A+B
# Metrics: macro AUC (12 labels, label >= .5, NaN masked: == fleet_train.macro_auc) vs --eval_labels on the --fold val
# studies, and vs --gold (rows with source == original_gold58) on gold58; paired bootstrap over studies (--boot) of
# variant - none: mean delta, P(delta > 0); per-label deltas of A+B.
# Gate (printed per member): PASS if A+B - none >= +.002 on val with P > .9 AND gold58 delta >= 0 -> serve pass A+B;
# else FAIL -> serve pass A only and spend the window on members.
# --fp_check N (CUDA only): on the first N val studies, pass A with fp32 weights + fp16 autocast (the worker) vs a
# half-precision backbone (cnn.half(), input cast) vs plain fp32: max |prob diff|.
# usage (box GPU, ~10 studies first; FLEET_DIR = this folder for fleet_common / fleet_model*):
#   python tta_bench.py --store ~/fleet_store --folds ~/rsna_train/folds5.csv --fold 0 \
#       --exclude ~/rsna_train/gold74_eval_set.csv --eval_labels ~/rsna_train/llm_labels_v4_blend.csv \
#       --gold ~/rsna_train/gold74_eval_set.csv --n 10 --out ~/bigjump_build/tta_n10 \
#       ~/fleet_runs/f0_s42/f0_s42_raw_*_best.pt ~/fleet_runs/f0_s42/f0_s42_ema_*_best.pt
#   then the same without --n (867 val + 58 gold; ~0.3-0.5 s per study-pass per member on a shared V100: GUESS)
import os, sys, json, time, glob, argparse, hashlib, inspect
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
import fleet_common as fc
from fleet_model import KneeFleetNet, mirror_views

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
LESION = ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion', 'Fracture']
PLANES = [p for p, _ in fc.VIEWS]
PASSES = {'A': (0.0, False), 'M': (0.0, True), 'P': (0.5, False), 'N': (-0.5, False), 'B': (0.5, True)}
MODS = ('fleet_model', 'fleet_model_f2', 'fleet_model_ttr')


def model_sha(mod='fleet_model'):                    # == fleet_worker._model_sha (bigjump)
    import fleet_model as _fm
    src = chr(10).join(inspect.getsource(_fm).splitlines())
    if mod == 'fleet_model_f2':
        import fleet_model_f2 as _f2
        src = chr(10).join(inspect.getsource(_f2).splitlines()) + chr(10) + src
    elif mod == 'fleet_model_ttr':
        import fleet_model_ttr as _tt
        src = chr(10).join(inspect.getsource(_tt).splitlines()) + chr(10) + src
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def load_member(p, dev, chunk=48):
    """== the fleet_worker.py member build (W2 contract, fp16-trained, W-F2b/W-TTR families, N-B model_sha)."""
    ck = torch.load(p, map_location='cpu', weights_only=False)
    a = ck['args']
    assert a.get('contract') == fc.CONTRACT, (p, 'ckpt contract', a.get('contract'), '!= fleet_common', fc.CONTRACT)
    assert a.get('amp') == 'fp16' and not a.get('bf16', False), (p, 'trained in', a.get('amp'), 'bf16', a.get('bf16'))
    mod = a.get('model_mod', 'fleet_model')
    assert mod in MODS, (p, 'unknown model_mod', mod)
    assert a.get('model_sha') in (None, model_sha(mod)), (p, f'{mod}.py changed since training', a.get('model_sha'))
    kw = dict(backbone=a['backbone'], n_views=a['n_views'], n_slices=a['n_slices'], dim=a['dim'],
              n_contrast=a['n_contrast'], contrast_emb=a['contrast_emb'], pretrained=False, drop_path=0.0, chunk=chunk)
    if mod == 'fleet_model_f2':
        from fleet_model_f2 import KneeFleetNetF2
        m = KneeFleetNetF2(freeze_bn=a['freeze_bn'], channels_last=a.get('channels_last', True), **kw)
    elif mod == 'fleet_model_ttr':
        from fleet_model_ttr import KneeFleetNetTTR
        m = KneeFleetNetTTR(ttr_size=a['ttr_size'], **kw)
    else:
        m = KneeFleetNet(**kw)
    miss, unexp = m.load_state_dict(ck['model'], strict=False)
    assert not miss and not unexp, (p, miss[:3], unexp[:3])
    m = m.eval().to(dev)
    if a.get('channels_last', True):
        m = m.to(memory_format=torch.channels_last)
    info = dict(path=p, name=os.path.basename(p), model_mod=mod, ttr_size=a.get('ttr_size'), fold=a.get('fold'),
                seed=a.get('seed'), train_labels=a.get('train_labels'), epoch_1based=int(ck.get('epoch', -2)) + 1,
                cv=ck.get('cv'), model_sha=a.get('model_sha'))
    del ck
    return m, info


def study_stacks(store, sid):
    """-> [(decoded stack uint8 (n, S, S), ctr) or None per view]: the worker's st[v] for this study."""
    out = []
    with np.load(f'{store}/{sid}.npz') as z:
        for j in range(len(fc.VIEWS)):
            if f'v{j}_blob' in z.files:
                out.append((fc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']), int(z[f'v{j}_ctr'])))
            else:
                out.append(None)
    return out


def make_input(st, off, mirror, dev):
    """== fleet_worker.py input build (+ the pass's slice offset / mirror_views on the raw uint8 tensor)."""
    vs, pr, ct = [], [], []
    for g in st:
        if g is None:
            vs.append(np.zeros((fc.N_SLICES, 3, fc.STORE, fc.STORE), np.uint8)); pr.append(False); ct.append(3)
        else:
            vs.append(fc.sample_triplets(g[0], fc.N_SLICES, off))
            pr.append(True); ct.append(g[1])
    x = torch.from_numpy(np.stack(vs)[None])
    if mirror:
        x = mirror_views(x, PLANES)
    return x.to(dev), torch.tensor([pr], device=dev), torch.tensor([ct], device=dev), int(sum(pr))


def forward(m, x, pr, ct, dev, mode='worker'):
    with torch.no_grad():
        if mode == 'worker':                                     # fleet_worker.py: fp32 weights + fp16 autocast
            with torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                return torch.sigmoid(m(x, pr, ct).float()).cpu().numpy()[0]
        return torch.sigmoid(m(x, pr, ct).float()).cpu().numpy()[0]       # fp32 / half-backbone (fp_check)


def _auc(y, s):                                      # rank AUC with average ties (== sklearn roc_auc_score)
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auc(P, Y, idx=None):
    """== fleet_train.macro_auc (T9): per target, finite label AND pred, label >= .5, both classes present."""
    if idx is not None:
        P, Y = P[idx], Y[idx]
    aucs = []
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        if ok.sum() and 0 < yb.mean() < 1:
            aucs.append(_auc(yb, P[ok, j]))
    return (float(np.mean(aucs)) if aucs else float('nan')), len(aucs)


def per_label(P, Y):
    out = {}
    for j, t in enumerate(TARGETS):
        ok = np.isfinite(Y[:, j]) & np.isfinite(P[:, j]); yb = Y[ok, j] >= 0.5
        out[t] = float(_auc(yb, P[ok, j])) if ok.sum() and 0 < yb.mean() < 1 else float('nan')
    return out


def boot(Pv, P0, Y, B, seed=0):
    """paired bootstrap over studies of macro_auc(Pv) - macro_auc(P0)."""
    if len(Y) < 3:
        return dict(mean=float('nan'), p_gt0=float('nan'), lo=float('nan'), hi=float('nan'))
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        i = rng.integers(0, len(Y), len(Y))
        a, _ = macro_auc(Pv, Y, i); b, _ = macro_auc(P0, Y, i)
        if np.isfinite(a) and np.isfinite(b):
            d.append(a - b)
    d = np.array(d)
    if not len(d):
        return dict(mean=float('nan'), p_gt0=float('nan'), lo=float('nan'), hi=float('nan'))
    return dict(mean=float(d.mean()), p_gt0=float((d > 0).mean()), lo=float(np.quantile(d, .05)),
                hi=float(np.quantile(d, .95)))


def variants(R):
    """R: {pass: (n, 12)} of one member -> {variant: (n, 12)}."""
    v = {'none': R['A'], 'mirror': (R['A'] + R['M']) / 2, 'offset': (R['A'] + R['P'] + R['N']) / 3,
         'A+B': (R['A'] + R['B']) / 2}
    lm = v['A+B'].copy()
    for t in LESION:
        j = TARGETS.index(t); lm[:, j] = np.maximum(R['A'][:, j], R['B'][:, j])
    v['A+B lesion-max'] = lm
    v['all5'] = sum(R[k] for k in PASSES) / len(PASSES)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--store', required=True); ap.add_argument('--folds', required=True)
    ap.add_argument('--fold', type=int, default=0); ap.add_argument('--exclude', default='')
    ap.add_argument('--eval_labels', required=True); ap.add_argument('--gold', default='')
    ap.add_argument('--n', type=int, default=0, help='first N val and first N gold58 studies (0 = all)')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--chunk', type=int, default=48)
    ap.add_argument('--passes', default='A,M,P,N,B'); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--fp_check', type=int, default=16); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--oof', nargs='*', default=[], help='trainer OOF csv per ckpt (same order): pass-A parity read')
    ap.add_argument('--out', required=True)
    A = ap.parse_args(argv)
    t0 = time.time()
    dev = A.device
    if dev == 'cuda':
        assert torch.cuda.is_available(), 'no CUDA (pass --device cpu for a smoke run)'
    passes = [p for p in A.passes.split(',') if p]
    assert set(passes) <= set(PASSES) and 'A' in passes, passes
    os.makedirs(A.out, exist_ok=True)
    ev = pd.read_csv(A.eval_labels, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    folds = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
    excl = set(pd.read_csv(A.exclude, dtype={'StudyInstanceUID': str}).StudyInstanceUID) if A.exclude else set()
    held = set(folds.index[folds < 0]) | excl
    have = lambda s: os.path.isfile(f'{A.store}/{s}.npz')
    val = [s for s in folds.index[folds == A.fold] if s not in held and s in ev.index and have(s)]   # trainer va_ids order
    val = [s for s in ev.index if s in set(val)]
    gold_tab = pd.read_csv(A.gold, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID') if A.gold else None
    g58 = ([s for s in gold_tab.index if have(s) and gold_tab.loc[s, 'source'] == 'original_gold58']
           if gold_tab is not None and 'source' in gold_tab.columns else [])
    if A.n:
        val, g58 = val[:A.n], g58[:A.n]
    sids = list(dict.fromkeys(val + g58))
    members = [load_member(p, dev, A.chunk) for p in A.ckpts]
    if len({i['name'] for _, i in members}) < len(members):              # same basename from two run dirs
        for k, (_, i) in enumerate(members):
            i['name'] = f'{k}:{i["name"]}'
    print(f'tta_bench | {len(members)} member(s) {[i["name"] for _, i in members]} | val {len(val)} (fold {A.fold}) '
          f'| gold58 {len(g58)} | passes {passes} | dev {dev} chunk {A.chunk} | torch {torch.__version__}', flush=True)
    R = {i['name']: {p: np.full((len(sids), 12), np.nan, np.float32) for p in passes} for _, i in members}
    NV = np.zeros(len(sids), np.int64)
    pool = ThreadPoolExecutor(A.threads)
    futs = {k: pool.submit(study_stacks, A.store, s) for k, s in enumerate(sids[:2 * A.threads])}
    nxt = len(futs)
    for k, s in enumerate(sids):
        st = futs.pop(k).result()
        if nxt < len(sids):
            futs[nxt] = pool.submit(study_stacks, A.store, sids[nxt]); nxt += 1
        for p in passes:
            off, mir = PASSES[p]
            x, pr, ct, nv = make_input(st, off, mir, dev)
            NV[k] = nv
            if nv == 0:
                continue                                                     # W4: row stays NaN
            for m, info in members:
                R[info['name']][p][k] = forward(m, x, pr, ct, dev)
        if (k + 1) % 50 == 0 or k + 1 == len(sids):
            print(f'  {k + 1}/{len(sids)} studies {(time.time() - t0) / 60:.1f} min', flush=True)
    pool.shutdown()
    np.savez(f'{A.out}/tta_passes.npz', sids=np.array(sids), NV=NV, names=np.array([i['name'] for _, i in members]),
             **{f'{n}|{p}': v for n, d in R.items() for p, v in d.items()})
    iv = [sids.index(s) for s in val]; ig = [sids.index(s) for s in g58]
    Yv = ev.loc[val, TARGETS].to_numpy(np.float64)
    Yg = gold_tab.loc[g58, TARGETS].to_numpy(np.float64) if g58 else np.zeros((0, 12))
    summ = dict(n_val=len(val), n_gold58=len(g58), passes=passes, members=[i for _, i in members], results={}, gate={})
    full = set(passes) == set(PASSES)
    combos = {}
    for n, d in R.items():
        combos[n] = variants(d) if full else {'none': d['A']}
    if len(members) >= 2:
        names = [i['name'] for _, i in members]
        combos['raw+EMA'] = {'none': sum(R[n]['A'] for n in names) / len(names)}
        if full:
            combos['raw+EMA']['A+B'] = sum((R[n]['A'] + R[n]['B']) / 2 for n in names) / len(names)
    base_name = members[0][1]['name']
    for cname, vs in combos.items():
        res = {}
        ref = combos[cname]['none'] if cname != 'raw+EMA' else combos[base_name]['none']
        for vname, P in vs.items():
            r = {}
            r['val'], r['val_nlab'] = macro_auc(P[iv], Yv)
            r['gold58'], _ = macro_auc(P[ig], Yg) if ig else (float('nan'), 0)
            if not (vname == 'none' and cname != 'raw+EMA'):
                r['val_d'] = boot(P[iv], ref[iv], Yv, A.boot)
                r['gold58_d'] = boot(P[ig], ref[ig], Yg, A.boot) if ig else None
            if vname == 'A+B':
                pv, p0 = per_label(P[iv], Yv), per_label(ref[iv], Yv)
                r['val_per_label_d'] = {t: round(pv[t] - p0[t], 5) for t in TARGETS}
            res[vname] = r
        summ['results'][cname] = res
        if 'A+B' in res and cname != 'raw+EMA':
            r = res['A+B']; g = r.get('gold58_d') or {}
            dv = r['val'] - res['none']['val']
            dg = (r['gold58'] - res['none']['gold58']) if ig else float('nan')
            ok = dv >= 0.002 and r['val_d']['p_gt0'] > 0.9 and (not ig or dg >= 0)
            summ['gate'][cname] = dict(val_delta=dv, p_gt0=r['val_d']['p_gt0'], gold58_delta=dg,
                                       verdict='PASS (serve A+B)' if ok else 'FAIL (serve pass A only)')
    # pass-A parity vs the trainer's own OOF (GPU: fp16 noise ~1e-4; CPU fp32 at equal chunk: 0)
    for (m, info), f in zip(members, A.oof):
        o = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        common = [s for s in val if s in o.index]
        d = float(np.nanmax(np.abs(R[info['name']]['A'][[sids.index(s) for s in common]]
                                   - o.loc[common, TARGETS].to_numpy(np.float32)))) if common else float('nan')
        summ.setdefault('oof_parity', {})[info['name']] = dict(n=len(common), max_abs=d)
        print(f'pass A vs trainer OOF {os.path.basename(f)}: {len(common)} studies, max|diff| {d:.3g}', flush=True)
    if dev == 'cuda' and A.fp_check and val:
        import copy
        fpc = {}
        for m, info in members:
            mh = copy.deepcopy(m); mh.cnn.half()
            mh.cnn.register_forward_pre_hook(lambda mod, inp: tuple(t.half() for t in inp))
            d16, d32 = 0.0, 0.0
            for s in val[:A.fp_check]:
                x, pr, ct, nv = make_input(study_stacks(A.store, s), 0.0, False, dev)
                if nv == 0:
                    continue
                pa = forward(m, x, pr, ct, dev, 'worker'); p32 = forward(m, x, pr, ct, dev, 'fp32')
                ph = forward(mh, x, pr, ct, dev, 'half')
                d16 = max(d16, float(np.abs(pa - ph).max())); d32 = max(d32, float(np.abs(pa - p32).max()))
            fpc[info['name']] = dict(n=min(A.fp_check, len(val)), autocast_vs_halfbackbone=d16, autocast_vs_fp32=d32)
            del mh
        summ['fp_check'] = fpc
    summ['minutes'] = round((time.time() - t0) / 60, 2)
    json.dump(summ, open(f'{A.out}/tta_summary.json', 'w'), indent=1, default=str)
    print(f'\n{"member/combo":44s} {"variant":15s} {"val AUC":>8s} {"d val":>8s} {"P>0":>5s} {"gold58":>7s} {"d g58":>8s}')
    for cname, res in summ['results'].items():
        for vname, r in res.items():
            vd = r.get('val_d') or {}; gd = r.get('gold58_d') or {}
            print(f'{cname[:44]:44s} {vname:15s} {r["val"]:8.4f} {vd.get("mean", float("nan")):+8.4f} '
                  f'{vd.get("p_gt0", float("nan")):5.2f} {r["gold58"]:7.4f} {gd.get("mean", float("nan")):+8.4f}')
    for cname, res in summ['results'].items():
        if 'A+B' in res and 'val_per_label_d' in res['A+B']:
            print(f'A+B per-label val delta [{cname[:40]}]: ' + ', '.join(f'{t} {v:+.4f}' for t, v in res['A+B']['val_per_label_d'].items()))
    for cname, g in summ['gate'].items():
        print(f'G-TTA {cname}: A+B - none = {g["val_delta"]:+.4f} val (P>0 {g["p_gt0"]:.2f}), gold58 {g["gold58_delta"]:+.4f} -> {g["verdict"]}')
    if 'fp_check' in summ:
        print('fp_check (max |prob diff| on', A.fp_check, 'val studies):', json.dumps(summ['fp_check']))
    print(f'tta_bench done in {summ["minutes"]} min -> {A.out}/tta_summary.json', flush=True)
    return summ


if __name__ == '__main__':
    main()
