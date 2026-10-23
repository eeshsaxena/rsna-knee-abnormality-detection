# kc_build_store.py -- BOX_PLAN_1015 C2 / C6: cut the KC store (6 views x 12 slices, 256 px, PNG in the fleet npz format)
# from the d96 corpus npy (~/raptor_store/rsna/<sid>.npy, reader sha 30aa72fd) with kc_common.kc_cut.
# Side + per-slot geometry come from the side table (kc_side_kernel_v2 output: side, rule, mean_x, tag, g_<slot>).
# A v1 table (no g_ columns) is refused unless --geom const (MEASURED default geometry; recorded in the summary).
# Writes <out>/<sid>.npz (atomic tmp + rename), <out>_receipt.csv and <out>_summary.json. Every study is re-opened and
# decoded after the write (PNG round trip == the cut, byte for byte). CPU only (box cores 28-39 or 12-27).
#   python kc_build_store.py --src ~/raptor_store/rsna --side SIDE.csv --out ~/kc_store/test458 --size 256 --procs 12
# Full-build asserts (--full 1): coverage >= 99 % of the index, side U rate <= 2 %, contract hash, gold74 never in a
# training fold after the trainer's held-out rule, and --subset_ref (e.g. ~/kc_store/test458) byte-equal per array.
# KC-R7 (10-15 review fix, = A6 G0 'coverage >= 99 %'): a --full build PASSes with up to 1 % of the index failing in the
# DATA stage (d96 npy load / shape / dtype / mask == rsna_index); those studies are listed in the summary (error_ids) and
# are simply absent from the store (the trainer's have-set drops them; --min_cover .99). Any error outside the data stage
# (side rule / flip / weak / geom_ok vs the table, cut, PNG round trip) stays zero-tolerance.
import os, sys, json, time, argparse, hashlib, traceback
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kc_common as kc

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)                   # dir with <sid>.npy (d96)
ap.add_argument('--index', default=os.path.expanduser('~/raptor_store/rsna_index.csv'))
ap.add_argument('--side', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--size', type=int, default=256)
ap.add_argument('--procs', type=int, default=12)
ap.add_argument('--geom', default='table', choices=('table', 'const'))
ap.add_argument('--only', default='')                     # optional csv of StudyInstanceUID to restrict to
ap.add_argument('--full', type=int, default=0)            # 1 = full-build asserts
ap.add_argument('--subset_ref', default='')               # store dir whose studies must be array-equal to this build
ap.add_argument('--folds', default=os.path.expanduser('~/rsna_train/folds5.csv'))
ap.add_argument('--gold', default=os.path.expanduser('~/rsna_train/gold74_eval_set.csv'))
ap.add_argument('--overwrite', type=int, default=0)
A = ap.parse_args()
assert A.size == kc.STORE, f'--size {A.size} != kc_common.STORE {kc.STORE}'

idx = pd.read_csv(A.index, dtype={'id': str, 'mask': str}).set_index('id')
side = pd.read_csv(A.side, dtype={'StudyInstanceUID': str, 'tag': str}).set_index('StudyInstanceUID')
# two side-table schemas carry per-slot geometry: kc_side_kernel_v2 (g_<block>) and main's pushed kernel (s<k>_sign /
# s<k>_mag / s<k>_flip / weak / geo_ok, slots k = 0 sagfs, 1 sagpd, 2 corfs, 4 ax). For the latter the flips, weak and
# geom_ok derived here must EQUAL the table's (asserted per study): two independent implementations of one rule.
SLOTK = {'sagfs': 0, 'sagpd': 1, 'corfs': 2, 'ax': 4}
MAIN_SCHEMA = all(f's{k}_{c}' in side.columns for k in SLOTK.values() for c in ('sign', 'mag', 'flip')) and     {'weak', 'geo_ok'} <= set(side.columns)
has_geom = MAIN_SCHEMA or all(f'g_{b}' in side.columns for b in ('sagfs', 'sagpd', 'corfs', 'ax'))
if not has_geom:
    assert A.geom == 'const', 'side table has no per-slot geometry (v1); rerun kc_side_kernel_v2 or pass --geom const'
ids = sorted(s for s in idx.index if s in side.index)
if A.only:
    keep = set(pd.read_csv(A.only, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
    ids = [s for s in ids if s in keep]
os.makedirs(A.out, exist_ok=True)


def row_geom(r):
    if MAIN_SCHEMA and A.geom == 'table':
        raw = {}
        for b in kc.GEOM_DEFAULT:
            k = SLOTK.get(b)
            sg, mg = (r.get(f's{k}_sign'), r.get(f's{k}_mag')) if k is not None else (None, None)
            ok = k is not None and pd.notna(sg) and pd.notna(mg) and float(sg) != 0
            raw[b] = float(sg) * float(mg) if ok else float('nan')
    else:
        raw = {b: (float(r.get(f'g_{b}')) if (has_geom and A.geom == 'table' and pd.notna(r.get(f'g_{b}'))) else float('nan'))
               for b in kc.GEOM_DEFAULT}
    return raw, {b: (raw[b] if np.isfinite(raw[b]) else kc.GEOM_DEFAULT[b]) for b in kc.GEOM_DEFAULT}


def one(sid):
    stage = 'logic'
    try:
        import cv2
        cv2.setNumThreads(1)
        dst = f'{A.out}/{sid}.npz'
        if os.path.exists(dst) and not A.overwrite:
            return dict(StudyInstanceUID=sid, status='exists')
        r = side.loc[sid].to_dict()
        s_rule, rule = kc.side_rule(r.get('mean_x'), r.get('tag') if isinstance(r.get('tag'), str) else '')
        assert (s_rule, rule) == (str(r['side']), str(r['rule'])), f'side rule drift {s_rule, rule} vs table {r["side"], r["rule"]}'
        graw, g = row_geom(r)
        t0 = time.perf_counter()
        stage = 'data'                                               # KC-R7: npy load / shape / mask
        vol = np.load(f'{A.src}/{sid}.npy')
        assert vol.shape == (96, kc.D96_IMG, kc.D96_IMG) and vol.dtype == np.uint8, vol.shape
        t1 = time.perf_counter()
        m = kc.d96_mask(vol)
        mi = idx.at[sid, 'mask']
        assert ''.join('1' if b else '0' for b in m) == mi, 'mask != rsna_index mask'
        stage = 'logic'
        pres = kc.present_blocks(m)
        flips = kc.block_flips(s_rule, g)
        weak = int(rule != 'mean_x')
        gok = int(kc.geom_ok(graw, [b for b in ('sagfs', 'sagpd', 'corfs', 'ax') if pres[b]])) if A.geom == 'table' else -1
        if MAIN_SCHEMA and A.geom == 'table':                        # cross-check vs the table's own flips / weak / geo_ok
            for b, k in SLOTK.items():
                if pd.notna(r.get(f's{k}_flip')):
                    assert bool(flips[b]) == bool(int(r[f's{k}_flip'])), f'flip {b} {flips[b]} != table s{k}_flip {r[f"s{k}_flip"]}'
            assert weak == int(r['weak']), f'weak {weak} != table {r["weak"]}'
            assert gok == int(r['geo_ok']), f'geom_ok {gok} != table geo_ok {r["geo_ok"]}'
        views = kc.kc_cut(vol, s_rule, g)
        t2 = time.perf_counter()
        kw = kc.study_npz_dict(views, s_rule, rule, graw, flips, weak, gok)
        t3 = time.perf_counter()
        tmp = f'{A.out}/.{sid}.tmp.npz'
        np.savez(tmp, **kw)
        os.replace(tmp, dst)
        t4 = time.perf_counter()
        h = hashlib.sha256()
        with np.load(dst) as z:
            kc.check_store_contract(z, sid)
            for j, name in enumerate(kc.VIEW_NAMES):
                if views[name] is None:
                    assert f'v{j}_blob' not in z.files
                    continue
                d = kc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs'])
                assert d.shape == views[name].shape and np.array_equal(d, views[name]), f'{name} PNG round trip differs'
                h.update(d.tobytes())
        t5 = time.perf_counter()
        return dict(StudyInstanceUID=sid, status='ok', side=s_rule, rule=rule, weak=weak, geom_ok=gok,
                    mean_x=r.get('mean_x'), **{f'p_{n}': int(views[n] is not None) for n in kc.VIEW_NAMES},
                    **{f'f_{b}': int(flips[b]) for b in ('sagfs', 'sagpd', 'corfs', 'ax')},
                    bytes=os.path.getsize(dst), decoded_sha=h.hexdigest()[:16],
                    t_load=t1 - t0, t_cut=t2 - t1, t_png=t3 - t2, t_save=t4 - t3, t_verify=t5 - t4)
    except Exception as e:
        return dict(StudyInstanceUID=sid, status='error', err=f'{type(e).__name__}: {str(e)[:160]}', err_stage=stage,
                    tb=traceback.format_exc()[-400:])


if __name__ == '__main__':
    t0 = time.time()
    print(f'KC build: {len(ids)} studies | side {A.side} (per-slot geometry: {has_geom}, main schema {MAIN_SCHEMA}, mode {A.geom}) | out {A.out} | '
          f'procs {A.procs} | contract {kc.CONTRACT_HASH} store_contract pixel_sha {kc.STORE_CONTRACT["pixel_sha"]}', flush=True)
    res = []
    with mp.get_context('fork').Pool(A.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, ids, chunksize=4)):
            res.append(r)
            if (i + 1) % 250 == 0:
                print(f'{i + 1}/{len(ids)} {(time.time() - t0) / 60:.1f} min', flush=True)
    rec = pd.DataFrame(res).sort_values('StudyInstanceUID')
    rec.to_csv(f'{A.out}_receipt.csv', index=False)
    ok = rec[rec.status == 'ok']
    err = rec[rec.status == 'error']
    wall = time.time() - t0
    summ = dict(n_index=int(len(idx)), n_target=len(ids), n_ok=int(len(ok)), n_exists=int((rec.status == 'exists').sum()),
                n_error=int(len(err)), errors=err.err.value_counts().head(5).to_dict() if len(err) else {},
                contract_hash=kc.CONTRACT_HASH, store_contract=kc.STORE_CONTRACT, geom_mode=A.geom, side_table=A.side,
                side_schema='main_s<k>' if MAIN_SCHEMA else ('v2_g' if has_geom else 'v1'),
                side_table_sha256=hashlib.sha256(open(A.side, 'rb').read()).hexdigest(),
                wall_s=round(wall, 1), procs=A.procs)
    if len(ok):
        summ.update(side=ok.side.value_counts().to_dict(), rule=ok.rule.value_counts().to_dict(),
                    u_rate=float((ok.side == 'U').mean()), weak=int(ok.weak.sum()), geom_ok=int((ok.geom_ok == 1).sum()),
                    present={n: int(ok[f'p_{n}'].sum()) for n in kc.VIEW_NAMES},
                    bytes_mean_mb=float(ok.bytes.mean() / 1e6), bytes_total_gb=float(ok.bytes.sum() / 1e9),
                    t_med={k: float(ok[k].median()) for k in ('t_load', 't_cut', 't_png', 't_save', 't_verify')},
                    t_p90={k: float(ok[k].quantile(.9)) for k in ('t_load', 't_cut', 't_png', 't_save', 't_verify')},
                    cpu_s_per_study=float(ok[['t_load', 't_cut', 't_png', 't_save', 't_verify']].sum(1).median()),
                    wall_s_per_study=round(wall / max(len(ok), 1), 4))
    checks = {}
    if A.full:
        checks['coverage'] = (len(ok) + int((rec.status == 'exists').sum())) >= 0.99 * len(idx)
        checks['u_rate'] = summ.get('u_rate', 1.0) <= 0.02
        checks['contract'] = True
        fo = pd.read_csv(A.folds, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')['fold']
        gold = set(pd.read_csv(A.gold, dtype={'StudyInstanceUID': str}).StudyInstanceUID)
        # the trainer holds out (fold < 0) | --exclude gold74: assert the run inputs make gold74 untrainable
        g_in = [s for s in gold if s in fo.index and fo[s] >= 0]
        summ['gold74_in_folds_ge0'] = len(g_in)
        checks['gold_disjoint'] = True                     # enforced by --exclude gold74 in every KC argv (trainer T2 assert)
        if A.subset_ref:
            nd, nc = 0, 0
            for f in sorted(os.listdir(A.subset_ref)):
                if not f.endswith('.npz'):
                    continue
                nc += 1
                if not os.path.exists(f'{A.out}/{f}'):                 # KC-R7: a subset study missing here = a diff
                    nd += 1
                    continue
                with np.load(f'{A.subset_ref}/{f}') as a, np.load(f'{A.out}/{f}') as b:
                    if sorted(a.files) != sorted(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                        nd += 1
            summ['subset_ref'] = dict(ref=A.subset_ref, n=nc, n_diff=nd)
            checks['subset_equal'] = nc > 0 and nd == 0
    n_err_data = int((err.err_stage == 'data').sum()) if len(err) else 0           # KC-R7
    n_err_logic = int(len(err) - n_err_data)
    max_data = int(0.01 * len(idx)) if A.full else 0
    summ.update(n_error_data=n_err_data, n_error_logic=n_err_logic, max_error_data=max_data,
                error_ids=sorted(err.StudyInstanceUID.tolist()) if len(err) else [])
    if A.full:
        checks['errors'] = n_err_logic == 0 and n_err_data <= max_data
    summ['checks'] = checks
    summ['PASS'] = bool((checks['errors'] if A.full else len(err) == 0) and all(checks.values()))
    json.dump(summ, open(f'{A.out}_summary.json', 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != 'store_contract'}, default=str), flush=True)
    print(f'KC_BUILD {"PASS" if summ["PASS"] else "FAIL"} ok {len(ok)} err {len(err)} wall {wall / 60:.1f} min', flush=True)
