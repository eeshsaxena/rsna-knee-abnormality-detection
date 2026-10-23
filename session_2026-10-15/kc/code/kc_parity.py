# kc_parity.py -- BOX_PLAN_1015 C3 serve parity: the KC serve path (kc_worker: verbatim d96 reader on the raw DICOMs ->
# header side + slot geometry -> canon -> cut) vs the KC store (cut from the d96 corpus npy) on the eda_data train studies.
# PASS = 458/458 studies compared, max |diff| <= 1 and mean |diff| < .05 per study, side / rule / flips equal the store's.
# Must PASS before the G3 pack (not part of KC_READY, critique adopted). Box CPU (cores 28-39).
#   python kc_parity.py --dicom ~/eda_data --store ~/kc_store/test458 --procs 12 --out ~/kc/logs/parity458.json
import os, sys, json, time, argparse, hashlib
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kc_common as kc
import kc_worker as kw

ap = argparse.ArgumentParser()
ap.add_argument('--dicom', required=True); ap.add_argument('--store', required=True)
ap.add_argument('--corpus', default=os.path.expanduser('~/raptor_store/rsna'))
ap.add_argument('--procs', type=int, default=12); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=0)
A = ap.parse_args()
ROOT = A.dicom if os.path.exists(f'{A.dicom}/train_series.csv') else f'{A.dicom}/rsna-knee-abnormality-detection'
SER = kw.series_records(f'{ROOT}/train_series.csv')
SDIR = f'{ROOT}/train_series'
sids = sorted(s for s in os.listdir(SDIR) if os.path.exists(f'{A.store}/{s}.npz'))
if A.n:
    sids = sids[:A.n]
R = None


def init():
    global R
    import cv2
    cv2.setNumThreads(1)
    R = kw.load_reader()


def one(sid):
    try:
        t0 = time.perf_counter()
        si = kw.study_inputs(sid, SER.get(sid, []), SDIR, R, both=False)
        t1 = time.perf_counter()
        vol_eq = bool(np.array_equal(si['vol'], np.load(f'{A.corpus}/{sid}.npy')))
        mx, mn, n_v = 0, 0.0, 0
        with np.load(f'{A.store}/{sid}.npz') as z:
            same_side = str(z['side']) == si['side'] and str(z['rule']) == si['rule']
            fl = json.loads(str(z['flips']))
            my = kc.block_flips(si['side'], si['geom'])
            same_flip = all(bool(fl[b]) == bool(my[b]) for b in ('sagfs', 'sagpd', 'corfs', 'ax'))
            pres_eq = True
            for j, name in enumerate(kc.VIEW_NAMES):
                st = f'v{j}_blob' in z.files
                if st != (si['views'][name] is not None):
                    pres_eq = False; continue
                if not st:
                    continue
                a = kc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']).astype(np.int16)
                d = np.abs(a - si['views'][name].astype(np.int16))
                mx = max(mx, int(d.max())); mn += float(d.mean()); n_v += 1
        return dict(sid=sid, ok=True, vol_eq=vol_eq, side_eq=same_side, flip_eq=same_flip, pres_eq=pres_eq, max=mx,
                    mean=mn / max(n_v, 1), serve=si['serve'], t_read=t1 - t0)
    except Exception as e:
        return dict(sid=sid, ok=False, err=f'{type(e).__name__}: {str(e)[:150]}')


if __name__ == '__main__':
    t0 = time.time()
    with mp.get_context('fork').Pool(A.procs, initializer=init) as pool:
        res = list(pool.imap_unordered(one, sids, chunksize=2))
    good = [r for r in res if r['ok']]
    agg = dict(n=len(sids), n_ok=len(good), n_err=len(res) - len(good), errors=[r.get('err') for r in res if not r['ok']][:5],
               vol_eq=sum(r['vol_eq'] for r in good), side_eq=sum(r['side_eq'] for r in good),
               flip_eq=sum(r['flip_eq'] for r in good), pres_eq=sum(r['pres_eq'] for r in good),
               max_diff=max([r['max'] for r in good] or [99]), mean_diff_max=max([r['mean'] for r in good] or [99.0]),
               serve_both=sum(r['serve'] == 'both' for r in good), s_per_study_thread=float(np.median([r['t_read'] for r in good] or [0])),
               wall_s=round(time.time() - t0, 1), procs=A.procs)
    n = agg['n']
    agg['PASS'] = bool(n > 0 and agg['n_ok'] == n and agg['max_diff'] <= 1 and agg['mean_diff_max'] < 0.05 and
                       agg['side_eq'] == n and agg['flip_eq'] == n and agg['pres_eq'] == n)
    json.dump(dict(agg, rows=res), open(A.out, 'w'), indent=1, default=str)
    print(json.dumps(agg))
    print(f"KC_PARITY {'PASS' if agg['PASS'] else 'FAIL'} {agg['n_ok']}/{n} max {agg['max_diff']} mean_max {agg['mean_diff_max']:.4f}")
