# kc_prob_parity.py -- BOX_PLAN_1015 C9 (and the G3 box side): probabilities of the SAME KC ckpt(s) on the same studies via
# (a) the store path (stored PNG views, exactly the trainer's eval tensor) and (b) the serve path (kc_worker: verbatim d96
# reader on the raw DICOMs -> header side + slot geometry -> canon -> cut). Bar (A6 G3): max |dp| <= 1e-3 over all
# studies and labels, on the study's own canon. The weak-zone both-canon serve probability is reported too.
#   python kc_prob_parity.py --ckpts m.pt [m2.pt] --store ~/kc_store/store --dicom ~/eda_data --n 20 --device cpu
#          --threads 12 --out ~/kc/logs/c9_prob_parity.json   (picks >= 1 SagPD-absent and >= 1 weak-zone study first)
import os, sys, json, time, argparse
import numpy as np, pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser('~/fast_build/code2'))
import kc_common as kc
import kc_worker as kw

ap = argparse.ArgumentParser()
ap.add_argument('--ckpts', nargs='+', required=True); ap.add_argument('--store', required=True)
ap.add_argument('--dicom', required=True); ap.add_argument('--n', type=int, default=20)
ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--out', required=True); ap.add_argument('--sids', nargs='*', default=[])
A = ap.parse_args()
import torch
torch.set_num_threads(A.threads)
ROOT = A.dicom if os.path.exists(f'{A.dicom}/train_series.csv') else f'{A.dicom}/rsna-knee-abnormality-detection'
SER = kw.series_records(f'{ROOT}/train_series.csv')
SDIR = f'{ROOT}/train_series'
have = sorted(s for s in os.listdir(SDIR) if os.path.exists(f'{A.store}/{s}.npz'))
pick = list(A.sids)
if not pick:
    absent, weak = [], []
    for s in have:
        with np.load(f'{A.store}/{s}.npz') as z:
            if len(absent) < 1 and any(f'v{j}_blob' not in z.files for j in range(len(kc.VIEWS))):
                absent.append(s)
            if len(weak) < 1 and int(z['weak']) == 1:
                weak.append(s)
        if absent and weak:
            break
    rest = [s for s in np.random.default_rng(1015).permutation(have) if s not in absent + weak]
    pick = (absent + weak + rest)[:A.n]
R = kw.load_reader()
ms = kw.load_members(A.ckpts, A.device)
rows, t0 = [], time.time()
for sid in pick:
    with np.load(f'{A.store}/{sid}.npz') as z:
        views_s = {n: (kc.decode_stack(z[f'v{j}_blob'], z[f'v{j}_offs']) if f'v{j}_blob' in z.files else None)
                   for j, n in enumerate(kc.VIEW_NAMES)}
        weak = int(z['weak'])
    p_store = kw.predict_views(ms, views_s, A.device)
    si = kw.study_inputs(sid, SER[sid], SDIR, R, both=True)
    p_dicom = kw.predict_views(ms, si['views'], A.device)
    p_serve = 0.5 * (kw.predict_views(ms, si['views_L'], A.device) + kw.predict_views(ms, si['views_R'], A.device)) \
        if si['serve'] == 'both' else p_dicom
    rows.append(dict(sid=sid, weak=weak, serve=si['serve'], absent=[n for n in kc.VIEW_NAMES if views_s[n] is None],
                     max_dp=float(np.abs(p_store - p_dicom).max()), serve_shift=float(np.abs(p_serve - p_store).max())))
res = dict(n=len(rows), max_dp=max(r['max_dp'] for r in rows), PASS=bool(max(r['max_dp'] for r in rows) <= 1e-3),
           s_per_study=(time.time() - t0) / max(len(rows), 1), ckpts=A.ckpts, rows=rows)
json.dump(res, open(A.out, 'w'), indent=1)
print(f"KC_PROB_PARITY {'PASS' if res['PASS'] else 'FAIL'} n {res['n']} max |dp| {res['max_dp']:.2e} | "
      f"{res['s_per_study']:.1f} s/study | both-canon studies {sum(r['serve'] == 'both' for r in rows)} | "
      f"SagPD-absent {sum(bool(r['absent']) for r in rows)}")
