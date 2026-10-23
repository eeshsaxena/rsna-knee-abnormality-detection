# === FN-IDS: the parent-free id cell (kfn/build_kfn.py, 2026-10-09, Claude). Replaces the 0.943 parent (31 cells). ===
# It runs NO model and loads NO parent dataset. It only:
#   I1 sets the kernel clock T0 (the fleet / N walls take min(/proc start, T0)) and the environment defaults the parent
#      used to leave for the child processes (OMP / OPENBLAS / MKL threads 4, CUDNN_CONV_WSCAP_DBG 1024, HF offline,
#      RSNA_COMP_ROOT), so the fleet workers and the N child start in the same environment as in the tested K9 / K9N kits;
#   I2 finds the competition root exactly as the fleet worker does (RSNA_COMP_ROOT, /kaggle/input/competitions/<slug>,
#      /kaggle/input/<slug>; test.csv + test_series.csv + test_series/ required), reads test.csv ids (str, unique,
#      non-empty) and, if sample_submission.csv exists, asserts its columns == the 12 targets and its id set == test.csv;
#   I3 writes _fn_ids.csv = the ids in test.csv order with every target = 0.5 (an ID-ONLY PLACEHOLDER, never a
#      prediction), copies it to submission.csv (the fleet cell reads its "parent" from there) and writes _fn_ids.json.
# The placeholder P gets weight 0 in every FN / N-only blend (asserted by the N cell, FN-P3). Fail-loud: on any failure
# NONE of the three files exists, so the fleet stage (no submission.csv: "parent wrote no submission.csv") and the N
# stage (no _fn_ids.csv) both refuse and the kernel ends with no submission.csv.
import os, time
T0 = time.time()
for _fn_v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_fn_v, '4')
os.environ.setdefault('CUDNN_CONV_WSCAP_DBG', '1024')
for _fn_v in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ.setdefault(_fn_v, '1')
import json, hashlib, traceback
import numpy as np, pandas as pd

_FN_WORK = '/kaggle/working'
_FN_FILES = ['submission.csv', '_fn_ids.csv', '_fn_ids.json']
_FN_TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
               'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _fn_log(msg):
    print(f'[fn-ids {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _fn_ids_remove():
    for f in _FN_FILES:
        for p in (f'{_FN_WORK}/{f}', f'{_FN_WORK}/{f}.tmp'):
            try:
                if os.path.lexists(p):
                    os.remove(p)
            except BaseException as e:
                _fn_log(f'could not remove {p}: {e!r}')


def _fn_ids_inner():
    _fn_ids_remove()
    root = None
    for c in (os.environ.get('RSNA_COMP_ROOT', ''), '/kaggle/input/competitions/rsna-knee-abnormality-detection',
              '/kaggle/input/rsna-knee-abnormality-detection'):
        if c and os.path.isfile(c + '/test.csv') and os.path.isfile(c + '/test_series.csv') and os.path.isdir(c + '/test_series'):
            root = c; break
    if root is None:
        raise RuntimeError('competition data (test.csv, test_series.csv, test_series/) not found')
    os.environ['RSNA_COMP_ROOT'] = root
    test = pd.read_csv(f'{root}/test.csv', dtype={'StudyInstanceUID': str})
    ids = test['StudyInstanceUID'].tolist() if 'StudyInstanceUID' in test.columns else []
    if not ids or any((not isinstance(s, str)) or not s.strip() for s in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f'test.csv StudyInstanceUID: {len(ids)} rows, {len(set(map(str, ids)))} unique (empty / NaN / duplicate)')
    ss_p = f'{root}/sample_submission.csv'
    if os.path.isfile(ss_p):
        ss = pd.read_csv(ss_p, dtype={'StudyInstanceUID': str})
        if ss.columns.tolist() != ['StudyInstanceUID', *_FN_TARGETS]:
            raise RuntimeError(f'sample_submission.csv columns {ss.columns.tolist()} != the 12 targets in order')
        if set(ss['StudyInstanceUID']) != set(ids) or len(ss) != len(ids):
            raise RuntimeError('sample_submission.csv ids != test.csv ids')
    ph = pd.DataFrame(np.full((len(ids), 12), 0.5), columns=_FN_TARGETS)
    ph.insert(0, 'StudyInstanceUID', ids)
    tmp = f'{_FN_WORK}/_fn_ids.csv.tmp'; ph.to_csv(tmp, index=False); os.replace(tmp, f'{_FN_WORK}/_fn_ids.csv')
    raw = open(f'{_FN_WORK}/_fn_ids.csv', 'rb').read()
    tmp = f'{_FN_WORK}/submission.csv.tmp'
    with open(tmp, 'wb') as fh:
        fh.write(raw)
    os.replace(tmp, f'{_FN_WORK}/submission.csv')
    rec = dict(n=len(ids), root=root, T0=T0, placeholder=0.5, sample_submission=os.path.isfile(ss_p),
               ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
               file_sha256=hashlib.sha256(raw).hexdigest())
    tmp = f'{_FN_WORK}/_fn_ids.json.tmp'; json.dump(rec, open(tmp, 'w'), indent=1); os.replace(tmp, f'{_FN_WORK}/_fn_ids.json')
    return rec


try:
    _fn_ids_rec = _fn_ids_inner()
    _fn_log(f'RESULT OK {_fn_ids_rec["n"]} test ids from {_fn_ids_rec["root"]}/test.csv; _fn_ids.csv = submission.csv = '
            f'ID-ONLY placeholder (0.5, weight 0), sha256 {_fn_ids_rec["file_sha256"][:16]}')
except BaseException as _fn_e:
    traceback.print_exc()
    _fn_ids_remove()
    _fn_ids_rec = None
    _fn_log(f'RESULT FAILED {type(_fn_e).__name__}: {_fn_e} -> no submission.csv / _fn_ids.csv; every later stage refuses '
            f'(the submission will error)')
