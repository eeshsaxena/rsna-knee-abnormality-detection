# R7-OOF merge + checks [eeshsaxena]. Writes /kaggle/working/r7_oof.csv (one row per scored study: its held-out fold, 12 probabilities
# exactly as the worker deployed them, the 12 raw fold-model logits, the Effusion fluid / Baker's state stacker terms, the 84-wide
# state read-out), r7_oof_status.csv (every requested study: ok / failed / not reached), r7_oof_receipt.json (timings, checks) and,
# when proof studies ran, r7_proof.csv (all four fold models on the same study: which one is the outlier).
import csv, json, math, platform, shutil
import numpy as np
LAYOUT = json.loads((BUNDLE / manifest['state_stacker']).read_text(encoding='utf-8'))['layout']
STATE_COLS = []
for c_ in LAYOUT['columns']:
    if 'categories' in c_:
        STATE_COLS += ['state|%s|%s' % (c_['class'], cat) for cat in c_['categories']]
    else:
        STATE_COLS.append('bin|%s|%s' % (c_['binary'], c_['class']))
assert len(STATE_COLS) == LAYOUT['width'] == 84, (len(STATE_COLS), LAYOUT['width'])
ADOPT_FLUID = ['Effusion']          # bundle fluid_stacker.json adopts Effusion; state_stacker.json adopts Baker's (asserted below)
ADOPT_STATE = ["Baker's"]
_fl = json.loads((BUNDLE / manifest['fluid_stacker']).read_text(encoding='utf-8'))
_st = json.loads((BUNDLE / manifest['state_stacker']).read_text(encoding='utf-8'))
assert [k for k, v in _fl['classes'].items() if v.get('adopt')] == ADOPT_FLUID
assert [k for k, v in _st['classes'].items() if v.get('adopt')] == ADOPT_STATE


def sig(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def read_csv_rows(p):
    if not p.is_file():
        return {}
    with open(p, encoding='utf-8', newline='') as fh:
        return {r['StudyInstanceUID']: r for r in csv.DictReader(fh)}


def read_job(job):
    wdir = JOBS_DIR / job['id']
    pred = read_csv_rows(wdir / 'predictions.csv')
    fluid = read_csv_rows(wdir / 'fluid_stacker.csv')
    state = read_csv_rows(wdir / 'state_stacker.csv')
    try:
        dump = json.loads((wdir / 'dump.json').read_text(encoding='utf-8'))
    except Exception:
        dump = {}
    try:
        rec = json.loads((wdir / 'receipt.json').read_text(encoding='utf-8'))
    except Exception:
        rec = None
    return pred, fluid, state, dump, rec


CHECKS, FAIL = [], []


def check(cond, what):
    CHECKS.append(('PASS ' if cond else 'FAIL ') + what)
    if not cond:
        FAIL.append(what)


rows, status, seen = [], {}, {}
max_dev_sig, max_dev_stk, n_full = 0.0, 0.0, 0
job_stats = []
for job in [j for j in JOBS if j['kind'] == 'oof']:
    pred, fluid, state, dump, rec = read_job(job)
    k = job['fold']
    if rec is not None:
        check(rec.get('models') == ['%s_s0_f%d.pt' % (ARM, k)], '%s receipt models == [fold %d file]: %s' % (job['id'], k, rec.get('models')))
        check(not (rec.get('flip_tta')), '%s no flip TTA in the receipt' % job['id'])
        st_ = rec.get('stats') or {}
        job_stats.append({'id': job['id'], 'fold': k, 'gpu': job.get('gpu'), 'studies': len(job['uids']), 'scored': rec.get('scored'),
                          'timed_out': rec.get('timed_out'), 'seconds': rec.get('seconds'), 'start_s': job.get('start_s'), 'end_s': job.get('end_s'),
                          'compile_check_s': (rec.get('compile') or {}).get('check_seconds'), 'compile_active': (rec.get('compile') or {}).get('active'),
                          'compile_fallback': (rec.get('compile') or {}).get('fallback'), 'stats': st_, 'per_study': rec.get('per_study'),
                          'amp': rec.get('amp'), 'torch': rec.get('torch'), 'timm': rec.get('timm'), 'side_check': rec.get('side_check'),
                          'anchor_methods': rec.get('anchor_methods')})
    else:
        job_stats.append({'id': job['id'], 'fold': k, 'gpu': job.get('gpu'), 'studies': len(job['uids']), 'status': job.get('status'), 'receipt': None})
    fb = (rec or {}).get('fallback') or {}
    for u in job['uids']:
        if r7_fold(u) != k:
            FAIL.append('study %s in the fold-%d job has hash fold %d' % (u, k, r7_fold(u)))
        if u not in pred:
            status[u] = {'fold': k, 'job': job['id'], 'status': 'not_scored', 'reason': fb.get(u, job.get('status') or 'no row (worker stopped)')}
            continue
        if u in seen:
            FAIL.append('study %s scored twice (%s, %s)' % (u, seen[u], job['id']))
            continue
        seen[u] = job['id']
        r = pred[u]
        p = {L: float(r[L]) for L in LABELS12}
        d = dump.get(u)
        z = None
        st_vec = None
        if d is not None:
            zz = d['z'] if isinstance(d, dict) else d
            assert len(zz) == 1 and len(zz[0]) == 12, 'one fold model per OOF study'
            z = [float(x) for x in zz[0]]
            if isinstance(d, dict) and d.get('state_plain'):
                assert len(d['state_plain']) == 1 and len(d['state_plain'][0]) == 84
                st_vec = [float(x) for x in d['state_plain'][0]]
        fr, sr = fluid.get(u), state.get(u)
        full = {}
        for j_, L in enumerate(LABELS12):
            if L in ADOPT_FLUID and fr is not None and fr.get(L + '|p_out', '') != '':
                full[L] = float(fr[L + '|p_out'])
                max_dev_stk = max(max_dev_stk, abs(full[L] - p[L]))
            elif L in ADOPT_STATE and sr is not None and sr.get(L + '|p_out', '') != '':
                full[L] = float(sr[L + '|p_out'])
                max_dev_stk = max(max_dev_stk, abs(full[L] - p[L]))
            else:
                full[L] = p[L]                    # float32 sigmoid of the fold logit, written %.9g = exact float32
                if z is not None:
                    max_dev_sig = max(max_dev_sig, abs(sig(z[j_]) - p[L]))
        n_full += int(z is not None)
        rec_row = {'StudyInstanceUID': u, 'r7_fold': k, 'shard': shard_of(u, int(C['shard_count'])), 'job': job['id'],
                   'precision': 'full' if z is not None else 'csv9g_no_dump'}
        rec_row.update({'p|' + L: full[L] for L in LABELS12})
        rec_row.update({'z|' + L: (z[j_] if z is not None else '') for j_, L in enumerate(LABELS12)})
        for L in ADOPT_FLUID:
            for s_ in ('z_feature', 'z_out', 'p_model'):
                rec_row['fluid|%s|%s' % (L, s_)] = (fr or {}).get(L + '|' + s_, '')
        rec_row['fluid|eff_sag_p99_open'] = (fr or {}).get('eff_sag_p99_open', '')
        rec_row['fluid|bak_sag_p99_open'] = (fr or {}).get('bak_sag_p99_open', '')
        rec_row['fluid|status'] = (fr or {}).get('status', '')
        for L in ADOPT_STATE:
            for s_ in ('z_feature', 'z_out', 'p_model'):
                rec_row['stk|%s|%s' % (L, s_)] = (sr or {}).get(L + '|' + s_, '')
        rec_row.update({c_: (st_vec[i_] if st_vec is not None else '') for i_, c_ in enumerate(STATE_COLS)})
        rows.append(rec_row)
        status[u] = {'fold': k, 'job': job['id'], 'status': 'ok', 'reason': ''}
for u in TODO:
    status.setdefault(u, {'fold': r7_fold(u), 'job': '', 'status': 'not_scored', 'reason': 'no job'})

# ---- checks on what was written
vals = np.array([[float(r['p|' + L]) for L in LABELS12] for r in rows], np.float64) if rows else np.zeros((0, 12))
check(bool(np.isfinite(vals).all()), 'all probabilities finite')
check(bool(((vals >= 0) & (vals <= 1)).all()), 'all probabilities in [0, 1]')
check(all(r7_fold(r['StudyInstanceUID']) == r['r7_fold'] for r in rows), 'every row r7_fold == sha1(uid)[:8] % 4')
check(len({r['StudyInstanceUID'] for r in rows}) == len(rows), 'each study scored by exactly one fold model (no duplicates)')
check(max_dev_sig <= 2e-7, 'non-stacked classes: |predictions.csv - sigmoid(dumped logit)| max %.3g <= 2e-7' % max_dev_sig)
check(max_dev_stk <= 1e-8, 'stacked classes (Effusion, Baker\'s): |predictions.csv - stacker p_out| max %.3g <= 1e-8' % max_dev_stk)

cols = (['StudyInstanceUID', 'r7_fold', 'shard', 'job', 'precision'] + ['p|' + L for L in LABELS12] + ['z|' + L for L in LABELS12]
        + ['fluid|Effusion|z_feature', 'fluid|Effusion|z_out', 'fluid|Effusion|p_model', 'fluid|eff_sag_p99_open', 'fluid|bak_sag_p99_open',
           'fluid|status', "stk|Baker's|z_feature", "stk|Baker's|z_out", "stk|Baker's|p_model"] + STATE_COLS)


def fmt(v):
    return repr(v) if isinstance(v, float) else v


with open(WORK / 'r7_oof.csv', 'w', encoding='utf-8', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for r in sorted(rows, key=lambda r: r['StudyInstanceUID']):
        w.writerow([fmt(r.get(c_, '')) for c_ in cols])
with open(WORK / 'r7_oof_status.csv', 'w', encoding='utf-8', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['StudyInstanceUID', 'r7_fold', 'job', 'status', 'reason'])
    for u in sorted(status):
        s_ = status[u]
        w.writerow([u, s_['fold'], s_['job'], s_['status'], s_['reason']])

# ---- proof studies: all four fold models; the held-out one should be the outlier (the other three trained on the study)
PROOF_ROWS = []
for job in [j for j in JOBS if j['kind'] == 'proof']:
    pred, fluid, state, dump, rec = read_job(job)
    if rec is not None:
        check(rec.get('models') == manifest['models'], '%s receipt models == the 4 bundle files in order' % job['id'])
    for u in job['uids']:
        d = dump.get(u)
        if d is None:
            continue
        Z = np.array(d['z'] if isinstance(d, dict) else d, np.float64)          # [4, 12], fold order f0..f3 (sorted file names)
        assert Z.shape == (4, 12)
        dist = [float(np.sqrt(np.mean((Z[k] - np.delete(Z, k, 0).mean(0)) ** 2))) for k in range(4)]
        oof = next((r for r in rows if r['StudyInstanceUID'] == u), None)
        k_ = r7_fold(u)
        det = None
        if oof is not None and oof['z|ACL'] != '':
            det = float(np.max(np.abs(np.array([oof['z|' + L] for L in LABELS12], np.float64) - Z[k_])))
        PROOF_ROWS.append({'StudyInstanceUID': u, 'r7_fold': k_, 'outlier_fold': int(np.argmax(dist)), 'match': int(np.argmax(dist)) == k_,
                           'dist_f0': dist[0], 'dist_f1': dist[1], 'dist_f2': dist[2], 'dist_f3': dist[3],
                           'mean_abs_z_heldout': float(np.abs(Z[k_]).mean()), 'mean_abs_z_infold': float(np.abs(np.delete(Z, k_, 0)).mean()),
                           'oof_vs_proof_max_abs_dz': det})
if PROOF_ROWS:
    with open(WORK / 'r7_proof.csv', 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(PROOF_ROWS[0]))
        w.writeheader()
        w.writerows(PROOF_ROWS)
    nm = sum(r['match'] for r in PROOF_ROWS)
    print('PROOF outlier fold == hash fold: %d / %d (chance 1/4 each)' % (nm, len(PROOF_ROWS)), flush=True)
    dets = [r['oof_vs_proof_max_abs_dz'] for r in PROOF_ROWS if r['oof_vs_proof_max_abs_dz'] is not None]
    if dets:
        print('DETERMINISM same fold model, same study, two separate worker runs: max |dz| %.3g (over %d studies)' % (max(dets), len(dets)), flush=True)

# ---- copy small artifacts, receipt
for job in JOBS:
    src_ = JOBS_DIR / job['id']
    dst_ = OUTW / job['id']
    dst_.mkdir(parents=True, exist_ok=True)
    for f_ in ('ids.json', 'log.txt', 'receipt.json', 'predictions.csv', 'fluid_stacker.csv', 'state_stacker.csv', 'dump.json'):
        if (src_ / f_).is_file():
            shutil.copyfile(src_ / f_, dst_ / f_)
try:
    mem_gb = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1)
except Exception:
    mem_gb = None
ok_ = [r for r in rows]
tot = {k_: 0.0 for k_ in ('decode_s', 'render_s', 'gpu_s', 'anchor_s', 'wait_s', 'windows', 'rows')}
for js in job_stats:
    for k_ in tot:
        tot[k_] += float((js.get('stats') or {}).get(k_, 0.0))
RECEIPT = {'config': {k_: (v if not isinstance(v, list) or len(v) <= 20 else '%d items' % len(v)) for k_, v in C.items()},
           'session_seconds': round(time.time() - NB_T0, 1), 'deadline_s': round(DEADLINE - NB_T0, 1),
           'requested': len(TODO), 'scored': len(rows), 'not_scored': sum(s_['status'] != 'ok' for s_ in status.values()),
           'scored_per_fold': [sum(r['r7_fold'] == k_ for r in rows) for k_ in range(4)], 'full_precision_rows': n_full,
           'checks': CHECKS, 'fail': FAIL, 'jobs': job_stats, 'proof': PROOF_ROWS,
           'per_scored_study_mean': {k_: round(v / max(1.0, tot['rows']), 3) for k_, v in tot.items() if k_ != 'rows'},
           'cpu_samples': SAMPLES, 'io_bench': IO_BENCH, 'cpu_count': os.cpu_count(), 'mem_gb': mem_gb, 'n_gpu': N_GPU,
           'python': platform.python_version(), 'bundle_id': manifest['bundle_id'],
           'compact_infer_sha256': hashlib.sha256((SRC / WORKER).read_bytes()).hexdigest(),
           'worker_args': [str(x) for x in WORKER_ARGS]}
(WORK / 'r7_oof_receipt.json').write_text(json.dumps(RECEIPT, indent=1), encoding='utf-8')
for c_ in CHECKS:
    print(c_)
print('R7OOF_DONE', json.dumps({k_: RECEIPT[k_] for k_ in ('session_seconds', 'requested', 'scored', 'not_scored', 'scored_per_fold',
                                                           'full_precision_rows', 'per_scored_study_mean', 'fail')}), flush=True)
