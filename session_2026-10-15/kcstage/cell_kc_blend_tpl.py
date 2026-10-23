# [eeshsaxena __TAG__ KC 3/3 BLEND + CANARY; not in any public notebook] Per label (weight fixed before any run, never fitted;
# PREREG_10-15 Amendment 6): submission = .8 pct(base) + .2 pct(KC leg), float32, not re-ranked; pct = average-tie percentile rank
# (the V224 / HF / S6F convention); base = the frozen output of the cells above (submission_s6f.csv); KC leg = the mean of the
# member sigmoid probabilities (kc_probs.csv, labels BY NAME). The KC leg is ranked over the studies it scored; a study it did not
# score takes its base rank in the KC slot. A column the base already holds constant (a canary of the cells above: Fracture =
# V224 r224, ACL = S6F) is left exactly as it is.
# KC is USED only if: setup + run clean, driver receipt ok, kc_probs.csv sha == the driver receipt, the 12 labels by name, every
# base id once, scored studies finite in [0, 1], coverage >= 98 %__RHO_DOC__. Otherwise __FAIL_DOC__
# Never raises.
def _kc_blend(setup, run):
    import os, json, time, hashlib
    import numpy as np, pandas as pd
    W_KC, COV_MIN, CANARY_COL, ON_FAIL = __W_KC__, __COV_MIN__, "__CANARY_COL__", '__ON_FAIL__'
    RHO_MEAN, RHO_MIN, RHO_MIN_N = __RHO_MEAN__, __RHO_MIN__, 100
    W = '/kaggle/working'
    rec = {'w_kc': W_KC, 'cov_min': COV_MIN, 'canary_col': CANARY_COL, 'on_fail': ON_FAIL, 'rho_floor': [RHO_MEAN, RHO_MIN],
           'mode': None, 'fired': []}
    pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    base = None
    try:
        bp = W + '/submission_s6f.csv'
        if not (isinstance(setup, dict) and setup.get('base_sha256') and sha(bp) == setup['base_sha256']):
            raise ValueError('frozen base missing or changed')
        base = pd.read_csv(bp, dtype={'StudyInstanceUID': str})
        rec['base_source'] = 'submission_s6f.csv'
    except Exception as e:
        rec['base_error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
        try:
            base = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str})   # KC never wrote it: still the base
            rec['base_source'] = 'submission.csv (not frozen)'
        except Exception as e2:
            rec['base_error2'] = type(e2).__name__
    if base is None:
        print('KC_NO_BASE: submission.csv left as written above ' + json.dumps(rec), flush=True)
        return rec
    labels = [c for c in base.columns if c != 'StudyInstanceUID']
    ids = base['StudyInstanceUID'].astype(str).tolist()
    n = len(ids)
    base[labels] = base[labels].astype(np.float32)
    blank = [c for c in labels if n and base[c].nunique(dropna=False) <= 1]
    rec.update(n=n, base_blanked=blank)
    try:
        if rec['base_source'] != 'submission_s6f.csv':
            raise ValueError('no frozen base')
        if not (isinstance(run, dict) and run.get('status') == 'RAN'):
            raise ValueError('KC did not run: %s' % (run or {}).get('status'))
        if run.get('killed'):
            raise ValueError('KC driver killed at the hard deadline')
        drv = run.get('driver') or {}
        if drv.get('ok') is not True:
            raise ValueError('KC driver failed: %s' % str(drv.get('error'))[:300])
        Kp = W + '/kc/kc_probs.csv'
        rec['kc_probs_sha256'] = sha(Kp)
        if rec['kc_probs_sha256'] != (drv.get('B') or {}).get('probs_sha256'):
            raise ValueError('kc_probs.csv differs from the driver receipt')
        K = pd.read_csv(Kp, dtype={'StudyInstanceUID': str})
        if not set(labels) <= set(K.columns):
            raise ValueError('kc_probs.csv lacks labels %s' % sorted(set(labels) - set(K.columns)))
        u = K['StudyInstanceUID'].astype(str)
        if u.duplicated().any() or sorted(u) != sorted(ids):
            raise ValueError('kc_probs.csv ids differ from the base rows')
        S = K.set_index('StudyInstanceUID').loc[ids, labels].to_numpy(np.float64)       # label order BY NAME
        ok = np.isfinite(S).all(axis=1)
        if ((S[ok] < 0) | (S[ok] > 1)).any():
            raise ValueError('KC probabilities outside [0, 1]')
        cov = float(ok.mean()) if n else 0.0
        rec.update(coverage=cov, n_scored=int(ok.sum()), n_unscored=int((~ok).sum()))
        if cov < COV_MIN:
            raise ValueError('coverage %.4f < %.2f (%d of %d scored)' % (cov, COV_MIN, int(ok.sum()), n))
        out = base.copy()
        rho = {}
        for j, c in enumerate(labels):
            if c in blank:
                continue
            rb = pct(base[c].values)
            rs = np.full(n, np.nan)
            rs[ok] = pct(S[ok, j])
            if ok.sum() >= 3 and np.ptp(rb[ok]) > 0 and np.ptp(rs[ok]) > 0:
                rho[c] = float(pd.Series(rb[ok]).corr(pd.Series(rs[ok]), method='spearman'))
            rs[~ok] = rb[~ok]
            out[c] = ((1.0 - W_KC) * rb + W_KC * rs).astype(np.float32)
        rec['rho_base_kc'] = {c: round(v, 4) for c, v in rho.items()}
        rv = np.array([rho.get(c, np.nan) for c in labels if c not in blank], np.float64)
        rec['rho_mean'] = float(np.nanmean(rv)) if np.isfinite(rv).any() else None
        if RHO_MEAN is not None and int(ok.sum()) >= RHO_MIN_N and (not np.isfinite(rv).all() or rv.mean() < RHO_MEAN
                                                                     or rv.min() < RHO_MIN):
            raise ValueError('sanity floor: rank corr(base, KC) mean %.3f / min %.3f below %.2f / %.2f'
                             % (np.nanmean(rv), np.nanmin(rv), RHO_MEAN, RHO_MIN))
        v = out[labels].values.astype(np.float64)
        if not (out['StudyInstanceUID'].tolist() == ids and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
            raise ValueError('blend output not finite / in [0, 1] / in base row order')
        tmp = W + '/submission.csv.tmp'
        out.to_csv(tmp, index=False)
        os.replace(tmp, W + '/submission.csv')
        rec['mode'] = 'KC'
        print('KC_BLEND_OK ' + json.dumps({k: rec.get(k) for k in ('w_kc', 'n', 'n_scored', 'coverage', 'rho_mean', 'base_blanked')}),
              flush=True)
    except Exception as e:
        rec['why'] = '%s: %s' % (type(e).__name__, str(e)[:600])
        try:
            out = base.copy()
            if ON_FAIL == 'canary':
                out[CANARY_COL] = np.float32(0.5)
            tmp = W + '/submission.csv.tmp'
            out.to_csv(tmp, index=False)
            os.replace(tmp, W + '/submission.csv')
            if ON_FAIL == 'canary':
                rec['mode'] = 'KC_CANARY'
                rec['fired'] = ['kc->' + CANARY_COL]
                print('KC_CANARY_FIRED KC leg not used (%s): %s set to 0.5 for all %d rows' % (rec['why'], CANARY_COL, n), flush=True)
            else:
                rec['mode'] = 'KC_BASE_ROWS'
                print('KC_NOT_USED KC leg not used (%s): base rows kept (final-pick form, no canary)' % rec['why'], flush=True)
        except Exception as e2:
            rec['mode'] = 'CANARY_WRITE_ERROR'
            rec['why2'] = '%s: %s' % (type(e2).__name__, str(e2)[:300])
            print('KC_CANARY_WRITE_ERROR ' + json.dumps(rec), flush=True)
    try:
        rec['submission_sha256'] = sha(W + '/submission.csv')
    except Exception:
        pass
    return rec


try:
    _KC_BLEND = _kc_blend(globals().get('_KC_SETUP'), globals().get('_KC_RUN'))
except Exception as _e:
    _KC_BLEND = {'mode': 'BLEND_CELL_ERROR', 'why': type(_e).__name__}
    print('KC_BLEND_CELL_ERROR ' + str(_KC_BLEND), flush=True)
try:
    import json as _kcj, time as _kct
    _KC_RECEIPT = {'tag': '__TAG__', 'setup': globals().get('_KC_SETUP'), 'run': globals().get('_KC_RUN'), 'blend': _KC_BLEND,
                   'fired': list((_KC_BLEND or {}).get('fired') or []), 'mode': (_KC_BLEND or {}).get('mode')}
    try:
        _KC_RECEIPT['total_age_h'] = round((_kct.time() - NB_T0) / 3600.0, 4)
    except Exception:
        pass
    with open('/kaggle/working/kc_receipt.json', 'w') as _f:
        _kcj.dump(_KC_RECEIPT, _f, indent=1, default=str)
    print('KC_DONE ' + _kcj.dumps({'mode': _KC_RECEIPT['mode'], 'fired': _KC_RECEIPT['fired'],
                                   'total_age_h': _KC_RECEIPT.get('total_age_h')}), flush=True)
except Exception as _e:
    print('KC_RECEIPT_ERROR ' + type(_e).__name__, flush=True)
