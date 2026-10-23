# [eeshsaxena V224-FLIP-S6F 3/3 BLEND + CANARY; not in any public notebook] Per label (weights fixed before any run, never fitted;
# PREREG_10-15 Amendment 3 R4): submission = .8 pct(V224-FLIP) + .2 pct(S6fall), float32, not re-ranked; pct = average-tie
# percentile rank (the V224 / HF convention). S6fall is ranked over the studies it scored; a study it did not score takes the
# V224-FLIP rank in the S6fall slot (= its V224-FLIP rank). A column the V224 canary blanked (Fracture = 0.5) stays 0.5.
# S6fall is USED only if: setup + run clean, driver receipt ok, B_probs.csv has the 12 labels (by name) and every test id once,
# scored studies finite in [0, 1], coverage >= 98 %, and (hidden set, N scored >= 100) rank corr(V224-FLIP, S6fall) mean >= .60
# and min >= .40 (nartaa's sanity floor). Otherwise THE S6F CANARY: the V224-FLIP output with ACL = 0.5 for every row (one
# constant column, about -.03 to -.04 macro AUC on the display), so a skipped / failed / partial S6fall can never pass for a read
# or a pick. Fracture (r224 / RM) and Synovitis (fleet) are the other kernels' canaries; ACL is S6F's alone. Never raises.
def _s6f_blend(setup, run):
    import os, json, time, hashlib
    import numpy as np, pandas as pd
    W_S6F, COV_MIN, CANARY_COL = 0.2, 0.98, 'ACL'
    MIN_RHO_MEAN, MIN_RHO_FINDING, RHO_MIN_N = 0.60, 0.40, 100
    W = '/kaggle/working'
    rec = {'w_s6f': W_S6F, 'cov_min': COV_MIN, 'canary_col': CANARY_COL, 'mode': None, 'fired': []}
    pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    # ---- the V224-FLIP output (the frozen copy; its sha must be the setup cell's) ------------------------------------------
    base, rec['base_source'] = None, None
    try:
        bp = W + '/submission_v224flip.csv'
        if not (isinstance(setup, dict) and setup.get('base_sha256') and sha(bp) == setup['base_sha256']):
            raise ValueError('frozen V224-FLIP output missing or changed')
        base = pd.read_csv(bp, dtype={'StudyInstanceUID': str})
        rec['base_source'] = 'submission_v224flip.csv'
    except Exception as e:
        rec['base_error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
        try:
            base = globals()['sub'].copy()                           # V224 stage 3 / canary's in-memory output
            rec['base_source'] = 'in-memory sub'
        except Exception as e2:
            rec['base_error2'] = type(e2).__name__
    if base is None:
        print('S6F_NO_BASE: submission.csv left as written above ' + json.dumps(rec), flush=True)
        return rec
    labels = [c for c in base.columns if c != 'StudyInstanceUID']
    ids = base['StudyInstanceUID'].astype(str).tolist()
    n = len(ids)
    base[labels] = base[labels].astype(np.float32)
    blank = ['Fracture'] if 'r224->Fracture' in ((setup or {}).get('v224_canary') or {}).get('fired', []) else []
    rec.update(n=n, base_blanked=blank)
    try:
        if not (isinstance(run, dict) and run.get('status') == 'RAN'):
            raise ValueError('S6F did not run: %s' % (run or {}).get('status'))
        if run.get('killed'):
            raise ValueError('S6F driver killed at the hard deadline')
        drv = run.get('driver') or {}
        if drv.get('ok') is not True:
            raise ValueError('S6F driver failed: %s' % str(drv.get('error'))[:300])
        Bp = W + '/s6f/B/B_probs.csv'
        rec['b_probs_sha256'] = sha(Bp)
        if rec['b_probs_sha256'] != (drv.get('B') or {}).get('probs_sha256'):
            raise ValueError('B_probs.csv differs from the driver receipt')
        B = pd.read_csv(Bp, dtype={'StudyInstanceUID': str})
        if not set(labels) <= set(B.columns):
            raise ValueError('B_probs.csv lacks labels %s' % sorted(set(labels) - set(B.columns)))
        u = B['StudyInstanceUID'].astype(str)
        if u.duplicated().any() or sorted(u) != sorted(ids):
            raise ValueError('B_probs.csv ids differ from the V224-FLIP rows')
        S = B.set_index('StudyInstanceUID').loc[ids, labels].to_numpy(np.float64)       # label order BY NAME
        ok = np.isfinite(S).all(axis=1)
        if ((S[ok] < 0) | (S[ok] > 1)).any():
            raise ValueError('S6fall probabilities outside [0, 1]')
        cov = float(ok.mean()) if n else 0.0
        rec.update(coverage=cov, n_scored=int(ok.sum()), n_unscored=int((~ok).sum()))
        if cov < COV_MIN:
            raise ValueError('coverage %.4f < %.2f (%d of %d scored)' % (cov, COV_MIN, int(ok.sum()), n))
        out = base.copy()
        rho = {}
        for j, c in enumerate(labels):
            rb = pct(base[c].values)
            rs = np.full(n, np.nan)
            rs[ok] = pct(S[ok, j])
            if ok.sum() >= 3 and np.ptp(rb[ok]) > 0 and np.ptp(rs[ok]) > 0:      # a constant column (a V224 canary) has no rho
                rho[c] = float(pd.Series(rb[ok]).corr(pd.Series(rs[ok]), method='spearman'))
            rs[~ok] = rb[~ok]
            out[c] = ((1.0 - W_S6F) * rb + W_S6F * rs).astype(np.float32)
        rec['rho_base_s6f'] = {c: round(v, 4) for c, v in rho.items()}
        rv = np.array([rho.get(c, np.nan) for c in labels if c not in blank], np.float64)
        rec['rho_mean'] = float(np.nanmean(rv)) if np.isfinite(rv).any() else None
        if int(ok.sum()) >= RHO_MIN_N and (not np.isfinite(rv).all() or rv.mean() < MIN_RHO_MEAN or rv.min() < MIN_RHO_FINDING):
            raise ValueError('sanity floor: rank corr(V224-FLIP, S6fall) mean %.3f / min %.3f below %.2f / %.2f'
                             % (np.nanmean(rv), np.nanmin(rv), MIN_RHO_MEAN, MIN_RHO_FINDING))
        for c in blank:
            out[c] = np.float32(0.5)
        v = out[labels].values.astype(np.float64)
        if not (out['StudyInstanceUID'].tolist() == ids and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
            raise ValueError('blend output not finite / in [0, 1] / in V224-FLIP row order')
        tmp = W + '/submission.csv.tmp'
        out.to_csv(tmp, index=False)
        os.replace(tmp, W + '/submission.csv')
        rec['mode'] = 'S6F'
        print('S6F_BLEND_OK ' + json.dumps({k: rec.get(k) for k in ('w_s6f', 'n', 'n_scored', 'coverage', 'rho_mean', 'base_blanked')}),
              flush=True)
    except Exception as e:
        rec['why'] = '%s: %s' % (type(e).__name__, str(e)[:600])
        try:
            out = base.copy()
            out[CANARY_COL] = np.float32(0.5)
            tmp = W + '/submission.csv.tmp'
            out.to_csv(tmp, index=False)
            os.replace(tmp, W + '/submission.csv')
            rec['mode'] = 'V224FLIP_CANARY'
            rec['fired'] = ['s6f->' + CANARY_COL]
            print('S6F_CANARY_FIRED S6fall not used (%s): %s set to 0.5 for all %d rows' % (rec['why'], CANARY_COL, n), flush=True)
        except Exception as e2:
            rec['mode'] = 'CANARY_WRITE_ERROR'
            rec['why2'] = '%s: %s' % (type(e2).__name__, str(e2)[:300])
            print('S6F_CANARY_WRITE_ERROR ' + json.dumps(rec), flush=True)
    try:
        rec['submission_sha256'] = sha(W + '/submission.csv')
    except Exception:
        pass
    return rec


try:
    _S6F_BLEND = _s6f_blend(globals().get('_S6F_SETUP'), globals().get('_S6F_RUN'))
except Exception as _e:
    _S6F_BLEND = {'mode': 'BLEND_CELL_ERROR', 'why': type(_e).__name__}
    print('S6F_BLEND_CELL_ERROR ' + str(_S6F_BLEND), flush=True)
try:
    import json as _s6j, time as _s6t
    _S6F_RECEIPT = {'tag': 'v224flips6f', 'setup': globals().get('_S6F_SETUP'), 'run': globals().get('_S6F_RUN'), 'blend': _S6F_BLEND,
                    'fired': list((_S6F_BLEND or {}).get('fired') or []), 'mode': (_S6F_BLEND or {}).get('mode')}
    try:
        _S6F_RECEIPT['total_age_h'] = round((_s6t.time() - NB_T0) / 3600.0, 4)
    except Exception:
        pass
    with open('/kaggle/working/s6f_receipt.json', 'w') as _f:
        _s6j.dump(_S6F_RECEIPT, _f, indent=1, default=str)
    print('S6F_DONE ' + _s6j.dumps({'mode': _S6F_RECEIPT['mode'], 'fired': _S6F_RECEIPT['fired'],
                                    'total_age_h': _S6F_RECEIPT.get('total_age_h')}), flush=True)
except Exception as _e:
    print('S6F_RECEIPT_ERROR ' + type(_e).__name__, flush=True)
