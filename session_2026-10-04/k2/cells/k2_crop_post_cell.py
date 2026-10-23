# === K2 CROP STAGE (CF writer), combined mode only: runs AFTER the verbatim K1 crop cell ===
# Legs: P = parent (_k2_parent.csv), F = ConvNeXt fleet raw-prob mean (submission_fleetsolo.csv of a CLEAN fleet
# stage, sha256-checked against _k2_stage_fleet.json), C = crop fleet raw-prob mean (the crop cell's own solo file in
# the sandbox, written only when its receipt says ok). Inside each leg the members are raw-prob averaged (as both
# verified cells do); ACROSS legs the blend is in rank space, one global weight per leg:
#   submission_cf.csv    = rank_pct((1 - CF_WC - CF_WF) r(P) + CF_WC r(C) + CF_WF r(F))   needs fleet clean + crop ok
#   submission_cXX.csv   = rank_pct((1 - C_W) r(P) + C_W r(C))                             needs crop ok
#   submission_cropsolo.csv = C (raw, parent row order)                                    needs crop ok
# Each group is fail-loud on its own: a file exists only if every stage it depends on finished cleanly. The fleet
# stage's files and submission.csv are never touched here.
# C6 after the CF writer has read the sandbox, every submission*.csv the K1 cell left in /kaggle/working/_k2_crop is
#    renamed k1out*.csv: those are parent copies (crop failed) or P+C blends under K2's own names (submission.csv,
#    submission_w20.csv), so none may be picked in the submit dialog. The only submittable files are top level.
import os, json, time, hashlib, shutil, traceback
import numpy as np, pandas as pd


def _k2c_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _k2c_rank(a):
    return pd.DataFrame(np.asarray(a, np.float64)).rank(method='average', pct=True).to_numpy()


def _k2c_leg(path, pub, what):
    d = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    if d.columns.tolist() != pub.columns.tolist():
        raise RuntimeError(f'{what}: columns {d.columns.tolist()}')
    if not (d['StudyInstanceUID'].values == pub['StudyInstanceUID'].values).all():
        raise RuntimeError(f'{what}: row order differs from the parent')
    v = d[pub.columns[1:]].to_numpy(np.float64)
    if not (np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()):
        raise RuntimeError(f'{what}: not finite / outside [0, 1]')
    if len(d) >= 11 and float(v.std(0).min()) < 1e-9:
        raise RuntimeError(f'{what}: constant column')
    return v


def _k2c_write_group(files, frames, rec, tag):
    """All files of one group or none (tmp + os.replace; partial writes deleted)."""
    try:
        for f in files:
            tmp = f'{_K2C_WORK}/{f}.tmp'; frames[f].to_csv(tmp, index=False); os.replace(tmp, f'{_K2C_WORK}/{f}')
        rec[tag] = 'written'
    except BaseException as e:
        rec[tag] = f'write failed {e!r}'
        for f in files:
            for p in (f'{_K2C_WORK}/{f}', f'{_K2C_WORK}/{f}.tmp'):
                try:
                    if os.path.lexists(p):
                        os.remove(p)
                except BaseException:
                    pass


def _k2c_post():
    rec = {'config': dict(K2C)}
    loud = _k2c_loud_names()
    cso, cw, cf = loud
    crop = globals().get('_cf_receipt')
    crop_ok = isinstance(crop, dict) and crop.get('ok') is True
    work_seen = globals().get('CF_WORK')
    sand_ok = work_seen is not None and os.path.realpath(str(work_seen)) == os.path.realpath(_K2C_SAND)
    if crop_ok and not sand_ok:
        crop_ok = False
        _k2c_log(f'ERROR the crop cell wrote to {work_seen!r}, not the sandbox: crop leg refused')
    rec.update(crop_ok=crop_ok, crop_why=(crop or {}).get('why') if isinstance(crop, dict) else 'crop cell wrote nothing',
               crop_cell_min=(crop or {}).get('cell_s', 0) / 60 if isinstance(crop, dict) else None,
               crop_members=[m.get('name') for m in (crop or {}).get('members', [])] if isinstance(crop, dict) else None)
    try:
        st = json.load(open(f'{_K2C_WORK}/_k2_stage_fleet.json'))
    except Exception as e:
        st = {'clean': False, 'why': f'no fleet status ({e!r})'}
    fleet_clean = st.get('clean') is True
    rec.update(fleet_clean=fleet_clean, fleet_why=st.get('why'))
    pub = pd.read_csv(_K2C_PAR, dtype={'StudyInstanceUID': str})
    r_p = _k2c_rank(pub[pub.columns[1:]].to_numpy(np.float64))

    def frame(v):
        d = pd.DataFrame(v, columns=pub.columns[1:]); d.insert(0, 'StudyInstanceUID', pub['StudyInstanceUID'].values)
        return d

    C = None
    if crop_ok:
        try:
            if _k2c_sha(f'{_K2C_SAND}/submission_fleetsolo.csv') == _k2c_sha(_K2C_PAR):
                raise RuntimeError('crop solo is a byte copy of the parent (crop cell restored)')
            C = _k2c_leg(f'{_K2C_SAND}/submission_fleetsolo.csv', pub, 'crop solo')
            w = float(K2C['C_W']); assert 0.0 < w < 1.0, w
            _k2c_write_group([cso, cw], {cso: frame(C), cw: frame(_k2c_rank((1 - w) * r_p + w * _k2c_rank(C)))},
                             rec, 'group_C')
        except BaseException as e:
            C = None; rec['group_C'] = f'not written: {type(e).__name__}: {e}'
    else:
        rec['group_C'] = 'not written: crop stage not ok'
    if C is not None and fleet_clean:
        try:
            fp = f'{_K2C_WORK}/submission_fleetsolo.csv'
            want = (st.get('files') or {}).get('submission_fleetsolo.csv')
            if not want or _k2c_sha(fp) != want:
                raise RuntimeError('submission_fleetsolo.csv is not the file the fleet stage wrote (sha256)')
            Fv = _k2c_leg(fp, pub, 'fleet solo')
            wc, wf = float(K2C['CF_WC']), float(K2C['CF_WF'])
            assert 0.0 < wc and 0.0 < wf and wc + wf < 1.0, (wc, wf)
            b = _k2c_rank((1 - wc - wf) * r_p + wc * _k2c_rank(C) + wf * _k2c_rank(Fv))
            _k2c_write_group([cf], {cf: frame(b)}, rec, 'group_CF')
            rr = [float(np.corrcoef(_k2c_rank(C)[:, j], _k2c_rank(Fv)[:, j])[0, 1]) for j in range(12)] if len(pub) > 2 else []
            rec['rho_crop_fleet_mean'] = float(np.nanmean(rr)) if rr else None
        except BaseException as e:
            rec['group_CF'] = f'not written: {type(e).__name__}: {e}'
    else:
        rec['group_CF'] = 'not written: ' + ('crop leg missing' if C is None else 'fleet stage not clean')
    return rec


def _k2c_entry():
    rec = {}
    try:
        rec = _k2c_post()
    except BaseException as e:
        traceback.print_exc()
        rec['error'] = f'{type(e).__name__}: {e}'
        for f in _k2c_loud_names():                                   # unexpected path: nothing half-done stays
            for p in (f'{_K2C_WORK}/{f}', f'{_K2C_WORK}/{f}.tmp'):
                try:
                    if os.path.lexists(p):
                        os.remove(p)
                except BaseException:
                    pass
    try:                                                              # C6 sandbox outputs never look submittable
        ren = []
        for f in (sorted(os.listdir(_K2C_SAND)) if os.path.isdir(_K2C_SAND) else []):
            if f.startswith('submission') and f.endswith('.csv'):
                os.replace(f'{_K2C_SAND}/{f}', f'{_K2C_SAND}/k1out{f[len("submission"):]}'); ren.append(f)
        rec['sandbox_renamed'] = ren
    except BaseException as e:
        rec['sandbox_rename'] = f'failed {e!r}'
    try:
        rec['stage_min'] = (time.time() - globals().get('_k2c_t0', time.time())) / 60
        for k in ('CROP_FLEET_WORK', 'CROP_FLEET_OVERRIDES'):
            os.environ.pop(k, None)
        for f in _k2c_loud_names() + ['submission.csv', 'submission_fleetsolo.csv']:
            p = f'{_K2C_WORK}/{f}'
            if os.path.isfile(p):
                d = pd.read_csv(p, dtype={'StudyInstanceUID': str})
                print(f'[k2-crop] {f:<27} rows={len(d)} nan={int(d.iloc[:, 1:].isna().sum().sum())}', flush=True)
            else:
                print(f'[k2-crop] {f:<27} ABSENT', flush=True)
        cm = rec.get('crop_cell_min')
        _k2c_log(f'STAGE crop end {rec["stage_min"]:.2f} min (crop cell {"-" if cm is None else f"{cm:.2f}"} min)')
        _k2c_log(f'RESULT crop ok={rec.get("crop_ok")} fleet clean={rec.get("fleet_clean")} | C files: '
                 f'{rec.get("group_C")} | CF file: {rec.get("group_CF")} | crop: {rec.get("crop_why")}')
        json.dump(rec, open(f'{_K2C_WORK}/k2_crop_receipt.json', 'w'), indent=1, default=str)
    except BaseException as e:
        try:
            _k2c_log(f'receipt/log error (outputs unaffected): {e!r}')
        except BaseException:
            pass
    return rec


_k2c_receipt = _k2c_entry()
