#!/usr/bin/env python3
"""k2_check.py: independent check of a K2 working dir after the appended cells ran (laptop smoke, box stage test,
box full-notebook run, or a downloaded Kaggle commit output). Never imports the cells: every blend is recomputed here.

  python k2_check.py --work <dir with submission*.csv> [--expect_fleet clean|fail|auto] [--expect_crop ok|fail|none|auto]

Checks: every expected file exists and every non-expected one is ABSENT (per-file fail-loud), schema, row order ==
the parent's (_k2_parent.csv), finite and in [0, 1], submission.csv == its wXX twin (clean) or == parent bytes (not
clean), each blend == an independent recomputation from the parent + solo files, the crop stage left every fleet
file untouched (sha256 vs _k2_stage_fleet.json), stage times from the receipts.
"""
import argparse, filecmp, hashlib, json, os, sys
import numpy as np, pandas as pd

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def _rank(a):
    return pd.DataFrame(np.asarray(a, np.float64)).rank(method='average', pct=True).to_numpy()


def wname(w):
    return f'submission_w{int(round(float(w) * 100)):02d}.csv'


def check(work, expect_fleet='auto', expect_crop='auto', tol=1e-9):
    res, info = [], {}

    def ok(name, cond, detail=''):
        res.append((name, bool(cond), str(detail)[:300]))

    par_p = f'{work}/_k2_parent.csv'
    ok('parent copy _k2_parent.csv exists', os.path.isfile(par_p))
    if not os.path.isfile(par_p):
        return res, info
    pub = pd.read_csv(par_p, dtype={'StudyInstanceUID': str})
    n = len(pub); r_p = _rank(pub[TARGETS].to_numpy(np.float64))
    rec = json.load(open(f'{work}/fleet_receipt.json')) if os.path.isfile(f'{work}/fleet_receipt.json') else {}
    st = json.load(open(f'{work}/_k2_stage_fleet.json')) if os.path.isfile(f'{work}/_k2_stage_fleet.json') else {}
    cfg = rec.get('config') or {}
    w_main, w_vars = float(cfg.get('W_MAIN', 0.30)), [float(x) for x in cfg.get('W_VARIANTS', [0.20, 0.40])]
    loud = ['submission_fleetsolo.csv'] + list(dict.fromkeys(wname(w) for w in [w_main] + w_vars))
    clean = st.get('clean') is True
    if expect_fleet != 'auto':
        ok(f'fleet stage outcome == {expect_fleet}', clean == (expect_fleet == 'clean'), st.get('why'))
    info.update(fleet_clean=clean, fleet_why=st.get('why'), n=n)

    def frame_ok(f):
        d = pd.read_csv(f'{work}/{f}', dtype={'StudyInstanceUID': str})
        v = d[TARGETS].to_numpy(np.float64) if d.columns.tolist() == pub.columns.tolist() else np.full((1, 1), np.nan)
        good = (d.columns.tolist() == pub.columns.tolist() and len(d) == n
                and (d['StudyInstanceUID'].values == pub['StudyInstanceUID'].values).all()
                and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all())
        ok(f'{f}: schema, {n} rows, parent row order, finite, in [0,1]', good)
        return v

    sub = f'{work}/submission.csv'
    if clean:
        for f in loud:
            ok(f'{f} written (fleet clean)', os.path.isfile(f'{work}/{f}'))
        if all(os.path.isfile(f'{work}/{f}') for f in loud):
            F = frame_ok('submission_fleetsolo.csv'); r_f = _rank(F)
            for w in [w_main] + w_vars:
                v = frame_ok(wname(w))
                d = float(np.abs(v - _rank((1 - w) * r_p + w * r_f)).max())
                ok(f'{wname(w)} == independent P+F rank blend at W {w}', d < tol, f'max|diff| {d:.2e}')
            frame_ok('submission.csv')
            ok('submission.csv == its fail-loud twin ' + wname(w_main), filecmp.cmp(sub, f'{work}/{wname(w_main)}', shallow=False))
            files = st.get('files') or {}
            for f in loud + ['submission.csv']:
                ok(f'{f} unchanged since the fleet stage wrote it (sha256)', files.get(f) == _sha(f'{work}/{f}'))
        ok('workers kept the full member count', rec.get('n_keep') and all(k == rec.get('models') for k in rec['n_keep']),
           f'{rec.get("n_keep")} of {rec.get("models")}')
    else:
        for f in loud:
            ok(f'{f} ABSENT (fleet not clean)', not os.path.exists(f'{work}/{f}'))
        ok('submission.csv == parent bytes (fleet not clean)', os.path.isfile(sub) and filecmp.cmp(sub, par_p, shallow=False))
    info.update(fleet_stage_min=rec.get('stage_min'), workers_s=rec.get('workers_s'), worker_min=rec.get('worker_min'),
                worker_loop_min=rec.get('worker_loop_min'), bench=rec.get('bench'),
                decode_s_per_study_per_thread=rec.get('decode_s_per_study_per_thread'), models=rec.get('names'),
                rho_mean=rec.get('rho_mean'), kernel_age_h=rec.get('kernel_age_h'), left_min=rec.get('left_min'))
    # ---- crop stage (combined kernels only)
    crec_p = f'{work}/k2_crop_receipt.json'
    if expect_crop == 'none' or (expect_crop == 'auto' and not os.path.isfile(crec_p)):
        for f in os.listdir(work):
            if f.startswith(('submission_cf', 'submission_c1', 'submission_c2', 'submission_c3', 'submission_cropsolo')):
                ok(f'no crop-stage file in a fleet-only run: {f}', False)
        return res, info
    crec = json.load(open(crec_p)) if os.path.isfile(crec_p) else {}
    ccfg = crec.get('config') or {}
    c_w, wc, wf = float(ccfg.get('C_W', 0.15)), float(ccfg.get('CF_WC', 0.15)), float(ccfg.get('CF_WF', 0.25))
    cso, cxx, cf = 'submission_cropsolo.csv', f'submission_c{int(round(c_w * 100)):02d}.csv', 'submission_cf.csv'
    crop_ok = crec.get('crop_ok') is True
    if expect_crop != 'auto':
        ok(f'crop stage outcome == {expect_crop}', crop_ok == (expect_crop == 'ok'), crec.get('crop_why'))
    if crop_ok:
        for f in (cso, cxx):
            ok(f'{f} written (crop ok)', os.path.isfile(f'{work}/{f}'))
        if os.path.isfile(f'{work}/{cso}') and os.path.isfile(f'{work}/{cxx}'):
            C = frame_ok(cso); r_c = _rank(C)
            v = frame_ok(cxx)
            d = float(np.abs(v - _rank((1 - c_w) * r_p + c_w * r_c)).max())
            ok(f'{cxx} == independent P+C rank blend at W {c_w}', d < tol, f'max|diff| {d:.2e}')
            sand = f'{work}/_k2_crop/submission_fleetsolo.csv'
            if os.path.isfile(sand):
                ok('cropsolo == the crop cell\'s own solo file (sandbox)', filecmp.cmp(sand, f'{work}/{cso}', shallow=False)
                   or float(np.abs(pd.read_csv(sand)[TARGETS].to_numpy() - C).max()) < 1e-12)
            if clean:
                ok(f'{cf} written (fleet clean + crop ok)', os.path.isfile(f'{work}/{cf}'))
                if os.path.isfile(f'{work}/{cf}'):
                    F = pd.read_csv(f'{work}/submission_fleetsolo.csv')[TARGETS].to_numpy(np.float64)
                    v = frame_ok(cf)
                    exp = _rank((1 - wc - wf) * r_p + wc * r_c + wf * _rank(F))
                    d = float(np.abs(v - exp).max())
                    ok(f'{cf} == independent rank blend P {1 - wc - wf:.2f} / C {wc} / F {wf}', d < tol, f'max|diff| {d:.2e}')
            else:
                ok(f'{cf} ABSENT (fleet not clean)', not os.path.exists(f'{work}/{cf}'))
    else:
        for f in (cso, cxx, cf):
            ok(f'{f} ABSENT (crop not ok)', not os.path.exists(f'{work}/{f}'))
    if clean:                                                         # the crop stage never touches fleet files
        files = st.get('files') or {}
        for f in loud + ['submission.csv']:
            if os.path.isfile(f'{work}/{f}'):
                ok(f'{f} untouched by the crop stage (sha256)', files.get(f) == _sha(f'{work}/{f}'))
    else:
        ok('submission.csv still == parent bytes after the crop stage', filecmp.cmp(sub, par_p, shallow=False))
    info.update(crop_ok=crop_ok, crop_why=crec.get('crop_why'), crop_members=crec.get('crop_members'),
                crop_cell_min=crec.get('crop_cell_min'), crop_stage_min=crec.get('stage_min'),
                rho_crop_fleet_mean=crec.get('rho_crop_fleet_mean'))
    return res, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--expect_fleet', default='auto', choices=('auto', 'clean', 'fail'))
    ap.add_argument('--expect_crop', default='auto', choices=('auto', 'ok', 'fail', 'none'))
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    res, info = check(a.work, a.expect_fleet, a.expect_crop)
    for name, good, detail in res:
        print(f'{"PASS" if good else "FAIL"}  {name}  {detail}')
    allok = all(g for _, g, _ in res)
    print('INFO', json.dumps(info, default=str))
    print(f'K2CHECK {"PASS" if allok else "FAIL"} ({sum(g for _, g, _ in res)}/{len(res)})')
    if a.json:
        json.dump({'pass': allok, 'checks': res, 'info': info}, open(a.json, 'w'), indent=1, default=str)
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
