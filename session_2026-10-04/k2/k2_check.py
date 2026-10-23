#!/usr/bin/env python3
"""k2_check.py: independent check of a K2 working dir after the appended cells ran (laptop smoke, box stage test,
box full-notebook run, or a downloaded Kaggle commit output). Never imports the cells: every blend is recomputed here.

  python k2_check.py --work <dir with submission*.csv> [--expect_fleet clean|fail|auto] [--expect_crop ok|fail|none|auto]
        [--kaggle_commit]

--kaggle_commit (the downloaded output of the <= 3-study Kaggle commit, fleet clean): also gates on the receipt, so
the decision never depends on whether the workers' [fleet g] lines reached the notebook log: 2 GPUs, every worker
decode probe match >= 1 and mismatch 0, shard-0 PARITY re-encode ran (frac_gt8 <= .01, no mismatch), one microbench
dict per GPU without an error, n_fail and n_blank 0 on every worker.

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


def check(work, expect_fleet='auto', expect_crop='auto', tol=1e-9, kaggle_commit=False):
    res, info = [], {}

    def ok(name, cond, detail=''):
        res.append((name, bool(cond), str(detail)[:300]))

    # C6: the crop sandbox keeps no submission*.csv (the parent's own g96_input/ and d4_input/
    # submission_checkpoint_rank_mean.csv are the 0.943 parent's, unchanged, and not checked here)
    sand_dir = f'{work}/_k2_crop'
    nested = sorted(f for f in (os.listdir(sand_dir) if os.path.isdir(sand_dir) else [])
                    if f.startswith('submission') and f.endswith('.csv'))
    ok('no submission*.csv left in the crop sandbox _k2_crop (C6)', not nested, nested[:6])

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
            twin = 'submission_fleetsolo.csv' if cfg.get('MAIN_MODE') == 'solo' else wname(w_main)
            ok('submission.csv == its fail-loud twin ' + twin, filecmp.cmp(sub, f'{work}/{twin}', shallow=False))
            files = st.get('files') or {}
            for f in loud + ['submission.csv']:
                ok(f'{f} unchanged since the fleet stage wrote it (sha256)', files.get(f) == _sha(f'{work}/{f}'))
        ok('workers kept the full member count', rec.get('n_keep') and all(k == rec.get('models') for k in rec['n_keep']),
           f'{rec.get("n_keep")} of {rec.get("models")}')
        if kaggle_commit:
            gpus, par, bench = rec.get('gpus') or [], rec.get('parity') or [], rec.get('bench') or []
            ok('commit: 2 GPUs used', len(gpus) == 2, gpus)
            ok('commit: decode probe match >= 1 and mismatch 0 on every worker',
               len(par) == len(gpus) and all(p.get('decode_match', 0) >= 1 and p.get('decode_mismatch', 1) == 0 for p in par),
               [(p.get('decode_match'), p.get('decode_mismatch')) for p in par])
            p0 = par[0] if par else {}
            ok('commit: shard-0 PARITY re-encode ran, frac_gt8 <= .01, no mismatch',
               'frac_gt8' in p0 and p0['frac_gt8'] <= 0.01 and p0.get('mismatch') == [],
               {k: p0.get(k) for k in ('checked', 'frac_gt8', 'maxdiff', 'mismatch')})
            ok('commit: one microbench per GPU, no error', len(bench) == len(gpus) and all(
                b and 'error' not in b and all(float(v) > 0 for v in b.values()) for b in bench), bench)
            ok('commit: n_fail 0 and n_blank 0 on every worker',
               rec.get('n_fail') == [0] * len(gpus) and rec.get('n_blank') == [0] * len(gpus),
               f'n_fail {rec.get("n_fail")} n_blank {rec.get("n_blank")}')
    else:
        for f in loud:
            ok(f'{f} ABSENT (fleet not clean)', not os.path.exists(f'{work}/{f}'))
        if cfg.get('MAIN_MODE') == 'solo' or st.get('main_mode') == 'solo':
            ok('submission.csv ABSENT (solo mode, fleet not clean)', not os.path.exists(sub))
        else:
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
            sand = f'{work}/_k2_crop/k1out_fleetsolo.csv'                 # C6: renamed by the CF writer
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
    ap.add_argument('--kaggle_commit', action='store_true', help='also gate on the commit receipt (parity, bench, GPUs)')
    a = ap.parse_args(argv)
    res, info = check(a.work, a.expect_fleet, a.expect_crop, kaggle_commit=a.kaggle_commit)
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
