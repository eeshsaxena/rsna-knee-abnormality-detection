#!/usr/bin/env python3
"""k2_check.py: independent check of a K2 working dir after the appended cells ran (laptop smoke, box stage test,
box full-notebook run, or a downloaded Kaggle commit output). Never imports the cells: every blend is recomputed here.

  python k2_check.py --work <dir with submission*.csv> [--expect_fleet clean|fail|auto] [--expect_crop ok|fail|none|auto]
        [--kaggle_commit] [--expect_members NAME[,NAME]] [--expect_mode blend|solo|strict] [--expect_manifest JSON]

K3 (scratchpad/k3): MAIN_MODE 'strict' = submission.csv is the W_MAIN blend twin when clean and ABSENT when not clean
(like 'solo', whose twin is the fleetsolo file); the member filter (K2_CFG MEMBERS) sets the expected member names and
count; --expect_members / --expect_mode pin both from outside (the commit check: the right files ran in the right mode).

--kaggle_commit (the downloaded output of the <= 3-study Kaggle commit, fleet clean): also gates on the receipt, so
the decision never depends on whether the workers' [fleet g] lines reached the notebook log: 2 GPUs, every worker
decode probe match >= 1 and mismatch 0, shard-0 PARITY re-encode ran (frac_gt8 <= .01, no mismatch), one microbench
dict per GPU without an error, n_fail and n_blank 0 on every worker.

K5 (scratchpad/k5): --expect_tta NAME=A|AB[,..] (or a K5_BUILD.json path, its 'tta' map): the receipt's TTA record
(rec['tta'], written by the K5 cell from the workers' npz) must show exactly these modes for exactly the members that
ran, pass B offset 0.5 - 1e-3 when any member is AB, rows written == studies - n_fail - n_blank, every member's pass A
count == rows, pass B count == rows for AB and 0 for A, and (with --kaggle_commit) microbench passes 2 for AB / 1 for A.
With --expect_manifest the receipt's TTA modes must also equal the manifest's sources tta (absent = A).
K5c (adversarial verify 10-05, with --expect_tta): the served numbers themselves, from the worker npz files the kernel
leaves in its output (_fleet_<g>.npz; required with --kaggle_commit, checked whenever present): one per worker, every AB
member P == (PA + PB) / 2 exactly on its written rows with PB != PA, every A member P == PA with PB never run, and
submission_fleetsolo.csv == the members' raw mean of those P rows (so a receipt cannot claim AB over pass-A numbers).

Checks: every expected file exists and every non-expected one is ABSENT (per-file fail-loud), schema, row order ==
the parent's (_k2_parent.csv), finite and in [0, 1], submission.csv == its wXX twin (clean) or == parent bytes (not
clean), each blend == an independent recomputation from the parent + solo files, the crop stage left every fleet
file untouched (sha256 vs _k2_stage_fleet.json), stage times from the receipts.
"""
import argparse, filecmp, glob, hashlib, json, os, sys
import numpy as np, pandas as pd

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def _rank(a):
    return pd.DataFrame(np.asarray(a, np.float64)).rank(method='average', pct=True).to_numpy()


def wname(w):
    return f'submission_w{int(round(float(w) * 100)):02d}.csv'


def check(work, expect_fleet='auto', expect_crop='auto', tol=1e-9, kaggle_commit=False, expect_members=None,
          expect_mode=None, expect_manifest=None, expect_w_main=None, expect_tta=None):
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
    mode = cfg.get('MAIN_MODE') or st.get('main_mode') or 'blend'                      # K3a
    want = cfg.get('MEMBERS') if 'MEMBERS' in cfg else st.get('members_filter')         # K3b
    if expect_fleet != 'auto':
        ok(f'fleet stage outcome == {expect_fleet}', clean == (expect_fleet == 'clean'), st.get('why'))
    if expect_mode:
        ok(f'MAIN_MODE == {expect_mode}', mode == expect_mode, mode)
    if expect_w_main is not None:                                                       # K4: W .20 vs W .50 kernels
        ok(f'W_MAIN == {expect_w_main}', 'W_MAIN' in cfg and abs(w_main - float(expect_w_main)) < 1e-9, cfg.get('W_MAIN'))
    if expect_members is not None:
        ok(f'member filter in the kernel config == {expect_members}', want is not None and sorted(want) == sorted(expect_members),
           want)
    info.update(fleet_clean=clean, fleet_why=st.get('why'), n=n, main_mode=mode, members_filter=want)

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
            twin = 'submission_fleetsolo.csv' if mode == 'solo' else wname(w_main)           # strict: as blend
            ok('submission.csv == its fail-loud twin ' + twin, filecmp.cmp(sub, f'{work}/{twin}', shallow=False))
            files = st.get('files') or {}
            for f in loud + ['submission.csv']:
                ok(f'{f} unchanged since the fleet stage wrote it (sha256)', files.get(f) == _sha(f'{work}/{f}'))
        ok('workers kept the full member count', rec.get('n_keep') and all(k == rec.get('models') for k in rec['n_keep']),
           f'{rec.get("n_keep")} of {rec.get("models")}')
        if want is not None:                                                            # K3b
            ok(f'member filter: the workers ran exactly {want}', sorted(rec.get('names') or []) == sorted(want)
               and rec.get('models') == len(want), f'names {rec.get("names")} models {rec.get("models")}')
        if expect_members is not None:
            ok(f'the workers ran exactly {expect_members}', sorted(rec.get('names') or []) == sorted(expect_members)
               and rec.get('models') == len(expect_members), rec.get('names'))
        if expect_manifest:                                                             # K3c: the bytes, not the names
            loc = json.load(open(expect_manifest))
            want_sha = {k: (v or {}).get('sha256') for k, v in (loc.get('sources') or {}).items()}
            ok('attached manifest checkpoints + sha256[:16] == the local build (same bytes, not only same names)',
               rec.get('manifest') == loc.get('checkpoints') and rec.get('manifest_sha16') == want_sha and bool(want_sha),
               f'commit {rec.get("manifest_sha16")} local {want_sha}')
            loc_tta = {k: ((v or {}).get('tta') or 'A') for k, v in (loc.get('sources') or {}).items()
                       if k in (rec.get('names') or [])}                                 # K5
            got_tta = (rec.get('tta') or {}).get('modes') if rec.get('tta') else {k: 'A' for k in rec.get('names') or []}
            ok('TTA modes the workers ran == the local manifest sources tta (absent = A)', got_tta == loc_tta,
               f'receipt {got_tta} manifest {loc_tta}')
        if expect_tta is not None:                                                      # K5
            t = rec.get('tta') or {}
            if 'TTA' in cfg:                                                            # K5b
                ok('kernel K2_CFG TTA == --expect_tta', cfg['TTA'] == expect_tta, cfg['TTA'])
            names_ran = sorted(rec.get('names') or [])
            ok(f'--expect_tta names == the members that ran', sorted(expect_tta) == names_ran,
               f'expect {sorted(expect_tta)} ran {names_ran}')
            ok(f'TTA modes in the receipt == {expect_tta}', t.get('modes') == expect_tta, t.get('modes'))
            if 'AB' in expect_tta.values():
                ok('TTA pass B offset == 0.5 - 1e-3 (HALF, FIX-1)', abs(float(t.get('half') or 0) - (0.5 - 1e-3)) < 1e-12,
                   t.get('half'))
            rows, fwd = t.get('rows'), t.get('fwd') or {}
            n_ok = n - sum(rec.get('n_fail') or [0]) - sum(rec.get('n_blank') or [0])
            ok('TTA rows written == studies - n_fail - n_blank', rows == n_ok, f'rows {rows} expected {n_ok}')
            good = rows is not None and all(
                (fwd.get(k) or {}).get('A') == rows and (fwd.get(k) or {}).get('B') == (rows if m == 'AB' else 0)
                for k, m in expect_tta.items())
            ok('TTA forwards: pass A on every row for every member, pass B on every row for AB members only', good,
               f'rows {rows} fwd {fwd}')
            if kaggle_commit:
                bp = t.get('bench_passes') or {}
                ok('commit: microbench timed 2 passes for AB members, 1 for A', bp == {k: (2 if m == 'AB' else 1)
                                                                                     for k, m in expect_tta.items()}, bp)
            # K5c (adversarial verify 10-05): the served NUMBERS, not only the receipt's counters. The workers' npz
            # files (_fleet_<g>.npz, left in /kaggle/working, so they are in the downloaded output) must show on every
            # written row P == (PA + PB) / 2 exactly with PB != PA for each AB member, and P == PA with pass B never run
            # for each A member; and submission_fleetsolo.csv (the file every blend and the solo main are built from)
            # must equal the members' raw mean of those P rows. Required with --kaggle_commit; else checked when present.
            zf = sorted(glob.glob(f'{work}/_fleet_*.npz'))
            nw = len(rec.get('n_keep') or [])
            if kaggle_commit or zf:
                ok('K5c worker npz files in the output, one per worker', bool(zf) and len(zf) == nw,
                   f'{[os.path.basename(f) for f in zf]} workers {nw}')
            if zf and len(zf) == nw:
                zs = [dict(np.load(f, allow_pickle=False)) for f in zf]
                bad, dpb = [], {}
                names0 = [str(x) for x in zs[0]['names']]
                for f, z in zip(zf, zs):
                    fn, nm = os.path.basename(f), [str(x) for x in z['names']]
                    if nm != names0 or sorted(nm) != sorted(expect_tta):
                        bad.append(f'{fn} names {nm}'); continue
                    for k, m in expect_tta.items():
                        j = nm.index(k); p = z['P'][j]; w = np.isfinite(p).all(1)
                        if m == 'AB':
                            if 'PA' not in z or 'PB' not in z:
                                bad.append(f'{fn}: no PA/PB for the AB member {k}'); continue
                            pa_, pb_ = z['PA'][j], z['PB'][j]
                            if not (np.isfinite(pa_[w]).all() and np.isfinite(pb_[w]).all() and np.isnan(p[~w]).all()
                                    and np.array_equal(p[w], (pa_[w] + pb_[w]) / 2)):
                                bad.append(f'{fn}: {k} P != (PA + PB) / 2 on the written rows')
                            if w.any():
                                dpb[k] = max(dpb.get(k, 0.0), float(np.abs(pb_[w].astype(np.float64) - pa_[w]).max()))
                        elif 'PA' in z or 'PB' in z:
                            if not (np.array_equal(p[w], z['PA'][j][w]) and np.isnan(z['PB'][j]).all()):
                                bad.append(f'{fn}: A member {k} P != PA or pass B ran')
                nab = [k for k, m in expect_tta.items() if m == 'AB']
                ok('K5c worker npz: AB members P == (PA + PB) / 2 exactly with PB != PA, A members P == PA (PB not run)',
                   not bad and all(dpb.get(k, 0.0) > 0.0 for k in nab), f'{bad[:3]} max|PB - PA| {dpb}')
                if not bad:
                    Pc = np.concatenate([z['P'] for z in zs], axis=1).astype(np.float64)
                    nvc = np.concatenate([z['NV'] for z in zs])
                    ids = np.concatenate([z['sids'] for z in zs]).astype(str)
                    okr = np.isfinite(Pc).all(axis=(0, 2)) & (nvc > 0)
                    fs_p = f'{work}/submission_fleetsolo.csv'
                    d = float('nan')
                    if os.path.isfile(fs_p) and okr.any():
                        fs = pd.read_csv(fs_p, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
                        mean = Pc.mean(0)[okr]
                        d = float(np.abs(fs.loc[list(ids[okr]), TARGETS].to_numpy(np.float64) - mean).max())
                    ok('K5c submission_fleetsolo.csv == the members raw mean of the worker npz P rows (served A+B / A)',
                       d <= tol, f'max|diff| {d:.2e} on {int(okr.sum())} rows')
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
        if mode in ('solo', 'strict'):                                                  # K3a
            ok(f'submission.csv ABSENT ({mode} mode, fleet not clean)', not os.path.exists(sub))
        else:
            ok('submission.csv == parent bytes (fleet not clean)', os.path.isfile(sub) and filecmp.cmp(sub, par_p, shallow=False))
    info.update(fleet_stage_min=rec.get('stage_min'), workers_s=rec.get('workers_s'), worker_min=rec.get('worker_min'),
                worker_loop_min=rec.get('worker_loop_min'), bench=rec.get('bench'),
                decode_s_per_study_per_thread=rec.get('decode_s_per_study_per_thread'), models=rec.get('names'),
                rho_mean=rec.get('rho_mean'), kernel_age_h=rec.get('kernel_age_h'), left_min=rec.get('left_min'),
                manifest=rec.get('manifest'), manifest_sha16=rec.get('manifest_sha16'), tta=rec.get('tta'))
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


def parse_expect_tta(spec):
    """K5: 'NAME=A,NAME2=AB' or a JSON file (K5_BUILD.json) holding {'tta': {NAME: MODE}} -> dict (None = no check)."""
    if not spec:
        return None
    if spec.endswith('.json') and os.path.isfile(spec):
        d = json.load(open(spec))
        d = d.get('tta', d)
        assert isinstance(d, dict) and d and all(v in ('A', 'AB') for v in d.values()), d
        return dict(d)
    out = {}
    for x in [y.strip() for y in spec.split(',') if y.strip()]:
        k, _, m = x.rpartition('=')
        assert k and m in ('A', 'AB') and k not in out, f'--expect_tta entry {x!r}: want NAME=A or NAME=AB'
        out[k] = m
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--expect_fleet', default='auto', choices=('auto', 'clean', 'fail'))
    ap.add_argument('--expect_crop', default='auto', choices=('auto', 'ok', 'fail', 'none'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--kaggle_commit', action='store_true', help='also gate on the commit receipt (parity, bench, GPUs)')
    ap.add_argument('--expect_members', default=None, help='K3: comma list of checkpoint file names that must have run')
    ap.add_argument('--expect_mode', default=None, choices=('blend', 'solo', 'strict'))
    ap.add_argument('--expect_manifest', default=None,
                    help='K3c: the LOCAL upload folder fleet_manifest.json; the commit must have attached the same bytes')
    ap.add_argument('--expect_w_main', type=float, default=None, help='K4: W_MAIN of the kernel config (0.2 main, 0.5 w50)')
    ap.add_argument('--expect_tta', default=None,
                    help="K5: NAME=A|AB[,NAME=A|AB] for every member, or a K5_BUILD.json path (its 'tta' map)")
    a = ap.parse_args(argv)
    em = [x.strip() for x in a.expect_members.split(',') if x.strip()] if a.expect_members else None
    res, info = check(a.work, a.expect_fleet, a.expect_crop, kaggle_commit=a.kaggle_commit, expect_members=em,
                      expect_mode=a.expect_mode, expect_manifest=a.expect_manifest, expect_w_main=a.expect_w_main,
                      expect_tta=parse_expect_tta(a.expect_tta))
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
