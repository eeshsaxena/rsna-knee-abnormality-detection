#!/usr/bin/env python3
"""t_k8_kf.py (laptop CPU; TEST ONLY; 2026-10-07, Claude): the K8 KF twin end to end + k2_check's K8-KF gate with positive
and negative controls. No GPU (stub torch), no network.

Drives the REAL cell 31 of a built K8 kit (rsna_sub_k8_kf.ipynb = KF cell + K2_CFG OUTER_FLAT 0.6, and
rsna_sub_k8_fleet.ipynb = the K5 cell) against a temp copy of a REAL parent working dir, with a stub fleet dataset
(fleet_manifest.json naming the kernel's 10 members with their TTA modes + a stub worker that serves a fixed prediction
table, as gold_prep/verify_parentaudit/kf_e2e_test.py does). Parent sources:
  k5w50 : SP/k5/build/out_w50_v1 (the real K5 w50 Kaggle commit, 3 studies; parent = _k2_parent.csv)
  mimic58: tests/_kf/mimic58 (the box gold58 parent mimic, 58 studies; parent = submission.csv), where the flat and the map
           parents RANK DIFFERENTLY, so the blend recompute itself can see KF.
Checks (each PASS / FAIL line; final 'TK8KF PASS n/n'):
  E1 KF kernel: clean stage; status + receipt carry outer_flat .6; _k2_parent.csv == the true parent bytes
  E2 main kernel (K5 cell) on the same inputs: clean; no outer_flat anywhere
  E3 the KF cell with OUTER_FLAT None writes submission*.csv byte-identical to the K5 cell (default = K5 behaviour)
  P1 k2_check KF output --expect_outer_flat 0.6 -> PASS;   P2 main output without it -> PASS
  N1 KF output WITHOUT --expect_outer_flat -> FAIL (the K8-KF absent check; on 58 studies also the map-parent recompute)
  N2 main output WITH --expect_outer_flat 0.6 -> FAIL
  N3 status outer_flat edited to 0.5 -> FAIL;  N4 receipt outer_flat removed -> FAIL;  N5 receipt rebuild_max_abs 1e-3 -> FAIL
  N6 receipt changed_labels altered -> FAIL;  N7 diagnostics/btkd_parent_after_rad.npz removed -> FAIL
  N8 _raptor.csv perturbed after the run (map rebuild != parent) -> FAIL;  N9 config MAIN_MODE edited to blend -> FAIL
  N10 consistent tampering: the wXX + submission.csv replaced by the MAP-parent blends, status sha256 updated, receipt left
      claiming KF: mimic58 -> FAIL (the flat recompute sees it); k5w50 (3 studies) -> PASS = the documented blind spot of
      a 3-study commit (the config / status / receipt assertions are the evidence there)
  C1 KF cell, live cell-28 map drift -> not clean, submission.csv ABSENT; k2_check --expect_fleet fail --expect_outer_flat
     0.6 -> PASS (consistent fail-loud);  C2 KF cell, stage-3 npz missing -> ABSENT, same check PASS
  python tests/t_k8_kf.py <build dir with kernel/rsna-sub-k8-kf and kernel/rsna-sub-k8-fleet>
"""
import contextlib, io, json, os, shutil, sys, tempfile, types
import numpy as np, pandas as pd

T_ = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K8 = os.path.dirname(T_)
SP = os.path.dirname(K8)
sys.path.insert(0, K8)
import k2_check  # noqa: E402

T = k2_check.TARGETS
MAP = {t: 0.60 for t in T}
MAP.update({'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75})
WORKER = r'''
import os, json, numpy as np, pandas as pd
out = os.environ['FLEET_OUT']; work = os.path.dirname(out); fd = os.environ['FLEET_DIR']
g, ng = int(os.environ['FLEET_SHARD']), int(os.environ['FLEET_NSHARD'])
man = json.load(open(os.path.join(fd, 'fleet_manifest.json'))); names = man['checkpoints']
ids = pd.read_csv(os.path.join(work, '_k2_parent.csv'), dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()[g::ng]
F = pd.read_csv(os.path.join(fd, 'stub_fleet.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
P = np.stack([F.loc[ids].to_numpy(np.float32) for _ in names])
modes = {n: man['sources'][n].get('tta', 'A') for n in names}; rows = len(ids)
fwd = {n: {'A': rows, 'B': rows if modes[n] == 'AB' else 0} for n in names}
np.savez(out, names=np.array(names), n_keep=len(names), hard_stopped=False, minutes=0.0, loop_min=0.0, n_fail=0,
         n_blank=0, parity=json.dumps({}), stats=json.dumps({'decode_s': 0.0}), P=P, NV=np.full(rows, 4),
         sids=np.array(ids), tta=json.dumps({'modes': modes, 'half': 0.5 - 1e-3, 'rows': rows, 'fwd': fwd}))
'''
R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {str(extra)[:240]}', flush=True)


def cell31(nb_path):
    nb = json.load(open(nb_path, encoding='utf-8'))
    return ''.join(nb['cells'][31]['source'])


def run_cell(src, parent_dir, par_name, fleet_csv, live_map='good', mutate=None, cfg_edit=None):
    """Run cell 31 text in a fresh namespace on a temp copy of the parent dir; returns (work dir, receipt, stdout)."""
    d = tempfile.mkdtemp(prefix='k8kf_'); fd = tempfile.mkdtemp(prefix='k8kf_fleet_')
    os.makedirs(f'{d}/diagnostics')
    for f in ('btk_v32_rest_input.npz', '_raptor.csv', 'diagnostics/btkd_parent_after_rad.npz'):
        shutil.copyfile(f'{parent_dir}/{f}', f'{d}/{f}')
    shutil.copyfile(f'{parent_dir}/{par_name}', f'{d}/submission.csv')
    if mutate:
        mutate(d)
    hdr, body = src.split('\n', 1)
    cfg = eval(hdr.split('=', 1)[1].split('#')[0])
    if cfg_edit:
        cfg = cfg_edit(dict(cfg)); hdr = f'K2_CFG = {cfg!r}   # edited by t_k8_kf.py'
    names, tta = cfg['MEMBERS'], cfg.get('TTA') or {}
    par = pd.read_csv(f'{d}/submission.csv', dtype={'StudyInstanceUID': str})
    fl = pd.read_csv(fleet_csv, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    fl.loc[par.StudyInstanceUID, T].reset_index().to_csv(f'{fd}/stub_fleet.csv', index=False)
    json.dump({'checkpoints': names, 'sources': {n: {'tta': tta.get(n, 'A'), 'sha256': 'stub'} for n in names}},
              open(f'{fd}/fleet_manifest.json', 'w'))
    open(f'{fd}/fleet_worker.py', 'w').write(WORKER)
    for n in names:
        open(f'{fd}/{n}', 'w').write('stub')
    s = hdr + '\n' + body
    for old, new in (("_K2_WORK = '/kaggle/working'", f'_K2_WORK = {d!r}'),
                     ("for _c in ('/kaggle/input/datasets/eeshsaxena/rsna-knee-cnxt-fleet', '/kaggle/input/rsna-knee-cnxt-fleet'):",
                      f'for _c in ({fd!r},):'),
                     ('_k2_receipt = _k2_entry()', '_k2_receipt = None')):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    fake = types.ModuleType('torch')
    fake.cuda = types.SimpleNamespace(device_count=lambda: 2, is_available=lambda: True, empty_cache=lambda: None)
    real = sys.modules.get('torch'); sys.modules['torch'] = fake
    G = {'__name__': '__k8kf__'}
    if live_map == 'good':
        G['_coatnet_weight'] = dict(MAP)
    elif live_map == 'drift':
        G['_coatnet_weight'] = {t: 0.60 for t in T}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(s, 'c31', 'exec'), G)
            rec = G['_k2_entry_inner']()
    finally:
        if real is None:
            sys.modules.pop('torch', None)
        else:
            sys.modules['torch'] = real
    shutil.rmtree(fd, ignore_errors=True)
    return d, rec, buf.getvalue()


def chk(work, *extra):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = k2_check.main(['--work', work, '--expect_mode', 'strict', '--expect_w_main', '0.5', *extra])
    out = buf.getvalue()
    fails = [l for l in out.splitlines() if l.startswith('FAIL')]
    last = [l for l in out.splitlines() if l.startswith('K2CHECK')]
    return rc, fails, (last[-1] if last else '')


def edit_json(p, fn):
    j = json.load(open(p)); fn(j); json.dump(j, open(p, 'w'), indent=1, default=str)


def copy(work):
    d = tempfile.mkdtemp(prefix='k8kfc_'); shutil.rmtree(d); shutil.copytree(work, d); return d


def main():
    bd = sys.argv[1] if len(sys.argv) > 1 else f'{K8}/build_sim'
    kf_src = cell31(f'{bd}/kernel/rsna-sub-k8-kf/rsna_sub_k8_kf.ipynb')
    main_src = cell31(f'{bd}/kernel/rsna-sub-k8-fleet/rsna_sub_k8_fleet.ipynb')
    assert "'OUTER_FLAT': 0.6" in kf_src.split('\n', 1)[0] and 'OUTER_FLAT' not in main_src.split('\n', 1)[0]
    srcs = {'k5w50': (f'{SP}/k5/build/out_w50_v1', '_k2_parent.csv', f'{SP}/k5/build/out_w50_v1/submission_fleetsolo.csv'),
            'mimic58': (f'{T_}/_kf/mimic58', 'submission.csv', f'{T_}/_kf/mimic58/stub_fleet_src.csv')}
    for tag, (pdir, pname, fcsv) in srcs.items():
        print(f'==== {tag} ({pdir})', flush=True)
        dk, rk, lk = run_cell(kf_src, pdir, pname, fcsv)
        dm, rm, lm = run_cell(main_src, pdir, pname, fcsv)
        st = json.load(open(f'{dk}/_k2_stage_fleet.json'))
        ok(f'{tag} E1 KF kernel clean, status + receipt outer_flat .6, _k2_parent.csv == true parent bytes',
           rk.get('clean') and st.get('outer_flat') == 0.6 and (rk.get('outer_flat') or {}).get('flat') == 0.6 and
           open(f'{dk}/_k2_parent.csv', 'rb').read() == open(f'{pdir}/{pname}', 'rb').read(),
           [l for l in lk.splitlines() if 'KF outer map' in l] or rk.get('why'))
        stm = json.load(open(f'{dm}/_k2_stage_fleet.json'))
        ok(f'{tag} E2 main kernel (K5 cell) clean, no outer_flat in status / receipt',
           rm.get('clean') and 'outer_flat' not in stm and 'outer_flat' not in rm and 'OUTER_FLAT' not in rm['config'], rm.get('why'))
        dn, rn, _ = run_cell(kf_src, pdir, pname, fcsv, cfg_edit=lambda c: {k: v for k, v in c.items() if k != 'OUTER_FLAT'})
        same = all(open(f'{dn}/{f}', 'rb').read() == open(f'{dm}/{f}', 'rb').read()
                   for f in ('submission.csv', 'submission_w40.csv', 'submission_w50.csv', 'submission_w60.csv',
                             'submission_fleetsolo.csv', '_k2_parent.csv'))
        ok(f'{tag} E3 KF cell with OUTER_FLAT None == K5 cell outputs byte for byte', rn.get('clean') and same)
        kd = k2_check.kf_independent(dk, pd.read_csv(f'{dk}/_k2_parent.csv', dtype={'StudyInstanceUID': str}), 0.6)[1]
        print(f'  {tag}: flat ranks differ from the map ranks: {kd["flat_ranks_differ_from_map"]} (MARS {kd["mars_flat_vs_parent"]:.4f})')
        rc, f, last = chk(dk, '--expect_fleet', 'clean', '--expect_outer_flat', '0.6')
        ok(f'{tag} P1 KF output + --expect_outer_flat 0.6 -> PASS', rc == 0, last + ' ' + ' | '.join(f))
        rc, f, last = chk(dm, '--expect_fleet', 'clean')
        ok(f'{tag} P2 main output without --expect_outer_flat -> PASS', rc == 0, last + ' ' + ' | '.join(f))
        rc, f, last = chk(dk, '--expect_fleet', 'clean')
        ok(f'{tag} N1 KF output WITHOUT --expect_outer_flat -> FAIL', rc == 1 and any('no KF flat parent' in x for x in f),
           f'{last} {len(f)} FAIL line(s): ' + ' | '.join(x[:80] for x in f))
        rc, f, last = chk(dm, '--expect_fleet', 'clean', '--expect_outer_flat', '0.6')
        ok(f'{tag} N2 main output WITH --expect_outer_flat 0.6 -> FAIL', rc == 1, last)
        muts = {
            'N3 status outer_flat 0.5': lambda w: edit_json(f'{w}/_k2_stage_fleet.json', lambda j: j.update(outer_flat=0.5)),
            'N4 receipt outer_flat removed': lambda w: edit_json(f'{w}/fleet_receipt.json', lambda j: j.pop('outer_flat')),
            'N5 receipt rebuild_max_abs 1e-3': lambda w: edit_json(f'{w}/fleet_receipt.json',
                                                                   lambda j: j['outer_flat'].update(rebuild_max_abs=1e-3)),
            'N6 receipt changed_labels altered': lambda w: edit_json(f'{w}/fleet_receipt.json',
                                                                     lambda j: j['outer_flat'].update(changed_labels=['ACL'])),
            'N7 diagnostics npz removed': lambda w: os.remove(f'{w}/diagnostics/btkd_parent_after_rad.npz'),
            'N8 _raptor.csv perturbed': lambda w: perturb_side(w),
            'N9 config MAIN_MODE blend': lambda w: edit_json(f'{w}/fleet_receipt.json', lambda j: j['config'].update(MAIN_MODE='blend')),
        }
        for nm, fn in muts.items():
            w = copy(dk); fn(w)
            rc, f, last = chk(w, '--expect_fleet', 'clean', '--expect_outer_flat', '0.6')
            ok(f'{tag} {nm} -> FAIL', rc == 1, f'{last} ' + ' | '.join(x[:90] for x in f))
            shutil.rmtree(w, ignore_errors=True)
        w = copy(dk); tamper_map(w)
        rc, f, last = chk(w, '--expect_fleet', 'clean', '--expect_outer_flat', '0.6')
        if kd['flat_ranks_differ_from_map']:
            ok(f'{tag} N10 consistent tampering (map-parent blends, sha updated, receipt claims KF) -> FAIL by the flat recompute',
               rc == 1 and any('independent P+F rank blend' in x for x in f), f'{last} ' + ' | '.join(x[:90] for x in f))
        else:
            ok(f'{tag} N10 consistent tampering on a {kd and "3"}-study output -> PASS (documented blind spot: identical ranks)',
               rc == 0, last)
        shutil.rmtree(w, ignore_errors=True)
        for nm, kw in (('C1 live cell-28 map drift', dict(live_map='drift')),
                       ('C2 stage-3 npz missing', dict(mutate=lambda d: os.remove(f'{d}/btk_v32_rest_input.npz')))):
            dc, rcp, lc = run_cell(kf_src, pdir, pname, fcsv, **kw)
            rc, f, last = chk(dc, '--expect_fleet', 'fail', '--expect_outer_flat', '0.6')
            ok(f'{tag} {nm}: not clean, submission.csv ABSENT, k2_check --expect_fleet fail -> PASS',
               not rcp.get('clean') and not os.path.exists(f'{dc}/submission.csv') and rc == 0 and 'KF' in str(rcp.get('why')),
               f'{rcp.get("why")} | {last}')
            shutil.rmtree(dc, ignore_errors=True)
        for x in (dk, dm, dn):
            shutil.rmtree(x, ignore_errors=True)
    print(f'TK8KF {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
    return 0 if all(R) else 1


def perturb_side(w):
    p = f'{w}/_raptor.csv'; x = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    j = 'Lateral Meniscus'; i = int(np.argmin(x[j].to_numpy())); x.loc[x.index[i], j] = 0.999999; x.to_csv(p, index=False)


def tamper_map(w):
    """Replace every wXX (and submission.csv) with the MAP-parent blend, update the status sha256 (receipt untouched)."""
    pub = pd.read_csv(f'{w}/_k2_parent.csv', dtype={'StudyInstanceUID': str})
    F = pd.read_csv(f'{w}/submission_fleetsolo.csv', dtype={'StudyInstanceUID': str})[T].to_numpy(np.float64)
    rk = k2_check._rank
    st = json.load(open(f'{w}/_k2_stage_fleet.json'))
    for wt in (0.4, 0.5, 0.6):
        b = rk((1 - wt) * rk(pub[T].to_numpy(np.float64)) + wt * rk(F))
        df = pd.DataFrame(b, columns=T); df.insert(0, 'StudyInstanceUID', pub['StudyInstanceUID'].values)
        df.to_csv(f'{w}/{k2_check.wname(wt)}', index=False)
        if wt == 0.5:
            df.to_csv(f'{w}/submission.csv', index=False)
    for f in ('submission_w40.csv', 'submission_w50.csv', 'submission_w60.csv', 'submission.csv'):
        st['files'][f] = k2_check._sha(f'{w}/{f}')
    json.dump(st, open(f'{w}/_k2_stage_fleet.json', 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
