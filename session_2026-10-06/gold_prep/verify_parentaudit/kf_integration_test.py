"""Verifier's integration test for the KF hedge (parentaudit workstream), CPU only, no network.
Splices kf_patch_block.py into the REAL K5 cell 31 (K-kit) exactly as its instructions say, then drives the real
_k2_entry_inner() against a temp copy of a parent working dir (Kaggle commit output or the gold58 box mimic).
No fleet dataset is attached, so a clean stage is impossible by design: the stage must fail AFTER the KF step with
'fleet dataset not attached' (proving KF ran first and passed), or AT the KF step for the adversarial cases.
Usage: python kf_integration_test.py <cell31_source.py> <kf_patch_block.py> <parent_work_dir> <parent_csv_name>
"""
import os, sys, json, shutil, tempfile, io, contextlib
import numpy as np, pandas as pd

c31_p, kf_p, src_work, par_name = sys.argv[1:5]
C31 = open(c31_p, encoding='utf-8').read()
KF = open(kf_p, encoding='utf-8').read()
KF_BLOCK = KF[:KF.index('# ---- KF call lines')]


def splice(src, flat):
    def rep(s, old, new):
        assert s.count(old) == 1, f'anchor not unique/absent: {old[:60]!r} ({s.count(old)})'
        return s.replace(old, new)
    first, rest = src.split('\n', 1)
    assert first.startswith('K2_CFG = {') and first.count('}') >= 1
    if flat is not None:
        first = first.replace("'MAIN_MODE': 'strict', ", f"'MAIN_MODE': 'strict', 'OUTER_FLAT': {flat!r}, ", 1)
        assert 'OUTER_FLAT' in first
    s = first + '\n' + rest
    s = rep(s, "      'MEMBERS': None}", "      'MEMBERS': None, 'OUTER_FLAT': None}")
    s = rep(s, '\ndef _k2_stage(t_cell, rec, procs):', '\n' + KF_BLOCK + '\n\ndef _k2_stage(t_cell, rec, procs):')
    s = rep(s, "    n = len(pub)\n    if not _K2_FLEET_DIR:",
            "    n = len(pub)\n"
            "    r_pub_src = pub[_K2_TARGETS].to_numpy(np.float64)                                          # KF\n"
            "    if K2.get('OUTER_FLAT') is not None:                                                        # KF\n"
            "        r_pub_src = _kf_outer_flat(pub, K2['OUTER_FLAT'], rec)                                  # KF\n"
            "    if not _K2_FLEET_DIR:")
    s = rep(s, "    r_pub = _k2_rank(pub[_K2_TARGETS].to_numpy(np.float64)); r_fl = _k2_rank(F)",
            "    r_pub = _k2_rank(r_pub_src); r_fl = _k2_rank(F)                                             # KF")
    s = rep(s, "members_filter=K2.get('MEMBERS'))", "members_filter=K2.get('MEMBERS'),\n                  outer_flat=K2.get('OUTER_FLAT'))   # KF")
    return s


def run(case, flat, mutate=None, live_map='good'):
    d = tempfile.mkdtemp(prefix='kfint_')
    os.makedirs(os.path.join(d, 'diagnostics'))
    for f in ('btk_v32_rest_input.npz', '_raptor.csv', 'diagnostics/btkd_parent_after_rad.npz'):
        if os.path.isfile(os.path.join(src_work, f)):
            shutil.copyfile(os.path.join(src_work, f), os.path.join(d, f))
    shutil.copyfile(os.path.join(src_work, par_name), os.path.join(d, 'submission.csv'))
    if mutate:
        mutate(d)
    s = splice(C31, flat).replace("_K2_WORK = '/kaggle/working'", f'_K2_WORK = {d!r}', 1)
    s = s.replace('_k2_receipt = _k2_entry()', '_k2_receipt = None', 1)
    G = {'__name__': '__kfint__'}
    if live_map == 'good':
        G['_coatnet_weight'] = {t: 0.60 for t in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
                                                  'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
                                                  'Contusion', 'Fracture']}
        G['_coatnet_weight'].update({'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00,
                                     'Lateral OA': 0.75, 'Fracture': 0.75})
    elif live_map == 'drift':
        G['_coatnet_weight'] = {t: 0.60 for t in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
                                                  'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
                                                  'Contusion', 'Fracture']}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(s, 'c31_kf', 'exec'), G)
        rec = G['_k2_entry_inner']()
    log = buf.getvalue()
    st = json.load(open(os.path.join(d, '_k2_stage_fleet.json')))
    r = {'case': case, 'why': rec.get('why'), 'sub_exists': os.path.isfile(os.path.join(d, 'submission.csv')),
         'status_outer_flat': st.get('outer_flat'), 'cfg_outer_flat': rec['config'].get('OUTER_FLAT'),
         'kf_log': [l for l in log.splitlines() if 'KF outer map' in l],
         'rebuild_max_abs': (rec.get('outer_flat') or {}).get('rebuild_max_abs'),
         'mars': (rec.get('outer_flat') or {}).get('mars_flat_vs_parent')}
    shutil.rmtree(d, ignore_errors=True)
    return r


def m_missing_npz(d):
    os.remove(os.path.join(d, 'btk_v32_rest_input.npz'))


def m_side_public(d):          # wrong side: the public Raptor before the CoAt substitution
    shutil.copyfile(os.path.join(src_work, '_raptor_public.csv'), os.path.join(d, '_raptor.csv'))


def m_side_shuffled(d):        # same side, rows shuffled: must still pass (id-aligned)
    p = os.path.join(d, '_raptor.csv'); x = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    x.sample(frac=1.0, random_state=1).to_csv(p, index=False)


def m_side_perturb(d):         # one side value pushed to the top of its column
    p = os.path.join(d, '_raptor.csv'); x = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    x.loc[x.index[0], 'Lateral Meniscus'] = 0.999999; x.to_csv(p, index=False)


def m_s3_perturb(d):           # stage 3: one Lateral Meniscus value pushed to the top (map w 1.00: rebuild-blind)
    p = os.path.join(d, 'btk_v32_rest_input.npz'); z = dict(np.load(p, allow_pickle=False))
    z['values'] = z['values'].copy(); j = list(z['labels']).index('Lateral Meniscus')
    i = int(np.argmin(z['values'][:, j])); z['values'][i, j] = 10.0
    np.savez_compressed(p, **z)


def m_extra_side_row(d):
    p = os.path.join(d, '_raptor.csv'); x = pd.read_csv(p, dtype={'StudyInstanceUID': str})
    pd.concat([x, x.iloc[[0]]]).to_csv(p, index=False)


if __name__ == '__main__':
    cases = [run('default None (K5 behaviour)', None),
             run('OUTER_FLAT .60', 0.60),
             run('OUTER_FLAT .60, no live map', 0.60, live_map=None),
             run('OUTER_FLAT .60, live map drift', 0.60, live_map='drift'),
             run('OUTER_FLAT .60, npz missing', 0.60, m_missing_npz),
             run('OUTER_FLAT .60, side = _raptor_public.csv', 0.60, m_side_public),
             run('OUTER_FLAT .60, side rows shuffled', 0.60, m_side_shuffled),
             run('OUTER_FLAT .60, side 1 cell perturbed (LatMen)', 0.60, m_side_perturb),
             run('OUTER_FLAT .60, stage3 1 cell perturbed (LatMen)', 0.60, m_s3_perturb),
             run('OUTER_FLAT .60, side duplicate row', 0.60, m_extra_side_row),
             run('OUTER_FLAT 1.0 (invalid)', 1.0)]
    for c in cases:
        print(json.dumps(c, default=str))
