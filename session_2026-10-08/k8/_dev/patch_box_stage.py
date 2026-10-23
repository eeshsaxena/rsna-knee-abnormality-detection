"""_dev/patch_box_stage.py (K8, 2026-10-07): K8-KF edits to k8/box/k2_box_stage.py (a K7 copy). Run once.
  --kf_parent ok|perturb|missing : synthesize the parent's cell-28 KF inputs for the A0 studies (the A0 never runs the
      parent): btk_v32_rest_input.npz (stage 3), diagnostics/btkd_parent_after_rad.npz (its cell-26 copy), _raptor.csv
      (side) and submission.csv = the cell-28 routing with the parent MAP (so the KF self-check can pass). perturb: one
      side cell moved AFTER the parent was built (the rebuild check must fail); missing: no stage-3 npz.
  --expect_outer_flat X : passed to k2_check.check (K8-KF assertions + the independent flat-parent recompute)."""
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(HERE, 'box', 'k2_box_stage.py')
s = open(p, encoding='utf-8').read()
assert 'K8-KF' not in s, 'already patched'


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''(PB NaN), and, when the microbench ran, the receipt's bench passes == 2 for AB / 1 for A.
"""''', '''(PB NaN), and, when the microbench ran, the receipt's bench passes == 2 for AB / 1 for A.
K8 (scratchpad/k8, 2026-10-07):
  K8-KF --kf_parent ok|perturb|missing: the A0 never runs the parent, so for the KF twin (OUTER_FLAT) the parent's cell-28
        inputs are SYNTHESIZED for the A0 studies: btk_v32_rest_input.npz (stage 3, random), its cell-26 copy
        diagnostics/btkd_parent_after_rad.npz (identical values), _raptor.csv (side, random) and submission.csv = the
        cell-28 routing rank((1 - w_l) rank(S3) + w_l side) with the parent MAP (ACL .75, MedMen .80, LatMen 1.00,
        LatOA .75, Fracture .75, else .60), so the KF self-check can pass. perturb: one Lateral Meniscus side cell is moved
        to the top AFTER the parent was built (the KF rebuild check must fail -> strict deletes submission.csv);
        missing: no stage-3 npz (KF input missing -> fail loud). --expect_outer_flat X goes to k2_check.check.
"""''')
rep('''    ap.add_argument('--threads', type=int, default=4)
''', '''    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--kf_parent', default=None, choices=('ok', 'perturb', 'missing'),
                    help='K8-KF: synthesize the parent cell-28 KF inputs (see the header)')
    ap.add_argument('--expect_outer_flat', type=float, default=None, help='K8-KF: k2_check --expect_outer_flat')
''')
rep('''    par = par.sample(frac=1, random_state=3).reset_index(drop=True)          # parent order != test.csv order
''', '''    if a.kf_parent:                                      # K8-KF: parent = the cell-28 map routing of synthetic inputs
        kmap = {'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75}
        rg = np.random.default_rng(11)
        s3 = rg.normal(0, 1, (len(ids), 12)); side = rg.uniform(0, 1, (len(ids), 12))
        rk = lambda x: pd.DataFrame(np.asarray(x, np.float64)).rank(method='average', pct=True).to_numpy()
        w = np.array([kmap.get(t, 0.60) for t in T], np.float64)[None, :]
        par = pd.DataFrame(rk((1.0 - w) * rk(s3) + w * side), columns=T); par.insert(0, 'StudyInstanceUID', ids)
        os.makedirs(f'{work}/diagnostics')
        if a.kf_parent != 'missing':
            np.savez(f'{work}/btk_v32_rest_input.npz', labels=np.array(T), study_uids=np.array(ids), values=s3)
        np.savez(f'{work}/diagnostics/btkd_parent_after_rad.npz', labels=np.array(T), study_uids=np.array(ids), values=s3)
        if a.kf_parent == 'perturb':                     # after the parent was built: the rebuild must now disagree
            j = T.index('Lateral Meniscus'); side = side.copy(); side[int(np.argmin(side[:, j])), j] = 0.999999
        sdf = pd.DataFrame(side, columns=T); sdf.insert(0, 'StudyInstanceUID', ids)
        sdf.sample(frac=1, random_state=5).to_csv(f'{work}/_raptor.csv', index=False)
    par = par.sample(frac=1, random_state=3).reset_index(drop=True)          # parent order != test.csv order
''')
rep('''    res, info = k2_check.check(work, expect_manifest=man_p if has_src else None, expect_tta=ett)       # K3c, K5
''', '''    res, info = k2_check.check(work, expect_manifest=man_p if has_src else None, expect_tta=ett,       # K3c, K5
                               expect_outer_flat=a.expect_outer_flat)                                  # K8-KF
''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched', p)
