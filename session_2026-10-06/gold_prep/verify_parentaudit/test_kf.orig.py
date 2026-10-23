"""Exercise kf_patch_block.py against a parent working dir (a Kaggle commit output or the box mimic's working dir).
Usage: python test_kf.py <work_dir> [parent_csv]   (parent_csv default: <work>/_k2_parent.csv, else <work>/submission.csv)
CPU only, read only on <work_dir>."""
import os, sys, json
import numpy as np, pandas as pd

work = sys.argv[1]
par_p = sys.argv[2] if len(sys.argv) > 2 else (f'{work}/_k2_parent.csv' if os.path.isfile(f'{work}/_k2_parent.csv')
                                                 else f'{work}/submission.csv')
G = {'np': np, 'pd': pd, 'os': os, '_K2_WORK': work,
     '_K2_TARGETS': ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
                     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']}


class _K2Fail(Exception):
    pass


G['_K2Fail'] = _K2Fail
G['_k2_rank'] = lambda a: pd.DataFrame(np.asarray(a, np.float64)).rank(method='average', pct=True).to_numpy()
G['_k2_log'] = lambda m: print('[kf]', m, flush=True)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kf_patch_block.py')).read(), G)
pub = pd.read_csv(par_p, dtype={'StudyInstanceUID': str})
res = {'work': work, 'parent': par_p, 'n': len(pub)}

# 1) normal path, with and without a live cell-28 map
rec = {}
P60 = G['_kf_outer_flat'](pub, 0.60, rec)
res['flat60'] = rec['outer_flat']
G['_coatnet_weight'] = {t: 0.60 for t in G['_K2_TARGETS']}
G['_coatnet_weight'].update(G['_KF_PARENT_MAP'])
rec2 = {}
P60b = G['_kf_outer_flat'](pub, 0.60, rec2)
res['live_map_ok_identical'] = bool(np.array_equal(P60, P60b))

# 2) negative tests: every one must raise _K2Fail
neg = {}
def expect_fail(name, fn):
    try:
        fn(); neg[name] = 'NO RAISE (BAD)'
    except _K2Fail as e:
        neg[name] = 'raised: ' + str(e)[:90]
G['_coatnet_weight'] = dict(G['_coatnet_weight'], **{'Lateral Meniscus': 0.9})
expect_fail('live map drift', lambda: G['_kf_outer_flat'](pub, 0.60, {}))
del G['_coatnet_weight']
bad = pub.copy(); bad.loc[bad.index[0], 'ACL'] = 1.0 - bad.loc[bad.index[0], 'ACL'] + 1e-3
expect_fail('parent bytes != rebuild', lambda: G['_kf_outer_flat'](bad, 0.60, {}))
expect_fail('flat out of range', lambda: G['_kf_outer_flat'](pub, 1.0, {}))
bad2 = pub.copy(); bad2.loc[bad2.index[0], 'StudyInstanceUID'] = 'x'
expect_fail('id mismatch', lambda: G['_kf_outer_flat'](bad2, 0.60, {}))
res['negative_tests'] = neg
res['all_negative_raise'] = all(v.startswith('raised') for v in neg.values())

# 3) what changes: per-label max abs shift; the 7 untouched labels must be identical to the parent
par = pub[G['_K2_TARGETS']].to_numpy(np.float64)
d = np.abs(P60 - par).max(0)
res['max_abs_shift_per_label'] = {t: round(float(x), 4) for t, x in zip(G['_K2_TARGETS'], d)}
same = [t for t in G['_K2_TARGETS'] if t not in G['_KF_PARENT_MAP']]
res['untouched_labels_identical'] = bool(all(d[G['_K2_TARGETS'].index(t)] == 0 for t in same))
print(json.dumps(res, indent=1))
