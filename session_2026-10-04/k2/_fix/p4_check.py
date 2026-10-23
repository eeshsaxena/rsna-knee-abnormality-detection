import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_lib import patch
K2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
patch(f'{K2}/k2_check.py', [
("""  python k2_check.py --work <dir with submission*.csv> [--expect_fleet clean|fail|auto] [--expect_crop ok|fail|none|auto]
""",
"""  python k2_check.py --work <dir with submission*.csv> [--expect_fleet clean|fail|auto] [--expect_crop ok|fail|none|auto]
        [--kaggle_commit]

--kaggle_commit (the downloaded output of the <= 3-study Kaggle commit, fleet clean): also gates on the receipt, so
the decision never depends on whether the workers' [fleet g] lines reached the notebook log: 2 GPUs, every worker
decode probe match >= 1 and mismatch 0, shard-0 PARITY re-encode ran (frac_gt8 <= .01, no mismatch), one microbench
dict per GPU without an error, n_fail and n_blank 0 on every worker.
"""),
("""def check(work, expect_fleet='auto', expect_crop='auto', tol=1e-9):
    res, info = [], {}

    def ok(name, cond, detail=''):
        res.append((name, bool(cond), str(detail)[:300]))
""",
"""def check(work, expect_fleet='auto', expect_crop='auto', tol=1e-9, kaggle_commit=False):
    res, info = [], {}

    def ok(name, cond, detail=''):
        res.append((name, bool(cond), str(detail)[:300]))

    nested = sorted(os.path.relpath(os.path.join(dp, f), work).replace('\\', '/') for dp, _, fs in os.walk(work)
                    if os.path.realpath(dp) != os.path.realpath(work) for f in fs
                    if f.startswith('submission') and f.endswith('.csv'))
    ok('no submission*.csv below the top level (C6: only top-level files are submittable)', not nested, nested[:6])
"""),
("""        ok('workers kept the full member count', rec.get('n_keep') and all(k == rec.get('models') for k in rec['n_keep']),
           f'{rec.get("n_keep")} of {rec.get("models")}')
""",
"""        ok('workers kept the full member count', rec.get('n_keep') and all(k == rec.get('models') for k in rec['n_keep']),
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
"""),
("""            sand = f'{work}/_k2_crop/submission_fleetsolo.csv'
""",
"""            sand = f'{work}/_k2_crop/k1out_fleetsolo.csv'                 # C6: renamed by the CF writer
"""),
("""    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    res, info = check(a.work, a.expect_fleet, a.expect_crop)
""",
"""    ap.add_argument('--json', default=None)
    ap.add_argument('--kaggle_commit', action='store_true', help='also gate on the commit receipt (parity, bench, GPUs)')
    a = ap.parse_args(argv)
    res, info = check(a.work, a.expect_fleet, a.expect_crop, kaggle_commit=a.kaggle_commit)
"""),
])
