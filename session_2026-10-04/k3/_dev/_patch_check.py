import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
p = 'k2_check.py'
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''  python k2_check.py --work <dir with submission*.csv> [--expect_fleet clean|fail|auto] [--expect_crop ok|fail|none|auto]
        [--kaggle_commit]
''', '''  python k2_check.py --work <dir with submission*.csv> [--expect_fleet clean|fail|auto] [--expect_crop ok|fail|none|auto]
        [--kaggle_commit] [--expect_members NAME[,NAME]] [--expect_mode blend|solo|strict]

K3 (scratchpad/k3): MAIN_MODE 'strict' = submission.csv is the W_MAIN blend twin when clean and ABSENT when not clean
(like 'solo', whose twin is the fleetsolo file); the member filter (K2_CFG MEMBERS) sets the expected member names and
count; --expect_members / --expect_mode pin both from outside (the commit check: the right files ran in the right mode).
''')
rep('''def check(work, expect_fleet='auto', expect_crop='auto', tol=1e-9, kaggle_commit=False):
    res, info = [], {}
''', '''def check(work, expect_fleet='auto', expect_crop='auto', tol=1e-9, kaggle_commit=False, expect_members=None,
          expect_mode=None):
    res, info = [], {}
''')
rep('''    clean = st.get('clean') is True
    if expect_fleet != 'auto':
        ok(f'fleet stage outcome == {expect_fleet}', clean == (expect_fleet == 'clean'), st.get('why'))
    info.update(fleet_clean=clean, fleet_why=st.get('why'), n=n)
''', '''    clean = st.get('clean') is True
    mode = cfg.get('MAIN_MODE') or st.get('main_mode') or 'blend'                      # K3a
    want = cfg.get('MEMBERS') if 'MEMBERS' in cfg else st.get('members_filter')         # K3b
    if expect_fleet != 'auto':
        ok(f'fleet stage outcome == {expect_fleet}', clean == (expect_fleet == 'clean'), st.get('why'))
    if expect_mode:
        ok(f'MAIN_MODE == {expect_mode}', mode == expect_mode, mode)
    if expect_members is not None:
        ok(f'member filter in the kernel config == {expect_members}', want is not None and sorted(want) == sorted(expect_members),
           want)
    info.update(fleet_clean=clean, fleet_why=st.get('why'), n=n, main_mode=mode, members_filter=want)
''')
rep('''            twin = 'submission_fleetsolo.csv' if cfg.get('MAIN_MODE') == 'solo' else wname(w_main)
            ok('submission.csv == its fail-loud twin ' + twin, filecmp.cmp(sub, f'{work}/{twin}', shallow=False))''',
    '''            twin = 'submission_fleetsolo.csv' if mode == 'solo' else wname(w_main)           # strict: as blend
            ok('submission.csv == its fail-loud twin ' + twin, filecmp.cmp(sub, f'{work}/{twin}', shallow=False))''')
rep('''        ok('workers kept the full member count', rec.get('n_keep') and all(k == rec.get('models') for k in rec['n_keep']),
           f'{rec.get("n_keep")} of {rec.get("models")}')''',
    '''        ok('workers kept the full member count', rec.get('n_keep') and all(k == rec.get('models') for k in rec['n_keep']),
           f'{rec.get("n_keep")} of {rec.get("models")}')
        if want is not None:                                                            # K3b
            ok(f'member filter: the workers ran exactly {want}', sorted(rec.get('names') or []) == sorted(want)
               and rec.get('models') == len(want), f'names {rec.get("names")} models {rec.get("models")}')
        if expect_members is not None:
            ok(f'the workers ran exactly {expect_members}', sorted(rec.get('names') or []) == sorted(expect_members)
               and rec.get('models') == len(expect_members), rec.get('names'))''')
rep('''        if cfg.get('MAIN_MODE') == 'solo' or st.get('main_mode') == 'solo':
            ok('submission.csv ABSENT (solo mode, fleet not clean)', not os.path.exists(sub))''',
    '''        if mode in ('solo', 'strict'):                                                  # K3a
            ok(f'submission.csv ABSENT ({mode} mode, fleet not clean)', not os.path.exists(sub))''')
rep('''                rho_mean=rec.get('rho_mean'), kernel_age_h=rec.get('kernel_age_h'), left_min=rec.get('left_min'))''',
    '''                rho_mean=rec.get('rho_mean'), kernel_age_h=rec.get('kernel_age_h'), left_min=rec.get('left_min'),
                manifest=rec.get('manifest'))''')
rep('''    ap.add_argument('--kaggle_commit', action='store_true', help='also gate on the commit receipt (parity, bench, GPUs)')
    a = ap.parse_args(argv)
    res, info = check(a.work, a.expect_fleet, a.expect_crop, kaggle_commit=a.kaggle_commit)''',
    '''    ap.add_argument('--kaggle_commit', action='store_true', help='also gate on the commit receipt (parity, bench, GPUs)')
    ap.add_argument('--expect_members', default=None, help='K3: comma list of checkpoint file names that must have run')
    ap.add_argument('--expect_mode', default=None, choices=('blend', 'solo', 'strict'))
    a = ap.parse_args(argv)
    em = [x.strip() for x in a.expect_members.split(',') if x.strip()] if a.expect_members else None
    res, info = check(a.work, a.expect_fleet, a.expect_crop, kaggle_commit=a.kaggle_commit, expect_members=em,
                      expect_mode=a.expect_mode)''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print('patched')
