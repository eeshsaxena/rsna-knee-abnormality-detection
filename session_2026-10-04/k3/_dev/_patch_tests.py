import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ run_scenario.py
p = 'tests/run_scenario.py'
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (p, s.count(old), old[:80])
    s = s.replace(old, new)


rep('''cfg = dict(nb='fleet', comp='big', fds='real', stub=None, crop=False, age_h=7.0, parent='normal', stale=False,
           write_fail=None, fast_clock=False, noclock=False, expect_fleet='clean', expect_crop='none')''',
    '''cfg = dict(nb='fleet', comp='big', fds='real', stub=None, crop=False, age_h=7.0, parent='normal', stale=False,
           write_fail=None, fast_clock=False, noclock=False, expect_fleet='clean', expect_crop='none',
           outer=False, members=None)''')
rep('''    'solo_ok':          dict(nb='solo', fds='stub', stub='ok'),
    'solo_fail':        dict(nb='solo', fds='stub', stub='fastfail', expect_fleet='fail'),''',
    '''    'solo_ok':          dict(nb='solo', fds='stub', stub='ok'),
    'solo_fail':        dict(nb='solo', fds='stub', stub='fastfail', expect_fleet='fail'),
    # ---- K3: strict main mode (K3a) and the member filter (K3b)
    'strict_ok':        dict(nb='strict', fds='stub', stub='ok'),
    'strict_fail':      dict(nb='strict', fds='stub', stub='fastfail', expect_fleet='fail'),
    'strict_writefail': dict(nb='strict', fds='stub', stub='ok', write_fail='submission.csv', expect_fleet='fail'),
    'strict_lowtime':   dict(nb='strict', fds='stub', stub='ok', age_h=8.3, expect_fleet='fail'),
    'strict_nods':      dict(nb='strict', fds='none', expect_fleet='fail'),
    'strict_outer':     dict(nb='strict', fds='stub', stub='ok', outer=True, expect_fleet='fail'),
    'solo_outer':       dict(nb='solo', fds='stub', stub='ok', outer=True, expect_fleet='fail'),
    'blend_outer':      dict(nb='fleet', fds='stub', stub='ok', outer=True, expect_fleet='fail'),
    'strict_ttr':       dict(nb='strict_ttr', fds='ttr', members='ttr_pair'),
    'strict_ttr_all':   dict(nb='strict', fds='ttr'),
    'ttrsolo_ok':       dict(nb='ttrsolo', fds='ttr', members='ttr_only'),
    'members_stub_ok':  dict(nb='stubfilter', fds='stubman', stub='okman', members='stub_b'),
    'members_mismatch': dict(nb='stubfilter', fds='stubman', stub='ok', expect_fleet='fail'),
    'members_unknown':  dict(nb='unknown', fds='stubman', stub='okman', expect_fleet='fail'),''')
rep('''      'solo': f'{S}/nb/rsna-sub-k2-solo/rsna_sub_k2_solo.ipynb'}[cfg['nb']]''',
    '''      'solo': f'{S}/nb/rsna-sub-k2-solo/rsna_sub_k2_solo.ipynb',
      'strict': f'{S}/nb/rsna-sub-k3-strict/rsna_sub_k3_strict.ipynb',
      'strict_ttr': f'{S}/nb/rsna-sub-k3-strictttr/rsna_sub_k3_strictttr.ipynb',
      'ttrsolo': f'{S}/nb/rsna-sub-k3-ttrsolo/rsna_sub_k3_ttrsolo.ipynb',
      'stubfilter': f'{S}/nb/rsna-sub-k3-stubfilter/rsna_sub_k3_stubfilter.ipynb',
      'unknown': f'{S}/nb/rsna-sub-k3-unknown/rsna_sub_k3_unknown.ipynb'}[cfg['nb']]
TTRM = json.load(open(f'{S}/ttr_members.json'))            # written by smoke stage(): the two real-worker member names
EXPECT_MEMBERS = {None: None, 'ttr_pair': [TTRM['f1'], TTRM['ttr']], 'ttr_only': [TTRM['ttr']],
                  'stub_b': ['stub_b.pt']}[cfg['members']]''')
rep('''for name, src in cells:
    if name == 'k2_fleet_stage':
        torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 2; torch.cuda.empty_cache = lambda: None
    t0 = time.perf_counter()
    try:
        exec(compile(redirect(src), name, 'exec'), g)
        escaped = None''',
    '''OUTER_PAR = None
for name, src in cells:
    if name == 'k2_fleet_stage':
        torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 2; torch.cuda.empty_cache = lambda: None
    t0 = time.perf_counter()
    try:
        if cfg['outer'] and name == 'k2_fleet_stage':
            # K3a outer-handler test (harness only): the cell text minus its last line, then an inner stage that copies
            # the parent, writes a fake fleet main file and raises OUTSIDE the guarded path; then the real _k2_entry()
            body = redirect(src)
            assert body.rstrip().endswith('_k2_receipt = _k2_entry()')
            exec(compile(body.rstrip()[:-len('_k2_receipt = _k2_entry()')], name, 'exec'), g)

            def _boom():
                shutil.copyfile(g['_K2_SUB'], g['_K2_PAR'])
                fake = pd.read_csv(g['_K2_SUB']); fake[TARGETS] = 0.5; fake.to_csv(g['_K2_SUB'], index=False)
                open(f'{WORK}/submission_fleetsolo.csv', 'w').write('x')
                raise RuntimeError('injected outside the guarded path')
            g['_k2_entry_inner'] = _boom
            g['_k2_receipt'] = g['_k2_entry']()
        else:
            exec(compile(redirect(src), name, 'exec'), g)
        escaped = None''')
rep('''if cfg['parent'] == 'normal' and cfg['nb'] != 'solo':
    res, info = k2_check.check(WORK, cfg['expect_fleet'], cfg['expect_crop'])
else:
    res, info = [], {}''',
    '''if cfg['parent'] == 'normal' and cfg['nb'] != 'solo' and not cfg['outer']:
    res, info = k2_check.check(WORK, cfg['expect_fleet'], cfg['expect_crop'], expect_members=EXPECT_MEMBERS)
else:
    res, info = [], {}''')
rep('''res += [(n, bool(c), '') for n, c in extra]
for n, c, d in res:''',
    '''# ---- K3 checks
rec = g.get('_k2_receipt') or {}
why = str(rec.get('why'))
sub, solo = f'{WORK}/submission.csv', f'{WORK}/submission_fleetsolo.csv'
loud_left = [f for f in os.listdir(WORK) if f.startswith('submission_')]
if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown') and not cfg['outer']:
    if cfg['expect_fleet'] == 'clean':
        extra.append(('strict ok: submission.csv == submission_w25.csv bytes (P+F at W .25)',
                      os.path.isfile(sub) and open(sub, 'rb').read() == open(f'{WORK}/submission_w25.csv', 'rb').read()))
        extra.append(('strict ok: submission.csv differs from the parent bytes', open(sub, 'rb').read() != par_bytes))
    else:
        extra.append(('strict fail: submission.csv DELETED (never the parent bytes)', not os.path.exists(sub)))
        extra.append(('strict fail: no fail-loud file left', not loud_left))
        extra.append(('strict fail: _k2_parent.csv kept (parent bytes, not submittable)',
                      not os.path.isfile(f'{WORK}/_k2_parent.csv') or open(f'{WORK}/_k2_parent.csv', 'rb').read() == par_bytes))
if SC == 'strict_writefail':
    extra.append(('strict writefail: why names the injected write error', 'No space left' in why))
if SC == 'strict_lowtime':
    extra.append(('strict lowtime: no worker was launched', not glob.glob(f'{WORK}/_fleet_*.npz')))
if SC == 'strict_nods':
    extra.append(('strict nods: why says not attached', 'not attached' in why))
if cfg['outer']:
    left = os.path.isfile(sub) and open(sub, 'rb').read()
    if cfg['nb'] == 'fleet':
        extra.append(('blend outer: submission.csv restored to the parent bytes (K2 behaviour kept)', left == par_bytes))
    else:
        extra.append((f'{cfg["nb"]} outer: submission.csv DELETED (K3a: never the parent, never the fake)', not os.path.exists(sub)))
    extra.append(('outer: fail-loud files removed', not loud_left))
    extra.append(('outer: _k2_entry returned None (exception contained)', g.get('_k2_receipt') is None))
if EXPECT_MEMBERS is not None and cfg['expect_fleet'] == 'clean':
    extra.append((f'members: receipt names == {EXPECT_MEMBERS}, n_keep == {len(EXPECT_MEMBERS)} per worker',
                  sorted(rec.get('names') or []) == sorted(EXPECT_MEMBERS) and rec.get('n_keep') == [len(EXPECT_MEMBERS)] * 2))
    md = rec.get('member_dir')
    extra.append(('members: private FLEET_DIR removed after the stage', bool(md) and not os.path.exists(md)))
    extra.append(('members: nothing extra in the working dir',
                  not [f for f in os.listdir(WORK) if 'k2_members' in f or f.endswith('.pt')]))
if SC == 'strict_ttr_all':
    extra.append(('no filter: both manifest members ran (TTR + family 1)', rec.get('models') == 2 and
                  sorted(rec.get('names') or []) == sorted([TTRM['f1'], TTRM['ttr']]) and rec.get('member_dir') is None))
if SC == 'ttrsolo_ok':
    extra.append(('ttrsolo: submission.csv == submission_fleetsolo.csv bytes (solo mode)',
                  os.path.isfile(sub) and open(sub, 'rb').read() == open(solo, 'rb').read()))
if SC == 'members_mismatch':
    extra.append(('mismatch: why names the member filter', 'member filter selected' in why))
if SC == 'members_unknown':
    extra.append(('unknown: why names the manifest', 'not in the fleet manifest' in why))
    extra.append(('unknown: no worker was launched', not glob.glob(f'{WORK}/_fleet_*.npz')))
    extra.append(('unknown: no private FLEET_DIR created', rec.get('member_dir') is None))
if SC in ('strict_fail', 'members_mismatch'):
    time.sleep(1)
    import psutil
    kids = [c for c in psutil.Process().children(recursive=True) if c.is_running() and c.status() != psutil.STATUS_ZOMBIE]
    extra.append(('no worker left running after the cell', not kids))
res += [(n, bool(c), '') for n, c in extra]
for n, c, d in res:''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print('run_scenario patched')
